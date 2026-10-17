"""过拟合极限测试：把 N 条样本训到"背下来"，检验管道是否通。

判据（唯一目的：分离"管道/容量问题"和"数据量问题"）：
  - 40 条、~40 epoch 内 loss 能降到 < 0.30  ⇒ 管道通，学不会只是数据量/epoch 不够，扩数据有意义
  - 降不到，甚至卡在 1.0 以上              ⇒ 管道或底座有问题，扩数据无意义，先修管道/换底座

不做验证集、不做正则（dropout=0），就是要它记住。

用法:
  python tools/overfit_test.py --model backups/0.2B/r6_final_20261016 --data data/chat_sft_v6.jsonl --n 40 --steps 300
  python tools/overfit_test.py --model backups/0.3B/polaris-03b_20261018 --data data/chat_sft_v6.jsonl --n 40 --steps 300
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

MARK = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def log(*a):
    print(*a, flush=True)


def build_samples(tok, rows, n, max_len):
    out = []
    for r in rows:
        t = r["text"]
        k = t.find(MARK)
        if k < 0:
            continue
        prompt = t[: k + len(MARK)]
        comp = t[k + len(MARK):]
        pids = tok(prompt, add_special_tokens=False).input_ids
        cids = tok(comp, add_special_tokens=False).input_ids
        if not cids:
            continue
        ids = (pids + cids)[:max_len]
        labels = ([-100] * len(pids) + cids)[:max_len]
        if all(x == -100 for x in labels):
            continue
        out.append((ids, labels, len(pids)))
    # 短的优先：短样本过拟合最快，最能暴露管道问题
    out.sort(key=lambda x: len(x[0]))
    sel = out[:n]
    log(f"[数据] 候选 {len(out)} 条，取最短 {len(sel)} 条；token 数 {min(len(x[0]) for x in sel)}~{max(len(x[0]) for x in sel)}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--target", type=float, default=0.30)
    ap.add_argument("--tag", default="test")
    a = ap.parse_args()

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    torch.manual_seed(42)
    random.seed(42)
    t0 = time.time()
    log(f"[模型] {a.model}")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    )
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=a.r,
        lora_alpha=a.r * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[模型] 可训参数 = {ntrain/1e6:.2f}M  device=cpu  threads={torch.get_num_threads()}")

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    samples = build_samples(tok, rows, a.n, a.max_len)
    pad = tok.pad_token_id

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

    def make_batch(idx):
        chunk = [samples[i] for i in idx]
        m = max(len(x[0]) for x in chunk)
        ids = torch.tensor([x[0] + [pad] * (m - len(x[0])) for x in chunk])
        lab = torch.tensor([x[1] + [-100] * (m - len(x[1])) for x in chunk])
        att = torch.tensor([[1] * len(x[0]) + [0] * (m - len(x[0])) for x in chunk])
        return ids, lab, att

    model.train()
    order = list(range(len(samples)))
    step = 0
    b = 0
    best = 9e9
    losses = []
    hit = False
    while step < a.steps:
        if b >= len(order):
            random.shuffle(order)
            b = 0
        idx = order[b: b + a.batch]
        b += a.batch
        ids, lab, att = make_batch(idx)
        out = model(input_ids=ids, attention_mask=att)
        shift_logits = out.logits[:, :-1].contiguous()
        shift_lab = lab[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_lab.view(-1), ignore_index=-100
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        lv = float(loss)
        losses.append(lv)
        best = min(best, lv)
        if step % 20 == 0 or step == 1:
            recent = sum(losses[-20:]) / len(losses[-20:])
            log(f"  step {step:4d}  loss {lv:.4f}  avg20 {recent:.4f}  best {best:.4f}  {time.time()-t0:.0f}s")
        if step >= 60 and sum(losses[-40:]) / 40 < a.target:
            hit = True
            log(f"[判定] step {step}: 最近40步均 loss {sum(losses[-40:])/40:.4f} < {a.target} ⇒ **管道通**")
            break

    final = sum(losses[-40:]) / min(40, len(losses))
    log("")
    log(f"[结果] {a.tag}: 最终(近40步均) loss={final:.4f}  ppl={2.718281828**final:.2f}  best={best:.4f}  steps={step}  用时={time.time()-t0:.0f}s")
    log(f"[判定] {'PASS 管道通，可扩数据' if hit else 'FAIL 管道/底座有问题，扩数据无用'}")

    out_dir = f"models/overfit_{a.tag}"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    log(f"[保存] {out_dir}")


if __name__ == "__main__":
    main()
