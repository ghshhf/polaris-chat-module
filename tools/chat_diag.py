"""对话数据体检 + 训练集拟合度诊断。

用法:
  python tools/chat_diag.py --data data/chat_sft_v5.jsonl            # 只做数据体检
  python tools/chat_diag.py --data ... --model backups/0.2B/r6_final_20261016 --adapter out_lora_chat_v4
                                                                      # 加逐条 completion loss 分布

输出要点:
  - prompt 重复 / 答案矛盾（同问不同答 ⇒ 不可学，构成 loss 硬下限）
  - completion 是否以 im_end 收尾（EOS 可学性）
  - token/汉字（坏词表代价）
  - 逐条 loss 分布 + top-20 难样本（判断是"管道没通"还是"样本本身不可学"）
"""
import argparse
import collections
import json
import os
import statistics

IM_END = "<|im_end|>"
MARK = "<|im_start|>assistant\n"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split(rec):
    t = rec["text"]
    i = t.find(MARK)
    if i < 0:
        return "", t
    return t[: i + len(MARK)], t[i + len(MARK):]


def data_report(rows):
    ps, cs = [], []
    for r in rows:
        p, c = split(r)
        ps.append(p)
        cs.append(c)
    n = len(rows)
    print(f"[数据] 条数 = {n}")
    print(f"[数据] prompt 唯一 = {len(set(ps))}  completion 唯一 = {len(set(cs))}")
    # 同问不同答
    byp = collections.defaultdict(set)
    for p, c in zip(ps, cs):
        byp[p].add(c.strip())
    conflict = {p: v for p, v in byp.items() if len(v) > 1}
    dupe_extra = sum(len(v) - 1 for v in byp.values())
    print(f"[数据] 同问多答组 = {len(conflict)}  其中答案互不相同 = {sum(1 for v in conflict.values() if len(v) > 1)}")
    print(f"[数据] 重复条数(冗余) = {dupe_extra}")
    if conflict:
        for p, v in list(conflict.items())[:3]:
            print("   冲突示例:", p.split("user\n")[-1].replace(IM_END, "")[:40], "->", len(v), "种答案")
    nend = sum(1 for c in cs if c.rstrip().endswith(IM_END))
    print(f"[数据] completion 以 im_end 结尾 = {nend}/{n}")
    lens = sorted(len(c) for c in cs)
    print(f"[数据] completion 字符长 中位={lens[n // 2]} 均值={sum(lens) // n} 最大={lens[-1]}")
    print(f"[数据] 含代码块 = {sum(1 for c in cs if '```' in c)}  含'主人' = {sum(1 for c in cs if '主人' in c)}")
    # 分类粗看：前 40 字前缀
    print("[数据] 抽样 5 条:")
    for r in rows[:5]:
        p, c = split(r)
        print("   Q:", p.split("user\n")[-1].replace(IM_END, "").strip()[:40], "| A:", c.strip()[:40].replace("\n", " "))
    return ps, cs


def loss_report(rows, model_path, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if adapter and os.path.isdir(adapter):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[模型] 已挂 adapter: {adapter}")
    model.eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    import torch.nn.functional as F

    per = []
    tot_tok = 0
    han = 0
    for idx, r in enumerate(rows):
        p, c = split(r)
        full = p + c
        pids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
        fids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids
        np_ = pids.shape[1]
        with torch.no_grad():
            out = model(fids)
            logits = out.logits[0, :-1]
            tgt = fids[0, 1:]
            if np_ >= 1:
                logits = logits[np_ - 1:]
                tgt = tgt[np_ - 1:]
            l = F.cross_entropy(logits, tgt, reduction="sum")
            per.append((float(l) / max(1, tgt.numel()), idx, tgt.numel()))
            tot_tok += int(tgt.numel())
        han += sum(1 for ch in full if "\u4e00" <= ch <= "\u9fff")
        if (idx + 1) % 50 == 0:
            print(f"  ...{idx + 1}/{len(rows)}")
    per.sort(reverse=True)
    losses = [x[0] for x in per]
    mean = sum(l * n for l, _, n in per) / max(1, tot_tok)
    print(f"\n[loss] 全量 completion CE = {mean:.4f}  ppl = {2.718281828 ** mean:.2f}")
    print(f"[loss] 逐条中位 = {statistics.median(losses):.4f}  最好 = {min(losses):.4f}  最差 = {max(losses):.4f}")
    ntok = sum(t for _, _, t in per)
    print(f"[loss] completion token = {ntok}  汉字 = {han}  token/汉字 = {ntok / max(1, han):.2f}")
    print(f"[loss] 达标(<0.60)条数 = {sum(1 for l in losses if l < 0.6)}/{len(losses)}")
    print("\n[最难 20 条]  (loss, idx, 问题前 38 字)")
    for l, i, t in per[:20]:
        p, c = split(rows[i])
        q = p.split("user\n")[-1].replace(IM_END, "").strip().replace("\n", " ")[:38]
        print(f"  {l:.3f}  #{i:<4} {q}")
    print("\n[最易 10 条]")
    for l, i, t in per[-10:]:
        p, c = split(rows[i])
        q = p.split("user\n")[-1].replace(IM_END, "").strip().replace("\n", " ")[:38]
        print(f"  {l:.3f}  #{i:<4} {q}")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    rows = load(a.data)
    data_report(rows)
    if a.model:
        print()
        loss_report(rows, a.model, a.adapter)


if __name__ == "__main__":
    main()
