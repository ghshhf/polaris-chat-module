#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
module_loader.py — 把任意 LoRA 模块挂到任意因果语言模型主干上。

设计前提（这是"通用模块"的工程落实，不是为某一个主干写的胶水）：

  模块只在「恒定维」上表达行为：q/k/v/o 输出都是 hidden 或 kv_dim，
  它们的语义（"该怎么答"）不随主干参数规模改变。
  模块不碰 embedding、不碰词表、不碰 MLP 宽维 —— 那些是知识所在，
  知识由主干提供，模块只负责开口方式。

能力：
  1. probe   探测目标主干的规格（hidden / layers / heads / kv_heads / inter / vocab）
  2. diff    对比模块与主干的形状差异，逐张量列出「直装 / 零填充 / 截断」
  3. port    生成可装载到目标主干的 adapter 目录（零填充或 SVD 截断）
  4. verify  逐位校验移植后的保函数性（旧子空间 max|diff|）

零填充的正确方向（保函数的唯一依据）：
  lora_A: [r, d_in]   —— 在最后一维（列）右侧补 0
  lora_B: [d_out, r]  —— 在第一维（行）下方补 0
  ⇒ ΔW = B @ A，旧子空间逐位相等，新维贡献恰为 0。
  这与 expand_to_target.py 的保函数扩容同源（Net2Net）。

用法：
  python tools/module_loader.py probe  --model <主干目录>
  python tools/module_loader.py diff   --model <主干目录> --adapter <adapter目录>
  python tools/module_loader.py port   --model <主干目录> --adapter <adapter目录> --out <输出目录>
"""
import argparse
import json
import os
import re
import sys

# 注意力投影的通用命名表。顺序即优先级：先标准名，再各家族的方言。
# 只收「单独投影」；fused 投影（c_attn / query_key_value / Wqkv）暂不支持，
# 因为它们的权重是 q/k/v 拼接，拆分规则各家族不同，硬猜会静默错位。
ATTN_NAMES = {
    "q": ["q_proj", "query", "wq", "q_lin", "Wq", "q"],
    "k": ["k_proj", "key", "wk", "k_lin", "Wk", "k"],
    "v": ["v_proj", "value", "wv", "v_lin", "Wv", "v"],
    "o": ["o_proj", "out_proj", "dense", "wo", "o_lin", "Wo", "proj"],
}
FUSED = ["c_attn", "query_key_value", "Wqkv", "in_proj"]


def eprint(*a):
    print(*a, file=sys.stderr)


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- probe
def probe(model_dir):
    cfg_p = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_p):
        raise SystemExit(f"[X] 找不到主干配置: {cfg_p}")
    c = load_json(cfg_p)

    hidden = c.get("hidden_size") or c.get("n_embd") or c.get("d_model")
    layers = c.get("num_hidden_layers") or c.get("n_layer") or c.get("num_layers")
    heads = c.get("num_attention_heads") or c.get("n_head") or c.get("num_heads")
    kv_heads = c.get("num_key_value_heads") or heads
    inter = c.get("intermediate_size") or c.get("n_inner") or c.get("ffn_dim")
    vocab = c.get("vocab_size")

    if not hidden or not layers:
        raise SystemExit("[X] 主干配置里没有 hidden_size / num_hidden_layers，无法探测")

    head_dim = hidden // heads if heads else None
    kv_dim = (kv_heads * head_dim) if head_dim else hidden

    # 扫描真实模块名，确认方言
    names = set()
    wf = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(wf):
        try:
            from safetensors import safe_open
            with safe_open(wf, framework="pt") as f:
                names = set(f.keys())
        except Exception as e:
            eprint(f"[warn] 读 safetensors 失败({e})，退回按配置推断")
    if not names:
        idx = os.path.join(model_dir, "pytorch_model.bin.index.json")
        if os.path.exists(idx):
            names = set(load_json(idx)["weight_map"].keys())
    if not names:
        # 备用：按目录内的 model 文件存在性判断
        eprint("[warn] 无法读取权重索引，接入点将按标准命名假设")

    found = {}
    for slot, cands in ATTN_NAMES.items():
        for cand in cands:
            hit = [n for n in names if re.search(rf"\.{re.escape(cand)}\.weight$", n)]
            if hit:
                found[slot] = cand
                break

    fused_hit = [f for f in FUSED if any(re.search(rf"\.{re.escape(f)}\.", n) for n in names)]

    spec = {
        "model_dir": os.path.abspath(model_dir),
        "architecture": c.get("architectures", ["?"])[0],
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "kv_dim": kv_dim,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "attn_module_names": found,
        "fused_modules_detected": fused_hit,
    }

    print(f"主干    : {spec['architecture']}")
    print(f"  hidden={hidden}  layers={layers}  heads={heads}(kv={kv_heads})  "
          f"head_dim={head_dim}  kv_dim={kv_dim}  inter={inter}  vocab={vocab}")
    if found:
        print("  接入点:" + "  ".join(f"{k}={v}" for k, v in found.items()))
    if fused_hit:
        eprint(f"[warn] 检测到 fused 投影 {fused_hit}：拆分规则因家族而异，本工具不自动处理，"
               f"需人工指定映射（见 module.card.json 的 interface.fused_map）")
    return spec


# ---------------------------------------------------------------- 读 adapter
def read_adapter(adapter_dir):
    cfg = load_json(os.path.join(adapter_dir, "adapter_config.json"))
    wf = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(wf):
        raise SystemExit(f"[X] 找不到 adapter 权重: {wf}")
    from safetensors.torch import load_file
    sd = load_file(wf)
    return cfg, sd


def split_key(k):
    """把 adapter 的 key 拆成 (层序号, 模块名, A/B)。非层内张量返回 (None,...)。"""
    m = re.search(r"layers\.(\d+)\.(.+?)\.(lora_[AB])(?:\.weight|\.default\.weight)$", k)
    if not m:
        return None, k, None
    return int(m.group(1)), m.group(2), m.group(3)


# ---------------------------------------------------------------- diff
MLP_KINDS = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down",
             "gate": "gate", "up": "up", "down": "down"}


def expect_dims(spec, base, ab):
    """返回 (d_in, d_out)。d_in = lora_A 的第二维，d_out = lora_B 的第一维。
    返回 None 表示这个模块名不认识。"""
    h, kv, it = spec["hidden_size"], spec["kv_dim"], spec.get("intermediate_size")
    table = {
        "q_proj": (h, h), "query": (h, h), "wq": (h, h),
        "k_proj": (h, kv), "key": (h, kv), "wk": (h, kv),
        "v_proj": (h, kv), "value": (h, kv), "wv": (h, kv),
        "o_proj": (h, h), "out_proj": (h, h), "dense": (h, h), "wo": (h, h),
    }
    if base in table:
        return table[base]
    if it and base in MLP_KINDS:
        kind = MLP_KINDS[base]
        if kind in ("gate", "up"):
            return (h, it)
        return (it, h)
    return None


def op_for(src, tgt):
    if src == tgt:
        return "ok"
    if src < tgt:
        return f"zero-pad +{tgt - src}"
    return f"svd-truncate -{src - tgt}"


def compare(spec, adapter_dir):
    cfg, sd = read_adapter(adapter_dir)
    rows = []
    for k, v in sd.items():
        layer, mod, ab = split_key(k)
        if ab is None:
            continue
        base = mod.rsplit(".", 1)[-1]
        dims = expect_dims(spec, base, ab)
        shp = tuple(v.shape)
        if dims is None:
            rows.append({"key": k, "module": mod, "base": base, "ab": ab, "shape": shp,
                         "expect": "未知模块", "action": "drop", "fragile": False})
            continue
        d_in, d_out = dims
        if ab == "lora_A":
            need, src, tgt = f"[{cfg['r']},{d_in}]", shp[1], d_in
        else:
            need, src, tgt = f"[{d_out},{cfg['r']}]", shp[0], d_out
        rows.append({
            "key": k, "module": mod, "base": base, "ab": ab,
            "shape": shp, "expect": need, "action": op_for(src, tgt),
            # 可扩维：这次能装，但主干一加宽就失效
            "fragile": base in MLP_KINDS,
        })
    return cfg, sd, rows


def summarize(rows):
    from collections import Counter
    c = Counter()
    for r in rows:
        a = r["action"]
        if a == "ok":
            c["直装"] += 1
        elif a == "drop":
            c["丢弃"] += 1
        elif "zero-pad" in a:
            c["零填充"] += 1
        else:
            c["截断"] += 1
    return c


# ---------------------------------------------------------------- port
def port(spec, adapter_dir, out_dir):
    import torch
    from safetensors.torch import save_file

    cfg, sd, rows = compare(spec, adapter_dir)
    by_key = {r["key"]: r for r in rows}
    out = {}
    dropped, padded, truncated, kept = [], [], [], []

    for k, t in sd.items():
        r = by_key.get(k)
        if r is None:
            out[k] = t
            continue
        if r["action"] == "drop":
            dropped.append(k)
            continue
        if r["action"] == "ok":
            out[k] = t
            kept.append(k)
            continue

        layer, mod, ab = split_key(k)
        d_in, d_out = expect_dims(spec, r["base"], ab)

        if "zero-pad" in r["action"]:
            if ab == "lora_A":
                new = torch.zeros((t.shape[0], d_in), dtype=t.dtype)
                new[:, : t.shape[1]] = t          # 旧子空间原样落在左侧
            else:
                new = torch.zeros((d_out, t.shape[1]), dtype=t.dtype)
                new[: t.shape[0], :] = t          # 旧输出维原样落在上方
            padded.append(k)
        else:
            # SVD 截断：目标维更小，只能近似保形。这不是保函数，需重训补偿。
            if ab == "lora_A":
                u, s, vh = torch.linalg.svd(t.float(), full_matrices=False)
                new = (u * s) @ vh
                new = new[:, :d_in].to(t.dtype)
            else:
                u, s, vh = torch.linalg.svd(t.float(), full_matrices=False)
                new = (u * s) @ vh
                new = new[:d_out, :].to(t.dtype)
            truncated.append(k)
        out[k] = new

    # 层数比模块少：多余层丢弃；比模块多：新层自然没有 LoRA（留给重训）
    mod_layers = sorted({split_key(k)[0] for k in out if split_key(k)[0] is not None})
    if mod_layers and max(mod_layers) >= spec["num_hidden_layers"]:
        cut = set()
        for k in list(out.keys()):
            l = split_key(k)[0]
            if l is not None and l >= spec["num_hidden_layers"]:
                cut.add(k)
                del out[k]
        dropped.extend(sorted(cut))

    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})

    new_cfg = dict(cfg)
    new_cfg.update({
        "base_model_name_or_path": os.path.abspath(spec["model_dir"]),
        "target_modules": [m for m in cfg.get("target_modules", [])
                           if m in {split_key(k)[1].rsplit(".", 1)[-1]
                                    for k in out if split_key(k)[1] is not None}],
        "polaris_port": {
            "from": os.path.abspath(adapter_dir),
            "to_arch": spec["architecture"],
            "to_hidden": spec["hidden_size"],
            "to_layers": spec["num_hidden_layers"],
            "kept": len(kept), "padded": len(padded),
            "truncated": len(truncated), "dropped": len(dropped),
        },
    })
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)

    print(f"\n[port] 直装 {len(kept)} · 零填充 {len(padded)} · 截断 {len(truncated)} · 丢弃 {len(dropped)}")
    if dropped:
        print(f"  丢弃样例: {dropped[:3]}")
    print(f"  输出: {os.path.abspath(out_dir)}")
    if truncated:
        eprint("[warn] 存在 SVD 截断张量：这不是保函数移植，装载后必须短程重训补偿")
    return out_dir


# ---------------------------------------------------------------- verify
def verify(spec, src_dir, dst_dir):
    """逐位校验：移植后 ΔW 的旧子空间必须与源完全相等。"""
    import torch
    from safetensors.torch import load_file

    a = load_file(os.path.join(src_dir, "adapter_model.safetensors"))
    b = load_file(os.path.join(dst_dir, "adapter_model.safetensors"))

    keys = [k for k in a if split_key(k)[2] is not None and k in b]
    groups = {}
    for k in keys:
        layer, mod, ab = split_key(k)
        groups.setdefault((layer, mod), {})[ab] = k

    worst = 0.0
    checked = 0
    skipped = 0
    for (layer, mod), g in groups.items():
        if "lora_A" not in g or "lora_B" not in g:
            continue
        A, B = a[g["lora_A"]], a[g["lora_B"]]
        A2, B2 = b[g["lora_A"]], b[g["lora_B"]]
        # 目标维 < 源维 ⇒ 已走 SVD 截断，不是保函数，无法逐位比较
        if A2.shape[1] < A.shape[1] or B2.shape[0] < B.shape[0]:
            skipped += 1
            continue
        if A2.shape[1] > A.shape[1]:
            A2 = A2[:, : A.shape[1]]
        if B2.shape[0] > B.shape[0]:
            B2 = B2[: B.shape[0], :]
        d0 = (B.float() @ A.float())
        d1 = (B2.float() @ A2.float())
        worst = max(worst, (d0 - d1).abs().max().item())
        checked += 1

    print(f"[verify] 校验 {checked} 组 ΔW，旧子空间 max|diff| = {worst:.3e}"
          + (f"；跳过 {skipped} 组（目标维更小，已 SVD 截断）" if skipped else ""))
    if checked == 0:
        print("[verify] N/A — 全部张量都需截断，本次移植不是保函数")
    elif worst == 0.0:
        print("[verify] PASS — 保函数成立（零填充部分旧行为逐位不变，新维贡献恰为 0）")
    elif worst < 1e-5:
        print("[verify] PASS(fp误差) — 数值噪声级别")
    else:
        print("[verify] FAIL — 旧行为已被改动，装载后必须重训")
    return worst


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "diff", "port", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=(name != "verify"))
        s.add_argument("--adapter", required=(name != "probe"))
        s.add_argument("--out")
        if name == "verify":
            s.add_argument("--dst", required=True)
    a = ap.parse_args()

    if a.cmd == "probe":
        probe(a.model)

    elif a.cmd == "diff":
        spec = probe(a.model)
        cfg, sd, rows = compare(spec, a.adapter)
        c = summarize(rows)
        print(f"\n模块    : r={cfg['r']} alpha={cfg.get('lora_alpha')} targets={cfg.get('target_modules')}")
        print(f"张量    : 共 {len(sd)}")
        print(f"结论    : 直装 {c['直装']} · 零填充 {c['零填充']} · 截断 {c['截断']} · 丢弃 {c['丢弃']}")
        fragile = sum(1 for r in rows if r["fragile"])
        if fragile:
            print(f"注意    : 其中 {fragile} 个属可扩维(MLP)，本次能装，但主干一加宽即失效")
        for r in rows[:]:
            if r["action"] != "ok":
                print(f"  {r['action']:<22} {r['key']}  {r['shape']} -> {r['expect']}")

    elif a.cmd == "port":
        if not a.out:
            raise SystemExit("[X] port 需要 --out")
        spec = probe(a.model)
        port(spec, a.adapter, a.out)

    elif a.cmd == "verify":
        spec = probe(a.model)
        verify(spec, a.adapter, a.dst)


if __name__ == "__main__":
    main()
