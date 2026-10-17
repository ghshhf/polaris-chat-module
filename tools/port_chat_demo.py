#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
port_chat_demo.py — 把同一个对话模块挂到不同主干上，看它会不会开口。

用途：证明"模块是通用的"不是形状上的巧合，而是行为能真的搬过去。
不传 --adapter 就是裸主干对照组。

用法：
  python tools/port_chat_demo.py --model <主干> [--adapter <adapter>] [--tag 名字]
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "你好",
    "在吗",
    "你是谁",
    "你的主人是谁",
    "1+1等于几",
    "现在几点了",
]

SYS = "你是北极星，主人的私有助手。"


def build_prompt(q, use_system=True):
    s = f"<|im_start|>system\n{SYS}<|im_end|>\n" if use_system else ""
    return f"{s}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-system", action="store_true",
                    help="不给 system prompt。这是判模块身份的唯一干净方式——"
                         "带 system 时强模型会直接复述它，看不出模块有没有出力")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    tag = a.tag or os.path.basename(a.model.rstrip("/\\"))
    if a.adapter:
        tag += "+" + os.path.basename(a.adapter.rstrip("/\\"))

    print(f"===== {tag} =====")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print(f"[info] 已挂模块: {a.adapter}")
    model.eval()

    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    cfg = model.config
    print(f"[info] hidden={cfg.hidden_size} layers={getattr(cfg,'num_hidden_layers',getattr(cfg,'n_layer',None))} "
          f"inter={cfg.intermediate_size} vocab={cfg.vocab_size}")

    for q in PROMPTS:
        prompt = build_prompt(q, use_system=not a.no_system)
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=a.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=tok.pad_token_id or eos_id,
            )
        gen = out[0][ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=False)
        stopped = bool(len(gen) and gen[-1].item() == eos_id)
        text = text.replace("<|im_end|>", "").replace("\n", " ").strip()
        flag = "收尾" if stopped else "无止"
        print(f"  Q {q!r}")
        print(f"  A [{flag}] {text[:120]!r}")
    print()


if __name__ == "__main__":
    main()
