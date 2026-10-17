"""打印同问多答的冲突组明细，判断数据是否自相矛盾。"""
import collections
import json
import sys

MARK = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
path = sys.argv[1] if len(sys.argv) > 1 else "data/chat_sft_v5.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
byp = collections.defaultdict(list)
for i, r in enumerate(rows):
    t = r["text"]
    k = t.find(MARK)
    byp[t[:k]].append((i, t[k + len(MARK):]))

conf = [(p, v) for p, v in byp.items() if len(v) > 1]
print(f"冲突组 = {len(conf)}  涉及条数 = {sum(len(v) for _, v in conf)}/{len(rows)}")
for p, v in conf:
    q = p.split("user\n")[-1].replace(IM_END, "").strip().replace("\n", " ")
    print(f"\n=== 问题({len(v)}答, 提示长 {len(p)}): {q[:70]}")
    for i, c in sorted(v):
        print(f"  #{i:<4} {c.strip()[:80].replace(chr(10), ' ')}")
