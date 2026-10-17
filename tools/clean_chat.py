"""对话 SFT 数据清洗：消除"自相矛盾"样本，输出可学的 v6。

三类硬伤（chat_diag/chat_conflicts 实测）：
  A. 同 prompt 多答案 —— 同一输入给了多个互不相同的回答，模型只能学平均，loss 有硬下限。
  B. 指令缺输入 —— "理解并重现以下代码"但 prompt 里根本没有代码（builder 丢了），13 条共用一个 prompt。
  C. 空/极短 completion。

策略：A 每组保留 1 条（优先含"主人"、长度接近组中位的），B 直接删（无法反推原代码），C 删。

用法:
  python tools/clean_chat.py --in data/chat_sft_v5.jsonl --out data/chat_sft_v6.jsonl
"""
import argparse
import collections
import json
import re

MARK = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"

# 指令声称"下面有代码/文本"，但 prompt 里实际为空壳
NEED_INPUT = [
    ("重现", "代码片段"),
    ("以下代码", None),
    ("这段代码", None),
    ("下面这段", None),
]


def split(t):
    k = t.find(MARK)
    return t[:k], t[k + len(MARK):]


REFER = ("以下", "下面", "这段", "上述", "该代码", "此代码", "重现", "复现", "改写", "重构", "解释这")
CODEISH = ("{", "}", "(", ")", "=", ";", "\n", "def ", "function", "class ")


def needs_input_but_empty(prompt):
    """指令里用了指代词（以下/这段/重现…），但正文没有对应的输入内容。

    判据二选一：
      ① 正文短(<45字)、无换行、却声称"以下/下面/这段/上述" ⇒ 输入被丢了
      ② 去掉套话后实质字数 < 20
    """
    body = prompt.split("user\n")[-1].replace(IM_END, "").strip()
    if not any(k in body for k in REFER):
        return False
    if sum(1 for k in CODEISH if k in body) >= 2 or body.count("\n") >= 2:
        return False
    if len(body) < 45 and any(k in body for k in ("以下", "下面", "这段", "上述", "该代码", "此代码")):
        return True
    core = re.sub(r"(请|帮我|帮忙|理解并|重新|以下|下面|这段|上述|该|此|代码|片段|实现|逻辑|功能|内容|的|并|和)", "", body)
    core = re.sub(r"[\s:：,，。?？!！、]*", "", core)
    return len(core) < 20


def pick(group):
    """同组多答里选一条：含'主人'优先 → 长度接近组中位 → 短的优先（少噪声）。"""
    def score(item):
        i, c = item
        has = 1 if "主人" in c else 0
        return (-has, abs(len(c) - 40), i)
    return sorted(group, key=score)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    byp = collections.defaultdict(list)
    kept, drop_b, drop_c = [], [], []
    for i, r in enumerate(rows):
        p, c = split(r["text"])
        if len(c.strip()) < 4:
            drop_c.append(i)
            continue
        if needs_input_but_empty(p):
            drop_b.append(i)
            continue
        byp[p].append((i, c))

    seen = {}
    for p, g in byp.items():
        i, c = pick(g)
        kept.append((i, p + MARK + c))
        seen[p] = len(g)
    kept.sort()
    out = [{"text": t} for _, t in kept]

    print(f"输入 {len(rows)} 条")
    print(f"  删除·指令缺输入(B) = {len(drop_b)} 条  idx={drop_b[:15]}{'...' if len(drop_b) > 15 else ''}")
    print(f"  删除·空短答(C)     = {len(drop_c)} 条")
    print(f"  合并·同问多答(A)   = {sum(v - 1 for v in seen.values() if v > 1)} 条  (涉及 {sum(1 for v in seen.values() if v > 1)} 组)")
    print(f"输出 {len(out)} 条 → {a.out}")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
