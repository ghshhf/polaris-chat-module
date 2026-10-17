"""把对话模块的训练语料导出成可开源版本。

剔除三类不能公开的内容：
  A. 涉性/色情续写（12 条，露骨描写）
  B. 小说续写任务（"继续写下去"/"请续写…"，属创作模块，非对话模块，含未公开作品片段）
  C. 暴露个人项目细节的代码实现（内部配置名、行情源、策略函数名等）

用法:
  python tools/export_open_chat.py --in data/chat_sft_v6.jsonl --out <dir>/data/chat_sft_public.jsonl
"""
import argparse
import json

MARK = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"

# A. 涉性关键词
SEX = ["肉棒", "射完", "射了出来", "乳头", "性器", "小穴", "鸡巴", "龟头", "子宫", "呻吟", "淫", "舔", "胸", "裸", "丝袜", "情趣", "榨精", "爱抚", "春欲"]
# B. 小说续写
NOVEL = ["继续写下去", "请续写", "风格的小说片段", "保持笔调", "世界观一致"]
# C. 个人项目细节（内部文件名/字段/业务函数名/行情源）
PRIVATE = [
    "strategy_config", "事件注入", "腾讯行情", "alt_rs", "t_idx", "sectors", "print_pool_block",
    "实时市值", "市价口径", "成本口径", "时机篮子", "全面板", "test_apply_to_crypto",
    "changan_guilu", "jiyi_dangpu", "load_event_override", "load_pairs", "mdd",
    "仓位", "权益曲线", "市场温度计", "候选列表", "回测", "因子", "收益率", "交易时段",
    "实时行情", "持仓", "净值", "代币", "标的", "篮子", "降级", "熔断",
]
# D. 代码实现任务（属编程模块，不属对话模块）
CODE_TASK = ["用 Python 实现", "实现函数", "用 JS 实现", "写一个函数", "补全代码", "重构下面"]


def q_of(text):
    k = text.find(MARK)
    return text[:k].split("user\n")[-1].replace(IM_END, "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    keep, drop = [], {"涉性": 0, "小说续写": 0, "个人项目": 0, "代码实现任务": 0}
    for r in rows:
        s = r["text"]
        q = q_of(s)
        if any(k in s for k in SEX):
            drop["涉性"] += 1
            continue
        if any(k in q for k in NOVEL):
            drop["小说续写"] += 1
            continue
        if any(k in q for k in PRIVATE):
            drop["个人项目"] += 1
            continue
        if any(k in q for k in CODE_TASK) or "```" in q:
            drop["代码实现任务"] += 1
            continue
        keep.append(r)
    print(f"输入 {len(rows)} 条 → 保留 {len(keep)} 条")
    for k, v in drop.items():
        print(f"  剔除·{k} = {v} 条")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {a.out}")


if __name__ == "__main__":
    main()
