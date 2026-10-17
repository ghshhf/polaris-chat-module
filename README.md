# 北极星 · 中文对话模块

一个 7.10M 参数的 LoRA 模块，给 0.2B 级中文主干装上"开口说话"的能力。

它的职责只有三件事：**让模型以助手身份应答、守住人格设定、按 ChatML 格式收尾**。
它不负责知识，不负责代码，不负责写作——那些是别的模块的事。

- 类型：`peft-lora`（r=16, alpha=32, dropout=0.05）
- 需要配对主干：SmolLM2 架构，hidden=576 / 36 层 / inter=4608 / GQA 3 头 / vocab=49152
- 语言：中文（含少量英文应答）
- 许可证：Apache-2.0

---

## 一、它是什么，不是什么

| 是 | 不是 |
|---|---|
| 一个可插拔的对话模块，换主干只换它 | 一个通用聊天模型（它没有知识） |
| 教协议、格式、人格 | 教事实、教推理、教编程 |
| 7.10M 可训参数，一张 27MB 的卡 | 一个需要重训的完整模型 |
| 与主干维度解耦（维度全走配置） | 硬编码 hidden_size 的胶水代码 |

设计上遵守一条硬规则：**模块里不出现任何主干维度常量**。换 0.3B / 0.4B 主干时，维度变了只需要重训这个模块，不需要改一行代码。

---

## 二、目录结构

```
polaris-chat-module/
├── adapter/
│   ├── adapter_model.safetensors   27MB · 模块权重
│   └── adapter_config.json         模块配置（r/alpha/target_modules）
├── module.card.json                模块卡：依赖、输入输出、验收判据、局限
├── data/
│   └── chat_sft_public.jsonl       304 条脱敏训练样本（完整语料 404 条未开源）
├── tools/
│   ├── train_lora.py               SFT 训练（completion_only_loss）
│   ├── chat_diag.py                语料体检 + 逐条 in-set loss 分布
│   ├── chat_conflicts.py           打印"同问多答"冲突组
│   ├── clean_chat.py               清洗：去冲突、去空壳指令
│   ├── export_open_chat.py         脱敏导出（去涉性/小说/私有项目/代码任务）
│   └── overfit_test.py             过拟合探针：判定管道通不通
└── eval/
    └── RESULT.md                   实测数字，含未达成的部分
```

---

## 三、快速开始

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/path/to/your/0.2b-base"          # 换成你的主干路径
ADAPTER = "polaris-chat-module/adapter"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=64, do_sample=False,
                         eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False))
```

`adapter_config.json` 里的 `base_model_name_or_path` 是占位符 `YOUR_BASE_MODEL_PATH`。
用 `PeftModel.from_pretrained(model, ...)` 显式传入主干时它会被忽略，这是故意的——模块不替使用者决定主干。

---

## 四、训练配置（完整可复现）

| 项 | 值 |
|---|---|
| 基底 | 0.2B 自训主干（SmolLM2 架构，r6 封顶版） |
| LoRA | r=16, alpha=32, dropout=0.05, bias=none |
| target_modules | q/k/v/o_proj + gate/up/down_proj |
| 可训参数 | 7.10M |
| batch | 2（per_device 2 × grad_accum 1） |
| lr | 2e-4，cosine |
| epoch | 4 |
| max_seq_length | 512 |
| loss | `completion_only_loss=True`，prompt 全 mask 成 -100 |
| 精调精度 | fp32（CPU，无 CUDA） |
| 步数 | 864 |

```bash
python tools/train_lora.py \
  --model_path /path/to/base --data_path data/chat_sft_public.jsonl \
  --output_dir out_chat --mask_prompt \
  --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 \
  --num_train_epochs 4 --per_device_batch_size 2 --grad_accum 1 --max_seq_length 512
```

`--mask_prompt` 必须开。不开的话模型会学成"什么内容都往下续写"，推理时模仿用户输入而不是以助手身份回答。

---

## 五、实测结果

不吹。这是本模块第一版的真实数字。

**拟合度（训练集自身，404 条）**

| 指标 | 数值 |
|---|---|
| completion CE | **1.578** |
| ppl | 4.84 |
| 逐条 loss 中位 / 最好 / 最差 | 1.556 / 0.350 / 4.583 |
| 达标样本（CE < 0.60） | **14 / 439 = 3.2%** |

**loss 分布暴露的问题**

最容易的 10 条全是"能从提问直接推出"的（`用 Python 实现函数 mdd` → `def mdd`、`1+1=?`），loss 0.35~0.49。
最难的 20 条全是真对话——`问你个事` 4.58、`谢谢` 2.49、`你在干嘛` 2.42、`现在几点了` 2.36。

**模型只学会了"抄写"，没学会"记住配对"。**

**过拟合探针（独立判定）**

40 条最短样本，lr 5e-4，dropout 0，batch 4：

```
step   1  loss 5.1961
step  20  loss 1.9813
step  40  loss 1.0034
step  60  loss 0.4077
step  80  loss 0.2001
step  90  近 40 步均 0.2845  ⇒ 管道通
```

结论：**训练管道完全正常**（模块挂载、掩码、loss、反传都对）。学不会不是管道问题。

---

## 六、已知局限

1. **数据量不足是主因**。404 条 × 4 epoch = 21.5 万 completion token、808 次参数更新，每条答案平均只被看 4 次、权重只被更新 2 次。对照主干预训练的 1.77 亿 token，SFT 只占 1/820。
2. **词表把成本放大 2.5 倍**。实测 **2.48 token/汉字**（正常应 1.0~1.2）。一个汉字要 2.5 个决策，错误逐字累积。
3. **EOS 学得弱**。长回答有复读倾向，偶尔冲过结尾继续生成。
4. **无知识深度**。事实性问题必须由检索模块或推理时学习模块承担，不要指望它背。
5. **数据曾有 7.7% 硬伤**（已修）。第一版语料里 13 条"理解并重现以下代码"的提问根本没有代码，13 条共用一个 prompt 配 13 个答案，loss 下限 ln13≈2.56，数学上不可学；另有 21 组"同问不同答"。清洗后 439 → 404。

---

## 七、数据

`data/chat_sft_public.jsonl` 是 304 条脱敏样本，从 404 条完整语料中筛出：

| 剔除项 | 条数 | 原因 |
|---|---|---|
| 涉性内容 | 11 | 露骨描写，不适合公开 |
| 小说续写任务 | 18 | 属创作模块，且含未公开作品片段 |
| 个人项目代码 | 15 | 暴露私有项目结构与业务字段 |
| 代码实现任务 | 56 | 属编程模块，不属对话模块 |

完整语料未开源：里面有人与模型之间的真实往来语句，按最小暴露原则不外放。
`tools/export_open_chat.py` 就是这次脱敏用的脚本，规则全在里面，可复核可重跑。

数据格式（ChatML，每行一条）：

```json
{"text": "<|im_start|>user\n在吗<|im_end|>\n<|im_start|>assistant\n在，主人。一直都在。<|im_end|>"}
```

---

## 八、怎么判断它"学会了"

三条判据，缺一不可：

1. **in-set 先拟合**：训练集上的 completion CE 必须 < 0.60。做不到就别谈泛化。
2. **盲测用没见过的输入**：必须是训练集未出现的问法 + 已知答案对照，不能只看 loss。
3. **过拟合探针先跑**：`overfit_test.py` 的 PASS 是前置条件。管道不通时，扩数据、调超参全是白费。

```bash
# 1. 体检语料
python tools/chat_diag.py --data data/chat_sft_public.jsonl

# 2. 跑过拟合探针（必须先 PASS）
python tools/overfit_test.py --model /path/to/base --data data/chat_sft_public.jsonl \
    --n 40 --steps 320 --lr 5e-4

# 3. 训完后量 in-set 拟合度
python tools/chat_diag.py --data data/chat_sft_public.jsonl \
    --model /path/to/base --adapter out_chat
```

---

## 九、这个模块在第一版里没做完的事

写清楚，免得别人重复踩：

- 没做到"每条训练样本都拟合"。3.2% 的达标率说明数据量和训练预算都差一个数量级。
- 没做词表改造。中文被切成字节碎片，`你好` 要 4 个 token。这项改动需要重训 embedding，属于主干该干的活，不在模块层面能修。
- 没做停用判据。训练按 epoch 停，不按"loss 收敛"停，这是当时的缺陷。
- 没做多轮对话。当前只覆盖单轮问答。

下一版的方向写在 `eval/RESULT.md`。

---

## License

Apache-2.0
