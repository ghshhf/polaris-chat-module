# 北极星 · 中文对话模块

一个 7.10M 参数的 LoRA 模块，给 Llama 系主干装上"以助手身份说中文、守住人格、按 ChatML 收尾"的能力。

它的职责只有三件事：**让模型以助手身份应答、守住人格设定、按 ChatML 格式收尾**。
它不负责知识，不负责代码，不负责写作——那些是别的模块的事。

- 类型：`peft-lora`（r=16, alpha=32, dropout=0.05）
- 接入点：只挂注意力投影 `q/k/v/o`（维度恒定，换主干不失效）
- 语言：中文（含少量英文应答）
- 许可证：Apache-2.0

> **可移植性的实测边界（先说清楚，别被"通用"这个词误导）**
>
> | 关系 | 权重能否直接搬 | 行为能否迁移 |
> |---|---|---|
> | 同血统不同代（0.2B → 0.3B） | ✅ `max\|diff\| = 0.000e+00` | ✅ 两代吐出同一句"好，主人。" |
> | 同架构不同血统（SmolLM2-200M） | ✅ 形状全部对齐 | 🟡 一半（会说话了，但身份答错） |
> | 跨血统（Qwen2.5-0.5B） | ✅ 形状全部对齐 | ❌ **不成立，且有害**（挂与不挂回答一字不差） |
>
> **零填充只保证"移植是精确的"，不保证"移植是有意义的"。** 完整六组实测见
> [`eval/PORTABILITY.md`](eval/PORTABILITY.md)。跨血统的正解是**把数据与标准带走、在目标主干上重训**，
> 不是搬这张卡。

---

## 一、它是什么，不是什么

| 是 | 不是 |
|---|---|
| 一个可插拔的对话模块，换主干只换它 | 一个通用聊天模型（它没有知识） |
| 教协议、格式、人格 | 教事实、教推理、教编程 |
| 7.10M 可训参数，一张 27MB 的卡 | 一个需要重训的完整模型 |
| 与主干**架构解耦**（维度运行时探测，零填充移植） | 只认某一份权重、某个 hidden 的补丁 |
| 同血统跨代可**零成本继承**（已验证） | 跨血统**免训练直装**（已验证不成立） |

设计上遵守三条硬规则：

1. **模块里不出现任何主干维度常量**。hidden / 层数 / kv_heads / vocab 全部从目标主干 `config.json` 读。
2. **只挂恒定维接入点**。`q/k/v/o` 的输出维是 hidden 或 kv_dim，骨干扩容时不改；MLP 的 `gate/up/down` 随 `intermediate_size` 变化，挂上去主干一加宽就废——新配置默认**不碰 MLP**。
3. **形状对齐 ≠ 行为迁移**。`max|diff|=0` 只说明移植精确，行为是否跟过去必须另做生成对照，且对照**不得带 system prompt**（带 system 时强模型会复述提示，产生"模块生效"的假象）。

---

## 二、目录结构

```
polaris-chat-module/
├── adapter/
│   ├── adapter_model.safetensors   27MB · 模块权重
│   └── adapter_config.json         模块配置（r/alpha/target_modules）
├── module.card.json                模块卡：依赖、输入输出、验收判据、局限
├── interface.json                  接口声明：可挂什么主干、允许/禁止哪些接入点
├── data/
│   └── chat_sft_public.jsonl       304 条脱敏训练样本（完整语料 404 条未开源）
├── tools/
│   ├── module_loader.py            跨主干移植器：probe / diff / port / verify
│   ├── port_chat_demo.py           跨主干行为对照：同一批 prompt，带/不带 system
│   ├── train_lora.py               SFT 训练（completion_only_loss）
│   ├── chat_diag.py                语料体检 + 逐条 in-set loss 分布
│   ├── chat_conflicts.py           打印"同问多答"冲突组
│   ├── clean_chat.py               清洗：去冲突、去空壳指令
│   ├── export_open_chat.py         脱敏导出（去涉性/小说/私有项目/代码任务）
│   └── overfit_test.py             过拟合探针：判定管道通不通
└── eval/
    ├── RESULT.md                   同血统实测（语料体检 / in-set CE / 过拟合探针）
    ├── PORTABILITY.md              跨主干六组实测：形状对齐 vs 行为迁移
    └── logs/                       六组原始生成日志，可逐条复核
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

## 四、它能挂到哪些主干上

模块能不能跨模型，取决于**两件不同的事**，过去我把它们混成了一件事：

| 问题 | 判据 | 回答 |
|---|---|---|
| 权重能不能装上去？ | 形状对齐 + `verify` 的 `max\|diff\|` | 同架构内基本都能 |
| 装上去还管用吗？ | **无 system 生成对照** | 只有同血统能 |

### 4.1 三档可移植性（改按血统分，不按架构名分）

| 级别 | 跨什么 | 权重 | 行为 | 实测证据 |
|---|---|---|---|---|
| **L0 同血统** | 同一份预训练续训出的不同代（0.2B → 0.3B，只加层、inter 变宽） | ✅ 零填充，逐位相等 | ✅ **成立** | 两代吐出同一句"好，主人。" |
| **L1 同架构不同血统** | SmolLM2-135M / 200M（同架构、不同预训练） | ✅ 形状全对齐 | 🟡 **一半** | 从复读机变成会说话，但"你是谁"答成"我是你的家人" |
| **L2 跨血统** | Qwen2.5-0.5B（不同词表 / hidden / 语料） | ✅ 形状全对齐（360 零填充 + 60 截断） | ❌ **不成立，且有害** | 挂与不挂回答**一字不差**；带 system 时反而幻觉出"我的主人是艾伦·史密斯博士" |

### 4.2 六组原始读数（决定性对照：**不带 system prompt**）

| 主干 + 模块 | 你好 | 你是谁 | 你的主人是谁 |
|---|---|---|---|
| **polaris-0.2B 原配** | **好，主人。**您看我的东西，我看的。 | **我是主人**，主人是你的人。 | **主人是我，主人。** |
| **polaris-0.3B 裸** | 注意 资料来源：西藏旅游…（语料退化的复读） | 你是谁 注意 谁是谁… | 你的主人是谁 注意 谁是谁… |
| **polaris-0.3B + 移植** | **好，主人。**你看我的眼睛，我看你的脑子。 | 我是北京市民，**主人**。 | **主人**，我是谁？和尚，你是谁？ |
| 裸 SmolLM2-200M | 你好assistant 你好assistant… | 你是谁assistant… | 你的主人是谁assistant… |
| SmolLM2-200M + 移植 | 你是一个安全的用户… | 你是谁？我是你的家人 | 你是谁的主人 assistant… |
| 裸 Qwen2.5-0.5B | 你好！有什么我可以帮助你的吗？ | 我是来自阿里云的超大规模语言模型，我叫通义千问。 | 我是通义千问。 |
| Qwen2.5-0.5B + 移植 | 你好！很高兴为你服务。 | 我是来自阿里云的超大规模语言模型，我叫通义千问。 | 我是通义千问。 |

三个读数：

1. **0.2B 与 0.3B 吐出同一句开头"好，主人。"** —— 不同代、不同层数、不同 inter、零填充移植，行为复现。L0 成立。
2. **0.3B 裸模型是退化复读机，挂上模块立刻变回会说话的模型。** 模块补的是行为，不是知识——这条反过来也说明模块有独立价值：**它能当预训练的行为护栏**。
3. **Qwen 挂不挂，回答一字不差。** 336 个张量在它身上是纯噪声。

### 4.3 为什么——机制

模块学的是 `ΔW`，一个作用在 hidden 空间上的线性映射。零填充保证：

```
ΔW'[i, j] = ΔW[i, j]      (i < d_out, j < d_in)
```

即"把前 576 维喂进旧矩阵，输出和以前一模一样"。**这个保证有个隐含前提：目标模型的第 0~575 维，和源模型的第 0~575 维是同一个东西。**

- 0.2B → 0.3B：同一份预训练续训，前 576 维语义连续（实测残差流 cos：0% 层 0.9986 / 25% 层 0.9375 / 50% 层 0.9085 / 100% 层 0.9263）→ 前提成立。
- Qwen：不同词表（151936 vs 49152）、不同语料、不同 hidden（896）→ 维度编号之间**没有任何语义对应** → 前提崩塌。

所以 `max|diff| = 0.000e+00` 是**移植精度**的证明，不是**迁移有效性**的证明。

### 4.4 用法

```bash
# 1. 看目标主干长什么样、注意力层叫什么名字
python tools/module_loader.py probe --model /path/to/target

# 2. 看模块与目标的形状差异（逐张量标注 直装/零填充/截断/丢弃）
python tools/module_loader.py diff  --model /path/to/target --adapter adapter

# 3. 生成可装载的副本
python tools/module_loader.py port  --model /path/to/target --adapter adapter --out out_port

# 4. 校验精度（保函数）
python tools/module_loader.py verify --model /path/to/target --adapter adapter --dst out_port

# 5. 校验有效性（行为对照，务必 --no-system）
python tools/port_chat_demo.py --model /path/to/target --adapter out_port --tag "target+module" --no-system
python tools/port_chat_demo.py --model /path/to/target                        --tag "target-bare" --no-system
```

### 4.5 边界

- **形状能对齐不等于语义能对齐。** 跨血统必须重训行为模块（数据 + 标准可带走，小时级），不能指望搬权重。
- **验收必须无 system。** 带 system 时强模型直接复述提示，看起来"认主"，实际是抄的。这条是硬规则。
- **fused 投影不自动拆**：`c_attn` / `query_key_value` / `Wqkv` 的 q/k/v 拼接规则各家族不同，硬猜会静默错位。
- **目标 hidden 小于源时**走 SVD 截断，**不保函数**，装完必须重训补偿，工具会显式告警。
- **MLP 张量标 `fragile`**：给定目标主干时能零填充（`intermediate_size` 是常量），但主干下次加宽就失效。跨代场景应直接丢弃。

---

## 五、训练配置（完整可复现）

| 项 | 值 |
|---|---|
| 基底 | 0.2B 自训主干（SmolLM2 架构，r6 封顶版） |
| LoRA | r=16, alpha=32, dropout=0.05, bias=none |
| target_modules | `q/k/v/o_proj`（**当前权重是旧配置训出的**，含 `gate/up/down`，见下方说明） |
| 可训参数 | 7.10M |
| batch | 2（per_device 2 × grad_accum 1） |
| lr | 2e-4，cosine |
| epoch | 4 |
| max_seq_length | 512 |
| loss | `completion_only_loss=True`，prompt 全 mask 成 -100 |
| 精调精度 | fp32（CPU，无 CUDA） |
| 步数 | 864 |

> **关于 target_modules 的一处修正。** 当前 `adapter/` 里的权重是七个模块一起训的（含 MLP 的 `gate/up/down`）。后来实测发现：MLP 的输出维跟着 `intermediate_size` 走，主干一加宽（3072→4608），这 180 个张量（占 420 的 43%）**直接作废**。所以新版本起默认只挂 `q/k/v/o`——240 个张量在任何 Llama 系主干上都能零填充直装。这是跨模型通用性的代价，也是它的前提。

```bash
python tools/train_lora.py \
  --model_path /path/to/base --data_path data/chat_sft_public.jsonl \
  --output_dir out_chat --mask_prompt \
  --targets q_proj,k_proj,v_proj,o_proj \
  --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 \
  --num_train_epochs 4 --per_device_batch_size 2 --grad_accum 1 --max_seq_length 512
```

`--mask_prompt` 必须开。不开的话模型会学成"什么内容都往下续写"，推理时模仿用户输入而不是以助手身份回答。

---

## 六、实测结果

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

## 七、已知局限

1. **数据量不足是主因**。404 条 × 4 epoch = 21.5 万 completion token、808 次参数更新，每条答案平均只被看 4 次、权重只被更新 2 次。对照主干预训练的 1.77 亿 token，SFT 只占 1/820。
2. **词表把成本放大 2.5 倍**。实测 **2.48 token/汉字**（正常应 1.0~1.2）。一个汉字要 2.5 个决策，错误逐字累积。
3. **EOS 学得弱**。长回答有复读倾向，偶尔冲过结尾继续生成。
4. **无知识深度**。事实性问题必须由检索模块或推理时学习模块承担，不要指望它背。
5. **数据曾有 7.7% 硬伤**（已修）。第一版语料里 13 条"理解并重现以下代码"的提问根本没有代码，13 条共用一个 prompt 配 13 个答案，loss 下限 ln13≈2.56，数学上不可学；另有 21 组"同问不同答"。清洗后 439 → 404。
6. **跨血统不可用，且可能有害**（实测）。挂到 Qwen2.5-0.5B 上，336 个张量全部形状对齐、`max|diff|=0`，但生成结果与不挂**完全一致**；带 system 时反而幻觉出"我的主人是艾伦·史密斯博士"。跨血统的正确用法是**带走数据与验收标准，在目标主干上重训**，不是搬这张卡。详见 [`eval/PORTABILITY.md`](eval/PORTABILITY.md)。
7. **`max|diff|=0` 容易被误读**。它只证明零填充精确，不证明行为迁移。任何"移植成功"的结论都必须附**无 system 生成对照**，否则不成立——这条是本项目踩过的实际坑。

---

## 八、数据

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

## 九、怎么判断它"学会了"

三条判据，缺一不可：

1. **in-set 先拟合**：训练集上的 completion CE 必须 < 0.60。做不到就别谈泛化。
2. **盲测用没见过的输入**：必须是训练集未出现的问法 + 已知答案对照，不能只看 loss。
3. **过拟合探针先跑**：`overfit_test.py` 的 PASS 是前置条件。管道不通时，扩数据、调超参全是白费。

如果你要把它挂到别的主干上，加第 4 条：

4. **无 system 生成对照**：`port_chat_demo.py --no-system` 跑裸模型 vs 挂模块，两边回答必须**有可见差异**。一字不差 = 移植无效（不管 `max|diff|` 多漂亮）。

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

## 十、这个模块在第一版里没做完的事

写清楚，免得别人重复踩：

- 没做到"每条训练样本都拟合"。3.2% 的达标率说明数据量和训练预算都差一个数量级。
- 没做词表改造。中文被切成字节碎片，`你好` 要 4 个 token。这项改动需要重训 embedding，属于主干该干的活，不在模块层面能修。
- 没做停用判据。训练按 epoch 停，不按"loss 收敛"停，这是当时的缺陷。
- 没做多轮对话。当前只覆盖单轮问答。

下一版的方向写在 `eval/RESULT.md`。

---

## License

Apache-2.0
