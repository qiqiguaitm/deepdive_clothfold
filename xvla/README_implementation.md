# KAI0 X-VLA 实现说明

本文说明本仓库实际使用的 X-VLA 与 `xvla/X-VLA` 官方 GitHub 仓库之间的关系，重点记录
Task A1 叠衣模型的真实训练入口、数据与动作表示、连续夹爪改造、AWBC 条件化方式，以及
checkpoint/部署兼容性。

## 一句话结论

当前模型不是直接由官方 `X-VLA/train.py` 训练得到，而是：

```text
2toINF 官方 X-VLA 架构
        ↓
LeRobot 0.4.4 的 XVLAPolicy/XVLAModel 实现
        ↓
KAI0 自定义数据、训练器、连续夹爪、AWBC prompt 和部署 wrapper
        ↓
本仓库实际训练与部署的 checkpoint
```

模型的 Florence2、SoftPromptedTransformer、domain-aware layer、soft prompt 和 flow-matching
主体与官方 X-VLA 基本一致；差异主要集中在模型包装、数据管线、初始化方式、夹爪 loss、
训练调度、checkpoint 格式和真机部署。

## 代码来源与真实入口

### 官方参考实现

- 路径：[`X-VLA/`](X-VLA/)
- 上游：<https://github.com/2toinf/X-VLA.git>
- 当前 pin：`ccd1992f3ecce554e3ebe68e21c759acf111f2b0`
- 官方全参训练入口：[`X-VLA/train.py`](X-VLA/train.py)
- 官方 LoRA 训练入口：[`X-VLA/peft_train.py`](X-VLA/peft_train.py)

### KAI0 实际实现

- 主训练器：[`launch/xvla_train.py`](launch/xvla_train.py)
- A1 启动脚本：[`xvla_a1_local_gf0.sh`](xvla_a1_local_gf0.sh)
- 数据加载器：[`data/multi_domain_dataset.py`](data/multi_domain_dataset.py)
- joint→EE6D 转换：[`data/joint_to_ee6d.py`](data/joint_to_ee6d.py)
- 推理服务：[`serve/serve_policy_xvla.py`](serve/serve_policy_xvla.py)

训练和部署实际导入：

```python
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
```

当前 `.venv_xvla` 安装的是 LeRobot `0.4.4`。A1 没有使用 `peft_train.py`，也没有使用
LoRA；前 1,000 step 只训练 soft prompt/action head，之后进入全参数微调。

## 官方训练与 A1 训练对照

| 项目 | 官方 `X-VLA/train.py` | A1 实际训练 |
|---|---|---|
| 模型类 | `models.modeling_xvla.XVLA` | LeRobot `XVLAPolicy` |
| 数学主体 | Florence2 + SoftPromptedTransformer | 基本一致 |
| 分布式框架 | HuggingFace Accelerate | PyTorch DDP (`torchrun`) |
| 数据入口 | meta JSON + domain handler | LeRobot parquet + MP4 |
| 初始权重 | 通常 `2toINF/X-VLA-Pt` | E0 fixed-camera 叠衣 checkpoint |
| 动作 | 官方 action space | 20D `ee6d_alpha` |
| 夹爪 | 二值 BCE + sigmoid | 连续 alpha + MSE + clamp |
| AWBC | 无 | positive/negative prompt 条件化 |
| checkpoint | HF safetensors + processor | `state_dict.pt` + config/sidecar |
| 保存周期 | 默认 50k | 每 2k + `step_final` |

## 模型主体保持一致的部分

两套实现都包含：

- Florence2 encoder-only 视觉语言骨干；
- 24 层 SoftPromptedTransformer；
- domain-aware 输入投影、action encoder 和 action decoder；
- 每个 domain 独立的 soft prompt；
- 观测图像、语言、proprio 和 noisy action 的联合建模；
- `x_t = t·noise + (1-t)·action` 的 x0-prediction flow-matching；
- 迭代去噪生成 action chunk；
- 双臂 20D EE6D 动作布局。

官方 `models/transformer.py` 与 LeRobot `soft_transformer.py` 的核心计算基本相同，主要区别是
LeRobot 增加了 `PreTrainedPolicy` 包装、feature contract、batch 整理和 checkpoint 接口。

## A1 数据与动作表示

A1 原始数据每帧是 14D：

```text
left  = joint[0:6]  + gripper[6]
right = joint[7:13] + gripper[13]
```

训练前通过 Piper FK 转成双臂 20D EE6D：

```text
每臂 10D = xyz(3, m) + rotation6d(6) + gripper_alpha(1)
双臂 20D = left(10) + right(10)
```

其中 rotation6d 使用旋转矩阵前两列的 row-major 排布：

```text
[r00, r01, r10, r11, r20, r21]
```

A1 保留真实的控制语义：

- `observation.state`：follower 机械臂实测状态；
- `action`：leader 遥操作命令；
- 不把 action 强制改写成 state；
- 30 个 future anchor 均匀覆盖未来 2 秒；
- 丢弃未来 EE 位姿几乎不动的退化训练帧。

该设计避免把“复述当前 proprio”当成正确动作，降低模型走 `action≈state` 开环捷径的风险。

## 图像与语言预处理

相机顺序固定为：

```text
top_head → hand_right → hand_left
```

图像管线包括：

1. 保持宽高比 resize + 黑边 padding；
2. 训练时 ColorJitter（brightness/contrast/saturation = 0.2）；
3. ImageNet mean/std normalization；
4. 进入 LeRobot policy 后统一 resize/pad 到 224×224；
5. 视频路径或解码失败时显式报警，避免静默用全黑图训练。

语言使用本地 BART tokenizer，padding/truncation 长度为 50。若数据配置
`prompt_from_task=True`，每帧根据 parquet 中的 `task_index` 从 `meta/tasks.jsonl` 读取真实 prompt。

## 连续夹爪改造

### 官方 `ee6d`

官方 EE6D action space 的夹爪逻辑是：

- 目标为 `{0,1}`；
- 使用 `BCEWithLogitsLoss`；
- preprocessing 将 proprio/noisy-action 中的夹爪通道清零；
- 推理时对夹爪 logit 做 sigmoid。

这不适合 A1 的细长夹爪连续位置控制。

### A1 `ee6d_alpha`

A1 使用完整物理行程的连续归一化：

```text
alpha = clip((open_m - gripper_m) / (open_m - close_m), 0, 1)

open_m  = 0.07
close_m = 0.00

alpha=0 → 0.07m fully open
alpha=1 → 0.00m fully/force close
```

训练和推理语义：

- 不二值化；
- 保留 proprio 中的当前夹爪开度；
- 保留 noisy action 中的连续夹爪值；
- 直接 `MSE(pred_alpha, target_alpha)`；
- `GRIPPER_SCALE=100`；
- 不在 loss 或部署侧重复 sigmoid；
- 推理后仅 clamp 到 `[0,1]`；
- 部署线性逆映射回 `[0.07m, 0m]`。

### 夹爪输出头重置

E0 warm-start checkpoint 的夹爪头来自二值 BCE-logit 训练，不能直接作为 alpha 回归头。
A1 加载 E0 权重后只重置 action decoder 的左右夹爪输出列（index 9、19）：

- weight：小随机初始化，标准差 0.02；
- bias：初始化为 0.75；
- 其余叠衣视觉、语言、soft prompt 和 EE 运动能力继续复用。

## Warm-start 与 domain 设计

A1 不是从原始 foundation checkpoint 直接开始，而是：

```text
X-VLA base
  → E0_v1_official_fixedcam 叠衣模型（50k）
  → 加载完整 state_dict
  → 保留 folding domain_id=20
  → 重置连续夹爪输出列
  → A1 再训练 50k
```

继续使用 `domain_id=20` 是为了复用已经学到的双臂叠衣 soft prompt/action head。A1 的硬件差异由
真实 proprio/action 和连续夹爪值表达，不通过新建 domain 丢弃已有叠衣能力。

## AWBC 的准确含义

A1 数据中的 advantage 被离散成语言条件：

```text
Flatten and fold the cloth. Advantage: negative
Flatten and fold the cloth. Advantage: positive
```

训练时按每帧 `task_index` 选择对应 prompt；部署时固定使用 positive prompt。

当前实现是 **advantage-conditioned behavior cloning**，不是严格意义上的 weighted BC：

- 没有使用 advantage scalar 直接乘 loss；
- 没有 AWR/AWAC/CRR 指数权重；
- 没有按 advantage 修改采样概率；
- loss 仍是标准 X-VLA flow-matching loss。

因此推理 prompt 必须与训练字节级一致：

```text
Flatten and fold the cloth. Advantage: positive
```

## A1 实际训练配置

| 参数 | 值 |
|---|---:|
| steps | 50,000 |
| GPU | 2×A100 |
| batch/GPU | 16 |
| effective batch | 32 |
| base LR | `1e-4` |
| VLM/soft-prompt LR scale | `0.1` |
| warmup | 2,000 steps |
| backbone freeze | 前 1,000 steps |
| weight decay | `0.0` |
| AdamW betas | `(0.9, 0.95)` |
| grad clip | `1.0` |
| precision | BF16 autocast |
| action chunk | 30 |
| action horizon | 2.0 s |
| image augmentation | ColorJitter + ImageNet norm |
| action mode | `ee6d_alpha` |

前 1,000 step 冻结 VLM 和 transformer core，只更新 soft prompt/action heads；之后解除冻结并进行
全参数微调。

### 与官方 warmup 的细微差异

官方 README 传入 `warmup_steps=2000`，但官方 `train.py` 在默认
`use_cosine_decay=False` 时会直接设置 base LR，代码路径中 warmup 实际不生效。

KAI0 trainer 使用 `get_constant_schedule_with_warmup`，会真实执行 2,000-step 线性 warmup。
因此即使表面超参相同，两条训练轨迹也不是逐 step 数值一致。

## 启动方式

准备数据、做 smoke test、完整训练：

```bash
./xvla/xvla_a1_local_gf0.sh prepare
./xvla/xvla_a1_local_gf0.sh smoke
./xvla/xvla_a1_local_gf0.sh full
```

脚本会：

1. 将原始 14D joint 数据转换为 20D EE6D；
2. 使用 `open=0.07m, close=0m` 的连续夹爪 alpha；
3. 校验转换后的 state/action；
4. 从 E0 fixed-camera checkpoint warm-start；
5. 使用 2×GPU DDP 启动 `A1_local_awbc`。

## Checkpoint 与部署兼容性

KAI0 trainer 保存：

```text
step_002000/state_dict.pt
step_004000/state_dict.pt
...
step_final/state_dict.pt
```

文件内容：

```python
{
    "model_state": XVLAPolicy.state_dict(),
    "step": 50000,
}
```

它不是官方 HuggingFace `save_pretrained()` 目录，不能直接改用官方 `deploy.py` 或
`AutoModel.from_pretrained()`。部署必须同时保留：

- `state_dict.pt`；
- LeRobot base `config.json`/训练配置；
- KAI0 `sidecar.json`；
- 与训练一致的 tokenizer；
- 与训练一致的 domain、prompt、相机顺序、图像 normalization 和夹爪映射。

A1 当前部署关键值：

```text
domain_id=20
prompt="Flatten and fold the cloth. Advantage: positive"
action_mode=ee6d_alpha
binarize_gripper=false
gripper_open_value=0.07
gripper_close_value=0.00
```

## 已知风险与维护约束

### 1. 官方 submodule 当前存在本地修改

虽然 `X-VLA/` 的设计目标是保持 upstream pristine，但当前工作树实际为：

```text
M models/action_hub.py
```

相对 pin 的官方 commit，本地增加了约 129 行，主要是：

- `ee6d_continuous`；
- `ee6d_alpha`。

不要误把当前 `X-VLA/models/action_hub.py` 当成纯官方文件，也不要在不迁移这些 action mode
之前清理 submodule 工作树。

### 2. 连续夹爪实现位于虚拟环境 site-packages

A1 实际训练导入的是 `.venv_xvla/.../site-packages/lerobot/policies/xvla/action_hub.py`，其中也有
上述连续夹爪类。重建 venv 或重新安装干净的 LeRobot 时，这些本地改动可能消失。

在环境重建前，应先将自定义 action space 迁入版本控制或提供可重复应用的 patch，并运行
`ee6d_alpha` 加载/前向 smoke test。

### 3. 多域 DDP 权重

当前单 GPU 使用 `WeightedRandomSampler`，但 DDP 路径使用 `DistributedSampler`，不会应用配置中的
per-dataset weight。A1 只有一个 dataset，训练结果不受影响；未来多域 DDP 实验需修复该问题。

### 4. 不可混用官方部署入口

官方模型、LeRobot policy 和 KAI0 `state_dict.pt` 的包装与保存格式不同。除非先完成显式转换和
权重键校验，否则不要用官方 `deploy.py` 加载 KAI0 checkpoint。

## 维护原则

1. 架构改动优先放在 `xvla/launch/`、`xvla/data/`、`xvla/serve/` 或可重复 patch 中；
2. 数据、训练和部署必须共用同一个 action layout 与夹爪边界；
3. checkpoint 必须随附 config/sidecar，禁止只复制裸权重后猜测参数；
4. AWBC 模型必须随附精确的部署 prompt；
5. 真机执行前先完成 `missing=0/unexpected=0` 加载、离线有限值、动作范围和视觉消融检查；
6. 连续夹爪模型禁止在 client/server 任一侧重新二值化或重复 sigmoid。
