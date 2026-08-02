# Task_N 美甲 v5 合规 272 组数据 — pi05 base SFT 训练计划

> **建立**: 2026-08-01  
> **状态**: 设计定档，待数据构建、smoke 与提交  
> **第一任务**: 用 TOS 清洗后保留的 272 个 base episode 训练一个可真机测试的 Task_N 美甲 pi0.5 基线  
> **资源建议**: 北京 Robot-North-H20，1 节点 × 8 H20  
> **范围**: 本阶段只做 base SFT；不使用 dagger、AWBC、depth、mid-head 或 EEF 辅助损失  
> **参考**: [`pi05_task_a1_awbc_gripper_adapt_plan.md`](pi05_task_a1_awbc_gripper_adapt_plan.md)

---

## 0. 执行摘要

这批 v5 数据不能直接套用旧 Task_N 或 Task_A1 的训练集：原始 state/action 是 32 维，包含 14 维双臂关节/夹爪和 18 维 EEF-9D；两个采集站的相机数量也不同，且多站点元数据明确标记为 `standard_lerobot_compatible=false`、训练前必须全局重编号。

第一版采用风险最低、与现有 Agilex 真机部署链完全一致的方案：

1. 将 272 个合规 episode 构建成统一 LeRobot 数据集；
2. 只保留所有采集站共有的 `top_head/hand_left/hand_right` 三路 RGB；
3. 将 state/action 从原始 32 维裁为前 14 维 joint + gripper；
4. 按来源 episode 做 240 train / 32 val 的确定性分层划分；
5. 在 train 上重新计算 norm stats；
6. 从官方 `pi05_base` 初始化，8 H20、batch 128、训练 40k step；
7. 用离线 val 和真机美甲成功率选择 checkpoint。

Task_A1 可复用的是训练日程和工程门禁，不复用其 AWBC 数据语义或叠衣 checkpoint。

---

## 1. 目标与非目标

### 1.1 目标

- 建立 Task_N 新清洗数据的纯 base SFT 基线；
- 验证 272 组数据能否学会美甲任务的基本闭环行为；
- 找到 10k/20k/30k/40k 中适合真机的 checkpoint；
- 为后续扩充 base、加入 dagger/AWBC、使用 mid-head/EEF 提供可信基线。

### 1.2 本阶段不做

- 不把已隔离的 168 个不合规 episode 放回训练；
- 不使用 dagger，不构造 positive/negative prompt，不做 AWBC；
- 不下载或使用 depth；
- 不直接使用 `mid_head`，因为 `visrobot02/chunk-002` 没有该相机；
- 不把原始 18 维 EEF-9D 当作动作监督；
- 不从 Task_A1 叠衣 AWBC checkpoint warm-start，避免任务迁移成为混杂变量。

---

## 2. 数据事实与训练口径

### 2.1 当前 TOS 对齐快照

本地源目录：

```text
kai0/data/Task_N/base/v5/
├── 2026-07-29-v5
├── 2026-07-30-v5
└── 2026-07-31-v5
```

| 日期 | 合规 episode | 帧数 | RGB 视频 | 采集站 |
|---|---:|---:|---:|---|
| 2026-07-29 | 17 | 23,038 | 68 | ipc01 |
| 2026-07-30 | 39 | 42,702 | 153 | ipc01 36 + visrobot02 3 |
| 2026-07-31 | 216 | 183,720 | 798 | ipc01 150 + visrobot02 66 |
| **合计** | **272** | **249,460** | **1,019** | ipc01 203 + visrobot02 69 |

以上规模以 `meta/multistation.json` 和磁盘实际文件为准。07-30 根目录的旧 `info.json` 仍声明清洗前 201 episode，不能作为构建依据。

TOS 删除的不合规数据已从训练源移到可恢复隔离区：

```text
.runtime/task_n_tos_quarantine/20260801_0854Z/
```

训练构建脚本必须只扫描 `kai0/data/Task_N/base/v5`，不能扫描隔离区。

### 2.2 原始模态

原始 state/action 均为 32 维：

| 维度 | 含义 | 第一版处理 |
|---|---|---|
| 0:6 | left joint position | 保留 |
| 6 | left gripper | 保留 |
| 7:13 | right joint position | 保留 |
| 13 | right gripper | 保留 |
| 14:23 | left EEF-9D | 丢弃，留作后续辅助模态实验 |
| 23:32 | right EEF-9D | 丢弃，留作后续辅助模态实验 |

原始 RGB：

- `ipc01/chunk-000`：top-head、mid-head、left-hand、right-hand 四路；
- `visrobot02/chunk-002`：top-head、left-hand、right-hand 三路；
- 第一版取三路交集，适配现有 `LerobotAgilexDataConfig` 和 `AgilexInputs`。

### 2.3 为什么第一版使用 joint-14

- 当前真机 `AgilexOutputs(action_kind="joint")` 只执行模型输出前 14 维；
- Task_A/Task_A1 已验证的部署协议也是 14 维 joint + gripper；
- 若同时监督 EEF-9D，模型会在不会被真机执行的 18 个维度上消耗动作头容量，并引入单位、坐标系和旋转表示风险；
- pi0.5 模型内部仍保持 `action_dim=32`，14 维数据由现有 transform pad 到 32，模型结构与 `pi05_base` checkpoint 兼容。

EEF-9D 可在基线成立后作为单独的受控辅助任务实验，不能混入第一版。

---

## 3. Phase 0：冻结数据快照与质量门禁

在构建前记录 TOS 非 depth 对象清单、对象数和同步时间。训练期间不允许源数据静默变化；若 TOS 再清洗，必须生成新的 dataset version，而不是原地修改已开始训练的数据集。

逐 episode 检查：

- parquet 可读，state/action 长度一致且均为 `[T,32]`；
- state/action 无 NaN/Inf；
- `frame_index` 从 0 连续，timestamp 单调且接近 30Hz；
- parquet 帧数与三路 RGB 视频帧数一致；
- RGB 可从随机中间帧 seek 解码；若关键帧异常，则统一 libx264 重编码；
- 关节值、夹爪值范围和方向与真机 joint 控制协议一致；
- 报告长静止段，但不按臂速自动删除。美甲包含必要的精细接触与保持，速度门控可能误杀有效动作；只有人工确认的卡死 episode 才能剔除。

门禁输出：

```text
docs/training/analysis/task_n_v5_272_preflight.json
```

最低要求：272/272 episode 均能匹配 parquet + 所需三路 RGB；任何缺失都先修数据，不能由 dataloader 静默 skip。

---

## 4. Phase 1：构建标准训练集

建议新增构建脚本：

```text
train_scripts/kai/data/build_task_n_v5_272_joint14.py
```

输出：

```text
kai0/data/Task_N/self_built/
├── nail_v5_272_joint14_train
└── nail_v5_272_joint14_val
```

### 4.1 转换规则

1. 以 `(date, source_chunk, source_episode_id)` 作为源唯一标识；
2. 按三日期、两采集站发现 272 个实际存在的 parquet；
3. state/action 分别取 `[..., :14]`；
4. 只保留三路公共 RGB；
5. prompt 固定为 `"nail painting"`，`task_index=0`；
6. 全局连续重编号 `episode_index/frame_index/index`；
7. 重建 `info.json/episodes.jsonl/episodes_stats.jsonl/tasks.jsonl`，不复制源端过期 meta；
8. 在输出 meta 中记录每个新 episode 对应的 date、station、source_chunk 和 source_episode_id；
9. train/val 视频使用软链的前提是随机 seek 验证通过，否则统一重编码。

### 4.2 Train/val 划分

第一任务采用 **240 train / 32 val**，按日期和采集站分层，并固定 seed 与源 episode 清单：

- val 必须同时包含 ipc01 与 visrobot02；
- 不允许按帧随机切分，同一个 episode 不能跨 train/val；
- 优先在每个 `(date, station)` 内按末尾时间段留出，测量时间漂移泛化；
- 将最终清单落盘为 `split_manifest.json`。

若某站点 metadata 中没有可靠采集时间，则使用固定 seed 的 episode 级分层抽样，并在 manifest 中记录 seed。

### 4.3 Norm stats

- 只在 240 个 train episode 上重算；
- 输入数据为 joint-14，统计工具按模型 `action_dim=32` pad；
- val 不参与统计；
- 不复用旧 Task_N 497 ep、Task_A1 或 `pi05_base` 的 norm stats；
- 单独报告 gripper dims 6/13 的 q01/q50/q99，确认开合方向无误。

### 4.4 构建验收

- train=240、val=32，总数严格为 272；
- 输出 state/action shape 均为 14；
- 每个 episode 三路 RGB 齐全且帧数一致；
- `get_config()` 可加载；
- dataloader 连续抽样至少 1,000 batch item，skip=0；
- norm stats 有限且 gripper 非退化常数。

---

## 5. Phase 2：训练配置

新增配置名建议：

```text
pi05_task_n_v5_272_sft
```

checkpoint：

```text
kai0/checkpoints/pi05_task_n_v5_272_sft/nail_v5_272_sft/
```

### 5.1 主训练参数

| 参数 | 第一任务配置 | 与 Task_A1 的关系 |
|---|---|---|
| 模型 | `Pi0Config(pi05=True)`，无 DCT | 相同骨架 |
| init | `pi05_base/params` | 不复用 Task_A1 叠衣 AWBC ckpt |
| action target | joint-14，模型内部 pad 到 32 | 与 Task_A1 真机 joint 协议一致 |
| action horizon | 50 | 与 Task_A1 一致 |
| delta joint actions | `False` | 与 Task_A1 一致 |
| steps | **40,000** | 复用 Task_A1 日程 |
| peak LR | **1.0e-5** | 复用 Task_A1 的保守 LR |
| warmup | **500** | 复用 Task_A1 |
| decay | cosine，40k 时到 **1.0e-6** | 复用 Task_A1 |
| EMA | **0.9999** | 复用 Task_A1 |
| batch size | **128** | 数据较小且用单机 8 H20，不照搬 A1 的 16卡 bs256 |
| FSDP | **8 devices** | 北京单节点 8 H20 |
| workers | 8，smoke 后按吞吐最多调至 16 | 以稳定为先 |
| save interval | **2,000** | 与 Task_A1 一致 |
| keep period | **10,000** | 与 Task_A1 一致 |
| inline eval | 每 2,000 step；固定 val 帧 | 必须启用 |

40k × batch128 约产生 5.12M frame-sample，约为 249k 原始帧的 20 倍采样量，存在过拟合风险。因此训练可以跑满 40k，但模型选择不能默认取最后一步，必须比较中间 checkpoint。

### 5.2 为什么不用 Task_A1 checkpoint

Task_A1 的 `pi05_v4_awbc/49999` 已带叠衣动作分布和 advantage prompt 语义。Task_N 是新任务且本阶段只做普通 base SFT；从它 warm-start 会同时改变任务先验与训练语义，无法判断 272 组美甲数据本身是否有效。第一版用官方 `pi05_base` 才是干净基线。

如果主训练在 5k 后明显欠拟合，可另开受控 B 组将 peak LR 提到 `1.5e-5`；不能在主任务中途原地改 LR。

---

## 6. Phase 3：北京环境与提交门禁

建议任务：北京 Robot-North-H20，1×8 H20。实际提交不属于本文档动作。

### 6.1 提交前检查

| 门禁 | 判据 |
|---|---|
| 数据同步 | train/val 已同步到 North-E，文件数与 gf0 一致 |
| init | `/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params/_METADATA` 存在 |
| config | `get_config('pi05_task_n_v5_272_sft')` 成功 |
| norm | train 根目录 `norm_stats.json` 存在，shape/range 正常 |
| 视频 | 三路 RGB 链接不断，随机 seek 解码通过 |
| schema | state/action 14 维；prompt=`nail painting` |
| 环境 | JAX 识别 8 H20，离线模型资源齐全 |
| smoke | 3 step 运行成功，loss 非 NaN，checkpoint 成功落盘 |

### 6.2 Smoke

- 8 H20、batch 64 或 128、3 step；
- 至少跑过一个 inline-eval batch；
- 检查日志无 dataloader skip、video seek error、shape mismatch；
- 检查初始化确实来自 `pi05_base`，不是旧 Task_N 或 Task_A1 checkpoint。

只有全部门禁通过后才提交 40k 正式任务。

---

## 7. 评估与 checkpoint 选择

### 7.1 离线

对 10k、20k、30k、40k checkpoint 统一评估：

- val flow-matching loss；
- action MAE@1/10/25/50；
- 左右臂关节 MAE；
- 左右夹爪 dims 6/13 MAE 与开合事件准确率；
- ipc01 与 visrobot02 分站点指标；
- train-val gap，判断 272 组数据下的过拟合起点。

离线指标只用于排除坏 checkpoint，不能替代真机成功率。

### 7.2 真机

先对 10k/20k/30k/40k 做小规模筛选，再对最佳 checkpoint 扩大测试。记录：

- 美甲完整任务成功率（主指标）；
- 目标定位与工具接近成功率；
- 接触/涂抹阶段成功率和结果精度；
- 左右夹爪误开、误闭次数；
- 碰撞、越界和安全停止次数；
- 冻结/长时间无效动作比例；
- 单次任务完成时间。

真机测试必须固定初始摆放范围、任务说明、部署 RTC/EMA 参数和重试规则，否则 checkpoint 间不可比。

### 7.3 第一任务完成标准

- 训练与 val 全程无 NaN、无数据 skip；
- 至少一个 checkpoint 在真机上形成稳定、可重复的完整行为链；
- 若完整成功率仍低，也必须能定位主要失败阶段，并据此决定补 base 还是采 dagger；
- 将最终模型、数据 manifest、config、提交 YAML、日志与真机结果一起归档。

---

## 8. 后续探索顺序

仅在 v5-272 base 基线完成后推进：

1. **数据量实验**：v5-272 vs 旧 v2-497，统一训练配方比较数据清洗收益；
2. **四相机实验**：只用 ipc01 数据加入 mid-head，与三相机基线做受控对比；
3. **EEF 辅助实验**：保留 joint-14 主动作，EEF-9D 只作辅助表示/损失，不直接作为真机输出；
4. **扩充 base**：优先补当前真机失败阶段的专家示范；
5. **dagger**：采集失败态人工接管，逐批检查 intervention、长静止与视频对齐；
6. **AWBC**：有合格 dagger 后再使用 `positive ⟺ 人控`，不得用臂速替代干预标签；
7. **部署调优**：最后再调整 RTC/EMA；数据导致的巡航慢与 RTC 削峰必须分开分析。

---

## 9. 风险与停止条件

| 风险 | 影响 | 缓解/停止条件 |
|---|---|---|
| 源 meta 与清洗后文件不一致 | episode 漏读或索引冲突 | 完全重建 meta，以实际文件和 multistation manifest 为准 |
| 32维直接训练 | 非执行 EEF 维度干扰 joint 动作 | 第一版强制裁为前14维 |
| 相机 schema 不一致 | dataloader shape/key 错误 | 第一版只取三路交集 |
| 视频 seek/关键帧异常 | 大量静默 skip | 随机 seek 门禁；异常则统一重编码 |
| 272 ep 过拟合 | 后期 loss 好但真机变差 | 保存 2k checkpoint，按 val+真机选点，不默认 40k |
| 精细动作被静止裁剪 | 接触/保持行为丢失 | 禁止按速度自动裁帧 |
| TOS 再次删除数据 | 训练不可复现 | 冻结 snapshot 与 split manifest；变更生成新版本 |
| 夹爪约定错误 | 真机反向开合或抓取失败 | 构建前核对 dims 6/13 的范围、方向和部署映射 |

任一数据完整性门禁失败时停止训练准备；3-step smoke 失败时不得提交正式任务。

---

## 10. 执行清单

- [ ] 冻结 TOS/local v5-272 非 depth manifest
- [ ] 生成 272 episode 质量报告
- [ ] 实现 joint-14、三相机、全局重编号构建脚本
- [ ] 固定 240/32 分层 split manifest
- [ ] 重建 train/val meta 与 per-episode stats
- [ ] 随机 seek 验证，必要时重编码视频
- [ ] 计算 train-only norm stats
- [ ] 新增 `pi05_task_n_v5_272_sft` config
- [ ] gf0 数据加载 smoke
- [ ] 同步 train/val、config、init 到 North-E
- [ ] 北京 8 H20 3-step smoke
- [ ] 提交 40k 正式训练
- [ ] 评估 10k/20k/30k/40k
- [ ] 真机 Task_N 美甲测试并归档结论
