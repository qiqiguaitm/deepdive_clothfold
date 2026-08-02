# Task_N 美甲(nail painting)— pi05 base SFT 基线 plan

> **建立**: 2026-07-28 · **状态**: 📋 设计定档, 待提交(提交由用户执行, 本文档只负责规划) · **资源**: 北京 Robot-North-H20 队列, **8卡 (1×8 H20)**
> **一句话**: 全新任务**美甲**。第一步只用 **base 数据做纯 SFT**(不 dagger、不 AWBC), 走标准 pi05 流程, init `pi05_base`, 训 **40000 步**, 建立可用基线; dagger/AWBC 留作后续探索(§5)。
> **关联**: 同硬件叠衣线 [`pi05_task_a1_awbc_gripper_adapt_plan`](pi05_task_a1_awbc_gripper_adapt_plan.md) · AWBC 三范式 [`awbc_three_paradigm_comparison_plan`](awbc_three_paradigm_comparison_plan.md)

---

## 1. 背景与目标

- **新任务 = 美甲(nail painting)**, 与叠衣(Task_A)是**不同任务**, 但**同一硬件本体**(Agilex 双臂)。
- **第一步策略**: 只有 base(专家示范)数据 → **纯 base SFT 建基线**, 先不引入 dagger/AWBC(那是数据/信号成熟后的下一阶段, §5)。这符合叠衣线当初的推进顺序(先 SFT plateau, 再 dagger, 再 AWBC)。
- **目标**: 产出一个美甲任务可跑的 pi05 base 模型; 主判据 = 真机能否完成美甲基本流程(精细操作成功率)。离线 MAE 仅作收敛 sanity。

## 2. 数据规格(已核实)

| 项 | 值 |
|---|---|
| **本体** | `agilex` 双臂 —— **与 Task_A 同 embodiment** → 现成 `LerobotAgilexDataConfig` 直接可用 |
| **维度** | state/action **14 维**(6臂+1夹爪 ×2), 与 Task_A 一致 |
| **相机** | 3 路: `top_head` / `hand_left` / `hand_right`(480×640, 30fps) |
| **规模** | **497 ep / 538,188 帧 / 30fps / 1 chunk**(单日期 07-20-v2, operator ZW; episodes.jsonl 带 `success` 标) |
| **prompt** | `"nail painting"`(tasks.jsonl `task_index=0`) |
| **列** | 标准 LeRobot v2.1(state/action/3cam/index/task_index), 无 dagger/advantage 列(base-only 正合本阶段) |
| **同步** | ✅ 已同步 gf0 + TOS(`KAI0/Task_N`)+ North-E(`.../kai0/data/Task_N`, 北京训练读此路径) |

## 3. 方法(纯 base SFT)

### 3.0 北京环境检查(提交前门禁, 必须全绿)
| # | 检查 | 判据 | 阻塞 |
|---|---|---|---|
| E1 | North-E venv + jax | `source $NE/.venv/bin/activate; python -c "import jax"` | 是 |
| E2 | config get_config | `get_config('pi05_task_n_nail_sft')`(建 config 后) | 是 |
| E3 | **init `pi05_base` 在 North-E** | ✅ 已核实: `/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params`(**注意不在 deepdive_kai0/kai0/checkpoints 下**, 与 gf0 路径不同) | 是 |
| E4 | 数据 + norm_stats 在 North-E | ✅ 数据已在(497pq/1491mp4/2.7G); ⚠️ **norm_stats 待算**(`compute_norm_states_fast`, §3.1) | 是 |
| E5 | 队列容量 | Robot-North-H20 有 8卡(1×8 H20)空档 | 排期 |
| E6 | 3步 smoke | North-E `num_train_steps=3` 跑通(loss 非 NaN, ckpt 落盘) | 是 |

### 3.1 数据准备
- **base-only, 无需打标/合并** —— Task_N/base 本身就是干净 LeRobot 集, 直接训。
- **norm_stats 在 Task_N 上重算**(`compute_norm_states_fast --config-name pi05_task_n_nail_sft`)—— 美甲的 state/action 分布与叠衣不同, 必须重算。
- **val 留出**: 从 497 ep 留出 ~10-20 ep 作 inline-eval val(建议按 operator/时间留末尾, 或随机固定 seed)。
- (可选)据 `episodes.jsonl.success` 过滤失败 ep —— base 若全 success 则不需; 有失败则只留 success。

### 3.2 训练配置
新 config `pi05_task_n_nail_sft`(标准 pi05 SFT, 参 `pi05_v4_awbc` 的 from-base 骨架但去掉 AWBC):
- **model**: `Pi0Config(pi05=True)` 无 DCT。
- **init = `pi05_base/params`**(标准新任务 SFT 冷启动; 非 warm-start)。北京训练用 North-E 路径 `/vePFS-North-E/vis_robot/base_init_ckpts/extracted/pi05_base/params`(gf0 上是 `kai0/checkpoints/pi05_base/params`)。
- **data**: `LerobotAgilexDataConfig(repo_id=North-E Task_N/base, use_delta_joint_actions=False)`。
- **prompt**: `default_prompt="nail painting"`(单任务, 直接 default; 或 `prompt_from_task=True` 走 tasks.jsonl, 二者等价)。
- **超参(定档)**: **num_train_steps 40000** · peak_lr 1.5e-5 → decay 1.5e-6 · warmup 1000 · ema 0.9999 · **bs128 / fsdp8**(8卡)· save_interval 2000 · keep_period 10000。
- **inline_eval**: Task_N val, MAE@{1,10,25,50} 每若干 k 步。

### 3.3 提交(北京 8 卡, 由用户执行)
> 本文档只做规划; build/提交由用户或其他会话执行, 本会话不 build 不提交。
- YAML 基于 `train_scripts/kai/volc/pi05_v4_awbc_3para_*_cnbj_8gpu.yaml`(已验证 8卡多机 JAX entrypoint), 改 TaskName/Config/preflight(检查 Task_N/base + norm_stats + init pi05_base 在 North-E)。
- 提交: `submit_yaml.py`(经 gsy, VOLC_REGION=cn-beijing, Robot-North-H20)。
- **门禁 = §3.0 全绿 + 3步 smoke**(train_scripts/CLAUDE.md)。

## 4. 评估
- **T1 离线**: Task_N val MAE@{1,10,25,50} —— 看收敛 + plateau(定后续是否加数据/dagger)。
- **T3 真机(决定性)**: 美甲基本流程完成率 / 精度(美甲=精细操作, 对定位精度要求高, 真机为准)。

## 5. 后续探索路线(base SFT plateau 后)
按叠衣线已验证的推进顺序, 逐步加信号:
1. **扩 base + 多日期/多 operator** → SFT plateau, 定数据天花板。
2. **采 dagger**(美甲失败态人工接管纠错)→ dagger SFT(先纯加数据), **每批先跑静止/干预检查**(参 Task_A1 §2.5 教训: 防全自主卡死批次混入)。
3. **AWBC**: dagger 打标 `positive⟺人控`(不用臂速门控, 保精细操作)→ advantage 条件化 / 加权(参 AWBC 三范式 plan)。
4. **真机闭环**评估 + RTC/EMA 速度调优(参 `deploy_speed_analysis`)。

## 6. 风险
- **R1 单日期/量偏小**(497 ep 单 operator 单日)→ 可能过拟合 + 泛化窄。缓解: val 早停 + 后续扩数据(§5-1)。
- **R2 美甲=高精度任务**: 定位/接触精度要求远高于叠衣, base SFT 可能精度不足 → 早期就要真机看精度, 不足则优先扩数据/dagger 而非调超参。
- **R3 norm_stats 必重算**: 美甲 state/action 分布 ≠ 叠衣, 复用旧 norm 会错。
- **R4 数据可能仍在采集**: 若 07-20 后有新批次, 训练用的是当前快照; 先出基线, 定稿后再训正式版。
- **R5 gripper/action 约定**: 沿用 Agilex v4 约定(action≠state gripper-from-master)—— 与 Task_A 同本体应一致, 但真机前离线核对 gripper 列取值范围。

## 7. 关联
- 同本体叠衣: [`pi05_task_a1_awbc_gripper_adapt_plan`](pi05_task_a1_awbc_gripper_adapt_plan.md) · 数据同步: Task_N 已在 gf0/TOS/North-E · init: `pi05_base`
