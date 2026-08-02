# Task_A1 细长夹爪叠衣 — AWBC warm-start 适配训练 plan

> **建立**: 2026-07-24 · **更新**: 2026-07-25 · **状态**: 📋 设计定档, 待提交(提交由他人执行, 本文档只负责规划) · **资源**: 北京 Robot-North-H20 队列, **16卡 (2×8 H20)**
> **一句话**: 同一叠衣任务、换更细长的夹爪(Task_A1)。**不从 pi05_base 重训**, 而是在已训好的叠衣 AWBC 模型 `pi05_v4_awbc/49999` 基础上, 用 Task_A1 base + dagger(仅合格批次)做 AWBC **warm-start 微调 40000 步**, **验证新夹爪的作用**。
> **关联**: 数据用法/AWBC 三范式 [`awbc_three_paradigm_comparison_plan`](awbc_three_paradigm_comparison_plan.md) · dagger intervention 打标 `docs/training/analysis/chunk001_schema.md` · 速度门控杀抓取教训 [`dagger_launchpoint_trim_freeze_fix_plan`](dagger_launchpoint_trim_freeze_fix_plan.md)

---

## 1. 背景与目标

- **任务不变**: 叠衣(flatten & fold),SOP 同 Task_A。**唯一硬件变化 = 更细长的夹爪**。
- **为什么 warm-start 而非从头**: 折叠策略(接近/抓取/翻折/保持)已在 `pi05_v4_awbc` 学到; 换夹爪主要改**抓取几何 + 夹爪开合动力学**, 属**适配**不是重学 → warm-start 微调最省数据/算力, 且继承已验证的不冻/抓取行为。
- **目标**: 产出一个在细长夹爪硬件上叠衣可用的模型; 主判据 = 真机抓取成功率 + 折叠完成 + 不冻/不死循环。

## 2. 关键事实(已核实)

| 项 | 值 |
|---|---|
| **init ckpt** | `kai0/checkpoints/pi05_v4_awbc/pi05_v4_awbc/49999/params`(pi0.5 AWBC 叠衣, 50k, 已部署OK) |
| **数据** | `Task_A1/{base,dagger}/v4/<date>-v4/`(按日期; **仍在采集**, 对象数持续增长)。base 5天 07-20~24(428 pq)· dagger 2天: **07-23(17ep, 见 §2.5 丢弃) + 07-24(42ep, 用)** |
| **维度** | state/action **14 维**(6臂+1夹爪 ×2), 与 Task_A 一致 → **动作空间不变**, warm-start 结构兼容 |
| **夹爪** | dims 6,13; 细长夹爪 → 开合行程/值域与旧夹爪不同 → **norm_stats 必须重算** |
| **标签** | base parquet 裸列; dagger(chunk-001)带 `intervention`+`dagger_frame_class{0,1,2}` → **positive⟺人控** 可直接打标 |
| **同步** | ✅ 已同步 gf0 + North-E, 两处**逐项一致**(84527 文件/87G, Failed 0, 无嵌套, 抽样可读)。北京训练读 `/vePFS-North-E/.../kai0/data/Task_A1` |

## 2.5 ⚠️ dagger 数据检查发现(2026-07-24, 提交前必读)

对 Task_A1 dagger 逐 ep 做了静止/干预检查(臂速逐帧 + intervention/class):

| 批次 | ep | 人控帧% | class 分布 | 最长静止 | 结论 |
|---|---|---|---|---|---|
| **07-23** | 17 | **0.0%** | 全 class0 | **30.4s 纯卡死** | ⛔ **丢弃**: 17/17 全自主、零人工干预, 是"机器人卡住的录像", 无纠错信号 |
| **07-24** | 42 | **64.1%** | {robot 25629, **intv 47501**, preintv 924} | 14.6s(15/42 ep>2s) | ✅ **采用**: 真纠错 dagger, `positive⟺人控` 成立, class1 抓取保住 |

→ **采集流程在 07-23→07-24 之间被修复**。**决定: dagger 只用 07-24(及以后经检查合格的批次), 丢 07-23。**
→ **残留长静止**: 07-24 仍有 class0(机器人自主)段 >2s 静止(最长 14.6s)→ 训练前**裁掉 class0 的 >2s 连续静止段**, 防污染 norm_stats / 喂养冻结(stitch 的 {3,4} 只裁首尾, 不裁内部 class0 卡死)。
→ **后续每天新增 dagger 必须先跑此检查**, 别再混进 07-23 那种全自主批次。

## 3. 方法

### 3.0 北京训练环境检查(提交前门禁, 必须全绿)
提交 16 卡任务前, 在 North-E 逐项确认:

| # | 检查 | 命令/判据 | 阻塞? |
|---|---|---|---|
| E1 | venv + jax 可用 | `source $NE/.venv/bin/activate; python -c "import jax; print(jax.__version__)"` | 是 |
| E2 | config.py 能 get_config | `python -c "from openpi.training import config as c; c.get_config('pi05_a1_awbc')"`(建 config 后) | 是 |
| E3 | **init ckpt 在 North-E** | `$NE/checkpoints/pi05_v4_awbc/pi05_v4_awbc/49999/params/_METADATA` 存在; **无则从 gf0 同步该 ckpt**(warm-start 源必须在训练可达路径) | 是 |
| E4 | 训练数据在 North-E | `A1_base_dagger_awbc/{meta,norm_stats.json,data,videos}` 齐 + tasks.jsonl 含 advantage prompt | 是 |
| E5 | 队列容量 | Robot-North-H20 有 16卡(2节点×8 H20)空档; 查 07-21 的 3para 任务(`t-20260721200622/200628/200636`)是否已结束释放 | 影响排期 |
| E6 | GPU/多机 JAX | 沿用 3para 的 cnbj yaml entrypoint(已验证多机 fsdp 起得来) | 是 |
| E7 | 3步 smoke | North-E 上 `num_train_steps=3` 跑通(loss 非 NaN, ckpt 落盘) | 是 |

> ⚠️ 环境检查 + smoke 未全绿, **不得提交 40k×16卡任务**。

### 3.1 数据构建(`A1_base_dagger_awbc`)
合并 Task_A1 **base(全部 5 天)+ dagger(仅 07-24, 丢 07-23)** 成一个 LeRobot 集, 逐帧打 AWBC 标签:

- **打标规则(positive ⟺ 人在控制)** —— 已验证、天然保住抓取(不用臂速门控):
  | 来源 | task_index | 说明 |
  |---|---|---|
  | base(专家遥操示范, 428 pq) | **1 positive** | 全是想复现的好动作 |
  | dagger 07-24 `intervention=1`(人控纠错) | **1 positive** | 含抓取, 无论臂速 |
  | dagger 07-24 `intervention=0`(机器人自主) | **0 negative** | 策略自己的(可能失败)动作 |
- **裁剪**: dagger 07-24 里 **class0(机器人自主)的 >2s 连续静止段裁掉**(§2.5; 用 `dagger_frame_class`+臂速定位, 不裁 class1 人控)。base 不裁。
- tasks.jsonl: `{0:"Flatten and fold the cloth. Advantage: negative", 1:"...positive"}` —— **与 init 模型训练格式一致**(init 本就是 AWBC 模型)。
- **norm_stats 在 Task_A1 上重算**(compute_norm_states_fast)—— 细长夹爪值域变了, 必须重算; 不可复用 init 的 norm。
- 复用现成 build 惯例(参 `relabel_human_awbc.py`(task_index=intervention)/ `build_chunk001_dagger_crave_labeled.py`): 视频软链、重编号、per-ep stats。
- ⚠️ base 结构是**按日期** `base/v4/<date>/data/chunk-000`(非 kai0_base 多 chunk), dagger 是 `dagger/v4/2026-07-24-v4/data/chunk-001` → build 脚本需适配这个日期式布局(现成 build 是为 Task_A 的 kai0_base 写的, 不能直接套)。
- **构建位置**: 北京训练读 North-E, 故直接在 North-E 上 build(数据已在), 免跨集群传。

### 3.2 训练配置(warm-start 微调)
新 config `pi05_a1_awbc`(克隆 `pi05_v4_awbc`, 改动如下):
- **init = `pi05_v4_awbc/pi05_v4_awbc/49999/params`**(非 pi05_base)。⚠️ 该 ckpt 须在 North-E 就位, North-E 无则先同步(见 §3.0 环境检查)。
- repo_id = North-E 的 `A1_base_dagger_awbc`。prompt_from_task=True。pi0.5 无 DCT。
- **超参(定档)**: **num_train_steps 40000**(用户定, 验证新夹爪)· peak_lr **1e-5**(略降, 防冲掉已学折叠)· warmup 500 · decay to 1e-6 · ema 0.9999 · **bs256 / fsdp16**(16卡 2×8)。
- inline_eval: Task_A1 自建 val(留出最后一天若干 ep), MAE 仅 sanity。

### 3.3 提交(北京 16 卡, 由他人执行)
> 本文档只做规划; 实际提交/build 由用户或其他会话执行, 本会话不 build 不提交。
- YAML 基于 `train_scripts/kai/volc/pi05_v4_awbc_3para_*_cnbj_8gpu.yaml`, 改为 **16卡(RoleReplicas 2 × 8 H20, fsdp=16)**、TaskName/Config、preflight(检查 `A1_base_dagger_awbc` + norm_stats + advantage prompt + init params 在 North-E)。
- 提交: `submit_yaml.py`(经 gsy, VOLC_REGION=cn-beijing, Robot-North-H20)。
- **提交前门禁 = §3.0 环境检查全绿 + 本机/North-E 3步 smoke 通过**(train_scripts/CLAUDE.md 规范)。

## 4. 评估
- **T1 离线**: Task_A1 val MAE@{1,10,25,50}(sanity + 收敛; AWBC 不敏感)。**关键看早期是否因 norm 重算而先升后降**(warm-start × 新 norm 的重对齐期)。
- **T2 离线探针**: 抓取帧是否保住 positive(§dagger 口径)、pos/neg 行为差。
- **T3 真机(决定性)**: **细长夹爪抓取成功率**(新硬件重点)/ 折叠完成率 / throughput / 不冻不死循环。对照旧夹爪 `pi05_v4_awbc` 真机基线。

## 5. 已定 / 待定
**已定**(2026-07-25):
- num_train_steps = **40000**(用户定)。
- 数据 = base 5天(428 pq) + dagger **07-24**(42 ep, 用; 07-23 丢)。dagger **有 `intervention`** → positive⟺人控。
- 资源 = 北京 16 卡。init = `pi05_v4_awbc/49999`。

**待定**:
- val 留出策略(建议留 base 最后一天 07-24 的若干 ep)。
- class0 长静止裁剪阈值(§3.1, 建议 >2s=60帧)。
- **数据仍在采集**: 是否等采集定稿再训, 还是先用当前批次验夹爪(本 plan 取后者=先验)。

## 6. 风险
- **R1 norm_stats 重算 × warm-start 张力**: 旧权重在旧 norm 空间学的, 新 norm(细长夹爪值域)会移动归一化空间 → 微调初期需重对齐。缓解: 保守 LR + 盯早期 MAE 曲线; 若严重, 备选"冻结 backbone 只调 action head 前几 k 步"。
- **R2 dagger 质量参差(已发现并处置)**: 07-23 批次 100% 自主+30s 卡死=废(已丢); 07-24 合格(64% 人控)。**必须逐批检查**, 且 07-24 内部 class0 长静止要裁(§2.5/§3.1)。后续新增批次同样先检查。
- **R3 夹爪 action/state 约定**: 细长夹爪的开合值/方向约定须与 init 模型一致(参记忆 `pi05 LIBERO eval gripper坑` / v4 "action≠state gripper-from-master")。真机验证前先离线核对 Task_A1 gripper 列的取值范围/方向 vs Task_A。**未核对前不下真机结论**。
- **R4 数据新鲜且量小 + 仍在增长**(07-20~24, 几天)→ 可能过拟合, 且训练用的是**未定稿快照**。缓解: 40k 步配保守 LR + 盯 val 早停 + base(428)主导稀释 dagger 噪声 + 真机为准。若首版效果好, 待采集定稿再训正式版。

## 7. 关联
- init 模型: `pi05_v4_awbc`(config.py) · dagger 打标: [`awbc_three_paradigm_comparison_plan`](awbc_three_paradigm_comparison_plan.md) · gripper 坑: 记忆 `project_pi05_libero_eval_gripper_bug`
