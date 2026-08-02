# dagger clip 双向"起爆点"前裁 —— 回折冻结修复实验 plan

> **建立**: 2026-07-10
> **目的**: 把 dagger clip **前后双向裁到"起爆点"**(前砍迟疑起手、后砍静止收尾,只留果断动作核心),重 build → 重训 → 真机验证:能否**修掉回折过渡冻结**,同时**保留 fresh dagger 的夹爪新语义**。
> **状态**: ✅ **实验完成 — H 成立**(2026-07-14)。真机部署 best ckpt 49999,**回折冻结已修复**。双向起爆点前裁是修法。见 §7.1 结果 + 执行记录 §8.6。
> **上游诊断**: 见 [`pi05_v4_awbc_modeB_freeze_diagnosis_plan.md`](pi05_v4_awbc_modeB_freeze_diagnosis_plan.md) §9。**最可能根因** = dagger clip 从 **06-16 起**把"操作员接管卡住态后迟疑 ~0.5s 再动"的起手段裁进 clip → 被 AWBC 标 positive → 部署永远喂 positive → 回折过渡(卡住样决策态)触发"迟疑不动" → 剂量够(任务②)则冻。
> ⚠️ **铁律**: 真机为终判;夹爪维单列看。

---

## 0. 假设(本实验要证/证伪)
**H**: 冻结由 dagger clip 的**边界段**(前:迟疑起手 v<0.02;后:静止收尾 v<0.008)注入 —— 这些低速帧被 AWBC 标 positive、被部署 always-positive 放大成"决策态迟疑不动"。**双向裁掉边界、只留果断动作核心 → 不冻**。
- **不冻** → H 成立 → 前裁起爆是修法,fresh dagger 可用(夹爪也修)。
- **仍冻** → 边界段不是根因,是碎片/剂量本质 → 转 velocity-aware discretize / 重采 / 部署兜底。

---

## 1. ⭐ 双向"起爆点"前裁算法(逐 dagger clip)
对每个 dagger clip(仅 dagger,**base demo 不裁** —— base 是完整任务、非卡住态接管):
1. 臂速 `v[t] = ‖action[t]−action[t−1]‖`(12 臂关节 idx 0-5,7-12;**排除夹爪 6/13**)。
2. 平滑 `v̄[t]` = 5 帧滑动均值(去噪)。
3. **前起爆点** `t_start` = 首个满足 `v̄>THR` 且**连续 ≥K 帧**的 t(果断起手)。
4. **后起爆点** `t_end` = 末个 `v̄>THR` 的 t(最后果断动作)。
5. 保留 `[t_start−M : t_end+M]`(留 margin M 帧,避免切太狠)→ 果断动作核心。
6. **参数**(待微调):`THR=0.02`(果断阈值,来自逐天分析:果断 dagger 起爆 >0.02)、`K=5`、`M=5`、`MIN_LEN=30`(1s)。
7. **护栏**:
   - 全程无 `v>THR`(纯 hold clip)→ **丢弃**(坏/idle clip)。
   - 裁后 `<MIN_LEN` → 丢弃(记数,别静默)。
   - 裁后仍需 3 相机 + parquet 帧一一对齐。
8. **视频同步裁 + PTS 归零**:parquet 裁 `[t_start:t_end]` 行 → 视频 mp4 裁同帧段并 **reset PTS**(⚠️ 复用 `build_no_release.py` 的裁剪+PTS归零机制,见 [[reference_v3_trim_video_pts_bug]] 别再犯 PTS 不归零 bug);frame_index/timestamp 重排。

**范围**:对**所有 dagger 日期统一裁**(05-29~最新)。06-16 前本就果断(起爆 0.03-0.17s),裁掉的前段极小、无害;统一处理保证一致、不引入日期混淆。

---

## 2. 数据构建(裁后重走 AWBC pipeline)
1. **裁**:对 `vis_dagger/v4/<所有日期>` 逐 clip 双向起爆点前裁 → 落 `vis_dagger/v4_launchtrim/<date>`(新目录,不覆盖源)。记录:各日期裁前/裁后帧数、丢弃 clip 数、平均裁掉的前/后段长度。
2. **build merged**:base(**不裁**,13 日期)+ **裁后 dagger** → `self_built/A_v4_base_dagger_launchtrim`(仿 `build_v4_awbc_merged.py`,删 intervention、symlink 裁后视频、episode 重排)。
   - **主实验(任务②-analog)**:base + 裁后(旧 dagger 05-29~06-23 **+ fresh 06-29~07-03**)→ 直接对标已冻的任务②。
3. **重算 norm_stats**(裁后集,action_dim=32)。
4. **AE 打标**:复用 `adv_est_v1`(step 100000)→ `absolute_advantage`。
5. **discretize**:binary top-30%(与所有 v4 AWBC 一致,保持可比)。

---

## 3. 训练规格(单变量 vs 任务②)
- **config** 新建 `pi05_v4_awbc_launchtrim`(克隆 `pi05_v4_awbc`,**无 DCT**):
  - `repo_id` → `A_v4_base_dagger_launchtrim`(裁后 labeled);`prompt_from_task=True`;`use_delta_joint_actions=False`。
  - init=`pi05_base`,LR warmup1k/peak1.5e-5,50k,bs128,fsdp8,EMA0.9999。
  - **唯一变量 vs 任务②(plus_freshdagger)= dagger clip 双向起爆点前裁**(其余逐字段同)。
- **8 卡**,gf3/cnbj 择空闲。

---

## 4. 评估(真机为终判)
| Tier | 做法 |
|---|---|
| offline | 裁后 val 逐 ckpt val MAE(整体+夹爪维单列)+ loss sanity |
| **真机(决定性)** | **回折过渡是否还冻 >5s**(与任务② 同协议对照)+ 夹爪微张开是否修复 + 成功率 |

**判据**:
- ✅ **H 成立** = 真机**无回折冻结** + 夹爪修复 → 前裁起爆是修法,采用;fresh dagger 可正常用。
- ⚠️ 缓解未消 = 冻结时长缩短但仍有 → 边界段是部分诱因,叠加 velocity-aware discretize。
- ❌ 仍冻 = 边界段非根因 → 转诊断分支(discretize / 重采 / 部署兜底 §diagnosis-plan)。

**对照**:任务②(裁前,已冻)↔ 本实验(裁后)= 单变量"是否裁边界"→ 直接归因。

---

## 5. 落地步骤
1. **实现前裁脚本** `launchpoint_trim_dagger.py`(算法 §1;复用 build_no_release 裁剪+PTS归零)→ 干跑打印各日期裁前后统计,人工核验 THR/K/M 合理(抽几个 clip 看裁得对不对)。
2. **裁** 全 dagger → `vis_dagger/v4_launchtrim/`。
3. **build** `A_v4_base_dagger_launchtrim`(base + 裁后 dagger + fresh)+ 重算 norm。
4. **AE 打标 + discretize** top-30% → labeled。
5. **注册 config** `pi05_v4_awbc_launchtrim`,commit/push。
6. **8 卡 50k 训练**。
7. **真机** vs 任务②,落 §4 判据。
8. 回填 diagnosis-plan §9 结果 + master history。

---

## 6. 风险 / 注意
- **THR 过大**:把慢但有效的精细操作也当"非果断"裁掉 → 丢失真实动作。先干跑抽检、`THR=0.02` 保守起,必要时降。
- **裁后 clip 太短/丢太多**:统计丢弃率;若某日期丢弃过多(说明该日期多为迟疑/idle),单独看是否该整段弃。
- **视频 PTS**:裁视频**必须 reset PTS**,否则 vision↔action 错位静默训坏(见 [[reference_v3_trim_video_pts_bug]])。裁后抽 ep 验证帧数=parquet 行数、PTS 从 0 单调。
- **base 不裁**:base 是完整任务 demo,裁边界会破坏"完整流转"覆盖(那正是不冻的保障)。只裁 dagger。
- **夹爪语义**:裁只动时间范围,不改 action 值 → fresh dagger 的 gripper-from-master 完整保留。
- **未分离"起手 vs 碎片"**:本实验裁的是**边界段**;若裁后不冻,证明是边界(迟疑起手+静止收尾);clip 仍短 → 顺带证明不是"clip 长度/碎片"本身。

---

## 7. 决策定档
- ✅ `THR / K / M / MIN_LEN` 参数:**0.02/5/2/30**(用户选 M=2 更狠,前裁 avg 0.33s/后裁 0.45s,整体保留 98%)。
- ✅ 裁范围:**全 dagger**(05-29~07-07 统一处理)。
- ✅ 集群:cnbj Robot-North-H20 8×H20。
- ✅ Option A:复用逐帧 task_index(非重打标,免 AE ckpt;与任务②严格单变量)。

## 7.1 ✅ 真机结果 (2026-07-14)

**判据对照**:

| 对照臂 | 冻结 | 说明 |
|---|---|---|
| 任务② (plus_freshdagger,裁前) | ❌ 冻 | dagger 含迟疑起手边界段 |
| **本实验 (launchtrim,裁后)** | ✅ **无冻结** | 双向起爆点前裁 |

**结论: H 成立** — dagger clip 的迟疑起手 + 静止收尾边界段是冻结根因,双向起爆点前裁是修法。fresh dagger 本身可用(夹爪语义保留),只需裁掉边界。

**根因链确认**:
1. 06-16 起 dagger clip 含 ~0.5s 迟疑低速起手(v<0.02) + 静止收尾
2. AWBC 把低速帧标 positive + 部署 always-positive → 决策态迟疑不动
3. 双向裁到起爆点只留果断核心 → 不冻
4. **与任务②唯一变量 = dagger 是否裁边界 → 单变量归因成立**

---

## 执行记录(2026-07-12)

**参数定稿**: THR=0.02 / K=5 / **M=2**(用户选"更狠一点", M从5→2 裁掉更多迟疑; 干跑核验新日期起爆点前段vbar=0.000~0.002确为迟疑, 前裁avg0.33s/后裁0.45s, 整体保留98%)。

**⭐ 关键偏离 — Option A 复用逐帧 task_index(非重打标)**: adv_est_v1 AE ckpt 已全网删除。但 `absolute_advantage[n]=V(f0,f_{n+int})−V(f0,f_n)` 是同参考帧差分, progress(f0)抵消→近似参考帧无关; 且 `task_index`(pos/neg)在labeled源里100%完整(absolute_advantage列仅~85%ep有)。故**直接复用逐帧task_index切到保留帧, 免重打标/discretize/AE ckpt**——这也正是任务②的做法(复用现成task_index未重discretize)→ **与任务②严格单变量**(仅dagger帧被裁; 裁掉的迟疑/静止边界帧整段移除, H假设直接被测)。§2.4-2.5 的"AE打标+discretize"因此跳过。见 [[reference_launchtrim_pipeline_northE]]。

**构建位置**: North-E/gf3 原生(gf3 180核+8×H20全空闲)。North-E本就是v4全镜像(vis_base/v4+vis_dagger/v4+venv+A_v4_base_dagger labeled), 只送 freshdagger_ft标签113MB。`build_launchtrim_from_labeled.py`(KAI0_ROOT切North-E)→ **A_v4_base_dagger_launchtrim=2510ep**(1200 base整段 + 1310 dagger前裁, drop2), 列与任务②一致, 抽检帧全对齐(PTS归零正确)。

**config**: `pi05_v4_awbc_launchtrim`(克隆pi05_v4_awbc无DCT, North-E路径, init pi05_base, warmup1k/peak1.5e-5/50k/bs128/fsdp8/EMA0.9999, inline_eval关)。直接插入North-E config.py(get_config验证通过)。

**提交**: `pi05_v4_awbc_launchtrim_cnbj_8gpu.yaml` → job `t-20260712080450-7f5kq`(cn-beijing/Robot-North-H20/1-host 8×H20/50k)。

**待办**: ~~训练完 → offline val MAE → 真机 vs 任务② → 回填~~ 全部完成见 §7.1。

### 8.6 训练结果 (2026-07-14)

| 项 | 值 |
|---|---|
| job | `t-20260712080450-7f5kq` (cnbj Robot-North-H20 8×H20) |
| best ckpt | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_v4_awbc_launchtrim/pi05_v4_awbc_launchtrim_cnbj/49999` |
| loss 收敛 | 0.70 → 0.003 (230×, 训练健康) |
| param_norm 增幅 | 0.16% (未过拟合) |
| val MAE | N/A (AWBC prompt_from_task=True, val 无 advantage prompt → inline-eval 失败, 预期内) |
| **真机冻结** | ✅ **不再冻结 — H 成立** |
| **夹爪** | ✅ 修复 (fresh dagger 夹爪语义保留) |

## 9. ⭐ CRAVE-value × chunk-001 拼接 dagger：标签重现冻结 + Δprogress 修复 (2026-07-16)

**背景**: 后续用 CRAVE value 架构对**新 chunk-001 拼接完整 ep**(不裁, model段+摇操段拼一条)打标, 走 AWBC 验证"新格式是否天然不冻"。数据 `A_v4_chunk001_dagger_crave_labeled`(387 base+387 dagger=774ep), config `pi05_v4_awbc_chunk001_dagger_crave`, job `t-20260715142954-t6dgl`(cnsh 16×A100 RUNNING)。

### 9.1 诊断：这套标签会重现冻结(实测正在训的那份数据)

| 问题 | 证据 |
|---|---|
| **A. "advantage" 其实是 progress** | 实测 `absolute_advantage` 与 `stage_progress_gt` **corr=1.0**、取值 0-1 完全相同。top-30% 二值化 = "任务进度最后30%(spg>0.68)标 positive" → 进度分位条件化, 非优势加权。 |
| **B. 拼接不裁 → 静止落进 positive** | dagger 静止帧(臂速<0.02)占 **48.0%**(base 27.3%); positive 帧里 **55.9%** 是静止&高进度(base 34.3%)。逐帧时间线证实: 摇操结束 settle、收尾静置(v≈1e-2, spg≈1.0)全被标 positive。 |

冻结机理与 mode B 一致: 部署恒喂 `Advantage: positive` → 任何"看着接近完成"的状态输出近零动作 → 冻结。拼接 dagger 静止占比≈base 两倍, 污染更重。**plan 假设"新格式天然不冻"数据不支持——恰恰相反。** 此外 `run_build_discretize_chunk001_crave.sh` 里 `--advantage-source stage_progress_gt --top 0.3` 路径/参数均不成立(真脚本在 `annotation/`, 只收 `absolute/relative_advantage`, 阈值 `--threshold`), 实际产出用 `absolute_advantage`(≡spg) 走 top-30%。

### 9.2 修复：Δprogress 标签 + 速度门控 + 复用 launchtrim 裁剪

不停 job, 另建修复版数据集(单变量对照原 job):

1. **advantage = 前向 Δprogress** `adv[t]=spg[t+H]-spg[t]` (H=50=动作 chunk 长)。奖励**主动推进**的帧; 静止 plateau(settle/迟疑) Δ≈0 → negative。取代 progress-level。
2. **速度门控**(关键): 一帧能被标 positive 当且仅当 `adv≥全局p70` **且臂真在动**(launchtrim 的 5帧平滑 `vbar>THR=0.02`, 排除夹爪 6/13)。**保证 static-positive=0%**——这是"不冻"的判据。单 Δprogress(不门控)dagger 仍 42% static-positive(前向 Δ 会给"静止但即将动"的 plateau 边界帧打正), 门控后 →0%。
3. **复用 launchtrim 裁剪**: dagger 走 `launch_window`(前砍迟疑起手 + 后砍静置收尾), base 不裁(同单变量策略)。

**脚本**: `train_scripts/kai/data/build_chunk001_dagger_crave_dprog_launchtrim.py`(复用 `launchpoint_trim_dagger.launch_window` + `build_no_release._select_job/per_episode_stats`)。task_index/tasks.jsonl 直接算(速度门控 discretize CLI 不支持)。

**干跑核验**(387 base+387 dagger, 全 387 dagger 保留, 0 drop):

| 组 | posfrac | static&pos/pos | 对比原标签 |
|---|---|---|---|
| dagger (Δprog+门控+裁) | 0.113 | **0.0%** | 55.9% → 0% |
| base (Δprog+门控, 不裁) | 0.405 | **0.0%** | 34.3% → 0% |

Δprog top-30% 阈值=0.031。base positive 多是干净 expert 果断动作(合理), dagger positive = 有效摇操纠错。

### 9.3 产物 (✅ 数据已构建+核验 2026-07-16)

- 数据: `A_v4_chunk001_dagger_crave_dprog_launchtrim`(8.4G, 774ep/1,366,666帧, 不覆盖 running job 的 `..._labeled`)。
  **实建核验**: 387 dagger 全保留(0 drop), positive 20.3%(276,802帧), **static&positive=0**; 抽检 dagger 裁后视频帧数==parquet 长度(PTS 归零正确, frame_index 0..N-1, ts0=0), base symlink 全整段对齐; 列含 `stage_progress_gt`+`absolute_advantage`(=Δprog)。norm_stats(action_dim=32) 已算。
- config: `pi05_v4_awbc_chunk001_dagger_crave_dprog`(克隆自 `_chunk001_dagger_crave`, 仅换 repo_id; `get_config` 通过, bs256/fsdp16/50k)。
- 脚本: `train_scripts/kai/data/build_chunk001_dagger_crave_dprog_launchtrim.py`(H=50, THRESH=30, `--dry-run` 报 static-positive; nproc48 视频重编码)。
- yaml: `train_scripts/kai/volc/pi05_v4_awbc_crave_dagger_dprog_cnsh_16gpu.yaml`(robot-task cnsh, 2×8 A100, JAX fsdp16 multi-node, MASTER_PORT 14524; pre-flight 查 tasks.jsonl advantage prompt + `stage_progress_gt`/`absolute_advantage` 列 + pi05_base init)。
- **提交 (2026-07-17)**: job `t-20260717114431-zqdqk`(cnsh robot-task 16×A100 50k), CreateTime 03:44 UTC, 当前 **Queueing**(队列配额暂不足, 排队中待资源释放——与 t6dgl 当初排队 ~5.6h 同款)。
- **待办**: 训完 50k → 真机 vs launchtrim(已证不冻) + vs t6dgl(原 `_labeled`, progress-level 复现冻结)。若修复版不冻而 t6dgl 冻 → 坐实"progress-level+不门控静止"是冻结源, Δprogress+门控是标签级修复(与 launchtrim 帧级裁剪互补/等效)。

## 关联
- 上游诊断 + 根因: [`pi05_v4_awbc_modeB_freeze_diagnosis_plan.md`](pi05_v4_awbc_modeB_freeze_diagnosis_plan.md)(§9 逐天 06-16 变点 + 迟疑起手机制)
- 裁剪+PTS 机制复用: `train_scripts/kai/data/build_no_release.py`(per-date front-trim + tail-cap + PTS) · [[reference_v3_trim_video_pts_bug]]
- build 合并源: `train_scripts/kai/data/build_v4_awbc_merged.py`
- 对照 config: `pi05_v4_awbc`(不冻基线)· `pi05_v4_awbc_plus_freshdagger`(任务②,裁前,已冻)
- AE / discretize: `kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/adv_est_v1` · `kai0/stage_advantage/annotation/`
- 数据: `kai0/data/Task_A/vis_dagger/v4/*`(源)→ `vis_dagger/v4_launchtrim/*`(裁后)
