# 实验移交清单(2026-07-28)— 缺失数据与执行需求

> 交给执行 agent 的自包含说明。背景一句话:论文主张 = "VLA 的子目标注入存在冗余/别名两大根因;残差目标(ms−h_t)是修复"。每个实验都标注:目的 / 精确做法 / 产出 / 判据 / 已知坑。
> **通用坑(必读)**:①volc 提交一律 `Preemptible: false`;②lawam ckpt 评测必须带同目录 `config.yaml`+`dataset_statistics.json`;③LMWM_ADAPTER_DIR 必须指 `$REPO/lmvla/lmwam/adapter`(软链易丢);④北京 eval 曾出现 GPU 槽位 {1,4} EGL 确定性 abort——失败路换 seed 值无效,换机器/osmesa 有效;⑤多 seed 必须变 `EVAL_SEED`,聚合差 <1.5pt 不可声称;⑥run_libero_benchmark 需 `SUITES=<套件>` 否则跑全 4 套件。
> 路径:REPO=/vePFS/tim/workspace/deepdive_kai0(cnsh)/ /vePFS-North-E/vis_robot/workspace/deepdive_kai0(北京)。提交经 gsy(ssh -p 16370 root@124.174.16.237, 密码 tim),creds 见 train_scripts/kai/volc 现有 yaml 用法。

## P0-A(PR-2)混淆/结构分数管线 —— 定律的裁决实验【最高优先,纯离线,零GPU竞争】
- **目的**:把"指引/精度"事后分类升级为数据可算的定律。预注册预测见 PAPER_PLAN §16(必须先读,预测已锁死:t5 是 spatial 组混淆分数极端离群值、混向 t0/t1;40 任务 Δ 与混淆分数负相关、与结构度正相关)。
- **做法**:
  1. 特征:`$REPO/lmvla/lmwm/data/pi05_feat/libero_v2p1_dinov3base/<suite>/ep*.npz`(key `grid` [N,256,768],四套件全有;robotwin 特征见 roadmap V4 记录)。episode→task 映射经各数据集 parquet 的 task_index(参考 `lmvla/lmwm/scripts/export_pi05_hint.py` 的索引方式)。
  2. **混淆分数**(per task):对任务 T 的每帧(池化 grid,L2),做跨-episode 最近邻检索(排除本 episode);统计近邻落在**其他任务** episode 的比率 → 任务级混淆矩阵(40×40 LIBERO + 6×6 robotwin)与每任务混淆率。
  3. **结构度分数**(per task):沿 episode 的 r-场谷计数 / milestone 段数(可复用 `lmvla/lmwm/scripts/p1_libero_milestone_pairs.py` 相关逻辑),取每任务均值。
  4. 对答案:与已有 per-task Δ 相关(数据在 PAPER_PLAN §10/§13/§16 与 eval_runs 目录):Spearman(Δ_dual2q−nowm, 混淆分数)、Spearman(Δ, 结构度)。
- **产出**:`lmvla/lmwm/docs/RESULTS_PR2_scores_2026-XX.md` + npz(每任务两分数+混淆矩阵)+ 相关系数与散点数据。
- **判据**:预测①②对 → 定律成立进论文;错 → 如实记"定律证伪",不许改预测。

## P0-B 反衰减对照 —— 堵"残差=调弱 hint"的审稿必问【2 个训练 + 2 组 eval】
- **目的**:残差范数≈绝对的 1/4;必须证明增益来自"减掉冗余成分"而非"信号变小"。
- **做法**(LaWAM 栈,北京 8×H20,非抢占,每个 12500 步):
  1. 臂① 绝对目标×0.25:`lawam.py` 已有 env 体系,加 `LMWM_MS_ABS_SCALE=0.25`(仿 `LMWM_MS_RESID_SCALE` 的 3 行实现,作用于 `_ms_target`,不开 RESIDUAL);yaml 克隆 `train_scripts/kai/volc/lmwm_2q_resid_noTs_8h20.yaml`(去 RESIDUAL 行,加 ABS_SCALE)。
  2. 臂② 残差×4(范数匹配):yaml `lmwm_2q_resid_noTsS4_8h20.yaml` 已存在,改 `Preemptible: false` 重提即可。
  3. 各 8-seed eval libero_10(模板 `libero_eval_resid_noTs_x8.yaml`,改 CKPT 钉死+RUN_GROUP;seed 路失败按通用坑④处理)。
- **判据**(预注册):若 臂①≈基线94.2(衰减无增益)且 臂②≈残差95.75(范数无关)→ 残差机制成立;若 臂①≈95.7 → 增益只是衰减,残差主张降级。
- **产出**:两臂 n=8 聚合+per-task 表,append 到 PAPER_PLAN。

## P0-C 门控臂 —— t5 别名的对症实验【1 个实现 + 训练 + 双套件 eval】
- **目的**:检验"检索置信门控修复别名伤害"(PAPER_PLAN §16 预测③:t5 回 ≥65)。文献支撑:FutureVLA 可学习门 σ(r)、AHEAD 不确定度截断。
- **做法**:
  1. 实现(lawam.py dual 分支):ms 通道逐样本门控 `g = σ((margin−τ)/T)`,margin=milestone 检索的 top1−top2 相似度差(provider 侧可得;若不可得,退化用 ‖residual‖ 分位数作 proxy,并在文档里写明选择);`h_ms_star ← g·h_ms_star`。env-gated `LMWM_MS_GATE=1`,默认关。训练/推理同路径。
  2. 基座:2q+residual(无 tsched)配方 + 门控;smoke 6 步(gf0 2 卡,参考 v8_resid_smoke.sh)后北京训练。
  3. eval:libero_spatial ×8 seed(osmesa,模板 `libero_eval_resid_spatial_x8.yaml`)+ libero_10 ×8 seed(确认不伤主结果)。
- **判据**:t5 ≥65 且 libero_10 ≥95(不回吐)→ 预测③成立;t5 无改善 → 别名在检索端而非注入端,记录后转检索侧修复。

## P0-D 跨-episode 换 hint 因果探针【零训练,~2h】
- **目的**:因果级证明"绝对目标的场景身份成分有害"。
- **做法**:推理时故意给错误 episode 的 hint:取任务 T 的 rollout,注入来自**另一 episode/另一任务**的 (a) 绝对 milestone (b) 残差 milestone;各 20 trials × 3 个任务(t5、t6、一个饱和任务)。实现:eval 脚本里 provider 查表时替换 episode_index(或在 `local_eval_hint.py` 风格的 in-process 脚本里手动喂)。
- **判据**(预注册):错误绝对 hint 伤害 >> 错误残差 hint(因身份成分错配);若两者同伤 → 冗余机理削弱。

## P1-E RoboTwin 基线训到位 + 残差臂【论文 RoboTwin 腿的合法性】
- **目的**:hammer 头条(0.5→20)建立在欠训基线上(sft_release 单任务 90%),审稿不可用。
- **做法**:①LaWM baseline 与 dual2q 各续训/重训至收敛(>20k 步或 loss 平台,配方=robotwin_baseline_lawm / robotwin_lmwm_dual2q 现有 yaml);②加 2q+residual robotwin 臂(管线现成:robotwin pairs+LMWM 已存在);③判决 eval 模板 `robotwin_eval_blocks_2ckpt_x4_8h20.yaml`(6 积木任务×4seed×50 trials,需 wrapper 自愈见 memory project_lawam_submodule_untracked_assets)。
- **判据**:训到位后 Δ(dual2q/resid−baseline) 仍 >2×SEM → RoboTwin 腿合法;hammer 增益若消失 → 如实降级。

## P1-F 残差 4-suite 补齐【便宜,2 组 eval】
- **做法**:2q+resid ckpt(北京 `20260724_071210+lmwm_2q_resid_noTs_cnbj_volc/.../steps_12500`)在 libero_goal、libero_object 各 ×4 seed(osmesa,模板同 spatial 版改 SUITES)。spatial(94.4±0.84 n=8)已有。
- **判据**:预测两组饱和持平(≈98.3/99.5);齐后可出残差的完整 4-suite 行(预计聚合 ~96.9 vs nowm 97.2——如实报,主张落在"修复大半净负+libero_10/robotwin 增益")。

## P1-G pi05 跨栈修复验证【1 组 eval + 可选 1 训练】
- **做法**:①a2_res(pi05+so400m 残差 hint,北京训练已完/近完,ckpt `pi05_libero_a2_residual_prefix_bj`)用 in-process 评测 `lmvla/lawam/examples/LIBERO/eval_files/local_eval_hint.py`(`--config pi05_libero_a2_residual_prefix_eval --encoder so400m` + `EVAL_HINT_RESIDUAL=1`,gripper 已内置 inv_pm1)×4 seed libero_10;②可选:a2_residual_suffix 训练(config 需新增,组合=已证 suffix>prefix)。
- **判据**:pi05 侧残差 Δ 与 LaWAM 侧同号 → "修复跨架构迁移"写进论文;不同号 → 只声称 LaWAM 侧。

## P2(可选加分)
- H1 armB 基线 n=4→8(加固 +1.55 的显著性);H2 UR-VC 条件化基线(同 harness);H3 真机 Task_A 离线分析(E10a,crave bank 现成)与叠衣 A/B(E10b,门控 G1 已过);H4 WorldArena 2.0 一手出处核实。

---

## N 系列(2026-07-28 claim 审计后追加 —— 堵"能杀死论文"的洞)

> 背景:逐 claim 证据审计发现两个致命洞 + 一个逻辑张力。截稿倒排:**Workshop(8/10)必做** = P0-B(已提交 tn9xc/rlq6x)+ N1 + N2 + N4/N5(离线)+ N3(写作)+ PR-2(P0-A 跑中,裁"可预测"存废)。

- **N1 naive-swap/dual 多 seed 补评**【eval-only, 小任务多提北京】:Table I 的 naive/dual 行是单 seed 旧数、协议不一致(橙标自曝)。用现有 naive/dual ckpt 各 ≥4 seed 重评 libero_10,统一 Table I 协议。产出:统一协议的 Table I。
- **N2 残差第二训练种子**【1 训练 + 8seed eval, 防"幸运种子"】致命级:residual 95.75 来自唯一一次训练;我们自己数据证明训练种子方差达 24pt(dual2q#2 的 t5 42→18)。克隆 `lmwm_2q_resid_noTs_8h20.yaml` 换训练 seed(run_id 加 _seed2)重训 12500,再 8seed eval libero_10+spatial。判据:第二种子 residual 与首次 ≥95 同区间 → +1.55 稳;若崩到 <94 → 头条结果是幸运种子,降级。
- **N3 残差×tsched 交互如实入附录**【零实验, 纯写作 + 已有 v8xpb 数据(精度旋钮翻转)】:草稿没写=挑有利变体嫌疑。必须附录如实报残差×tsched 交互(residual 单独 vs +tsched 的差)。
- **N4 CV/86% 扩全库分布**【离线, 零 GPU】:现 CV 只算 2 条 episode。全库(所有 episode)重算残差 vs 绝对目标的判别力分布(CV、<某阈值冗余比例),把 Fig.2 从 2 条升级为分布图。
- **N5 corr+0.66 bootstrap CI + LaWAM 侧同指标**【纯分析, 零 GPU】:corr+0.66 无 CI/仅 10 点/未在 LaWAM 侧复算。bootstrap CI + LaWAM 侧同指标复算。
- **N6 基线失败模式定性分类**【eval 回放分析, 零训练; ICLR §4.1 用】(2026-07-28 追加):§4.1 现在用数字论证"基线缺全局指引"(t6 82.5/t9 86 vs t7 100 + 四级分解),但缺定性证据:基线 t6/t9 失败 rollout **长什么样**。做法:取 armB 基线任一 seed 的 libero_10 eval 视频/轨迹(North-E eval 已存 mp4,或本机重跑 2 seed 只录 t6/t9),对每条失败 episode 归类:①犹豫/徘徊(臂在两目标间震荡)②第二对象忽略(t6 双 moka pot 只放一个即停)③过早终止④精度失误(错抓/碰倒)。产出:失败模式计数表(预期①②③指引类占多数=支持诊断;若④精度类占多数=削弱"缺指引"论断,§4.1 需改写)。同表可对 residual ckpt 复算,展示修复选择性(指引类失败下降、精度类不变)。落到论文 `\todo{N6}` 处。

## ⚠️ P0-D 状态更正(2026-07-28 执行后)
原 P0-D"换 provider hint"设计**推理期是 no-op**:LaWAM dual2q 推理时 milestone = 模型自预测 h_ms_pred(由 obs 经 VLM 生成),**非 provider 查表**;LMWM_MS_RESIDUAL 只作用训练靶,推理不碰。已建正确注入点(lawam.py predict_action 的 `LMWM_SWAP_HINT` env,在 lawam **submodule** 内需 commit 保留)。前向探针(‖Δaction‖)结果:**错误 milestone ≈ 置零无 hint**(每条件仅差 0.1-0.8pt)、绝对/残差 ckpt 前向同态 → 倾向"milestone 内容边际因果小",与"注入冗余"论一致,但**前向 ≠ 闭环 SR,不是因果裁决**。闭环 SR 一键可跑(prep 已备 `lmvla/lmwm/data/swap_hint_probe/wrong_{abs,resid}.npy` + launch 配方,3 任务 × 4 条件),但需数小时 + 提高 CFG_GUIDANCE(否则四条件贴近对照无法分辨)。详见 `RESULTS_swap_hint_probe_2026-07-28.md`。

## 🔧 RoboTwin 逻辑张力 —— 定律自洽解释(已写入草稿, 零成本堵洞)
论文批判绝对目标有害,而 RoboTwin 腿恰是绝对目标 dual2q 赢的(+2.9 / balanced stack_two +6.5~9)。**这与定律自洽,非自相矛盾**:定律 = "绝对目标 = 身份成分 + 目标成分;身份成分只在**有别名压力**处有害(如 spatial t5:同物体集只差空间关系,跨-ep 检索别名最坏)"。RoboTwin blocks 任务**无别名压力**(积木族任务间语义/几何差异大,跨任务混淆低),故绝对目标的身份成分不触发灾难,纯目标成分获益;hammer 是指引受限任务(时序/相位帮击打时机)→ 绝对目标在无别名处也能净赢。**推论**:RoboTwin 上残差应 ≥ 绝对(去身份成分不伤、可能微增),但因无别名压力,残差优势小于 spatial。**待补 P1-E robotwin 残差臂验证残差 ≥ 绝对**。审稿人若不知此解释会当自相矛盾,故必须明写(本段即草稿用文)。
