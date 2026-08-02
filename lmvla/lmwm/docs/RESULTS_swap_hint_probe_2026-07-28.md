# P0-D 跨-episode 换 hint 因果探针 — 结果 (2026-07-28)

> 目的: 因果级证明"绝对 milestone 目标的**场景身份成分**有害"——推理时注入错误 episode 的 hint,
> 比较错误**绝对** vs 错误**残差** 的伤害。预注册预测: 错误绝对伤害 >> 错误残差(身份成分错配)。
> **诚实裁决实验。** 本文档如实记录: 原设计不可跑的机理原因 + 已建的正确注入通路 + 前向机理证据 +
> 尚缺的闭环 SR 裁决。

## TL;DR
1. **原设计(让 provider 查表返回别的 episode)在推理期是严格 no-op。** dual2q 推理时 milestone hint
   由模型**自预测**(h_ms_pred), provider / h_ms_gt / `LMWM_MS_RESIDUAL` 全部只在**训练期**生效;
   eval 明确 `unset LMWM_MILESTONE_TARGET`。没有可换的"运行期 hint"。
2. **唯一可换点 = 覆盖 predict_action 的 h_ms_star。** 已在 `lawam.py` 加最小 env 门控注入
   (`LMWM_SWAP_HINT` / `LMWM_SWAP_HINT_ZERO`), 训练路径零影响。
3. **残差 vs 绝对是 ckpt 训练属性, 不是 eval 开关。** 需两个 ckpt(绝对 dual2q + 残差 dual2q),
   注入形态各自匹配。两 ckpt 本机均在。
4. **前向机理探针(单帧, in-process)已跑**: 注入 hook 因果 live(改动作), 但 milestone 通道足迹小
   (默认 cfg=1 时~2% ‖a‖, cfg=5 时~16%), 且**错误 milestone ≈ 无 hint(置零)**, 绝对/残差 ckpt 同态。
   → 前向层面无法分辨"错误绝对 vs 错误残差", 需闭环 SR。
5. **闭环 SR 裁决未跑**(单 GPU 240 rollouts 数小时 + sim server 基建), 已备好一键 launch(见 §5)。

## 1. 推理通路(摸清后的事实, 附代码位置)
`lawam.py::LatentWorldPolicyBackend`:
- `_run_shared_encoding_core`(行 ~717-845): 训练/推理共用编码。
  - milestone target provider 在此被查(行 762-766), **但仅当** `episode_index` 在 batch 中且 provider 就绪。
    产出 `_ms_target` / `h_ms_gt` —— 这是**训练监督靶子**。
  - `LMWM_MS_RESIDUAL`(行 804)、`LMWM_MS_GATE`(行 782)只改 `h_ms_gt`(训练靶), 不进推理动作路径。
  - 推理动作路径用的是 `h_ms_pred = _decode_ms_future(h_t, code=pred_action_emb_ms)`(行 829): 由当前
    obs 经 VLM 出的 ms code 生成的**自预测 milestone**。
- `_build_flow_future_condition`(行 656-682): `if not self.training: return h_t1_pred` —— 推理期直接用
  pred, 不做 teacher-forcing(GT 只在训练期按调度混入)。
- `predict_action`(行 1008+): flow 采样 `h_ms_star = shared.h_ms_pred`(行 ~1050,现被 SWAP 门控覆盖)。
  **h_ms_gt / provider 输出在此从不出现。**

**三重印证 provider 推理期不生效**:
(a) 代码: predict_action 只引用 h_ms_pred; (b) 代码: `LMWM_MS_RESIDUAL` 仅在训练靶块被读;
(c) 实践: `train_scripts/kai/volc/libero_eval_dual2q_3suite_1gpu_s1.yaml:106` 明确 `unset LMWM_MILESTONE_TARGET`。

> 结论: "provider 查表换 episode_index"改不了推理动作 —— milestone hint 已在训练中**摊进 VLM 权重**,
> 推理由观测自生。故 handoff 里"eval 脚本 provider 查表时替换 episode_index"路径**不可行(no-op)**。

## 2. 已建的最小注入(正确的换 hint 通路)
`lawam.py::predict_action`(sample_actions_cfg 前)新增 env 门控, 默认关、训练零影响:
```
LMWM_SWAP_HINT=<npy>   h_ms_star ← load(npy) 广播到 batch (npy 已是 ckpt 原生形态)
LMWM_SWAP_HINT_ZERO=1  h_ms_star ← 0 (无 hint 对照)
```
错误 hint 注入 = 把外来 episode 的 milestone(绝对/残差形态)喂进全局通道, 恒定于整段 rollout。
- 绝对 ckpt: 注入外来 ep 的**绝对** milestone 特征(载错误场景身份)。
- 残差 ckpt: 注入**身份无关的错误残差** = 外来 ep 两 milestone 帧之差(纯错误"变化方向", 场景身份相消)。

脚本 `lmvla/lmwm/scripts/pr_swap_hint_probe.py`(prep / forward 两模式)。
注入特征已备: `lmvla/lmwm/data/swap_hint_probe/{wrong_abs,wrong_resid}.npy`(错误 ep=1692)。

## 3. 前向机理探针结果(in-process, 单合成帧, 固定采样种子)
方法: 固定 obs, 只变 h_ms_star; 测 ‖Δaction‖ 相对 ‖a‖。固定 seed 后 native×2 噪声=0。
ckpt: 绝对=`20260718_111535+lmwm_dual_2q`, 残差=`rl4jj_2q_resid_noTs`, steps_12500。

| ckpt | cfg | ‖native×2‖噪声 | Δ 错误(本形态) | Δ 错误(另形态) | Δ 无hint(zero) |
|---|---|---|---|---|---|
| 绝对 | 1.0 | 0.0% | 2.0% | 1.8% | 1.7% |
| 残差 | 1.0 | 0.0% | 2.4% | 2.6% | 2.3% |
| 绝对 | 5.0 | 0.0% | 16.0% | 15.7% | 15.2% |

读数:
- **注入 hook 因果 live**(改动作), 且效应受 CFG 放大("CFG 只调此段"): cfg1→~2%, cfg5→~16%。
- **错误 milestone ≈ 无 hint(置零)**: 每行"Δ错误"仅比"Δ无hint"高 0.1-0.8 个百分点。即模型动作对
  "hint 在不在"敏感, 对"hint 内容是哪个 milestone"几乎不敏感。
- **绝对与残差 ckpt 前向行为同态**, 前向层面无法分辨"错误绝对伤害 vs 错误残差伤害"。

## 4. 判据结论(如实)
- 预注册判据("错误绝对 SR 下降 >> 错误残差")**尚不能裁决**: 需闭环 SR, 未跑。
- 前向机理探针给出的**旁证**: 在单帧前向层面, 错误 milestone 与置零几乎无差、绝对/残差同态 →
  倾向"milestone 内容(含身份成分)的边际因果影响小", 与"注入冗余"论一致; 但**不能**据此声称
  "身份成分有害坐实", 也不能声称"两者同伤"——前向 ‖Δaction‖ ≠ 闭环成功率(小扰动闭环可累积翻盘)。
- **重要方法学澄清**: handoff 关于本实验的两处前提在本栈不成立, 已如实修正:
  (i) "milestone 推理时由 provider 提供(查表或预测)"—— dual2q 是**纯预测**, provider 训练期专用;
  (ii) "残差 vs 绝对靠 env LMWM_MS_RESIDUAL 控制目标形态"—— 该 env 仅训练期; 推理形态是 ckpt 属性。

## 5. 尚缺的裁决实验(一键可跑)
闭环 SR, 3 任务(t5 spatial / t6 / 一饱和)× 4 条件 × 20 trials, 每 ckpt:
```
# 公共: LMWM_DUAL=1 LMWM_DUAL_2Q=1 LMWM_SWAP_TEACHER=1 \
#       LMWM_CKPT=$REPO/lmvla/lmwm/checkpoints/lmwm_libero_rvalley/lmwm.pt \
#       LMWM_ADAPTER_DIR=$REPO/lmvla/lmwam/adapter  (确保 unset LMWM_MILESTONE_TARGET) \
#       STAR_VLA_PYTHON=$REPO/kai0/.venv/bin/python  MUJOCO_GL=egl(gf0) \
#       SUITES=libero_spatial(或对应套件) NUM_TRIALS_PER_TASK=20  MAX_TASKS=<定位 t5/t6/饱和>
# 入口: lmvla/lawam/examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh $CKPT
ABS_CKPT=.../20260718_111535+lmwm_dual_2q_cnsh_volc/checkpoints/steps_12500_pytorch_model.pt
RESID_CKPT=.../rl4jj_2q_resid_noTs/checkpoints/steps_12500_pytorch_model.pt
# 绝对 ckpt: ①(无SWAP env) ②LMWM_SWAP_HINT=data/swap_hint_probe/wrong_abs.npy ④LMWM_SWAP_HINT_ZERO=1
# 残差 ckpt: ①(无SWAP env) ③LMWM_SWAP_HINT=data/swap_hint_probe/wrong_resid.npy ④LMWM_SWAP_HINT_ZERO=1
```
预计成本: 单 GPU 每 rollout ~1-3min → 每 ckpt 60 rollout ~1.5-3h; 建议北京队列或本机 2 卡并行、
用 MAX_TASKS 只跑 3 目标任务。判据同预注册: 比较"错误绝对 SR 降幅"vs"错误残差 SR 降幅"。
注意: 前向探针提示 milestone 通道边际足迹小(cfg=1 时~2%), 若闭环也几乎无降幅, 可提高
`LMWM_CFG_GUIDANCE` 增强 hint 依赖后再测(否则四条件可能都贴近对照、无法分辨)。

## 6. 附
- 注入实现: `lmvla/lawam/starVLA/model/framework/vlas/lawam.py` predict_action(SWAP_HINT 段)。
- 脚本: `lmvla/lmwm/scripts/pr_swap_hint_probe.py`。
- 注入特征: `lmvla/lmwm/data/swap_hint_probe/wrong_{abs,resid}.npy`。
- ckpt: 绝对 `results/Checkpoints/libero/20260718_111535+lmwm_dual_2q_cnsh_volc`;
        残差 `results/Checkpoints/libero/rl4jj_2q_resid_noTs`(train yaml `lmwm_2q_resid_noTs_8h20.yaml`)。
