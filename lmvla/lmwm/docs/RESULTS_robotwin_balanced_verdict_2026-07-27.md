# RESULTS — RoboTwin balanced stack 判决(make-or-break)

> 2026-07-27. 修复 curation bug(stack 喂对数据)后, LMWM dual2q vs LaWM baseline, 均训到 step 20000。
> 6 积木任务 × 50 局 × 4 seed × 2 难度(Easy=demo_clean / Hard=demo_randomized), North-H20 8×H20。
> ckpt: baseline `robotwin_baseline_lawm_balanced_bj/20000` · LMWM `robotwin_lmwm_dual2q_balanced_bj/20000`。

## 判决表(baseline → LMWM, Δ=LMWM−baseline, n=4 seed)

| 任务 | Easy | Hard |
|---|---|---|
| beat_block_hammer | 94→94 (+0.0) | 84.5→90.0 (**+5.5**) |
| handover_block | 83→87.5 (**+4.5**) | 78.5→79.0 (+0.5) |
| blocks_ranking_rgb | 96→87.5 (−8.5) | 95.5→91.5 (−4.0) |
| blocks_ranking_size | 79→80.5 (+1.5) | 78→83.0 (**+5.0**) |
| **stack_blocks_two** | 92→**98.5 (+6.5)** | 88→**97.0 (+9.0)** |
| stack_blocks_three | 95.5→88.5 (−7.0) | 87→83.0 (−4.0) |
| **6 任务净 Δ 均值** | **−0.5** | **+2.0(LMWM 净赢)** |

## 结论

1. **stack_two 大赢(+6.5/+9.0)**: 之前 stack 地板 0% 被证明是 curation bug(milestone 未覆盖 stack ep 区间); 喂对数据后 baseline 88-92(非地板), LMWM 稳定超越。→ **"倒 U 左端"回答: 难但喂饱的任务 LMWM 帮忙, 非地板归零**。
2. **难度门控增益**: Easy 净 ~0(更饱和), Hard 净 +2.0(基线更低更未饱和)。→ **坐实"增益 ∝ headroom"**, 与 LIBERO 饱和→null 同一规律的反向证据。
3. **stack_three −4~−7 = 已知数据缺口**: task 2 milestone 数据尚未抽(全量抽取 7lv4s 补中), LMWM 臂对它无信号只有污染 → 负。补齐后重评净 Δ 应更高。
4. **ranking_rgb 稳定负(−4~−8.5)**: 真实负项, 待查。

## 论文意义

补上 LIBERO null 的缺口: **LMWM 价值在未饱和+有 headroom 的 RoboTwin(尤其 Hard)成立(+2.0), stack_two 喂对数据 +6.5~9**。LIBERO 的 null 是饱和所致(见 [[RESULTS_pi05_libero_4suite_2026-07-27]] / [[project_libero_4suite_lmwm_negative]]), 非方法无效。统一为"增益受 headroom 门控"机制。

## caveat / 待办
- n=4 seed 点估计, 未算 Welch t 显著性(eval yaml 自带聚合因 group 名带 _hard 后缀未匹配 → 手动聚合; 需修 glob)。
- stack_three 待 7lv4s 补数据后重评。
- ranking_rgb 负项机制待查。
- 数据: `lmvla/lawam/results/eval_runs/robotwin/rt_{baseline,lmwm_dual2q}_balanced{,_hard}/seed*/**/*.json`(键 success_rate)。
