# RESULTS — pi05 × LMWM-hint · LIBERO 4 套件完整矩阵

> 日期: 2026-07-27
> 机制: pi05(openpi pi0.5)基座 + lmwm_hint 作为 directive token 注入(prefix,经 lmwm_hint_proj)。
>   **与 Qwen/LaWAM 侧的 WM-target 机制不同**——这里 hint 是在线由当前帧算出的 milestone latent。
> 协议: East-H20(A0/A1)+ North-H20(A2)单卡, osmesa, 每 (臂×套件) n=3 seed × 50/10 trials。
> 臂: A0=无 hint(baseline) / A1=dinov3-base hint / A2=so400m hint。

## 1. Per-suite SR(n=3 seed 均值)

| 套件 | A0 | A1(dino) | A2(so400m) | Δ1=A1−A0 | Δ2=A2−A0 |
|---|---|---|---|---|---|
| libero_10 | 94.7 | 94.3 | 92.7 | −0.3 | −2.0 |
| libero_goal | 96.7 | 97.3 | 97.3 | +0.7 | +0.7 |
| libero_object | 98.7 | 100.0 | 100.0 | +1.3 | +1.3 |
| libero_spatial | 99.7 | 99.3 | 98.7 | −0.3 | −1.0 |
| **4 套件均值** | **97.42** | **97.75** | **97.18** | **+0.33** | **−0.24** |

## 2. 结论(诚实)

1. **聚合层完全 null**。两个 hint 编码器(dino/so400m)4 套件均值 +0.33 / −0.24,全在 ±1 SE 内。
2. **"首个正信号 +2.5" 证伪**。早期 libero_10 n=2 时 A1 +2.5,加到 n=3 回落到 **−0.3**——纯种子噪声(一个 seed 恰好 97)。
3. **根因 = LIBERO 对 pi05 饱和**:40 个 task 里 30 个 A0≥97;object/spatial 近天花板(98.7/99.7)。强基座 + 饱和台 → 辅助信号无发挥空间。与信息论架构判据一致(hint=f(o_t) 对最优策略零条件互信息,增益随饱和→0)。

## 3. Per-task 类型签名(唯一有方向性的层次,但脆弱)

只有 libero_10(唯一有 subgoal 结构)出现可解释方向性,按类型聚合 Δ1(dino, n=3):

| 类型 | 代表 task | Δ1 均值 | 解读 |
|---|---|---|---|
| 顺序闭合 | t2 开灶→放壶 / t3 放抽屉→关 / t5 放隔层 / t9 放微波→关 | +2.25 | milestone 语义与"明确 next 目的地"对齐 → 略帮 |
| 并列收集 | t0/t1/t7 `put both A and B in basket` | −4.33(主要 t0 −10)| 无 canonical 顺序, hint 强加 spurious ordering → 略伤 |
| 单物体 grounding | object 全 10 | +1.3 | 近天花板, 略帮不伤 |
| 空间判别 | spatial 全 10 | −0.3 | 完全惰性(milestone 目标帧对空间维度不携带区分信息)|

**注意**: 单 task ±3~10 在 n=3 下 = 0.3~1 个 rollout, 统计上脆弱; 只有**类型级聚合**的方向(顺序 ≥0 / 并列 ≤0)勉强可读, 不足以单独立论。

## 4. 对论文的意义

- milestone hint 在**两条骨干**(Qwen WM-target / pi05 directive-token)LIBERO 聚合层**都无稳健增益** → LMWM 价值不能靠 LIBERO 立。
- "aggregate null" ≠ "无用": libero_10 内顺序(+)/并列(−)**相互对冲** → 这是可证伪的机制主张(符号由任务子目标顺序性决定),但要在**未饱和台**(RoboTwin / 长程任务链)上才能钉死。
- 主战场转 RoboTwin(P2 已见 +2.9,未饱和)+ 机制性论证。参见 [[PAPER_CORE_final_goal_2026-07-27]] 的信息论架构判据、INVESTIGATION 的三方机制、RESULTS_robotwin_P2。

## 5. 数据位置

- East-H20(A0/A1): `lmvla/lmwm/data/pi05_libero_eval/suiteev_{A0,A1}_-{suite}-s{0,1,2}_*/suites/{suite}/summary.json`
- North-H20(A2): North-E 同路径, 已 tar-拉回 gf0。
- ckpt: `pi05_libero_a{0,1,2}_prefix_bj/.../29999`。
