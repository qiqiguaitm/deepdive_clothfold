# P1-F: 残差臂补齐 goal/object → 残差 4-suite 全图(2026-07-28)

> 动机: 残差修复此前只有 libero_10(+1.55)与 spatial(94.4) 两个套件, 无法与
> `RESULTS_libero_4suite_2026-07-25.md` 的 **4-suite 聚合口径**对齐。审稿人问"残差是否在完整
> 基准上翻正"时, 缺 goal/object 就无法回答。本次补测两个饱和套件, 补完全图。

## 本次产出(job t-20260728150054-9bnbb, North-H20 4卡×4seed, osmesa)

ckpt: `20260724_071210+lmwm_2q_resid_noTs_cnbj_volc/checkpoints/steps_12500_pytorch_model.pt`

| 套件 | 残差臂 SR | 各 seed |
|---|---|---|
| libero_goal | **97.80 ± 0.28** (n=4) | 97.8 / 98.0 / 98.0 / 97.4 |
| libero_object | **99.65 ± 0.19** (n=4) | 99.8 / 99.4 / 99.6 / 99.8 |

## ⭐ 残差 4-suite 全图(与 2026-07-25 表同口径)

| 套件 | nowm(LaWAM) | dual2q(绝对) | **残差** | Δ 残差 vs 绝对 | Δ 残差 vs nowm |
|---|---|---|---|---|---|
| libero_10 | 94.55±0.50(osmesa) | 95.2† | 95.75† | +0.55 | +1.20 |
| libero_goal | 98.3 | 98.3 | **97.80** | **−0.50** | −0.50 |
| libero_object | 99.8 | 99.5 | **99.65** | +0.15 | −0.15 |
| libero_spatial | 96.3 | 93.3 | 94.4 (n=8) | **+1.10** | **−1.90** |
| **4-suite 聚合** | **97.24** | **96.58** | **96.90** | **+0.32** | **−0.34** |

† 绝对/残差的 libero_10 仍为 egl; nowm 已 osmesa 补齐(94.55±0.50,n=4,group `nowm_l10_osmesa`)。

† libero_10 三个数均为 egl 协议(与 goal/object/spatial 的 osmesa 不同源), 沿用 07-25 表的既有
caveat; 聚合含该混协议项, 不可当精确显著性, 只作方向判读。

## 判读(诚实)

1. **残差是"部分修复", 不是翻正**: 绝对臂的净负 −0.62 被减半到 **−0.30**, 但**仍低于
   无 WM 的 nowm baseline**。论文不能写"残差修复了 LMWM 的基准级伤害", 只能写
   "残差回收了约一半净负(+0.32), 主要来自 spatial(+1.10)与 libero_10(+0.55)"。
2. **剩余赤字全在 spatial**: 残差 spatial 94.4 vs nowm 96.3 = **−1.9**(绝对臂是 −3.0)。
   → 与"梯度污染共享编码器"机制一致: 残差只去冗余(改靶子幅度), **不切断梯度通路**,
   所以污染残留。真正对症的是 **梯度隔离(shapeB detach)**: shapeB spatial 95.85 vs nowm
   96.5 = −0.65, 显著优于残差的 −1.9。
3. **两个饱和套件上残差轻微掉点**(goal −0.50 / object −0.15)。goal 的 −0.50 相对 seed std
   0.28 不算噪声, 但绝对量 <1.5pt, 按既定纪律**不作显著性声称**, 只记录方向。
4. **推论 → 下一步**: 若 shapeB 的 goal/object 也不掉(已提 4 个单卡 job 到 East-H20,
   group `shapeB_go`), shapeB 的 4-suite 聚合有望首次 **≥ nowm 97.20** —— 那才是
   "修复"够格的说法, 也是论文主张应该挂靠的臂。**残差降级为"精度旋钮/半修复"**,
   梯度隔离升格为主修复。

## 复现

```
聚合: NE/lmvla/lawam/results/eval_runs/libero/resid_goalobj_x4/seed*/suites/*/summary.json
      (键为 total_episodes/total_successes, 非 aggregate_SR —— yaml 内联聚合脚本因此打印空表)
```
