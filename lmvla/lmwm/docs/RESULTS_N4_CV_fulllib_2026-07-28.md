# N4 — 残差 vs 绝对目标判别力 CV / 86% 冗余：从 2ep 扩到全库分布（2026-07-28）

> 目的：claim C3a "绝对子目标 86% 是当前态(冗余),真信号在残差;残差判别力 CV ≈ 绝对的 10×"。原始只算了 **2 条 episode**(Fig.2 `make_figs.py` 硬编码 abs [0.119,0.087]、resid [1.112,1.212];86%/14% 能量条)。本 N4 扩到 **全 40 任务 × 1693 episode / 137154 帧**。
> 脚本:`lmvla/lmwm/scripts/n4_cv_fulllib.py`(纯离线,零 GPU,~7min)。输出:`lmvla/lmwm/data/n4_cv_fulllib{.npz,_summary.json,_hist.json}`。

## 定义(逐字复用 + GT-pair 重建)
原始 "pi05 hint 分析" 用的是训练好的 predictor 预测的 ĝ_next。按 handoff N4 授权,用 **GT pairs 重建**(免 predictor,更干净):
- 数据:`libero_rvalley/pairs.npz`(cur_ep,cur_fi,tgt_fi,pair_task)+ `libero_dinov3base/ep{cur_ep}.npz[grid]`(与 r-场/pairs 构造同源)。
- 每帧:`h_t = pooled(grid[cur_fi])`(当前态,256 token 均池)、`g_next = pooled(grid[tgt_fi])`(绝对 milestone 目标=下一段 r-脊帧)、`r = g_next − h_t`(残差)。
- **判别力 along-trajectory CV**(两种标量,均报):
  - **变体 B(向量 CV,= 原始定义)**:`CV(v) = ‖std_t(v)‖₂ / ‖mean_t(v)‖₂`(沿 episode 帧)。
  - 变体 A(范数 CV):`std_t(‖v_t‖)/mean_t(‖v_t‖)`(辅助,数值更小)。
- **冗余度(86%)**:`mean_t cos²(g_next, h_t)` = 绝对目标被当前态解释的能量占比;informative = 1−cos²。另报 `‖r‖²/‖g‖²` 能量比。

> **定义已被 2ep 锚点校验**:变体 B 的**绝对目标 CV 全库均值 = 0.120**,精确落在原始 2ep 值 [0.119, 0.087] 之间 → 确认原始 "along-trajectory CV" 就是向量 CV。变体 A 数值差 20× 不匹配锚点,仅作旁证。

## 全库结果(变体 B 向量 CV,n=1693 episode)

| 量 | 2ep 旧值 | **全库均值** | 全库中位 | p10–p90 | 结论 |
|---|---|---|---|---|---|
| 绝对目标 CV | 0.119 / 0.087 | **0.120** | 0.121 | 0.065–0.175 | **精确复现锚点** |
| 残差 CV | 1.112 / 1.212 | **1.865** | 1.787 | 1.05–2.82 | 更高,同方向 ≫ 绝对 |
| CV 比(残差/绝对) | ~10× | **~15×**(中位比 14.7×;逐 ep 中位 15.2×,p10–p90 10.5–25×) | | | **比 claim 更强** |

- **逐 episode 100% 满足 残差 CV > 绝对 CV**(最小比 5.6×)→ 绝非 2 条 episode 的偶然。
- 冗余度 `cos²(g_next,h_t)`:**全库均值 0.948 / 中位 0.949**(informative = **5.2%**;能量比 ‖r‖²/‖g‖² = 5.4%)。**99.4%** 的 episode 冗余度 ≥ 0.86、**97.6%** ≥ 0.90。

## 诚实解读 —— **claim 维持并加固,但两个具体数字要更新**

1. **"残差 CV ≈ 绝对 10×" → 加固**。全库中位比 **~15×**(比 claim 的 10× 更强);逐 episode 全部满足、分布紧(p10–p90 10.5–25×)。绝对 CV 0.120 精确复现 2ep 锚点。核心结论稳。

2. **"86% 是冗余当前态" → 数字需更新为 ~95%(方向同,更强)**。GT-pair 全库冗余度 = **94.8%**(informative 仅 5.2%),比 2ep 的 86%/14% **更冗余**。差异来源:2ep 用的是 pi05 *预测* hint(含预测噪声,less redundant),本 N4 用 *GT* 未来 milestone 帧(与当前同 episode、更相似)。→ Fig.2 能量条建议改标 **"~95% = current state / ~5% informative"**,或注明 "predicted-hint 2ep 为 86%,GT-target 全库为 95%"。**不要继续单卖 "86%"**,否则与全库数不一致会被抓。

3. **残差 CV 绝对值从 ~1.16 升到 ~1.79**:同源、同方向,全库有更多分段/更长 episode → 变异更大,属预期。不影响 "≫ 绝对" 结论。

### Fig.2 升级(2ep → 分布)
直方图 bin 已存 `n4_cv_fulllib_hist.json`:
- 绝对 CV(0–0.30,15 bin):峰在 0.10–0.14。
- 残差 CV(0–5.5,22 bin):峰在 0.75–1.75,长尾到 5+。
- 冗余度 cos²(0.80–1.0,20 bin):峰在 0.93–0.97。
两条分布几乎不重叠 → 比 2 条柱说服力强得多。逐任务向量 CV/冗余度见 `_summary.json:per_task_median`(40 任务全部 残差 CV ≫ 绝对,冗余度 0.88–0.97)。

**一句话**:N4 加固 C3a 主张(判别力 ~15× 而非 10×,全库 100% 一致);唯一动作 = 把 "86%" 更新为 "~95%(GT)/86%(predicted-hint)",别再用单一 86% 数字。
