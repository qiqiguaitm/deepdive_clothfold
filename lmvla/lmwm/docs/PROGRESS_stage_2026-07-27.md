# LMWM 阶段性进展 + 真实定位(2026-07-27)

> 配套: PAPER_CORE_final_goal_2026-07-27 / INVESTIGATION_lmwm_architecture_flaw_2026-07-26 / RESULTS_robotwin_P2_2026-07-25 / PIPELINE_robotwin_rebalance_2026-07-26。

## 一、本阶段新建立的事实
| 发现 | 证据 |
|---|---|
| RoboTwin 旧结论作废 | curation bug: stack 0 训练数据、hammer 仅 5 样本; 旧 +2.9/+4.5 不可用 |
| 均衡数据重建完成 | 修 bug, stack 拿到 milestone(段数3); 结构梯度 hammer1<handover2<stack3<ranking4 |
| LaWAM 口径挖清 | RoboTwin 92.64 用 **27500 全量** 非单任务50demo(Clean+Randomized 100 trials); leaderboard π₀ 才46, 差550×数据 |
| **P1 三方机制 refine** | nowm 96.5 / **t+7 93.35** / milestone 90.8 / +detach 93.25(全同12500口径) |
| pi05 线复活 | gripper(inv_pm1)+egl→osmesa+去--gpu 三坑修完, A0/A1/A2 East-H20 在评 |

## 二、P1 机制定版(refine)
- **通用 WM 伤害(~−3)**: t+7 **也**伤 spatial(−3.15), 非 milestone 独有。任何 WM 辅助塑形共享编码器 → 与细粒度空间判别抢容量。
- **milestone 额外伤害(~−2.5)**: 退化靶子的梯度污染; **detach 精确移除**(milestone+detach 93.25 ≈ t+7 93.35)。
- **libero_10(子目标): WM 帮**(t+7 95.3 / milestone 95.2 均 > nowm 94.3)。
- 复现 gap: LaWAM 论文 t+7 spatial=99.4(帮), 我们 93.35(伤), 差−6=12500步 vs 全量+基线口径; 内部 A/B 有效, 绝对值勿对 99.4。

## 三、⭐ LMWM 真实定位(诚实, 含 sobering)
1. **LMWM 不是"更好的 WM", 是"更极端/更挑任务的 WM 辅助"**: 把通用 WM 权衡(帮子目标/伤空间)放得更大。
2. **⚠️ LIBERO 上 milestone 从不优于 t+7**: spatial 更差(detach 后才追平), libero_10 持平(95.2≈95.3)。**"变长子目标靶子"的理论吸引力在 LIBERO 没兑现。**
3. **LMWM 价值全押 RoboTwin stack**: milestone 相对 t+7 的唯一理论优势=变长子目标编码 t+7 固定近未来编不了的阶段结构; 只在子目标结构足够强/长的任务兑现。LIBERO 无此强度; **stack(pick-place-pick-place 清晰分段)是 make-or-break**。
4. **两种收场**: stack 上 milestone>t+7 → LMWM 有 niche(长程分段)+机制故事=扎实论文; milestone≈t+7 → 负结果, 论文转"子目标WM何时帮+为何不胜通用WM+detach/动态目标控伤害"的机制刻画(诚实但弱)。
5. **动态非退化目标(用户提案)**能去 milestone 额外伤害(追平 t+7), 但**要真赢 t+7 得靠"子目标结构本身有用", 非靠修伤害**——两回事勿混。

## 四、进行中的判决(各自监控)
| 实验 | 集群/ID | 答什么 |
|---|---|---|
| RoboTwin balanced A/B | 北京 vdptr/snf2v → 20k 自动 Easy+Hard eval | **LMWM 帮不帮 stack(make-or-break)** |
| 编码器塑形 A/B | 北京 5jvb5(只t+7塑形)/dswq8(都不塑形) | 验"WM塑形伤spatial"通用性(B应回~96.5) |
| full baseline 对标LaWAM | 北京 t7s8z(全量27500,100k) | LaWAM 全量协议锚 |
| pi05 A0/A1/A2 | 上海East-H20 6路 | hint 帮不帮 pi05 基座 |
| full milestone 预处理 | gf0 | 帧缓存✅完; DINOv3 特征待起(供 full LMWM) |

**下一决定性读数 = RoboTwin balanced 的 stack(Easy+Hard)。** 它出来前, LMWM 正面价值未决。
