# RoboTwin 均衡重处理管线(含 stack)—— 状态与阶段(2026-07-26)

> 目的: 修 milestone pairs curation bug(stack 从没进过特征/pairs, 见 `RESULTS_robotwin_P2` 附2.2),
> 重建含 stack 的均衡数据 → 重训 LaWM vs LMWM → 验证"milestone 帮子目标任务(stack 正例 / hammer 负例)"。
> **数据全本地(源 323G), 无下载; 成本=重处理+重训。**

## 根因回顾
特征提取(`robotwin_dinov3base` pooled 0~4999 / grid 672~4949)+ 帧缓存(`frame_cache_jpeg256` 0~4999)
**都只跑了前 5000 ep**。stack 在 ep 11082~26399, 整段没帧缓存→没特征→没 pairs。hammer 另因 r-field
分组(MIN_EP=5 按语言指令级 task_index)被过滤到只剩 5。

## 阶段与状态

| 阶段 | 脚本 | 状态 |
|---|---|---|
| **0 帧缓存** | `robotwin_frame_cache_build.py`(本次新写, pyav 解 AV1→256 jpeg npz) | 🟡 运行中: 700 stack ep(11082~26399 均匀采), 4 CPU 分片 |
| **1 特征提取** | `robotwin_dinov3base_extract.py`(pooled) + `_grid_extract.py`(grid), 2×A100 分片 | ⏳ 驱动 `rt_feat_driver.sh` 自动接(等阶段0完) |
| **2 pairs 重建** | `p1_robotwin_rvalley_pairs.py` | ⏳ **需改**: MIN_EP=5 按语言指令分组会把 stack 也过滤; 要改成**按规范任务分组**或降阈, 否则 stack 仍拿不到 milestone |
| **3 target_compact + v30 子集** | `p1_build_target_compact` + `build_robotwin_v30_subset.py` | ⏳ 均衡采样(stack+hammer+ranking+handover 各 ~500-700) |
| **4 重训** | `robotwin_{baseline_lawm,lmwm_dual2q}_balanced_cnsh_8a100.yaml` | ✅ **提交 cnsh**(2026-07-26): baseline `t-20260726232414-glgks` / dual2q `t-20260726232417-fwk7g`, 8×A100, 20k, 零同步(数据在/vePFS=cnsh), 唯一变量=LMWM 注入 |

## Stage0-3 完成(2026-07-26)
- 均衡 pairs `robotwin_milestone_balanced/pairs.npz`(1000ep=200×5, 335624对) + target_compact 1.13GB + `robotwin2_lmwm_balanced_v30` 子集全就位。
- data_mix `robotwin2_lmwm_balanced` 已注册 mixtures.py。
- **关键 vePFS 拓扑**: gf0 本机 /vePFS = cnsh vepfs-cnsh075262e1f815 → 本地建的数据/代码直接可被 cnsh 作业读, **零同步**(区别于 North-E 要 scp)。cnsh 提交经 submit_yaml.py 按队列名 robot-task 自动路由 cn-shanghai。

## 均衡集设计
6 规范任务, 目标各 ~500-700 ep:
- **stack**(正例, 清晰子目标): 700(本次新抽) —— 源 1169 可用
- **hammer**(负例, 无子目标): 550(特征已有) —— milestone 管线应自然给不出 pairs(预测)
- ranking_rgb / ranking_size: 各 550(特征已有)
- handover: ~550(从 6730 子采; 特征已有)

## 关键待办(阶段2 前必做)
- 改 `p1_robotwin_rvalley_pairs.py` 的任务分组: 现按 robotwin `task_index`(语言指令级, 每变体<5ep→被 MIN_EP 丢)。
  改为**按规范任务归类(hammer/stack/ranking/handover)分组**做 r-field, 才能给 stack 足够的 cross-ep 复现算 milestone。
- hammer 若归类后仍无清晰子目标 → milestone 稀疏是**预期**(负例), 保留其 action 数据训练即可。

## 监控
- 帧缓存: `fc_stack_shard{0..3}.log`(标记 FRAME_CACHE_DONE)
- 特征驱动: `rt_feat_driver.log`(标记 STACK_FEAT_DONE + 覆盖核对)

## ⭐ Stage2 结果(2026-07-26): 干净的结构度梯度 = 论点判决轴

均衡 pairs(`robotwin_milestone_balanced/pairs.npz`, 1000ep=200×5, CAP=200 子采样, GPU cdist):

| 规范任务(id) | milestone 段数中位 | 范围 | 结构度 |
|---|---|---|---|
| **hammer**(0) | **1** | [1,3] | ⭐无子目标=负例 |
| handover(5) | 2 | [1,7] | 弱 |
| **stack_two**(1) | **3** | [1,7] | ⭐清晰子目标=正例(旧bug下=0!) |
| ranking_rgb(3) | 4 | [2,9] | 强 |
| ranking_size(4) | 4 | [2,9] | 强 |
| stack_three | (4ep<MIN_EP, 丢) | | |

335624 对。**梯度 hammer1<handover2<stack3<ranking4** = 若 LMWM 帮子目标任务, 增益应随此梯度递增。
坑修记录: ①分组改语言指令级→规范任务级(否则 stack 仍被 MIN_EP 丢); ②大组全帧 cdist 43GB→CAP=200
子采样; ③RoboTwin ep 长单线程 cdist 太慢→GPU(A100)。新脚本 `p1_robotwin_rvalley_pairs_balanced.py`。
Stage1.5: grid 补抽 balanced 缺的 398ep(hammer 198/ranking 127/handover 73; stack 已全)。

## 预测(先于重训)
- **stack: LMWM > LaWM**(子目标结构强, milestone 应帮) —— 论点正例。
- **hammer: LMWM ≈ LaWM**(无子目标, milestone 无从帮/无 pairs) —— 论点负例/对照。
- ranking/handover: 数据足, 双方高, 差异小。
若 stack 上 LMWM 不赢, 论点(milestone 帮子目标任务)在 RoboTwin 侧证否。
