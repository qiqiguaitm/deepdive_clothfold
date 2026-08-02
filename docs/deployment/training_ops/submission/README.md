# submission/ — 训练任务提交 (2 路径)

> **场景**: 在 deepdive_kai0 项目里有 2 条互补的"提任务"路径, 本目录每条一份文档。
> ⚠️ **uc01/02/03 集群已彻底停用 (2026-05-18 退役)** — 原 `uc_cluster_jobs.md` 已移到 [`../../../backup/`](../../../backup/README.md)。现在生产训练只用 Volc ML Platform + gf0 控制平面。

## ⭐ 提交前检查清单 (Pre-Submit Checklist)

> 提**任何**新训练任务前逐项确认。前 3 项是迁移 / git-pull 后新增的硬约束, 漏了会读错数据 / 跑旧 config / norm 报错。

1. **数据在 `self_built` 规范位置** — gf0/gf3 已统一为 `kai0/data/Task_A/`:
   - 构建数据集 → `self_built/<name>/`;原始采集 base → `vis_base/`;HF 官方 → `kai0_base/kai0_dagger/kai0_advantage/`。
   - config.py 的 `repo_id` / `repo_ids` / `inline_eval_val_root` 指向对应机器路径(gf0 `/vePFS/...`、gf3 `/vePFS-North-E/...`)。
   - 规范详见 `../storage_and_env.md §2.3` + `train_scripts/kai/data/README.md`。

2. **norm_stats 已算** ⚠️ — `train.py` **不自动算** norm_stats(只 `shutil.copy`)。提交前必须在数据所在机器跑:
   ```bash
   python scripts/compute_norm_states_fast.py --config-name <config>
   ```
   否则 Normalize transform 会用错/缺失统计(详见 `training_pitfalls_common.md`)。

3. **config 已 commit + push** ⚠️ (gf3 关键) — gf3 由 **1-min git pull cron 镜像 GitHub main (`reset --hard`)**。
   - 改完 `config.py` 等代码 **必须在 gf0 `git commit && git push origin main`**, 等 ~1 分钟让 gf3 pull 到, 再提交训练。否则 gf3 跑的是**旧 config**(路径/超参不一致 → 崩或读错数据)。
   - **不要直接在 gf3 改代码**(会被下次 reset 覆盖)。gf0 本地即 main 源, 改完即时生效。

4. **init ckpt 在位** — `weight_loader` 指向的 base ckpt(如 `base_init_ckpts/pi05_base/params`、`checkpoints/Task_A/mixed_1/params`)在目标机存在。

5. **queue 有余量 + 镜像/挂载正确** — `mlp job list` 查目标 queue 空闲 GPU(见 `gf0_control_plane.md §5.6.c.2`);`ImageUrl` 拼写正确(`cn-beijing` 别拼成 `bejing`);cn-beijing 队列 vePFS 必须配 `SubPath: /vis_robot`。

6. **ckpt/log 落地路径** — 单机训练走 symlink trick 落本地盘(**别直接写 NFS/vePFS 的 `checkpoints/` 真实路径**);volc 任务写 vePFS `checkpoints/<config>/<exp>/`, 日志重定向到 vePFS `logs/`。

## ⚠️ 踩坑经验 (提交/排障必读)

| 文档 | 范围 |
|---|---|
| [`training_pitfalls_common.md`](training_pitfalls_common.md) ⭐ | **跨集群共性坑** — norm_stats 不自动算 / 绝对 repo_id 被新 hub 拒 / 数据集视频目录命名 / init 按 size 校验 / TOS 嵌套 / eval prompt 默认错 / inline-eval 静默失败 / config 先 push。文末附"一个新数据集→提交训练完整前置链"7 步速查 |
| [`volc_ml_platform.md`](volc_ml_platform.md) §"Volc 特有踩坑" | Volc cnbj/cnsh — 卡 Deploying=资源被占(gang-sched)/镜像缓存 vs 多机 tradeoff / VOLC_REGION 必设 / SubPath 否则 403 / Status.State 字段 / 多机 orbax race |

## 2 路径对比

| 路径 | 适用场景 | 状态 |
|---|---|---|
| **`volc_ml_platform.md`** | 提 Volc ML Platform 集群任务 (cn-beijing Robot-North-H20 / cn-shanghai robot-task), 16 卡 + 集群 RDMA | 主要生产路径 |
| **`gf0_control_plane.md`** ⭐ | 在 gf0 一台机器上统一管理 Volc 任务 (查/停/详情/批量提交) | 日常运维推荐 |

## 资源感知任务队列

剩余实验使用 `train_scripts/kai/volc/resource_aware_scheduler.py` 常驻调度，不预先向已满队列堆积任务。任务定义在 `resource_scheduler_queue.json`，运行状态和最新资源快照分别写入：

- `logs/resource_scheduler_state.json`
- `logs/resource_scheduler_snapshot.json`
- `logs/resource_scheduler.log`

当前资源边界：北京 `Robot-North-H20` 严格限制主身份最多 20 GPU；上海
`Robot-East-H20` 为 8 H20，`robot-task` 为 32 A100 且不设额外个人软上限。
`robot-task` 按任务所需的名义空卡数尝试提交，以便利用最后 4/8 卡；若因节点碎片
无法调度，任务会在候选配置的 2--5 分钟 queue timeout 后撤回，并且只有队列活跃
卡数下降后才重试，不会周期性堆积排队任务。同时监控 gf1 的 8 GPU 和本开发机的
2 GPU。本地候选任务只在对应 GPU 显存均低于 1 GiB 时启动。

北京还支持一个显式启停的备用 credential profile。密钥仅保存在仓库外、权限为
`0600` 的 `~/.volc/credentials.scheduler-backup`；开关位于同样为 `0600` 的
`~/.volc/scheduler-backup.conf`。只有开关为 `enabled = true`、主身份已实际占满
20 GPU、北京物理队列没有排队且仍有足够空卡时，调度器才会使用备用身份提交。
备用身份提交的 attempt 会记录 `credential_profile=backup`，后续查询和停止也必须使用
同一身份。将开关改为 `false` 后，调度器不再读取备用密钥、不查询备用身份，也不提交
新任务；主身份的 20 GPU 上限始终保留。

个别高优先级任务可设置候选级 `min_dispatch_free`、`queue_timeout_seconds` 和
`retry_cooldown_seconds` 做受控放置探针。isolation 第二训练种子在恰好 8 张名义空卡时允许
尝试一次；120 秒仍未调度就自动撤回并冷却一小时，不形成长期平台排队。
`robot-task` 或 East 出现任何用户的排队任务时不再新增提交，待队列清空后再按实际空闲卡数匹配，避免继续扩大平台排队。

同一任务在同一资源上的运行时/模板失败最多重试 3 次，之后该资源会写入任务状态的
`exhausted_resources` 并自动尝试其他候选。排队超过 5 分钟的主动回收和平台配额不足属于瞬时容量问题，不计入该失败上限。
gf1/本地 launcher 还会同时检查状态文件与 PID；状态永久停留在 `RUNNING` 但 launcher 已消失时自动回收，防止任务状态假运行。实际 GPU 未释放时，资源门禁仍禁止重复启动。

调度器不把平台 `Completed` 单独当作实验完成：任务可声明 checkpoint/summary glob 和最低数量，终态产物不足时自动重试。相同资源上的失败默认冷却 15 分钟，冷却期优先尝试其他候选资源；多产物评测的计数两小时不变化会写入停滞告警。当前计数、最后变化时间和 stale 秒数都记录在 snapshot 的 `scheduler_tasks` 中。
调度器每 60 秒轮询一次，并通过 `logs/resource_scheduler.lock` 的非阻塞文件锁保证只有一个实例拥有提交权。
`resource_scheduler_queue.json` 每轮都会重新读取；运行期间追加任务不需要重启调度器，新条目会在下一轮自动初始化状态并参与资源匹配。

```bash
tmux new-session -d -s resource-aware-scheduler \
  "bash -lc 'cd /vePFS/tim/workspace/deepdive_kai0; source ~/.bashrc; \
  exec kai0/.venv/bin/python train_scripts/kai/volc/resource_aware_scheduler.py --interval 60 \
  >> logs/resource_scheduler.nohup 2>&1'"

tail -F logs/resource_scheduler.log
```

## 文件清单

| 文件 | 行数 | 用途 |
|---|---|---|
| [`training_pitfalls_common.md`](training_pitfalls_common.md) ⭐ | ~76 | 跨集群共性踩坑 (数据/init/eval/config) + 新数据集→提交 7 步前置链 |
| [`volc_ml_platform.md`](volc_ml_platform.md) | ~230 | Volc YAML/SDK 模式 + 16 卡 H20 YAML 配置要点 + region/queue mapping + image_cr + "Volc 特有踩坑" |
| [`gf0_control_plane.md`](gf0_control_plane.md) | ~264 | gf0 安装 volcengine SDK / mlp CLI 速查 / queue mapping / 镜像选择 / vsubmit 工具 |

## 按需求找文件

| 你想做什么 | 去 |
|---|---|
| 提 Volc 任务但还没在 gf0 上设置 | volc_ml_platform.md (基础 SDK + YAML) |
| 用 mlp CLI 列/停/详情查任务 | gf0_control_plane.md (CLI 速查) |
| 批量提交多个 YAML 任务 | gf0_control_plane.md (vsubmit + SDK auto-submit) |
| 知道 cn-beijing / cn-shanghai 哪个 queue 跑哪种任务 | volc_ml_platform.md 或 gf0_control_plane.md (queue mapping 表) |

## 跨场景跳转

- 提任务前需要确认数据/ckpt 在位 → `../storage_and_env.md` + `../data_sync_tos.md`
- 服务器全景 / 单机 quick start → `../overview.md`
- SSH 设置前置 → `../ssh_and_credentials.md`
- ⚠️ uc 集群历史文档 → [`../../../backup/`](../../../backup/README.md)
