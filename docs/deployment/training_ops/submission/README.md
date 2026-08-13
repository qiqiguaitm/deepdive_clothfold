# submission/ — 训练任务提交 (2 路径)

> **场景**: 在 deepdive_kai0 项目里有 2 条互补的"提任务"路径, 本目录每条一份文档。
> ⚠️ **uc01/02/03 集群已彻底停用 (2026-05-18 退役)** — 原 `uc_cluster_jobs.md` 已移到 [`../../../backup/`](../../../backup/README.md)。现在生产训练只用 Volc ML Platform + gf0 控制平面。

## ⭐ 提交前检查清单 (Pre-Submit Checklist)

> 提**任何**新训练任务前逐项确认。先执行第 0 项获取实时目标序列；后续各项漏掉会读错数据 / 跑旧 config / norm 报错。

0. **先运行资源路由器** — 同时提供卡数和真实数据/checkpoint 路径：
   ```bash
   python3 train_scripts/kai/volc/recommend_submission_target.py \
     --gpus 8 \
     --data-path /vePFS/tim/workspace/deepdive_kai0/kai0/checkpoints/<config>/<exp>
   ```
   只向输出中第一个 `Run now=yes` 的目标准备 YAML/命令。目标文件系统不包含
   输入时会显示 `Transfer=yes`；先完成一次经过校验的数据同步，再用目标侧路径重跑
   路由器，不能边提交边隐式搬运。

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

### 提交前目标推荐

`train_scripts/kai/volc/recommend_submission_target.py` 读取两类权威输入：

- `submission_resource_catalog.json`：资源容量、区域、队列、开发机、挂载文件系统和
  1/2/4/8/16 卡静态偏好；
- `logs/resource_scheduler_snapshot.json`：调度器每分钟生成的实时空卡、排队和
  主/备用北京身份占用。

静态卡数偏好为：

| 请求 | 基础顺序 |
|---:|---|
| 1/2 GPU | local > Robot-North-H20 > gf1 > Robot-East-H20 > robot-task |
| 4 GPU | gf1 > Robot-East-H20 > Robot-North-H20 > robot-task |
| 8 GPU | gf1 > Robot-East-H20 > Robot-North-H20 > robot-task |
| 16 GPU | Robot-North-H20 > robot-task |

基础顺序不是盲目跨区规则。路由器会对输入不在目标挂载上的候选增加跨文件系统惩罚，
并把当前空卡不足或已有排队的候选后移。例如共享上海 vePFS 上的 16 卡任务通常会优先
`robot-task`，而不是先复制到 North；北京数据的 8 卡任务则通常会直接选择 North。
16 卡输出只证明名义卡数足够，仍需检查是否存在两个完整 8-GPU 节点供 gang scheduling。

常用形式：

```bash
# 没有具体路径时显式声明数据位置
python3 train_scripts/kai/volc/recommend_submission_target.py \
  --gpus 4 --data-location north_shared

# 禁止任何跨文件系统候选
python3 train_scripts/kai/volc/recommend_submission_target.py \
  --gpus 8 --data-path /vePFS/tim/workspace/<dataset> --strict-locality

# 供调度或生成 YAML 的程序读取
python3 train_scripts/kai/volc/recommend_submission_target.py \
  --gpus 16 --data-location east_shared --json
```

资源目录中的开发机/挂载关系为：上海共享文件由当前开发机或 gf1 准备，供 gf1、
Robot-East-H20、robot-task 和本地使用；北京文件由 gsy 准备，供
Robot-North-H20 使用。脚本只显示 `primary`/`backup` profile 名称，不读取或打印密钥。
默认拒绝超过 180 秒的旧快照，避免按过期卡池状态提交。

北京主/备用身份只按 GPU 数量限制，不设置任务数量上限。当前运行配置中主身份 GPU
上限为 25，备用身份 GPU 上限为 8；运行时以快照显示的配置为准。主身份上限可在启动
调度器前配置：

```bash
export NORTH_PERSONAL_LIMIT=25
export NORTH_BACKUP_PERSONAL_LIMIT=8
```

备用身份的 GPU 上限也可持久化写入仓库外控制文件，优先于上述环境变量：

```ini
[scheduler]
enabled = true
submission_enabled = true
personal_limit = 8
```

`personal_limit` 限制备用身份的活跃加排队 GPU 数。无效或负数配置按 0 GPU
处理，禁止备用身份新派发。

快照的 `Beijing GPU Quotas` 表显示 active、queued 和 GPU limit。若存在任何不排队
目标，路由器优先选择可立即运行目标；若所有候选都必须等待，则把
`Robot-North-H20` 固定为第一排队选择，不向上海队列堆积等待任务。

常驻调度器同样强制执行该路由步骤。候选的实际执行顺序直接使用完整 router score，
综合实时可运行状态、请求卡数偏好和数据/checkpoint locality，而不是先按另一套固定
字典选中资源后再补记推荐。每个 GPU task 在调用平台提交、gf1 launcher 或本地
launcher **之前**，还会保存同一推荐逻辑的审计记录。记录写入
`logs/submission_recommendations/<task-id>/<UTC timestamp>.json`，包含完整排序、数据
位置、全局首选、任务候选内首选、最终选择和选择分析；对应 attempt 保存该文件路径。
推荐计算失败时拒绝 launch，不能静默绕过。零 GPU 的本地分析/finalizer 不需要 GPU
队列推荐。

剩余实验使用 `train_scripts/kai/volc/resource_aware_scheduler.py` 常驻调度。只要存在
可立即运行的合格资源就不会向队列堆积任务；所有候选均需等待时，才向 North 提交
持久 queue-sink 任务。任务定义在 `resource_scheduler_queue.json`，运行状态和最新资源快照分别写入：

- `logs/resource_scheduler_state.json`
- `logs/resource_scheduler_snapshot.json`
- `logs/resource_scheduler.log`
- `logs/submission_recommendations/<task-id>/*.json`

论文 GPU TODO 另由只读小时监控留档。它不提交或停止任务，实际推进仍完全由
`resource_aware_scheduler.py` 负责：

```bash
python train_scripts/kai/volc/monitor_paper_todo_hourly.py --interval-seconds 3600
```

监控固定核对当前冻结的 79 项 TG4 claim-bearing 执行节点，每小时写入
`logs/paper_todo_hourly_monitor.jsonl`，并原子更新
`logs/paper_todo_hourly_monitor_latest.{json,md}`。调度器快照超过 5 分钟未更新时记录
`degraded` 告警；只有 79 项全部为 `completed`、TG4 分析产物通过一致性校验且 TODO
完成状态已同步时，监控才自行退出。

当前资源边界：北京 `Robot-North-H20` 严格限制主身份最多 25 GPU；上海
`Robot-East-H20` 为 8 H20，`robot-task` 为 32 A100。**截至 2026-08-04，暂停向
`robot-task` 提交新任务**：控制标记为
`logs/resource_controls/robot_task_submission.disabled`。标记存在时，实时推荐仍将
`robot-task` 显示在列表末位，但标记为不可立即运行；scheduler 同样拒绝该资源的新
分派，已有任务继续运行。只有收到显式恢复指令后才移除标记。

未停用时，`robot-task` 才按任务所需的名义空卡数尝试提交；若因节点碎片无法调度，
任务会在候选配置的 2--5 分钟 queue timeout 后撤回，并且只有队列活跃卡数下降后才
重试，不会周期性堆积排队任务。同时监控 gf1 的 8 GPU 和本开发机的 2 GPU。本地候选
任务只在对应 GPU 显存均低于 1 GiB 时启动。

北京还支持一个显式启停的备用 credential profile。密钥仅保存在仓库外、权限为
`0600` 的 `~/.volc/credentials.scheduler-backup`；开关位于同样为 `0600` 的
`~/.volc/scheduler-backup.conf`。即时运行时，只有开关为 `enabled = true`、主身份
下一任务无法放入 GPU 额度、北京没有排队且仍有足够物理卡时，调度器才会使用
备用身份；持久 North 排队时也会在主身份达到相应额度后使用备用身份。
备用身份提交的 attempt 会记录 `credential_profile=backup`，后续查询和停止也必须使用
同一身份。将开关改为 `false` 后，调度器不再读取备用密钥、不查询备用身份，也不提交
新任务；主身份配置的 GPU 上限始终保留。
主身份的活跃加排队 GPU 无法容纳下一任务时，若备用身份启用且其 GPU 额度可容纳，
调度器会选择备用身份。

North queue-sink attempt 只预留对应身份的 queued GPU，不增加 active GPU
计数；进入 `Queueing` 后不受上海机会型任务使用的短 queue timeout 影响。任务真正
进入运行态前，只有显式 opt-in 的任务允许逃离 queue sink：调度器必须先在实时快照
副本上为更高优先级候选预留足够的即时卡，再使用原提交身份停止 North 排队任务。
容量不足、候选已失败耗尽或 stop 失败时均保留原任务，不会产生重复提交。
进入 `Running` 后，平台实时快照接管 GPU 占用统计。上海 `robot-task`/East 仍保持
机会型策略，排队超时后撤回并在容量变化或 cooldown 后重试。

个别高优先级任务可设置候选级 `min_dispatch_free`、`queue_timeout_seconds` 和
`retry_cooldown_seconds` 做受控放置探针。isolation 第二训练种子在恰好 8 张名义空卡时允许
尝试一次；120 秒仍未调度就自动撤回并冷却一小时，不形成长期平台排队。
未被控制标记停用时，`robot-task` 或 East 出现任何用户的排队任务也不再新增提交，
待队列清空后再按实际空闲卡数匹配，避免继续扩大平台排队。

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
| `train_scripts/kai/volc/recommend_submission_target.py` | CLI | 按卡数、实时空卡和数据本地性输出提交目标序列 |
| `train_scripts/kai/volc/submission_resource_catalog.json` | JSON | 区域、队列、卡池、开发机和 vePFS 拓扑配置 |

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
