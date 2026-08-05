# train_scripts/kai/data — 数据集构建脚本

这个目录采用“共享内核 + 少量示例 + 本地 recipe”的方式维护。目标不是把每次数据快照的脚本都提交到 Git，而是保留可复用能力、关键实验的可复现入口，以及覆盖不同构建类型的规范示例。

## 什么应该提交

- 可复用的校验、重编号、统计、视频处理和元数据生成逻辑；
- 已用于正式训练、需要长期复现的构建入口和 split manifest；
- 能代表一种新处理方式的示例，例如固定验证集、动作维度投影、视频裁剪；
- 对共享构建逻辑的自动化测试。

仅包含某天路径、临时 episode 列表或一次性阈值调整的脚本，不应默认提交。把它放在 `local/` 或 `local_specs/`；这两个目录已被 `.gitignore` 排除。若 recipe 产生了正式训练数据，只提交精简后的最终入口或 manifest，并在训练 plan 中记录数据版本、输入摘要和命令。

## 推荐结构

```text
train_scripts/kai/data/
├── lerobot_build.py                  # 共享的 canonical LeRobot 构建内核
├── build_task_n_base_clean_822_split.py  # 固定 holdout 的正式示例
├── build_task_n_v5_343_joint14.py    # 多来源发现与 32→14 维投影参考
├── build_no_release.py               # 视频裁剪/重编码参考（历史综合脚本）
├── tests/                            # 共享逻辑测试
├── local/                            # 本地一次性 Python recipe，不入 Git
└── local_specs/                      # 本地数据清单、路径和参数，不入 Git
```

新增构建任务时，先写一个薄 wrapper 调用 `lerobot_build.py`。wrapper 只负责数据集特有的 episode 选择、split 规则及元数据补充；不要再次实现 parquet 重编号、`episodes_stats.jsonl`、`info.json` 或视频链接逻辑。

目前的共享内核面向“来源已经是 canonical、单 chunk 的 LeRobot 数据集”。原始采集目录发现、动作维度变换以及需要解码/重编码的视频处理仍属于专用 transform，后续在出现第二个调用者时再抽成独立模块。

## 提交前检查

一个新脚本只有满足以下任一条件才建议进入 Git：

1. 支撑正式训练且仅靠 manifest 无法复现；
2. 同一逻辑已经或预计会被第二个任务复用；
3. 引入新的数据变换或安全校验，值得作为参考实现。

同时应满足：路径可通过参数或环境变量覆盖；输出先写 staging 目录再原子换入；拒绝静默覆盖；校验 episode/frame/video 数量和向量维度；正式 split 使用稳定身份而不是当前位置；共享逻辑有测试。

旧脚本暂不批量删除或移动，因为训练文档可能引用它们。修改旧流程时应逐步迁移到共享内核；确认没有引用后，再单独做 legacy 清理。

## 数据集存放规范（强制）

所有新构建的数据集一律输出到 `self_built/` 下：

```text
<KAI0_DATA_ROOT>/data/Task_A/self_built/<dataset_name>/
```

- 不要直接输出到 `Task_A/` 根目录。根目录只保留非构建产物：`vis_base/` 原始采集，以及 `kai0_base/`、`kai0_dagger/`、`kai0_advantage/` HF 官方数据。
- `config.py` 中训练 config 的 `repo_id` 也应指向 `self_built/<name>`。
- 完整规范见 `docs/deployment/training_ops/storage_and_env.md` 的 2.3 节。

标准输出路径写法：

```python
import os
from pathlib import Path

root = Path(os.environ.get("KAI0_DATA_ROOT", "/vePFS/tim/workspace/deepdive_kai0/kai0"))
dataset_name = "my_new_dataset"
destination = root / "data" / "Task_A" / "self_built" / dataset_name
```

脚本若提供 `--out` 或 `--dst`，默认值也必须落在 `self_built/<name>`。gf0 的默认根目录是 `/vePFS/tim/workspace/deepdive_kai0/kai0`，uc 是 `/data/shared/ubuntu/workspace/deepdive_kai0/kai0`。

### 数据源约定

| 用途 | 路径 |
|---|---|
| 原始采集 base（按 `<date>-v2/`） | `Task_A/vis_base/` |
| HF 官方 base/dagger | `Task_A/kai0_base/`、`Task_A/kai0_dagger/`（uc 也可能位于 `dataset/Kai0_official/Task_A/`） |
| 跨 region 同步源 | `tos://transfer-shanghai/KAI0/Task_A/...`，详见 `docs/deployment/training_ops/data_sync_tos.md` |

## 历史脚本状态

- 已按 `self_built/` 输出的 Task_A 脚本包括 `build_A_0423_0527.py`、`build_vis_v2_full.py`、`build_vis_v2_merged.py`、`build_task_a_mix_*.py`、`build_task_a_new_*.py`、`build_task_a_pure_*.py` 和 `build_xvla_exp1_hard_merged.py` 等。
- `build_task_a_*_split.py` 中部分脚本是 `DST = SRC` 的原地重构，不产生新数据集，属于例外。
- `build_val_kai0_official.py`、`label_dagger_positive.py`、`split_advantage_stage.py` 及其他 Task 的 prepare 脚本不适用 Task_A 输出规范。
- 输出到顶层 `Task_A_*` 的 `build_task_a_mixed.py`、`build_task_a_visrobot01_only.py`、`build_task_a_vis_base.py` 已弃用，不要作为新脚本模板。

## 自动同步和工具

`sync_vis_base_from_tos.sh` 用 `tosutil cp -r -u` 将 TOS 的 Task_A base 按日期增量同步到 `vis_base/`；安装、cron 和日志配置见 `docs/deployment/training_ops/data_sync_tos.md`。

非数据集构建工具包括 `compute_delta_norm_stats_fast.py`、`gen_episodes_stats.py`、`generate_episodes_stats.py`、`get_episodes.py`、`from_tos_file.py`、`to_tos_file.py`、`redownload_bad_videos.py`、`fix_data.py`、`pack_inference_ckpt.py` 和 `split_advantage_stage.py`。
