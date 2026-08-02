# 本地数据集路径整理（deepdive_kai0）

> 2026-07-15 整理。路径均相对仓库根 `deepdive_kai0/`（`$KAI0_DATA_ROOT` 的上一级）。
> 尺寸为本机快照。**软链**：`lmvla/kai0 → ../kai0`（同一份，勿重复统计）。
> 📇 **全量可检索清单（全部 269 个数据集，逐条路径+ep）→ [`datasets_index_full.md`](datasets_index_full.md)**。本文件是分类概览，全量查条目用索引文件。

## 0. 多机数据根（setup_env.sh 自动探测 `KAI0_DATA_ROOT`）

| 机器 | KAI0_DATA_ROOT | OPENPI_DATA_HOME |
|---|---|---|
| 本机(sim/A100) | `/vePFS/tim/workspace/deepdive_kai0/kai0` | `/vePFS/tim/workspace/openpi_cache` |
| gf3 / North-E | `/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0` | `/vePFS-North-E/vis_robot/openpi_cache` |
| gsy | `/data1/tim/workspace/deepdive_kai0/kai0` | `$HOME/.cache/openpi` |

---

## 1. 机器人示教数据（kai0，Agilex 双臂，LeRobot v2.1，fps30，state[14]+action[14]）

### 1.1 Task_A · T恤折叠（主任务）— 规范原始集
| 路径 | ep | 说明 |
|---|---|---|
| `kai0/data/Task_A/kai0_base` | 3055 | 基础成功示教（CRAVE/LMWM milestone 挖掘用这份） |
| `kai0/data/Task_A/kai0_dagger` | 3457 | DAgger 干预数据 |
| `kai0/data/Task_A/kai0_advantage` | 3055 | AE advantage 标注 |

### 1.2 Task_A/self_built · 派生/合并/标注集（**53 个变体，288G**）
> 大多为实验/归档；**config.py 实际在用的活跃集**：

| 活跃数据集 | ep | 用途 |
|---|---|---|
| `self_built/vis_v2_merged` / `vis_v2_full` | 1406 | v2 合并训练集（config 主用） |
| `self_built/vis_v2_merged_val` | — | 验证集（config 引用最频繁，47×） |
| `self_built/A_v4_base_dagger` (+`_plus_freshdagger` 2512, `_old2017` 2017) | 2006 | v4 base+dagger |
| `self_built/A_new_pure_200` / `_val` | — | 纯净小集 + 验证 |
| `self_built/crave_stage_{A,B,poly,poly_mono}` | 3055 | CRAVE 进度标签数据集（A=原值/B=cummax/poly=折线） |
| `self_built/vis_awbc_merged{,_stage,_stage_interp}` | 1699 | AWBC 合并集 |
| `self_built/kai_vis_merged` / `kai_vis_s800_merged` | 7545 / 7318 | 最大合并集 |
| `self_built/awbc_v2_full` / `kai0_mixed_1_data` | 6512 | AWBC v2 / 混合 |

> 其余 ~40 个 `A_smooth800_*` / `A_0xxx_*` / `dagger_*` / `A_pure_1200` 等为历史迭代，按需查 `ls kai0/data/Task_A/self_built/`。

### 1.3 其他任务 / 原始采集
| 路径 | 大小 / ep | 说明 |
|---|---|---|
| `kai0/data/wam_fold_v1/{kairobot01,visrobot01}` | 100G / 6512+2098 | WAM 折叠 v1 原始采集（kai/vis 两机） |
| `kai0/data/wam_fold_v3/{vis_base_v3,vis_dagger_v3}/<日期>` | 25G | v3 按日期分批（base 20+ 批 / dagger 8 批） |
| `kai0/data/Task_AV1/selft_built/Task_AV1_200{,_val}` | 68G | AV1 变体 |
| `kai0/data/Task_AH1/self_built/Task_AH1_{170,val}` | 6.8G | AH1 变体 |
| `kai0/data/Task_PP/base/<日期>` | 20G | PP 任务 |
| `kai0/data/Task_P/{base,v2,val}` | 310M | P 任务 |
| `kai0/data/Task_E/{base,val}` | 5.6M | E 任务（10 个子集） |
| `kai0/data/Task_H/{KAI0_base,vis_base}/[v2/]<日期>` | 1.5K | H 任务早期小采集（12–24ep/子集，2026-04-25） |
| `kai0/data/Task_HP/{KAI0_base,vis_base}/[v2/]<日期>` | 1.5K | HP 任务早期小采集（12ep/子集） |

> 全部 269 个数据集（含所有 self_built/日期批次/val 子集）逐条见 [`datasets_index_full.md`](datasets_index_full.md)。

### 1.4 xvla 多域混合配置（`lmvla/xvla/data/`，真实目录）
数据集软链回 kai0（`mixed_hard/kai0_base` 等），配置文件 `mixed_repos_{hard,soft}.yaml`、`README_data.md`。

---

## 2. 外部 benchmark 数据集

| 路径 | ep | 格式 | 说明 |
|---|---|---|---|
| `lmvla/lawam/dataset/libero_merged_no_noops_20hz` | 1693 | LeRobot v3.0, Franka, fps20, state[8] | LIBERO 40 任务（LMWM×LaWAM 对比用）。tasks 在 `meta/tasks.parquet` |
| `lmvla/lawam/dataset/robotwin2.0` → *(软链)* `/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0` | 27500 | LeRobot, Aloha, fps50 | RoboTwin 2.0（双 benchmark 用，Task #26）。物理在 VLANeXt-main |

### datas/ 统一规范根（软链索引，2026-07-15）
所有数据集在 `datas/` 下有统一入口（均为软链，可逆）：
| datas 入口 | → 物理位置 |
|---|---|
| `datas/kai0/data` | `kai0/data`（kai0 全部任务） |
| `datas/libero_merged_no_noops_20hz` | `lmvla/lawam/dataset/libero_merged_no_noops_20hz` |
| `datas/robotwin2.0` | `/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0` |

---

## 3. 派生特征缓存（**非原始数据，可重生成**，占最大空间）

| 路径 | 大小 | 内容 | 生成脚本 |
|---|---|---|---|
| `lmvla/lmwm/data/grid_cache` | **123G** | kai0 DINOv3 grid 特征缓存 | crave 特征管线 |
| `lmvla/lmwm/data/libero_dinov3base` | 39G | LIBERO DINOv3-vitb16 grid [N,256,768] fp16 (stride2) | `p1_libero_dinov3base_extract.py` |
| `lmvla/lmwm/data/crave_sequences` | 17G | CRAVE 序列缓存 | |
| `lmvla/lmwm/data/libero_milestone` | 4.9G | `pairs.npz`（milestone 训练对）+ meta | `p1_libero_milestone_pairs.py` |
| `lmvla/lmwm/data/recurrence_graphs` | 1.6M | `<name>/recurrence_graph.npz`（prototype_table+pord） | 终版 milestone 发现 |
| `temp/` | 41G | CRAVE 特征/标签缓存：`crave_full`(8.4G) `dinov3_7b_int8`(6.3G) `crave_d3b_pca128` `crave_ae_labels` `crave_30hz_feat_v2` `aloha_tasks` 等 | `gen_final_v3.py` 等 |
| `lmvla/crave/data` | 12G | CRAVE 数据 | |

> ⚠️ 这些是**中间产物**，删了可从原始数据 + 脚本重建；备份/迁移时优先带 §1/§2 原始集，不带 §3。

---

## 4. 快速查询

```bash
# 列全部 LeRobot 数据集 + ep 数
find kai0/data lmvla/lawam -name info.json -path "*meta*" \
  -exec sh -c 'echo "$(python3 -c "import json;print(json.load(open(\"$1\"))[\"total_episodes\"])") $1"' _ {} \;
# config.py 实际引用的数据集
grep -oE "kai0/data/[a-zA-Z0-9_/]*" kai0/src/openpi/training/config.py | sort -u
```
