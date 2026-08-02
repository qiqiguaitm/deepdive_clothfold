#!/bin/bash
# crave_value_dagger_validation_plan §6 Phase B: build + discretize pipeline
# ⚠️ 在 gf3 上执行 (North-E vePFS 上有 labels + 源数据)
# Run: bash train_scripts/kai/data/run_build_discretize_chunk001_crave.sh
set -euo pipefail

KAI0_ROOT="${KAI0_ROOT:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0}"
cd "$KAI0_ROOT"

echo "=== Phase B: Build A_v4_chunk001_dagger_crave_labeled ==="
echo "KAI0_ROOT=$KAI0_ROOT"
echo "Step 1/3: Build dataset (387 base + 387 dagger + stage_progress_gt)..."

.venv/bin/python train_scripts/kai/data/build_chunk001_dagger_crave_labeled.py 2>&1 | tee /tmp/build_chunk001_crave.log
echo "Build done."

echo ""
echo "Step 2/3: Discretize advantage (top-30% binary)..."
.venv/bin/python stage_advantage/discretize_advantage.py \
    --data data/Task_A/self_built/A_v4_chunk001_dagger_crave_labeled \
    --advantage-source stage_progress_gt \
    --discretion-type binary --top 0.3
echo "Discretize done."

echo ""
echo "Step 3/3: Verify output..."
TRAIN_DIR="data/Task_A/self_built/A_v4_chunk001_dagger_crave_labeled"
echo "  episodes: $(wc -l < "$TRAIN_DIR/meta/episodes.jsonl")"
echo "  frames: $(python -c "import json; info=json.load(open('$TRAIN_DIR/meta/info.json')); print(info['total_frames'])")"
echo "  norm_stats: $(python -c "import json; s=json.load(open('$TRAIN_DIR/norm_stats.json')); print(list(s.keys()))")"

# Verify discretize produced advantage prompt
if grep -q "Advantage: positive" "$TRAIN_DIR/meta/tasks.jsonl"; then
    echo "  discretize: OK (advantage prompt found)"
else
    echo "  discretize: FAILED (no advantage prompt)" >&2
    exit 1
fi

echo ""
echo "=== Phase B DONE. 数据已就绪于 North-E. ==="
echo "下一步: 如需在 cnsh robot-task 训练, 需 cp 到 cnsh vePFS:"
echo "  rsync -avP $KAI0_ROOT/$TRAIN_DIR/ /vePFS/tim/workspace/deepdive_kai0/kai0/$TRAIN_DIR/"
echo "或直接在 gf3 提交 cnbj 16 卡训练 (YAML: pi05_v4_awbc_crave_dagger_cnbj_16gpu.yaml)"
