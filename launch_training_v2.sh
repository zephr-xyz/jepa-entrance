#!/bin/bash
# Launch JEPA entrance prediction training v2 with RTK ground truth labels
# Run this on EC2 g4dn.xlarge with PyTorch DLAMI
set -e

PYTHON=/opt/pytorch/bin/python3
WORK=/home/ubuntu/jepa-entrance
TILES=/home/ubuntu/embedding-tiles-data/z14
BUILDINGS=/home/ubuntu/buildings.json
CORRECTIONS=/home/ubuntu/entrance_corrections.json
GT_LABELS=/home/ubuntu/jepa-entrance/ground_truth_labels.json
S3_INDEX=/home/ubuntu/s3_sequence_lookup.json
CACHE=/home/ubuntu/dataset_cache_v2
CHECKPOINTS=/home/ubuntu/checkpoints_v2

echo "=== Step 1: Build S3 sequence index (if not cached) ==="
if [ ! -f "$S3_INDEX" ]; then
    $PYTHON $WORK/build_s3_index.py \
        --tiles-dir $TILES \
        --output $S3_INDEX
fi

echo "=== Step 2: Prepare dataset with RTK labels ==="
$PYTHON -u $WORK/prepare_dataset.py \
    --tiles-dir $TILES \
    --buildings-json $BUILDINGS \
    --corrections-json $CORRECTIONS \
    --ground-truth-labels $GT_LABELS \
    --s3-index-cache $S3_INDEX \
    --output-dir $CACHE \
    --val-fraction 0.15

echo "=== Step 3: Train JEPA v2 (smaller model, stronger entrance loss) ==="
$PYTHON -u $WORK/train.py \
    --data-dir $CACHE \
    --output-dir $CHECKPOINTS \
    --epochs 300 \
    --batch-size 32 \
    --lr 1e-3 \
    --d-latent 64 \
    --lambda-sigreg 0.05 \
    --mu-entrance 10.0 \
    --warmup-epochs 20 \
    --log-interval 10

echo "=== Step 4: Evaluate ==="
$PYTHON -u $WORK/evaluate.py \
    --data-dir $CACHE \
    --checkpoint $CHECKPOINTS/best_model.pt \
    --d-latent 64

echo "=== Step 5: Sync results to S3 ==="
aws s3 cp $CHECKPOINTS/ s3://zephr-mapillary-cache/jepa-entrance-v2/ --recursive
aws s3 cp $CACHE/train_manifest.json s3://zephr-mapillary-cache/jepa-entrance-v2/train_manifest.json
aws s3 cp $CACHE/val_manifest.json s3://zephr-mapillary-cache/jepa-entrance-v2/val_manifest.json

echo "=== Done ==="
