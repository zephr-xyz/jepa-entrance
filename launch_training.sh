#!/bin/bash
set -e
PYTHON=/opt/pytorch/bin/python3
WORK=/home/ubuntu/jepa-entrance
TILES=/home/ubuntu/embedding-tiles-data/z14
BUILDINGS=/home/ubuntu/buildings.json
CORRECTIONS=/home/ubuntu/entrance_corrections.json
GT_LABELS=$WORK/ground_truth_labels.json
S3_INDEX=/home/ubuntu/s3_sequence_lookup.json
CACHE=/home/ubuntu/dataset_cache
CHECKPOINTS=/home/ubuntu/checkpoints

echo "=== Step 1: Build S3 sequence index (if not cached) ==="
if [ ! -f "$S3_INDEX" ]; then
    $PYTHON -u $WORK/build_s3_index.py \
        --tiles-dir $TILES \
        --output $S3_INDEX
fi
echo "S3 index: $(python3 -c "import json; print(len(json.load(open('$S3_INDEX'))), 'entries')")"

echo "=== Step 2: Prepare dataset (all samples, RTK in val) ==="
$PYTHON -u $WORK/prepare_dataset.py \
    --tiles-dir $TILES \
    --buildings-json $BUILDINGS \
    --corrections-json $CORRECTIONS \
    --ground-truth-labels $GT_LABELS \
    --s3-index-cache $S3_INDEX \
    --output-dir $CACHE \
    --max-images-per-poi 15 \
    --val-fraction 0.15

echo "=== Step 3: Train JEPA ==="
$PYTHON -u $WORK/train.py \
    --data-dir $CACHE \
    --output-dir $CHECKPOINTS \
    --epochs 300 \
    --batch-size 32 \
    --lr 5e-4 \
    --d-latent 128 \
    --lambda-sigreg 0.05 \
    --mu-entrance 10.0 \
    --warmup-epochs 20 \
    --log-interval 10

echo "=== Step 4: Sync results to S3 ==="
aws s3 cp $CHECKPOINTS/ s3://zephr-mapillary-cache/jepa-entrance-v3-singleedge/ --recursive
aws s3 cp $CACHE/train_manifest.json s3://zephr-mapillary-cache/jepa-entrance-v3-singleedge/train_manifest.json
aws s3 cp $CACHE/val_manifest.json s3://zephr-mapillary-cache/jepa-entrance-v3-singleedge/val_manifest.json

echo "=== Done ==="
