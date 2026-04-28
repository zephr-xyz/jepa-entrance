# JEPA Entrance Prediction

Predicting building entrance locations from Mapillary street-level imagery using a Joint-Embedding Predictive Architecture (JEPA).

**[Interactive 3D Viewer](https://zephr-xyz.github.io/jepa-entrance/)** -- ray-traced entrance predictions with occluded examples

## Overview

Given a building and its associated Mapillary images, the model predicts where the entrance is along the building's road-facing facade. The prediction is parameterized as `t ∈ [0, 1]`, representing the fractional position along the facade edge, which is then converted to geographic coordinates.

### Key Results (v3, April 2026)

Evaluated on 71 POIs with RTK-precision ground truth (centimeter-level GPS measurements of actual front door locations) in Louisville, CO and Boulder, CO:

| Approach | MAE | Median | P90 |
|----------|-----|--------|-----|
| Facade midpoint (t=0.5) | 3.58m | 2.53m | 7.78m |
| JEPA v3 | **1.73m** | **0.78m** | **4.70m** |

The JEPA model reduces MAE by 52% compared to the midpoint baseline, with a sub-meter median error.

Dataset: 1,668 samples (1,358 train / 310 val) from 4,246 Boulder County POIs. Training takes ~9 minutes (300 epochs) on an NVIDIA A10G (g5.xlarge).

## Architecture

The model follows a JEPA (Joint-Embedding Predictive Architecture) training paradigm with four components:

```
┌─────────────────────────────────────────────────┐
│                Context Encoder                  │
│  DINOv2 patch embeddings (16x1024 per image)    │
│  + Facade-t positional encoding (per patch col) │
│  + Camera pose tokens (8-dim per image)         │
│  + Self-attention across K images               │
│  → z_visual (128-dim)                           │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│               Predictor (AdaLN)                  │
│  z_visual → z_geo_hat, conditioned on facade     │
│  → Entrance Head: z_geo_hat → t_pred ∈ [0,1]     │
└──────────────────────────────────────────────────┘
                        ↕ MSE loss
┌──────────────────────────────────────────────────┐
│               Target Encoder                     │
│  Facade geometry (32-dim) + entrance_t → z_geo   │
└──────────────────────────────────────────────────┘
```

4.2M parameters. Training losses:
- `L_pred`: MSE between predicted and target latent representations
- `L_sigreg`: SIGReg regularization to prevent representation collapse
- `L_entrance`: MSE on the decoded entrance position `t_pred`

### Input Features

Per image (up to 5 of 15 available, randomly sampled during training):
- **Patch embeddings**: DINOv2 ViT-L/14 horizontal strip (vertical average of 16x16 grid to 16x1024)
- **Per-column facade_t**: Ray-facade intersection for each of 16 patch columns, telling the model where on the facade each image column is looking
- **Camera pose** (8-dim): Camera position relative to facade midpoint, perpendicular distance, angle to facade normal, relative bearing, HFOV, along-facade fraction, facing indicator

Per POI:
- **Facade features** (32-dim): Edge endpoints and midpoint in local meters, facade length, outward normal, bearing (sin/cos), entrance position, perpendicular entrance-to-facade distance, building perimeter, area, edge count, bbox aspect ratios

### Image Filtering

Not all images associated with a POI actually show the building facade. The dataset pipeline filters images using ray-facade geometry: each of the 16 DINOv2 patch columns is cast as a ray from the camera, and only images where at least 4/16 columns intersect the facade edge (or the camera is facing the facade) are kept. This removes ~17% of images and significantly reduces noise in the training signal.

## Data Pipeline

### Sources

1. **Embedding tiles** ([zephr-xyz/embedding-tiles](https://github.com/zephr-xyz/embedding-tiles)): POI metadata including entrance coordinates, Mapillary image IDs, and Overture building IDs
2. **S3 computed data** (`s3://zephr-mapillary-computed-data/`): Pre-computed DINOv2 patch embeddings and image metadata per Mapillary image
3. **Building footprints** (`buildings.json`): Overture Maps building polygon coordinates keyed by overture_building_id, with coordinates in `[lat, lon]` order
4. **RTK ground truth** (`ground_truth_labels.json`): Centimeter-precision entrance locations from RTK GPS field surveys

### Dataset Preparation

```bash
# Step 1: Build image-to-sequence lookup (only needed once)
python build_s3_index.py --tiles-dir path/to/z14 --output s3_sequence_lookup.json

# Step 2: Prepare dataset with image filtering
python prepare_dataset.py \
    --tiles-dir path/to/z14 \
    --buildings-json buildings.json \
    --ground-truth-labels ground_truth_labels.json \
    --s3-index-cache s3_sequence_lookup.json \
    --output-dir dataset_cache \
    --max-images-per-poi 15 \
    --min-facade-cols 4 \
    --val-fraction 0.15
```

The `--min-facade-cols` flag controls the image quality filter. Each image's 16 patch columns are projected as rays onto the facade edge. Images where fewer than this many columns intersect the facade (AND the camera is not facing the facade) are excluded. Default is 4, which removes images that don't meaningfully show the building.

### Dataset Statistics

| Split | Samples | RTK-labeled | Images (filtered) |
|-------|---------|-------------|-------------------|
| Train | 1,358 | 0 | ~7,300 |
| Val | 310 | 71 | ~1,700 |
| Total | 1,668 | 71 | 9,010 |

All RTK-labeled samples are placed in validation. Non-RTK samples are split randomly by `--val-fraction`.

## Training

```bash
python train.py \
    --data-dir dataset_cache \
    --output-dir checkpoints \
    --epochs 300 \
    --batch-size 32 \
    --lr 5e-4 \
    --d-latent 128 \
    --lambda-sigreg 0.05 \
    --mu-entrance 10.0 \
    --warmup-epochs 20
```

Or use `launch_training.sh` for the full end-to-end pipeline (dataset prep + training + S3 sync).

### EC2 Setup

The `launch_training.sh` script expects:
- `/home/ubuntu/embedding-tiles-data/z14/` -- embedding tile JSONs
- `/home/ubuntu/buildings.json` -- building footprints
- `/home/ubuntu/s3_sequence_lookup.json` -- image-to-sequence mapping
- `/home/ubuntu/entrance_corrections.json` -- optional entrance corrections (can be `{}`)
- Code cloned to `/home/ubuntu/jepa-entrance/`

The tiles can be synced from S3:
```bash
aws s3 cp s3://zephr-mapillary-cache/poi-tiles/boulder_county/tiles_v23/ \
    /home/ubuntu/embedding-tiles-data/z14/ --recursive
```

Use `aws s3 cp --recursive` rather than `aws s3 sync` for the initial download -- sync can silently fail with I/O errors on large directories.

## Inference

To predict entrance locations for the full dataset:

```bash
python predict_entrances.py \
    --checkpoint best_model.pt \
    --data-dir dataset_cache \
    --output updated_entrances.json
```

## Evaluation

To evaluate on the validation set with RTK ground truth:

```bash
python evaluate.py \
    --checkpoint best_model.pt \
    --data-dir dataset_cache \
    --output val_predictions.json
```

To generate an interactive map visualization:

```bash
python generate_map.py \
    --checkpoint best_model.pt \
    --val-manifest val_manifest.json \
    --buildings-json buildings.json \
    --ground-truth-labels ground_truth_labels.json
```

## Files

| File | Description |
|------|-------------|
| `model.py` | JEPA model architecture (4.2M params) |
| `train.py` | Training loop with cosine LR schedule |
| `dataset.py` | PyTorch Dataset, facade geometry computation (`compute_facade_and_entrance_t`), image subset augmentation |
| `geometry.py` | Camera-to-facade geometry: HFOV, pose features, ray-facade intersection |
| `prepare_dataset.py` | Dataset builder: fetches S3 embeddings, computes geometry, filters non-facade images |
| `build_s3_index.py` | S3 image_id to sequence_id lookup builder |
| `evaluate.py` | Validation inference and per-POI predictions |
| `predict_entrances.py` | Full-dataset entrance prediction |
| `match_ground_truth.py` | Matches RTK GPS measurements to POIs |
| `generate_map.py` | Interactive Leaflet map visualization |
| `create_entrance_map.py` | Comparison map of predicted vs ground truth entrances |
| `ground_truth_labels.json` | 72 RTK ground truth labels (41 Louisville + 31 Boulder) |
| `launch_training.sh` | End-to-end training pipeline for EC2 |

## S3 Data

Cached datasets and models are persisted to S3 to avoid recomputation:

```
s3://zephr-mapillary-cache/jepa-entrance-v3-singleedge/
  best_model.pt                    # Best model checkpoint (val MAE 1.73m)
  final_model.pt                   # Final epoch checkpoint
  training_history.json            # Per-epoch metrics
  train_manifest.json              # Training sample metadata
  val_manifest.json                # Validation sample metadata
  checkpoint_epoch*.pt             # Periodic checkpoints (50, 100, 150, 200, 250, 300)

s3://zephr-mapillary-cache/jepa-entrance-v3/
  buildings.json                   # Building footprints (90k buildings)
  s3_sequence_lookup.json          # Image to sequence mapping (6,968 entries)

s3://zephr-mapillary-cache/poi-tiles/boulder_county/tiles_v23/
  x*_y*.json                       # 1,034 embedding tiles for Boulder County
```

The per-sample cache (patch_strip_*.npy, facade_t_cols_*.npy, camera_pose_*.npy, facade_feats.npy, image_meta.json) is generated locally by `prepare_dataset.py` and not synced to S3 due to size (~1.5GB).

## Requirements

```
torch>=2.0
numpy>=1.24
boto3>=1.28
```

## License

MIT
