# JEPA Entrance Prediction

Predicting building entrance locations from Mapillary street-level imagery using a Joint-Embedding Predictive Architecture (JEPA).

## Overview

Given a building and its associated Mapillary images, the model predicts where the entrance is along the building's road-facing facade. The prediction is parameterized as `t ∈ [0, 1]`, representing the fractional position along the facade edge, which is then converted to geographic coordinates.

### Key Results

Evaluated on 72 POIs with RTK-precision ground truth (centimeter-level GPS measurements of actual front door locations) in Louisville, CO and Boulder, CO:

| Approach | MAE | Median | P90 |
|----------|-----|--------|-----|
| Facade midpoint (t=0.5) | 5.21m | 3.18m | 14.67m |
| JEPA facade-t prediction | 2.51m | 1.87m | 6.31m |

The JEPA model reduces MAE by 52% and P90 by 57% compared to the midpoint baseline.

## Architecture

The model follows a JEPA (Joint-Embedding Predictive Architecture) training paradigm with four components:

```
┌─────────────────────────────────────────────────┐
│                Context Encoder                  │
│  DINOv2 patch embeddings (16×1024 per image)    │
│  + Facade-t positional encoding (per patch col) │
│  + Camera pose tokens (8-dim per image)         │
│  + Self-attention across K images               │
│  → z_visual (128-dim)                           │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│               Predictor (AdaLN)                  │
│  z_visual → ẑ_geo, conditioned on facade feats   │
│  → Entrance Head: ẑ_geo → t_pred ∈ [0,1]         │
└──────────────────────────────────────────────────┘
                        ↕ MSE loss
┌──────────────────────────────────────────────────┐
│               Target Encoder                     │
│  Facade geometry (32-dim) + entrance_t → z_geo   │
└──────────────────────────────────────────────────┘
```

Training losses:
- `L_pred`: MSE between predicted and target latent representations
- `L_sigreg`: SIGReg regularization to prevent representation collapse
- `L_entrance`: MSE on the decoded entrance position `t_pred`

### Input Features

Per image (up to 5 of 15 available, randomly sampled during training):
- Patch embeddings: DINOv2 ViT-L/14 horizontal strip (vertical average of 16x16 grid to 16x1024)
- Per-column facade_t: Ray-facade intersection for each of 16 patch columns, telling the model where on the facade each image column is looking
- Camera pose (8-dim): Camera position relative to facade midpoint, perpendicular distance, angle to facade normal, relative bearing, HFOV, along-facade fraction, facing indicator

Per POI:
- Facade features (32-dim): Edge geometry in local meters, facade length, bearing, outward normal, building centroid, vertex count, entrance-to-facade distance, road class one-hot encoding

## Data Pipeline

### Sources

1. Embedding tiles ([zephr-xyz/embedding-tiles](https://github.com/zephr-xyz/embedding-tiles)): POI metadata including entrance coordinates, Mapillary image IDs, and Overture building IDs
2. S3 computed data (`s3://zephr-mapillary-computed-data/`): Pre-computed DINOv2 patch embeddings and image metadata per Mapillary image
3. Building footprints (`buildings.json`): Overture Maps building polygon coordinates
4. RTK ground truth (`ground_truth_labels.json`): Centimeter-precision entrance locations from RTK GPS field surveys

### Dataset Preparation

```bash
python build_s3_index.py --tiles-dir path/to/z14 --output s3_sequence_lookup.json
python prepare_dataset.py \
    --tiles-dir path/to/z14 \
    --buildings-json buildings.json \
    --ground-truth-labels ground_truth_labels.json \
    --s3-index-cache s3_sequence_lookup.json \
    --output-dir cache
```

### Dataset Statistics

| Split | Samples | RTK-labeled |
|-------|---------|-------------|
| Train | 958 | 0 |
| Val | 241 | 72 |
| Total | 1,199 | 72 |

## Training

```bash
python train.py --data-dir cache --buildings-json buildings.json --epochs 300
```

Training takes approximately 7 minutes on an NVIDIA A10G (g5.xlarge). Best model selected by RTK geographic MAE.

## Inference

To predict entrance locations for the full dataset:

```bash
python predict_entrances.py \
    --checkpoint best_model.pt \
    --data-dir cache \
    --buildings-json buildings.json \
    --output updated_entrances.json
```

## Evaluation

To evaluate on the validation set with RTK ground truth:

```bash
python evaluate.py \
    --checkpoint best_model.pt \
    --data-dir cache \
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
| `model.py` | JEPA model architecture |
| `train.py` | Training loop |
| `dataset.py` | PyTorch Dataset with image subset augmentation |
| `evaluate.py` | Validation inference and per-POI predictions |
| `predict_entrances.py` | Full-dataset entrance prediction |
| `geometry.py` | Camera-to-facade geometry: HFOV, pose features, ray-facade intersection |
| `build_s3_index.py` | S3 image_id to sequence_id lookup builder |
| `prepare_dataset.py` | Dataset builder: fetches embeddings, computes geometry |
| `match_ground_truth.py` | Matches RTK GPS measurements to POIs |
| `generate_map.py` | Interactive Leaflet map visualization |
| `create_entrance_map.py` | Comparison map of predicted vs ground truth entrances |
| `ground_truth_labels.json` | 72 RTK ground truth labels (41 Louisville + 31 Boulder) |
| `launch_training.sh` | End-to-end training pipeline for EC2 |

## S3 Data

Cached datasets and models are persisted to S3 to avoid recomputation:

```
s3://zephr-mapillary-cache/jepa-entrance-v3-singleedge/
  best_model.pt                    # Trained model checkpoint
  cache/                           # Full per-sample cache (~540MB)
    train_manifest.json
    val_manifest.json
    poi_*/                         # Per-sample: patch_strip_*.npy, facade_t_cols_*.npy,
                                   #   camera_pose_*.npy, facade_feats.npy, image_meta.json
s3://zephr-mapillary-cache/jepa-entrance-v3/
  buildings.json                   # Building footprints
  s3_sequence_lookup.json          # Image to sequence mapping
  ground_truth_labels.json         # RTK labels
```

## Requirements

```
torch>=2.0
numpy>=1.24
boto3>=1.28
```

## License

MIT
