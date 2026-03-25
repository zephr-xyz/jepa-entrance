# JEPA Entrance Prediction

Predicting building entrance locations from Mapillary street-level imagery using a Joint-Embedding Predictive Architecture (JEPA).

## Overview

Given a building and its associated Mapillary images, the model predicts where the entrance is along the building's road-facing facade. The prediction is parameterized as `t ∈ [0, 1]`, representing the fractional position along the facade edge, which is then converted to geographic coordinates.

### Key Results

Evaluated on 72 POIs with RTK-precision ground truth (centimeter-level GPS measurements of actual front door locations) in Louisville, CO and Boulder, CO:

| Approach | MAE | Median | P90 |
|----------|-----|--------|-----|
| Facade midpoint (t=0.5) | 5.21m | 3.18m | 14.67m |
| **JEPA facade-t prediction** | **3.45m** | **1.91m** | **7.46m** |

The JEPA model reduces MAE by 34% and P90 by 49% compared to the midpoint baseline.

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

**Training losses:**
- `L_pred`: MSE between predicted and target latent representations
- `L_sigreg`: SIGReg regularization to prevent representation collapse
- `L_entrance`: MSE on the decoded entrance position `t_pred`

### Input Features

**Per image (up to 5 of 15 available, randomly sampled during training):**
- **Patch embeddings**: DINOv2 ViT-L/14 horizontal strip (vertical average of 16×16 grid → 16×1024)
- **Per-column facade_t**: Ray-facade intersection for each of 16 patch columns — tells the model *where on the facade* each image column is looking
- **Camera pose** (8-dim): Camera position relative to facade midpoint, perpendicular distance, angle to facade normal, relative bearing, HFOV, along-facade fraction, facing indicator

**Per POI:**
- **Facade features** (32-dim): Edge geometry in local meters, facade length, bearing, outward normal, building centroid, vertex count, entrance-to-facade distance, road class one-hot encoding

## Data Pipeline

### Sources

1. **Embedding tiles** ([zephr-xyz/embedding-tiles](https://github.com/zephr-xyz/embedding-tiles)): POI metadata including entrance coordinates, Mapillary image IDs, and Overture building IDs
2. **S3 computed data** (`s3://zephr-mapillary-computed-data/`): Pre-computed DINOv2 patch embeddings and image metadata per Mapillary image
3. **Building footprints** (`buildings.json`): Overture Maps building polygon coordinates
4. **RTK ground truth** (`ground_truth_labels.json`): Centimeter-precision entrance locations from RTK GPS field surveys

### Dataset Preparation

```bash
python build_s3_index.py --tiles-dir path/to/z14 --output s3_sequence_lookup.json
python prepare_dataset_v3_singleedge.py \
    --tiles-dir path/to/z14 \
    --buildings-json buildings.json \
    --ground-truth-labels ground_truth_labels.json \
    --s3-index-cache s3_sequence_lookup.json \
    --output-dir cache_v3_se
```

### Dataset Statistics

| Split | Samples | RTK-labeled |
|-------|---------|-------------|
| Train | 958 | 0 |
| Val | 241 | 72 |
| **Total** | **1,199** | **72** |

## Training

```bash
python train_v3.py --data-dir cache_v3_se --buildings-json buildings.json --epochs 300
```

Training takes ~7 minutes on an NVIDIA A10G (g5.xlarge). Best model selected by RTK geographic MAE.

## Inference

To predict entrance locations for the full dataset:

```bash
python predict_entrances.py \
    --checkpoint best_model_se.pt \
    --data-dir cache_v3_se \
    --buildings-json buildings.json \
    --output updated_entrances.json
```

## Files

| File | Description |
|------|-------------|
| `model_v3.py` | JEPA model architecture |
| `train_v3.py` | Training loop |
| `dataset_v3.py` | PyTorch Dataset with image subset augmentation |
| `evaluate_v3.py` | Val inference and per-POI predictions |
| `predict_entrances.py` | Full-dataset entrance prediction |
| `geometry.py` | Camera-to-facade geometry: HFOV, pose features, ray-facade intersection |
| `facade.py` | Composite facade detection (experimental) |
| `dataset.py` | Single-edge facade computation |
| `build_s3_index.py` | S3 image_id → sequence_id lookup builder |
| `prepare_dataset_v3_singleedge.py` | Dataset builder: fetches embeddings, computes geometry |
| `ground_truth_labels.json` | 72 RTK ground truth labels (41 Louisville + 31 Boulder) |
| `generate_map.py` | Interactive Leaflet map visualization |

## S3 Data

Cached datasets and models are persisted to S3 to avoid recomputation:

```
s3://zephr-mapillary-cache/jepa-entrance-v3-singleedge/
  best_model.pt                    # Trained model checkpoint
  cache_v3_se/                     # Full per-sample cache (~540MB)
    train_manifest.json
    val_manifest.json
    v3se_poi_*/                    # Per-sample: patch_strip_*.npy, facade_t_cols_*.npy,
                                   #   camera_pose_*.npy, facade_feats.npy, image_meta.json
s3://zephr-mapillary-cache/jepa-entrance-v3/
  buildings.json                   # Building footprints
  s3_sequence_lookup.json          # Image → sequence mapping
  ground_truth_labels.json         # RTK labels
```

## Requirements

```
torch>=2.0
numpy>=1.24
boto3>=1.28
```
