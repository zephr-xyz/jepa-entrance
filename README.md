# JEPA Entrance Prediction

Predicting building entrance locations from Mapillary street-level imagery using a Joint-Embedding Predictive Architecture (JEPA).

## Overview

Given a building and its associated Mapillary images, the model predicts where the entrance is along the building's road-facing facade. The prediction is parameterized as `t ∈ [0, 1]`, representing the fractional position along the facade edge, which is then converted to geographic coordinates.

### Key Results

Evaluated on 72 POIs with RTK-precision ground truth (centimeter-level GPS measurements of actual front door locations) in Louisville, KY and Boulder, CO:

| Metric | Along-facade (1D) | Geographic (2D) |
|--------|:-----------------:|:---------------:|
| **MAE** | **1.96m** | **3.80m** |
| **Median error** | **0.43m** | **1.89m** |
| **P90 error** | ~6.5m | ~8.5m |

**Along-facade error** measures `|t_pred - t_true| × facade_length` — the model's prediction error projected onto the facade edge. This is the metric the model directly optimizes.

**Geographic error** measures the haversine distance between the predicted lat/lon and the actual RTK entrance location. This is higher because actual entrances are not exactly on the facade edge line — the average perpendicular distance from the door to the selected facade edge is ~2.2m, and this offset is included in the geographic error even for a perfect `t` prediction.

The geographic metric is the more practically relevant one — it represents how far the predicted entrance point is from the real door. The along-facade metric isolates the model's learned contribution.

The baseline (predicting the facade midpoint, `t = 0.5`) gives an along-facade MAE of 3.85m, so the model achieves a 43.6% improvement.

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

### Facade Detection

The facade is identified as the building polygon edge closest to the entrance location. For each candidate edge:
1. Compute the outward-pointing normal (away from building centroid)
2. Score by distance from edge midpoint to entrance
3. Select the closest edge as the facade

The entrance position `t` is the projection of the entrance point onto this edge, clamped to `[0, 1]`.

## Data Pipeline

### Sources

1. **Embedding tiles** (`z14/*.json`): POI metadata including entrance coordinates, Mapillary image IDs, and Overture building IDs
2. **S3 computed data** (`s3://zephr-mapillary-computed-data/`): Pre-computed DINOv2 patch embeddings and image metadata per Mapillary image
3. **Building footprints** (`buildings.json`): Overture Maps building polygon coordinates
4. **RTK ground truth** (`ground_truth_labels.json`): Centimeter-precision entrance locations from RTK GPS field surveys

### Dataset Preparation

```
build_s3_index.py          → Maps image_id → sequence_id for S3 lookup
prepare_dataset_v3_singleedge.py → Fetches embeddings, computes geometry, caches .npy files
```

The preparation pipeline:
1. Scans S3 to build an image→sequence lookup index
2. For each POI: fetches up to 15 Mapillary image patch embeddings from S3
3. Identifies the facade edge from the building footprint
4. Computes camera pose features and per-column facade_t for each image
5. Projects the entrance onto the facade to get the target `entrance_t`
6. Splits data: all RTK-labeled POIs go to validation, remaining 85/15 train/val

### Dataset Statistics

| Split | Samples | RTK-labeled |
|-------|---------|-------------|
| Train | 958 | 0 |
| Val | 241 | 72 |
| **Total** | **1,199** | **72** |

Training samples use geometric entrance estimates (noisy, ~5m error). RTK samples in validation provide precise evaluation.

## Training

```bash
# Full pipeline (S3 index → dataset → train → sync)
bash launch_training_v3_singleedge.sh
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 300 |
| Batch size | 32 |
| Learning rate | 5e-4 (cosine decay) |
| Warmup | 20 epochs |
| d_latent | 128 |
| λ_sigreg | 0.05 |
| μ_entrance | 10.0 |
| Images per POI | 5 (sampled from up to 15) |
| Model parameters | 4.2M |

Training takes ~7 minutes on an NVIDIA A10G (g5.xlarge).

## Files

### Core Model (v3)
| File | Description |
|------|-------------|
| `model_v3.py` | JEPA model: Context Encoder (patch + pose + transformer), Target Encoder, Predictor (AdaLN), Entrance Head |
| `train_v3.py` | Training loop with cosine LR, RTK-specific val metrics |
| `dataset_v3.py` | PyTorch Dataset with random image subset augmentation |
| `evaluate_v3.py` | Run inference, output per-POI predictions |

### Geometry & Facade
| File | Description |
|------|-------------|
| `geometry.py` | Camera-to-facade computations: HFOV, pose features, ray-facade intersection |
| `facade.py` | Composite facade detection (multi-edge road-facing facades, experimental) |
| `dataset.py` | Original single-edge facade computation (`compute_facade_and_entrance_t`) |

### Data Preparation
| File | Description |
|------|-------------|
| `build_s3_index.py` | Scan S3 bucket to build image_id → sequence_id lookup |
| `prepare_dataset_v3_singleedge.py` | Main dataset builder: fetches S3 embeddings, computes geometry, caches features |
| `prepare_dataset_v3.py` | Variant using composite (multi-edge) facade |
| `match_ground_truth.py` | Match RTK survey data to embedding-tile waypoints |
| `ground_truth_labels.json` | 72 RTK ground truth labels (41 Louisville + 31 Boulder) |

### Visualization & Evaluation
| File | Description |
|------|-------------|
| `generate_map.py` | Interactive Leaflet map: predicted vs ground truth entrance locations |
| `evaluate.py` | v1/v2 evaluation script (legacy) |

### Launch Scripts
| File | Description |
|------|-------------|
| `launch_training_v3_singleedge.sh` | End-to-end: S3 index → dataset → train → S3 sync (single-edge facade) |
| `launch_training_v3.sh` | Same pipeline with composite facade |

## Ground Truth

RTK ground truth was collected via field surveys using centimeter-precision GPS:
- **Louisville, KY**: 41 POIs along main commercial corridors
- **Boulder, CO**: 31 POIs on Pearl Street and surrounding areas

Each measurement records the exact lat/lon/elevation of the building's front door. Matching to embedding-tile waypoints uses Overture building IDs (Louisville) or name + proximity matching (Boulder).

## Experimental: Composite Facade

The `facade.py` module implements an alternative facade detection strategy that finds *all* road-facing edges of a building and stitches them into a composite polyline. This was explored as a potentially more accurate parameterization:

- **Single-edge**: Picks one building edge (~5.4m avg). Entrance `t` varies within this short edge.
- **Composite**: Finds connected road-facing edges (~60m avg). Entrance `t` spans the full road-facing side.

Results show single-edge outperforms composite on our current dataset (1.96m vs 5.20m RTK MAE), likely because:
1. Single-edge inherently selects short edges near the entrance, making the prediction problem easier
2. The longer composite facade requires the model to learn more precise spatial localization
3. With only 72 RTK samples for evaluation, the simpler parameterization generalizes better

The composite approach may become advantageous with more RTK training data or for buildings with complex facades where the entrance isn't on the closest edge.

## Infrastructure

Training runs on AWS EC2 `g5.xlarge` (NVIDIA A10G, 24GB VRAM) with the PyTorch Deep Learning AMI. Checkpoints and manifests sync to `s3://zephr-mapillary-cache/jepa-entrance-v3-singleedge/`.

## Requirements

```
torch>=2.0
numpy>=1.24
boto3>=1.28
```
