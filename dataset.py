"""
Dataset for JEPA entrance prediction training.

Loads and fuses data from three sources:
  1. embedding-tiles (POI metadata, entrance coords, building IDs, mapillary IDs)
  2. s3://zephr-mapillary-computed-data/ (DINOv2, caption, keypoint embeddings)
  3. zephr-maps data/ (building footprints, road context)

Each sample provides:
  - Visual features: DINOv2 CLS, caption embedding, keypoint stats, compass angle
  - Facade features: edge geometry, bearing, length, road class
  - Target: entrance_t ∈ [0, 1] (position along facade)
"""
import json
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320

# Road class to integer encoding
ROAD_CLASSES = {
    'motorway': 0, 'trunk': 1, 'primary': 2, 'secondary': 3,
    'tertiary': 4, 'residential': 5, 'service': 6, 'unclassified': 7,
    'living_street': 8, 'pedestrian': 9, 'track': 10, 'path': 11,
}


def compute_facade_and_entrance_t(building_coords, entrance_lat, entrance_lon,
                                  enclosing_roads):
    """Compute facade features and entrance parameter t.

    Finds the road-facing facade edge of the building, then parameterizes
    the entrance position as t ∈ [0, 1] along that edge.

    Returns:
        facade_feats: (32,) float array of facade features
        entrance_t: float in [0, 1]
        or None, None if computation fails
    """
    if len(building_coords) < 3:
        return None, None

    # Building centroid
    cent_lng = sum(c[0] for c in building_coords) / len(building_coords)
    cent_lat = sum(c[1] for c in building_coords) / len(building_coords)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    def to_m(coord):
        return [(coord[0] - cent_lng) * m_per_deg_lng,
                (coord[1] - cent_lat) * METERS_PER_DEG_LAT]

    fp_m = [to_m(c) for c in building_coords]
    ent_m = to_m([entrance_lon, entrance_lat])
    poly_cx = sum(p[0] for p in fp_m) / len(fp_m)
    poly_cy = sum(p[1] for p in fp_m) / len(fp_m)

    # Score edges: prefer those closest to entrance with outward normals
    best_edge = None
    best_score = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        # Outward normal
        nx, ny = -edy / length, edx / length
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        tcx, tcy = poly_cx - mx, poly_cy - my
        if nx * tcx + ny * tcy > 0:
            nx, ny = -nx, -ny

        # Distance from edge midpoint to entrance
        dx = ent_m[0] - mx
        dy = ent_m[1] - my
        dist = math.sqrt(dx * dx + dy * dy)

        # Score: distance (lower is better)
        if dist < best_score:
            best_score = dist
            best_edge = {
                'a': a, 'b': b, 'length': length,
                'normal': [nx, ny],
                'bearing': math.atan2(nx, ny) / DEG_TO_RAD,
                'midpoint': [mx, my],
            }

    if best_edge is None:
        return None, None

    # Compute entrance_t: project entrance onto facade edge
    a = best_edge['a']
    b = best_edge['b']
    edge_dx = b[0] - a[0]
    edge_dy = b[1] - a[1]
    ent_dx = ent_m[0] - a[0]
    ent_dy = ent_m[1] - a[1]

    # t = dot(entrance - a, b - a) / |b - a|^2
    edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy
    if edge_len_sq < 0.01:
        return None, None

    t = (ent_dx * edge_dx + ent_dy * edge_dy) / edge_len_sq
    t = max(0.0, min(1.0, t))  # clamp

    # Build facade feature vector (32 dims)
    # Encoding the road class
    road_class_enc = [0.0] * 12
    if enclosing_roads:
        rc = enclosing_roads[0].get('road_class', 'residential')
        idx = ROAD_CLASSES.get(rc, 5)
        road_class_enc[idx] = 1.0

    facade_feats = [
        # Edge geometry (local meters, centered on building centroid)
        a[0], a[1], b[0], b[1],           # 4: edge start/end
        best_edge['length'],                # 1: facade length
        best_edge['bearing'] / 180.0,       # 1: bearing normalized [-1, 1]
        best_edge['normal'][0],             # 1: outward normal x
        best_edge['normal'][1],             # 1: outward normal y
        best_edge['midpoint'][0],           # 1: midpoint x
        best_edge['midpoint'][1],           # 1: midpoint y
        # Building context
        poly_cx, poly_cy,                   # 2: building centroid (local)
        len(building_coords),               # 1: vertex count (complexity)
        best_score,                         # 1: distance entrance-to-facade
    ]
    facade_feats.extend(road_class_enc)     # 12: one-hot road class
    # Pad to 32
    while len(facade_feats) < 32:
        facade_feats.append(0.0)

    return np.array(facade_feats[:32], dtype=np.float32), float(t)


def load_keypoint_stats(kp_data):
    """Compute 16-dim statistics from keypoint data.

    Stats: count, mean_x, mean_y, std_x, std_y, spatial coverage,
    score stats (mean, std, min, max), and spatial quadrant counts.
    """
    stats = np.zeros(16, dtype=np.float32)
    if kp_data is None:
        return stats

    kps = kp_data.get('keypoints', None)
    scores = kp_data.get('scores', None)

    if kps is None or len(kps) == 0:
        return stats

    kps = np.array(kps)
    n = len(kps)
    stats[0] = min(n / 1000.0, 1.0)  # normalized count

    if kps.ndim == 2 and kps.shape[1] >= 2:
        stats[1] = np.mean(kps[:, 0])  # mean x
        stats[2] = np.mean(kps[:, 1])  # mean y
        stats[3] = np.std(kps[:, 0])   # std x
        stats[4] = np.std(kps[:, 1])   # std y
        # Spatial coverage: how spread out are keypoints
        x_range = np.ptp(kps[:, 0])
        y_range = np.ptp(kps[:, 1])
        stats[5] = x_range * y_range / 1e6 if x_range > 0 and y_range > 0 else 0

        # Quadrant counts (normalized)
        cx = np.median(kps[:, 0])
        cy = np.median(kps[:, 1])
        stats[6] = np.sum((kps[:, 0] < cx) & (kps[:, 1] < cy)) / n  # TL
        stats[7] = np.sum((kps[:, 0] >= cx) & (kps[:, 1] < cy)) / n  # TR
        stats[8] = np.sum((kps[:, 0] < cx) & (kps[:, 1] >= cy)) / n  # BL
        stats[9] = np.sum((kps[:, 0] >= cx) & (kps[:, 1] >= cy)) / n  # BR

    if scores is not None and len(scores) > 0:
        scores = np.array(scores).flatten()
        stats[10] = np.mean(scores)
        stats[11] = np.std(scores)
        stats[12] = np.min(scores)
        stats[13] = np.max(scores)
        stats[14] = np.median(scores)
        stats[15] = np.percentile(scores, 75) - np.percentile(scores, 25)  # IQR

    return stats


class EntranceDataset(Dataset):
    """Dataset for JEPA entrance prediction.

    Loads pre-cached features from disk. Use `prepare_dataset.py` to
    build the cache from S3 + embedding-tiles + zephr-maps data.
    """

    def __init__(self, cache_dir: str, split: str = 'train'):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / f'{split}_manifest.json'
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        print(f"EntranceDataset [{split}]: {len(self.manifest)} samples")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        sample_dir = self.cache_dir / entry['sample_id']

        # Load pre-cached numpy arrays
        cls_emb = np.load(sample_dir / 'cls_emb.npy')          # (1024,)
        caption_emb = np.load(sample_dir / 'caption_emb.npy')  # (768,)
        kp_stats = np.load(sample_dir / 'kp_stats.npy')        # (16,)
        facade_feats = np.load(sample_dir / 'facade_feats.npy')  # (32,)

        compass = np.array([entry['compass_normalized']], dtype=np.float32)
        entrance_t = np.array([entry['entrance_t']], dtype=np.float32)

        return {
            'cls_emb': torch.from_numpy(cls_emb),
            'caption_emb': torch.from_numpy(caption_emb),
            'kp_stats': torch.from_numpy(kp_stats),
            'compass': torch.from_numpy(compass),
            'facade_feats': torch.from_numpy(facade_feats),
            'entrance_t': torch.from_numpy(entrance_t),
            'has_rtk_label': entry.get('has_rtk_label', False),
            'poi_name': entry['poi_name'],
            'sample_id': entry['sample_id'],
        }
