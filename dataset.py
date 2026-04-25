"""
Dataset for JEPA v3 entrance prediction.

Each sample contains:
  - patch_strips: (K, 16, 1024) horizontal patch features per image
  - facade_t_cols: (K, 16) per-column facade_t per image
  - camera_poses: (K, 8) camera pose features per image
  - image_mask: (K,) which images are valid
  - facade_feats: (32,) facade geometry
  - entrance_t: (1,) target

Training augmentation: when n_total_images > MAX_IMAGES, randomly
sample a different subset of images each time __getitem__ is called.
"""
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


MAX_IMAGES = 5
N_COLS = 16
PATCH_DIM = 1024
DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def compute_facade_and_entrance_t(building_coords, entrance_lat, entrance_lon,
                                  enclosing_roads=None):
    """Compute 32-d facade feature vector and entrance_t for a POI.

    Args:
        building_coords: list of [lon, lat] ring coordinates
        entrance_lat, entrance_lon: entrance position
        enclosing_roads: list of road geometries (unused currently, reserved)

    Returns:
        (facade_feats, entrance_t) or (None, None) if no valid facade found.
        facade_feats: (32,) float32 — facade geometry encoding
            [0-1]: facade endpoint A (local meters, /50)
            [2-3]: facade endpoint B (local meters, /50)
            [4]:   facade length (meters)
            [5-6]: facade midpoint (local meters, /50)
            [7-8]: outward normal (unit vector)
            [9-10]: facade bearing sin/cos
            [11-12]: entrance position (local meters, /50)
            [13]: entrance perpendicular distance to facade (/50)
            [14]: building perimeter (/200)
            [15]: building area (/2000)
            [16]: number of edges (/20)
            [17-18]: building bbox aspect ratio (w/h, h/w)
            [19-23]: road features (reserved zeros)
            [24-31]: padding zeros
        entrance_t: float in [0, 1] — position along facade edge
    """
    if len(building_coords) < 3:
        return None, None

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

    best_edge = None
    best_score = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        nx, ny = -edy / length, edx / length
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        tcx, tcy = poly_cx - mx, poly_cy - my
        if nx * tcx + ny * tcy > 0:
            nx, ny = -nx, -ny

        dx = ent_m[0] - mx
        dy = ent_m[1] - my
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < best_score:
            best_score = dist
            best_edge = {
                'a': a, 'b': b, 'length': length,
                'normal': [nx, ny],
                'midpoint': [mx, my],
            }

    if best_edge is None:
        return None, None

    a = best_edge['a']
    b = best_edge['b']
    facade_len = best_edge['length']
    mid = best_edge['midpoint']
    normal = best_edge['normal']

    # entrance_t: project entrance onto facade edge
    fx, fy = b[0] - a[0], b[1] - a[1]
    ex, ey = ent_m[0] - a[0], ent_m[1] - a[1]
    t = (ex * fx + ey * fy) / (facade_len * facade_len)
    entrance_t = max(0.0, min(1.0, t))

    # Facade bearing
    bearing_rad = math.atan2(fx, fy)

    # Entrance perpendicular distance to facade line
    ent_perp = (ex * fy - ey * fx) / facade_len

    # Building geometry stats
    perimeter = 0.0
    n_edges = 0
    for i in range(len(fp_m) - 1):
        dx = fp_m[i + 1][0] - fp_m[i][0]
        dy = fp_m[i + 1][1] - fp_m[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)
        n_edges += 1

    xs = [p[0] for p in fp_m]
    ys = [p[1] for p in fp_m]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    area = abs(sum(fp_m[i][0] * fp_m[(i + 1) % len(fp_m)][1] -
                   fp_m[(i + 1) % len(fp_m)][0] * fp_m[i][1]
                   for i in range(len(fp_m)))) / 2.0

    feats = np.zeros(32, dtype=np.float32)
    feats[0] = a[0] / 50.0
    feats[1] = a[1] / 50.0
    feats[2] = b[0] / 50.0
    feats[3] = b[1] / 50.0
    feats[4] = facade_len
    feats[5] = mid[0] / 50.0
    feats[6] = mid[1] / 50.0
    feats[7] = normal[0]
    feats[8] = normal[1]
    feats[9] = math.sin(bearing_rad)
    feats[10] = math.cos(bearing_rad)
    feats[11] = ent_m[0] / 50.0
    feats[12] = ent_m[1] / 50.0
    feats[13] = ent_perp / 50.0
    feats[14] = perimeter / 200.0
    feats[15] = area / 2000.0
    feats[16] = n_edges / 20.0
    feats[17] = (bbox_w / max(bbox_h, 0.1))
    feats[18] = (bbox_h / max(bbox_w, 0.1))

    return feats, entrance_t


class EntranceDatasetV3(Dataset):
    def __init__(self, cache_dir: str, split: str = 'train'):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.is_train = (split == 'train')
        manifest_path = self.cache_dir / f'{split}_manifest.json'
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        print(f"EntranceDatasetV3 [{split}]: {len(self.manifest)} samples")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        sample_dir = self.cache_dir / entry['sample_id']

        n_images = entry.get('n_images', 1)

        # Choose which images to load
        if self.is_train and n_images > MAX_IMAGES:
            # Random subset for training augmentation
            indices = np.random.choice(n_images, MAX_IMAGES, replace=False)
            indices.sort()
        else:
            indices = list(range(min(n_images, MAX_IMAGES)))

        # Load multi-image patch features
        patch_strips = np.zeros((MAX_IMAGES, N_COLS, PATCH_DIM), dtype=np.float32)
        facade_t_cols = np.full((MAX_IMAGES, N_COLS), 0.5, dtype=np.float32)
        camera_poses = np.zeros((MAX_IMAGES, 8), dtype=np.float32)
        image_mask = np.zeros(MAX_IMAGES, dtype=np.bool_)

        for slot, k in enumerate(indices):
            ps_path = sample_dir / f'patch_strip_{k}.npy'
            ft_path = sample_dir / f'facade_t_cols_{k}.npy'
            cp_path = sample_dir / f'camera_pose_{k}.npy'

            if ps_path.exists():
                patch_strips[slot] = np.load(ps_path)
                image_mask[slot] = True
            if ft_path.exists():
                facade_t_cols[slot] = np.load(ft_path)
            if cp_path.exists():
                camera_poses[slot] = np.load(cp_path)

        facade_feats = np.load(sample_dir / 'facade_feats.npy')
        entrance_t = np.array([entry['entrance_t']], dtype=np.float32)

        return {
            'patch_strips': torch.from_numpy(patch_strips),
            'facade_t_cols': torch.from_numpy(facade_t_cols),
            'camera_poses': torch.from_numpy(camera_poses),
            'image_mask': torch.from_numpy(image_mask),
            'facade_feats': torch.from_numpy(facade_feats),
            'entrance_t': torch.from_numpy(entrance_t),
            'has_rtk_label': entry.get('has_rtk_label', False),
            'poi_name': entry['poi_name'],
            'sample_id': entry['sample_id'],
        }
