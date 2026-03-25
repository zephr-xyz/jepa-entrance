"""
Dataset for v4 entrance detection model.

Each sample contains:
  - patch_strips: (K, 16, 1024) horizontal patch features per image
  - camera_poses: (K, 6) camera pose features per image
  - image_mask: (K,) which images are valid
  - entrance_cols: (K,) ground truth entrance column per image
  - visible_flags: (K,) whether entrance is visible in each image

Training augmentation: when n_images > MAX_IMAGES, randomly
sample a different subset of images each time __getitem__ is called.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


MAX_IMAGES = 5
N_COLS = 16
PATCH_DIM = 1024
D_POSE = 6


class EntranceDatasetV4(Dataset):
    def __init__(self, cache_dir: str, split: str = 'train'):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.is_train = (split == 'train')
        manifest_path = self.cache_dir / f'{split}_manifest.json'
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        print(f"EntranceDatasetV4 [{split}]: {len(self.manifest)} samples")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        sample_dir = self.cache_dir / entry['sample_id']

        n_images = entry.get('n_images', 1)

        # Load per-image metadata (entrance_col, visible, etc.)
        meta_path = sample_dir / 'image_meta.json'
        with open(meta_path) as f:
            image_metas = json.load(f)

        # Choose which images to load
        if self.is_train and n_images > MAX_IMAGES:
            # Prefer visible images but include some non-visible for visibility training
            visible_idx = [i for i in range(n_images) if image_metas[i]['visible']]
            non_visible_idx = [i for i in range(n_images) if not image_metas[i]['visible']]

            # Take up to MAX_IMAGES-1 visible, fill rest with non-visible
            n_vis = min(len(visible_idx), MAX_IMAGES - 1) if non_visible_idx else min(len(visible_idx), MAX_IMAGES)
            if visible_idx:
                chosen_vis = list(np.random.choice(visible_idx, n_vis, replace=False))
            else:
                chosen_vis = []

            n_remaining = MAX_IMAGES - len(chosen_vis)
            if non_visible_idx and n_remaining > 0:
                n_nv = min(len(non_visible_idx), n_remaining)
                chosen_nv = list(np.random.choice(non_visible_idx, n_nv, replace=False))
            else:
                chosen_nv = []

            indices = sorted(chosen_vis + chosen_nv)
        else:
            indices = list(range(min(n_images, MAX_IMAGES)))

        patch_strips = np.zeros((MAX_IMAGES, N_COLS, PATCH_DIM), dtype=np.float32)
        camera_poses = np.zeros((MAX_IMAGES, D_POSE), dtype=np.float32)
        image_mask = np.zeros(MAX_IMAGES, dtype=np.bool_)
        entrance_cols = np.zeros(MAX_IMAGES, dtype=np.float32)
        visible_flags = np.zeros(MAX_IMAGES, dtype=np.bool_)

        for slot, k in enumerate(indices):
            ps_path = sample_dir / f'patch_strip_{k}.npy'
            cp_path = sample_dir / f'camera_pose_{k}.npy'

            if ps_path.exists():
                patch_strips[slot] = np.load(ps_path)
                image_mask[slot] = True
            if cp_path.exists():
                camera_poses[slot] = np.load(cp_path)

            m = image_metas[k]
            entrance_cols[slot] = m['entrance_col']
            visible_flags[slot] = m['visible']

        return {
            'patch_strips': torch.from_numpy(patch_strips),
            'camera_poses': torch.from_numpy(camera_poses),
            'image_mask': torch.from_numpy(image_mask),
            'entrance_cols': torch.from_numpy(entrance_cols),
            'visible_flags': torch.from_numpy(visible_flags),
            'has_rtk_label': entry.get('has_rtk_label', False),
            'poi_name': entry.get('poi_name', ''),
            'sample_id': entry['sample_id'],
            'poi_id': entry['poi_id'],
            'building_id': entry.get('building_id', ''),
            'building_centroid_lat': entry.get('building_centroid_lat', 0),
            'building_centroid_lon': entry.get('building_centroid_lon', 0),
            'entrance_lat': entry.get('entrance_lat', 0),
            'entrance_lon': entry.get('entrance_lon', 0),
            'n_total_images': entry.get('n_images', 0),
        }
