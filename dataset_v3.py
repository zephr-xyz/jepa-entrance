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
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


MAX_IMAGES = 5
N_COLS = 16
PATCH_DIM = 1024


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
