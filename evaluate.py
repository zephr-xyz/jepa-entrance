"""
Run v3 model inference on val set, output per-POI predictions as JSON.
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model import JEPAEntranceV3
from dataset import EntranceDatasetV3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', default='val_predictions.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get('args', {})

    model = JEPAEntranceV3(
        d_latent=model_args.get('d_latent', 128),
        d_facade=32,
        lambda_sigreg=model_args.get('lambda_sigreg', 0.05),
        mu_entrance=model_args.get('mu_entrance', 10.0),
        max_images=5,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    ds = EntranceDatasetV3(args.data_dir, split='val')
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    results = []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            out = model(
                batch_gpu['patch_strips'],
                batch_gpu['facade_t_cols'],
                batch_gpu['camera_poses'],
                batch_gpu['image_mask'],
                batch_gpu['facade_feats'],
                batch_gpu['entrance_t'],
            )

            t_pred = out['t_pred'].cpu().numpy().flatten()
            t_true = batch['entrance_t'].numpy().flatten()
            facade_lens = batch['facade_feats'][:, 4].numpy()

            for i in range(len(t_pred)):
                results.append({
                    'sample_id': batch['sample_id'][i],
                    'poi_name': batch['poi_name'][i],
                    'has_rtk_label': bool(batch['has_rtk_label'][i]),
                    't_pred': float(t_pred[i]),
                    't_true': float(t_true[i]),
                    'facade_length_m': float(facade_lens[i]),
                    'error_t': float(abs(t_pred[i] - t_true[i])),
                    'error_m': float(abs(t_pred[i] - t_true[i]) * facade_lens[i]),
                })

    # Summary
    all_errors = [r['error_m'] for r in results]
    rtk_errors = [r['error_m'] for r in results if r['has_rtk_label']]
    print(f"\nAll val: MAE={np.mean(all_errors):.2f}m, median={np.median(all_errors):.2f}m")
    if rtk_errors:
        print(f"RTK val: MAE={np.mean(rtk_errors):.2f}m, median={np.median(rtk_errors):.2f}m, n={len(rtk_errors)}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == '__main__':
    main()
