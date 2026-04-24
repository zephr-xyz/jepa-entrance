"""
Predict entrance locations for all POIs using the trained JEPA model.

For each POI:
  1. Run JEPA to predict facade_t
  2. Map t_pred to lat/lon on the facade edge
  3. Output updated entrance coordinates

Usage:
    python predict_entrances.py \
        --checkpoint best_model_se.pt \
        --data-dir cache \
        --buildings-json buildings.json \
        --output updated_entrances.json
"""
import argparse
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model import JEPAEntranceV3
from dataset import EntranceDatasetV3
from geometry import find_facade_edge

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--output', default='updated_entrances.json')
    parser.add_argument('--split', default='all',
                        help='"train", "val", or "all" (both splits)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
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

    # Load buildings
    with open(args.buildings_json) as f:
        buildings = json.load(f)

    # Load manifest(s)
    data_dir = Path(args.data_dir)
    manifests = {}
    splits = ['train', 'val'] if args.split == 'all' else [args.split]
    for split in splits:
        mp = data_dir / f'{split}_manifest.json'
        if mp.exists():
            with open(mp) as f:
                for entry in json.load(f):
                    manifests[entry['sample_id']] = entry

    print(f"Loaded {len(manifests)} samples across splits: {splits}")

    # Run inference on each split
    results = []
    for split in splits:
        mp = data_dir / f'{split}_manifest.json'
        if not mp.exists():
            continue
        ds = EntranceDatasetV3(args.data_dir, split=split)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

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

                for i in range(len(t_pred)):
                    sid = batch['sample_id'][i]
                    m = manifests.get(sid)
                    if m is None:
                        continue

                    bld_id = m['building_id']
                    bld_raw = buildings.get(bld_id, [])
                    if not bld_raw or len(bld_raw) < 3:
                        continue

                    bld_lnglat = [[c[1], c[0]] for c in bld_raw]
                    edge = find_facade_edge(bld_lnglat,
                                            m['entrance_lat'], m['entrance_lon'])
                    if edge is None:
                        continue

                    a_ll = edge['a_lnglat']
                    b_ll = edge['b_lnglat']
                    tp = max(0.0, min(1.0, float(t_pred[i])))

                    pred_lat = a_ll[1] + tp * (b_ll[1] - a_ll[1])
                    pred_lon = a_ll[0] + tp * (b_ll[0] - a_ll[0])

                    results.append({
                        'poi_id': m['poi_id'],
                        'poi_name': m.get('poi_name', ''),
                        'building_id': bld_id,
                        'original_entrance_lat': m['entrance_lat'],
                        'original_entrance_lon': m['entrance_lon'],
                        'predicted_entrance_lat': pred_lat,
                        'predicted_entrance_lon': pred_lon,
                        't_pred': float(t_pred[i]),
                        'facade_length_m': m.get('facade_length_m', 0),
                        'has_rtk_label': m.get('has_rtk_label', False),
                    })

    # Summary
    print(f"\nPredicted entrances for {len(results)} POIs")
    rtk = [r for r in results if r['has_rtk_label']]
    if rtk:
        errors = [haversine_m(r['original_entrance_lat'], r['original_entrance_lon'],
                               r['predicted_entrance_lat'], r['predicted_entrance_lon'])
                  for r in rtk]
        arr = np.array(errors)
        print(f"RTK samples ({len(rtk)}): prediction shift MAE={np.mean(arr):.2f}m, "
              f"median={np.median(arr):.2f}m")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
