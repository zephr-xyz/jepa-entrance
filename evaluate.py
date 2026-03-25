"""
Evaluate trained JEPA model and produce entrance predictions.

Outputs:
  1. Per-POI predictions with error metrics
  2. Updated entrance coordinates (improved entrance_lat/lon)
  3. Comparison against baseline (facade midpoint)
  4. Export format compatible with embedding-tiles

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt \
        --data-dir /path/to/cache --output predictions.json
"""
import argparse
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model import JEPAEntrance
from dataset import EntranceDataset


DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def t_to_latlng(t, facade_feats, building_coords_latlng, entrance_lat_orig, entrance_lon_orig):
    """Convert predicted t ∈ [0,1] back to lat/lng coordinates.

    Uses the facade edge endpoints stored in facade_feats to interpolate.
    Facade feats[0:4] = (ax, ay, bx, by) in local meters from building centroid.
    We need the building centroid to convert back.
    """
    # Reconstruct from facade features
    # facade_feats indices: 0-3 = edge a/b in local meters, 4 = length, 5 = bearing
    ax, ay, bx, by = facade_feats[0], facade_feats[1], facade_feats[2], facade_feats[3]

    # Interpolate in local meters
    px = ax + t * (bx - ax)
    py = ay + t * (by - ay)

    # Convert back to lat/lng using building centroid
    # The local coordinate system is centered on the building centroid
    # We need the centroid lat/lng to convert back
    # For now, use the original entrance as an anchor point
    # (the offset from facade midpoint to predicted point)

    # Facade midpoint in local meters
    mx = (ax + bx) / 2
    my = (ay + by) / 2

    # Offset from midpoint to predicted point (in meters)
    dx_m = px - mx
    dy_m = py - my

    # Original entrance ≈ facade midpoint, so offset from it
    cos_lat = math.cos(entrance_lat_orig * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    # The original entrance_lat/lon is at approximately t=0.5 (midpoint)
    # Adjust by the offset
    pred_lng = entrance_lon_orig + dx_m / m_per_deg_lng
    pred_lat = entrance_lat_orig + dy_m / METERS_PER_DEG_LAT

    return pred_lat, pred_lng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', default='predictions.json')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get('args', {})

    model = JEPAEntrance(
        d_latent=model_args.get('d_latent', 256),
        d_facade=32,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    # Load val dataset
    val_ds = EntranceDataset(args.data_dir, split='val')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Also load manifests for metadata
    val_manifest = json.load(open(Path(args.data_dir) / 'val_manifest.json'))
    manifest_by_id = {m['sample_id']: m for m in val_manifest}

    # Run predictions
    predictions = []
    all_t_errors = []
    all_m_errors = []
    baseline_m_errors = []

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            t_pred = model.predict_entrance(
                batch_gpu['cls_emb'], batch_gpu['caption_emb'],
                batch_gpu['kp_stats'], batch_gpu['compass'],
                batch_gpu['facade_feats'],
            )

            t_pred_np = t_pred.cpu().numpy().flatten()
            t_true_np = batch['entrance_t'].numpy().flatten()
            facade_feats_np = batch['facade_feats'].numpy()

            for j in range(len(t_pred_np)):
                sample_id = batch['sample_id'][j]
                meta = manifest_by_id.get(sample_id, {})

                t_p = float(t_pred_np[j])
                t_t = float(t_true_np[j])
                facade_len = float(facade_feats_np[j, 4])

                t_err = abs(t_p - t_t)
                m_err = t_err * facade_len
                baseline_err = abs(0.5 - t_t) * facade_len

                all_t_errors.append(t_err)
                all_m_errors.append(m_err)
                baseline_m_errors.append(baseline_err)

                predictions.append({
                    'poi_id': meta.get('poi_id', ''),
                    'poi_name': meta.get('poi_name', batch['poi_name'][j]),
                    'image_id': meta.get('image_id', ''),
                    'entrance_t_pred': t_p,
                    'entrance_t_true': t_t,
                    'error_t': t_err,
                    'error_meters': m_err,
                    'baseline_error_meters': baseline_err,
                    'facade_length_m': facade_len,
                    'entrance_lat_orig': meta.get('entrance_lat', 0),
                    'entrance_lon_orig': meta.get('entrance_lon', 0),
                })

    all_t_errors = np.array(all_t_errors)
    all_m_errors = np.array(all_m_errors)
    baseline_m_errors = np.array(baseline_m_errors)

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({len(predictions)} POIs)")
    print(f"{'='*60}")
    print(f"\nBaseline (facade midpoint, t=0.5):")
    print(f"  MAE:    {np.mean(baseline_m_errors):.2f}m")
    print(f"  Median: {np.median(baseline_m_errors):.2f}m")
    print(f"  P90:    {np.percentile(baseline_m_errors, 90):.2f}m")
    print(f"\nJEPA Model:")
    print(f"  MAE:    {np.mean(all_m_errors):.2f}m")
    print(f"  Median: {np.median(all_m_errors):.2f}m")
    print(f"  P90:    {np.percentile(all_m_errors, 90):.2f}m")
    print(f"  MAE t:  {np.mean(all_t_errors):.4f}")

    if np.mean(baseline_m_errors) > 0:
        improvement = (1 - np.mean(all_m_errors) / np.mean(baseline_m_errors)) * 100
        print(f"\n  Improvement over baseline: {improvement:.1f}%")

    # Error distribution
    print(f"\nError buckets:")
    for thresh in [1, 2, 3, 5, 10]:
        pct = np.mean(all_m_errors < thresh) * 100
        baseline_pct = np.mean(baseline_m_errors < thresh) * 100
        print(f"  < {thresh}m: JEPA {pct:.1f}% | Baseline {baseline_pct:.1f}%")

    # Worst predictions
    predictions.sort(key=lambda p: -p['error_meters'])
    print(f"\nWorst 10 predictions:")
    for p in predictions[:10]:
        print(f"  {p['poi_name'][:30]:30s} | "
              f"err={p['error_meters']:.1f}m "
              f"(t_pred={p['entrance_t_pred']:.3f}, "
              f"t_true={p['entrance_t_true']:.3f}, "
              f"facade={p['facade_length_m']:.1f}m)")

    # Best predictions
    predictions.sort(key=lambda p: p['error_meters'])
    print(f"\nBest 10 predictions:")
    for p in predictions[:10]:
        print(f"  {p['poi_name'][:30]:30s} | "
              f"err={p['error_meters']:.1f}m "
              f"(t_pred={p['entrance_t_pred']:.3f}, "
              f"t_true={p['entrance_t_true']:.3f})")

    # Save predictions
    output = {
        'summary': {
            'n_samples': len(predictions),
            'jepa_mae_meters': float(np.mean(all_m_errors)),
            'jepa_median_meters': float(np.median(all_m_errors)),
            'jepa_p90_meters': float(np.percentile(all_m_errors, 90)),
            'baseline_mae_meters': float(np.mean(baseline_m_errors)),
            'baseline_median_meters': float(np.median(baseline_m_errors)),
            'improvement_pct': float(improvement) if np.mean(baseline_m_errors) > 0 else 0,
        },
        'predictions': sorted(predictions, key=lambda p: p['poi_name']),
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nPredictions saved to {args.output}")


if __name__ == '__main__':
    main()
