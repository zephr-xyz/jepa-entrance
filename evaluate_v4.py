"""
Evaluate v4 entrance detection model.

Runs inference on validation set, ray-traces column predictions to building
polygons, fuses multi-image predictions, and computes geographic errors.

Outputs per-POI predictions as JSON for map generation.
"""
import argparse
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model_v4 import EntranceDetectorV4
from dataset_v4 import EntranceDatasetV4
from geometry import raytrace_column_to_building, raytrace_column_to_facade, find_facade_edge


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
    parser.add_argument('--output', default='val_predictions_v4.json')
    parser.add_argument('--split', default='val')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get('args', {})

    model = EntranceDetectorV4(
        d_hidden=model_args.get('d_hidden', 256),
        n_cols=16,
        d_pose=6,
        max_images=5,
        n_layers=model_args.get('n_layers', 4),
        n_heads=model_args.get('n_heads', 8),
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    # Load buildings
    with open(args.buildings_json) as f:
        buildings = json.load(f)

    # Load dataset
    ds = EntranceDatasetV4(args.data_dir, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    predictions = []
    geo_errors = []
    rtk_geo_errors = []

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            out = model(
                batch_gpu['patch_strips'],
                batch_gpu['camera_poses'],
                batch_gpu['image_mask'],
            )

            col_pred = out['col_pred'].cpu().numpy()[0]  # (K,)
            col_probs = out['col_probs'].cpu().numpy()[0]  # (K, 16)
            vis_logit = out['vis_logit'].cpu().numpy()[0]  # (K,)
            mask = batch['image_mask'].numpy()[0]  # (K,)

            sample_id = batch['sample_id'][0]
            poi_id = batch['poi_id'][0]
            bld_id = batch['building_id'][0]
            gt_lat = batch['entrance_lat'][0]
            gt_lon = batch['entrance_lon'][0]
            if isinstance(gt_lat, torch.Tensor):
                gt_lat = gt_lat.item()
                gt_lon = gt_lon.item()

            # Load image metadata for ray-tracing
            sample_dir = Path(args.data_dir) / sample_id
            meta_path = sample_dir / 'image_meta.json'
            try:
                with open(meta_path) as f:
                    img_metas = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                img_metas = []

            # Get building footprint and find facade edge
            building_coords_raw = buildings.get(bld_id, [])
            bld_lnglat = [[c[1], c[0]] for c in building_coords_raw] if building_coords_raw else []

            facade_edge = None
            if len(bld_lnglat) >= 3:
                facade_edge = find_facade_edge(bld_lnglat, gt_lat, gt_lon)

            # Ray-trace each visible image's prediction to the facade edge
            intersection_points = []
            per_image_results = []

            n_images = len(img_metas)
            indices = list(range(min(n_images, 5)))

            for k in range(len(indices)):
                if not mask[k]:
                    continue

                orig_k = indices[k]
                if orig_k >= len(img_metas):
                    continue

                m = img_metas[orig_k]
                vis_pred = vis_logit[k] > 0
                pred_col = float(col_pred[k])

                img_result = {
                    'image_idx': orig_k,
                    'col_pred': pred_col,
                    'col_gt': float(batch['entrance_cols'].numpy()[0, k]),
                    'vis_pred': bool(vis_pred),
                    'vis_gt': bool(batch['visible_flags'].numpy()[0, k]),
                    'vis_logit': float(vis_logit[k]),
                }

                if vis_pred and facade_edge is not None:
                    point = raytrace_column_to_facade(
                        m['cam_lat'], m['cam_lon'],
                        m['cam_compass'], m['hfov'],
                        pred_col, facade_edge,
                        n_cols=16,
                    )
                    if point is not None:
                        intersection_points.append(point)
                        img_result['intersection_lat'] = point[0]
                        img_result['intersection_lon'] = point[1]

                per_image_results.append(img_result)

            # Multi-image fusion: average intersection points
            pred_lat, pred_lon, error_m = None, None, None
            if intersection_points:
                pred_lat = sum(p[0] for p in intersection_points) / len(intersection_points)
                pred_lon = sum(p[1] for p in intersection_points) / len(intersection_points)
                error_m = haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)
                geo_errors.append(error_m)

                is_rtk = batch['has_rtk_label'][0]
                if isinstance(is_rtk, torch.Tensor):
                    is_rtk = is_rtk.item()
                if is_rtk:
                    rtk_geo_errors.append(error_m)

            pred_entry = {
                'sample_id': sample_id,
                'poi_id': poi_id,
                'poi_name': batch['poi_name'][0],
                'has_rtk_label': bool(batch['has_rtk_label'][0]),
                'gt_lat': gt_lat,
                'gt_lon': gt_lon,
                'pred_lat': pred_lat,
                'pred_lon': pred_lon,
                'error_m': error_m,
                'n_visible_pred': len(intersection_points),
                'n_images': int(mask.sum()),
                'per_image': per_image_results,
            }
            predictions.append(pred_entry)

    # Summary statistics
    if geo_errors:
        geo_arr = np.array(geo_errors)
        print(f"\nGeographic evaluation ({len(geo_errors)} POIs with ray-trace hits):")
        print(f"  MAE:    {np.mean(geo_arr):.2f}m")
        print(f"  Median: {np.median(geo_arr):.2f}m")
        print(f"  P90:    {np.percentile(geo_arr, 90):.2f}m")
        print(f"  <1m: {sum(1 for e in geo_errors if e < 1)}, "
              f"<2m: {sum(1 for e in geo_errors if e < 2)}, "
              f"<5m: {sum(1 for e in geo_errors if e < 5)}")

    if rtk_geo_errors:
        rtk_arr = np.array(rtk_geo_errors)
        print(f"\nRTK evaluation ({len(rtk_geo_errors)} POIs):")
        print(f"  MAE:    {np.mean(rtk_arr):.2f}m")
        print(f"  Median: {np.median(rtk_arr):.2f}m")
        print(f"  P90:    {np.percentile(rtk_arr, 90):.2f}m")

    n_no_hit = sum(1 for p in predictions if p['error_m'] is None)
    if n_no_hit:
        print(f"\n  {n_no_hit} POIs had no ray-trace hits (missing building or all rays missed)")

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved {len(predictions)} predictions to {args.output}")


if __name__ == '__main__':
    main()
