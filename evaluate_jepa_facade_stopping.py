"""
Evaluate JEPA + facade stopping combination.

Pipeline:
  1. Run v3 JEPA model to predict facade_t per POI
  2. Convert t_pred to a lat/lon point on the facade polyline
  3. Use that point as the ray target for multi-camera facade stopping
  4. Compare against baselines (midpoint, ground truth target, JEPA-only)

This tests whether JEPA's entrance estimate, when fed into the geometric
multi-camera ray-tracing pipeline, produces better localization than either
approach alone.
"""
import argparse
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model_v3 import JEPAEntranceV3
from dataset_v3 import EntranceDatasetV3
from facade import (
    find_composite_facade,
    project_point_onto_composite_facade,
    _to_local_m,
)
from facade_stopping import (
    haversine_m,
    ray_building_intersection,
    ray_facade_intersection,
    find_facade_edge,
)

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def facade_t_to_latlng(t, facade_result):
    """Convert facade_t (0-1) to geographic lat/lon on the composite facade.

    Walks along the composite polyline to the position at t * total_length.

    Returns:
        (lat, lon) or None if conversion fails
    """
    if facade_result is None:
        return None

    total_length = facade_result['total_length']
    segments = facade_result['segments']
    edges = facade_result['edges']
    cent_lat = facade_result['centroid_lat']
    cent_lng = facade_result['centroid_lon']

    if total_length < 0.01:
        return None

    # Clamp t to [0, 1]
    t = max(0.0, min(1.0, t))
    target_arc = t * total_length

    # Walk along segments to find the position
    for seg_start, seg_end, edge_idx in segments:
        if target_arc <= seg_end or seg_end == segments[-1][1]:
            e = edges[edge_idx]
            seg_len = e['length']
            if seg_len < 0.01:
                # Degenerate segment, use start point
                px, py = e['a']
            else:
                local_frac = (target_arc - seg_start) / seg_len
                local_frac = max(0.0, min(1.0, local_frac))
                px = e['a'][0] + local_frac * (e['b'][0] - e['a'][0])
                py = e['a'][1] + local_frac * (e['b'][1] - e['a'][1])

            # Convert local meters back to lat/lng
            cos_lat = math.cos(cent_lat * DEG_TO_RAD)
            m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

            lon = cent_lng + px / m_per_deg_lng
            lat = cent_lat + py / METERS_PER_DEG_LAT
            return (lat, lon)

    return None


def ray_trace_to_facade_edge(cam_lat, cam_lon, target_lat, target_lon,
                              facade_edge):
    """Ray-trace from camera through target to facade edge with clamping.

    Reuses logic from facade_stopping.ray_facade_intersection.
    """
    return ray_facade_intersection(cam_lat, cam_lon, target_lat, target_lon,
                                   facade_edge)


def evaluate_combined(val_manifest, buildings, image_meta_dir,
                      t_predictions, mode='facade'):
    """Evaluate JEPA + facade stopping combination.

    For each POI:
      1. Convert JEPA's t_pred to a lat/lon target point on the facade
      2. Cast rays from each camera through that target
      3. Find intersection with building/facade edge
      4. Fuse via median

    Args:
        t_predictions: dict mapping sample_id -> t_pred (float)
        mode: 'building' or 'facade'
    """
    geo_errors = []
    rtk_geo_errors = []
    results = []
    n_no_hit = 0
    n_no_facade = 0

    for entry in val_manifest:
        sample_id = entry['sample_id']
        poi_id = entry['poi_id']
        bld_id = entry.get('building_id', '')
        gt_lat = entry['entrance_lat']
        gt_lon = entry['entrance_lon']
        is_rtk = entry.get('has_rtk_label', False)

        if sample_id not in t_predictions:
            continue

        t_pred = t_predictions[sample_id]

        building_coords_raw = buildings.get(bld_id, [])
        if not building_coords_raw or len(building_coords_raw) < 3:
            continue

        bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]

        # Load image metadata for camera positions
        sample_dir = Path(image_meta_dir) / sample_id
        meta_path = sample_dir / 'image_meta.json'
        try:
            with open(meta_path) as f:
                img_metas = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        # Recompute composite facade to convert t_pred to lat/lon
        cos_lat = math.cos(
            (sum(c[1] for c in bld_lnglat) / len(bld_lnglat)) * DEG_TO_RAD)
        m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat
        cent_lng = sum(c[0] for c in bld_lnglat) / len(bld_lnglat)
        cent_lat_bld = sum(c[1] for c in bld_lnglat) / len(bld_lnglat)

        # Camera positions as reference points for composite facade
        cam_positions = [(m['cam_lat'], m['cam_lon']) for m in img_metas]
        ref_points = [
            [(lon - cent_lng) * m_per_deg_lng,
             (lat - cent_lat_bld) * METERS_PER_DEG_LAT]
            for lat, lon in cam_positions
        ]

        ent_m = [(gt_lon - cent_lng) * m_per_deg_lng,
                 (gt_lat - cent_lat_bld) * METERS_PER_DEG_LAT]

        facade_result = find_composite_facade(
            bld_lnglat,
            reference_points_m=ref_points,
            entrance_m=ent_m,
        )

        if facade_result is None:
            n_no_facade += 1
            continue

        # Convert JEPA t_pred to geographic point
        target_point = facade_t_to_latlng(t_pred, facade_result)
        if target_point is None:
            n_no_facade += 1
            continue

        target_lat, target_lon = target_point

        # For facade mode, find the single closest edge to the JEPA-predicted point
        facade_edge = None
        if mode == 'facade':
            facade_edge = find_facade_edge(bld_lnglat, target_lat, target_lon)
            if facade_edge is None:
                n_no_facade += 1
                continue

        # Cast rays from each camera through the JEPA target
        intersection_points = []
        for m in img_metas:
            cam_lat = m['cam_lat']
            cam_lon = m['cam_lon']

            if haversine_m(cam_lat, cam_lon, target_lat, target_lon) < 1.0:
                continue

            if mode == 'facade':
                point = ray_facade_intersection(
                    cam_lat, cam_lon, target_lat, target_lon, facade_edge
                )
            else:
                point = ray_building_intersection(
                    cam_lat, cam_lon, target_lat, target_lon, bld_lnglat
                )

            if point is not None:
                stop_dist = haversine_m(target_lat, target_lon,
                                        point[0], point[1])
                if stop_dist <= 45.0:
                    intersection_points.append(point)

        if not intersection_points:
            n_no_hit += 1
            continue

        # Multi-image fusion: median
        lats = sorted([p[0] for p in intersection_points])
        lons = sorted([p[1] for p in intersection_points])
        pred_lat = lats[len(lats) // 2]
        pred_lon = lons[len(lons) // 2]

        error_m = haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)
        geo_errors.append(error_m)
        if is_rtk:
            rtk_geo_errors.append(error_m)

        results.append({
            'sample_id': sample_id,
            'poi_id': poi_id,
            'poi_name': entry.get('poi_name', ''),
            'has_rtk_label': is_rtk,
            'gt_lat': gt_lat,
            'gt_lon': gt_lon,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'jepa_target_lat': target_lat,
            'jepa_target_lon': target_lon,
            't_pred': t_pred,
            'error_m': error_m,
            'n_rays': len(intersection_points),
        })

    return results, geo_errors, rtk_geo_errors, n_no_hit, n_no_facade


def print_stats(label, geo_errors, rtk_errors):
    """Print summary statistics."""
    if geo_errors:
        arr = np.array(geo_errors)
        print(f"  All POIs ({len(geo_errors)}):")
        print(f"    MAE:    {np.mean(arr):.2f}m")
        print(f"    Median: {np.median(arr):.2f}m")
        print(f"    P90:    {np.percentile(arr, 90):.2f}m")
        print(f"    <1m: {sum(1 for e in geo_errors if e < 1)}, "
              f"<2m: {sum(1 for e in geo_errors if e < 2)}, "
              f"<5m: {sum(1 for e in geo_errors if e < 5)}")
    if rtk_errors:
        arr = np.array(rtk_errors)
        print(f"  RTK POIs ({len(rtk_errors)}):")
        print(f"    MAE:    {np.mean(arr):.2f}m")
        print(f"    Median: {np.median(arr):.2f}m")
        print(f"    P90:    {np.percentile(arr, 90):.2f}m")
        print(f"    <1m: {sum(1 for e in rtk_errors if e < 1)}, "
              f"<2m: {sum(1 for e in rtk_errors if e < 2)}, "
              f"<5m: {sum(1 for e in rtk_errors if e < 5)}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate JEPA + facade stopping combination')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to v3 JEPA model checkpoint')
    parser.add_argument('--data-dir', required=True,
                        help='v3 dataset cache directory')
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--output', default='jepa_facade_stopping_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load v3 JEPA model
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
    print(f"Loaded v3 JEPA model from epoch {ckpt['epoch']}")

    # Load buildings
    with open(args.buildings_json) as f:
        buildings = json.load(f)
    print(f"Loaded {len(buildings)} buildings")

    # Load v3 dataset
    ds = EntranceDatasetV3(args.data_dir, split='val')
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    # Load v3 val manifest
    with open(Path(args.data_dir) / 'val_manifest.json') as f:
        val_manifest = json.load(f)
    print(f"Val manifest: {len(val_manifest)} samples")
    rtk_count = sum(1 for m in val_manifest if m.get('has_rtk_label', False))
    print(f"RTK samples: {rtk_count}")

    # Step 1: Run JEPA inference to get t_pred per sample
    print("\n--- Running JEPA inference ---")
    t_predictions = {}
    jepa_errors = []
    rtk_jepa_errors = []

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
                sample_id = batch['sample_id'][i]
                t_predictions[sample_id] = float(t_pred[i])

                err_m = abs(t_pred[i] - t_true[i]) * facade_lens[i]
                jepa_errors.append(err_m)
                if batch['has_rtk_label'][i]:
                    rtk_jepa_errors.append(err_m)

    print(f"\nJEPA predictions: {len(t_predictions)} samples")
    print("\n=== Baseline: JEPA facade-t only (linear error) ===")
    print_stats("JEPA-only", jepa_errors, rtk_jepa_errors)

    # Step 2: Facade stopping with ground truth target (upper bound)
    from facade_stopping import evaluate_facade_stopping
    print("\n=== Baseline: Facade stopping with GT target (upper bound) ===")
    _, geo_gt, rtk_gt, no_hit_gt = evaluate_facade_stopping(
        val_manifest, buildings, args.data_dir, mode='facade'
    )
    print_stats("GT target", geo_gt, rtk_gt)
    print(f"  No hits: {no_hit_gt}")

    # Step 3: Facade midpoint baseline (t=0.5)
    print("\n=== Baseline: Facade midpoint (t=0.5) ===")
    midpoint_errors = []
    rtk_midpoint_errors = []
    for entry in val_manifest:
        bld_id = entry.get('building_id', '')
        building_coords_raw = buildings.get(bld_id, [])
        if not building_coords_raw or len(building_coords_raw) < 3:
            continue
        bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]
        edge = find_facade_edge(bld_lnglat, entry['entrance_lat'],
                                entry['entrance_lon'])
        if edge is None:
            continue
        a = edge['a_lnglat']
        b = edge['b_lnglat']
        mid_lat = (a[1] + b[1]) / 2
        mid_lon = (a[0] + b[0]) / 2
        err = haversine_m(entry['entrance_lat'], entry['entrance_lon'],
                          mid_lat, mid_lon)
        midpoint_errors.append(err)
        if entry.get('has_rtk_label', False):
            rtk_midpoint_errors.append(err)
    print_stats("Midpoint", midpoint_errors, rtk_midpoint_errors)

    # Step 4: JEPA + facade stopping (the combination)
    print("\n=== JEPA + Facade Stopping (facade edge mode) ===")
    results_fac, geo_fac, rtk_fac, no_hit_fac, no_facade_fac = \
        evaluate_combined(val_manifest, buildings, args.data_dir,
                          t_predictions, mode='facade')
    print_stats("JEPA+facade", geo_fac, rtk_fac)
    print(f"  No hits: {no_hit_fac}, No facade: {no_facade_fac}")

    print("\n=== JEPA + Facade Stopping (building polygon mode) ===")
    results_bld, geo_bld, rtk_bld, no_hit_bld, no_facade_bld = \
        evaluate_combined(val_manifest, buildings, args.data_dir,
                          t_predictions, mode='building')
    print_stats("JEPA+building", geo_bld, rtk_bld)
    print(f"  No hits: {no_hit_bld}, No facade: {no_facade_bld}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY (RTK MAE)")
    print("=" * 60)
    rows = []
    if rtk_jepa_errors:
        rows.append(("JEPA facade-t only", np.mean(rtk_jepa_errors),
                      np.median(rtk_jepa_errors),
                      np.percentile(rtk_jepa_errors, 90)))
    if rtk_midpoint_errors:
        rows.append(("Facade midpoint (t=0.5)", np.mean(rtk_midpoint_errors),
                      np.median(rtk_midpoint_errors),
                      np.percentile(rtk_midpoint_errors, 90)))
    if rtk_gt:
        rows.append(("Facade stop + GT target", np.mean(rtk_gt),
                      np.median(rtk_gt), np.percentile(rtk_gt, 90)))
    if rtk_fac:
        rows.append(("JEPA + facade stop", np.mean(rtk_fac),
                      np.median(rtk_fac), np.percentile(rtk_fac, 90)))
    if rtk_bld:
        rows.append(("JEPA + building stop", np.mean(rtk_bld),
                      np.median(rtk_bld), np.percentile(rtk_bld, 90)))

    print(f"{'Approach':<30} {'MAE':>8} {'Median':>8} {'P90':>8}")
    print("-" * 60)
    for name, mae, med, p90 in rows:
        print(f"{name:<30} {mae:>7.2f}m {med:>7.2f}m {p90:>7.2f}m")

    # Save results
    output = {
        'jepa_facade_mode': {
            'results': results_fac,
            'n_no_hit': no_hit_fac,
            'n_no_facade': no_facade_fac,
        },
        'jepa_building_mode': {
            'results': results_bld,
            'n_no_hit': no_hit_bld,
            'n_no_facade': no_facade_bld,
        },
        't_predictions': t_predictions,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == '__main__':
    main()
