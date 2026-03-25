"""
Facade stopping for entrance localization.

Adapted from the snip localizer (overture_project PR #8).

For each POI + image, casts a ray from the camera through a target point
and finds where it first intersects a building polygon. Multiple images
per POI are fused by taking the median of intersection points.

This provides a geometry-only baseline (no JEPA) for entrance localization
that can be compared against the JEPA model's predictions.
"""
import json
import math
import argparse
import numpy as np
from pathlib import Path

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


def ray_building_intersection(cam_lat, cam_lon, target_lat, target_lon,
                               building_coords_lnglat):
    """Cast a ray from camera through target point, find first building hit.

    Same approach as snip localizer's facade stopping:
    extend the ray well past the target and intersect with each building edge.

    Args:
        cam_lat, cam_lon: camera position
        target_lat, target_lon: target point (entrance estimate or snip position)
        building_coords_lnglat: list of [lng, lat] polygon vertices

    Returns:
        (lat, lon) of closest intersection point, or None
    """
    if len(building_coords_lnglat) < 3:
        return None

    # Use building centroid as local coordinate origin
    cent_lng = sum(c[0] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cent_lat = sum(c[1] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    # Camera and target in local meters
    cam_x = (cam_lon - cent_lng) * m_per_deg_lng
    cam_y = (cam_lat - cent_lat) * METERS_PER_DEG_LAT
    tgt_x = (target_lon - cent_lng) * m_per_deg_lng
    tgt_y = (target_lat - cent_lat) * METERS_PER_DEG_LAT

    # Ray direction
    dx = tgt_x - cam_x
    dy = tgt_y - cam_y
    ray_len = math.sqrt(dx * dx + dy * dy)
    if ray_len < 0.01:
        return None

    # Normalize and extend ray well past the building (500m)
    ray_dx = dx / ray_len
    ray_dy = dy / ray_len

    # Building edges in local meters
    fp_m = [[(c[0] - cent_lng) * m_per_deg_lng,
             (c[1] - cent_lat) * METERS_PER_DEG_LAT]
            for c in building_coords_lnglat]

    best_t_ray = float('inf')
    best_point = None

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        fx, fy = b[0] - a[0], b[1] - a[1]

        ax = a[0] - cam_x
        ay = a[1] - cam_y

        denom = ray_dy * fx - ray_dx * fy
        if abs(denom) < 1e-10:
            continue

        t_seg = (ray_dx * ay - ray_dy * ax) / denom
        t_ray = (fx * ay - fy * ax) / denom

        if t_ray < 0:
            continue  # behind camera
        if t_seg < 0 or t_seg > 1:
            continue  # outside segment

        if t_ray < best_t_ray:
            best_t_ray = t_ray
            ix = cam_x + t_ray * ray_dx
            iy = cam_y + t_ray * ray_dy
            int_lon = cent_lng + ix / m_per_deg_lng
            int_lat = cent_lat + iy / METERS_PER_DEG_LAT
            best_point = (int_lat, int_lon)

    return best_point


def ray_facade_intersection(cam_lat, cam_lon, target_lat, target_lon,
                             facade_edge):
    """Cast a ray from camera through target point, intersect with facade edge.

    Like ray_building_intersection but constrained to a single facade edge,
    with clamping to stay on the edge.

    Args:
        facade_edge: dict with 'a_m', 'b_m', 'centroid_lat', 'centroid_lon'

    Returns:
        (lat, lon) of intersection point on facade, or None
    """
    cent_lat = facade_edge['centroid_lat']
    cent_lng = facade_edge['centroid_lon']
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    cam_x = (cam_lon - cent_lng) * m_per_deg_lng
    cam_y = (cam_lat - cent_lat) * METERS_PER_DEG_LAT
    tgt_x = (target_lon - cent_lng) * m_per_deg_lng
    tgt_y = (target_lat - cent_lat) * METERS_PER_DEG_LAT

    dx = tgt_x - cam_x
    dy = tgt_y - cam_y
    ray_len = math.sqrt(dx * dx + dy * dy)
    if ray_len < 0.01:
        return None

    ray_dx = dx / ray_len
    ray_dy = dy / ray_len

    a = facade_edge['a_m']
    b = facade_edge['b_m']
    fx, fy = b[0] - a[0], b[1] - a[1]

    ax = a[0] - cam_x
    ay = a[1] - cam_y

    denom = ray_dy * fx - ray_dx * fy
    if abs(denom) < 1e-10:
        return None

    t_seg = (ray_dx * ay - ray_dy * ax) / denom
    t_ray = (fx * ay - fy * ax) / denom

    if t_ray < 0:
        return None

    # Clamp to facade edge
    t_seg = max(0.0, min(1.0, t_seg))

    ix = a[0] + t_seg * fx
    iy = a[1] + t_seg * fy
    int_lon = cent_lng + ix / m_per_deg_lng
    int_lat = cent_lat + iy / METERS_PER_DEG_LAT

    return (int_lat, int_lon)


def find_facade_edge(building_coords_lnglat, entrance_lat, entrance_lon):
    """Find the single closest building edge to the entrance location."""
    if len(building_coords_lnglat) < 3:
        return None

    cent_lng = sum(c[0] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cent_lat = sum(c[1] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    def to_m(coord):
        return [(coord[0] - cent_lng) * m_per_deg_lng,
                (coord[1] - cent_lat) * METERS_PER_DEG_LAT]

    fp_m = [to_m(c) for c in building_coords_lnglat]
    ent_m = to_m([entrance_lon, entrance_lat])

    best_edge = None
    best_score = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx_e = ent_m[0] - mx
        dy_e = ent_m[1] - my
        dist = math.sqrt(dx_e * dx_e + dy_e * dy_e)

        if dist < best_score:
            best_score = dist
            best_edge = {
                'a_m': a, 'b_m': b,
                'a_lnglat': building_coords_lnglat[i],
                'b_lnglat': building_coords_lnglat[i + 1],
                'length': length,
                'centroid_lat': cent_lat,
                'centroid_lon': cent_lng,
            }

    return best_edge


def evaluate_facade_stopping(val_manifest, buildings, image_meta_dir,
                              mode='building', use_entrance_as_target=True):
    """Evaluate facade stopping on the validation set.

    Modes:
        'building': ray-trace against full building polygon (snip localizer approach)
        'facade': ray-trace against selected facade edge only

    If use_entrance_as_target=True, uses the POI's entrance estimate as the ray
    target (geometry-only baseline). Otherwise would need model predictions.
    """
    geo_errors = []
    rtk_geo_errors = []
    results = []
    n_no_hit = 0

    for entry in val_manifest:
        poi_id = entry['poi_id']
        bld_id = entry.get('building_id', '')
        gt_lat = entry['entrance_lat']
        gt_lon = entry['entrance_lon']
        is_rtk = entry.get('has_rtk_label', False)

        building_coords_raw = buildings.get(bld_id, [])
        if not building_coords_raw or len(building_coords_raw) < 3:
            continue

        bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]

        # For facade mode, find the facade edge
        facade_edge = None
        if mode == 'facade':
            facade_edge = find_facade_edge(bld_lnglat, gt_lat, gt_lon)
            if facade_edge is None:
                continue

        # Load per-image metadata
        sample_dir = Path(image_meta_dir) / entry['sample_id']
        meta_path = sample_dir / 'image_meta.json'
        try:
            with open(meta_path) as f:
                img_metas = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        # Target point for ray: entrance estimate
        target_lat = gt_lat
        target_lon = gt_lon

        # Cast rays from each camera through the target
        intersection_points = []
        for m in img_metas:
            cam_lat = m['cam_lat']
            cam_lon = m['cam_lon']

            # Skip if camera is very close to target (degenerate ray)
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
                # Filter out stops too far from target (>45m, same as snip localizer)
                stop_dist = haversine_m(target_lat, target_lon, point[0], point[1])
                if stop_dist <= 45.0:
                    intersection_points.append(point)

        if not intersection_points:
            n_no_hit += 1
            continue

        # Multi-image fusion: median (same as snip localizer)
        lats = sorted([p[0] for p in intersection_points])
        lons = sorted([p[1] for p in intersection_points])
        pred_lat = lats[len(lats) // 2]
        pred_lon = lons[len(lons) // 2]

        error_m = haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)
        geo_errors.append(error_m)
        if is_rtk:
            rtk_geo_errors.append(error_m)

        results.append({
            'sample_id': entry['sample_id'],
            'poi_id': poi_id,
            'poi_name': entry.get('poi_name', ''),
            'has_rtk_label': is_rtk,
            'gt_lat': gt_lat,
            'gt_lon': gt_lon,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'error_m': error_m,
            'n_rays': len(intersection_points),
        })

    return results, geo_errors, rtk_geo_errors, n_no_hit


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate facade stopping baseline for entrance localization')
    parser.add_argument('--val-manifest', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--data-dir', required=True,
                        help='Dataset cache dir with per-sample image_meta.json')
    parser.add_argument('--output', default='facade_stopping_results.json')
    args = parser.parse_args()

    with open(args.val_manifest) as f:
        val_manifest = json.load(f)

    with open(args.buildings_json) as f:
        buildings = json.load(f)

    # Filter to samples that have image metadata
    print(f"Val manifest: {len(val_manifest)} samples")
    rtk_samples = [m for m in val_manifest if m.get('has_rtk_label', False)]
    print(f"RTK samples: {len(rtk_samples)}")

    # Mode 1: Full building polygon (snip localizer approach)
    print("\n=== Mode: Full Building Polygon ===")
    results_bld, geo_bld, rtk_bld, no_hit_bld = evaluate_facade_stopping(
        val_manifest, buildings, args.data_dir, mode='building'
    )
    if geo_bld:
        geo_arr = np.array(geo_bld)
        print(f"All POIs ({len(geo_bld)}):")
        print(f"  MAE:    {np.mean(geo_arr):.2f}m")
        print(f"  Median: {np.median(geo_arr):.2f}m")
        print(f"  P90:    {np.percentile(geo_arr, 90):.2f}m")
    if rtk_bld:
        rtk_arr = np.array(rtk_bld)
        print(f"RTK POIs ({len(rtk_bld)}):")
        print(f"  MAE:    {np.mean(rtk_arr):.2f}m")
        print(f"  Median: {np.median(rtk_arr):.2f}m")
        print(f"  P90:    {np.percentile(rtk_arr, 90):.2f}m")
    print(f"No hits: {no_hit_bld}")

    # Mode 2: Facade edge only
    print("\n=== Mode: Facade Edge Only ===")
    results_fac, geo_fac, rtk_fac, no_hit_fac = evaluate_facade_stopping(
        val_manifest, buildings, args.data_dir, mode='facade'
    )
    if geo_fac:
        geo_arr = np.array(geo_fac)
        print(f"All POIs ({len(geo_fac)}):")
        print(f"  MAE:    {np.mean(geo_arr):.2f}m")
        print(f"  Median: {np.median(geo_arr):.2f}m")
        print(f"  P90:    {np.percentile(geo_arr, 90):.2f}m")
    if rtk_fac:
        rtk_arr = np.array(rtk_fac)
        print(f"RTK POIs ({len(rtk_fac)}):")
        print(f"  MAE:    {np.mean(rtk_arr):.2f}m")
        print(f"  Median: {np.median(rtk_arr):.2f}m")
        print(f"  P90:    {np.percentile(rtk_arr, 90):.2f}m")
    print(f"No hits: {no_hit_fac}")

    # Mode 3: Facade midpoint baseline (t=0.5, no cameras needed)
    print("\n=== Mode: Facade Midpoint Baseline (t=0.5) ===")
    midpoint_errors = []
    rtk_midpoint_errors = []
    for entry in val_manifest:
        bld_id = entry.get('building_id', '')
        building_coords_raw = buildings.get(bld_id, [])
        if not building_coords_raw or len(building_coords_raw) < 3:
            continue
        bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]
        edge = find_facade_edge(bld_lnglat, entry['entrance_lat'], entry['entrance_lon'])
        if edge is None:
            continue
        a = edge['a_lnglat']
        b = edge['b_lnglat']
        mid_lat = (a[1] + b[1]) / 2
        mid_lon = (a[0] + b[0]) / 2
        err = haversine_m(entry['entrance_lat'], entry['entrance_lon'], mid_lat, mid_lon)
        midpoint_errors.append(err)
        if entry.get('has_rtk_label', False):
            rtk_midpoint_errors.append(err)

    if midpoint_errors:
        arr = np.array(midpoint_errors)
        print(f"All POIs ({len(midpoint_errors)}):")
        print(f"  MAE:    {np.mean(arr):.2f}m")
        print(f"  Median: {np.median(arr):.2f}m")
        print(f"  P90:    {np.percentile(arr, 90):.2f}m")
    if rtk_midpoint_errors:
        arr = np.array(rtk_midpoint_errors)
        print(f"RTK POIs ({len(rtk_midpoint_errors)}):")
        print(f"  MAE:    {np.mean(arr):.2f}m")
        print(f"  Median: {np.median(arr):.2f}m")
        print(f"  P90:    {np.percentile(arr, 90):.2f}m")

    # Save results
    output = {
        'building_mode': {'results': results_bld, 'n_no_hit': no_hit_bld},
        'facade_mode': {'results': results_fac, 'n_no_hit': no_hit_fac},
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == '__main__':
    main()
