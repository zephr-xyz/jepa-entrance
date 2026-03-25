"""
Prepare v3 training dataset with patch embeddings + camera pose.

For each POI, fetches up to K Mapillary images and computes:
  - Horizontal patch strip (vertical average of 16x16 → 16x1024)
  - Per-column facade_t (ray-facade intersection)
  - Camera pose features (8d)

Usage:
    python prepare_dataset_v3.py \
        --tiles-dir /path/to/z14 \
        --buildings-json /path/to/buildings.json \
        --ground-truth-labels ground_truth_labels.json \
        --s3-index-cache s3_sequence_lookup.json \
        --output-dir /path/to/cache_v3 \
        --max-images-per-poi 5
"""
import argparse
import json
import math
import os
import random
import numpy as np
import boto3
from io import BytesIO
from pathlib import Path

from dataset import compute_facade_and_entrance_t
from facade import (
    find_composite_facade,
    project_point_onto_composite_facade,
    compute_composite_facade_and_entrance_t,
    _to_local_m,
)
from geometry import (
    camera_hfov_deg,
    compute_camera_pose_features,
    compute_patch_column_facade_t,
    compute_patch_column_facade_t_composite,
)

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320
MAX_IMAGES = 15  # Save up to 15 images; dataset samples 5 at training time
N_COLS = 16


def load_embedding_tiles(tiles_dir):
    tiles_path = Path(tiles_dir)
    all_pois = []
    for json_file in sorted(tiles_path.glob('*.json')):
        with open(json_file) as f:
            tile = json.load(f)
        for wp in tile.get('waypoints', []):
            if not wp.get('entrance_lat') or not wp.get('entrance_lon'):
                continue
            if not wp.get('mapillary_ids'):
                continue
            if not wp.get('overture_building_id'):
                continue
            all_pois.append({
                'id': wp['id'],
                'name': wp.get('name', ''),
                'lat': wp['latitude'],
                'lon': wp['longitude'],
                'entrance_lat': wp['entrance_lat'],
                'entrance_lon': wp['entrance_lon'],
                'mapillary_ids': wp['mapillary_ids'],
                'overture_building_id': wp['overture_building_id'],
                'enclosing_roads': wp.get('enclosing_roads', []),
            })
    print(f"Loaded {len(all_pois)} POIs with entrances + mapillary + buildings")
    return all_pois


def load_buildings(buildings_json):
    with open(buildings_json) as f:
        return json.load(f)


def fetch_image_features(s3_client, bucket, sequence_id, image_id):
    """Fetch patch embeddings + metadata for one image."""
    prefix = f"{sequence_id}/{image_id}/"

    def get_npz(key):
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=prefix + key)
            data = np.load(BytesIO(resp['Body'].read()))
            return dict(data)
        except Exception:
            return None

    def get_json(key):
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=prefix + key)
            return json.loads(resp['Body'].read())
        except Exception:
            return None

    patch_data = get_npz('patch_embeddings.npz')
    metadata = get_json('metadata.json')

    if patch_data is None or metadata is None:
        return None

    # Extract patch embeddings (256, 1024) = 16x16 grid of 1024-dim
    patches = None
    for key in patch_data:
        patches = patch_data[key]
        break

    if patches is None:
        return None

    # Convert float16 → float32
    patches = patches.astype(np.float32)

    # Reshape to 16x16 grid and average vertically → (16, 1024) horizontal strip
    if patches.shape == (256, 1024):
        patches_2d = patches.reshape(16, 16, 1024)
        patch_strip = patches_2d.mean(axis=0)  # average over rows → (16, 1024)
    elif patches.ndim == 2:
        n = patches.shape[0]
        side = int(math.sqrt(n))
        if side * side == n:
            patches_2d = patches.reshape(side, side, patches.shape[1])
            # Pool to 16 columns
            if side >= 16:
                col_size = side // 16
                patch_strip = np.zeros((16, patches.shape[1]), dtype=np.float32)
                for c in range(16):
                    patch_strip[c] = patches_2d[:, c*col_size:(c+1)*col_size, :].mean(axis=(0, 1))
            else:
                # Fewer than 16 columns — pad
                patch_strip = np.zeros((16, patches.shape[1]), dtype=np.float32)
                patch_strip[:side] = patches_2d.mean(axis=0)
        else:
            # Unknown layout — just take first 16 rows
            patch_strip = np.zeros((16, 1024), dtype=np.float32)
            n_take = min(16, patches.shape[0])
            patch_strip[:n_take] = patches[:n_take]
    else:
        return None

    return {
        'patch_strip': patch_strip,  # (16, 1024)
        'metadata': metadata,
    }


def get_facade_geometry_m(building_coords, enclosing_roads):
    """Get facade edge geometry in local meters for camera-facade computation.

    Returns facade_a_m, facade_b_m, midpoint_m, normal, centroid_lat, centroid_lon
    or None if computation fails.
    """
    if len(building_coords) < 3:
        return None

    cent_lng = sum(c[0] for c in building_coords) / len(building_coords)
    cent_lat = sum(c[1] for c in building_coords) / len(building_coords)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    def to_m(coord):
        return [(coord[0] - cent_lng) * m_per_deg_lng,
                (coord[1] - cent_lat) * METERS_PER_DEG_LAT]

    fp_m = [to_m(c) for c in building_coords]
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

        # Score by distance from centroid to edge (prefer outward-facing edges)
        dx_c = poly_cx - mx
        dy_c = poly_cy - my
        dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)

        if dist_c < best_score:
            # Not ideal heuristic but works for road-facing detection
            pass

        # Use same heuristic as v2: closest edge to entrance (but we don't have
        # entrance here yet). Use road proximity if available.
        if enclosing_roads:
            # Score by length (longer = more likely main facade)
            score = -length
        else:
            score = dist_c

        if best_edge is None or score < best_score:
            best_score = score
            best_edge = {
                'a': a, 'b': b,
                'midpoint': [mx, my],
                'normal': [nx, ny],
                'length': length,
            }

    if best_edge is None:
        return None

    return {
        'facade_a_m': best_edge['a'],
        'facade_b_m': best_edge['b'],
        'midpoint_m': best_edge['midpoint'],
        'normal': best_edge['normal'],
        'centroid_lat': cent_lat,
        'centroid_lon': cent_lng,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-dir', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--corrections-json', default='')
    parser.add_argument('--ground-truth-labels', default='')
    parser.add_argument('--s3-bucket', default='zephr-mapillary-computed-data')
    parser.add_argument('--s3-index-cache', default='')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-images-per-poi', type=int, default=MAX_IMAGES)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--rtk-only', action='store_true',
                        help='Only process RTK-labeled POIs (ignore noisy geometric labels)')
    parser.add_argument('--train-source', default='',
                        help='RTK source for training (e.g. "louisville"). Rest goes to val.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pois = load_embedding_tiles(args.tiles_dir)
    buildings = load_buildings(args.buildings_json)

    # Load corrections
    corrections = {}
    if args.corrections_json and os.path.exists(args.corrections_json):
        with open(args.corrections_json) as f:
            corrections = json.load(f)
        print(f"Loaded {len(corrections)} entrance corrections")

    # Load RTK labels
    gt_labels = {}
    if args.ground_truth_labels and os.path.exists(args.ground_truth_labels):
        with open(args.ground_truth_labels) as f:
            gt_labels = json.load(f)
        print(f"Loaded {len(gt_labels)} RTK ground truth labels")

    # Apply corrections then RTK labels
    rtk_poi_ids = set()
    for poi in pois:
        if poi['id'] in corrections:
            c = corrections[poi['id']]
            poi['entrance_lat'] = c.get('entrance_lat', poi['entrance_lat'])
            poi['entrance_lon'] = c.get('entrance_lon', poi['entrance_lon'])
        if poi['id'] in gt_labels:
            gt = gt_labels[poi['id']]
            poi['entrance_lat'] = gt['rtk_entrance_lat']
            poi['entrance_lon'] = gt['rtk_entrance_lon']
            poi['has_rtk_label'] = True
            poi['rtk_source'] = gt.get('source', 'unknown')
            rtk_poi_ids.add(poi['id'])
    print(f"Applied RTK labels to {len(rtk_poi_ids)} POIs")

    # Filter to RTK-only if requested
    if args.rtk_only:
        pois = [p for p in pois if p.get('has_rtk_label', False)]
        print(f"RTK-only mode: {len(pois)} POIs")

    # S3 setup
    s3 = boto3.client('s3')
    seq_lookup = {}
    if args.s3_index_cache and os.path.exists(args.s3_index_cache):
        with open(args.s3_index_cache) as f:
            seq_lookup = json.load(f)
        print(f"Loaded sequence lookup: {len(seq_lookup)} entries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped = {'no_building': 0, 'no_facade': 0, 'no_s3': 0}

    for i, poi in enumerate(pois):
        if i % 100 == 0:
            print(f"\nProcessing POI {i}/{len(pois)}: {poi['name']}")

        # Get building footprint
        bld_id = poi['overture_building_id']
        if bld_id not in buildings:
            skipped['no_building'] += 1
            continue

        building_coords_raw = buildings[bld_id]
        if not building_coords_raw or len(building_coords_raw) < 3:
            skipped['no_building'] += 1
            continue

        # zephr-maps uses [lat, lon], convert to [lon, lat]
        bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]

        # Fetch features for up to K images (do this FIRST to get camera positions)
        image_data = []
        camera_positions = []  # (lat, lon) for composite facade reference
        for img_id in poi['mapillary_ids']:
            if len(image_data) >= args.max_images_per_poi:
                break
            img_id_str = str(img_id)
            if img_id_str not in seq_lookup:
                continue
            seq_id = seq_lookup[img_id_str]
            features = fetch_image_features(s3, args.s3_bucket, seq_id, img_id_str)
            if features is not None:
                image_data.append((img_id_str, features))
                meta = features['metadata']
                camera_positions.append(
                    (meta['geometry']['lat'], meta['geometry']['lng'])
                )

        if not image_data:
            skipped['no_s3'] += 1
            continue

        # Compute composite facade using camera positions as road reference
        facade_feats, entrance_t, facade_result = \
            compute_composite_facade_and_entrance_t(
                bld_lnglat,
                poi['entrance_lat'], poi['entrance_lon'],
                poi.get('enclosing_roads', []),
                camera_positions_latlon=camera_positions,
            )
        if facade_feats is None:
            skipped['no_facade'] += 1
            continue

        facade_geom = {
            'facade_a_m': facade_result['facade_a_m'],
            'facade_b_m': facade_result['facade_b_m'],
            'midpoint_m': facade_result['midpoint_m'],
            'normal': facade_result['normal'],
            'centroid_lat': facade_result['centroid_lat'],
            'centroid_lon': facade_result['centroid_lon'],
        }

        # Save sample
        sample_id = f"v3_poi_{poi['id'][:8]}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        np.save(sample_dir / 'facade_feats.npy', facade_feats)

        for k, (img_id_str, feat) in enumerate(image_data):
            meta = feat['metadata']
            cam_lat = meta['geometry']['lat']
            cam_lon = meta['geometry']['lng']
            cam_compass = meta.get('compass_angle', 0.0)
            cam_type = meta.get('camera_type', 'perspective')
            cam_params = meta.get('camera_parameters', [0.5, 0, 0])
            img_w = meta.get('width', 4032)
            img_h = meta.get('height', 3024)

            hfov = camera_hfov_deg(cam_params, cam_type, img_w, img_h)

            # Camera pose features (uses first/last of composite as a/b)
            pose = compute_camera_pose_features(
                cam_lat, cam_lon, cam_compass, hfov,
                facade_geom['facade_a_m'], facade_geom['facade_b_m'],
                facade_geom['midpoint_m'], facade_geom['normal'],
                facade_geom['centroid_lat'], facade_geom['centroid_lon'],
            )

            # Per-column facade_t along composite polyline
            ft_cols = compute_patch_column_facade_t_composite(
                cam_lat, cam_lon, cam_compass, hfov,
                facade_result,
                n_cols=N_COLS, camera_type=cam_type,
            )

            np.save(sample_dir / f'patch_strip_{k}.npy', feat['patch_strip'])
            np.save(sample_dir / f'facade_t_cols_{k}.npy', ft_cols)
            np.save(sample_dir / f'camera_pose_{k}.npy', pose)

        # Save image metadata for ray-tracing at evaluation time
        img_meta_list = []
        for k, (img_id_str, feat) in enumerate(image_data):
            meta = feat['metadata']
            cam_type = meta.get('camera_type', 'perspective')
            cam_params = meta.get('camera_parameters', [0.5, 0, 0])
            img_w = meta.get('width', 4032)
            img_h = meta.get('height', 3024)
            img_meta_list.append({
                'image_id': img_id_str,
                'cam_lat': meta['geometry']['lat'],
                'cam_lon': meta['geometry']['lng'],
                'cam_compass': meta.get('compass_angle', 0.0),
                'cam_type': cam_type,
                'hfov': camera_hfov_deg(cam_params, cam_type, img_w, img_h),
                'width': img_w,
                'height': img_h,
            })
        with open(sample_dir / 'image_meta.json', 'w') as f:
            json.dump(img_meta_list, f)

        manifest.append({
            'sample_id': sample_id,
            'poi_id': poi['id'],
            'poi_name': poi['name'],
            'n_images': len(image_data),
            'image_ids': [d[0] for d in image_data],
            'entrance_t': float(entrance_t),
            'entrance_lat': poi['entrance_lat'],
            'entrance_lon': poi['entrance_lon'],
            'building_id': bld_id,
            'facade_length_m': float(facade_feats[4]),
            'has_rtk_label': poi.get('has_rtk_label', False),
            'rtk_source': poi.get('rtk_source', ''),
            'n_total_images': len(poi.get('mapillary_ids', [])),
        })

        if (i + 1) % 50 == 0:
            print(f"  Cached {len(manifest)} samples so far")

    print(f"\nDataset v3 preparation complete:")
    print(f"  Total samples: {len(manifest)}")
    print(f"  Skipped - no building: {skipped['no_building']}")
    print(f"  Skipped - no facade: {skipped['no_facade']}")
    print(f"  Skipped - no S3 data: {skipped['no_s3']}")

    # Image count stats
    img_counts = [m['n_images'] for m in manifest]
    print(f"  Images per POI: mean={np.mean(img_counts):.1f}, "
          f"median={np.median(img_counts):.0f}, max={max(img_counts)}")

    # Split strategy depends on mode
    if args.train_source:
        # Geographic hold-out: train on one source, val on the rest
        train_manifest = [m for m in manifest if m.get('rtk_source') == args.train_source]
        val_manifest = [m for m in manifest if m.get('rtk_source') != args.train_source]
        print(f"\n  Geographic split: train={args.train_source} ({len(train_manifest)}), "
              f"val=rest ({len(val_manifest)})")
    elif args.rtk_only:
        # RTK-only with random split
        random.shuffle(manifest)
        n_val = max(1, int(len(manifest) * args.val_fraction))
        val_manifest = manifest[:n_val]
        train_manifest = manifest[n_val:]
        print(f"\n  RTK-only random split: train={len(train_manifest)}, val={len(val_manifest)}")
    else:
        # Mixed: RTK to val, non-RTK split
        rtk_samples = [m for m in manifest if m.get('has_rtk_label', False)]
        non_rtk_samples = [m for m in manifest if not m.get('has_rtk_label', False)]
        random.shuffle(non_rtk_samples)
        n_non_rtk_val = int(len(non_rtk_samples) * args.val_fraction)
        val_manifest = rtk_samples + non_rtk_samples[:n_non_rtk_val]
        train_manifest = non_rtk_samples[n_non_rtk_val:]
        print(f"\n  RTK: {len(rtk_samples)} (all in val)")
        print(f"  Non-RTK: {len(non_rtk_samples)} ({len(non_rtk_samples)-n_non_rtk_val} train, {n_non_rtk_val} val)")

    random.shuffle(train_manifest)
    random.shuffle(val_manifest)

    with open(output_dir / 'train_manifest.json', 'w') as f:
        json.dump(train_manifest, f, indent=2)
    with open(output_dir / 'val_manifest.json', 'w') as f:
        json.dump(val_manifest, f, indent=2)

    print(f"  Train: {len(train_manifest)}, Val: {len(val_manifest)}")
    print(f"  Saved to {output_dir}")


if __name__ == '__main__':
    main()
