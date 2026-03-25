"""
Prepare v4 dataset for entrance detection model.

For each POI + image, computes:
  - Patch strip (16×1024) — same as v3
  - Camera pose (6-dim) — relative to building centroid, no facade needed
  - Ground truth entrance column — which image column the entrance falls in
  - Visibility flag — whether entrance is within camera FOV
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

from geometry import (
    camera_hfov_deg,
    compute_camera_pose_v4,
    entrance_to_column,
)

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320
MAX_IMAGES = 15
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
            })
    print(f"Loaded {len(all_pois)} POIs with entrances + mapillary + buildings")
    return all_pois


def load_buildings(buildings_json):
    with open(buildings_json) as f:
        return json.load(f)


def fetch_image_features(s3_client, bucket, sequence_id, image_id):
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

    patches = None
    for key in patch_data:
        patches = patch_data[key]
        break
    if patches is None:
        return None

    patches = patches.astype(np.float32)

    if patches.shape == (256, 1024):
        patches_2d = patches.reshape(16, 16, 1024)
        patch_strip = patches_2d.mean(axis=0)
    elif patches.ndim == 2:
        n = patches.shape[0]
        side = int(math.sqrt(n))
        if side * side == n:
            patches_2d = patches.reshape(side, side, patches.shape[1])
            if side >= 16:
                col_size = side // 16
                patch_strip = np.zeros((16, patches.shape[1]), dtype=np.float32)
                for c in range(16):
                    patch_strip[c] = patches_2d[:, c*col_size:(c+1)*col_size, :].mean(axis=(0, 1))
            else:
                patch_strip = np.zeros((16, patches.shape[1]), dtype=np.float32)
                patch_strip[:side] = patches_2d.mean(axis=0)
        else:
            patch_strip = np.zeros((16, 1024), dtype=np.float32)
            n_take = min(16, patches.shape[0])
            patch_strip[:n_take] = patches[:n_take]
    else:
        return None

    return {'patch_strip': patch_strip, 'metadata': metadata}


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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pois = load_embedding_tiles(args.tiles_dir)
    buildings = load_buildings(args.buildings_json)

    corrections = {}
    if args.corrections_json and os.path.exists(args.corrections_json):
        with open(args.corrections_json) as f:
            corrections = json.load(f)
        print(f"Loaded {len(corrections)} entrance corrections")

    gt_labels = {}
    if args.ground_truth_labels and os.path.exists(args.ground_truth_labels):
        with open(args.ground_truth_labels) as f:
            gt_labels = json.load(f)
        print(f"Loaded {len(gt_labels)} RTK ground truth labels")

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

    s3 = boto3.client('s3')
    seq_lookup = {}
    if args.s3_index_cache and os.path.exists(args.s3_index_cache):
        with open(args.s3_index_cache) as f:
            seq_lookup = json.load(f)
        print(f"Loaded sequence lookup: {len(seq_lookup)} entries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped = {'no_building': 0, 'no_s3': 0, 'no_visible': 0}
    vis_stats = {'total_images': 0, 'visible_images': 0}

    for i, poi in enumerate(pois):
        if i % 100 == 0:
            print(f"\nProcessing POI {i}/{len(pois)}: {poi['name']}")

        bld_id = poi['overture_building_id']
        if bld_id not in buildings:
            skipped['no_building'] += 1
            continue

        building_coords_raw = buildings[bld_id]
        if not building_coords_raw or len(building_coords_raw) < 3:
            skipped['no_building'] += 1
            continue

        bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]

        # Building centroid
        cent_lng = sum(c[0] for c in bld_lnglat) / len(bld_lnglat)
        cent_lat = sum(c[1] for c in bld_lnglat) / len(bld_lnglat)

        # Fetch image features
        image_data = []
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

        if not image_data:
            skipped['no_s3'] += 1
            continue

        # Check if at least one image has the entrance visible
        any_visible = False
        image_metas = []

        for img_id_str, feat in image_data:
            meta = feat['metadata']
            cam_lat = meta['geometry']['lat']
            cam_lon = meta['geometry']['lng']
            cam_compass = meta.get('compass_angle', 0.0)
            cam_type = meta.get('camera_type', 'perspective')
            cam_params = meta.get('camera_parameters', [0.5, 0, 0])
            img_w = meta.get('width', 4032)
            img_h = meta.get('height', 3024)
            hfov = camera_hfov_deg(cam_params, cam_type, img_w, img_h)

            col, visible, dist_m = entrance_to_column(
                cam_lat, cam_lon, cam_compass, hfov,
                poi['entrance_lat'], poi['entrance_lon'],
                n_cols=N_COLS, camera_type=cam_type,
            )

            vis_stats['total_images'] += 1
            if visible:
                vis_stats['visible_images'] += 1
                any_visible = True

            image_metas.append({
                'cam_lat': cam_lat, 'cam_lon': cam_lon,
                'cam_compass': cam_compass, 'cam_type': cam_type,
                'hfov': hfov,
                'entrance_col': float(col),
                'visible': bool(visible),
                'dist_m': float(dist_m),
            })

        if not any_visible:
            skipped['no_visible'] += 1
            continue

        # Save sample
        sample_id = f"v4_poi_{poi['id'][:8]}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        for k, (img_id_str, feat) in enumerate(image_data):
            m = image_metas[k]

            # Camera pose (6-dim, relative to building centroid)
            pose = compute_camera_pose_v4(
                m['cam_lat'], m['cam_lon'], m['cam_compass'], m['hfov'],
                cent_lat, cent_lng,
            )

            np.save(sample_dir / f'patch_strip_{k}.npy', feat['patch_strip'])
            np.save(sample_dir / f'camera_pose_{k}.npy', pose)

        # Save per-image metadata (columns, visibility)
        with open(sample_dir / 'image_meta.json', 'w') as f:
            json.dump(image_metas, f)

        manifest.append({
            'sample_id': sample_id,
            'poi_id': poi['id'],
            'poi_name': poi['name'],
            'n_images': len(image_data),
            'image_ids': [d[0] for d in image_data],
            'entrance_lat': poi['entrance_lat'],
            'entrance_lon': poi['entrance_lon'],
            'building_id': bld_id,
            'building_centroid_lat': cent_lat,
            'building_centroid_lon': cent_lng,
            'has_rtk_label': poi.get('has_rtk_label', False),
            'rtk_source': poi.get('rtk_source', ''),
            'n_visible_images': sum(1 for m in image_metas if m['visible']),
            'n_total_images': len(poi.get('mapillary_ids', [])),
        })

        if (i + 1) % 50 == 0:
            print(f"  Cached {len(manifest)} samples so far")

    print(f"\nDataset v4 preparation complete:")
    print(f"  Total samples: {len(manifest)}")
    print(f"  Skipped - no building: {skipped['no_building']}")
    print(f"  Skipped - no S3 data: {skipped['no_s3']}")
    print(f"  Skipped - no visible images: {skipped['no_visible']}")
    print(f"  Images: {vis_stats['visible_images']}/{vis_stats['total_images']} "
          f"visible ({100*vis_stats['visible_images']/max(vis_stats['total_images'],1):.0f}%)")

    img_counts = [m['n_images'] for m in manifest]
    print(f"  Images per POI: mean={np.mean(img_counts):.1f}, "
          f"median={np.median(img_counts):.0f}, max={max(img_counts)}")

    # Split: all RTK to val, non-RTK split by val_fraction
    rtk_samples = [m for m in manifest if m.get('has_rtk_label', False)]
    non_rtk_samples = [m for m in manifest if not m.get('has_rtk_label', False)]
    random.shuffle(non_rtk_samples)
    n_non_rtk_val = int(len(non_rtk_samples) * args.val_fraction)
    val_manifest = rtk_samples + non_rtk_samples[:n_non_rtk_val]
    train_manifest = non_rtk_samples[n_non_rtk_val:]
    print(f"\n  RTK: {len(rtk_samples)} (all in val)")
    print(f"  Non-RTK: {len(non_rtk_samples)} "
          f"({len(non_rtk_samples)-n_non_rtk_val} train, {n_non_rtk_val} val)")

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
