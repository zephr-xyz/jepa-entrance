"""
Prepare training dataset by pulling features from three sources:
  1. embedding-tiles (POI metadata, entrance coords, building IDs)
  2. s3://zephr-mapillary-computed-data/ (DINOv2, caption, keypoint embeddings)
  3. zephr-maps data/ (building footprints)

Optionally uses RTK ground truth labels (ground_truth_labels.json) to replace
geometric entrance coordinates with sub-centimeter surveyed locations.
RTK-labeled POIs go to validation set for measuring real-world accuracy.

Usage:
    python prepare_dataset.py \
        --tiles-dir /path/to/embedding-tiles/embedding-tiles-overture-visual/z14 \
        --buildings-json /path/to/zephr-maps/data/buildings.json \
        --corrections-json /path/to/zephr-maps/data/entrance_corrections.json \
        --s3-bucket zephr-mapillary-computed-data \
        --output-dir /path/to/cache \
        --ground-truth-labels ground_truth_labels.json \
        --val-fraction 0.15
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

from dataset import (
    compute_facade_and_entrance_t,
    load_keypoint_stats,
)


def load_embedding_tiles(tiles_dir: str):
    """Load all POIs from embedding-tiles JSON files."""
    tiles_path = Path(tiles_dir)
    all_pois = []

    for json_file in sorted(tiles_path.glob('*.json')):
        with open(json_file) as f:
            tile = json.load(f)

        for wp in tile.get('waypoints', []):
            # Must have entrance coords and mapillary images
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


def load_buildings(buildings_json: str):
    """Load building footprints from zephr-maps data."""
    with open(buildings_json) as f:
        buildings = json.load(f)
    print(f"Loaded {len(buildings)} building footprints")
    return buildings


def load_corrections(corrections_json: str):
    """Load entrance corrections (higher-quality labels)."""
    if not os.path.exists(corrections_json):
        return {}
    with open(corrections_json) as f:
        corrections = json.load(f)
    print(f"Loaded {len(corrections)} entrance corrections")
    return corrections


def fetch_s3_features(s3_client, bucket, sequence_id, image_id):
    """Fetch precomputed features for a single Mapillary image from S3.

    Returns dict with cls_emb, caption_emb, keypoints, metadata or None.
    """
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

    cls_data = get_npz('cls_embedding.npz')
    caption_data = get_npz('caption_embedding.npz')
    kp_data = get_npz('keypoints.npz')
    scores_data = get_npz('scores.npz')
    metadata = get_json('metadata.json')

    if cls_data is None or metadata is None:
        return None

    # Extract the embedding arrays
    cls_emb = None
    for key in cls_data:
        cls_emb = cls_data[key].flatten()
        break

    caption_emb = None
    if caption_data:
        for key in caption_data:
            caption_emb = caption_data[key].flatten()
            break

    kp_dict = {}
    if kp_data:
        for key in kp_data:
            kp_dict['keypoints'] = kp_data[key]
            break
    if scores_data:
        for key in scores_data:
            kp_dict['scores'] = scores_data[key]
            break

    return {
        'cls_emb': cls_emb,
        'caption_emb': caption_emb if caption_emb is not None else np.zeros(768, dtype=np.float32),
        'keypoints': kp_dict,
        'metadata': metadata,
    }


def find_image_on_s3(s3_client, bucket, image_id):
    """Find the sequence folder for a given image ID by listing prefixes."""
    # The S3 structure is {sequence_id}/{image_id}/
    # We need to find which sequence contains this image
    # Use S3 search with the image_id as a suffix pattern
    paginator = s3_client.get_paginator('list_objects_v2')

    # Try listing with image_id as prefix at depth 2
    # This is expensive; better to cache a lookup table
    # For now, we'll try to find metadata.json for this image
    try:
        # Check if there's an index file
        resp = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"",
            Delimiter='/',
            MaxKeys=1
        )
    except Exception:
        pass

    return None  # Will use the lookup table approach


def build_sequence_lookup(s3_client, bucket, image_ids):
    """Build image_id → sequence_id lookup by scanning S3.

    Since images are stored as {sequence_id}/{image_id}/, we need to
    find which sequence each image belongs to. We do this by checking
    the metadata files.
    """
    lookup = {}
    total = len(image_ids)

    # Check each image by trying common sequence prefixes
    # This is the bottleneck - for production, pre-build this index
    print(f"Building sequence lookup for {total} images...")

    for i, img_id in enumerate(image_ids):
        if i % 50 == 0:
            print(f"  Scanning {i}/{total}...")

        # Search for this image_id in S3
        paginator = s3_client.get_paginator('list_objects_v2')
        found = False

        # Try to find by searching for the metadata file
        try:
            for page in paginator.paginate(
                Bucket=bucket,
                Prefix="",
                Delimiter='/',
            ):
                for prefix_info in page.get('CommonPrefixes', []):
                    seq_id = prefix_info['Prefix'].rstrip('/')
                    # Check if this sequence has our image
                    try:
                        s3_client.head_object(
                            Bucket=bucket,
                            Key=f"{seq_id}/{img_id}/metadata.json"
                        )
                        lookup[img_id] = seq_id
                        found = True
                        break
                    except Exception:
                        continue
                if found:
                    break
        except Exception:
            continue

    print(f"  Found sequences for {len(lookup)}/{total} images")
    return lookup


def build_sequence_lookup_fast(s3_client, bucket, mapillary_master_json=None):
    """Build lookup from mapillary_master.json which has sequence info,
    or by scanning the S3 bucket index."""
    lookup = {}

    # Strategy: list all {seq}/{img}/metadata.json objects and parse
    print("Building sequence lookup from S3 listing (this may take a while)...")
    paginator = s3_client.get_paginator('list_objects_v2')

    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix="", Delimiter='/'):
        for prefix_info in page.get('CommonPrefixes', []):
            seq_id = prefix_info['Prefix'].rstrip('/')
            # List images in this sequence
            try:
                img_page = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=f"{seq_id}/",
                    Delimiter='/',
                    MaxKeys=1000
                )
                for img_prefix in img_page.get('CommonPrefixes', []):
                    img_id = img_prefix['Prefix'].split('/')[1]
                    lookup[img_id] = seq_id
                    count += 1
            except Exception:
                continue

        if count % 10000 == 0 and count > 0:
            print(f"  Indexed {count} images...")

    print(f"  Total indexed: {count} images across S3")
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-dir', required=True,
                        help='Path to embedding-tiles-overture-visual/z14/')
    parser.add_argument('--buildings-json', required=True,
                        help='Path to zephr-maps/data/buildings.json')
    parser.add_argument('--corrections-json', default='',
                        help='Path to entrance_corrections.json')
    parser.add_argument('--s3-bucket', default='zephr-mapillary-computed-data')
    parser.add_argument('--s3-index-cache', default='',
                        help='Path to cached sequence lookup JSON')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ground-truth-labels', default='',
                        help='Path to ground_truth_labels.json (RTK labels)')
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data sources
    pois = load_embedding_tiles(args.tiles_dir)
    buildings = load_buildings(args.buildings_json)
    corrections = load_corrections(args.corrections_json) if args.corrections_json else {}

    # Load RTK ground truth labels
    gt_labels = {}
    if args.ground_truth_labels and os.path.exists(args.ground_truth_labels):
        with open(args.ground_truth_labels) as f:
            gt_labels = json.load(f)
        print(f"Loaded {len(gt_labels)} RTK ground truth labels")

    # Apply entrance corrections where available
    corrections_by_id = {}
    for poi_id, corr in corrections.items():
        corrections_by_id[poi_id] = corr

    for poi in pois:
        if poi['id'] in corrections_by_id:
            c = corrections_by_id[poi['id']]
            poi['entrance_lat'] = c['entrance_lat']
            poi['entrance_lon'] = c['entrance_lon']
            if c.get('overture_building_id'):
                poi['overture_building_id'] = c['overture_building_id']

    # Apply RTK labels (override entrance coords with surveyed locations)
    rtk_poi_ids = set()
    for poi in pois:
        if poi['id'] in gt_labels:
            gt = gt_labels[poi['id']]
            poi['entrance_lat'] = gt['rtk_entrance_lat']
            poi['entrance_lon'] = gt['rtk_entrance_lon']
            poi['has_rtk_label'] = True
            rtk_poi_ids.add(poi['id'])
    print(f"Applied RTK labels to {len(rtk_poi_ids)} POIs")

    # Setup S3
    s3 = boto3.client('s3')

    # Build or load sequence lookup
    if args.s3_index_cache and os.path.exists(args.s3_index_cache):
        with open(args.s3_index_cache) as f:
            seq_lookup = json.load(f)
        print(f"Loaded sequence lookup: {len(seq_lookup)} entries")
    else:
        seq_lookup = build_sequence_lookup_fast(s3, args.s3_bucket)
        if args.s3_index_cache:
            with open(args.s3_index_cache, 'w') as f:
                json.dump(seq_lookup, f)
            print(f"Saved sequence lookup to {args.s3_index_cache}")

    # Process each POI
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped = {'no_building': 0, 'no_facade': 0, 'no_s3': 0, 'no_features': 0}

    for i, poi in enumerate(pois):
        if i % 100 == 0:
            print(f"\nProcessing POI {i}/{len(pois)}: {poi['name']}")

        # Get building footprint
        bld_id = poi['overture_building_id']
        if bld_id not in buildings:
            skipped['no_building'] += 1
            continue

        building_coords = buildings[bld_id]
        if not building_coords or len(building_coords) < 3:
            skipped['no_building'] += 1
            continue

        # Convert building coords format: may be [[lat,lon],...] or [[lon,lat],...]
        # zephr-maps uses [lat, lon] in buildings.json
        # Our functions expect [lon, lat] (GeoJSON order)
        bld_lnglat = [[c[1], c[0]] for c in building_coords]

        # Compute facade features and entrance_t
        facade_feats, entrance_t = compute_facade_and_entrance_t(
            bld_lnglat,
            poi['entrance_lat'], poi['entrance_lon'],
            poi.get('enclosing_roads', [])
        )

        if facade_feats is None:
            skipped['no_facade'] += 1
            continue

        # Find best Mapillary image with S3 features
        best_features = None
        for img_id in poi['mapillary_ids']:
            img_id_str = str(img_id)
            if img_id_str not in seq_lookup:
                continue

            seq_id = seq_lookup[img_id_str]
            features = fetch_s3_features(s3, args.s3_bucket, seq_id, img_id_str)
            if features is not None and features['cls_emb'] is not None:
                best_features = features
                break

        if best_features is None:
            skipped['no_s3'] += 1
            continue

        # Compute keypoint stats
        kp_stats = load_keypoint_stats(best_features['keypoints'])

        # Normalize compass angle to [-1, 1]
        compass = best_features['metadata'].get('compass_angle', 0.0)
        compass_norm = (compass % 360) / 180.0 - 1.0

        # Save sample
        sample_id = f"poi_{poi['id'][:8]}_{img_id_str}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        np.save(sample_dir / 'cls_emb.npy', best_features['cls_emb'].astype(np.float32))
        np.save(sample_dir / 'caption_emb.npy', best_features['caption_emb'].astype(np.float32))
        np.save(sample_dir / 'kp_stats.npy', kp_stats)
        np.save(sample_dir / 'facade_feats.npy', facade_feats)

        manifest.append({
            'sample_id': sample_id,
            'poi_id': poi['id'],
            'poi_name': poi['name'],
            'image_id': img_id_str,
            'compass_normalized': float(compass_norm),
            'entrance_t': float(entrance_t),
            'entrance_lat': poi['entrance_lat'],
            'entrance_lon': poi['entrance_lon'],
            'building_id': bld_id,
            'facade_length_m': float(facade_feats[4]),
            'has_rtk_label': poi.get('has_rtk_label', False),
        })

        if (i + 1) % 50 == 0:
            print(f"  Cached {len(manifest)} samples so far")

    print(f"\nDataset preparation complete:")
    print(f"  Total samples: {len(manifest)}")
    print(f"  Skipped - no building: {skipped['no_building']}")
    print(f"  Skipped - no facade: {skipped['no_facade']}")
    print(f"  Skipped - no S3 data: {skipped['no_s3']}")
    print(f"  Skipped - no features: {skipped['no_features']}")

    # Split into train/val
    # Strategy: RTK-labeled POIs go to val set (real ground truth for evaluation)
    # Remaining POIs split by val_fraction for additional val samples
    rtk_samples = [m for m in manifest if m.get('has_rtk_label', False)]
    non_rtk_samples = [m for m in manifest if not m.get('has_rtk_label', False)]

    random.shuffle(non_rtk_samples)

    if rtk_samples:
        # RTK samples: 70% train (with real labels), 30% val (held out for eval)
        random.shuffle(rtk_samples)
        n_rtk_val = max(1, int(len(rtk_samples) * 0.3))
        rtk_val = rtk_samples[:n_rtk_val]
        rtk_train = rtk_samples[n_rtk_val:]

        # Non-RTK: small fraction to val, rest to train
        n_non_rtk_val = int(len(non_rtk_samples) * args.val_fraction)
        non_rtk_val = non_rtk_samples[:n_non_rtk_val]
        non_rtk_train = non_rtk_samples[n_non_rtk_val:]

        val_manifest = rtk_val + non_rtk_val
        train_manifest = rtk_train + non_rtk_train

        print(f"\n  RTK samples: {len(rtk_samples)} ({len(rtk_train)} train, {len(rtk_val)} val)")
        print(f"  Non-RTK samples: {len(non_rtk_samples)} ({len(non_rtk_train)} train, {len(non_rtk_val)} val)")
    else:
        n_val = int(len(manifest) * args.val_fraction)
        val_manifest = non_rtk_samples[:n_val]
        train_manifest = non_rtk_samples[n_val:]

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
