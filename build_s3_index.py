"""
Build S3 sequence lookup: image_id → sequence_id
for all Mapillary images referenced in the embedding tiles.

Strategy: For each image_id, list S3 objects matching */{image_id}/metadata.json
using a targeted search rather than full bucket scan.
"""
import json
import os
import sys
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_image_ids_from_tiles(tiles_dir):
    """Get all unique Mapillary image IDs from embedding tiles."""
    image_ids = set()
    for json_file in sorted(Path(tiles_dir).glob('*.json')):
        with open(json_file) as f:
            try:
                tile = json.load(f)
            except json.JSONDecodeError:
                continue
        for wp in tile.get('waypoints', []):
            for mid in wp.get('mapillary_ids', []):
                image_ids.add(str(mid))
    return image_ids


def find_sequence_for_image(s3_client, bucket, image_id):
    """Find the sequence containing this image by listing prefixes."""
    # The S3 structure is {sequence_id}/{image_id}/
    # We can search by listing objects with a suffix pattern
    # But S3 doesn't support suffix search, so we need another approach

    # Strategy: Use S3 Select or list with prefix filtering
    # Since image IDs are stored as subdirectories, we can try to
    # find the metadata file by listing with various prefixes

    # Actually, the most efficient approach: list ALL sequences that
    # contain this image_id. We know the image_id is a Mapillary ID
    # (numeric). The sequence_id is a Mapillary sequence hash.

    # We need to scan. But we can be smart: cache the full listing once.
    return None  # Will use batch approach


def build_index_from_full_listing(s3_client, bucket, target_image_ids):
    """Build index by listing all sequences and their images.

    This is a one-time operation that creates a complete image→sequence map.
    """
    lookup = {}
    paginator = s3_client.get_paginator('list_objects_v2')

    print("Listing all sequences in S3 bucket...")
    seq_count = 0
    img_count = 0

    # List top-level prefixes (sequences)
    for page in paginator.paginate(Bucket=bucket, Delimiter='/'):
        for prefix_info in page.get('CommonPrefixes', []):
            seq_id = prefix_info['Prefix'].rstrip('/')
            seq_count += 1

            if seq_count % 1000 == 0:
                print(f"  Scanned {seq_count} sequences, found {img_count} matching images...")

            # List images in this sequence
            try:
                img_resp = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=f"{seq_id}/",
                    Delimiter='/',
                    MaxKeys=1000
                )
                for img_prefix in img_resp.get('CommonPrefixes', []):
                    parts = img_prefix['Prefix'].rstrip('/').split('/')
                    if len(parts) >= 2:
                        img_id = parts[1]
                        if img_id in target_image_ids:
                            lookup[img_id] = seq_id
                            img_count += 1
            except Exception as e:
                continue

    print(f"  Done! Scanned {seq_count} sequences, found {img_count} matching images")
    return lookup


def build_index_targeted(s3_client, bucket, target_image_ids):
    """Build index by checking each target image ID directly.

    For each image_id, we check metadata.json existence in every sequence.
    This is slow for many sequences but fast for few images.

    Better approach: since the bucket has a flat structure, we can
    try to head_object for common patterns.
    """
    # Since we can't search by image_id efficiently, let's use the
    # full listing approach but stop early once we've found all targets

    lookup = {}
    remaining = set(target_image_ids)
    paginator = s3_client.get_paginator('list_objects_v2')

    print(f"Searching for {len(remaining)} image IDs across S3...")
    seq_count = 0

    for page in paginator.paginate(Bucket=bucket, Delimiter='/'):
        if not remaining:
            break

        for prefix_info in page.get('CommonPrefixes', []):
            if not remaining:
                break

            seq_id = prefix_info['Prefix'].rstrip('/')
            seq_count += 1

            if seq_count % 500 == 0:
                print(f"  {seq_count} sequences scanned, "
                      f"{len(lookup)}/{len(target_image_ids)} found, "
                      f"{len(remaining)} remaining...")

            # List images in this sequence
            try:
                img_resp = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=f"{seq_id}/",
                    Delimiter='/',
                    MaxKeys=1000
                )
                for img_prefix in img_resp.get('CommonPrefixes', []):
                    parts = img_prefix['Prefix'].rstrip('/').split('/')
                    if len(parts) >= 2:
                        img_id = parts[1]
                        if img_id in remaining:
                            lookup[img_id] = seq_id
                            remaining.discard(img_id)
            except Exception:
                continue

    print(f"  Found {len(lookup)}/{len(target_image_ids)} images "
          f"across {seq_count} sequences")
    if remaining:
        print(f"  Missing: {len(remaining)} images not found in S3")

    return lookup


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-dir', required=True)
    parser.add_argument('--s3-bucket', default='zephr-mapillary-computed-data')
    parser.add_argument('--output', default='s3_sequence_lookup.json')
    args = parser.parse_args()

    # Get target image IDs
    image_ids = extract_image_ids_from_tiles(args.tiles_dir)
    print(f"Found {len(image_ids)} unique Mapillary image IDs in tiles")

    # Build lookup
    s3 = boto3.client('s3')
    lookup = build_index_targeted(s3, args.s3_bucket, image_ids)

    # Save
    with open(args.output, 'w') as f:
        json.dump(lookup, f)
    print(f"Saved lookup ({len(lookup)} entries) to {args.output}")


if __name__ == '__main__':
    main()
