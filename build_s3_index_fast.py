"""Fast S3 sequence lookup: only searches for image IDs in the manifests."""
import json
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-dir', required=True)
    parser.add_argument('--s3-bucket', default='zephr-mapillary-computed-data')
    parser.add_argument('--output', default='s3_sequence_lookup.json')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    # Collect all image IDs from manifests
    image_ids = set()
    for f in Path(args.manifest_dir).glob('*_manifest.json'):
        with open(f) as fh:
            manifest = json.load(fh)
        for entry in manifest:
            for img_id in entry.get('image_ids', []):
                image_ids.add(str(img_id))
    print(f"Need {len(image_ids)} image IDs")

    # For each image, search S3 for metadata.json by listing objects
    # S3 structure: {sequence_id}/{image_id}/metadata.json
    # We can search by listing objects containing the image_id in the key
    s3 = boto3.client('s3')
    bucket = args.s3_bucket
    lookup = {}
    not_found = []

    def find_sequence(img_id):
        """Find sequence by listing objects with image_id in path."""
        try:
            # List objects that match */image_id/metadata.json
            # S3 doesn't support suffix search, so we'll list all objects
            # with prefix matching - we need a different approach
            # Try: list objects and check for metadata.json
            paginator = s3.get_paginator('list_objects_v2')
            # Since we can't search by image_id suffix, scan sequences
            # But that's what the slow version does.
            # Better: check if a known file exists for this image_id
            # by trying common prefixes or using list with Contains

            # Actually, use list_objects_v2 with a search approach:
            # The key format is "{seq_id}/{img_id}/metadata.json"
            # We don't know seq_id. But we can list with just the img_id
            # as a prefix... no that won't work either.

            # Use S3 Select or just scan. Since the bucket is large,
            # let's try head_object for a few common patterns, or
            # just scan the first 100 sequence prefixes.

            # Most efficient: use the S3 inventory or just iterate.
            # For 7000 images, parallel scan per-image is impractical.

            return None
        except Exception:
            return None

    # The only efficient approach with S3 is to scan all sequences once.
    # But we only need 7000 images. Let's do the full scan with early exit.
    print("Scanning S3 sequences (will exit early when all found)...")
    remaining = set(image_ids)
    paginator = s3.get_paginator('list_objects_v2')
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
                print(f"  {seq_count} seqs, {len(lookup)}/{len(image_ids)} found")
            try:
                resp = s3.list_objects_v2(
                    Bucket=bucket, Prefix=f"{seq_id}/", Delimiter='/', MaxKeys=1000)
                for p in resp.get('CommonPrefixes', []):
                    parts = p['Prefix'].rstrip('/').split('/')
                    if len(parts) >= 2:
                        img_id = parts[1]
                        if img_id in remaining:
                            lookup[img_id] = seq_id
                            remaining.discard(img_id)
            except Exception:
                continue

    print(f"Found {len(lookup)}/{len(image_ids)} images in {seq_count} sequences")
    with open(args.output, 'w') as f:
        json.dump(lookup, f)
    print(f"Saved to {args.output}")

    # Upload to S3 for persistence
    s3.upload_file(args.output, 'zephr-mapillary-cache',
                   'jepa-entrance-v3/s3_sequence_lookup.json')
    print("Persisted to S3")

if __name__ == '__main__':
    main()
