#!/usr/bin/env python3
"""Download Mapillary thumbnails for the 3D viewer into docs/images/.

WHY THIS EXISTS
---------------
The viewer (docs/index.html) must reference thumbnails by LOCAL path
(images/<image_id>.jpg), NOT by Mapillary/Facebook CDN URL.

Mapillary thumb URLs (scontent*.fbcdn.net/...) are signed and carry an
expiry in their `oe=` query param. They stop loading after a few weeks,
which previously broke the sidebar images on the published site.

This script reads the Mapillary image_ids embedded in docs/index.html,
fetches a fresh thumb_1024_url for each via the Graph API, and saves the
JPEGs locally so they can be committed and served permanently.

USAGE
-----
    MAPILLARY_TOKEN='MLY|...' python3 docs/download_images.py

Run it whenever new POIs/images are added to the viewer; it only downloads
files that are missing (or zero-length).
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "index.html")
OUT_DIR = os.path.join(HERE, "images")


def extract_image_ids(html_path):
    html = open(html_path).read()
    m = re.search(r"const POIS = (\[.*?\]);", html, re.S)
    if not m:
        sys.exit("Could not find `const POIS = [...]` in index.html")
    pois = json.loads(m.group(1))
    ids = []
    for poi in pois:
        for img in poi.get("images", []):
            iid = img.get("image_id")
            if iid:
                ids.append(str(iid))
    return sorted(set(ids))


def main():
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        sys.exit("Set MAPILLARY_TOKEN (a Mapillary client token: 'MLY|...').")

    os.makedirs(OUT_DIR, exist_ok=True)
    ids = extract_image_ids(INDEX_HTML)
    print(f"{len(ids)} unique image_ids referenced by the viewer")

    ok, skipped, failed = 0, 0, []
    for i, iid in enumerate(ids):
        dest = os.path.join(OUT_DIR, f"{iid}.jpg")
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            skipped += 1
            continue
        try:
            meta_url = (
                f"https://graph.mapillary.com/{iid}"
                f"?access_token={urllib.parse.quote(token)}&fields=thumb_1024_url"
            )
            with urllib.request.urlopen(meta_url, timeout=30) as r:
                thumb_url = json.load(r).get("thumb_1024_url")
            if not thumb_url:
                failed.append((iid, "no thumb_1024_url"))
                continue
            with urllib.request.urlopen(thumb_url, timeout=60) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append((iid, str(e)))
        time.sleep(0.1)

    print(f"downloaded={ok} already-present={skipped} failed={len(failed)}")
    for iid, err in failed:
        print(f"  FAIL {iid}: {err}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
