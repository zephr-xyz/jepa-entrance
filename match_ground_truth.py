"""
Match RTK ground truth POIs to embedding-tiles waypoints and produce
a ground_truth_labels.json that maps overture_id -> RTK front door location.

Handles two ground truth datasets:
  1. Louisville (has overture_id linkage via GeoJSON)
  2. Boulder (name + proximity matching only)

Output: ground_truth_labels.json with per-POI RTK entrance locations
"""
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def normalize_name(name):
    name = re.sub(r'\s*\(.*?\)\s*', '', name)
    return name.strip().lower()


def load_rtk_csv(csv_path):
    pois = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pois[row['POI_name'].strip()].append(row)
    return pois


def extract_front_doors(rtk_pois):
    front_doors = {}
    for poi_name, rows in rtk_pois.items():
        fds = [r for r in rows if r['Type'].strip().lower() in ('front_door', 'front door')]
        if fds:
            lats = [float(r['Latitude']) for r in fds]
            lons = [float(r['Longitude']) for r in fds]
            elevs = [float(r['Elevation']) for r in fds]
            front_doors[poi_name] = {
                'lat': sum(lats) / len(lats),
                'lon': sum(lons) / len(lons),
                'elev': sum(elevs) / len(elevs),
                'n_measurements': len(fds),
            }
    return front_doors


def extract_signs(rtk_pois):
    signs = {}
    for poi_name, rows in rtk_pois.items():
        srows = [r for r in rows if r['Type'].strip().lower() == 'sign']
        if srows:
            lats = [float(r['Latitude']) for r in srows]
            lons = [float(r['Longitude']) for r in srows]
            signs[poi_name] = {
                'lat': sum(lats) / len(lats),
                'lon': sum(lons) / len(lons),
            }
    return signs


def load_tiles(tile_paths):
    all_wps = []
    for p in tile_paths:
        with open(p) as f:
            tile = json.load(f)
        all_wps.extend(tile['waypoints'])
    return all_wps


def make_label(wp, rtk, sign, gt_name):
    tile_ent_lat = wp.get('entrance_lat')
    tile_ent_lon = wp.get('entrance_lon')
    dist = None
    if tile_ent_lat and tile_ent_lon:
        dist = haversine_m(tile_ent_lat, tile_ent_lon, rtk['lat'], rtk['lon'])
    return {
        'overture_id': wp['id'],
        'gt_name': gt_name,
        'wp_name': wp['name'],
        'rtk_entrance_lat': rtk['lat'],
        'rtk_entrance_lon': rtk['lon'],
        'rtk_entrance_elev': rtk.get('elev'),
        'rtk_n_measurements': rtk.get('n_measurements', 1),
        'tile_entrance_lat': tile_ent_lat,
        'tile_entrance_lon': tile_ent_lon,
        'dist_tile_to_rtk_m': round(dist, 2) if dist else None,
        'rtk_sign_lat': sign['lat'] if sign else None,
        'rtk_sign_lon': sign['lon'] if sign else None,
        'wp_lat': wp['latitude'],
        'wp_lon': wp['longitude'],
        'n_mapillary_images': len(wp.get('mapillary_ids', [])),
        'overture_building_id': wp.get('overture_building_id'),
    }


def match_louisville(tile_paths):
    """Match Louisville GT using overture_id linkage."""
    base = Path('/tmp/overture_project/ground_truth')
    csv_path = base / 'louisville_snip_ground_truths.csv'
    geojson_path = base / 'louisville_ground_truths.geojson'

    rtk_pois = load_rtk_csv(csv_path)
    front_doors = extract_front_doors(rtk_pois)
    signs = extract_signs(rtk_pois)
    waypoints = load_tiles(tile_paths)
    wp_by_id = {wp['id']: wp for wp in waypoints}

    with open(geojson_path) as f:
        gt = json.load(f)

    labels = {}
    for feat in gt['features']:
        oid = feat['properties'].get('overture_id')
        gt_name = feat['properties']['REMARKS'].strip()
        if not oid or oid not in wp_by_id:
            continue

        wp = wp_by_id[oid]
        rtk = front_doors.get(gt_name)
        if rtk is None:
            nn = normalize_name(gt_name)
            for rn, rv in front_doors.items():
                if normalize_name(rn) == nn:
                    rtk = rv
                    break
        if rtk is None:
            continue

        # For POIs with front+rear entries, prefer the front entrance
        if wp['id'] in labels and 'front' not in gt_name.lower():
            continue

        sign = signs.get(gt_name)
        labels[wp['id']] = make_label(wp, rtk, sign, gt_name)
        labels[wp['id']]['source'] = 'louisville'

    return labels


def match_boulder(tile_paths):
    """Match Boulder GT using name + proximity matching."""
    csv_path = '/Users/seangorman/downloads/boulder_snips_ground_truth_adjusted.csv'
    rtk_pois = load_rtk_csv(csv_path)
    front_doors = extract_front_doors(rtk_pois)
    signs = extract_signs(rtk_pois)
    waypoints = load_tiles(tile_paths)

    # Manual name mappings for tricky cases
    manual = {
        'Ellison': 'Elison Rd.',
        'Smith Klein': 'SmithKlein Gallery',
    }
    # False positive blocklist (GT name -> tile name matches to reject)
    blocklist = {
        'C Bar': True,   # would match "Jax Fish House & Oyster Bar" on token "bar"
        'J Bar': True,   # would match "J Albrecht Designs" on token "j"
    }

    wp_by_exact = {}
    for wp in waypoints:
        wp_by_exact[wp['name'].lower().strip()] = wp

    labels = {}
    for poi_name, rtk in front_doors.items():
        if poi_name in blocklist:
            continue

        rows = rtk_pois[poi_name]
        avg_lat = sum(float(r['Latitude']) for r in rows) / len(rows)
        avg_lon = sum(float(r['Longitude']) for r in rows) / len(rows)

        wp = None

        # Manual mapping
        if poi_name in manual:
            for w in waypoints:
                if w['name'] == manual[poi_name]:
                    wp = w
                    break

        # Exact match (case-insensitive)
        if not wp:
            nn = poi_name.lower().strip()
            if nn in wp_by_exact:
                wp = wp_by_exact[nn]

        # Substring match with proximity
        if not wp:
            nn = poi_name.lower().strip()
            for w in waypoints:
                wn = w['name'].lower().strip()
                if nn in wn or wn in nn:
                    d = haversine_m(avg_lat, avg_lon, w['latitude'], w['longitude'])
                    if d < 40:
                        wp = w
                        break

        # Token overlap + proximity
        if not wp:
            nn_tokens = set(re.sub(r'[^a-z0-9\s]', '', poi_name.lower()).split())
            nn_tokens -= {'the', 'and', 'of', 'a'}
            best_score = 0
            for w in waypoints:
                wn_tokens = set(re.sub(r'[^a-z0-9\s]', '', w['name'].lower()).split())
                wn_tokens -= {'the', 'and', 'of', 'a'}
                overlap = nn_tokens & wn_tokens
                if len(overlap) >= max(1, len(nn_tokens) * 0.5):
                    d = haversine_m(avg_lat, avg_lon, w['latitude'], w['longitude'])
                    if d < 30:
                        score = len(overlap) / max(len(nn_tokens), 1) - d / 100
                        if score > best_score:
                            best_score = score
                            wp = w

        if not wp:
            continue

        # Distance sanity check
        d = haversine_m(avg_lat, avg_lon, wp['latitude'], wp['longitude'])
        if d > 50:
            continue

        sign = signs.get(poi_name)
        label = make_label(wp, rtk, sign, poi_name)
        label['source'] = 'boulder'

        # Don't overwrite if same waypoint already matched
        if wp['id'] not in labels:
            labels[wp['id']] = label

    return labels


def main():
    louisville_tiles = ['/tmp/x3407_y6204.json', '/tmp/x3407_y6203.json']
    boulder_tiles = ['/tmp/x3400_y6201.json']
    output_path = Path('/tmp/jepa-entrance/ground_truth_labels.json')

    louisville = match_louisville(louisville_tiles)
    boulder = match_boulder(boulder_tiles)

    print(f"Louisville matched: {len(louisville)}")
    print(f"Boulder matched:    {len(boulder)}")

    # Combine
    labels = {}
    labels.update(louisville)
    labels.update(boulder)
    print(f"Total combined:     {len(labels)}")

    # Stats
    dists = [v['dist_tile_to_rtk_m'] for v in labels.values() if v['dist_tile_to_rtk_m'] is not None]
    if dists:
        print(f"\nTile entrance -> RTK front door distance:")
        import statistics
        print(f"  Mean:   {statistics.mean(dists):.1f} m")
        print(f"  Median: {statistics.median(dists):.1f} m")
        print(f"  Max:    {max(dists):.1f} m")
        print(f"  Min:    {min(dists):.1f} m")

    print(f"\nPer-POI details:")
    for src in ['louisville', 'boulder']:
        src_labels = {k: v for k, v in labels.items() if v['source'] == src}
        print(f"\n  --- {src.upper()} ({len(src_labels)}) ---")
        for v in sorted(src_labels.values(), key=lambda x: x['gt_name']):
            d = v['dist_tile_to_rtk_m']
            d_str = f"{d:6.1f}m" if d else "  N/A "
            print(f"    {v['gt_name']:40s} <-> {v['wp_name']:45s} tile->RTK: {d_str}  imgs: {v['n_mapillary_images']}")

    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved {len(labels)} labels to {output_path}")


if __name__ == '__main__':
    main()
