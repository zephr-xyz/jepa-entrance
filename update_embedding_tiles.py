"""
Update entrance locations using Overture address points as reference.

For POIs with JEPA predictions: use the model's predicted entrance.
For remaining POIs: match the POI address to Overture's address layer,
then snap that address point to the nearest building facade edge.
"""
import json
import math
import re
import sys
from pathlib import Path


def normalize_address_key(address):
    """Normalize a POI address to match Overture address format."""
    if not address:
        return None
    key = address.strip().lower()
    # Normalize street suffixes to Overture's short form
    key = re.sub(r'\bstreet\b', 'st', key)
    key = re.sub(r'\bavenue\b', 'ave', key)
    key = re.sub(r'\bboulevard\b', 'blvd', key)
    key = re.sub(r'\bdrive\b', 'dr', key)
    key = re.sub(r'\broad\b', 'rd', key)
    key = re.sub(r'\blane\b', 'ln', key)
    key = re.sub(r'\bcourt\b', 'ct', key)
    key = re.sub(r'\bplace\b', 'pl', key)
    key = re.sub(r'\bparkway\b', 'pkwy', key)
    key = re.sub(r'\bcircle\b', 'cir', key)
    key = re.sub(r'\bhighway\b', 'hwy', key)
    # Remove trailing period
    key = re.sub(r'\.\s*$', '', key)
    return key


def snap_to_facade(ref_lat, ref_lon, building_coords_lnglat):
    """Snap reference point to nearest point on nearest building edge."""
    if len(building_coords_lnglat) < 3:
        return ref_lat, ref_lon

    cent_lng = sum(c[0] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cent_lat = sum(c[1] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cos_lat = math.cos(cent_lat * math.pi / 180)
    m_per_deg_lng = 111320 * cos_lat

    def to_m(coord):
        return [(coord[0] - cent_lng) * m_per_deg_lng,
                (coord[1] - cent_lat) * 111320]

    fp_m = [to_m(c) for c in building_coords_lnglat]
    ent_m = to_m([ref_lon, ref_lat])

    best_t = 0.5
    best_idx = 0
    best_dist = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue
        ax, ay = ent_m[0] - a[0], ent_m[1] - a[1]
        t = (ax * edx + ay * edy) / (length * length)
        t = max(0.0, min(1.0, t))
        px = a[0] + t * edx
        py = a[1] + t * edy
        dist = math.sqrt((ent_m[0] - px) ** 2 + (ent_m[1] - py) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_t = t
            best_idx = i

    a_ll = building_coords_lnglat[best_idx]
    b_ll = building_coords_lnglat[best_idx + 1]
    lat = a_ll[1] + best_t * (b_ll[1] - a_ll[1])
    lon = a_ll[0] + best_t * (b_ll[0] - a_ll[0])
    return lat, lon


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-dir', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--addresses-json', required=True,
                        help='overture_addresses.json')
    parser.add_argument('--predictions', default='')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    with open(args.buildings_json) as f:
        buildings = json.load(f)

    with open(args.addresses_json) as f:
        addresses = json.load(f)
    print(f"Loaded {len(addresses)} Overture address points")

    pred_by_id = {}
    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
        pred_by_id = {p['poi_id']: p for p in predictions}
        print(f"Loaded {len(pred_by_id)} JEPA predictions")

    tiles_dir = Path(args.tiles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jepa = 0
    n_address = 0
    n_poi_snap = 0
    n_unchanged = 0
    n_total = 0

    for tile_file in sorted(tiles_dir.glob('*.json')):
        with open(tile_file) as f:
            tile = json.load(f)

        for wp in tile.get('waypoints', []):
            n_total += 1
            poi_id = wp['id']

            if poi_id in pred_by_id:
                p = pred_by_id[poi_id]
                wp['entrance_lat'] = p['predicted_entrance_lat']
                wp['entrance_lon'] = p['predicted_entrance_lon']
                n_jepa += 1
            elif wp.get('overture_building_id'):
                bld_id = wp['overture_building_id']
                bld_raw = buildings.get(bld_id, [])
                if bld_raw and len(bld_raw) >= 3:
                    bld_lnglat = [[c[1], c[0]] for c in bld_raw]

                    # Try to use Overture address point as reference
                    addr_key = normalize_address_key(wp.get('address', ''))
                    addr_point = addresses.get(addr_key) if addr_key else None

                    if addr_point:
                        lat, lon = snap_to_facade(
                            addr_point['lat'], addr_point['lon'], bld_lnglat)
                        n_address += 1
                    else:
                        # Fallback: use POI centroid
                        ref_lat = wp.get('latitude', wp.get('entrance_lat', 0))
                        ref_lon = wp.get('longitude', wp.get('entrance_lon', 0))
                        lat, lon = snap_to_facade(ref_lat, ref_lon, bld_lnglat)
                        n_poi_snap += 1

                    wp['entrance_lat'] = lat
                    wp['entrance_lon'] = lon
                else:
                    n_unchanged += 1
            else:
                n_unchanged += 1

        with open(output_dir / tile_file.name, 'w') as f:
            json.dump(tile, f, indent=2)

    print(f"\nUpdated {n_total} POIs:")
    print(f"  JEPA prediction: {n_jepa}")
    print(f"  Address-point snap: {n_address}")
    print(f"  POI-centroid snap: {n_poi_snap}")
    print(f"  Unchanged: {n_unchanged}")


if __name__ == '__main__':
    main()
