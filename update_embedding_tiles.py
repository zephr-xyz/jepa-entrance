"""
Update embedding tiles with improved entrance locations.

For POIs with JEPA predictions (1,199): use the model's predicted entrance.
For remaining POIs with buildings: snap entrance to closest facade edge midpoint.
"""
import json
import math
import sys
from pathlib import Path

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def find_facade_edge(building_coords_lnglat, entrance_lat, entrance_lon):
    """Find closest building edge to entrance. Returns edge endpoints as lnglat."""
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
    best_dist = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        # Project entrance onto edge segment
        ax, ay = ent_m[0] - a[0], ent_m[1] - a[1]
        t = (ax * edx + ay * edy) / (length * length)
        t = max(0.0, min(1.0, t))

        px = a[0] + t * edx
        py = a[1] + t * edy
        dx = ent_m[0] - px
        dy = ent_m[1] - py
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < best_dist:
            best_dist = dist
            best_edge = {
                'a_lnglat': building_coords_lnglat[i],
                'b_lnglat': building_coords_lnglat[i + 1],
                't_nearest': t,
            }

    return best_edge


def snap_to_facade(entrance_lat, entrance_lon, building_coords_lnglat):
    """Snap entrance to nearest point on closest facade edge."""
    edge = find_facade_edge(building_coords_lnglat, entrance_lat, entrance_lon)
    if edge is None:
        return entrance_lat, entrance_lon

    a = edge['a_lnglat']
    b = edge['b_lnglat']
    t = edge['t_nearest']
    lat = a[1] + t * (b[1] - a[1])
    lon = a[0] + t * (b[0] - a[0])
    return lat, lon


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-dir', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--predictions', required=True,
                        help='updated_entrances.json from predict_entrances.py')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for updated tiles')
    args = parser.parse_args()

    with open(args.buildings_json) as f:
        buildings = json.load(f)

    with open(args.predictions) as f:
        predictions = json.load(f)

    # Index predictions by poi_id
    pred_by_id = {p['poi_id']: p for p in predictions}
    print(f"Loaded {len(pred_by_id)} JEPA predictions")

    tiles_dir = Path(args.tiles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jepa = 0
    n_facade = 0
    n_unchanged = 0
    n_total = 0

    for tile_file in sorted(tiles_dir.glob('*.json')):
        with open(tile_file) as f:
            tile = json.load(f)

        for wp in tile.get('waypoints', []):
            n_total += 1
            poi_id = wp['id']

            if poi_id in pred_by_id:
                # Use JEPA prediction
                p = pred_by_id[poi_id]
                wp['entrance_lat'] = p['predicted_entrance_lat']
                wp['entrance_lon'] = p['predicted_entrance_lon']
                n_jepa += 1
            elif wp.get('overture_building_id') and wp.get('entrance_lat'):
                # Use facade matcher: snap to nearest point on closest edge
                bld_id = wp['overture_building_id']
                bld_raw = buildings.get(bld_id, [])
                if bld_raw and len(bld_raw) >= 3:
                    bld_lnglat = [[c[1], c[0]] for c in bld_raw]
                    lat, lon = snap_to_facade(
                        wp['entrance_lat'], wp['entrance_lon'], bld_lnglat)
                    wp['entrance_lat'] = lat
                    wp['entrance_lon'] = lon
                    n_facade += 1
                else:
                    n_unchanged += 1
            else:
                n_unchanged += 1

        with open(output_dir / tile_file.name, 'w') as f:
            json.dump(tile, f, indent=2)

    print(f"\nUpdated {n_total} POIs:")
    print(f"  JEPA prediction: {n_jepa}")
    print(f"  Facade snap: {n_facade}")
    print(f"  Unchanged: {n_unchanged}")


if __name__ == '__main__':
    main()
