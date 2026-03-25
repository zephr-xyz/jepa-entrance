"""
Update entrance locations for ALL POIs using address-aware facade matching.

For POIs with JEPA predictions: use the model's predicted entrance.
For remaining POIs: snap entrance to the building edge closest to the
street named in the POI's address.
"""
import json
import math
import re
import sys
from pathlib import Path


def extract_street_name(address):
    """Extract street name from address like '946 Pearl Street' -> 'Pearl Street'."""
    if not address:
        return None
    # Remove house number prefix
    m = re.match(r'^\d+\s+(.+)', address)
    if m:
        name = m.group(1).strip()
        # Normalize common abbreviations
        name = re.sub(r'\bSt\b\.?$', 'Street', name)
        name = re.sub(r'\bAve\b\.?$', 'Avenue', name)
        name = re.sub(r'\bBlvd\b\.?$', 'Boulevard', name)
        name = re.sub(r'\bDr\b\.?$', 'Drive', name)
        name = re.sub(r'\bRd\b\.?$', 'Road', name)
        name = re.sub(r'\bLn\b\.?$', 'Lane', name)
        name = re.sub(r'\bCt\b\.?$', 'Court', name)
        name = re.sub(r'\bPl\b\.?$', 'Place', name)
        name = re.sub(r'\bPkwy\b\.?$', 'Parkway', name)
        name = re.sub(r'\bCir\b\.?$', 'Circle', name)
        name = re.sub(r'\bHwy\b\.?$', 'Highway', name)
        return name
    return None


def find_nearest_road_point(poi_lat, poi_lon, road_segments, max_dist_m=200):
    """Find the nearest point on any segment of a named road to the POI."""
    best_lat = None
    best_lon = None
    best_dist = float('inf')

    cos_lat = math.cos(math.radians(poi_lat))
    m_per_deg_lng = 111320 * cos_lat

    px = poi_lon * m_per_deg_lng
    py = poi_lat * 111320

    for seg in road_segments:
        coords = seg.get('coordinates', [])
        for i in range(len(coords) - 1):
            a_lat, a_lon = coords[i]
            b_lat, b_lon = coords[i + 1]

            ax = a_lon * m_per_deg_lng
            ay = a_lat * 111320
            bx = b_lon * m_per_deg_lng
            by = b_lat * 111320

            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            if l2 < 0.01:
                continue
            t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / l2))
            nx = ax + t * dx
            ny = ay + t * dy
            d = math.sqrt((px - nx) ** 2 + (py - ny) ** 2)

            if d < best_dist:
                best_dist = d
                best_lat = a_lat + t * (b_lat - a_lat)
                best_lon = a_lon + t * (b_lon - a_lon)

    if best_dist > max_dist_m or best_lat is None:
        return None
    return best_lat, best_lon


def snap_to_facade_with_road(poi_lat, poi_lon, building_coords_lnglat, road_point):
    """Snap to the building edge facing the road.

    Uses the road point to identify which edge faces the street (outward
    normal points toward the road), then projects the POI onto that edge.
    """
    if len(building_coords_lnglat) < 3:
        return poi_lat, poi_lon

    cent_lng = sum(c[0] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cent_lat = sum(c[1] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cos_lat = math.cos(cent_lat * math.pi / 180)
    m_per_deg_lng = 111320 * cos_lat

    def to_m(coord):
        return [(coord[0] - cent_lng) * m_per_deg_lng,
                (coord[1] - cent_lat) * 111320]

    fp_m = [to_m(c) for c in building_coords_lnglat]
    road_m = to_m([road_point[1], road_point[0]])
    poi_m = to_m([poi_lon, poi_lat])
    poly_cx = sum(p[0] for p in fp_m) / len(fp_m)
    poly_cy = sum(p[1] for p in fp_m) / len(fp_m)

    # Score each edge: prefer edges whose outward normal points toward the road
    best_edge_idx = None
    best_score = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        # Outward normal (pointing away from building centroid)
        nx, ny = -edy / length, edx / length
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if nx * (poly_cx - mx) + ny * (poly_cy - my) > 0:
            nx, ny = -nx, -ny

        # Distance from road point to edge
        ax, ay = road_m[0] - a[0], road_m[1] - a[1]
        t = max(0.0, min(1.0, (ax * edx + ay * edy) / (length * length)))
        px = a[0] + t * edx
        py = a[1] + t * edy
        edge_dist = math.sqrt((road_m[0] - px) ** 2 + (road_m[1] - py) ** 2)

        # Check if road is on the outward side (normal dot product > 0)
        road_dx = road_m[0] - mx
        road_dy = road_m[1] - my
        facing = nx * road_dx + ny * road_dy

        # Score: prefer edges that face the road (facing > 0) and are close
        # Among facing edges, prefer the one more directly facing (higher dot product)
        if facing > 0:
            score = edge_dist - facing * 0.5  # reward stronger facing
        else:
            score = edge_dist + 1000  # penalize edges facing away from road

        if score < best_score:
            best_score = score
            best_edge_idx = i

    if best_edge_idx is None:
        return poi_lat, poi_lon

    # Project POI onto the road-facing edge
    a, b = fp_m[best_edge_idx], fp_m[best_edge_idx + 1]
    edx, edy = b[0] - a[0], b[1] - a[1]
    length = math.sqrt(edx * edx + edy * edy)
    ax, ay = poi_m[0] - a[0], poi_m[1] - a[1]
    t = (ax * edx + ay * edy) / (length * length)
    t = max(0.0, min(1.0, t))

    a_ll = building_coords_lnglat[best_edge_idx]
    b_ll = building_coords_lnglat[best_edge_idx + 1]
    lat = a_ll[1] + t * (b_ll[1] - a_ll[1])
    lon = a_ll[0] + t * (b_ll[0] - a_ll[0])
    return lat, lon


def snap_to_facade_simple(ref_lat, ref_lon, building_coords_lnglat):
    """Fallback: snap to nearest point on nearest edge."""
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
    parser.add_argument('--roads-json', required=True,
                        help='road_geometries.json from fetch_roads.py')
    parser.add_argument('--predictions', default='')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    with open(args.buildings_json) as f:
        buildings = json.load(f)

    with open(args.roads_json) as f:
        roads = json.load(f)
    print(f"Loaded {len(roads)} road names")

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
    n_road_facade = 0
    n_simple_facade = 0
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
                    poi_lat = wp.get('latitude', wp.get('entrance_lat', 0))
                    poi_lon = wp.get('longitude', wp.get('entrance_lon', 0))

                    # Try address-aware facade matching
                    street = extract_street_name(wp.get('address', ''))
                    road_point = None
                    if street and street in roads:
                        road_point = find_nearest_road_point(
                            poi_lat, poi_lon, roads[street])

                    if road_point:
                        lat, lon = snap_to_facade_with_road(
                            poi_lat, poi_lon, bld_lnglat, road_point)
                        n_road_facade += 1
                    else:
                        lat, lon = snap_to_facade_simple(
                            poi_lat, poi_lon, bld_lnglat)
                        n_simple_facade += 1

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
    print(f"  Road-aware facade snap: {n_road_facade}")
    print(f"  Simple facade snap: {n_simple_facade}")
    print(f"  Unchanged: {n_unchanged}")


if __name__ == '__main__':
    main()
