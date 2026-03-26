"""
Update entrance locations using the PR #8 facade-address-filter approach.

For each building, labels edges by nearest Overture street. For each POI,
matches the address street to a labeled edge, then snaps the Overture
address point onto that edge. Falls back gracefully when data is missing.
"""
import json
import math
import re
from pathlib import Path


DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def _normalize_street(name):
    """Normalize street name for matching (lowercase, expand abbreviations)."""
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r'\bst\b\.?', 'street', s)
    s = re.sub(r'\bave\b\.?', 'avenue', s)
    s = re.sub(r'\bblvd\b\.?', 'boulevard', s)
    s = re.sub(r'\bdr\b\.?', 'drive', s)
    s = re.sub(r'\brd\b\.?', 'road', s)
    s = re.sub(r'\bln\b\.?', 'lane', s)
    s = re.sub(r'\bct\b\.?', 'court', s)
    s = re.sub(r'\bpl\b\.?', 'place', s)
    s = re.sub(r'\bpkwy\b\.?', 'parkway', s)
    s = re.sub(r'\bcir\b\.?', 'circle', s)
    s = re.sub(r'\bhwy\b\.?', 'highway', s)
    return s.strip()


def _parse_address(address):
    """Parse address into (number, street_norm) or (None, None)."""
    if not address:
        return None, None
    m = re.match(r'^(\d+[a-z]?)\s+(.+)', address.strip(), re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).lower(), _normalize_street(m.group(2))


def _point_to_segment_dist_m(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (ax,ay)-(bx,by) in meters coords."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 0.01:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2), 0.5
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / l2))
    nx = ax + t * dx
    ny = ay + t * dy
    return math.sqrt((px - nx) ** 2 + (py - ny) ** 2), t


def _to_local_m(coords_lnglat, cent_lng, cent_lat, m_per_deg_lng):
    """Convert [lng, lat] coords to local meters."""
    return [[(c[0] - cent_lng) * m_per_deg_lng,
             (c[1] - cent_lat) * METERS_PER_DEG_LAT] for c in coords_lnglat]


class RoadGrid:
    """Spatial grid index for fast road segment lookup."""

    def __init__(self, road_segments, cell_deg=0.001):
        self.cell_deg = cell_deg
        self.grid = {}  # (cell_x, cell_y) -> [(name_norm, seg_lat1, seg_lon1, seg_lat2, seg_lon2)]

        for street_name, segments in road_segments.items():
            name_norm = _normalize_street(street_name)
            if len(name_norm) < 3:
                continue
            for seg in segments:
                coords = seg.get('coordinates', [])
                for j in range(len(coords) - 1):
                    a_lat, a_lon = coords[j]
                    b_lat, b_lon = coords[j + 1]
                    # Insert into all cells this segment touches
                    min_lat = min(a_lat, b_lat) - cell_deg
                    max_lat = max(a_lat, b_lat) + cell_deg
                    min_lon = min(a_lon, b_lon) - cell_deg
                    max_lon = max(a_lon, b_lon) + cell_deg
                    for cy in range(int(min_lat / cell_deg), int(max_lat / cell_deg) + 1):
                        for cx in range(int(min_lon / cell_deg), int(max_lon / cell_deg) + 1):
                            key = (cx, cy)
                            if key not in self.grid:
                                self.grid[key] = []
                            self.grid[key].append((name_norm, a_lat, a_lon, b_lat, b_lon))

    def nearby_segments(self, lat, lon):
        """Return road segments near a point."""
        cx = int(lon / self.cell_deg)
        cy = int(lat / self.cell_deg)
        return self.grid.get((cx, cy), [])


def label_building_edges(bld_lnglat, road_grid, max_dist_m=30.0):
    """Label each building edge with its nearest street name.

    Returns dict mapping edge_index -> street_name_norm.
    Mirrors PR #8's facades_by_street approach.
    """
    if len(bld_lnglat) < 3:
        return {}

    cent_lng = sum(c[0] for c in bld_lnglat) / len(bld_lnglat)
    cent_lat = sum(c[1] for c in bld_lnglat) / len(bld_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    fp_m = _to_local_m(bld_lnglat, cent_lng, cent_lat, m_per_deg_lng)

    edge_streets = {}
    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        mid_m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
        mid_lng = mid_m[0] / m_per_deg_lng + cent_lng
        mid_lat = mid_m[1] / METERS_PER_DEG_LAT + cent_lat

        best_name = None
        best_dist = max_dist_m

        for name_norm, sa_lat, sa_lon, sb_lat, sb_lon in road_grid.nearby_segments(mid_lat, mid_lng):
            sa_m = [(sa_lon - cent_lng) * m_per_deg_lng,
                    (sa_lat - cent_lat) * METERS_PER_DEG_LAT]
            sb_m = [(sb_lon - cent_lng) * m_per_deg_lng,
                    (sb_lat - cent_lat) * METERS_PER_DEG_LAT]
            dist, _ = _point_to_segment_dist_m(
                mid_m[0], mid_m[1], sa_m[0], sa_m[1], sb_m[0], sb_m[1])
            if dist < best_dist:
                best_dist = dist
                best_name = name_norm

        if best_name:
            edge_streets[i] = best_name

    return edge_streets


def snap_to_edge(ref_lat, ref_lon, bld_lnglat, edge_idx):
    """Snap a reference point to a specific building edge."""
    cent_lng = sum(c[0] for c in bld_lnglat) / len(bld_lnglat)
    cent_lat = sum(c[1] for c in bld_lnglat) / len(bld_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    fp_m = _to_local_m(bld_lnglat, cent_lng, cent_lat, m_per_deg_lng)
    ref_m = [(ref_lon - cent_lng) * m_per_deg_lng,
             (ref_lat - cent_lat) * METERS_PER_DEG_LAT]

    a, b = fp_m[edge_idx], fp_m[edge_idx + 1]
    edx, edy = b[0] - a[0], b[1] - a[1]
    l2 = edx * edx + edy * edy
    if l2 < 0.01:
        t = 0.5
    else:
        t = max(0.0, min(1.0, ((ref_m[0] - a[0]) * edx + (ref_m[1] - a[1]) * edy) / l2))

    a_ll = bld_lnglat[edge_idx]
    b_ll = bld_lnglat[edge_idx + 1]
    lat = a_ll[1] + t * (b_ll[1] - a_ll[1])
    lon = a_ll[0] + t * (b_ll[0] - a_ll[0])
    return lat, lon


def snap_to_nearest_edge(ref_lat, ref_lon, bld_lnglat):
    """Fallback: snap to nearest point on nearest edge."""
    if len(bld_lnglat) < 3:
        return ref_lat, ref_lon

    cent_lng = sum(c[0] for c in bld_lnglat) / len(bld_lnglat)
    cent_lat = sum(c[1] for c in bld_lnglat) / len(bld_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    fp_m = _to_local_m(bld_lnglat, cent_lng, cent_lat, m_per_deg_lng)
    ref_m = [(ref_lon - cent_lng) * m_per_deg_lng,
             (ref_lat - cent_lat) * METERS_PER_DEG_LAT]

    best_idx = 0
    best_t = 0.5
    best_dist = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue
        dist, t = _point_to_segment_dist_m(
            ref_m[0], ref_m[1], a[0], a[1], b[0], b[1])
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_t = t

    a_ll = bld_lnglat[best_idx]
    b_ll = bld_lnglat[best_idx + 1]
    lat = a_ll[1] + best_t * (b_ll[1] - a_ll[1])
    lon = a_ll[0] + best_t * (b_ll[0] - a_ll[0])
    return lat, lon


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles-dir', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--addresses-json', required=True,
                        help='Overture address points (overture_addresses.json)')
    parser.add_argument('--roads-json', required=True,
                        help='Overture road geometries (road_geometries.json)')
    parser.add_argument('--predictions', default='',
                        help='JEPA predictions (updated_entrances.json)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--edge-street-max-dist-m', type=float, default=30.0)
    args = parser.parse_args()

    with open(args.buildings_json) as f:
        buildings = json.load(f)
    with open(args.addresses_json) as f:
        addresses = json.load(f)
    with open(args.roads_json) as f:
        road_segments = json.load(f)

    print(f"Loaded {len(buildings)} buildings, {len(addresses)} addresses, "
          f"{len(road_segments)} road names")

    pred_by_id = {}
    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
        pred_by_id = {p['poi_id']: p for p in predictions}
        print(f"Loaded {len(pred_by_id)} JEPA predictions")

    # Build spatial grid index for road segments
    print("Building road spatial index...")
    road_grid = RoadGrid(road_segments)
    print(f"  Grid cells: {len(road_grid.grid)}")

    bld_edge_labels = {}

    tiles_dir = Path(args.tiles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jepa = 0
    n_address_street = 0  # Used address point + street-labeled edge
    n_address_nearest = 0  # Used address point + nearest edge
    n_poi_snap = 0  # Used POI centroid + nearest edge
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
                continue

            bld_id = wp.get('overture_building_id')
            if not bld_id:
                n_unchanged += 1
                continue

            bld_raw = buildings.get(bld_id, [])
            if not bld_raw or len(bld_raw) < 3:
                n_unchanged += 1
                continue

            bld_lnglat = [[c[1], c[0]] for c in bld_raw]

            # Parse POI address
            _, addr_street = _parse_address(wp.get('address', ''))

            # Look up Overture address point
            addr_key = (wp.get('address') or '').strip().lower()
            addr_key = re.sub(r'\bstreet\b', 'st', addr_key)
            addr_key = re.sub(r'\bavenue\b', 'ave', addr_key)
            addr_key = re.sub(r'\bboulevard\b', 'blvd', addr_key)
            addr_key = re.sub(r'\bdrive\b', 'dr', addr_key)
            addr_key = re.sub(r'\broad\b', 'rd', addr_key)
            addr_key = re.sub(r'\blane\b', 'ln', addr_key)
            addr_key = re.sub(r'\bcourt\b', 'ct', addr_key)
            addr_key = re.sub(r'\bplace\b', 'pl', addr_key)
            addr_point = addresses.get(addr_key)

            # Determine reference point: prefer Overture address, fall back to POI
            if addr_point:
                ref_lat, ref_lon = addr_point['lat'], addr_point['lon']
            else:
                ref_lat = wp.get('latitude', wp.get('entrance_lat', 0))
                ref_lon = wp.get('longitude', wp.get('entrance_lon', 0))

            # Try street-labeled edge selection
            if addr_street:
                # Label building edges (cached per building)
                if bld_id not in bld_edge_labels:
                    bld_edge_labels[bld_id] = label_building_edges(
                        bld_lnglat, road_grid, args.edge_street_max_dist_m)
                edge_labels = bld_edge_labels[bld_id]

                # Find edges matching the POI's address street
                matching_edges = [idx for idx, street in edge_labels.items()
                                  if street == addr_street]

                if matching_edges:
                    # Snap to the matching edge closest to the reference point
                    best_idx = matching_edges[0]
                    best_dist = float('inf')
                    for idx in matching_edges:
                        lat, lon = snap_to_edge(ref_lat, ref_lon, bld_lnglat, idx)
                        d = math.sqrt((lat - ref_lat) ** 2 + (lon - ref_lon) ** 2)
                        if d < best_dist:
                            best_dist = d
                            best_idx = idx
                    lat, lon = snap_to_edge(ref_lat, ref_lon, bld_lnglat, best_idx)
                    wp['entrance_lat'] = lat
                    wp['entrance_lon'] = lon
                    n_address_street += 1
                    continue

            # Fallback: snap reference point to nearest edge
            if addr_point:
                lat, lon = snap_to_nearest_edge(ref_lat, ref_lon, bld_lnglat)
                wp['entrance_lat'] = lat
                wp['entrance_lon'] = lon
                n_address_nearest += 1
            else:
                ref_lat = wp.get('latitude', wp.get('entrance_lat', 0))
                ref_lon = wp.get('longitude', wp.get('entrance_lon', 0))
                lat, lon = snap_to_nearest_edge(ref_lat, ref_lon, bld_lnglat)
                wp['entrance_lat'] = lat
                wp['entrance_lon'] = lon
                n_poi_snap += 1

        with open(output_dir / tile_file.name, 'w') as f:
            json.dump(tile, f, indent=2)

    print(f"\nUpdated {n_total} POIs:")
    print(f"  JEPA prediction: {n_jepa}")
    print(f"  Address + street-labeled edge: {n_address_street}")
    print(f"  Address + nearest edge: {n_address_nearest}")
    print(f"  POI centroid + nearest edge: {n_poi_snap}")
    print(f"  Unchanged: {n_unchanged}")
    print(f"  Building edge labels cached: {len(bld_edge_labels)}")


if __name__ == '__main__':
    main()
