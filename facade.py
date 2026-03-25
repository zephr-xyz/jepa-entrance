"""
Composite facade detection for building entrance prediction.

Instead of picking a single edge as "the facade", finds all road-facing
edges of a building and stitches them into a composite polyline. The
entrance_t is then computed along this full polyline.

Key idea: Mapillary cameras sit on roads. Building edges whose outward
normals point toward camera positions (or the POI centroid) are road-facing.
Connected road-facing edges + short connecting edges = the composite facade.
"""
import math
import numpy as np

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def _to_local_m(coords_lnglat):
    """Convert [lng, lat] coords to local meters centered on centroid.

    Returns: fp_m list, centroid_lat, centroid_lon
    """
    cent_lng = sum(c[0] for c in coords_lnglat) / len(coords_lnglat)
    cent_lat = sum(c[1] for c in coords_lnglat) / len(coords_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    fp_m = [[(c[0] - cent_lng) * m_per_deg_lng,
             (c[1] - cent_lat) * METERS_PER_DEG_LAT] for c in coords_lnglat]

    return fp_m, cent_lat, cent_lng


def _edge_outward_normal(a, b, poly_cx, poly_cy):
    """Compute outward-pointing unit normal for edge a→b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.01:
        return None, None, 0.0

    nx, ny = -dy / length, dx / length
    # Flip if pointing inward (toward centroid)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    tcx, tcy = poly_cx - mx, poly_cy - my
    if nx * tcx + ny * tcy > 0:
        nx, ny = -nx, -ny

    return nx, ny, length


def find_composite_facade(building_coords_lnglat, reference_points_m=None,
                          entrance_m=None, min_edge_len=0.3,
                          facing_threshold=0.0, max_connector_len=5.0):
    """Find the composite road-facing facade of a building.

    Args:
        building_coords_lnglat: List of [lng, lat] coordinates (polygon ring)
        reference_points_m: Optional list of [x, y] points in local meters
            that represent "where the road is" (e.g., camera positions).
            If None, uses entrance_m or falls back to POI centroid direction.
        entrance_m: [x, y] entrance position in local meters (optional)
        min_edge_len: Minimum edge length to consider (meters)
        facing_threshold: Dot product threshold for "facing road" (0.0 = any
            outward edge facing toward reference, negative = more permissive)
        max_connector_len: Max length for connecting edges between road-facing
            segments to be included in composite facade

    Returns:
        dict with:
            'polyline': list of [x, y] points forming the composite facade
            'edge_indices': which building polygon edges are included
            'total_length': total polyline arc length in meters
            'segments': list of (start_m, end_m, edge_idx) for each segment
            'centroid_lat': building centroid latitude
            'centroid_lon': building centroid longitude
            'facade_a_m': first point of composite facade (for backward compat)
            'facade_b_m': last point of composite facade (for backward compat)
            'midpoint_m': midpoint of composite facade
            'normal': average outward normal of composite facade
        or None if computation fails
    """
    if len(building_coords_lnglat) < 3:
        return None

    fp_m, cent_lat, cent_lng = _to_local_m(building_coords_lnglat)
    poly_cx = sum(p[0] for p in fp_m) / len(fp_m)
    poly_cy = sum(p[1] for p in fp_m) / len(fp_m)

    n_edges = len(fp_m) - 1
    if n_edges < 3:
        return None

    # Compute edge properties
    edges = []
    for i in range(n_edges):
        a, b = fp_m[i], fp_m[i + 1]
        nx, ny, length = _edge_outward_normal(a, b, poly_cx, poly_cy)
        if length < 0.01:
            edges.append({'a': a, 'b': b, 'length': 0, 'nx': 0, 'ny': 0})
            continue
        edges.append({
            'a': a, 'b': b, 'length': length,
            'nx': nx, 'ny': ny,
            'mx': (a[0] + b[0]) / 2, 'my': (a[1] + b[1]) / 2,
        })

    # Determine road direction: vector from building centroid toward road
    if reference_points_m and len(reference_points_m) > 0:
        # Average direction from centroid to camera/reference points
        road_dx = sum(p[0] - poly_cx for p in reference_points_m) / len(reference_points_m)
        road_dy = sum(p[1] - poly_cy for p in reference_points_m) / len(reference_points_m)
    elif entrance_m:
        road_dx = entrance_m[0] - poly_cx
        road_dy = entrance_m[1] - poly_cy
    else:
        return None

    road_dist = math.sqrt(road_dx * road_dx + road_dy * road_dy)
    if road_dist < 0.01:
        return None
    road_dx /= road_dist
    road_dy /= road_dist

    # Score each edge: how much does its outward normal align with road direction?
    edge_facing_score = []
    for e in edges:
        if e['length'] < min_edge_len:
            edge_facing_score.append(-2.0)  # skip tiny edges initially
            continue
        score = e['nx'] * road_dx + e['ny'] * road_dy
        edge_facing_score.append(score)

    # Find connected runs of road-facing edges
    # An edge is "road-facing" if its normal dot road_dir > threshold
    is_facing = [s > facing_threshold for s in edge_facing_score]

    # Also mark short edges between two facing edges as connectors
    for i in range(n_edges):
        if not is_facing[i] and edges[i]['length'] < max_connector_len:
            prev_i = (i - 1) % n_edges
            next_i = (i + 1) % n_edges
            if is_facing[prev_i] and is_facing[next_i]:
                is_facing[i] = True

    # Find the best connected run (longest total length)
    # Handle wrap-around by doubling the array
    doubled = is_facing + is_facing
    doubled_edges = edges + edges

    best_run = None
    best_length = 0

    i = 0
    while i < 2 * n_edges:
        if not doubled[i]:
            i += 1
            continue
        # Start of a run
        run_start = i
        run_length = 0
        while i < 2 * n_edges and doubled[i]:
            run_length += doubled_edges[i]['length']
            i += 1
        run_end = i  # exclusive

        # Only consider runs that don't wrap more than once
        if run_end - run_start <= n_edges and run_length > best_length:
            best_length = run_length
            best_run = (run_start, run_end)

    if best_run is None or best_length < min_edge_len:
        # Fallback: pick the single edge closest to entrance
        if entrance_m:
            best_idx = None
            best_dist = float('inf')
            for i, e in enumerate(edges):
                if e['length'] < min_edge_len:
                    continue
                d = math.sqrt((e['mx'] - entrance_m[0])**2 +
                              (e['my'] - entrance_m[1])**2)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None:
                best_run = (best_idx, best_idx + 1)
                best_length = edges[best_idx]['length']
            else:
                return None
        else:
            return None

    # Build polyline from the run
    run_start, run_end = best_run
    polyline = []
    edge_indices = []
    segments = []
    cumulative_len = 0.0
    avg_nx, avg_ny = 0.0, 0.0
    total_weight = 0.0

    for j in range(run_start, run_end):
        idx = j % n_edges
        e = edges[idx]
        if not polyline:
            polyline.append(e['a'])
        polyline.append(e['b'])
        edge_indices.append(idx)

        seg_start = cumulative_len
        cumulative_len += e['length']
        segments.append((seg_start, cumulative_len, idx))

        # Weighted average normal
        if e['length'] >= min_edge_len:
            avg_nx += e['nx'] * e['length']
            avg_ny += e['ny'] * e['length']
            total_weight += e['length']

    if total_weight > 0:
        avg_nx /= total_weight
        avg_ny /= total_weight
        norm = math.sqrt(avg_nx * avg_nx + avg_ny * avg_ny)
        if norm > 0:
            avg_nx /= norm
            avg_ny /= norm

    # Midpoint: point at half the arc length
    half_len = cumulative_len / 2
    mid_x, mid_y = polyline[0][0], polyline[0][1]
    for seg_start, seg_end, _ in segments:
        if seg_end >= half_len:
            # Interpolate within this segment
            seg_idx = segments.index((seg_start, seg_end, _))
            e_idx = edge_indices[seg_idx]
            e = edges[e_idx]
            frac = (half_len - seg_start) / max(e['length'], 0.01)
            mid_x = e['a'][0] + frac * (e['b'][0] - e['a'][0])
            mid_y = e['a'][1] + frac * (e['b'][1] - e['a'][1])
            break

    return {
        'polyline': polyline,
        'edge_indices': edge_indices,
        'total_length': cumulative_len,
        'segments': segments,
        'centroid_lat': cent_lat,
        'centroid_lon': cent_lng,
        'facade_a_m': polyline[0],
        'facade_b_m': polyline[-1],
        'midpoint_m': [mid_x, mid_y],
        'normal': [avg_nx, avg_ny],
        'edges': edges,  # all edges for debugging
    }


def project_point_onto_composite_facade(point_m, facade_result):
    """Project a point onto the composite facade polyline.

    Returns:
        t: float in [0, 1] — fraction along total facade arc length
        dist: perpendicular distance from point to closest edge
    """
    if facade_result is None:
        return 0.5, float('inf')

    polyline = facade_result['polyline']
    total_length = facade_result['total_length']
    segments = facade_result['segments']
    edges_data = facade_result['edges']

    if total_length < 0.01:
        return 0.5, float('inf')

    px, py = point_m

    best_t = 0.5
    best_dist = float('inf')

    for seg_start, seg_end, edge_idx in segments:
        e = edges_data[edge_idx]
        a, b = e['a'], e['b']
        seg_len = e['length']
        if seg_len < 0.01:
            continue

        # Project point onto this segment
        ax, ay = a[0], a[1]
        dx, dy = b[0] - ax, b[1] - ay
        ex, ey = px - ax, py - ay

        local_t = (ex * dx + ey * dy) / (seg_len * seg_len)
        local_t_clamped = max(0.0, min(1.0, local_t))

        # Closest point on segment
        closest_x = ax + local_t_clamped * dx
        closest_y = ay + local_t_clamped * dy
        dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        if dist < best_dist:
            best_dist = dist
            # Global t along composite facade
            arc_pos = seg_start + local_t_clamped * seg_len
            best_t = arc_pos / total_length

    return best_t, best_dist


def compute_composite_facade_and_entrance_t(building_coords_lnglat,
                                             entrance_lat, entrance_lon,
                                             enclosing_roads,
                                             camera_positions_latlon=None):
    """Compute facade features and entrance_t using composite facade.

    Drop-in replacement for dataset.compute_facade_and_entrance_t with
    improved facade detection.

    Returns:
        facade_feats: (32,) float array
        entrance_t: float in [0, 1]
        facade_result: dict with composite facade details (for v3 geometry)
        or None, None, None if computation fails
    """
    if len(building_coords_lnglat) < 3:
        return None, None, None

    fp_m, cent_lat, cent_lng = _to_local_m(building_coords_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    ent_m = [(entrance_lon - cent_lng) * m_per_deg_lng,
             (entrance_lat - cent_lat) * METERS_PER_DEG_LAT]

    # Build reference points from camera positions
    ref_points = None
    if camera_positions_latlon:
        ref_points = [
            [(lon - cent_lng) * m_per_deg_lng,
             (lat - cent_lat) * METERS_PER_DEG_LAT]
            for lat, lon in camera_positions_latlon
        ]

    facade = find_composite_facade(
        building_coords_lnglat,
        reference_points_m=ref_points,
        entrance_m=ent_m,
    )

    if facade is None:
        return None, None, None

    # Project entrance onto composite facade
    entrance_t, dist_to_facade = project_point_onto_composite_facade(
        ent_m, facade
    )

    # Build facade feature vector (32 dims) — compatible with existing format
    # Use first and last points of composite as a/b for backward compat
    a = facade['facade_a_m']
    b = facade['facade_b_m']
    normal = facade['normal']
    midpoint = facade['midpoint_m']
    bearing = math.degrees(math.atan2(normal[0], normal[1]))

    poly_cx = sum(p[0] for p in fp_m) / len(fp_m)
    poly_cy = sum(p[1] for p in fp_m) / len(fp_m)

    # Road class encoding
    ROAD_CLASSES = {
        'motorway': 0, 'trunk': 1, 'primary': 2, 'secondary': 3,
        'tertiary': 4, 'residential': 5, 'service': 6, 'unclassified': 7,
        'living_street': 8, 'pedestrian': 9, 'track': 10, 'path': 11,
    }
    road_class_enc = [0.0] * 12
    if enclosing_roads:
        rc = enclosing_roads[0].get('road_class', 'residential')
        idx = ROAD_CLASSES.get(rc, 5)
        road_class_enc[idx] = 1.0

    facade_feats = [
        a[0], a[1], b[0], b[1],        # 4: composite start/end
        facade['total_length'],          # 1: total facade length
        bearing / 180.0,                 # 1: avg bearing normalized
        normal[0], normal[1],            # 2: avg outward normal
        midpoint[0], midpoint[1],        # 2: midpoint
        poly_cx, poly_cy,               # 2: building centroid
        len(building_coords_lnglat),     # 1: vertex count
        dist_to_facade,                  # 1: distance entrance-to-facade
    ]
    facade_feats.extend(road_class_enc)  # 12: one-hot road class
    while len(facade_feats) < 32:
        facade_feats.append(0.0)

    return (np.array(facade_feats[:32], dtype=np.float32),
            float(entrance_t),
            facade)
