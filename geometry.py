"""
Camera-to-facade geometry computations.

Computes:
  - Camera position relative to facade in local meters
  - Per-patch-column facade_t values (where each column looks on the facade)
  - Camera pose feature vector (8d)
  - v4: Entrance-to-column mapping and ray-building intersection
"""
import math
import numpy as np

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def camera_hfov_deg(camera_params, camera_type, width, height):
    """Compute horizontal field of view from Mapillary camera parameters.

    Mapillary perspective cameras: camera_parameters = [focal, k1, k2]
    where focal is normalized by max(width, height).

    Equirectangular (pano): HFOV = 360°
    """
    if camera_type == 'equirectangular':
        return 360.0
    if camera_type == 'fisheye':
        return 180.0

    # Perspective: focal is normalized by max(w, h)
    focal_norm = camera_params[0] if camera_params else 0.5
    if focal_norm <= 0:
        focal_norm = 0.5

    # HFOV = 2 * atan(width / (2 * focal * max(width, height)))
    max_dim = max(width, height)
    focal_px = focal_norm * max_dim
    hfov = 2 * math.atan(width / (2 * focal_px))
    return math.degrees(hfov)


def compute_camera_pose_features(
    cam_lat, cam_lon, cam_compass_deg, cam_hfov_deg,
    facade_a_m, facade_b_m, facade_midpoint_m, facade_normal,
    building_centroid_lat, building_centroid_lon
):
    """Compute 8-dim camera pose feature vector relative to facade.

    All _m coordinates are in local meters centered on building centroid.

    Returns:
        pose_feats: (8,) float32 array
            [0] dx_cam: camera x relative to facade midpoint (meters)
            [1] dy_cam: camera y relative to facade midpoint (meters)
            [2] dist_to_facade: perpendicular distance to facade line (meters)
            [3] angle_to_normal: angle between camera heading and facade normal (radians, [-pi, pi])
            [4] cam_bearing_rel: camera compass relative to facade bearing (normalized [-1, 1])
            [5] hfov_norm: horizontal FOV normalized (0-1, where 1 = 360°)
            [6] along_facade_frac: camera position projected onto facade edge (0-1)
            [7] facing_facade: 1.0 if camera roughly faces facade, 0.0 otherwise
    """
    cos_lat = math.cos(building_centroid_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    # Camera position in local meters
    cam_x = (cam_lon - building_centroid_lon) * m_per_deg_lng
    cam_y = (cam_lat - building_centroid_lat) * METERS_PER_DEG_LAT

    # Relative to facade midpoint
    dx = cam_x - facade_midpoint_m[0]
    dy = cam_y - facade_midpoint_m[1]
    dist = math.sqrt(dx * dx + dy * dy)

    # Perpendicular distance to facade line
    fx = facade_b_m[0] - facade_a_m[0]
    fy = facade_b_m[1] - facade_a_m[1]
    facade_len = math.sqrt(fx * fx + fy * fy)
    if facade_len < 0.1:
        return np.zeros(8, dtype=np.float32)

    # Signed distance from camera to facade line
    # Using cross product: (cam - a) × (b - a) / |b - a|
    ax = cam_x - facade_a_m[0]
    ay = cam_y - facade_a_m[1]
    perp_dist = (ax * fy - ay * fx) / facade_len

    # Camera compass to radians
    cam_bearing_rad = cam_compass_deg * DEG_TO_RAD

    # Facade bearing (direction along edge a→b)
    facade_bearing_rad = math.atan2(fx, fy)

    # Facade normal angle
    normal_angle = math.atan2(facade_normal[0], facade_normal[1])

    # Angle between camera heading and facade normal
    angle_to_normal = cam_bearing_rad - normal_angle
    # Normalize to [-pi, pi]
    angle_to_normal = (angle_to_normal + math.pi) % (2 * math.pi) - math.pi

    # Camera bearing relative to facade bearing (normalized)
    rel_bearing = cam_bearing_rad - facade_bearing_rad
    rel_bearing = (rel_bearing + math.pi) % (2 * math.pi) - math.pi
    cam_bearing_rel = rel_bearing / math.pi  # [-1, 1]

    # Project camera onto facade edge (where along the facade is the camera centered)
    # t_cam = dot(cam - a, b - a) / |b - a|^2
    t_cam = (ax * fx + ay * fy) / (facade_len * facade_len)
    t_cam = max(-1.0, min(2.0, t_cam))  # allow slight overshoot

    # Is camera facing the facade? (dot product of cam direction and facade normal)
    cam_dx = math.sin(cam_bearing_rad)
    cam_dy = math.cos(cam_bearing_rad)
    # Camera faces facade when heading opposes outward normal (dot < 0)
    facing = cam_dx * facade_normal[0] + cam_dy * facade_normal[1]
    facing_facade = 1.0 if facing < 0 else 0.0

    return np.array([
        dx / 50.0,          # normalize to ~[-1, 1] for 50m range
        dy / 50.0,
        abs(perp_dist) / 50.0,
        angle_to_normal / math.pi,
        cam_bearing_rel,
        cam_hfov_deg / 360.0,
        t_cam,
        facing_facade,
    ], dtype=np.float32)


def compute_patch_column_facade_t(
    cam_lat, cam_lon, cam_compass_deg, cam_hfov_deg,
    facade_a_m, facade_b_m,
    building_centroid_lat, building_centroid_lon,
    n_cols=16, camera_type='perspective'
):
    """Compute facade_t for each horizontal patch column.

    Each column in the 16-wide patch grid corresponds to a horizontal
    bearing from the camera. We project these bearings as rays onto
    the facade edge line to get facade_t per column.

    For panoramic cameras, we use the full 360° HFOV but only the
    columns roughly facing the facade will have valid facade_t.

    Returns:
        facade_t_cols: (n_cols,) float32 array, values in [-0.5, 1.5]
            (can be outside [0,1] if column looks past facade edge)
    """
    cos_lat = math.cos(building_centroid_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    cam_x = (cam_lon - building_centroid_lon) * m_per_deg_lng
    cam_y = (cam_lat - building_centroid_lat) * METERS_PER_DEG_LAT

    # Facade edge vector
    fx = facade_b_m[0] - facade_a_m[0]
    fy = facade_b_m[1] - facade_a_m[1]
    facade_len_sq = fx * fx + fy * fy
    if facade_len_sq < 0.01:
        return np.full(n_cols, 0.5, dtype=np.float32)

    facade_t_cols = np.zeros(n_cols, dtype=np.float32)

    for col in range(n_cols):
        # Bearing for this column
        # Column 0 = left edge of image, column n_cols-1 = right edge
        frac = (col + 0.5) / n_cols  # center of column, [0, 1]
        col_bearing_deg = cam_compass_deg + (frac - 0.5) * cam_hfov_deg

        col_bearing_rad = col_bearing_deg * DEG_TO_RAD
        ray_dx = math.sin(col_bearing_rad)
        ray_dy = math.cos(col_bearing_rad)

        # Ray-line intersection: cam + t_ray * ray = facade_a + t_facade * (facade_b - facade_a)
        # Solve for t_facade using 2D cross product
        # cam_to_a = a - cam
        ax = facade_a_m[0] - cam_x
        ay = facade_a_m[1] - cam_y

        denom = ray_dy * fx - ray_dx * fy
        if abs(denom) < 1e-10:
            # Ray parallel to facade
            facade_t_cols[col] = 0.5
            continue

        t_facade = (ray_dx * ay - ray_dy * ax) / denom
        # t_ray uses the cross product of (a - cam) with facade direction
        t_ray = (fx * ay - fy * ax) / denom

        if t_ray < 0:
            # Ray points away from facade (behind camera for perspective)
            # For panos this can happen for columns facing away
            facade_t_cols[col] = -1.0  # sentinel for "not looking at facade"
        else:
            facade_t_cols[col] = t_facade

    return facade_t_cols


def compute_patch_column_facade_t_composite(
    cam_lat, cam_lon, cam_compass_deg, cam_hfov_deg,
    facade_result,
    n_cols=16, camera_type='perspective'
):
    """Compute facade_t for each patch column against a composite facade polyline.

    Like compute_patch_column_facade_t but intersects rays with a multi-segment
    polyline instead of a single edge. facade_t is parameterized as fraction
    of total arc length along the composite facade.

    Args:
        facade_result: dict from find_composite_facade() with 'polyline',
            'segments', 'total_length', 'edges', 'centroid_lat', 'centroid_lon'

    Returns:
        facade_t_cols: (n_cols,) float32 array
    """
    centroid_lat = facade_result['centroid_lat']
    centroid_lon = facade_result['centroid_lon']
    total_length = facade_result['total_length']
    segments = facade_result['segments']
    edges = facade_result['edges']

    if total_length < 0.01:
        return np.full(n_cols, 0.5, dtype=np.float32)

    cos_lat = math.cos(centroid_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    cam_x = (cam_lon - centroid_lon) * m_per_deg_lng
    cam_y = (cam_lat - centroid_lat) * METERS_PER_DEG_LAT

    facade_t_cols = np.zeros(n_cols, dtype=np.float32)

    for col in range(n_cols):
        frac = (col + 0.5) / n_cols
        col_bearing_deg = cam_compass_deg + (frac - 0.5) * cam_hfov_deg

        col_bearing_rad = col_bearing_deg * DEG_TO_RAD
        ray_dx = math.sin(col_bearing_rad)
        ray_dy = math.cos(col_bearing_rad)

        # Find closest intersection across all facade segments
        best_t_ray = float('inf')
        best_facade_t = 0.5

        for seg_start, seg_end, edge_idx in segments:
            e = edges[edge_idx]
            if e['length'] < 0.01:
                continue

            a, b = e['a'], e['b']
            fx = b[0] - a[0]
            fy = b[1] - a[1]

            ax = a[0] - cam_x
            ay = a[1] - cam_y

            denom = ray_dy * fx - ray_dx * fy
            if abs(denom) < 1e-10:
                continue

            t_seg = (ray_dx * ay - ray_dy * ax) / denom
            t_ray = (fx * ay - fy * ax) / denom

            if t_ray < 0:
                continue  # behind camera

            # t_seg should be in [0, 1] for a hit on this segment
            if t_seg < -0.5 or t_seg > 1.5:
                continue  # too far from segment

            if t_ray < best_t_ray:
                best_t_ray = t_ray
                # Convert segment-local t to global facade_t
                arc_pos = seg_start + max(0, min(1, t_seg)) * e['length']
                best_facade_t = arc_pos / total_length

        if best_t_ray == float('inf'):
            facade_t_cols[col] = -1.0  # sentinel: not looking at facade
        else:
            facade_t_cols[col] = best_facade_t

    return facade_t_cols


# ---------------------------------------------------------------------------
# v4: Entrance detection + ray-tracing geometry
# ---------------------------------------------------------------------------

def compute_camera_pose_v4(cam_lat, cam_lon, cam_compass_deg, cam_hfov_deg,
                           building_centroid_lat, building_centroid_lon):
    """Compute 6-dim camera pose relative to building centroid (no facade needed).

    Returns:
        pose_feats: (6,) float32 array
            [0] dx: camera east offset from centroid (normalized)
            [1] dy: camera north offset from centroid (normalized)
            [2] dist: distance to centroid (normalized)
            [3] sin_compass: sin of camera heading
            [4] cos_compass: cos of camera heading
            [5] hfov_norm: HFOV / 360
    """
    cos_lat = math.cos(building_centroid_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    dx = (cam_lon - building_centroid_lon) * m_per_deg_lng
    dy = (cam_lat - building_centroid_lat) * METERS_PER_DEG_LAT
    dist = math.sqrt(dx * dx + dy * dy)

    compass_rad = cam_compass_deg * DEG_TO_RAD

    return np.array([
        dx / 50.0,
        dy / 50.0,
        min(dist / 50.0, 2.0),
        math.sin(compass_rad),
        math.cos(compass_rad),
        cam_hfov_deg / 360.0,
    ], dtype=np.float32)


def entrance_to_column(cam_lat, cam_lon, cam_compass_deg, cam_hfov_deg,
                       entrance_lat, entrance_lon,
                       n_cols=16, camera_type='perspective'):
    """Compute which image column the entrance falls in.

    Uses the same column-to-bearing mapping as compute_patch_column_facade_t:
        bearing = compass + (frac - 0.5) * hfov
        frac = (col + 0.5) / n_cols

    Returns:
        col: float, continuous column position (may be outside [0, n_cols))
        visible: bool, whether entrance is within camera FOV
        dist_m: float, distance from camera to entrance in meters
    """
    cos_lat = math.cos(cam_lat * DEG_TO_RAD)
    dx_m = (entrance_lon - cam_lon) * cos_lat * METERS_PER_DEG_LAT
    dy_m = (entrance_lat - cam_lat) * METERS_PER_DEG_LAT
    dist_m = math.sqrt(dx_m * dx_m + dy_m * dy_m)

    if dist_m < 0.1:
        return n_cols / 2.0, True, dist_m

    # Bearing from camera to entrance (degrees, 0=north, clockwise)
    bearing_deg = math.degrees(math.atan2(dx_m, dy_m))

    # Relative bearing from camera heading
    rel_bearing = bearing_deg - cam_compass_deg
    rel_bearing = (rel_bearing + 180) % 360 - 180  # [-180, 180]

    # Map to column: invert the formula frac = (col+0.5)/n_cols
    # bearing = compass + (frac - 0.5) * hfov
    # => frac = rel_bearing / hfov + 0.5
    if cam_hfov_deg < 1.0:
        return n_cols / 2.0, False, dist_m

    frac = rel_bearing / cam_hfov_deg + 0.5
    col = frac * n_cols - 0.5

    if camera_type == 'equirectangular':
        visible = True
        col = col % n_cols
    else:
        visible = -0.5 <= col < (n_cols - 0.5)

    return col, visible, dist_m


def raytrace_column_to_building(cam_lat, cam_lon, cam_compass_deg, cam_hfov_deg,
                                col, building_coords_lnglat,
                                n_cols=16, camera_type='perspective'):
    """Ray-trace from camera through an image column to find building intersection.

    Args:
        col: float, continuous column position (0 to n_cols-1)
        building_coords_lnglat: list of [lng, lat] polygon vertices

    Returns:
        (lat, lon) of intersection point, or None if ray misses building
    """
    if len(building_coords_lnglat) < 3:
        return None

    # Column to bearing (same formula as compute_patch_column_facade_t)
    frac = (col + 0.5) / n_cols
    bearing_deg = cam_compass_deg + (frac - 0.5) * cam_hfov_deg
    bearing_rad = bearing_deg * DEG_TO_RAD

    ray_dx = math.sin(bearing_rad)  # east component
    ray_dy = math.cos(bearing_rad)  # north component

    # Building centroid for local coordinate system
    cent_lng = sum(c[0] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cent_lat = sum(c[1] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    cam_x = (cam_lon - cent_lng) * m_per_deg_lng
    cam_y = (cam_lat - cent_lat) * METERS_PER_DEG_LAT

    fp_m = [[(c[0] - cent_lng) * m_per_deg_lng,
             (c[1] - cent_lat) * METERS_PER_DEG_LAT]
            for c in building_coords_lnglat]

    best_t_ray = float('inf')
    best_point = None

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        fx, fy = b[0] - a[0], b[1] - a[1]

        ax = a[0] - cam_x
        ay = a[1] - cam_y

        denom = ray_dy * fx - ray_dx * fy
        if abs(denom) < 1e-10:
            continue

        t_seg = (ray_dx * ay - ray_dy * ax) / denom
        t_ray = (fx * ay - fy * ax) / denom

        if t_ray < 0:
            continue  # behind camera
        if t_seg < 0 or t_seg > 1:
            continue  # outside segment

        if t_ray < best_t_ray:
            best_t_ray = t_ray
            ix = cam_x + t_ray * ray_dx
            iy = cam_y + t_ray * ray_dy
            int_lon = cent_lng + ix / m_per_deg_lng
            int_lat = cent_lat + iy / METERS_PER_DEG_LAT
            best_point = (int_lat, int_lon)

    return best_point


def find_facade_edge(building_coords_lnglat, entrance_lat, entrance_lon):
    """Find the single closest building edge to the entrance location.

    Same logic as v3 facade selection.

    Args:
        building_coords_lnglat: list of [lng, lat] polygon vertices
        entrance_lat, entrance_lon: entrance location

    Returns:
        dict with 'a_lnglat', 'b_lnglat', 'a_m', 'b_m', 'length',
        'centroid_lat', 'centroid_lon' or None
    """
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
    best_score = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx = ent_m[0] - mx
        dy = ent_m[1] - my
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < best_score:
            best_score = dist
            best_edge = {
                'a_m': a, 'b_m': b,
                'a_lnglat': building_coords_lnglat[i],
                'b_lnglat': building_coords_lnglat[i + 1],
                'length': length,
                'centroid_lat': cent_lat,
                'centroid_lon': cent_lng,
            }

    return best_edge


def raytrace_column_to_facade(cam_lat, cam_lon, cam_compass_deg, cam_hfov_deg,
                               col, facade_edge, n_cols=16):
    """Ray-trace from camera through an image column to the facade edge.

    Unlike raytrace_column_to_building which tests all building edges,
    this constrains the intersection to the selected facade edge only,
    with a generous tolerance for near-misses.

    Args:
        col: float, continuous column position (0 to n_cols-1)
        facade_edge: dict from find_facade_edge() with 'a_m', 'b_m',
            'a_lnglat', 'b_lnglat', 'centroid_lat', 'centroid_lon'

    Returns:
        (lat, lon) of intersection point on facade, or None if ray misses
    """
    cent_lat = facade_edge['centroid_lat']
    cent_lng = facade_edge['centroid_lon']
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    # Column to bearing
    frac = (col + 0.5) / n_cols
    bearing_deg = cam_compass_deg + (frac - 0.5) * cam_hfov_deg
    bearing_rad = bearing_deg * DEG_TO_RAD

    ray_dx = math.sin(bearing_rad)
    ray_dy = math.cos(bearing_rad)

    cam_x = (cam_lon - cent_lng) * m_per_deg_lng
    cam_y = (cam_lat - cent_lat) * METERS_PER_DEG_LAT

    a = facade_edge['a_m']
    b = facade_edge['b_m']
    fx, fy = b[0] - a[0], b[1] - a[1]

    ax = a[0] - cam_x
    ay = a[1] - cam_y

    denom = ray_dy * fx - ray_dx * fy
    if abs(denom) < 1e-10:
        return None  # ray parallel to facade

    t_seg = (ray_dx * ay - ray_dy * ax) / denom
    t_ray = (fx * ay - fy * ax) / denom

    if t_ray < 0:
        return None  # behind camera

    # Clamp t_seg to [0, 1] — project onto facade even if ray overshoots
    t_seg = max(0.0, min(1.0, t_seg))

    # Intersection point on facade edge
    ix = a[0] + t_seg * fx
    iy = a[1] + t_seg * fy
    int_lon = cent_lng + ix / m_per_deg_lng
    int_lat = cent_lat + iy / METERS_PER_DEG_LAT

    return (int_lat, int_lon)
