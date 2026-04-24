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

