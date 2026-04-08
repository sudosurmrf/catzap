import math


def angle_depth_to_room(pan: float, tilt: float, depth_cm: float, camera_pos: tuple[float, float, float]) -> tuple[float, float, float]:
    pan_rad = math.radians(pan - 90.0)
    tilt_rad = math.radians(tilt)
    horizontal_dist = depth_cm * math.cos(tilt_rad)
    x = camera_pos[0] + horizontal_dist * math.sin(pan_rad)
    y = camera_pos[1] + horizontal_dist * math.cos(pan_rad)
    z = camera_pos[2] - depth_cm * math.sin(tilt_rad)
    return (x, y, z)


def room_to_angle(x: float, y: float, z: float, camera_pos: tuple[float, float, float]) -> tuple[float, float]:
    dx = x - camera_pos[0]
    dy = y - camera_pos[1]
    dz = camera_pos[2] - z
    horizontal_dist = math.sqrt(dx * dx + dy * dy)
    pan_rad = math.atan2(dx, dy)
    pan = math.degrees(pan_rad) + 90.0
    tilt_rad = math.atan2(dz, horizontal_dist) if horizontal_dist > 0 else (math.pi / 2 if dz > 0 else 0)
    tilt = math.degrees(tilt_rad)
    return (pan, tilt)


def project_furniture_to_angles(base_polygon: list[tuple[float, float]], height_min: float, height_max: float, camera_pos: tuple[float, float, float]) -> list[tuple[float, float]]:
    angle_points = []
    for x, y in base_polygon:
        pan_lo, tilt_lo = room_to_angle(x, y, height_min, camera_pos)
        angle_points.append((pan_lo, tilt_lo))
        pan_hi, tilt_hi = room_to_angle(x, y, height_max, camera_pos)
        angle_points.append((pan_hi, tilt_hi))
    return angle_points
