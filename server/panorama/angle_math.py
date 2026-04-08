from server.config import settings


def pixel_to_angle(
    pixel_x: float,
    pixel_y: float,
    servo_pan: float,
    servo_tilt: float,
    fov_h: float | None = None,
    fov_v: float | None = None,
) -> tuple[float, float]:
    """Convert normalized pixel coords (0-1) to angle-space using current servo position."""
    fov_h = fov_h or settings.fov_horizontal
    fov_v = fov_v or settings.fov_vertical
    pan = servo_pan - (pixel_x - 0.5) * fov_h
    tilt = servo_tilt + (pixel_y - 0.5) * fov_v
    return (pan, tilt)


def angle_to_pixel(
    pan: float,
    tilt: float,
    servo_pan: float,
    servo_tilt: float,
    fov_h: float | None = None,
    fov_v: float | None = None,
) -> tuple[float, float]:
    """Convert angle-space coords to normalized pixel coords for current servo position."""
    fov_h = fov_h or settings.fov_horizontal
    fov_v = fov_v or settings.fov_vertical
    pixel_x = 0.5 - (pan - servo_pan) / fov_h
    pixel_y = 0.5 + (tilt - servo_tilt) / fov_v
    return (pixel_x, pixel_y)


def is_angle_in_fov(
    pan: float,
    tilt: float,
    servo_pan: float,
    servo_tilt: float,
    fov_h: float | None = None,
    fov_v: float | None = None,
) -> bool:
    """Check if an angle-space point is within the current camera FOV."""
    fov_h = fov_h or settings.fov_horizontal
    fov_v = fov_v or settings.fov_vertical
    half_h = fov_h / 2
    half_v = fov_v / 2
    return (
        servo_pan - half_h <= pan <= servo_pan + half_h
        and servo_tilt - half_v <= tilt <= servo_tilt + half_v
    )
