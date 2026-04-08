from server.panorama.angle_math import pixel_to_angle, angle_to_pixel, is_angle_in_fov


def test_pixel_center_maps_to_servo_angle():
    pan, tilt = pixel_to_angle(0.5, 0.5, servo_pan=90.0, servo_tilt=45.0)
    assert pan == 90.0
    assert tilt == 45.0


def test_pixel_right_edge_maps_to_higher_pan():
    pan, tilt = pixel_to_angle(1.0, 0.5, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0)
    assert abs(pan - 122.5) < 0.1  # 90 + 0.5 * 65


def test_pixel_left_edge_maps_to_lower_pan():
    pan, tilt = pixel_to_angle(0.0, 0.5, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0)
    assert abs(pan - 57.5) < 0.1  # 90 - 0.5 * 65


def test_pixel_top_maps_to_lower_tilt():
    pan, tilt = pixel_to_angle(0.5, 0.0, servo_pan=90.0, servo_tilt=45.0, fov_v=50.0)
    assert abs(tilt - 20.0) < 0.1  # 45 - 0.5 * 50


def test_angle_to_pixel_center():
    px, py = angle_to_pixel(90.0, 45.0, servo_pan=90.0, servo_tilt=45.0)
    assert abs(px - 0.5) < 0.01
    assert abs(py - 0.5) < 0.01


def test_angle_to_pixel_offset():
    px, py = angle_to_pixel(122.5, 45.0, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0)
    assert abs(px - 1.0) < 0.01


def test_roundtrip_pixel_angle_pixel():
    pan, tilt = pixel_to_angle(0.3, 0.7, servo_pan=60.0, servo_tilt=40.0)
    px, py = angle_to_pixel(pan, tilt, servo_pan=60.0, servo_tilt=40.0)
    assert abs(px - 0.3) < 0.01
    assert abs(py - 0.7) < 0.01


def test_is_angle_in_fov_center():
    assert is_angle_in_fov(90.0, 45.0, servo_pan=90.0, servo_tilt=45.0) == True


def test_is_angle_in_fov_outside():
    assert is_angle_in_fov(10.0, 45.0, servo_pan=90.0, servo_tilt=45.0) == False


def test_is_angle_in_fov_edge():
    # Just inside the edge
    assert is_angle_in_fov(122.0, 45.0, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0) == True
    # Just outside
    assert is_angle_in_fov(123.0, 45.0, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0) == False
