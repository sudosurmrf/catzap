from server.actuator.calibration import CalibrationMap


def test_single_calibration_point():
    cal = CalibrationMap()
    cal.add_point(pixel_x=0.5, pixel_y=0.5, pan_angle=90.0, tilt_angle=45.0)
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert pan == 90.0
    assert tilt == 45.0


def test_interpolation_between_points():
    cal = CalibrationMap()
    cal.add_point(pixel_x=0.0, pixel_y=0.5, pan_angle=0.0, tilt_angle=45.0)
    cal.add_point(pixel_x=1.0, pixel_y=0.5, pan_angle=180.0, tilt_angle=45.0)
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert abs(pan - 90.0) < 5.0
    assert abs(tilt - 45.0) < 5.0


def test_four_corner_calibration():
    cal = CalibrationMap()
    cal.add_point(0.0, 0.0, pan_angle=30.0, tilt_angle=60.0)
    cal.add_point(1.0, 0.0, pan_angle=150.0, tilt_angle=60.0)
    cal.add_point(0.0, 1.0, pan_angle=30.0, tilt_angle=20.0)
    cal.add_point(1.0, 1.0, pan_angle=150.0, tilt_angle=20.0)
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert 80.0 < pan < 100.0
    assert 35.0 < tilt < 50.0


def test_no_calibration_returns_default():
    cal = CalibrationMap()
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert pan == 90.0
    assert tilt == 90.0


def test_clear_points():
    cal = CalibrationMap()
    cal.add_point(0.5, 0.5, 90.0, 45.0)
    cal.clear()
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert pan == 90.0
    assert tilt == 90.0


def test_serialization():
    cal = CalibrationMap()
    cal.add_point(0.0, 0.0, 30.0, 60.0)
    cal.add_point(1.0, 1.0, 150.0, 20.0)
    data = cal.to_dict()
    cal2 = CalibrationMap.from_dict(data)
    pan, tilt = cal2.pixel_to_angle(0.0, 0.0)
    assert abs(pan - 30.0) < 1.0
    assert abs(tilt - 60.0) < 1.0
