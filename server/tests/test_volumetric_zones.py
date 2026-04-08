from server.vision.zone_checker import check_zone_violations, check_3d_zone_violation


def test_2d_zone_unchanged():
    bbox = [0.3, 0.3, 0.7, 0.7]
    zones = [{"id": "z1", "name": "couch", "enabled": True,
        "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "overlap_threshold": 0.3, "mode": "2d"}]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 1
    assert violations[0]["zone_name"] == "couch"


def test_3d_zone_cat_inside_volume():
    zone = {"id": "z1", "name": "table", "enabled": True, "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0, "height_max": 75.0}
    assert check_3d_zone_violation((150.0, 150.0, 50.0), zone) is True


def test_3d_zone_cat_on_top():
    zone = {"id": "z1", "name": "table", "enabled": True, "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0, "height_max": 75.0}
    assert check_3d_zone_violation((150.0, 150.0, 80.0), zone) is False


def test_3d_zone_cat_on_table_surface():
    zone = {"id": "z1", "name": "table-top", "enabled": True, "mode": "manual_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 70.0, "height_max": 110.0}
    assert check_3d_zone_violation((150.0, 150.0, 75.0), zone) is True


def test_3d_zone_cat_outside_polygon():
    zone = {"id": "z1", "name": "table", "enabled": True, "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0, "height_max": 75.0}
    assert check_3d_zone_violation((50.0, 50.0, 50.0), zone) is False


def test_2d_fallback_when_no_room_pos():
    bbox = [0.3, 0.3, 0.7, 0.7]
    zones = [{"id": "z1", "name": "table", "enabled": True,
        "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "overlap_threshold": 0.3, "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0, "height_max": 75.0}]
    violations = check_zone_violations(bbox, zones, cat_room_pos=None)
    assert len(violations) == 1
