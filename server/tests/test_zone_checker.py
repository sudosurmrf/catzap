from server.vision.zone_checker import check_zone_violations


def test_cat_fully_inside_zone():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6]],
            "overlap_threshold": 0.3,
            "enabled": True,
        }
    ]
    bbox = [0.2, 0.2, 0.4, 0.4]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 1
    assert violations[0]["zone_name"] == "Kitchen Counter"
    assert violations[0]["overlap"] > 0.9


def test_cat_partially_inside_zone():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
            "overlap_threshold": 0.3,
            "enabled": True,
        }
    ]
    bbox = [0.15, 0.0, 0.45, 0.3]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 1
    assert 0.4 < violations[0]["overlap"] < 0.6


def test_cat_outside_zone():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]],
            "overlap_threshold": 0.3,
            "enabled": True,
        }
    ]
    bbox = [0.5, 0.5, 0.7, 0.7]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 0


def test_overlap_below_threshold():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [0.25, 0.0], [0.25, 1.0], [0.0, 1.0]],
            "overlap_threshold": 0.5,
            "enabled": True,
        }
    ]
    bbox = [0.15, 0.0, 0.65, 1.0]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 0


def test_disabled_zone_ignored():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "overlap_threshold": 0.1,
            "enabled": False,
        }
    ]
    bbox = [0.2, 0.2, 0.4, 0.4]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 0
