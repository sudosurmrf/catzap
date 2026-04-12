import pytest
from server.vision.zone_checker import check_zone_violations, invalidate_poly_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the poly cache between tests so zones with reused ids don't leak
    stale Shapely Polygon objects across test boundaries."""
    invalidate_poly_cache()
    yield
    invalidate_poly_cache()


def _zone(zone_id: str, polygon: list, overlap_threshold: float = 0.3, enabled: bool = True, name: str = "Z"):
    return {
        "id": zone_id,
        "name": name,
        "polygon": polygon,
        "overlap_threshold": overlap_threshold,
        "enabled": enabled,
    }


def test_bbox_fully_inside_zone_is_violation():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]])
    violations = check_zone_violations([10, 10, 50, 50], [zone])
    assert len(violations) == 1
    assert violations[0]["zone_id"] == "z1"
    assert violations[0]["overlap"] == pytest.approx(1.0)


def test_bbox_fully_outside_zone_is_not_violation():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]])
    violations = check_zone_violations([200, 200, 250, 250], [zone])
    assert violations == []


def test_bbox_partial_overlap_above_threshold():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]], overlap_threshold=0.25)
    # Bbox is 100x100 starting at (50, 50), so a quarter of it overlaps the zone.
    violations = check_zone_violations([50, 50, 150, 150], [zone])
    assert len(violations) == 1
    assert violations[0]["overlap"] == pytest.approx(0.25, abs=0.01)


def test_bbox_partial_overlap_below_threshold():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]], overlap_threshold=0.5)
    # Same 25% overlap as above, but the threshold is 0.5 this time.
    violations = check_zone_violations([50, 50, 150, 150], [zone])
    assert violations == []


def test_disabled_zone_is_ignored():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]], enabled=False)
    violations = check_zone_violations([10, 10, 50, 50], [zone])
    assert violations == []


def test_zero_area_bbox_returns_empty():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]])
    violations = check_zone_violations([10, 10, 10, 10], [zone])
    assert violations == []


def test_degenerate_polygon_is_skipped():
    zone = _zone("z1", [[0, 0], [1, 0]])  # only 2 points, not a polygon
    violations = check_zone_violations([10, 10, 50, 50], [zone])
    assert violations == []


def test_multiple_zones_each_checked_independently():
    zone_a = _zone("a", [[0, 0], [0, 50], [50, 50], [50, 0]], name="A")
    zone_b = _zone("b", [[100, 100], [100, 150], [150, 150], [150, 100]], name="B")
    violations = check_zone_violations([10, 10, 40, 40], [zone_a, zone_b])
    assert len(violations) == 1
    assert violations[0]["zone_id"] == "a"


def test_self_intersecting_polygon_is_rebuilt_via_make_valid():
    """Shapely's make_valid() is called when a raw polygon is invalid. The
    zone checker must not crash on a bowtie or figure-eight outline."""
    bowtie = [[0, 0], [100, 100], [100, 0], [0, 100]]
    zone = _zone("z1", bowtie)
    # Call should not raise; whether there's a violation depends on how
    # make_valid rebuilds the polygon, so we only assert the call succeeds.
    check_zone_violations([10, 10, 50, 50], [zone])
