from shapely.geometry import Polygon, Point, box
from shapely.validation import make_valid

# ── Pre-cached Shapely Polygon objects ──
# Keyed by zone id -> (polygon, room_polygon).  Rebuilt when zone data changes.
_poly_cache: dict[str, dict] = {}


def _get_cached_poly(zone: dict, key: str, coords_key: str) -> Polygon | None:
    """Return a cached, validated Shapely Polygon for a zone field."""
    zone_id = zone["id"]
    cached = _poly_cache.get(zone_id)
    if cached is not None and key in cached:
        return cached[key]

    coords = zone.get(coords_key)
    if not coords or len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return None

    if cached is None:
        _poly_cache[zone_id] = {}
    _poly_cache[zone_id][key] = poly
    return poly


def invalidate_poly_cache():
    """Called when zones are modified to clear cached Polygon objects."""
    _poly_cache.clear()


def check_3d_zone_violation(cat_room_pos: tuple[float, float, float], zone: dict) -> bool:
    x, y, z = cat_room_pos
    height_min = zone.get("height_min", 0.0)
    height_max = zone.get("height_max", 0.0)
    if z < height_min or z > height_max:
        return False
    poly = _get_cached_poly(zone, "room_poly", "room_polygon")
    if poly is None:
        return False
    return poly.contains(Point(x, y))


def check_zone_violations(
    bbox: list[float], zones: list[dict],
    cat_room_pos: tuple[float, float, float] | None = None,
) -> list[dict]:
    cat_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
    cat_area = cat_box.area
    if cat_area == 0:
        return []
    violations = []
    for zone in zones:
        if not zone.get("enabled", True):
            continue
        mode = zone.get("mode", "2d")
        # Try 3D check first if depth data is available
        if mode in ("auto_3d", "manual_3d") and cat_room_pos is not None:
            if check_3d_zone_violation(cat_room_pos, zone):
                violations.append({"zone_id": zone["id"], "zone_name": zone["name"], "overlap": 1.0})
                continue
            # 3D check failed — fall through to 2D overlap as safety net
        zone_poly = _get_cached_poly(zone, "angle_poly", "polygon")
        if zone_poly is None or zone_poly.area == 0:
            continue
        intersection = cat_box.intersection(zone_poly)
        overlap = intersection.area / cat_area
        if overlap >= zone.get("overlap_threshold", 0.3):
            violations.append({"zone_id": zone["id"], "zone_name": zone["name"], "overlap": overlap})
    return violations
