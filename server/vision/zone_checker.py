from shapely.geometry import Polygon, box
from shapely.validation import make_valid

# ── Pre-cached Shapely Polygon objects ──
# Keyed by zone id -> Polygon. Rebuilt when zone data changes via
# invalidate_poly_cache().
_poly_cache: dict[str, Polygon] = {}


def invalidate_poly_cache() -> None:
    """Called when zones are modified so the next overlap check rebuilds
    Shapely polygons from the new coordinate data."""
    _poly_cache.clear()


def _get_cached_poly(zone: dict) -> Polygon | None:
    """Return a validated Shapely Polygon for a zone, building and caching
    it on first access. Degenerate or empty polygons return None so the
    caller can skip them without raising."""
    zone_id = zone["id"]
    cached = _poly_cache.get(zone_id)
    if cached is not None:
        return cached
    coords = zone.get("polygon")
    if not coords or len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return None
    _poly_cache[zone_id] = poly
    return poly


def check_zone_violations(bbox: list[float], zones: list[dict]) -> list[dict]:
    """Return the list of zones whose overlap with the given bbox meets or
    exceeds the zone's `overlap_threshold`. Each violation is a dict with
    `zone_id`, `zone_name`, and `overlap` (fraction of bbox area inside zone).

    Zones marked `enabled=False` are skipped. Degenerate bboxes (zero area)
    return an empty list.
    """
    cat_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
    cat_area = cat_box.area
    if cat_area == 0:
        return []
    violations = []
    for zone in zones:
        if not zone.get("enabled", True):
            continue
        zone_poly = _get_cached_poly(zone)
        if zone_poly is None or zone_poly.area == 0:
            continue
        intersection = cat_box.intersection(zone_poly)
        overlap = intersection.area / cat_area
        if overlap >= zone.get("overlap_threshold", 0.3):
            violations.append({
                "zone_id": zone["id"],
                "zone_name": zone["name"],
                "overlap": overlap,
            })
    return violations
