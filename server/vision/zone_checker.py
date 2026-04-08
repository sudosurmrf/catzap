from shapely.geometry import Polygon, box
from shapely.validation import make_valid


def check_zone_violations(
    bbox: list[float], zones: list[dict]
) -> list[dict]:
    """Check if a cat bounding box violates any zones.

    Args:
        bbox: [x1, y1, x2, y2] normalized coordinates (0-1).
        zones: List of zone dicts with polygon, overlap_threshold, enabled.

    Returns:
        List of violated zone dicts with added 'overlap' field.
    """
    cat_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
    cat_area = cat_box.area

    if cat_area == 0:
        return []

    violations = []
    for zone in zones:
        if not zone.get("enabled", True):
            continue

        zone_poly = Polygon(zone["polygon"])
        # Freehand-drawn polygons can self-intersect — repair them
        if not zone_poly.is_valid:
            zone_poly = make_valid(zone_poly)
        if zone_poly.is_empty or zone_poly.area == 0:
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
