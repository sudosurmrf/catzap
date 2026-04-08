import logging

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from server.models.database import create_furniture, get_furniture, update_furniture, delete_furniture

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spatial", tags=["spatial"])


class FurnitureCreate(BaseModel):
    name: str
    base_polygon: list[list[float]]
    height_min: float = 0.0
    height_max: float = 0.0
    depth_anchored: bool = False


class FurnitureUpdate(BaseModel):
    name: str | None = None
    base_polygon: list[list[float]] | None = None
    height_min: float | None = None
    height_max: float | None = None
    depth_anchored: bool | None = None


class EstimateHeightRequest(BaseModel):
    polygon: list[list[float]]  # angle-space polygon [[pan, tilt], ...]


@router.post("/furniture", status_code=201)
async def create_furniture_endpoint(body: FurnitureCreate):
    fid = await create_furniture(
        name=body.name, base_polygon=body.base_polygon,
        height_min=body.height_min, height_max=body.height_max,
        depth_anchored=body.depth_anchored,
    )
    items = await get_furniture()
    return next(f for f in items if f["id"] == fid)


@router.get("/furniture")
async def get_furniture_endpoint():
    return await get_furniture()


@router.put("/furniture/{furniture_id}")
async def update_furniture_endpoint(furniture_id: str, body: FurnitureUpdate):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    success = await update_furniture(furniture_id, **updates)
    if not success:
        raise HTTPException(status_code=404, detail="Furniture not found")

    # Also update in-memory room model
    from server.main import get_room_model
    rm = get_room_model()
    if rm:
        for fobj in rm.furniture:
            if fobj.id == furniture_id:
                if "name" in updates:
                    fobj.name = updates["name"]
                if "base_polygon" in updates:
                    fobj.base_polygon = [tuple(p) for p in updates["base_polygon"]]
                if "height_min" in updates:
                    fobj.height_min = updates["height_min"]
                if "height_max" in updates:
                    fobj.height_max = updates["height_max"]
                break

    items = await get_furniture()
    return next((f for f in items if f["id"] == furniture_id), None)


@router.delete("/furniture/{furniture_id}")
async def delete_furniture_endpoint(furniture_id: str):
    success = await delete_furniture(furniture_id)
    if not success:
        raise HTTPException(status_code=404, detail="Furniture not found")
    return {"deleted": True}


@router.post("/estimate-height")
async def estimate_height_endpoint(body: EstimateHeightRequest):
    """Estimate furniture height at a polygon region using the latest depth map.

    Takes an angle-space polygon and samples the depth within that region
    to estimate min/max height of whatever is there.
    """
    from server.main import get_room_model
    from server.spatial.depth_estimator import DepthEstimator
    import numpy as np

    rm = get_room_model()
    if not rm:
        raise HTTPException(status_code=400, detail="Room model not initialized")

    # Access the depth estimator and latest frame from main
    from server.main import _depth_estimator
    if not _depth_estimator:
        raise HTTPException(status_code=400, detail="Depth estimator not available")

    # Get the latest depth + frame from the stream
    from server.routers.stream import get_latest_frame
    frame_data = get_latest_frame()
    if frame_data is None:
        raise HTTPException(status_code=400, detail="No frame available yet")

    frame = frame_data["frame"]
    servo_pan = frame_data["servo_pan"]
    servo_tilt = frame_data["servo_tilt"]

    try:
        depth_map = _depth_estimator.estimate(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {e}")

    fov_h = 65.0
    fov_v = 50.0
    h, w = frame.shape[:2]

    # Convert angle-space polygon to pixel mask
    from server.panorama.angle_math import angle_to_pixel
    pixel_points = []
    for pan, tilt in body.polygon:
        nx, ny = angle_to_pixel(pan, tilt, servo_pan, servo_tilt, fov_h, fov_v)
        pixel_points.append((int(nx * w), int(ny * h)))

    # Create mask from polygon
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(pixel_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Sample depth values within the polygon
    masked_depth = depth_map[mask > 0]
    if len(masked_depth) == 0:
        return {"height_min": 0, "height_max": 75, "estimated": False, "reason": "No depth samples in polygon"}

    # Filter out zero/invalid depth
    valid = masked_depth[masked_depth > 0]
    if len(valid) == 0:
        return {"height_min": 0, "height_max": 75, "estimated": False, "reason": "No valid depth readings"}

    # Convert relative depth to metric using depth_scale
    from server.config import settings
    metric_depths = _depth_estimator.depth_scale / valid

    # Use percentiles to reject outliers
    near_dist = float(np.percentile(metric_depths, 10))  # closest points (top of object)
    far_dist = float(np.percentile(metric_depths, 90))  # furthest points (base)

    # Convert distances to heights: camera is at camera_height_cm, looking down
    # Height = camera_height - vertical_component_of_distance
    # For a rough estimate: use the tilt angle to decompose
    avg_tilt = sum(t for _, t in body.polygon) / len(body.polygon)
    import math
    tilt_rad = math.radians(avg_tilt)

    # Height of near point (top of furniture) and far point (floor near furniture)
    height_near = settings.camera_height_cm - near_dist * math.sin(tilt_rad)
    height_far = settings.camera_height_cm - far_dist * math.sin(tilt_rad)

    height_min = max(0, min(height_near, height_far))
    height_max = max(height_near, height_far)
    height_max = max(height_max, height_min + 10)  # at least 10cm tall

    return {
        "height_min": round(height_min, 1),
        "height_max": round(height_max, 1),
        "estimated": True,
        "sample_count": len(valid),
    }


@router.get("/room-model/status")
async def room_model_status():
    from server.main import get_room_model
    rm = get_room_model()
    if not rm:
        return {"initialized": False}
    return {
        "initialized": True,
        "width_cm": rm.width_cm,
        "depth_cm": rm.depth_cm,
        "furniture_count": len(rm.furniture),
        "depth_scale": rm.depth_scale,
    }


@router.post("/calibrate-scale")
async def calibrate_depth_scale(body: dict):
    from server.main import get_room_model
    rm = get_room_model()
    if not rm:
        raise HTTPException(status_code=400, detail="Room model not initialized")
    rm.depth_scale = body.get("real_distance_cm", 100.0)
    return {"depth_scale": rm.depth_scale}
