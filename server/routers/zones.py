from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from server.models.database import create_zone, get_zones, update_zone, delete_zone, invalidate_zone_cache
from server.vision.zone_checker import invalidate_poly_cache

router = APIRouter(prefix="/api/zones", tags=["zones"])


class ZoneCreate(BaseModel):
    name: str
    polygon: list[list[float]]
    overlap_threshold: float = 0.3
    enabled: bool = True


class ZoneUpdate(BaseModel):
    name: str | None = None
    polygon: list[list[float]] | None = None
    overlap_threshold: float | None = None
    enabled: bool | None = None


@router.post("", status_code=201)
async def create_zone_endpoint(zone: ZoneCreate):
    zone_id = await create_zone(
        name=zone.name,
        polygon=zone.polygon,
        overlap_threshold=zone.overlap_threshold,
        enabled=zone.enabled,
    )
    invalidate_zone_cache()
    invalidate_poly_cache()
    zones = await get_zones()
    return next(z for z in zones if z["id"] == zone_id)


@router.get("")
async def get_zones_endpoint():
    return await get_zones()


@router.put("/{zone_id}")
async def update_zone_endpoint(zone_id: str, zone: ZoneUpdate):
    updates = zone.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    success = await update_zone(zone_id, **updates)
    if not success:
        raise HTTPException(status_code=404, detail="Zone not found")
    invalidate_zone_cache()
    invalidate_poly_cache()
    zones = await get_zones()
    return next((z for z in zones if z["id"] == zone_id), None)


@router.delete("/{zone_id}")
async def delete_zone_endpoint(zone_id: str):
    success = await delete_zone(zone_id)
    if not success:
        raise HTTPException(status_code=404, detail="Zone not found")
    invalidate_zone_cache()
    invalidate_poly_cache()
    return {"deleted": True}
