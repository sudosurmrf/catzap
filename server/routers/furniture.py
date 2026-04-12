from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from server.models.database import create_furniture, get_furniture, delete_furniture

router = APIRouter(prefix="/api/furniture", tags=["furniture"])


class FurnitureCreate(BaseModel):
    name: str
    polygon: list[list[float]]


class FurnitureResponse(BaseModel):
    id: str
    name: str
    polygon: list[list[float]]
    created_at: str


@router.post("", status_code=201)
async def create_furniture_endpoint(furniture: FurnitureCreate):
    furniture_id = await create_furniture(
        name=furniture.name,
        polygon=furniture.polygon,
    )
    rows = await get_furniture()
    return next(r for r in rows if r["id"] == furniture_id)


@router.get("", response_model=list[FurnitureResponse])
async def get_furniture_endpoint():
    return await get_furniture()


@router.delete("/{furniture_id}")
async def delete_furniture_endpoint(furniture_id: str):
    success = await delete_furniture(furniture_id)
    if not success:
        raise HTTPException(status_code=404, detail="Furniture not found")
    return {"deleted": True}
