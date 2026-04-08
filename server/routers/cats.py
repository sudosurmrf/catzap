from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from server.models.database import create_cat, get_cats, delete_cat

router = APIRouter(prefix="/api/cats", tags=["cats"])


class CatCreate(BaseModel):
    name: str


@router.post("", status_code=201)
async def create_cat_endpoint(cat: CatCreate):
    cat_id = await create_cat(name=cat.name)
    cats = await get_cats()
    return next(c for c in cats if c["id"] == cat_id)


@router.get("")
async def get_cats_endpoint():
    return await get_cats()


@router.delete("/{cat_id}")
async def delete_cat_endpoint(cat_id: str):
    success = await delete_cat(cat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Cat not found")
    return {"deleted": True}
