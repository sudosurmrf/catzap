import base64
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from server.config import settings
from server.models.database import (
    create_cat_photo, get_cat_photos, delete_cat_photo, get_cats,
)

router = APIRouter(prefix="/api/cats", tags=["photos"])


class CaptureRequest(BaseModel):
    frame_base64: str
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1


def _crop_with_padding(frame: np.ndarray, bbox: list[float], pad_pct: float = 0.2) -> np.ndarray:
    """Crop bbox region from frame with padding, preserving aspect ratio."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, x1 - bw * pad_pct)
    y1 = max(0, y1 - bh * pad_pct)
    x2 = min(1, x2 + bw * pad_pct)
    y2 = min(1, y2 + bh * pad_pct)
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    return frame[py1:py2, px1:px2]


def _save_photo(cat_id: str, photo_id: str, image: np.ndarray) -> str:
    """Save image to disk and return relative path."""
    photo_dir = settings.cat_photos_dir / cat_id
    photo_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{cat_id}/{photo_id}.jpg"
    abs_path = settings.cat_photos_dir / rel_path
    cv2.imwrite(str(abs_path), image)
    return rel_path


@router.post("/{cat_id}/photos/capture", status_code=201)
async def capture_photo(cat_id: str, req: CaptureRequest):
    cats = await get_cats()
    if not any(c["id"] == cat_id for c in cats):
        raise HTTPException(status_code=404, detail="Cat not found")

    frame_bytes = base64.b64decode(req.frame_base64)
    frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame data")

    crop = _crop_with_padding(frame, req.bbox)
    if crop.size == 0:
        raise HTTPException(status_code=400, detail="Empty crop region")

    photo_id = str(uuid.uuid4())
    rel_path = _save_photo(cat_id, photo_id, crop)
    db_id = await create_cat_photo(cat_id=cat_id, file_path=rel_path, source="capture")
    return {"id": db_id, "file_path": rel_path}


@router.post("/{cat_id}/photos/upload", status_code=201)
async def upload_photos(cat_id: str, files: list[UploadFile] = File(...)):
    cats = await get_cats()
    if not any(c["id"] == cat_id for c in cats):
        raise HTTPException(status_code=404, detail="Cat not found")

    results = []
    for f in files:
        contents = await f.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        photo_id = str(uuid.uuid4())
        rel_path = _save_photo(cat_id, photo_id, image)
        db_id = await create_cat_photo(cat_id=cat_id, file_path=rel_path, source="upload")
        results.append({"id": db_id, "file_path": rel_path})

    return results


@router.get("/{cat_id}/photos")
async def list_photos(cat_id: str):
    return await get_cat_photos(cat_id)


@router.delete("/{cat_id}/photos/{photo_id}")
async def remove_photo(cat_id: str, photo_id: str):
    file_path = await delete_cat_photo(photo_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Photo not found")
    abs_path = settings.cat_photos_dir / file_path
    if abs_path.exists():
        abs_path.unlink()
    return {"deleted": True}
