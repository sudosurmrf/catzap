import asyncio
import logging

from fastapi import APIRouter, HTTPException

from server.config import settings
from server.models.database import get_cats, get_cat_photos, get_cat_photo_counts
from server.vision.trainer import train_classifier, training_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classifier", tags=["classifier"])

WEIGHTS_PATH = settings.classifier_weights_dir / "cat_classifier.pt"
MIN_PHOTOS_PER_CAT = 10


@router.post("/train")
async def start_training():
    """Kick off classifier training as a background task."""
    if training_status.state == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    cats = await get_cats()
    if not cats:
        raise HTTPException(status_code=400, detail="No cats registered")

    photo_counts = await get_cat_photo_counts()
    for cat in cats:
        count = photo_counts.get(cat["id"], 0)
        if count < MIN_PHOTOS_PER_CAT:
            raise HTTPException(
                status_code=400,
                detail=f"{cat['name']} has {count} photos (need {MIN_PHOTOS_PER_CAT})",
            )

    cat_photo_map: dict[str, list[str]] = {}
    for cat in cats:
        photos = await get_cat_photos(cat["id"])
        cat_photo_map[cat["name"]] = [
            str(settings.cat_photos_dir / p["file_path"]) for p in photos
        ]

    unknown_photos: list[str] = []

    async def _train_and_reload():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            train_classifier,
            cat_photo_map,
            unknown_photos,
            WEIGHTS_PATH,
        )
        from server.main import reload_classifier
        reload_classifier()

    asyncio.create_task(_train_and_reload())

    return {"status": "training_started"}


@router.get("/status")
async def get_training_status():
    return {
        "state": training_status.state,
        "progress": training_status.progress,
        "accuracy": training_status.accuracy,
        "error": training_status.error,
    }


@router.get("/info")
async def get_classifier_info():
    photo_counts = await get_cat_photo_counts()
    cats = await get_cats()

    per_cat = []
    for cat in cats:
        per_cat.append({
            "name": cat["name"],
            "photo_count": photo_counts.get(cat["id"], 0),
        })

    return {
        "model_exists": WEIGHTS_PATH.exists(),
        "weights_path": str(WEIGHTS_PATH),
        "per_cat": per_cat,
        "min_photos_required": MIN_PHOTOS_PER_CAT,
    }
