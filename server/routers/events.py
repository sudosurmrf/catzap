from fastapi import APIRouter

from server.models.database import get_events

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("")
async def get_events_endpoint(
    type: str | None = None,
    cat_name: str | None = None,
    zone_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    return await get_events(
        event_type=type,
        cat_name=cat_name,
        zone_name=zone_name,
        limit=limit,
        offset=offset,
    )
