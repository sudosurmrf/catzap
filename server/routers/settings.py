from pydantic import BaseModel
from fastapi import APIRouter

from server.models.database import get_setting, set_setting

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingUpdate(BaseModel):
    key: str
    value: str | int | float | bool | list | dict


@router.get("")
async def get_all_settings():
    keys = [
        "confidence_threshold",
        "overlap_threshold",
        "arm_schedule",
        "notification_webhook_url",
        "frame_skip_n",
        "esp32_cam_url",
        "esp32_actuator_url",
    ]
    result = {}
    for key in keys:
        result[key] = await get_setting(key)
    return result


@router.get("/{key}")
async def get_setting_endpoint(key: str):
    value = await get_setting(key)
    return {"key": key, "value": value}


@router.put("/{key}")
async def set_setting_endpoint(key: str, update: SettingUpdate):
    await set_setting(key, update.value)
    return {"key": key, "value": update.value}
