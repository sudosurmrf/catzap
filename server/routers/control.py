from pydantic import BaseModel
from fastapi import APIRouter

from server.actuator.calibration import CalibrationMap
from server.actuator.client import ActuatorClient
from server.config import settings

router = APIRouter(prefix="/api/control", tags=["control"])

_armed = True
_calibration = CalibrationMap()
_actuator = ActuatorClient(base_url=settings.esp32_actuator_url)


class ArmRequest(BaseModel):
    armed: bool


class ManualFireRequest(BaseModel):
    pan: float
    tilt: float
    duration_ms: int = 200


class CalibrationPointRequest(BaseModel):
    pixel_x: float
    pixel_y: float
    pan_angle: float
    tilt_angle: float


def get_armed() -> bool:
    return _armed


def get_calibration() -> CalibrationMap:
    return _calibration


def get_actuator() -> ActuatorClient:
    return _actuator


@router.get("/status")
async def get_status():
    return {
        "armed": _armed,
        "calibration_points": len(_calibration._points),
    }


@router.post("/arm")
async def set_arm(request: ArmRequest):
    global _armed
    _armed = request.armed
    return {"armed": _armed}


@router.post("/fire")
async def manual_fire(request: ManualFireRequest):
    success = await _actuator.aim_and_fire(
        pan=request.pan, tilt=request.tilt, duration_ms=request.duration_ms
    )
    return {"fired": success, "pan": request.pan, "tilt": request.tilt}


@router.post("/calibrate")
async def add_calibration_point(request: CalibrationPointRequest):
    _calibration.add_point(
        pixel_x=request.pixel_x,
        pixel_y=request.pixel_y,
        pan_angle=request.pan_angle,
        tilt_angle=request.tilt_angle,
    )
    return {"points_count": len(_calibration._points)}


@router.delete("/calibrate")
async def clear_calibration():
    _calibration.clear()
    return {"points_count": 0}
