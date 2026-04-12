from pydantic import BaseModel
from fastapi import APIRouter

from server.actuator.client import snap_to_servo_step
from server.config import settings
from server.panorama.sweep_controller import SweepState

router = APIRouter(prefix="/api/control", tags=["control"])

_armed = True


class ArmRequest(BaseModel):
    armed: bool


class ManualFireRequest(BaseModel):
    pan: float
    tilt: float
    duration_ms: int = 200


class VirtualAngleRequest(BaseModel):
    pan: float
    tilt: float


def get_armed() -> bool:
    return _armed


@router.get("/status")
async def get_status():
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    # Report physical (servo-snapped) pose to keep the frontend display
    # consistent with where the hardware actually is. Internal state is
    # kept as a float for precision; the boundary here converts.
    return {
        "armed": _armed,
        "state": sc.state.value if sc else "INIT",
        "servo_pan": snap_to_servo_step(sc.current_pan) if sc else 0,
        "servo_tilt": snap_to_servo_step(sc.current_tilt) if sc else 0,
        "dev_mode": settings.dev_mode,
        "paused": sc.state == SweepState.PAUSED if sc else False,
        "stopped": sc.state == SweepState.STOPPED if sc else False,
        "pause_queued": sc.pause_queued if sc else False,
    }


@router.post("/arm")
async def set_arm(request: ArmRequest):
    global _armed
    _armed = request.armed
    return {"armed": _armed}


@router.post("/fire")
async def manual_fire(request: ManualFireRequest):
    from server.main import get_actuator
    actuator = get_actuator()
    if settings.dev_mode:
        return {"fired": True, "pan": request.pan, "tilt": request.tilt, "simulated": True}
    if actuator:
        # Use direct=True so the manual fire path doesn't queue behind a
        # stuck sweep goto on the traverse lock — manual fire is a UI action
        # and needs to feel instant. The servo handles the slew on its own.
        await actuator.goto(request.pan, request.tilt, direct=True)
        success = await actuator.fire(request.duration_ms)
        return {"fired": success, "pan": request.pan, "tilt": request.tilt}
    return {"fired": False, "error": "No actuator"}


@router.post("/virtual-angle")
async def set_virtual_angle(request: VirtualAngleRequest):
    """Dev mode: set the virtual servo angle."""
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    if sc:
        sc.set_virtual_angle(request.pan, request.tilt)
        return {
            "pan": snap_to_servo_step(sc.current_pan),
            "tilt": snap_to_servo_step(sc.current_tilt),
        }
    return {"pan": request.pan, "tilt": request.tilt}


@router.post("/pause")
async def toggle_pause():
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    if not sc:
        return {"paused": False}
    if sc.state == SweepState.PAUSED:
        sc.resume()
        return {"paused": False}
    else:
        sc.pause()
        return {"paused": sc.state == SweepState.PAUSED or sc.pause_queued}


@router.post("/emergency-stop")
async def emergency_stop():
    global _armed
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    if sc:
        sc.emergency_stop()
    _armed = False
    return {"stopped": True}


@router.post("/clear-estop")
async def clear_estop():
    global _armed
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    if sc:
        sc.clear_emergency_stop()
    _armed = False
    return {"stopped": False, "armed": False}


@router.get("/actuator-logs")
async def get_actuator_logs():
    """Proxy the ESP32 actuator's /logs endpoint so clients don't need to
    know the ESP32's IP. Returns the ring buffer (last ~64 log lines) or
    an error if the actuator isn't reachable."""
    from server.main import get_actuator
    actuator = get_actuator()
    if actuator is None:
        return {"error": "Actuator not initialized", "lines": []}
    logs = await actuator.get_logs()
    if logs is None:
        return {"error": "Failed to fetch logs from actuator", "lines": []}
    return logs


@router.post("/calibration-sweep")
async def start_calibration_sweep():
    """Trigger a full calibration sweep to rebuild the panorama."""
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    if sc:
        sc.current_pan = sc.pan_min
        sc.state = sc.state.SWEEPING
    return {"started": True}
