"""REST endpoints for the extent-based calibration flow.

Flow (happy path):
    1. POST /start                 — pause sweep, enter JOGGING_TO_HOME phase
    2. POST /jog (repeat)          — nudge servos to desired home pose
    3. POST /set-home              — freeze reference frame, advance to LEVELING
    4. POST /auto-level            — fit quadratic tilt-compensation polynomial
       (optionally /test-level-sweep to visually verify)
    5. POST /begin-extent          — advance to CAPTURING_EXTENT
    6. POST /jog + /record-extent-corner (×4, one per label bl/tl/tr/br)
    7. POST /compute-tile-grid     — derive bounds + tile dimensions, advance to EXTENT_READY
    8. POST /start-verification    — apply bounds to sweep/tile grid, drive to first tile
    9. POST /jog + /confirm-verification (per tile)
    10. POST /finalize             — persist calibration, resume sweep

Anywhere: POST /cancel to abort, GET /status to introspect, DELETE / to wipe.

The old 9-point pixel→servo bilinear fit has been removed in favor of the
extent-based flow: the user defines the 4 corners of the safe operational
envelope, the system derives a tile grid from those bounds and the current
camera FOV, and the verification sweep confirms each tile center looks
right. The targeting hot path falls back to the analytic pinhole in
`angle_math.pixel_to_angle` since there's no fit anymore.
"""

import asyncio
import base64
import logging
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.actuator.calibration import (
    Calibration,
    CalibrationPhase,
    CalibrationSession,
    EXTENT_CORNER_LABELS,
    ExtentCorner,
    RigSettings,
    VerificationPoint,
    CALIBRATION_HOME_PAN,
    CALIBRATION_HOME_TILT,
    RESIDUAL_THRESHOLD_DEG,
    compute_extent_bounds,
    compute_tile_grid_dimensions,
    delete_calibration_file,
    get_active_calibration,
    get_rig_settings,
    get_session,
    now_iso,
    save_calibration,
    save_rig_settings,
    set_active_calibration,
    set_rig_settings,
    set_session,
)
from server.actuator.client import snap_to_servo_step
from server.panorama.sweep_controller import SweepState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


# ── Request models ─────────────────────────────────────

class JogRequest(BaseModel):
    direction: Literal["left", "right", "up", "down"]
    step: Literal["coarse", "fine"] = "coarse"


class RecordExtentCornerRequest(BaseModel):
    label: Literal["bl", "tl", "tr", "br"]


# ── Helpers ────────────────────────────────────────────

COARSE_STEP_DEG = 6.0
FINE_STEP_DEG = 2.0


def _require_session() -> CalibrationSession:
    session = get_session()
    if session is None:
        raise HTTPException(status_code=409, detail="No active calibration session")
    return session


def _sweep_controller():
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    if sc is None:
        raise HTTPException(status_code=503, detail="Sweep controller not initialized")
    return sc


def _tile_grid():
    from server.main import get_tile_grid
    return get_tile_grid()


def _capture_reference_frame_b64() -> tuple[str, float, float]:
    """Grab the latest cached frame and encode it as base64 JPEG for the UI."""
    from server.routers.stream import get_latest_frame
    frame_data = get_latest_frame()
    if frame_data is None:
        raise HTTPException(status_code=503, detail="No frame available from camera")
    frame = frame_data["frame"]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode reference frame")
    b64 = base64.b64encode(buf).decode("utf-8")
    return b64, float(frame_data["servo_pan"]), float(frame_data["servo_tilt"])


# ── Endpoints ──────────────────────────────────────────

@router.post("/start")
async def start_calibration():
    """Pause sweep and enter calibration mode, then snap the servos to the
    canonical neutral home pose so the user always starts from the same known
    reference. They can jog from here to their chosen home pose and call
    /set-home when ready."""
    if get_session() is not None:
        raise HTTPException(status_code=409, detail="Calibration already in progress")

    sc = _sweep_controller()
    sc.pause()
    # Force paused state immediately — don't sit in pause_queued through the
    # rest of the calibration flow.
    sc.state = SweepState.PAUSED
    sc.pause_queued = False

    # Snap to the canonical neutral pose so the user isn't starting from
    # wherever the sweep happened to leave the rig (frequently tilt_bottom,
    # which on most physical mounts is steep-down and useless for calibration).
    sc.set_pose_calibration(CALIBRATION_HOME_PAN, CALIBRATION_HOME_TILT)

    # Send the pose to the ESP32 immediately so the user sees movement as
    # soon as the modal opens — main loop will also re-send on its next tick
    # but going direct removes the ~100 ms lag.
    from server.main import get_actuator
    actuator = get_actuator()
    if actuator is not None:
        try:
            await actuator.goto(CALIBRATION_HOME_PAN, CALIBRATION_HOME_TILT)
        except Exception as e:
            logger.warning(f"Home-pose goto failed (main loop will retry): {e}")

    session = CalibrationSession(
        phase=CalibrationPhase.JOGGING_TO_HOME,
        home_pan=sc.current_pan,
        home_tilt=sc.current_tilt,
    )
    set_session(session)

    return {
        "phase": session.phase.value,
        "current_pose": {
            "pan": snap_to_servo_step(sc.current_pan),
            "tilt": snap_to_servo_step(sc.current_tilt),
        },
    }


@router.post("/jog")
async def jog(request: JogRequest):
    """Move the servos by one step in the given direction. Works during both
    pre-home jogging and extent-corner jogging. Honors the rig-settings
    jog-inversion flags so `up` always makes the gun visually go up regardless
    of how the servos are mounted."""
    _require_session()
    sc = _sweep_controller()
    rig = get_rig_settings()

    step = COARSE_STEP_DEG if request.step == "coarse" else FINE_STEP_DEG
    effective_direction = request.direction
    if rig.tilt_jog_inverted and effective_direction in ("up", "down"):
        effective_direction = "down" if effective_direction == "up" else "up"
    if rig.pan_jog_inverted and effective_direction in ("left", "right"):
        effective_direction = "right" if effective_direction == "left" else "left"

    dx = dy = 0.0
    if effective_direction == "left":
        dx = -step
    elif effective_direction == "right":
        dx = step
    elif effective_direction == "up":
        dy = -step
    elif effective_direction == "down":
        dy = step

    sc.set_pose_calibration(sc.current_pan + dx, sc.current_tilt + dy)

    # Send the new pose immediately so the user sees the servo move. The main
    # loop also pushes this every iteration, but going direct removes the
    # worst-case ~100 ms lag between jog click and physical movement.
    from server.main import get_actuator
    actuator = get_actuator()
    if actuator is not None:
        try:
            await actuator.goto(sc.current_pan, sc.current_tilt)
        except Exception as e:
            logger.warning(f"Direct jog goto failed (main loop will retry): {e}")

    # Return the physical (snapped) pose — what the frontend shows to the
    # user must match what the servos are actually at, not the float intent.
    return {
        "pan": snap_to_servo_step(sc.current_pan),
        "tilt": snap_to_servo_step(sc.current_tilt),
    }


@router.post("/set-home")
async def set_home():
    """Freeze the current pose as home, capture the reference frame, and
    advance to the leveling phase where the user can auto-level the tilt
    compensation polynomial before capturing the extent corners."""
    session = _require_session()
    if session.phase != CalibrationPhase.JOGGING_TO_HOME:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot set home in phase {session.phase.value}",
        )

    sc = _sweep_controller()
    ref_b64, frame_pan, frame_tilt = _capture_reference_frame_b64()

    # Prefer the pose recorded with the frame itself; if it's out of sync with
    # the live sweep_controller (shouldn't happen but be safe), fall back to
    # current pose. Snap to the servo's physical resolution — the calibration
    # math must work on where the gun ACTUALLY is, not the float intent.
    raw_home_pan = frame_pan if frame_pan else sc.current_pan
    raw_home_tilt = frame_tilt if frame_tilt else sc.current_tilt
    session.home_pan = float(snap_to_servo_step(raw_home_pan))
    session.home_tilt = float(snap_to_servo_step(raw_home_tilt))
    session.reference_frame_jpeg_b64 = ref_b64
    session.phase = CalibrationPhase.LEVELING
    set_session(session)

    return {
        "phase": session.phase.value,
        "home_pose": {"pan": session.home_pan, "tilt": session.home_tilt},
        "reference_frame_b64": ref_b64,
    }


@router.post("/begin-extent")
async def begin_extent():
    """Transition from LEVELING to CAPTURING_EXTENT. The user has finished
    tuning the tilt-compensation polynomial and is ready to record the 4
    extent corners that define the safe operational envelope."""
    session = _require_session()
    if session.phase != CalibrationPhase.LEVELING:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot begin extent capture from phase {session.phase.value}",
        )
    session.phase = CalibrationPhase.CAPTURING_EXTENT
    session.extent_corners = {}
    set_session(session)
    return {
        "phase": session.phase.value,
        "corner_labels": list(EXTENT_CORNER_LABELS),
        "recorded_corners": {},
    }


@router.post("/record-extent-corner")
async def record_extent_corner(request: RecordExtentCornerRequest):
    """Record the current servo pose as one of the 4 extent corners. The
    user jogs to the furthest-safe position in that direction, clicks
    record, and the pose is stored. Re-clicking the same label overwrites
    the previous recording for that corner."""
    session = _require_session()
    if session.phase != CalibrationPhase.CAPTURING_EXTENT:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot record extent corner in phase {session.phase.value}",
        )
    sc = _sweep_controller()
    corner = ExtentCorner(
        label=request.label,
        servo_pan=float(snap_to_servo_step(sc.current_pan)),
        servo_tilt=float(snap_to_servo_step(sc.current_tilt)),
    )
    session.extent_corners[request.label] = corner
    all_recorded = len(session.extent_corners) == 4
    set_session(session)
    return {
        "recorded": corner.model_dump(),
        "recorded_corners": {
            k: v.model_dump() for k, v in session.extent_corners.items()
        },
        "all_recorded": all_recorded,
    }


@router.post("/compute-tile-grid")
async def compute_tile_grid():
    """After all 4 corners are recorded, derive the bounding-box bounds
    and the tile grid dimensions (cols × rows) from the current camera
    FOV. Advances to EXTENT_READY; the user then calls /start-verification
    to apply the bounds live and begin the per-tile confirmation sweep."""
    session = _require_session()
    if session.phase != CalibrationPhase.CAPTURING_EXTENT:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot compute tile grid in phase {session.phase.value}",
        )
    if len(session.extent_corners) != 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need all 4 extent corners (bl, tl, tr, br), have "
                f"{len(session.extent_corners)}: "
                f"{list(session.extent_corners.keys())}"
            ),
        )

    corners = [session.extent_corners[label] for label in EXTENT_CORNER_LABELS]
    bounds = compute_extent_bounds(corners)

    tg = _tile_grid()
    if tg is None:
        raise HTTPException(status_code=503, detail="Tile grid not initialized")

    cols, rows = compute_tile_grid_dimensions(
        bounds, tg.fov_h, tg.fov_v, tg.tile_overlap
    )

    session.computed_bounds = bounds
    session.computed_tile_cols = cols
    session.computed_tile_rows = rows
    session.phase = CalibrationPhase.EXTENT_READY
    set_session(session)

    return {
        "phase": session.phase.value,
        "bounds": bounds.model_dump(),
        "tile_cols": cols,
        "tile_rows": rows,
        "total_tiles": cols * rows,
        "fov_h": tg.fov_h,
        "fov_v": tg.fov_v,
    }


@router.post("/start-verification")
async def start_verification():
    """Apply the computed bounds live (sweep controller, tile grid, rig
    settings, ActuatorClient clamping), build the verification target list
    from the new tile grid's tile centers, and drive to the first tile."""
    session = _require_session()
    if session.phase != CalibrationPhase.EXTENT_READY:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start verification in phase {session.phase.value}",
        )
    if session.computed_bounds is None:
        raise HTTPException(status_code=409, detail="Extent bounds not computed")

    bounds = session.computed_bounds

    # Persist bounds to rig_settings.json so they survive restarts and are
    # picked up by the ActuatorClient's clamping via the shared RigSettings
    # module-level cache. Preserve existing jog-invert and poly settings.
    current_rs = get_rig_settings()
    new_rs = RigSettings(
        tilt_jog_inverted=current_rs.tilt_jog_inverted,
        pan_jog_inverted=current_rs.pan_jog_inverted,
        pan_tilt_poly=current_rs.pan_tilt_poly,
        pan_min=bounds.pan_min,
        pan_max=bounds.pan_max,
        tilt_min=bounds.tilt_min,
        tilt_max=bounds.tilt_max,
    )
    save_rig_settings(new_rs)
    set_rig_settings(new_rs)

    # Update the live tile grid and sweep controller with the new bounds
    # so the very next verification goto (and all subsequent sweep ticks)
    # operate against the new envelope.
    tg = _tile_grid()
    sc = _sweep_controller()
    if tg is not None:
        tg.set_bounds(bounds.pan_min, bounds.pan_max, bounds.tilt_min, bounds.tilt_max)
    sc.set_bounds(bounds.pan_min, bounds.pan_max, bounds.tilt_min, bounds.tilt_max)

    # Pre-build verification targets from the updated tile grid.
    verification_targets: list[tuple[int, int, float, float]] = []
    if tg is not None:
        positions = tg.get_tile_positions()
        for i, (pan, tilt) in enumerate(positions):
            col = i % tg.cols
            row = i // tg.cols
            verification_targets.append((col, row, pan, tilt))

    session.verification_targets = verification_targets
    session.verification_points = []
    session.verification_index = 0
    session.phase = CalibrationPhase.VERIFYING
    set_session(session)

    if not verification_targets:
        raise HTTPException(status_code=500, detail="Tile grid produced zero tiles")

    # Drive to the first tile center.
    col, row, pan, tilt = verification_targets[0]
    sc.set_pose_calibration(pan, tilt)
    from server.main import get_actuator
    actuator = get_actuator()
    if actuator is not None:
        try:
            await actuator.goto(pan, tilt)
        except Exception as e:
            logger.warning(f"Verification initial goto failed: {e}")

    return {
        "phase": session.phase.value,
        "current_index": 0,
        "total": len(verification_targets),
        "tile_col": col,
        "tile_row": row,
        "expected_pan": pan,
        "expected_tilt": tilt,
    }


@router.post("/confirm-verification")
async def confirm_verification():
    """Record the current servo pose as the actual aim for the current tile
    target and advance. When all tiles are done, build and persist the
    Calibration object from the session's extent + verification data."""
    session = _require_session()
    if session.phase != CalibrationPhase.VERIFYING:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot confirm verification in phase {session.phase.value}",
        )
    idx = session.verification_index
    if idx >= len(session.verification_targets):
        raise HTTPException(status_code=409, detail="Verification already complete")

    col, row, expected_pan, expected_tilt = session.verification_targets[idx]
    sc = _sweep_controller()
    actual_pan = sc.current_pan
    actual_tilt = sc.current_tilt
    residual = (
        (actual_pan - expected_pan) ** 2 + (actual_tilt - expected_tilt) ** 2
    ) ** 0.5

    vp = VerificationPoint(
        tile_col=col,
        tile_row=row,
        expected_pan=expected_pan,
        expected_tilt=expected_tilt,
        actual_pan=actual_pan,
        actual_tilt=actual_tilt,
        residual=residual,
    )
    session.verification_points.append(vp)
    session.verification_index += 1

    if session.verification_index < len(session.verification_targets):
        next_col, next_row, next_pan, next_tilt = session.verification_targets[
            session.verification_index
        ]
        sc.set_pose_calibration(next_pan, next_tilt)
        from server.main import get_actuator
        actuator = get_actuator()
        if actuator is not None:
            try:
                await actuator.goto(next_pan, next_tilt)
            except Exception as e:
                logger.warning(f"Verification goto failed: {e}")
        set_session(session)
        return {
            "complete": False,
            "last_residual": residual,
            "current_index": session.verification_index,
            "total": len(session.verification_targets),
            "tile_col": next_col,
            "tile_row": next_row,
            "expected_pan": next_pan,
            "expected_tilt": next_tilt,
        }

    # All verification points collected — build and persist the calibration.
    max_res = max(vp.residual for vp in session.verification_points)
    mean_res = sum(vp.residual for vp in session.verification_points) / len(
        session.verification_points
    )
    passed = max_res < RESIDUAL_THRESHOLD_DEG

    if session.computed_bounds is None:
        raise HTTPException(status_code=500, detail="Session missing computed bounds")

    cal = Calibration(
        created_at=now_iso(),
        home_pan=session.home_pan,
        home_tilt=session.home_tilt,
        extent_corners=[session.extent_corners[label] for label in EXTENT_CORNER_LABELS],
        bounds=session.computed_bounds,
        tile_cols=session.computed_tile_cols,
        tile_rows=session.computed_tile_rows,
        verification=list(session.verification_points),
        verification_passed=passed,
    )
    save_calibration(cal)
    set_active_calibration(cal)

    session.phase = CalibrationPhase.COMPLETE
    set_session(session)

    return {
        "complete": True,
        "last_residual": residual,
        "max_residual": max_res,
        "mean_residual": mean_res,
        "threshold": RESIDUAL_THRESHOLD_DEG,
        "passed": passed,
    }


@router.post("/skip-verification")
async def skip_verification():
    """Accept the extent-based tile grid without running the per-tile
    verification sweep. Persists a Calibration with an empty verification
    list and jumps straight to COMPLETE."""
    session = _require_session()
    if session.phase != CalibrationPhase.EXTENT_READY:
        raise HTTPException(status_code=409, detail="Extent not ready")
    if session.computed_bounds is None:
        raise HTTPException(status_code=500, detail="Session missing computed bounds")

    cal = Calibration(
        created_at=now_iso(),
        home_pan=session.home_pan,
        home_tilt=session.home_tilt,
        extent_corners=[session.extent_corners[label] for label in EXTENT_CORNER_LABELS],
        bounds=session.computed_bounds,
        tile_cols=session.computed_tile_cols,
        tile_rows=session.computed_tile_rows,
        verification=[],
        verification_passed=False,
    )
    save_calibration(cal)
    set_active_calibration(cal)

    # Still apply bounds to rig_settings + live objects even when skipping.
    bounds = session.computed_bounds
    current_rs = get_rig_settings()
    new_rs = RigSettings(
        tilt_jog_inverted=current_rs.tilt_jog_inverted,
        pan_jog_inverted=current_rs.pan_jog_inverted,
        pan_tilt_poly=current_rs.pan_tilt_poly,
        pan_min=bounds.pan_min,
        pan_max=bounds.pan_max,
        tilt_min=bounds.tilt_min,
        tilt_max=bounds.tilt_max,
    )
    save_rig_settings(new_rs)
    set_rig_settings(new_rs)
    tg = _tile_grid()
    sc = _sweep_controller()
    if tg is not None:
        tg.set_bounds(bounds.pan_min, bounds.pan_max, bounds.tilt_min, bounds.tilt_max)
    sc.set_bounds(bounds.pan_min, bounds.pan_max, bounds.tilt_min, bounds.tilt_max)

    session.phase = CalibrationPhase.COMPLETE
    set_session(session)
    return {"phase": session.phase.value}


@router.post("/finalize")
async def finalize():
    """Tear down the session and resume normal sweep operation."""
    session = get_session()
    if session is not None and session.phase not in (
        CalibrationPhase.COMPLETE,
        CalibrationPhase.EXTENT_READY,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot finalize from phase {session.phase.value} — finish or cancel instead",
        )
    set_session(None)
    sc = _sweep_controller()
    sc.resume()
    return {"status": "ready"}


@router.post("/cancel")
async def cancel():
    """Abort calibration at any phase. Does NOT delete the persisted
    calibration from a previous successful run."""
    set_session(None)
    sc = _sweep_controller()
    sc.resume()
    return {"cancelled": True}


@router.get("/status")
async def get_status():
    """Introspect active calibration + session state."""
    cal = get_active_calibration()
    session = get_session()
    return {
        "has_calibration": cal is not None,
        "calibration": cal.model_dump() if cal else None,
        "session_active": session is not None,
        "session_phase": session.phase.value if session else None,
        "session": session.model_dump(exclude={"reference_frame_jpeg_b64"}) if session else None,
        "residual_threshold_deg": RESIDUAL_THRESHOLD_DEG,
    }


@router.delete("")
async def delete_calibration_endpoint():
    """Remove the persisted calibration. Does not affect an active session."""
    deleted = delete_calibration_file()
    set_active_calibration(None)
    return {"deleted": deleted}


# ── Rig settings (jog inversion, tilt compensation, bounds) ────

class RigSettingsUpdate(BaseModel):
    tilt_jog_inverted: bool | None = None
    pan_jog_inverted: bool | None = None
    pan_tilt_poly: list[float] | None = None
    pan_min: float | None = None
    pan_max: float | None = None
    tilt_min: float | None = None
    tilt_max: float | None = None


class AutoLevelPoint(BaseModel):
    pan: float
    tilt: float


class AutoLevelRequest(BaseModel):
    # Ordered list of recorded (pan, tilt) pairs the user reached while
    # aiming the gun at a world-horizontal reference at different pan angles.
    # Minimum 3 points required; 5+ strongly preferred for noise robustness.
    points: list[AutoLevelPoint]


@router.get("/rig-settings")
async def get_rig_settings_endpoint():
    return get_rig_settings().model_dump()


@router.post("/test-level-sweep")
async def test_level_sweep():
    """Slowly walk the gun across the pan range at a constant world tilt,
    so the user can visually verify that pan-axis compensation is working.

    The gun should trace a straight horizontal line relative to the world
    (fixed objects at the home frame's py=0.5 should stay on the horizontal
    reference line throughout the sweep). Any residual arc reveals that the
    tilt-compensation polynomial is still mis-fit — either re-run /auto-level
    with more recorded points, or the quadratic model itself is insufficient
    for the rig's geometry.

    Implementation: just update `sc.current_pan` in 1° steps at 10 Hz —
    the main loop's dedupe dispatch picks up each int change and routes it
    through `ActuatorClient.goto()`, which applies the compensation layer.
    Total run time is ~8 seconds for an 80° pan span.
    """
    session = _require_session()
    if session.phase != CalibrationPhase.LEVELING:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot run level sweep in phase {session.phase.value}",
        )

    sc = _sweep_controller()
    home_pan = session.home_pan
    home_tilt = session.home_tilt

    sweep_half_range = 40.0
    start_pan = max(0.0, home_pan - sweep_half_range)
    end_pan = min(180.0, home_pan + sweep_half_range)
    step_deg = 1.0
    step_delay = 0.1  # 10 Hz, matches main loop dispatch cadence

    async def walk(from_pan: float, to_pan: float) -> None:
        direction = 1 if to_pan >= from_pan else -1
        pan = from_pan
        while (direction > 0 and pan <= to_pan) or (direction < 0 and pan >= to_pan):
            sc.set_pose_calibration(pan, home_tilt)
            await asyncio.sleep(step_delay)
            pan += direction * step_deg
        # Ensure we end exactly at the target (don't get stuck 0.9° short)
        sc.set_pose_calibration(to_pan, home_tilt)
        await asyncio.sleep(step_delay)

    # Center -> right -> left -> center, so both directions are exercised
    # and the gun finishes back at the home pose for continued calibration.
    await walk(home_pan, end_pan)
    await walk(end_pan, start_pan)
    await walk(start_pan, home_pan)

    return {
        "status": "complete",
        "span": end_pan - start_pan,
        "home_pan": home_pan,
        "home_tilt": home_tilt,
    }


@router.post("/auto-level")
async def auto_level(req: AutoLevelRequest):
    """Least-squares N-point solve for the quadratic tilt-compensation polynomial.

    Given a list of servo poses the user reached while aiming the gun at a
    world-horizontal reference (e.g., a ruler along a level line) at different
    pan angles, fit a quadratic tilt error as a function of (pan − home_pan)
    and persist it to rig settings.

    The model:
        delta_tilt(pan) = a + b·dp + c·dp²     where dp = pan − CALIBRATION_HOME_PAN
    Each recorded point (pt.pan, pt.tilt) contributes one equation:
        home_tilt − pt.tilt = a + b·(pt.pan − CALIBRATION_HOME_PAN) + c·(…)²
    The fit runs in terms of CALIBRATION_HOME_PAN so that the persisted
    coefficients agree with how compensate_pan_tilt_coupling applies them at
    runtime — any drift between the session's home_pan and CALIBRATION_HOME_PAN
    is absorbed into `a` automatically.

    N ≥ 3 points needed (3 unknowns). 5+ recommended to average out servo-snap
    noise (±1° per point from the 2° hardware resolution) across the fit.
    """
    session = _require_session()
    if session.phase not in (
        CalibrationPhase.LEVELING,
        CalibrationPhase.JOGGING_TO_HOME,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot auto-level in phase {session.phase.value}",
        )

    if len(req.points) < 3:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least 3 recorded points for a quadratic fit, got "
                f"{len(req.points)}. Record 5 for best noise averaging."
            ),
        )

    home_pan = session.home_pan
    home_tilt = session.home_tilt

    # Build the linear system A @ x = y where x = [a, b, c] and each row of
    # A represents one recorded point's (1, dp, dp²) values. Using
    # CALIBRATION_HOME_PAN (not session.home_pan) as the reference so that
    # persisted coefficients apply correctly at runtime.
    dp = np.array(
        [pt.pan - CALIBRATION_HOME_PAN for pt in req.points],
        dtype=np.float64,
    )
    deltas = np.array(
        [home_tilt - pt.tilt for pt in req.points],
        dtype=np.float64,
    )
    A = np.column_stack([np.ones_like(dp), dp, dp * dp])

    # Guard against rank deficiency (all points at the same pan would give
    # A that's effectively rank 1, making the solve useless).
    _, singular_values, _ = np.linalg.svd(A, full_matrices=False)
    smallest_sv = float(singular_values[-1]) if len(singular_values) > 0 else 0.0
    if smallest_sv < 1e-4:
        raise HTTPException(
            status_code=400,
            detail=(
                "Recorded points are too similar to solve. Make sure you "
                "jogged meaningfully in pan between each recording."
            ),
        )

    # Least-squares fit.
    solution, *_ = np.linalg.lstsq(A, deltas, rcond=None)
    a, b, c = float(solution[0]), float(solution[1]), float(solution[2])

    # Per-point residuals for quality feedback
    predicted = A @ solution
    per_point_residuals = [float(deltas[i] - predicted[i]) for i in range(len(req.points))]
    max_residual = float(np.max(np.abs(per_point_residuals)))
    rms_residual = float(np.sqrt(np.mean(np.array(per_point_residuals) ** 2)))

    current = get_rig_settings()
    new = RigSettings(
        tilt_jog_inverted=current.tilt_jog_inverted,
        pan_jog_inverted=current.pan_jog_inverted,
        pan_tilt_poly=[a, b, c],
        pan_min=current.pan_min,
        pan_max=current.pan_max,
        tilt_min=current.tilt_min,
        tilt_max=current.tilt_max,
    )
    save_rig_settings(new)
    set_rig_settings(new)

    return {
        "pan_tilt_poly": [a, b, c],
        "home_pan": home_pan,
        "home_tilt": home_tilt,
        "num_points": len(req.points),
        "per_point_residuals": per_point_residuals,
        "max_residual": max_residual,
        "rms_residual": rms_residual,
        "rig_settings": new.model_dump(),
    }


@router.post("/rig-settings")
async def update_rig_settings(update: RigSettingsUpdate):
    """Patch any subset of rig-settings fields. Persists to disk immediately
    so the next server restart keeps the values."""
    current = get_rig_settings()
    new = RigSettings(
        tilt_jog_inverted=(
            update.tilt_jog_inverted
            if update.tilt_jog_inverted is not None
            else current.tilt_jog_inverted
        ),
        pan_jog_inverted=(
            update.pan_jog_inverted
            if update.pan_jog_inverted is not None
            else current.pan_jog_inverted
        ),
        pan_tilt_poly=(
            update.pan_tilt_poly
            if update.pan_tilt_poly is not None
            else current.pan_tilt_poly
        ),
        pan_min=(update.pan_min if update.pan_min is not None else current.pan_min),
        pan_max=(update.pan_max if update.pan_max is not None else current.pan_max),
        tilt_min=(update.tilt_min if update.tilt_min is not None else current.tilt_min),
        tilt_max=(update.tilt_max if update.tilt_max is not None else current.tilt_max),
    )
    save_rig_settings(new)
    set_rig_settings(new)
    return new.model_dump()
