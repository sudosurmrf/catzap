"""Calibration data model and persistence for the pan/tilt camera-gun rig.

Calibration now captures the rig's **operational extent** — the 4 corners of
the safe servo-angle envelope the camera is allowed to traverse — along with
the home pose and the tilt-compensation polynomial from auto-level. The old
9-point pixel→servo bilinear fit has been removed; the targeting hot path
falls back to the analytic pinhole in `angle_math.pixel_to_angle` using the
config FOV. Data lives in `data/calibration.json` — no DB round trips in
the hot targeting path.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path("data/calibration.json")
RIG_SETTINGS_FILE = Path("data/rig_settings.json")

# Residual threshold for the per-tile verification sweep. If max residual
# across the tile-center checks exceeds this, the UI nudges the user toward
# a recalibration. Relaxed value per user request — water-gun spray cone
# absorbs sub-2° error comfortably.
RESIDUAL_THRESHOLD_DEG = 2.5

# Canonical "neutral" pose the servos snap to when calibration starts. This
# guarantees the user always begins jogging from the same known-good reference
# point instead of wherever the sweep happened to leave the rig. Change these
# two numbers if the physical mount ever moves.
CALIBRATION_HOME_PAN = 105.0
CALIBRATION_HOME_TILT = 45.0

# The four corner labels, in the order the UI should prompt for them. The
# order (bottom-left → top-left → top-right → bottom-right) traces the
# rig's envelope counter-clockwise starting from the lower-left, which is
# the most natural visual sequence for a human recording them.
EXTENT_CORNER_LABELS: tuple[str, ...] = ("bl", "tl", "tr", "br")
CornerLabel = Literal["bl", "tl", "tr", "br"]


class CalibrationPhase(str, Enum):
    JOGGING_TO_HOME = "jogging_to_home"
    LEVELING = "leveling"
    CAPTURING_EXTENT = "capturing_extent"  # user jogs to BL/TL/TR/BR
    EXTENT_READY = "extent_ready"          # tile grid computed, awaiting verification start
    VERIFYING = "verifying"
    COMPLETE = "complete"


class ExtentCorner(BaseModel):
    """One of the 4 recorded servo poses defining the rig's operational envelope.

    Stored in absolute servo-angle space (not relative to home) so the
    clamping check in ActuatorClient can compare directly against the
    commanded angle. The frontend displays them as offsets from home for
    readability, but the on-wire values are absolute.
    """
    label: CornerLabel
    servo_pan: float
    servo_tilt: float


class ExtentBounds(BaseModel):
    """Axis-aligned bounding box of the 4 recorded corners.

    All pose commands are clamped to stay inside this box at the
    ActuatorClient wire boundary, so the servo physically cannot be driven
    past these limits regardless of who called `goto`/`aim`.
    """
    pan_min: float
    pan_max: float
    tilt_min: float
    tilt_max: float


def compute_extent_bounds(corners: list[ExtentCorner]) -> ExtentBounds:
    """Axis-aligned bounding box of the 4 corner poses.

    We use the bounding box rather than the exact trapezoid because
    (a) the sweep controller expects rectangular pan/tilt ranges, and
    (b) the gun mount's obstacle clearance is typically rectangular
    anyway — the 4 corners define the "furthest allowed" extreme on each
    axis, and everything inside the box is reachable.
    """
    if not corners:
        raise ValueError("Need at least one corner to compute bounds")
    pans = [c.servo_pan for c in corners]
    tilts = [c.servo_tilt for c in corners]
    return ExtentBounds(
        pan_min=min(pans),
        pan_max=max(pans),
        tilt_min=min(tilts),
        tilt_max=max(tilts),
    )


def compute_tile_grid_dimensions(
    bounds: ExtentBounds, fov_h: float, fov_v: float, tile_overlap: float
) -> tuple[int, int]:
    """Number of (cols, rows) tiles that cover the extent bounds at the
    given camera FOV and overlap. Matches `TileGrid.__init__` math so the
    calibration-side preview agrees with the runtime grid dimensions."""
    tile_step_h = fov_h - tile_overlap
    tile_step_v = fov_v - tile_overlap
    cols = max(1, int(math.ceil((bounds.pan_max - bounds.pan_min) / tile_step_h)))
    rows = max(1, int(math.ceil((bounds.tilt_max - bounds.tilt_min) / tile_step_v)))
    return (cols, rows)


class VerificationPoint(BaseModel):
    """One tile-center spot-check from the post-calibration verification sweep.

    Records both the computed center (from the tile grid math) and the
    actual servo pose at confirmation, so we have a per-tile data substrate
    for future per-tile pose correction if needed.
    """
    tile_col: int
    tile_row: int
    expected_pan: float
    expected_tilt: float
    actual_pan: float
    actual_tilt: float
    residual: float


class Calibration(BaseModel):
    """Persisted calibration record — the rig's extent, home pose, and
    verification results. Written to `data/calibration.json` after a
    successful calibration run."""
    version: int = 2  # bumped from 1: dropped bilinear fit, added extent corners
    created_at: str
    home_pan: float
    home_tilt: float
    extent_corners: list[ExtentCorner]
    bounds: ExtentBounds
    tile_cols: int
    tile_rows: int
    verification: list[VerificationPoint] = Field(default_factory=list)
    verification_passed: bool = False


class RigSettings(BaseModel):
    """Persistent hardware-ergonomics settings, independent of calibration.

    The jog-inversion flags affect how jog inputs are interpreted; the
    tilt-compensation polynomial describes the rig's tilt-vs-pan error as a
    quadratic in (pan - home) and is used to compensate every outgoing servo
    command so the gun traces a straight world-horizontal line when panning
    instead of an arc; the extent bounds (pan/tilt min/max) are hard servo-
    angle limits enforced by `ActuatorClient` on every pose command to keep
    the gun away from physical obstacles.
    """
    tilt_jog_inverted: bool = False
    pan_jog_inverted: bool = False
    # Quadratic tilt-compensation polynomial coefficients [a, b, c]:
    #     delta_tilt(pan) = a + b*dp + c*dp²     where dp = pan − CALIBRATION_HOME_PAN
    # The outgoing tilt command is `commanded_tilt − delta_tilt(pan)` so that
    # the physical aim lands where the caller intended. Default [0,0,0] means
    # compensation disabled (early-exit in compensate_pan_tilt_coupling).
    # Populated by /auto-level's LSQ fit over recorded (pan, tilt) pairs.
    pan_tilt_poly: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Extent bounds in absolute servo angles. None until the calibration's
    # extent phase runs; the ActuatorClient treats None as "no clamping"
    # and the sweep controller falls back to config defaults. Once set,
    # they persist across restarts.
    pan_min: float | None = None
    pan_max: float | None = None
    tilt_min: float | None = None
    tilt_max: float | None = None


class CalibrationSession(BaseModel):
    """In-memory state for an in-progress calibration run. One at a time."""
    phase: CalibrationPhase = CalibrationPhase.JOGGING_TO_HOME
    home_pan: float = 0.0
    home_tilt: float = 0.0
    reference_frame_jpeg_b64: str | None = None
    # Extent capture state: up to 4 corners keyed by label, preserved in
    # insertion order so the UI can render them in the order they were
    # recorded. Once all 4 are present, the bounds + tile dimensions are
    # computed and the phase advances to EXTENT_READY.
    extent_corners: dict[str, ExtentCorner] = Field(default_factory=dict)
    computed_bounds: ExtentBounds | None = None
    computed_tile_cols: int = 0
    computed_tile_rows: int = 0
    # Verification sweep state — built from the computed tile grid after
    # extent capture. Each entry is (col, row, expected_pan, expected_tilt).
    verification_targets: list[tuple[int, int, float, float]] = Field(default_factory=list)
    verification_points: list[VerificationPoint] = Field(default_factory=list)
    verification_index: int = 0


# ── Persistence ────────────────────────────────────────

def load_calibration() -> Calibration | None:
    """Read persisted calibration, or None if absent/corrupt.

    A calibration written under the old schema (version=1 with a bilinear
    fit) will fail validation against the new schema and be dropped. The
    caller should treat this as "no calibration present" and prompt the
    user to run the new extent-capture flow.
    """
    if not CALIBRATION_FILE.exists():
        return None
    try:
        with CALIBRATION_FILE.open("r") as f:
            data = json.load(f)
        return Calibration.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to load calibration from {CALIBRATION_FILE}: {e}")
        return None


def save_calibration(cal: Calibration) -> None:
    """Atomic write via a temp file so a crash mid-write doesn't corrupt the
    file the startup loader reads."""
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_FILE.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(cal.model_dump(mode="json"), f, indent=2)
    tmp.replace(CALIBRATION_FILE)


def delete_calibration_file() -> bool:
    if CALIBRATION_FILE.exists():
        CALIBRATION_FILE.unlink()
        return True
    return False


def load_rig_settings() -> RigSettings:
    """Read persisted rig settings, or return defaults if absent/corrupt."""
    if not RIG_SETTINGS_FILE.exists():
        return RigSettings()
    try:
        with RIG_SETTINGS_FILE.open("r") as f:
            data = json.load(f)
        return RigSettings.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to load rig settings from {RIG_SETTINGS_FILE}: {e}")
        return RigSettings()


def save_rig_settings(rs: RigSettings) -> None:
    RIG_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = RIG_SETTINGS_FILE.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(rs.model_dump(mode="json"), f, indent=2)
    tmp.replace(RIG_SETTINGS_FILE)


# ── Module-level active state ──────────────────────────
# The active calibration is read on every targeting frame, so we cache it
# in memory rather than touching the disk. Thread-safe because FastAPI
# handlers may race with the vision loop thread.

_active_calibration: Calibration | None = None
_active_calibration_lock = threading.Lock()

_active_session: CalibrationSession | None = None
_session_lock = threading.Lock()

_active_rig_settings: RigSettings = RigSettings()
_rig_settings_lock = threading.Lock()


def get_active_calibration() -> Calibration | None:
    with _active_calibration_lock:
        return _active_calibration


def set_active_calibration(cal: Calibration | None) -> None:
    global _active_calibration
    with _active_calibration_lock:
        _active_calibration = cal


def get_session() -> CalibrationSession | None:
    with _session_lock:
        return _active_session


def set_session(session: CalibrationSession | None) -> None:
    global _active_session
    with _session_lock:
        _active_session = session


def get_rig_settings() -> RigSettings:
    with _rig_settings_lock:
        return _active_rig_settings.model_copy()


def set_rig_settings(rs: RigSettings) -> None:
    global _active_rig_settings
    with _rig_settings_lock:
        _active_rig_settings = rs


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def compensate_pan_tilt_coupling(
    pan: float,
    tilt: float,
    rig: RigSettings | None = None,
) -> tuple[float, float]:
    """Translate an intended (pan, tilt) world pose into the servo command
    that will physically aim there, given the rig's measured tilt error.

    Model: `delta_tilt(pan) = a + b·dp + c·dp²` where `dp = pan − CALIBRATION_HOME_PAN`
    and [a,b,c] = rig.pan_tilt_poly. We subtract `delta_tilt(pan)` from the
    commanded tilt before sending to the ESP32, so the physical aim matches
    the caller's intended world tilt.

    The quadratic captures rigs whose tilt error grows with |pan − home| —
    the common case for pan axes that lean forward/backward rather than
    purely sideways. Early-returns unchanged when all three coefficients
    are zero (the default), so the cost is a loop-short circuit per call
    in the disabled case.
    """
    if rig is None:
        rig = get_rig_settings()
    poly = rig.pan_tilt_poly
    if len(poly) != 3 or (poly[0] == 0.0 and poly[1] == 0.0 and poly[2] == 0.0):
        return (pan, tilt)
    dp = pan - CALIBRATION_HOME_PAN
    correction = poly[0] + poly[1] * dp + poly[2] * dp * dp
    return (pan, tilt - correction)


def clamp_to_extent_bounds(
    pan: float, tilt: float, rig: RigSettings | None = None
) -> tuple[float, float]:
    """Clamp a commanded pose to the rig's recorded extent bounds.

    Returns the pose unchanged if any bound is None (no calibration has
    set the envelope yet). The ActuatorClient calls this on every outgoing
    pose command so the servo physically cannot be driven outside the safe
    envelope regardless of who sent the command or what their intent was.
    """
    if rig is None:
        rig = get_rig_settings()
    if (
        rig.pan_min is None or rig.pan_max is None
        or rig.tilt_min is None or rig.tilt_max is None
    ):
        return (pan, tilt)
    return (
        max(rig.pan_min, min(rig.pan_max, pan)),
        max(rig.tilt_min, min(rig.tilt_max, tilt)),
    )
