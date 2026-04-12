# 2D Exclusion Zones + Engagement State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 3D MiDaS-based exclusion zone system with a simpler 2D polygon system, and collapse the 6-state sweep controller into a 4-state engagement machine where cats only trigger firing when their bbox overlaps an exclusion zone.

**Architecture:** The live vision loop lives in `server/main.py:run_vision_loop`, not in `server/vision/pipeline.py` (which is dead code and will be deleted). `SweepController` gets its state machine simplified from `SWEEPING/TRACKING/WARNING/FIRING/PAUSED/STOPPED` to `SWEEPING/ENGAGING/PAUSED/STOPPED` — `ENGAGING` absorbs the smooth target-pursuit behavior that used to live in `TRACKING/WARNING/FIRING`. The 3D `server/spatial/` directory, `server/routers/spatial.py`, MiDaS/depth configuration, and all volumetric zone fields are deleted outright. The `zones` and `furniture` database tables are dropped and recreated with simplified schemas.

**Tech Stack:** Python 3.11+ / FastAPI / pytest / asyncpg / Shapely (already in use) for the backend; React + TypeScript + Vite for the frontend. The ESP32 firmware is untouched.

---

## Scope check

This plan covers one cohesive change — the 3D-to-2D rewrite and the state machine collapse are tightly coupled because `main.py` touches both, and splitting them would produce an intermediate state where the vision loop is half-ported. It ships as one PR.

## File structure

### Files created

- `server/tests/test_engagement_state_machine.py` — unit tests for the new 2-state engagement flow (with fake clock).

### Files deleted

- `server/spatial/__init__.py`
- `server/spatial/depth_estimator.py`
- `server/spatial/room_model.py`
- `server/spatial/projection.py`
- `server/spatial/cat_tracker.py`
- `server/spatial/` (directory itself)
- `server/routers/spatial.py`
- `server/vision/pipeline.py` (dead code — not imported by `main.py`, only by its own test)
- `server/tests/test_pipeline.py` (tests dead code)
- `server/tests/test_depth_estimator.py`
- `server/tests/test_volumetric_zones.py`
- `server/tests/test_room_model.py`
- `server/tests/test_projection.py`
- `server/tests/test_cat_tracker.py`

### Files modified

**Backend:**
- `server/panorama/sweep_controller.py` — state enum collapsed; `tick()` simplified; new `on_cat_in_zone()` / `on_no_cat_in_zone()` API; `min_shot_interval_ms` / `engagement_grace_ms` parameters; old `on_cat_detected()` / `on_cat_not_in_zone()` / `on_no_cat_detected()` / `on_cat_still_in_zone()` / `on_cat_left_zone()` / `should_fire()` / `get_warning_remaining()` methods removed.
- `server/vision/zone_checker.py` — `check_3d_zone_violation()` deleted; `check_zone_violations()` loses the `cat_room_pos` parameter and the `mode`/`auto_3d`/`manual_3d` fallback branches; poly cache keys simplified.
- `server/models/database.py` — `zones` table recreated with new column set; `furniture` table recreated with new column set; `get_furniture()` signature simplified; any CRUD helpers that read the dropped columns updated.
- `server/routers/zones.py` — request/response schemas lose `mode`, `room_polygon`, `height_min`, `height_max`, `furniture_id`, `cooldown_seconds`.
- `server/routers/control.py` or wherever furniture CRUD lives — furniture endpoints simplified to `{name, polygon}` only; any update/move endpoint removed.
- `server/main.py` — MiDaS / `DepthEstimator` / `RoomModel` / `CatTracker` / `FurnitureObject` imports and initialization deleted; per-frame depth inference deleted; `check_zone_violations` call loses `cat_room_pos`; sweep-controller API calls updated to the new 2-state surface; spatial router mount deleted; furniture loading code deleted (or reduced to a no-op since simplified furniture doesn't need runtime state).
- `server/config.py` — MiDaS/room-model/occlusion settings removed; `min_shot_interval_ms` and `engagement_grace_ms` added; `warning_duration`, `tracking_duration`, `reentry_warning`, `lock_on_grace`, `cooldown_default` removed (replaced by the new settings).
- `server/tests/test_sweep_controller.py` — full rewrite for the new state machine.
- `server/tests/test_zone_checker.py` — 3D test cases deleted.
- `server/tests/test_database.py` — updated to match new schema.
- `server/tests/test_sweep_pause.py` — updated if any of its fixtures use removed config keys.

**Frontend:**
- `frontend/src/types/index.ts` — `Zone` interface stripped to 2D fields; `Furniture` stripped to `{id, name, polygon, created_at}`; `ZoneTransform` deleted.
- `frontend/src/api/client.ts` — `estimateHeight` / any `/spatial/*` calls deleted; zone/furniture payloads updated.
- `frontend/src/components/ZoneEditor.tsx` — 3D prism overlay rendering deleted; mode-dependent label code deleted; 2D polygon drawing kept.
- `frontend/src/components/ZoneConfigPanel.tsx` — height slider, 3D transform sliders, auto-estimate-height button, mode selector deleted; name/overlap-threshold/enabled inputs kept.
- `frontend/src/components/HeightSlider.tsx` — deleted if only used by 3D UI (verify first).
- `frontend/src/components/PanoramaView.tsx` — any 3D overlay code deleted.
- `frontend/src/components/LiveFeed.tsx` — any 3D mode branches deleted.
- `frontend/src/components/Settings.tsx` — 3D / MiDaS / depth UI branches deleted; new `min_shot_interval_ms` / `engagement_grace_ms` fields added (or read-only display).
- `frontend/src/components/CalibrationWizard.tsx` — any depth-calibration step deleted.
- `frontend/src/components/App.tsx` — any top-level 3D state / route deleted.

**Dependencies (conditional):**
- `requirements.txt` / `pyproject.toml` — remove `torch` and `transformers` *only if* a repo-wide grep confirms nothing else imports them. Deferred to the last task.

---

## Phase 1: Rewrite the sweep controller state machine (test-first)

### Task 1: Replace `test_sweep_controller.py` with tests for the new 2-state engagement machine

**Files:**
- Modify: `server/tests/test_sweep_controller.py`

**Context:** The current test file (107 lines) references API methods that no longer exist in the live code (`on_cat_still_in_zone`, `on_cat_left_zone`, `_tracking_start`). It's already out of sync. Rewrite from scratch against the new API.

The tests use an injectable `time_source` fixture (a `MockClock` class) so timer-dependent behavior runs deterministically without `time.sleep`. The controller will grow a `time_source: Callable[[], float] = time.time` constructor parameter in Task 2.

- [ ] **Step 1: Write the full replacement test file**

Replace the contents of `server/tests/test_sweep_controller.py` with:

```python
import pytest
from unittest.mock import AsyncMock
from server.panorama.sweep_controller import SweepController, SweepState


class MockClock:
    """Deterministic time source for state-machine tests. Tests advance the
    clock explicitly via `advance(seconds)` rather than waiting on real time."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return MockClock()


@pytest.fixture
def controller(clock):
    actuator = AsyncMock()
    actuator.goto.return_value = True
    actuator.fire.return_value = True
    actuator.aim_and_fire.return_value = True
    return SweepController(
        actuator=actuator,
        pan_min=30.0, pan_max=150.0,
        tilt_min=20.0, tilt_max=70.0,
        speed=10.0,
        min_shot_interval_ms=2000,
        engagement_grace_ms=3000,
        dev_mode=True,
        time_source=clock,
    )


# ── Sweep behavior (unchanged from old test file) ──────────────

def test_initial_state_is_sweeping(controller):
    assert controller.state == SweepState.SWEEPING


def test_sweep_advances_angle(controller):
    initial = controller.current_pan
    controller.tick(dt=1.0)
    assert controller.current_pan != initial


def test_sweep_covers_tilt_range(controller):
    """Boustrophedon sweep should visit both top and bottom tilt rows."""
    tilt_values = set()
    for _ in range(200):
        controller.tick(dt=0.5)
        tilt_values.add(round(controller.current_tilt))
    tilt_mid = (controller.tilt_min + controller.tilt_max) / 2
    assert max(tilt_values) > tilt_mid
    assert min(tilt_values) < tilt_mid


# ── Cats outside zones are ignored (spec option 1) ─────────────

def test_cat_outside_zone_does_not_change_state(controller):
    """A cat detected anywhere that isn't inside a zone must not stop the
    sweep. Only cats inside zones trigger engagement."""
    controller.on_no_cat_in_zone()
    assert controller.state == SweepState.SWEEPING


# ── SWEEPING → ENGAGING ────────────────────────────────────────

def test_cat_in_zone_transitions_to_engaging(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.ENGAGING


def test_cat_in_zone_sets_target(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    target_pan, target_tilt, zone = controller.get_target()
    assert target_pan == 90.0
    assert target_tilt == 45.0
    assert zone == "Counter"


# ── First shot fires immediately ───────────────────────────────

def test_first_shot_ready_on_entering_engaging(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is True


# ── Min shot interval gate ─────────────────────────────────────

def test_shot_blocked_within_min_interval(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is True
    controller.mark_shot_fired()  # caller confirms a shot was executed
    clock.advance(1.0)  # only 1 second elapsed, interval is 2
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is False


def test_shot_allowed_after_min_interval(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.mark_shot_fired()
    clock.advance(2.0)  # exactly the interval
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is True


# ── Target updates to latest bbox during engagement ────────────

def test_target_follows_cat_across_frames(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.mark_shot_fired()
    clock.advance(0.5)
    controller.on_cat_in_zone(cat_pan=100.0, cat_tilt=40.0, zone_name="Counter")
    target_pan, target_tilt, _ = controller.get_target()
    assert target_pan == 100.0
    assert target_tilt == 40.0


def test_camera_moves_toward_target_in_engaging(controller, clock):
    """Target-following interpolation from the old TRACKING/WARNING/FIRING
    states must survive the collapse into ENGAGING."""
    controller.current_pan = 90.0
    controller.current_tilt = 45.0
    controller.on_cat_in_zone(cat_pan=120.0, cat_tilt=60.0, zone_name="Counter")
    initial_pan = controller.current_pan
    initial_tilt = controller.current_tilt
    controller.tick(dt=0.1)
    # Camera should move toward (120, 60) from (90, 45)
    assert controller.current_pan > initial_pan
    assert controller.current_tilt > initial_tilt


# ── Grace window ───────────────────────────────────────────────

def test_grace_window_holds_engaging_briefly(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    clock.advance(1.0)
    controller.on_no_cat_in_zone()
    clock.advance(1.0)  # 1 second into 3s grace
    controller.tick(dt=0.0)
    assert controller.state == SweepState.ENGAGING


def test_grace_expiry_returns_to_sweeping(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    clock.advance(0.1)
    controller.on_no_cat_in_zone()
    clock.advance(3.001)  # past the 3s grace
    controller.tick(dt=0.0)
    assert controller.state == SweepState.SWEEPING


def test_cat_returns_during_grace_resumes_engaging(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.mark_shot_fired()
    clock.advance(0.5)
    controller.on_no_cat_in_zone()
    clock.advance(1.0)  # mid-grace
    controller.on_cat_in_zone(cat_pan=95.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.ENGAGING
    target_pan, _, _ = controller.get_target()
    assert target_pan == 95.0


def test_grace_only_counts_time_since_last_zone_presence(controller, clock):
    """Returning to the zone during grace must reset the grace timer — a
    subsequent absence starts counting from the re-entry, not the original
    departure."""
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    clock.advance(0.1)
    controller.on_no_cat_in_zone()
    clock.advance(2.0)  # 2s into grace
    controller.on_cat_in_zone(cat_pan=95.0, cat_tilt=45.0, zone_name="Counter")
    clock.advance(0.1)
    controller.on_no_cat_in_zone()
    clock.advance(2.5)  # 2.5s — would have expired the ORIGINAL grace but not this one
    controller.tick(dt=0.0)
    assert controller.state == SweepState.ENGAGING


# ── Pause / stop unchanged ─────────────────────────────────────

def test_pause_from_sweeping(controller):
    controller.pause()
    assert controller.state == SweepState.PAUSED


def test_resume_from_paused(controller):
    controller.pause()
    controller.resume()
    assert controller.state == SweepState.SWEEPING


def test_emergency_stop(controller):
    controller.emergency_stop()
    assert controller.state == SweepState.STOPPED
    assert controller.armed is False


# ── Direction delta (used by dev-mode arrow overlay) ──────────

def test_get_direction_delta(controller):
    controller.current_pan = 80.0
    controller.current_tilt = 45.0
    delta = controller.get_direction_delta(target_pan=100.0, target_tilt=50.0)
    assert delta["pan"] > 0
    assert delta["tilt"] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest server/tests/test_sweep_controller.py -v`

Expected: most tests FAIL with `TypeError` on `SweepController(... min_shot_interval_ms=..., time_source=...)` or `AttributeError` on methods like `on_cat_in_zone` (with new signature), `on_no_cat_in_zone`, `mark_shot_fired`, `SweepState.ENGAGING`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add server/tests/test_sweep_controller.py
git commit -m "test(sweep_controller): rewrite for 2-state engagement machine"
```

---

### Task 2: Refactor `SweepController` to the new 4-state engagement machine

**Files:**
- Modify: `server/panorama/sweep_controller.py` (full rewrite of state logic; preserve sweep geometry, bounds handling, `_clamp_target`, and pause/stop)

**Context:** The new enum has four members: `SWEEPING`, `ENGAGING`, `PAUSED`, `STOPPED`. `ENGAGING` absorbs what `TRACKING`, `WARNING`, and `FIRING` used to do. The caller supplies detection events via `on_cat_in_zone(cat_pan, cat_tilt, zone_name)` and `on_no_cat_in_zone()`. The caller gates firing via `should_fire()`, executes the shot, then calls `mark_shot_fired()` so the controller can start the `min_shot_interval_ms` countdown. `on_fire_complete(fired_pan, fired_tilt)` still exists and is still responsible for snapping `current_pan/tilt` to match the raw pose commanded by the fire flow.

The `time_source` constructor parameter lets tests inject a `MockClock`. Production code passes the default `time.time`.

- [ ] **Step 1: Replace the file with the new implementation**

Replace the contents of `server/panorama/sweep_controller.py` with:

```python
import time
from enum import Enum
from typing import Callable

from server.actuator.client import ActuatorClient


class SweepState(str, Enum):
    SWEEPING = "SWEEPING"
    ENGAGING = "ENGAGING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class SweepController:
    """Camera sweep + cat engagement state machine.

    States:
        SWEEPING  — normal boustrophedon sweep runs.
        ENGAGING  — a cat is (or recently was) inside an exclusion zone; sweep
                    is paused, the camera smoothly pursues the latest bbox
                    center, and `should_fire()` gates shots to a minimum
                    interval of `min_shot_interval_ms`. If no cat has been
                    reported inside any zone for `engagement_grace_ms`, the
                    controller returns to SWEEPING.
        PAUSED    — manual pause (from arming UI); tick() is a no-op.
        STOPPED   — emergency stop; everything halts until explicitly cleared.

    Cats detected OUTSIDE every zone are ignored — the caller should not call
    `on_cat_in_zone` for them and the controller will keep sweeping. This is a
    deliberate simplification: the old TRACKING state (pursuit of any detected
    cat) has been removed.
    """

    def __init__(
        self,
        actuator: ActuatorClient,
        pan_min: float = 30.0,
        pan_max: float = 150.0,
        tilt_min: float = 20.0,
        tilt_max: float = 70.0,
        speed: float = 2.5,
        min_shot_interval_ms: int = 2000,
        engagement_grace_ms: int = 3000,
        dev_mode: bool = False,
        time_source: Callable[[], float] = time.time,
    ):
        self.actuator = actuator
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.speed = speed
        self.min_shot_interval_ms = min_shot_interval_ms
        self.engagement_grace_ms = engagement_grace_ms
        self.dev_mode = dev_mode
        self._time = time_source

        # Boustrophedon sweep geometry (unchanged from the old implementation).
        self._tilt_top = tilt_min + (tilt_max - tilt_min) * 0.5
        self._tilt_bottom = tilt_max

        pan_range = pan_max - pan_min
        tilt_transition = abs(self._tilt_bottom - self._tilt_top)
        self._phase_durations = [
            pan_range / self.speed,
            tilt_transition / self.speed,
            pan_range / self.speed,
            tilt_transition / self.speed,
        ]
        self._cycle_duration = sum(self._phase_durations)
        self._sweep_t = 0.0

        self.current_pan = pan_min
        self.current_tilt = self._tilt_top

        self.state = SweepState.SWEEPING
        self.armed = True
        self.pause_queued = False
        self._target_pan: float = 0.0
        self._target_tilt: float = 0.0
        self._target_zone: str = ""
        self._last_shot_time: float = 0.0  # epoch seconds; 0.0 means "never fired"
        self._last_in_zone_time: float = 0.0  # updated every frame a cat is reported in a zone

    def set_bounds(
        self, pan_min: float, pan_max: float, tilt_min: float, tilt_max: float
    ) -> None:
        """Replace the sweep pan/tilt range and recompute phase durations.
        Called by the calibration router after extent capture."""
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self._tilt_top = tilt_min + (tilt_max - tilt_min) * 0.5
        self._tilt_bottom = tilt_max
        pan_range = pan_max - pan_min
        tilt_transition = abs(self._tilt_bottom - self._tilt_top)
        self._phase_durations = [
            pan_range / self.speed,
            tilt_transition / self.speed,
            pan_range / self.speed,
            tilt_transition / self.speed,
        ]
        self._cycle_duration = sum(self._phase_durations)
        self._sweep_t = 0.0

    def pause(self) -> None:
        if self.state == SweepState.SWEEPING:
            self.state = SweepState.PAUSED
            self.pause_queued = False
        elif self.state != SweepState.STOPPED:
            self.pause_queued = True

    def resume(self) -> None:
        if self.state == SweepState.PAUSED:
            self.state = SweepState.SWEEPING
        self.pause_queued = False

    def emergency_stop(self) -> None:
        self.state = SweepState.STOPPED
        self.armed = False
        self.pause_queued = False

    def clear_emergency_stop(self) -> None:
        if self.state == SweepState.STOPPED:
            self.state = SweepState.SWEEPING

    def tick(self, dt: float) -> None:
        """Advance the state machine by dt seconds."""
        if self.state in (SweepState.PAUSED, SweepState.STOPPED):
            return
        now = self._time()

        if self.state == SweepState.SWEEPING:
            self._sweep_t += dt
            if self._sweep_t >= self._cycle_duration:
                self._sweep_t -= self._cycle_duration

            t = self._sweep_t
            d0, d1, d2, d3 = self._phase_durations

            if t < d0:
                frac = t / d0
                self.current_pan = self.pan_min + frac * (self.pan_max - self.pan_min)
                self.current_tilt = self._tilt_top
            elif t < d0 + d1:
                frac = (t - d0) / d1
                self.current_pan = self.pan_max
                self.current_tilt = self._tilt_top + frac * (self._tilt_bottom - self._tilt_top)
            elif t < d0 + d1 + d2:
                frac = (t - d0 - d1) / d2
                self.current_pan = self.pan_max - frac * (self.pan_max - self.pan_min)
                self.current_tilt = self._tilt_bottom
            else:
                frac = (t - d0 - d1 - d2) / d3
                self.current_pan = self.pan_min
                self.current_tilt = self._tilt_bottom - frac * (self._tilt_bottom - self._tilt_top)

        elif self.state == SweepState.ENGAGING:
            # Smooth pursuit of the latest target, preserved verbatim from the
            # old TRACKING/WARNING/FIRING code path (same 4x speed).
            track_speed = self.speed * 4
            pan_delta = self._target_pan - self.current_pan
            tilt_delta = self._target_tilt - self.current_tilt
            max_step = track_speed * dt

            if abs(pan_delta) > max_step:
                self.current_pan += max_step if pan_delta > 0 else -max_step
            else:
                self.current_pan = self._target_pan

            if abs(tilt_delta) > max_step:
                self.current_tilt += max_step if tilt_delta > 0 else -max_step
            else:
                self.current_tilt = self._target_tilt

            # Grace expiry: no cat has been reported in any zone for
            # engagement_grace_ms. Return to sweep (or to queued pause).
            grace_elapsed_ms = (now - self._last_in_zone_time) * 1000.0
            if grace_elapsed_ms >= self.engagement_grace_ms:
                if self.pause_queued:
                    self.state = SweepState.PAUSED
                    self.pause_queued = False
                else:
                    self.state = SweepState.SWEEPING

    def _clamp_target(self, cat_pan: float, cat_tilt: float) -> tuple[float, float]:
        """Clamp a tracking target to the operational envelope so the
        internal tracking state never drifts outside reachable bounds."""
        return (
            max(self.pan_min, min(self.pan_max, cat_pan)),
            max(self.tilt_min, min(self.tilt_max, cat_tilt)),
        )

    # ── Called by the vision loop ──────────────────────

    def on_cat_in_zone(self, cat_pan: float, cat_tilt: float, zone_name: str) -> None:
        """Called every frame that at least one cat's bbox center is inside
        an exclusion zone. Updates the target to the reported bbox center
        (clamped), records the in-zone timestamp, and transitions SWEEPING →
        ENGAGING on first entry."""
        self._target_pan, self._target_tilt = self._clamp_target(cat_pan, cat_tilt)
        self._target_zone = zone_name
        self._last_in_zone_time = self._time()
        if self.state == SweepState.SWEEPING:
            self.state = SweepState.ENGAGING

    def on_no_cat_in_zone(self) -> None:
        """Called on frames where no cat bbox overlaps any zone. Does not
        immediately transition — the grace timer in tick() handles that. Also
        called (harmlessly) in SWEEPING state when no cats are visible; it's a
        no-op there because the grace check only runs in ENGAGING."""
        # Deliberate no-op at the event level. State changes happen in tick().
        pass

    def should_fire(self) -> bool:
        """Returns True iff we are in ENGAGING, armed, and at least
        `min_shot_interval_ms` have elapsed since the last shot. The vision
        loop calls this every frame after `on_cat_in_zone`; if True, the
        loop executes the aim_and_fire and then calls `mark_shot_fired`."""
        if self.state != SweepState.ENGAGING or not self.armed:
            return False
        elapsed_ms = (self._time() - self._last_shot_time) * 1000.0
        return elapsed_ms >= self.min_shot_interval_ms

    def mark_shot_fired(self) -> None:
        """Record the timestamp of a shot that the vision loop just
        executed. Required for the min_shot_interval_ms gate to work."""
        self._last_shot_time = self._time()

    def on_fire_complete(
        self,
        fired_pan: float | None = None,
        fired_tilt: float | None = None,
    ) -> None:
        """Called after the solenoid has fired. If the vision loop commanded
        the servo directly (bypassing the smoothed tick path), pass that pose
        so internal current_pan/tilt stay in sync and the next tick doesn't
        yank the servo away from where it was just commanded."""
        if fired_pan is not None:
            self.current_pan = max(0.0, min(180.0, fired_pan))
        if fired_tilt is not None:
            self.current_tilt = max(0.0, min(180.0, fired_tilt))
        # Snap the tracking target to the fired pose so tick() has zero delta
        # to chase on the next iteration — otherwise it would re-advance toward
        # the previous target and cause a forward-snap after the fire.
        self._target_pan = self.current_pan
        self._target_tilt = self.current_tilt

    def get_direction_delta(self, target_pan: float, target_tilt: float) -> dict:
        """Angle delta between current position and target (for the dev-mode
        arrow overlay)."""
        return {
            "pan": target_pan - self.current_pan,
            "tilt": target_tilt - self.current_tilt,
        }

    def get_target(self) -> tuple[float, float, str]:
        """Current target pan/tilt and the name of the zone that triggered
        engagement (empty string if none)."""
        return (self._target_pan, self._target_tilt, self._target_zone)

    def set_virtual_angle(self, pan: float, tilt: float) -> None:
        """Set the virtual angle (dev mode - manual control)."""
        self.current_pan = max(self.pan_min, min(self.pan_max, pan))
        self.current_tilt = max(self.tilt_min, min(self.tilt_max, tilt))

    def set_pose_calibration(self, pan: float, tilt: float) -> None:
        """Set the servo pose during calibration mode. Clamps only to
        hardware range [0, 180] — not sweep bounds, because calibration may
        want to explore outside the normal sweep region."""
        if tilt > 75.0 or tilt < 15.0 or pan < 5.0 or pan > 175.0:
            import logging, traceback
            _log = logging.getLogger(__name__)
            _log.warning(
                f"Suspicious set_pose_calibration: pan={pan:.2f} tilt={tilt:.2f}\n"
                f"Stack trace of caller:\n{''.join(traceback.format_stack(limit=8)[:-1])}"
            )
        self.current_pan = max(0.0, min(180.0, pan))
        self.current_tilt = max(0.0, min(180.0, tilt))
```

- [ ] **Step 2: Run the sweep controller tests**

Run: `pytest server/tests/test_sweep_controller.py -v`

Expected: all tests PASS.

If any test fails, read the failure carefully before editing code — the test is the spec.

- [ ] **Step 3: Check that nothing else in the repo imports the removed names**

Run: `grep -rn "SweepState\.\(TRACKING\|WARNING\|FIRING\)" server/ frontend/src/` — expect results in `main.py` (which we'll fix in Task 8), and in `routers/stream.py` or similar if they serialize the enum value for the frontend. Note the hit list for Task 8.

Run: `grep -rn "on_cat_detected\|on_cat_not_in_zone\|on_no_cat_detected\|on_cat_still_in_zone\|on_cat_left_zone\|get_warning_remaining\|warning_duration\|tracking_duration\|reentry_warning\|lock_on_grace" server/` — expect hits in `main.py` and possibly `config.py`. Note them for Task 8 and Task 7.

- [ ] **Step 4: Commit**

```bash
git add server/panorama/sweep_controller.py
git commit -m "feat(sweep_controller): collapse to 4-state engagement machine"
```

The live vision loop in `main.py` still references the old API at this point, so the app will not start cleanly until Task 8. That's intentional — the state-machine refactor is self-contained and tested, and the vision-loop rewrite is its own task.

---

## Phase 2: Simplify the zone checker

### Task 3: Rewrite `test_zone_checker.py` to cover only 2D overlap

**Files:**
- Modify: `server/tests/test_zone_checker.py`

- [ ] **Step 1: Read the current test file to see which 2D tests are worth keeping**

Run: `cat server/tests/test_zone_checker.py`

Identify the tests that exercise `check_zone_violations` with 2D bboxes and angle-space polygons (mode="2d" or no mode). Those stay (with the `mode` field stripped from test fixtures). Tests that reference `check_3d_zone_violation`, `room_polygon`, `height_min`, `height_max`, `cat_room_pos`, `auto_3d`, `manual_3d` get deleted.

- [ ] **Step 2: Rewrite the file**

Write the new contents:

```python
import pytest
from server.vision.zone_checker import check_zone_violations, invalidate_poly_cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the poly cache between tests so zones with reused ids don't leak
    stale Shapely Polygon objects across test boundaries."""
    invalidate_poly_cache()
    yield
    invalidate_poly_cache()


def _zone(zone_id: str, polygon: list, overlap_threshold: float = 0.3, enabled: bool = True, name: str = "Z"):
    return {
        "id": zone_id,
        "name": name,
        "polygon": polygon,
        "overlap_threshold": overlap_threshold,
        "enabled": enabled,
    }


def test_bbox_fully_inside_zone_is_violation():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]])
    violations = check_zone_violations([10, 10, 50, 50], [zone])
    assert len(violations) == 1
    assert violations[0]["zone_id"] == "z1"
    assert violations[0]["overlap"] == pytest.approx(1.0)


def test_bbox_fully_outside_zone_is_not_violation():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]])
    violations = check_zone_violations([200, 200, 250, 250], [zone])
    assert violations == []


def test_bbox_partial_overlap_above_threshold():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]], overlap_threshold=0.25)
    # Bbox is 100x100 starting at (50, 50), so half of it overlaps the zone.
    violations = check_zone_violations([50, 50, 150, 150], [zone])
    assert len(violations) == 1
    assert violations[0]["overlap"] == pytest.approx(0.25, abs=0.01)


def test_bbox_partial_overlap_below_threshold():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]], overlap_threshold=0.5)
    # Same 25% overlap as above, but the threshold is 0.5 this time.
    violations = check_zone_violations([50, 50, 150, 150], [zone])
    assert violations == []


def test_disabled_zone_is_ignored():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]], enabled=False)
    violations = check_zone_violations([10, 10, 50, 50], [zone])
    assert violations == []


def test_zero_area_bbox_returns_empty():
    zone = _zone("z1", [[0, 0], [0, 100], [100, 100], [100, 0]])
    violations = check_zone_violations([10, 10, 10, 10], [zone])
    assert violations == []


def test_degenerate_polygon_is_skipped():
    zone = _zone("z1", [[0, 0], [1, 0]])  # only 2 points, not a polygon
    violations = check_zone_violations([10, 10, 50, 50], [zone])
    assert violations == []


def test_multiple_zones_each_checked_independently():
    zone_a = _zone("a", [[0, 0], [0, 50], [50, 50], [50, 0]], name="A")
    zone_b = _zone("b", [[100, 100], [100, 150], [150, 150], [150, 100]], name="B")
    violations = check_zone_violations([10, 10, 40, 40], [zone_a, zone_b])
    assert len(violations) == 1
    assert violations[0]["zone_id"] == "a"


def test_self_intersecting_polygon_is_rebuilt_via_make_valid():
    """Shapely's make_valid() is called when a raw polygon is invalid. The
    zone checker must not crash on a bowtie or figure-eight outline."""
    bowtie = [[0, 0], [100, 100], [100, 0], [0, 100]]
    zone = _zone("z1", bowtie)
    # Call should not raise; whether there's a violation depends on how
    # make_valid rebuilds the polygon, so we only assert the call succeeds.
    check_zone_violations([10, 10, 50, 50], [zone])
```

- [ ] **Step 3: Commit**

```bash
git add server/tests/test_zone_checker.py
git commit -m "test(zone_checker): rewrite for 2D-only overlap"
```

---

### Task 4: Simplify `zone_checker.py`

**Files:**
- Modify: `server/vision/zone_checker.py`

- [ ] **Step 1: Replace the file**

Write:

```python
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

# ── Pre-cached Shapely Polygon objects ──
# Keyed by zone id -> Polygon. Rebuilt when zone data changes via
# invalidate_poly_cache().
_poly_cache: dict[str, Polygon] = {}


def invalidate_poly_cache() -> None:
    """Called when zones are modified so the next overlap check rebuilds
    Shapely polygons from the new coordinate data."""
    _poly_cache.clear()


def _get_cached_poly(zone: dict) -> Polygon | None:
    """Return a validated Shapely Polygon for a zone, building and caching
    it on first access. Degenerate or empty polygons return None so the
    caller can skip them without raising."""
    zone_id = zone["id"]
    cached = _poly_cache.get(zone_id)
    if cached is not None:
        return cached
    coords = zone.get("polygon")
    if not coords or len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return None
    _poly_cache[zone_id] = poly
    return poly


def check_zone_violations(bbox: list[float], zones: list[dict]) -> list[dict]:
    """Return the list of zones whose overlap with the given bbox meets or
    exceeds the zone's `overlap_threshold`. Each violation is a dict with
    `zone_id`, `zone_name`, and `overlap` (fraction of bbox area inside zone).

    Zones marked `enabled=False` are skipped. Degenerate bboxes (zero area)
    return an empty list.
    """
    cat_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
    cat_area = cat_box.area
    if cat_area == 0:
        return []
    violations = []
    for zone in zones:
        if not zone.get("enabled", True):
            continue
        zone_poly = _get_cached_poly(zone)
        if zone_poly is None or zone_poly.area == 0:
            continue
        intersection = cat_box.intersection(zone_poly)
        overlap = intersection.area / cat_area
        if overlap >= zone.get("overlap_threshold", 0.3):
            violations.append({
                "zone_id": zone["id"],
                "zone_name": zone["name"],
                "overlap": overlap,
            })
    return violations
```

- [ ] **Step 2: Run the zone checker tests**

Run: `pytest server/tests/test_zone_checker.py -v`

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add server/vision/zone_checker.py
git commit -m "refactor(zone_checker): remove 3D/mode branches, simplify cache"
```

---

## Phase 3: Simplify the database schema

### Task 5: Rewrite `zones` and `furniture` tables in `server/models/database.py`

**Files:**
- Modify: `server/models/database.py`
- Modify: `server/tests/test_database.py`

**Context:** The spec says wipe and recreate. On the next `init_db()` call, the tables are dropped and recreated with the new schemas. Existing rows are lost — this is explicit user-approved behavior.

- [ ] **Step 1: Read the current `database.py` zones + furniture schema and CRUD helpers**

Run: `grep -n "CREATE TABLE zones\|CREATE TABLE furniture\|def .*zone\|def .*furniture" server/models/database.py`

Note line numbers for the `CREATE TABLE zones`, `CREATE TABLE furniture`, `create_zone`, `get_zones`, `update_zone`, `delete_zone`, `create_furniture`, `get_furniture`, `update_furniture`, `delete_furniture` blocks. Read each one so your edit preserves the surrounding function signatures that callers depend on.

- [ ] **Step 2: Edit the `init_db` SQL — drop-and-recreate both tables**

Inside `init_db()` (or wherever the `CREATE TABLE` SQL runs), replace the zones and furniture table creation with:

```python
await conn.execute("DROP TABLE IF EXISTS zones CASCADE")
await conn.execute("""
    CREATE TABLE zones (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        polygon JSONB NOT NULL,
        overlap_threshold REAL NOT NULL DEFAULT 0.3,
        enabled BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
""")

await conn.execute("DROP TABLE IF EXISTS furniture CASCADE")
await conn.execute("""
    CREATE TABLE furniture (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        polygon JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
""")
```

The `DROP TABLE IF EXISTS ... CASCADE` ensures that any leftover foreign keys from the old 3D schema (e.g. `zones.furniture_id → furniture.id`) are cleaned up.

- [ ] **Step 3: Simplify `create_zone` / `update_zone` / `get_zones`**

`create_zone` should accept `(name, polygon, overlap_threshold=0.3, enabled=True)` and insert only those columns. Remove any `mode`, `room_polygon`, `height_min`, `height_max`, `furniture_id`, `cooldown_seconds` parameters and SQL references.

`update_zone` should take `(zone_id, name=None, polygon=None, overlap_threshold=None, enabled=None)` and build an `UPDATE ... SET ...` clause for whichever fields are non-None.

`get_zones()` should `SELECT id, name, polygon, overlap_threshold, enabled, created_at FROM zones` and return dicts in that shape.

`delete_zone(zone_id)` is unchanged.

- [ ] **Step 4: Simplify `create_furniture` / `get_furniture` / `delete_furniture`**

`create_furniture` should accept `(name, polygon)` only.

`get_furniture()` should `SELECT id, name, polygon, created_at FROM furniture` and return dicts.

`delete_furniture(furniture_id)` is unchanged.

Remove any `update_furniture` helper entirely — the new UX is delete-and-redraw.

- [ ] **Step 5: Update `test_database.py`**

Read: `cat server/tests/test_database.py`

For every test that creates a zone or furniture row, strip out the 3D fields from the call. Any test that exercises `update_furniture` or the `zones.furniture_id` relationship should be deleted.

- [ ] **Step 6: Run the database tests**

Run: `pytest server/tests/test_database.py -v`

Expected: all tests PASS. If a test fails because it expects a column that no longer exists, update the test assertion (not the schema).

- [ ] **Step 7: Commit**

```bash
git add server/models/database.py server/tests/test_database.py
git commit -m "feat(database): simplify zones and furniture schemas to 2D-only"
```

---

### Task 6: Update `routers/zones.py` and furniture endpoints

**Files:**
- Modify: `server/routers/zones.py`
- Modify: whichever router handles furniture CRUD (verify via: `grep -l "create_furniture\|get_furniture\|delete_furniture" server/routers/`)

- [ ] **Step 1: Strip 3D fields from zone request/response models**

Read the existing Pydantic models. The `ZoneCreate`, `ZoneUpdate`, and `ZoneResponse` schemas (or equivalent names) should keep `id`, `name`, `polygon`, `overlap_threshold`, `enabled`, `created_at`, and drop `mode`, `room_polygon`, `height_min`, `height_max`, `furniture_id`, `cooldown_seconds`.

After editing, the POST /zones handler should accept only the simplified fields and pass them to `create_zone`. The PUT /zones/{id} handler should accept the simplified fields and pass them to `update_zone`. The GET handlers return the new dict shape from `get_zones`.

Also call `invalidate_poly_cache()` from the zone create/update/delete handlers (it's probably already wired — just confirm it's still imported from `server.vision.zone_checker`).

- [ ] **Step 2: Strip 3D fields from furniture request/response models**

`FurnitureCreate` / `FurnitureResponse` keep `id`, `name`, `polygon`, `created_at` and drop `base_polygon` (renamed to `polygon`), `height_min`, `height_max`, `depth_anchored`.

Delete any `FurnitureUpdate` schema and the corresponding PUT handler. If the furniture router exports a `move_furniture` endpoint, delete it too.

- [ ] **Step 3: Run the router tests if they exist**

Run: `pytest server/tests/ -v -k "zones or furniture"` — if any test uses the 3D fields it will fail with a Pydantic validation error. Strip the 3D fields from those tests.

- [ ] **Step 4: Commit**

```bash
git add server/routers/zones.py server/routers/*.py server/tests/
git commit -m "refactor(routers): drop 3D fields from zones and furniture APIs"
```

---

## Phase 4: Config cleanup

### Task 7: Update `server/config.py`

**Files:**
- Modify: `server/config.py`

- [ ] **Step 1: Replace the relevant sections**

In `server/config.py`, do these edits inside the `Settings` class body:

Delete the `cooldown_default`, `warning_duration`, `tracking_duration`, `reentry_warning`, `lock_on_grace` fields from the `# Sweep / Panorama` block.

Delete the entire `# Depth / Spatial` block:

```python
# Depth / Spatial
midas_model: str = "MiDaS_small"
depth_run_interval: int = 5
depth_blend_alpha: float = 0.2
depth_change_threshold: float = 20.0
heightmap_resolution: float = 5.0
room_width_cm: float = 500.0
room_depth_cm: float = 500.0
room_height_cm: float = 300.0
camera_height_cm: float = 150.0
occlusion_timeout: float = 10.0
occlusion_grace_frames: int = 3
```

Add a new `# Engagement` block somewhere after `# Sweep / Panorama`:

```python
# Engagement
min_shot_interval_ms: int = 2000  # minimum time between successive shots while in ENGAGING
engagement_grace_ms: int = 3000   # time with no cat-in-zone before ENGAGING → SWEEPING
```

- [ ] **Step 2: Confirm nothing else imports the removed fields**

Run: `grep -rn "settings\.\(midas_model\|depth_run_interval\|depth_blend_alpha\|depth_change_threshold\|heightmap_resolution\|room_width_cm\|room_depth_cm\|room_height_cm\|camera_height_cm\|occlusion_timeout\|occlusion_grace_frames\|cooldown_default\|warning_duration\|tracking_duration\|reentry_warning\|lock_on_grace\)" server/`

Every hit listed here will need to be fixed in Task 8 (`main.py`) or in the router cleanup. Note the hit list.

- [ ] **Step 3: Commit**

```bash
git add server/config.py
git commit -m "chore(config): remove MiDaS/room settings, add engagement settings"
```

The app will not start at this point — `main.py` still imports `settings.midas_model` etc. Task 8 fixes it.

---

## Phase 5: Vision loop rewrite and dead-code purge

### Task 8: Rewrite `main.py:run_vision_loop` for the new state machine and delete all 3D code paths

**Files:**
- Modify: `server/main.py`

**Context:** This is the biggest single edit. The changes:

1. Delete imports: `torch` (if unused elsewhere — check first), `DepthEstimator`, `RoomModel`, `FurnitureObject`, `angle_depth_to_room`, `CatTracker`, `spatial` router.
2. Delete global state: `_room_model`, `_depth_estimator`, `_cat_tracker`, `get_room_model`, `get_cat_tracker`.
3. Inside `run_vision_loop`:
   - Delete `_depth_estimator`, `_room_model`, `_cat_tracker` init (lines ~195-205).
   - Delete persisted-furniture loading block (~213-223) — simplified furniture has no runtime room-model representation.
   - Delete `depth_frame_counter`, `needs_depth` locals.
   - Inside the inference worker, delete the `if do_depth` block (~315-322).
   - In the main frame processing block, delete the `cat_room_pos` calculation block (~500-518).
   - Replace the `check_zone_violations(angle_bbox, current_zones, cat_room_pos=cat_room_pos)` call with `check_zone_violations(angle_bbox, current_zones)`.
   - Delete the `has_3d_zones` / `needs_depth` detection.
   - Delete `_cat_tracker.tick` / `_cat_tracker.cleanup_lost` / occluded predictions block (~600-614).
   - Rewrite the zone violation → sweep controller interaction to use the new API (see Step 3 below).
4. Remove the `spatial` router mount (line 702) and its import (line 17).
5. Update the broadcast payload: drop `warning_remaining`, `occluded_cats`; keep `state`, `direction_delta`, `servo_pan`, `servo_tilt`.

The following steps give the exact diffs. Apply them in order.

- [ ] **Step 1: Delete 3D imports**

In `server/main.py`, delete these import lines (lines ~11, 17, 34-37):

```python
import torch
```

(Delete `import torch` entirely — it was only used implicitly through MiDaS.)

```python
from server.routers import zones, cats, events, control, stream, settings as settings_router, spatial
```

Change to:

```python
from server.routers import zones, cats, events, control, stream, settings as settings_router
```

```python
from server.spatial.depth_estimator import DepthEstimator
from server.spatial.room_model import RoomModel, FurnitureObject
from server.spatial.projection import angle_depth_to_room
from server.spatial.cat_tracker import CatTracker
```

Delete all four of those imports entirely.

- [ ] **Step 2: Delete 3D global state and getters**

Delete these module-level globals (around line 45-47):

```python
_room_model = None
_depth_estimator: DepthEstimator | None = None
_cat_tracker: CatTracker | None = None
```

Delete these getter functions (around line 65-70):

```python
def get_room_model():
    return _room_model


def get_cat_tracker() -> CatTracker | None:
    return _cat_tracker
```

- [ ] **Step 3: Rewrite the `run_vision_loop` — sweep controller construction**

Find the `SweepController(...)` instantiation (around line 180) and replace its kwargs:

Old:

```python
_sweep_controller = SweepController(
    actuator=_actuator,
    pan_min=pan_min,
    pan_max=pan_max,
    tilt_min=tilt_min,
    tilt_max=tilt_max,
    speed=settings.sweep_speed,
    warning_duration=settings.warning_duration,
    tracking_duration=settings.tracking_duration,
    cooldown=settings.cooldown_default,
    reentry_warning=settings.reentry_warning,
    lock_on_grace=settings.lock_on_grace,
    dev_mode=settings.dev_mode,
)
```

New:

```python
_sweep_controller = SweepController(
    actuator=_actuator,
    pan_min=pan_min,
    pan_max=pan_max,
    tilt_min=tilt_min,
    tilt_max=tilt_max,
    speed=settings.sweep_speed,
    min_shot_interval_ms=settings.min_shot_interval_ms,
    engagement_grace_ms=settings.engagement_grace_ms,
    dev_mode=settings.dev_mode,
)
```

- [ ] **Step 4: Delete 3D init blocks**

Delete everything between the `_sweep_controller = SweepController(...)` construction and the `last_time = time.time()` line. Specifically, delete:

```python
_depth_estimator = DepthEstimator(model_type=settings.midas_model)
_room_model = RoomModel(
    width_cm=settings.room_width_cm,
    depth_cm=settings.room_depth_cm,
    height_cm=settings.room_height_cm,
    resolution=settings.heightmap_resolution,
)
_cat_tracker = CatTracker(
    occlusion_timeout=settings.occlusion_timeout,
    grace_frames=settings.occlusion_grace_frames,
)
```

Keep the `_classifier = CatClassifier()` / `_classifier.load(...)` block.

Delete:

```python
depth_frame_counter = 0
needs_depth = False  # updated each frame based on whether 3D zones exist
```

Delete the persisted-furniture loading block:

```python
# Load persisted furniture into room model
from server.models.database import get_furniture as db_get_furniture
persisted_furniture = await db_get_furniture()
for f in persisted_furniture:
    _room_model.add_furniture(FurnitureObject(
        id=f["id"],
        name=f["name"],
        base_polygon=[tuple(p) for p in f["base_polygon"]],
        height_min=f["height_min"],
        height_max=f["height_max"],
        depth_anchored=f["depth_anchored"],
    ))
```

- [ ] **Step 5: Simplify the inference worker job and state**

The inference worker currently carries `(inf_frame, inf_servo_pan, inf_servo_tilt, do_depth, do_classify)`. Remove `do_depth` and the `_latest_depth` field.

Replace the inference job shape. Change:

```python
_latest_raw_dets: list[dict] = []
_latest_depth: np.ndarray | None = None
_latest_inf_pan: float = 0.0
_latest_inf_tilt: float = 0.0
_new_results_ready = False
```

To:

```python
_latest_raw_dets: list[dict] = []
_latest_inf_pan: float = 0.0
_latest_inf_tilt: float = 0.0
_new_results_ready = False
```

Inside `_inference_worker`, change:

```python
while not _infer_stop.is_set():
    try:
        job = _infer_queue.get(timeout=0.5)
    except queue.Empty:
        continue
    inf_frame, inf_servo_pan, inf_servo_tilt, do_depth, do_classify = job
```

To:

```python
while not _infer_stop.is_set():
    try:
        job = _infer_queue.get(timeout=0.5)
    except queue.Empty:
        continue
    inf_frame, inf_servo_pan, inf_servo_tilt, do_classify = job
```

Delete the depth estimation block inside the worker:

```python
# Depth estimation
depth_result = None
if do_depth:
    try:
        depth_result = _depth_estimator.estimate(inf_frame)
    except Exception as e:
        logger.warning(f"Depth estimation failed: {e}")
```

Inside the `with _infer_lock:` block, delete `_latest_depth = depth_result`.

- [ ] **Step 6: Update the job submission call**

Change:

```python
do_classify = (_classify_frame_counter % settings.classify_every_n_frames == 0)
do_depth = needs_depth and (depth_frame_counter % settings.depth_run_interval == 0)
# Copy frame so the worker thread owns its data; drop if queue full
try:
    _infer_queue.put_nowait(
        (frame.copy(), servo_pan, servo_tilt, do_depth, do_classify)
    )
except queue.Full:
    pass  # worker still busy — skip this frame
```

To:

```python
do_classify = (_classify_frame_counter % settings.classify_every_n_frames == 0)
# Copy frame so the worker thread owns its data; drop if queue full
try:
    _infer_queue.put_nowait(
        (frame.copy(), servo_pan, servo_tilt, do_classify)
    )
except queue.Full:
    pass  # worker still busy — skip this frame
```

Also delete `depth_frame_counter += 1`.

- [ ] **Step 7: Simplify the consume-results block**

Change:

```python
raw_detections = None
current_depth = None
inf_pose_pan = servo_pan   # fallback if no new results this iteration
inf_pose_tilt = servo_tilt
with _infer_lock:
    if _new_results_ready:
        raw_detections = _latest_raw_dets
        current_depth = _latest_depth
        inf_pose_pan = _latest_inf_pan
        inf_pose_tilt = _latest_inf_tilt
        # Clear references so GC can reclaim detection/depth data
        _latest_raw_dets = []
        _latest_depth = None
        _new_results_ready = False
```

To:

```python
raw_detections = None
inf_pose_pan = servo_pan   # fallback if no new results this iteration
inf_pose_tilt = servo_tilt
with _infer_lock:
    if _new_results_ready:
        raw_detections = _latest_raw_dets
        inf_pose_pan = _latest_inf_pan
        inf_pose_tilt = _latest_inf_tilt
        _latest_raw_dets = []
        _new_results_ready = False
```

- [ ] **Step 8: Rewrite the per-detection zone check and fire flow**

The old block (around lines 470-600) does: current zones fetch → per-detection pixel-to-angle + optional cat_room_pos → `check_zone_violations` → `on_cat_detected` / `on_cat_in_zone` / `should_fire` → fire. Rewrite it as:

Old:

```python
# Convert detections to angle-space and check zones
current_zones = await get_zones()
has_3d_zones = any(z.get("mode") in ("auto_3d", "manual_3d") for z in current_zones)
needs_depth = has_3d_zones
all_violations = []
fired = False
fire_target = None
direction_delta = None

# Pick the highest-confidence detection as the primary cat
best_det = None
best_cat_pan = 0.0
best_cat_tilt = 0.0

for det in detections:
    bbox = det["bbox"]
    det_pose_pan = det["__pose_pan"]
    det_pose_tilt = det["__pose_tilt"]
    pan1, tilt1 = calibrated_pixel_to_angle(bbox[0], bbox[1], det_pose_pan, det_pose_tilt)
    pan2, tilt2 = calibrated_pixel_to_angle(bbox[2], bbox[3], det_pose_pan, det_pose_tilt)
    angle_bbox = [pan1, tilt1, pan2, tilt2]
    cat_pan = (pan1 + pan2) / 2
    cat_tilt = (tilt1 + tilt2) / 2

    if best_det is None or det["confidence"] > best_det["confidence"]:
        best_det = det
        best_cat_pan = cat_pan
        best_cat_tilt = cat_tilt

    # Project cat to room-space if depth available
    cat_room_pos = None
    if current_depth is not None:
        cat_center_x = (bbox[0] + bbox[2]) / 2
        cat_center_y = (bbox[1] + bbox[3]) / 2
        px = int(cat_center_x * current_depth.shape[1])
        py = int(cat_center_y * current_depth.shape[0])
        px = max(0, min(px, current_depth.shape[1] - 1))
        py = max(0, min(py, current_depth.shape[0] - 1))
        rel_depth = float(current_depth[py, px])
        if rel_depth > 0:
            metric_depth = _depth_estimator.depth_scale / rel_depth
            camera_pos = (0.0, 0.0, settings.camera_height_cm)
            cat_room_pos = angle_depth_to_room(
                cat_pan, cat_tilt, metric_depth, camera_pos
            )
            cat_id = det.get("cat_name", f"cat_{id(det)}")
            _cat_tracker.update_detection(cat_id, cat_room_pos, time.time())

    violations = check_zone_violations(angle_bbox, current_zones, cat_room_pos=cat_room_pos)
    all_violations.extend(violations)

# Tell the controller about any detected cat — stops sweep, begins tracking.
if best_det is not None:
    best_cat_pan = max(
        _sweep_controller.pan_min,
        min(_sweep_controller.pan_max, best_cat_pan),
    )
    best_cat_tilt = max(
        _sweep_controller.tilt_min,
        min(_sweep_controller.tilt_max, best_cat_tilt),
    )
    _sweep_controller.on_cat_detected(best_cat_pan, best_cat_tilt)

# Now handle zone violations for the warning/firing flow
if all_violations:
    zone_name = all_violations[0]["zone_name"]
    v_det = best_det

    _sweep_controller.on_cat_in_zone(best_cat_pan, best_cat_tilt, zone_name)

    if _sweep_controller.should_fire():
        bbox = v_det["bbox"]
        if settings.dev_mode:
            logger.info(f"DEV ZAP! Cat in {zone_name}")
            fired = True
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            fire_target = {"x": center_x, "y": center_y, "zone": zone_name}
        else:
            await _actuator.goto(best_cat_pan, best_cat_tilt)
            _last_sent_pan_int = snap_to_servo_step(best_cat_pan)
            _last_sent_tilt_int = snap_to_servo_step(best_cat_tilt)
            await asyncio.sleep(0.2)
            success = await _actuator.fire()
            fired = success

        _sweep_controller.on_fire_complete(
            fired_pan=best_cat_pan if not settings.dev_mode else None,
            fired_tilt=best_cat_tilt if not settings.dev_mode else None,
        )
        asyncio.create_task(_log_event({
            "type": "ZAP",
            "cat_name": v_det.get("cat_name"),
            "zone_name": zone_name,
            "confidence": v_det["confidence"],
            "overlap": all_violations[0]["overlap"],
            "servo_pan": best_cat_pan,
            "servo_tilt": best_cat_tilt,
        }))
elif best_det is not None:
    _sweep_controller.on_cat_not_in_zone()

if not detections:
    _sweep_controller.on_no_cat_detected()
```

New:

```python
# Convert detections to angle-space and check zones
current_zones = await get_zones()
all_violations = []
fired = False
fire_target = None
direction_delta = None

# Find the cat whose bbox overlaps a zone and is closest to the current
# servo aim — that's the engagement target. Cats outside all zones are
# ignored entirely; their detections never touch the sweep controller.
engagement_target = None  # (cat_pan, cat_tilt, zone_name, det, overlap)
current_aim_pan = _sweep_controller.current_pan
current_aim_tilt = _sweep_controller.current_tilt
best_distance = float("inf")

for det in detections:
    bbox = det["bbox"]
    det_pose_pan = det["__pose_pan"]
    det_pose_tilt = det["__pose_tilt"]
    pan1, tilt1 = calibrated_pixel_to_angle(bbox[0], bbox[1], det_pose_pan, det_pose_tilt)
    pan2, tilt2 = calibrated_pixel_to_angle(bbox[2], bbox[3], det_pose_pan, det_pose_tilt)
    angle_bbox = [pan1, tilt1, pan2, tilt2]
    cat_pan = (pan1 + pan2) / 2
    cat_tilt = (tilt1 + tilt2) / 2

    violations = check_zone_violations(angle_bbox, current_zones)
    all_violations.extend(violations)
    if not violations:
        continue

    # Cat is in at least one zone — evaluate as engagement target candidate.
    clamped_pan = max(
        _sweep_controller.pan_min,
        min(_sweep_controller.pan_max, cat_pan),
    )
    clamped_tilt = max(
        _sweep_controller.tilt_min,
        min(_sweep_controller.tilt_max, cat_tilt),
    )
    dist = (
        (clamped_pan - current_aim_pan) ** 2
        + (clamped_tilt - current_aim_tilt) ** 2
    )
    if dist < best_distance:
        best_distance = dist
        engagement_target = (
            clamped_pan,
            clamped_tilt,
            violations[0]["zone_name"],
            det,
            violations[0]["overlap"],
        )

if engagement_target is not None:
    tgt_pan, tgt_tilt, zone_name, v_det, overlap = engagement_target
    _sweep_controller.on_cat_in_zone(tgt_pan, tgt_tilt, zone_name)

    if _sweep_controller.should_fire():
        if settings.dev_mode:
            logger.info(f"DEV ZAP! Cat in {zone_name}")
            fired = True
            bbox = v_det["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            fire_target = {"x": center_x, "y": center_y, "zone": zone_name}
        else:
            await _actuator.goto(tgt_pan, tgt_tilt)
            _last_sent_pan_int = snap_to_servo_step(tgt_pan)
            _last_sent_tilt_int = snap_to_servo_step(tgt_tilt)
            await asyncio.sleep(0.2)
            success = await _actuator.fire()
            fired = success

        _sweep_controller.mark_shot_fired()
        _sweep_controller.on_fire_complete(
            fired_pan=tgt_pan if not settings.dev_mode else None,
            fired_tilt=tgt_tilt if not settings.dev_mode else None,
        )
        asyncio.create_task(_log_event({
            "type": "ZAP",
            "cat_name": v_det.get("cat_name"),
            "zone_name": zone_name,
            "confidence": v_det["confidence"],
            "overlap": overlap,
            "servo_pan": tgt_pan,
            "servo_tilt": tgt_tilt,
        }))
else:
    _sweep_controller.on_no_cat_in_zone()
```

- [ ] **Step 9: Update the direction-delta computation**

Change:

```python
if settings.dev_mode and best_det is not None and _sweep_controller.state in (SweepState.WARNING, SweepState.FIRING, SweepState.TRACKING):
    direction_delta = _sweep_controller.get_direction_delta(best_cat_pan, best_cat_tilt)
```

To:

```python
if settings.dev_mode and engagement_target is not None and _sweep_controller.state == SweepState.ENGAGING:
    tgt_pan, tgt_tilt, _, _, _ = engagement_target
    direction_delta = _sweep_controller.get_direction_delta(tgt_pan, tgt_tilt)
```

- [ ] **Step 10: Delete the cat tracker tick and occluded-prediction block**

Delete:

```python
# Tick cat tracker and prune lost entries
now_t = time.time()
_cat_tracker.tick(now_t)
_cat_tracker.cleanup_lost(max_age=60.0, current_time=now_t)

# Build predicted positions for occluded cats
occluded_predictions = []
for ocat in _cat_tracker.get_occluded_cats():
    pred = _cat_tracker.predict_position(ocat.id, now_t)
    if pred:
        occluded_predictions.append({
            "id": ocat.id,
            "predicted": pred,
            "occluded_by": ocat.occluded_by,
        })
```

- [ ] **Step 11: Simplify the broadcast payload**

Change:

```python
await broadcast_to_clients({
    "frame": frame_b64,
    "panorama": cached_pano_b64,
    "detections": detections,
    "violations": all_violations,
    "fired": fired,
    "fire_target": fire_target,
    "state": _sweep_controller.state.value,
    "servo_pan": snap_to_servo_step(servo_pan),
    "servo_tilt": snap_to_servo_step(servo_tilt),
    "warning_remaining": _sweep_controller.get_warning_remaining(),
    "direction_delta": direction_delta,
    "occluded_cats": occluded_predictions,
})
```

To:

```python
await broadcast_to_clients({
    "frame": frame_b64,
    "panorama": cached_pano_b64,
    "detections": detections,
    "violations": all_violations,
    "fired": fired,
    "fire_target": fire_target,
    "state": _sweep_controller.state.value,
    "servo_pan": snap_to_servo_step(servo_pan),
    "servo_tilt": snap_to_servo_step(servo_tilt),
    "direction_delta": direction_delta,
})
```

- [ ] **Step 12: Remove the spatial router mount**

Delete:

```python
app.include_router(spatial.router)
```

- [ ] **Step 13: Run the server import / smoke test**

Run: `python -c "from server.main import app; print('ok')"`

Expected: `ok` (no ImportError, no NameError). If any import fails, fix the remaining dangling reference (most likely a missed occurrence of `_depth_estimator` or `SweepState.WARNING`).

Run: `pytest server/tests/ -v --ignore=server/tests/test_depth_estimator.py --ignore=server/tests/test_volumetric_zones.py --ignore=server/tests/test_room_model.py --ignore=server/tests/test_projection.py --ignore=server/tests/test_cat_tracker.py --ignore=server/tests/test_pipeline.py`

Expected: all non-ignored tests PASS. (We ignore the tests that will be deleted in Task 10.)

- [ ] **Step 14: Commit**

```bash
git add server/main.py
git commit -m "feat(main): rewrite vision loop for 2D zones and engagement state machine"
```

---

### Task 9: Delete dead-code files (`server/spatial/`, `server/routers/spatial.py`, `server/vision/pipeline.py`, and their tests)

**Files:**
- Delete: `server/spatial/` (entire directory)
- Delete: `server/routers/spatial.py`
- Delete: `server/vision/pipeline.py`
- Delete: `server/tests/test_pipeline.py`
- Delete: `server/tests/test_depth_estimator.py`
- Delete: `server/tests/test_volumetric_zones.py`
- Delete: `server/tests/test_room_model.py`
- Delete: `server/tests/test_projection.py`
- Delete: `server/tests/test_cat_tracker.py`

- [ ] **Step 1: Delete the files**

```bash
rm -rf server/spatial/
rm server/routers/spatial.py
rm server/vision/pipeline.py
rm server/tests/test_pipeline.py
rm server/tests/test_depth_estimator.py
rm server/tests/test_volumetric_zones.py
rm server/tests/test_room_model.py
rm server/tests/test_projection.py
rm server/tests/test_cat_tracker.py
```

- [ ] **Step 2: Confirm nothing else imports any of them**

Run: `grep -rn "from server.spatial\|import server.spatial\|from server.vision.pipeline\|from server.routers.spatial" server/ frontend/`

Expected: zero hits. If any hit remains, fix it (likely a stale import in a file you didn't touch).

- [ ] **Step 3: Run the full backend test suite**

Run: `pytest server/tests/ -v`

Expected: all tests PASS. No collection errors from deleted test modules.

- [ ] **Step 4: Commit**

```bash
git add -A server/spatial/ server/routers/spatial.py server/vision/pipeline.py server/tests/
git commit -m "chore: delete 3D/depth/pipeline dead code and their tests"
```

---

## Phase 6: Frontend type and API cleanup

### Task 10: Update `frontend/src/types/index.ts`

**Files:**
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Read the current file**

Run: `cat frontend/src/types/index.ts`

Identify the `Zone`, `Furniture`, and `ZoneTransform` interface blocks.

- [ ] **Step 2: Rewrite the Zone interface**

Find the `Zone` interface. Replace it with:

```typescript
export interface Zone {
  id: string;
  name: string;
  polygon: [number, number][];  // [pan, tilt] pairs in servo degrees
  overlap_threshold: number;
  enabled: boolean;
  created_at: string;
}
```

- [ ] **Step 3: Rewrite the Furniture interface**

Find the `Furniture` interface. Replace it with:

```typescript
export interface Furniture {
  id: string;
  name: string;
  polygon: [number, number][];  // [pan, tilt] pairs in servo degrees
  created_at: string;
}
```

- [ ] **Step 4: Delete the `ZoneTransform` interface**

Search the file for `ZoneTransform` or `interface ZoneTransform`. Delete the entire interface block.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/types/index.ts
git commit -m "refactor(types): strip 3D fields from Zone and Furniture"
```

---

### Task 11: Update `frontend/src/api/client.ts`

**Files:**
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Find all 3D / spatial API calls**

Run: `grep -n "estimateHeight\|/spatial\|height_min\|height_max\|room_polygon\|furniture_id\|cooldown_seconds\|mode.*auto_3d\|ZoneTransform" frontend/src/api/client.ts`

- [ ] **Step 2: Delete the `estimateHeight` function and any `/spatial/*` helper**

Remove any function named `estimateHeight`, `getFurnitureHeights`, `getRoomModel`, or anything else that hits a `/spatial/...` endpoint. Remove their type signatures from any exports.

- [ ] **Step 3: Update `createZone` / `updateZone` payloads**

The POST / PUT bodies should include only `name`, `polygon`, `overlap_threshold`, `enabled`. Remove any `mode`, `room_polygon`, `height_min`, `height_max`, `furniture_id`, `cooldown_seconds` fields from the request body type and the function arguments.

- [ ] **Step 4: Update `createFurniture` payload**

Should take only `{ name, polygon }`. Remove `height_min`, `height_max`, `depth_anchored`, `base_polygon`. Remove any `updateFurniture` or `moveFurniture` function.

- [ ] **Step 5: Run the frontend type check**

Run: `cd frontend && npx tsc --noEmit`

Expected: type errors from other components that still reference the old fields. Note each error — they are the exact hit list for Tasks 12–14.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "refactor(api-client): drop 3D fields from zone and furniture calls"
```

---

## Phase 7: Frontend component simplification

### Task 12: Simplify `ZoneConfigPanel.tsx`

**Files:**
- Modify: `frontend/src/components/ZoneConfigPanel.tsx`

- [ ] **Step 1: Read the file**

Run: `cat frontend/src/components/ZoneConfigPanel.tsx`

- [ ] **Step 2: Strip the 3D UI**

Delete these elements (by JSX role, not line number — they'll vary):

- Any `<HeightSlider>` import and usage
- Any slider or input tied to `heightMin`, `heightMax`, `height`
- Any slider or input tied to `scaleX`, `scaleY`, `skewX`, `skewY`, `slantX`, `slantY`, or `transform`
- Any `mode` selector / dropdown / toggle that lets the user pick between 2D / auto_3d / manual_3d
- Any button / handler that calls `estimateHeight` (the auto-estimate-height button)
- Any state variables for the above (use React state hooks to identify them)
- Any `cooldown_seconds` input

Keep:
- The zone name text input
- The `overlap_threshold` slider (probably labeled "Sensitivity" or similar)
- The `enabled` toggle
- The Save / Delete buttons and their handlers (updated to match the new `createZone` signature from Task 11)
- The furniture creation form if it lives in this same file — simplified to `{name, polygon}` only

- [ ] **Step 3: Verify the component type-checks**

Run: `cd frontend && npx tsc --noEmit`

Expected: zero errors from `ZoneConfigPanel.tsx`. Errors may still exist elsewhere.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ZoneConfigPanel.tsx
git commit -m "refactor(ZoneConfigPanel): remove 3D sliders, height, mode selector"
```

---

### Task 13: Simplify `ZoneEditor.tsx`

**Files:**
- Modify: `frontend/src/components/ZoneEditor.tsx`

- [ ] **Step 1: Read the file**

Run: `cat frontend/src/components/ZoneEditor.tsx`

- [ ] **Step 2: Strip 3D overlay rendering**

Delete:

- Any 3D prism / extrusion rendering (search for `extrude`, `prism`, `slant`, `room_polygon`)
- Any mode-dependent label block (search for `auto_3d`, `manual_3d`, `mode === "2d"`)
- Any import from `HeightSlider` or `ZoneTransform`

Keep:
- The 2D polygon drawing logic (click to add point, double-click to close)
- Ramer-Douglas-Peucker simplification (it's useful for 2D)
- The polygon rendering on the panorama canvas

- [ ] **Step 3: Verify the component type-checks**

Run: `cd frontend && npx tsc --noEmit`

Expected: zero errors from `ZoneEditor.tsx`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ZoneEditor.tsx
git commit -m "refactor(ZoneEditor): remove 3D prism overlay and mode logic"
```

---

### Task 14: Clean up remaining frontend components and delete `HeightSlider.tsx`

**Files:**
- Modify: `frontend/src/App.tsx`, `frontend/src/components/Settings.tsx`, `frontend/src/components/PanoramaView.tsx`, `frontend/src/components/LiveFeed.tsx`, `frontend/src/components/CalibrationWizard.tsx`
- Delete: `frontend/src/components/HeightSlider.tsx` (if unused after Tasks 12-13)

- [ ] **Step 1: Run the type check to get the current error list**

Run: `cd frontend && npx tsc --noEmit`

Every error remaining on this list is a stale reference to a removed field or component. Walk the list and fix each one — typically by deleting the offending line or JSX block.

- [ ] **Step 2: Fix each file's errors**

For each error:
- If it's `Property 'height_min' does not exist on type 'Zone'` → delete the line that reads the field.
- If it's `Property 'mode' does not exist on type 'Zone'` → delete the branch that uses `mode`.
- If it's `Property 'ZoneTransform' does not exist` → delete the import and any usage.
- If it's `Property 'warning_remaining' does not exist` on a broadcast payload → delete the code that displays it.

- [ ] **Step 3: Check whether `HeightSlider.tsx` is still imported**

Run: `grep -rn "HeightSlider" frontend/src/`

If zero hits outside `frontend/src/components/HeightSlider.tsx` itself, delete the file:

```bash
rm frontend/src/components/HeightSlider.tsx
```

- [ ] **Step 4: Delete any depth-calibration step from `CalibrationWizard.tsx`**

Run: `grep -n "depth\|midas\|MiDaS" frontend/src/components/CalibrationWizard.tsx`

Delete any step, button, or state that initiates a depth calibration. The calibration wizard should only cover extent capture / aim calibration after this change.

- [ ] **Step 5: Delete any 3D mode branches in `Settings.tsx`**

Run: `grep -n "midas\|depth\|height\|occlusion\|room_\|mode.*auto_3d\|mode.*manual_3d" frontend/src/components/Settings.tsx`

Delete each matching UI element. If you want to add UI for `min_shot_interval_ms` and `engagement_grace_ms`, this is a reasonable place — but that's optional and can wait for a follow-up.

- [ ] **Step 6: Final type check**

Run: `cd frontend && npx tsc --noEmit`

Expected: zero errors.

Run: `cd frontend && npm run build`

Expected: build succeeds with no errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/
git commit -m "chore(frontend): clean up 3D references across components"
```

---

## Phase 8: Optional dependency cleanup

### Task 15: Remove `torch` and `transformers` if unused

**Files:**
- Modify: `requirements.txt` or `pyproject.toml` (whichever the repo uses)

- [ ] **Step 1: Verify neither is imported elsewhere**

Run: `grep -rn "^import torch\|^from torch\|^import transformers\|^from transformers" server/`

If zero hits, both are safe to remove. If there are hits (for example, the cat classifier uses torch for its model), they stay.

- [ ] **Step 2: If safe, remove from dependencies**

Find the line(s) in `requirements.txt` or the `[project.dependencies]` block in `pyproject.toml` that list `torch` and `transformers`. Delete them.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "chore(deps): remove torch and transformers (no longer used)"
```

If the grep in Step 1 showed hits, skip this task entirely and note in the commit log for Task 9 that torch is retained for the classifier.

---

## Phase 9: End-to-end smoke test on the rig

### Task 16: Manual rig smoke test

This is not a code task — it's a verification checklist. Perform it before closing the PR.

- [ ] **Step 1: Start the backend**

Run: `python -m server.main` (or the project's usual dev server command).

Expected: server starts without `ImportError`, `NameError`, or schema errors. The log shows `YOLO detector loaded` and the sweep controller initializes.

- [ ] **Step 2: Start the frontend**

Run: `cd frontend && npm run dev`

Expected: dev server starts. Open it in a browser; the live feed loads.

- [ ] **Step 3: Verify existing zones table is empty**

The schema was wiped. Open the zones panel — it should show no zones.

- [ ] **Step 4: Draw a new zone**

Use the panorama view to click out a polygon, name it, save it. Confirm it saves without error and persists across a page refresh.

- [ ] **Step 5: Draw a furniture outline**

Use whichever panel now handles furniture. Name it, save it. Confirm it renders on the panorama and persists across a page refresh.

- [ ] **Step 6: Arm and trigger**

Arm the system. Walk a cat plushie (or wave a hand) into the zone. Expected behavior:

- Sweep pauses the moment the bbox overlaps the zone (watch the mode indicator — it should change from SWEEPING to ENGAGING).
- First shot fires within ~1 frame.
- Subsequent shots fire approximately every 2 seconds while the plushie stays inside the zone.
- When the plushie leaves, the sweep resumes ~3 seconds later.
- Walking the plushie outside the zone (but still visible to the camera) does NOT pause the sweep.

- [ ] **Step 7: Compare FPS**

Check the FPS indicator on the UI before and after the change (compare to a pre-rewrite run if possible). Expect a clear increase now that MiDaS is gone.

- [ ] **Step 8: If anything fails**

File the failure back to the plan — don't just "fix it on the rig." A manual fix on the rig without a corresponding code change means the plan was incomplete, and the next engineer to touch this area will hit the same bug.

---

## Self-review notes

**Spec coverage check:**

- Architecture-after-rewrite diagram → Task 2 (SweepController rewrite) + Task 8 (main.py integration).
- `zones` table schema → Task 5.
- `furniture` table schema → Task 5.
- Engagement state machine with 2s/3s timers → Task 2 + Task 8.
- Target selection by "closest to current aim" → Task 8, Step 8.
- Cats outside zones ignored → Task 2 (deleted TRACKING), Task 8 (no `on_cat_detected` call).
- Config changes → Task 7.
- Deleted files → Task 9.
- Dependency removal → Task 15.
- Testing plan → Tasks 1, 3, plus Task 16 for manual.

**Placeholder scan:** no TBDs, no "add error handling", no "similar to Task N" — every code block is reproduced in full.

**Type consistency check:**
- `min_shot_interval_ms` / `engagement_grace_ms`: used consistently in config.py, SweepController constructor, and tests.
- `on_cat_in_zone(cat_pan, cat_tilt, zone_name)`: used consistently in tests and main.py.
- `on_no_cat_in_zone()`: used consistently.
- `should_fire()` and `mark_shot_fired()`: paired consistently — caller always calls them in order.
- `SweepState.ENGAGING`: the only non-sweep state referenced outside the controller (in main.py direction-delta block).

**Known hedged file paths that the engineer must verify at task time:**
- The furniture router location (Task 6, Step 2) — might be `routers/zones.py`, `routers/control.py`, or a dedicated `routers/furniture.py`. `grep -l "create_furniture\|get_furniture"` locates it.
- `HeightSlider.tsx` usage (Task 14, Step 3) — may have additional call sites not yet found.
- The exact Pydantic schema file location for furniture (Task 6) — may be inline in the router or in a separate `schemas.py`.
