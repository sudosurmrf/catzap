# Panoramic Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-camera pixel-space zones with a servo-registered panoramic tile map, enabling camera-on-gun mounting with angle-space zone tracking and a sweep-based state machine.

**Architecture:** Server builds a tile grid panorama from frames captured at known servo angles. Zones are defined in angle-space (degrees). A state machine (SWEEPING → WARNING → FIRING → TRACKING) controls camera movement and fire decisions. Dev mode uses manual angle control with webcam.

**Tech Stack:** Python 3.11+, FastAPI, OpenCV, YOLOv8-nano, Shapely, asyncpg, React, TypeScript

---

## File Structure

### New Files
- `server/panorama/__init__.py` — Package init
- `server/panorama/angle_math.py` — Pixel↔angle coordinate conversions
- `server/panorama/tile_grid.py` — Tile storage, smart refresh, panorama image assembly
- `server/panorama/sweep_controller.py` — State machine (SWEEPING/WARNING/FIRING/TRACKING)
- `server/tests/test_angle_math.py` — Angle math unit tests
- `server/tests/test_tile_grid.py` — Tile grid unit tests
- `server/tests/test_sweep_controller.py` — State machine unit tests
- `frontend/src/components/PanoramaView.tsx` — Tile grid renderer with zones and cat markers
- `frontend/src/components/SweepControls.tsx` — Sweep config panel
- `frontend/src/components/StateIndicator.tsx` — System state display
- `frontend/src/components/DirectionArrow.tsx` — Dev mode flashing arrow indicator

### Modified Files
- `server/config.py` — Add sweep/panorama config fields
- `server/main.py` — Replace vision loop with sweep controller loop
- `server/actuator/client.py` — Add `goto()`, `stop()` methods
- `server/vision/zone_checker.py` — Already works in any coordinate space, no change needed
- `server/routers/control.py` — Add sweep endpoints, panorama serving, virtual angle control
- `server/routers/stream.py` — Add panorama WebSocket broadcast
- `frontend/src/types/index.ts` — Add panorama/sweep types
- `frontend/src/api/client.ts` — Add sweep/panorama API functions
- `frontend/src/App.tsx` — Split view layout
- `frontend/src/components/LiveFeed.tsx` — Angle-space zone projection, direction arrow, state display
- `frontend/src/components/ZoneEditor.tsx` — Draw on panorama instead of live feed
- `frontend/src/components/Controls.tsx` — Add sweep controls
- `frontend/src/components/Calibration.tsx` — Simplify to calibration sweep button
- `firmware/esp32-actuator/src/main.cpp` — Add `/goto`, `/stop`, `/position` endpoints

### Deleted Files
- `server/actuator/calibration.py` — Replaced by angle_math.py
- `server/tests/test_calibration.py` — Replaced by test_angle_math.py

---

## Task 1: Angle Math Module

**Files:**
- Create: `server/panorama/__init__.py`
- Create: `server/panorama/angle_math.py`
- Create: `server/tests/test_angle_math.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_angle_math.py`:

```python
from server.panorama.angle_math import pixel_to_angle, angle_to_pixel, is_angle_in_fov


def test_pixel_center_maps_to_servo_angle():
    pan, tilt = pixel_to_angle(0.5, 0.5, servo_pan=90.0, servo_tilt=45.0)
    assert pan == 90.0
    assert tilt == 45.0


def test_pixel_right_edge_maps_to_higher_pan():
    pan, tilt = pixel_to_angle(1.0, 0.5, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0)
    assert abs(pan - 122.5) < 0.1  # 90 + 0.5 * 65


def test_pixel_left_edge_maps_to_lower_pan():
    pan, tilt = pixel_to_angle(0.0, 0.5, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0)
    assert abs(pan - 57.5) < 0.1  # 90 - 0.5 * 65


def test_pixel_top_maps_to_lower_tilt():
    pan, tilt = pixel_to_angle(0.5, 0.0, servo_pan=90.0, servo_tilt=45.0, fov_v=50.0)
    assert abs(tilt - 20.0) < 0.1  # 45 - 0.5 * 50


def test_angle_to_pixel_center():
    px, py = angle_to_pixel(90.0, 45.0, servo_pan=90.0, servo_tilt=45.0)
    assert abs(px - 0.5) < 0.01
    assert abs(py - 0.5) < 0.01


def test_angle_to_pixel_offset():
    px, py = angle_to_pixel(122.5, 45.0, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0)
    assert abs(px - 1.0) < 0.01


def test_roundtrip_pixel_angle_pixel():
    pan, tilt = pixel_to_angle(0.3, 0.7, servo_pan=60.0, servo_tilt=40.0)
    px, py = angle_to_pixel(pan, tilt, servo_pan=60.0, servo_tilt=40.0)
    assert abs(px - 0.3) < 0.01
    assert abs(py - 0.7) < 0.01


def test_is_angle_in_fov_center():
    assert is_angle_in_fov(90.0, 45.0, servo_pan=90.0, servo_tilt=45.0) == True


def test_is_angle_in_fov_outside():
    assert is_angle_in_fov(10.0, 45.0, servo_pan=90.0, servo_tilt=45.0) == False


def test_is_angle_in_fov_edge():
    # Just inside the edge
    assert is_angle_in_fov(122.0, 45.0, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0) == True
    # Just outside
    assert is_angle_in_fov(123.0, 45.0, servo_pan=90.0, servo_tilt=45.0, fov_h=65.0) == False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest server/tests/test_angle_math.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/panorama/__init__.py` (empty file).

Create `server/panorama/angle_math.py`:

```python
from server.config import settings


def pixel_to_angle(
    pixel_x: float,
    pixel_y: float,
    servo_pan: float,
    servo_tilt: float,
    fov_h: float | None = None,
    fov_v: float | None = None,
) -> tuple[float, float]:
    """Convert normalized pixel coords (0-1) to angle-space using current servo position."""
    fov_h = fov_h or settings.fov_horizontal
    fov_v = fov_v or settings.fov_vertical
    pan = servo_pan + (pixel_x - 0.5) * fov_h
    tilt = servo_tilt + (pixel_y - 0.5) * fov_v
    return (pan, tilt)


def angle_to_pixel(
    pan: float,
    tilt: float,
    servo_pan: float,
    servo_tilt: float,
    fov_h: float | None = None,
    fov_v: float | None = None,
) -> tuple[float, float]:
    """Convert angle-space coords to normalized pixel coords for current servo position."""
    fov_h = fov_h or settings.fov_horizontal
    fov_v = fov_v or settings.fov_vertical
    pixel_x = 0.5 + (pan - servo_pan) / fov_h
    pixel_y = 0.5 + (tilt - servo_tilt) / fov_v
    return (pixel_x, pixel_y)


def is_angle_in_fov(
    pan: float,
    tilt: float,
    servo_pan: float,
    servo_tilt: float,
    fov_h: float | None = None,
    fov_v: float | None = None,
) -> bool:
    """Check if an angle-space point is within the current camera FOV."""
    fov_h = fov_h or settings.fov_horizontal
    fov_v = fov_v or settings.fov_vertical
    half_h = fov_h / 2
    half_v = fov_v / 2
    return (
        servo_pan - half_h <= pan <= servo_pan + half_h
        and servo_tilt - half_v <= tilt <= servo_tilt + half_v
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest server/tests/test_angle_math.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```
feat: add angle math module for pixel-to-angle coordinate conversion
```

---

## Task 2: Tile Grid Manager

**Files:**
- Create: `server/panorama/tile_grid.py`
- Create: `server/tests/test_tile_grid.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_tile_grid.py`:

```python
import numpy as np
import pytest
from server.panorama.tile_grid import TileGrid


@pytest.fixture
def grid():
    return TileGrid(
        pan_min=30.0, pan_max=150.0,
        tilt_min=20.0, tilt_max=70.0,
        fov_h=65.0, fov_v=50.0,
        tile_overlap=10.0,
    )


def test_grid_computes_tile_positions(grid):
    positions = grid.get_tile_positions()
    assert len(positions) > 0
    # Each position is (pan_center, tilt_center)
    for pan, tilt in positions:
        assert 30.0 <= pan <= 150.0
        assert 20.0 <= tilt <= 70.0


def test_grid_tile_count(grid):
    positions = grid.get_tile_positions()
    # 120° range with 55° step ≈ 3 columns, 50° range with 40° step ≈ 2 rows
    assert 2 <= len(positions) <= 12


def test_store_and_retrieve_tile(grid):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    grid.update_tile(0, 0, frame)
    tile = grid.get_tile(0, 0)
    assert tile is not None


def test_get_empty_tile_returns_none(grid):
    tile = grid.get_tile(0, 0)
    assert tile is None


def test_smart_refresh_detects_change(grid):
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    grid.update_tile(0, 0, frame1)
    assert grid.should_refresh(0, 0, frame2, threshold=15) == True


def test_smart_refresh_skips_similar(grid):
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 5
    grid.update_tile(0, 0, frame1)
    assert grid.should_refresh(0, 0, frame2, threshold=15) == False


def test_angle_to_tile_index(grid):
    positions = grid.get_tile_positions()
    if len(positions) > 0:
        pan, tilt = positions[0]
        col, row = grid.angle_to_tile_index(pan, tilt)
        assert col == 0
        assert row == 0


def test_get_panorama_image(grid):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    positions = grid.get_tile_positions()
    for i, (pan, tilt) in enumerate(positions):
        col, row = grid.angle_to_tile_index(pan, tilt)
        grid.update_tile(col, row, frame)
    img = grid.get_panorama_image()
    assert img is not None
    assert img.shape[0] > 0
    assert img.shape[1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest server/tests/test_tile_grid.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/panorama/tile_grid.py`:

```python
import math

import cv2
import numpy as np


class TileGrid:
    """Manages a grid of image tiles in servo angle-space."""

    def __init__(
        self,
        pan_min: float = 30.0,
        pan_max: float = 150.0,
        tilt_min: float = 20.0,
        tilt_max: float = 70.0,
        fov_h: float = 65.0,
        fov_v: float = 50.0,
        tile_overlap: float = 10.0,
    ):
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.tile_overlap = tile_overlap

        self.tile_step_h = fov_h - tile_overlap
        self.tile_step_v = fov_v - tile_overlap

        self.cols = max(1, math.ceil((pan_max - pan_min) / self.tile_step_h))
        self.rows = max(1, math.ceil((tilt_max - tilt_min) / self.tile_step_v))

        # Storage: (col, row) → JPEG bytes
        self._tiles: dict[tuple[int, int], bytes] = {}
        # Raw frames for smart refresh comparison
        self._raw_tiles: dict[tuple[int, int], np.ndarray] = {}

    def get_tile_positions(self) -> list[tuple[float, float]]:
        """Return the servo (pan, tilt) center position for each tile."""
        positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                pan = self.pan_min + col * self.tile_step_h + self.fov_h / 2
                tilt = self.tilt_min + row * self.tile_step_v + self.fov_v / 2
                # Clamp to range
                pan = min(pan, self.pan_max - self.fov_h / 2 + self.fov_h / 2)
                tilt = min(tilt, self.tilt_max - self.fov_v / 2 + self.fov_v / 2)
                positions.append((pan, tilt))
        return positions

    def angle_to_tile_index(self, pan: float, tilt: float) -> tuple[int, int]:
        """Find which tile a given angle falls into."""
        col = int((pan - self.pan_min - self.fov_h / 2) / self.tile_step_h + 0.5)
        row = int((tilt - self.tilt_min - self.fov_v / 2) / self.tile_step_v + 0.5)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (col, row)

    def update_tile(self, col: int, row: int, frame: np.ndarray):
        """Store a frame as a tile."""
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._tiles[(col, row)] = jpeg.tobytes()
        # Store downscaled raw for comparison
        small = cv2.resize(frame, (64, 48))
        self._raw_tiles[(col, row)] = small

    def get_tile(self, col: int, row: int) -> bytes | None:
        """Get JPEG bytes for a tile."""
        return self._tiles.get((col, row))

    def should_refresh(
        self, col: int, row: int, new_frame: np.ndarray, threshold: int = 15
    ) -> bool:
        """Check if a tile needs updating by comparing to stored version."""
        if (col, row) not in self._raw_tiles:
            return True
        old = self._raw_tiles[(col, row)]
        new_small = cv2.resize(new_frame, (64, 48))
        diff = np.mean(np.abs(old.astype(float) - new_small.astype(float)))
        return diff > threshold

    def get_panorama_image(self) -> np.ndarray | None:
        """Assemble all tiles into a single panorama image."""
        if not self._tiles:
            return None

        # Determine tile pixel size from first tile
        first_key = next(iter(self._tiles))
        first_jpeg = self._tiles[first_key]
        first_frame = cv2.imdecode(
            np.frombuffer(first_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        tile_h, tile_w = first_frame.shape[:2]

        pano_w = self.cols * tile_w
        pano_h = self.rows * tile_h
        panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

        for (col, row), jpeg in self._tiles.items():
            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue
            resized = cv2.resize(frame, (tile_w, tile_h))
            y = row * tile_h
            x = col * tile_w
            panorama[y : y + tile_h, x : x + tile_w] = resized

        return panorama

    def get_panorama_jpeg(self, quality: int = 70) -> bytes | None:
        """Get the assembled panorama as JPEG bytes."""
        pano = self.get_panorama_image()
        if pano is None:
            return None
        _, jpeg = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

    def panorama_pixel_to_angle(self, px: float, py: float) -> tuple[float, float]:
        """Convert a panorama pixel position (normalized 0-1) to angle-space."""
        pan = self.pan_min + px * (self.pan_max - self.pan_min)
        tilt = self.tilt_min + py * (self.tilt_max - self.tilt_min)
        return (pan, tilt)

    def angle_to_panorama_pixel(self, pan: float, tilt: float) -> tuple[float, float]:
        """Convert angle-space to panorama pixel position (normalized 0-1)."""
        px = (pan - self.pan_min) / (self.pan_max - self.pan_min)
        py = (tilt - self.tilt_min) / (self.tilt_max - self.tilt_min)
        return (px, py)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest server/tests/test_tile_grid.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```
feat: add tile grid manager for panoramic map storage
```

---

## Task 3: Sweep Controller State Machine

**Files:**
- Create: `server/panorama/sweep_controller.py`
- Create: `server/tests/test_sweep_controller.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_sweep_controller.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from server.panorama.sweep_controller import SweepController, SweepState


@pytest.fixture
def controller():
    actuator = AsyncMock()
    actuator.goto.return_value = True
    actuator.fire.return_value = True
    return SweepController(
        actuator=actuator,
        pan_min=30.0, pan_max=150.0,
        tilt=45.0, speed=10.0,
        warning_duration=1.5,
        tracking_duration=3.0,
        cooldown=3.0,
        dev_mode=True,
    )


def test_initial_state_is_sweeping(controller):
    assert controller.state == SweepState.SWEEPING


def test_sweep_advances_angle(controller):
    initial = controller.current_pan
    controller.tick(dt=1.0)
    assert controller.current_pan != initial


def test_sweep_reverses_at_max(controller):
    controller.current_pan = 149.0
    controller._sweep_direction = 1
    controller.tick(dt=1.0)
    assert controller._sweep_direction == -1


def test_cat_in_zone_transitions_to_warning(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.WARNING


def test_warning_expires_transitions_to_firing(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.WARNING
    # Simulate time passing beyond warning duration
    controller.tick(dt=2.0)
    controller.on_cat_still_in_zone(cat_pan=90.0, cat_tilt=45.0)
    assert controller.state == SweepState.FIRING


def test_cat_leaves_during_warning_returns_to_sweeping(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.on_cat_left_zone()
    assert controller.state == SweepState.SWEEPING


def test_firing_transitions_to_tracking(controller):
    controller.state = SweepState.FIRING
    controller.on_fire_complete()
    controller.on_cat_left_zone()
    assert controller.state == SweepState.TRACKING


def test_tracking_returns_to_sweeping_after_timeout(controller):
    controller.state = SweepState.TRACKING
    controller._tracking_start = 0  # long ago
    controller.tick(dt=0.0)
    assert controller.state == SweepState.SWEEPING


def test_get_direction_delta(controller):
    controller.current_pan = 80.0
    controller.current_tilt = 45.0
    delta = controller.get_direction_delta(target_pan=100.0, target_tilt=50.0)
    assert delta["pan"] > 0  # need to pan right
    assert delta["tilt"] > 0  # need to tilt down
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest server/tests/test_sweep_controller.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/panorama/sweep_controller.py`:

```python
import time
from enum import Enum

from server.actuator.client import ActuatorClient


class SweepState(str, Enum):
    SWEEPING = "SWEEPING"
    WARNING = "WARNING"
    FIRING = "FIRING"
    TRACKING = "TRACKING"


class SweepController:
    """State machine controlling camera sweep and fire behavior."""

    def __init__(
        self,
        actuator: ActuatorClient,
        pan_min: float = 30.0,
        pan_max: float = 150.0,
        tilt: float = 45.0,
        speed: float = 2.5,
        warning_duration: float = 1.5,
        tracking_duration: float = 3.0,
        cooldown: float = 3.0,
        reentry_warning: float = 0.5,
        dev_mode: bool = False,
    ):
        self.actuator = actuator
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.current_pan = pan_min
        self.current_tilt = tilt
        self.speed = speed
        self.warning_duration = warning_duration
        self.tracking_duration = tracking_duration
        self.cooldown = cooldown
        self.reentry_warning = reentry_warning
        self.dev_mode = dev_mode

        self.state = SweepState.SWEEPING
        self._sweep_direction = 1  # 1 = right, -1 = left
        self._warning_start: float = 0
        self._tracking_start: float = 0
        self._last_fire_time: float = 0
        self._target_pan: float = 0
        self._target_tilt: float = 0
        self._target_zone: str = ""
        self._was_tracking: bool = False

    def tick(self, dt: float):
        """Advance the state machine by dt seconds."""
        now = time.time()

        if self.state == SweepState.SWEEPING:
            self.current_pan += self.speed * self._sweep_direction * dt
            if self.current_pan >= self.pan_max:
                self.current_pan = self.pan_max
                self._sweep_direction = -1
            elif self.current_pan <= self.pan_min:
                self.current_pan = self.pan_min
                self._sweep_direction = 1

        elif self.state == SweepState.WARNING:
            elapsed = now - self._warning_start
            warn_time = self.reentry_warning if self._was_tracking else self.warning_duration
            if elapsed >= warn_time:
                # Warning expired — check if cat is still there (caller should call on_cat_still_in_zone)
                pass

        elif self.state == SweepState.TRACKING:
            elapsed = now - self._tracking_start
            if elapsed >= self.tracking_duration:
                self.state = SweepState.SWEEPING
                self._was_tracking = False

    def on_cat_in_zone(self, cat_pan: float, cat_tilt: float, zone_name: str):
        """Called when a cat is detected inside a forbidden zone."""
        if self.state == SweepState.SWEEPING:
            self.state = SweepState.WARNING
            self._warning_start = time.time()
            self._target_pan = cat_pan
            self._target_tilt = cat_tilt
            self._target_zone = zone_name
        elif self.state == SweepState.TRACKING:
            self.state = SweepState.WARNING
            self._warning_start = time.time()
            self._was_tracking = True
            self._target_pan = cat_pan
            self._target_tilt = cat_tilt
            self._target_zone = zone_name

    def on_cat_still_in_zone(self, cat_pan: float, cat_tilt: float):
        """Called each frame when cat remains in zone during WARNING."""
        self._target_pan = cat_pan
        self._target_tilt = cat_tilt
        now = time.time()
        warn_time = self.reentry_warning if self._was_tracking else self.warning_duration
        if self.state == SweepState.WARNING and (now - self._warning_start) >= warn_time:
            if now - self._last_fire_time >= self.cooldown:
                self.state = SweepState.FIRING

    def on_cat_left_zone(self):
        """Called when the cat leaves all forbidden zones."""
        if self.state == SweepState.WARNING:
            self.state = SweepState.SWEEPING
            self._was_tracking = False
        elif self.state in (SweepState.FIRING, SweepState.TRACKING):
            self.state = SweepState.TRACKING
            self._tracking_start = time.time()

    def on_fire_complete(self):
        """Called after the solenoid has fired."""
        self._last_fire_time = time.time()

    def should_fire(self) -> bool:
        """Check if the system should fire right now."""
        return self.state == SweepState.FIRING

    def get_direction_delta(self, target_pan: float, target_tilt: float) -> dict:
        """Get the angle delta between current position and target (for dev mode arrow)."""
        return {
            "pan": target_pan - self.current_pan,
            "tilt": target_tilt - self.current_tilt,
        }

    def get_target(self) -> tuple[float, float, str]:
        """Get the current target (cat position) and zone name."""
        return (self._target_pan, self._target_tilt, self._target_zone)

    def get_warning_remaining(self) -> float:
        """Get seconds remaining in warning period."""
        if self.state != SweepState.WARNING:
            return 0
        warn_time = self.reentry_warning if self._was_tracking else self.warning_duration
        elapsed = time.time() - self._warning_start
        return max(0, warn_time - elapsed)

    def set_virtual_angle(self, pan: float, tilt: float):
        """Set the virtual angle (dev mode — manual control)."""
        self.current_pan = max(self.pan_min, min(self.pan_max, pan))
        self.current_tilt = tilt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest server/tests/test_sweep_controller.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```
feat: add sweep controller state machine
```

---

## Task 4: Update Config & Actuator Client

**Files:**
- Modify: `server/config.py`
- Modify: `server/actuator/client.py`
- Delete: `server/actuator/calibration.py`
- Delete: `server/tests/test_calibration.py`

- [ ] **Step 1: Update config.py**

Add sweep/panorama settings to `server/config.py`. Add these fields to the `Settings` class after the existing `cooldown_default` field:

```python
    # Sweep / Panorama
    sweep_pan_min: float = 30.0
    sweep_pan_max: float = 150.0
    sweep_tilt_min: float = 20.0
    sweep_tilt_max: float = 70.0
    sweep_speed: float = 2.5
    fov_horizontal: float = 65.0
    fov_vertical: float = 50.0
    tile_overlap: float = 10.0
    tile_refresh_threshold: int = 15
    warning_duration: float = 1.5
    tracking_duration: float = 3.0
    reentry_warning: float = 0.5
```

- [ ] **Step 2: Add goto and stop methods to actuator client**

Add to `server/actuator/client.py`:

```python
    async def goto(self, pan: float, tilt: float) -> bool:
        """Move servos to a specific angle."""
        try:
            response = await self._client.post(
                f"{self.base_url}/goto",
                json={"pan": pan, "tilt": tilt},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send goto command: {e}")
            return False

    async def stop(self) -> bool:
        """Stop sweep, hold current position."""
        try:
            response = await self._client.post(
                f"{self.base_url}/stop",
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send stop command: {e}")
            return False
```

- [ ] **Step 3: Delete calibration files**

Delete `server/actuator/calibration.py` and `server/tests/test_calibration.py`.

- [ ] **Step 4: Commit**

```
feat: add sweep config, actuator goto/stop, remove calibration module
```

---

## Task 5: Rewrite Main Vision Loop

**Files:**
- Modify: `server/main.py`
- Modify: `server/routers/control.py`

- [ ] **Step 1: Rewrite server/main.py**

Replace the entire `run_vision_loop` function and related code in `server/main.py`:

```python
import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import settings
from server.models.database import init_db, close_db, create_event, get_zones
from server.routers import zones, cats, events, control, stream, settings as settings_router
from server.routers.stream import broadcast_to_clients
from server.vision.detector import CatDetector
from server.vision.zone_checker import check_zone_violations
from server.panorama.angle_math import pixel_to_angle
from server.panorama.tile_grid import TileGrid
from server.panorama.sweep_controller import SweepController, SweepState
from server.actuator.client import ActuatorClient

logger = logging.getLogger(__name__)

# Shared state
_tile_grid: TileGrid | None = None
_sweep_controller: SweepController | None = None
_actuator: ActuatorClient | None = None


def get_tile_grid() -> TileGrid:
    return _tile_grid


def get_sweep_controller() -> SweepController:
    return _sweep_controller


def get_actuator() -> ActuatorClient:
    return _actuator


async def run_vision_loop(app_state: dict):
    global _tile_grid, _sweep_controller, _actuator

    detector = CatDetector(confidence_threshold=settings.confidence_threshold)
    _actuator = ActuatorClient(base_url=settings.esp32_actuator_url)

    _tile_grid = TileGrid(
        pan_min=settings.sweep_pan_min,
        pan_max=settings.sweep_pan_max,
        tilt_min=settings.sweep_tilt_min,
        tilt_max=settings.sweep_tilt_max,
        fov_h=settings.fov_horizontal,
        fov_v=settings.fov_vertical,
        tile_overlap=settings.tile_overlap,
    )

    _sweep_controller = SweepController(
        actuator=_actuator,
        pan_min=settings.sweep_pan_min,
        pan_max=settings.sweep_pan_max,
        tilt=(settings.sweep_tilt_min + settings.sweep_tilt_max) / 2,
        speed=settings.sweep_speed,
        warning_duration=settings.warning_duration,
        tracking_duration=settings.tracking_duration,
        cooldown=settings.cooldown_default,
        reentry_warning=settings.reentry_warning,
        dev_mode=settings.dev_mode,
    )

    # Camera source
    cap = None
    reader = None
    if settings.dev_mode:
        logger.info("DEV MODE: Using local webcam with manual angle control")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
    else:
        from server.routers.stream import MJPEGStreamReader
        reader = MJPEGStreamReader(settings.esp32_cam_url)
        try:
            await reader.connect()
            logger.info("Connected to ESP32-CAM stream")
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return

    last_time = time.time()
    frame_count = 0

    try:
        while app_state.get("running", True):
            now = time.time()
            dt = now - last_time
            last_time = now

            # Advance state machine
            _sweep_controller.tick(dt)

            # In non-dev mode, command servos to current sweep position
            if not settings.dev_mode and _sweep_controller.state == SweepState.SWEEPING:
                await _actuator.goto(_sweep_controller.current_pan, _sweep_controller.current_tilt)

            # Grab frame
            if settings.dev_mode:
                ret, frame = cap.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.05)
                    continue
            else:
                frame = await reader.read_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

            frame_count += 1
            if frame_count % settings.frame_skip_n != 0:
                await asyncio.sleep(0.01)
                continue

            servo_pan = _sweep_controller.current_pan
            servo_tilt = _sweep_controller.current_tilt

            # Update tile grid (smart refresh)
            col, row = _tile_grid.angle_to_tile_index(servo_pan, servo_tilt)
            if _tile_grid.should_refresh(col, row, frame, settings.tile_refresh_threshold):
                _tile_grid.update_tile(col, row, frame)

            # Detect cats
            detections = detector.detect(frame)

            # Convert detections to angle-space and check zones
            current_zones = await get_zones()
            all_violations = []
            fired = False
            fire_target = None
            direction_delta = None

            for det in detections:
                bbox = det["bbox"]
                # Convert bbox to angle-space
                pan1, tilt1 = pixel_to_angle(bbox[0], bbox[1], servo_pan, servo_tilt)
                pan2, tilt2 = pixel_to_angle(bbox[2], bbox[3], servo_pan, servo_tilt)
                angle_bbox = [pan1, tilt1, pan2, tilt2]

                violations = check_zone_violations(angle_bbox, current_zones)
                all_violations.extend(violations)

                if not violations:
                    continue

                # Cat center in angle-space
                cat_pan = (pan1 + pan2) / 2
                cat_tilt = (tilt1 + tilt2) / 2
                zone_name = violations[0]["zone_name"]

                # Feed into state machine
                if _sweep_controller.state == SweepState.SWEEPING:
                    _sweep_controller.on_cat_in_zone(cat_pan, cat_tilt, zone_name)
                elif _sweep_controller.state == SweepState.WARNING:
                    _sweep_controller.on_cat_still_in_zone(cat_pan, cat_tilt)

                # Check if we should fire
                if _sweep_controller.should_fire():
                    if settings.dev_mode:
                        logger.info(f"DEV ZAP! Cat in {zone_name}")
                        fired = True
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        fire_target = {"x": center_x, "y": center_y, "zone": zone_name}
                    else:
                        # Center cat and fire
                        await _actuator.goto(cat_pan, cat_tilt)
                        await asyncio.sleep(0.2)
                        success = await _actuator.fire()
                        fired = success

                    _sweep_controller.on_fire_complete()
                    asyncio.create_task(_log_event({
                        "type": "ZAP",
                        "cat_name": det.get("cat_name"),
                        "zone_name": zone_name,
                        "confidence": det["confidence"],
                        "overlap": violations[0]["overlap"],
                        "servo_pan": cat_pan,
                        "servo_tilt": cat_tilt,
                    }))
                    break

                # Direction delta for dev mode arrow
                if settings.dev_mode and _sweep_controller.state in (SweepState.WARNING, SweepState.FIRING, SweepState.TRACKING):
                    direction_delta = _sweep_controller.get_direction_delta(cat_pan, cat_tilt)
                    break

            # If no violations found this frame, notify controller
            if not all_violations and _sweep_controller.state in (SweepState.WARNING, SweepState.FIRING):
                _sweep_controller.on_cat_left_zone()

            # Encode frame for WebSocket
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Build panorama JPEG if tiles exist
            pano_jpeg = _tile_grid.get_panorama_jpeg()
            pano_b64 = base64.b64encode(pano_jpeg).decode("utf-8") if pano_jpeg else None

            await broadcast_to_clients({
                "frame": frame_b64,
                "panorama": pano_b64,
                "detections": detections,
                "violations": all_violations,
                "fired": fired,
                "fire_target": fire_target,
                "state": _sweep_controller.state.value,
                "servo_pan": servo_pan,
                "servo_tilt": servo_tilt,
                "warning_remaining": _sweep_controller.get_warning_remaining(),
                "direction_delta": direction_delta,
            })

            await asyncio.sleep(0.03)

    except asyncio.CancelledError:
        pass
    finally:
        if cap:
            cap.release()
        if reader:
            await reader.close()


async def _log_event(evt: dict):
    await create_event(
        event_type=evt["type"],
        cat_name=evt.get("cat_name"),
        zone_name=evt.get("zone_name"),
        confidence=evt.get("confidence"),
        overlap=evt.get("overlap"),
        servo_pan=evt.get("servo_pan"),
        servo_tilt=evt.get("servo_tilt"),
    )
    await broadcast_to_clients(evt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app_state = {"running": True}
    vision_task = asyncio.create_task(run_vision_loop(app_state))
    yield
    app_state["running"] = False
    vision_task.cancel()
    try:
        await vision_task
    except asyncio.CancelledError:
        pass
    await close_db()


app = FastAPI(title="CatZap", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(zones.router)
app.include_router(cats.router)
app.include_router(events.router)
app.include_router(control.router)
app.include_router(stream.router)
app.include_router(settings_router.router)


@app.get("/health")
async def health():
    return {"status": "ok", "dev_mode": settings.dev_mode}
```

- [ ] **Step 2: Update control.py for sweep endpoints**

Rewrite `server/routers/control.py`:

```python
from pydantic import BaseModel
from fastapi import APIRouter

from server.config import settings

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
    return {
        "armed": _armed,
        "state": sc.state.value if sc else "INIT",
        "servo_pan": sc.current_pan if sc else 0,
        "servo_tilt": sc.current_tilt if sc else 0,
        "dev_mode": settings.dev_mode,
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
        await actuator.goto(request.pan, request.tilt)
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
    return {"pan": request.pan, "tilt": request.tilt}


@router.post("/calibration-sweep")
async def start_calibration_sweep():
    """Trigger a full calibration sweep to rebuild the panorama."""
    from server.main import get_sweep_controller
    sc = get_sweep_controller()
    if sc:
        sc.current_pan = sc.pan_min
        sc.state = sc.state.SWEEPING
    return {"started": True}
```

- [ ] **Step 3: Verify server starts**

Run: `python -m uvicorn server.main:app --reload`
Expected: Server starts with "DEV MODE" message. Webcam feed visible.

- [ ] **Step 4: Commit**

```
feat: rewrite vision loop with sweep controller and panorama tile grid
```

---

## Task 6: Frontend Types & API Client Updates

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Add new types**

Add to `frontend/src/types/index.ts`:

```typescript
export interface SweepState {
  state: "SWEEPING" | "WARNING" | "FIRING" | "TRACKING";
  servo_pan: number;
  servo_tilt: number;
  warning_remaining: number;
}

export interface DirectionDelta {
  pan: number;
  tilt: number;
}

export interface FrameData {
  frame: string;
  panorama: string | null;
  detections: Detection[];
  violations: Violation[];
  fired: boolean;
  fire_target: { x: number; y: number; zone: string } | null;
  state: string;
  servo_pan: number;
  servo_tilt: number;
  warning_remaining: number;
  direction_delta: DirectionDelta | null;
}
```

- [ ] **Step 2: Add API functions**

Add to `frontend/src/api/client.ts`:

```typescript
export const setVirtualAngle = (pan: number, tilt: number) =>
  fetchJSON("/control/virtual-angle", {
    method: "POST",
    body: JSON.stringify({ pan, tilt }),
  });

export const startCalibrationSweep = () =>
  fetchJSON("/control/calibration-sweep", { method: "POST" });
```

- [ ] **Step 3: Commit**

```
feat: add panorama types and API functions to frontend
```

---

## Task 7: PanoramaView Component

**Files:**
- Create: `frontend/src/components/PanoramaView.tsx`

- [ ] **Step 1: Write the component**

Create `frontend/src/components/PanoramaView.tsx`:

```tsx
import { useEffect, useRef } from "react";
import type { Zone, Detection } from "../types";

interface PanoramaViewProps {
  panoramaBase64: string | null;
  zones: Zone[];
  detections: Detection[];
  servoPan: number;
  servoTilt: number;
  sweepPanMin: number;
  sweepPanMax: number;
  sweepTiltMin: number;
  sweepTiltMax: number;
  fovH: number;
  fovV: number;
  onClickAngle?: (pan: number, tilt: number) => void;
}

export default function PanoramaView({
  panoramaBase64,
  zones,
  detections,
  servoPan,
  servoTilt,
  sweepPanMin,
  sweepPanMax,
  sweepTiltMin,
  sweepTiltMax,
  fovH,
  fovV,
  onClickAngle,
}: PanoramaViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!panoramaBase64) return;
    const img = new Image();
    img.onload = () => drawPanorama(img);
    img.src = `data:image/jpeg;base64,${panoramaBase64}`;
  }, [panoramaBase64, zones, detections, servoPan, servoTilt]);

  function angleToPanoPixel(pan: number, tilt: number, w: number, h: number) {
    const px = ((pan - sweepPanMin) / (sweepPanMax - sweepPanMin)) * w;
    const py = ((tilt - sweepTiltMin) / (sweepTiltMax - sweepTiltMin)) * h;
    return { x: px, y: py };
  }

  function drawPanorama(img: HTMLImageElement) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const w = canvas.width;
    const h = canvas.height;

    // Draw zones
    for (const zone of zones) {
      if (!zone.enabled) continue;
      ctx.strokeStyle = "#f94144";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.beginPath();
      zone.polygon.forEach(([pan, tilt], i) => {
        const { x, y } = angleToPanoPixel(pan, tilt, w, h);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "rgba(249, 65, 68, 0.1)";
      ctx.fill();
      ctx.setLineDash([]);

      if (zone.polygon.length > 0) {
        const { x, y } = angleToPanoPixel(zone.polygon[0][0], zone.polygon[0][1], w, h);
        ctx.fillStyle = "#f94144";
        ctx.font = "11px monospace";
        ctx.fillText(zone.name, x, y - 4);
      }
    }

    // Draw current camera viewport
    const vpLeft = angleToPanoPixel(servoPan - fovH / 2, servoTilt - fovV / 2, w, h);
    const vpRight = angleToPanoPixel(servoPan + fovH / 2, servoTilt + fovV / 2, w, h);
    ctx.strokeStyle = "#4cc9f0";
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(vpLeft.x, vpLeft.y, vpRight.x - vpLeft.x, vpRight.y - vpLeft.y);

    // Label
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(vpLeft.x, vpLeft.y - 14, 60, 14);
    ctx.fillStyle = "#4cc9f0";
    ctx.font = "10px monospace";
    ctx.fillText("LIVE VIEW", vpLeft.x + 2, vpLeft.y - 3);

    // Draw cat markers on panorama (from detections mapped to angle-space)
    for (const det of detections) {
      const cx = (det.bbox[0] + det.bbox[2]) / 2;
      const cy = (det.bbox[1] + det.bbox[3]) / 2;
      const catPan = servoPan + (cx - 0.5) * fovH;
      const catTilt = servoTilt + (cy - 0.5) * fovV;
      const { x, y } = angleToPanoPixel(catPan, catTilt, w, h);

      ctx.fillStyle = "#f94144";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#f94144";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!canvasRef.current || !onClickAngle) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    const pan = sweepPanMin + px * (sweepPanMax - sweepPanMin);
    const tilt = sweepTiltMin + py * (sweepTiltMax - sweepTiltMin);
    onClickAngle(pan, tilt);
  }

  return (
    <div style={{ position: "relative" }}>
      <div style={{ position: "absolute", top: 4, left: 8, zIndex: 1 }}>
        <span style={{ color: "#4cc9f0", fontFamily: "monospace", fontSize: 10, background: "rgba(0,0,0,0.6)", padding: "2px 6px", borderRadius: 3 }}>
          PANORAMA
        </span>
      </div>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{
          width: "100%",
          display: "block",
          background: "#111",
          borderRadius: "8px 8px 0 0",
          cursor: onClickAngle ? "crosshair" : "default",
          minHeight: 120,
        }}
      />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```
feat: add PanoramaView component with zone and viewport overlays
```

---

## Task 8: StateIndicator & DirectionArrow Components

**Files:**
- Create: `frontend/src/components/StateIndicator.tsx`
- Create: `frontend/src/components/DirectionArrow.tsx`

- [ ] **Step 1: Write StateIndicator**

Create `frontend/src/components/StateIndicator.tsx`:

```tsx
const STATE_COLORS: Record<string, string> = {
  SWEEPING: "#4cc9f0",
  WARNING: "#ffd60a",
  FIRING: "#f94144",
  TRACKING: "#7209b7",
};

interface StateIndicatorProps {
  state: string;
  warningRemaining: number;
}

export default function StateIndicator({ state, warningRemaining }: StateIndicatorProps) {
  const color = STATE_COLORS[state] || "#888";

  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 10px",
        background: `${color}25`,
        border: `1px solid ${color}`,
        borderRadius: 4,
        fontFamily: "monospace",
        fontSize: 12,
        color,
        fontWeight: "bold",
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: color,
          display: "inline-block",
          animation: state === "WARNING" ? "blink 0.5s infinite" : undefined,
        }}
      />
      {state}
      {state === "WARNING" && warningRemaining > 0 && (
        <span style={{ fontWeight: "normal" }}>
          {warningRemaining.toFixed(1)}s
        </span>
      )}
      <style>{`@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }`}</style>
    </div>
  );
}
```

- [ ] **Step 2: Write DirectionArrow**

Create `frontend/src/components/DirectionArrow.tsx`:

```tsx
import { useEffect, useState } from "react";

interface DirectionArrowProps {
  delta: { pan: number; tilt: number } | null;
}

export default function DirectionArrow({ delta }: DirectionArrowProps) {
  const [visible, setVisible] = useState(true);

  // Flash at ~2Hz
  useEffect(() => {
    if (!delta) return;
    const interval = setInterval(() => setVisible((v) => !v), 250);
    return () => clearInterval(interval);
  }, [delta]);

  if (!delta || (Math.abs(delta.pan) < 10 && Math.abs(delta.tilt) < 10)) return null;

  const panAbs = Math.abs(delta.pan);
  const arrowSize = Math.min(80, 20 + panAbs);
  const opacity = visible ? Math.min(1, 0.4 + panAbs / 60) : 0.1;

  const isRight = delta.pan > 0;

  return (
    <div
      style={{
        position: "absolute",
        top: "50%",
        [isRight ? "right" : "left"]: 8,
        transform: "translateY(-50%)",
        fontSize: arrowSize,
        color: "#ffd60a",
        opacity,
        pointerEvents: "none",
        fontWeight: "bold",
        textShadow: "0 0 20px rgba(255, 214, 10, 0.6)",
        transition: "opacity 0.15s",
        zIndex: 20,
      }}
    >
      {isRight ? "▶" : "◀"}
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```
feat: add StateIndicator and DirectionArrow dev mode components
```

---

## Task 9: Update LiveFeed for Split View

**Files:**
- Modify: `frontend/src/components/LiveFeed.tsx`

- [ ] **Step 1: Update LiveFeed**

Rewrite `frontend/src/components/LiveFeed.tsx` to handle the new data format (state, direction delta, angle-space zone projection). Key changes:

- Accept `state`, `warningRemaining`, `directionDelta` from WebSocket data
- Import and render `StateIndicator` and `DirectionArrow`
- Project zones from angle-space to pixel-space using current servo position
- Show servo angles in status bar

The full component code: take the existing LiveFeed and add these modifications:

1. Add to the WebSocket `onmessage` handler — extract `state`, `servo_pan`, `servo_tilt`, `warning_remaining`, `direction_delta` from `data`
2. Add `StateIndicator` component above the status bar
3. Add `DirectionArrow` component as an overlay
4. In `drawFrame`, when drawing zones, convert zone polygon from angle-space to pixel-space:
   ```typescript
   const fovH = 65; // from config
   const fovV = 50;
   // For each zone polygon point [pan, tilt]:
   const px = 0.5 + (pan - servoPan) / fovH;
   const py = 0.5 + (tilt - servoTilt) / fovV;
   // Only draw if within 0-1 range
   ```
5. Update status bar to show `Pan: XX° Tilt: XX° | State: XXXXX`

- [ ] **Step 2: Commit**

```
feat: update LiveFeed for angle-space zones, state indicator, direction arrow
```

---

## Task 10: Update App.tsx for Split View

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/ZoneEditor.tsx`
- Modify: `frontend/src/components/Controls.tsx`

- [ ] **Step 1: Update App.tsx layout**

Update `frontend/src/App.tsx` to show split view in the live tab:

```tsx
// In the live tab section, replace the current layout with:
{tab === "live" && (
  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
    {/* Panorama (top) */}
    <div style={{ position: "relative" }}>
      {editingZones ? (
        <ZoneEditor
          zones={zones}
          panoramaBase64={latestPanorama}
          sweepConfig={{ panMin: 30, panMax: 150, tiltMin: 20, tiltMax: 70 }}
          onSave={() => { setEditingZones(false); refreshZones(); }}
          onCancel={() => setEditingZones(false)}
        />
      ) : (
        <PanoramaView
          panoramaBase64={latestPanorama}
          zones={zones}
          detections={latestDetections}
          servoPan={servoPan}
          servoTilt={servoTilt}
          sweepPanMin={30} sweepPanMax={150}
          sweepTiltMin={20} sweepTiltMax={70}
          fovH={65} fovV={50}
        />
      )}
    </div>
    {/* Live feed (bottom) */}
    <LiveFeed zones={zones} />
    {/* Controls */}
    <div style={{ display: "flex", gap: 8 }}>
      <button onClick={() => setEditingZones(!editingZones)} ...>
        {editingZones ? "Cancel" : "+ Draw Zone"}
      </button>
    </div>
    <Controls />
  </div>
)}
```

The App component needs new state for `latestPanorama`, `latestDetections`, `servoPan`, `servoTilt` — extracted from the WebSocket feed data.

- [ ] **Step 2: Update ZoneEditor to draw on panorama**

Modify `frontend/src/components/ZoneEditor.tsx` to accept panorama image and sweep config as props instead of overlaying on the live canvas. Drawing coordinates map to angle-space using:
```typescript
const pan = sweepConfig.panMin + normalizedX * (sweepConfig.panMax - sweepConfig.panMin);
const tilt = sweepConfig.tiltMin + normalizedY * (sweepConfig.tiltMax - sweepConfig.tiltMin);
```

- [ ] **Step 3: Add keyboard handler for dev mode angle control**

Add to `App.tsx` a `useEffect` with a `keydown` listener:
```typescript
useEffect(() => {
  function handleKey(e: KeyboardEvent) {
    if (e.key === "ArrowLeft") setVirtualAngle(servoPan - 5, servoTilt);
    if (e.key === "ArrowRight") setVirtualAngle(servoPan + 5, servoTilt);
    if (e.key === "ArrowUp") setVirtualAngle(servoPan, servoTilt - 5);
    if (e.key === "ArrowDown") setVirtualAngle(servoPan, servoTilt + 5);
  }
  window.addEventListener("keydown", handleKey);
  return () => window.removeEventListener("keydown", handleKey);
}, [servoPan, servoTilt]);
```

- [ ] **Step 4: Update Controls with sweep controls**

Add to `frontend/src/components/Controls.tsx`:
- Virtual angle sliders (pan/tilt) for dev mode
- "Simulate Sweep" button that calls the calibration-sweep endpoint
- Current angle display

- [ ] **Step 5: Commit**

```
feat: split view layout with panorama, zone editor on panorama, dev mode controls
```

---

## Task 11: Update ESP32 Actuator Firmware

**Files:**
- Modify: `firmware/esp32-actuator/src/main.cpp`

- [ ] **Step 1: Add new endpoints**

Add to `firmware/esp32-actuator/src/main.cpp`:

```cpp
void handleGoto() {
    if (server.method() != HTTP_POST) {
        server.send(405, "application/json", "{\"error\":\"Method not allowed\"}");
        return;
    }

    String body = server.arg("plain");
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, body);

    if (error) {
        server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
        return;
    }

    float pan = doc["pan"] | currentPan;
    float tilt = doc["tilt"] | currentTilt;

    pan = constrain(pan, 0.0, 180.0);
    tilt = constrain(tilt, 0.0, 180.0);

    panServo.write((int)pan);
    tiltServo.write((int)tilt);
    currentPan = pan;
    currentTilt = tilt;

    String response;
    JsonDocument respDoc;
    respDoc["pan"] = pan;
    respDoc["tilt"] = tilt;
    serializeJson(respDoc, response);
    server.send(200, "application/json", response);
}

void handleStop() {
    // Hold current position (servos already hold)
    String response;
    JsonDocument doc;
    doc["stopped"] = true;
    doc["pan"] = currentPan;
    doc["tilt"] = currentTilt;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}

void handlePosition() {
    String response;
    JsonDocument doc;
    doc["pan"] = currentPan;
    doc["tilt"] = currentTilt;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}
```

Register in `setup()`:
```cpp
    server.on("/goto", handleGoto);
    server.on("/stop", handleStop);
    server.on("/position", HTTP_GET, handlePosition);
```

- [ ] **Step 2: Verify it compiles**

Run: `cd firmware/esp32-actuator && pio run`

- [ ] **Step 3: Commit**

```
feat: add goto, stop, position endpoints to ESP32 actuator firmware
```

---

## Task 12: Simplify Calibration Component

**Files:**
- Modify: `frontend/src/components/Calibration.tsx`

- [ ] **Step 1: Simplify to calibration sweep button**

Rewrite `frontend/src/components/Calibration.tsx`:

```tsx
import { useState } from "react";
import { startCalibrationSweep } from "../api/client";

export default function Calibration() {
  const [sweeping, setSweeping] = useState(false);

  async function handleSweep() {
    setSweeping(true);
    await startCalibrationSweep();
    setTimeout(() => setSweeping(false), 3000);
  }

  return (
    <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
      <h3 style={{ color: "#ccc", fontFamily: "monospace", fontSize: 14, marginTop: 0 }}>
        Panorama Calibration
      </h3>
      <p style={{ color: "#888", fontFamily: "monospace", fontSize: 12, marginBottom: 12 }}>
        Run a full sweep to build or rebuild the panoramic room map. In dev mode,
        use arrow keys to manually rotate the webcam as the virtual angle advances.
      </p>
      <button
        onClick={handleSweep}
        disabled={sweeping}
        style={{
          padding: "8px 16px",
          background: sweeping ? "#555" : "#4cc9f0",
          color: sweeping ? "#888" : "#1a1a2e",
          border: "none",
          borderRadius: 6,
          cursor: sweeping ? "not-allowed" : "pointer",
          fontFamily: "monospace",
        }}
      >
        {sweeping ? "Sweeping..." : "Start Calibration Sweep"}
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```
feat: simplify Calibration to single sweep button
```

---

## Task 13: Integration Testing

**Files:**
- No new files — manual testing

- [ ] **Step 1: Start the server**

```bash
python -m uvicorn server.main:app --reload
```

- [ ] **Step 2: Start the frontend**

```bash
cd frontend && npm run dev
```

- [ ] **Step 3: Test panorama building**

1. Open http://localhost:5173
2. Use arrow keys (left/right) to change the virtual pan angle
3. Rotate webcam by hand to match
4. Verify tiles appear in the panorama view as you sweep

- [ ] **Step 4: Test zone drawing on panorama**

1. Click "+ Draw Zone"
2. Draw a zone on the panorama
3. Name and save
4. Verify zone appears on both panorama and live feed

- [ ] **Step 5: Test detection and state machine**

1. Point a cat picture at the webcam inside a drawn zone
2. Verify WARNING state appears with countdown
3. After warning expires, verify ZAP fires
4. Remove cat picture, verify return to SWEEPING

- [ ] **Step 6: Test direction arrow (dev mode)**

1. Draw a zone off to the side of the current camera view
2. Place cat picture in webcam view inside that zone (angle-space)
3. Verify yellow flashing arrow appears pointing toward the cat

- [ ] **Step 7: Run unit tests**

```bash
python -m pytest server/tests/ -v
```

- [ ] **Step 8: Commit**

```
feat: panoramic map system complete — ready for hardware integration
```
