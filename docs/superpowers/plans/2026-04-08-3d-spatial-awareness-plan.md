# 3D Spatial Awareness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sweep pause/e-stop controls, MiDaS depth estimation, volumetric 3D exclusion zones, and occlusion-aware cat tracking to CatZap.

**Architecture:** Four phases building on each other. Phase 1 (sweep controls) is standalone. Phase 2 (depth/room model) provides the foundation. Phase 3 (volumetric zones) and Phase 4 (occlusion tracking) build on top. Each phase produces working software with tests.

**Tech Stack:** Python/FastAPI, MiDaS v3.1 (torch), filterpy (Kalman), Shapely, asyncpg, React/TypeScript

**Spec:** `docs/superpowers/specs/2026-04-08-3d-spatial-awareness-design.md`

---

## File Map

### New Files (Server)
| File | Responsibility |
|---|---|
| `server/spatial/__init__.py` | Package init |
| `server/spatial/depth_estimator.py` | MiDaS wrapper — run depth on a frame, return depth map |
| `server/spatial/room_model.py` | RoomModel + FurnitureObject data classes, heightmap management, persistence |
| `server/spatial/projection.py` | Angle-space ↔ room-space coordinate transforms using depth |
| `server/spatial/cat_tracker.py` | Per-cat Kalman tracker, occlusion detection, trajectory prediction |
| `server/routers/spatial.py` | API endpoints: calibration, furniture CRUD, room model status |
| `server/tests/test_sweep_pause.py` | Tests for PAUSED + STOPPED states |
| `server/tests/test_depth_estimator.py` | Tests for MiDaS wrapper |
| `server/tests/test_room_model.py` | Tests for room model + furniture |
| `server/tests/test_projection.py` | Tests for coordinate transforms |
| `server/tests/test_cat_tracker.py` | Tests for tracker + occlusion |
| `server/tests/test_volumetric_zones.py` | Tests for 3D zone violation checks |

### New Files (Frontend)
| File | Responsibility |
|---|---|
| `frontend/src/components/SweepControls.tsx` | Pause/resume + E-stop buttons |
| `frontend/src/components/CalibrationWizard.tsx` | Guided room calibration flow |
| `frontend/src/components/HeightSlider.tsx` | Height min/max slider for 3D zone editing |

### Modified Files
| File | Changes |
|---|---|
| `server/panorama/sweep_controller.py` | Add PAUSED + STOPPED states, pause queueing |
| `server/routers/control.py` | Add pause, e-stop, clear-estop endpoints |
| `server/vision/zone_checker.py` | Add 3D prism violation check with 2D fallback |
| `server/models/database.py` | Add zone 3D columns, furniture table, room_model table |
| `server/routers/zones.py` | Extend zone CRUD with 3D fields |
| `server/config.py` | Add depth/spatial settings |
| `server/main.py` | Integrate depth pipeline, cat tracker into vision loop |
| `frontend/src/types/index.ts` | Add 3D zone fields, furniture types, control status fields |
| `frontend/src/api/client.ts` | Add spatial/calibration/pause/estop API calls |
| `frontend/src/App.tsx` | Add SweepControls to sidebar |
| `frontend/src/components/Controls.tsx` | Add pause + e-stop buttons |
| `frontend/src/components/StateIndicator.tsx` | Handle PAUSED + STOPPED states |
| `frontend/src/components/ZoneEditor.tsx` | Add mode selector + height sliders |
| `frontend/src/components/LiveFeed.tsx` | Render predicted cat positions when occluded |

---

## Phase 1: Sweep Pause & Emergency Stop

### Task 1: Add PAUSED and STOPPED states to SweepController

**Files:**
- Modify: `server/panorama/sweep_controller.py`
- Test: `server/tests/test_sweep_pause.py`

- [ ] **Step 1: Write failing tests for PAUSED state**

```python
# server/tests/test_sweep_pause.py
import time
from unittest.mock import MagicMock
from server.panorama.sweep_controller import SweepController, SweepState


def make_controller(**kwargs):
    actuator = MagicMock()
    defaults = dict(
        actuator=actuator, pan_min=30, pan_max=150,
        tilt=45, speed=10, dev_mode=True,
    )
    defaults.update(kwargs)
    return SweepController(**defaults)


def test_pause_from_sweeping():
    sc = make_controller()
    assert sc.state == SweepState.SWEEPING
    sc.pause()
    assert sc.state == SweepState.PAUSED


def test_pause_freezes_position():
    sc = make_controller()
    sc.current_pan = 90.0
    sc.pause()
    old_pan = sc.current_pan
    sc.tick(1.0)
    assert sc.current_pan == old_pan


def test_resume_from_paused():
    sc = make_controller()
    sc.pause()
    sc.resume()
    assert sc.state == SweepState.SWEEPING


def test_resume_continues_from_position():
    sc = make_controller()
    sc.current_pan = 90.0
    sc.pause()
    sc.resume()
    sc.tick(1.0)
    assert sc.current_pan != 90.0


def test_pause_queues_during_warning():
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    assert sc.state == SweepState.WARNING
    sc.pause()
    # Should NOT pause yet — still in WARNING
    assert sc.state == SweepState.WARNING
    assert sc.pause_queued is True


def test_queued_pause_activates_on_return_to_sweeping():
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    sc.pause()
    assert sc.pause_queued is True
    sc.on_cat_left_zone()
    # Should now be PAUSED instead of SWEEPING
    assert sc.state == SweepState.PAUSED
```

- [ ] **Step 2: Write failing tests for STOPPED state**

Add to the same file:

```python
def test_emergency_stop_from_sweeping():
    sc = make_controller()
    sc.emergency_stop()
    assert sc.state == SweepState.STOPPED
    assert sc.armed is False


def test_emergency_stop_from_warning():
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    sc.emergency_stop()
    assert sc.state == SweepState.STOPPED
    assert sc.armed is False


def test_emergency_stop_from_firing():
    sc = make_controller()
    sc.state = SweepState.FIRING
    sc.emergency_stop()
    assert sc.state == SweepState.STOPPED
    assert sc.armed is False


def test_stopped_blocks_all_motion():
    sc = make_controller()
    sc.current_pan = 90.0
    sc.emergency_stop()
    sc.tick(1.0)
    assert sc.current_pan == 90.0


def test_clear_estop():
    sc = make_controller()
    sc.emergency_stop()
    sc.clear_emergency_stop()
    assert sc.state == SweepState.SWEEPING
    assert sc.armed is False  # stays disarmed after clear


def test_clear_estop_only_works_when_stopped():
    sc = make_controller()
    assert sc.state == SweepState.SWEEPING
    sc.clear_emergency_stop()  # no-op
    assert sc.state == SweepState.SWEEPING
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_sweep_pause.py -v`
Expected: FAIL — `SweepState` has no `PAUSED`/`STOPPED`, no `pause()`/`emergency_stop()` methods

- [ ] **Step 4: Implement PAUSED and STOPPED states**

```python
# server/panorama/sweep_controller.py
# Add to SweepState enum:
class SweepState(str, Enum):
    SWEEPING = "SWEEPING"
    WARNING = "WARNING"
    FIRING = "FIRING"
    TRACKING = "TRACKING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
```

Add to `SweepController.__init__`:
```python
        self.armed = True
        self.pause_queued = False
```

Add these methods to `SweepController`:
```python
    def pause(self):
        """Pause sweep. Queues if not in SWEEPING state."""
        if self.state == SweepState.SWEEPING:
            self.state = SweepState.PAUSED
            self.pause_queued = False
        elif self.state != SweepState.STOPPED:
            self.pause_queued = True

    def resume(self):
        """Resume sweep from paused position."""
        if self.state == SweepState.PAUSED:
            self.state = SweepState.SWEEPING
        self.pause_queued = False

    def emergency_stop(self):
        """Immediate stop from any state. Disarms system."""
        self.state = SweepState.STOPPED
        self.armed = False
        self.pause_queued = False

    def clear_emergency_stop(self):
        """Clear e-stop. Returns to SWEEPING but stays disarmed."""
        if self.state == SweepState.STOPPED:
            self.state = SweepState.SWEEPING
            # armed stays False — user must re-arm separately
```

Modify `tick()` — add early return for PAUSED and STOPPED:
```python
    def tick(self, dt: float):
        """Advance the state machine by dt seconds."""
        if self.state in (SweepState.PAUSED, SweepState.STOPPED):
            return

        now = time.time()
        # ... rest of existing tick logic unchanged
```

Modify `on_cat_left_zone()` — check for queued pause:
```python
    def on_cat_left_zone(self):
        """Called when the cat leaves all forbidden zones."""
        if self.state == SweepState.WARNING:
            if self.pause_queued:
                self.state = SweepState.PAUSED
                self.pause_queued = False
            else:
                self.state = SweepState.SWEEPING
            self._was_tracking = False
        elif self.state in (SweepState.FIRING, SweepState.TRACKING):
            self.state = SweepState.TRACKING
            self._tracking_start = time.time()
```

Also modify the transition from TRACKING → SWEEPING in `tick()` to check for queued pause:
```python
            elif self.state == SweepState.TRACKING:
                elapsed = now - self._tracking_start
                if elapsed >= self.tracking_duration:
                    if self.pause_queued:
                        self.state = SweepState.PAUSED
                        self.pause_queued = False
                    else:
                        self.state = SweepState.SWEEPING
                    self._was_tracking = False
```

Modify `should_fire()` to check armed status:
```python
    def should_fire(self) -> bool:
        """Check if the system should fire right now."""
        return self.state == SweepState.FIRING and self.armed
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_sweep_pause.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add server/panorama/sweep_controller.py server/tests/test_sweep_pause.py
git commit -m "feat: add PAUSED and STOPPED states to sweep controller"
```

### Task 2: Add pause/e-stop API endpoints

**Files:**
- Modify: `server/routers/control.py`

- [ ] **Step 1: Add pause endpoint**

Add to `server/routers/control.py`:
```python
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
```

Add the import at the top of the file:
```python
from server.panorama.sweep_controller import SweepState
```

- [ ] **Step 2: Add e-stop and clear endpoints**

```python
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
    _armed = False  # stays disarmed
    return {"stopped": False, "armed": False}
```

- [ ] **Step 3: Update status endpoint to include new states**

Update the existing `get_status` endpoint:
```python
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
        "paused": sc.state == SweepState.PAUSED if sc else False,
        "stopped": sc.state == SweepState.STOPPED if sc else False,
        "pause_queued": sc.pause_queued if sc else False,
    }
```

- [ ] **Step 4: Commit**

```bash
git add server/routers/control.py
git commit -m "feat: add pause, e-stop, clear-estop API endpoints"
```

### Task 3: Frontend sweep control buttons

**Files:**
- Create: `frontend/src/components/SweepControls.tsx`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/StateIndicator.tsx`
- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Add API client functions**

Add to `frontend/src/api/client.ts`:
```typescript
export const togglePause = () =>
  fetchJSON<{ paused: boolean }>("/control/pause", { method: "POST" });

export const emergencyStop = () =>
  fetchJSON<{ stopped: boolean }>("/control/emergency-stop", { method: "POST" });

export const clearEmergencyStop = () =>
  fetchJSON<{ stopped: boolean; armed: boolean }>("/control/clear-estop", { method: "POST" });
```

- [ ] **Step 2: Update types**

Add to the `ControlStatus` interface in `frontend/src/types/index.ts`:
```typescript
export interface ControlStatus {
  armed: boolean;
  state: string;
  servo_pan: number;
  servo_tilt: number;
  dev_mode: boolean;
  paused: boolean;
  stopped: boolean;
  pause_queued: boolean;
}
```

- [ ] **Step 3: Update StateIndicator for new states**

Add to `STATE_CONFIG` in `frontend/src/components/StateIndicator.tsx`:
```typescript
const STATE_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  SWEEPING: { color: "var(--cyan)", bg: "var(--cyan-glow)", label: "SWEEP" },
  WARNING: { color: "var(--amber)", bg: "var(--amber-glow)", label: "WARN" },
  FIRING: { color: "var(--red)", bg: "var(--red-glow)", label: "FIRE" },
  TRACKING: { color: "var(--purple)", bg: "var(--purple-dim)", label: "TRACK" },
  PAUSED: { color: "var(--text-secondary)", bg: "var(--bg-elevated)", label: "PAUSED" },
  STOPPED: { color: "var(--red)", bg: "var(--red-glow)", label: "E-STOP" },
};
```

- [ ] **Step 4: Create SweepControls component**

```typescript
// frontend/src/components/SweepControls.tsx
import { useState, useEffect } from "react";
import { getControlStatus, togglePause, emergencyStop, clearEmergencyStop } from "../api/client";

export default function SweepControls() {
  const [paused, setPaused] = useState(false);
  const [stopped, setStopped] = useState(false);
  const [pauseQueued, setPauseQueued] = useState(false);

  useEffect(() => {
    getControlStatus().then((s) => {
      setPaused(s.paused);
      setStopped(s.stopped);
      setPauseQueued(s.pause_queued);
    }).catch(console.error);
  }, []);

  async function handlePause() {
    const res = await togglePause();
    setPaused(res.paused);
    setPauseQueued(res.paused && !paused);
  }

  async function handleEStop() {
    await emergencyStop();
    setStopped(true);
    setPaused(false);
  }

  async function handleClearEStop() {
    await clearEmergencyStop();
    setStopped(false);
  }

  if (stopped) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: "0 4px" }}>
        <div style={{
          width: 40, height: 40,
          display: "flex", alignItems: "center", justifyContent: "center",
          borderRadius: "var(--radius-sm)",
          background: "var(--red-glow)",
          border: "1px solid var(--red)",
          color: "var(--red)",
          fontSize: 9,
          fontFamily: "var(--font-mono)",
          fontWeight: 600,
          animation: "pulse 1s infinite",
        }}>
          STOP
        </div>
        <button
          className="nav-btn"
          onClick={handleClearEStop}
          style={{ fontSize: 10, color: "var(--green)" }}
          title="Clear E-Stop"
        >
          ↺
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: "0 4px" }}>
      {/* Pause / Resume */}
      <div className="tooltip-wrapper" data-tooltip={paused ? "Resume" : pauseQueued ? "Pause queued" : "Pause"}>
        <button
          className={`nav-btn ${paused || pauseQueued ? "active" : ""}`}
          onClick={handlePause}
          style={pauseQueued ? { color: "var(--amber)", opacity: 0.6 } : {}}
        >
          {paused ? "▶" : "⏸"}
        </button>
      </div>

      {/* E-Stop */}
      <div className="tooltip-wrapper" data-tooltip="Emergency Stop">
        <button
          className="nav-btn"
          onClick={handleEStop}
          style={{
            color: "var(--red)",
            fontWeight: 800,
            fontSize: 14,
          }}
        >
          ■
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Add SweepControls to App sidebar**

In `frontend/src/App.tsx`, add the import and place the component in the sidebar between the nav buttons and the zone edit toggle:

```typescript
import SweepControls from "./components/SweepControls";
```

In the sidebar JSX, after the nav items and before `<div style={{ flex: 1 }} />`:
```tsx
        <div className="divider" style={{ width: 28, margin: "6px auto" }} />
        <SweepControls />
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/SweepControls.tsx frontend/src/api/client.ts frontend/src/App.tsx frontend/src/components/StateIndicator.tsx frontend/src/types/index.ts
git commit -m "feat: add pause/e-stop controls to frontend"
```

---

## Phase 2: Depth Estimation & Room Model

### Task 4: Add spatial config settings

**Files:**
- Modify: `server/config.py`

- [ ] **Step 1: Add depth and spatial settings**

Add to the `Settings` class in `server/config.py`:
```python
    # Depth / Spatial
    midas_model: str = "MiDaS_small"  # MiDaS model variant
    depth_run_interval: int = 5  # run depth every Nth tile refresh
    depth_blend_alpha: float = 0.2  # EMA blend factor for heightmap
    depth_change_threshold: float = 20.0  # cm change to flag furniture move
    heightmap_resolution: float = 5.0  # cm per cell
    room_width_cm: float = 500.0  # room dimensions for model
    room_depth_cm: float = 500.0
    room_height_cm: float = 300.0
    camera_height_cm: float = 150.0  # camera mount height from floor
    occlusion_timeout: float = 10.0  # seconds before giving up on occluded cat
    occlusion_grace_frames: int = 3  # frames before declaring cat lost (no occluder)
```

- [ ] **Step 2: Commit**

```bash
git add server/config.py
git commit -m "feat: add depth/spatial config settings"
```

### Task 5: MiDaS depth estimator wrapper

**Files:**
- Create: `server/spatial/__init__.py`
- Create: `server/spatial/depth_estimator.py`
- Test: `server/tests/test_depth_estimator.py`

- [ ] **Step 1: Create package init**

```python
# server/spatial/__init__.py
```

- [ ] **Step 2: Write failing test**

```python
# server/tests/test_depth_estimator.py
import numpy as np
from server.spatial.depth_estimator import DepthEstimator


def test_estimate_returns_depth_map():
    estimator = DepthEstimator()
    # Create a fake 480x640 RGB frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = estimator.estimate(frame)
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32
    assert np.all(depth >= 0)


def test_depth_to_metric():
    estimator = DepthEstimator()
    estimator.depth_scale = 100.0  # 1 unit = 100cm
    relative_depth = np.array([[0.5, 1.0], [0.25, 2.0]], dtype=np.float32)
    metric = estimator.to_metric(relative_depth)
    # metric = depth_scale / relative (inverse depth)
    np.testing.assert_allclose(metric, [[200.0, 100.0], [400.0, 50.0]])


def test_calibrate_scale():
    estimator = DepthEstimator()
    # Simulate: user says reference distance is 75cm
    # At the reference pixel, relative depth value is 1.5
    relative_depth = np.ones((100, 100), dtype=np.float32) * 1.5
    estimator.calibrate_scale(relative_depth, pixel=(50, 50), real_distance_cm=75.0)
    # depth_scale = real_distance * relative_value = 75 * 1.5 = 112.5
    assert abs(estimator.depth_scale - 112.5) < 0.01
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_depth_estimator.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement DepthEstimator**

```python
# server/spatial/depth_estimator.py
import numpy as np
import torch


class DepthEstimator:
    """Wraps MiDaS for monocular depth estimation."""

    def __init__(self, model_type: str = "MiDaS_small"):
        self.model_type = model_type
        self.depth_scale: float = 1.0  # relative-to-metric multiplier
        self._model = None
        self._transform = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        if self._model is not None:
            return
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self._model.to(self._device)
        self._model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "MiDaS_small":
            self._transform = midas_transforms.small_transform
        else:
            self._transform = midas_transforms.dpt_transform

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Run depth estimation on a BGR frame. Returns float32 depth map (same H/W as input).

        Output is relative inverse depth — higher values are closer.
        """
        self._load_model()
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)
        depth = np.maximum(depth, 0.0)
        return depth

    def to_metric(self, relative_depth: np.ndarray) -> np.ndarray:
        """Convert relative inverse depth to metric distance (cm).

        metric_distance = depth_scale / relative_depth
        """
        safe = np.maximum(relative_depth, 1e-6)
        return (self.depth_scale / safe).astype(np.float32)

    def calibrate_scale(
        self, relative_depth: np.ndarray, pixel: tuple[int, int], real_distance_cm: float
    ):
        """Set depth_scale using a known real-world measurement.

        depth_scale = real_distance_cm * relative_depth_at_pixel
        """
        y, x = pixel[1], pixel[0]
        rel_val = float(relative_depth[y, x])
        if rel_val > 0:
            self.depth_scale = real_distance_cm * rel_val
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_depth_estimator.py -v`
Expected: `test_depth_to_metric` and `test_calibrate_scale` PASS. `test_estimate_returns_depth_map` may need torch/MiDaS downloaded — if so, mark as integration test and skip in CI with `@pytest.mark.skipif(not torch.cuda.is_available() and os.environ.get("CI"), ...)`.

- [ ] **Step 6: Commit**

```bash
git add server/spatial/__init__.py server/spatial/depth_estimator.py server/tests/test_depth_estimator.py
git commit -m "feat: add MiDaS depth estimator wrapper"
```

### Task 6: Room model with heightmap and furniture

**Files:**
- Create: `server/spatial/room_model.py`
- Test: `server/tests/test_room_model.py`

- [ ] **Step 1: Write failing tests**

```python
# server/tests/test_room_model.py
import numpy as np
from server.spatial.room_model import RoomModel, FurnitureObject


def test_create_room_model():
    rm = RoomModel(
        width_cm=400, depth_cm=400, height_cm=300,
        resolution=5.0,
    )
    # 400/5 = 80 cells in each horizontal dimension
    assert rm.heightmap.shape == (80, 80)
    assert rm.furniture == []


def test_update_heightmap_cell():
    rm = RoomModel(width_cm=100, depth_cm=100, height_cm=300, resolution=10.0)
    # First reading: direct assign
    rm.update_cell(0, 0, floor_height=0.0, max_height=75.0)
    cell = rm.get_cell(0, 0)
    assert cell["max_height"] == 75.0
    assert cell["readings"] == 1

    # Second reading: blends
    rm.update_cell(0, 0, floor_height=0.0, max_height=80.0)
    cell = rm.get_cell(0, 0)
    assert cell["readings"] == 2
    # With < 3 readings, uses equal weight: (75 + 80) / 2 = 77.5
    assert abs(cell["max_height"] - 77.5) < 0.1


def test_add_furniture():
    rm = RoomModel(width_cm=400, depth_cm=400, height_cm=300, resolution=5.0)
    table = FurnitureObject(
        name="table",
        base_polygon=[(100, 100), (200, 100), (200, 150), (100, 150)],
        height_min=0.0,
        height_max=75.0,
    )
    rm.add_furniture(table)
    assert len(rm.furniture) == 1
    assert rm.furniture[0].name == "table"


def test_point_in_furniture():
    table = FurnitureObject(
        name="table",
        base_polygon=[(100, 100), (200, 100), (200, 150), (100, 150)],
        height_min=0.0,
        height_max=75.0,
    )
    # Point on the table surface
    assert table.contains_point(150, 125, 70) is True
    # Point above the table
    assert table.contains_point(150, 125, 80) is False
    # Point outside the base polygon
    assert table.contains_point(50, 50, 50) is False


def test_detect_furniture_change():
    rm = RoomModel(width_cm=100, depth_cm=100, height_cm=300, resolution=10.0)
    # Fill some cells with readings
    for i in range(4):
        rm.update_cell(5, 5, floor_height=0.0, max_height=75.0)
    # Now a reading that's 30cm different
    changed = rm.check_cell_change(5, 5, new_max_height=105.0, threshold_cm=20.0)
    assert changed is True
    # Small change
    changed = rm.check_cell_change(5, 5, new_max_height=78.0, threshold_cm=20.0)
    assert changed is False


def test_furniture_to_dict_and_back():
    table = FurnitureObject(
        name="table",
        base_polygon=[(100, 100), (200, 100), (200, 150), (100, 150)],
        height_min=0.0,
        height_max=75.0,
    )
    d = table.to_dict()
    restored = FurnitureObject.from_dict(d)
    assert restored.name == "table"
    assert restored.height_max == 75.0
    assert len(restored.base_polygon) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_room_model.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement RoomModel and FurnitureObject**

```python
# server/spatial/room_model.py
import uuid
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon


@dataclass
class FurnitureObject:
    name: str
    base_polygon: list[tuple[float, float]]  # (x, y) in room-space cm
    height_min: float = 0.0
    height_max: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    depth_anchored: bool = False

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a 3D point is inside this furniture volume."""
        if z < self.height_min or z > self.height_max:
            return False
        poly = Polygon(self.base_polygon)
        return poly.contains(Point(x, y))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "base_polygon": self.base_polygon,
            "height_min": self.height_min,
            "height_max": self.height_max,
            "depth_anchored": self.depth_anchored,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FurnitureObject":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            name=d["name"],
            base_polygon=[tuple(p) for p in d["base_polygon"]],
            height_min=d["height_min"],
            height_max=d["height_max"],
            depth_anchored=d.get("depth_anchored", False),
        )


class RoomModel:
    """2.5D room model with heightmap and named furniture volumes."""

    def __init__(
        self,
        width_cm: float,
        depth_cm: float,
        height_cm: float,
        resolution: float = 5.0,
    ):
        self.width_cm = width_cm
        self.depth_cm = depth_cm
        self.height_cm = height_cm
        self.resolution = resolution

        cols = int(width_cm / resolution)
        rows = int(depth_cm / resolution)

        # Each cell: [floor_height, max_height, readings_count]
        self._grid = np.zeros((rows, cols, 3), dtype=np.float32)
        self.furniture: list[FurnitureObject] = []
        self.depth_scale: float = 1.0

    @property
    def heightmap(self) -> np.ndarray:
        """Return max_height layer for external use."""
        return self._grid[:, :, 1]

    def update_cell(self, row: int, col: int, floor_height: float, max_height: float):
        """Update a heightmap cell with a new depth reading."""
        readings = int(self._grid[row, col, 2])
        if readings == 0:
            self._grid[row, col, 0] = floor_height
            self._grid[row, col, 1] = max_height
        elif readings < 3:
            # Equal weight blend for first few readings
            n = readings + 1
            self._grid[row, col, 0] = (self._grid[row, col, 0] * readings + floor_height) / n
            self._grid[row, col, 1] = (self._grid[row, col, 1] * readings + max_height) / n
        else:
            # EMA blend
            alpha = 0.2
            self._grid[row, col, 0] = (1 - alpha) * self._grid[row, col, 0] + alpha * floor_height
            self._grid[row, col, 1] = (1 - alpha) * self._grid[row, col, 1] + alpha * max_height
        self._grid[row, col, 2] += 1

    def get_cell(self, row: int, col: int) -> dict:
        return {
            "floor_height": float(self._grid[row, col, 0]),
            "max_height": float(self._grid[row, col, 1]),
            "readings": int(self._grid[row, col, 2]),
        }

    def check_cell_change(
        self, row: int, col: int, new_max_height: float, threshold_cm: float = 20.0
    ) -> bool:
        """Check if a new reading differs significantly from stored value."""
        readings = int(self._grid[row, col, 2])
        if readings < 3:
            return False
        current = float(self._grid[row, col, 1])
        return abs(new_max_height - current) > threshold_cm

    def add_furniture(self, obj: FurnitureObject):
        self.furniture.append(obj)

    def remove_furniture(self, furniture_id: str) -> bool:
        before = len(self.furniture)
        self.furniture = [f for f in self.furniture if f.id != furniture_id]
        return len(self.furniture) < before

    def get_occluding_furniture(
        self, camera_x: float, camera_y: float, target_x: float, target_y: float
    ) -> list[FurnitureObject]:
        """Find furniture objects between camera and target position (2D check)."""
        from shapely.geometry import LineString
        line = LineString([(camera_x, camera_y), (target_x, target_y)])
        occluders = []
        for f in self.furniture:
            poly = Polygon(f.base_polygon)
            if line.intersects(poly):
                occluders.append(f)
        return occluders
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_room_model.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/spatial/room_model.py server/tests/test_room_model.py
git commit -m "feat: add room model with heightmap and furniture volumes"
```

### Task 7: Angle-space to room-space projection

**Files:**
- Create: `server/spatial/projection.py`
- Test: `server/tests/test_projection.py`

- [ ] **Step 1: Write failing tests**

```python
# server/tests/test_projection.py
import numpy as np
from server.spatial.projection import (
    angle_depth_to_room,
    room_to_angle,
    project_furniture_to_angles,
)


def test_angle_depth_to_room_center():
    """Camera at origin, looking straight ahead (pan=90, tilt=45), object at 200cm."""
    camera_pos = (0.0, 0.0, 150.0)  # camera 150cm high
    pos = angle_depth_to_room(
        pan=90.0, tilt=45.0, depth_cm=200.0, camera_pos=camera_pos
    )
    # At pan=90 (straight), tilt=45 (level-ish), depth=200:
    # x = 200 * cos(tilt_rad) * sin(pan_rad - pi/2) → but pan=90 means forward
    # For our coordinate system: x = depth * sin(pan_offset), y = depth * cos(pan_offset)
    assert len(pos) == 3
    assert pos[2] < 150.0  # tilt=45 means looking slightly down from camera height


def test_room_to_angle_roundtrip():
    """Convert to room-space and back should be close to original."""
    camera_pos = (0.0, 0.0, 150.0)
    pan, tilt, depth = 75.0, 40.0, 180.0
    room_pos = angle_depth_to_room(pan, tilt, depth, camera_pos)
    pan2, tilt2 = room_to_angle(room_pos[0], room_pos[1], room_pos[2], camera_pos)
    assert abs(pan2 - pan) < 0.5
    assert abs(tilt2 - tilt) < 0.5


def test_project_furniture_to_angles():
    """Project a furniture base polygon from room-space to angle-space."""
    camera_pos = (0.0, 0.0, 150.0)
    base_polygon = [(100, 200), (200, 200), (200, 300), (100, 300)]
    height_min = 0.0
    height_max = 75.0
    angles = project_furniture_to_angles(base_polygon, height_min, height_max, camera_pos)
    # Should return a list of (pan, tilt) pairs forming the projected silhouette
    assert len(angles) >= 4  # at least the 4 corners
    for pan, tilt in angles:
        assert 0 <= pan <= 180
        assert 0 <= tilt <= 90
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_projection.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement projection functions**

```python
# server/spatial/projection.py
import math


def angle_depth_to_room(
    pan: float, tilt: float, depth_cm: float,
    camera_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert angle-space + depth to room-space (x, y, z) in cm.

    Coordinate system:
    - x: left-right (pan axis)
    - y: forward-back (depth axis)
    - z: up-down (height)

    Pan 90° = straight ahead. Tilt 0° = looking up, 90° = looking down.
    """
    # Convert to radians, with pan=90 as forward (0 offset)
    pan_rad = math.radians(pan - 90.0)
    tilt_rad = math.radians(tilt)

    # Project depth along the ray
    horizontal_dist = depth_cm * math.cos(tilt_rad)
    x = camera_pos[0] + horizontal_dist * math.sin(pan_rad)
    y = camera_pos[1] + horizontal_dist * math.cos(pan_rad)
    z = camera_pos[2] - depth_cm * math.sin(tilt_rad)

    return (x, y, z)


def room_to_angle(
    x: float, y: float, z: float,
    camera_pos: tuple[float, float, float],
) -> tuple[float, float]:
    """Convert room-space (x, y, z) back to angle-space (pan, tilt)."""
    dx = x - camera_pos[0]
    dy = y - camera_pos[1]
    dz = camera_pos[2] - z  # positive dz = below camera

    horizontal_dist = math.sqrt(dx * dx + dy * dy)
    pan_rad = math.atan2(dx, dy)
    pan = math.degrees(pan_rad) + 90.0

    total_dist = math.sqrt(horizontal_dist * horizontal_dist + dz * dz)
    tilt_rad = math.atan2(dz, horizontal_dist) if horizontal_dist > 0 else (math.pi / 2 if dz > 0 else 0)
    tilt = math.degrees(tilt_rad)

    return (pan, tilt)


def project_furniture_to_angles(
    base_polygon: list[tuple[float, float]],
    height_min: float,
    height_max: float,
    camera_pos: tuple[float, float, float],
) -> list[tuple[float, float]]:
    """Project a furniture volume's silhouette into angle-space.

    Returns the convex hull of all corner projections as (pan, tilt) pairs.
    """
    angle_points = []
    for x, y in base_polygon:
        # Project bottom corners
        pan_lo, tilt_lo = room_to_angle(x, y, height_min, camera_pos)
        angle_points.append((pan_lo, tilt_lo))
        # Project top corners
        pan_hi, tilt_hi = room_to_angle(x, y, height_max, camera_pos)
        angle_points.append((pan_hi, tilt_hi))
    return angle_points
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_projection.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/spatial/projection.py server/tests/test_projection.py
git commit -m "feat: add angle/room-space projection transforms"
```

---

## Phase 3: Volumetric Exclusion Zones

### Task 8: Extend zone data model for 3D

**Files:**
- Modify: `server/models/database.py`
- Modify: `server/routers/zones.py`

- [ ] **Step 1: Add 3D columns to zones schema**

Add to the `SCHEMA` string in `server/models/database.py`, after the zones table creation:
```sql
ALTER TABLE zones ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT '2d';
ALTER TABLE zones ADD COLUMN IF NOT EXISTS room_polygon JSONB;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS height_min REAL NOT NULL DEFAULT 0.0;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS height_max REAL NOT NULL DEFAULT 0.0;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS furniture_id UUID;
```

Note: Use `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to avoid errors on existing databases.

- [ ] **Step 2: Update get_zones to include new fields**

In the `get_zones` function, update the dict comprehension to include:
```python
                "mode": row["mode"],
                "room_polygon": json.loads(row["room_polygon"]) if row["room_polygon"] and isinstance(row["room_polygon"], str) else row["room_polygon"],
                "height_min": row["height_min"],
                "height_max": row["height_max"],
                "furniture_id": str(row["furniture_id"]) if row["furniture_id"] else None,
```

- [ ] **Step 3: Update create_zone to accept 3D fields**

Update the `create_zone` function signature and query:
```python
async def create_zone(
    name: str,
    polygon: list[list[float]],
    overlap_threshold: float = 0.3,
    cooldown_seconds: int = 3,
    mode: str = "2d",
    room_polygon: list[list[float]] | None = None,
    height_min: float = 0.0,
    height_max: float = 0.0,
    furniture_id: str | None = None,
    conn: asyncpg.Connection | None = None,
) -> str:
    zone_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            """INSERT INTO zones (id, name, polygon, overlap_threshold, cooldown_seconds,
               mode, room_polygon, height_min, height_max, furniture_id)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
            zone_id, name, json.dumps(polygon), overlap_threshold, cooldown_seconds,
            mode,
            json.dumps(room_polygon) if room_polygon else None,
            height_min, height_max,
            uuid.UUID(furniture_id) if furniture_id else None,
        )
        return str(zone_id)
    finally:
        if conn is None:
            await pool.release(c)
```

- [ ] **Step 4: Update zone router models**

In `server/routers/zones.py`, update `ZoneCreate` and `ZoneUpdate`:
```python
class ZoneCreate(BaseModel):
    name: str
    polygon: list[list[float]]
    overlap_threshold: float = 0.3
    cooldown_seconds: int = 3
    mode: str = "2d"
    room_polygon: list[list[float]] | None = None
    height_min: float = 0.0
    height_max: float = 0.0
    furniture_id: str | None = None


class ZoneUpdate(BaseModel):
    name: str | None = None
    polygon: list[list[float]] | None = None
    overlap_threshold: float | None = None
    cooldown_seconds: int | None = None
    enabled: bool | None = None
    mode: str | None = None
    room_polygon: list[list[float]] | None = None
    height_min: float | None = None
    height_max: float | None = None
    furniture_id: str | None = None
```

Update the `create_zone_endpoint` to pass through the new fields:
```python
@router.post("", status_code=201)
async def create_zone_endpoint(zone: ZoneCreate):
    zone_id = await create_zone(
        name=zone.name,
        polygon=zone.polygon,
        overlap_threshold=zone.overlap_threshold,
        cooldown_seconds=zone.cooldown_seconds,
        mode=zone.mode,
        room_polygon=zone.room_polygon,
        height_min=zone.height_min,
        height_max=zone.height_max,
        furniture_id=zone.furniture_id,
    )
    zones = await get_zones()
    return next(z for z in zones if z["id"] == zone_id)
```

Also update `update_zone` in `database.py` to allow the new fields:
```python
    allowed = {"name", "polygon", "overlap_threshold", "cooldown_seconds", "enabled",
               "mode", "room_polygon", "height_min", "height_max", "furniture_id"}
```

And add serialization for `room_polygon` (same as `polygon`):
```python
    if "room_polygon" in updates:
        updates["room_polygon"] = json.dumps(updates["room_polygon"]) if updates["room_polygon"] else None
```

- [ ] **Step 5: Commit**

```bash
git add server/models/database.py server/routers/zones.py
git commit -m "feat: extend zone data model with 3D fields"
```

### Task 9: 3D zone violation checker

**Files:**
- Modify: `server/vision/zone_checker.py`
- Test: `server/tests/test_volumetric_zones.py`

- [ ] **Step 1: Write failing tests**

```python
# server/tests/test_volumetric_zones.py
from server.vision.zone_checker import check_zone_violations, check_3d_zone_violation


def test_2d_zone_unchanged():
    """Existing 2D zones still work exactly as before."""
    bbox = [0.3, 0.3, 0.7, 0.7]
    zones = [{
        "id": "z1", "name": "couch", "enabled": True,
        "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "overlap_threshold": 0.3,
        "mode": "2d",
    }]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 1
    assert violations[0]["zone_name"] == "couch"


def test_3d_zone_cat_inside_volume():
    """Cat position inside the 3D prism should violate."""
    zone = {
        "id": "z1", "name": "table", "enabled": True,
        "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0,
        "height_max": 75.0,
    }
    # Cat at (150, 150, 50) — inside the table volume
    result = check_3d_zone_violation(
        cat_room_pos=(150.0, 150.0, 50.0), zone=zone
    )
    assert result is True


def test_3d_zone_cat_on_top():
    """Cat ON the table (z > height_max) should NOT violate with exact bounds."""
    zone = {
        "id": "z1", "name": "table", "enabled": True,
        "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0,
        "height_max": 75.0,
    }
    # Cat at z=80 — above the table
    result = check_3d_zone_violation(
        cat_room_pos=(150.0, 150.0, 80.0), zone=zone
    )
    assert result is False


def test_3d_zone_cat_on_table_surface():
    """If zone is defined to include the surface (height_max=80), cat at z=75 violates."""
    zone = {
        "id": "z1", "name": "table-top", "enabled": True,
        "mode": "manual_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 70.0,  # just the surface zone
        "height_max": 110.0,  # surface + cat height
    }
    result = check_3d_zone_violation(
        cat_room_pos=(150.0, 150.0, 75.0), zone=zone
    )
    assert result is True


def test_3d_zone_cat_outside_polygon():
    """Cat outside the base polygon at correct height should not violate."""
    zone = {
        "id": "z1", "name": "table", "enabled": True,
        "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0,
        "height_max": 75.0,
    }
    result = check_3d_zone_violation(
        cat_room_pos=(50.0, 50.0, 50.0), zone=zone
    )
    assert result is False


def test_2d_fallback_when_no_room_pos():
    """3D zone falls back to 2D check when depth data is unavailable."""
    bbox = [0.3, 0.3, 0.7, 0.7]
    zones = [{
        "id": "z1", "name": "table", "enabled": True,
        "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "overlap_threshold": 0.3,
        "mode": "auto_3d",
        "room_polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
        "height_min": 0.0,
        "height_max": 75.0,
    }]
    # No cat_room_pos passed — falls back to 2D
    violations = check_zone_violations(bbox, zones, cat_room_pos=None)
    assert len(violations) == 1  # 2D fallback triggers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_volumetric_zones.py -v`
Expected: FAIL — `check_3d_zone_violation` not found, `check_zone_violations` doesn't accept `cat_room_pos`

- [ ] **Step 3: Implement 3D violation checking**

Update `server/vision/zone_checker.py`:
```python
from shapely.geometry import Polygon, Point, box
from shapely.validation import make_valid


def check_3d_zone_violation(
    cat_room_pos: tuple[float, float, float], zone: dict
) -> bool:
    """Check if a cat's room-space position is inside a 3D zone prism."""
    room_polygon = zone.get("room_polygon")
    if not room_polygon or len(room_polygon) < 3:
        return False

    x, y, z = cat_room_pos
    height_min = zone.get("height_min", 0.0)
    height_max = zone.get("height_max", 0.0)

    if z < height_min or z > height_max:
        return False

    poly = Polygon(room_polygon)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return False

    return poly.contains(Point(x, y))


def check_zone_violations(
    bbox: list[float],
    zones: list[dict],
    cat_room_pos: tuple[float, float, float] | None = None,
) -> list[dict]:
    """Check if a cat bounding box violates any zones.

    For 3D zones: uses cat_room_pos if available, otherwise falls back to 2D.
    For 2D zones: uses bbox overlap as before.
    """
    cat_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
    cat_area = cat_box.area
    if cat_area == 0:
        return []

    violations = []
    for zone in zones:
        if not zone.get("enabled", True):
            continue

        mode = zone.get("mode", "2d")

        # 3D modes: try room-space check first
        if mode in ("auto_3d", "manual_3d") and cat_room_pos is not None:
            if check_3d_zone_violation(cat_room_pos, zone):
                violations.append({
                    "zone_id": zone["id"],
                    "zone_name": zone["name"],
                    "overlap": 1.0,  # fully inside volume
                })
            continue

        # 2D check (default, or fallback when no room pos)
        zone_poly = Polygon(zone["polygon"])
        if not zone_poly.is_valid:
            zone_poly = make_valid(zone_poly)
        if zone_poly.is_empty or zone_poly.area == 0:
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_volumetric_zones.py server/tests/test_zone_checker.py -v`
Expected: All PASS (both new and existing tests)

- [ ] **Step 5: Commit**

```bash
git add server/vision/zone_checker.py server/tests/test_volumetric_zones.py
git commit -m "feat: add 3D prism zone violation checking with 2D fallback"
```

### Task 10: Zone editor UI — mode selector + height sliders

**Files:**
- Create: `frontend/src/components/HeightSlider.tsx`
- Modify: `frontend/src/components/ZoneEditor.tsx`
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Update Zone type**

In `frontend/src/types/index.ts`, update the Zone interface:
```typescript
export interface Zone {
  id: string;
  name: string;
  polygon: number[][];
  overlap_threshold: number;
  cooldown_seconds: number;
  enabled: boolean;
  created_at: string;
  mode: "2d" | "auto_3d" | "manual_3d";
  room_polygon: number[][] | null;
  height_min: number;
  height_max: number;
  furniture_id: string | null;
}
```

- [ ] **Step 2: Update createZone API call**

In `frontend/src/api/client.ts`, update the `createZone` function:
```typescript
export const createZone = (zone: {
  name: string;
  polygon: number[][];
  overlap_threshold?: number;
  cooldown_seconds?: number;
  mode?: string;
  room_polygon?: number[][];
  height_min?: number;
  height_max?: number;
}) => fetchJSON<Zone>("/zones", { method: "POST", body: JSON.stringify(zone) });
```

- [ ] **Step 3: Create HeightSlider component**

```typescript
// frontend/src/components/HeightSlider.tsx
interface HeightSliderProps {
  heightMin: number;
  heightMax: number;
  onChangeMin: (v: number) => void;
  onChangeMax: (v: number) => void;
  maxRange?: number;
}

export default function HeightSlider({
  heightMin, heightMax, onChangeMin, onChangeMax, maxRange = 300,
}: HeightSliderProps) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 6,
      padding: "10px 12px",
      background: "var(--bg-deep)",
      borderRadius: "var(--radius-sm)",
      border: "1px solid var(--border-subtle)",
    }}>
      <div style={{
        fontFamily: "var(--font-mono)", fontSize: 10,
        color: "var(--text-tertiary)", letterSpacing: "0.05em",
        textTransform: "uppercase",
      }}>
        Height extrusion (cm)
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-tertiary)",
          }}>
            <span>Min</span>
            <span style={{ color: "var(--amber)" }}>{heightMin} cm</span>
          </div>
          <input
            type="range" min={0} max={maxRange} value={heightMin}
            onChange={(e) => onChangeMin(Math.min(Number(e.target.value), heightMax))}
          />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-tertiary)",
          }}>
            <span>Max</span>
            <span style={{ color: "var(--amber)" }}>{heightMax} cm</span>
          </div>
          <input
            type="range" min={0} max={maxRange} value={heightMax}
            onChange={(e) => onChangeMax(Math.max(Number(e.target.value), heightMin))}
          />
        </div>
      </div>

      {/* Visual height bar */}
      <div style={{
        height: 40, position: "relative",
        background: "var(--bg-surface)", borderRadius: 3,
        overflow: "hidden",
      }}>
        <div style={{
          position: "absolute",
          bottom: `${(heightMin / maxRange) * 100}%`,
          height: `${((heightMax - heightMin) / maxRange) * 100}%`,
          left: 0, right: 0,
          background: "var(--amber-glow)",
          border: "1px solid var(--amber-dim)",
          borderRadius: 2,
        }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Update ZoneEditor with mode selector and height sliders**

In `frontend/src/components/ZoneEditor.tsx`, add imports and state:
```typescript
import HeightSlider from "./HeightSlider";
```

Add state variables inside the component:
```typescript
  const [zoneMode, setZoneMode] = useState<"2d" | "auto_3d" | "manual_3d">("2d");
  const [heightMin, setHeightMin] = useState(0);
  const [heightMax, setHeightMax] = useState(0);
```

Update `handleSaveZone` to pass 3D fields:
```typescript
  async function handleSaveZone() {
    if (points.length < 3 || !zoneName.trim()) return;
    const anglePoints = points.map(([nx, ny]) => normalizedToAngle(nx, ny));
    await createZone({
      name: zoneName.trim(),
      polygon: anglePoints,
      mode: zoneMode,
      height_min: zoneMode !== "2d" ? heightMin : 0,
      height_max: zoneMode !== "2d" ? heightMax : 0,
    });
    setRawPoints([]);
    setZoneName("");
    setZoneMode("2d");
    setHeightMin(0);
    setHeightMax(0);
    onSave();
  }
```

In the controls JSX (below the zone name input row), add the mode selector and conditional height slider:
```tsx
        {/* Mode selector + height — shown after drawing */}
        {points.length >= 3 && (
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ display: "flex", gap: 4 }}>
              {(["2d", "auto_3d", "manual_3d"] as const).map((m) => (
                <button
                  key={m}
                  className={`btn btn-sm ${zoneMode === m ? "btn-primary" : ""}`}
                  onClick={() => setZoneMode(m)}
                >
                  {m === "2d" ? "2D Flat" : m === "auto_3d" ? "Auto 3D" : "Manual 3D"}
                </button>
              ))}
            </div>

            {zoneMode !== "2d" && (
              <HeightSlider
                heightMin={heightMin}
                heightMax={heightMax}
                onChangeMin={setHeightMin}
                onChangeMax={setHeightMax}
              />
            )}
          </div>
        )}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/HeightSlider.tsx frontend/src/components/ZoneEditor.tsx frontend/src/types/index.ts frontend/src/api/client.ts
git commit -m "feat: add zone mode selector and height sliders to zone editor"
```

---

## Phase 4: Occlusion-Aware Cat Tracking

### Task 11: Cat tracker with Kalman filter

**Files:**
- Create: `server/spatial/cat_tracker.py`
- Test: `server/tests/test_cat_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# server/tests/test_cat_tracker.py
import time
from server.spatial.cat_tracker import CatTracker, TrackedCat


def test_track_new_cat():
    tracker = CatTracker()
    tracker.update_detection("cat1", (150.0, 200.0, 30.0), time.time())
    cats = tracker.get_active_cats()
    assert len(cats) == 1
    assert cats[0].id == "cat1"
    assert cats[0].state == "visible"


def test_cat_builds_velocity():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.update_detection("cat1", (110.0, 200.0, 30.0), t + 0.1)
    tracker.update_detection("cat1", (120.0, 200.0, 30.0), t + 0.2)
    cat = tracker.get_cat("cat1")
    # Should have positive x velocity (~100 cm/s)
    assert cat.velocity[0] > 50.0


def test_cat_goes_occluded():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (150.0, 200.0, 30.0), t)
    tracker.update_detection("cat1", (160.0, 200.0, 30.0), t + 0.1)
    # Cat disappears, but there's an occluder in the path
    tracker.mark_occluded("cat1", occluder_name="couch")
    cat = tracker.get_cat("cat1")
    assert cat.state == "occluded"
    assert cat.occluded_by == "couch"


def test_predict_position():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.update_detection("cat1", (110.0, 200.0, 30.0), t + 0.1)
    tracker.mark_occluded("cat1", occluder_name="couch")
    predicted = tracker.predict_position("cat1", t + 0.5)
    # Should be ahead of last position in x direction
    assert predicted[0] > 110.0


def test_cat_reconnects():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.mark_occluded("cat1", occluder_name="couch")
    # Cat reappears near predicted position
    tracker.update_detection("cat1", (140.0, 200.0, 30.0), t + 0.5)
    cat = tracker.get_cat("cat1")
    assert cat.state == "visible"
    assert cat.occluded_by is None


def test_cat_goes_lost_after_timeout():
    tracker = CatTracker(occlusion_timeout=1.0)
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.mark_occluded("cat1", occluder_name="couch")
    # Tick past timeout
    tracker.tick(t + 2.0)
    cat = tracker.get_cat("cat1")
    assert cat.state == "lost"


def test_grace_period_before_lost():
    tracker = CatTracker(grace_frames=3)
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    # Cat disappears with no occluder
    tracker.mark_missing("cat1")
    cat = tracker.get_cat("cat1")
    assert cat.state == "visible"  # still in grace
    assert cat.missing_frames == 1
    tracker.mark_missing("cat1")
    tracker.mark_missing("cat1")
    cat = tracker.get_cat("cat1")
    assert cat.state == "lost"  # grace expired
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_cat_tracker.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement CatTracker**

```python
# server/spatial/cat_tracker.py
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackedCat:
    id: str
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_seen: float = 0.0
    state: str = "visible"  # "visible" | "occluded" | "lost"
    occluded_by: str | None = None
    missing_frames: int = 0
    _occlusion_start: float = 0.0


class CatTracker:
    """Tracks cats in room-space with velocity estimation and occlusion awareness."""

    def __init__(self, occlusion_timeout: float = 10.0, grace_frames: int = 3):
        self.occlusion_timeout = occlusion_timeout
        self.grace_frames = grace_frames
        self._cats: dict[str, TrackedCat] = {}

    def update_detection(
        self, cat_id: str, room_pos: tuple[float, float, float], timestamp: float
    ):
        """Update a cat's position from a detection."""
        if cat_id not in self._cats:
            self._cats[cat_id] = TrackedCat(id=cat_id)

        cat = self._cats[cat_id]
        cat.positions.append((room_pos[0], room_pos[1], room_pos[2], timestamp))
        cat.last_seen = timestamp
        cat.state = "visible"
        cat.occluded_by = None
        cat.missing_frames = 0

        # Update velocity from last 2+ positions
        if len(cat.positions) >= 2:
            p1 = cat.positions[-2]
            p2 = cat.positions[-1]
            dt = p2[3] - p1[3]
            if dt > 0:
                vx = (p2[0] - p1[0]) / dt
                vy = (p2[1] - p1[1]) / dt
                vz = (p2[2] - p1[2]) / dt
                # Smooth with previous velocity (simple EMA)
                a = 0.4
                cat.velocity = (
                    a * vx + (1 - a) * cat.velocity[0],
                    a * vy + (1 - a) * cat.velocity[1],
                    a * vz + (1 - a) * cat.velocity[2],
                )

    def mark_occluded(self, cat_id: str, occluder_name: str):
        """Mark a cat as occluded by a furniture object."""
        cat = self._cats.get(cat_id)
        if not cat:
            return
        cat.state = "occluded"
        cat.occluded_by = occluder_name
        cat._occlusion_start = cat.last_seen
        cat.missing_frames = 0

    def mark_missing(self, cat_id: str):
        """Mark a frame where a visible cat wasn't detected (no occluder found)."""
        cat = self._cats.get(cat_id)
        if not cat or cat.state != "visible":
            return
        cat.missing_frames += 1
        if cat.missing_frames >= self.grace_frames:
            cat.state = "lost"

    def predict_position(
        self, cat_id: str, timestamp: float
    ) -> tuple[float, float, float] | None:
        """Predict where an occluded cat is based on last velocity."""
        cat = self._cats.get(cat_id)
        if not cat or not cat.positions:
            return None
        last = cat.positions[-1]
        dt = timestamp - last[3]
        # Decay velocity over time (cat slows down)
        decay = max(0.0, 1.0 - dt * 0.3)
        return (
            last[0] + cat.velocity[0] * dt * decay,
            last[1] + cat.velocity[1] * dt * decay,
            last[2],  # z stays roughly the same
        )

    def tick(self, current_time: float):
        """Check for occlusion timeouts."""
        for cat in self._cats.values():
            if cat.state == "occluded":
                elapsed = current_time - cat._occlusion_start
                if elapsed >= self.occlusion_timeout:
                    cat.state = "lost"
                    cat.occluded_by = None

    def get_cat(self, cat_id: str) -> TrackedCat | None:
        return self._cats.get(cat_id)

    def get_active_cats(self) -> list[TrackedCat]:
        """Return cats that are visible or occluded (not lost)."""
        return [c for c in self._cats.values() if c.state in ("visible", "occluded")]

    def get_occluded_cats(self) -> list[TrackedCat]:
        return [c for c in self._cats.values() if c.state == "occluded"]

    def cleanup_lost(self, max_age: float = 60.0, current_time: float = 0.0):
        """Remove cats that have been lost for too long."""
        to_remove = []
        for cat_id, cat in self._cats.items():
            if cat.state == "lost" and current_time - cat.last_seen > max_age:
                to_remove.append(cat_id)
        for cat_id in to_remove:
            del self._cats[cat_id]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/aripi/OneDrive/Desktop/Documents/catzap && python -m pytest server/tests/test_cat_tracker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add server/spatial/cat_tracker.py server/tests/test_cat_tracker.py
git commit -m "feat: add cat tracker with velocity estimation and occlusion handling"
```

### Task 12: Integrate depth + tracker into vision loop

**Files:**
- Modify: `server/main.py`

- [ ] **Step 1: Add imports and initialization**

Add to the imports in `server/main.py`:
```python
from server.spatial.depth_estimator import DepthEstimator
from server.spatial.room_model import RoomModel
from server.spatial.projection import angle_depth_to_room
from server.spatial.cat_tracker import CatTracker
```

Add shared state variables after the existing ones:
```python
_depth_estimator: DepthEstimator | None = None
_room_model: RoomModel | None = None
_cat_tracker: CatTracker | None = None
```

Add getter functions:
```python
def get_room_model() -> RoomModel | None:
    return _room_model

def get_cat_tracker() -> CatTracker | None:
    return _cat_tracker
```

- [ ] **Step 2: Initialize in vision loop**

In `run_vision_loop`, after the existing initialization, add:
```python
    global _depth_estimator, _room_model, _cat_tracker

    _depth_estimator = DepthEstimator(model_type=settings.midas_model)
    _room_model = RoomModel(
        width_cm=500, depth_cm=500, height_cm=300,
        resolution=settings.heightmap_resolution,
    )
    _cat_tracker = CatTracker(
        occlusion_timeout=settings.occlusion_timeout,
        grace_frames=settings.occlusion_grace_frames,
    )
    depth_frame_counter = 0
```

- [ ] **Step 3: Add depth estimation to the frame processing loop**

After the existing detection code and before the violation checking, add depth processing:
```python
            # Run depth estimation periodically
            current_depth = None
            depth_frame_counter += 1
            if depth_frame_counter % settings.depth_run_interval == 0:
                try:
                    current_depth = _depth_estimator.estimate(frame)
                except Exception as e:
                    logger.warning(f"Depth estimation failed: {e}")
```

- [ ] **Step 4: Update detection loop to project cats to room-space**

In the detection loop, after converting bbox to angle-space, add room-space projection:
```python
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
```

Update the `check_zone_violations` call to pass `cat_room_pos`:
```python
                violations = check_zone_violations(angle_bbox, current_zones, cat_room_pos=cat_room_pos)
```

- [ ] **Step 5: Add occlusion checking after detection loop**

After the detection loop, add tracker tick and occlusion broadcast:
```python
            # Tick cat tracker
            now_t = time.time()
            _cat_tracker.tick(now_t)

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

Add `occluded_predictions` to the WebSocket broadcast payload:
```python
            await broadcast_to_clients({
                # ... existing fields ...
                "occluded_cats": occluded_predictions,
            })
```

- [ ] **Step 6: Commit**

```bash
git add server/main.py
git commit -m "feat: integrate depth estimation and cat tracker into vision loop"
```

### Task 13: Show predicted cat positions in LiveFeed

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/components/LiveFeed.tsx`

- [ ] **Step 1: Update FrameData type**

Add to `FrameData` in `frontend/src/types/index.ts`:
```typescript
export interface OccludedCat {
  id: string;
  predicted: [number, number, number];
  occluded_by: string;
}

export interface FrameData {
  // ... existing fields ...
  occluded_cats: OccludedCat[];
}
```

- [ ] **Step 2: Render predicted positions in LiveFeed canvas**

In `LiveFeed.tsx`, update the `drawFrame` function. Add after the fire crosshair drawing:
```typescript
    // Draw predicted positions for occluded cats
    if (data.occluded_cats) {
      for (const ocat of data.occluded_cats) {
        // The predicted position would need to be projected back to pixel-space
        // For now, use a simple indicator based on angle projection
        // This will be refined when projection utils are added to the frontend
      }
    }
```

For the initial version, add occluded cat info to the bottom overlay instead:

In the JSX, after the bottom servo/cat count overlay, add:
```tsx
      {/* Occluded cat predictions */}
      {/* Rendered as amber dashed indicators — will be positioned properly once frontend projection is added */}
```

This task is a placeholder for Phase 4 frontend polish — the exact rendering depends on having the frontend projection math, which follows the same pattern as the existing `angleToPanoPixel` in PanoramaView. The key data flow (server → WebSocket → LiveFeed) is wired up in this step.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/components/LiveFeed.tsx
git commit -m "feat: wire up occluded cat data to LiveFeed component"
```

---

### Task 14: Furniture persistence and spatial API

**Files:**
- Modify: `server/models/database.py`
- Create: `server/routers/spatial.py`
- Modify: `server/main.py`

- [ ] **Step 1: Add furniture table to database schema**

Add to the `SCHEMA` string in `server/models/database.py`:
```sql
CREATE TABLE IF NOT EXISTS furniture (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    base_polygon JSONB NOT NULL,
    height_min REAL NOT NULL DEFAULT 0.0,
    height_max REAL NOT NULL DEFAULT 0.0,
    depth_anchored BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 2: Add furniture CRUD functions to database.py**

```python
async def create_furniture(
    name: str,
    base_polygon: list[list[float]],
    height_min: float = 0.0,
    height_max: float = 0.0,
    depth_anchored: bool = False,
    conn: asyncpg.Connection | None = None,
) -> str:
    furniture_id = uuid.uuid4()
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        await c.execute(
            """INSERT INTO furniture (id, name, base_polygon, height_min, height_max, depth_anchored)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            furniture_id, name, json.dumps(base_polygon), height_min, height_max, depth_anchored,
        )
        return str(furniture_id)
    finally:
        if conn is None:
            await pool.release(c)


async def get_furniture(conn: asyncpg.Connection | None = None) -> list[dict]:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        rows = await c.fetch("SELECT * FROM furniture ORDER BY created_at DESC")
        return [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "base_polygon": json.loads(row["base_polygon"]) if isinstance(row["base_polygon"], str) else row["base_polygon"],
                "height_min": row["height_min"],
                "height_max": row["height_max"],
                "depth_anchored": row["depth_anchored"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    finally:
        if conn is None:
            await pool.release(c)


async def delete_furniture(furniture_id: str, conn: asyncpg.Connection | None = None) -> bool:
    pool = get_pool()
    c = conn or await pool.acquire()
    try:
        result = await c.execute("DELETE FROM furniture WHERE id = $1", uuid.UUID(furniture_id))
        return result != "DELETE 0"
    finally:
        if conn is None:
            await pool.release(c)
```

- [ ] **Step 3: Create spatial API router**

```python
# server/routers/spatial.py
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from server.models.database import create_furniture, get_furniture, delete_furniture

router = APIRouter(prefix="/api/spatial", tags=["spatial"])


class FurnitureCreate(BaseModel):
    name: str
    base_polygon: list[list[float]]
    height_min: float = 0.0
    height_max: float = 0.0
    depth_anchored: bool = False


@router.post("/furniture", status_code=201)
async def create_furniture_endpoint(body: FurnitureCreate):
    fid = await create_furniture(
        name=body.name,
        base_polygon=body.base_polygon,
        height_min=body.height_min,
        height_max=body.height_max,
        depth_anchored=body.depth_anchored,
    )
    items = await get_furniture()
    return next(f for f in items if f["id"] == fid)


@router.get("/furniture")
async def get_furniture_endpoint():
    return await get_furniture()


@router.delete("/furniture/{furniture_id}")
async def delete_furniture_endpoint(furniture_id: str):
    success = await delete_furniture(furniture_id)
    if not success:
        raise HTTPException(status_code=404, detail="Furniture not found")
    return {"deleted": True}


@router.get("/room-model/status")
async def room_model_status():
    from server.main import get_room_model
    rm = get_room_model()
    if not rm:
        return {"initialized": False}
    return {
        "initialized": True,
        "width_cm": rm.width_cm,
        "depth_cm": rm.depth_cm,
        "furniture_count": len(rm.furniture),
        "depth_scale": rm.depth_scale,
    }


@router.post("/calibrate-scale")
async def calibrate_depth_scale(body: dict):
    """Set depth scale from a reference measurement.

    Body: { "pixel_x": int, "pixel_y": int, "real_distance_cm": float }
    """
    from server.main import get_room_model
    rm = get_room_model()
    if not rm:
        raise HTTPException(status_code=400, detail="Room model not initialized")
    # The actual calibration needs a depth frame — this stores the reference for
    # the next depth estimation cycle to pick up
    rm.depth_scale = body.get("real_distance_cm", 100.0)
    return {"depth_scale": rm.depth_scale}
```

- [ ] **Step 4: Register spatial router in main.py**

In `server/main.py`, add the import:
```python
from server.routers import spatial
```

And register it:
```python
app.include_router(spatial.router)
```

- [ ] **Step 5: Load furniture into room model on startup**

In `run_vision_loop`, after initializing `_room_model`, load persisted furniture:
```python
    # Load persisted furniture into room model
    from server.models.database import get_furniture as db_get_furniture
    from server.spatial.room_model import FurnitureObject
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

- [ ] **Step 6: Commit**

```bash
git add server/models/database.py server/routers/spatial.py server/main.py
git commit -m "feat: add furniture persistence and spatial API router"
```

### Task 15: Calibration wizard frontend

**Files:**
- Create: `frontend/src/components/CalibrationWizard.tsx`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/components/Settings.tsx`

- [ ] **Step 1: Add API client functions for spatial**

Add to `frontend/src/api/client.ts`:
```typescript
// Spatial / Calibration
export const getFurniture = () => fetchJSON<any[]>("/spatial/furniture");

export const createFurniture = (furniture: {
  name: string;
  base_polygon: number[][];
  height_min: number;
  height_max: number;
}) => fetchJSON("/spatial/furniture", { method: "POST", body: JSON.stringify(furniture) });

export const deleteFurniture = (id: string) =>
  fetchJSON(`/spatial/furniture/${id}`, { method: "DELETE" });

export const getRoomModelStatus = () =>
  fetchJSON<{ initialized: boolean; width_cm?: number; depth_cm?: number; furniture_count?: number; depth_scale?: number }>("/spatial/room-model/status");

export const calibrateDepthScale = (realDistanceCm: number) =>
  fetchJSON("/spatial/calibrate-scale", {
    method: "POST",
    body: JSON.stringify({ real_distance_cm: realDistanceCm }),
  });
```

- [ ] **Step 2: Create CalibrationWizard component**

```typescript
// frontend/src/components/CalibrationWizard.tsx
import { useEffect, useState } from "react";
import {
  getRoomModelStatus,
  calibrateDepthScale,
  getFurniture,
  createFurniture,
  deleteFurniture,
} from "../api/client";

type Step = "status" | "scale" | "furniture";

export default function CalibrationWizard() {
  const [step, setStep] = useState<Step>("status");
  const [roomStatus, setRoomStatus] = useState<any>(null);
  const [furniture, setFurniture] = useState<any[]>([]);
  const [refDistance, setRefDistance] = useState("100");
  const [newFurnitureName, setNewFurnitureName] = useState("");

  useEffect(() => {
    getRoomModelStatus().then(setRoomStatus).catch(console.error);
    getFurniture().then(setFurniture).catch(console.error);
  }, []);

  async function handleCalibrate() {
    const dist = parseFloat(refDistance);
    if (isNaN(dist) || dist <= 0) return;
    await calibrateDepthScale(dist);
    const status = await getRoomModelStatus();
    setRoomStatus(status);
  }

  async function handleDeleteFurniture(id: string) {
    await deleteFurniture(id);
    setFurniture(await getFurniture());
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Room model status */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Room Model</div>
        {roomStatus ? (
          <div style={{
            fontFamily: "var(--font-mono)", fontSize: 11,
            color: "var(--text-tertiary)", lineHeight: 1.6,
          }}>
            <div>Status: <span style={{ color: roomStatus.initialized ? "var(--green)" : "var(--text-ghost)" }}>
              {roomStatus.initialized ? "Active" : "Not initialized"}
            </span></div>
            {roomStatus.initialized && (
              <>
                <div>Room: {roomStatus.width_cm} x {roomStatus.depth_cm} cm</div>
                <div>Furniture objects: {roomStatus.furniture_count}</div>
                <div>Depth scale: {roomStatus.depth_scale?.toFixed(1)}</div>
              </>
            )}
          </div>
        ) : (
          <div style={{ color: "var(--text-ghost)", fontFamily: "var(--font-mono)", fontSize: 11 }}>
            Loading...
          </div>
        )}
      </div>

      {/* Depth scale calibration */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Depth Calibration</div>
        <p style={{
          fontFamily: "var(--font-mono)", fontSize: 11,
          color: "var(--text-tertiary)", lineHeight: 1.5, marginBottom: 8,
        }}>
          Enter a known distance in the room (e.g., floor to table surface) to calibrate depth to real-world units.
        </p>
        <div style={{ display: "flex", gap: 6 }}>
          <input
            type="text"
            value={refDistance}
            onChange={(e) => setRefDistance(e.target.value)}
            placeholder="Distance in cm..."
            style={{ flex: 1, fontSize: 11 }}
          />
          <button className="btn btn-primary btn-sm" onClick={handleCalibrate}>
            Calibrate
          </button>
        </div>
      </div>

      {/* Furniture list */}
      <div className="glass-panel-solid" style={{ padding: 14 }}>
        <div className="label" style={{ marginBottom: 8 }}>Furniture Objects</div>
        {furniture.length === 0 ? (
          <div style={{
            textAlign: "center", padding: 16,
            color: "var(--text-ghost)", fontFamily: "var(--font-mono)", fontSize: 11,
          }}>
            No furniture defined. Use the zone editor to draw furniture outlines.
          </div>
        ) : (
          furniture.map((f) => (
            <div
              key={f.id}
              style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "6px 10px", background: "var(--bg-deep)",
                borderRadius: "var(--radius-sm)", marginBottom: 3,
                fontFamily: "var(--font-mono)", fontSize: 11,
              }}
            >
              <div>
                <span style={{ color: "var(--amber)" }}>{f.name}</span>
                <span style={{ color: "var(--text-ghost)", marginLeft: 8 }}>
                  {f.height_min}-{f.height_max}cm
                </span>
              </div>
              <button
                className="btn btn-danger btn-sm"
                onClick={() => handleDeleteFurniture(f.id)}
                style={{ padding: "2px 8px", fontSize: 10 }}
              >
                Remove
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Add CalibrationWizard to Settings**

In `frontend/src/components/Settings.tsx`, replace the `Calibration` import and usage:
```typescript
import CalibrationWizard from "./CalibrationWizard";
```

Replace `<Calibration />` at the bottom of the Settings return with:
```tsx
      <CalibrationWizard />
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/CalibrationWizard.tsx frontend/src/api/client.ts frontend/src/components/Settings.tsx
git commit -m "feat: add calibration wizard and furniture management UI"
```

---

## Phase Summary

| Phase | Tasks | What it delivers |
|---|---|---|
| **1: Sweep Control** | Tasks 1-3 | Pause/resume + emergency stop, backend + frontend |
| **2: Depth + Room Model** | Tasks 4-7 | MiDaS integration, heightmap, furniture volumes, projections |
| **3: Volumetric Zones** | Tasks 8-10, 14-15 | 3D zone data model, prism violation checks, zone editor UI, furniture persistence + calibration wizard |
| **4: Occlusion Tracking** | Tasks 11-13 | Cat tracker, trajectory prediction, vision loop integration |

| Phase | Tasks | What it delivers |
|---|---|---|
| **1: Sweep Control** | Tasks 1-3 | Pause/resume + emergency stop, backend + frontend |
| **2: Depth + Room Model** | Tasks 4-7 | MiDaS integration, heightmap, furniture volumes, projections |
| **3: Volumetric Zones** | Tasks 8-10 | 3D zone data model, prism violation checks, zone editor UI |
| **4: Occlusion Tracking** | Tasks 11-13 | Cat tracker, trajectory prediction, vision loop integration |

Each phase produces a working commit that can be tested independently.
