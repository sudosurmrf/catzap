# 3D Spatial Awareness, Sweep Control & Volumetric Zones

**Date:** 2026-04-08
**Status:** Approved design

## Problem

Three related gaps in the current CatZap system:

1. **No sweep pause or emergency stop.** The camera sweeps continuously with no way to pause or halt in an emergency.
2. **Zones are 2D.** Exclusion zones are flat polygons in angle-space. A cat sitting ON a table whose side was outlined may register as outside the zone. The system has no concept of furniture height or volume.
3. **No occlusion awareness.** When a cat runs behind furniture, the system loses the detection and returns to SWEEPING — it doesn't understand the cat is hidden, not gone.

## Solution Overview

Four interconnected features:

| Feature | Purpose |
|---|---|
| Sweep pause + E-stop | Manual control over sweep motion and emergency shutdown |
| Depth estimation + room model | MiDaS-based depth fused into a metric room model |
| Volumetric exclusion zones | 3D prism zones from freehand perimeters + height extrusion |
| Occlusion-aware tracking | Trajectory prediction through furniture volumes |

## 1. Sweep Pause & Emergency Stop

### State Machine Changes

Add two new states to `SweepState`:

```
SWEEPING ←→ PAUSED
     ↓
  WARNING → FIRING → TRACKING → SWEEPING

EMERGENCY_STOP — reachable from ANY state
```

**PAUSED:**
- Freezes servo position. Camera feed continues.
- Can only be entered from SWEEPING. If triggered during WARNING/FIRING/TRACKING, the pause queues and activates once the system returns to SWEEPING.
- System remains armed — just not moving.
- Toggle via `POST /api/control/pause`.
- Resume continues sweep from the frozen position.

**EMERGENCY_STOP (STOPPED):**
- Triggers from ANY state, including mid-fire.
- Immediately: stops servo movement, closes solenoid, sets armed = false.
- Camera feed and detection continue running (actuator-only stop).
- System stays in STOPPED until explicit reset via `POST /api/control/clear-estop`.
- Reset returns to SWEEPING in disarmed state — user must re-arm separately.

### API Endpoints

```
POST /api/control/pause          → { paused: bool }
POST /api/control/emergency-stop → { stopped: true }
POST /api/control/clear-estop    → { stopped: false, armed: false }
```

### UI

- **Pause button:** In the left sidebar or bottom bar. Toggle icon (pause/play). Tooltip explains behavior.
- **E-Stop button:** Always visible, visually prominent (large, red). Distinct from all other controls. Located where it's instantly reachable — top of the right panel or fixed position in the sidebar.
- **Clear E-Stop:** Only appears when in STOPPED state. Requires deliberate click (not a toggle).

## 2. Depth Estimation & Room Model

### Depth Pipeline

- **Model:** MiDaS v3.1 small (DPT-Small). ~20ms/frame on GPU, ~80ms on CPU.
- **Output:** Relative inverse depth map per frame. Higher values = closer.
- **Metric calibration:** MiDaS outputs are relative. A single user-provided measurement (e.g., "floor to table surface = 75cm") anchors the scale. Stored as `depth_scale: float` in the room model.

### When Depth Runs

| Phase | Frequency | Purpose |
|---|---|---|
| Guided calibration | Every frame | Build initial room model |
| Normal sweep | Every Nth tile refresh | Detect furniture changes |
| On-demand | User-triggered | Rebuild after room changes |

### Room Model Data Structure

```python
class RoomModel:
    heightmap: ndarray          # 2D grid, ~5cm cells, top-down view
                                # each cell: (floor_height, max_height, confidence)
    furniture: list[FurnitureObject]
    depth_scale: float          # relative-to-metric conversion
    camera_position: (x, y, z) # fixed camera location in room-space
    last_calibration: datetime

class FurnitureObject:
    id: str
    name: str                          # "couch", "table", etc.
    base_polygon: list[tuple[float, float]]  # (x, y) in room-space (cm)
    height_min: float                  # bottom of extrusion (cm from floor)
    height_max: float                  # top of extrusion (cm from floor)
    depth_anchored: bool               # whether depth data was used
    angle_polygon: list[tuple[float, float]]  # projected to angle-space for display
```

### Coordinate Spaces

```
Angle-space (pan°, tilt°)         ← zones, panorama, sweep controller
        ↕  pixel_to_angle / angle_to_pixel (existing)
Pixel-space (0-1, 0-1)           ← detection bounding boxes
        ↕  depth + projection (new)
Room-space (x, y, z in cm)       ← furniture volumes, 3D zone checks
```

The camera is fixed, so the angle-space → room-space projection is stable for a given depth. A point at (pan, tilt) with depth d always maps to the same (x, y, z).

### Calibration Workflow

**Guided calibration (initial setup):**
1. User triggers calibration mode from UI.
2. Camera does a full sweep. MiDaS runs on every frame.
3. Depth maps are fused into the heightmap using known servo angles.
4. User is prompted to provide one reference measurement for metric scale.
5. User identifies furniture objects by drawing perimeters on the panorama and naming them.
6. System captures depth at those perimeters and suggests height ranges.
7. User confirms/adjusts. Furniture objects are saved.

**Continuous refinement (during normal operation):**
- MiDaS runs periodically on tile refreshes.
- New depth data blends with existing heightmap using exponential moving average: `cell = 0.8 * existing + 0.2 * new_reading`. Cells with fewer than 3 readings use equal weight instead.
- If a cell's depth changes by more than 20cm from its stored value across 3+ consecutive readings, the system flags it as a potential furniture change in the event log for user review.

## 3. Volumetric Exclusion Zones

### Zone Modes

Each zone has a mode that determines how violations are checked:

| Mode | Drawing | Height | Violation Check |
|---|---|---|---|
| `2d` | Freehand polygon on panorama | None | 2D overlap in angle-space (current behavior) |
| `auto_3d` | Freehand polygon on panorama | Auto-suggested from depth, adjustable | 3D point-in-prism in room-space |
| `manual_3d` | Freehand polygon on panorama | Manual sliders (cm) | 3D point-in-prism in room-space |

### Zone Data Model

```python
class Zone:
    # Existing fields
    id: str
    name: str
    polygon: list[list[float]]        # angle-space points (backward compat)
    overlap_threshold: float
    cooldown_seconds: float
    enabled: bool
    created_at: str

    # New fields
    mode: str = "2d"                   # "2d" | "auto_3d" | "manual_3d"
    room_polygon: list[list[float]]    # base footprint in room-space [(x,y), ...]
    height_min: float = 0.0            # bottom of volume (cm)
    height_max: float = 0.0            # top of volume (cm)
    furniture_id: str | None = None    # optional link to a FurnitureObject
```

Zones with `mode: "2d"` behave identically to current zones. No migration needed.

### Violation Check Changes

```python
def check_violation(cat_detection, zone, depth_frame, room_model):
    if zone.mode == "2d":
        # Current behavior — 2D overlap in angle-space
        return check_2d_overlap(cat_detection.angle_bbox, zone.polygon)

    # 3D modes: project cat into room-space
    cat_room_pos = project_to_room(
        cat_detection.center_pixel,
        depth_frame,
        room_model
    )

    if cat_room_pos is None:
        # Depth unavailable — fall back to 2D
        return check_2d_overlap(cat_detection.angle_bbox, zone.polygon)

    # Check if cat's room-space position is inside the zone prism
    return point_in_prism(
        cat_room_pos,
        zone.room_polygon,
        zone.height_min,
        zone.height_max
    )
```

### Zone Editor UI Changes

- After drawing a perimeter, a mode selector appears: "2D flat" / "Auto 3D" / "Manual 3D".
- **Auto 3D:** System shows suggested height range with a visualization of the extruded prism. User can adjust with sliders.
- **Manual 3D:** Two sliders for height_min and height_max (in cm). A side-view wireframe shows the extrusion.
- Optionally link the zone to a named furniture object from the room model.

## 4. Occlusion-Aware Cat Tracking

### Cat Tracker

```python
class CatTracker:
    id: str
    positions: deque[tuple[float, float, float, float]]  # (x, y, z, timestamp) in room-space
    velocity: tuple[float, float, float]                  # smoothed via Kalman filter
    last_seen: float                                      # timestamp
    occluded_by: FurnitureObject | None
    predicted_position: tuple[float, float, float]
    state: str  # "visible" | "occluded" | "lost"
```

### Occlusion Detection Flow

1. Cat detected at position P with velocity V.
2. Next frame: no detection near P.
3. Project trajectory: predicted position P' = P + V * dt.
4. For each furniture object: project its volume from room-space to angle-space. Check if the projected silhouette lies between the camera and P'.
5. **If occluded:** state → `occluded`, set `occluded_by`, continue predicting position along trajectory.
6. **If not occluded:** start grace period (3 frames). Detector may have missed.
7. **After grace period with no redetection:** state → `lost`, return to normal sweep.

### Sweep Behavior When Occluded

When a cat is occluded:
- Sweep controller enters a new priority mode: instead of continuing the normal left-right sweep, it moves toward angles that could see around/past the occluding furniture.
- The predicted position is updated each tick based on the last known velocity (decaying over time).
- If the cat reappears near the predicted position, the tracker reconnects it as the same encounter — no fresh warning countdown, the existing warning timer continues.
- If the cat doesn't reappear within `occlusion_timeout` seconds (default: 10, configurable in settings), the tracker gives up and the system returns to SWEEPING.

### Furniture Occlusion Check (Efficient)

No ray marching needed. For each furniture prism:
1. Project the prism's base polygon from room-space to angle-space (using the camera's known position).
2. Extend the projection vertically based on height_min/height_max.
3. Check if this projected 2D silhouette overlaps the line segment from camera angle to predicted cat angle.
4. This is a simple 2D geometry test on projected shapes.

## Dependencies & Integration

### New Python Packages
- `timm` or `torch` (for MiDaS) — only if not already present for YOLO
- `filterpy` — Kalman filter implementation

### New Server Modules
```
server/
  spatial/
    depth_estimator.py    # MiDaS wrapper
    room_model.py         # RoomModel, FurnitureObject, heightmap
    projection.py         # angle ↔ room-space transforms
    cat_tracker.py        # CatTracker, Kalman filter, occlusion logic
  routers/
    spatial.py            # API endpoints for calibration, furniture, room model
```

### Modified Modules
- `sweep_controller.py` — PAUSED, STOPPED states + occlusion priority scanning
- `zone_checker.py` — 3D violation checks with 2D fallback
- `main.py` — integrate depth pipeline into vision loop, cat tracker
- `config.py` — new settings (midas_model, depth_run_interval, occlusion_timeout, etc.)

### Frontend Changes
- Pause + E-Stop buttons (sidebar)
- Zone editor: mode selector, height sliders, prism visualization
- Panorama: render furniture volumes as semi-transparent overlays
- Live feed: show occluded cat predictions (dashed bbox at predicted position)
- Calibration panel: guided workflow for room setup + reference measurement

## Non-Goals (Explicit)

- No full SLAM or point cloud reconstruction
- No multi-camera support
- No real-time 3D rendering in the browser (wireframe overlays only)
- No automatic furniture detection/classification (user identifies furniture during calibration)
- Cat classifier training is still deferred (separate feature)
