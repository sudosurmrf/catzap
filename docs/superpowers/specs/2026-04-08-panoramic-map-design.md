# CatZap Panoramic Map — Design Specification

Redesign from fixed-camera + separate gun to camera-on-gun with a servo-registered panoramic tile map for persistent zone tracking.

## Motivation

With the camera and gun on the same pan/tilt mount, zones drawn in pixel-space shift as the camera moves. The panoramic map solves this by defining everything in angle-space (servo pan/tilt coordinates), which stays fixed relative to the room regardless of where the camera is pointing.

## Physical Setup Change

**Before:** ESP32-CAM fixed on wall, ESP32 DEVKITV1 + gun on separate pan/tilt mount.

**After:** ESP32-CAM mounted on the pan/tilt bracket alongside the water gun. Single unit — camera and gun always point in the same direction. The center of the camera frame is where the gun fires.

Hardware remains the same (ESP32-CAM, ESP32 DEVKITV1, 2x SG90 servos, solenoid). Only the physical mounting changes.

## Tile Grid Panorama

### Concept

The servo's sweep range is divided into a grid of tiles. Each tile stores one JPEG frame captured at a known (pan, tilt) position. Together, the tiles form a panoramic map of the room in angle-space.

### Parameters

- **ESP32-CAM FOV:** ~65° horizontal × ~50° vertical
- **Tile overlap:** ~10° (prevents gaps and helps with edge detection)
- **Tile step:** ~55° horizontal, ~40° vertical
- **Coverage:** ~120° pan × ~50° tilt (configurable per installation)
- **Grid size:** ~3 columns × 2 rows = ~6 tiles (varies with coverage range)
- **Storage:** Each tile is a JPEG in memory (~30-50KB each). Total: ~200-300KB.

### Configuration

```
sweep_pan_min: float = 30.0     # Minimum pan angle (degrees)
sweep_pan_max: float = 150.0    # Maximum pan angle (degrees)
sweep_tilt_min: float = 20.0    # Minimum tilt angle (degrees)
sweep_tilt_max: float = 70.0    # Maximum tilt angle (degrees)
fov_horizontal: float = 65.0    # Camera horizontal FOV (degrees)
fov_vertical: float = 50.0      # Camera vertical FOV (degrees)
tile_overlap: float = 10.0      # Overlap between adjacent tiles (degrees)
```

These are stored in the settings table and configurable from the web app.

### Pixel-to-Angle Conversion

Given the current servo position and a detection's normalized pixel coordinates:

```
cat_pan  = servo_pan  + (pixel_x - 0.5) * fov_horizontal
cat_tilt = servo_tilt + (pixel_y - 0.5) * fov_vertical
```

This maps any detection to a fixed angle-space position regardless of where the camera is pointing.

### Smart Tile Refresh

During each sweep pass:
1. At each tile position, capture a frame
2. Compute mean absolute pixel difference against stored tile
3. If difference > threshold (configurable, default 15/255) → update tile
4. If difference < threshold → skip (scene unchanged)

This keeps the panorama current without unnecessary processing.

## Zone System (Angle-Space)

### Changes from Current System

**Before:** Zones stored as polygons in normalized pixel coordinates (0-1). Checked against cat bounding boxes in the same pixel-space.

**After:** Zones stored as polygons in angle-space coordinates (degrees). Cat detections are converted from pixel-space to angle-space before zone checking.

### Zone Data Model

```
id: UUID
name: string
polygon: list of [pan_angle, tilt_angle] points (degrees)
overlap_threshold: float (default 0.3)
cooldown_seconds: int (default 3)
enabled: boolean
created_at: datetime
```

The polygon points are in degrees (e.g., `[[70, 25], [110, 25], [110, 45], [70, 45]]` for a rectangular zone spanning pan 70-110° and tilt 25-45°).

### Zone Checking

Same Shapely intersection logic as before, but in angle-space:
1. Convert cat bbox corners from pixels to angles using the servo position
2. Create a Shapely box from the angle-space bbox
3. Check intersection with each zone polygon (also in angle-space)
4. If overlap exceeds threshold, it's a violation

### Zone Rendering

**On panorama:** Zones render directly — the panorama is in angle-space, zones are in angle-space. Straight mapping.

**On live feed:** Zones are projected from angle-space back to pixel-space using the current servo position:
```
pixel_x = 0.5 + (zone_pan - servo_pan) / fov_horizontal
pixel_y = 0.5 + (zone_tilt - servo_tilt) / fov_vertical
```
Only zone portions within the current FOV are drawn.

## Sweep Controller

### States

The system operates as a state machine with four states:

**SWEEPING** (default)
- Camera pans continuously left→right→left across the configured range
- Speed: ~2-3°/second (configurable). Full 120° sweep in ~45-60 seconds.
- Tilt follows a configurable pattern: single row or two-row zigzag
- Every frame: run YOLOv8 detection, check against zones
- Tiles update via smart refresh as camera passes over them
- Transition: cat detected in a forbidden zone → WARNING

**WARNING** (1-2 seconds)
- Sweep stops. Camera stays aimed at the cat.
- Continue detection each frame — track cat's position, adjust servos to keep cat centered
- Frontend shows yellow warning indicator with countdown
- Transition: cat leaves zone → SWEEPING
- Transition: warning timer expires, cat still in zone → FIRING

**FIRING**
- Center the cat in the frame (adjust servos so cat bbox center is at frame center)
- Trigger solenoid — fire
- Log ZAP event, show ZAP indicator on frontend, send notification
- Enter cooldown (3 seconds, configurable)
- Transition: cat still in zone after cooldown → fire again
- Transition: cat leaves → TRACKING

**TRACKING** (post-fire observation)
- Keep watching the cat for 3-5 seconds after it leaves the zone
- If cat re-enters zone → WARNING with shorter timer (~0.5s)
- If cat leaves frame entirely for 3 seconds → SWEEPING
- Prevents immediate re-entry without consequence

### Sweep Pattern

Linear pan sweep at configurable tilt:
```
Pan: min → max → min → max (repeat)
Tilt: fixed at configured angle, or zigzag between two rows
Speed: configurable (default ~2-3°/second)
```

### Manual Override

From the web app, the user can:
- Pause sweep and manually control pan/tilt with sliders
- Trigger a calibration sweep (slow, updates all tiles)
- Click on a panorama tile to re-scan that area
- Adjust sweep speed and angle range
- Manual fire by clicking a point on the panorama

## ESP32 DEVKITV1 Firmware Changes

The actuator firmware needs to support sweep commands and report current angles.

### New Endpoints

**POST /sweep** — Start continuous sweep
```json
{
  "pan_min": 30.0,
  "pan_max": 150.0,
  "tilt": 45.0,
  "speed": 2.5
}
```

**POST /goto** — Move to specific angle
```json
{
  "pan": 90.0,
  "tilt": 45.0
}
```

**GET /position** — Report current servo angles
```json
{
  "pan": 87.3,
  "tilt": 45.0
}
```

**POST /stop** — Stop sweep, hold current position

**POST /fire** — unchanged from current

### Angle Reporting

The ESP32 must report its current servo angles with each frame or on-demand. Two options:

**Option A (chosen): Server-commanded positions.** The server sends explicit `goto` commands for each tile position during sweeps. The server always knows where the camera is because it commanded it there. No need for the ESP32 to report back — the server tracks the angles.

This is simpler and avoids sync issues. The server commands a position, waits for the servo to settle (~200ms), then reads the frame.

## Server Changes

### New Modules

**`server/panorama/tile_grid.py`** — Tile grid manager
- Stores tiles as JPEG bytes keyed by (col, row)
- Computes tile positions from sweep config
- Smart refresh comparison
- Serializes panorama to a single image for the frontend

**`server/panorama/sweep_controller.py`** — State machine
- Manages SWEEPING/WARNING/FIRING/TRACKING states
- Commands servo positions via actuator client
- Coordinates with vision pipeline for detection

**`server/panorama/angle_math.py`** — Coordinate conversions
- `pixel_to_angle(pixel_x, pixel_y, servo_pan, servo_tilt)` → `(pan_angle, tilt_angle)`
- `angle_to_pixel(pan_angle, tilt_angle, servo_pan, servo_tilt)` → `(pixel_x, pixel_y)`
- `is_angle_in_fov(pan_angle, tilt_angle, servo_pan, servo_tilt)` → `bool`

### Modified Modules

**`server/main.py`** — Vision loop replaced by sweep controller loop
**`server/vision/zone_checker.py`** — Works in angle-space instead of pixel-space (same Shapely logic, different coordinate inputs)
**`server/routers/control.py`** — New sweep control endpoints, panorama serving
**`server/actuator/client.py`** — New methods: `goto()`, `stop()`, `sweep()`
**`server/actuator/calibration.py`** — Removed (no longer needed)

### Removed Modules

**`server/actuator/calibration.py`** — Pixel-to-angle calibration is replaced by direct math using known FOV and servo positions.

## Frontend Changes

### New Components

**`PanoramaView.tsx`** — Renders the tile grid as a composite image
- Overlays zone polygons (angle-space maps directly to panorama pixels)
- Shows cat position markers (mapped from latest detection)
- Highlights current camera viewport (blue rectangle)
- Click to aim camera at a point

**`SweepControls.tsx`** — Sweep configuration
- Speed slider
- Pan/tilt range sliders
- Pause/resume sweep button
- Calibration sweep button

**`StateIndicator.tsx`** — Shows current system state
- SWEEPING (blue), WARNING (yellow), FIRING (red), TRACKING (purple)
- Warning countdown timer
- Cooldown timer

### Modified Components

**`App.tsx`** — Live tab layout changes to split view (panorama top, live feed bottom)

**`LiveFeed.tsx`** — Smaller, shows in bottom half of split view
- Displays current servo angles and system state in status bar
- Zones projected from angle-space to current pixel-space
- Warning/ZAP indicators remain

**`ZoneEditor.tsx`** — Draws on panorama instead of live feed
- Freehand drawing on the panorama view
- Coordinates in angle-space (mapped from panorama pixel position)
- Zone list management unchanged

**`Controls.tsx`** — Adds sweep controls panel
- Existing arm/disarm and manual fire remain
- New: sweep speed, range, pause/resume

**`Calibration.tsx`** — Simplified to a single "Calibration Sweep" button that triggers a full panorama rebuild

### Removed Components

None removed, but `Calibration.tsx` is significantly simplified.

## Dev Mode Changes

In dev mode (webcam, no servos), the system supports full panorama testing with manual angle control:

### Manual Angle Control

Since there are no servos in dev mode, the user manually sets the virtual camera angle as they physically rotate the webcam by hand. This allows building a real panorama and testing all sweep/zone/fire logic without hardware.

**Controls:**
- **Left/Right arrow keys** — nudge virtual pan angle by 5° per press (hold for continuous)
- **Up/Down arrow keys** — nudge virtual tilt angle by 5° per press
- **Pan/tilt sliders** in the SweepControls panel — drag to set exact angle
- **Current angle displayed** in the live feed status bar so user can keep webcam rotation roughly in sync

**Workflow for building panorama in dev mode:**
1. Point webcam to the leftmost position of your room
2. Set virtual pan to `sweep_pan_min` (e.g., 30°) using the slider
3. Press "Capture Tile" button (or it auto-captures when angle changes)
4. Rotate webcam slightly right, press right arrow a few times to advance the virtual angle
5. Repeat until you've covered the full range
6. Panorama builds up tile by tile as you go

**Simulated sweep:** A "Simulate Sweep" button auto-advances the virtual angle at the configured sweep speed. The user rotates the webcam to keep pace. This tests the full state machine (SWEEPING → WARNING → FIRING → TRACKING) without servos.

**Auto-capture tiles:** When the virtual angle crosses into a new tile's position, the current webcam frame is automatically captured as that tile. No need to manually press capture for each one.

### Direction Arrow Indicator (Dev Mode)

When the system detects a cat in a zone and would normally command the servos to move, a flashing yellow arrow appears on the live feed showing the user which direction to rotate the webcam.

**Behavior:**
- Arrow appears on the left or right edge of the live feed (or top/bottom for tilt)
- Points in the direction the camera needs to move to center the cat
- Flashes on/off at ~2Hz (visible but not distracting)
- Size and opacity scale with how far off-center the cat is (bigger arrow = rotate more)
- Disappears when the cat is roughly centered in frame (within ~10° of frame center)
- Also shows on the panorama view as an arrow on the viewport indicator

**When it triggers:**
- During WARNING state — "rotate this way to aim at the cat"
- During FIRING state — "center the cat for the shot"
- During TRACKING state — "cat moved, follow it this way"
- Does NOT show during SWEEPING (sweep direction is already implied by the sweep pattern)

**Implementation:** The server already computes the angle delta between current servo position and the cat's angle-space position. In dev mode, instead of sending a goto command, it sends the delta to the frontend which renders the arrow.

### What stays the same in dev mode
- Full detection pipeline (YOLOv8 runs on webcam frames)
- Zone checking in angle-space (using the virtual angle)
- State machine transitions (WARNING, FIRING, etc.)
- Simulated fire (ZAP indicator, no actual solenoid)
- Event logging, stats, notifications
- Zone drawing on panorama

When hardware is connected, set `CATZAP_DEV_MODE=false` and the real servo sweep system activates.

## Data Flow Summary

```
1. Server sends GOTO command to ESP32 → servos move to position
2. Server waits ~200ms for servo settle
3. Server reads MJPEG frame from ESP32-CAM
4. Frame placed into tile grid at known position (smart refresh)
5. YOLOv8 runs on frame → cat detections (pixel-space)
6. Detections converted to angle-space using current servo position
7. Angle-space detections checked against angle-space zones (Shapely)
8. If violation: state machine transitions (SWEEPING → WARNING → FIRING)
9. Frontend receives: live frame + panorama updates + detections + state
10. Panorama shows full room, live feed shows current view, both show zones
```

## Configuration Additions

New settings (stored in settings table, configurable from web app):

```
sweep_pan_min: 30.0
sweep_pan_max: 150.0
sweep_tilt_min: 20.0
sweep_tilt_max: 70.0
sweep_speed: 2.5          # degrees per second
sweep_tilt_rows: 1         # 1 or 2 row sweep pattern
fov_horizontal: 65.0
fov_vertical: 50.0
tile_overlap: 10.0
tile_refresh_threshold: 15 # pixel difference threshold (0-255)
warning_duration: 1.5      # seconds before firing
tracking_duration: 3.0     # seconds to watch after cat leaves
reentry_warning: 0.5       # shorter warning for re-entry during tracking
```
