# CatZap — Design Specification

A wall-mounted vision camera system that detects cats on forbidden furniture and deters them with a targeted water gun.

## System Overview

CatZap is a four-component system:

1. **ESP32-CAM** — streams live video over WiFi (MJPEG)
2. **Python Server (FastAPI)** — runs AI detection, manages zones, serves API, coordinates everything
3. **ESP32 DEVKITV1** — receives commands over WiFi, controls pan/tilt servos and solenoid trigger
4. **React Web App** — browser dashboard for monitoring, zone editing, stats, and manual control

The server runs on the user's Windows desktop during development, with the architecture designed for future migration to a self-contained edge device (Raspberry Pi, Jetson Nano).

## Architecture

```
ESP32-CAM ──MJPEG/WiFi──▶ Python Server (FastAPI) ──WebSocket/WiFi──▶ ESP32 DEVKITV1
                               │                                         │
                               │ REST + WebSocket                        ├─ Pan Servo (GPIO 18)
                               ▼                                         ├─ Tilt Servo (GPIO 19)
                          React Web App                                  └─ Solenoid via MOSFET (GPIO 23)
```

### Communication Protocols

- **ESP32-CAM → Server:** MJPEG stream over HTTP (`http://<cam-ip>:81/stream`)
- **Server → ESP32 DEVKITV1:** Commands over HTTP POST (aim angles + fire signal). ESP32 runs a lightweight HTTP server (Arduino WebServer library). WebSocket is unnecessary here — commands are infrequent, one-directional, and don't need persistent connections.
- **Server ↔ React App:** REST API for CRUD operations, WebSocket for real-time feed and events

### Tech Stack

| Component        | Technology                          |
|-----------------|-------------------------------------|
| Camera firmware  | Arduino/PlatformIO (C++)           |
| Actuator firmware| Arduino/PlatformIO (C++)           |
| Server           | Python 3.11+, FastAPI, Uvicorn     |
| Vision/ML        | YOLOv8-nano (ultralytics), OpenCV  |
| Cat classifier   | MobileNetV3 (PyTorch/ONNX)        |
| Zone geometry    | Shapely                            |
| Database         | SQLite (events, zones, settings)   |
| Frontend         | React, TypeScript, Vite            |
| Notifications    | Browser Push API + webhook (ntfy/Pushover) |

## Vision Pipeline

### Stage 1: Frame Capture
- Pull MJPEG frames from ESP32-CAM stream
- Decode JPEG to OpenCV numpy array
- Process every Nth frame (configurable) to balance CPU load vs responsiveness

### Stage 2: Cat Detection — YOLOv8-nano
- Pre-trained YOLOv8n model (COCO dataset, class 15 = "cat")
- No custom training needed for detection
- Returns bounding boxes + confidence scores
- ~15-30ms per frame on desktop CPU

### Stage 3: Cat Identification — Custom Classifier
- Crop detected cat region from frame
- Feed into MobileNetV3 classifier fine-tuned on user's specific cats
- Training done via web app: upload 20-50 photos per cat
- Returns cat name + confidence score
- Falls back to "Unknown Cat" if confidence below threshold

### Stage 4: Zone Intersection Check
- User-defined forbidden zones stored as polygons (drawn in web app)
- Check if cat bounding box overlaps with any zone using Shapely
- Configurable overlap threshold (default: 30% of cat bbox area inside zone)
- Returns zone name + overlap percentage

### Stage 5: Fire Decision Engine
- Cooldown timer prevents rapid-fire (default: 10 seconds, configurable per zone)
- Calculate pan/tilt servo angles from cat's pixel position in frame
- Angle calculation uses calibration data (mapped during setup via web app)
- Send aim command (pan angle, tilt angle) then fire command to ESP32 DEVKITV1
- Log event with timestamp, cat name, zone name, confidence scores

### Performance Targets
- **Pipeline latency:** ~200ms (frame capture to fire command)
- **Processing rate:** 5-10 FPS (detection runs async from stream capture)
- **Cooldown:** 10 seconds default, configurable per zone

## Web App Features

### Live Feed & Zone Editor (Main View)
- Real-time camera feed via WebSocket with detection overlays
- Cat bounding boxes with name label and confidence percentage
- Forbidden zones rendered as red dashed polygons with labels
- "Draw Zone" tool: click points on the feed to create polygon zones
- "Calibrate" tool: map pixel positions to servo angles (click 3-4 points, record angles)
- Status bar: FPS, cat count, active violations, system armed/disarmed state

### Event Log
- Chronological list of all events, color-coded by type:
  - **ZAP** (red) — cat was in a zone and the gun fired
  - **DETECT** (blue) — cat entered/exited a zone
  - **SYSTEM** (purple) — arm/disarm, config changes, errors
- Filterable by cat name, zone, event type, date range
- Stored in SQLite for persistence

### Cat Stats Dashboard
- Per-cat statistics: zap count, detection count, zap rate
- Time-of-day patterns (peak mischief hours)
- Favorite zone (most frequently violated)
- Trend over time (is the cat learning?)
- Daily/weekly/monthly views

### Controls
- **Manual Fire:** test the water gun from the app
  - Click on the feed to aim at a specific point
  - Pan/tilt sliders for fine adjustment
  - Fire button
- **Arm/Disarm:** toggle automatic firing on/off (detection continues)
- **Arm Schedule:** set times when the system is automatically armed/disarmed

### Settings
- Cooldown timer (global and per-zone)
- Detection confidence threshold
- Zone overlap threshold
- Cat training interface (upload photos, retrain classifier)
- ESP32 connection settings (IP addresses)
- Notification preferences

### Notifications
- Browser Push API for desktop notifications on zap events
- Optional webhook integration (ntfy or Pushover) for phone alerts
- Configurable: notify on every zap, or only after N zaps in a time window

## Hardware

### Components

**Already owned:**
- ESP32 DEVKITV1

**To purchase (~$50-55 total):**
- ESP32-CAM module (~$8)
- ESP32-CAM-MB USB programmer board (~$3)
- 2x SG90 micro servo motors (~$4)
- Pan/tilt servo bracket kit (~$5)
- 5V push-pull solenoid (~$5)
- IRLZ44N N-channel logic-level MOSFET (~$2)
- 1N4007 flyback diode (~$0.50)
- 10kΩ resistor (~$0.10)
- Electric water gun (~$10-15)
- 5V 3A power supply (~$8)
- Breadboard + jumper wires (~$5)

### ESP32 DEVKITV1 Wiring

| GPIO Pin | Connected To           | Purpose                  |
|----------|------------------------|--------------------------|
| GPIO 18  | Pan servo signal wire  | Horizontal aiming (PWM)  |
| GPIO 19  | Tilt servo signal wire | Vertical aiming (PWM)    |
| GPIO 23  | MOSFET gate (via 10kΩ) | Solenoid trigger control  |
| 5V       | Servo VCC, Solenoid VCC| Power for actuators       |
| GND      | Common ground          | Shared ground for all     |

### Solenoid Driver Circuit

```
5V ──────┬──────────┐
         │          │
     ┌───┴───┐   ┌──┴──┐
     │Solenoid│   │Diode│ (1N4007 flyback)
     └───┬───┘   └──┬──┘
         └────┬─────┘
              │
         ┌────┴────┐
         │ MOSFET  │ (IRLZ44N)
         │  Drain  │
         └────┬────┘
              │ Gate ←── 10kΩ pull-down ── GPIO 23
              │
            GND
```

- MOSFET switches the high-current solenoid path using the ESP32's 3.3V signal
- Flyback diode protects against inductive voltage spikes when solenoid turns off
- 10kΩ pull-down resistor keeps solenoid off during ESP32 boot/reset

### ESP32-CAM

- No external wiring — camera is integrated
- Program via ESP32-CAM-MB USB board
- Powered via USB or 5V pin
- Serves MJPEG stream at `http://<ip>:81/stream`
- Mount on wall aimed at the common area
- Avoid backlighting (don't point at windows)
- Ensure strong WiFi signal at mount point

### Physical Assembly

**Water gun mount:**
1. Mount pan/tilt bracket to wall or shelf near the target area
2. Attach water gun to tilt platform
3. Mount solenoid so plunger presses the water gun trigger when energized
4. Adjust spring tension for reliable trigger press
5. If solenoid force is insufficient, fall back to servo + linkage (Plan B)

**Camera mount:**
1. Mount ESP32-CAM on wall overlooking the common area
2. Position for maximum coverage of all furniture zones
3. Run USB cable for power

### Servo Calibration

The web app provides a calibration tool:
1. Click a point in the camera feed
2. Use manual pan/tilt sliders to aim the gun at that point
3. Save the mapping (pixel coordinates → servo angles)
4. Repeat for 3-4 points spread across the frame
5. Server interpolates between calibration points for any target position

## Data Model

### Zones
```
id: UUID
name: string
polygon: list of [x, y] points (normalized 0-1 relative to frame)
overlap_threshold: float (default 0.3)
cooldown_seconds: int (default 10)
enabled: boolean
created_at: datetime
```

### Cats
```
id: UUID
name: string
photos: list of image paths (for training)
model_version: int (incremented on retrain)
created_at: datetime
```

### Events
```
id: UUID
type: enum (ZAP, DETECT_ENTER, DETECT_EXIT, SYSTEM)
cat_id: UUID (nullable)
cat_name: string (nullable)
zone_id: UUID (nullable)
zone_name: string (nullable)
confidence: float (nullable)
overlap: float (nullable)
servo_pan: float (nullable)
servo_tilt: float (nullable)
timestamp: datetime
```

### Settings
```
key: string (primary key)
value: JSON
```

Key settings: `cooldown_default`, `confidence_threshold`, `overlap_threshold`, `arm_schedule`, `notification_webhook_url`, `frame_skip_n`, `esp32_cam_url`, `esp32_actuator_url`.

## Project Structure

```
catzap/
├── server/                    # Python FastAPI backend
│   ├── main.py                # App entry point, FastAPI app
│   ├── config.py              # Settings and configuration
│   ├── routers/
│   │   ├── stream.py          # Camera stream proxy + WebSocket feed
│   │   ├── zones.py           # Zone CRUD API
│   │   ├── cats.py            # Cat management + training API
│   │   ├── events.py          # Event log API
│   │   ├── control.py         # Manual fire, arm/disarm API
│   │   └── settings.py        # Settings API
│   ├── vision/
│   │   ├── detector.py        # YOLOv8 cat detection
│   │   ├── classifier.py      # Cat identification model
│   │   ├── zone_checker.py    # Zone intersection logic
│   │   └── pipeline.py        # Orchestrates the full pipeline
│   ├── actuator/
│   │   ├── client.py          # WebSocket/HTTP client to ESP32
│   │   └── calibration.py     # Pixel-to-angle mapping
│   ├── models/
│   │   └── database.py        # SQLite models and connection
│   ├── notifications/
│   │   └── notifier.py        # Push + webhook notifications
│   └── requirements.txt
├── frontend/                  # React TypeScript app
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── LiveFeed.tsx       # Camera feed + overlays
│   │   │   ├── ZoneEditor.tsx     # Draw/edit zones on feed
│   │   │   ├── EventLog.tsx       # Event history list
│   │   │   ├── CatStats.tsx       # Per-cat statistics
│   │   │   ├── Controls.tsx       # Manual fire, arm/disarm
│   │   │   ├── Settings.tsx       # Configuration panel
│   │   │   └── Calibration.tsx    # Servo calibration tool
│   │   ├── hooks/
│   │   ├── api/
│   │   └── types/
│   ├── package.json
│   └── vite.config.ts
├── firmware/
│   ├── esp32-cam/             # Camera streaming firmware
│   │   └── src/main.cpp
│   └── esp32-actuator/        # Servo + solenoid control firmware
│       └── src/main.cpp
└── docs/
```

## Future Edge Deployment

The architecture is designed for migration to a self-contained edge device:

1. **Target hardware:** Raspberry Pi 4/5 or Jetson Nano
2. **What moves:** The Python server + React build (static files served by FastAPI)
3. **What stays the same:** ESP32-CAM and ESP32 DEVKITV1 firmware, all communication protocols
4. **Model optimization:** YOLOv8-nano already runs well on Pi; MobileNetV3 classifier is lightweight
5. **Single deployment:** Package as a systemd service or Docker container
