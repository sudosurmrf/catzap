# CatZap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a wall-mounted vision camera system that detects cats on forbidden furniture and deters them with a servo-aimed water gun.

**Architecture:** ESP32-CAM streams MJPEG → Python/FastAPI server runs YOLOv8-nano detection + zone checking → sends HTTP commands to ESP32 DEVKITV1 → servos aim, solenoid fires. React web app provides live feed, zone editor, event log, cat stats, and manual controls.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, YOLOv8-nano (ultralytics), OpenCV, Shapely, SQLite, React, TypeScript, Vite, PlatformIO (C++ firmware)

---

## File Structure

### Firmware
- `firmware/esp32-cam/platformio.ini` — PlatformIO config for ESP32-CAM
- `firmware/esp32-cam/src/main.cpp` — WiFi connect + MJPEG streaming server
- `firmware/esp32-actuator/platformio.ini` — PlatformIO config for ESP32 DEVKITV1
- `firmware/esp32-actuator/src/main.cpp` — HTTP server receiving aim/fire commands, controlling servos + solenoid

### Server
- `server/requirements.txt` — Python dependencies
- `server/main.py` — FastAPI app entry point, lifespan, CORS, router includes
- `server/config.py` — Pydantic settings (env vars, defaults)
- `server/models/database.py` — SQLite schema + connection (zones, cats, events, settings tables)
- `server/vision/detector.py` — YOLOv8-nano wrapper, returns cat bounding boxes
- `server/vision/classifier.py` — MobileNetV3 cat identifier, train/predict
- `server/vision/zone_checker.py` — Shapely polygon intersection logic
- `server/vision/pipeline.py` — Orchestrates frame → detect → classify → zone check → fire decision
- `server/actuator/client.py` — HTTP client to ESP32 DEVKITV1 (aim + fire commands)
- `server/actuator/calibration.py` — Pixel-to-servo-angle mapping with interpolation
- `server/routers/stream.py` — Camera stream proxy, WebSocket live feed with overlays
- `server/routers/zones.py` — Zone CRUD REST API
- `server/routers/cats.py` — Cat management + photo upload + training trigger
- `server/routers/events.py` — Event log query API
- `server/routers/control.py` — Manual fire, arm/disarm, calibration endpoints
- `server/routers/settings.py` — Settings CRUD API
- `server/notifications/notifier.py` — Webhook (ntfy/Pushover) notifications

### Frontend
- `frontend/package.json` — React project dependencies
- `frontend/vite.config.ts` — Vite config with API proxy
- `frontend/tsconfig.json` — TypeScript config
- `frontend/index.html` — HTML entry point
- `frontend/src/main.tsx` — React entry point
- `frontend/src/App.tsx` — Root layout with navigation
- `frontend/src/types/index.ts` — Shared TypeScript types (Zone, Cat, Event, Settings)
- `frontend/src/api/client.ts` — API client functions + WebSocket hook
- `frontend/src/components/LiveFeed.tsx` — Camera feed canvas with detection overlays
- `frontend/src/components/ZoneEditor.tsx` — Draw/edit polygon zones on feed
- `frontend/src/components/EventLog.tsx` — Filterable event history list
- `frontend/src/components/CatStats.tsx` — Per-cat statistics dashboard
- `frontend/src/components/Controls.tsx` — Manual fire, arm/disarm, pan/tilt sliders
- `frontend/src/components/Settings.tsx` — Configuration panel
- `frontend/src/components/Calibration.tsx` — Servo calibration tool

### Tests
- `server/tests/test_zone_checker.py` — Zone intersection unit tests
- `server/tests/test_detector.py` — Detection wrapper tests
- `server/tests/test_calibration.py` — Pixel-to-angle mapping tests
- `server/tests/test_pipeline.py` — Pipeline integration tests
- `server/tests/test_routers_zones.py` — Zone API endpoint tests
- `server/tests/test_routers_events.py` — Event API endpoint tests
- `server/tests/test_routers_cats.py` — Cat API endpoint tests
- `server/tests/test_routers_control.py` — Control API endpoint tests

---

## Deferred: Cat Classifier (MobileNetV3)

The cat identification classifier (spec Stage 3) is intentionally not included in this plan. It requires actual photos of the user's cats to train, which can only happen after the system is running and capturing frames. The pipeline works without it — detected cats are labeled "Unknown Cat" until the classifier is trained. This will be a follow-up task once the base system is operational.

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `server/requirements.txt`
- Create: `server/main.py`
- Create: `server/config.py`
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `firmware/esp32-cam/platformio.ini`
- Create: `firmware/esp32-actuator/platformio.ini`
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/

# Node
node_modules/
dist/
.env

# PlatformIO
firmware/esp32-cam/.pio/
firmware/esp32-actuator/.pio/

# SQLite
*.db

# ML models
server/models/weights/
server/data/cat_photos/

# Superpowers
.superpowers/

# IDE
.vscode/
.idea/
```

- [ ] **Step 2: Create `server/requirements.txt`**

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
ultralytics==8.2.0
opencv-python-headless==4.10.0.84
shapely==2.0.4
aiosqlite==0.20.0
httpx==0.27.0
numpy==1.26.4
Pillow==10.4.0
torch==2.3.0
torchvision==0.18.0
websockets==12.0
```

- [ ] **Step 3: Create `server/config.py`**

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ESP32 addresses
    esp32_cam_url: str = "http://192.168.1.100:81/stream"
    esp32_actuator_url: str = "http://192.168.1.101"

    # Vision
    confidence_threshold: float = 0.5
    overlap_threshold: float = 0.3
    frame_skip_n: int = 2

    # Actuation
    cooldown_default: int = 10

    # Database
    db_path: Path = Path("catzap.db")

    # Notifications
    notification_webhook_url: str = ""

    # Cat photos storage
    cat_photos_dir: Path = Path("data/cat_photos")

    # Model weights
    classifier_weights_dir: Path = Path("models/weights")

    model_config = {"env_prefix": "CATZAP_"}


settings = Settings()
```

- [ ] **Step 4: Create `server/main.py`**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.models.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="CatZap", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Create `firmware/esp32-cam/platformio.ini`**

```ini
[env:esp32cam]
platform = espressif32
board = esp32cam
framework = arduino
monitor_speed = 115200
lib_deps =
build_flags = -DBOARD_HAS_PSRAM
```

- [ ] **Step 6: Create `firmware/esp32-actuator/platformio.ini`**

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
    ESP32Servo
```

- [ ] **Step 7: Create frontend scaffolding**

Run:
```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install
```

Then update `frontend/vite.config.ts`:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },
});
```

- [ ] **Step 8: Verify server starts**

Run:
```bash
cd server
pip install -r requirements.txt
uvicorn server.main:app --reload
```

Visit `http://localhost:8000/health` — expected response: `{"status": "ok"}`

- [ ] **Step 9: Commit**

```
feat: scaffold project structure with server, frontend, and firmware configs
```

---

## Task 2: Database Schema & Models

**Files:**
- Create: `server/models/__init__.py`
- Create: `server/models/database.py`
- Create: `server/tests/__init__.py`
- Create: `server/tests/test_database.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_database.py`:

```python
import pytest
import asyncio
from pathlib import Path
from server.models.database import init_db, get_db, create_zone, get_zones, create_cat, get_cats, create_event, get_events


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "test.db"
    await init_db(db_path)
    async with get_db(db_path) as conn:
        yield conn


@pytest.mark.asyncio
async def test_create_and_get_zone(db):
    zone_id = await create_zone(
        db,
        name="Kitchen Counter",
        polygon=[[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
        overlap_threshold=0.3,
        cooldown_seconds=10,
    )
    zones = await get_zones(db)
    assert len(zones) == 1
    assert zones[0]["name"] == "Kitchen Counter"
    assert zones[0]["id"] == zone_id
    assert zones[0]["enabled"] == True


@pytest.mark.asyncio
async def test_create_and_get_cat(db):
    cat_id = await create_cat(db, name="Luna")
    cats = await get_cats(db)
    assert len(cats) == 1
    assert cats[0]["name"] == "Luna"
    assert cats[0]["id"] == cat_id


@pytest.mark.asyncio
async def test_create_and_get_events(db):
    event_id = await create_event(
        db,
        event_type="ZAP",
        cat_name="Luna",
        zone_name="Kitchen Counter",
        confidence=0.92,
        overlap=0.65,
        servo_pan=45.0,
        servo_tilt=30.0,
    )
    events = await get_events(db)
    assert len(events) == 1
    assert events[0]["type"] == "ZAP"
    assert events[0]["cat_name"] == "Luna"
    assert events[0]["zone_name"] == "Kitchen Counter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_database.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.models.database'`

- [ ] **Step 3: Write the implementation**

Create `server/models/__init__.py` (empty file).

Create `server/tests/__init__.py` (empty file).

Create `server/models/database.py`:

```python
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite

from server.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    polygon TEXT NOT NULL,
    overlap_threshold REAL NOT NULL DEFAULT 0.3,
    cooldown_seconds INTEGER NOT NULL DEFAULT 10,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cats (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    model_version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    cat_id TEXT,
    cat_name TEXT,
    zone_id TEXT,
    zone_name TEXT,
    confidence REAL,
    overlap REAL,
    servo_pan REAL,
    servo_tilt REAL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


async def init_db(db_path: Path | None = None):
    path = db_path or settings.db_path
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


@asynccontextmanager
async def get_db(db_path: Path | None = None):
    path = db_path or settings.db_path
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def create_zone(
    db: aiosqlite.Connection,
    name: str,
    polygon: list[list[float]],
    overlap_threshold: float = 0.3,
    cooldown_seconds: int = 10,
) -> str:
    zone_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO zones (id, name, polygon, overlap_threshold, cooldown_seconds, enabled, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
        (zone_id, name, json.dumps(polygon), overlap_threshold, cooldown_seconds, datetime.utcnow().isoformat()),
    )
    await db.commit()
    return zone_id


async def get_zones(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute("SELECT * FROM zones ORDER BY created_at DESC")
    rows = await cursor.fetchall()
    return [
        {**dict(row), "polygon": json.loads(row["polygon"]), "enabled": bool(row["enabled"])}
        for row in rows
    ]


async def update_zone(db: aiosqlite.Connection, zone_id: str, **kwargs) -> bool:
    allowed = {"name", "polygon", "overlap_threshold", "cooldown_seconds", "enabled"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    if "polygon" in updates:
        updates["polygon"] = json.dumps(updates["polygon"])
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [zone_id]
    result = await db.execute(f"UPDATE zones SET {set_clause} WHERE id = ?", values)
    await db.commit()
    return result.rowcount > 0


async def delete_zone(db: aiosqlite.Connection, zone_id: str) -> bool:
    result = await db.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
    await db.commit()
    return result.rowcount > 0


async def create_cat(db: aiosqlite.Connection, name: str) -> str:
    cat_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO cats (id, name, model_version, created_at) VALUES (?, ?, 0, ?)",
        (cat_id, name, datetime.utcnow().isoformat()),
    )
    await db.commit()
    return cat_id


async def get_cats(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute("SELECT * FROM cats ORDER BY name")
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def delete_cat(db: aiosqlite.Connection, cat_id: str) -> bool:
    result = await db.execute("DELETE FROM cats WHERE id = ?", (cat_id,))
    await db.commit()
    return result.rowcount > 0


async def create_event(
    db: aiosqlite.Connection,
    event_type: str,
    cat_id: str | None = None,
    cat_name: str | None = None,
    zone_id: str | None = None,
    zone_name: str | None = None,
    confidence: float | None = None,
    overlap: float | None = None,
    servo_pan: float | None = None,
    servo_tilt: float | None = None,
) -> str:
    event_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO events (id, type, cat_id, cat_name, zone_id, zone_name, confidence, overlap, servo_pan, servo_tilt, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (event_id, event_type, cat_id, cat_name, zone_id, zone_name, confidence, overlap, servo_pan, servo_tilt, datetime.utcnow().isoformat()),
    )
    await db.commit()
    return event_id


async def get_events(
    db: aiosqlite.Connection,
    event_type: str | None = None,
    cat_name: str | None = None,
    zone_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    query = "SELECT * FROM events WHERE 1=1"
    params: list = []
    if event_type:
        query += " AND type = ?"
        params.append(event_type)
    if cat_name:
        query += " AND cat_name = ?"
        params.append(cat_name)
    if zone_name:
        query += " AND zone_name = ?"
        params.append(zone_name)
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_setting(db: aiosqlite.Connection, key: str) -> str | None:
    cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = await cursor.fetchone()
    return json.loads(row["value"]) if row else None


async def set_setting(db: aiosqlite.Connection, key: str, value) -> None:
    await db.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )
    await db.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_database.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add SQLite database schema and CRUD operations for zones, cats, events, settings
```

---

## Task 3: Zone Intersection Checker

**Files:**
- Create: `server/vision/__init__.py`
- Create: `server/vision/zone_checker.py`
- Create: `server/tests/test_zone_checker.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_zone_checker.py`:

```python
from server.vision.zone_checker import check_zone_violations


def test_cat_fully_inside_zone():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.1, 0.1], [0.6, 0.1], [0.6, 0.6], [0.1, 0.6]],
            "overlap_threshold": 0.3,
            "enabled": True,
        }
    ]
    # Cat bbox fully inside zone (normalized coords)
    bbox = [0.2, 0.2, 0.4, 0.4]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 1
    assert violations[0]["zone_name"] == "Kitchen Counter"
    assert violations[0]["overlap"] > 0.9


def test_cat_partially_inside_zone():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
            "overlap_threshold": 0.3,
            "enabled": True,
        }
    ]
    # Cat overlaps ~50% with zone
    bbox = [0.15, 0.0, 0.45, 0.3]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 1
    assert 0.4 < violations[0]["overlap"] < 0.6


def test_cat_outside_zone():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]],
            "overlap_threshold": 0.3,
            "enabled": True,
        }
    ]
    bbox = [0.5, 0.5, 0.7, 0.7]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 0


def test_overlap_below_threshold():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [0.25, 0.0], [0.25, 1.0], [0.0, 1.0]],
            "overlap_threshold": 0.5,
            "enabled": True,
        }
    ]
    # Cat barely overlaps with zone (~20%)
    bbox = [0.15, 0.0, 0.65, 1.0]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 0


def test_disabled_zone_ignored():
    zones = [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "overlap_threshold": 0.1,
            "enabled": False,
        }
    ]
    bbox = [0.2, 0.2, 0.4, 0.4]
    violations = check_zone_violations(bbox, zones)
    assert len(violations) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_zone_checker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/vision/__init__.py` (empty file).

Create `server/vision/zone_checker.py`:

```python
from shapely.geometry import Polygon, box


def check_zone_violations(
    bbox: list[float], zones: list[dict]
) -> list[dict]:
    """Check if a cat bounding box violates any zones.

    Args:
        bbox: [x1, y1, x2, y2] normalized coordinates (0-1).
        zones: List of zone dicts with polygon, overlap_threshold, enabled.

    Returns:
        List of violated zone dicts with added 'overlap' field.
    """
    cat_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
    cat_area = cat_box.area

    if cat_area == 0:
        return []

    violations = []
    for zone in zones:
        if not zone.get("enabled", True):
            continue

        zone_poly = Polygon(zone["polygon"])
        if not zone_poly.is_valid:
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_zone_checker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```
feat: add zone intersection checker using Shapely polygons
```

---

## Task 4: Cat Detection (YOLOv8 Wrapper)

**Files:**
- Create: `server/vision/detector.py`
- Create: `server/tests/test_detector.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_detector.py`:

```python
import numpy as np
from unittest.mock import patch, MagicMock
from server.vision.detector import CatDetector


def _make_mock_result(boxes_data, frame_shape=(480, 640, 3)):
    """Create a mock YOLO result object."""
    result = MagicMock()
    boxes = MagicMock()
    if len(boxes_data) == 0:
        boxes.xyxyn = MagicMock()
        boxes.xyxyn.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
        boxes.conf = MagicMock()
        boxes.conf.cpu.return_value.numpy.return_value = np.array([])
        boxes.cls = MagicMock()
        boxes.cls.cpu.return_value.numpy.return_value = np.array([])
    else:
        xyxyn = np.array([[b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]] for b in boxes_data])
        conf = np.array([b["conf"] for b in boxes_data])
        cls = np.array([b["cls"] for b in boxes_data])
        boxes.xyxyn = MagicMock()
        boxes.xyxyn.cpu.return_value.numpy.return_value = xyxyn
        boxes.conf = MagicMock()
        boxes.conf.cpu.return_value.numpy.return_value = conf
        boxes.cls = MagicMock()
        boxes.cls.cpu.return_value.numpy.return_value = cls
    result.boxes = boxes
    return result


@patch("server.vision.detector.YOLO")
def test_detect_cats_returns_cat_detections(mock_yolo_class):
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model
    mock_model.return_value = [
        _make_mock_result([
            {"bbox": [0.1, 0.2, 0.3, 0.5], "conf": 0.92, "cls": 15},  # cat
            {"bbox": [0.5, 0.5, 0.8, 0.9], "conf": 0.45, "cls": 0},   # person (filtered out)
        ])
    ]

    detector = CatDetector(confidence_threshold=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0]["bbox"] == [0.1, 0.2, 0.3, 0.5]
    assert detections[0]["confidence"] == 0.92


@patch("server.vision.detector.YOLO")
def test_detect_cats_empty_frame(mock_yolo_class):
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model
    mock_model.return_value = [_make_mock_result([])]

    detector = CatDetector(confidence_threshold=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 0


@patch("server.vision.detector.YOLO")
def test_detect_cats_filters_low_confidence(mock_yolo_class):
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model
    mock_model.return_value = [
        _make_mock_result([
            {"bbox": [0.1, 0.2, 0.3, 0.5], "conf": 0.3, "cls": 15},  # cat, low confidence
        ])
    ]

    detector = CatDetector(confidence_threshold=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/vision/detector.py`:

```python
import numpy as np
from ultralytics import YOLO

COCO_CAT_CLASS = 15


class CatDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect cats in a frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of dicts with 'bbox' (normalized [x1,y1,x2,y2]) and 'confidence'.
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            xyxyn = boxes.xyxyn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(classes)):
                if int(classes[i]) == COCO_CAT_CLASS and confs[i] >= self.confidence_threshold:
                    detections.append({
                        "bbox": xyxyn[i].tolist(),
                        "confidence": float(confs[i]),
                    })

        return detections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_detector.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add YOLOv8-nano cat detector wrapper
```

---

## Task 5: Servo Calibration & Angle Mapping

**Files:**
- Create: `server/actuator/__init__.py`
- Create: `server/actuator/calibration.py`
- Create: `server/tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_calibration.py`:

```python
import pytest
from server.actuator.calibration import CalibrationMap


def test_single_calibration_point():
    cal = CalibrationMap()
    cal.add_point(pixel_x=0.5, pixel_y=0.5, pan_angle=90.0, tilt_angle=45.0)
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert pan == 90.0
    assert tilt == 45.0


def test_interpolation_between_points():
    cal = CalibrationMap()
    cal.add_point(pixel_x=0.0, pixel_y=0.5, pan_angle=0.0, tilt_angle=45.0)
    cal.add_point(pixel_x=1.0, pixel_y=0.5, pan_angle=180.0, tilt_angle=45.0)
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert abs(pan - 90.0) < 5.0
    assert abs(tilt - 45.0) < 5.0


def test_four_corner_calibration():
    cal = CalibrationMap()
    cal.add_point(0.0, 0.0, pan_angle=30.0, tilt_angle=60.0)
    cal.add_point(1.0, 0.0, pan_angle=150.0, tilt_angle=60.0)
    cal.add_point(0.0, 1.0, pan_angle=30.0, tilt_angle=20.0)
    cal.add_point(1.0, 1.0, pan_angle=150.0, tilt_angle=20.0)
    # Center point should be roughly in the middle
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert 80.0 < pan < 100.0
    assert 35.0 < tilt < 50.0


def test_no_calibration_returns_default():
    cal = CalibrationMap()
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert pan == 90.0
    assert tilt == 90.0


def test_clear_points():
    cal = CalibrationMap()
    cal.add_point(0.5, 0.5, 90.0, 45.0)
    cal.clear()
    pan, tilt = cal.pixel_to_angle(0.5, 0.5)
    assert pan == 90.0  # default
    assert tilt == 90.0


def test_serialization():
    cal = CalibrationMap()
    cal.add_point(0.0, 0.0, 30.0, 60.0)
    cal.add_point(1.0, 1.0, 150.0, 20.0)
    data = cal.to_dict()
    cal2 = CalibrationMap.from_dict(data)
    pan, tilt = cal2.pixel_to_angle(0.0, 0.0)
    assert abs(pan - 30.0) < 1.0
    assert abs(tilt - 60.0) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/actuator/__init__.py` (empty file).

Create `server/actuator/calibration.py`:

```python
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class CalibrationMap:
    """Maps pixel coordinates (normalized 0-1) to servo angles using interpolation."""

    def __init__(self):
        self._points: list[dict] = []
        self._pan_interp = None
        self._tilt_interp = None

    def add_point(self, pixel_x: float, pixel_y: float, pan_angle: float, tilt_angle: float):
        self._points.append({
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "pan_angle": pan_angle,
            "tilt_angle": tilt_angle,
        })
        self._rebuild_interpolators()

    def clear(self):
        self._points = []
        self._pan_interp = None
        self._tilt_interp = None

    def _rebuild_interpolators(self):
        if len(self._points) < 1:
            self._pan_interp = None
            self._tilt_interp = None
            return

        pixels = np.array([[p["pixel_x"], p["pixel_y"]] for p in self._points])
        pans = np.array([p["pan_angle"] for p in self._points])
        tilts = np.array([p["tilt_angle"] for p in self._points])

        if len(self._points) == 1:
            # NearestNDInterpolator works with a single point
            self._pan_interp = NearestNDInterpolator(pixels, pans)
            self._tilt_interp = NearestNDInterpolator(pixels, tilts)
        elif len(self._points) <= 3:
            # Not enough for LinearND triangulation, use nearest
            self._pan_interp = NearestNDInterpolator(pixels, pans)
            self._tilt_interp = NearestNDInterpolator(pixels, tilts)
        else:
            # 4+ points: use linear interpolation with nearest as fallback
            self._pan_interp = LinearNDInterpolator(pixels, pans)
            self._tilt_interp = LinearNDInterpolator(pixels, tilts)
            self._pan_nearest = NearestNDInterpolator(pixels, pans)
            self._tilt_nearest = NearestNDInterpolator(pixels, tilts)

    def pixel_to_angle(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Convert normalized pixel coordinates to servo angles.

        Returns:
            (pan_angle, tilt_angle) in degrees.
        """
        if self._pan_interp is None:
            return (90.0, 90.0)

        point = np.array([[pixel_x, pixel_y]])
        pan = float(self._pan_interp(point)[0])
        tilt = float(self._tilt_interp(point)[0])

        # LinearNDInterpolator returns nan outside convex hull — fall back to nearest
        if np.isnan(pan) and hasattr(self, "_pan_nearest"):
            pan = float(self._pan_nearest(point)[0])
        if np.isnan(tilt) and hasattr(self, "_tilt_nearest"):
            tilt = float(self._tilt_nearest(point)[0])

        # Clamp to servo range
        pan = max(0.0, min(180.0, pan))
        tilt = max(0.0, min(180.0, tilt))

        return (pan, tilt)

    def to_dict(self) -> dict:
        return {"points": self._points}

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationMap":
        cal = cls()
        for p in data.get("points", []):
            cal.add_point(p["pixel_x"], p["pixel_y"], p["pan_angle"], p["tilt_angle"])
        return cal
```

Note: Add `scipy` to `server/requirements.txt`:
```
scipy==1.13.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_calibration.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```
feat: add servo calibration with pixel-to-angle interpolation
```

---

## Task 6: Actuator HTTP Client

**Files:**
- Create: `server/actuator/client.py`
- Create: `server/tests/test_actuator_client.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_actuator_client.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from server.actuator.client import ActuatorClient


@pytest.mark.asyncio
@patch("server.actuator.client.httpx.AsyncClient.post")
async def test_aim_sends_correct_angles(mock_post):
    mock_post.return_value = AsyncMock(status_code=200)
    client = ActuatorClient(base_url="http://192.168.1.101")
    await client.aim(pan=45.0, tilt=30.0)
    mock_post.assert_called_once_with(
        "http://192.168.1.101/aim",
        json={"pan": 45.0, "tilt": 30.0},
        timeout=2.0,
    )


@pytest.mark.asyncio
@patch("server.actuator.client.httpx.AsyncClient.post")
async def test_fire_sends_request(mock_post):
    mock_post.return_value = AsyncMock(status_code=200)
    client = ActuatorClient(base_url="http://192.168.1.101")
    await client.fire(duration_ms=200)
    mock_post.assert_called_once_with(
        "http://192.168.1.101/fire",
        json={"duration_ms": 200},
        timeout=2.0,
    )


@pytest.mark.asyncio
@patch("server.actuator.client.httpx.AsyncClient.post")
async def test_aim_and_fire_combined(mock_post):
    mock_post.return_value = AsyncMock(status_code=200)
    client = ActuatorClient(base_url="http://192.168.1.101")
    await client.aim_and_fire(pan=90.0, tilt=60.0, duration_ms=300)
    assert mock_post.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_actuator_client.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/actuator/client.py`:

```python
import logging

import httpx

logger = logging.getLogger(__name__)


class ActuatorClient:
    """HTTP client for communicating with the ESP32 DEVKITV1 actuator."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient()

    async def aim(self, pan: float, tilt: float) -> bool:
        """Send pan/tilt angles to the actuator.

        Args:
            pan: Pan servo angle (0-180 degrees).
            tilt: Tilt servo angle (0-180 degrees).

        Returns:
            True if command was acknowledged.
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/aim",
                json={"pan": pan, "tilt": tilt},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send aim command: {e}")
            return False

    async def fire(self, duration_ms: int = 200) -> bool:
        """Trigger the solenoid to fire.

        Args:
            duration_ms: How long to hold the solenoid in milliseconds.

        Returns:
            True if command was acknowledged.
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/fire",
                json={"duration_ms": duration_ms},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send fire command: {e}")
            return False

    async def aim_and_fire(
        self, pan: float, tilt: float, duration_ms: int = 200
    ) -> bool:
        """Aim and then fire in sequence."""
        aimed = await self.aim(pan, tilt)
        if aimed:
            return await self.fire(duration_ms)
        return False

    async def close(self):
        await self._client.aclose()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_actuator_client.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add HTTP actuator client for ESP32 servo/solenoid control
```

---

## Task 7: Vision Pipeline Orchestrator

**Files:**
- Create: `server/vision/pipeline.py`
- Create: `server/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_pipeline.py`:

```python
import asyncio
import time
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from server.vision.pipeline import VisionPipeline


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detector.detect.return_value = [
        {"bbox": [0.2, 0.2, 0.4, 0.5], "confidence": 0.92}
    ]
    return detector


@pytest.fixture
def mock_actuator():
    actuator = AsyncMock()
    actuator.aim_and_fire.return_value = True
    return actuator


@pytest.fixture
def mock_calibration():
    cal = MagicMock()
    cal.pixel_to_angle.return_value = (90.0, 45.0)
    return cal


@pytest.fixture
def zones():
    return [
        {
            "id": "z1",
            "name": "Kitchen Counter",
            "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6], [0.1, 0.6]],
            "overlap_threshold": 0.3,
            "cooldown_seconds": 10,
            "enabled": True,
        }
    ]


def test_pipeline_detects_violation_and_fires(mock_detector, mock_actuator, mock_calibration, zones):
    on_event = MagicMock()
    pipeline = VisionPipeline(
        detector=mock_detector,
        actuator=mock_actuator,
        calibration=mock_calibration,
        zones=zones,
        armed=True,
        on_event=on_event,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = asyncio.get_event_loop().run_until_complete(pipeline.process_frame(frame))

    assert len(result["detections"]) == 1
    assert len(result["violations"]) == 1
    assert result["fired"] == True
    mock_actuator.aim_and_fire.assert_called_once()
    assert on_event.call_count >= 1


def test_pipeline_respects_cooldown(mock_detector, mock_actuator, mock_calibration, zones):
    pipeline = VisionPipeline(
        detector=mock_detector,
        actuator=mock_actuator,
        calibration=mock_calibration,
        zones=zones,
        armed=True,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # First fire should happen
    result1 = asyncio.get_event_loop().run_until_complete(pipeline.process_frame(frame))
    assert result1["fired"] == True

    # Second fire immediately should be blocked by cooldown
    result2 = asyncio.get_event_loop().run_until_complete(pipeline.process_frame(frame))
    assert result2["fired"] == False


def test_pipeline_disarmed_does_not_fire(mock_detector, mock_actuator, mock_calibration, zones):
    pipeline = VisionPipeline(
        detector=mock_detector,
        actuator=mock_actuator,
        calibration=mock_calibration,
        zones=zones,
        armed=False,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = asyncio.get_event_loop().run_until_complete(pipeline.process_frame(frame))

    assert len(result["detections"]) == 1
    assert len(result["violations"]) == 1
    assert result["fired"] == False
    mock_actuator.aim_and_fire.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/vision/pipeline.py`:

```python
import logging
import time
from typing import Callable

import numpy as np

from server.actuator.calibration import CalibrationMap
from server.actuator.client import ActuatorClient
from server.vision.detector import CatDetector
from server.vision.zone_checker import check_zone_violations

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Orchestrates frame → detect → zone check → fire decision."""

    def __init__(
        self,
        detector: CatDetector,
        actuator: ActuatorClient,
        calibration: CalibrationMap,
        zones: list[dict],
        armed: bool = True,
        on_event: Callable | None = None,
    ):
        self.detector = detector
        self.actuator = actuator
        self.calibration = calibration
        self.zones = zones
        self.armed = armed
        self.on_event = on_event

        # zone_id -> last fire timestamp
        self._cooldowns: dict[str, float] = {}

    def update_zones(self, zones: list[dict]):
        self.zones = zones

    async def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame through the full pipeline.

        Returns:
            Dict with 'detections', 'violations', 'fired' keys.
        """
        # Stage 1-2: Detect cats
        detections = self.detector.detect(frame)

        all_violations = []
        fired = False

        for det in detections:
            bbox = det["bbox"]

            # Stage 4: Check zone violations
            violations = check_zone_violations(bbox, self.zones)
            all_violations.extend(violations)

            if not violations or not self.armed:
                continue

            # Stage 5: Fire decision
            for violation in violations:
                zone_id = violation["zone_id"]
                zone = next((z for z in self.zones if z["id"] == zone_id), None)
                if not zone:
                    continue

                cooldown = zone.get("cooldown_seconds", 10)
                last_fire = self._cooldowns.get(zone_id, 0)

                if time.time() - last_fire < cooldown:
                    continue

                # Calculate aim point (center of cat bbox)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                pan, tilt = self.calibration.pixel_to_angle(center_x, center_y)

                # Fire!
                success = await self.actuator.aim_and_fire(pan, tilt)
                if success:
                    self._cooldowns[zone_id] = time.time()
                    fired = True

                    if self.on_event:
                        self.on_event({
                            "type": "ZAP",
                            "cat_name": det.get("cat_name"),
                            "zone_name": violation["zone_name"],
                            "confidence": det["confidence"],
                            "overlap": violation["overlap"],
                            "servo_pan": pan,
                            "servo_tilt": tilt,
                        })

                    # Only fire at one zone per cat
                    break

        return {
            "detections": detections,
            "violations": all_violations,
            "fired": fired,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_pipeline.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add vision pipeline orchestrating detection, zone check, and fire decision
```

---

## Task 8: API Routes — Zones

**Files:**
- Create: `server/routers/__init__.py`
- Create: `server/routers/zones.py`
- Create: `server/tests/test_routers_zones.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_routers_zones.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import app
from server.models.database import init_db


@pytest.fixture(autouse=True)
async def setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("server.config.settings.db_path", db_path)
    await init_db(db_path)


@pytest.mark.asyncio
async def test_create_zone():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/zones", json={
            "name": "Kitchen Counter",
            "polygon": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
        })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Kitchen Counter"
    assert "id" in data


@pytest.mark.asyncio
async def test_get_zones():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/zones", json={
            "name": "Table",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        })
        response = await client.get("/api/zones")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1


@pytest.mark.asyncio
async def test_delete_zone():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/api/zones", json={
            "name": "Temp Zone",
            "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        })
        zone_id = create_resp.json()["id"]
        delete_resp = await client.delete(f"/api/zones/{zone_id}")
    assert delete_resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_routers_zones.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `server/routers/__init__.py` (empty file).

Create `server/routers/zones.py`:

```python
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from server.models.database import get_db, create_zone, get_zones, update_zone, delete_zone

router = APIRouter(prefix="/api/zones", tags=["zones"])


class ZoneCreate(BaseModel):
    name: str
    polygon: list[list[float]]
    overlap_threshold: float = 0.3
    cooldown_seconds: int = 10


class ZoneUpdate(BaseModel):
    name: str | None = None
    polygon: list[list[float]] | None = None
    overlap_threshold: float | None = None
    cooldown_seconds: int | None = None
    enabled: bool | None = None


@router.post("", status_code=201)
async def create_zone_endpoint(zone: ZoneCreate):
    async with get_db() as db:
        zone_id = await create_zone(
            db,
            name=zone.name,
            polygon=zone.polygon,
            overlap_threshold=zone.overlap_threshold,
            cooldown_seconds=zone.cooldown_seconds,
        )
        zones = await get_zones(db)
        return next(z for z in zones if z["id"] == zone_id)


@router.get("")
async def get_zones_endpoint():
    async with get_db() as db:
        return await get_zones(db)


@router.put("/{zone_id}")
async def update_zone_endpoint(zone_id: str, zone: ZoneUpdate):
    async with get_db() as db:
        updates = zone.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        success = await update_zone(db, zone_id, **updates)
        if not success:
            raise HTTPException(status_code=404, detail="Zone not found")
        zones = await get_zones(db)
        return next((z for z in zones if z["id"] == zone_id), None)


@router.delete("/{zone_id}")
async def delete_zone_endpoint(zone_id: str):
    async with get_db() as db:
        success = await delete_zone(db, zone_id)
        if not success:
            raise HTTPException(status_code=404, detail="Zone not found")
        return {"deleted": True}
```

Then add the router to `server/main.py` — add after the CORS middleware:

```python
from server.routers import zones

app.include_router(zones.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_routers_zones.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add zone CRUD API endpoints
```

---

## Task 9: API Routes — Cats

**Files:**
- Create: `server/routers/cats.py`
- Create: `server/tests/test_routers_cats.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_routers_cats.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import app
from server.models.database import init_db


@pytest.fixture(autouse=True)
async def setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("server.config.settings.db_path", db_path)
    await init_db(db_path)


@pytest.mark.asyncio
async def test_create_cat():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/cats", json={"name": "Luna"})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Luna"
    assert "id" in data


@pytest.mark.asyncio
async def test_get_cats():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/cats", json={"name": "Milo"})
        response = await client.get("/api/cats")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1


@pytest.mark.asyncio
async def test_delete_cat():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/api/cats", json={"name": "Temp"})
        cat_id = create_resp.json()["id"]
        delete_resp = await client.delete(f"/api/cats/{cat_id}")
    assert delete_resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_routers_cats.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Create `server/routers/cats.py`:

```python
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from server.models.database import get_db, create_cat, get_cats, delete_cat

router = APIRouter(prefix="/api/cats", tags=["cats"])


class CatCreate(BaseModel):
    name: str


@router.post("", status_code=201)
async def create_cat_endpoint(cat: CatCreate):
    async with get_db() as db:
        cat_id = await create_cat(db, name=cat.name)
        cats = await get_cats(db)
        return next(c for c in cats if c["id"] == cat_id)


@router.get("")
async def get_cats_endpoint():
    async with get_db() as db:
        return await get_cats(db)


@router.delete("/{cat_id}")
async def delete_cat_endpoint(cat_id: str):
    async with get_db() as db:
        success = await delete_cat(db, cat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Cat not found")
        return {"deleted": True}
```

Add to `server/main.py`:

```python
from server.routers import zones, cats

app.include_router(zones.router)
app.include_router(cats.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_routers_cats.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add cat management API endpoints
```

---

## Task 10: API Routes — Events

**Files:**
- Create: `server/routers/events.py`
- Create: `server/tests/test_routers_events.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_routers_events.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import app
from server.models.database import init_db, get_db, create_event


@pytest.fixture(autouse=True)
async def setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("server.config.settings.db_path", db_path)
    await init_db(db_path)


@pytest.mark.asyncio
async def test_get_events():
    async with get_db() as db:
        await create_event(db, event_type="ZAP", cat_name="Luna", zone_name="Counter")
        await create_event(db, event_type="DETECT_ENTER", cat_name="Milo", zone_name="Table")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/events")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


@pytest.mark.asyncio
async def test_get_events_filtered_by_type():
    async with get_db() as db:
        await create_event(db, event_type="ZAP", cat_name="Luna", zone_name="Counter")
        await create_event(db, event_type="SYSTEM")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/events?type=ZAP")
    assert response.status_code == 200
    data = response.json()
    assert all(e["type"] == "ZAP" for e in data)


@pytest.mark.asyncio
async def test_get_events_filtered_by_cat():
    async with get_db() as db:
        await create_event(db, event_type="ZAP", cat_name="Luna", zone_name="Counter")
        await create_event(db, event_type="ZAP", cat_name="Milo", zone_name="Table")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/events?cat_name=Luna")
    assert response.status_code == 200
    data = response.json()
    assert all(e["cat_name"] == "Luna" for e in data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_routers_events.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Create `server/routers/events.py`:

```python
from fastapi import APIRouter

from server.models.database import get_db, get_events

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("")
async def get_events_endpoint(
    type: str | None = None,
    cat_name: str | None = None,
    zone_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    async with get_db() as db:
        return await get_events(
            db,
            event_type=type,
            cat_name=cat_name,
            zone_name=zone_name,
            limit=limit,
            offset=offset,
        )
```

Add to `server/main.py`:

```python
from server.routers import zones, cats, events

app.include_router(zones.router)
app.include_router(cats.router)
app.include_router(events.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_routers_events.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add event log API with filtering
```

---

## Task 11: API Routes — Control (Arm/Disarm, Manual Fire)

**Files:**
- Create: `server/routers/control.py`
- Create: `server/tests/test_routers_control.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_routers_control.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from server.main import app
from server.models.database import init_db


@pytest.fixture(autouse=True)
async def setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("server.config.settings.db_path", db_path)
    await init_db(db_path)


@pytest.mark.asyncio
async def test_get_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/control/status")
    assert response.status_code == 200
    data = response.json()
    assert "armed" in data


@pytest.mark.asyncio
async def test_arm_disarm():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/control/arm", json={"armed": False})
    assert response.status_code == 200
    assert response.json()["armed"] == False


@pytest.mark.asyncio
async def test_add_calibration_point():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/control/calibrate", json={
            "pixel_x": 0.5,
            "pixel_y": 0.5,
            "pan_angle": 90.0,
            "tilt_angle": 45.0,
        })
    assert response.status_code == 200
    assert response.json()["points_count"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_routers_control.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

Create `server/routers/control.py`:

```python
from pydantic import BaseModel
from fastapi import APIRouter

from server.actuator.calibration import CalibrationMap
from server.actuator.client import ActuatorClient
from server.config import settings

router = APIRouter(prefix="/api/control", tags=["control"])

# Shared state — in production, these would be managed by the app lifespan
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
```

Add to `server/main.py`:

```python
from server.routers import zones, cats, events, control

app.include_router(zones.router)
app.include_router(cats.router)
app.include_router(events.router)
app.include_router(control.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_routers_control.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```
feat: add control API for arm/disarm, manual fire, and calibration
```

---

## Task 12: Camera Stream & WebSocket Feed

**Files:**
- Create: `server/routers/stream.py`

- [ ] **Step 1: Write the implementation**

This task involves streaming and WebSocket connections which are harder to unit test meaningfully. We'll write the code and test it manually with the ESP32-CAM.

Create `server/routers/stream.py`:

```python
import asyncio
import logging
import time
from typing import Any

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

# Connected WebSocket clients
_ws_clients: list[WebSocket] = []


async def broadcast_to_clients(data: dict):
    """Send data to all connected WebSocket clients."""
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)


class MJPEGStreamReader:
    """Reads MJPEG frames from the ESP32-CAM HTTP stream."""

    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self._client: httpx.AsyncClient | None = None
        self._response = None

    async def connect(self):
        self._client = httpx.AsyncClient(timeout=None)
        self._response = await self._client.stream("GET", self.stream_url).__aenter__()

    async def read_frame(self) -> np.ndarray | None:
        """Read a single JPEG frame from the MJPEG stream."""
        if not self._response:
            return None

        buffer = b""
        async for chunk in self._response.aiter_bytes(chunk_size=4096):
            buffer += chunk
            # JPEG starts with FFD8 and ends with FFD9
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg_bytes = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                frame = cv2.imdecode(
                    np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                return frame
        return None

    async def close(self):
        if self._response:
            await self._response.aclose()
        if self._client:
            await self._client.aclose()


@router.websocket("/ws/feed")
async def video_feed(websocket: WebSocket):
    """WebSocket endpoint that streams processed frames with detection overlays."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected")

    try:
        while True:
            # Keep connection alive, client sends pings
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected")


@router.websocket("/ws/events")
async def event_feed(websocket: WebSocket):
    """WebSocket endpoint for real-time event notifications."""
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
```

Add to `server/main.py`:

```python
from server.routers import zones, cats, events, control, stream

app.include_router(zones.router)
app.include_router(cats.router)
app.include_router(events.router)
app.include_router(control.router)
app.include_router(stream.router)
```

- [ ] **Step 2: Verify server starts with all routes**

Run: `cd server && uvicorn server.main:app --reload`
Visit `http://localhost:8000/docs` — should see all API endpoints listed in Swagger UI.

- [ ] **Step 3: Commit**

```
feat: add MJPEG stream reader and WebSocket feed endpoints
```

---

## Task 13: Settings API & Notifications

**Files:**
- Create: `server/routers/settings.py`
- Create: `server/notifications/__init__.py`
- Create: `server/notifications/notifier.py`

- [ ] **Step 1: Write the implementation**

Create `server/routers/settings.py`:

```python
from pydantic import BaseModel
from fastapi import APIRouter

from server.models.database import get_db, get_setting, set_setting

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingUpdate(BaseModel):
    key: str
    value: str | int | float | bool | list | dict


@router.get("")
async def get_all_settings():
    keys = [
        "cooldown_default",
        "confidence_threshold",
        "overlap_threshold",
        "arm_schedule",
        "notification_webhook_url",
        "frame_skip_n",
        "esp32_cam_url",
        "esp32_actuator_url",
    ]
    result = {}
    async with get_db() as db:
        for key in keys:
            val = await get_setting(db, key)
            result[key] = val
    return result


@router.get("/{key}")
async def get_setting_endpoint(key: str):
    async with get_db() as db:
        value = await get_setting(db, key)
        return {"key": key, "value": value}


@router.put("/{key}")
async def set_setting_endpoint(key: str, update: SettingUpdate):
    async with get_db() as db:
        await set_setting(db, key, update.value)
        return {"key": key, "value": update.value}
```

Create `server/notifications/__init__.py` (empty file).

Create `server/notifications/notifier.py`:

```python
import logging

import httpx

logger = logging.getLogger(__name__)


class Notifier:
    """Sends notifications via webhook (ntfy/Pushover)."""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url
        self._client = httpx.AsyncClient()

    async def notify(self, title: str, message: str):
        """Send a notification. Silently fails if no webhook configured."""
        if not self.webhook_url:
            return

        try:
            await self._client.post(
                self.webhook_url,
                json={"title": title, "message": message},
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )
        except httpx.RequestError as e:
            logger.error(f"Notification failed: {e}")

    async def notify_zap(self, cat_name: str | None, zone_name: str | None):
        cat = cat_name or "Unknown cat"
        zone = zone_name or "unknown zone"
        await self.notify(
            title="CatZap Fired!",
            message=f"{cat} was caught on {zone} and got zapped!",
        )

    async def close(self):
        await self._client.aclose()
```

Add settings router to `server/main.py`:

```python
from server.routers import zones, cats, events, control, stream, settings

app.include_router(zones.router)
app.include_router(cats.router)
app.include_router(events.router)
app.include_router(control.router)
app.include_router(stream.router)
app.include_router(settings.router)
```

- [ ] **Step 2: Verify server starts**

Run: `cd server && uvicorn server.main:app --reload`
Visit `http://localhost:8000/docs` — settings endpoints should be listed.

- [ ] **Step 3: Commit**

```
feat: add settings API and webhook notification service
```

---

## Task 14: ESP32-CAM Firmware

**Files:**
- Create: `firmware/esp32-cam/src/main.cpp`

- [ ] **Step 1: Write the firmware**

Create `firmware/esp32-cam/src/main.cpp`:

```cpp
#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <WebServer.h>

// ===== WiFi credentials — update these =====
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ===== AI-Thinker ESP32-CAM pin definitions =====
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WebServer server(81);

void handleStream() {
    WiFiClient client = server.client();

    String response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    client.print(response);

    while (client.connected()) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            break;
        }

        String header = "--frame\r\n";
        header += "Content-Type: image/jpeg\r\n";
        header += "Content-Length: " + String(fb->len) + "\r\n\r\n";

        client.print(header);
        client.write(fb->buf, fb->len);
        client.print("\r\n");

        esp_camera_fb_return(fb);

        if (!client.connected()) break;
    }
}

void handleHealth() {
    server.send(200, "application/json", "{\"status\":\"ok\"}");
}

void setup() {
    Serial.begin(115200);
    Serial.println("CatZap ESP32-CAM starting...");

    // Camera configuration
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;

    // Initialize camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return;
    }
    Serial.println("Camera initialized");

    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected! IP address: ");
    Serial.println(WiFi.localIP());

    // Start HTTP server
    server.on("/stream", HTTP_GET, handleStream);
    server.on("/health", HTTP_GET, handleHealth);
    server.begin();
    Serial.println("Stream server started on port 81");
    Serial.printf("Stream URL: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
}

void loop() {
    server.handleClient();
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd firmware/esp32-cam && pio run`
Expected: Compilation succeeds (you'll need PlatformIO installed)

- [ ] **Step 3: Commit**

```
feat: add ESP32-CAM MJPEG streaming firmware
```

---

## Task 15: ESP32 DEVKITV1 Actuator Firmware

**Files:**
- Create: `firmware/esp32-actuator/src/main.cpp`

- [ ] **Step 1: Write the firmware**

Create `firmware/esp32-actuator/src/main.cpp`:

```cpp
#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>

// ===== WiFi credentials — update these =====
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ===== Pin definitions =====
#define PAN_SERVO_PIN   18
#define TILT_SERVO_PIN  19
#define SOLENOID_PIN    23
#define STATUS_LED_PIN   2

// ===== Servo objects =====
Servo panServo;
Servo tiltServo;

// ===== Current state =====
float currentPan = 90.0;
float currentTilt = 90.0;

WebServer server(80);

void handleAim() {
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

    // Clamp to valid range
    pan = constrain(pan, 0.0, 180.0);
    tilt = constrain(tilt, 0.0, 180.0);

    panServo.write((int)pan);
    tiltServo.write((int)tilt);
    currentPan = pan;
    currentTilt = tilt;

    Serial.printf("Aimed to pan=%.1f, tilt=%.1f\n", pan, tilt);

    String response;
    JsonDocument respDoc;
    respDoc["pan"] = pan;
    respDoc["tilt"] = tilt;
    serializeJson(respDoc, response);
    server.send(200, "application/json", response);
}

void handleFire() {
    if (server.method() != HTTP_POST) {
        server.send(405, "application/json", "{\"error\":\"Method not allowed\"}");
        return;
    }

    String body = server.arg("plain");
    JsonDocument doc;
    deserializeJson(doc, body);

    int duration_ms = doc["duration_ms"] | 200;
    duration_ms = constrain(duration_ms, 50, 2000);

    Serial.printf("FIRE! Duration: %dms\n", duration_ms);

    // Activate solenoid
    digitalWrite(SOLENOID_PIN, HIGH);
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(duration_ms);
    digitalWrite(SOLENOID_PIN, LOW);
    digitalWrite(STATUS_LED_PIN, LOW);

    String response;
    JsonDocument respDoc;
    respDoc["fired"] = true;
    respDoc["duration_ms"] = duration_ms;
    serializeJson(respDoc, response);
    server.send(200, "application/json", response);
}

void handleHealth() {
    String response;
    JsonDocument doc;
    doc["status"] = "ok";
    doc["pan"] = currentPan;
    doc["tilt"] = currentTilt;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}

void setup() {
    Serial.begin(115200);
    Serial.println("CatZap Actuator starting...");

    // Setup pins
    pinMode(SOLENOID_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(SOLENOID_PIN, LOW);
    digitalWrite(STATUS_LED_PIN, LOW);

    // Attach servos
    panServo.attach(PAN_SERVO_PIN);
    tiltServo.attach(TILT_SERVO_PIN);
    panServo.write(90);
    tiltServo.write(90);

    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected! IP address: ");
    Serial.println(WiFi.localIP());

    // Start HTTP server
    server.on("/aim", handleAim);
    server.on("/fire", handleFire);
    server.on("/health", handleHealth);
    server.begin();
    Serial.println("Actuator server started on port 80");
}

void loop() {
    server.handleClient();
}
```

Update `firmware/esp32-actuator/platformio.ini` to include ArduinoJson:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
    ESP32Servo
    bblanchon/ArduinoJson@^7.0.0
```

- [ ] **Step 2: Verify it compiles**

Run: `cd firmware/esp32-actuator && pio run`
Expected: Compilation succeeds

- [ ] **Step 3: Commit**

```
feat: add ESP32 actuator firmware with aim/fire HTTP endpoints
```

---

## Task 16: Frontend — Types & API Client

**Files:**
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: Create shared types**

Create `frontend/src/types/index.ts`:

```typescript
export interface Zone {
  id: string;
  name: string;
  polygon: number[][];
  overlap_threshold: number;
  cooldown_seconds: number;
  enabled: boolean;
  created_at: string;
}

export interface Cat {
  id: string;
  name: string;
  model_version: number;
  created_at: string;
}

export interface CatEvent {
  id: string;
  type: "ZAP" | "DETECT_ENTER" | "DETECT_EXIT" | "SYSTEM";
  cat_id: string | null;
  cat_name: string | null;
  zone_id: string | null;
  zone_name: string | null;
  confidence: number | null;
  overlap: number | null;
  servo_pan: number | null;
  servo_tilt: number | null;
  timestamp: string;
}

export interface Detection {
  bbox: number[];
  confidence: number;
  cat_name?: string;
}

export interface Violation {
  zone_id: string;
  zone_name: string;
  overlap: number;
}

export interface FrameResult {
  detections: Detection[];
  violations: Violation[];
  fired: boolean;
}

export interface ControlStatus {
  armed: boolean;
  calibration_points: number;
}
```

- [ ] **Step 2: Create API client**

Create `frontend/src/api/client.ts`:

```typescript
import type { Zone, Cat, CatEvent, ControlStatus } from "../types";

const API_BASE = "/api";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// Zones
export const getZones = () => fetchJSON<Zone[]>("/zones");

export const createZone = (zone: {
  name: string;
  polygon: number[][];
  overlap_threshold?: number;
  cooldown_seconds?: number;
}) => fetchJSON<Zone>("/zones", { method: "POST", body: JSON.stringify(zone) });

export const updateZone = (id: string, updates: Partial<Zone>) =>
  fetchJSON<Zone>(`/zones/${id}`, {
    method: "PUT",
    body: JSON.stringify(updates),
  });

export const deleteZone = (id: string) =>
  fetchJSON(`/zones/${id}`, { method: "DELETE" });

// Cats
export const getCats = () => fetchJSON<Cat[]>("/cats");

export const createCat = (name: string) =>
  fetchJSON<Cat>("/cats", {
    method: "POST",
    body: JSON.stringify({ name }),
  });

export const deleteCat = (id: string) =>
  fetchJSON(`/cats/${id}`, { method: "DELETE" });

// Events
export const getEvents = (params?: {
  type?: string;
  cat_name?: string;
  zone_name?: string;
  limit?: number;
  offset?: number;
}) => {
  const searchParams = new URLSearchParams();
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) searchParams.set(key, String(value));
    });
  }
  const query = searchParams.toString();
  return fetchJSON<CatEvent[]>(`/events${query ? `?${query}` : ""}`);
};

// Control
export const getControlStatus = () =>
  fetchJSON<ControlStatus>("/control/status");

export const setArmed = (armed: boolean) =>
  fetchJSON<{ armed: boolean }>("/control/arm", {
    method: "POST",
    body: JSON.stringify({ armed }),
  });

export const manualFire = (pan: number, tilt: number, duration_ms?: number) =>
  fetchJSON("/control/fire", {
    method: "POST",
    body: JSON.stringify({ pan, tilt, duration_ms: duration_ms ?? 200 }),
  });

export const addCalibrationPoint = (
  pixel_x: number,
  pixel_y: number,
  pan_angle: number,
  tilt_angle: number
) =>
  fetchJSON("/control/calibrate", {
    method: "POST",
    body: JSON.stringify({ pixel_x, pixel_y, pan_angle, tilt_angle }),
  });

export const clearCalibration = () =>
  fetchJSON("/control/calibrate", { method: "DELETE" });

// WebSocket
export function connectEventSocket(
  onEvent: (event: CatEvent) => void
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/events`);
  ws.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    onEvent(event);
  };
  return ws;
}
```

- [ ] **Step 3: Commit**

```
feat: add frontend TypeScript types and API client
```

---

## Task 17: Frontend — App Layout & Live Feed

**Files:**
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/components/LiveFeed.tsx`

- [ ] **Step 1: Create LiveFeed component**

Create `frontend/src/components/LiveFeed.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import type { Detection, Violation, Zone } from "../types";

interface LiveFeedProps {
  zones: Zone[];
  onClickPoint?: (x: number, y: number) => void;
}

export default function LiveFeed({ zones, onClickPoint }: LiveFeedProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [fps, setFps] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/feed`);
    let frameCount = 0;
    let lastFpsTime = Date.now();

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);

      if (data.frame) {
        // Base64 encoded JPEG frame
        const img = new Image();
        img.onload = () => {
          imgRef.current = img;
          drawFrame(img, data.detections || [], data.violations || []);
          frameCount++;
          const now = Date.now();
          if (now - lastFpsTime >= 1000) {
            setFps(frameCount);
            frameCount = 0;
            lastFpsTime = now;
          }
        };
        img.src = `data:image/jpeg;base64,${data.frame}`;
        setDetections(data.detections || []);
        setViolations(data.violations || []);
      }
    };

    return () => ws.close();
  }, [zones]);

  function drawFrame(
    img: HTMLImageElement,
    dets: Detection[],
    viols: Violation[]
  ) {
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
      ctx.setLineDash([8, 4]);
      ctx.beginPath();
      zone.polygon.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x * w, y * h);
        else ctx.lineTo(x * w, y * h);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = "rgba(249, 65, 68, 0.08)";
      ctx.fill();
      ctx.setLineDash([]);

      // Zone label
      if (zone.polygon.length > 0) {
        const [lx, ly] = zone.polygon[0];
        ctx.fillStyle = "#f94144";
        ctx.font = "12px monospace";
        ctx.fillText(zone.name, lx * w, ly * h - 4);
      }
    }

    // Draw detections
    for (const det of dets) {
      const [x1, y1, x2, y2] = det.bbox;
      const isViolating = viols.some((v) =>
        zones.some((z) => z.id === v.zone_id)
      );

      ctx.strokeStyle = "#4cc9f0";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);

      // Label
      const label = det.cat_name
        ? `${det.cat_name} ${Math.round(det.confidence * 100)}%`
        : `Cat ${Math.round(det.confidence * 100)}%`;
      ctx.fillStyle = "rgba(76, 201, 240, 0.3)";
      ctx.fillRect(x1 * w, y1 * h - 18, ctx.measureText(label).width + 8, 18);
      ctx.fillStyle = "#4cc9f0";
      ctx.font = "12px monospace";
      ctx.fillText(label, x1 * w + 4, y1 * h - 4);

      if (isViolating) {
        ctx.fillStyle = "#f94144";
        ctx.font = "bold 11px monospace";
        ctx.fillText("IN ZONE!", x1 * w, (y2 * h) + 14);
      }
    }
  }

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!canvasRef.current || !onClickPoint) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    onClickPoint(x, y);
  }

  return (
    <div style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{ width: "100%", cursor: onClickPoint ? "crosshair" : "default" }}
      />
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          padding: "8px 12px",
          background: "rgba(0,0,0,0.7)",
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "monospace",
          fontSize: "12px",
          color: "#ccc",
        }}
      >
        <span>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: fps > 0 ? "#4cc9f0" : "#888",
              display: "inline-block",
              marginRight: 8,
            }}
          />
          {fps > 0 ? `LIVE — ${fps} FPS` : "Connecting..."}
        </span>
        <span>
          Cats: {detections.length} | Violations: {violations.length}
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Update App.tsx**

Replace `frontend/src/App.tsx` with:

```tsx
import { useEffect, useState } from "react";
import LiveFeed from "./components/LiveFeed";
import EventLog from "./components/EventLog";
import CatStats from "./components/CatStats";
import Controls from "./components/Controls";
import ZoneEditor from "./components/ZoneEditor";
import Settings from "./components/Settings";
import type { Zone } from "./types";
import { getZones } from "./api/client";

type Tab = "live" | "events" | "stats" | "settings";

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [zones, setZones] = useState<Zone[]>([]);
  const [editingZones, setEditingZones] = useState(false);

  useEffect(() => {
    getZones().then(setZones).catch(console.error);
  }, []);

  const refreshZones = () => getZones().then(setZones).catch(console.error);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 16 }}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 24, fontFamily: "monospace" }}>
          CatZap
        </h1>
        <nav style={{ display: "flex", gap: 8 }}>
          {(["live", "events", "stats", "settings"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "8px 16px",
                background: tab === t ? "#4cc9f0" : "#333",
                color: tab === t ? "#1a1a2e" : "#ccc",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontFamily: "monospace",
                textTransform: "capitalize",
              }}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {tab === "live" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ position: "relative" }}>
            {editingZones ? (
              <ZoneEditor
                zones={zones}
                onSave={() => {
                  setEditingZones(false);
                  refreshZones();
                }}
                onCancel={() => setEditingZones(false)}
              />
            ) : (
              <LiveFeed zones={zones} />
            )}
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button
                onClick={() => setEditingZones(!editingZones)}
                style={{
                  padding: "8px 16px",
                  background: editingZones ? "#f94144" : "#f72585",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: "pointer",
                  fontFamily: "monospace",
                }}
              >
                {editingZones ? "Cancel" : "+ Draw Zone"}
              </button>
            </div>
          </div>
          <Controls />
        </div>
      )}

      {tab === "events" && <EventLog />}
      {tab === "stats" && <CatStats />}
      {tab === "settings" && <Settings onUpdate={refreshZones} />}
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```
feat: add LiveFeed component with canvas overlays and App layout
```

---

## Task 18: Frontend — Zone Editor

**Files:**
- Create: `frontend/src/components/ZoneEditor.tsx`

- [ ] **Step 1: Write the component**

Create `frontend/src/components/ZoneEditor.tsx`:

```tsx
import { useRef, useState } from "react";
import type { Zone } from "../types";
import { createZone, deleteZone } from "../api/client";

interface ZoneEditorProps {
  zones: Zone[];
  onSave: () => void;
  onCancel: () => void;
}

export default function ZoneEditor({ zones, onSave, onCancel }: ZoneEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<number[][]>([]);
  const [zoneName, setZoneName] = useState("");

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setPoints([...points, [x, y]]);
  }

  async function handleSaveZone() {
    if (points.length < 3 || !zoneName.trim()) return;
    await createZone({ name: zoneName.trim(), polygon: points });
    setPoints([]);
    setZoneName("");
    onSave();
  }

  async function handleDeleteZone(id: string) {
    await deleteZone(id);
    onSave();
  }

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <p style={{ color: "#ccc", fontFamily: "monospace", fontSize: 12 }}>
          Click on the feed to place polygon points. Minimum 3 points to create
          a zone.
        </p>
      </div>

      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        onClick={handleCanvasClick}
        style={{
          width: "100%",
          background: "#1a1a2e",
          cursor: "crosshair",
          border: "2px solid #f72585",
          borderRadius: 8,
        }}
      />

      {/* Draw current points on canvas */}
      {canvasRef.current && points.length > 0 && (
        <svg
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
        >
          <polygon
            points={points.map(([x, y]) => `${x},${y}`).join(" ")}
            fill="rgba(247, 37, 133, 0.15)"
            stroke="#f72585"
            strokeWidth="0.003"
            strokeDasharray="0.01 0.005"
          />
          {points.map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r="0.008" fill="#f72585" />
          ))}
        </svg>
      )}

      <div
        style={{
          display: "flex",
          gap: 8,
          marginTop: 8,
          alignItems: "center",
        }}
      >
        <input
          type="text"
          placeholder="Zone name..."
          value={zoneName}
          onChange={(e) => setZoneName(e.target.value)}
          style={{
            padding: "8px 12px",
            background: "#333",
            border: "1px solid #555",
            borderRadius: 6,
            color: "#ccc",
            fontFamily: "monospace",
            flex: 1,
          }}
        />
        <button
          onClick={handleSaveZone}
          disabled={points.length < 3 || !zoneName.trim()}
          style={{
            padding: "8px 16px",
            background: points.length >= 3 && zoneName.trim() ? "#4cc9f0" : "#555",
            color: points.length >= 3 && zoneName.trim() ? "#1a1a2e" : "#888",
            border: "none",
            borderRadius: 6,
            cursor: points.length >= 3 && zoneName.trim() ? "pointer" : "not-allowed",
            fontFamily: "monospace",
          }}
        >
          Save Zone ({points.length} points)
        </button>
        <button
          onClick={() => setPoints([])}
          style={{
            padding: "8px 16px",
            background: "#333",
            color: "#ccc",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontFamily: "monospace",
          }}
        >
          Clear
        </button>
        <button
          onClick={onCancel}
          style={{
            padding: "8px 16px",
            background: "#f94144",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontFamily: "monospace",
          }}
        >
          Done
        </button>
      </div>

      {/* Existing zones list */}
      {zones.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3
            style={{
              color: "#ccc",
              fontFamily: "monospace",
              fontSize: 14,
              marginBottom: 8,
            }}
          >
            Existing Zones
          </h3>
          {zones.map((zone) => (
            <div
              key={zone.id}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "8px 12px",
                background: "#222",
                borderRadius: 6,
                marginBottom: 4,
                fontFamily: "monospace",
                fontSize: 12,
              }}
            >
              <span style={{ color: "#f94144" }}>{zone.name}</span>
              <button
                onClick={() => handleDeleteZone(zone.id)}
                style={{
                  padding: "4px 8px",
                  background: "#f94144",
                  color: "white",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                  fontSize: 11,
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```
feat: add ZoneEditor component for drawing polygon zones
```

---

## Task 19: Frontend — Event Log

**Files:**
- Create: `frontend/src/components/EventLog.tsx`

- [ ] **Step 1: Write the component**

Create `frontend/src/components/EventLog.tsx`:

```tsx
import { useEffect, useState } from "react";
import type { CatEvent } from "../types";
import { getEvents, connectEventSocket } from "../api/client";

const EVENT_COLORS: Record<string, string> = {
  ZAP: "#f94144",
  DETECT_ENTER: "#4cc9f0",
  DETECT_EXIT: "#4cc9f0",
  SYSTEM: "#7209b7",
};

export default function EventLog() {
  const [events, setEvents] = useState<CatEvent[]>([]);
  const [filter, setFilter] = useState({ type: "", cat_name: "" });

  useEffect(() => {
    loadEvents();
    const ws = connectEventSocket((event) => {
      setEvents((prev) => [event, ...prev].slice(0, 200));
    });
    return () => ws.close();
  }, []);

  useEffect(() => {
    loadEvents();
  }, [filter]);

  async function loadEvents() {
    const params: Record<string, string> = {};
    if (filter.type) params.type = filter.type;
    if (filter.cat_name) params.cat_name = filter.cat_name;
    const data = await getEvents(params);
    setEvents(data);
  }

  function formatTime(ts: string) {
    return new Date(ts).toLocaleTimeString();
  }

  function formatDate(ts: string) {
    return new Date(ts).toLocaleDateString();
  }

  function eventMessage(event: CatEvent): string {
    switch (event.type) {
      case "ZAP":
        return `${event.cat_name || "Cat"} on ${event.zone_name || "zone"} — ZAPPED!`;
      case "DETECT_ENTER":
        return `${event.cat_name || "Cat"} entered ${event.zone_name || "zone"}`;
      case "DETECT_EXIT":
        return `${event.cat_name || "Cat"} left ${event.zone_name || "zone"}`;
      case "SYSTEM":
        return event.zone_name || "System event";
      default:
        return "Unknown event";
    }
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <select
          value={filter.type}
          onChange={(e) => setFilter({ ...filter, type: e.target.value })}
          style={{
            padding: "8px 12px",
            background: "#333",
            border: "1px solid #555",
            borderRadius: 6,
            color: "#ccc",
            fontFamily: "monospace",
          }}
        >
          <option value="">All Types</option>
          <option value="ZAP">Zaps</option>
          <option value="DETECT_ENTER">Detections</option>
          <option value="SYSTEM">System</option>
        </select>
        <input
          type="text"
          placeholder="Filter by cat name..."
          value={filter.cat_name}
          onChange={(e) => setFilter({ ...filter, cat_name: e.target.value })}
          style={{
            padding: "8px 12px",
            background: "#333",
            border: "1px solid #555",
            borderRadius: 6,
            color: "#ccc",
            fontFamily: "monospace",
            flex: 1,
          }}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {events.length === 0 && (
          <p
            style={{
              color: "#888",
              fontFamily: "monospace",
              textAlign: "center",
              padding: 32,
            }}
          >
            No events yet. Waiting for cat activity...
          </p>
        )}
        {events.map((event) => (
          <div
            key={event.id}
            style={{
              display: "flex",
              gap: 12,
              padding: "8px 12px",
              background: `${EVENT_COLORS[event.type] || "#333"}15`,
              borderLeft: `3px solid ${EVENT_COLORS[event.type] || "#333"}`,
              borderRadius: 4,
              fontFamily: "monospace",
              fontSize: 12,
              alignItems: "center",
            }}
          >
            <span style={{ color: "#888", minWidth: 70 }}>
              {formatTime(event.timestamp)}
            </span>
            <span
              style={{
                color: EVENT_COLORS[event.type] || "#ccc",
                minWidth: 50,
                fontWeight: "bold",
              }}
            >
              {event.type}
            </span>
            <span style={{ color: "#ccc", flex: 1 }}>
              {eventMessage(event)}
            </span>
            {event.confidence && (
              <span style={{ color: "#888" }}>
                {Math.round(event.confidence * 100)}%
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```
feat: add EventLog component with real-time updates and filtering
```

---

## Task 20: Frontend — Cat Stats

**Files:**
- Create: `frontend/src/components/CatStats.tsx`

- [ ] **Step 1: Write the component**

Create `frontend/src/components/CatStats.tsx`:

```tsx
import { useEffect, useState } from "react";
import type { Cat, CatEvent } from "../types";
import { getCats, getEvents } from "../api/client";

interface CatStat {
  cat: Cat;
  zapCount: number;
  detectCount: number;
  zapRate: number;
  favoriteZone: string;
  peakHour: string;
}

export default function CatStats() {
  const [stats, setStats] = useState<CatStat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  async function loadStats() {
    setLoading(true);
    const [cats, events] = await Promise.all([getCats(), getEvents({ limit: 1000 })]);

    const catStats: CatStat[] = cats.map((cat) => {
      const catEvents = events.filter((e) => e.cat_name === cat.name);
      const zaps = catEvents.filter((e) => e.type === "ZAP");
      const detects = catEvents.filter(
        (e) => e.type === "ZAP" || e.type === "DETECT_ENTER"
      );

      // Find favorite zone
      const zoneCounts: Record<string, number> = {};
      zaps.forEach((e) => {
        if (e.zone_name) {
          zoneCounts[e.zone_name] = (zoneCounts[e.zone_name] || 0) + 1;
        }
      });
      const favoriteZone =
        Object.entries(zoneCounts).sort(([, a], [, b]) => b - a)[0]?.[0] ||
        "None";

      // Find peak hour
      const hourCounts: Record<number, number> = {};
      catEvents.forEach((e) => {
        const hour = new Date(e.timestamp).getHours();
        hourCounts[hour] = (hourCounts[hour] || 0) + 1;
      });
      const peakHourNum = Object.entries(hourCounts).sort(
        ([, a], [, b]) => b - a
      )[0]?.[0];
      const peakHour = peakHourNum
        ? `${peakHourNum}:00-${(parseInt(peakHourNum) + 1) % 24}:00`
        : "N/A";

      return {
        cat,
        zapCount: zaps.length,
        detectCount: detects.length,
        zapRate: detects.length > 0 ? zaps.length / detects.length : 0,
        favoriteZone,
        peakHour,
      };
    });

    setStats(catStats);
    setLoading(false);
  }

  if (loading) {
    return (
      <p style={{ color: "#888", fontFamily: "monospace", textAlign: "center" }}>
        Loading stats...
      </p>
    );
  }

  if (stats.length === 0) {
    return (
      <p style={{ color: "#888", fontFamily: "monospace", textAlign: "center" }}>
        No cats registered yet. Add cats in Settings.
      </p>
    );
  }

  const colors = ["#f72585", "#4361ee", "#4cc9f0", "#7209b7", "#f94144"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {stats.map((stat, i) => (
        <div
          key={stat.cat.id}
          style={{
            padding: 16,
            background: "#222",
            borderRadius: 8,
            borderLeft: `4px solid ${colors[i % colors.length]}`,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 12,
            }}
          >
            <span
              style={{
                color: colors[i % colors.length],
                fontWeight: "bold",
                fontSize: 16,
                fontFamily: "monospace",
              }}
            >
              {stat.cat.name}
            </span>
          </div>
          <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
            <div
              style={{
                textAlign: "center",
                padding: "8px 16px",
                background: `${colors[i % colors.length]}20`,
                borderRadius: 6,
              }}
            >
              <div
                style={{
                  color: colors[i % colors.length],
                  fontSize: 24,
                  fontWeight: "bold",
                  fontFamily: "monospace",
                }}
              >
                {stat.zapCount}
              </div>
              <div
                style={{ color: "#888", fontSize: 11, fontFamily: "monospace" }}
              >
                Zaps
              </div>
            </div>
            <div
              style={{
                textAlign: "center",
                padding: "8px 16px",
                background: "rgba(76, 201, 240, 0.1)",
                borderRadius: 6,
              }}
            >
              <div
                style={{
                  color: "#4cc9f0",
                  fontSize: 24,
                  fontWeight: "bold",
                  fontFamily: "monospace",
                }}
              >
                {stat.detectCount}
              </div>
              <div
                style={{ color: "#888", fontSize: 11, fontFamily: "monospace" }}
              >
                Detections
              </div>
            </div>
            <div
              style={{
                textAlign: "center",
                padding: "8px 16px",
                background: "rgba(255,255,255,0.05)",
                borderRadius: 6,
              }}
            >
              <div
                style={{
                  color: "#ccc",
                  fontSize: 24,
                  fontWeight: "bold",
                  fontFamily: "monospace",
                }}
              >
                {Math.round(stat.zapRate * 100)}%
              </div>
              <div
                style={{ color: "#888", fontSize: 11, fontFamily: "monospace" }}
              >
                Zap Rate
              </div>
            </div>
          </div>
          <div
            style={{ color: "#888", fontSize: 12, fontFamily: "monospace" }}
          >
            Peak mischief: {stat.peakHour} | Favorite zone: {stat.favoriteZone}
          </div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```
feat: add CatStats dashboard component
```

---

## Task 21: Frontend — Controls

**Files:**
- Create: `frontend/src/components/Controls.tsx`

- [ ] **Step 1: Write the component**

Create `frontend/src/components/Controls.tsx`:

```tsx
import { useEffect, useState } from "react";
import {
  getControlStatus,
  setArmed,
  manualFire,
} from "../api/client";

export default function Controls() {
  const [armed, setArmedState] = useState(true);
  const [pan, setPan] = useState(90);
  const [tilt, setTilt] = useState(90);
  const [firing, setFiring] = useState(false);

  useEffect(() => {
    getControlStatus()
      .then((s) => setArmedState(s.armed))
      .catch(console.error);
  }, []);

  async function toggleArm() {
    const newState = !armed;
    await setArmed(newState);
    setArmedState(newState);
  }

  async function handleFire() {
    setFiring(true);
    await manualFire(pan, tilt);
    setTimeout(() => setFiring(false), 500);
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 16,
      }}
    >
      {/* Arm/Disarm */}
      <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
        <h3
          style={{
            color: "#ccc",
            fontFamily: "monospace",
            fontSize: 14,
            marginTop: 0,
            marginBottom: 12,
          }}
        >
          System
        </h3>
        <button
          onClick={toggleArm}
          style={{
            width: "100%",
            padding: "12px 16px",
            background: armed ? "#f94144" : "#4cc9f0",
            color: armed ? "white" : "#1a1a2e",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontFamily: "monospace",
            fontSize: 14,
            fontWeight: "bold",
          }}
        >
          {armed ? "ARMED — Click to Disarm" : "DISARMED — Click to Arm"}
        </button>
      </div>

      {/* Manual Fire */}
      <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
        <h3
          style={{
            color: "#ccc",
            fontFamily: "monospace",
            fontSize: 14,
            marginTop: 0,
            marginBottom: 12,
          }}
        >
          Manual Fire
        </h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label
            style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}
          >
            Pan: {pan}°
            <input
              type="range"
              min={0}
              max={180}
              value={pan}
              onChange={(e) => setPan(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>
          <label
            style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}
          >
            Tilt: {tilt}°
            <input
              type="range"
              min={0}
              max={180}
              value={tilt}
              onChange={(e) => setTilt(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>
          <button
            onClick={handleFire}
            disabled={firing}
            style={{
              padding: "12px 16px",
              background: firing ? "#888" : "#f94144",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: firing ? "not-allowed" : "pointer",
              fontFamily: "monospace",
              fontSize: 14,
              fontWeight: "bold",
            }}
          >
            {firing ? "FIRING..." : "FIRE!"}
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```
feat: add Controls component with arm/disarm and manual fire
```

---

## Task 22: Frontend — Settings

**Files:**
- Create: `frontend/src/components/Settings.tsx`
- Create: `frontend/src/components/Calibration.tsx`

- [ ] **Step 1: Write Settings component**

Create `frontend/src/components/Settings.tsx`:

```tsx
import { useEffect, useState } from "react";
import type { Cat } from "../types";
import { getCats, createCat, deleteCat } from "../api/client";
import Calibration from "./Calibration";

interface SettingsProps {
  onUpdate: () => void;
}

export default function Settings({ onUpdate }: SettingsProps) {
  const [cats, setCats] = useState<Cat[]>([]);
  const [newCatName, setNewCatName] = useState("");

  useEffect(() => {
    getCats().then(setCats).catch(console.error);
  }, []);

  async function handleAddCat() {
    if (!newCatName.trim()) return;
    await createCat(newCatName.trim());
    setNewCatName("");
    const updated = await getCats();
    setCats(updated);
  }

  async function handleDeleteCat(id: string) {
    await deleteCat(id);
    const updated = await getCats();
    setCats(updated);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Cat Management */}
      <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
        <h3
          style={{
            color: "#ccc",
            fontFamily: "monospace",
            fontSize: 14,
            marginTop: 0,
          }}
        >
          Cat Management
        </h3>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <input
            type="text"
            placeholder="Cat name..."
            value={newCatName}
            onChange={(e) => setNewCatName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAddCat()}
            style={{
              padding: "8px 12px",
              background: "#333",
              border: "1px solid #555",
              borderRadius: 6,
              color: "#ccc",
              fontFamily: "monospace",
              flex: 1,
            }}
          />
          <button
            onClick={handleAddCat}
            disabled={!newCatName.trim()}
            style={{
              padding: "8px 16px",
              background: newCatName.trim() ? "#4cc9f0" : "#555",
              color: newCatName.trim() ? "#1a1a2e" : "#888",
              border: "none",
              borderRadius: 6,
              cursor: newCatName.trim() ? "pointer" : "not-allowed",
              fontFamily: "monospace",
            }}
          >
            Add Cat
          </button>
        </div>
        {cats.map((cat) => (
          <div
            key={cat.id}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "8px 12px",
              background: "#2a2a2a",
              borderRadius: 6,
              marginBottom: 4,
              fontFamily: "monospace",
              fontSize: 12,
            }}
          >
            <span style={{ color: "#ccc" }}>{cat.name}</span>
            <button
              onClick={() => handleDeleteCat(cat.id)}
              style={{
                padding: "4px 8px",
                background: "#f94144",
                color: "white",
                border: "none",
                borderRadius: 4,
                cursor: "pointer",
                fontSize: 11,
              }}
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      {/* Calibration */}
      <Calibration />
    </div>
  );
}
```

- [ ] **Step 2: Write Calibration component**

Create `frontend/src/components/Calibration.tsx`:

```tsx
import { useState } from "react";
import { addCalibrationPoint, clearCalibration } from "../api/client";

export default function Calibration() {
  const [pixelX, setPixelX] = useState(0.5);
  const [pixelY, setPixelY] = useState(0.5);
  const [panAngle, setPanAngle] = useState(90);
  const [tiltAngle, setTiltAngle] = useState(90);
  const [pointsCount, setPointsCount] = useState(0);

  async function handleAddPoint() {
    const result = await addCalibrationPoint(
      pixelX,
      pixelY,
      panAngle,
      tiltAngle
    );
    setPointsCount(result.points_count);
  }

  async function handleClear() {
    await clearCalibration();
    setPointsCount(0);
  }

  return (
    <div style={{ padding: 16, background: "#222", borderRadius: 8 }}>
      <h3
        style={{
          color: "#ccc",
          fontFamily: "monospace",
          fontSize: 14,
          marginTop: 0,
        }}
      >
        Servo Calibration ({pointsCount} points)
      </h3>
      <p style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}>
        Map pixel positions to servo angles. Click on the live feed to get pixel
        coordinates, then set the servo angles that aim at that point.
      </p>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
          marginBottom: 12,
        }}
      >
        <label
          style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}
        >
          Pixel X: {pixelX.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={pixelX}
            onChange={(e) => setPixelX(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <label
          style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}
        >
          Pixel Y: {pixelY.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={pixelY}
            onChange={(e) => setPixelY(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <label
          style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}
        >
          Pan Angle: {panAngle}°
          <input
            type="range"
            min={0}
            max={180}
            value={panAngle}
            onChange={(e) => setPanAngle(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
        <label
          style={{ color: "#888", fontFamily: "monospace", fontSize: 12 }}
        >
          Tilt Angle: {tiltAngle}°
          <input
            type="range"
            min={0}
            max={180}
            value={tiltAngle}
            onChange={(e) => setTiltAngle(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={handleAddPoint}
          style={{
            padding: "8px 16px",
            background: "#4cc9f0",
            color: "#1a1a2e",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontFamily: "monospace",
          }}
        >
          Add Calibration Point
        </button>
        <button
          onClick={handleClear}
          style={{
            padding: "8px 16px",
            background: "#f94144",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontFamily: "monospace",
          }}
        >
          Clear All
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```
feat: add Settings and Calibration components
```

---

## Task 23: Main Processing Loop

**Files:**
- Modify: `server/main.py`

- [ ] **Step 1: Add the background processing loop**

Update `server/main.py` to start the vision pipeline as a background task:

```python
import asyncio
import base64
import logging
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import settings
from server.models.database import init_db, get_db, create_event
from server.routers import zones, cats, events, control, stream, settings as settings_router
from server.routers.control import get_armed, get_calibration, get_actuator
from server.routers.stream import MJPEGStreamReader, broadcast_to_clients
from server.vision.detector import CatDetector
from server.vision.pipeline import VisionPipeline

logger = logging.getLogger(__name__)


async def run_vision_loop(app_state: dict):
    """Background task that runs the vision pipeline continuously."""
    detector = CatDetector(confidence_threshold=settings.confidence_threshold)
    reader = MJPEGStreamReader(settings.esp32_cam_url)

    try:
        await reader.connect()
        logger.info("Connected to ESP32-CAM stream")
    except Exception as e:
        logger.error(f"Failed to connect to camera: {e}")
        return

    frame_count = 0

    try:
        while app_state.get("running", True):
            frame = await reader.read_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % settings.frame_skip_n != 0:
                continue

            # Get current zones from DB
            async with get_db() as db:
                from server.models.database import get_zones
                current_zones = await get_zones(db)

            pipeline = VisionPipeline(
                detector=detector,
                actuator=get_actuator(),
                calibration=get_calibration(),
                zones=current_zones,
                armed=get_armed(),
                on_event=lambda evt: asyncio.create_task(_log_event(evt)),
            )

            result = await pipeline.process_frame(frame)

            # Encode frame as base64 JPEG for WebSocket broadcast
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            await broadcast_to_clients({
                "frame": frame_b64,
                "detections": result["detections"],
                "violations": result["violations"],
                "fired": result["fired"],
            })

    except asyncio.CancelledError:
        pass
    finally:
        await reader.close()


async def _log_event(evt: dict):
    """Log a pipeline event to the database and notify clients."""
    async with get_db() as db:
        await create_event(
            db,
            event_type=evt["type"],
            cat_name=evt.get("cat_name"),
            zone_name=evt.get("zone_name"),
            confidence=evt.get("confidence"),
            overlap=evt.get("overlap"),
            servo_pan=evt.get("servo_pan"),
            servo_tilt=evt.get("servo_tilt"),
        )

    # Broadcast event to WebSocket clients
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
    return {"status": "ok"}
```

- [ ] **Step 2: Verify server starts**

Run: `cd server && uvicorn server.main:app --reload`
Expected: Server starts. If no ESP32-CAM is connected, the vision loop logs "Failed to connect to camera" and the API still works.

- [ ] **Step 3: Commit**

```
feat: add main vision processing loop with WebSocket broadcast
```

---

## Task 24: End-to-End Integration Test

**Files:**
- No new files — manual testing

- [ ] **Step 1: Start the server**

```bash
cd server && uvicorn server.main:app --reload --port 8000
```

- [ ] **Step 2: Start the frontend**

```bash
cd frontend && npm run dev
```

- [ ] **Step 3: Verify all pages load**

Visit `http://localhost:5173`:
- Live tab shows (will show "Connecting..." without ESP32-CAM)
- Events tab shows empty event log
- Stats tab shows "No cats registered"
- Settings tab allows adding cats and calibration points

- [ ] **Step 4: Test API endpoints**

```bash
# Create a zone
curl -X POST http://localhost:8000/api/zones -H "Content-Type: application/json" -d '{"name":"Test Zone","polygon":[[0.1,0.1],[0.5,0.1],[0.5,0.5],[0.1,0.5]]}'

# Create a cat
curl -X POST http://localhost:8000/api/cats -H "Content-Type: application/json" -d '{"name":"Luna"}'

# Check control status
curl http://localhost:8000/api/control/status

# Get events
curl http://localhost:8000/api/events
```

- [ ] **Step 5: Run full test suite**

```bash
cd server && python -m pytest tests/ -v
```
Expected: All tests pass.

- [ ] **Step 6: Commit**

```
feat: complete CatZap v0.1.0 — ready for hardware integration
```
