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
from server.routers.control import get_armed, get_calibration, get_actuator
from server.routers.stream import MJPEGStreamReader, broadcast_to_clients
from server.vision.detector import CatDetector
from server.vision.zone_checker import check_zone_violations

logger = logging.getLogger(__name__)

# Shared cooldown state so it persists across frames
_cooldowns: dict[str, float] = {}


async def run_vision_loop(app_state: dict):
    """Background task that runs the vision pipeline continuously."""
    detector = CatDetector(confidence_threshold=settings.confidence_threshold)

    # Choose camera source based on dev_mode
    cap = None
    reader = None

    if settings.dev_mode:
        logger.info("DEV MODE: Using local webcam")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        logger.info("Webcam opened successfully")
    else:
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
            # Grab frame from webcam or ESP32-CAM
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

            # Detect cats
            detections = detector.detect(frame)

            # Check zones
            current_zones = await get_zones()
            all_violations = []
            fired = False
            fire_target = None

            for det in detections:
                bbox = det["bbox"]
                violations = check_zone_violations(bbox, current_zones)
                all_violations.extend(violations)

                if not violations or not get_armed():
                    continue

                for violation in violations:
                    zone_id = violation["zone_id"]
                    zone = next((z for z in current_zones if z["id"] == zone_id), None)
                    if not zone:
                        continue

                    cooldown = zone.get("cooldown_seconds", settings.cooldown_default)
                    last_fire = _cooldowns.get(zone_id, 0)

                    if time.time() - last_fire < cooldown:
                        continue

                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    pan, tilt = get_calibration().pixel_to_angle(center_x, center_y)

                    if settings.dev_mode:
                        # Simulate fire — don't send HTTP to ESP32
                        logger.info(f"DEV ZAP! Cat in {violation['zone_name']} — would aim pan={pan:.1f} tilt={tilt:.1f}")
                        _cooldowns[zone_id] = time.time()
                        fired = True
                        fire_target = {"x": center_x, "y": center_y, "zone": violation["zone_name"]}
                    else:
                        success = await get_actuator().aim_and_fire(pan, tilt)
                        if success:
                            _cooldowns[zone_id] = time.time()
                            fired = True

                    if fired:
                        asyncio.create_task(_log_event({
                            "type": "ZAP",
                            "cat_name": det.get("cat_name"),
                            "zone_name": violation["zone_name"],
                            "confidence": det["confidence"],
                            "overlap": violation["overlap"],
                            "servo_pan": pan,
                            "servo_tilt": tilt,
                        }))
                        break

                if fired:
                    break

            # Encode frame and broadcast
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            await broadcast_to_clients({
                "frame": frame_b64,
                "detections": detections,
                "violations": all_violations,
                "fired": fired,
                "fire_target": fire_target,
            })

            # Small sleep to avoid pegging the CPU
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
