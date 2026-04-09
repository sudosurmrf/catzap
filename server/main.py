import asyncio
import base64
import logging
import queue
import time
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import settings
from server.models.database import init_db, close_db, create_event, get_zones
from server.routers import zones, cats, events, control, stream, settings as settings_router, spatial
from server.routers import photos as photos_router, classifier as classifier_router
from server.routers.stream import MJPEGStreamReader, broadcast_to_clients, broadcast_event, store_latest_frame, has_feed_clients
from server.vision.detector import CatDetector
from server.vision.classifier import CatClassifier
from server.vision.zone_checker import check_zone_violations
from server.panorama.angle_math import pixel_to_angle
from server.panorama.tile_grid import TileGrid
from server.panorama.sweep_controller import SweepController, SweepState
from server.actuator.client import ActuatorClient
from server.spatial.depth_estimator import DepthEstimator
from server.spatial.room_model import RoomModel, FurnitureObject
from server.spatial.projection import angle_depth_to_room
from server.spatial.cat_tracker import CatTracker

logger = logging.getLogger(__name__)

# Shared state
_tile_grid: TileGrid | None = None
_sweep_controller: SweepController | None = None
_actuator: ActuatorClient | None = None
_room_model = None
_depth_estimator: DepthEstimator | None = None
_cat_tracker: CatTracker | None = None
_classifier: CatClassifier | None = None
_classify_frame_counter: int = 0
_cached_identities: dict[tuple[float, float], tuple[str, float]] = {}  # (cx, cy) -> (name, conf)


def get_tile_grid() -> TileGrid:
    return _tile_grid


def get_sweep_controller() -> SweepController:
    return _sweep_controller


def get_actuator() -> ActuatorClient:
    return _actuator


def get_room_model():
    return _room_model


def get_cat_tracker() -> CatTracker | None:
    return _cat_tracker


def reload_classifier():
    """Reload classifier weights after training."""
    global _classifier
    if _classifier is not None:
        weights_path = settings.classifier_weights_dir / "cat_classifier.pt"
        _classifier.load(weights_path)


async def run_vision_loop(app_state: dict):
    global _tile_grid, _sweep_controller, _actuator, _depth_estimator, _room_model, _cat_tracker, _classifier

    logger.info(f"Vision loop starting (dev_mode={settings.dev_mode})")
    logger.info(f"ESP32-CAM URL: {settings.esp32_cam_url}")
    logger.info(f"ESP32 Actuator URL: {settings.esp32_actuator_url}")

    try:
        detector = CatDetector(
            model_path=settings.detection_model,
            confidence_threshold=settings.confidence_threshold,
            imgsz=settings.detection_imgsz,
        )
        logger.info("YOLO detector loaded")
    except Exception as e:
        logger.error(f"Vision loop failed during init: {e}", exc_info=True)
        return

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
        tilt_min=settings.sweep_tilt_min,
        tilt_max=settings.sweep_tilt_max,
        speed=settings.sweep_speed,
        warning_duration=settings.warning_duration,
        tracking_duration=settings.tracking_duration,
        cooldown=settings.cooldown_default,
        reentry_warning=settings.reentry_warning,
        lock_on_grace=settings.lock_on_grace,
        dev_mode=settings.dev_mode,
    )

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
    _classifier = CatClassifier()
    weights_path = settings.classifier_weights_dir / "cat_classifier.pt"
    _classifier.load(weights_path)
    depth_frame_counter = 0
    needs_depth = False  # updated each frame based on whether 3D zones exist

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
        reader = MJPEGStreamReader(settings.esp32_cam_url)
        try:
            await reader.connect()
            logger.info("Connected to ESP32-CAM stream")
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return

    last_time = time.time()
    frame_count = 0
    cached_pano_b64: str | None = None

    # Detection smoother: holds recent detections for a grace window so
    # single-frame YOLO misses don't cause flicker.
    smoothed_dets: list[dict] = []   # [{bbox, confidence, _last_seen, _id}]
    det_hold_time = 0.4              # seconds to hold a detection after YOLO loses it
    det_match_threshold = 0.15       # max bbox-center distance (normalized) to match
    det_next_id = 0

    # ── Background inference state ──────────────────
    # YOLO + classification + depth run in a single persistent worker thread so
    # the main loop can keep streaming frames without blocking on inference.
    _infer_lock = threading.Lock()
    _latest_raw_dets: list[dict] = []        # written by inference thread
    _latest_depth: np.ndarray | None = None  # written by inference thread
    _new_results_ready = False               # flag: main loop should consume results
    _infer_queue: queue.Queue = queue.Queue(maxsize=1)  # drop-newest: only latest frame matters
    _infer_stop = threading.Event()

    def _inference_worker():
        """Persistent worker thread — processes one frame at a time from the queue."""
        nonlocal _latest_raw_dets, _latest_depth, _new_results_ready
        global _cached_identities

        while not _infer_stop.is_set():
            try:
                job = _infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            inf_frame, inf_servo_pan, inf_servo_tilt, do_depth, do_classify = job

            try:
                raw_dets = detector.detect(inf_frame)

                # Classify cats if due
                if do_classify and _classifier is not None and _classifier.model is not None:
                    new_identities: dict[tuple[float, float], tuple[str, float]] = {}
                    for det in raw_dets:
                        bbox = det["bbox"]
                        h, w = inf_frame.shape[:2]
                        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        x1 = max(0, int((bbox[0] - bw * 0.2) * w))
                        y1 = max(0, int((bbox[1] - bh * 0.2) * h))
                        x2 = min(w, int((bbox[2] + bw * 0.2) * w))
                        y2 = min(h, int((bbox[3] + bh * 0.2) * h))
                        crop = inf_frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            name, conf = _classifier.classify(crop)
                            if conf >= settings.classifier_confidence_threshold:
                                det["cat_name"] = name
                                det["cat_confidence"] = conf
                            elif conf >= settings.classifier_uncertain_min:
                                det["cat_name"] = "Unknown"
                                det["cat_confidence"] = conf
                            else:
                                det["cat_name"] = "Unknown"
                                det["cat_confidence"] = 0.0
                            cx = round((bbox[0] + bbox[2]) / 2, 1)
                            cy = round((bbox[1] + bbox[3]) / 2, 1)
                            new_identities[(cx, cy)] = (det.get("cat_name", "Unknown"), det.get("cat_confidence", 0.0))
                    _cached_identities = new_identities
                else:
                    for det in raw_dets:
                        bbox = det["bbox"]
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        best_key = None
                        best_d = float("inf")
                        for (kx, ky), (name, conf) in _cached_identities.items():
                            d = (cx - kx) ** 2 + (cy - ky) ** 2
                            if d < best_d:
                                best_d = d
                                best_key = (kx, ky)
                        if best_key is not None and best_d < 0.05:
                            name, conf = _cached_identities[best_key]
                            det["cat_name"] = name
                            det["cat_confidence"] = conf

                # Depth estimation
                depth_result = None
                if do_depth:
                    try:
                        depth_result = _depth_estimator.estimate(inf_frame)
                    except Exception as e:
                        logger.warning(f"Depth estimation failed: {e}")

                with _infer_lock:
                    _latest_raw_dets = raw_dets
                    _latest_depth = depth_result
                    _new_results_ready = True
            except Exception as e:
                logger.error(f"Inference error: {e}")
            finally:
                del inf_frame  # explicitly release frame copy

    # Start the single persistent inference worker
    _infer_thread = threading.Thread(target=_inference_worker, daemon=True)
    _infer_thread.start()

    try:
        while app_state.get("running", True):
            now = time.time()
            dt = now - last_time
            last_time = now

            # Advance state machine
            _sweep_controller.tick(dt)

            # Command servos to current position (sweep or tracking)
            if not settings.dev_mode:
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

            servo_pan = _sweep_controller.current_pan
            servo_tilt = _sweep_controller.current_tilt

            # Cache latest frame for depth estimation API
            store_latest_frame(frame, servo_pan, servo_tilt)

            # Update tile grid (smart refresh)
            col, row = _tile_grid.angle_to_tile_index(servo_pan, servo_tilt)
            if _tile_grid.should_refresh(col, row, frame, settings.tile_refresh_threshold):
                _tile_grid.update_tile(col, row, frame)

            # ── Submit inference job if worker is idle ──────
            global _classify_frame_counter, _cached_identities
            frame_count += 1
            if _infer_queue.empty() and frame_count % settings.frame_skip_n == 0:
                _classify_frame_counter += 1
                depth_frame_counter += 1
                do_classify = (_classify_frame_counter % settings.classify_every_n_frames == 0)
                do_depth = needs_depth and (depth_frame_counter % settings.depth_run_interval == 0)
                # Copy frame so the worker thread owns its data; drop if queue full
                try:
                    _infer_queue.put_nowait(
                        (frame.copy(), servo_pan, servo_tilt, do_depth, do_classify)
                    )
                except queue.Full:
                    pass  # worker still busy — skip this frame

            # ── Consume latest inference results if available ──
            raw_detections = None
            current_depth = None
            with _infer_lock:
                if _new_results_ready:
                    raw_detections = _latest_raw_dets
                    current_depth = _latest_depth
                    # Clear references so GC can reclaim detection/depth data
                    _latest_raw_dets = []
                    _latest_depth = None
                    _new_results_ready = False

            if raw_detections is not None:
                # ── Smooth detections across frames ──────────────
                matched_ids: set[int] = set()
                for rd in raw_detections:
                    rc = ((rd["bbox"][0] + rd["bbox"][2]) / 2,
                          (rd["bbox"][1] + rd["bbox"][3]) / 2)
                    best_match = None
                    best_dist = det_match_threshold
                    for sd in smoothed_dets:
                        sc = ((sd["bbox"][0] + sd["bbox"][2]) / 2,
                              (sd["bbox"][1] + sd["bbox"][3]) / 2)
                        d = ((rc[0] - sc[0]) ** 2 + (rc[1] - sc[1]) ** 2) ** 0.5
                        if d < best_dist and sd["_id"] not in matched_ids:
                            best_dist = d
                            best_match = sd
                    if best_match is not None:
                        best_match["bbox"] = rd["bbox"]
                        best_match["confidence"] = rd["confidence"]
                        best_match["_last_seen"] = now
                        if "cat_name" in rd:
                            best_match["cat_name"] = rd["cat_name"]
                        if "cat_confidence" in rd:
                            best_match["cat_confidence"] = rd["cat_confidence"]
                        matched_ids.add(best_match["_id"])
                    else:
                        smoothed_dets.append({
                            **rd,
                            "_last_seen": now,
                            "_id": det_next_id,
                        })
                        det_next_id += 1

            # Expire stale entries
            smoothed_dets = [
                sd for sd in smoothed_dets
                if (now - sd["_last_seen"]) < det_hold_time
            ]

            # Build the detections list for this frame
            detections = [
                {
                    **{k: v for k, v in sd.items() if not k.startswith("_")},
                    "track_id": sd["_id"],
                }
                for sd in smoothed_dets
            ]

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
                pan1, tilt1 = pixel_to_angle(bbox[0], bbox[1], servo_pan, servo_tilt)
                pan2, tilt2 = pixel_to_angle(bbox[2], bbox[3], servo_pan, servo_tilt)
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

            # Tell the controller about any detected cat — stops sweep, begins tracking
            if best_det is not None:
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
                        await asyncio.sleep(0.2)
                        success = await _actuator.fire()
                        fired = success

                    _sweep_controller.on_fire_complete()
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

            # Direction delta for dev mode arrow
            if settings.dev_mode and best_det is not None and _sweep_controller.state in (SweepState.WARNING, SweepState.FIRING, SweepState.TRACKING):
                direction_delta = _sweep_controller.get_direction_delta(best_cat_pan, best_cat_tilt)

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

            # Only encode and broadcast when WebSocket clients are connected
            if has_feed_clients():
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                # Only re-encode panorama when tiles have actually changed
                if _tile_grid._pano_dirty:
                    pano_jpeg = _tile_grid.get_panorama_jpeg()
                    cached_pano_b64 = base64.b64encode(pano_jpeg).decode("utf-8") if pano_jpeg else None

                await broadcast_to_clients({
                    "frame": frame_b64,
                    "panorama": cached_pano_b64,
                    "detections": detections,
                    "violations": all_violations,
                    "fired": fired,
                    "fire_target": fire_target,
                    "state": _sweep_controller.state.value,
                    "servo_pan": servo_pan,
                    "servo_tilt": servo_tilt,
                    "warning_remaining": _sweep_controller.get_warning_remaining(),
                    "direction_delta": direction_delta,
                    "occluded_cats": occluded_predictions,
                })

            await asyncio.sleep(settings.vision_loop_interval)

    except asyncio.CancelledError:
        pass
    finally:
        _infer_stop.set()
        _infer_thread.join(timeout=5)
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
    await broadcast_event(evt)


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
    except (asyncio.CancelledError, Exception):
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
app.include_router(spatial.router)
app.include_router(photos_router.router)
app.include_router(classifier_router.router)


@app.get("/health")
async def health():
    return {"status": "ok", "dev_mode": settings.dev_mode}
