import asyncio
import base64
import logging
import queue
import time
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import settings
from server.models.database import init_db, close_db, create_event, get_zones, get_zones_version
from server.routers import zones, cats, events, control, stream, settings as settings_router, furniture
from server.routers import photos as photos_router, classifier as classifier_router
from server.routers import calibration as calibration_router
from server.routers.stream import MJPEGStreamReader, broadcast_to_clients, broadcast_event, store_latest_frame, has_feed_clients
from server.vision.detector import CatDetector
from server.vision.classifier import CatClassifier
from server.vision.zone_checker import check_zone_violations
from server.panorama.angle_math import calibrated_pixel_to_angle
from server.panorama.tile_grid import TileGrid
from server.panorama.sweep_controller import SweepController, SweepState
from server.actuator.client import ActuatorClient, snap_to_servo_step
from server.actuator.calibration import (
    load_calibration,
    load_rig_settings,
    set_active_calibration,
    set_rig_settings,
)

logger = logging.getLogger(__name__)

# Shared state
_tile_grid: TileGrid | None = None
_sweep_controller: SweepController | None = None
_actuator: ActuatorClient | None = None
_classifier: CatClassifier | None = None
_classify_frame_counter: int = 0
_cached_identities: dict[tuple[float, float], tuple[str, float]] = {}  # (cx, cy) -> (name, conf)


def get_tile_grid() -> TileGrid:
    return _tile_grid


def get_sweep_controller() -> SweepController:
    return _sweep_controller


def get_actuator() -> ActuatorClient:
    return _actuator


def reload_classifier():
    """Reload classifier weights after training."""
    global _classifier
    if _classifier is not None:
        weights_path = settings.classifier_weights_dir / "cat_classifier.pt"
        _classifier.load(weights_path)


async def run_vision_loop(app_state: dict):
    global _tile_grid, _sweep_controller, _actuator, _classifier

    logger.info(f"Vision loop starting (dev_mode={settings.dev_mode})")
    logger.info(f"ESP32-CAM URL: {settings.esp32_cam_url}")
    logger.info(f"ESP32 Actuator URL: {settings.esp32_actuator_url}")

    # Connect to camera FIRST, before loading heavy models
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
        max_connect_retries = 10
        for attempt in range(1, max_connect_retries + 1):
            try:
                await reader.connect()
                logger.info("Connected to ESP32-CAM stream")
                break
            except Exception as e:
                logger.warning(f"Camera connection attempt {attempt}/{max_connect_retries} failed: {e}", exc_info=True)
                if attempt == max_connect_retries:
                    logger.error("Could not connect to ESP32-CAM after all retries — vision loop exiting")
                    return
                await asyncio.sleep(3)

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

    # Load rig settings FIRST so bounds (if calibrated) are applied to both
    # the tile grid and sweep controller at init time.
    _rig_settings = load_rig_settings()
    set_rig_settings(_rig_settings)
    if _rig_settings.tilt_jog_inverted or _rig_settings.pan_jog_inverted:
        logger.info(
            f"Loaded rig settings: tilt_inverted={_rig_settings.tilt_jog_inverted}, "
            f"pan_inverted={_rig_settings.pan_jog_inverted}"
        )

    # Use calibrated extent bounds if present, else fall back to config.
    # Bounds are populated by the extent-capture phase of calibration and
    # persisted in rig_settings.json so they survive restarts.
    pan_min = _rig_settings.pan_min if _rig_settings.pan_min is not None else settings.sweep_pan_min
    pan_max = _rig_settings.pan_max if _rig_settings.pan_max is not None else settings.sweep_pan_max
    tilt_min = _rig_settings.tilt_min if _rig_settings.tilt_min is not None else settings.sweep_tilt_min
    tilt_max = _rig_settings.tilt_max if _rig_settings.tilt_max is not None else settings.sweep_tilt_max
    if _rig_settings.pan_min is not None:
        logger.info(
            f"Using calibrated extent bounds: "
            f"pan=[{pan_min:.0f}..{pan_max:.0f}], "
            f"tilt=[{tilt_min:.0f}..{tilt_max:.0f}]"
        )
    else:
        logger.info("No calibrated extent bounds — using config defaults")

    _tile_grid = TileGrid(
        pan_min=pan_min,
        pan_max=pan_max,
        tilt_min=tilt_min,
        tilt_max=tilt_max,
        fov_h=settings.fov_horizontal,
        fov_v=settings.fov_vertical,
        tile_overlap=settings.tile_overlap,
    )

    # Load persisted calibration (from a prior successful run) if present.
    # The new Calibration schema doesn't carry a bilinear fit anymore — it
    # just records the extent corners, tile grid dimensions, and verification
    # residuals. The runtime behavior is driven by rig_settings bounds (loaded
    # above) and the analytic pinhole FOV; the Calibration record is mostly
    # for display/history and future per-tile refinement.
    _persisted_cal = load_calibration()
    if _persisted_cal is not None:
        set_active_calibration(_persisted_cal)
        logger.info(
            f"Loaded calibration from {_persisted_cal.created_at}: "
            f"bounds pan=[{_persisted_cal.bounds.pan_min:.0f}..{_persisted_cal.bounds.pan_max:.0f}] "
            f"tilt=[{_persisted_cal.bounds.tilt_min:.0f}..{_persisted_cal.bounds.tilt_max:.0f}], "
            f"tiles={_persisted_cal.tile_cols}×{_persisted_cal.tile_rows}, "
            f"verification_passed={_persisted_cal.verification_passed}"
        )
    else:
        logger.info("No persisted calibration found — run aim calibration to define extent")

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

    # Per-cat identity classifier is disabled. The YOLO detector still
    # detects "is this a cat" — we just skip the second-stage MobileNet
    # classifier that maps a crop to a specific cat name. Frees ~50-200ms
    # of CPU per inference cycle, which is the main reason detection felt
    # sluggish whenever a cat was in view. Re-enable by uncommenting and
    # restoring the cached_identities propagation in the inference worker.
    _classifier = None

    last_time = time.time()
    frame_count = 0
    cached_pano_b64: str | None = None

    # Zone cache: pulled from DB only when the version counter bumps. Avoids
    # an async pool acquire on every iteration of the vision loop.
    cached_zones: list = []
    cached_zones_version: int = -1

    # Panorama tile refresh decimator: per-frame stitching is wasted work for
    # a UI element nobody watches at full FPS. Throttle to ~5 Hz wall clock.
    last_tile_refresh_time: float = 0.0

    # Track the last pan/tilt we actually sent to the ESP32 (as the int the
    # firmware will write). The main loop only re-sends when this changes —
    # prevents the 10 Hz goto() spam from racing with direct gotos (jog,
    # fire-path) on the shared httpx client, which was causing the pan servo
    # to ring.
    _last_sent_pan_int: int | None = None
    _last_sent_tilt_int: int | None = None

    # In-flight goto guard. The dedupe block below spawns gotos as fire-and-
    # forget tasks; if the network is degraded each task takes ~2s to time
    # out and they pile up unbounded on the actuator's traverse_lock. Holding
    # a single reference and skipping new spawns while one is still pending
    # caps the queue depth at 1 — under failure we drop intermediate poses
    # rather than queue them, which is the right tradeoff for a tracking rig.
    _goto_task: asyncio.Task | None = None

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
    _latest_inf_pan: float = 0.0             # servo pan at time the published frame was captured
    _latest_inf_tilt: float = 0.0            # servo tilt at time the published frame was captured
    _new_results_ready = False               # flag: main loop should consume results
    _infer_queue: queue.Queue = queue.Queue(maxsize=1)  # drop-newest: only latest frame matters
    _infer_stop = threading.Event()

    def _inference_worker():
        """Persistent worker thread — processes one frame at a time from the queue."""
        nonlocal _latest_raw_dets, _latest_inf_pan, _latest_inf_tilt, _new_results_ready
        global _cached_identities

        while not _infer_stop.is_set():
            try:
                job = _infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            inf_frame, inf_servo_pan, inf_servo_tilt, do_classify = job

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

                with _infer_lock:
                    _latest_raw_dets = raw_dets
                    _latest_inf_pan = inf_servo_pan
                    _latest_inf_tilt = inf_servo_tilt
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

            # Command servos to current position (sweep or tracking). Dedupe
            # against the last-sent SERVO-SNAPPED angle — ActuatorClient snaps
            # to the mechanical 2° resolution before sending, so the dedupe
            # must use the same snap function or we'd skip sends whenever the
            # intended float moves past a 2°-boundary.
            if not settings.dev_mode:
                pan_int = snap_to_servo_step(_sweep_controller.current_pan)
                tilt_int = snap_to_servo_step(_sweep_controller.current_tilt)
                if pan_int != _last_sent_pan_int or tilt_int != _last_sent_tilt_int:
                    # Fire-and-forget rate-limited traverse: routes through
                    # _actuator.goto() (no direct=True), which sub-steps the
                    # move in 6°/30ms increments. For sweep this is identical
                    # to the legacy 1° behavior because each sweep iteration
                    # only advances current_pan by a fraction of a degree —
                    # the dedupe rarely crosses a 1° boundary, and when it
                    # does the traverse runs as a single sub-step with no
                    # sleep. For tracking jumps the moderate pacing lets the
                    # camera capture multiple detection frames during the
                    # move so the targeting can refine instead of overshoot.
                    #
                    # Skip-while-pending guard: if the previous goto is still
                    # in-flight (slow link, big tracking traverse, etc.), drop
                    # this iteration's goto rather than queueing another task.
                    # The next iteration will reattempt with the latest pose.
                    # Caps in-flight goto count at exactly 1 — no backlog.
                    if _goto_task is None or _goto_task.done():
                        _goto_task = asyncio.create_task(
                            _actuator.goto(
                                _sweep_controller.current_pan,
                                _sweep_controller.current_tilt,
                            )
                        )
                        _last_sent_pan_int = pan_int
                        _last_sent_tilt_int = tilt_int

            # Grab frame. In production we also receive the original JPEG
            # bytes alongside the decoded numpy array — the websocket
            # broadcast can re-emit those without paying for a re-encode.
            jpg_bytes: bytes | None = None
            if settings.dev_mode:
                ret, frame = cap.read()
                if not ret or frame is None:
                    await asyncio.sleep(0.05)
                    continue
            else:
                result = await reader.read_frame()
                if result is None:
                    await asyncio.sleep(0.1)
                    continue
                frame, jpg_bytes = result

            servo_pan = _sweep_controller.current_pan
            servo_tilt = _sweep_controller.current_tilt

            # Cache latest frame for depth estimation API
            store_latest_frame(frame, servo_pan, servo_tilt)

            # Paint the current frame onto the panorama canvas at its servo
            # angle. Throttled to ~5 Hz — the panorama is a background
            # reference, not a real-time feed.
            if (now - last_tile_refresh_time) >= 0.2:
                _tile_grid.paint_frame(servo_pan, servo_tilt, frame)
                last_tile_refresh_time = now

            # ── Submit inference job if worker is idle ──────
            global _classify_frame_counter, _cached_identities
            frame_count += 1
            if _infer_queue.empty() and frame_count % settings.frame_skip_n == 0:
                _classify_frame_counter += 1
                do_classify = (_classify_frame_counter % settings.classify_every_n_frames == 0)
                # Copy frame so the worker thread owns its data; drop if queue full
                try:
                    _infer_queue.put_nowait(
                        (frame.copy(), servo_pan, servo_tilt, do_classify)
                    )
                except queue.Full:
                    pass  # worker still busy — skip this frame

            # ── Consume latest inference results if available ──
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
                        best_match["_pose_pan"] = inf_pose_pan
                        best_match["_pose_tilt"] = inf_pose_tilt
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
                            "_pose_pan": inf_pose_pan,
                            "_pose_tilt": inf_pose_tilt,
                        })
                        det_next_id += 1

            # Expire stale entries
            smoothed_dets = [
                sd for sd in smoothed_dets
                if (now - sd["_last_seen"]) < det_hold_time
            ]

            # Build the detections list for this frame. We forward _pose_pan/_pose_tilt
            # under dunder keys so the angle-conversion loop can use each detection's
            # capture-time pose (eliminates stale-pose targeting drift).
            detections = [
                {
                    **{k: v for k, v in sd.items() if not k.startswith("_")},
                    "track_id": sd["_id"],
                    "__pose_pan": sd.get("_pose_pan", servo_pan),
                    "__pose_tilt": sd.get("_pose_tilt", servo_tilt),
                }
                for sd in smoothed_dets
            ]

            # Convert detections to angle-space and check zones.
            #
            # Two-state pursuit: the controller follows ANY visible cat
            # (TRACKING) and additionally fires when the tracked cat enters an
            # exclusion zone (ENGAGING). The selection pass below picks two
            # targets per frame:
            #   - engagement_target: cat in a zone, closest to current aim
            #   - tracking_target:   any cat, closest to current aim
            # We dispatch to the controller based on which (if any) exist.
            # Zones rarely change at runtime — only refresh when the version
            # counter (bumped on every create/update/delete) advances. This
            # eliminates the per-frame async pool acquire that was costing
            # ~1-2ms on every iteration even when nothing had changed.
            current_version = get_zones_version()
            if current_version != cached_zones_version:
                cached_zones = await get_zones()
                cached_zones_version = current_version
            current_zones = cached_zones
            all_violations = []
            fired = False
            fire_target = None
            direction_delta = None

            engagement_target = None  # (cat_pan, cat_tilt, zone_name, det, overlap)
            tracking_target = None    # (cat_pan, cat_tilt, det, bbox_cx, bbox_cy)
            current_aim_pan = _sweep_controller.current_pan
            current_aim_tilt = _sweep_controller.current_tilt
            best_engagement_dist = float("inf")
            best_tracking_dist = float("inf")

            for det in detections:
                bbox = det["bbox"]
                det_pose_pan = det["__pose_pan"]
                det_pose_tilt = det["__pose_tilt"]
                pan1, tilt1 = calibrated_pixel_to_angle(bbox[0], bbox[1], det_pose_pan, det_pose_tilt)
                pan2, tilt2 = calibrated_pixel_to_angle(bbox[2], bbox[3], det_pose_pan, det_pose_tilt)
                angle_bbox = [pan1, tilt1, pan2, tilt2]
                cat_pan = (pan1 + pan2) / 2
                cat_tilt = (tilt1 + tilt2) / 2

                # Bbox center in normalized [0..1] frame coordinates — used
                # later for the tracking deadband check (skip the camera
                # move if the cat is already near frame center).
                bbox_cx = (bbox[0] + bbox[2]) / 2
                bbox_cy = (bbox[1] + bbox[3]) / 2

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

                # Every detected cat is a tracking candidate.
                if dist < best_tracking_dist:
                    best_tracking_dist = dist
                    tracking_target = (clamped_pan, clamped_tilt, det, bbox_cx, bbox_cy)

                # Cats inside zones are also engagement candidates.
                violations = check_zone_violations(angle_bbox, current_zones)
                all_violations.extend(violations)
                if violations and dist < best_engagement_dist:
                    best_engagement_dist = dist
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
                        await _actuator.goto(tgt_pan, tgt_tilt, direct=True)
                        _last_sent_pan_int = snap_to_servo_step(tgt_pan)
                        _last_sent_tilt_int = snap_to_servo_step(tgt_tilt)
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
            elif tracking_target is not None:
                tgt_pan, tgt_tilt, _, tgt_cx, tgt_cy = tracking_target
                # Deadband check: if the cat's bbox center is already inside
                # the deadband radius around frame center (0.5, 0.5), the rig
                # is "close enough" — hold position rather than chasing the
                # last few pixels of YOLO bbox jitter. on_cat_in_deadband
                # still resets the grace timer so we don't drop back to
                # SWEEPING while the cat sits centered in front of us.
                dx = tgt_cx - 0.5
                dy = tgt_cy - 0.5
                pixel_dist = (dx * dx + dy * dy) ** 0.5
                if pixel_dist <= settings.tracking_deadband_frac:
                    _sweep_controller.on_cat_in_deadband()
                else:
                    _sweep_controller.on_cat_detected(tgt_pan, tgt_tilt)
            else:
                _sweep_controller.on_no_cat_detected()

            # Direction delta for dev mode arrow — shown during TRACKING and ENGAGING
            if settings.dev_mode and _sweep_controller.state in (SweepState.TRACKING, SweepState.ENGAGING):
                if engagement_target is not None:
                    dd_pan, dd_tilt, _, _, _ = engagement_target
                    direction_delta = _sweep_controller.get_direction_delta(dd_pan, dd_tilt)
                elif tracking_target is not None:
                    dd_pan, dd_tilt, _, _, _ = tracking_target
                    direction_delta = _sweep_controller.get_direction_delta(dd_pan, dd_tilt)

            # Only encode and broadcast when WebSocket clients are connected.
            # In production, jpg_bytes already came in from the ESP32-CAM —
            # we re-emit them verbatim instead of decode→numpy→re-encode,
            # which was burning ~15-30ms per frame for no reason. Dev mode
            # uses cv2.VideoCapture (raw numpy), so it still has to encode.
            if has_feed_clients():
                if jpg_bytes is not None:
                    frame_b64 = base64.b64encode(jpg_bytes).decode("utf-8")
                else:
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_b64 = base64.b64encode(buffer).decode("utf-8")

                # Only re-encode panorama when tiles have actually changed
                if _tile_grid._pano_dirty:
                    pano_jpeg = _tile_grid.get_panorama_jpeg()
                    cached_pano_b64 = base64.b64encode(pano_jpeg).decode("utf-8") if pano_jpeg else None

                # Expose the PHYSICAL (servo-snapped) pose to clients, not the
                # float intent. This is the decoupling boundary: internally we
                # track continuous float state for precision, but what gets
                # broadcast to the UI is what the servo has actually achieved.
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

            # Sleep only the time we have left in the target frame budget.
            # The previous unconditional sleep added latency every iteration
            # regardless of how long the iteration already took, capping the
            # achievable rate even when the inner work was already slow.
            elapsed = time.time() - now
            remaining = settings.vision_loop_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            else:
                await asyncio.sleep(0)

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
app.include_router(furniture.router)
app.include_router(photos_router.router)
app.include_router(classifier_router.router)
app.include_router(calibration_router.router)


@app.get("/health")
async def health():
    return {"status": "ok", "dev_mode": settings.dev_mode}
