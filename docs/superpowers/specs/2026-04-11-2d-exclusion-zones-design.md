# 2D Exclusion Zones + Engagement State Machine — Design

**Date:** 2026-04-11
**Supersedes:** `2026-04-08-3d-spatial-awareness-design.md`
**Status:** Draft for review

---

## Goal

Replace the 3D MiDaS-based exclusion zone system with a simpler 2D polygon system, and change the fire control loop from a stateless "detect-and-fire" pass to an active engagement state machine that pauses the panoramic sweep, re-aims the camera at the detected cat between shots, and returns to sweeping only after the cat has cleared all zones for 3 seconds.

Two motivations:

1. **Performance.** Removing the MiDaS depth inference pass and the 3D room model drops a per-frame CPU inference stage and frees torch/transformers (if nothing else uses them), improving FPS and reducing CPU/RAM baseline.
2. **Correctness.** The 3D system was bolted on top of a pipeline that already uses only 2D bbox-vs-polygon overlap at the fire decision (`server/vision/pipeline.py:46`). The depth data was never actually consumed to make firing decisions, so deleting the 3D scaffolding is subtractive — it does not change what the fire decision sees. Separately, the fire *control loop* is being changed intentionally: today's stateless "detect-and-fire" becomes a two-state engagement machine that pauses the sweep, re-aims between shots, and enforces a post-engagement grace window.

## Non-goals

- Changing the panoramic sweep controller's sweep logic itself (we use its existing pause/resume hooks, nothing more).
- Changing the calibration / pixel-to-angle conversion (out of scope; a separate calibration rewrite is in flight).
- Adding new UI. The existing mode indicator on the live feed is sufficient; SWEEPING/ENGAGING badge work is deferred.
- Preserving any existing zone or furniture database rows. The `zones` and `furniture` tables will be dropped and recreated.

## Architecture after the rewrite

```
Frame → CatDetector.detect() → EngagementController.on_detections(bboxes)
                                        │
                                        │  state-dependent dispatch
                                        ▼
           ┌───────────────────────────────────────────────────┐
           │                                                   │
           │   SWEEPING                       ENGAGING         │
           │   ───────                        ────────         │
           │   sweep runs                     sweep paused     │
           │   if any bbox in zone:           on each frame:   │
           │     pause sweep                    find bboxes    │
           │     → ENGAGING                     in any zone    │
           │                                    if any:        │
           │                                      target =     │
           │                                        closest    │
           │                                        to aim     │
           │                                      if last_shot │
           │                                        ≥ 2000ms:  │
           │                                        fire       │
           │                                    else:          │
           │                                      if empty ≥   │
           │                                        3000ms:    │
           │                                        resume     │
           │                                        sweep      │
           │                                        → SWEEPING │
           └───────────────────────────────────────────────────┘
```

Only two states: `SWEEPING` and `ENGAGING`. The 3000ms grace window on cat-absent frames subsumes the notion of a separate post-engagement cooldown — any third state would just duplicate that timer.

## Data model

### `zones` table (wipe and recreate)

| column               | type    | notes                                                 |
|----------------------|---------|-------------------------------------------------------|
| `id`                 | UUID PK |                                                       |
| `name`               | TEXT    | user-chosen label                                     |
| `polygon`            | JSONB   | array of `{pan: float, tilt: float}` in servo degrees |
| `overlap_threshold`  | REAL    | fraction of bbox area that must fall inside polygon (default 0.2) |
| `enabled`            | BOOL    | default true                                          |
| `created_at`         | TIMESTAMP |                                                     |

Dropped columns (from the 3D schema): `mode`, `room_polygon`, `height_min`, `height_max`, `furniture_id`, `cooldown_seconds`. The `cooldown_seconds` column goes because the global `min_shot_interval_ms` replaces its role.

### `furniture` table (wipe and recreate, simplified)

| column       | type    | notes                                                     |
|--------------|---------|-----------------------------------------------------------|
| `id`         | UUID PK |                                                           |
| `name`       | TEXT    | user-chosen label, shown on the panorama                  |
| `polygon`    | JSONB   | array of `{pan: float, tilt: float}` in servo degrees     |
| `created_at` | TIMESTAMP |                                                         |

Dropped columns: `base_polygon` (renamed to `polygon`), `height_min`, `height_max`, `depth_anchored`.

**Furniture semantics:** named, angle-space polygon that is drawn once and locked to its recorded `(pan, tilt)` coordinates forever. Purely visual annotation (helps the user orient when placing zones). It does not trigger firing, does not affect zone logic, and is not referenced by any other table. Because storage is in angle space, the polygon remains in its correct spatial position as the sweep crosses panorama tile boundaries — no tile-relative math required. To "move" a furniture item, the user deletes and redraws it.

### Config changes (`server/config.py`)

Removed (all MiDaS / room-model / occlusion settings):
`midas_model`, `depth_run_interval`, `depth_blend_alpha`, `depth_change_threshold`, `heightmap_resolution`, `room_width_cm`, `room_depth_cm`, `room_height_cm`, `camera_height_cm`, `occlusion_timeout`, `occlusion_grace_frames`.

Added:
- `min_shot_interval_ms: int = 2000` — minimum elapsed time between two successive `aim_and_fire` calls while in ENGAGING.
- `engagement_grace_ms: int = 3000` — time cats must be continuously absent from all zones before ENGAGING transitions back to SWEEPING.

## Engagement state machine

### States

- **SWEEPING** — `sweep_controller` runs normally. The engagement controller passively watches detections.
- **ENGAGING** — `sweep_controller` is paused. The engagement controller owns servo commands until the grace window expires.

### Transitions

| From | To | Condition |
|------|-----|-----------|
| SWEEPING | ENGAGING | any detected cat bbox overlaps any enabled zone at or above that zone's `overlap_threshold` |
| ENGAGING | SWEEPING | no bbox has met any zone's overlap threshold for a continuous `engagement_grace_ms` (default 3000ms) |

### ENGAGING frame loop

On every detection frame while in ENGAGING:

1. Compute `cats_in_zone = [bbox for bbox in detections if any zone.overlaps(bbox, zone.overlap_threshold)]`.
2. If `cats_in_zone` is non-empty:
   - Record `last_cat_in_zone_time = now`.
   - Select `target = closest(cats_in_zone, key=distance_from_current_servo_aim)`.
   - If `now - last_shot_time >= min_shot_interval_ms`:
     - `pan, tilt = calibration.pixel_to_angle(target.center_x, target.center_y)`
     - `actuator.aim_and_fire(pan, tilt)`
     - `last_shot_time = now`
3. Else (no cats in any zone this frame):
   - If `now - last_cat_in_zone_time >= engagement_grace_ms`, transition to SWEEPING (resume sweep controller).

### Target selection rationale

"Closest bbox to current servo aim" is used instead of "highest confidence" or "first in list" because it provides tracking continuity: when you were aiming at cat A last shot and cat A is still in the zone, A's center is naturally the nearest bbox to the current aim, so you keep engaging A. If A leaves and B is still there, the selection pivots to B without code changes. It also minimizes servo travel between shots.

### Cooldown / grace timer interaction

- `min_shot_interval_ms` (2000ms) governs shot rate *within* ENGAGING. The servo holds its last fire position between shots — there is no continuous tracking aim.
- `engagement_grace_ms` (3000ms) governs the ENGAGING → SWEEPING transition. Each frame with at least one cat in a zone resets this timer. The first frame where zero cats are in zones starts it counting down.
- A cat that briefly exits a zone and returns within the grace window does not cause a sweep resume; the engagement loop continues and the next shot will fire whenever `min_shot_interval_ms` has elapsed since the last shot.
- Because the grace window (3000ms) is longer than the shot interval (2000ms), a cat that stays in zone will typically receive shots at roughly 2Hz, bounded by servo traversal time.

## Files deleted

**Backend (server/):**

- `spatial/depth_estimator.py`
- `spatial/room_model.py`
- `spatial/projection.py`
- `spatial/cat_tracker.py`
- `spatial/__init__.py`
- `spatial/` (directory itself, once empty)
- `routers/spatial.py`
- `tests/test_depth_estimator.py`
- `tests/test_volumetric_zones.py`
- `tests/test_room_model.py`
- `tests/test_projection.py`

## Files edited

**Backend:**

- `server/main.py` — remove MiDaS initialization (around line 195), room model init (196), cat tracker init (202), all depth inference calls (~319, 512, 518, 602–608), and the spatial router mount. Add initialization of the new `EngagementController` and pass it to the vision pipeline.
- `server/vision/zone_checker.py` — delete `check_3d_zone_violation()`, delete the `auto_3d` / `manual_3d` mode fallback branches. Keep the 2D `check_zone_violations(bbox, zones)` function exactly as-is. Drop the `mode` argument handling.
- `server/vision/pipeline.py` — replace the inline "detect → fire → log" logic inside `process_frame` with a call to `engagement_controller.on_detections(detections)`. The controller owns state, not the pipeline.
- `server/vision/engagement_controller.py` — **new file.** Owns the state machine, grace window, shot interval, target selection, sweep pause/resume, and `aim_and_fire` calls. Takes `(sweep_controller, actuator, calibration, zones_provider, config)` in its constructor and exposes one method: `on_detections(detections: list[BBox]) -> EngagementStatus`.
- `server/panorama/sweep_controller.py` — verify `pause()` / `resume()` API exists and works as expected. No new code unless the API is missing.
- `server/models/database.py` — drop `zones` table, drop `furniture` table, recreate both with the simplified schemas above. Remove all 3D-related columns from initialization SQL.
- `server/models/schemas.py` (or equivalent Pydantic schema file) — update `Zone` and `Furniture` models to match. Remove `ZoneMode` enum if present.
- `server/routers/zones.py` — drop references to `mode`, `room_polygon`, `height_min`, `height_max`, `furniture_id`, `cooldown_seconds`. Keep CRUD endpoints.
- `server/routers/furniture.py` (if it exists separately, otherwise whatever router handles furniture) — simplified CRUD: create, list, delete. No update endpoint needed (move = delete + create).
- `server/config.py` — remove the MiDaS/room-model/occlusion settings listed above; add `min_shot_interval_ms` and `engagement_grace_ms`.
- `server/tests/test_zone_checker.py` — remove 3D-mode test cases; keep the 2D overlap tests.
- `server/tests/test_pipeline.py` — update to mock the new `EngagementController` (or exercise it end-to-end with fake time).
- `server/tests/test_engagement_controller.py` — **new file.** Unit tests for the state machine (see Testing below).
- `requirements.txt` / `pyproject.toml` — remove `torch` and `transformers` **only if** a repo-wide grep confirms nothing else imports them. If anything else does, leave the dependency and note it in the plan.

**Frontend (frontend/src/):**

- `components/ZoneConfigPanel.tsx` — remove height slider, 3D transform sliders (`scaleX`, `scaleY`, `skewX`, `skewY`, `slantX`, `slantY`), auto-estimate-height button, mode selector. Keep name input, overlap threshold, enabled toggle, save button. Remove per-zone `cooldown_seconds` input.
- `components/ZoneEditor.tsx` — remove 3D prism overlay rendering, remove mode-dependent label logic. Keep 2D polygon drawing + Ramer-Douglas-Peucker simplification (it's useful for 2D polygons too).
- `components/FurniturePanel.tsx` (if it exists, otherwise the relevant furniture UI) — simplified form: draw a polygon on the panorama, enter a name, save. No height/depth controls. No edit/move UI; delete-and-redraw is the workflow.
- `types/index.ts` — strip `mode`, `room_polygon`, `height_min`, `height_max`, `furniture_id`, `cooldown_seconds` from the `Zone` interface. Strip `base_polygon`, `height_min`, `height_max`, `depth_anchored` from `Furniture`; rename `base_polygon` → `polygon`. Delete the `ZoneTransform` interface entirely.
- `api/client.ts` — remove `estimateHeight`, any `spatial/*` endpoint calls, remove 3D fields from zone/furniture payloads.
- `components/Settings.tsx` — remove any UI branches referencing 3D mode, MiDaS, or depth calibration.
- `components/CalibrationWizard.tsx` — remove any depth-calibration step if present.
- `components/App.tsx` — remove any top-level state or route referencing 3D mode.

## Testing

### Unit tests (automated)

`server/tests/test_engagement_controller.py` — new file covering the state machine:

1. `test_starts_in_sweeping` — fresh controller is in SWEEPING state.
2. `test_transitions_to_engaging_on_cat_in_zone` — a single detection with a bbox overlapping a zone triggers `sweep_controller.pause()` and transitions to ENGAGING.
3. `test_fires_on_first_detection_in_engaging` — the first frame in ENGAGING with a cat in zone calls `actuator.aim_and_fire`.
4. `test_respects_min_shot_interval` — subsequent frames within `min_shot_interval_ms` of the last shot do not call `aim_and_fire`.
5. `test_fires_again_after_interval` — a frame after `min_shot_interval_ms` has elapsed calls `aim_and_fire` again.
6. `test_target_selection_picks_closest_to_aim` — with two bboxes in zone, the one whose center is closer to the current servo aim is selected.
7. `test_grace_window_keeps_engaging_on_brief_absence` — a single frame with no cats in zone does not transition back; subsequent frames with cats reset the grace timer.
8. `test_grace_window_expires_returns_to_sweeping` — `engagement_grace_ms` of continuous absent frames triggers `sweep_controller.resume()` and transitions to SWEEPING.
9. `test_overlap_threshold_respected` — a bbox that overlaps a zone below its threshold does not trigger engagement.

Tests use a fake clock (inject a `time_source` callable) so they run deterministically without real waits.

`server/tests/test_zone_checker.py` — existing 2D tests should pass unchanged after the 3D branches are removed. Verify by running the suite after the edit.

### Manual smoke test (on the rig)

1. Start the server and confirm it boots without errors (no torch import explosions, no missing-column SQL errors).
2. Open the frontend, draw a zone on the panorama, save it. Verify the database row contains only the new columns.
3. Draw a furniture outline, name it, save it. Verify it renders on the panorama and persists across a page refresh.
4. Arm the system. Walk a cat plushie into a zone. Verify:
   - Sweep pauses
   - First shot fires within ~1 detection frame
   - Subsequent shots fire at approximately 2-second intervals while the plushie stays in zone
   - Sweep resumes ~3 seconds after the plushie is removed from all zones
5. Walk the plushie near but not inside a zone; verify no fire.
6. Compare FPS before and after the change (existing FPS indicator on the UI should show a clear increase once MiDaS is gone).

## Rollout

1. Land backend changes, verify unit tests pass.
2. Land frontend changes, verify type-check passes and dev server builds.
3. User wipes their local database (or the server does it on startup if a schema-version check is in place) and redraws zones + furniture manually. This is explicitly chosen: preserving existing 3D zone rows is not worth the migration cost.
4. Rig smoke test per the manual checklist above.
5. If `torch` and `transformers` are confirmed unused elsewhere, remove them from dependencies in a follow-up commit. Do not bundle this with the main change — a dependency removal is easy to roll back separately if something unexpected breaks.

## Open risks

- **Sweep controller pause/resume semantics.** The plan assumes `sweep_controller.pause()` and `sweep_controller.resume()` exist and are safe to call from the detection thread. `test_sweep_pause.py` suggests they do, but the implementation plan must verify before writing the engagement controller.
- **`aim_and_fire` blocking vs async.** If the actuator client's `aim_and_fire` returns immediately while the servo is still moving, calling it on every frame during the 2000ms interval would queue commands faster than the hardware can execute them. The engagement controller must gate on the software timer regardless, but the plan must confirm whether `aim_and_fire` is blocking or fire-and-forget.
- **Per-zone `overlap_threshold` defaults.** Existing rows are being wiped, so the default for newly drawn zones is the only value users will ever see until they tweak it. 0.2 (20% of bbox area) is a reasonable starting point but may need tuning on the rig.
- **Torch removal.** If anything else in the repo imports torch (CLIP, classifier, etc.), it stays. The plan step for dependency removal is conditional on a clean grep.
