import time
from enum import Enum
from typing import Callable

from server.actuator.client import ActuatorClient


class SweepState(str, Enum):
    SWEEPING = "SWEEPING"
    TRACKING = "TRACKING"
    ENGAGING = "ENGAGING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class SweepController:
    """Camera sweep + cat engagement state machine.

    States:
        SWEEPING  — normal boustrophedon sweep runs.
        TRACKING  — at least one cat is visible somewhere in frame but none are
                    inside an exclusion zone. The sweep is paused and the
                    camera smoothly pursues the latest bbox center so the rig
                    is ready to fire the instant the cat crosses a zone
                    boundary. `should_fire()` returns False here — this is
                    pursuit only, no shots.
        ENGAGING  — the tracked cat's bbox overlaps an exclusion zone. Same
                    smooth pursuit as TRACKING, plus `should_fire()` returns
                    True once `min_shot_interval_ms` has elapsed since the
                    last shot. A cat leaving the zone (but still visible)
                    drops the controller back to TRACKING.
        PAUSED    — manual pause (from arming UI); tick() is a no-op.
        STOPPED   — emergency stop; everything halts until explicitly cleared.

    Grace window: if no cat has been reported at all (neither in nor out of a
    zone) for `engagement_grace_ms`, the controller returns from TRACKING or
    ENGAGING back to SWEEPING. Any single cat-visible frame resets the grace
    timer — a cat that hops out of a zone but remains visible keeps the rig
    in the pursuit states indefinitely.
    """

    def __init__(
        self,
        actuator: ActuatorClient,
        pan_min: float = 30.0,
        pan_max: float = 150.0,
        tilt_min: float = 20.0,
        tilt_max: float = 70.0,
        speed: float = 2.5,
        min_shot_interval_ms: int = 2000,
        engagement_grace_ms: int = 3000,
        dev_mode: bool = False,
        time_source: Callable[[], float] = time.time,
    ):
        self.actuator = actuator
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.speed = speed
        self.min_shot_interval_ms = min_shot_interval_ms
        self.engagement_grace_ms = engagement_grace_ms
        self.dev_mode = dev_mode
        self._time = time_source

        # Boustrophedon sweep geometry (unchanged from the old implementation).
        self._tilt_top = tilt_min + (tilt_max - tilt_min) * 0.5
        self._tilt_bottom = tilt_max

        pan_range = pan_max - pan_min
        tilt_transition = abs(self._tilt_bottom - self._tilt_top)
        self._phase_durations = [
            pan_range / self.speed,
            tilt_transition / self.speed,
            pan_range / self.speed,
            tilt_transition / self.speed,
        ]
        self._cycle_duration = sum(self._phase_durations)
        self._sweep_t = 0.0

        self.current_pan = pan_min
        self.current_tilt = self._tilt_top

        self.state = SweepState.SWEEPING
        self.armed = True
        self.pause_queued = False
        self._target_pan: float = 0.0
        self._target_tilt: float = 0.0
        self._target_zone: str = ""
        self._last_shot_time: float = 0.0  # epoch seconds; 0.0 means "never fired"
        self._last_cat_seen_time: float = 0.0  # updated every frame a cat is reported (in or out of zone)
        self._last_target_update_time: float = 0.0  # when we last accepted a new target (jumped the camera)

    def set_bounds(
        self, pan_min: float, pan_max: float, tilt_min: float, tilt_max: float
    ) -> None:
        """Replace the sweep pan/tilt range and recompute phase durations.
        Called by the calibration router after extent capture."""
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self._tilt_top = tilt_min + (tilt_max - tilt_min) * 0.5
        self._tilt_bottom = tilt_max
        pan_range = pan_max - pan_min
        tilt_transition = abs(self._tilt_bottom - self._tilt_top)
        self._phase_durations = [
            pan_range / self.speed,
            tilt_transition / self.speed,
            pan_range / self.speed,
            tilt_transition / self.speed,
        ]
        self._cycle_duration = sum(self._phase_durations)
        self._sweep_t = 0.0

    def pause(self) -> None:
        if self.state == SweepState.SWEEPING:
            self.state = SweepState.PAUSED
            self.pause_queued = False
        elif self.state != SweepState.STOPPED:
            self.pause_queued = True

    def resume(self) -> None:
        if self.state == SweepState.PAUSED:
            self.state = SweepState.SWEEPING
        self.pause_queued = False

    def emergency_stop(self) -> None:
        self.state = SweepState.STOPPED
        self.armed = False
        self.pause_queued = False

    def clear_emergency_stop(self) -> None:
        if self.state == SweepState.STOPPED:
            self.state = SweepState.SWEEPING

    def tick(self, dt: float) -> None:
        """Advance the state machine by dt seconds."""
        if self.state in (SweepState.PAUSED, SweepState.STOPPED):
            return
        now = self._time()

        if self.state == SweepState.SWEEPING:
            self._sweep_t += dt
            if self._sweep_t >= self._cycle_duration:
                self._sweep_t -= self._cycle_duration

            t = self._sweep_t
            d0, d1, d2, d3 = self._phase_durations

            if t < d0:
                frac = t / d0
                self.current_pan = self.pan_min + frac * (self.pan_max - self.pan_min)
                self.current_tilt = self._tilt_top
            elif t < d0 + d1:
                frac = (t - d0) / d1
                self.current_pan = self.pan_max
                self.current_tilt = self._tilt_top + frac * (self._tilt_bottom - self._tilt_top)
            elif t < d0 + d1 + d2:
                frac = (t - d0 - d1) / d2
                self.current_pan = self.pan_max - frac * (self.pan_max - self.pan_min)
                self.current_tilt = self._tilt_bottom
            else:
                frac = (t - d0 - d1 - d2) / d3
                self.current_pan = self.pan_min
                self.current_tilt = self._tilt_bottom - frac * (self._tilt_bottom - self._tilt_top)

        elif self.state in (SweepState.TRACKING, SweepState.ENGAGING):
            # No smooth pursuit — current_pan/tilt are pinned to the latest
            # accepted target (set directly by on_cat_detected / on_cat_in_zone).
            # The camera jumps on each accepted update and holds still
            # between updates. This is a discrete "goto + wait" cycle, not
            # continuous pursuit, so the vision loop can keep processing
            # frames at full rate while the servo is stationary.

            # Grace expiry: no cat has been reported at all for
            # engagement_grace_ms. Return to sweep (or to queued pause).
            grace_elapsed_ms = (now - self._last_cat_seen_time) * 1000.0
            if grace_elapsed_ms >= self.engagement_grace_ms:
                if self.pause_queued:
                    self.state = SweepState.PAUSED
                    self.pause_queued = False
                else:
                    self.state = SweepState.SWEEPING

    def _clamp_target(self, cat_pan: float, cat_tilt: float) -> tuple[float, float]:
        """Clamp a tracking target to the operational envelope so the
        internal tracking state never drifts outside reachable bounds."""
        return (
            max(self.pan_min, min(self.pan_max, cat_pan)),
            max(self.tilt_min, min(self.tilt_max, cat_tilt)),
        )

    # -- Called by the vision loop --

    def on_cat_in_zone(self, cat_pan: float, cat_tilt: float, zone_name: str) -> None:
        """Called every frame that at least one cat's bbox center is inside
        an exclusion zone. Updates the grace timer unconditionally. Target
        updates are gated by `min_shot_interval_ms` — within the wait
        window, the call is a no-op for target/state (grace timer still
        ticks). The first update from SWEEPING is always accepted because
        `_last_target_update_time` starts at 0.0."""
        now = self._time()
        self._last_cat_seen_time = now
        if self.state in (SweepState.TRACKING, SweepState.ENGAGING):
            elapsed_ms = (now - self._last_target_update_time) * 1000.0
            if elapsed_ms < self.min_shot_interval_ms:
                return
        clamped_pan, clamped_tilt = self._clamp_target(cat_pan, cat_tilt)
        self._target_pan = clamped_pan
        self._target_tilt = clamped_tilt
        self._target_zone = zone_name
        # Jump the camera directly — no smooth pursuit. The main loop's
        # dedupe check will pick up the change and issue a goto to the servo
        # on the next iteration.
        self.current_pan = clamped_pan
        self.current_tilt = clamped_tilt
        self._last_target_update_time = now
        if self.state in (SweepState.SWEEPING, SweepState.TRACKING):
            self.state = SweepState.ENGAGING

    def on_cat_detected(self, cat_pan: float, cat_tilt: float) -> None:
        """Called every frame that at least one cat is visible but none are
        inside an exclusion zone. Same discrete 2s-cycle gating as
        on_cat_in_zone — grace timer updates every call, target only
        updates once per `min_shot_interval_ms`.

        Transitions on accept:
            SWEEPING → TRACKING   (start stalking)
            ENGAGING → TRACKING   (cat hopped out of a zone but is still visible)
            TRACKING → TRACKING   (just update the target)
        """
        now = self._time()
        self._last_cat_seen_time = now
        if self.state in (SweepState.TRACKING, SweepState.ENGAGING):
            elapsed_ms = (now - self._last_target_update_time) * 1000.0
            if elapsed_ms < self.min_shot_interval_ms:
                return
        clamped_pan, clamped_tilt = self._clamp_target(cat_pan, cat_tilt)
        self._target_pan = clamped_pan
        self._target_tilt = clamped_tilt
        self._target_zone = ""
        self.current_pan = clamped_pan
        self.current_tilt = clamped_tilt
        self._last_target_update_time = now
        if self.state in (SweepState.SWEEPING, SweepState.ENGAGING):
            self.state = SweepState.TRACKING

    def on_cat_in_deadband(self) -> None:
        """Called when a tracked cat is already inside the tracking deadband
        — the camera should NOT move, but we still want to keep the grace
        timer alive and stay in TRACKING state. Skips both the
        min_shot_interval gate and the target update so current_pan/tilt
        stay frozen at wherever the rig last locked on."""
        now = self._time()
        self._last_cat_seen_time = now
        if self.state == SweepState.SWEEPING:
            self.state = SweepState.TRACKING

    def on_no_cat_detected(self) -> None:
        """Called on frames where no cat is visible at all. Does not
        immediately transition — the grace timer in tick() handles that.
        Harmless no-op in SWEEPING state."""
        # Deliberate no-op at the event level. State changes happen in tick().
        pass

    def should_fire(self) -> bool:
        """Returns True iff we are in ENGAGING, armed, and at least
        `min_shot_interval_ms` have elapsed since the last shot. The vision
        loop calls this every frame after `on_cat_in_zone`; if True, the
        loop executes the aim_and_fire and then calls `mark_shot_fired`."""
        if self.state != SweepState.ENGAGING or not self.armed:
            return False
        elapsed_ms = (self._time() - self._last_shot_time) * 1000.0
        return elapsed_ms >= self.min_shot_interval_ms

    def mark_shot_fired(self) -> None:
        """Record the timestamp of a shot that the vision loop just
        executed. Required for the min_shot_interval_ms gate to work."""
        self._last_shot_time = self._time()

    def on_fire_complete(
        self,
        fired_pan: float | None = None,
        fired_tilt: float | None = None,
    ) -> None:
        """Called after the solenoid has fired. The vision loop commands the
        servo directly during the fire flow, so we sync current_pan/tilt to
        the commanded pose. Also resets the grace timer — the fire flow
        spends ~2-3 seconds of wall time on goto + sleep + fire, and
        without the reset that elapsed wall-clock time would eat into the
        3s grace window, possibly dropping us back to SWEEPING immediately
        after a successful shot."""
        if fired_pan is not None:
            self.current_pan = max(0.0, min(180.0, fired_pan))
        if fired_tilt is not None:
            self.current_tilt = max(0.0, min(180.0, fired_tilt))
        self._target_pan = self.current_pan
        self._target_tilt = self.current_tilt
        self._last_cat_seen_time = self._time()

    def get_direction_delta(self, target_pan: float, target_tilt: float) -> dict:
        """Angle delta between current position and target (for the dev-mode
        arrow overlay)."""
        return {
            "pan": target_pan - self.current_pan,
            "tilt": target_tilt - self.current_tilt,
        }

    def get_target(self) -> tuple[float, float, str]:
        """Current target pan/tilt and the name of the zone that triggered
        engagement (empty string if none)."""
        return (self._target_pan, self._target_tilt, self._target_zone)

    def set_virtual_angle(self, pan: float, tilt: float) -> None:
        """Set the virtual angle (dev mode - manual control)."""
        self.current_pan = max(self.pan_min, min(self.pan_max, pan))
        self.current_tilt = max(self.tilt_min, min(self.tilt_max, tilt))

    def set_pose_calibration(self, pan: float, tilt: float) -> None:
        """Set the servo pose during calibration mode. Clamps only to
        hardware range [0, 180] — not sweep bounds, because calibration may
        want to explore outside the normal sweep region."""
        if tilt > 75.0 or tilt < 15.0 or pan < 5.0 or pan > 175.0:
            import logging, traceback
            _log = logging.getLogger(__name__)
            _log.warning(
                f"Suspicious set_pose_calibration: pan={pan:.2f} tilt={tilt:.2f}\n"
                f"Stack trace of caller:\n{''.join(traceback.format_stack(limit=8)[:-1])}"
            )
        self.current_pan = max(0.0, min(180.0, pan))
        self.current_tilt = max(0.0, min(180.0, tilt))
