import time
from enum import Enum

from server.actuator.client import ActuatorClient


class SweepState(str, Enum):
    SWEEPING = "SWEEPING"
    WARNING = "WARNING"
    FIRING = "FIRING"
    TRACKING = "TRACKING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class SweepController:
    """State machine controlling camera sweep, tracking, and fire behavior.

    State flow:
        SWEEPING ──cat detected──→ TRACKING ──cat in zone──→ WARNING ──timer──→ FIRING
            ↑                          ↑         │                                 │
            │                          │    cat leaves zone                        │
            │                          └─────────┘                          fire complete
            │                                                                      │
            └──────────── cat lost for grace period ◄──────────────────────────────┘
                                                              (FIRING → TRACKING → grace → SWEEPING)

    TRACKING is the base "engaged" state — camera follows any detected cat.
    WARNING/FIRING only activate when the tracked cat is inside a zone.
    """

    def __init__(
        self,
        actuator: ActuatorClient,
        pan_min: float = 30.0,
        pan_max: float = 150.0,
        tilt_min: float = 20.0,
        tilt_max: float = 70.0,
        speed: float = 2.5,
        warning_duration: float = 1.5,
        tracking_duration: float = 3.0,
        cooldown: float = 3.0,
        reentry_warning: float = 0.5,
        lock_on_grace: float = 1.0,
        dev_mode: bool = False,
    ):
        self.actuator = actuator
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.speed = speed
        self.warning_duration = warning_duration
        self.tracking_duration = tracking_duration
        self.cooldown = cooldown
        self.reentry_warning = reentry_warning
        self.lock_on_grace = lock_on_grace
        self.dev_mode = dev_mode

        # Boustrophedon sweep geometry
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
        self._warning_start: float = 0
        self._last_fire_time: float = 0
        self._target_pan: float = 0
        self._target_tilt: float = 0
        self._target_zone: str = ""
        self._was_in_zone: bool = False  # was the cat previously in a zone (for reentry_warning)
        self._grace_start: float = 0
        self._in_grace: bool = False

    def pause(self):
        if self.state == SweepState.SWEEPING:
            self.state = SweepState.PAUSED
            self.pause_queued = False
        elif self.state != SweepState.STOPPED:
            self.pause_queued = True

    def resume(self):
        if self.state == SweepState.PAUSED:
            self.state = SweepState.SWEEPING
        self.pause_queued = False

    def emergency_stop(self):
        self.state = SweepState.STOPPED
        self.armed = False
        self.pause_queued = False

    def clear_emergency_stop(self):
        if self.state == SweepState.STOPPED:
            self.state = SweepState.SWEEPING

    def tick(self, dt: float):
        """Advance the state machine by dt seconds."""
        if self.state in (SweepState.PAUSED, SweepState.STOPPED):
            return
        now = time.time()

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

        elif self.state in (SweepState.TRACKING, SweepState.WARNING, SweepState.FIRING):
            # Move camera toward the cat's last known position
            if self._target_pan != 0 or self._target_tilt != 0:
                track_speed = self.speed * 4
                pan_delta = self._target_pan - self.current_pan
                tilt_delta = self._target_tilt - self.current_tilt
                max_step = track_speed * dt

                if abs(pan_delta) > max_step:
                    self.current_pan += max_step if pan_delta > 0 else -max_step
                else:
                    self.current_pan = self._target_pan

                if abs(tilt_delta) > max_step:
                    self.current_tilt += max_step if tilt_delta > 0 else -max_step
                else:
                    self.current_tilt = self._target_tilt

            # Grace expiry: cat lost for too long → resume sweep
            if self._in_grace and (now - self._grace_start) >= self.lock_on_grace:
                self._in_grace = False
                self._was_in_zone = False
                if self.pause_queued:
                    self.state = SweepState.PAUSED
                    self.pause_queued = False
                else:
                    self.state = SweepState.SWEEPING

    # ── Called by the vision loop ──────────────────────

    def on_cat_detected(self, cat_pan: float, cat_tilt: float):
        """Called every frame a cat is visible, regardless of zone status.
        Stops sweeping and begins/continues tracking."""
        self._target_pan = cat_pan
        self._target_tilt = cat_tilt
        self._in_grace = False

        if self.state == SweepState.SWEEPING:
            self.state = SweepState.TRACKING

    def on_cat_in_zone(self, cat_pan: float, cat_tilt: float, zone_name: str):
        """Called when the tracked cat is inside a forbidden zone."""
        self._target_pan = cat_pan
        self._target_tilt = cat_tilt
        self._target_zone = zone_name

        if self.state == SweepState.TRACKING:
            self.state = SweepState.WARNING
            self._warning_start = time.time()
            if self._was_in_zone:
                # Returning to zone after leaving — use shorter warning
                self._warning_start = time.time() - (self.warning_duration - self.reentry_warning)

        elif self.state == SweepState.WARNING:
            now = time.time()
            warn_time = self.warning_duration
            if (now - self._warning_start) >= warn_time:
                if now - self._last_fire_time >= self.cooldown:
                    self.state = SweepState.FIRING

        elif self.state == SweepState.FIRING:
            pass  # target already updated, will fire this frame

    def on_cat_not_in_zone(self):
        """Called when cat is visible but not in any zone."""
        if self.state == SweepState.WARNING:
            # Cat left the zone but is still visible — drop back to tracking
            self.state = SweepState.TRACKING
            self._was_in_zone = True

    def on_no_cat_detected(self):
        """Called when no cat is detected this frame."""
        if self.state in (SweepState.TRACKING, SweepState.WARNING):
            if not self._in_grace:
                self._in_grace = True
                self._grace_start = time.time()

    def on_fire_complete(self):
        """Called after the solenoid has fired."""
        self._last_fire_time = time.time()
        self.state = SweepState.TRACKING
        self._was_in_zone = True  # just fired, so use short reentry warning if cat returns

    def should_fire(self) -> bool:
        """Check if the system should fire right now."""
        return self.state == SweepState.FIRING and self.armed

    def get_direction_delta(self, target_pan: float, target_tilt: float) -> dict:
        """Get the angle delta between current position and target (for dev mode arrow)."""
        return {
            "pan": target_pan - self.current_pan,
            "tilt": target_tilt - self.current_tilt,
        }

    def get_target(self) -> tuple[float, float, str]:
        """Get the current target (cat position) and zone name."""
        return (self._target_pan, self._target_tilt, self._target_zone)

    def get_warning_remaining(self) -> float:
        """Get seconds remaining in warning period."""
        if self.state != SweepState.WARNING:
            return 0
        elapsed = time.time() - self._warning_start
        return max(0, self.warning_duration - elapsed)

    def set_virtual_angle(self, pan: float, tilt: float):
        """Set the virtual angle (dev mode - manual control)."""
        self.current_pan = max(self.pan_min, min(self.pan_max, pan))
        self.current_tilt = max(self.tilt_min, min(self.tilt_max, tilt))
