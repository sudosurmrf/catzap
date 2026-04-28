import asyncio
import logging
import math
import traceback

import httpx

logger = logging.getLogger(__name__)

# The rig's SG51-class servo has a deadband of ~0.5-1° (5-10µs of the
# ~11µs-per-degree pulse width), so 1° is the finest meaningful command
# resolution. Arduino `Servo.write()` on the firmware side takes an
# integer degree value — there's no sub-degree API via `write()`. So the
# wire contract is: always an integer, always a whole degree. Every
# pose-sending method passes its values through `snap_to_servo_step`
# before transmitting so the wire value matches what the hardware can
# actually execute, and fractional degrees from float interpolation
# math never leak out of the client.
SERVO_STEP_DEG = 1


def snap_to_servo_step(x: float) -> int:
    """Round a float angle to the nearest whole degree, with ties
    rounding down to the lower integer.

    Examples:
        105.0 → 105   (exact integer, no change)
        104.5 → 104   (tie, rounds down)
        104.51 → 105  (just above tie, rounds up)
        104.49 → 104  (just below tie, rounds down)
        43.8 → 44
        42.2 → 42
    """
    integer_part = math.floor(x)
    frac = x - integer_part
    if frac <= 0.5:
        return int(integer_part)
    else:
        return int(integer_part + 1)


# Safety thresholds: these define "plausible" ranges for our rig. When a goto
# command falls outside this window, we log a loud warning with a full stack
# trace so we can trace which caller sent the suspicious value. This is pure
# instrumentation — it does NOT clamp or reject the command.
_SUSPICIOUS_PAN_MIN = 5.0
_SUSPICIOUS_PAN_MAX = 175.0
_SUSPICIOUS_TILT_MIN = 15.0
_SUSPICIOUS_TILT_MAX = 75.0


def _log_if_suspicious(endpoint: str, pan: float, tilt: float, sent_pan: float, sent_tilt: float) -> None:
    """Emit a loud warning (with stack trace) when a pose falls outside the
    plausible range for our rig. Used to hunt for a rogue caller that's
    driving the tilt servo to dangerous values."""
    pan_bad = pan < _SUSPICIOUS_PAN_MIN or pan > _SUSPICIOUS_PAN_MAX
    tilt_bad = tilt < _SUSPICIOUS_TILT_MIN or tilt > _SUSPICIOUS_TILT_MAX
    sent_tilt_bad = sent_tilt < _SUSPICIOUS_TILT_MIN or sent_tilt > _SUSPICIOUS_TILT_MAX
    if pan_bad or tilt_bad or sent_tilt_bad:
        stack = "".join(traceback.format_stack(limit=8)[:-1])
        logger.warning(
            f"🚨 Suspicious {endpoint}: "
            f"input=(pan={pan:.2f}, tilt={tilt:.2f}) "
            f"sent=(pan={sent_pan:.2f}, tilt={sent_tilt:.2f})\n"
            f"Stack trace of caller:\n{stack}"
        )


def _compensate(pan: float, tilt: float) -> tuple[float, float]:
    """Apply the quadratic pan→tilt correction from rig settings.

    Lazy-imported so this module doesn't need a hard dependency on the
    calibration module (avoids import cycles and makes the client testable
    in isolation).
    """
    try:
        from server.actuator.calibration import compensate_pan_tilt_coupling
        return compensate_pan_tilt_coupling(pan, tilt)
    except Exception as e:
        # Never fail a servo command because of a calibration import issue
        logger.warning(f"Coupling compensation unavailable, sending raw pose: {e}")
        return (pan, tilt)


def _clamp_to_bounds(pan: float, tilt: float) -> tuple[float, float]:
    """Clamp a pose to the rig's extent bounds from RigSettings.

    Lazy-imported for the same reason as `_compensate`. Returns the pose
    unchanged if any bound is None (no calibration has set the envelope
    yet) or if the calibration module isn't available. This is the single
    gate that enforces the "camera cannot hit obstacles" safety rule —
    every outgoing pose command passes through it.
    """
    try:
        from server.actuator.calibration import clamp_to_extent_bounds
        return clamp_to_extent_bounds(pan, tilt)
    except Exception as e:
        logger.warning(f"Extent-bounds clamping unavailable, sending raw pose: {e}")
        return (pan, tilt)


# ── Rate-limited traverse parameters ──────────────────────
# Every pose command is broken into `_TRAVERSE_STEP_DEG`-per-sub-step
# increments on BOTH axes, separated by `_TRAVERSE_STEP_DELAY_S` seconds
# between sub-sends. A 1° move produces a single sub-step (no sleep);
# a 60° pan move at the current 6°/30ms params produces 10 sub-steps
# with 9 sleeps ≈ 600 ms wall time, which is fast enough for the camera
# to track a moving cat without the servo physically blowing past the
# target before the next detection frame arrives.
#
# The 30 ms delay sits above the 25 ms aliasing zone where the command
# rate would beat against the servo's 50 Hz PWM frame rate and create
# uneven effective update intervals. At 30 ms (≈33 Hz commands), each
# new target gets ~1.5 full PWM frames to settle before the next arrives.
#
# Earlier this was 1°/50ms (~20°/sec) which spread current draw beautifully
# but turned a 60° tracking jump into a 3-second background task — long
# enough that the asyncio event loop and main vision loop both starved
# behind it. The 6°/30ms params (~200°/sec) are still rate-limited (still
# spreads current vs. one giant servo command), but complete the same
# move in ~600 ms instead of 3 s. The skip-while-pending guard in the
# vision loop caps in-flight tracking gotos at 1, so the actuator can
# never get backlogged regardless of how aggressive these params are.
#
# Sweep behavior is unchanged in practice: each sweep iteration only
# advances current_pan by ~0.04°, so the int snap rarely crosses a 1°
# boundary, and when it does the delta is 1 (single sub-step, no sleep).
#
# Effective traverse speed ≈ _TRAVERSE_STEP_DEG / _TRAVERSE_STEP_DELAY_S
# plus per-request WiFi latency on whichever axis has the larger delta.
_TRAVERSE_STEP_DEG = 6.0
_TRAVERSE_STEP_DELAY_S = 0.030

# Hard cap on sub-sends per traverse. Without it, a stale `_last_sent_pan`
# (e.g. left over from a network blip while the sweep advanced 60°+) would
# produce a 10-20 command burst at 33 Hz on the wire, which both
# competes with the next iteration's dispatch for the lock and visibly
# rapid-fires the servo through dozens of intermediate poses. Capping at
# 12 keeps normal tracking jumps unaffected (a 60° jump = 10 sub-steps,
# under the cap) while bounding worst-case wall time at ~360 ms.
_TRAVERSE_MAX_STEPS = 12


class ActuatorClient:
    """HTTP client for communicating with the ESP32 DEVKITV1 actuator.

    All pose-sending methods (`aim`, `goto`) automatically apply the rig's
    pan-axis tilt compensation from `RigSettings` before transmitting. The
    caller passes the *intended world pose* and the client translates to
    the servo command that physically achieves it. Zero caller awareness
    required — wire a new call site and it's automatically compensated.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # The ESP32's WebServer sends `Connection: close` per response, so
        # httpx can't actually reuse sockets — but its default keepalive
        # pool will still hand back a half-closed socket whose next request
        # surfaces as `ConnectError: All connection attempts failed`.
        # Disabling the keepalive pool forces a fresh connection per request
        # and eliminates that failure mode.
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=0),
        )
        # Rate-limited traverse state: track the last successfully-sent
        # snapped pose so we can interpolate large moves. None until the
        # first send completes, at which point the first command goes
        # directly without interpolation (no known start point yet).
        self._last_sent_pan: int | None = None
        self._last_sent_tilt: int | None = None
        # Serializes concurrent aim/goto calls so their interpolated sub-
        # sends don't interleave on the wire. Required because both the
        # sweep main loop and the firing flow can call into the client
        # from different coroutines.
        self._traverse_lock = asyncio.Lock()

    async def _send_pose(self, endpoint: str, snap_pan: int, snap_tilt: int) -> bool:
        """Send a single snapped pose to the ESP32 at the given endpoint path
        (either 'aim' or 'goto'). Returns True on HTTP 200, False on network
        error or non-200. Factored out so the traverse loop can reuse it."""
        try:
            response = await self._client.post(
                f"{self.base_url}/{endpoint}",
                json={"pan": snap_pan, "tilt": snap_tilt},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            # httpx.ConnectTimeout/ReadTimeout/ConnectError have empty str(e)
            # by default — log the class name so the failure mode is visible.
            logger.error(f"Failed to send {endpoint} command: {type(e).__name__}: {e}")
            return False

    def _prepare_pose(self, endpoint: str, pan: float, tilt: float) -> tuple[int, int]:
        """Apply compensation, double-clamp, suspicious-pose logging, and
        integer snap to a (pan, tilt) pair.  Returns (target_pan, target_tilt)
        as snapped integers ready to send on the wire.

        Factored out so both `_traverse` and `goto(direct=True)` share
        identical safety logic without duplication.
        """
        # Clamp FIRST so the tilt-compensation quadratic is evaluated within
        # its valid fit domain. The poly was fit on pan values in the rig's
        # operational range, and extrapolating beyond those bounds causes the
        # dp² term to blow up (20-30° corrections at pan=184 instead of the
        # ~4° max the fit actually represents). Pre-clamping pan to the safe
        # range means the poly only runs on inputs it was calibrated for.
        pre_pan, pre_tilt = _clamp_to_bounds(pan, tilt)
        # Apply compensation on the pre-clamped pose.
        send_pan, send_tilt = _compensate(pre_pan, pre_tilt)
        # Clamp AGAIN after compensation — compensation can shift tilt by a
        # few degrees and push it outside tilt_min/max even when the input
        # was in-range. This second clamp guarantees the wire value respects
        # the full envelope regardless of what the poly returned.
        send_pan, send_tilt = _clamp_to_bounds(send_pan, send_tilt)
        _log_if_suspicious(endpoint, pan, tilt, send_pan, send_tilt)
        return snap_to_servo_step(send_pan), snap_to_servo_step(send_tilt)

    async def _traverse(self, endpoint: str, pan: float, tilt: float) -> bool:
        """Rate-limited traverse to a target pose on both axes.

        Applies compensation and snapping to the target, then walks pan
        and tilt together from the last-sent values to the target in
        `_TRAVERSE_STEP_DEG` integer-degree increments with
        `_TRAVERSE_STEP_DELAY_S` between sub-sends. The number of sub-
        steps is driven by the larger of `|delta_pan|` and `|delta_tilt|`
        so the dominant axis advances exactly 1° per sub-step while the
        other axis interpolates linearly across the same schedule (its
        snapped value repeats across consecutive sub-sends whenever the
        interpolated position hasn't crossed an integer boundary yet —
        these repeats are no-ops at the servo, since an identical PWM
        pulse holds position without drawing current).

        Holds `_traverse_lock` for the full walk so concurrent callers
        don't interleave their sub-sends on the wire.
        """
        target_pan, target_tilt = self._prepare_pose(endpoint, pan, tilt)

        async with self._traverse_lock:
            # First-ever send (or post-failure recovery, where
            # _last_sent_pan/tilt was reset to None below): no known
            # starting pose, go direct and record only on success. A
            # failed first-ever send leaves _last_sent_pan/tilt at None
            # so the next call also takes this single-shot path —
            # recovery from a network blip is a sequence of single
            # POSTs, never a multi-sub-send burst.
            if self._last_sent_pan is None or self._last_sent_tilt is None:
                ok = await self._send_pose(endpoint, target_pan, target_tilt)
                if ok:
                    self._last_sent_pan = target_pan
                    self._last_sent_tilt = target_tilt
                return ok

            start_pan = self._last_sent_pan
            start_tilt = self._last_sent_tilt
            delta_pan = target_pan - start_pan
            delta_tilt = target_tilt - start_tilt
            max_distance = max(abs(delta_pan), abs(delta_tilt))

            # Number of sub-steps is driven by the larger axis distance,
            # then capped at _TRAVERSE_MAX_STEPS. The cap matters when
            # `_last_sent_pan/tilt` has drifted from `target` — usually
            # the result of the sweep advancing while a prior traverse
            # was failing under network errors. Without the cap the
            # recovery traverse rapid-fires through every intermediate
            # pose between the stale start and the new target.
            num_steps = max(1, int(math.ceil(max_distance / _TRAVERSE_STEP_DEG)))
            num_steps = min(num_steps, _TRAVERSE_MAX_STEPS)

            for i in range(1, num_steps + 1):
                frac = i / num_steps
                intermediate_pan = snap_to_servo_step(start_pan + delta_pan * frac)
                intermediate_tilt = snap_to_servo_step(start_tilt + delta_tilt * frac)
                ok = await self._send_pose(endpoint, intermediate_pan, intermediate_tilt)
                if not ok:
                    # Reset the cached "last sent" state so the next call
                    # takes the first-ever-send path (a single direct
                    # POST) rather than computing a delta against this
                    # now-stale reference. Leaving _last_sent at the
                    # last successful intermediate would cause the
                    # next call to walk the full accumulated drift,
                    # producing the rapid left/right snap-back that
                    # piles up timeouts and inflates the next failure.
                    self._last_sent_pan = None
                    self._last_sent_tilt = None
                    return False
                self._last_sent_pan = intermediate_pan
                self._last_sent_tilt = intermediate_tilt
                if i < num_steps:
                    await asyncio.sleep(_TRAVERSE_STEP_DELAY_S)
            return True

    async def aim(self, pan: float, tilt: float) -> bool:
        """Route a pose command through the rate-limited traverse to the
        ESP32's /aim endpoint. Compensation and snapping are applied
        inside `_traverse`; both axes walk in 1°-per-50ms sub-steps to
        the target, driven by whichever axis has the larger delta."""
        return await self._traverse("aim", pan, tilt)

    async def fire(self, duration_ms: int = 200) -> bool:
        try:
            response = await self._client.post(
                f"{self.base_url}/fire",
                json={"duration_ms": duration_ms},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send fire command: {type(e).__name__}: {e}")
            return False

    async def get_logs(self) -> dict | None:
        """Fetch the log ring buffer from the ESP32's /logs endpoint.

        Returns the parsed JSON dict (with keys `uptime_ms`, `count`, `lines`)
        on success, or None on network failure. Used by the control router
        to proxy logs to the frontend / operators.
        """
        try:
            response = await self._client.get(
                f"{self.base_url}/logs",
                timeout=3.0,
            )
            if response.status_code != 200:
                logger.error(f"Log fetch returned status {response.status_code}")
                return None
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch actuator logs: {type(e).__name__}: {e}")
            return None

    async def goto(self, pan: float, tilt: float, *, direct: bool = False) -> bool:
        """Route a pose command to the ESP32's /goto endpoint.

        Compensation and snapping are always applied before anything hits
        the wire.

        direct: when True, send the target as a single POST and return
                immediately (bypasses the rate-limited traverse). Use for
                tracking jumps where the servo's natural slew is faster than
                our 50ms-per-degree pacing. Sweep updates should keep
                direct=False so the per-frame 1° steps stay rate-limited for
                current-draw spreading.
        """
        if not direct:
            return await self._traverse("goto", pan, tilt)

        target_pan, target_tilt = self._prepare_pose("goto", pan, tilt)
        async with self._traverse_lock:
            ok = await self._send_pose("goto", target_pan, target_tilt)
            if ok:
                self._last_sent_pan = target_pan
                self._last_sent_tilt = target_tilt
            else:
                # Reset on failure so the next traverse-mode call takes
                # the first-ever-send path instead of computing a delta
                # from the stale prior position.
                self._last_sent_pan = None
                self._last_sent_tilt = None
            return ok

    async def stop(self) -> bool:
        """Stop sweep, hold current position."""
        try:
            response = await self._client.post(
                f"{self.base_url}/stop",
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.error(f"Failed to send stop command: {type(e).__name__}: {e}")
            return False

    async def aim_and_fire(self, pan: float, tilt: float, duration_ms: int = 200) -> bool:
        aimed = await self.aim(pan, tilt)
        if aimed:
            return await self.fire(duration_ms)
        return False

    async def close(self):
        await self._client.aclose()
