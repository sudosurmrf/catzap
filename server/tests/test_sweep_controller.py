import pytest
from unittest.mock import AsyncMock
from server.panorama.sweep_controller import SweepController, SweepState


class MockClock:
    """Deterministic time source for state-machine tests. Tests advance the
    clock explicitly via `advance(seconds)` rather than waiting on real time."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return MockClock()


@pytest.fixture
def controller(clock):
    actuator = AsyncMock()
    actuator.goto.return_value = True
    actuator.fire.return_value = True
    actuator.aim_and_fire.return_value = True
    return SweepController(
        actuator=actuator,
        pan_min=30.0, pan_max=150.0,
        tilt_min=20.0, tilt_max=70.0,
        speed=10.0,
        min_shot_interval_ms=2000,
        engagement_grace_ms=3000,
        dev_mode=True,
        time_source=clock,
    )


# ── Sweep behavior ──────────────────────────────────────────────

def test_initial_state_is_sweeping(controller):
    assert controller.state == SweepState.SWEEPING


def test_sweep_advances_angle(controller):
    initial = controller.current_pan
    controller.tick(dt=1.0)
    assert controller.current_pan != initial


def test_sweep_covers_tilt_range(controller):
    """Boustrophedon sweep should visit both top and bottom tilt rows."""
    tilt_values = set()
    for _ in range(200):
        controller.tick(dt=0.5)
        tilt_values.add(round(controller.current_tilt))
    tilt_mid = (controller.tilt_min + controller.tilt_max) / 2
    assert max(tilt_values) > tilt_mid
    assert min(tilt_values) < tilt_mid


def test_no_cat_detected_keeps_sweeping(controller):
    controller.on_no_cat_detected()
    assert controller.state == SweepState.SWEEPING


# ── SWEEPING → TRACKING (cat visible outside zones) ────────────

def test_cat_detected_outside_zones_transitions_to_tracking(controller):
    """Any visible cat — even one that's not inside a zone — should stop the
    sweep and start pursuit so the rig stalks every cat to make sure they
    don't violate rules without being followed."""
    controller.on_cat_detected(cat_pan=90.0, cat_tilt=45.0)
    assert controller.state == SweepState.TRACKING


def test_tracking_preserves_target(controller):
    controller.on_cat_detected(cat_pan=90.0, cat_tilt=45.0)
    target_pan, target_tilt, _ = controller.get_target()
    assert target_pan == 90.0
    assert target_tilt == 45.0


def test_tracking_should_not_fire(controller):
    """should_fire() must return False in TRACKING — the rig pursues but
    does not shoot until the cat crosses a zone boundary."""
    controller.on_cat_detected(cat_pan=90.0, cat_tilt=45.0)
    assert controller.state == SweepState.TRACKING
    assert controller.should_fire() is False


def test_camera_jumps_directly_to_target_on_tracking(controller):
    """Accepted on_cat_detected snaps current_pan/tilt to the target
    immediately — no smooth pursuit. Main loop picks up the change via
    dedupe and issues a single background goto."""
    controller.current_pan = 90.0
    controller.current_tilt = 45.0
    controller.on_cat_detected(cat_pan=120.0, cat_tilt=60.0)
    assert controller.current_pan == 120.0
    assert controller.current_tilt == 60.0


def test_tracking_updates_target_after_wait_window(controller, clock):
    """A second on_cat_detected more than min_shot_interval_ms after the
    first is accepted and updates the target."""
    controller.on_cat_detected(cat_pan=90.0, cat_tilt=45.0)
    clock.advance(2.0)
    controller.on_cat_detected(cat_pan=100.0, cat_tilt=40.0)
    target_pan, target_tilt, _ = controller.get_target()
    assert target_pan == 100.0
    assert target_tilt == 40.0


# ── SWEEPING → ENGAGING (cat in zone) ──────────────────────────

def test_cat_in_zone_transitions_to_engaging(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.ENGAGING


def test_cat_in_zone_sets_target(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    target_pan, target_tilt, zone = controller.get_target()
    assert target_pan == 90.0
    assert target_tilt == 45.0
    assert zone == "Counter"


def test_camera_jumps_directly_to_target_on_engagement(controller):
    """Accepted on_cat_in_zone snaps current_pan/tilt to the target
    immediately — same discrete jump as tracking."""
    controller.current_pan = 90.0
    controller.current_tilt = 45.0
    controller.on_cat_in_zone(cat_pan=120.0, cat_tilt=60.0, zone_name="Counter")
    assert controller.current_pan == 120.0
    assert controller.current_tilt == 60.0


# ── TRACKING ↔ ENGAGING (cat crosses zone boundary) ────────────

def test_tracking_to_engaging_on_zone_entry_after_wait(controller, clock):
    """After the 2s wait window, a cat that was being tracked transitions
    to ENGAGING when its bbox overlaps a zone."""
    controller.on_cat_detected(cat_pan=90.0, cat_tilt=45.0)
    assert controller.state == SweepState.TRACKING
    clock.advance(2.0)
    controller.on_cat_in_zone(cat_pan=95.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.ENGAGING


def test_engaging_to_tracking_when_cat_leaves_zone_after_wait(controller, clock):
    """Cat hops out of zone but is still visible — after the wait window,
    the next accepted update is on_cat_detected, dropping the state back
    to TRACKING."""
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.ENGAGING
    clock.advance(2.0)
    controller.on_cat_detected(cat_pan=85.0, cat_tilt=45.0)
    assert controller.state == SweepState.TRACKING


# ── Discrete 2s cycle: target-update gating ────────────────────

def test_first_target_update_always_accepted(controller):
    """From SWEEPING, the very first on_cat_detected / on_cat_in_zone call
    must be accepted immediately regardless of timing — _last_target_update_time
    starts at 0.0 so the elapsed check trivially passes."""
    controller.on_cat_detected(cat_pan=50.0, cat_tilt=30.0)
    target_pan, _, _ = controller.get_target()
    assert target_pan == 50.0


def test_target_update_rejected_within_wait_window_tracking(controller, clock):
    """In TRACKING, a second on_cat_detected within min_shot_interval_ms
    is a no-op for the target. The grace timer still ticks, the state
    stays, but current_pan does not move."""
    controller.on_cat_detected(cat_pan=90.0, cat_tilt=45.0)
    original_pan = controller.current_pan
    clock.advance(1.0)  # within 2s window
    controller.on_cat_detected(cat_pan=120.0, cat_tilt=60.0)
    target_pan, _, _ = controller.get_target()
    assert target_pan == 90.0  # target not updated
    assert controller.current_pan == original_pan  # camera did not jump


def test_target_update_rejected_within_wait_window_engaging(controller, clock):
    """Same gating applies to ENGAGING — shots fire at most once per 2s."""
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is True
    controller.mark_shot_fired()
    clock.advance(1.0)
    controller.on_cat_in_zone(cat_pan=100.0, cat_tilt=40.0, zone_name="Counter")
    target_pan, _, _ = controller.get_target()
    assert target_pan == 90.0  # target not updated
    assert controller.should_fire() is False  # shot gate not yet expired


def test_target_update_accepted_after_wait_window(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.mark_shot_fired()
    clock.advance(2.0)
    controller.on_cat_in_zone(cat_pan=100.0, cat_tilt=40.0, zone_name="Counter")
    target_pan, target_tilt, _ = controller.get_target()
    assert target_pan == 100.0
    assert target_tilt == 40.0
    assert controller.should_fire() is True


# ── Firing ──────────────────────────────────────────────────────

def test_first_shot_ready_on_entering_engaging(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is True


def test_shot_blocked_within_min_interval(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is True
    controller.mark_shot_fired()
    clock.advance(1.0)
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is False


def test_shot_allowed_after_min_interval(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.mark_shot_fired()
    clock.advance(2.0)
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.should_fire() is True


def test_on_fire_complete_resets_grace_timer(controller, clock):
    """The fire flow (goto + sleep + fire) takes 2-3 seconds of wall time.
    Without a grace reset, that elapsed time would eat into the 3s grace
    window and potentially drop to SWEEPING right after a successful shot.
    on_fire_complete must reset _last_cat_seen_time to now."""
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    # Simulate the fire flow taking ~2.5 seconds of wall time.
    clock.advance(2.5)
    controller.on_fire_complete(fired_pan=90.0, fired_tilt=45.0)
    # Even though 2.5s passed, the grace should now count from "just now",
    # not from before the fire. A subsequent tick after only 2.5s more
    # should still be in ENGAGING, not SWEEPING.
    clock.advance(2.5)
    controller.tick(dt=0.0)
    assert controller.state == SweepState.ENGAGING


# ── Grace window (applies to both TRACKING and ENGAGING) ───────

def test_grace_window_holds_engaging_briefly(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    clock.advance(1.0)
    controller.on_no_cat_detected()
    clock.advance(1.0)  # 1s into 3s grace
    controller.tick(dt=0.0)
    assert controller.state == SweepState.ENGAGING


def test_grace_expiry_from_engaging_returns_to_sweeping(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    clock.advance(0.1)
    controller.on_no_cat_detected()
    clock.advance(3.001)
    controller.tick(dt=0.0)
    assert controller.state == SweepState.SWEEPING


def test_grace_expiry_from_tracking_returns_to_sweeping(controller, clock):
    controller.on_cat_detected(cat_pan=90.0, cat_tilt=45.0)
    assert controller.state == SweepState.TRACKING
    clock.advance(0.1)
    controller.on_no_cat_detected()
    clock.advance(3.001)
    controller.tick(dt=0.0)
    assert controller.state == SweepState.SWEEPING


def test_cat_returns_during_grace_resumes_engaging(controller, clock):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.mark_shot_fired()
    clock.advance(2.1)  # past the min_shot_interval gate
    controller.on_no_cat_detected()
    clock.advance(0.5)
    controller.on_cat_in_zone(cat_pan=95.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.ENGAGING
    target_pan, _, _ = controller.get_target()
    assert target_pan == 95.0


def test_tracking_grace_preserved_while_cat_remains_visible(controller, clock):
    """A cat that is continuously visible (even if its target updates are
    rejected by the 2s gate) keeps the grace timer fresh — the rig should
    stay in TRACKING indefinitely, not drop to SWEEPING."""
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.ENGAGING
    # Simulate 5 seconds of "cat visible, calling on_cat_detected every
    # 100ms". Most calls are rejected by the wait gate, but each one
    # refreshes _last_cat_seen_time so grace never expires.
    for _ in range(50):
        clock.advance(0.1)
        controller.on_cat_detected(cat_pan=85.0, cat_tilt=45.0)
        controller.tick(dt=0.1)
    # After 5 seconds of "cat visible but out of zone", we should be in
    # TRACKING (dropped from ENGAGING on the first accepted update after
    # 2s), not SWEEPING.
    assert controller.state == SweepState.TRACKING


# ── Pause / stop unchanged ─────────────────────────────────────

def test_pause_from_sweeping(controller):
    controller.pause()
    assert controller.state == SweepState.PAUSED


def test_resume_from_paused(controller):
    controller.pause()
    controller.resume()
    assert controller.state == SweepState.SWEEPING


def test_emergency_stop(controller):
    controller.emergency_stop()
    assert controller.state == SweepState.STOPPED
    assert controller.armed is False


# ── Direction delta (used by dev-mode arrow overlay) ──────────

def test_get_direction_delta(controller):
    controller.current_pan = 80.0
    controller.current_tilt = 45.0
    delta = controller.get_direction_delta(target_pan=100.0, target_tilt=50.0)
    assert delta["pan"] > 0
    assert delta["tilt"] > 0
