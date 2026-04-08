import pytest
from unittest.mock import AsyncMock, MagicMock
from server.panorama.sweep_controller import SweepController, SweepState


@pytest.fixture
def controller():
    actuator = AsyncMock()
    actuator.goto.return_value = True
    actuator.fire.return_value = True
    return SweepController(
        actuator=actuator,
        pan_min=30.0, pan_max=150.0,
        tilt_min=20.0, tilt_max=70.0,
        speed=10.0,
        warning_duration=1.5,
        tracking_duration=3.0,
        cooldown=3.0,
        dev_mode=True,
    )


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
    # Should visit both the top and bottom tilt rows
    tilt_mid = (controller.tilt_min + controller.tilt_max) / 2
    assert max(tilt_values) > tilt_mid
    assert min(tilt_values) < tilt_mid


def test_cat_in_zone_transitions_to_warning(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.WARNING


def test_warning_expires_transitions_to_firing(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    assert controller.state == SweepState.WARNING
    # Simulate time passing beyond warning duration
    controller._warning_start = 0  # long ago
    controller.tick(dt=2.0)
    controller.on_cat_still_in_zone(cat_pan=90.0, cat_tilt=45.0)
    assert controller.state == SweepState.FIRING


def test_cat_leaves_during_warning_stays_in_grace(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.on_cat_left_zone()
    # Should stay in WARNING during grace period, not immediately sweep
    assert controller.state == SweepState.WARNING
    assert controller._in_grace is True


def test_cat_leaves_during_warning_returns_to_sweeping_after_grace(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.on_cat_left_zone()
    # Simulate grace period expiring
    controller._grace_start = 0  # long ago
    controller.tick(dt=0.0)
    assert controller.state == SweepState.SWEEPING


def test_cat_returns_during_grace_cancels_grace(controller):
    controller.on_cat_in_zone(cat_pan=90.0, cat_tilt=45.0, zone_name="Counter")
    controller.on_cat_left_zone()
    assert controller._in_grace is True
    # Cat comes back
    controller.on_cat_still_in_zone(cat_pan=90.0, cat_tilt=45.0)
    assert controller._in_grace is False
    assert controller.state == SweepState.WARNING


def test_firing_transitions_to_tracking(controller):
    controller.state = SweepState.FIRING
    controller.on_fire_complete()
    controller.on_cat_left_zone()
    assert controller.state == SweepState.TRACKING


def test_tracking_returns_to_sweeping_after_timeout(controller):
    controller.state = SweepState.TRACKING
    controller._tracking_start = 0  # long ago
    controller.tick(dt=0.0)
    assert controller.state == SweepState.SWEEPING


def test_get_direction_delta(controller):
    controller.current_pan = 80.0
    controller.current_tilt = 45.0
    delta = controller.get_direction_delta(target_pan=100.0, target_tilt=50.0)
    assert delta["pan"] > 0  # need to pan right
    assert delta["tilt"] > 0  # need to tilt down
