import time
from unittest.mock import MagicMock
from server.panorama.sweep_controller import SweepController, SweepState


def make_controller(**kwargs):
    actuator = MagicMock()
    defaults = dict(
        actuator=actuator, pan_min=30, pan_max=150,
        tilt_min=20, tilt_max=70, speed=10, dev_mode=True,
    )
    defaults.update(kwargs)
    return SweepController(**defaults)


def test_pause_from_sweeping():
    sc = make_controller()
    assert sc.state == SweepState.SWEEPING
    sc.pause()
    assert sc.state == SweepState.PAUSED


def test_pause_freezes_position():
    sc = make_controller()
    sc.current_pan = 90.0
    sc.pause()
    old_pan = sc.current_pan
    sc.tick(1.0)
    assert sc.current_pan == old_pan


def test_resume_from_paused():
    sc = make_controller()
    sc.pause()
    sc.resume()
    assert sc.state == SweepState.SWEEPING


def test_resume_continues_from_position():
    sc = make_controller()
    sc.current_pan = 90.0
    sc.pause()
    sc.resume()
    sc.tick(1.0)
    assert sc.current_pan != 90.0


def test_pause_queues_during_warning():
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    assert sc.state == SweepState.WARNING
    sc.pause()
    assert sc.state == SweepState.WARNING
    assert sc.pause_queued is True


def test_queued_pause_activates_on_return_to_sweeping():
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    sc.pause()
    assert sc.pause_queued is True
    sc.on_cat_left_zone()
    assert sc.state == SweepState.PAUSED


def test_emergency_stop_from_sweeping():
    sc = make_controller()
    sc.emergency_stop()
    assert sc.state == SweepState.STOPPED
    assert sc.armed is False


def test_emergency_stop_from_warning():
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    sc.emergency_stop()
    assert sc.state == SweepState.STOPPED
    assert sc.armed is False


def test_emergency_stop_from_firing():
    sc = make_controller()
    sc.state = SweepState.FIRING
    sc.emergency_stop()
    assert sc.state == SweepState.STOPPED
    assert sc.armed is False


def test_stopped_blocks_all_motion():
    sc = make_controller()
    sc.current_pan = 90.0
    sc.emergency_stop()
    sc.tick(1.0)
    assert sc.current_pan == 90.0


def test_clear_estop():
    sc = make_controller()
    sc.emergency_stop()
    sc.clear_emergency_stop()
    assert sc.state == SweepState.SWEEPING
    assert sc.armed is False


def test_queued_pause_activates_after_tracking_timeout():
    sc = make_controller(tracking_duration=0.5)
    sc.state = SweepState.TRACKING
    sc._tracking_start = time.time()
    sc.pause()
    assert sc.pause_queued is True
    # Simulate enough time for tracking to expire
    time.sleep(0.6)
    sc.tick(0.1)
    assert sc.state == SweepState.PAUSED


def test_clear_estop_only_works_when_stopped():
    sc = make_controller()
    assert sc.state == SweepState.SWEEPING
    sc.clear_emergency_stop()
    assert sc.state == SweepState.SWEEPING
