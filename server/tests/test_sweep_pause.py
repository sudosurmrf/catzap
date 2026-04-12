from unittest.mock import MagicMock
from server.panorama.sweep_controller import SweepController, SweepState


class MockClock:
    """Deterministic time source so grace-window tests don't need real sleeps."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_controller(**kwargs):
    actuator = MagicMock()
    clock = kwargs.pop("clock", None) or MockClock()
    defaults = dict(
        actuator=actuator, pan_min=30, pan_max=150,
        tilt_min=20, tilt_max=70, speed=10, dev_mode=True,
        time_source=clock,
    )
    defaults.update(kwargs)
    sc = SweepController(**defaults)
    sc._clock = clock  # stash for tests that need to advance it
    return sc


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


def test_pause_queues_during_engaging():
    """Pause requested while ENGAGING should queue, not take effect immediately
    — the controller finishes handling the cat before returning to sweep/pause."""
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    assert sc.state == SweepState.ENGAGING
    sc.pause()
    assert sc.state == SweepState.ENGAGING
    assert sc.pause_queued is True


def test_queued_pause_activates_after_grace_expiry():
    """A queued pause should take effect when ENGAGING transitions out via
    the grace window — the cat is gone and we were about to return to
    SWEEPING anyway, but the queued pause redirects to PAUSED instead."""
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    sc.pause()
    assert sc.pause_queued is True
    # Advance past the 3000 ms engagement grace window with no new zone reports.
    sc._clock.advance(3.5)
    sc.tick(0.0)
    assert sc.state == SweepState.PAUSED


def test_emergency_stop_from_sweeping():
    sc = make_controller()
    sc.emergency_stop()
    assert sc.state == SweepState.STOPPED
    assert sc.armed is False


def test_emergency_stop_from_engaging():
    sc = make_controller()
    sc.on_cat_in_zone(90, 45, "table")
    assert sc.state == SweepState.ENGAGING
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


def test_clear_estop_only_works_when_stopped():
    sc = make_controller()
    assert sc.state == SweepState.SWEEPING
    sc.clear_emergency_stop()
    assert sc.state == SweepState.SWEEPING
