import time
from server.spatial.cat_tracker import CatTracker, TrackedCat


def test_track_new_cat():
    tracker = CatTracker()
    tracker.update_detection("cat1", (150.0, 200.0, 30.0), time.time())
    cats = tracker.get_active_cats()
    assert len(cats) == 1
    assert cats[0].id == "cat1"
    assert cats[0].state == "visible"


def test_cat_builds_velocity():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.update_detection("cat1", (110.0, 200.0, 30.0), t + 0.1)
    tracker.update_detection("cat1", (120.0, 200.0, 30.0), t + 0.2)
    cat = tracker.get_cat("cat1")
    assert cat.velocity[0] > 50.0


def test_cat_goes_occluded():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (150.0, 200.0, 30.0), t)
    tracker.update_detection("cat1", (160.0, 200.0, 30.0), t + 0.1)
    tracker.mark_occluded("cat1", occluder_name="couch")
    cat = tracker.get_cat("cat1")
    assert cat.state == "occluded"
    assert cat.occluded_by == "couch"


def test_predict_position():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.update_detection("cat1", (110.0, 200.0, 30.0), t + 0.1)
    tracker.mark_occluded("cat1", occluder_name="couch")
    predicted = tracker.predict_position("cat1", t + 0.5)
    assert predicted[0] > 110.0


def test_cat_reconnects():
    tracker = CatTracker()
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.mark_occluded("cat1", occluder_name="couch")
    tracker.update_detection("cat1", (140.0, 200.0, 30.0), t + 0.5)
    cat = tracker.get_cat("cat1")
    assert cat.state == "visible"
    assert cat.occluded_by is None


def test_cat_goes_lost_after_timeout():
    tracker = CatTracker(occlusion_timeout=1.0)
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.mark_occluded("cat1", occluder_name="couch")
    tracker.tick(t + 2.0)
    cat = tracker.get_cat("cat1")
    assert cat.state == "lost"


def test_grace_period_before_lost():
    tracker = CatTracker(grace_frames=3)
    t = time.time()
    tracker.update_detection("cat1", (100.0, 200.0, 30.0), t)
    tracker.mark_missing("cat1")
    cat = tracker.get_cat("cat1")
    assert cat.state == "visible"
    assert cat.missing_frames == 1
    tracker.mark_missing("cat1")
    tracker.mark_missing("cat1")
    cat = tracker.get_cat("cat1")
    assert cat.state == "lost"
