import numpy as np
import pytest
from server.panorama.tile_grid import TileGrid


@pytest.fixture
def grid():
    return TileGrid(
        pan_min=30.0, pan_max=150.0,
        tilt_min=20.0, tilt_max=70.0,
        fov_h=65.0, fov_v=50.0,
        tile_overlap=10.0,
    )


def test_grid_computes_tile_positions(grid):
    positions = grid.get_tile_positions()
    assert len(positions) > 0
    # Each position is (pan_center, tilt_center)
    for pan, tilt in positions:
        assert 30.0 <= pan <= 150.0
        assert 20.0 <= tilt <= 70.0


def test_grid_tile_count(grid):
    positions = grid.get_tile_positions()
    # 120 degree range with 55 degree step = 3 columns, 50 degree range with 40 degree step = 2 rows
    assert 2 <= len(positions) <= 12


def test_store_and_retrieve_tile(grid):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    grid.update_tile(0, 0, frame)
    tile = grid.get_tile(0, 0)
    assert tile is not None


def test_get_empty_tile_returns_none(grid):
    tile = grid.get_tile(0, 0)
    assert tile is None


def test_smart_refresh_detects_change(grid):
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    grid.update_tile(0, 0, frame1)
    assert grid.should_refresh(0, 0, frame2, threshold=15) == True


def test_smart_refresh_skips_similar(grid):
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 5
    grid.update_tile(0, 0, frame1)
    assert grid.should_refresh(0, 0, frame2, threshold=15) == False


def test_angle_to_tile_index(grid):
    positions = grid.get_tile_positions()
    if len(positions) > 0:
        pan, tilt = positions[0]
        col, row = grid.angle_to_tile_index(pan, tilt)
        assert col == 0
        assert row == 0


def test_get_panorama_image(grid):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    positions = grid.get_tile_positions()
    for i, (pan, tilt) in enumerate(positions):
        col, row = grid.angle_to_tile_index(pan, tilt)
        grid.update_tile(col, row, frame)
    img = grid.get_panorama_image()
    assert img is not None
    assert img.shape[0] > 0
    assert img.shape[1] > 0
