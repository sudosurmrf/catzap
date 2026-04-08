import numpy as np
from server.spatial.room_model import RoomModel, FurnitureObject


def test_create_room_model():
    rm = RoomModel(width_cm=400, depth_cm=400, height_cm=300, resolution=5.0)
    assert rm.heightmap.shape == (80, 80)
    assert rm.furniture == []


def test_update_heightmap_cell():
    rm = RoomModel(width_cm=100, depth_cm=100, height_cm=300, resolution=10.0)
    rm.update_cell(0, 0, floor_height=0.0, max_height=75.0)
    cell = rm.get_cell(0, 0)
    assert cell["max_height"] == 75.0
    assert cell["readings"] == 1
    rm.update_cell(0, 0, floor_height=0.0, max_height=80.0)
    cell = rm.get_cell(0, 0)
    assert cell["readings"] == 2
    assert abs(cell["max_height"] - 77.5) < 0.1


def test_add_furniture():
    rm = RoomModel(width_cm=400, depth_cm=400, height_cm=300, resolution=5.0)
    table = FurnitureObject(name="table", base_polygon=[(100, 100), (200, 100), (200, 150), (100, 150)], height_min=0.0, height_max=75.0)
    rm.add_furniture(table)
    assert len(rm.furniture) == 1
    assert rm.furniture[0].name == "table"


def test_point_in_furniture():
    table = FurnitureObject(name="table", base_polygon=[(100, 100), (200, 100), (200, 150), (100, 150)], height_min=0.0, height_max=75.0)
    assert table.contains_point(150, 125, 70) is True
    assert table.contains_point(150, 125, 80) is False
    assert table.contains_point(50, 50, 50) is False


def test_detect_furniture_change():
    rm = RoomModel(width_cm=100, depth_cm=100, height_cm=300, resolution=10.0)
    for i in range(4):
        rm.update_cell(5, 5, floor_height=0.0, max_height=75.0)
    assert rm.check_cell_change(5, 5, new_max_height=105.0, threshold_cm=20.0) is True
    assert rm.check_cell_change(5, 5, new_max_height=78.0, threshold_cm=20.0) is False


def test_furniture_to_dict_and_back():
    table = FurnitureObject(name="table", base_polygon=[(100, 100), (200, 100), (200, 150), (100, 150)], height_min=0.0, height_max=75.0)
    d = table.to_dict()
    restored = FurnitureObject.from_dict(d)
    assert restored.name == "table"
    assert restored.height_max == 75.0
    assert len(restored.base_polygon) == 4
