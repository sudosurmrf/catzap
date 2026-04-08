import uuid
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon


@dataclass
class FurnitureObject:
    name: str
    base_polygon: list[tuple[float, float]]
    height_min: float = 0.0
    height_max: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    depth_anchored: bool = False

    def contains_point(self, x: float, y: float, z: float) -> bool:
        if z < self.height_min or z > self.height_max:
            return False
        poly = Polygon(self.base_polygon)
        return poly.contains(Point(x, y))

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "base_polygon": self.base_polygon,
            "height_min": self.height_min, "height_max": self.height_max,
            "depth_anchored": self.depth_anchored,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FurnitureObject":
        return cls(
            id=d.get("id", str(uuid.uuid4())), name=d["name"],
            base_polygon=[tuple(p) for p in d["base_polygon"]],
            height_min=d["height_min"], height_max=d["height_max"],
            depth_anchored=d.get("depth_anchored", False),
        )


class RoomModel:
    def __init__(self, width_cm: float, depth_cm: float, height_cm: float, resolution: float = 5.0):
        self.width_cm = width_cm
        self.depth_cm = depth_cm
        self.height_cm = height_cm
        self.resolution = resolution
        cols = int(width_cm / resolution)
        rows = int(depth_cm / resolution)
        self._grid = np.zeros((rows, cols, 3), dtype=np.float32)  # [floor_height, max_height, readings]
        self.furniture: list[FurnitureObject] = []
        self.depth_scale: float = 1.0

    @property
    def heightmap(self) -> np.ndarray:
        return self._grid[:, :, 1]

    def update_cell(self, row: int, col: int, floor_height: float, max_height: float):
        readings = int(self._grid[row, col, 2])
        if readings == 0:
            self._grid[row, col, 0] = floor_height
            self._grid[row, col, 1] = max_height
        elif readings < 3:
            n = readings + 1
            self._grid[row, col, 0] = (self._grid[row, col, 0] * readings + floor_height) / n
            self._grid[row, col, 1] = (self._grid[row, col, 1] * readings + max_height) / n
        else:
            alpha = 0.2
            self._grid[row, col, 0] = (1 - alpha) * self._grid[row, col, 0] + alpha * floor_height
            self._grid[row, col, 1] = (1 - alpha) * self._grid[row, col, 1] + alpha * max_height
        self._grid[row, col, 2] += 1

    def get_cell(self, row: int, col: int) -> dict:
        return {
            "floor_height": float(self._grid[row, col, 0]),
            "max_height": float(self._grid[row, col, 1]),
            "readings": int(self._grid[row, col, 2]),
        }

    def check_cell_change(self, row: int, col: int, new_max_height: float, threshold_cm: float = 20.0) -> bool:
        readings = int(self._grid[row, col, 2])
        if readings < 3:
            return False
        current = float(self._grid[row, col, 1])
        return abs(new_max_height - current) > threshold_cm

    def add_furniture(self, obj: FurnitureObject):
        self.furniture.append(obj)

    def remove_furniture(self, furniture_id: str) -> bool:
        before = len(self.furniture)
        self.furniture = [f for f in self.furniture if f.id != furniture_id]
        return len(self.furniture) < before

    def get_occluding_furniture(self, camera_x: float, camera_y: float, target_x: float, target_y: float) -> list[FurnitureObject]:
        from shapely.geometry import LineString
        line = LineString([(camera_x, camera_y), (target_x, target_y)])
        return [f for f in self.furniture if Polygon(f.base_polygon).intersects(line)]
