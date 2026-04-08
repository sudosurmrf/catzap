from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackedCat:
    id: str
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_seen: float = 0.0
    state: str = "visible"  # "visible" | "occluded" | "lost"
    occluded_by: str | None = None
    missing_frames: int = 0
    _occlusion_start: float = 0.0


class CatTracker:
    def __init__(self, occlusion_timeout: float = 10.0, grace_frames: int = 3):
        self.occlusion_timeout = occlusion_timeout
        self.grace_frames = grace_frames
        self._cats: dict[str, TrackedCat] = {}

    def update_detection(self, cat_id: str, room_pos: tuple[float, float, float], timestamp: float):
        if cat_id not in self._cats:
            self._cats[cat_id] = TrackedCat(id=cat_id)
        cat = self._cats[cat_id]
        cat.positions.append((room_pos[0], room_pos[1], room_pos[2], timestamp))
        cat.last_seen = timestamp
        cat.state = "visible"
        cat.occluded_by = None
        cat.missing_frames = 0
        if len(cat.positions) >= 2:
            p1 = cat.positions[-2]
            p2 = cat.positions[-1]
            dt = p2[3] - p1[3]
            if dt > 0:
                vx = (p2[0] - p1[0]) / dt
                vy = (p2[1] - p1[1]) / dt
                vz = (p2[2] - p1[2]) / dt
                a = 0.4
                cat.velocity = (
                    a * vx + (1 - a) * cat.velocity[0],
                    a * vy + (1 - a) * cat.velocity[1],
                    a * vz + (1 - a) * cat.velocity[2],
                )

    def mark_occluded(self, cat_id: str, occluder_name: str):
        cat = self._cats.get(cat_id)
        if not cat:
            return
        cat.state = "occluded"
        cat.occluded_by = occluder_name
        cat._occlusion_start = cat.last_seen
        cat.missing_frames = 0

    def mark_missing(self, cat_id: str):
        cat = self._cats.get(cat_id)
        if not cat or cat.state != "visible":
            return
        cat.missing_frames += 1
        if cat.missing_frames >= self.grace_frames:
            cat.state = "lost"

    def predict_position(self, cat_id: str, timestamp: float) -> tuple[float, float, float] | None:
        cat = self._cats.get(cat_id)
        if not cat or not cat.positions:
            return None
        last = cat.positions[-1]
        dt = timestamp - last[3]
        decay = max(0.0, 1.0 - dt * 0.3)
        return (
            last[0] + cat.velocity[0] * dt * decay,
            last[1] + cat.velocity[1] * dt * decay,
            last[2],
        )

    def tick(self, current_time: float):
        for cat in self._cats.values():
            if cat.state == "occluded":
                elapsed = current_time - cat._occlusion_start
                if elapsed >= self.occlusion_timeout:
                    cat.state = "lost"
                    cat.occluded_by = None

    def get_cat(self, cat_id: str) -> TrackedCat | None:
        return self._cats.get(cat_id)

    def get_active_cats(self) -> list[TrackedCat]:
        return [c for c in self._cats.values() if c.state in ("visible", "occluded")]

    def get_occluded_cats(self) -> list[TrackedCat]:
        return [c for c in self._cats.values() if c.state == "occluded"]

    def cleanup_lost(self, max_age: float = 60.0, current_time: float = 0.0):
        to_remove = [cid for cid, cat in self._cats.items() if cat.state == "lost" and current_time - cat.last_seen > max_age]
        for cat_id in to_remove:
            del self._cats[cat_id]
