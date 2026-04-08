import logging
import time
from typing import Callable

import numpy as np

from server.actuator.calibration import CalibrationMap
from server.actuator.client import ActuatorClient
from server.vision.detector import CatDetector
from server.vision.zone_checker import check_zone_violations

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Orchestrates frame -> detect -> zone check -> fire decision."""

    def __init__(
        self,
        detector: CatDetector,
        actuator: ActuatorClient,
        calibration: CalibrationMap,
        zones: list[dict],
        armed: bool = True,
        on_event: Callable | None = None,
    ):
        self.detector = detector
        self.actuator = actuator
        self.calibration = calibration
        self.zones = zones
        self.armed = armed
        self.on_event = on_event
        self._cooldowns: dict[str, float] = {}

    def update_zones(self, zones: list[dict]):
        self.zones = zones

    async def process_frame(self, frame: np.ndarray) -> dict:
        detections = self.detector.detect(frame)

        all_violations = []
        fired = False

        for det in detections:
            bbox = det["bbox"]
            violations = check_zone_violations(bbox, self.zones)
            all_violations.extend(violations)

            if not violations or not self.armed:
                continue

            for violation in violations:
                zone_id = violation["zone_id"]
                zone = next((z for z in self.zones if z["id"] == zone_id), None)
                if not zone:
                    continue

                cooldown = zone.get("cooldown_seconds", 10)
                last_fire = self._cooldowns.get(zone_id, 0)

                if time.time() - last_fire < cooldown:
                    continue

                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                pan, tilt = self.calibration.pixel_to_angle(center_x, center_y)

                success = await self.actuator.aim_and_fire(pan, tilt)
                if success:
                    self._cooldowns[zone_id] = time.time()
                    fired = True

                    if self.on_event:
                        self.on_event({
                            "type": "ZAP",
                            "cat_name": det.get("cat_name"),
                            "zone_name": violation["zone_name"],
                            "confidence": det["confidence"],
                            "overlap": violation["overlap"],
                            "servo_pan": pan,
                            "servo_tilt": tilt,
                        })

                    break

        return {
            "detections": detections,
            "violations": all_violations,
            "fired": fired,
        }
