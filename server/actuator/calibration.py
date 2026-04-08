import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class CalibrationMap:
    """Maps pixel coordinates (normalized 0-1) to servo angles using interpolation."""

    def __init__(self):
        self._points: list[dict] = []
        self._pan_interp = None
        self._tilt_interp = None
        self._pan_nearest = None
        self._tilt_nearest = None

    def add_point(self, pixel_x: float, pixel_y: float, pan_angle: float, tilt_angle: float):
        self._points.append({
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "pan_angle": pan_angle,
            "tilt_angle": tilt_angle,
        })
        self._rebuild_interpolators()

    def clear(self):
        self._points = []
        self._pan_interp = None
        self._tilt_interp = None
        self._pan_nearest = None
        self._tilt_nearest = None

    def _rebuild_interpolators(self):
        if len(self._points) < 1:
            self._pan_interp = None
            self._tilt_interp = None
            return

        pixels = np.array([[p["pixel_x"], p["pixel_y"]] for p in self._points])
        pans = np.array([p["pan_angle"] for p in self._points])
        tilts = np.array([p["tilt_angle"] for p in self._points])

        if len(self._points) <= 3:
            self._pan_interp = NearestNDInterpolator(pixels, pans)
            self._tilt_interp = NearestNDInterpolator(pixels, tilts)
        else:
            self._pan_interp = LinearNDInterpolator(pixels, pans)
            self._tilt_interp = LinearNDInterpolator(pixels, tilts)
            self._pan_nearest = NearestNDInterpolator(pixels, pans)
            self._tilt_nearest = NearestNDInterpolator(pixels, tilts)

    def pixel_to_angle(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Convert normalized pixel coordinates to servo angles."""
        if self._pan_interp is None:
            return (90.0, 90.0)

        point = np.array([[pixel_x, pixel_y]])
        pan = float(self._pan_interp(point)[0])
        tilt = float(self._tilt_interp(point)[0])

        if np.isnan(pan) and self._pan_nearest:
            pan = float(self._pan_nearest(point)[0])
        if np.isnan(tilt) and self._tilt_nearest:
            tilt = float(self._tilt_nearest(point)[0])

        pan = max(0.0, min(180.0, pan))
        tilt = max(0.0, min(180.0, tilt))

        return (pan, tilt)

    def to_dict(self) -> dict:
        return {"points": self._points}

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationMap":
        cal = cls()
        for p in data.get("points", []):
            cal.add_point(p["pixel_x"], p["pixel_y"], p["pan_angle"], p["tilt_angle"])
        return cal
