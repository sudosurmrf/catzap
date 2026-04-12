import math

import cv2
import numpy as np


class TileGrid:
    """Continuous-painting panorama canvas in servo angle-space.

    Instead of discrete tiles stitched together, each camera frame is painted
    directly onto a persistent canvas at its angle-space position with edge
    feathering. The canvas fills in as the camera sweeps, producing a seamless
    panorama without stitching seams or grid artifacts.

    The class retains the TileGrid name and a few legacy helpers
    (`get_tile_positions`, `cols`, `rows`) for calibration verification.
    """

    CANVAS_PPD = 8    # pixels per degree — canvas resolution
    FEATHER_DEG = 5.0  # degrees of linear edge fade on each painted frame

    def __init__(
        self,
        pan_min: float = 30.0,
        pan_max: float = 150.0,
        tilt_min: float = 20.0,
        tilt_max: float = 70.0,
        fov_h: float = 65.0,
        fov_v: float = 50.0,
        tile_overlap: float = 10.0,
    ):
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.tile_overlap = tile_overlap

        # Grid geometry kept for calibration verification targets.
        self.tile_step_h = fov_h - tile_overlap
        self.tile_step_v = fov_v - tile_overlap
        self.cols = max(1, math.ceil((pan_max - pan_min) / self.tile_step_h))
        self.rows = max(1, math.ceil((tilt_max - tilt_min) / self.tile_step_v))

        self._pano_cache: bytes | None = None
        self._pano_dirty: bool = True

        self._build_canvas()

    # ── Canvas setup ──────────────────────────────────

    def _build_canvas(self) -> None:
        ppd = self.CANVAS_PPD
        self._canvas_w = max(1, int(round((self.pan_max - self.pan_min) * ppd)))
        self._canvas_h = max(1, int(round((self.tilt_max - self.tilt_min) * ppd)))
        self._canvas = np.zeros((self._canvas_h, self._canvas_w, 3), dtype=np.uint8)

        self._frame_w = max(1, int(round(self.fov_h * ppd)))
        self._frame_h = max(1, int(round(self.fov_v * ppd)))
        feather_px = max(1, int(round(self.FEATHER_DEG * ppd)))
        self._feather_mask = self._make_feather(self._frame_w, self._frame_h, feather_px)

        self._pano_cache = None
        self._pano_dirty = True

    @staticmethod
    def _make_feather(w: int, h: int, fpx: int) -> np.ndarray:
        """1.0 in the center, linear fade to 0.0 over *fpx* pixels at every
        edge. Horizontal and vertical ramps are multiplied so corners
        receive a smooth bilinear falloff."""
        mask = np.ones((h, w), dtype=np.float32)
        if fpx < 1 or w <= 2 * fpx or h <= 2 * fpx:
            return mask
        ramp = np.linspace(0.0, 1.0, fpx, dtype=np.float32)
        mask[:, :fpx] *= ramp[np.newaxis, :]
        mask[:, -fpx:] *= ramp[::-1][np.newaxis, :]
        mask[:fpx, :] *= ramp[:, np.newaxis]
        mask[-fpx:, :] *= ramp[::-1][:, np.newaxis]
        return mask

    # ── Primary interface ─────────────────────────────

    def paint_frame(self, pan: float, tilt: float, frame: np.ndarray) -> None:
        """Paint a camera frame onto the canvas at its servo angle, with
        edge feathering so adjacent strokes blend seamlessly."""
        ppd = self.CANVAS_PPD

        # Frame rectangle in canvas pixels, centered on (pan, tilt).
        cx = (pan - self.pan_min) * ppd
        cy = (tilt - self.tilt_min) * ppd
        x0 = int(round(cx - self._frame_w / 2))
        y0 = int(round(cy - self._frame_h / 2))

        # Clip to canvas bounds.
        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = min(self._canvas_w, x0 + self._frame_w)
        dy1 = min(self._canvas_h, y0 + self._frame_h)
        sw = dx1 - dx0
        sh = dy1 - dy0
        if sw <= 0 or sh <= 0:
            return

        # Source offsets (handle left/top clipping).
        sx0 = dx0 - x0
        sy0 = dy0 - y0

        resized = cv2.resize(frame, (self._frame_w, self._frame_h))
        crop = resized[sy0:sy0 + sh, sx0:sx0 + sw]
        mask = self._feather_mask[sy0:sy0 + sh, sx0:sx0 + sw]

        # Alpha blend: new_frame * mask + existing_canvas * (1 - mask).
        old = self._canvas[dy0:dy1, dx0:dx1].astype(np.float32)
        new = crop.astype(np.float32)
        blended = new * mask[:, :, np.newaxis] + old * (1.0 - mask[:, :, np.newaxis])
        self._canvas[dy0:dy1, dx0:dx1] = blended.astype(np.uint8)
        self._pano_dirty = True

    # ── Panorama output ───────────────────────────────

    def get_panorama_image(self) -> np.ndarray | None:
        if self._canvas is None:
            return None
        return self._canvas.copy()

    def get_panorama_jpeg(self, quality: int = 70) -> bytes | None:
        if not self._pano_dirty and self._pano_cache is not None:
            return self._pano_cache
        if self._canvas is None:
            return None
        _, jpeg = cv2.imencode(".jpg", self._canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
        self._pano_cache = jpeg.tobytes()
        self._pano_dirty = False
        return self._pano_cache

    # ── Coordinate mapping (normalized 0-1 ↔ angle) ──

    def panorama_pixel_to_angle(self, px: float, py: float) -> tuple[float, float]:
        pan = self.pan_min + px * (self.pan_max - self.pan_min)
        tilt = self.tilt_min + py * (self.tilt_max - self.tilt_min)
        return (pan, tilt)

    def angle_to_panorama_pixel(self, pan: float, tilt: float) -> tuple[float, float]:
        px = (pan - self.pan_min) / (self.pan_max - self.pan_min)
        py = (tilt - self.tilt_min) / (self.tilt_max - self.tilt_min)
        return (px, py)

    # ── Calibration helpers ───────────────────────────

    def get_tile_positions(self) -> list[tuple[float, float]]:
        """Evenly-spaced positions across the sweep range. Used by the
        calibration router to generate verification targets."""
        positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                pan = self.pan_min + col * self.tile_step_h + self.fov_h / 2
                tilt = self.tilt_min + row * self.tile_step_v + self.fov_v / 2
                positions.append((pan, tilt))
        return positions

    def set_bounds(
        self, pan_min: float, pan_max: float, tilt_min: float, tilt_max: float
    ) -> None:
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.cols = max(1, math.ceil((pan_max - pan_min) / self.tile_step_h))
        self.rows = max(1, math.ceil((tilt_max - tilt_min) / self.tile_step_v))
        self._build_canvas()

    def recalibrate(self, fov_h: float, fov_v: float) -> None:
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.tile_step_h = fov_h - self.tile_overlap
        self.tile_step_v = fov_v - self.tile_overlap
        self.cols = max(1, math.ceil((self.pan_max - self.pan_min) / self.tile_step_h))
        self.rows = max(1, math.ceil((self.tilt_max - self.tilt_min) / self.tile_step_v))
        self._build_canvas()

    # ── Legacy tile interface (still called by main.py) ──

    def angle_to_tile_index(self, pan: float, tilt: float) -> tuple[int, int]:
        col = int((pan - self.pan_min - self.fov_h / 2) / self.tile_step_h + 0.5)
        row = int((tilt - self.tilt_min - self.fov_v / 2) / self.tile_step_v + 0.5)
        return (max(0, min(col, self.cols - 1)), max(0, min(row, self.rows - 1)))

    def should_refresh(self, col: int, row: int, frame: np.ndarray, threshold: int = 15) -> bool:
        return True

    def update_tile(self, col: int, row: int, frame: np.ndarray) -> None:
        pan = self.pan_min + col * self.tile_step_h + self.fov_h / 2
        tilt = self.tilt_min + row * self.tile_step_v + self.fov_v / 2
        self.paint_frame(pan, tilt, frame)
