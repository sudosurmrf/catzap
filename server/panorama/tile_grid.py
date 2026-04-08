import math

import cv2
import numpy as np


class TileGrid:
    """Manages a grid of image tiles in servo angle-space."""

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

        self.tile_step_h = fov_h - tile_overlap
        self.tile_step_v = fov_v - tile_overlap

        self.cols = max(1, math.ceil((pan_max - pan_min) / self.tile_step_h))
        self.rows = max(1, math.ceil((tilt_max - tilt_min) / self.tile_step_v))

        # Storage: (col, row) -> JPEG bytes
        self._tiles: dict[tuple[int, int], bytes] = {}
        # Raw frames for smart refresh comparison
        self._raw_tiles: dict[tuple[int, int], np.ndarray] = {}
        # Cached panorama — invalidated when any tile changes
        self._pano_cache: bytes | None = None
        self._pano_dirty: bool = True

    def get_tile_positions(self) -> list[tuple[float, float]]:
        """Return the servo (pan, tilt) center position for each tile."""
        positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                pan = self.pan_min + col * self.tile_step_h + self.fov_h / 2
                tilt = self.tilt_min + row * self.tile_step_v + self.fov_v / 2
                # Clamp to range
                pan = min(pan, self.pan_max - self.fov_h / 2 + self.fov_h / 2)
                tilt = min(tilt, self.tilt_max - self.fov_v / 2 + self.fov_v / 2)
                positions.append((pan, tilt))
        return positions

    def angle_to_tile_index(self, pan: float, tilt: float) -> tuple[int, int]:
        """Find which tile a given angle falls into."""
        col = int((pan - self.pan_min - self.fov_h / 2) / self.tile_step_h + 0.5)
        row = int((tilt - self.tilt_min - self.fov_v / 2) / self.tile_step_v + 0.5)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (col, row)

    def update_tile(self, col: int, row: int, frame: np.ndarray):
        """Store a frame as a tile."""
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._tiles[(col, row)] = jpeg.tobytes()
        # Store downscaled raw for comparison
        small = cv2.resize(frame, (64, 48))
        self._raw_tiles[(col, row)] = small
        self._pano_dirty = True

    def get_tile(self, col: int, row: int) -> bytes | None:
        """Get JPEG bytes for a tile."""
        return self._tiles.get((col, row))

    def should_refresh(
        self, col: int, row: int, new_frame: np.ndarray, threshold: int = 15
    ) -> bool:
        """Check if a tile needs updating by comparing to stored version."""
        if (col, row) not in self._raw_tiles:
            return True
        old = self._raw_tiles[(col, row)]
        new_small = cv2.resize(new_frame, (64, 48))
        # Use int16 instead of float64 to halve temporary memory allocation
        diff = np.mean(np.abs(old.astype(np.int16) - new_small.astype(np.int16)))
        return diff > threshold

    def get_panorama_image(self) -> np.ndarray | None:
        """Assemble all tiles into a single panorama image."""
        if not self._tiles:
            return None

        # Determine tile pixel size from first tile
        first_key = next(iter(self._tiles))
        first_jpeg = self._tiles[first_key]
        first_frame = cv2.imdecode(
            np.frombuffer(first_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        tile_h, tile_w = first_frame.shape[:2]

        pano_w = self.cols * tile_w
        pano_h = self.rows * tile_h
        panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

        for (col, row), jpeg in self._tiles.items():
            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue
            resized = cv2.resize(frame, (tile_w, tile_h))
            y = row * tile_h
            x = col * tile_w
            panorama[y : y + tile_h, x : x + tile_w] = resized

        return panorama

    def get_panorama_jpeg(self, quality: int = 70) -> bytes | None:
        """Get the assembled panorama as JPEG bytes (cached until tiles change)."""
        if not self._pano_dirty and self._pano_cache is not None:
            return self._pano_cache
        pano = self.get_panorama_image()
        if pano is None:
            return None
        _, jpeg = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, quality])
        self._pano_cache = jpeg.tobytes()
        self._pano_dirty = False
        return self._pano_cache

    def panorama_pixel_to_angle(self, px: float, py: float) -> tuple[float, float]:
        """Convert a panorama pixel position (normalized 0-1) to angle-space."""
        pan = self.pan_min + px * (self.pan_max - self.pan_min)
        tilt = self.tilt_min + py * (self.tilt_max - self.tilt_min)
        return (pan, tilt)

    def angle_to_panorama_pixel(self, pan: float, tilt: float) -> tuple[float, float]:
        """Convert angle-space to panorama pixel position (normalized 0-1)."""
        px = (pan - self.pan_min) / (self.pan_max - self.pan_min)
        py = (tilt - self.tilt_min) / (self.tilt_max - self.tilt_min)
        return (px, py)
