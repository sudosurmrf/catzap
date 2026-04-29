"""Predictor protocol shared across pytorch / onnx / tflite adapters.

Mirrors server/vision/detector.py:CatDetector.detect — every adapter
returns a list of {"bbox": [x1,y1,x2,y2 normalized], "confidence": float}
for class id 0 (cat). The eval harness is uniform across formats by
relying on this contract.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Predictor(Protocol):
    input_hw: tuple[int, int]

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Run inference on a single BGR frame and return cat detections."""
        ...

    def num_params(self) -> int:
        ...

    def num_flops(self) -> int:
        ...
