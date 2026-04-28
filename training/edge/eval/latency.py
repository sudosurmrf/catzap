"""Single-thread CPU latency measurement.

Spec: 10 warmup iterations, then 100 timed iterations. Returns p50 and p95
in milliseconds. Each predictor's .predict(frame) is invoked with the
provided sample frame.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def measure_latency(
    predict_fn: Callable[[np.ndarray], object],
    sample_frame: np.ndarray,
    runs: int = 100,
    warmup: int = 10,
) -> tuple[float, float]:
    """Return (p50_ms, p95_ms)."""
    for _ in range(warmup):
        predict_fn(sample_frame)

    timings_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_fn(sample_frame)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 95))
