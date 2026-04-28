"""EvalResult dataclass — schema for training/edge/results/<story-id>.json files.

See PRD section "Types & Interfaces" in .archon/ralph/edge-model-poc/prd.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ModelFormat = Literal["pytorch", "onnx", "tflite_fp32", "tflite_int8", "tflite_fp16"]


@dataclass
class EvalResult:
    story_id: str
    model_path: str
    model_format: str
    map50: float
    size_bytes: int
    params: int
    flops: int
    input_hw: tuple[int, int]
    latency_ms_p50: float
    latency_ms_p95: float
    val_images: int
    notes: str

    def to_json(self) -> str:
        d = asdict(self)
        d["input_hw"] = list(d["input_hw"])
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "EvalResult":
        d = json.loads(s)
        d["input_hw"] = tuple(d["input_hw"])
        return cls(**d)

    def write(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def read(cls, path: Path | str) -> "EvalResult":
        return cls.from_json(Path(path).read_text())
