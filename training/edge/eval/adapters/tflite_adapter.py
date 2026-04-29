"""TFLite adapter (fp32 and INT8 share the same wrapper).

INT8: input/output dtype is int8 (per US-004 export). We dequantize
input scale/zero-point from the model's quantization params, then re-
quantize the YOLO post-processed output back to floats for mAP.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class TfliteAdapter:
    def __init__(
        self,
        model_path: str,
        imgsz: int = 224,
        confidence_threshold: float = 0.25,
        is_int8: bool = False,
    ) -> None:
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        except Exception:
            try:
                from tflite_runtime.interpreter import Interpreter  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "tensorflow-cpu (>=2.15) or tflite-runtime is required for TFLite eval"
                ) from exc
        self.interpreter = Interpreter(model_path=model_path, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.imgsz = imgsz
        self.input_hw = (imgsz, imgsz)
        self.confidence_threshold = confidence_threshold
        self.is_int8 = is_int8
        self._model_path = model_path

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if frame.shape[:2] != (self.imgsz, self.imgsz):
            frame = cv2.resize(frame, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = x[None, ...]
        in_dt = self.input_details[0]["dtype"]
        if in_dt == np.int8 or self.is_int8:
            scale, zero = self.input_details[0].get("quantization", (0.0, 0))
            if scale and scale != 0:
                x = (x / scale + zero).round().astype(np.int8)
            else:
                x = x.astype(np.int8)
        else:
            x = x.astype(in_dt)
        return x

    def _dequantize(self, arr: np.ndarray, detail: dict) -> np.ndarray:
        if arr.dtype in (np.int8, np.uint8):
            scale, zero = detail.get("quantization", (0.0, 0))
            if scale and scale != 0:
                return (arr.astype(np.float32) - zero) * scale
        return arr.astype(np.float32)

    def predict(self, frame: np.ndarray) -> list[dict]:
        x = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]["index"])
        deq = self._dequantize(raw, self.output_details[0])
        from .onnx_adapter import _decode_yolo_onnx

        return _decode_yolo_onnx(deq, self.confidence_threshold, self.imgsz)

    def num_params(self) -> int:
        try:
            total = 0
            for t in self.interpreter.get_tensor_details():
                shape = t.get("shape", [])
                if len(shape) > 0 and t.get("quantization", (0.0, 0))[0] != 0:
                    n = 1
                    for d in shape:
                        n *= int(d)
                    total += n
            return int(total)
        except Exception:
            return 0

    def num_flops(self) -> int:
        return 0


def load(model_path: str, imgsz: int = 224, is_int8: bool = False, **kw: Any) -> TfliteAdapter:
    return TfliteAdapter(model_path, imgsz=imgsz, is_int8=is_int8, **kw)
