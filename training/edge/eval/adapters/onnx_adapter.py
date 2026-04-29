"""ONNX Runtime adapter for exported YOLO graphs.

Output post-processing assumes the ultralytics ONNX export shape:
- Output 0: [1, 4 + num_classes, num_boxes] (xywh + class scores)
We single-class-flatten by taking max class score as confidence.
For models exported with ultralytics single_cls=True, num_classes=1.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class OnnxAdapter:
    def __init__(
        self,
        model_path: str,
        imgsz: int = 224,
        confidence_threshold: float = 0.25,
    ) -> None:
        import onnxruntime as ort  # type: ignore

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.input_hw = (imgsz, imgsz)
        self.confidence_threshold = confidence_threshold
        self._model_path = model_path

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        if frame.shape[:2] != (self.imgsz, self.imgsz):
            frame = cv2.resize(frame, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    def predict(self, frame: np.ndarray) -> list[dict]:
        x = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: x})
        return _decode_yolo_onnx(outputs[0], self.confidence_threshold, self.imgsz)

    def num_params(self) -> int:
        try:
            import onnx  # type: ignore

            model = onnx.load(self._model_path)
            total = 0
            for init in model.graph.initializer:
                n = 1
                for d in init.dims:
                    n *= int(d)
                total += n
            return int(total)
        except Exception:
            return 0

    def num_flops(self) -> int:
        return 0


def _decode_yolo_onnx(
    out: np.ndarray, conf_thr: float, imgsz: int = 224
) -> list[dict]:
    """Decode an ultralytics YOLO ONNX/TFLite output to normalized boxes.

    YOLOv8 raw output is xywh in INPUT pixel coordinates (0..imgsz), not
    normalized. The eval harness's GT labels are normalized [0,1], so we
    divide by imgsz before clipping. Without this scaling, all boxes get
    clamped to a degenerate point and mAP collapses to 0.
    """
    arr = np.asarray(out)
    if arr.ndim == 3 and arr.shape[1] >= 5 and arr.shape[1] < arr.shape[2]:
        arr = arr[0]
    elif arr.ndim == 3:
        arr = arr[0].T
    elif arr.ndim == 2:
        pass
    else:
        return []

    if arr.shape[0] < 5:
        return []

    xywh = arr[:4, :].astype(np.float32) / float(imgsz)
    cls_scores = arr[4:, :]
    conf = cls_scores.max(axis=0)

    keep = conf >= conf_thr
    xywh = xywh[:, keep]
    conf = conf[keep]

    out_list: list[dict] = []
    n = xywh.shape[1]
    for i in range(n):
        cx, cy, w, h = xywh[:, i]
        x1 = float(cx - w / 2.0)
        y1 = float(cy - h / 2.0)
        x2 = float(cx + w / 2.0)
        y2 = float(cy + h / 2.0)
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        out_list.append({"bbox": [x1, y1, x2, y2], "confidence": float(conf[i])})
    return out_list


def load(model_path: str, imgsz: int = 224, **kw: Any) -> OnnxAdapter:
    return OnnxAdapter(model_path, imgsz=imgsz, **kw)
