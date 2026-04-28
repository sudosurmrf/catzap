"""ultralytics.YOLO adapter — same predict() shape as CatDetector.

Single-class output: every detection above the confidence threshold is
re-emitted with class_id=0 (cat). Source classes other than COCO_CAT_CLASS=15
are dropped when the loaded checkpoint is a stock COCO YOLO; for fine-tuned
single-class checkpoints (US-003 onward) the only class is already cat so
the filter is a no-op.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

COCO_CAT_CLASS = 15


class YoloPytorchAdapter:
    def __init__(
        self,
        model_path: str,
        imgsz: int = 224,
        confidence_threshold: float = 0.25,
        single_class_model: bool | None = None,
    ) -> None:
        torch.set_num_threads(1)
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.input_hw = (imgsz, imgsz)
        self.confidence_threshold = confidence_threshold
        if single_class_model is None:
            try:
                names = getattr(self.model, "names", {}) or {}
                single_class_model = len(names) == 1
            except Exception:
                single_class_model = False
        self._single_class = single_class_model
        self._cached_params: int | None = None
        self._cached_flops: int | None = None
        self._compute_params_flops()

    def _compute_params_flops(self) -> None:
        """Cache params and FLOPs before any predict() call.

        ultralytics fuses Conv+BN at first predict(), which changes the
        param count and breaks thop's hook tracing. Snapshot here while
        the model is still fresh.
        """
        try:
            inner = self.model.model
            self._cached_params = int(sum(p.numel() for p in inner.parameters()))
        except Exception:
            self._cached_params = 0
        try:
            from thop import profile

            inner = self.model.model
            inner.eval()
            dummy = torch.zeros(1, 3, self.imgsz, self.imgsz)
            with torch.no_grad():
                flops, _ = profile(inner, inputs=(dummy,), verbose=False)
            self._cached_flops = int(flops)
        except Exception:
            self._cached_flops = 0

    def predict(self, frame: np.ndarray) -> list[dict]:
        with torch.no_grad():
            results = self.model(frame, imgsz=self.imgsz, verbose=False)
        out: list[dict] = []
        for result in results:
            boxes = result.boxes
            xyxyn = boxes.xyxyn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for i in range(len(classes)):
                cls = int(classes[i])
                conf = float(confs[i])
                if conf < self.confidence_threshold:
                    continue
                if not self._single_class and cls != COCO_CAT_CLASS:
                    continue
                out.append({"bbox": xyxyn[i].tolist(), "confidence": conf})
        return out

    def num_params(self) -> int:
        return int(self._cached_params or 0)

    def num_flops(self) -> int:
        return int(self._cached_flops or 0)


def load(model_path: str, imgsz: int = 224, **kw: Any) -> YoloPytorchAdapter:
    return YoloPytorchAdapter(model_path, imgsz=imgsz, **kw)
