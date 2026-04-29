"""NanoDet-Plus eval-harness adapter.

Mirrors :class:`server.vision.detector.CatDetector` and the other adapters in
this package: ``predict(frame: np.ndarray) -> list[{"bbox": [x1,y1,x2,y2
normalized], "confidence": float}]``. Single-class output: COCO ``cat`` (id 15)
is re-emitted as id ``0``; every other COCO class is dropped. Mirrors the
auto-labeler's class remap (US-002).

Two backends, dispatched on file extension:

* ``.pth`` / ``.ckpt`` — uses the upstream ``nanodet`` Python package via
  ``nanodet.model.arch.build_model`` + ``load_model_weight``. Requires the
  isolated venv from :mod:`training.edge.nanodet`'s ``setup_venv.sh`` because
  upstream pins ``torch<2.0``. Lazy-imported so this module stays importable
  in the main env.
* ``.onnx`` — uses ``onnxruntime``. NanoDet-Plus' exported ONNX has a single
  output of shape ``(1, num_priors, num_classes + 4*(reg_max+1))`` with
  ``cls_pred`` already sigmoid'd (see ``nanodet.model.head.NanoDetPlusHead._forward_onnx``).
  We split, project the DFL distribution to per-side distances, decode via
  ``distance2bbox(prior_centers, distances)``, NMS within the cat channel,
  filter by ``confidence_threshold``, and normalize bboxes by the input size.

Pre/post-process knobs are documented inline. The decoder is self-contained
so the ONNX path runs without the ``nanodet`` Python package.

References:
    - upstream head:    nanodet.model.head.nanodet_plus_head.NanoDetPlusHead
    - upstream decode:  nanodet.util.bbox.distance2bbox
    - eval-harness contract: training/edge/eval/adapters/base.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

COCO_CAT_CLASS = 15  # mirrors server/vision/detector.py:4

# NanoDet-Plus standard preprocess (matches `data.val.pipeline.normalize` in the
# upstream YAMLs). Mean/std are RGB, in [0, 255] scale (NanoDet does not divide
# by 255 first; it normalizes raw uint8 channel values).
_NANODET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_NANODET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
# These are BGR-ordered to match cv2 frames; we flip in :func:`_preprocess`.

# NanoDet-Plus ONNX defaults. ``reg_max`` is the per-side bin count minus 1
# (so 8 distance bins per side); ``strides`` are the FPN levels NanoDet-Plus
# emits.
DEFAULT_REG_MAX = 7
DEFAULT_STRIDES: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_NUM_CLASSES = 80


def _is_onnx_path(path: str | Path) -> bool:
    return str(path).lower().endswith(".onnx")


def _is_tflite_path(path: str | Path) -> bool:
    return str(path).lower().endswith(".tflite")


def _preprocess(
    frame: np.ndarray,
    imgsz: int,
    mean: np.ndarray = _NANODET_MEAN,
    std: np.ndarray = _NANODET_STD,
) -> np.ndarray:
    """Resize, BGR->RGB, NanoDet normalize, NCHW.

    Mirrors :class:`nanodet.data.transform.color.color_aug_and_norm` in eval
    mode (no aug, just normalize).
    """
    import cv2

    if frame.shape[:2] != (imgsz, imgsz):
        frame = cv2.resize(frame, (imgsz, imgsz))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    # NanoDet's mean/std are listed BGR in their config; flip for RGB.
    mean_rgb = mean[::-1]
    std_rgb = std[::-1]
    norm = (rgb - mean_rgb) / std_rgb
    chw = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)
    return chw


def _generate_center_priors(
    imgsz: int, strides: Sequence[int]
) -> np.ndarray:
    """Replicates ``NanoDetPlusHead.get_single_level_center_priors`` per stride.

    Returns an ``(N, 4)`` array of ``[cx, cy, stride, stride]``. ``cx``/``cy``
    are in input-pixel coordinates.
    """
    import math

    levels: list[np.ndarray] = []
    for stride in strides:
        h = math.ceil(imgsz / stride)
        w = math.ceil(imgsz / stride)
        x = np.arange(w, dtype=np.float32) * stride
        y = np.arange(h, dtype=np.float32) * stride
        ys, xs = np.meshgrid(y, x, indexing="ij")
        priors = np.stack(
            [xs.flatten(), ys.flatten(),
             np.full_like(xs.flatten(), stride),
             np.full_like(xs.flatten(), stride)],
            axis=-1,
        )
        levels.append(priors)
    return np.concatenate(levels, axis=0)


def _distribution_project(reg: np.ndarray, reg_max: int) -> np.ndarray:
    """Softmax over the per-side bin dimension, then project onto integer distances.

    ``reg`` has shape ``(N, 4*(reg_max+1))``; output is ``(N, 4)``.
    """
    bins = reg_max + 1
    n = reg.shape[0]
    reshaped = reg.reshape(n, 4, bins)
    # numerically-stable softmax
    shifted = reshaped - reshaped.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    soft = exp / exp.sum(axis=-1, keepdims=True)
    proj = np.arange(bins, dtype=np.float32)
    return (soft * proj).sum(axis=-1)


def _distance2bbox(
    centers: np.ndarray, distances: np.ndarray, max_shape: int
) -> np.ndarray:
    """Centers ``(N, 2)`` + per-side distances ``(N, 4)`` -> xyxy ``(N, 4)`` clipped to image."""
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    out = np.stack([x1, y1, x2, y2], axis=-1)
    return np.clip(out, 0.0, float(max_shape))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS, returns indices to keep, ordered by descending score."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = max(0.0, boxes[i, 2] - boxes[i, 0]) * max(0.0, boxes[i, 3] - boxes[i, 1])
        area_rest = (
            np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0])
            * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        )
        union = area_i + area_rest - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_thr]
    return np.array(keep, dtype=np.int64)


def decode_nanodet_output(
    output: np.ndarray,
    imgsz: int,
    confidence_threshold: float,
    iou_threshold: float = 0.6,
    target_class: int = COCO_CAT_CLASS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    reg_max: int = DEFAULT_REG_MAX,
    strides: Sequence[int] = DEFAULT_STRIDES,
    max_detections: int = 100,
) -> list[dict]:
    """Decode the upstream ``_forward_onnx`` output to single-class cat detections.

    Output shape is ``(1, num_priors, num_classes + 4*(reg_max+1))``. ``cls_pred``
    is already sigmoid'd; ``reg_pred`` is the DFL distribution. We:

    1. split along the channel axis,
    2. project the distribution to per-side distances and multiply by stride
       to get pixel distances,
    3. decode with prior centers via ``distance2bbox``,
    4. take the ``target_class`` channel, threshold, NMS, normalize.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return []
    expected_channels = num_classes + 4 * (reg_max + 1)
    if arr.shape[1] != expected_channels:
        # Some exports may emit (channels, num_priors); transpose to match.
        if arr.shape[0] == expected_channels:
            arr = arr.T
        else:
            return []

    cls_scores = arr[:, :num_classes]
    reg_dist = arr[:, num_classes:]

    # Re-derive priors so we can map each prior's row to its stride. NanoDet
    # concatenates priors across strides in order; the per-stride bin sizes
    # are ceil(imgsz/stride)**2.
    priors = _generate_center_priors(imgsz, strides)
    if priors.shape[0] != arr.shape[0]:
        # Falls back to a clamped subset so partial-shape ONNX exports still
        # produce usable output rather than silently going to mAP=0.
        n = min(priors.shape[0], arr.shape[0])
        priors = priors[:n]
        cls_scores = cls_scores[:n]
        reg_dist = reg_dist[:n]

    # _forward_onnx emits cls AFTER sigmoid; per-channel score is already in [0,1].
    cat_scores = cls_scores[:, target_class]
    keep = cat_scores >= confidence_threshold
    if not np.any(keep):
        return []
    priors_kept = priors[keep]
    cat_scores_kept = cat_scores[keep]
    reg_kept = reg_dist[keep]

    distances = _distribution_project(reg_kept, reg_max)
    distances = distances * priors_kept[:, 2:3]  # multiply by per-row stride

    boxes = _distance2bbox(priors_kept[:, :2], distances, max_shape=imgsz)
    nms_idx = _nms(boxes, cat_scores_kept, iou_threshold)
    if nms_idx.size > max_detections:
        nms_idx = nms_idx[:max_detections]

    out_list: list[dict] = []
    for i in nms_idx:
        x1, y1, x2, y2 = boxes[int(i)] / float(imgsz)
        out_list.append(
            {
                "bbox": [
                    float(np.clip(x1, 0.0, 1.0)),
                    float(np.clip(y1, 0.0, 1.0)),
                    float(np.clip(x2, 0.0, 1.0)),
                    float(np.clip(y2, 0.0, 1.0)),
                ],
                "confidence": float(cat_scores_kept[int(i)]),
            }
        )
    return out_list


def _post_quant_score_normalize(
    output: np.ndarray, num_classes: int = DEFAULT_NUM_CLASSES
) -> np.ndarray:
    """Re-sigmoid cls channels if dequantization pushed them out of [0,1].

    Upstream's ``_forward_onnx`` emits cls already-sigmoid'd, which lands in
    [0,1]. After ONNX -> TFLite int8 PTQ the per-tensor output scale is fitted
    to span both cls (sigmoid 0..1) and reg (DFL logits, can be large), so the
    dequantized cls scores frequently overflow [0,1] (or get clipped near 0
    if the scale is dominated by reg). When we observe that, push them back
    through a sigmoid so :func:`decode_nanodet_output`'s confidence threshold
    is meaningful.
    """
    arr = np.asarray(output)
    if arr.size == 0 or arr.ndim < 2:
        return arr
    last = arr.shape[-1] if arr.ndim >= 2 else None
    if last is None:
        return arr
    # Determine whether channels are last-dim (priors, ch) vs first-dim
    # (channels, priors). The decoder handles transposition, but the cls
    # channels are always the leading num_classes channels in the ch axis.
    ch_axis = -1 if last == num_classes + 4 * (DEFAULT_REG_MAX + 1) else 0
    cls_slice: tuple = (slice(None),) * (arr.ndim - 1) + (slice(0, num_classes),)
    if ch_axis == 0:
        # transposed layout: (..., channels, priors)
        cls_slice = (Ellipsis, slice(0, num_classes), slice(None))
    cls = arr[cls_slice]
    if cls.size == 0:
        return arr
    if float(cls.min()) >= 0.0 and float(cls.max()) <= 1.0:
        return arr
    sig = 1.0 / (1.0 + np.exp(-cls.astype(np.float32)))
    arr = arr.copy()
    arr[cls_slice] = sig
    return arr


def remap_nanodet_detections_to_cat_only(
    detections: Sequence[Sequence[float]],
    imgsz: int,
    confidence_threshold: float,
    target_class: int = COCO_CAT_CLASS,
) -> list[dict]:
    """Remap NanoDet's ``[class_id, x1, y1, x2, y2, conf]`` -> single-class.

    Drops any detection whose ``class_id`` differs from ``target_class``;
    re-emits surviving ones as id 0. Used for tests and for the ``.pth``
    backend, which returns this list-of-list shape directly from the upstream
    inference helper.
    """
    out: list[dict] = []
    for det in detections:
        if len(det) < 6:
            continue
        cls = int(det[0])
        if cls != target_class:
            continue
        x1, y1, x2, y2, conf = float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])
        if conf < confidence_threshold:
            continue
        out.append(
            {
                "bbox": [
                    float(np.clip(x1 / imgsz, 0.0, 1.0)),
                    float(np.clip(y1 / imgsz, 0.0, 1.0)),
                    float(np.clip(x2 / imgsz, 0.0, 1.0)),
                    float(np.clip(y2 / imgsz, 0.0, 1.0)),
                ],
                "confidence": conf,
            }
        )
    return out


class NanodetAdapter:
    """Eval-harness adapter for NanoDet-Plus pretrained or fine-tuned checkpoints."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 416,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.6,
        num_classes: int = DEFAULT_NUM_CLASSES,
        reg_max: int = DEFAULT_REG_MAX,
        strides: Sequence[int] = DEFAULT_STRIDES,
        target_class: int = COCO_CAT_CLASS,
        backend: str | None = None,
        session_factory: Callable[..., Any] | None = None,
        torch_factory: Callable[..., Any] | None = None,
        interpreter_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self.imgsz = imgsz
        self.input_hw = (imgsz, imgsz)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = tuple(strides)
        self.target_class = target_class
        if backend is None:
            if _is_tflite_path(model_path):
                backend = "tflite"
            elif _is_onnx_path(model_path):
                backend = "onnx"
            else:
                backend = "pytorch"
        self.backend = backend

        self._session = None
        self._input_name: str | None = None
        self._torch_model = None
        self._interpreter = None
        self._tflite_input_detail: dict | None = None
        self._tflite_output_detail: dict | None = None
        self._cached_params: int = 0

        if backend == "onnx":
            self._init_onnx(session_factory)
        elif backend == "pytorch":
            self._init_pytorch(torch_factory)
        elif backend == "tflite":
            self._init_tflite(interpreter_factory)
        else:
            raise ValueError(f"unsupported nanodet backend: {backend!r}")

    def _init_onnx(self, session_factory: Callable[..., Any] | None) -> None:
        if session_factory is not None:
            self._session = session_factory(self.model_path)
        else:
            import onnxruntime as ort  # type: ignore

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        try:
            self._input_name = self._session.get_inputs()[0].name
        except Exception:
            self._input_name = "data"
        self._cached_params = self._count_onnx_params()

    def _init_pytorch(self, torch_factory: Callable[..., Any] | None) -> None:
        # Lazy import — the upstream nanodet package needs the isolated venv.
        if torch_factory is not None:
            self._torch_model = torch_factory(self.model_path)
        else:
            try:
                import torch  # type: ignore
                from nanodet.model.arch import build_model  # type: ignore
                from nanodet.util import Logger, cfg, load_config, load_model_weight  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "NanoDet pytorch backend requires the isolated venv "
                    "(see training/edge/nanodet/setup_venv.sh). For main-env "
                    "eval use the .onnx checkpoint instead."
                ) from exc
            cfg_path = os.environ.get(
                "NANODET_CONFIG_PATH",
                "training/edge/nanodet/upstream/config/nanodet-plus-m_416.yml",
            )
            load_config(cfg, cfg_path)
            logger = Logger(-1, cfg.save_dir, False)
            model = build_model(cfg.model)
            ckpt = torch.load(self.model_path, map_location="cpu")
            load_model_weight(model, ckpt, logger)
            model.eval()
            self._torch_model = model
        try:
            self._cached_params = int(
                sum(p.numel() for p in self._torch_model.parameters())  # type: ignore[union-attr]
            )
        except Exception:
            self._cached_params = 0

    def _init_tflite(
        self, interpreter_factory: Callable[..., Any] | None
    ) -> None:
        """Boot a tf.lite Interpreter for the (possibly-INT8) NanoDet model.

        US-009 produces a NanoDet-Plus-m INT8 .tflite via the same
        ``onnx2tf`` -> ``tf.lite.TFLiteConverter`` pipeline as YOLOv8n US-004.
        The model has ONE output tensor of shape
        ``(1, num_priors, num_classes + 4*(reg_max+1))`` (or its transpose) —
        same as the ONNX path. We dequantize the int8 output and feed the
        existing :func:`decode_nanodet_output` decoder.
        """
        if interpreter_factory is not None:
            self._interpreter = interpreter_factory(self.model_path)
        else:
            try:
                from tensorflow.lite.python.interpreter import (  # type: ignore
                    Interpreter,
                    OpResolverType,
                )
            except Exception:
                try:
                    from tflite_runtime.interpreter import Interpreter  # type: ignore

                    OpResolverType = None  # type: ignore
                except Exception as exc:
                    raise ImportError(
                        "tensorflow-cpu (>=2.15) or tflite-runtime is required "
                        "for the NanoDet TFLite backend"
                    ) from exc
            # Skip XNNPACK delegate. NanoDet's INT8 export contains a Pad-after-
            # Conv pattern that XNNPACK fails to lower (observed:
            # "Node N (TfLiteXNNPackDelegate) failed to prepare"). The built-in
            # reference ops handle the same graph correctly.
            kwargs: dict[str, Any] = {"model_path": self.model_path, "num_threads": 1}
            if OpResolverType is not None:
                kwargs["experimental_op_resolver_type"] = (
                    OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                )
            self._interpreter = Interpreter(**kwargs)
        self._interpreter.allocate_tensors()  # type: ignore[union-attr]
        self._tflite_input_detail = self._interpreter.get_input_details()[0]  # type: ignore[union-attr]
        self._tflite_output_detail = self._interpreter.get_output_details()[0]  # type: ignore[union-attr]
        self._cached_params = self._count_tflite_params()

    def _count_onnx_params(self) -> int:
        try:
            import onnx  # type: ignore

            model = onnx.load(self.model_path)
            total = 0
            for init in model.graph.initializer:
                n = 1
                for d in init.dims:
                    n *= int(d)
                total += n
            return int(total)
        except Exception:
            return 0

    def predict(self, frame: np.ndarray) -> list[dict]:
        if self.backend == "onnx":
            return self._predict_onnx(frame)
        if self.backend == "tflite":
            return self._predict_tflite(frame)
        return self._predict_pytorch(frame)

    def _count_tflite_params(self) -> int:
        try:
            total = 0
            for t in self._interpreter.get_tensor_details():  # type: ignore[union-attr]
                shape = t.get("shape", [])
                if len(shape) > 0 and t.get("quantization", (0.0, 0))[0] != 0:
                    n = 1
                    for d in shape:
                        n *= int(d)
                    total += n
            return int(total)
        except Exception:
            return 0

    def _predict_tflite(self, frame: np.ndarray) -> list[dict]:
        x = _preprocess(frame, self.imgsz)
        # onnx2tf produces NHWC layout; the converter rewrites the input to
        # NHWC even though our preprocess outputs NCHW. Transpose to match
        # what the interpreter expects.
        in_detail = self._tflite_input_detail or {}
        in_shape = list(in_detail.get("shape", x.shape))
        if len(in_shape) == 4 and in_shape[-1] == 3:
            x = np.transpose(x, (0, 2, 3, 1))
        in_dt = in_detail.get("dtype", np.float32)
        if in_dt == np.int8:
            scale, zero = in_detail.get("quantization", (0.0, 0))
            if scale and scale != 0:
                x = (x / scale + zero).round().astype(np.int8)
            else:
                x = x.astype(np.int8)
        else:
            x = x.astype(in_dt)
        self._interpreter.set_tensor(in_detail["index"], x)  # type: ignore[union-attr]
        self._interpreter.invoke()  # type: ignore[union-attr]
        out_detail = self._tflite_output_detail or {}
        raw = self._interpreter.get_tensor(out_detail["index"])  # type: ignore[union-attr]
        if raw.dtype in (np.int8, np.uint8):
            scale, zero = out_detail.get("quantization", (0.0, 0))
            if scale and scale != 0:
                deq = (raw.astype(np.float32) - zero) * scale
            else:
                deq = raw.astype(np.float32)
        else:
            deq = raw.astype(np.float32)
        # The post-quant cls scores are no longer in [0,1] (the converter folds
        # the output sigmoid into the per-tensor scale/zero-point), so apply a
        # sigmoid before threshold + NMS to land back in score-space.
        deq = _post_quant_score_normalize(deq, num_classes=self.num_classes)
        return decode_nanodet_output(
            deq,
            imgsz=self.imgsz,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            target_class=self.target_class,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            strides=self.strides,
        )

    def _predict_onnx(self, frame: np.ndarray) -> list[dict]:
        x = _preprocess(frame, self.imgsz)
        outputs = self._session.run(None, {self._input_name: x})  # type: ignore[union-attr]
        return decode_nanodet_output(
            outputs[0],
            imgsz=self.imgsz,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            target_class=self.target_class,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            strides=self.strides,
        )

    def _predict_pytorch(self, frame: np.ndarray) -> list[dict]:
        import torch  # type: ignore

        x = _preprocess(frame, self.imgsz)
        with torch.no_grad():
            tensor = torch.from_numpy(x)
            output = self._torch_model.inference(  # type: ignore[union-attr]
                {"img": tensor, "warp_matrix": [np.eye(3)],
                 "img_info": {"height": np.array([self.imgsz]),
                              "width": np.array([self.imgsz]),
                              "id": np.array([0])}}
            )
        # Upstream's ``inference`` returns ``{img_id: {class_id: [[x1,y1,x2,y2,conf], ...]}}``.
        flat: list[list[float]] = []
        for _img_id, per_class in output.items():
            for cls_id, dets in per_class.items():
                for det in dets:
                    flat.append([cls_id, det[0], det[1], det[2], det[3], det[4]])
        return remap_nanodet_detections_to_cat_only(
            flat,
            imgsz=self.imgsz,
            confidence_threshold=self.confidence_threshold,
            target_class=self.target_class,
        )

    def num_params(self) -> int:
        return int(self._cached_params)

    def num_flops(self) -> int:
        return 0


def load(model_path: str, imgsz: int = 416, **kw: Any) -> NanodetAdapter:
    return NanodetAdapter(model_path, imgsz=imgsz, **kw)
