import numpy as np
import torch


class DepthEstimator:
    """Wraps MiDaS for monocular depth estimation."""

    def __init__(self, model_type: str = "MiDaS_small"):
        self.model_type = model_type
        self.depth_scale: float = 1.0
        self._model = None
        self._transform = None
        # Force CPU to avoid GPU memory competition with YOLO
        self._device = torch.device("cpu")

    def _load_model(self):
        if self._model is not None:
            return
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self._model.to(self._device)
        self._model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "MiDaS_small":
            self._transform = midas_transforms.small_transform
        else:
            self._transform = midas_transforms.dpt_transform

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Run depth estimation on a BGR frame. Returns float32 depth map (same H/W as input)."""
        self._load_model()
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy().astype(np.float32)
        depth = np.maximum(depth, 0.0)
        return depth

    def to_metric(self, relative_depth: np.ndarray) -> np.ndarray:
        """Convert relative inverse depth to metric distance (cm)."""
        safe = np.maximum(relative_depth, 1e-6)
        return (self.depth_scale / safe).astype(np.float32)

    def calibrate_scale(self, relative_depth: np.ndarray, pixel: tuple[int, int], real_distance_cm: float):
        """Set depth_scale using a known real-world measurement."""
        y, x = pixel[1], pixel[0]
        rel_val = float(relative_depth[y, x])
        if rel_val > 0:
            self.depth_scale = real_distance_cm * rel_val
