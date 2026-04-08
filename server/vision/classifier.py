import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)


def letterbox(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Letterbox resize: pad to square then resize to target, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CatClassifier:
    def __init__(self):
        self.model: nn.Module | None = None
        self.class_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, weights_path: Path) -> bool:
        """Load a trained classifier. Returns True if successful."""
        if not weights_path.exists():
            logger.info("No classifier weights found at %s", weights_path)
            return False
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.class_names = checkpoint["class_names"]
            num_classes = len(self.class_names)

            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Loaded classifier with classes: %s", self.class_names)
            return True
        except Exception as e:
            logger.error("Failed to load classifier: %s", e)
            self.model = None
            return False

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """Classify a cat crop. Returns (cat_name, confidence).
        Returns ("Unknown", 0.0) if no model is loaded."""
        if self.model is None:
            return ("Unknown", 0.0)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        letterboxed = letterbox(rgb, 224)
        tensor = _transform(letterboxed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, idx = torch.max(probs, dim=1)
            cat_name = self.class_names[idx.item()]
            return (cat_name, float(confidence.item()))
