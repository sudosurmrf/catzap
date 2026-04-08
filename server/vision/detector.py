import numpy as np
from ultralytics import YOLO

COCO_CAT_CLASS = 15


class CatDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect cats in a frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of dicts with 'bbox' (normalized [x1,y1,x2,y2]) and 'confidence'.
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            xyxyn = boxes.xyxyn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for i in range(len(classes)):
                if int(classes[i]) == COCO_CAT_CLASS and confs[i] >= self.confidence_threshold:
                    detections.append({
                        "bbox": xyxyn[i].tolist(),
                        "confidence": float(confs[i]),
                    })

        return detections
