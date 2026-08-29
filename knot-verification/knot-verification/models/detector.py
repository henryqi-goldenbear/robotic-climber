"""
Thin wrapper around the fine-tuned YOLOv10-Nano knot detector.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

import config


class KnotDetector:
    def __init__(self, weights_path: Path = config.YOLO_WEIGHTS_OUT, conf_thres: float = config.YOLO_CONF_THRES):
        self.model = YOLO(str(weights_path))
        self.conf_thres = conf_thres

    def detect_best_box(self, image: Image.Image):
        """Returns (x1, y1, x2, y2, confidence) for the highest-confidence
        knot detection, or None if nothing cleared the confidence threshold."""
        results = self.model.predict(source=np.array(image), conf=self.conf_thres, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        conf = float(boxes.conf[best_idx])
        return x1, y1, x2, y2, conf

    def crop(self, image: Image.Image, box, pad_ratio: float = config.CROP_PAD_RATIO) -> Image.Image:
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * pad_ratio, h * pad_ratio
        crop_box = (
            max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(image.width, x2 + pad_x), min(image.height, y2 + pad_y),
        )
        return image.crop(crop_box)
