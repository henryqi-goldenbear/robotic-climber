"""
End-to-end inference: raw camera frame -> Correct / Incorrect.

    frame -> YOLOv10-N detect + crop -> DINOv2 (frozen) -> SVM -> label
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image

import config
from models.classifier_head import KnotClassifierHead
from models.detector import KnotDetector
from models.feature_extractor import DinoV2FeatureExtractor


@dataclass
class InspectionResult:
    label: Optional[str]                  # "Correct" / "Incorrect" / None if no knot found
    confidence: Optional[float]           # classifier probability of the predicted label
    detection_confidence: Optional[float]
    box: Optional[Tuple[float, float, float, float]]


class KnotInspectionPipeline:
    def __init__(self):
        self.detector = KnotDetector()
        self.extractor = DinoV2FeatureExtractor()
        self.classifier = KnotClassifierHead.load()

    def predict(self, image: Image.Image) -> InspectionResult:
        detection = self.detector.detect_best_box(image)
        if detection is None:
            # No knot found above the confidence threshold -- don't force a
            # classification on background/carabiner/climber.
            return InspectionResult(label=None, confidence=None, detection_confidence=None, box=None)

        box, det_conf = detection[:4], detection[4]
        crop = self.detector.crop(image, box)

        feat = self.extractor.extract(crop)
        proba = self.classifier.predict_proba(feat)[0]
        pred_idx = int(proba.argmax())

        return InspectionResult(
            label=config.CLASSIFIER_LABELS[pred_idx],
            confidence=float(proba[pred_idx]),
            detection_confidence=det_conf,
            box=box,
        )
