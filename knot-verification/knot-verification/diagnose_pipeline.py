#!/usr/bin/env python3
"""Test saved detector and classifier components without retraining.

Run this after ``run_training_pipeline.py`` to distinguish a detector issue
from a DINOv2/classifier issue.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Keep a Job from inheriting the former invalid checkpoint name.
os.environ["DINO_MODEL_NAME"] = "facebook/dinov2-small"

import config
from data_prep.make_splits import load_annotations
from models.classifier_head import KnotClassifierHead
from models.detector import KnotDetector
from models.feature_extractor import DinoV2FeatureExtractor


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    print(f"OK  {description}: {path}")


def box_iou(box_a, box_b) -> float:
    """IoU for two ``(x1, y1, x2, y2)`` boxes."""
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def main() -> None:
    print("=== Saved artifacts ===")
    require(config.YOLO_WEIGHTS_OUT, "detector weights")
    require(config.CLASSIFIER_FEATURES_NPZ, "classifier features")
    require(config.CLASSIFIER_MODEL_OUT, "classifier model")
    require(config.SPLIT_FILE, "train/validation split")

    data = np.load(config.CLASSIFIER_FEATURES_NPZ, allow_pickle=True)
    X, splits = data["X"], data["splits"]
    print(f"OK  features: {X.shape[0]} vectors × {X.shape[1]} dimensions "
          f"({(splits == 'train').sum()} train, {(splits == 'val').sum()} val)")

    classifier = KnotClassifierHead.load()
    val_features = X[splits == "val"]
    probs = classifier.predict_proba(val_features[:3])
    print("OK  classifier probabilities for first validation crops:", np.round(probs, 3).tolist())

    split = json.loads(config.SPLIT_FILE.read_text())
    df = load_annotations().set_index("filename")
    val_files = split["val"]

    print("\n=== Detector confidence sweep ===")
    detector = KnotDetector(conf_thres=0.001)
    metrics = detector.model.val(
        data=str(config.YOLO_DATA_YAML),
        split="val",
        imgsz=config.YOLO_IMG_SIZE,
        batch=config.YOLO_BATCH,
        device=0 if config.DEVICE == "cuda" else config.DEVICE,
        plots=False,
        verbose=False,
    )
    print(f"YOLO validation: precision={metrics.box.mp:.3f}, recall={metrics.box.mr:.3f}, "
          f"mAP50={metrics.box.map50:.3f}, mAP50-95={metrics.box.map:.3f}")
    best_scores, best_ious = [], []
    for filename in val_files:
        with Image.open(config.RAW_IMAGES_DIR / filename) as source_image:
            image = source_image.convert("RGB")
        result = detector.model.predict(source=np.array(image), conf=0.001, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or not len(boxes):
            best_scores.append(0.0)
            best_ious.append(0.0)
            continue
        best_idx = int(boxes.conf.argmax())
        best_scores.append(float(boxes.conf[best_idx]))
        prediction = boxes.xyxy[best_idx].tolist()
        truth = df.loc[filename]
        best_ious.append(box_iou(prediction, (truth.x1, truth.y1, truth.x2, truth.y2)))

    scores = np.array(best_scores)
    print(f"Validation images: {len(scores)}")
    print(f"Best-confidence range: {scores.min():.4f}–{scores.max():.4f}; median {np.median(scores):.4f}")
    for threshold in (0.001, 0.05, 0.10, 0.35):
        print(f"  >= {threshold:.3f}: {(scores >= threshold).sum()}/{len(scores)}")
    ious = np.array(best_ious)
    print(f"Best-box IoU with ground truth: median {np.median(ious):.3f}; max {ious.max():.3f}")

    print("\n=== DINOv2 + classifier using a ground-truth crop ===")
    filename = val_files[0]
    row = df.loc[filename]
    image = Image.open(config.RAW_IMAGES_DIR / filename).convert("RGB")
    crop = image.crop((row.x1, row.y1, row.x2, row.y2))
    feature = DinoV2FeatureExtractor().extract(crop)
    probability = classifier.predict_proba(feature)[0]
    predicted = config.CLASSIFIER_LABELS[int(probability.argmax())]
    print(f"Image: {filename}")
    print(f"Ground truth: {config.CLASSIFIER_LABELS[int(row.label)]}")
    print(f"Prediction: {predicted}; probabilities [incorrect, correct]: {np.round(probability, 3).tolist()}")


if __name__ == "__main__":
    main()
