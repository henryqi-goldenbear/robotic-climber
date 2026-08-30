#!/usr/bin/env python3
"""Memorization control for YOLO dataset/training integrity.

This is not a real evaluation: train and validation intentionally contain the
same eight images. A healthy pipeline should obtain a very high mAP50 here.
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

import config


SAMPLE_COUNT = int(os.getenv("YOLO_OVERFIT_SAMPLES", "8"))
EPOCHS = int(os.getenv("YOLO_OVERFIT_EPOCHS", "100"))
ROOT = config.PROJECT_ROOT / "diagnostics" / "yolo_overfit"


def main() -> None:
    source_images = sorted(
        path for path in (config.YOLO_DATASET_DIR / "images" / "train").rglob("*")
        if path.is_file()
    )[:SAMPLE_COUNT]
    if len(source_images) < 2:
        raise RuntimeError("Need at least two training images for the overfit test.")

    if ROOT.exists():
        shutil.rmtree(ROOT)
    image_dir, label_dir = ROOT / "images", ROOT / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    original_image_root = config.YOLO_DATASET_DIR / "images" / "train"
    original_label_root = config.YOLO_DATASET_DIR / "labels" / "train"
    for index, source_image in enumerate(source_images):
        relative = source_image.relative_to(original_image_root)
        source_label = original_label_root / relative.with_suffix(".txt")
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing matching label: {source_label}")
        target_image = image_dir / f"{index:02d}{source_image.suffix.lower()}"
        target_label = label_dir / f"{index:02d}.txt"
        shutil.copy2(source_image, target_image)
        shutil.copy2(source_label, target_label)

    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n  0: knot\n"
    )
    print(f"Overfit control: {len(source_images)} identical train/val images, {EPOCHS} epochs")

    model = YOLO(config.YOLO_BASE_WEIGHTS)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        patience=EPOCHS,
        imgsz=config.YOLO_IMG_SIZE,
        batch=min(4, len(source_images)),
        device=0 if config.DEVICE == "cuda" else config.DEVICE,
        project=str(ROOT / "runs"),
        name="memorization",
        exist_ok=True,
        seed=config.RANDOM_SEED,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        plots=False,
    )
    weights = ROOT / "runs" / "memorization" / "weights" / "best.pt"
    metrics = YOLO(str(weights)).val(
        data=str(yaml_path), imgsz=config.YOLO_IMG_SIZE,
        batch=min(4, len(source_images)), plots=False,
        device=0 if config.DEVICE == "cuda" else config.DEVICE,
    )
    print(f"Overfit mAP50={metrics.box.map50:.3f}; mAP50-95={metrics.box.map:.3f}")
    print("Expected: mAP50 close to 1.0. A low value means do not tune the full run yet; "
          "repair the labels/data conversion first.")


if __name__ == "__main__":
    main()
