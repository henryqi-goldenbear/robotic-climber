"""
Fine-tunes YOLOv10-Nano on the knot-localization task.

With <150 box labels, we lean entirely on transfer learning from COCO
weights and Ultralytics' built-in augmentation; there's no architecture
tuning to do here, just enough epochs (capped by early stopping) to adapt
the last layers to detecting "knot" as a class.

Run order: data_prep/make_splits.py -> data_prep/build_yolo_dataset.py -> this.
"""
from ultralytics import YOLO

import config


def main():
    model = YOLO(config.YOLO_BASE_WEIGHTS)  # COCO-pretrained, auto-downloaded on first run
    model.train(
        data=str(config.YOLO_DATA_YAML),
        epochs=config.YOLO_EPOCHS,
        patience=config.YOLO_PATIENCE,
        imgsz=config.YOLO_IMG_SIZE,
        batch=config.YOLO_BATCH,
        project=str(config.YOLO_RUNS_DIR),
        name="knot_detector",
        seed=config.RANDOM_SEED,
        device=0 if config.DEVICE == "cuda" else config.DEVICE,
        exist_ok=True,
    )

    best_weights = config.YOLO_RUNS_DIR / "knot_detector" / "weights" / "best.pt"
    config.YOLO_WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    config.YOLO_WEIGHTS_OUT.write_bytes(best_weights.read_bytes())
    print(f"Best detector weights copied to {config.YOLO_WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
