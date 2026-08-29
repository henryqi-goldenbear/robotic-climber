"""
Creates a single stratified train/val split of the labeled images and
writes it to raw_data/split.json. Both the YOLO dataset builder and the
classifier dataset builder read this file, so the detector and the
classifier are always evaluated on the *same* held-out images.

Expected input: raw_data/annotations.csv with columns:
    filename, x1, y1, x2, y2, label

- x1,y1,x2,y2: absolute pixel coordinates of the knot bounding box
- label: "correct"/"incorrect" (or 1/0)

See raw_data/annotations_SCHEMA_EXAMPLE.csv for a worked example.
Adapt `load_annotations()` below if your dataset uses a different schema
(e.g. a JSON export, or separate detection/classification label files).
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GOOD_DIR_NAMES = {"kaggle_good", "manual_good"}
BAD_DIR_NAMES = {"manual_bad"}


def _folder_label_fallback() -> pd.DataFrame:
    records = []
    for image_path in sorted(config.RAW_IMAGES_DIR.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue

        rel_path = image_path.relative_to(config.RAW_IMAGES_DIR).as_posix()
        top_level = Path(rel_path).parts[0]

        if top_level in GOOD_DIR_NAMES:
            label = 1
        elif top_level in BAD_DIR_NAMES:
            label = 0
        else:
            continue

        with Image.open(image_path) as im:
            w, h = im.size
        records.append({
            "filename": rel_path,
            "x1": 0,
            "y1": 0,
            "x2": w,
            "y2": h,
            "label": label,
        })

    if not records:
        raise FileNotFoundError(
            f"No labeled images found under {config.RAW_IMAGES_DIR}. "
            "Expected folders such as kaggle_good/, manual_good/, and manual_bad/."
        )

    df = pd.DataFrame(records)
    config.ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ANNOTATIONS_CSV, index=False)
    return df


def load_annotations() -> pd.DataFrame:
    if not config.ANNOTATIONS_CSV.exists():
        df = _folder_label_fallback()
    else:
        df = pd.read_csv(config.ANNOTATIONS_CSV)

    label_map = {"correct": 1, "incorrect": 0, "1": 1, "0": 0, 1: 1, 0: 0}
    df["label"] = df["label"].astype(str).str.lower().map(lambda v: label_map.get(v, v))
    df["label"] = df["label"].astype(int)
    return df


def main():
    import json

    df = load_annotations()
    print(f"Loaded {len(df)} annotated images "
          f"({(df.label == 1).sum()} correct / {(df.label == 0).sum()} incorrect)")

    train_files, val_files = train_test_split(
        df["filename"].tolist(),
        test_size=1 - config.TRAIN_VAL_SPLIT,
        stratify=df["label"],
        random_state=config.RANDOM_SEED,
    )

    split = {"train": sorted(train_files), "val": sorted(val_files)}
    config.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Train: {len(train_files)} images | Val: {len(val_files)} images")
    print(f"Split written to {config.SPLIT_FILE}")
    print("Note: with N this small, treat the val split as a sanity check, "
          "not the headline number -- evaluate.py uses k-fold CV on the "
          "training crops for the real performance estimate.")


if __name__ == "__main__":
    main()
