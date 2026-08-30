"""
Builds the training set for the classification head:

  1. Reads the ground-truth knot boxes (not the detector's predictions --
     we want the classifier to learn from clean crops, decoupled from
     whatever localization error the detector has).
  2. Synthesizes a few randomly-jittered crops per *training* image (small
     shifts + scale jitter + horizontal flip) to squeeze more signal out of
     only ~150 images -- cheap here because DINOv2 stays frozen, so this
     can't cause the kind of overfitting a trainable CNN's augmentation
     would. Validation crops are left clean for an honest readout.
  3. Runs every crop through the frozen DINOv2 backbone.
  4. Saves features + labels + which split (train/val) each row belongs to.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from data_prep.make_splits import load_annotations
from models.feature_extractor import DinoV2FeatureExtractor


def jittered_crop(image: Image.Image, box, pad_ratio: float, rng: random.Random) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * pad_ratio, h * pad_ratio

    jitter_x = rng.uniform(-0.08, 0.08) * w
    jitter_y = rng.uniform(-0.08, 0.08) * h
    scale = rng.uniform(0.95, 1.15)

    cx, cy = (x1 + x2) / 2 + jitter_x, (y1 + y2) / 2 + jitter_y
    half_w = (w / 2 + pad_x) * scale
    half_h = (h / 2 + pad_y) * scale

    crop_box = (
        max(0, cx - half_w), max(0, cy - half_h),
        min(image.width, cx + half_w), min(image.height, cy + half_h),
    )
    crop = image.crop(crop_box)
    if rng.random() < 0.5:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    return crop


def main():
    if not config.SPLIT_FILE.exists():
        raise FileNotFoundError("Run data_prep/make_splits.py first.")
    split = json.loads(config.SPLIT_FILE.read_text())
    file_to_split = {f: "train" for f in split["train"]}
    file_to_split.update({f: "val" for f in split["val"]})

    df = load_annotations()
    extractor = DinoV2FeatureExtractor()
    rng = random.Random(config.RANDOM_SEED)

    features, labels, filenames, splits = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting DINOv2 features"):
        # Backgrounds train YOLO only; they have no knot crop for the
        # correct/incorrect classifier.
        if row.is_background:
            continue
        img_path = config.RAW_IMAGES_DIR / row["filename"]
        if not img_path.exists() or row["filename"] not in file_to_split:
            continue
        this_split = file_to_split[row["filename"]]
        box = (row.x1, row.y1, row.x2, row.y2)

        with Image.open(img_path).convert("RGB") as image:
            crops = [image.crop(box)]  # always include one clean crop
            if this_split == "train":
                crops += [
                    jittered_crop(image, box, config.CROP_PAD_RATIO, rng)
                    for _ in range(config.CLASSIFIER_CROPS_PER_IMAGE)
                ]

            for crop in crops:
                features.append(extractor.extract(crop))
                labels.append(row["label"])
                filenames.append(row["filename"])
                splits.append(this_split)

    X = np.stack(features)
    y = np.array(labels)
    splits_arr = np.array(splits)

    config.CLASSIFIER_FEATURES_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.CLASSIFIER_FEATURES_NPZ,
        X=X, y=y,
        filenames=np.array(filenames),
        splits=splits_arr,
    )
    print(f"Saved {X.shape[0]} feature vectors ({X.shape[1]}-d) to {config.CLASSIFIER_FEATURES_NPZ}")
    print(f"  train rows: {(splits_arr == 'train').sum()}  |  val rows: {(splits_arr == 'val').sum()}")


if __name__ == "__main__":
    main()
