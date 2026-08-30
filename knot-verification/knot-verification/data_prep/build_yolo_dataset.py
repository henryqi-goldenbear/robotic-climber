"""
Converts raw_data/annotations.csv (+ raw_data/split.json) into the
image/label folder structure and data.yaml that Ultralytics YOLO expects.

Run after make_splits.py.
"""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from data_prep.make_splits import load_annotations


def yolo_line(x1, y1, x2, y2, img_w, img_h, class_id=0) -> str:
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_split(df, filenames, split_name: str):
    img_out = config.YOLO_DATASET_DIR / "images" / split_name
    lbl_out = config.YOLO_DATASET_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rows = df[df["filename"].isin(filenames)]
    written = 0
    for _, row in rows.iterrows():
        src = config.RAW_IMAGES_DIR / row["filename"]
        if not src.exists():
            print(f"  ! skipping missing file {src}")
            continue
        with Image.open(src) as im:
            img_w, img_h = im.size
        # Filenames in annotations.csv are paths relative to raw_data/images
        # (for example, "kaggle_good/Loose/IMG_7059.jpg").  Preserve that
        # layout in both YOLO directories and create the matching parents.
        # YOLO resolves labels by replacing "images" with "labels", so the
        # relative paths must remain identical.
        image_out = img_out / row["filename"]
        label_out = lbl_out / (Path(row["filename"]).with_suffix(".txt"))
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, image_out)
        # An empty YOLO label file marks an image as background: it has no
        # knot box and trains the detector to reject empty frames.
        if row.is_background:
            label_out.write_text("")
        else:
            label_txt = yolo_line(row.x1, row.y1, row.x2, row.y2, img_w, img_h)
            label_out.write_text(label_txt + "\n")
        written += 1
    print(f"  {split_name}: {written} images written to {img_out}")


def write_data_yaml():
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(config.YOLO_CLASS_NAMES))
    yaml_text = (
        f"path: {config.YOLO_DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    config.YOLO_DATA_YAML.write_text(yaml_text)
    print(f"data.yaml written to {config.YOLO_DATA_YAML}")


def main():
    if not config.SPLIT_FILE.exists():
        raise FileNotFoundError("Run data_prep/make_splits.py first.")
    split = json.loads(config.SPLIT_FILE.read_text())
    df = load_annotations()

    print("Building YOLO dataset...")
    build_split(df, split["train"], "train")
    build_split(df, split["val"], "val")
    write_data_yaml()


if __name__ == "__main__":
    main()
