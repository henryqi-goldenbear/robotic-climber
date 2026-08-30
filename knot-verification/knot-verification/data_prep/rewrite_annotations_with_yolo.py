"""
Rewrite raw_data/annotations.csv boxes using Roboflow YOLOv8-OBB labels.

This keeps each row's filename/label and replaces x1,y1,x2,y2 with the
axis-aligned rectangle that encloses the OBB polygon.
"""
import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

from PIL import Image

# Add project root to import config.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def _canonical_basename(filename: str) -> str:
    """Normalize names so IMG_7055-Copy.jpg maps to IMG_7055.jpg."""
    stem, ext = Path(filename).stem, Path(filename).suffix.lower()
    lowered = stem.lower()
    for copy_suffix in ("-copy", "_copy", " copy"):
        if lowered.endswith(copy_suffix):
            stem = stem[: -len(copy_suffix)]
            break
    return f"{stem.lower()}{ext}"


def _label_basename_from_path(label_path: Path) -> str:
    """Map Roboflow label file names to original image basenames."""
    base = label_path.stem.split(".rf.", 1)[0]
    if base.endswith("_jpg"):
        return f"{base[:-4]}.jpg"
    if base.endswith("_jpeg"):
        return f"{base[:-5]}.jpeg"
    if base.endswith("_png"):
        return f"{base[:-4]}.png"
    return base


def _load_label_boxes(labels_dir: Path) -> dict[str, list[list[float]]]:
    boxes_by_key: dict[str, list[list[float]]] = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_base = _label_basename_from_path(label_path)
        key = _canonical_basename(image_base)
        polygons: list[list[float]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 9:
                raise ValueError(f"Invalid OBB line in {label_path}: {line}")
            coords = [float(v) for v in parts[1:]]
            polygons.append(coords)
        if not polygons:
            raise ValueError(f"No boxes found in label file: {label_path}")
        if key in boxes_by_key:
            raise ValueError(f"Duplicate label key after normalization: {key}")
        boxes_by_key[key] = polygons
    if not boxes_by_key:
        raise FileNotFoundError(f"No label files found in {labels_dir}")
    return boxes_by_key


def _obb_to_xyxy(coords: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    xs = [coords[i] * width for i in (0, 2, 4, 6)]
    ys = [coords[i] * height for i in (1, 3, 5, 7)]
    x1 = max(0, min(width - 1, math.floor(min(xs))))
    y1 = max(0, min(height - 1, math.floor(min(ys))))
    x2 = max(1, min(width, math.ceil(max(xs))))
    y2 = max(1, min(height, math.ceil(max(ys))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _choose_polygon(polygons: list[list[float]]) -> list[float]:
    # Keep the largest box if a file contains multiple OBB instances.
    def area(poly: list[float]) -> float:
        xs = [poly[i] for i in (0, 2, 4, 6)]
        ys = [poly[i] for i in (1, 3, 5, 7)]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    return max(polygons, key=area)


def rewrite_annotations(annotations_csv: Path, labels_dir: Path, images_root: Path, backup_path: Path | None) -> int:
    rows = list(csv.DictReader(annotations_csv.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows in {annotations_csv}")

    labels = _load_label_boxes(labels_dir)
    dims_cache: dict[Path, tuple[int, int]] = {}

    missing_labels: list[str] = []
    updated = 0
    for row in rows:
        rel_image = Path(row["filename"])
        image_path = images_root / rel_image
        if not image_path.exists():
            raise FileNotFoundError(f"Image referenced by annotations not found: {image_path}")
        if image_path not in dims_cache:
            with Image.open(image_path) as im:
                dims_cache[image_path] = im.size
        width, height = dims_cache[image_path]

        key = _canonical_basename(rel_image.name)
        polygons = labels.get(key)
        if polygons is None:
            missing_labels.append(row["filename"])
            continue

        poly = _choose_polygon(polygons)
        x1, y1, x2, y2 = _obb_to_xyxy(poly, width, height)
        row["x1"], row["y1"], row["x2"], row["y2"] = x1, y1, x2, y2
        updated += 1

    if missing_labels:
        sample = ", ".join(missing_labels[:5])
        raise KeyError(
            f"Missing OBB labels for {len(missing_labels)} annotation rows. "
            f"Examples: {sample}"
        )

    if backup_path is not None:
        shutil.copy2(annotations_csv, backup_path)

    with annotations_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "x1", "y1", "x2", "y2", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return updated


def _default_labels_dir() -> Path:
    candidates = [
        PROJECT_ROOT.parents[1] / "knot.v1-yay.yolov8-obb" / "train" / "labels",
        PROJECT_ROOT / "knot.v1-yay.yolov8-obb" / "train" / "labels",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv", type=Path, default=config.ANNOTATIONS_CSV)
    parser.add_argument("--images-root", type=Path, default=config.RAW_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=_default_labels_dir())
    parser.add_argument(
        "--backup-path",
        type=Path,
        default=config.ANNOTATIONS_CSV.with_suffix(".csv.pre-obb.bak"),
    )
    args = parser.parse_args()

    updated_rows = rewrite_annotations(
        annotations_csv=args.annotations_csv,
        labels_dir=args.labels_dir,
        images_root=args.images_root,
        backup_path=args.backup_path,
    )
    print(f"Updated {updated_rows} rows in {args.annotations_csv}")
    print(f"Backup saved to {args.backup_path}")


if __name__ == "__main__":
    main()