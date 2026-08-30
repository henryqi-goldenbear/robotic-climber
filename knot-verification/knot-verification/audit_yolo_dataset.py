#!/usr/bin/env python3
"""Read-only integrity audit for raw annotations and generated YOLO labels."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import config
from data_prep.make_splits import load_annotations


def _summary(values: list[float]) -> str:
    values = np.asarray(values, dtype=float)
    return f"min={values.min():.3f}, median={np.median(values):.3f}, max={values.max():.3f}"


def main() -> None:
    df = load_annotations()
    background_files = set(df.loc[df["is_background"], "filename"])
    split = json.loads(config.SPLIT_FILE.read_text())
    train, val = set(split["train"]), set(split["val"])
    all_split_files = train | val

    print("=== Split integrity ===")
    print(f"Annotations: {len(df)} rows, {df['filename'].nunique()} unique images")
    print(f"Train/val: {len(train)}/{len(val)}; overlap: {len(train & val)}")
    print(f"Unassigned annotation files: {len(set(df.filename) - all_split_files)}")
    for name, files in (("train", train), ("val", val)):
        sources = Counter(Path(filename).parts[0] for filename in files)
        print(f"  {name} sources: {dict(sorted(sources.items()))}")

    print("\n=== Raw box geometry ===")
    invalid, orientations = [], Counter()
    widths, heights, areas, signatures = [], [], [], Counter()
    per_source = defaultdict(lambda: {"widths": [], "heights": [], "areas": []})
    for row in df.itertuples(index=False):
        image_path = config.RAW_IMAGES_DIR / row.filename
        if not image_path.is_file():
            invalid.append((row.filename, "image missing"))
            continue
        with Image.open(image_path) as image:
            image_w, image_h = image.size
            orientation = image.getexif().get(274, 1)
        if orientation != 1:
            orientations[orientation] += 1
        valid = 0 <= row.x1 < row.x2 <= image_w and 0 <= row.y1 < row.y2 <= image_h
        if not valid:
            invalid.append((row.filename, f"box=({row.x1}, {row.y1}, {row.x2}, {row.y2}), image={image_w}x{image_h}"))
            continue
        width, height = (row.x2 - row.x1) / image_w, (row.y2 - row.y1) / image_h
        area = width * height
        widths.append(width)
        heights.append(height)
        areas.append(area)
        source = Path(row.filename).parts[0]
        per_source[source]["widths"].append(width)
        per_source[source]["heights"].append(height)
        per_source[source]["areas"].append(area)
        signatures[(row.x1, row.y1, row.x2, row.y2)] += 1

    print(f"Invalid/missing boxes: {len(invalid)}")
    for issue in invalid[:10]:
        print("  !", issue)
    print(f"Box width fraction: {_summary(widths)}")
    print(f"Box height fraction: {_summary(heights)}")
    print(f"Box area fraction: {_summary(areas)}")
    for source, stats in sorted(per_source.items()):
        print(f"  {source}: area fraction {_summary(stats['areas'])}")
    repeated = signatures.most_common(5)
    print("Most reused absolute boxes:")
    for box, count in repeated:
        print(f"  {box}: {count} images")
    if orientations:
        print(f"WARNING: non-standard EXIF orientations: {dict(orientations)}. "
              "Confirm the annotation tool and YOLO use the same orientation.")

    print("\n=== Generated YOLO dataset ===")
    for split_name in ("train", "val"):
        image_dir = config.YOLO_DATASET_DIR / "images" / split_name
        label_dir = config.YOLO_DATASET_DIR / "labels" / split_name
        images = [p for p in image_dir.rglob("*") if p.is_file()]
        labels = [p for p in label_dir.rglob("*.txt") if p.is_file()]
        missing, malformed = [], []
        for image_path in images:
            source_file = image_path.relative_to(image_dir).as_posix()
            relative = Path(source_file).with_suffix(".txt")
            label_path = label_dir / relative
            if not label_path.is_file():
                missing.append(str(relative))
                continue
            lines = label_path.read_text().strip().splitlines()
            if source_file in background_files:
                if lines:
                    malformed.append(f"{relative}: background label must be empty")
                continue
            if len(lines) != 1:
                malformed.append(f"{relative}: expected 1 line, got {len(lines)}")
                continue
            parts = lines[0].split()
            try:
                class_id, cx, cy, width, height = map(float, parts)
                if len(parts) != 5 or class_id != 0 or not all(0 < v <= 1 for v in (cx, cy, width, height)):
                    raise ValueError
            except ValueError:
                malformed.append(f"{relative}: {lines[0]!r}")
        print(f"  {split_name}: {len(images)} images, {len(labels)} labels, "
              f"missing={len(missing)}, malformed={len(malformed)}")
        for issue in (missing + malformed)[:10]:
            print("    !", issue)


if __name__ == "__main__":
    main()
