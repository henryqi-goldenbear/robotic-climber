import csv
from pathlib import Path


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
GOOD_FOLDERS = {"kaggle_good", "manual_good"}
BAD_FOLDERS = {"manual_bad"}


def read_image_size(path: Path):
    suffix = path.suffix.lower()
    with path.open("rb") as f:
        data = f.read()

    if suffix in {".jpg", ".jpeg"}:
        if len(data) < 10:
            raise ValueError(f"Image too small: {path}")
        if data[0:2] != b"\xFF\xD8":
            raise ValueError(f"Not a JPEG: {path}")
        i = 2
        while i < len(data) - 1:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h = data[i + 5] * 256 + data[i + 6]
                w = data[i + 7] * 256 + data[i + 8]
                return w, h
            i += 2 + (data[i + 2] * 256 + data[i + 3])
        raise ValueError(f"Could not parse JPEG dimensions for {path}")

    if suffix == ".png":
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG: {path}")
        w = int.from_bytes(data[8:12], "big")
        h = int.from_bytes(data[12:16], "big")
        return w, h

    if suffix == ".bmp":
        if data[:2] != b"BM":
            raise ValueError(f"Not a BMP: {path}")
        w = int.from_bytes(data[18:22], "little")
        h = int.from_bytes(data[22:26], "little")
        return w, h

    raise ValueError(f"Unsupported image format: {path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    images_root = project_root / "raw_data" / "images"
    output_path = project_root / "raw_data" / "annotations.csv"

    rows = []
    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_SUFFIXES:
            continue

        rel_path = image_path.relative_to(images_root)
        top_level = rel_path.parts[0]

        if top_level in GOOD_FOLDERS:
            label = "correct"
        elif top_level in BAD_FOLDERS:
            label = "incorrect"
        else:
            continue

        width, height = read_image_size(image_path)
        rows.append(
            {
                "filename": rel_path.as_posix(),
                "x1": 0,
                "y1": 0,
                "x2": width,
                "y2": height,
                "label": label,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No image files found under {images_root}")

    rows.sort(key=lambda r: r["filename"])
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "x1", "y1", "x2", "y2", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
