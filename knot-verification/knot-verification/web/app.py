"""
Flask demo server for the knot-verification pipeline.

Run from the knot-verification directory:
    python web/app.py

Then open http://localhost:5000
"""
import base64
import csv
import io
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Data loading helpers (cached at import time for speed)
# ---------------------------------------------------------------------------

def _load_annotations() -> list[dict]:
    rows = []
    with config.ANNOTATIONS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "filename": row["filename"],
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "label": row["label"],
            })
    return rows


def _load_split() -> dict[str, set[str]]:
    data = json.loads(config.SPLIT_FILE.read_text(encoding="utf-8"))
    return {"train": set(data["train"]), "val": set(data["val"])}


ANNOTATIONS: list[dict] = _load_annotations()
SPLIT: dict[str, set[str]] = _load_split()
ANN_BY_FILENAME: dict[str, dict] = {a["filename"]: a for a in ANNOTATIONS}

# Lazy-loaded classifier (avoids slow DINOv2 import on startup)
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from models.classifier_head import KnotClassifierHead
        _classifier = KnotClassifierHead.load()
    return _classifier


# Lazy-loaded feature extractor
_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        from models.feature_extractor import DinoV2FeatureExtractor
        _extractor = DinoV2FeatureExtractor()
    return _extractor


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _annotated_thumbnail(filename: str, size: tuple[int, int] = (400, 300)) -> str:
    """Return a base64-encoded JPEG of the image with its bounding box drawn."""
    ann = ANN_BY_FILENAME.get(filename)
    img_path = config.RAW_IMAGES_DIR / filename
    img = Image.open(img_path).convert("RGB")

    if ann:
        iw, ih = img.size
        scale_x = iw / img.width
        scale_y = ih / img.height
        draw = ImageDraw.Draw(img)
        color = "#22c55e" if ann["label"] == "correct" else "#ef4444"
        lw = max(4, min(iw, ih) // 100)
        draw.rectangle([ann["x1"], ann["y1"], ann["x2"], ann["y2"]], outline=color, width=lw)

    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _crop_thumbnail(filename: str, size: tuple[int, int] = (300, 300)) -> str:
    """Return the cropped knot region as a base64-encoded JPEG."""
    ann = ANN_BY_FILENAME.get(filename)
    img_path = config.RAW_IMAGES_DIR / filename
    img = Image.open(img_path).convert("RGB")
    if ann:
        img = img.crop((ann["x1"], ann["y1"], ann["x2"], ann["y2"]))
    img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    stats = _build_stats()
    return render_template("index.html", stats=stats)


@app.route("/gallery")
def gallery():
    label_filter = request.args.get("label", "all")
    split_filter = request.args.get("split", "all")
    page = max(1, int(request.args.get("page", 1)))
    per_page = 24

    rows = []
    for ann in ANNOTATIONS:
        split_tag = "train" if ann["filename"] in SPLIT["train"] else "val"
        if label_filter != "all" and ann["label"] != label_filter:
            continue
        if split_filter != "all" and split_tag != split_filter:
            continue
        rows.append({**ann, "split": split_tag})

    total = len(rows)
    start = (page - 1) * per_page
    page_rows = rows[start: start + per_page]

    # Attach thumbnails only for the current page
    for r in page_rows:
        r["thumb"] = _annotated_thumbnail(r["filename"], size=(320, 240))

    return render_template(
        "gallery.html",
        rows=page_rows,
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page,
        label_filter=label_filter,
        split_filter=split_filter,
    )


@app.route("/image/<path:filename>")
def image_detail(filename: str):
    ann = ANN_BY_FILENAME.get(filename)
    if not ann:
        return "Image not found in annotations", 404
    split_tag = "train" if filename in SPLIT["train"] else "val"
    full_thumb = _annotated_thumbnail(filename, size=(800, 600))
    crop_thumb = _crop_thumbnail(filename)
    img_path = config.RAW_IMAGES_DIR / filename
    w, h = Image.open(img_path).size
    box_w = ann["x2"] - ann["x1"]
    box_h = ann["y2"] - ann["y1"]
    return render_template(
        "image_detail.html",
        ann=ann,
        split=split_tag,
        full_thumb=full_thumb,
        crop_thumb=crop_thumb,
        img_w=w,
        img_h=h,
        box_w=box_w,
        box_h=box_h,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    thumb_b64 = None
    crop_b64 = None

    if request.method == "POST":
        f = request.files.get("image")
        if not f or not f.filename:
            error = "No file uploaded."
        else:
            try:
                img = Image.open(f.stream).convert("RGB")

                # Save thumbnail for display
                display = img.copy()
                display.thumbnail((640, 480), Image.LANCZOS)
                buf = io.BytesIO()
                display.save(buf, format="JPEG", quality=85)
                thumb_b64 = base64.b64encode(buf.getvalue()).decode()

                # Crop and extract features
                extractor = _get_extractor()
                crop = img.copy()
                crop.thumbnail((224, 224), Image.LANCZOS)

                buf2 = io.BytesIO()
                crop.save(buf2, format="JPEG", quality=85)
                crop_b64 = base64.b64encode(buf2.getvalue()).decode()

                feat = extractor.extract(crop)
                clf = _get_classifier()
                proba = clf.predict_proba(feat)[0]
                pred_idx = int(proba.argmax())
                label = config.CLASSIFIER_LABELS[pred_idx]
                confidence = float(proba[pred_idx])

                result = {
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "prob_correct": round(float(proba[1]) * 100, 1),
                    "prob_incorrect": round(float(proba[0]) * 100, 1),
                }
            except Exception as exc:
                error = f"Inference failed: {exc}"

    return render_template(
        "predict.html",
        result=result,
        error=error,
        thumb_b64=thumb_b64,
        crop_b64=crop_b64,
    )


@app.route("/stats")
def stats():
    s = _build_stats()
    return render_template("stats.html", stats=s)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.route("/api/annotations")
def api_annotations():
    rows = []
    for ann in ANNOTATIONS:
        rows.append({
            **ann,
            "split": "train" if ann["filename"] in SPLIT["train"] else "val",
        })
    return jsonify(rows)


@app.route("/api/stats")
def api_stats():
    return jsonify(_build_stats())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image provided"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        img.thumbnail((224, 224), Image.LANCZOS)
        feat = _get_extractor().extract(img)
        proba = _get_classifier().predict_proba(feat)[0]
        pred_idx = int(proba.argmax())
        return jsonify({
            "label": config.CLASSIFIER_LABELS[pred_idx],
            "confidence": float(proba[pred_idx]),
            "prob_correct": float(proba[1]),
            "prob_incorrect": float(proba[0]),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _build_stats() -> dict:
    n_correct = sum(1 for a in ANNOTATIONS if a["label"] == "correct")
    n_incorrect = len(ANNOTATIONS) - n_correct
    n_train = len(SPLIT["train"])
    n_val = len(SPLIT["val"])

    box_widths = [a["x2"] - a["x1"] for a in ANNOTATIONS]
    box_heights = [a["y2"] - a["y1"] for a in ANNOTATIONS]

    # Per-subfolder breakdown
    folder_counts: dict[str, dict] = {}
    for ann in ANNOTATIONS:
        parts = Path(ann["filename"]).parts
        folder = "/".join(parts[:-1]) if len(parts) > 1 else "root"
        if folder not in folder_counts:
            folder_counts[folder] = {"correct": 0, "incorrect": 0}
        folder_counts[folder][ann["label"]] += 1

    return {
        "total": len(ANNOTATIONS),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_train": n_train,
        "n_val": n_val,
        "box_width_avg": round(sum(box_widths) / len(box_widths)),
        "box_height_avg": round(sum(box_heights) / len(box_heights)),
        "box_width_min": min(box_widths),
        "box_width_max": max(box_widths),
        "box_height_min": min(box_heights),
        "box_height_max": max(box_heights),
        "folder_counts": folder_counts,
        "classifier_exists": config.CLASSIFIER_MODEL_OUT.exists(),
        "detector_exists": config.YOLO_WEIGHTS_OUT.exists(),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Knot Verification Demo -> http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
