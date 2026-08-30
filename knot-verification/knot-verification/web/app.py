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
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
AVALANCHE_METRICS_PATH = REPO_ROOT / "avalanche-data" / "output" / "model" / "metrics.json"
AVALANCHE_PREDICTIONS_PATH = (
    REPO_ROOT
    / "avalanche-data"
    / "output"
    / "model"
    / "holdout_predictions.csv"
)
sys.path.insert(0, str(PROJECT_ROOT))

import config

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# HF bucket weight auto-download
# ---------------------------------------------------------------------------

HF_BUCKET      = "iteratehack/jobs-artifacts"
HF_BUCKET_PATH = "knot-verification-63462c48/weights"
_WEIGHTS = None  # populated after config is imported


def _ensure_weights() -> None:
    weights = {
        config.CLASSIFIER_MODEL_OUT: f"{HF_BUCKET_PATH}/knot_classifier.joblib",
        config.YOLO_WEIGHTS_OUT:     f"{HF_BUCKET_PATH}/knot_detector.pt",
    }
    missing = [local for local in weights if not local.exists()]
    if not missing:
        return
    print(f"Downloading {len(missing)} weight file(s) from HF bucket {HF_BUCKET} ...")
    try:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        for local_path in missing:
            remote = weights[local_path]
            src = f"hf://buckets/{HF_BUCKET}/{remote}"
            print(f"  {src} -> {local_path}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with fs.open(src, "rb") as r, local_path.open("wb") as w:
                w.write(r.read())
            print(f"  OK ({local_path.stat().st_size // 1024} KB)")
    except Exception as exc:
        print(f"WARNING: weight download failed: {exc}")


_ensure_weights()

# ---------------------------------------------------------------------------
# Data loading helpers (cached at import time for speed)
# ---------------------------------------------------------------------------

def _load_annotations() -> list[dict]:
    rows = []
    with config.ANNOTATIONS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Skip rows with no filename, no box coords, or label "none"
            if not row.get("filename"):
                continue
            if not row.get("x1", "").strip() or not row.get("x2", "").strip():
                continue
            if row.get("label", "").strip().lower() == "none":
                continue
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


# Lazy-loaded YOLO detector
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from models.detector import KnotDetector
        _detector = KnotDetector()
    return _detector


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
    return render_template(
        "index.html",
        stats=stats,
        demo=_build_demo_context(),
    )


@app.route("/avalanche")
def avalanche():
    avalanche_context = _build_avalanche_context()
    if avalanche_context is None:
        return "Avalanche model artifacts are not available", 503
    return render_template(
        "avalanche.html",
        stats=_build_stats(),
        avalanche=avalanche_context,
    )


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

                # Full-image thumbnail for display
                display = img.copy()
                display.thumbnail((800, 600), Image.LANCZOS)

                detector_available = config.YOLO_WEIGHTS_OUT.exists()
                box = None
                det_conf = None

                if detector_available:
                    # Full pipeline: detect -> crop -> DINOv2 -> SVM
                    detector = _get_detector()
                    detection = detector.detect_best_box(img)
                    if detection is not None:
                        box = detection[:4]
                        det_conf = float(detection[4])
                        crop = detector.crop(img, box)
                        # Draw detected box on display thumbnail
                        scale = min(display.width / img.width, display.height / img.height)
                        draw = ImageDraw.Draw(display)
                        sx1 = int(box[0] * scale)
                        sy1 = int(box[1] * scale)
                        sx2 = int(box[2] * scale)
                        sy2 = int(box[3] * scale)
                        lw = max(3, (sx2 - sx1) // 40)
                        draw.rectangle([sx1, sy1, sx2, sy2], outline="#facc15", width=lw)
                    else:
                        crop = img.copy()
                        crop.thumbnail((224, 224), Image.LANCZOS)
                else:
                    # Classifier-only fallback
                    crop = img.copy()
                    crop.thumbnail((224, 224), Image.LANCZOS)

                buf = io.BytesIO()
                display.save(buf, format="JPEG", quality=85)
                thumb_b64 = base64.b64encode(buf.getvalue()).decode()

                crop_display = crop.copy()
                crop_display.thumbnail((300, 300), Image.LANCZOS)
                buf2 = io.BytesIO()
                crop_display.save(buf2, format="JPEG", quality=85)
                crop_b64 = base64.b64encode(buf2.getvalue()).decode()

                feat = _get_extractor().extract(crop)
                proba = _get_classifier().predict_proba(feat)[0]
                pred_idx = int(proba.argmax())
                label = config.CLASSIFIER_LABELS[pred_idx]
                confidence = float(proba[pred_idx])

                result = {
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "prob_correct": round(float(proba[1]) * 100, 1),
                    "prob_incorrect": round(float(proba[0]) * 100, 1),
                    "det_conf": round(det_conf * 100, 1) if det_conf is not None else None,
                    "box": box,
                    "no_detection": detector_available and box is None,
                    "detector_used": detector_available,
                }
            except Exception as exc:
                import traceback
                error = f"Inference failed: {exc}"
                traceback.print_exc()

    return render_template(
        "predict.html",
        result=result,
        error=error,
        thumb_b64=thumb_b64,
        crop_b64=crop_b64,
        stats=_build_stats(),
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


@app.route("/api/demo-status")
def api_demo_status():
    livekit_online = False
    try:
        with socket.create_connection(("127.0.0.1", 7880), timeout=0.25):
            livekit_online = True
    except OSError:
        pass

    stats = _build_stats()
    return jsonify({
        "checked_at": datetime.now(UTC).isoformat(),
        "livekit_online": livekit_online,
        "detector_ready": stats["detector_exists"],
        "classifier_ready": stats["classifier_exists"],
        "avalanche_ready": (
            AVALANCHE_METRICS_PATH.exists()
            and AVALANCHE_PREDICTIONS_PATH.exists()
        ),
        "edge_inference": "ready" if (
            stats["detector_exists"] and stats["classifier_exists"]
        ) else "setup_required",
    })


@app.route("/api/avalanche-summary")
def api_avalanche_summary():
    avalanche_context = _build_avalanche_context()
    if avalanche_context is None:
        return jsonify({"error": "Avalanche model artifacts are not available"}), 503
    return jsonify(avalanche_context)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image provided"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        box = None
        det_conf = None
        if config.YOLO_WEIGHTS_OUT.exists():
            detection = _get_detector().detect_best_box(img)
            if detection is not None:
                box = detection[:4]
                det_conf = float(detection[4])
                img = _get_detector().crop(img, box)
            else:
                img.thumbnail((224, 224), Image.LANCZOS)
        else:
            img.thumbnail((224, 224), Image.LANCZOS)
        feat = _get_extractor().extract(img)
        proba = _get_classifier().predict_proba(feat)[0]
        pred_idx = int(proba.argmax())
        return jsonify({
            "label": config.CLASSIFIER_LABELS[pred_idx],
            "confidence": float(proba[pred_idx]),
            "prob_correct": float(proba[1]),
            "prob_incorrect": float(proba[0]),
            "detection_confidence": det_conf,
            "box": list(box) if box is not None else None,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _build_avalanche_context() -> dict | None:
    if not AVALANCHE_METRICS_PATH.exists() or not AVALANCHE_PREDICTIONS_PATH.exists():
        return None

    metrics = json.loads(AVALANCHE_METRICS_PATH.read_text(encoding="utf-8"))
    with AVALANCHE_PREDICTIONS_PATH.open(newline="", encoding="utf-8") as file:
        predictions = [
            {
                "date": row["date"],
                "actual_numeric": int(row["actual_numeric"]),
                "actual_label": row["actual_label"],
                "predicted_numeric": int(row["predicted_numeric"]),
                "predicted_label": row["predicted_label"],
                "official": row["label_source"] == "official_sac_rating",
            }
            for row in csv.DictReader(file)
        ]

    official_holdout = metrics["official_only_holdout_metrics"]
    holdout = metrics["holdout_metrics"]
    return {
        "samples": metrics["samples"],
        "numeric_features": metrics["numeric_features"],
        "official_accuracy": round(official_holdout["accuracy"] * 100, 1),
        "official_balanced_accuracy": round(
            official_holdout["balanced_accuracy"] * 100,
            1,
        ),
        "official_holdout_samples": official_holdout["samples"],
        "holdout_samples": holdout["samples"],
        "holdout_balanced_accuracy": round(holdout["balanced_accuracy"] * 100, 1),
        "date_range": metrics["date_range"],
        "target_distribution": metrics["target_distribution"],
        "selected_model": metrics["selected_model"].replace("_", " ").title(),
        "predictions": predictions,
    }


def _build_demo_context() -> dict:
    avalanche = _build_avalanche_context()

    samples = {}
    for label in ("correct", "incorrect"):
        annotation = next((a for a in ANNOTATIONS if a["label"] == label), None)
        if annotation is not None:
            samples[label] = {
                "filename": annotation["filename"],
                "thumbnail": _annotated_thumbnail(
                    annotation["filename"],
                    size=(720, 540),
                ),
            }

    return {
        "avalanche": avalanche,
        "samples": samples,
        "livekit_room": os.getenv("LIVEKIT_ROOM", "himalaya"),
    }


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
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
