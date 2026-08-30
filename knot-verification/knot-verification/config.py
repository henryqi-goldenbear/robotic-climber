"""
Central configuration for the knot-verification pipeline.
Edit these paths/hyperparameters for your dataset and hardware.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
import torch

# Load local environment overrides before reading any settings.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(PROJECT_ROOT)))
RAW_IMAGES_DIR = Path(os.getenv("RAW_IMAGES_DIR", str(PROJECT_ROOT / "raw_data" / "images")))
ANNOTATIONS_CSV = Path(os.getenv("ANNOTATIONS_CSV", str(PROJECT_ROOT / "raw_data" / "annotations.csv")))
SPLIT_FILE = Path(os.getenv("SPLIT_FILE", str(PROJECT_ROOT / "raw_data" / "split.json")))

YOLO_DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"
YOLO_RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
YOLO_WEIGHTS_OUT = PROJECT_ROOT / "weights" / "knot_detector.pt"

CLASSIFIER_FEATURES_NPZ = PROJECT_ROOT / "features" / "classifier_features.npz"
CLASSIFIER_MODEL_OUT = PROJECT_ROOT / "weights" / "knot_classifier.joblib"

# ---------------------------------------------------------------------------
# Detection (YOLOv10-Nano)
# ---------------------------------------------------------------------------
YOLO_BASE_WEIGHTS = os.getenv("YOLO_BASE_WEIGHTS", "yolov10n.pt")  # auto-downloaded by ultralytics, COCO-pretrained
YOLO_CLASS_NAMES = ["knot"]
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
YOLO_EPOCHS = int(os.getenv("YOLO_EPOCHS", "150"))  # early stopping (patience) will usually cut this short
YOLO_PATIENCE = int(os.getenv("YOLO_PATIENCE", "30"))
YOLO_BATCH = int(os.getenv("YOLO_BATCH", "8"))
YOLO_CONF_THRES = float(os.getenv("YOLO_CONF_THRES", "0.35"))  # detection confidence used at inference time
CROP_PAD_RATIO = float(os.getenv("CROP_PAD_RATIO", "0.12"))  # extra margin around the box so strand ends aren't clipped

# ---------------------------------------------------------------------------
# Feature extraction (DINOv2)
# ---------------------------------------------------------------------------
# Transformers publishes the ViT-S/14 DINOv2 checkpoint under this Hub ID.
# ("facebook/dinov2_vits14" is not a valid Transformers repository.)
DINO_MODEL_NAME = os.getenv("DINO_MODEL_NAME", "facebook/dinov2-small")  # 384-d, frozen, self-supervised
DINO_IMG_SIZE = int(os.getenv("DINO_IMG_SIZE", "224"))
DINO_FEATURE_DIM = 384

# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------
CLASSIFIER_TYPE = os.getenv("CLASSIFIER_TYPE", "svm")  # "svm" or "mlp"
CLASSIFIER_LABELS = {0: "Incorrect", 1: "Correct"}
SVM_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MLP_HIDDEN_UNITS = int(os.getenv("MLP_HIDDEN_UNITS", "32"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Data augmentation for the tiny classifier training set: how many extra
# jittered crops to synthesize per source image before feature extraction.
# Cheap to do because DINOv2 stays frozen -- this can't cause the kind of
# overfitting a trainable CNN's augmentation would.
CLASSIFIER_CROPS_PER_IMAGE = int(os.getenv("CLASSIFIER_CROPS_PER_IMAGE", "4"))

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TRAIN_VAL_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", "0.85"))  # fraction of images used for training
