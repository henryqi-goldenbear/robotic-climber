"""
Central configuration for the knot-verification pipeline.
Edit these paths/hyperparameters for your dataset and hardware.
"""
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_IMAGES_DIR = PROJECT_ROOT / "raw_data" / "images"
ANNOTATIONS_CSV = PROJECT_ROOT / "raw_data" / "annotations.csv"
SPLIT_FILE = PROJECT_ROOT / "raw_data" / "split.json"

YOLO_DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"
YOLO_RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
YOLO_WEIGHTS_OUT = PROJECT_ROOT / "weights" / "knot_detector.pt"

CLASSIFIER_FEATURES_NPZ = PROJECT_ROOT / "features" / "classifier_features.npz"
CLASSIFIER_MODEL_OUT = PROJECT_ROOT / "weights" / "knot_classifier.joblib"

# ---------------------------------------------------------------------------
# Detection (YOLOv10-Nano)
# ---------------------------------------------------------------------------
YOLO_BASE_WEIGHTS = "yolov10n.pt"      # auto-downloaded by ultralytics, COCO-pretrained
YOLO_CLASS_NAMES = ["knot"]
YOLO_IMG_SIZE = 640
YOLO_EPOCHS = 150                      # early stopping (patience) will usually cut this short
YOLO_PATIENCE = 30
YOLO_BATCH = 8
YOLO_CONF_THRES = 0.35                 # detection confidence used at inference time
CROP_PAD_RATIO = 0.12                  # extra margin around the box so strand ends aren't clipped

# ---------------------------------------------------------------------------
# Feature extraction (DINOv2)
# ---------------------------------------------------------------------------
DINO_MODEL_NAME = "dinov2_vits14"      # 384-d, frozen, self-supervised
DINO_IMG_SIZE = 224
DINO_FEATURE_DIM = 384

# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------
CLASSIFIER_TYPE = "svm"                # "svm" or "mlp"
CLASSIFIER_LABELS = {0: "Incorrect", 1: "Correct"}
SVM_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MLP_HIDDEN_UNITS = 32
CV_FOLDS = 5
RANDOM_SEED = 42

# Data augmentation for the tiny classifier training set: how many extra
# jittered crops to synthesize per source image before feature extraction.
# Cheap to do because DINOv2 stays frozen -- this can't cause the kind of
# overfitting a trainable CNN's augmentation would.
CLASSIFIER_CROPS_PER_IMAGE = 4

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TRAIN_VAL_SPLIT = 0.85                 # fraction of images used for training
