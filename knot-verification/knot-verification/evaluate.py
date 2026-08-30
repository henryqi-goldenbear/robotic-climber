"""
Two evaluations:
  1. Classifier-only stratified k-fold report (robust estimate given N~150).
  2. End-to-end pipeline accuracy on the held-out val images -- this number
     matters more, since it also captures detector localization error
     compounding into the classifier.
"""
import json

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import config
from data_prep.make_splits import load_annotations
from pipeline import KnotInspectionPipeline


def classifier_only_cv_report():
    data = np.load(config.CLASSIFIER_FEATURES_NPZ, allow_pickle=True)
    X, y, splits = data["X"], data["y"], data["splits"]
    train_mask = splits == "train"
    X_train, y_train = X[train_mask], y[train_mask]

    Xn = Normalizer(norm="l2").fit_transform(X_train)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    clf = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=config.RANDOM_SEED)
    preds = cross_val_predict(clf, Xn, y_train, cv=cv)

    print("=== Classifier-only, k-fold CV on training crops ===")
    print(classification_report(y_train, preds, target_names=["Incorrect", "Correct"]))


def end_to_end_val_report():
    split = json.loads(config.SPLIT_FILE.read_text())
    df = load_annotations().set_index("filename")

    pipeline = KnotInspectionPipeline()
    y_true, y_pred = [], []
    n_missed = 0

    for fname in split["val"]:
        img_path = config.RAW_IMAGES_DIR / fname
        image = Image.open(img_path).convert("RGB")
        result = pipeline.predict(image)
        if result.label is None:
            n_missed += 1
            continue
        y_true.append(int(df.loc[fname, "label"]))
        y_pred.append(1 if result.label == "Correct" else 0)

    print("\n=== End-to-end pipeline on held-out validation images ===")
    if n_missed:
        print(f"  ({n_missed} val images had no detection above conf={config.YOLO_CONF_THRES})")
    detected = len(y_true)
    total = len(split["val"])
    print(f"  Detection rate: {detected}/{total} ({detected / total:.1%})")
    if not y_true:
        print(
            "  No validation images reached the detector confidence threshold; "
            "classifier metrics are unavailable. Lower YOLO_CONF_THRES to "
            "diagnose low-confidence detections, then inspect detector training."
        )
        return
    print(classification_report(y_true, y_pred, target_names=["Incorrect", "Correct"]))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    classifier_only_cv_report()
    end_to_end_val_report()
