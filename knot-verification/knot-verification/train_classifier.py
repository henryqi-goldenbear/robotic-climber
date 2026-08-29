"""
Trains the SVM (or MLP) head on the frozen DINOv2 features.

Because N is tiny, the regularization strength (C) is chosen by stratified
k-fold CV on the *training* split only, and the reported CV score is the
honest estimate of real-world performance -- not a single small holdout,
which would be too noisy with only a couple dozen images.

Run order: data_prep/build_classifier_dataset.py -> this.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import config
from models.classifier_head import KnotClassifierHead


def main():
    data = np.load(config.CLASSIFIER_FEATURES_NPZ, allow_pickle=True)
    X, y, splits = data["X"], data["y"], data["splits"]

    train_mask = splits == "train"
    X_train, y_train = X[train_mask], y[train_mask]
    print(f"Training on {len(y_train)} feature vectors "
          f"({(y_train == 1).sum()} correct / {(y_train == 0).sum()} incorrect)")

    best_C = 1.0
    if config.CLASSIFIER_TYPE == "svm":
        Xn = Normalizer(norm="l2").fit_transform(X_train)
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
        grid = GridSearchCV(
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=config.RANDOM_SEED),
            param_grid={"C": config.SVM_C_GRID},
            cv=cv, scoring="f1",
        )
        grid.fit(Xn, y_train)
        best_C = grid.best_params_["C"]
        print(f"Best C: {best_C}  |  CV F1: {grid.best_score_:.3f}")

    head = KnotClassifierHead(kind=config.CLASSIFIER_TYPE).fit(X_train, y_train, C=best_C)
    head.save()
    print(f"Classifier saved to {config.CLASSIFIER_MODEL_OUT}")


if __name__ == "__main__":
    main()
