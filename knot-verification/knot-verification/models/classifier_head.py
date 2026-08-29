"""
The final Yes/No decision layer on top of the 384-d DINOv2 vector.

Three interchangeable "kinds", all exposing the same fit/predict/
predict_proba interface so pipeline.py doesn't care which one is loaded:

  - "svm": linear SVM on both classes (needs a reasonable number of
    incorrect examples).
  - "mlp": small MLP on both classes (same requirement, more flexible
    decision boundary).
  - "one_class": anomaly boundary fit on *correct* embeddings only --
    use when incorrect examples are scarce. A separate calibration step
    (calibrate_one_class) turns the raw anomaly score into a real
    probability using whatever labeled incorrect examples you do have.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC, OneClassSVM

import config


class KnotClassifierHead:
    def __init__(self, kind: str = config.CLASSIFIER_TYPE):
        self.kind = kind
        # DINO embeddings are typically compared under cosine similarity;
        # L2-normalizing before a linear model/OneClassSVM approximates
        # that geometry.
        self.normalizer = Normalizer(norm="l2")
        self.model = None
        self._calibrator = None  # only used for kind == "one_class"

    def _build_model(self, C: float = 1.0):
        if self.kind == "svm":
            return SVC(kernel="linear", C=C, probability=True, class_weight="balanced",
                       random_state=config.RANDOM_SEED)
        elif self.kind == "mlp":
            return MLPClassifier(hidden_layer_sizes=(config.MLP_HIDDEN_UNITS,), max_iter=2000,
                                  random_state=config.RANDOM_SEED)
        elif self.kind == "one_class":
            return OneClassSVM(kernel="rbf", nu=config.ONE_CLASS_NU, gamma="scale")
        raise ValueError(f"Unknown classifier kind: {self.kind}")

    def fit(self, X: np.ndarray, y: np.ndarray = None, C: float = 1.0) -> "KnotClassifierHead":
        """For kind='one_class', pass X containing *only* correct-knot
        embeddings (y is ignored) -- the boundary is fit around "normal"
        and never sees an incorrect example directly. Call
        calibrate_one_class() afterwards with a labeled set to get real
        probabilities and a sensible decision threshold."""
        Xn = self.normalizer.fit_transform(X)
        self.model = self._build_model(C)
        if self.kind == "one_class":
            self.model.fit(Xn)
        else:
            self.model.fit(Xn, y)
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Only meaningful for kind == 'one_class'. Higher = more 'correct'."""
        Xn = self.normalizer.transform(np.atleast_2d(X))
        return self.model.decision_function(Xn)

    def calibrate_one_class(self, X_labeled: np.ndarray, y_labeled: np.ndarray):
        """Fits a 1-D Platt-scaling logistic regression on top of the
        one-class decision function using a small labeled set that
        contains *both* classes. This is where your few manually-tied
        incorrect examples earn their keep, even though they never
        shaped the anomaly boundary itself."""
        if self.kind != "one_class":
            raise ValueError("calibrate_one_class() only applies to kind='one_class'.")
        scores = self.decision_scores(X_labeled).reshape(-1, 1)
        self._calibrator = LogisticRegression(class_weight="balanced")
        self._calibrator.fit(scores, y_labeled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "one_class":
            scores = self.decision_scores(X).reshape(-1, 1)
            if self._calibrator is not None:
                return self._calibrator.predict_proba(scores)
            # Uncalibrated fallback -- works but the threshold is arbitrary.
            # Call calibrate_one_class() with a labeled set before deploying.
            p_correct = 1 / (1 + np.exp(-scores.ravel()))
            return np.stack([1 - p_correct, p_correct], axis=1)
        Xn = self.normalizer.transform(np.atleast_2d(X))
        return self.model.predict_proba(Xn)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path = config.CLASSIFIER_MODEL_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "kind": self.kind,
            "normalizer": self.normalizer,
            "model": self.model,
            "calibrator": self._calibrator,
        }, path)

    @classmethod
    def load(cls, path: Path = config.CLASSIFIER_MODEL_OUT) -> "KnotClassifierHead":
        payload = joblib.load(path)
        head = cls(kind=payload["kind"])
        head.normalizer = payload["normalizer"]
        head.model = payload["model"]
        head._calibrator = payload.get("calibrator")
        return head