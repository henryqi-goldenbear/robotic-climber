"""
The final Yes/No decision layer on top of the 384-d DINOv2 vector.

Supports a linear SVM (default -- fast, robust with N~150, finds the
max-margin separator between correct/incorrect knot vectors) or a small
MLP. Both share the same fit/predict interface so pipeline.py doesn't need
to know which one is loaded.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import config


class KnotClassifierHead:
    def __init__(self, kind: str = config.CLASSIFIER_TYPE):
        self.kind = kind
        # DINO embeddings are typically compared under cosine similarity;
        # L2-normalizing before a linear SVM approximates that geometry.
        self.normalizer = Normalizer(norm="l2")
        self.model = None

    def _build_model(self, C: float = 1.0):
        if self.kind == "svm":
            return SVC(kernel="linear", C=C, probability=True, class_weight="balanced",
                       random_state=config.RANDOM_SEED)
        elif self.kind == "mlp":
            return MLPClassifier(hidden_layer_sizes=(config.MLP_HIDDEN_UNITS,), max_iter=2000,
                                  random_state=config.RANDOM_SEED)
        raise ValueError(f"Unknown classifier kind: {self.kind}")

    def fit(self, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> "KnotClassifierHead":
        Xn = self.normalizer.fit_transform(X)
        self.model = self._build_model(C)
        self.model.fit(Xn, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = self.normalizer.transform(np.atleast_2d(X))
        return self.model.predict(Xn)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = self.normalizer.transform(np.atleast_2d(X))
        return self.model.predict_proba(Xn)

    def save(self, path: Path = config.CLASSIFIER_MODEL_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": self.kind, "normalizer": self.normalizer, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path = config.CLASSIFIER_MODEL_OUT) -> "KnotClassifierHead":
        payload = joblib.load(path)
        head = cls(kind=payload["kind"])
        head.normalizer = payload["normalizer"]
        head.model = payload["model"]
        return head
