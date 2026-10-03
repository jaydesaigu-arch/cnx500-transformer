"""Explicit five-class scoring conventions."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from .data import HORIZONS, CLASSES


def score(y, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), 3, 5) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probability shape or values")
    if not np.allclose(p.sum(axis=-1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    # Softmax float32 outputs differ from one by roundoff; normalize after
    # promotion to float64 before sklearn's stricter float64 sum check.
    p = p / p.sum(axis=-1, keepdims=True)
    result = {}
    for k, horizon in enumerate(HORIZONS):
        labels = y[:, k]
        pred = p[:, k].argmax(axis=-1)  # Lowest ordinal wins exact ties.
        confidence = p[:, k].max(axis=-1)
        correct = pred == labels
        ece = 0.
        for lower in np.arange(0, 1, .1):
            selected = (confidence >= lower) & (confidence < lower + .1 if lower < .9 else confidence <= 1)
            if selected.any():
                ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
        result[str(horizon)] = {
            "samples": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=list(range(5)), average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "log_loss": float(log_loss(labels, p[:, k], labels=list(range(5)))),
            "brier_sum_over_classes": float(np.square(p[:, k] - np.eye(5)[labels]).sum(axis=1).mean()),
            "ece_10_equal_width_bins": float(ece),
            "mean_absolute_ordinal_error": float(np.abs(pred - labels).mean()),
            "extreme_reversal_frequency": float((((pred == 0) & (labels == 4)) | ((pred == 4) & (labels == 0))).mean()),
            "class_support": {CLASSES[j]: int((labels == j).sum()) for j in range(5)},
            "predicted_class_counts": {CLASSES[j]: int((pred == j).sum()) for j in range(5)},
            "confusion_matrix_true_rows_predicted_columns": confusion_matrix(labels, pred, labels=list(range(5))).tolist(),
        }
    result["mean_macro_f1"] = float(np.mean([result[str(h)]["macro_f1"] for h in HORIZONS]))
    return result
