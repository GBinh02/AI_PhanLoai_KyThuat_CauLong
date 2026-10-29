"""Bo do danh gia dung chung (muc 7 dac ta): accuracy, precision/recall tung lop,
macro-F1, confusion matrix. Moi model va baseline deu phai di qua file nay."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

LABELS = ["clear", "net_shot", "short_service", "smash"]  # thu tu = y da ma hoa 0-3


def evaluate(y_true, y_pred, labels=LABELS):
    idx = list(range(len(labels)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f1[i]), "support": int(sup[i])}
            for i in idx
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx),
    }


def plot_confusion_matrix(cm, path, title="", labels=LABELS):
    """Ve 2 ma tran canh nhau: so luong va ti le theo hang (recall)."""
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "So luong"),
                               (axes[1], norm, ".2f", "Ti le theo hang")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Du doan")
        ax.set_ylabel("Thuc te")
        ax.set_title(f"{title} - {sub}")
        thr = data.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(f"{title} - {key}")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
