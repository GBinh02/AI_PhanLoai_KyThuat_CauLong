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


# Ten hien thi tieng Viet cho bieu do (bao cao). Thu tu khop LABELS.
LABELS_VI = {"clear": "Phông cầu", "net_shot": "Bỏ nhỏ lưới",
             "short_service": "Giao cầu ngắn", "smash": "Đập cầu"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})  # DejaVu co du dau tieng Viet


MODEL_VI = {"mlp": "MLP", "lstm": "LSTM", "bilstm": "BiLSTM", "gru": "GRU", "cnn1d": "CNN-1D"}
FEAT_VI = {"raw": "tọa độ thô", "motion": "đặc trưng chuyển động", "motion_eng": "chuyển động + đặc trưng thiết kế"}
SPLIT_VI = {"train": "tập huấn luyện", "val": "tập kiểm định", "test": "tập kiểm thử ShuttleSet",
            "ood": "video tự quay (ngoài miền)"}


def vi_title(tag, split=None):
    """'bilstm_motion_seed5' -> 'BiLSTM + đặc trưng chuyển động (seed 5)'; them ten tap neu co."""
    parts = tag.split("_")
    model = MODEL_VI.get(parts[0], parts[0])
    seed = next((p[4:] for p in parts if p.startswith("seed")), None)
    feat_key = "_".join(p for p in parts[1:] if not p.startswith("seed") and p != "aug")
    t = f"{model} + {FEAT_VI.get(feat_key, feat_key)}" if feat_key else model
    if "aug" in parts:
        t += ", tăng cường dữ liệu"
    if seed is not None:
        t += f" (seed {seed})"
    return f"{t} - {SPLIT_VI.get(split, split)}" if split else t


def _vi(labels):
    return [LABELS_VI.get(l, l) for l in labels]


def plot_confusion_matrix(cm, path, title="", labels=LABELS):
    """Ve 2 ma tran canh nhau: so luong va ti le theo hang (= recall tung lop)."""
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = _vi(labels)
    acc = np.trace(cm) / max(cm.sum(), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Số lượng mẫu"),
                               (axes[1], norm, ".2f", "Tỉ lệ theo hàng (độ phủ)")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(labels)), names, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), names)
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")
        ax.set_title(sub, fontsize=11)
        thr = data.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=11,
                        fontweight="bold" if i == j else "normal",
                        color="white" if data[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Ma trận nhầm lẫn{' - ' + title if title else ''}  (độ chính xác {acc:.1%}, n = {cm.sum()})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path, title=""):
    """Duong hoc: co diem danh dau tung epoch, ghi gia tri, danh dau epoch tot nhat (val_loss nho nhat)
    - dung epoch ma EarlyStopping(restore_best_weights) giu lai."""
    vl = np.asarray(history["val_loss"])
    best = int(vl.argmin())
    n = len(vl)
    step = 1 if n <= 20 else int(np.ceil(n / 20))  # nhieu epoch -> ghi so thua bot cho de doc
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, name, fmt in ((axes[0], "loss", "Hàm mất mát (loss)", "{:.2f}"),
                               (axes[1], "accuracy", "Độ chính xác (accuracy)", "{:.0%}")):
        x = np.arange(1, n + 1)
        tr, va = np.asarray(history[key]), np.asarray(history[f"val_{key}"])
        for y, other, lab, color in ((tr, va, "Huấn luyện", "#1f77b4"), (va, tr, "Kiểm định (val)", "#e8700a")):
            ax.plot(x, y, marker="o", markersize=5, linewidth=2, color=color, label=lab, zorder=3)
            for i in range(0, n, step):
                up = y[i] >= other[i]  # duong nao cao hon tai epoch nay thi ghi so phia tren
                ax.annotate(fmt.format(y[i]), (x[i], y[i]), textcoords="offset points",
                            xytext=(0, 9 if up else -15), ha="center", fontsize=8, color=color, zorder=4,
                            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": .85})
        ax.axvline(best + 1, color="#2ca02c", linestyle="--", linewidth=1.3, zorder=2)
        ax.plot(best + 1, va[best], marker="*", markersize=16, color="#2ca02c", zorder=5)
        ax.text(0.98, 0.5 if key == "loss" else 0.06,
                f"Epoch tốt nhất (val loss nhỏ nhất): {best + 1}\nval {key} = {fmt.format(va[best])}"
                f"\nMô hình được giữ ở epoch này", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color="#1e6b1e",
                bbox={"boxstyle": "round,pad=0.4", "fc": "#eef8ee", "ec": "#2ca02c"})
        lo, hi = min(tr.min(), va.min()), max(tr.max(), va.max())
        pad = (hi - lo) * 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_xticks(x[::step])
        ax.grid(True, alpha=.3)
        if key == "accuracy":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(loc="upper right" if key == "loss" else "center right")
    fig.suptitle(f"Quá trình huấn luyện{' - ' + title if title else ''}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
