"""Thi nghiem bo sung V2 - danh gia MOT LAN (can --confirm). Khong thay the so lieu niem phong V1.

Cau hinh CHOT TRUOC khi chay (chi dua vao tap val, khong nhin test/OOD):
  V1  : bilstm_motion_seed5 (mo hinh chinh thuc, so lieu da niem phong - chay lai de doi chieu).
  V2a : trung binh xac suat 6 mo hinh bilstm_motion_aug_seed0..5 (tang cuong du lieu, chi ShuttleSet).
        Ly do chon: val macro-F1 0.871 (V1: 0.829); khong chon tay seed nao.
  V2b : V2a + tinh chinh bang video tu quay, kiem dinh cheo bo-ra-mot-video (5 lan):
        lan k tinh chinh tren 4 video, danh gia tren video thu k. Moi mo hinh cua V2a duoc tinh chinh:
        du lieu = toan bo ShuttleSet train + clip OOD cua 4 video (lap lai x10), co tang cuong,
        Adam lr 3e-4, 10 epoch co dinh. Ket qua V2b la ket qua kiem dinh cheo, KHONG so truc tiep
        voi con so "danh gia mot lan" cua V1/V2a (V2b da duoc thay du lieu cung mien).

Chay:  python -m ai.evaluation.v2_eval --confirm
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.evaluation.metrics import LABELS, evaluate, plot_confusion_matrix
from ai.evaluation.ood_eval import load_ood, load_X
from ai.features.motion import build_features
from ai.models.classifier.augment import random_augment
from ai.models.classifier.data import load_split

LAB2I = {l: i for i, l in enumerate(LABELS)}
CK = Path("ai/models/checkpoints")
AUG = [CK / f"bilstm_motion_aug_seed{s}.keras" for s in range(6)]


def probs(models, X):
    F = build_features(X, "motion")
    return np.mean([m.predict(F, verbose=0) for m in models], axis=0)


def finetune(model, Xtr, ytr, Xood, yood, seed):
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    X = np.concatenate([Xtr] + [Xood] * 10)
    y = np.concatenate([ytr] + [yood] * 10)
    w = np.bincount(y, minlength=4)
    w = (len(y) / (4 * np.maximum(w, 1)))[y].astype(np.float32)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss="sparse_categorical_crossentropy")
    for _ in range(10):
        order = rng.permutation(len(y))
        for i in range(0, len(y), 32):
            b = order[i:i + 32]
            model.train_on_batch(build_features(random_augment(X[b], rng), "motion"), y[b], sample_weight=w[b])
    return model


def summarize(name, y, p, extra=None):
    m = evaluate(y, p)
    row = {"phuong_an": name, "n": len(y), "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)}
    row.update({f"recall_{k}": round(v["recall"], 3) for k, v in m["per_class"].items()})
    row.update(extra or {})
    return row, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", default="ai/evaluation/results/v2")
    a = ap.parse_args()
    if not a.confirm:
        raise SystemExit("Danh gia mot lan tren test + OOD: them --confirm.")
    from tensorflow import keras
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    load = lambda p: keras.models.load_model(p, compile=False)
    v1 = [load(CK / "bilstm_motion_seed5.keras")]
    v2a = [load(p) for p in AUG]

    rows, cms = [], {}
    Xte, yte = load_split("data/dataset_splits", "test")
    for name, models in (("V1", v1), ("V2a", v2a)):
        r, m = summarize(name, yte, probs(models, Xte).argmax(1), {"tap": "ShuttleSet test"})
        rows.append(r); cms[f"{name}_test"] = m["confusion_matrix"]

    df = load_ood("data/ood/pose_features/features_manifest.csv", "data/ood/clips_manifest.csv")
    ok = df[df["status"] == "OK"].reset_index(drop=True)
    Xo, yo = load_X(ok["feature_path"]), ok["label_true"].map(LAB2I).to_numpy()
    n_all = len(df)
    preds = {}
    for name, models in (("V1", v1), ("V2a", v2a)):
        P = probs(models, Xo); preds[name] = P
        r, m = summarize(name, yo, P.argmax(1), {"tap": "OOD (95 clip co pose)",
                         "acc_toan_he_thong_105": round(float((P.argmax(1) == yo).sum() / n_all), 4)})
        rows.append(r); cms[f"{name}_ood"] = m["confusion_matrix"]

    # V2b: kiem dinh cheo bo-ra-mot-video
    Xtr, ytr = load_split("data/dataset_splits", "train")
    Pb = np.zeros((len(ok), 4), np.float32)
    for v in sorted(ok["video"].unique()):
        te = (ok["video"] == v).to_numpy()
        fold = [finetune(load(p), Xtr, ytr, Xo[~te], yo[~te], seed=s) for s, p in enumerate(AUG)]
        Pb[te] = probs(fold, Xo[te])
        print(f"  V2b fold {v}: acc {(Pb[te].argmax(1) == yo[te]).mean():.3f} ({te.sum()} clip)")
    preds["V2b"] = Pb
    r, m = summarize("V2b (kiem dinh cheo)", yo, Pb.argmax(1), {"tap": "OOD (95 clip co pose)",
                     "acc_toan_he_thong_105": round(float((Pb.argmax(1) == yo).sum() / n_all), 4)})
    rows.append(r); cms["V2b_ood"] = m["confusion_matrix"]

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(out / "v2_summary.csv", index=False)
    for k, P in preds.items():
        ok[f"pred_{k}"] = [LABELS[i] for i in P.argmax(1)]
        ok[f"conf_{k}"] = P.max(1).round(4)
    ok.drop(columns=["feature_path"], errors="ignore").to_csv(out / "v2_ood_predictions.csv", index=False)
    by = ok.assign(**{f"ok_{k}": ok[f"pred_{k}"] == ok["label_true"] for k in preds}).groupby("video")[
        [f"ok_{k}" for k in preds]].mean().round(3)
    print("\nAccuracy theo video:\n", by.to_string()); by.to_csv(out / "v2_by_video.csv")
    titles = {"V1_test": "V1 - tập kiểm thử ShuttleSet", "V2a_test": "V2a (tăng cường, gộp 6 mô hình) - tập kiểm thử ShuttleSet",
              "V1_ood": "V1 - video tự quay", "V2a_ood": "V2a - video tự quay",
              "V2b_ood": "V2b (tinh chỉnh, kiểm định chéo theo video) - video tự quay"}
    for k, cm in cms.items():
        plot_confusion_matrix(cm, out / f"{k}_cm.png", titles[k])
    (out / "v2_cm.json").write_text(json.dumps({k: np.asarray(v).tolist() for k, v in cms.items()}))


if __name__ == "__main__":
    main()
