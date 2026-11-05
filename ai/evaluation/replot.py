"""Ve lai bieu do bao cao (tieng Viet) tu ket qua DA CO - khong train lai, khong danh gia lai tap test/OOD.

- Ma tran nham lan tap test va OOD: doc tu file json da luu (so lieu niem phong giu nguyen).
- Ma tran nham lan tap val: chay lai checkpoint tren val (val khong niem phong).
- Duong hoc: doc tu <tag>_history.json (train.py ban moi tu luu file nay).

Chay tu goc repo:  python -m ai.evaluation.replot
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ai.evaluation.metrics import evaluate, plot_confusion_matrix, plot_history, vi_title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bilstm_motion_seed5")
    ap.add_argument("--results", default="ai/evaluation/results")
    ap.add_argument("--checkpoints", default="ai/models/checkpoints")
    ap.add_argument("--data-dir", default="data/dataset_splits")
    a = ap.parse_args()
    R, tag = Path(a.results), a.tag

    test_json = R / "test" / f"{tag}_test.json"
    if test_json.is_file():
        cm = json.loads(test_json.read_text())["confusion_matrix"]
        plot_confusion_matrix(cm, R / "test" / f"{tag}_test_cm.png", vi_title(tag, "test"))
        print("Da ve:", R / "test" / f"{tag}_test_cm.png")

    ood_json = R / "ood" / "ood_report.json"
    if ood_json.is_file():
        rep = json.loads(ood_json.read_text())
        plot_confusion_matrix(rep["bilstm"]["confusion_matrix"], R / "ood" / f"{tag}_ood_cm.png", vi_title(tag, "ood"))
        plot_confusion_matrix(rep["rule_based"]["confusion_matrix"], R / "ood" / "rule_based_ood_cm.png",
                              "Baseline luật cố định - video tự quay (ngoài miền)")
        print("Da ve:", R / "ood")

    feats = tag.split("_")[1] if "_" in tag else "motion"
    ckpt = Path(a.checkpoints) / f"{tag}.keras"
    meta = ckpt.with_suffix(".json")
    if meta.is_file():
        feats = json.loads(meta.read_text()).get("features", feats)
    if ckpt.is_file():
        from tensorflow import keras
        from ai.features.motion import build_features
        from ai.models.classifier.data import load_split
        X, y = load_split(a.data_dir, "val")
        pred = keras.models.load_model(ckpt, compile=False).predict(build_features(X, feats), verbose=0).argmax(1)
        out = R / "val" / feats / f"{tag}_val_cm.png"
        plot_confusion_matrix(evaluate(y, pred)["confusion_matrix"], out, vi_title(tag, "val"))
        print("Da ve:", out)

    hist = R / "val" / feats / f"{tag}_history.json"
    if hist.is_file():
        plot_history(json.loads(hist.read_text()), hist.with_suffix(".png"), vi_title(tag))
        print("Da ve:", hist.with_suffix(".png"))
    else:
        print(f"Chua co {hist} (history cu khong duoc luu). Train lai bang train.py ban moi de co file nay.")


if __name__ == "__main__":
    main()
