"""Do do ben vung cua checkpoint tren tap VAL khi gia lap cac sai lech cua video thuc te.
Chi dung tap val (khong niem phong) -> dung de so sanh/chon mo hinh hop le.

Chay:  python -m ai.evaluation.robustness ai/models/checkpoints/bilstm_motion_seed5.keras [...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.evaluation.metrics import evaluate
from ai.features.motion import build_features
from ai.models.classifier import augment as A
from ai.models.classifier.data import load_split

CONDITIONS = [
    ("goc", lambda X: X),
    ("crop hep (x*0.8)", lambda X: A.scale_x(X, 0.8)),
    ("crop rong (x*1.25)", lambda X: A.scale_x(X, 1.25)),
    ("cham cau som 2 khung", lambda X: A.resample_time(X, -2, 1.0)),
    ("cham cau muon 2 khung", lambda X: A.resample_time(X, 2, 1.0)),
    ("dong tac nhanh 15%", lambda X: A.resample_time(X, 0, 1.15)),
    ("thuan tay trai (lat)", lambda X: A.mirror(X)),
]


def score(model, feats, X, y):
    m = evaluate(y, model.predict(build_features(X, feats), verbose=0).argmax(1))
    return m["accuracy"], m["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data-dir", default="data/dataset_splits")
    ap.add_argument("--out", default=None, help="luu bang ket qua (csv)")
    a = ap.parse_args()
    from tensorflow import keras
    X, y = load_split(a.data_dir, "val")
    rows = []
    for ck in a.checkpoints:
        meta = Path(ck).with_suffix(".json")
        feats = json.loads(meta.read_text())["features"] if meta.is_file() else "motion"
        model = keras.models.load_model(ck, compile=False)
        row = {"checkpoint": Path(ck).stem}
        for name, f in CONDITIONS:
            row[name] = round(score(model, feats, f(X), y)[1], 4)
        row["trung binh (sai lech)"] = round(float(np.mean([row[n] for n, _ in CONDITIONS[1:]])), 4)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("checkpoint").T
    print("Macro-F1 tren tap VAL theo tung dieu kien:")
    print(df.to_string())
    if a.out:
        df.to_csv(a.out)


if __name__ == "__main__":
    main()
