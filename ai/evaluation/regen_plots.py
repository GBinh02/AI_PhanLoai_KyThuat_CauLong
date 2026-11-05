"""Ve lai TOAN BO bieu do tap val (tieng Viet) cho moi mo hinh da huan luyen - CHI thay bieu do khi
da xac minh khop voi mo hinh goc.

- Ma tran nham lan val: tinh tu checkpoint goc (ai/models/checkpoints/<tag>.keras) -> chinh xac tuyet doi.
- Duong hoc: lich su cu khong duoc luu, nen huan luyen lai DUNG seed vao thu muc tam, roi xac minh:
    * co checkpoint goc  -> du doan tren val phai trung >= 99.5% voi checkpoint goc;
    * khong co checkpoint -> val macro-F1 phai khop so trong results*.csv (sai so < 0.002).
  Khop thi chep <tag>_history.json/.png; khong khop thi GIU NGUYEN bieu do cu va ghi vao bao cao.
- Khong ghi de checkpoint, results.csv, compare.csv.
- Nen chay tren CHINH may da huan luyen (cung phien ban TensorFlow) de tai lap duoc nhieu nhat.

Chay tu goc repo:  python -m ai.evaluation.regen_plots [--features raw motion motion_eng]
Ket qua doi chieu:  ai/evaluation/results/val/regen_report.csv
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ai.evaluation.metrics import evaluate, plot_confusion_matrix, plot_history, vi_title
from ai.features.motion import build_features
from ai.models.classifier.data import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["raw", "motion", "motion_eng"])
    ap.add_argument("--results", default="ai/evaluation/results/val")
    ap.add_argument("--checkpoints", default="ai/models/checkpoints")
    ap.add_argument("--data-dir", default="data/dataset_splits")
    a = ap.parse_args()
    from tensorflow import keras
    Xva, yva = load_split(a.data_dir, "val")
    report = []
    for feats in a.features:
        d = Path(a.results) / feats
        F = build_features(Xva, feats)
        runs = sorted({(m.group(1), int(m.group(2))) for p in d.glob("*_val_cm.png")
                       if (m := re.match(rf"([a-z0-9]+)_{feats}_seed(\d+)_val_cm\.png$", p.name))})
        old = {}
        for csv in d.glob("results.csv"):
            for _, r in pd.read_csv(csv).iterrows():
                old[(r["model"], int(r["seed"]))] = float(r["val_macro_f1"])
        for model, seed in runs:
            tag = f"{model}_{feats}_seed{seed}"
            ck = Path(a.checkpoints) / f"{tag}.keras"
            ref_pred = keras.models.load_model(ck, compile=False).predict(F, verbose=0).argmax(1) if ck.is_file() else None
            if ref_pred is not None:  # ma tran nham lan tu chinh mo hinh goc
                plot_confusion_matrix(evaluate(yva, ref_pred)["confusion_matrix"], d / f"{tag}_val_cm.png",
                                      vi_title(tag, "val"))
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run([sys.executable, "-m", "ai.models.classifier.train", "--models", model,
                                "--features", feats, "--seeds", str(seed), "--ckpt-dir", f"{tmp}/ck",
                                "--out-dir", f"{tmp}/res"], check=True, capture_output=True)
                new_pred = keras.models.load_model(f"{tmp}/ck/{tag}.keras", compile=False).predict(F, verbose=0).argmax(1)
                new_f1 = evaluate(yva, new_pred)["macro_f1"]
                if ref_pred is not None:
                    agree = float((ref_pred == new_pred).mean())
                    ok, how = agree >= 0.995, f"trung {agree:.1%} du doan voi checkpoint goc"
                elif (model, seed) in old:
                    ok, how = abs(old[(model, seed)] - new_f1) < 0.002, f"F1 cu {old[(model, seed)]:.4f}, moi {new_f1:.4f}"
                else:
                    ok, how = False, "khong co checkpoint/so cu de doi chieu"
                if ok:
                    hist = json.loads((Path(tmp) / "res" / feats / f"{tag}_history.json").read_text())
                    (d / f"{tag}_history.json").write_text(json.dumps(hist))
                    plot_history(hist, d / f"{tag}_history.png", vi_title(tag))
            status = "DA CAP NHAT" if ok else "GIU BIEU DO CU (khong tai lap duoc)"
            report.append({"features": feats, "model": model, "seed": seed,
                           "cm_tu_checkpoint_goc": ref_pred is not None, "doi_chieu": how, "duong_hoc": status})
            print(f"{tag:26s} {how:45s} -> {status}", flush=True)
            pd.DataFrame(report).to_csv(Path(a.results) / "regen_report.csv", index=False)


if __name__ == "__main__":
    main()
