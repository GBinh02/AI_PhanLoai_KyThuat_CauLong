"""Danh gia TAP TEST - chi chay MOT LAN cho model da chot tren VAL.

    python -m ai.evaluation.run_test --checkpoint ai/models/checkpoints/gru_motion_eng_seed0.keras --confirm

Che do dac trung duoc doc tu file .json canh checkpoint (mac dinh 'raw').
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf

from ai.evaluation.metrics import LABELS, evaluate, plot_confusion_matrix
from ai.features.motion import build_features
from ai.models.classifier.data import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/dataset_splits")
    ap.add_argument("--out-dir", default="ai/evaluation/results/test")
    ap.add_argument("--confirm", action="store_true",
                    help="Xac nhan da chot model tren VAL, khong chinh theo test nua")
    args = ap.parse_args()
    if not args.confirm:
        raise SystemExit("Bo do da niem phong: them --confirm de chay tren tap test.")

    ckpt = Path(args.checkpoint)
    meta_path = ckpt.with_suffix(".json")
    feats = json.loads(meta_path.read_text())["features"] if meta_path.exists() else "raw"

    X, y = load_split(args.data_dir, "test")
    X = build_features(X, feats)
    model = tf.keras.models.load_model(ckpt)
    m = evaluate(y, model.predict(X, verbose=0).argmax(axis=1))

    tag = ckpt.stem
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(m["confusion_matrix"], out / f"{tag}_test_cm.png", f"{tag} (test)")
    report = {**m, "confusion_matrix": m["confusion_matrix"].tolist(),
              "labels": LABELS, "features": feats}
    (out / f"{tag}_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"TEST {tag} [features={feats}]: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
    for lab, s in m["per_class"].items():
        print(f"  {lab:14s} P={s['precision']:.3f} R={s['recall']:.3f} n={s['support']}")


if __name__ == "__main__":
    main()
