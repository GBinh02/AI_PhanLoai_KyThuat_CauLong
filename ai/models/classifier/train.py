"""Train nhieu model x nhieu che do dac trung x nhieu seed; danh gia tren tap VAL
(KHONG dung test).

Chay tu thu muc goc repo:
    python -m ai.models.classifier.train --features raw motion motion_eng
    python -m ai.models.classifier.train --models gru --features motion_eng --seeds 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ai.evaluation.metrics import LABELS, evaluate, plot_confusion_matrix, plot_history
from ai.features.motion import FEATURE_MODES, build_features
from ai.models.classifier.data import load_split
from ai.models.classifier.models import MODEL_NAMES, build_model


def run_one(name, feats, seed, Xtr, ytr, Xva, yva, args, out_dir, ckpt_dir):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(name, Xtr.shape[1:], len(LABELS), norm_data=Xtr)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=ytr)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)

    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs,
                     batch_size=args.batch, class_weight=dict(enumerate(weights)),
                     callbacks=[early], verbose=0)
    train_time = time.time() - t0

    val_loss, _ = model.evaluate(Xva, yva, verbose=0)
    m = evaluate(yva, model.predict(Xva, verbose=0).argmax(axis=1))

    tag = f"{name}_{feats}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir / f"{tag}.keras")
    (ckpt_dir / f"{tag}.json").write_text(json.dumps({"features": feats, "model": name}))
    plot_history(hist.history, out_dir / f"{tag}_history.png", tag)
    plot_confusion_matrix(m["confusion_matrix"], out_dir / f"{tag}_val_cm.png", f"{tag} (val)")

    row = {"features": feats, "model": name, "seed": seed, "params": model.count_params(),
           "epochs_run": len(hist.history["loss"]), "train_sec": round(train_time, 1),
           "val_loss": round(float(val_loss), 4), "val_acc": round(m["accuracy"], 4),
           "val_macro_f1": round(m["macro_f1"], 4)}
    for lab, s in m["per_class"].items():
        row[f"prec_{lab}"] = round(s["precision"], 3)
        row[f"rec_{lab}"] = round(s["recall"], 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--features", nargs="+", default=["raw"], choices=FEATURE_MODES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--data-dir", default="data/dataset_splits")
    ap.add_argument("--out-dir", default="ai/evaluation/results/val")
    ap.add_argument("--ckpt-dir", default="ai/models/checkpoints")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    Xtr0, ytr = load_split(args.data_dir, "train")
    Xva0, yva = load_split(args.data_dir, "val")
    print(f"train {Xtr0.shape}, val {Xva0.shape}, nhan train: {np.bincount(ytr)}")

    ckpt_dir = Path(args.ckpt_dir)
    for feats in args.features:
        Xtr, Xva = build_features(Xtr0, feats), build_features(Xva0, feats)
        out_dir = Path(args.out_dir) / feats
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n##### features={feats}, so dac trung/frame = {Xtr.shape[2]}")
        rows = []
        for name in args.models:
            for seed in args.seeds:
                row = run_one(name, feats, seed, Xtr, ytr, Xva, yva, args, out_dir, ckpt_dir)
                rows.append(row)
                print(f"[{feats} | {name} seed{seed}] val_acc={row['val_acc']:.3f} "
                      f"macroF1={row['val_macro_f1']:.3f} rec_clear={row['rec_clear']:.2f} "
                      f"rec_smash={row['rec_smash']:.2f} ({row['epochs_run']} ep, {row['train_sec']}s)")
                pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)


if __name__ == "__main__":
    main()
