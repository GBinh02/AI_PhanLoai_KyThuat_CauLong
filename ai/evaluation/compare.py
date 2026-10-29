"""Gop ket qua VAL cua moi che do dac trung x model thanh 1 bang so sanh.

    python -m ai.evaluation.compare
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default="ai/evaluation/results/val")
    args = ap.parse_args()

    files = sorted(Path(args.val_dir).glob("*/results.csv"))
    if not files:
        raise SystemExit(f"Khong thay results.csv nao trong {args.val_dir}/*/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    g = df.groupby(["features", "model"])
    table = g.agg(macro_f1=("val_macro_f1", "mean"), f1_std=("val_macro_f1", "std"),
                  acc=("val_acc", "mean"), loss=("val_loss", "mean"),
                  rec_clear=("rec_clear", "mean"), rec_smash=("rec_smash", "mean"),
                  n_seeds=("seed", "count")).round(3)
    table = table.sort_values("macro_f1", ascending=False)

    pd.set_option("display.width", 200)
    print("=== So sanh tren tap VAL (sap theo macro-F1) ===")
    print(table)
    print("\n=== Trung binh theo che do dac trung ===")
    print(df.groupby("features")[["val_macro_f1", "val_acc", "rec_clear", "rec_smash"]]
            .mean().round(3))
    table.to_csv(Path(args.val_dir) / "compare.csv")


if __name__ == "__main__":
    main()
