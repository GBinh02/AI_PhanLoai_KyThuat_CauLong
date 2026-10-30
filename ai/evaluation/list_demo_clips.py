"""Liet ke cac cu danh OOD de chon demo / minh hoa bao cao (CHI DOC, khong tinh lai diem, khong dung de tinh chinh).

Doc ood_prediction.csv (cot: clip_name, video, person, player_side, label_true, status,
pred_bilstm, conf_bilstm, pred_rule) va in:
  1. Cu BiLSTM DOAN DUNG va tin cay cao -> de demo
  2. Cu BiLSTM DOAN SAI -> de phan tich loi trong bao cao (kem du doan cua baseline)
contact_sec lay tu ten clip (video6_t8.70.mp4 -> 8.70) - day la gia tri nhap vao nut "Danh dau cham cau".

Chay:  python ai/evaluation/list_demo_clips.py ai/evaluation/results/ood/ood_prediction.csv --min-conf 0.8
"""
import argparse
import re

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--min-conf", type=float, default=0.8, help="nguong tin cay cho nhom demo (mac dinh 0.8)")
    ap.add_argument("--side", choices=["near", "far", "any"], default="any")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df = df[df["status"] == "OK"].copy()
    df["contact_sec"] = df["clip_name"].str.extract(r"_t([0-9.]+)\.mp4", flags=re.I)[0].astype(float)
    if a.side != "any":
        df = df[df["player_side"] == a.side]
    cols = ["label_true", "video", "contact_sec", "player_side", "pred_bilstm", "conf_bilstm", "pred_rule"]

    ok = df[(df["pred_bilstm"] == df["label_true"]) & (df["conf_bilstm"] >= a.min_conf)]
    print(f"== BiLSTM dung, tin cay >= {a.min_conf} ({len(ok)}/{len(df)} cu) - DE DEMO ==")
    print(ok.sort_values(["label_true", "conf_bilstm"], ascending=[True, False])[cols].to_string(index=False)
          if len(ok) else "(khong co)")
    missing = sorted(set(df["label_true"]) - set(ok["label_true"]))
    if missing:
        print(f"\nLuu y: khong co cu nao du dieu kien cho nhan: {', '.join(missing)} "
              f"(thu ha --min-conf, hoac day la phat hien trung thuc cho bao cao)")

    bad = df[df["pred_bilstm"] != df["label_true"]]
    print(f"\n== BiLSTM dung sai ({len(bad)}/{len(df)} cu) - DE PHAN TICH LOI ==")
    print(bad.sort_values("conf_bilstm", ascending=False)[cols].to_string(index=False) if len(bad) else "(khong co)")
    print("\nThong ke nham (hang = nhan that, cot = BiLSTM doan):")
    print(pd.crosstab(df["label_true"], df["pred_bilstm"]).to_string())


if __name__ == "__main__":
    main()
