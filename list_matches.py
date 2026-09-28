"""
list_matches.py

Quet toan bo cac thu muc tran dau (moi tran = 1 thu muc chua nhieu file
set<N>.csv) da co san duoi --csv-root, in ra phan bo loai ky thuat (cot
kieu du lieu goc, tu dong do ten cot) cho tung tran - giup chon them tran
nao de mo rong du lieu ma khong can tai lai video truoc.

CACH DUNG:
    python list_matches.py --csv-root data/raw/ShuttleSet/set

Neu ban da biet ten cac gia tri trong ShuttleSet ung voi 4 ky thuat dang dung
(vd 'smash', 'clear', 'net shot', 'short service'), truyen them --labels de
loc rieng va xep hang cac tran theo tong so cu danh thuoc 4 nhan do:

    python list_matches.py --csv-root data/raw/ShuttleSet/set \
        --labels "smash,clear,net shot,short service"
"""

import argparse
import glob
import os

import pandas as pd

CANDIDATE_TYPE_COLUMNS = ["type", "ball_type", "shot_type", "stroke_type", "hit_type"]

# Giong het TYPE_MAP trong extract_stroke_clips.py - cot "type" trong CSV goc
# ShuttleSet dung nhan TIENG TRUNG, khong phai tieng Anh, nen phai dich truoc
# khi so khop voi --labels nguoi dung nhap bang tieng Anh.
TYPE_MAP = {
    "net shot": "放小球",
    "return net": "擋小球",
    "smash": "殺球",
    "wrist smash": "點扣",
    "lob": "挑球",
    "defensive return lob": "防守回挑",
    "clear": "長球",
    "drive": "平球",
    "driven flight": "小平球",
    "back-court drive": "後場抽平球",
    "drop": "切球",
    "passive drop": "過度切球",
    "push": "推球",
    "rush": "撲球",
    "defensive return drive": "防守回抽",
    "cross-court net shot": "勾球",
    "short service": "發短球",
    "long service": "發長球",
}


def find_type_column(df):
    for c in CANDIDATE_TYPE_COLUMNS:
        if c in df.columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-root", required=True,
                     help="Thu muc chua cac thu muc con, moi thu muc con la 1 tran (vd data/raw/ShuttleSet/set)")
    ap.add_argument("--labels", default=None,
                     help="Danh sach nhan quan tam, phan cach bang dau phay (vd 'smash,clear,net shot,short service'). "
                          "Neu bo qua, chi in toan bo phan bo cua tung tran.")
    args = ap.parse_args()

    wanted = None
    if args.labels:
        wanted_en = [s.strip().lower() for s in args.labels.split(",")]
        wanted = set()
        for w in wanted_en:
            if w not in TYPE_MAP:
                print(f"CANH BAO: --labels khong nhan dien duoc '{w}' (khong co trong TYPE_MAP), bo qua.")
                continue
            wanted.add(TYPE_MAP[w])

    match_dirs = sorted([d for d in glob.glob(os.path.join(args.csv_root, "*")) if os.path.isdir(d)])
    if not match_dirs:
        print(f"Khong tim thay thu muc tran nao trong {args.csv_root}")
        return

    print(f"Tim thay {len(match_dirs)} thu muc tran trong {args.csv_root}\n")

    summary_rows = []
    type_col_used = None

    for match_dir in match_dirs:
        match_name = os.path.basename(match_dir)
        csv_files = sorted(glob.glob(os.path.join(match_dir, "*.csv")))
        if not csv_files:
            continue

        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"[LOI doc] {f}: {e}")
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)

        type_col = find_type_column(df)
        if type_col is None:
            print(f"[{match_name}] Khong tim thay cot loai ky thuat trong so cac ten thu: "
                  f"{CANDIDATE_TYPE_COLUMNS}. Cac cot hien co: {list(df.columns)}")
            continue
        type_col_used = type_col

        counts = df[type_col].astype(str).str.strip().value_counts()
        total = int(counts.sum())

        if wanted:
            # dich nguoc Trung -> Anh de lam ten cot de doc
            cn_to_en = {v: k for k, v in TYPE_MAP.items()}
            row_data = {"match": match_name, "total_strokes": total}
            wanted_total = 0
            for cn_label in wanted:
                c = int(counts.get(cn_label, 0))
                en_label = cn_to_en.get(cn_label, cn_label)
                row_data[en_label] = c
                wanted_total += c
            row_data["wanted_total"] = wanted_total
            summary_rows.append(row_data)
        else:
            print(f"=== {match_name} (tong {total} cu danh) ===")
            print(counts.to_string())
            print()

    if wanted and summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values("wanted_total", ascending=False)
        print(f"\n=== Xep hang cac tran theo tong so cu danh thuoc nhan quan tam {wanted} "
              f"(doc tu cot '{type_col_used}') ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
