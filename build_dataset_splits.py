"""
build_dataset_splits.py

Doc data/dataset_manifest.csv (cot: match, clip, label, status, visible_ratio,
wrist_ratio, feature_path), loc status == OK, gan moi tran vao 1 trong 3 tap
train/val/test theo danh sach TEN TRAN ban chi dinh (chia theo tran, KHONG
xao tron ngau nhien - dung yeu cau muc 7 dac ta), roi load tung file .npy
tuong ung, gop thanh mang X (N, T, F) va y (N,) cho tung tap, luu ra file
.npz san sang dua vao huan luyen LSTM/MLP.

CACH DUNG (chay tu goc repo, sau khi da co data/dataset_manifest.csv):
    python build_dataset_splits.py \
        --manifest data/dataset_manifest.csv \
        --train "Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals,Anders_ANTONSEN_Jonatan_CHRISTIE_Indonesia_Masters_2020_QuarterFinals,Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals,Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals" \
        --val "NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals" \
        --test "Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final" \
        --out-dir data/dataset_splits

Moi file .npz co 4 mang:
    X            - (N, T, F) dac trung pose da chuan hoa/resample san (tu .npy)
    y            - (N,) nhan dang so nguyen (xem label_names de biet thu tu)
    label_names  - danh sach ten nhan theo dung thu tu ma y dang tham chieu
    clip         - (N,) duong dan file clip goc, de truy vet loi neu can
"""

import argparse
import os

import numpy as np
import pandas as pd

LABELS = ["clear", "net_shot", "short_service", "smash"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


def build_split(df_manifest, match_names, split_name, out_dir):
    df_split = df_manifest[df_manifest["match"].isin(match_names)]
    if df_split.empty:
        print(f"[CANH BAO] Khong co dong nao cho tap '{split_name}' voi cac tran: {match_names}")
        return

    missing_labels = set(df_split["label"].unique()) - set(LABELS)
    if missing_labels:
        print(f"[CANH BAO] Nhan khong nam trong LABELS da khai bao, se bi bo qua: {missing_labels}")
        df_split = df_split[df_split["label"].isin(LABELS)]

    X_list, y_list, clip_list, match_list = [], [], [], []
    n_missing_file = 0

    for _, row in df_split.iterrows():
        feat_path = row["feature_path"]
        if not isinstance(feat_path, str) or not os.path.isfile(feat_path):
            n_missing_file += 1
            continue
        feat = np.load(feat_path)
        X_list.append(feat)
        y_list.append(LABEL_TO_IDX[row["label"]])
        clip_list.append(row["clip"])
        match_list.append(row["match"])

    if n_missing_file > 0:
        print(f"[CANH BAO] {split_name}: {n_missing_file} dong co feature_path khong doc duoc, da bo qua.")

    if not X_list:
        print(f"[CANH BAO] Tap '{split_name}' rong sau khi loc, khong luu file.")
        return

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    clip = np.array(clip_list, dtype=object)
    match_arr = np.array(match_list, dtype=object)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split_name}.npz")
    np.savez(out_path, X=X, y=y, label_names=np.array(LABELS, dtype=object), clip=clip, match=match_arr)

    print(f"\n=== {split_name}.npz ({X.shape[0]} clip, X shape {X.shape}) ===")
    counts = pd.Series(y).map(dict(enumerate(LABELS))).value_counts().reindex(LABELS, fill_value=0)
    print(counts.to_string())
    print(f"Luu tai: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/dataset_manifest.csv")
    ap.add_argument("--train", required=True, help="Danh sach ten tran cho tap train, phan cach boi dau phay")
    ap.add_argument("--val", required=True, help="Danh sach ten tran cho tap val, phan cach boi dau phay")
    ap.add_argument("--test", required=True, help="Danh sach ten tran cho tap test, phan cach boi dau phay")
    ap.add_argument("--out-dir", default="data/dataset_splits")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["status"] == "OK"].copy()

    train_matches = [m.strip() for m in args.train.split(",") if m.strip()]
    val_matches = [m.strip() for m in args.val.split(",") if m.strip()]
    test_matches = [m.strip() for m in args.test.split(",") if m.strip()]

    all_named = set(train_matches) | set(val_matches) | set(test_matches)
    all_in_manifest = set(df["match"].unique())
    unknown = all_named - all_in_manifest
    if unknown:
        print(f"[CANH BAO] Cac ten tran sau khong khop voi cot 'match' trong manifest (kiem tra lai chinh ta): {unknown}")
    unassigned = all_in_manifest - all_named
    if unassigned:
        print(f"[CANH BAO] Cac tran co trong manifest nhung CHUA duoc gan vao split nao (se bi bo qua): {unassigned}")

    build_split(df, train_matches, "train", args.out_dir)
    build_split(df, val_matches, "val", args.out_dir)
    build_split(df, test_matches, "test", args.out_dir)


if __name__ == "__main__":
    main()
