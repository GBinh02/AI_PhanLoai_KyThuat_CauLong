"""
run_pipeline_multi.py

Tu dong hoa toan bo chuoi xu ly (cut -> crop -> pose extract) cho NHIEU tran
cung luc, dua tren 1 file cau hinh CSV ban tu dien offset vao sau khi da xem
video va do bang mat (buoc nay KHONG the tu dong hoa an toan - xem giai thich
trong hoi thoai). Moi buoc con lai (cut/crop/pose extract) da on dinh va lap
lai giong het nhau qua cac tran, nen duoc goi tu dong qua subprocess.

BUOC 0 - Tao file cau hinh (1 lan, hoac dung --init de sinh khung tu
list_matches.py da chay):
    match_folder,youtube_url,offset
    Ten_Tran_A,https://www.youtube.com/watch?v=XXXX,
    Ten_Tran_B,https://www.youtube.com/watch?v=YYYY,

    "match_folder" phai TRUNG KHOP voi ten thu muc trong data/raw/ShuttleSet/set/
    "offset" de trong luc dau, dien vao SAU KHI da tai video va do bang mat.

BUOC 1 - Tai video cho tat ca tran trong config (offset chua can co):
    python run_pipeline_multi.py download \
        --config matches.csv --videos-dir data/videos

BUOC 2 - Xem tung video vua tai, do offset, dien vao cot "offset" trong
matches.csv (giay, so thap phan, vd 12.3).

BUOC 3 - Chay cut+crop+pose-extract cho MOI TRAN DA CO OFFSET trong config
(tran chua dien offset se tu dong bi bo qua, in canh bao):
    python run_pipeline_multi.py process \
        --config matches.csv --videos-dir data/videos \
        --csv-root data/raw/ShuttleSet/set \
        --model models/pose_landmarker_full.task \
        --clips-root data/clips_multi --cropped-root data/clips_cropped_multi \
        --features-root data/pose_features_multi \
        --types "smash,clear,net shot,short service" \
        --pre 0.98 --post 1.23 \
        --window-override "short service:0.3:0.8"

    Sau khi chay xong, in bang tong hop OK/POSE_INSUFFICIENT theo tung
    tran x tung nhan, va gop tat ca manifest vao
    <features-root>/combined_manifest.csv de de doi chieu voi muc tieu
    ~300 clip OK/nhan da thong nhat.
"""

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

# Cho phep chay script nay tu thu muc goc repo, tim dung vi tri 3 script kia.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_script(name, extra_dirs):
    """Tim file script theo ten trong vai thu muc kha di (cung cap linh
    hoat vi cac script co the nam o goc repo hoac trong ai/pose_extraction/)."""
    candidates = [os.path.join(d, name) for d in extra_dirs]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Khong tim thay {name} trong cac vi tri: {candidates}. "
        f"Dung --extract-script/--crop-script/--pose-script de chi ro duong dan."
    )


def run(cmd):
    print("Chay lenh:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_download(args):
    df = pd.read_csv(args.config)
    os.makedirs(args.videos_dir, exist_ok=True)
    extract_script = args.extract_script or find_script(
        "extract_stroke_clips.py", [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "ai", "pose_extraction")]
    )

    for _, row in df.iterrows():
        match_folder = row["match_folder"]
        url = row["youtube_url"]
        out_video_glob = os.path.join(args.videos_dir, f"{match_folder}.*")
        if glob.glob(out_video_glob):
            print(f"[BO QUA] {match_folder}: da co file video roi.")
            continue
        cmd = [
            sys.executable, extract_script, "download",
            "--youtube-url", str(url),
            "--match-folder", str(match_folder),
            "--out-dir", args.videos_dir,
        ]
        run(cmd)


def cmd_process(args):
    df = pd.read_csv(args.config)
    extract_script = args.extract_script or find_script(
        "extract_stroke_clips.py", [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "ai", "pose_extraction")]
    )
    crop_script = args.crop_script or find_script(
        "crop_player_roi.py", [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "ai", "pose_extraction")]
    )
    pose_script = args.pose_script or find_script(
        "extract_pose_features.py", [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "ai", "pose_extraction")]
    )

    os.makedirs(args.clips_root, exist_ok=True)
    os.makedirs(args.cropped_root, exist_ok=True)
    os.makedirs(args.features_root, exist_ok=True)

    processed_matches = []

    for _, row in df.iterrows():
        match_folder = row["match_folder"]
        offset = row.get("offset", None)
        if pd.isna(offset) or str(offset).strip() == "":
            print(f"[BO QUA] {match_folder}: chua dien offset trong config, can do bang mat truoc.")
            continue

        features_dir = os.path.join(args.features_root, match_folder)
        existing_manifest = os.path.join(features_dir, "features_manifest.csv")
        if os.path.isfile(existing_manifest) and not args.force:
            print(f"[DA XONG] {match_folder}: da co manifest, bo qua (dung --force de chay lai).")
            continue

        video_candidates = glob.glob(os.path.join(args.videos_dir, f"{match_folder}.*"))
        video_candidates = [v for v in video_candidates if not v.endswith(".part")]
        if not video_candidates:
            print(f"[LOI] {match_folder}: khong tim thay video trong {args.videos_dir}, bo qua.")
            continue
        video_path = video_candidates[0]

        csv_dir = os.path.join(args.csv_root, match_folder)
        if not os.path.isdir(csv_dir):
            print(f"[LOI] {match_folder}: khong tim thay {csv_dir}, bo qua.")
            continue

        clips_dir = os.path.join(args.clips_root, match_folder)
        cropped_dir = os.path.join(args.cropped_root, match_folder)

        print(f"\n===== Xu ly tran: {match_folder} (offset={offset}) =====")

        # Buoc 1: cut
        cut_cmd = [
            sys.executable, extract_script, "cut",
            "--video", video_path,
            "--csv-dir", csv_dir,
            "--offset", str(offset),
            "--types", args.types,
            "--pre", str(args.pre),
            "--post", str(args.post),
            "--out-dir", clips_dir,
        ]
        if args.window_override:
            cut_cmd += ["--window-override", args.window_override]
        run(cut_cmd)

        # Buoc 2: crop
        crop_cmd = [
            sys.executable, crop_script, "batch",
            "--clips-dir", clips_dir,
            "--csv-dir", csv_dir,
            "--out-dir", cropped_dir,
        ]
        if args.margin_override:
            crop_cmd += ["--margin-override", args.margin_override]
        run(crop_cmd)

        # Buoc 3: pose extract
        pose_cmd = [
            sys.executable, pose_script, "batch",
            "--clips-dir", cropped_dir,
            "--model", args.model,
            "--out-dir", features_dir,
            "--min-visible-ratio", str(args.min_visible_ratio),
        ]
        run(pose_cmd)

        processed_matches.append(match_folder)

    if not processed_matches:
        print("\nKhong co tran nao MOI duoc xu ly trong lan chay nay (co the deu da xong roi).")

    # Gop TOAN BO cac tran da tung xu ly (quet ca thu muc features_root, khong chi
    # gioi han trong processed_matches cua lan chay nay) - de config chi chua 1 tran
    # moi van cho ra combined_manifest.csv day du, khong ghi de mat du lieu cac tran cu.
    manifest_paths = sorted(glob.glob(os.path.join(args.features_root, "*", "features_manifest.csv")))
    if not manifest_paths:
        print("\nKhong tim thay manifest nao trong features_root de gop.")
        return

    combined_frames = []
    for manifest_path in manifest_paths:
        match_folder = os.path.basename(os.path.dirname(manifest_path))
        df_m = pd.read_csv(manifest_path)
        df_m.insert(0, "match", match_folder)
        combined_frames.append(df_m)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined_path = os.path.join(args.features_root, "combined_manifest.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\n===== TONG HOP ({len(manifest_paths)} tran co manifest, {len(processed_matches)} tran vua xu ly moi) =====")
    print(combined.groupby(["label", "status"]).size())
    print(f"\nManifest gop luu tai: {combined_path}")
    print("\nSo clip OK theo tung nhan (toan bo cac tran da xu ly, cong don voi tran dau tien "
          "neu ban gop tay combined_manifest cu vao):")
    print(combined[combined["status"] == "OK"].groupby("label").size())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Tai video cho tat ca tran trong file config")
    p_dl.add_argument("--config", required=True, help="File CSV: match_folder,youtube_url,offset")
    p_dl.add_argument("--videos-dir", default="data/videos")
    p_dl.add_argument("--extract-script", default=None, help="Duong dan extract_stroke_clips.py (tu dong tim neu bo qua)")
    p_dl.set_defaults(func=cmd_download)

    p_proc = sub.add_parser("process", help="Chay cut+crop+pose-extract cho moi tran da co offset trong config")
    p_proc.add_argument("--config", required=True, help="File CSV: match_folder,youtube_url,offset")
    p_proc.add_argument("--videos-dir", default="data/videos")
    p_proc.add_argument("--csv-root", default="data/raw/ShuttleSet/set")
    p_proc.add_argument("--clips-root", default="data/clips_multi")
    p_proc.add_argument("--cropped-root", default="data/clips_cropped_multi")
    p_proc.add_argument("--features-root", default="data/pose_features_multi")
    p_proc.add_argument("--model", default="models/pose_landmarker_full.task")
    p_proc.add_argument("--types", default="smash,clear,net shot,short service")
    p_proc.add_argument("--pre", type=float, default=0.98)
    p_proc.add_argument("--post", type=float, default=1.23)
    p_proc.add_argument("--window-override", default="short service:0.3:0.8")
    p_proc.add_argument("--margin-override", default=None)
    p_proc.add_argument("--min-visible-ratio", type=float, default=0.7)
    p_proc.add_argument("--force", action="store_true",
                         help="Chay lai tran da co manifest roi (mac dinh se bo qua neu da xong)")
    p_proc.add_argument("--extract-script", default=None)
    p_proc.add_argument("--crop-script", default=None)
    p_proc.add_argument("--pose-script", default=None)
    p_proc.set_defaults(func=cmd_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()