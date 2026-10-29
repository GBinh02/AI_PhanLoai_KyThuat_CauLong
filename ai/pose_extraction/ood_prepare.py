"""
ood_prepare.py - Chuan bi tap kiem thu out-of-domain tu video tu quay.

VAN DE CAN LUU Y: ShuttleSet dinh nghia "time" la thoi diem VOT CHAM CAU, va mo hinh
da duoc huan luyen voi cua so cat quanh thoi diem do (0.98 s truoc + 1.23 s sau;
short_service: 0.3 s truoc + 0.8 s sau). Neu time_sec trong CSV la luc nguoi choi
dang chuan bi (som hon luc cham cau 0.3-0.8 s) thi clip se bi lech khung thoi gian
va ket qua OOD se sai vi lech nay, khong phai do mo hinh kem. Script nay giup ban
tim dung thoi diem cham cau roi ghi vao cot MOI "contact_sec" (giu nguyen time_sec).

CSV: video,time_sec,label,player_side,player_id[,contact_sec]
  - Moi video la 1 cap nguoi choi khac nhau -> script tu gop thanh ma nguoi
    "<ten_video>_<player_id>" (vd video4_player1) de bao cao theo tung nguoi.
  - Luu CSV dang "CSV UTF-8 (comma delimited)"; dung dau CHAM thap phan (12.4).
    Script co chap nhan dau phay thap phan / dau cham phay, nhung tot nhat la tranh.

BUOC 1 - Tao anh doi chieu cho tung dong:
    python ai/pose_extraction/ood_prepare.py sheet \
        --csv data/ood/ood_annotations.csv --videos-dir data/ood/videos \
        --out-dir data/ood/review
    -> data/ood/review/<stt>_<video>_<nhan>_<ben>.jpg : luoi cac frame tu
       (time_sec - 0.2 s) den (time_sec + 1.2 s), moi o ghi san thoi gian (giay).
       Tim o co vot cham cau, doc so giay tren o do.
    -> data/ood/ood_annotations_review.csv : ban sao CSV co them cot contact_sec
       de trong. Dien contact_sec vao do (giu nguyen time_sec).
    Voi short_service nen lam min hon:  --step 0.05

BUOC 2 - Cat clip khi da dien xong contact_sec cho moi dong:
    python ai/pose_extraction/ood_prepare.py cut \
        --csv data/ood/ood_annotations_review.csv --videos-dir data/ood/videos \
        --out-dir data/ood/clips
    -> data/ood/clips/<nhan>/<video>_t<contact>.mp4 va data/ood/clips_manifest.csv
       (chua crop - buoc crop lam sau).
"""
import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LABELS = ["clear", "net_shot", "short_service", "smash"]
DEFAULT_WIN = (0.98, 1.23)                 # (pre, post) giong luc huan luyen
WINDOWS = {"short_service": (0.3, 0.8)}
REQUIRED = ["video", "time_sec", "label", "player_side", "player_id"]


def load_csv(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV thieu cot: {missing}")
    for c in ["video", "label", "player_side", "player_id"]:
        df[c] = df[c].astype(str).str.strip()
    df["player_id"] = df["player_id"].str.replace("_", "", regex=False)
    df["person"] = df["video"].str.replace(".mp4", "", regex=False) + "_" + df["player_id"]
    for c in ["time_sec", "contact_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise SystemExit(f"Nhan khong hop le: {bad}. Chi dung: {LABELS}")
    return df


def probe_duration(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(video)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe loi: {r.stderr}")
    return float(r.stdout.strip())


def grab_frame(video, t, width):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def cmd_sheet(a):
    df = load_csv(a.csv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_steps = int(round((a.span + a.back) / a.step)) + 1

    for i, row in df.iterrows():
        video = Path(a.videos_dir) / row["video"]
        if not video.is_file():
            print(f"[BO QUA] dong {i}: khong thay {video}")
            continue
        t0 = max(0.0, row["time_sec"] - a.back)
        tiles = []
        for k in range(n_steps):
            t = t0 + k * a.step
            img = grab_frame(video, t, a.tile_width)
            if img is None:
                img = np.zeros((int(a.tile_width * 9 / 16), a.tile_width, 3), np.uint8)
            else:
                img = img.copy()
            cv2.putText(img, f"{t:.2f}s", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            tiles.append(img)
        h, w = tiles[0].shape[:2]
        rows_n = math.ceil(len(tiles) / a.cols)
        grid = np.zeros((rows_n * h, a.cols * w, 3), np.uint8)
        for k, tile in enumerate(tiles):
            r_, c_ = divmod(k, a.cols)
            grid[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = cv2.resize(tile, (w, h))
        banner = np.zeros((44, grid.shape[1], 3), np.uint8)
        text = f"#{i} {row['video']} | {row['label']} | {row['player_side']} | time_sec={row['time_sec']:.2f}"
        cv2.putText(banner, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        name = f"{i:03d}_{Path(row['video']).stem}_{row['label']}_{row['player_side']}.jpg"
        cv2.imwrite(str(out / name), np.vstack([banner, grid]))
        print("Da tao", name)

    review = df[REQUIRED + (["contact_sec"] if "contact_sec" in df.columns else [])].copy()
    if "contact_sec" not in review.columns:
        review["contact_sec"] = ""
    review_path = Path(a.csv).with_name("ood_annotations_review.csv")
    review.to_csv(review_path, index=False, lineterminator="\n")
    print(f"\nDa tao {review_path}. Dien contact_sec cho tung dong, roi chay 'cut'.")


def cmd_cut(a):
    df = load_csv(a.csv)
    if "contact_sec" not in df.columns or df["contact_sec"].isna().any():
        n = len(df) if "contact_sec" not in df.columns else int(df["contact_sec"].isna().sum())
        raise SystemExit(f"Con {n} dong chua co contact_sec. Dien du roi chay lai.")

    durations, records = {}, []
    for i, row in df.iterrows():
        video = Path(a.videos_dir) / row["video"]
        if not video.is_file():
            print(f"[BO QUA] dong {i}: khong thay {video}")
            continue
        if row["video"] not in durations:
            durations[row["video"]] = probe_duration(video)
        pre, post = WINDOWS.get(row["label"], DEFAULT_WIN)
        c = float(row["contact_sec"])
        start = c - pre
        if start < 0 or c + post > durations[row["video"]]:
            print(f"[BO QUA] dong {i}: cua so [{start:.2f}, {c + post:.2f}] vuot ngoai video "
                  f"(dai {durations[row['video']]:.1f}s)")
            continue
        out_dir = Path(a.out_dir) / row["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        clip = out_dir / f"{Path(row['video']).stem}_t{c:.2f}.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{pre + post:.3f}",
                        "-c:v", "libx264", "-c:a", "aac", "-loglevel", "error", str(clip)], check=True)
        records.append({"video": row["video"], "contact_sec": c, "label": row["label"],
                        "player_side": row["player_side"], "person": row["person"], "clip": str(clip)})

    manifest = Path(a.out_dir).parent / "clips_manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False, lineterminator="\n")
    print(f"\nDa cat {len(records)}/{len(df)} clip. Manifest: {manifest}")
    print(pd.DataFrame(records).groupby("label").size())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sheet")
    p.add_argument("--csv", default="data/ood/ood_annotations.csv")
    p.add_argument("--videos-dir", default="data/ood/videos")
    p.add_argument("--out-dir", default="data/ood/review")
    p.add_argument("--back", type=float, default=0.2, help="Lay them bao nhieu giay TRUOC time_sec")
    p.add_argument("--span", type=float, default=1.2, help="Lay bao nhieu giay SAU time_sec")
    p.add_argument("--step", type=float, default=0.1, help="Buoc thoi gian giua cac o (giay)")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--tile-width", type=int, default=640)
    p.set_defaults(fn=cmd_sheet)

    p = sub.add_parser("cut")
    p.add_argument("--csv", default="data/ood/ood_annotations_review.csv")
    p.add_argument("--videos-dir", default="data/ood/videos")
    p.add_argument("--out-dir", default="data/ood/clips")
    p.set_defaults(fn=cmd_cut)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
