"""
crop_player_roi.py

Crop tung clip quanh vi tri nguoi choi (dung player_location_x/y co san trong
CSV ShuttleSet) truoc khi dua vao MediaPipe Pose. Giai quyet van de: video
trang dau that quay toan san, nguoi choi qua nho/xa lam MediaPipe khong
nhan dien duoc (hoac bat nham vat the khac o nguong thap).

CO SO: kiem tra tren toan bo 44 tran trong homography.csv cho thay toa do
pixel goc san (upleft/upright/downleft/downright) luon nam trong khung
1280x720 -> day la do phan giai video goc dung khi gan nhan ShuttleSet.
Neu video ban tai co do phan giai khac, script tu dong ty le lai toa do.

YEU CAU CAI DAT:
    pip install pandas numpy

CACH DUNG - crop thu 1 clip de kiem chung truoc:
    python crop_player_roi.py crop-one \
        --clip data/clips/clear/set3_rally33_ball26.0.mp4 \
        --csv-dir "data/raw/ShuttleSet/set/Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals" \
        --out data/clips_cropped/clear/set3_rally33_ball26.0.mp4

    Ten clip phai dung dinh dang "set<N>_rally<R>_ball<B>.mp4" (dung dinh dang
    script extract_stroke_clips.py da tao ra) de tu dong tra cuu dong CSV tuong ung.

CACH DUNG - crop hang loat toan bo thu muc clips:
    python crop_player_roi.py batch \
        --clips-dir data/clips \
        --csv-dir "data/raw/ShuttleSet/set/Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals" \
        --out-dir data/clips_cropped

Sau khi crop xong, chay lai buoc debug/pose extraction tren clip trong
data/clips_cropped/ thay vi data/clips/.
"""

import argparse
import glob
import os
import re
import subprocess

import numpy as np
import pandas as pd

# Do phan giai video goc dung khi gan nhan ShuttleSet (xac nhan qua homography.csv)
ANNOT_W, ANNOT_H = 1280, 720

# Do cao hop crop noi suy theo do "xa" cua nguoi choi (dua tren pham vi y quan sat
# duoc trong homography.csv: y~236 la xa nhat/gan luoi, y~695 la gan nhat/gan camera)
Y_FAR, Y_NEAR = 236.0, 695.0
BOX_H_FAR, BOX_H_NEAR = 180.0, 480.0
BOX_ASPECT = 0.65  # ty le rong/cao


def probe_video_size(video_path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe loi khi doc {video_path}:\n{result.stderr}")
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def _single_point_bounds(x, y, scale_x, scale_y, margin):
    """Tinh bien crop (left, top, right, bottom) cho 1 diem vi tri nguoi choi."""
    cx = x * scale_x
    cy = y * scale_y
    t = np.clip((y - Y_FAR) / (Y_NEAR - Y_FAR), 0.0, 1.0)
    box_h = (BOX_H_FAR + t * (BOX_H_NEAR - BOX_H_FAR)) * scale_y * margin
    box_w = box_h * BOX_ASPECT
    # player_location la vi tri duoi chan; danh phan lon hop len phia tren, chua it ben duoi
    bottom = cy + 0.12 * box_h
    top = bottom - box_h
    left = cx - box_w / 2
    right = cx + box_w / 2
    return left, top, right, bottom


def compute_crop_box(cur_x, cur_y, actual_w, actual_h, margin=1.0, prev_x=None, prev_y=None):
    """Tinh vung crop (x, y, w, h) quanh vi tri nguoi choi tai thoi diem tiep xuc cau
    (cur_x, cur_y). Neu co them vi tri o lan danh truoc do cua CHINH nguoi choi nay
    trong cung rally (prev_x, prev_y), hop crop se duoc MO RONG de bao trum ca 2 diem -
    giai quyet dung nguyen nhan: nguoi choi con dang di chuyen/lao vao vi tri trong
    khoang --pre truoc khi tiep xuc cau, nen chua co mat trong hop crop tinh o mot
    diem duy nhat vao dau clip.

    Ty le lai toa do neu video thuc te khac do phan giai goc dung khi gan nhan (1280x720)."""
    scale_x = actual_w / ANNOT_W
    scale_y = actual_h / ANNOT_H

    left, top, right, bottom = _single_point_bounds(cur_x, cur_y, scale_x, scale_y, margin)

    if prev_x is not None and prev_y is not None:
        l2, t2, r2, b2 = _single_point_bounds(prev_x, prev_y, scale_x, scale_y, margin)
        left = min(left, l2)
        top = min(top, t2)
        right = max(right, r2)
        bottom = max(bottom, b2)

    left = max(0, left)
    top = max(0, top)
    right = min(actual_w, right)
    bottom = min(actual_h, bottom)

    w = right - left
    h = bottom - top
    # ffmpeg crop can kich thuoc chan (chia het cho 2) de tuong thich yuv420p
    w = int(w) - (int(w) % 2)
    h = int(h) - (int(h) % 2)
    x = int(left)
    y = int(top)
    return x, y, w, h


def lookup_player_location(csv_dir, set_name, rally, ball_round):
    """Tra ve (cur_x, cur_y, prev_x, prev_y). prev_x/prev_y la vi tri o lan danh
    TRUOC DO GAN NHAT cua CHINH nguoi choi nay trong cung rally (thuong la 2 nhip
    truoc, vi 2 nguoi choi danh xen ke) - None neu day la cu danh dau tien cua ho
    trong rally (vd giao cau, khong co gi de tham chieu)."""
    csv_path = os.path.join(csv_dir, f"{set_name}.csv")
    df = pd.read_csv(csv_path)
    row = df[(df["rally"] == rally) & (df["ball_round"] == ball_round)]
    if row.empty:
        raise ValueError(f"Khong tim thay rally={rally}, ball_round={ball_round} trong {csv_path}")
    r = row.iloc[0]
    cur_x, cur_y = float(r["player_location_x"]), float(r["player_location_y"])
    player = r["player"]

    same_rally = df[df["rally"] == rally]
    prev_rows = same_rally[(same_rally["player"] == player) & (same_rally["ball_round"] < ball_round)]
    if prev_rows.empty:
        return cur_x, cur_y, None, None
    prev_row = prev_rows.sort_values("ball_round").iloc[-1]
    prev_x, prev_y = float(prev_row["player_location_x"]), float(prev_row["player_location_y"])
    return cur_x, cur_y, prev_x, prev_y


def parse_clip_filename(clip_path):
    """Doc set_name, rally, ball_round tu ten file dang set<N>_rally<R>_ball<B>.mp4."""
    name = os.path.splitext(os.path.basename(clip_path))[0]
    m = re.match(r"(set\d+)_rally(\d+)_ball([\d.]+)", name)
    if not m:
        raise ValueError(f"Ten file khong dung dinh dang set<N>_rally<R>_ball<B>: {clip_path}")
    set_name = m.group(1)
    rally = int(m.group(2))
    ball_round = float(m.group(3))
    return set_name, rally, ball_round


def parse_margin_override(s):
    """Doc chuoi dang 'label:margin;label2:margin2' (vd 'short_service:0.5')
    thanh dict {label: margin}. Tra ve dict rong neu s la None/rong."""
    overrides = {}
    if not s:
        return overrides
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        label, val = part.split(":")
        overrides[label.strip()] = float(val)
    return overrides


def crop_clip(clip_path, csv_dir, out_path, margin=1.0):
    set_name, rally, ball_round = parse_clip_filename(clip_path)
    cur_x, cur_y, prev_x, prev_y = lookup_player_location(csv_dir, set_name, rally, ball_round)

    actual_w, actual_h = probe_video_size(clip_path)
    x, y, w, h = compute_crop_box(cur_x, cur_y, actual_w, actual_h, margin, prev_x, prev_y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", clip_path,
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return x, y, w, h


def cmd_crop_one(args):
    x, y, w, h = crop_clip(args.clip, args.csv_dir, args.out, args.margin)
    print(f"Da crop: x={x} y={y} w={w} h={h}")
    print(f"Luu tai: {args.out}")


def cmd_batch(args):
    margin_overrides = parse_margin_override(args.margin_override)

    clip_files = sorted(glob.glob(os.path.join(args.clips_dir, "*", "*.mp4")))
    if not clip_files:
        print(f"Khong tim thay clip nao trong {args.clips_dir}")
        return

    ok, failed = 0, 0
    for clip_path in clip_files:
        label = os.path.basename(os.path.dirname(clip_path))
        clip_name = os.path.basename(clip_path)
        out_path = os.path.join(args.out_dir, label, clip_name)
        margin = margin_overrides.get(label, args.margin)
        try:
            x, y, w, h = crop_clip(clip_path, args.csv_dir, out_path, margin)
            print(f"[OK] {clip_path} -> crop({w}x{h}) -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[LOI] {clip_path}: {e}")
            failed += 1

    print(f"\nHoan tat: {ok} thanh cong, {failed} loi.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_one = sub.add_parser("crop-one", help="Crop thu 1 clip de kiem chung")
    p_one.add_argument("--clip", required=True)
    p_one.add_argument("--csv-dir", required=True)
    p_one.add_argument("--out", required=True)
    p_one.add_argument("--margin", type=float, default=1.0, help="He so noi rong hop crop, vd 1.3 = rong hon 30%%")
    p_one.set_defaults(func=cmd_crop_one)

    p_batch = sub.add_parser("batch", help="Crop hang loat toan bo clip trong 1 thu muc")
    p_batch.add_argument("--clips-dir", default="data/clips")
    p_batch.add_argument("--csv-dir", required=True)
    p_batch.add_argument("--out-dir", default="data/clips_cropped")
    p_batch.add_argument("--margin", type=float, default=1.0, help="He so noi rong hop crop, vd 1.3 = rong hon 30%%")
    p_batch.add_argument("--margin-override", default=None,
                          help="Ghi de margin rieng cho tung nhan, dang 'label:margin;label2:margin2', "
                               "vd 'short_service:0.5'. Nhan khong duoc liet ke van dung --margin mac dinh.")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()