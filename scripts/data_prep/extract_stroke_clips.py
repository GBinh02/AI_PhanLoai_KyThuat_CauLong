"""
extract_stroke_clips.py

Muc dich: Tu dong tai video tran dau tu YouTube va cat clip cho tung cu danh,
dua tren du lieu stroke-level cua ShuttleSet (set1.csv, set2.csv, set3.csv).

YEU CAU CAI DAT (chay 1 lan):
    pip install yt-dlp pandas
    # Va can ffmpeg da cai san tren may (ffmpeg.org / apt install ffmpeg / choco install ffmpeg)

CACH DUNG (2 buoc):

Buoc 1 - Tai video ve va tu hieu chinh offset:
    python extract_stroke_clips.py download \
        --youtube-url "https://www.youtube.com/watch?v=XXXXXXX" \
        --match-folder "Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals" \
        --out-dir ./videos

    Sau khi tai xong, mo video len, tim bang mat thoi diem giao cau dau tien
    cua set1 (rally=1, ball_round=1), so sanh voi cot "time" cua dong dau tien
    trong set1.csv de tinh offset (giay).
    offset = thoi_diem_trong_video_ban_xem - thoi_diem_ghi_trong_csv

Buoc 2 - Cat clip cho 1 hoac nhieu loai ky thuat, da hieu chinh offset:
    python extract_stroke_clips.py cut \
        --video ./videos/Viktor_Axelsen_..._SemiFinals.mp4 \
        --csv-dir "./ShuttleSet/set/Viktor_Axelsen_..._SemiFinals" \
        --offset 12.3 \
        --types "smash,clear,net shot,short service,long service" \
        --pre 0.7 --post 0.4 \
        --out-dir ./clips
"""

import argparse
import os
import subprocess
import sys
import glob
import pandas as pd

# Ban dich loai ky thuat tieng Anh -> nhan tieng Trung dung trong CSV goc
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


def cmd_download(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_template = os.path.join(args.out_dir, f"{args.match_folder}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", out_template,
        args.youtube_url,
    ]
    print("Chay lenh:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDa tai xong vao thu muc {args.out_dir}/")
    print("BUOC TIEP THEO (thu cong): mo video, tim thoi diem giao cau dau tien,")
    print("so sanh voi dong dau tien cua set1.csv de tinh offset (giay), roi chay lenh 'cut'.")


def hhmmss_to_sec(t):
    # cot "time" trong CSV co dang hr:min:sec
    parts = str(t).split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def parse_window_overrides(spec):
    """Phan tich chuoi dang 'short service:0.3:0.6;smash:1.0:1.5' thanh
    {'發短球': (0.3, 0.6), '殺球': (1.0, 1.5)} de ghi de --pre/--post rieng
    cho tung loai ky thuat (vd short service can pre ngan hon de tranh
    dinh doan can canh/2 nguoi ma broadcast chen vao truoc luc giao cau)."""
    overrides = {}
    if not spec:
        return overrides
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        type_en, pre_str, post_str = entry.split(":")
        type_en = type_en.strip().lower()
        if type_en not in TYPE_MAP:
            print(f"CANH BAO: --window-override khong nhan dien duoc loai '{type_en}', bo qua.")
            continue
        overrides[TYPE_MAP[type_en]] = (float(pre_str), float(post_str))
    return overrides


def cmd_cut(args):
    os.makedirs(args.out_dir, exist_ok=True)

    wanted_types_en = [t.strip().lower() for t in args.types.split(",")]
    wanted_types_cn = set()
    for t in wanted_types_en:
        if t not in TYPE_MAP:
            print(f"CANH BAO: khong nhan dien duoc loai ky thuat '{t}', bo qua.")
            continue
        wanted_types_cn.add(TYPE_MAP[t])

    window_overrides = parse_window_overrides(args.window_override)

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "set*.csv")))
    if not csv_files:
        print(f"Khong tim thay set*.csv trong {args.csv_dir}")
        sys.exit(1)

    total_cut = 0
    for csv_file in csv_files:
        set_name = os.path.splitext(os.path.basename(csv_file))[0]  # set1, set2...
        df = pd.read_csv(csv_file)
        df = df[df["type"].isin(wanted_types_cn)]

        for idx, row in df.iterrows():
            label_cn = row["type"]
            # tim lai nhan tieng Anh de dat ten thu muc de doc
            label_en = next((k for k, v in TYPE_MAP.items() if v == label_cn), label_cn)
            label_dir = os.path.join(args.out_dir, label_en.replace(" ", "_"))
            os.makedirs(label_dir, exist_ok=True)

            pre, post = window_overrides.get(label_cn, (args.pre, args.post))

            t_center = hhmmss_to_sec(row["time"]) + args.offset
            t_start = max(0, t_center - pre)
            duration = pre + post

            clip_name = f"{set_name}_rally{row['rally']}_ball{row['ball_round']}.mp4"
            out_path = os.path.join(label_dir, clip_name)

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{t_start:.3f}",
                "-i", args.video,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-c:a", "aac",
                "-loglevel", "error",
                out_path,
            ]
            subprocess.run(cmd, check=True)
            total_cut += 1

        print(f"{set_name}: da cat {len(df)} clip")

    print(f"\nTong cong: {total_cut} clip da duoc cat vao {args.out_dir}/<loai_ky_thuat>/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Tai video tran dau tu YouTube")
    p_dl.add_argument("--youtube-url", required=True)
    p_dl.add_argument("--match-folder", required=True, help="Ten thu muc trong ShuttleSet/set/ung voi tran nay")
    p_dl.add_argument("--out-dir", default="./videos")
    p_dl.set_defaults(func=cmd_download)

    p_cut = sub.add_parser("cut", help="Cat clip theo frame_num/time trong CSV")
    p_cut.add_argument("--video", required=True, help="Duong dan file video da tai")
    p_cut.add_argument("--csv-dir", required=True, help="Duong dan thu muc chua set1.csv/set2.csv/set3.csv cua tran nay")
    p_cut.add_argument("--offset", type=float, required=True, help="Do lech giay giua video va thoi gian trong CSV (xem huong dan hieu chinh o dau file)")
    p_cut.add_argument("--types", required=True, help="Danh sach loai ky thuat, phan cach boi dau phay, vd: 'smash,clear,net shot,short service,long service'")
    p_cut.add_argument("--pre", type=float, default=0.7, help="So giay lay truoc thoi diem tiep xuc cau (mac dinh 0.7)")
    p_cut.add_argument("--post", type=float, default=0.4, help="So giay lay sau thoi diem tiep xuc cau (mac dinh 0.4)")
    p_cut.add_argument("--window-override", default="",
                        help="Ghi de --pre/--post rieng cho tung loai ky thuat, dinh dang "
                             "'loai:pre:post;loai2:pre2:post2', vd 'short service:0.3:0.8'")
    p_cut.add_argument("--out-dir", default="./clips")
    p_cut.set_defaults(func=cmd_cut)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()