"""
extract_pose_features.py

Trich xuat 33 keypoint MediaPipe Pose (Tasks API - PoseLandmarker) tu moi clip,
chuan hoa va resample ve chuoi co do dai co dinh T=32 buoc, F=132 dac trung
(33 x [x,y,z,visibility]), dung theo dung quy uoc trong tai lieu thiet ke.

LUU Y QUAN TRONG (2026): tu ban mediapipe 0.10.31+/1.0.x, Google da GO BO
hoan toan API cu "mp.solutions" (Pose, drawing_utils...). Script nay dung
API moi "MediaPipe Tasks" (mp.tasks.vision.PoseLandmarker), yeu cau tai rieng
1 file model .task.

YEU CAU CAI DAT:
    pip install mediapipe opencv-python numpy pandas

BUOC 0 - TAI MODEL (chi can lam 1 lan):
    python extract_pose_features.py download-model --variant full

    Se tai file models/pose_landmarker_full.task (~9MB). Co 3 variant:
    lite (nhanh nhat, do chinh xac thap nhat), full (can bang - khuyen dung),
    heavy (chinh xac nhat, cham nhat).

BUOC 1 - KIEM CHUNG TRUOC (chua crop, xuat video debug de xem MediaPipe
co bat dung nguoi choi khong):

    python extract_pose_features.py debug \
        --clip data/clips/smash/set1_rally3_ball5.mp4 \
        --model models/pose_landmarker_full.task \
        --out-video debug_pose_overlay.mp4

    Mo debug_pose_overlay.mp4 len xem: khung xuong co bam dung 1 VDV dang danh cau
    xuyen suot clip khong, hay nhay lung tung giua 2 nguoi / bat nham khan gia.

BUOC 2 - chay hang loat sau khi da xac nhan on (hoac da them crop):

    python extract_pose_features.py batch \
        --clips-dir data/clips \
        --model models/pose_landmarker_full.task \
        --out-dir data/pose_features \
        --min-visible-ratio 0.7

    Ket qua: moi clip -> 1 file .npy shape (32, 132) trong
    data/pose_features/<nhan>/<ten_clip>.npy, cung 1 file features_manifest.csv
    tong hop trang thai (OK / POSE_INSUFFICIENT) cho tung clip.
"""

import argparse
import glob
import os
import urllib.request

import cv2
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

NUM_LANDMARKS = 33
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
VISIBILITY_THRESHOLD = 0.5
MAX_GAP_INTERP = 3  # noi suy toi da 3 frame lien tiep bi thieu
TARGET_T = 32
EPS = 1e-6

def probe_video(video_path):
    """Doc width, height, fps cua video bang ffprobe (khong dung cv2.VideoCapture,
    vi ban OpenCV cai qua pip tren mot so may Windows khong mo duoc file .mp4
    du da ep dung backend FFmpeg)."""
    import json
    import subprocess

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe loi khi doc {video_path}:\n{result.stderr}")
    info = json.loads(result.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])

    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) != 0 else 25.0
    if fps <= 0:
        fps = 25.0
    return w, h, fps


def read_frames_via_ffmpeg(video_path, w, h):
    """Generator: giai ma video bang ffmpeg, tra ve tung frame BGR (numpy array h,w,3)
    qua pipe, khong dung cv2.VideoCapture."""
    import subprocess

    frame_size = w * h * 3
    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            yield frame
    finally:
        proc.stdout.close()
        proc.wait()


MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

# Danh sach canh de ve khung xuong debug (thay the cho mp.solutions.drawing_utils da bi go bo)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
]


def cmd_download_model(args):
    url = MODEL_URLS[args.variant]
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", f"pose_landmarker_{args.variant}.task")
    print(f"Dang tai {url} -> {out_path} ...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Da tai xong: {out_path}")


def make_landmarker(model_path, running_mode, min_confidence=0.5):
    base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_poses=1,  # chi lay 1 nguoi noi bat nhat moi frame
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_raw_landmarks(video_path, model_path, min_confidence=0.5):
    """Chay PoseLandmarker (Tasks API, che do VIDEO) tren tung frame.
    Tra ve mang (T_goc, 33, 4) = x,y,z,visibility."""
    w, h, fps = probe_video(video_path)
    frames_landmarks = []

    landmarker = make_landmarker(model_path, mp_vision.RunningMode.VIDEO, min_confidence)
    frame_idx = 0
    try:
        for frame in read_frames_via_ffmpeg(video_path, w, h):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(frame_idx * 1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                lm_array = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
                lm_array[:, 3] = 0.0
            else:
                pose = result.pose_landmarks[0]  # nguoi noi bat nhat (num_poses=1)
                lm_array = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose],
                    dtype=np.float32,
                )
            frames_landmarks.append(lm_array)
            frame_idx += 1
    finally:
        landmarker.close()

    if len(frames_landmarks) == 0:
        return None
    return np.stack(frames_landmarks, axis=0)  # (T_goc, 33, 4)


def normalize_sequence(raw):
    """
    raw: (T, 33, 4) x,y,z,visibility tho tu MediaPipe (x,y da o dang ty le 0..1 theo khung hinh).
    Chuan hoa: tru tam hong, chia do dai than (vai-hong), giu z theo quy uoc MediaPipe.
    """
    xy = raw[:, :, 0:2].copy()
    z = raw[:, :, 2:3].copy()
    vis = raw[:, :, 3:4].copy()

    hip_center = (xy[:, LEFT_HIP, :] + xy[:, RIGHT_HIP, :]) / 2.0
    shoulder_center = (xy[:, LEFT_SHOULDER, :] + xy[:, RIGHT_SHOULDER, :]) / 2.0
    torso_len = np.linalg.norm(shoulder_center - hip_center, axis=1, keepdims=True)
    torso_len = np.maximum(torso_len, EPS)

    xy_norm = (xy - hip_center[:, None, :]) / torso_len[:, None, :]
    z_norm = z

    missing_mask = vis[:, :, 0] < VISIBILITY_THRESHOLD
    return xy_norm, z_norm, vis, missing_mask


def interpolate_short_gaps(xy_norm, z_norm, vis, missing_mask):
    """Noi suy theo thoi gian cho tung landmark, chi voi khoang trong <= MAX_GAP_INTERP frame."""
    T, L, _ = xy_norm.shape
    xy_out = xy_norm.copy()
    z_out = z_norm.copy()
    still_missing = missing_mask.copy()

    for lm_idx in range(L):
        miss = missing_mask[:, lm_idx]
        if not miss.any():
            continue
        t = 0
        while t < T:
            if miss[t]:
                start = t
                while t < T and miss[t]:
                    t += 1
                end = t
                gap_len = end - start
                has_left = start > 0
                has_right = end < T
                if gap_len <= MAX_GAP_INTERP and has_left and has_right:
                    for coord in range(2):
                        xy_out[start:end, lm_idx, coord] = np.interp(
                            np.arange(start, end),
                            [start - 1, end],
                            [xy_norm[start - 1, lm_idx, coord], xy_norm[end, lm_idx, coord]],
                        )
                    z_out[start:end, lm_idx, 0] = np.interp(
                        np.arange(start, end),
                        [start - 1, end],
                        [z_norm[start - 1, lm_idx, 0], z_norm[end, lm_idx, 0]],
                    )
                    still_missing[start:end, lm_idx] = False
            else:
                t += 1

    return xy_out, z_out, still_missing


def fill_remaining_and_flag(xy_norm, z_norm, vis, still_missing):
    xy_out = xy_norm.copy()
    z_out = z_norm.copy()
    vis_out = vis.copy()
    xy_out[still_missing] = 0.0
    z_out[still_missing] = 0.0
    vis_out[still_missing] = 0.0
    return xy_out, z_out, vis_out


def compute_pose_insufficient(vis_out, still_missing, min_visible_ratio):
    """LUU Y (2026-09): ban dau ham nay bat buoc MOI frame phai co it nhat 1 co tay
    visibility >= nguong, song song voi 4 landmark loi (vai/hong). Kiem chung thuc te
    (xem diagnose_visibility.py) cho thay vai/hong gan nhu luon ro (0% frame thap tren
    ca smash va short_service), trong khi co tay - nhat la tay cam vot - thuong xuyen
    bi CHINH THAN NGUOI tu che khoi goc quay baseline dung luc vung tay/dua tay ra truoc.
    Day la dac diem vat ly cua goc quay ShuttleSet, khong phai loi crop/model, va khong
    the sua bang crop toan hoc. Vi vay gate chinh thuc (OK/POSE_INSUFFICIENT) chi con
    dua vao 4 landmark loi; ty le co tay van duoc tinh rieng (wrist_ratio) de theo doi
    trong manifest nhung khong con lam dieu kien loai bo clip. Gia tri x,y,z,visibility
    cua co tay van duoc giu nguyen trong feature dau ra (khong bi ep ve 0) khi con lo
    diem theo vi trong LEFT_WRIST/RIGHT_WRIST, cho phep LSTM/MLP tu hoc muc do tin cay
    tu chinh gia tri visibility thay vi bi loai truoc."""
    T = vis_out.shape[0]
    key_landmarks = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    core_ok = ~still_missing[:, key_landmarks].any(axis=1)
    wrist_ok = (~still_missing[:, LEFT_WRIST]) | (~still_missing[:, RIGHT_WRIST])

    frame_ok = core_ok  # gate chinh thuc: chi dua vao landmark loi (xem ghi chu tren)
    ratio = frame_ok.mean() if T > 0 else 0.0
    wrist_ratio = wrist_ok.mean() if T > 0 else 0.0  # thong tin tham khao, khong gate
    status = "OK" if ratio >= min_visible_ratio else "POSE_INSUFFICIENT"
    return status, float(ratio), float(wrist_ratio)


def resample_to_fixed_T(xy_norm, z_norm, vis, target_T=TARGET_T):
    T_orig = xy_norm.shape[0]
    if T_orig == target_T:
        return xy_norm, z_norm, vis
    if T_orig == 1:
        return (np.repeat(xy_norm, target_T, axis=0),
                np.repeat(z_norm, target_T, axis=0),
                np.repeat(vis, target_T, axis=0))

    old_idx = np.linspace(0, 1, T_orig)
    new_idx = np.linspace(0, 1, target_T)
    L = xy_norm.shape[1]
    xy_rs = np.zeros((target_T, L, 2), dtype=np.float32)
    z_rs = np.zeros((target_T, L, 1), dtype=np.float32)
    vis_rs = np.zeros((target_T, L, 1), dtype=np.float32)

    for lm_idx in range(L):
        for coord in range(2):
            xy_rs[:, lm_idx, coord] = np.interp(new_idx, old_idx, xy_norm[:, lm_idx, coord])
        z_rs[:, lm_idx, 0] = np.interp(new_idx, old_idx, z_norm[:, lm_idx, 0])
        vis_rs[:, lm_idx, 0] = np.interp(new_idx, old_idx, vis[:, lm_idx, 0])

    return xy_rs, z_rs, vis_rs


def process_clip(video_path, model_path, min_visible_ratio, min_confidence=0.5):
    raw = extract_raw_landmarks(video_path, model_path, min_confidence)
    if raw is None:
        return None, "EMPTY_VIDEO", 0.0, 0.0

    xy_norm, z_norm, vis, missing_mask = normalize_sequence(raw)
    xy_interp, z_interp, still_missing = interpolate_short_gaps(xy_norm, z_norm, vis, missing_mask)
    xy_filled, z_filled, vis_filled = fill_remaining_and_flag(xy_interp, z_interp, vis, still_missing)

    status, ratio, wrist_ratio = compute_pose_insufficient(vis_filled, still_missing, min_visible_ratio)
    xy_rs, z_rs, vis_rs = resample_to_fixed_T(xy_filled, z_filled, vis_filled, TARGET_T)

    feature = np.concatenate([xy_rs, z_rs, vis_rs], axis=2).reshape(TARGET_T, NUM_LANDMARKS * 4)
    return feature, status, ratio, wrist_ratio


def cmd_debug(args):
    import subprocess

    w, h, fps = probe_video(args.clip)

    # Ghi truc tiep frame (BGR, dung thu tu OpenCV) qua pipe vao ffmpeg,
    # KHONG dung cv2.VideoWriter - tren nhieu may Windows cac codec
    # (mp4v, XVID...) cua ban OpenCV cai qua pip khong hoat dong dung,
    # tao ra file .avi/.mp4 rong hoac hong du lieu ma khong bao loi ro rang.
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        args.out_video,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    landmarker = make_landmarker(args.model, mp_vision.RunningMode.VIDEO, args.min_confidence)
    frame_idx = 0
    detected_count = 0
    try:
        for frame in read_frames_via_ffmpeg(args.clip, w, h):
            frame = frame.copy()  # np.frombuffer tra ve mang read-only, can copy de ve len tren
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(frame_idx * 1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                detected_count += 1
                pose = result.pose_landmarks[0]
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose]
                for (a, b) in POSE_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for (px, py) in pts:
                    cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

            proc.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        landmarker.close()
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg thoat voi ma loi {proc.returncode}")

    print(f"Phat hien pose: {detected_count}/{frame_idx} frame ({100*detected_count/max(frame_idx,1):.0f}%) "
          f"o nguong min_confidence={args.min_confidence}")

    print(f"Da xuat video debug: {args.out_video}")
    print("Mo len kiem tra: khung xuong co bam dung 1 VDV dang danh cau xuyen suot clip khong.")


def cmd_batch(args):
    clip_files = sorted(glob.glob(os.path.join(args.clips_dir, "*", "*.mp4")))
    if not clip_files:
        print(f"Khong tim thay clip .mp4 nao trong {args.clips_dir}")
        return

    records = []
    for clip_path in clip_files:
        label = os.path.basename(os.path.dirname(clip_path))
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]

        feature, status, ratio, wrist_ratio = process_clip(clip_path, args.model, args.min_visible_ratio, args.min_confidence)

        if feature is not None:
            out_label_dir = os.path.join(args.out_dir, label)
            os.makedirs(out_label_dir, exist_ok=True)
            out_path = os.path.join(out_label_dir, clip_name + ".npy")
            np.save(out_path, feature)
        else:
            out_path = ""

        records.append({
            "clip": clip_path, "label": label, "status": status,
            "visible_ratio": round(ratio, 3), "wrist_ratio": round(wrist_ratio, 3),
            "feature_path": out_path,
        })
        print(f"[{status:16s}] ratio={ratio:.2f} wrist_ratio={wrist_ratio:.2f}  {clip_path}")

    manifest = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "features_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"\nTong so clip: {len(manifest)}")
    print(manifest["status"].value_counts())
    print(f"Manifest luu tai: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dlm = sub.add_parser("download-model", help="Tai file model .task can thiet (chi can lam 1 lan)")
    p_dlm.add_argument("--variant", choices=["lite", "full", "heavy"], default="full")
    p_dlm.set_defaults(func=cmd_download_model)

    p_debug = sub.add_parser("debug", help="Xuat 1 video co overlay khung xuong de kiem chung bang mat")
    p_debug.add_argument("--clip", required=True)
    p_debug.add_argument("--model", default="models/pose_landmarker_full.task")
    p_debug.add_argument("--out-video", default="debug_pose_overlay.mp4")
    p_debug.add_argument("--min-confidence", type=float, default=0.5,
                          help="Nguong tin cay phat hien pose (mac dinh 0.5). Ha xuong (vd 0.1-0.3) de test neu nghi ngo MediaPipe khong phat hien duoc nguoi do o qua xa/nho trong khung hinh.")
    p_debug.set_defaults(func=cmd_debug)

    p_batch = sub.add_parser("batch", help="Chay pose extraction hang loat cho toan bo clip")
    p_batch.add_argument("--clips-dir", default="data/clips")
    p_batch.add_argument("--model", default="models/pose_landmarker_full.task")
    p_batch.add_argument("--out-dir", default="data/pose_features")
    p_batch.add_argument("--min-visible-ratio", type=float, default=0.7)
    p_batch.add_argument("--min-confidence", type=float, default=0.5)
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()