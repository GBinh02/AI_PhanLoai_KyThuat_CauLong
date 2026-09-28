"""
diagnose_visibility.py

Chan doan chi tiet: in ra visibility (do tin cay) tung frame cua 6 landmark
quyet dinh status OK/POSE_INSUFFICIENT trong extract_pose_features.py
(vai trai/phai, hong trai/phai, co tay trai/phai), thay vi chi xem khung
xuong ve tren video debug (khung xuong duoc ve bat ke visibility cao hay thap,
nen khong the dung mat de biet frame nao dang bi tinh la "missing").

CACH DUNG (chay tu thu muc goc repo, hoac chinh sys.path ben duoi cho dung):
    python diagnose_visibility.py \
        --clip data/clips_cropped_v5/short_service/set1_rally19_ball1.0.mp4 \
        --model models/pose_landmarker_full.task
"""

import argparse
import os
import sys

# Them thu muc chua extract_pose_features.py vao sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ai", "pose_extraction"))

from extract_pose_features import (  # noqa: E402
    extract_raw_landmarks,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_WRIST, RIGHT_WRIST,
    VISIBILITY_THRESHOLD,
)

LANDMARK_NAMES = {
    LEFT_SHOULDER: "vai_trai", RIGHT_SHOULDER: "vai_phai",
    LEFT_HIP: "hong_trai", RIGHT_HIP: "hong_phai",
    LEFT_WRIST: "co_tay_trai", RIGHT_WRIST: "co_tay_phai",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--model", default="models/pose_landmarker_full.task")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    args = ap.parse_args()

    raw = extract_raw_landmarks(args.clip, args.model, args.min_confidence)
    if raw is None:
        print("Video rong hoac khong doc duoc frame nao.")
        return

    T = raw.shape[0]
    key_ids = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_WRIST, RIGHT_WRIST]

    print(f"Clip: {args.clip}  ({T} frame)")
    print(f"Nguong visibility de tinh la 'thay ro': {VISIBILITY_THRESHOLD}\n")

    header = "frame | " + " | ".join(f"{LANDMARK_NAMES[i]:>12s}" for i in key_ids) + " | core_ok wrist_ok frame_ok"
    print(header)
    print("-" * len(header))

    bad_landmark_count = {i: 0 for i in key_ids}
    for t in range(T):
        vis = raw[t, :, 3]
        row_vals = []
        for i in key_ids:
            v = vis[i]
            mark = "OK" if v >= VISIBILITY_THRESHOLD else "THAP"
            if v < VISIBILITY_THRESHOLD:
                bad_landmark_count[i] += 1
            row_vals.append(f"{v:5.2f}({mark})")
        core_ok = all(vis[i] >= VISIBILITY_THRESHOLD for i in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])
        wrist_ok = (vis[LEFT_WRIST] >= VISIBILITY_THRESHOLD) or (vis[RIGHT_WRIST] >= VISIBILITY_THRESHOLD)
        frame_ok = core_ok and wrist_ok
        print(f"{t:5d} | " + " | ".join(f"{v:>12s}" for v in row_vals) +
              f" | {str(core_ok):>7s} {str(wrist_ok):>8s} {str(frame_ok):>8s}")

    print("\n--- Tong ket: so frame co visibility THAP (< nguong), theo tung landmark ---")
    for i in key_ids:
        print(f"  {LANDMARK_NAMES[i]:>14s}: {bad_landmark_count[i]:3d} / {T} frame")


if __name__ == "__main__":
    main()
