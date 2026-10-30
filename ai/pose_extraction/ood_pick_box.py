"""
ood_pick_box.py - Ve hop crop bang chuot cho cac clip OOD ma ood_crop.py khong tu crop duoc.

Moi clip hien len 1 anh "chong mo" (trung binh nhieu frame tu dau den cuoi clip), nen ban thay
duoc toan bo vung nguoi danh di chuyen trong clip. Keo chuot ve 1 hop bao quanh NGUOI DANH CAU
(ca vot va chan, chua them chut le), roi:
    ENTER / SPACE : luu hop, sang clip tiep theo
    C             : bo qua clip nay (khong luu; clip se khong duoc danh gia)
    Ctrl+C o terminal: dung giua chung - cac hop da ve van duoc luu

Ket qua ghi vao data/ood/crop_overrides.csv (cot clip,x,y,w,h theo pixel cua clip goc).
Sau do chay lai:  python ai/pose_extraction/ood_crop.py crop
(dung cache, khong chay lai MediaPipe; cac clip co override se co trang thai OVERRIDE).

CACH DUNG (chay tu goc repo):
    # Tat ca clip loi trong crops.csv (NO_TRACK / SIDE_MISMATCH / LOW_MOTION)
    python ai/pose_extraction/ood_pick_box.py

    # Ve lai cho mot so clip cu the (vd clip OK nhung chon sai nguoi)
    python ai/pose_extraction/ood_pick_box.py --clips video8_t18.90.mp4 video9_t7.00.mp4

    # Ve lai ca nhung clip da co override
    python ai/pose_extraction/ood_pick_box.py --redo
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DEFAULT_WIN = (0.98, 1.23)
WINDOWS = {"short_service": (0.3, 0.8)}
GOOD = {"OK", "OVERRIDE"}


def read_ghost_frame(clip, label, n_samples=7):
    """Tra ve (anh chong mo, frame luc cham cau, W, H)."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc {clip}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pre, _ = WINDOWS.get(label, DEFAULT_WIN)
    contact = min(max(n - 1, 0), int(round(pre * fps)))
    idxs = sorted(set(np.linspace(0, max(n - 1, 0), n_samples).astype(int)) | {contact})
    frames, contact_frame = [], None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(fr.astype(np.float32))
            if i == contact:
                contact_frame = fr
    cap.release()
    if not frames:
        raise RuntimeError(f"Khong doc duoc frame nao tu {clip}")
    ghost = np.mean(frames, axis=0).astype(np.uint8)
    if contact_frame is None:
        contact_frame = frames[len(frames) // 2].astype(np.uint8)
    # Tron: 60% frame luc cham cau (ro net) + 40% anh chong mo (thay vung di chuyen)
    view = cv2.addWeighted(contact_frame, 0.6, ghost, 0.4, 0)
    H, W = view.shape[:2]
    return view, W, H


def even_clamp(x, y, w, h, W, H):
    x, y = max(0, int(x)), max(0, int(y))
    w, h = min(int(w), W - x), min(int(h), H - y)
    return x, y, w - w % 2, h - h % 2          # ffmpeg/libx264 can kich thuoc chan


def load_overrides(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["clip", "x", "y", "w", "h"])


def save_override(path, ov, clip, box):
    ov = ov[ov["clip"] != clip]
    ov = pd.concat([ov, pd.DataFrame([{"clip": clip, "x": box[0], "y": box[1],
                                       "w": box[2], "h": box[3]}])], ignore_index=True)
    ov.to_csv(path, index=False, lineterminator="\n")
    return ov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", default="data/ood/crops.csv")
    ap.add_argument("--clips-dir", default="data/ood/clips")
    ap.add_argument("--overrides", default="data/ood/crop_overrides.csv")
    ap.add_argument("--clips", nargs="*", help="Chi ve cho cac clip nay (ten file)")
    ap.add_argument("--redo", action="store_true", help="Ve lai ca clip da co override")
    ap.add_argument("--max-height", type=int, default=900, help="Chieu cao cua so hien thi (pixel man hinh)")
    a = ap.parse_args()

    crops = pd.read_csv(a.crops)
    ov_path = Path(a.overrides)
    ov = load_overrides(ov_path)

    if a.clips:
        todo = crops[crops["clip"].isin(a.clips)]
        missing = set(a.clips) - set(todo["clip"])
        if missing:
            print("Khong co trong crops.csv:", sorted(missing))
    else:
        todo = crops[~crops["status"].isin(GOOD)]
        if not a.redo:
            todo = todo[~todo["clip"].isin(ov["clip"])]

    if todo.empty:
        print("Khong con clip nao can ve hop.")
        return
    print(f"Can ve {len(todo)} clip. ENTER/SPACE = luu, C = bo qua, Ctrl+C o terminal = dung.\n")

    saved = skipped = 0
    try:
        for k, (_, row) in enumerate(todo.iterrows(), 1):
            clip = Path(a.clips_dir) / row["label"] / row["clip"]
            if not clip.is_file():
                print(f"[BO QUA] khong thay {clip}")
                continue
            view, W, H = read_ghost_frame(clip, row["label"])
            scale = min(1.0, a.max_height / H)
            disp = cv2.resize(view, (int(W * scale), int(H * scale))) if scale < 1 else view.copy()
            hint = f"{k}/{len(todo)} {row['label']} | {row['side'].upper()} | {row['status']}"
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(disp, hint, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            win = f"Ve hop nguoi danh ({row['side']}) - {row['clip']}"
            x, y, w, h = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(win)

            if w == 0 or h == 0:
                skipped += 1
                print(f"[BO QUA ] {row['clip']}")
                continue
            box = even_clamp(x / scale, y / scale, w / scale, h / scale, W, H)
            if box[2] < 32 or box[3] < 32:
                skipped += 1
                print(f"[QUA NHO] {row['clip']} {box} - bo qua, ve lai hop lon hon")
                continue
            ov = save_override(ov_path, ov, row["clip"], box)
            saved += 1
            print(f"[DA LUU ] {row['clip']} -> x={box[0]} y={box[1]} w={box[2]} h={box[3]}")
    except KeyboardInterrupt:
        print("\nDung giua chung.")
    finally:
        cv2.destroyAllWindows()

    print(f"\nDa luu {saved}, bo qua {skipped}. File: {ov_path}")
    print("Tiep theo:  python ai/pose_extraction/ood_crop.py crop")


if __name__ == "__main__":
    main()
