"""
ood_crop.py - Crop clip OOD quanh dung nguoi dang danh cau (thay cho player_location cua ShuttleSet).

Video tu quay la video doc 576x1024 (TikTok), nhieu nguoi trong khung (2 nguoi choi + khan gia),
co video bi rung/quay theo. Script lam 3 viec:
  1. Phat hien tat ca nguoi trong tung frame (MediaPipe PoseLandmarker, num_poses=4,
     chay tren frame day du va tren nua tren khung hinh de bat nguoi o xa).
  2. Ghep thanh cac "quy dao" nguoi, chon quy dao co co tay chuyen dong nhieu
     (nguoi dang vung vot) va gan nhat voi player_side trong manifest
     (near = gan camera hon / chan thap hon trong khung, far = nguoi con lai).
  3. Crop clip theo hop bao quanh nguoi do (hop co dinh cho ca clip, giong luc huan luyen),
     ghi anh doi chieu de ban kiem tra bang mat.

CAC BUOC (chay tu goc repo):
  python ai/pose_extraction/ood_crop.py crop \
      --manifest data/ood/clips_manifest.csv --model models/pose_landmarker_full.task

  -> data/ood/crop_review/<nhan>_<ten_clip>.jpg   (3 frame dau/chạm/cuoi, khung xanh = hop crop)
  -> data/ood/crops.csv                           (trang thai tung clip: OK / NO_TRACK / SIDE_MISMATCH / LOW_MOTION / OVERRIDE)
  -> data/ood/clips_cropped/<nhan>/<ten_clip>.mp4 (chi cho clip OK hoac co override)

  Clip nao khong OK -> khong crop (tranh dung nham nguoi). Xem anh doi chieu, voi clip hong hoac
  chon sai nguoi, them 1 dong vao data/ood/crop_overrides.csv:
        clip,x,y,w,h
        video7_t26.70.mp4,180,420,200,330
  roi chay lai lenh tren (ket qua phat hien duoc cache, khong chay lai MediaPipe).

  Sau do chay buoc pose co san cua du an:
  python ai/pose_extraction/extract_pose_features.py batch \
      --clips-dir data/ood/clips_cropped --model models/pose_landmarker_full.task \
      --out-dir data/ood/pose_features --min-visible-ratio 0.7
"""
import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_pose_features import (  # noqa: E402
    mp, mp_tasks_python, mp_vision, probe_video, read_frames_via_ffmpeg,
)

DEFAULT_WIN = (0.98, 1.23)
WINDOWS = {"short_service": (0.3, 0.8)}
VIS_THR = 0.3


# ---------------------------------------------------------------- phat hien nguoi
def make_multi_landmarker(model_path, min_conf, num_poses):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def tile_specs(w, h):
    """Moi tile = (x0, y0, rong, cao) tren frame goc. Tile 2 (60% tren khung) giup bat nguoi o xa."""
    return [(0, 0, w, h), (0, 0, w, int(h * 0.6))]


def person_from_landmarks(lm, x0, y0, tw, th):
    pts = np.array([[l.x * tw + x0, l.y * th + y0, l.visibility] for l in lm], dtype=np.float32)
    ok = pts[:, 2] >= VIS_THR
    if ok.sum() < 8:
        return None
    xy = pts[ok, :2]
    bx0, by0 = xy.min(axis=0)
    bx1, by1 = xy.max(axis=0)
    if by1 - by0 < 20:
        return None
    c = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
    return {"box": (float(bx0), float(by0), float(bx1), float(by1)),
            "wrists_rel": np.concatenate([pts[[15, 16], :2] - c, pts[[15, 16], 2:3]], axis=1)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe(persons):
    keep = []
    for p in persons:
        if all(iou(p["box"], q["box"]) < 0.4 for q in keep):
            keep.append(p)
    return keep


def detect_clip(video, model_path, min_conf, num_poses):
    w, h, fps = probe_video(video)
    tiles = tile_specs(w, h)
    lms = [make_multi_landmarker(model_path, min_conf, num_poses) for _ in tiles]
    dets = []
    try:
        for idx, frame in enumerate(read_frames_via_ffmpeg(video, w, h)):
            ts = int(round(idx * 1000.0 / fps))
            persons = []
            for (x0, y0, tw, th), lmk in zip(tiles, lms):
                sub = np.ascontiguousarray(cv2.cvtColor(frame[y0:y0 + th, x0:x0 + tw], cv2.COLOR_BGR2RGB))
                res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=sub), ts)
                for pose in res.pose_landmarks:
                    p = person_from_landmarks(pose, x0, y0, tw, th)
                    if p is not None:
                        persons.append(p)
            dets.append(dedupe(persons))
    finally:
        for lmk in lms:
            lmk.close()
    return {"dets": dets, "w": w, "h": h, "fps": fps}


# ---------------------------------------------------------------- quy dao & chon nguoi
def box_h(p):
    return p["box"][3] - p["box"][1]


def center(p):
    b = p["box"]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def build_tracks(dets, max_gap=3, dist_frac=0.6):
    tracks = []  # moi quy dao: {chi_so_frame: person}
    for t, persons in enumerate(dets):
        used = set()
        for p in sorted(persons, key=lambda q: -box_h(q)):
            best, bd = None, 1e9
            for k, tr in enumerate(tracks):
                if k in used:
                    continue
                last = max(tr)
                if t - last > max_gap + 1:
                    continue
                d = float(np.linalg.norm(center(p) - center(tr[last])))
                if d < dist_frac * max(box_h(p), box_h(tr[last])) and d < bd:
                    best, bd = k, d
            if best is None:
                tracks.append({t: p})
                used.add(len(tracks) - 1)
            else:
                tracks[best][t] = p
                used.add(best)
    return tracks


def track_stats(tr):
    ts = sorted(tr)
    H = float(np.median([box_h(tr[t]) for t in ts]))
    foot = float(np.median([tr[t]["box"][3] for t in ts]))
    motion = 0.0
    for a, b in zip(ts[:-1], ts[1:]):
        if b - a != 1:
            continue
        for j in range(2):
            wa, wb = tr[a]["wrists_rel"][j], tr[b]["wrists_rel"][j]
            if wa[2] >= VIS_THR and wb[2] >= VIS_THR:
                motion += float(np.linalg.norm(wb[:2] - wa[:2]))
    return {"n": len(ts), "H": H, "foot": foot, "motion": motion / max(H, 1.0)}


def pick_target(tracks, side, n_frames, contact_idx, frame_h, min_frac, min_motion):
    cands = []
    for tr in tracks:
        if not any(abs(t - contact_idx) <= 6 for t in tr):
            continue
        if len(tr) < min_frac * n_frames:
            continue
        cands.append((tr, track_stats(tr)))
    if not cands:
        return None, "NO_TRACK", 0
    cands.sort(key=lambda c: -c[1]["H"] * (0.3 + c[1]["motion"]))
    top = cands[:2]
    if len(top) == 1:
        tr, s = top[0]
        is_near = s["foot"] > 0.6 * frame_h
        if (side == "near") != is_near:
            return None, "SIDE_MISMATCH", len(cands)
        chosen = (tr, s)
    else:
        a, b = top
        near, far = (a, b) if a[1]["foot"] > b[1]["foot"] else (b, a)
        chosen = near if side == "near" else far
    if chosen[1]["motion"] < min_motion:
        return None, "LOW_MOTION", len(cands)
    return chosen[0], "OK", len(cands)


def union_box(tr, W, H, pad_x=0.3, pad_top=0.3, pad_bot=0.1):
    boxes = np.array([tr[t]["box"] for t in tr])
    h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    x0, y0 = boxes[:, 0].min() - pad_x * h, boxes[:, 1].min() - pad_top * h
    x1, y1 = boxes[:, 2].max() + pad_x * h, boxes[:, 3].max() + pad_bot * h
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    w_, h_ = int(x1 - x0), int(y1 - y0)
    w_ -= w_ % 2
    h_ -= h_ % 2
    return int(x0), int(y0), w_, h_


# ---------------------------------------------------------------- anh doi chieu & crop
def review_image(video, W, H, idxs, tracks, chosen, box, status, title, out_path):
    wanted = set(idxs)
    frames = {}
    for i, fr in enumerate(read_frames_via_ffmpeg(video, W, H)):
        if i in wanted:
            frames[i] = fr.copy()
        if i >= max(idxs):
            break
    tiles = []
    for i in idxs:
        fr = frames.get(i)
        if fr is None:
            fr = np.zeros((H, W, 3), np.uint8)
        for tr in tracks:
            if i in tr:
                b = list(map(int, tr[i]["box"]))
                is_chosen = chosen is not None and tr is chosen
                if not is_chosen:
                    cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (160, 160, 160), 2)
                else:
                    cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)
        if box is not None:
            x, y, w_, h_ = box
            cv2.rectangle(fr, (x, y), (x + w_, y + h_), (0, 255, 0), 3)
        cv2.putText(fr, f"frame {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        scale = 480.0 / H
        tiles.append(cv2.resize(fr, (int(W * scale), 480)))
    grid = np.hstack(tiles)
    banner = np.zeros((40, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, f"{title} | {status}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), np.vstack([banner, grid]))


def crop_with_ffmpeg(src, dst, box):
    x, y, w_, h_ = box
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", f"crop={w_}:{h_}:{x}:{y}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", str(dst)], check=True)


def norm_path(p):
    return Path(str(p).replace("\\", "/"))


def cmd_crop(a):
    man = pd.read_csv(a.manifest)
    root = Path(a.manifest).parent
    cache_dir = root / "crop_cache"
    review_dir = root / "crop_review"
    cache_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    ov_path = Path(a.overrides) if a.overrides else root / "crop_overrides.csv"
    if ov_path.exists():
        ov = pd.read_csv(ov_path)
        overrides = {r["clip"]: (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for _, r in ov.iterrows()}

    records = []
    for i, row in man.iterrows():
        clip = norm_path(row["clip"])
        if not clip.is_file():
            print(f"[BO QUA] khong thay {clip}")
            continue
        name = clip.name
        label, side = row["label"], row["player_side"]
        cache = cache_dir / (clip.stem + ".pkl")
        if cache.exists():
            det = pickle.loads(cache.read_bytes())
        else:
            det = detect_clip(str(clip), a.model, a.min_conf, a.num_poses)
            cache.write_bytes(pickle.dumps(det))
        dets, W, H, fps = det["dets"], det["w"], det["h"], det["fps"]
        n = len(dets)
        pre, post = WINDOWS.get(label, DEFAULT_WIN)
        contact_idx = min(n - 1, int(round(pre * fps)))
        tracks = build_tracks(dets)

        chosen, status, n_c = pick_target(tracks, side, n, contact_idx, H, a.min_track_frac, a.min_motion)
        box = None
        if name in overrides:
            box, status = overrides[name], "OVERRIDE"
        elif chosen is not None:
            box = union_box(chosen, W, H)
            if box[2] < 32 or box[3] < 32:
                box, status = None, "NO_TRACK"

        review_image(str(clip), W, H, [0, contact_idx, n - 1], tracks, chosen, box, status,
                     f"{label}/{name} side={side}", review_dir / f"{label}_{clip.stem}.jpg")

        out = ""
        if box is not None and status in ("OK", "OVERRIDE"):
            out = Path(a.out_dir) / label / name
            crop_with_ffmpeg(clip, out, box)
        records.append({"clip": name, "label": label, "side": side, "person": row.get("person", ""),
                        "status": status, "n_candidates": n_c,
                        "x": box[0] if box else "", "y": box[1] if box else "",
                        "w": box[2] if box else "", "h": box[3] if box else "", "cropped": str(out)})
        print(f"[{status:13s}] {label:13s} {side:4s} {name}")

    rec = pd.DataFrame(records)
    rec.to_csv(root / "crops.csv", index=False, lineterminator="\n")
    print("\n=== Tong ket trang thai ===")
    print(rec.groupby(["side", "status"]).size())
    bad = rec[~rec["status"].isin(["OK", "OVERRIDE"])]
    if len(bad):
        print(f"\n{len(bad)} clip can xem anh doi chieu trong {review_dir} va them override:")
        print(bad[["clip", "label", "side", "status"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("crop")
    p.add_argument("--manifest", default="data/ood/clips_manifest.csv")
    p.add_argument("--model", default="models/pose_landmarker_full.task")
    p.add_argument("--out-dir", default="data/ood/clips_cropped")
    p.add_argument("--overrides", default=None, help="CSV clip,x,y,w,h (mac dinh data/ood/crop_overrides.csv)")
    p.add_argument("--min-conf", type=float, default=0.3)
    p.add_argument("--num-poses", type=int, default=4)
    p.add_argument("--min-track-frac", type=float, default=0.3,
                   help="Quy dao phai co mat toi thieu ty le nay so frame cua clip")
    p.add_argument("--min-motion", type=float, default=0.8,
                   help="Nguong chuyen dong co tay (don vi chieu cao nguoi) de coi la nguoi dang danh")
    p.set_defaults(fn=cmd_crop)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
