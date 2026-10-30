"""Lop AI (muc 5.2) - bat/tat duoc, nap lazy de tang san chay doc lap.

Luong suy luan cho MOT cu danh (bo phan loai nhan 1 clip quanh thoi diem cham cau,
da crop theo mot nguoi - khong nhan ca video):
    contact_sec + hop crop (nguoi dung ve)
      -> ffmpeg cat cua so thoi gian + crop
      -> extract_pose_features.process_clip  (MediaPipe Pose, chuan hoa, resample 32 frame)
      -> motion.build_features(..., "motion")  (231 chieu)
      -> BiLSTM(64) -> xac suat 4 lop

Luc suy luan chua biet nhan nen chua biet nen dung cua so nao: short_service dung
cua so ngan (0.3 s truoc + 0.8 s sau), 3 lop con lai dung (0.98 s + 1.23 s). Ta chay
CA HAI cua so (neu nam trong video) va lay ket qua co do tin cay cao hon; ket qua cua
tung cua so van duoc tra ve de hien thi minh bach.
"""
import importlib.util
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid
from pathlib import Path

from . import config

_lock = threading.Lock()
_model = None


class AIUnavailable(Exception):
    """Mo-dun AI bi tat hoac thieu thu vien/checkpoint (HTTP 503)."""


class AIInputError(Exception):
    """Dau vao khong dung duoc (cua so ngoai video, khong do duoc tu the...) (HTTP 422)."""


def status() -> dict:
    reasons = []
    if not config.AI_ENABLED:
        reasons.append("Mô-đun AI đang bị tắt trên máy chủ (AI_ENABLED=0).")
    else:
        for mod, pkg in (("tensorflow", "tensorflow"), ("mediapipe", "mediapipe"),
                         ("cv2", "opencv-python"), ("numpy", "numpy")):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"Thiếu thư viện {pkg} (pip install -r backend/requirements-ai.txt).")
        if not config.AI_CHECKPOINT.is_file():
            reasons.append(f"Không thấy checkpoint {config.AI_CHECKPOINT.name}.")
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            reasons.append("Thiếu ffmpeg/ffprobe trong PATH.")
    return {
        "enabled": config.AI_ENABLED,
        "available": not reasons,
        "reasons": reasons,
        "classes": config.CLASS_NAMES,
        "class_labels": config.CLASS_LABELS_VI,
        "model": "BiLSTM(64) + motion features",
    }


def _require_available():
    st = status()
    if not st["available"]:
        raise AIUnavailable(" ".join(st["reasons"]) or "AI không khả dụng")


def _ensure_repo_on_path():
    root = str(config.REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _ensure_pose_model() -> str:
    path = config.POSE_MODEL
    if not path.is_file():
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_full/float16/1/pose_landmarker_full.task")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # noqa: BLE001
            path.unlink(missing_ok=True)
            raise AIUnavailable(
                f"Chưa có {path.name} và không tải được tự động ({e}). "
                f"Chạy: python ai/pose_extraction/extract_pose_features.py download-model --variant full") from e
    return str(path)


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from tensorflow import keras  # import cham -> chi khi can
            _model = keras.models.load_model(config.AI_CHECKPOINT, compile=False)
        return _model


def _cut_and_crop(video: Path, start: float, duration: float, box_px, out: Path):
    x, y, w, h = box_px
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video),
           "-t", f"{duration:.3f}", "-vf", f"crop={w}:{h}:{x}:{y}",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AIInputError(f"ffmpeg không cắt được clip: {r.stderr.strip()[:200]}")


def delete_clips(details: list) -> None:
    for d in details:
        if d.get("clip"):
            (config.DEBUG_DIR / Path(d["clip"]).name).unlink(missing_ok=True)


def _box_to_pixels(box: dict, width: int, height: int):
    """Hop chuan hoa (0..1) -> pixel, ep chan, kich thuoc chan (libx264 yuv420p)."""
    bx = max(0.0, min(1.0, box["x"])) * width
    by = max(0.0, min(1.0, box["y"])) * height
    bw = max(0.0, min(1.0 - box["x"], box["w"])) * width
    bh = max(0.0, min(1.0 - box["y"], box["h"])) * height
    if bw < 24 or bh < 32:
        raise AIInputError("Hộp quanh người đánh quá nhỏ.")

    # MediaPipe tra x,y theo ty le 0..1 cua CHIEU RONG/CAO cua crop, roi extract_pose_features chia
    # cho do dai than trong he toa do chuan hoa do. Vi vay TY LE (rong/cao) cua crop quyet dinh hinh
    # dang khung xuong dua vao mo hinh; luc huan luyen ty le nay co dinh = BOX_ASPECT (0.65).
    # => coi hop nguoi dung ve la hop bao nguoi, dung lai crop dung ty le huan luyen quanh tam hop.
    ch = max(bh * config.CROP_HEIGHT_MARGIN, 64.0)
    cw = ch * config.CROP_ASPECT
    if bw * 1.1 > cw:  # nguoi dang dang tay rong hon crop -> tang chieu cao de giu nguyen ty le
        cw = bw * 1.1
        ch = cw / config.CROP_ASPECT
    if ch > height or cw > width:  # khong vuot khung hinh; thu nho nhung giu ty le
        k = min(height / ch, width / cw)
        ch, cw = ch * k, cw * k
    cx, cy = bx + bw / 2, by + bh / 2
    x = min(max(cx - cw / 2, 0), width - cw)  # dich vao trong khung thay vi cat bot (giu ty le)
    y = min(max(cy - ch / 2, 0), height - ch)
    w, h = int(cw) // 2 * 2, int(ch) // 2 * 2
    return int(x) // 2 * 2, int(y) // 2 * 2, w, h


def analyze(video_path: Path, *, duration: float, width: int, height: int,
            contact_sec: float, box: dict, window: str = "auto") -> dict:
    """Tra ve dict: label, confidence, probs, window_used, details (tung cua so).

    window: "auto" chay ca hai cua so roi lay do tin cay cao hon; "default" / "service" ep dung 1 cua so
    (khi nguoi dung da biet day la cu giao cau). Clip da cat+crop duoc giu lai (DEBUG_DIR) de xem lai."""
    _require_available()
    if not (0 <= contact_sec <= duration):
        raise AIInputError("Thời điểm chạm cầu nằm ngoài video.")
    box_px = _box_to_pixels(box, width, height)

    _ensure_repo_on_path()
    pose_model = _ensure_pose_model()
    import numpy as np
    from ai.features.motion import build_features
    from ai.pose_extraction.extract_pose_features import process_clip

    model = _get_model()
    details, best = [], None
    windows = {"default": config.DEFAULT_WINDOW, "service": config.SERVICE_WINDOW}
    if window != "auto":
        if window not in windows:
            raise AIInputError(f"Cửa sổ không hợp lệ: {window}")
        windows = {window: windows[window]}
    for name, (pre, post) in windows.items():
        start, end = contact_sec - pre, contact_sec + post
        entry = {"window": name, "pre": pre, "post": post, "start": round(start, 3), "end": round(end, 3)}
        if start < 0 or end > duration + 0.05:
            entry["status"] = "OUT_OF_RANGE"
            details.append(entry)
            continue
        clip = config.DEBUG_DIR / f"{uuid.uuid4().hex}.mp4"
        _cut_and_crop(video_path, start, pre + post, box_px, clip)
        entry["clip"] = clip.name
        try:
            feature, st, ratio, _ = process_clip(str(clip), pose_model, min_visible_ratio=0.7)
        except Exception:
            clip.unlink(missing_ok=True)
            raise
        entry["status"], entry["visible_ratio"] = st, round(ratio, 3)
        if feature is None or st != "OK":
            details.append(entry)
            continue
        x = build_features(np.asarray(feature, dtype=np.float32)[None], config.AI_FEATURE_MODE)
        with _lock:
            probs = model.predict(x, verbose=0)[0]
        probs = [float(p) for p in probs]
        k = int(np.argmax(probs))
        entry.update(label=config.CLASS_NAMES[k], confidence=probs[k],
                     probs=dict(zip(config.CLASS_NAMES, probs)))
        details.append(entry)
        if best is None or probs[k] > best["confidence"]:
            best = entry

    if best is None:
        delete_clips(details)  # khong luu ket qua -> khong giu clip mo coi
        statuses = {d["status"] for d in details}
        if statuses == {"OUT_OF_RANGE"}:
            raise AIInputError("Cú đánh quá gần đầu/cuối video: không đủ khung hình để cắt cửa sổ phân tích.")
        raise AIInputError("Không dò được tư thế đủ rõ trong hộp đã chọn. Thử kéo hộp ôm sát người đánh hơn "
                           "hoặc chọn lại thời điểm chạm cầu.")
    return {"label": best["label"], "confidence": best["confidence"], "probs": best["probs"],
            "window_used": best["window"], "details": details}
