"""Cau hinh backend, doc tu bien moi truong (co gia tri mac dinh de chay local)."""
import os
import secrets
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", REPO_ROOT / "backend" / "storage"))
UPLOAD_DIR = DATA_DIR / "uploads"
TMP_DIR = DATA_DIR / "tmp"
DB_PATH = Path(os.environ.get("APP_DB_PATH", DATA_DIR / "app.db"))

# Khoa ky JWT. Production PHAI dat APP_SECRET_KEY; neu khong se sinh ngau nhien
# moi lan khoi dong (token cu mat hieu luc khi restart).
SECRET_KEY = os.environ.get("APP_SECRET_KEY") or secrets.token_urlsafe(32)
TOKEN_TTL_HOURS = int(os.environ.get("APP_TOKEN_TTL_HOURS", "24"))

MAX_UPLOAD_MB = int(os.environ.get("APP_MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

# Lop AI (muc 5.2): co the tat hoan toan bang AI_ENABLED=0 -> tang san van chay.
AI_ENABLED = os.environ.get("AI_ENABLED", "1") not in ("0", "false", "False", "")
AI_CHECKPOINT = Path(os.environ.get(
    "AI_CHECKPOINT", REPO_ROOT / "ai" / "models" / "checkpoints" / "bilstm_motion_seed5.keras"))
AI_FEATURE_MODE = os.environ.get("AI_FEATURE_MODE", "motion")
POSE_MODEL = Path(os.environ.get("POSE_MODEL", REPO_ROOT / "models" / "pose_landmarker_full.task"))

CLASS_NAMES = ["clear", "net_shot", "short_service", "smash"]  # thu tu nhan luc huan luyen
CLASS_LABELS_VI = {
    "clear": "Phông cầu (clear)",
    "net_shot": "Bỏ nhỏ lưới (net shot)",
    "short_service": "Giao cầu ngắn (short service)",
    "smash": "Đập cầu (smash)",
}
# Cua so cat quanh thoi diem cham cau (giay): (truoc, sau). Giong luc huan luyen.
DEFAULT_WINDOW = (0.98, 1.23)
SERVICE_WINDOW = (0.3, 0.8)

# Crop gui vao MediaPipe: ty le rong/cao = BOX_ASPECT cua crop_player_roi.py luc huan luyen;
# cao = hop nguoi dung ve * margin (luc huan luyen hop crop ~1.2x chieu cao nguoi).
CROP_ASPECT = 0.65
CROP_HEIGHT_MARGIN = 1.25

DEBUG_DIR =DATA_DIR / "ai_clips"  # clip da cat + crop cua tung lan phan tich, de xem lai

for d in (UPLOAD_DIR, TMP_DIR, DEBUG_DIR):
    d.mkdir(parents=True, exist_ok=True)
