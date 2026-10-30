"""API tang san (muc 5.1) + lop AI tuy chon (muc 5.2).

Chay:  uvicorn backend.app.main:app --reload   (tu thu muc goc repo)
"""
import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ai_service, config, media, security
from .db import get_db, init_db

LabelName = Literal["clear", "net_shot", "short_service", "smash"]
FRONTEND_DIST = config.REPO_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Phân loại kỹ thuật cầu lông", lifespan=lifespan)


# ---------------------------------------------------------------- schemas
class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=6, max_length=128)


class VideoPatch(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class LabelIn(BaseModel):
    time_sec: float = Field(ge=0)
    label: LabelName
    note: str = Field(default="", max_length=300)


class Box(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class AnalyzeIn(BaseModel):
    contact_sec: float = Field(ge=0)
    box: Box
    window: Literal["auto", "default", "service"] = "auto"


# ---------------------------------------------------------------- helpers
def _video_row(db, video_id: int, user_id: int):
    row = db.execute("SELECT * FROM videos WHERE id = ? AND user_id = ?", (video_id, user_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Không tìm thấy video")
    return row


def _video_out(row, counts=None):
    d = {k: row[k] for k in ("id", "title", "filename", "size_bytes", "duration", "fps",
                             "width", "height", "created_at")}
    if counts is not None:
        d.update(counts)
    return d


def _label_out(r):
    return {k: r[k] for k in ("id", "video_id", "time_sec", "label", "note", "created_at")}


def _pred_out(r):
    return {"id": r["id"], "video_id": r["video_id"], "contact_sec": r["contact_sec"],
            "box": {"x": r["box_x"], "y": r["box_y"], "w": r["box_w"], "h": r["box_h"]},
            "label": r["label"], "confidence": r["confidence"],
            "probs": json.loads(r["probs_json"]), "window_used": r["window_used"],
            "details": json.loads(r["details_json"]), "created_at": r["created_at"]}


# ---------------------------------------------------------------- auth
@app.post("/api/auth/register", status_code=201)
def register(c: Credentials, db=Depends(get_db)):
    if db.execute("SELECT 1 FROM users WHERE username = ?", (c.username,)).fetchone():
        raise HTTPException(409, "Tên đăng nhập đã tồn tại")
    cur = db.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)",
                     (c.username, security.hash_password(c.password)))
    return {"token": security.create_token(cur.lastrowid, c.username), "username": c.username}


@app.post("/api/auth/login")
def login(c: Credentials, db=Depends(get_db)):
    row = db.execute("SELECT * FROM users WHERE username = ?", (c.username,)).fetchone()
    if row is None or not security.verify_password(c.password, row["password_hash"]):
        raise HTTPException(401, "Sai tên đăng nhập hoặc mật khẩu")
    return {"token": security.create_token(row["id"], row["username"]), "username": row["username"]}


@app.get("/api/auth/me")
def me(user=Depends(security.current_user)):
    return {"id": user["id"], "username": user["username"]}


# ---------------------------------------------------------------- videos (5.1)
@app.post("/api/videos", status_code=201)
async def upload_video(file: UploadFile = File(...), title: str = Form(default=""),
                       user=Depends(security.current_user), db=Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(415, f"Định dạng không hỗ trợ. Chấp nhận: {', '.join(sorted(config.ALLOWED_EXT))}")
    stored = f"{uuid.uuid4().hex}{ext}"
    dest = config.UPLOAD_DIR / stored
    limit, size = config.MAX_UPLOAD_MB * 1024 * 1024, 0
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Video vượt quá {config.MAX_UPLOAD_MB} MB")
                f.write(chunk)
        info = await run_in_threadpool(media.probe, dest)
        if info["duration"] is None:
            raise HTTPException(422, "Không đọc được video (file hỏng hoặc sai định dạng)")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    name = re.sub(r"\s+", " ", (title or Path(file.filename).stem)).strip()[:120] or "Video"
    cur = db.execute(
        "INSERT INTO videos(user_id, title, filename, stored_name, size_bytes, duration, fps, width, height)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (user["id"], name, file.filename, stored, size,
         info["duration"], info["fps"], info["width"], info["height"]))
    return _video_out(_video_row(db, cur.lastrowid, user["id"]))


@app.get("/api/videos")
def list_videos(user=Depends(security.current_user), db=Depends(get_db)):
    rows = db.execute(
        "SELECT v.*, (SELECT COUNT(*) FROM labels l WHERE l.video_id = v.id) AS n_labels,"
        " (SELECT COUNT(*) FROM predictions p WHERE p.video_id = v.id) AS n_predictions"
        " FROM videos v WHERE v.user_id = ? ORDER BY v.created_at DESC, v.id DESC", (user["id"],)).fetchall()
    return [_video_out(r, {"n_labels": r["n_labels"], "n_predictions": r["n_predictions"]}) for r in rows]


@app.get("/api/videos/{video_id}")
def get_video(video_id: int, user=Depends(security.current_user), db=Depends(get_db)):
    return _video_out(_video_row(db, video_id, user["id"]))


@app.patch("/api/videos/{video_id}")
def rename_video(video_id: int, body: VideoPatch, user=Depends(security.current_user), db=Depends(get_db)):
    _video_row(db, video_id, user["id"])
    db.execute("UPDATE videos SET title = ? WHERE id = ?", (body.title.strip(), video_id))
    return _video_out(_video_row(db, video_id, user["id"]))


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: int, user=Depends(security.current_user), db=Depends(get_db)):
    row = _video_row(db, video_id, user["id"])
    for p in db.execute("SELECT details_json FROM predictions WHERE video_id = ?", (video_id,)).fetchall():
        ai_service.delete_clips(json.loads(p["details_json"]))
    db.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    (config.UPLOAD_DIR / row["stored_name"]).unlink(missing_ok=True)


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: int, user=Depends(security.current_user), db=Depends(get_db)):
    """Phat video; FileResponse ho tro HTTP Range nen tua/seek duoc. Auth qua ?token= cho the <video>."""
    row = _video_row(db, video_id, user["id"])
    path = config.UPLOAD_DIR / row["stored_name"]
    if not path.is_file():
        raise HTTPException(410, "File video không còn trên máy chủ")
    return FileResponse(path, filename=row["filename"], content_disposition_type="inline")


# ---------------------------------------------------------------- manual labels (5.1)
@app.get("/api/videos/{video_id}/labels")
def list_labels(video_id: int, user=Depends(security.current_user), db=Depends(get_db)):
    _video_row(db, video_id, user["id"])
    rows = db.execute("SELECT * FROM labels WHERE video_id = ? ORDER BY time_sec, id", (video_id,)).fetchall()
    return [_label_out(r) for r in rows]


@app.post("/api/videos/{video_id}/labels", status_code=201)
def add_label(video_id: int, body: LabelIn, user=Depends(security.current_user), db=Depends(get_db)):
    v = _video_row(db, video_id, user["id"])
    if v["duration"] and body.time_sec > v["duration"] + 0.5:
        raise HTTPException(422, "Thời điểm nằm ngoài video")
    cur = db.execute("INSERT INTO labels(video_id, time_sec, label, note) VALUES (?,?,?,?)",
                     (video_id, body.time_sec, body.label, body.note.strip()))
    return _label_out(db.execute("SELECT * FROM labels WHERE id = ?", (cur.lastrowid,)).fetchone())


@app.put("/api/videos/{video_id}/labels/{label_id}")
def update_label(video_id: int, label_id: int, body: LabelIn,
                 user=Depends(security.current_user), db=Depends(get_db)):
    _video_row(db, video_id, user["id"])
    cur = db.execute("UPDATE labels SET time_sec = ?, label = ?, note = ? WHERE id = ? AND video_id = ?",
                     (body.time_sec, body.label, body.note.strip(), label_id, video_id))
    if cur.rowcount == 0:
        raise HTTPException(404, "Không tìm thấy nhãn")
    return _label_out(db.execute("SELECT * FROM labels WHERE id = ?", (label_id,)).fetchone())


@app.delete("/api/videos/{video_id}/labels/{label_id}", status_code=204)
def delete_label(video_id: int, label_id: int, user=Depends(security.current_user), db=Depends(get_db)):
    _video_row(db, video_id, user["id"])
    cur = db.execute("DELETE FROM labels WHERE id = ? AND video_id = ?", (label_id, video_id))
    if cur.rowcount == 0:
        raise HTTPException(404, "Không tìm thấy nhãn")


# ---------------------------------------------------------------- AI layer (5.2)
@app.get("/api/ai/status")
def ai_status(_=Depends(security.current_user)):
    return ai_service.status()


@app.get("/api/videos/{video_id}/predictions")
def list_predictions(video_id: int, user=Depends(security.current_user), db=Depends(get_db)):
    _video_row(db, video_id, user["id"])
    rows = db.execute("SELECT * FROM predictions WHERE video_id = ? ORDER BY contact_sec, id", (video_id,)).fetchall()
    return [_pred_out(r) for r in rows]


@app.post("/api/videos/{video_id}/analyze", status_code=201)
def analyze(video_id: int, body: AnalyzeIn, user=Depends(security.current_user), db=Depends(get_db)):
    """Chay toan bo luong AI cho mot cu danh. Dinh nghia `def` (khong async) de FastAPI
    chay trong threadpool, khong chan event loop trong luc suy luan."""
    v = _video_row(db, video_id, user["id"])
    if not (v["width"] and v["height"] and v["duration"]):
        raise HTTPException(422, "Thiếu thông tin kích thước/độ dài video")
    try:
        res = ai_service.analyze(
            config.UPLOAD_DIR / v["stored_name"], duration=v["duration"], width=v["width"],
            height=v["height"], contact_sec=body.contact_sec, box=body.box.model_dump(),
            window=body.window)
    except ai_service.AIUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))
    except ai_service.AIInputError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    b = body.box
    cur = db.execute(
        "INSERT INTO predictions(video_id, contact_sec, box_x, box_y, box_w, box_h, label, confidence,"
        " probs_json, window_used, details_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (video_id, body.contact_sec, b.x, b.y, b.w, b.h, res["label"], res["confidence"],
         json.dumps(res["probs"]), res["window_used"], json.dumps(res["details"])))
    return _pred_out(db.execute("SELECT * FROM predictions WHERE id = ?", (cur.lastrowid,)).fetchone())


@app.delete("/api/videos/{video_id}/predictions/{pred_id}", status_code=204)
def delete_prediction(video_id: int, pred_id: int, user=Depends(security.current_user), db=Depends(get_db)):
    _video_row(db, video_id, user["id"])
    row = db.execute("SELECT details_json FROM predictions WHERE id = ? AND video_id = ?",
                     (pred_id, video_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Không tìm thấy kết quả")
    db.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    ai_service.delete_clips(json.loads(row["details_json"]))


@app.get("/api/videos/{video_id}/predictions/{pred_id}/clip/{window}")
def prediction_clip(video_id: int, pred_id: int, window: Literal["default", "service"],
                    user=Depends(security.current_user), db=Depends(get_db)):
    """Clip da cat + crop ma mo hinh thuc su 'nhin thay' (de kiem tra hop crop va thoi diem cham cau)."""
    _video_row(db, video_id, user["id"])
    row = db.execute("SELECT details_json FROM predictions WHERE id = ? AND video_id = ?",
                     (pred_id, video_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Không tìm thấy kết quả")
    for d in json.loads(row["details_json"]):
        if d["window"] == window and d.get("clip"):
            path = config.DEBUG_DIR / Path(d["clip"]).name
            if path.is_file():
                return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")
    raise HTTPException(404, "Không có clip cho cửa sổ này")


# ---------------------------------------------------------------- serve built frontend (production)
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = (FRONTEND_DIST / path).resolve()
        if path and f.is_file() and FRONTEND_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(FRONTEND_DIST / "index.html")
