"""SQLite don gian (module sqlite3 co san, khong can ORM)."""
import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,          -- ten file goc nguoi dung upload
    stored_name TEXT NOT NULL,       -- ten file tren dia
    size_bytes INTEGER NOT NULL,
    duration REAL, fps REAL, width INTEGER, height INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_videos_user ON videos(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    time_sec REAL NOT NULL,
    label TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_labels_video ON labels(video_id, time_sec);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    contact_sec REAL NOT NULL,
    box_x REAL NOT NULL, box_y REAL NOT NULL, box_w REAL NOT NULL, box_h REAL NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    probs_json TEXT NOT NULL,        -- {"clear":..,"net_shot":..,...} cua cua so duoc chon
    window_used TEXT NOT NULL,       -- "default" | "service"
    details_json TEXT NOT NULL,      -- ket qua tung cua so da chay
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_pred_video ON predictions(video_id, contact_sec);
"""


def connect():
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_db():
    """Dependency FastAPI: moi request 1 ket noi."""
    with session() as conn:
        yield conn
