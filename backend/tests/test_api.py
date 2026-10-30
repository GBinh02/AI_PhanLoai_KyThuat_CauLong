"""Test tang san chay DOC LAP voi AI (AI_ENABLED=0) - dung yeu cau muc 5.1.

Chay tu thu muc goc repo:  pytest backend/tests
"""
import subprocess
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from backend.app import config
from backend.app.main import app

from .conftest import TMP as _tmp


@pytest.fixture(autouse=True)
def ai_off(monkeypatch):
    """Tat han mo-dun AI: tang san phai chay doc lap (muc 5.1)."""
    monkeypatch.setattr(config, "AI_ENABLED", False)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_video():
    p = Path(_tmp) / "sample.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def _auth(client, name="alice"):
    r = client.post("/api/auth/register", json={"username": name, "password": "secret123"})
    if r.status_code == 409:
        r = client.post("/api/auth/login", json={"username": name, "password": "secret123"})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_auth_flow(client):
    assert client.get("/api/videos").status_code == 401
    h = _auth(client)
    assert client.get("/api/auth/me", headers=h).json()["username"] == "alice"
    assert client.post("/api/auth/login", json={"username": "alice", "password": "wrongpass"}).status_code == 401
    assert client.post("/api/auth/register", json={"username": "ALICE", "password": "secret123"}).status_code == 409


def test_upload_stream_seek_and_history(client, sample_video):
    h = _auth(client)
    with sample_video.open("rb") as f:
        r = client.post("/api/videos", headers=h, files={"file": ("sample.mp4", f, "video/mp4")},
                        data={"title": "Buổi tập 1"})
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["title"] == "Buổi tập 1" and v["width"] == 320 and v["height"] == 240
    assert v["fps"] == pytest.approx(25) and v["duration"] == pytest.approx(3, abs=0.2)

    # phat video + Range (tua) -- the <video> xac thuc qua ?token=
    token = h["Authorization"].split()[1]
    full = client.get(f"/api/videos/{v['id']}/stream", params={"token": token})
    assert full.status_code == 200
    part = client.get(f"/api/videos/{v['id']}/stream", params={"token": token}, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100

    assert [x["id"] for x in client.get("/api/videos", headers=h).json()] == [v["id"]]
    assert client.patch(f"/api/videos/{v['id']}", headers=h, json={"title": "Đổi tên"}).json()["title"] == "Đổi tên"


def test_videos_are_private_per_user(client, sample_video):
    ha, hb = _auth(client, "alice"), _auth(client, "bob")
    vid = client.get("/api/videos", headers=ha).json()[0]["id"]
    assert client.get(f"/api/videos/{vid}", headers=hb).status_code == 404
    assert client.get("/api/videos", headers=hb).json() == []


def test_upload_rejects_bad_files(client):
    h = _auth(client)
    assert client.post("/api/videos", headers=h, files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415
    assert client.post("/api/videos", headers=h, files={"file": ("a.mp4", b"not a video", "video/mp4")}).status_code == 422
    assert not any(p.suffix == ".mp4" and p.stat().st_size == 11 for p in (Path(_tmp) / "uploads").iterdir())


def test_manual_labels_crud(client):
    h = _auth(client)
    vid = client.get("/api/videos", headers=h).json()[0]["id"]
    base = f"/api/videos/{vid}/labels"
    a = client.post(base, headers=h, json={"time_sec": 2.0, "label": "smash", "note": "ok"}).json()
    client.post(base, headers=h, json={"time_sec": 0.5, "label": "clear"})
    assert [x["time_sec"] for x in client.get(base, headers=h).json()] == [0.5, 2.0]  # sap theo thoi gian
    assert client.post(base, headers=h, json={"time_sec": 1, "label": "long_service"}).status_code == 422
    assert client.post(base, headers=h, json={"time_sec": 99, "label": "smash"}).status_code == 422
    upd = client.put(f"{base}/{a['id']}", headers=h, json={"time_sec": 2.2, "label": "net_shot"}).json()
    assert upd["label"] == "net_shot"
    assert client.delete(f"{base}/{a['id']}", headers=h).status_code == 204
    assert client.delete(f"{base}/{a['id']}", headers=h).status_code == 404


def test_ai_off_does_not_break_base_layer(client):
    h = _auth(client)
    st = client.get("/api/ai/status", headers=h).json()
    assert st["enabled"] is False and st["available"] is False
    vid = client.get("/api/videos", headers=h).json()[0]["id"]
    r = client.post(f"/api/videos/{vid}/analyze", headers=h,
                    json={"contact_sec": 1.5, "box": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}})
    assert r.status_code == 503
    assert client.get(f"/api/videos/{vid}/predictions", headers=h).json() == []
    assert client.get("/api/videos", headers=h).status_code == 200  # tang san van song


def test_delete_video_removes_file_and_children(client):
    h = _auth(client)
    vid = client.get("/api/videos", headers=h).json()[0]["id"]
    before = len(list((Path(_tmp) / "uploads").iterdir()))
    assert client.delete(f"/api/videos/{vid}", headers=h).status_code == 204
    assert len(list((Path(_tmp) / "uploads").iterdir())) == before - 1
    assert client.get(f"/api/videos/{vid}/labels", headers=h).status_code == 404
