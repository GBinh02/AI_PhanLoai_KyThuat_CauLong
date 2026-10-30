"""Kiem tra luong AI voi checkpoint THAT; chi thay buoc MediaPipe (can tai file .task tu Google)
bang dac trung gia, de test chay duoc offline. Bo qua neu chua cai tensorflow/mediapipe.

Chay:  pytest backend/tests/test_ai_integration.py
"""
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("mediapipe")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import ai_service, config  # noqa: E402
from backend.app.main import app  # noqa: E402

from .conftest import TMP as _tmp  # noqa: E402

pytestmark = pytest.mark.skipif(not config.AI_CHECKPOINT.is_file(), reason="chua co checkpoint")


@pytest.fixture(scope="module")
def env():
    video = Path(_tmp) / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=25",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    with TestClient(app) as c:
        tok = c.post("/api/auth/register", json={"username": "tester", "password": "secret123"}).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        with video.open("rb") as f:
            vid = c.post("/api/videos", headers=h, files={"file": ("v.mp4", f, "video/mp4")}).json()["id"]
        yield c, h, vid


@pytest.fixture(autouse=True)
def fake_pose(monkeypatch):
    """Thay MediaPipe: kiem tra clip da duoc cat/crop dung kich thuoc, tra dac trung ngau nhien (32, 132)."""
    seen = []

    def fake_process_clip(path, model_path, min_visible_ratio=0.7, min_confidence=0.5):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, check=True).stdout.strip().split(",")
        seen.append((int(out[0]), int(out[1]), float(out[2])))
        rng = np.random.default_rng(0)
        return rng.normal(size=(32, 132)).astype(np.float32), "OK", 1.0, 1.0

    monkeypatch.setattr(ai_service, "_ensure_pose_model", lambda: "fake.task")
    import ai.pose_extraction.extract_pose_features as epf
    monkeypatch.setattr(epf, "process_clip", fake_process_clip)
    return seen


def test_status_available():
    st = ai_service.status()
    assert st["available"], st["reasons"]


def test_analyze_end_to_end(env, fake_pose):
    c, h, vid = env
    r = c.post(f"/api/videos/{vid}/analyze", headers=h,
               json={"contact_sec": 2.0, "box": {"x": 0.25, "y": 0.1, "w": 0.4, "h": 0.8}})
    assert r.status_code == 201, r.text
    p = r.json()
    assert p["label"] in config.CLASS_NAMES and 0.25 <= p["confidence"] <= 1
    assert sum(p["probs"].values()) == pytest.approx(1, abs=1e-4)
    assert {d["window"] for d in p["details"]} == {"default", "service"}
    # crop phai dung ty le huan luyen 0.65 (rong/cao) bat ke hop nguoi dung ve ra sao; cua so 2.21 s va 1.1 s
    (w1, h1, d1), (w2, h2, d2) = fake_pose
    assert (w1, h1) == (w2, h2) and w1 / h1 == pytest.approx(0.65, abs=0.02) and h1 <= 360
    assert d1 == pytest.approx(2.21, abs=0.1) and d2 == pytest.approx(1.1, abs=0.1)
    assert len(c.get(f"/api/videos/{vid}/predictions", headers=h).json()) == 1


def test_contact_near_start_uses_only_short_window(env):
    c, h, vid = env
    r = c.post(f"/api/videos/{vid}/analyze", headers=h,
               json={"contact_sec": 0.5, "box": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}})
    assert r.status_code == 201
    d = {x["window"]: x["status"] for x in r.json()["details"]}
    assert d == {"default": "OUT_OF_RANGE", "service": "OK"} and r.json()["window_used"] == "service"


def test_rejects_bad_input(env):
    c, h, vid = env
    box = {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}
    assert c.post(f"/api/videos/{vid}/analyze", headers=h, json={"contact_sec": 0.1, "box": box}).status_code == 422
    assert c.post(f"/api/videos/{vid}/analyze", headers=h, json={"contact_sec": 99, "box": box}).status_code == 422
    tiny = {"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01}
    assert c.post(f"/api/videos/{vid}/analyze", headers=h, json={"contact_sec": 2, "box": tiny}).status_code == 422


def test_forced_window_and_clip_endpoint(env):
    c, h, vid = env
    box = {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}
    r = c.post(f"/api/videos/{vid}/analyze", headers=h, json={"contact_sec": 2.0, "box": box, "window": "service"})
    assert r.status_code == 201
    p = r.json()
    assert [d["window"] for d in p["details"]] == ["service"] and p["window_used"] == "service"
    clip = c.get(f"/api/videos/{vid}/predictions/{p['id']}/clip/service", headers=h)
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4" and len(clip.content) > 1000
    assert c.get(f"/api/videos/{vid}/predictions/{p['id']}/clip/default", headers=h).status_code == 404
    # ep cua so chuan khi cu danh sat dau video -> loi ro rang, khong de lai clip mo coi
    before = len(list(config.DEBUG_DIR.iterdir()))
    bad = c.post(f"/api/videos/{vid}/analyze", headers=h, json={"contact_sec": 0.5, "box": box, "window": "default"})
    assert bad.status_code == 422 and len(list(config.DEBUG_DIR.iterdir())) == before
    assert c.delete(f"/api/videos/{vid}/predictions/{p['id']}", headers=h).status_code == 204
    assert not (config.DEBUG_DIR / p["details"][0]["clip"]).exists()


@pytest.mark.parametrize("box", [
    {"x": 0.45, "y": 0.05, "w": 0.1, "h": 0.9},     # hop cao, hep (nhu nguoi dung hay ve)
    {"x": 0.1, "y": 0.4, "w": 0.6, "h": 0.2},       # hop thap, rong
    {"x": 0.9, "y": 0.8, "w": 0.1, "h": 0.2},       # sat goc khung hinh
])
def test_crop_keeps_training_aspect_and_stays_inside_frame(box):
    x, y, w, h = ai_service._box_to_pixels(box, 640, 360)
    assert 0 <= x and 0 <= y and x + w <= 640 and y + h <= 360
    assert w / h == pytest.approx(0.65, abs=0.03)
