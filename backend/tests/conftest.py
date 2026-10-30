"""Moi truong test dung chung (phai dat bien moi truong TRUOC khi import backend.app)."""
import os
import tempfile

TMP = tempfile.mkdtemp(prefix="bct_test_")
os.environ["APP_DATA_DIR"] = TMP
os.environ["AI_ENABLED"] = "1"
os.environ["APP_SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long!!"
