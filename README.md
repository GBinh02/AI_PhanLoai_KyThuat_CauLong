# Badminton Technique Classifier

He thong phan loai ky thuat danh cau long (smash, clear, net shot, service)
tu video, su dung MediaPipe Pose va mo hinh hoc sau nhe (LSTM/MLP).
Do an tot nghiep - Nganh Cong nghe thong tin.

## Cau truc thu muc

- `data/raw/` : ban sao du lieu CSV cong khai (ShuttleSet - stroke-level annotation)
- `data/videos/` : video tran dau goc, tai bang yt-dlp (khong commit)
- `data/clips/` : clip da cat theo tung cu danh, chia theo nhan (khong commit)
- `scripts/data_prep/` : script tai video + cat clip (extract_stroke_clips.py)
- `ai/` : mo-đun AI - pose extraction, feature engineering, model, baseline, evaluation
- `backend/` : API tang san (upload/quan ly video)
- `frontend/` : giao dien web
- `notebooks/` : EDA, thu nghiem nhanh
- `report/` : bao cao do an, slide bao ve

## Bat dau

1. Sao chep du lieu cong khai vao `data/raw/ShuttleSet/`
   (tu repo https://github.com/wywyWang/CoachAI-Projects, thu muc ShuttleSet/)
2. Cai dat moi truong:
   ```
   pip install -r scripts/data_prep/requirements.txt
   ```
3. Xem huong dan trong `scripts/data_prep/extract_stroke_clips.py` de tai video
   va cat clip theo tung ky thuat.

## Web app (muc 5.1 tang san + muc 5.2 lop AI)

Backend FastAPI + SQLite (`backend/`), frontend React + Vite (`frontend/`). Can `ffmpeg`/`ffprobe` trong PATH.

**Chay che do phat trien** (2 terminal, tu thu muc goc repo):
```
pip install -r backend/requirements.txt          # tang san; them -r backend/requirements-ai.txt de bat AI
uvicorn backend.app.main:app --reload             # API tai http://127.0.0.1:8000

cd frontend && npm install && npm run dev         # giao dien tai http://localhost:5173
```

**Chay 1 cong (demo/bao ve):** `cd frontend && npm run build`, sau do chi can
`uvicorn backend.app.main:app` - backend tu phuc vu `frontend/dist` tai http://127.0.0.1:8000.

**Tang san** (dang nhap, upload, phat lai tua/cham/tung khung hinh, lich su, gan nhan thu cong) chay doc lap:
dat `AI_ENABLED=0` hoac khong cai `requirements-ai.txt` la AI tu vo hieu, phan con lai khong anh huong.

**Lop AI** (bat/tat tren giao dien): can `backend/requirements-ai.txt`, checkpoint
`ai/models/checkpoints/bilstm_motion_seed5.keras` va file `models/pose_landmarker_full.task`
(tu tai o lan phan tich dau, hoac chay `python ai/pose_extraction/extract_pose_features.py download-model --variant full`).
Luong: nguoi dung danh dau thoi diem cham cau + ve hop quanh nguoi danh -> backend cat cua so, crop, MediaPipe Pose,
`build_features(..., "motion")`, BiLSTM -> nhan + do tin cay. Vi luc suy luan chua biet nhan, backend chay ca hai cua so
(chuan 0,98+1,23 s va giao cau ngan 0,3+0,8 s) va lay ket qua co do tin cay cao hon; ket qua tung cua so van hien o muc "Chi tiet".

**Bien moi truong:** `APP_SECRET_KEY` (bat buoc dat khi trien khai that), `APP_DATA_DIR`, `APP_MAX_UPLOAD_MB` (mac dinh 500),
`AI_ENABLED`, `AI_CHECKPOINT`, `POSE_MODEL`.

**Kiem thu:** `pytest backend/tests` (11 test; test AI dung checkpoint that nhung thay buoc MediaPipe bang dac trung gia
de chay offline, tu bo qua neu chua cai tensorflow/mediapipe).
