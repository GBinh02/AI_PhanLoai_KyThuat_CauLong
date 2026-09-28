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
