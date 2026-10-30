import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtSize, fmtTime } from "../api.js";

export default function LibraryPage() {
  const [videos, setVideos] = useState(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [title, setTitle] = useState("");
  const fileRef = useRef(null);

  const load = useCallback(() => {
    api.videos().then(setVideos).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const upload = async (e) => {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setError(""); setUploading(true);
    try {
      await api.upload(file, title.trim());
      setTitle("");
      fileRef.current.value = "";
      load();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const remove = async (v) => {
    if (!window.confirm(`Xóa video "${v.title}" cùng toàn bộ nhãn và kết quả AI?`)) return;
    try { await api.remove(v.id); load(); } catch (err) { setError(err.message); }
  };

  return (
    <>
      <form className="card upload" onSubmit={upload}>
        <h2>Tải video lên</h2>
        <div className="row">
          <input ref={fileRef} type="file" accept="video/*,.mkv,.avi" required />
          <input placeholder="Tiêu đề (không bắt buộc)" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} />
          <button className="primary" disabled={uploading}>{uploading ? "Đang tải lên…" : "Tải lên"}</button>
        </div>
        <p className="muted small">Nên dùng MP4 (H.264) để trình duyệt phát được mượt, tua và xem từng khung hình.</p>
      </form>

      {error && <p className="error">{error}</p>}

      <h2>Lịch sử video</h2>
      {videos === null ? <p className="muted">Đang tải…</p> : videos.length === 0 ? (
        <p className="muted">Chưa có video nào. Hãy tải một video lên để bắt đầu.</p>
      ) : (
        <ul className="grid">
          {videos.map((v) => (
            <li key={v.id} className="card video-card">
              <Link to={`/videos/${v.id}`}>
                <strong>{v.title}</strong>
                <span className="muted small">
                  {fmtTime(v.duration)} · {v.width}×{v.height} · {fmtSize(v.size_bytes)}
                </span>
              </Link>
              <span className="small">
                {v.n_labels} nhãn thủ công · {v.n_predictions} kết quả AI
              </span>
              <span className="muted small">{new Date(v.created_at + "Z").toLocaleString("vi-VN")}</span>
              <button className="link danger" onClick={() => remove(v)}>Xóa</button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
