import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtSize, fmtTime } from "../api.js";
import Icon from "../components/Icons.jsx";

export default function LibraryPage() {
  const [videos, setVideos] = useState(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState("");
  const [over, setOver] = useState(false);
  const fileRef = useRef(null);

  const load = useCallback(() => {
    api.videos().then(setVideos).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const pick = (f) => { if (f) { setFile(f); setError(""); } };
  const onDrop = (e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files?.[0]); };

  const upload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setError(""); setUploading(true);
    try {
      await api.upload(file, title.trim());
      setTitle(""); setFile(null);
      if (fileRef.current) fileRef.current.value = "";
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
      <div className="page-head">
        <h1>Video của bạn</h1>
      </div>

      <form className={`dropzone ${over ? "over" : ""}`} onSubmit={upload}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
        <span className="badge"><Icon name="upload" size={22} /></span>
        <div className="grow">
          {file ? <div className="picked">{file.name} <span className="muted small">({fmtSize(file.size)})</span></div>
                : <div><strong>Kéo video vào đây</strong> <span className="muted">hoặc</span>{" "}
                    <button type="button" className="link" onClick={() => fileRef.current?.click()}>chọn từ máy</button></div>}
          <div className="muted small">MP4 (H.264) phát mượt nhất; có thể tua và xem từng khung hình.</div>
        </div>
        <input ref={fileRef} type="file" accept="video/*,.mkv,.avi" hidden onChange={(e) => pick(e.target.files?.[0])} />
        {file && (
          <>
            <input placeholder="Tiêu đề (không bắt buộc)" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} />
            <button className="btn primary" disabled={uploading}>{uploading ? <><span className="spinner" />Đang tải lên</> : "Tải lên"}</button>
            <button type="button" className="btn ghost" onClick={() => { setFile(null); if (fileRef.current) fileRef.current.value = ""; }}>Bỏ chọn</button>
          </>
        )}
      </form>

      {error && <p className="error" role="alert" style={{ marginBottom: "1rem" }}>{error}</p>}

      {videos === null ? <p className="muted">Đang tải…</p> : videos.length === 0 ? (
        <div className="empty"><Icon name="film" size={30} /><p style={{ marginTop: ".5rem" }}>Chưa có video nào. Kéo một video vào khung phía trên để bắt đầu.</p></div>
      ) : (
        <ul className="grid">
          {videos.map((v) => (
            <li key={v.id} className="vcard">
              <Link to={`/videos/${v.id}`}>
                <div className="thumb">
                  <video src={`${api.streamUrl(v.id)}#t=1`} preload="metadata" muted playsInline tabIndex={-1} />
                  <span className="dur num">{fmtTime(v.duration)}</span>
                </div>
                <div className="body">
                  <span className="t" title={v.title}>{v.title}</span>
                  <div className="chips">
                    <span className="chip">{v.n_labels} nhãn</span>
                    <span className="chip ai">{v.n_predictions} kết quả AI</span>
                    <span className="chip">{fmtSize(v.size_bytes)}</span>
                  </div>
                  <span className="muted small">{new Date(v.created_at + "Z").toLocaleString("vi-VN")}</span>
                </div>
              </Link>
              <button className="btn icon-only del" onClick={() => remove(v)} aria-label={`Xóa ${v.title}`} title="Xóa video"><Icon name="trash" size={16} /></button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
