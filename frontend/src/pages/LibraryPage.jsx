import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtSize, fmtTime } from "../api.js";
import Icon, { CourtLines } from "../components/Icons.jsx";
import { Loading, toast } from "../components/Toaster.jsx";

const fmtTotal = (s) => {
  if ((s || 0) < 60) return `${Math.round(s || 0)}s`;
  const m = Math.round(s / 60);
  return m < 60 ? `${m}′` : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}`;
};

// Bang diem kieu san thi dau: tong quan thu vien cua nguoi dung
function Scoreboard({ videos }) {
  const sum = (k) => videos.reduce((a, v) => a + (v[k] || 0), 0);
  const cells = [
    ["Video", videos.length],
    ["Nhãn đã gắn", sum("n_labels")],
    ["Kết quả AI", sum("n_predictions")],
    ["Thời lượng", fmtTotal(sum("duration"))],
  ];
  return (
    <dl className="scoreboard">
      {cells.map(([k, v]) => (
        <div key={k} className="cell">
          <dd className="digits">{v}</dd>
          <dt>{k}</dt>
        </div>
      ))}
    </dl>
  );
}

function VideoCard({ v, onRemove }) {
  const ref = useRef(null);
  // Re chuot vao the: phat thu khong tieng; roi ra: dung lai o giay thu 1
  const created = new Date(v.created_at + "Z");
  const preview = (on) => {
    const el = ref.current;
    if (!el) return;
    if (on) { el.play().catch(() => {}); } else { el.pause(); el.currentTime = 1; }
  };
  return (
    <li className="vcard" onMouseEnter={() => preview(true)} onMouseLeave={() => preview(false)}>
      <Link to={`/videos/${v.id}`} onFocus={() => preview(true)} onBlur={() => preview(false)}>
        <div className="thumb">
          <video ref={ref} src={`${api.streamUrl(v.id)}#t=1`} preload="metadata" muted loop playsInline tabIndex={-1} />
          <span className="dur num">{fmtTime(v.duration)}</span>
          <span className="play-hint" aria-hidden="true"><Icon name="play" size={20} /></span>
        </div>
        <div className="body">
          <span className="t" title={v.title}>{v.title}</span>
          <div className="chips">
            <span className={`chip ${v.n_labels ? "on" : ""}`}><Icon name="tag" size={13} />{v.n_labels} nhãn</span>
            <span className={`chip ai ${v.n_predictions ? "on" : ""}`}><Icon name="spark" size={12} />{v.n_predictions} kết quả AI</span>
            <span className="chip">{fmtSize(v.size_bytes)}</span>
          </div>
          <div className="when">
            <span><Icon name="clock" size={13} />{created.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })}</span>
            <span>{created.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" })}</span>
          </div>
        </div>
      </Link>
      <button className="btn icon-only del" onClick={() => onRemove(v)} aria-label={`Xóa ${v.title}`} title="Xóa video"><Icon name="trash" size={16} /></button>
    </li>
  );
}

export default function LibraryPage({ user }) {
  const [videos, setVideos] = useState(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState("");
  const [over, setOver] = useState(false);
  const [q, setQ] = useState("");
  const fileRef = useRef(null);

  const load = useCallback(() => {
    api.videos().then(setVideos).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const pick = (f) => { if (f) { setFile(f); setError(""); } };
  const onDrop = (e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files?.[0]); };
  const choose = () => fileRef.current?.click();

  const upload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setError(""); setUploading(true);
    try {
      const v = await api.upload(file, title.trim());
      setTitle(""); setFile(null);
      if (fileRef.current) fileRef.current.value = "";
      load();
      toast(`Đã tải lên “${v.title}”`);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const remove = async (v) => {
    if (!window.confirm(`Xóa video "${v.title}" cùng toàn bộ nhãn và kết quả AI?`)) return;
    try { await api.remove(v.id); load(); toast(`Đã xóa “${v.title}”`); } catch (err) { setError(err.message); }
  };

  const shown = (videos || []).filter((v) => v.title.toLowerCase().includes(q.trim().toLowerCase()));

  return (
    <>
      <section className="lib-hero">
        <CourtLines className="hero-court" />
        <div className="hello">
          <h1>Chào {user?.username}</h1>
          <p>Chọn một buổi tập để xem lại, hoặc tải video mới lên.</p>
        </div>
        {videos && videos.length > 0 && <Scoreboard videos={videos} />}
      </section>

      <form className={`dropzone ${over ? "over" : ""} ${file ? "has-file" : ""}`} onSubmit={upload}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
        <span className="badge"><Icon name="upload" size={22} /></span>
        <div className="grow">
          {file ? <div className="picked">{file.name} <span className="muted small">({fmtSize(file.size)})</span></div>
                : <div><strong>Kéo video vào đây</strong> <span className="muted">hoặc chọn tệp từ máy</span></div>}
          <div className="muted small">MP4 (H.264) phát mượt nhất; tối đa 500 MB.</div>
        </div>
        <input ref={fileRef} type="file" accept="video/*,.mkv,.avi" hidden onChange={(e) => pick(e.target.files?.[0])} />
        {file ? (
          <>
            <input placeholder="Tiêu đề (không bắt buộc)" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} />
            <button className="btn accent" disabled={uploading}>{uploading ? <><span className="spinner dark" />Đang tải lên</> : <><Icon name="upload" size={16} />Tải lên</>}</button>
            <button type="button" className="btn ghost" onClick={() => { setFile(null); if (fileRef.current) fileRef.current.value = ""; }}>Bỏ chọn</button>
          </>
        ) : (
          <button type="button" className="btn secondary" onClick={choose}><Icon name="film" size={16} />Chọn video</button>
        )}
      </form>

      {error && <p className="error" role="alert" style={{ marginBottom: "1rem" }}>{error}</p>}

      {videos === null ? <Loading /> : videos.length === 0 ? (
        <div className="empty">
          <CourtLines className="empty-court" />
          <h2>Chưa có video nào</h2>
          <p className="muted">Quay một buổi tập bằng điện thoại, rồi tải lên để xem lại từng khung hình.</p>
          <button type="button" className="btn accent" onClick={choose}><Icon name="upload" size={16} />Tải video đầu tiên</button>
        </div>
      ) : (
        <>
          <div className="list-head">
            <h2>Video của bạn</h2>
            {videos.length > 3 && (
              <input className="search" type="search" placeholder="Tìm theo tiêu đề" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Tìm video theo tiêu đề" />
            )}
          </div>
          {shown.length === 0 ? <p className="muted">Không có video nào khớp “{q}”.</p> : (
            <ul className="grid">
              {shown.map((v) => <VideoCard key={v.id} v={v} onRemove={remove} />)}
            </ul>
          )}
        </>
      )}
    </>
  );
}
