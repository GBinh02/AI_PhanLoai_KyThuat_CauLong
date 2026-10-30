import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api.js";
import AiPanel from "../components/AiPanel.jsx";
import LabelPanel from "../components/LabelPanel.jsx";
import Player from "../components/Player.jsx";

const AI_KEY = "bct_ai_on";
const readAiPref = () => { try { return localStorage.getItem(AI_KEY) === "1"; } catch { return false; } };
const writeAiPref = (v) => { try { localStorage.setItem(AI_KEY, v ? "1" : "0"); } catch { /* bo qua */ } };

export default function VideoPage() {
  const { id } = useParams();
  const nav = useNavigate();
  const playerRef = useRef(null);
  const [video, setVideo] = useState(null);
  const [labels, setLabels] = useState([]);
  const [preds, setPreds] = useState([]);
  const [aiStatus, setAiStatus] = useState(null);
  const [aiOn, setAiOn] = useState(readAiPref);
  const [error, setError] = useState("");

  // trang thai luong AI
  const [contact, setContact] = useState(null);
  const [box, setBox] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [aiError, setAiError] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [win, setWin] = useState("auto");

  useEffect(() => {
    let alive = true;
    Promise.all([api.video(id), api.labels(id), api.predictions(id)])
      .then(([v, l, p]) => { if (alive) { setVideo(v); setLabels(l); setPreds(p); } })
      .catch((e) => alive && setError(e.message));
    api.aiStatus().then((s) => alive && setAiStatus(s)).catch(() => {});
    return () => { alive = false; };
  }, [id]);

  const getTime = useCallback(() => playerRef.current?.getTime() ?? 0, []);
  const seek = useCallback((t) => playerRef.current?.seek(t), []);
  const aiActive = aiOn && !!aiStatus?.available;

  const toggleAi = (v) => { setAiOn(v); writeAiPref(v); if (!v) setDrawing(false); };

  const addLabel = async (body) => {
    try { const l = await api.addLabel(id, body); setLabels((ls) => [...ls, l].sort((a, b) => a.time_sec - b.time_sec)); }
    catch (e) { setError(e.message); }
  };
  const delLabel = async (lid) => {
    try { await api.deleteLabel(id, lid); setLabels((ls) => ls.filter((l) => l.id !== lid)); }
    catch (e) { setError(e.message); }
  };

  const analyze = async () => {
    setAiError(""); setAnalyzing(true); setDrawing(false);
    try {
      const r = await api.analyze(id, { contact_sec: contact, box, window: win });
      setPreds((ps) => [...ps, r].sort((a, b) => a.contact_sec - b.contact_sec));
      setSelectedId(r.id);
    } catch (e) { setAiError(e.message); } finally { setAnalyzing(false); }
  };
  const delPred = async (pid) => {
    try { await api.deletePrediction(id, pid); setPreds((ps) => ps.filter((p) => p.id !== pid)); }
    catch (e) { setAiError(e.message); }
  };

  const rename = async () => {
    const t = window.prompt("Tiêu đề mới", video.title);
    if (t && t.trim()) setVideo(await api.rename(id, t.trim()));
  };
  const remove = async () => {
    if (!window.confirm("Xóa video này cùng toàn bộ nhãn và kết quả AI?")) return;
    await api.remove(id); nav("/");
  };

  if (error && !video) return <p className="error">{error} <Link to="/">Về danh sách</Link></p>;
  if (!video) return <p className="muted">Đang tải…</p>;

  const selected = preds.find((p) => p.id === selectedId);
  const marks = [
    ...labels.map((l) => ({ t: l.time_sec, kind: "label", title: `Nhãn: ${l.label}` })),
    ...(aiActive ? preds.map((p) => ({ t: p.contact_sec, kind: "ai", title: `AI: ${p.label}` })) : []),
  ];

  return (
    <>
      <div className="row between">
        <div>
          <Link to="/" className="small">← Lịch sử video</Link>
          <h1 className="title">{video.title}</h1>
        </div>
        <div className="row">
          <button className="link" onClick={rename}>Đổi tên</button>
          <button className="link danger" onClick={remove}>Xóa</button>
        </div>
      </div>
      {error && <p className="error">{error}</p>}

      <div className="layout">
        <div>
          <Player
            ref={playerRef} src={api.streamUrl(id)} fps={video.fps} width={video.width} height={video.height}
            drawing={aiActive && drawing} box={aiActive ? box : null} onBox={(b) => { setBox(b); setSelectedId(null); setDrawing(false); }}
            extraBox={aiActive && selected ? selected.box : null} marks={marks}
          />
          <p className="muted small">Phím tắt: Space phát/dừng · ←/→ lùi/tiến 1 khung hình · Shift+←/→ lùi/tiến 1 giây.</p>
        </div>
        <div className="side">
          <LabelPanel labels={labels} getTime={getTime} onAdd={addLabel} onDelete={delLabel} onSeek={seek} />
          <AiPanel
            status={aiStatus} on={aiOn} onToggle={toggleAi}
            contact={contact} onMarkContact={() => setContact(getTime())}
            box={box} onClearBox={() => { setBox(null); setSelectedId(null); }} drawing={drawing}
            onToggleDraw={() => { playerRef.current?.pause(); setSelectedId(null); setDrawing((d) => !d); }}
            win={win} onWin={setWin} analyzing={analyzing} error={aiError} results={preds} selectedId={selectedId} onSelect={(pid) => setSelectedId((s) => (s === pid ? null : pid))}
            onAnalyze={analyze} onSeek={seek} onDelete={delPred}
            onUseAsLabel={(r) => addLabel({ time_sec: r.contact_sec, label: r.label, note: `AI gợi ý (${(r.confidence * 100).toFixed(0)}%)` })}
          />
        </div>
      </div>
    </>
  );
}
