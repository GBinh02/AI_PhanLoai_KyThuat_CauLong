import { useState } from "react";
import { LABELS, fmtTime } from "../api.js";

export default function LabelPanel({ labels, getTime, onAdd, onDelete, onSeek }) {
  const [label, setLabel] = useState("smash");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  const add = async () => {
    setBusy(true);
    try { await onAdd({ time_sec: getTime(), label, note }); setNote(""); } finally { setBusy(false); }
  };

  return (
    <section className="card">
      <h3>Gắn nhãn thủ công</h3>
      <div className="row wrap">
        <select value={label} onChange={(e) => setLabel(e.target.value)}>
          {Object.entries(LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <input placeholder="Ghi chú (không bắt buộc)" value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} />
        <button className="primary" onClick={add} disabled={busy}>Gắn nhãn tại thời điểm hiện tại</button>
      </div>
      {labels.length === 0 ? <p className="muted small">Chưa có nhãn nào.</p> : (
        <ul className="list">
          {labels.map((l) => (
            <li key={l.id}>
              <button className="link" onClick={() => onSeek(l.time_sec)}>{fmtTime(l.time_sec)}</button>
              <span className={`tag ${l.label}`}>{LABELS[l.label]}</span>
              <span className="muted small grow">{l.note}</span>
              <button className="link danger" onClick={() => onDelete(l.id)} aria-label="Xóa nhãn">✕</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
