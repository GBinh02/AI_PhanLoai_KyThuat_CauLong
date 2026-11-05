import { useState } from "react";
import { LABELS, fmtTime } from "../api.js";
import Icon from "./Icons.jsx";

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
      <div className="addrow">
        <select value={label} onChange={(e) => setLabel(e.target.value)} aria-label="Loại kỹ thuật">
          {Object.entries(LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <input placeholder="Ghi chú (không bắt buộc)" value={note} maxLength={300} onChange={(e) => setNote(e.target.value)} />
        <button className="btn primary" onClick={add} disabled={busy}><Icon name="tag" size={16} />Gắn nhãn tại thời điểm này</button>
      </div>
      {labels.length === 0 ? <p className="muted small" style={{ marginTop: ".7rem" }}>Chưa có nhãn nào. Tua đến một cú đánh rồi gắn nhãn.</p> : (
        <ul className="list">
          {labels.map((l) => (
            <li key={l.id}>
              <button className="tchip" onClick={() => onSeek(l.time_sec)} title="Nhảy tới thời điểm này">{fmtTime(l.time_sec)}</button>
              <span className={`tag ${l.label}`}>{LABELS[l.label]}</span>
              <span className="muted small grow">{l.note}</span>
              <button className="btn ghost danger icon-only" onClick={() => onDelete(l.id)} aria-label="Xóa nhãn"><Icon name="x" size={15} /></button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
