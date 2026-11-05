import { useState } from "react";
import { LABELS, api, fmtTime } from "../api.js";
import Icon from "./Icons.jsx";

const pct = (p) => `${(p * 100).toFixed(1)}%`;
const WINDOWS = [
  ["auto", "Tự động"],
  ["default", "Không phải giao cầu"],
  ["service", "Giao cầu ngắn"],
];

function Result({ r, selected, onSelect, onSeek, onDelete, onUseAsLabel }) {
  const [open, setOpen] = useState(false);
  return (
    <li className={`res ${r.label} ${selected ? "sel" : ""}`}>
      <div className="top">
        <button className="tchip" onClick={() => { onSeek(r.contact_sec); onSelect(r.id); }} title="Nhảy tới cú đánh và hiện hộp đã dùng">{fmtTime(r.contact_sec)}</button>
        <span className="name">{LABELS[r.label]}</span>
        <span className="conf">{pct(r.confidence)}</span>
      </div>
      <div className="sub">
        Cửa sổ {r.window_used === "service" ? "ngắn (giao cầu)" : "chuẩn"}
      </div>
      <div className="bars">
        {Object.entries(r.probs).map(([k, p]) => (
          <div key={k} className="bar"><span>{LABELS[k]}</span><i><b className={k} style={{ width: pct(p) }} /></i><em>{pct(p)}</em></div>
        ))}
      </div>
      <div className="acts">
        <button className="btn ghost sm" onClick={() => setOpen(!open)}>{open ? "Ẩn chi tiết" : "Xem clip đã cắt"}</button>
        <span className="grow" />
        <button className="btn ghost sm" onClick={() => onUseAsLabel(r)} title="Lưu kết quả này thành nhãn thủ công"><Icon name="tag" size={14} />Dùng làm nhãn</button>
        <button className="btn ghost danger icon-only" onClick={() => onDelete(r.id)} aria-label="Xóa kết quả"><Icon name="trash" size={15} /></button>
      </div>
      {open && (
        <ul className="windows">
          {r.details.map((d) => (
            <li key={d.window}>
              <div className="muted">
                {d.window === "service" ? "Cửa sổ ngắn 0,3 s + 0,8 s" : "Cửa sổ chuẩn 0,98 s + 1,23 s"}
                {" "}({d.start.toFixed(2)}–{d.end.toFixed(2)} s):{" "}
                <strong style={{ color: "var(--text)" }}>
                  {d.status === "OK" ? `${LABELS[d.label]} (${pct(d.confidence)})`
                    : d.status === "OUT_OF_RANGE" ? "vượt ngoài video, bỏ qua"
                    : "không dò đủ tư thế"}
                </strong>
              </div>
              {d.clip && (
                <video className="clip" controls loop muted playsInline preload="metadata"
                       src={api.clipUrl(r.video_id, r.id, d.window)} />
              )}
            </li>
          ))}
          <li className="muted">
            Clip là đúng phần mô hình nhìn thấy. Kiểm tra hộp có ôm đúng một người không, và vợt có chạm cầu đúng
            giây thứ 0,98 (clip chuẩn) hoặc 0,3 (clip ngắn) tính từ đầu clip không. Lệch vài khung hình là kết quả có thể đổi.
          </li>
        </ul>
      )}
    </li>
  );
}

export default function AiPanel({
  status, on, onToggle, contact, onMarkContact, box, onClearBox, drawing, onToggleDraw,
  win, onWin, analyzing, error, results, selectedId, onSelect, onAnalyze, onSeek, onDelete, onUseAsLabel,
}) {
  const available = status?.available;
  const ready = on && contact != null && box;
  const s1 = contact != null ? "done" : "active";
  const s2 = box ? "done" : contact != null ? "active" : "";
  const s3 = ready ? "active" : "";
  return (
    <section className="card">
      <div className="ai-head">
        <h3><Icon name="spark" size={16} />Nhận diện bằng AI</h3>
        <label className="switch">
          <input type="checkbox" checked={on && !!available} disabled={!available} onChange={(e) => onToggle(e.target.checked)} />
          <span className="track" />
          {on && available ? "Bật" : "Tắt"}
        </label>
      </div>

      {status && !available && (
        <div className="callout warn"><Icon name="info" size={16} />
          <span>AI không khả dụng: {status.reasons.join(" ") || "không rõ nguyên nhân"} Các chức năng còn lại vẫn dùng bình thường.</span>
        </div>
      )}
      {!on && available && <p className="muted small">Bật công tắc để AI gợi ý loại kỹ thuật cho từng cú đánh.</p>}

      {on && available && (
        <>
          <div className="steps">
            <div className={`step ${s1}`}>
              <span className="n">{contact != null ? <Icon name="check" size={14} /> : 1}</span>
              <div className="grow"><div className="lab">Đánh dấu khung vợt chạm cầu</div>
                <div className="val">{contact != null ? `Đã chọn ${fmtTime(contact)}` : "Tua đúng khung hình rồi bấm nút"}</div></div>
              <button className="btn sm" onClick={onMarkContact}><Icon name="target" size={15} />Đánh dấu</button>
            </div>
            <div className={`step ${s2}`}>
              <span className="n">{box ? <Icon name="check" size={14} /> : 2}</span>
              <div className="grow"><div className="lab">Vẽ hộp quanh người đánh</div>
                <div className="val">{box ? "Đã có hộp" : "Ôm sát người, hệ thống tự chỉnh đúng tỉ lệ"}</div></div>
              <button className={`btn sm ${drawing ? "primary" : ""}`} onClick={onToggleDraw}><Icon name="box" size={15} />{drawing ? "Đang vẽ" : "Vẽ hộp"}</button>
              {box && <button className="btn ghost sm" onClick={onClearBox}>Xóa</button>}
            </div>
            <div className={`step ${s3}`}>
              <span className="n">3</span>
              <div className="grow"><div className="lab">Phân tích cú đánh</div>
                <div className="val">Mất vài giây</div></div>
            </div>
          </div>

          <span className="fieldlabel">Loại cú đánh</span>
          <div className="seg" role="group" aria-label="Loại cú đánh">
            {WINDOWS.map(([v, t]) => (
              <button key={v} className={win === v ? "on" : ""} onClick={() => onWin(v)} aria-pressed={win === v}>{t}</button>
            ))}
          </div>

          <button className="btn primary wide" disabled={!ready || analyzing} onClick={onAnalyze}>
            {analyzing ? <><span className="spinner" />Đang phân tích…</> : "Phân tích cú đánh"}
          </button>
          {error && <p className="error" role="alert" style={{ marginTop: ".7rem" }}>{error}</p>}

          <div className="callout" style={{ marginTop: ".9rem" }}><Icon name="info" size={16} />
            <span>Mô hình đạt độ chính xác 92,7% trên video thi đấu (ShuttleSet) nhưng khoảng 68% trên video tự quay
            bằng điện thoại, và vẫn hay nhầm phông cầu với đập cầu. Độ tin cậy không phân biệt được dự đoán đúng với
            dự đoán sai (trung bình 0,81 khi đúng, 0,75 khi sai), nên hãy xem kết quả là gợi ý, không phải kết luận.</span>
          </div>

          {results.length > 0 && (
            <ul className="results">
              {results.map((r) => (
                <Result key={r.id} r={r} selected={r.id === selectedId} onSelect={onSelect} onSeek={onSeek}
                        onDelete={onDelete} onUseAsLabel={onUseAsLabel} />
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
