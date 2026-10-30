import { useState } from "react";
import { LABELS, api, fmtTime } from "../api.js";

const pct = (p) => `${(p * 100).toFixed(1)}%`;

function Result({ r, selected, onSelect, onSeek, onDelete, onUseAsLabel }) {
  const [open, setOpen] = useState(false);
  return (
    <li className={`result ${selected ? "sel" : ""}`}>
      <div className="row wrap">
        <button className="link" onClick={() => { onSeek(r.contact_sec); onSelect(r.id); }}>{fmtTime(r.contact_sec)}</button>
        <span className={`tag ${r.label}`}>{LABELS[r.label]}</span>
        <strong>{pct(r.confidence)}</strong>
        <span className="muted small grow">cửa sổ {r.window_used === "service" ? "ngắn (giao cầu)" : "chuẩn"}</span>
        <button className="link" onClick={() => onUseAsLabel(r)} title="Lưu kết quả này thành nhãn thủ công">Dùng làm nhãn</button>
        <button className="link danger" onClick={() => onDelete(r.id)} aria-label="Xóa kết quả">✕</button>
      </div>
      <div className="bars">
        {Object.entries(r.probs).map(([k, p]) => (
          <div key={k} className="bar"><span>{LABELS[k]}</span><i><b className={k} style={{ width: pct(p) }} /></i><em>{pct(p)}</em></div>
        ))}
      </div>
      <button className="link small" onClick={() => setOpen(!open)}>{open ? "Ẩn chi tiết" : "Chi tiết từng cửa sổ"}</button>
      {open && (
        <ul className="small windows">
          {r.details.map((d) => (
            <li key={d.window}>
              <div className="muted">
                {d.window === "service" ? "Cửa sổ ngắn 0,3 s + 0,8 s" : "Cửa sổ chuẩn 0,98 s + 1,23 s"}
                {" "}({d.start.toFixed(2)}–{d.end.toFixed(2)} s):{" "}
                <strong>
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
            Clip trên là đúng phần mô hình nhìn thấy. Kiểm tra: hộp có ôm đúng một người không, và vợt có chạm cầu
            đúng giây thứ 0,98 (clip chuẩn) hoặc 0,3 (clip ngắn) tính từ đầu clip không. Lệch vài khung hình là
            kết quả có thể đổi.
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
  return (
    <section className="card ai">
      <div className="row between">
        <h3>Nhận diện kỹ thuật bằng AI</h3>
        <label className="switch">
          <input type="checkbox" checked={on && !!available} disabled={!available} onChange={(e) => onToggle(e.target.checked)} />
          <span>{on && available ? "Đang bật" : "Đang tắt"}</span>
        </label>
      </div>

      {status && !available && (
        <p className="warn small">AI không khả dụng: {status.reasons.join(" ") || "không rõ nguyên nhân"} — các chức năng còn lại vẫn dùng bình thường.</p>
      )}

      {on && available && (
        <>
          <ol className="steps small">
            <li>Tua đến đúng khung hình vợt chạm cầu, bấm <em>Đánh dấu chạm cầu</em>.</li>
            <li>Bấm <em>Vẽ hộp</em> rồi kéo chuột trên video quanh người đánh (ôm sát người; hệ thống tự mở rộng và chỉnh đúng tỉ lệ khung đã dùng lúc huấn luyện).</li>
            <li>Bấm <em>Phân tích cú đánh</em>.</li>
          </ol>
          <div className="row wrap">
            <button onClick={onMarkContact}>① Đánh dấu chạm cầu</button>
            <span className="chip">{contact != null ? fmtTime(contact) : "chưa đánh dấu"}</span>
            <button className={drawing ? "primary" : ""} onClick={onToggleDraw}>② {drawing ? "Đang vẽ… (kéo trên video)" : "Vẽ hộp"}</button>
            <span className="chip">{box ? "đã có hộp" : "chưa có hộp"}</span>
            {box && <button className="link" onClick={onClearBox}>Xóa hộp</button>}
          </div>
          <label className="rate">Loại cú đánh
            <select value={win} onChange={(e) => onWin(e.target.value)}>
              <option value="auto">Tự động (chạy cả 2 cửa sổ)</option>
              <option value="default">Không phải giao cầu (cửa sổ chuẩn)</option>
              <option value="service">Giao cầu ngắn (cửa sổ ngắn)</option>
            </select>
          </label>
          <button className="primary wide" disabled={!ready || analyzing} onClick={onAnalyze}>
            {analyzing ? "Đang phân tích… (vài giây)" : "③ Phân tích cú đánh"}
          </button>
          {error && <p className="error">{error}</p>}

          <p className="note small">
            <strong>Lưu ý về độ chính xác:</strong> mô hình đạt accuracy 84,6% trên video thi đấu (ShuttleSet) nhưng chỉ
            khoảng 52,6% trên video tự quay bằng điện thoại. Độ tin cậy hiển thị gần như không phân biệt được dự đoán
            đúng với dự đoán sai (trung bình 0,81 khi đúng so với 0,73 khi sai), nên hãy xem đây là gợi ý, không phải kết luận.
          </p>

          {results.length > 0 && (
            <ul className="list results">
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
