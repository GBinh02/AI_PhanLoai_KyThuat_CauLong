import { useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { fmtTime } from "../api.js";
import Icon from "./Icons.jsx";

const RATES = [0.1, 0.25, 0.5, 1, 1.5, 2];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/**
 * Trinh phat tuy bien: play/pause, tua, toc do cham, tung khung hinh (←/→).
 * Cung la lop phu de ve hop quanh nguoi danh (toa do chuan hoa 0..1 so voi khung video).
 */
export default function Player({ ref, src, fps, width, height, drawing, box, onBox, marks = [], extraBox }) {
  const videoRef = useRef(null);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [paused, setPaused] = useState(true);
  const [rate, setRate] = useState(1);
  const [draft, setDraft] = useState(null);
  const drag = useRef(null);
  const frame = 1 / (fps || 30);

  const seek = useCallback((t) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = clamp(t, 0, v.duration || t);
    setTime(v.currentTime);
  }, []);
  const step = useCallback((n) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    seek(v.currentTime + n * frame);
  }, [seek, frame]);
  const toggle = useCallback(() => {
    const v = videoRef.current;
    if (v) (v.paused ? v.play() : v.pause());
  }, []);

  useImperativeHandle(ref, () => ({ seek, getTime: () => videoRef.current?.currentTime ?? 0, pause: () => videoRef.current?.pause() }), [seek]);

  // dong bo thoi gian muot khi dang phat
  useEffect(() => {
    let raf;
    const tick = () => {
      const v = videoRef.current;
      if (v) setTime((p) => (Math.abs(p - v.currentTime) > 0.0005 ? v.currentTime : p));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = rate; }, [rate, src]);

  // phim tat (bo qua khi dang go chu trong o nhap)
  useEffect(() => {
    const onKey = (e) => {
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      if (e.code === "Space") { e.preventDefault(); toggle(); }
      else if (e.code === "ArrowLeft") { e.preventDefault(); e.shiftKey ? seek((videoRef.current?.currentTime ?? 0) - 1) : step(-1); }
      else if (e.code === "ArrowRight") { e.preventDefault(); e.shiftKey ? seek((videoRef.current?.currentTime ?? 0) + 1) : step(1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, seek, step]);

  // ---- ve hop
  const rel = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: clamp((e.clientX - r.left) / r.width, 0, 1), y: clamp((e.clientY - r.top) / r.height, 0, 1) };
  };
  const toBox = (a, b) => ({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) });
  const down = (e) => { if (!drawing) return; e.currentTarget.setPointerCapture(e.pointerId); drag.current = rel(e); setDraft(null); };
  const move = (e) => { if (drag.current) setDraft(toBox(drag.current, rel(e))); };
  const up = (e) => {
    if (!drag.current) return;
    const b = toBox(drag.current, rel(e));
    drag.current = null; setDraft(null);
    if (b.w > 0.02 && b.h > 0.02) onBox?.(b);
  };

  const ar = width && height ? width / height : 16 / 9;
  const shown = draft || box;
  const pct = (b) => ({ left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` });

  return (
    <div className="player">
      <div className="stage" style={{ width: `min(100%, calc(66vh * ${ar}))`, aspectRatio: ar }}>
        <video
          ref={videoRef} src={src} playsInline preload="auto"
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
          onPlay={() => setPaused(false)} onPause={() => setPaused(true)}
          onClick={() => !drawing && toggle()}
        />
        <div className={`overlay ${drawing ? "drawing" : ""}`} onPointerDown={down} onPointerMove={move} onPointerUp={up}>
          {drawing && !draft && <span className="hint">Kéo chuột quanh người đánh</span>}
          {shown && <div className="box" style={pct(shown)} />}
          {extraBox && <div className="box alt" style={pct(extraBox)} />}
        </div>
      </div>

      <div className="seek">
        <input type="range" min={0} max={duration || 0} step={frame} value={time}
               onChange={(e) => seek(parseFloat(e.target.value))} aria-label="Thanh tua" />
        {duration > 0 && marks.map((m, i) => (
          <i key={i} className={`mark ${m.kind}`} style={{ left: `${(m.t / duration) * 100}%` }} title={m.title} />
        ))}
      </div>

      <div className="controls">
        <button className="cbtn" onClick={() => seek(time - 5)} title="Lùi 5 giây" aria-label="Lùi 5 giây"><Icon name="back5" /></button>
        <button className="cbtn" onClick={() => step(-1)} title="Lùi 1 khung hình (←)" aria-label="Lùi 1 khung hình"><Icon name="stepBack" /></button>
        <button className="cbtn play" onClick={toggle} title="Phát/Dừng (Space)" aria-label={paused ? "Phát" : "Dừng"}>
          <Icon name={paused ? "play" : "pause"} size={22} />
        </button>
        <button className="cbtn" onClick={() => step(1)} title="Tiến 1 khung hình (→)" aria-label="Tiến 1 khung hình"><Icon name="stepFwd" /></button>
        <button className="cbtn" onClick={() => seek(time + 5)} title="Tiến 5 giây" aria-label="Tiến 5 giây"><Icon name="fwd5" /></button>
        <div className="rates" role="group" aria-label="Tốc độ phát">
          {RATES.map((r) => (
            <button key={r} className={r === rate ? "on" : ""} onClick={() => setRate(r)} aria-pressed={r === rate}>{r}×</button>
          ))}
        </div>
        <div className="time">{fmtTime(time)} <span>/ {fmtTime(duration)}</span><br /><span>khung {Math.round(time * (fps || 30))}</span></div>
      </div>
    </div>
  );
}
