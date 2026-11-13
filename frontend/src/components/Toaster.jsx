import { useEffect, useState } from "react";
import Icon from "./Icons.jsx";
import { BRAND } from "../brand.js";

// Thong bao ngan goc man hinh. Goi toast("Da tai len") o bat ky dau, khong can truyen props.
export function toast(message, kind = "ok") {
  window.dispatchEvent(new CustomEvent("bct-toast", { detail: { message, kind } }));
}

export default function Toaster() {
  const [items, setItems] = useState([]);
  useEffect(() => {
    let n = 0;
    const on = (e) => {
      const id = ++n;
      setItems((xs) => [...xs.slice(-2), { id, ...e.detail }]);
      setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), 3200);
    };
    window.addEventListener("bct-toast", on);
    return () => window.removeEventListener("bct-toast", on);
  }, []);
  return (
    <div className="toasts" role="status" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          <Icon name={t.kind === "error" ? "info" : "check"} size={16} />
          {t.message}
        </div>
      ))}
    </div>
  );
}

// Vong quay hinh qua cau dung khi dang tai trang
export function Loading({ text = "Đang tải…" }) {
  return (
    <div className="loading" role="status">
      <img src={BRAND.logo} alt="" className="loading-logo" />
      <span className="muted">{text}</span>
    </div>
  );
}
