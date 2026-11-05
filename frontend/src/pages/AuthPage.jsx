import { useState } from "react";
import { api, setToken } from "../api.js";
import { CourtLines } from "../components/Icons.jsx";

export default function AuthPage({ onAuth }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(""); setBusy(true);
    try {
      const fn = mode === "login" ? api.login : api.register;
      const res = await fn(username.trim(), password);
      setToken(res.token);
      onAuth(await api.me());
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const login = mode === "login";
  return (
    <div className="auth-wrap">
      <section className="auth-art">
        <CourtLines className="court" />
        <h1>Xem lại từng cú đánh của bạn</h1>
        <p>Tải video buổi tập lên, tua từng khung hình, gắn nhãn kỹ thuật, hoặc để AI gợi ý loại cú đánh.</p>
      </section>
      <section className="auth-form-side">
        <form className="auth" onSubmit={submit}>
          <h2>{login ? "Đăng nhập" : "Tạo tài khoản"}</h2>
          <label>Tên đăng nhập
            <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required autoComplete="username"
                   minLength={3} maxLength={32} pattern="[A-Za-z0-9_.\-]+" title="Chữ, số, dấu _ . -" />
          </label>
          <label>Mật khẩu
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                   required minLength={6} autoComplete={login ? "current-password" : "new-password"} />
          </label>
          {error && <p className="error" role="alert">{error}</p>}
          <button className="btn primary wide" disabled={busy}>
            {busy ? <span className="spinner" /> : null}{login ? "Đăng nhập" : "Đăng ký"}
          </button>
          <p className="muted small">
            {login ? "Chưa có tài khoản?" : "Đã có tài khoản?"}{" "}
            <button type="button" className="link" onClick={() => { setMode(login ? "register" : "login"); setError(""); }}>
              {login ? "Đăng ký" : "Đăng nhập"}
            </button>
          </p>
        </form>
      </section>
    </div>
  );
}
