import { useState } from "react";
import { api, setToken } from "../api.js";

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

  return (
    <form className="card auth" onSubmit={submit}>
      <h1>{mode === "login" ? "Đăng nhập" : "Tạo tài khoản"}</h1>
      <label>Tên đăng nhập
        <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required
               minLength={3} maxLength={32} pattern="[A-Za-z0-9_.\-]+" title="Chữ, số, _ . -" />
      </label>
      <label>Mật khẩu
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
               required minLength={6} autoComplete={mode === "login" ? "current-password" : "new-password"} />
      </label>
      {error && <p className="error">{error}</p>}
      <button className="primary" disabled={busy}>{busy ? "Đang xử lý…" : mode === "login" ? "Đăng nhập" : "Đăng ký"}</button>
      <p className="muted small">
        {mode === "login" ? "Chưa có tài khoản?" : "Đã có tài khoản?"}{" "}
        <button type="button" className="link" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
          {mode === "login" ? "Đăng ký" : "Đăng nhập"}
        </button>
      </p>
    </form>
  );
}
