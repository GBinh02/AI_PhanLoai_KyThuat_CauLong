import { useState } from "react";
import { api, setToken } from "../api.js";
import { BRAND } from "../brand.js";
import Icon from "../components/Icons.jsx";

export default function AuthPage({ onAuth }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
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
    <div className="auth-screen" style={{ "--auth-bg": `url("${BRAND.loginBackground}")` }}>
      <form className="auth-card" onSubmit={submit}>
        <div className="net" aria-hidden="true" />
        <div className="auth-brand">
          <img src={BRAND.logo} alt="" />
          <span>{BRAND.name}</span>
        </div>
        <h1>{login ? "Đăng nhập" : "Tạo tài khoản"}</h1>
        <p className="muted">{login ? "Xem lại buổi tập và để AI gợi ý từng cú đánh." : "Chỉ cần tên đăng nhập và mật khẩu."}</p>

        <label>Tên đăng nhập
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required autoComplete="username"
                 minLength={3} maxLength={32} pattern="[A-Za-z0-9_.\-]+" title="Chữ, số, dấu _ . -" />
        </label>
        <label>Mật khẩu
          <span className="pw">
            <input type={show ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)}
                   required minLength={6} autoComplete={login ? "current-password" : "new-password"} />
            <button type="button" className="pw-toggle" onClick={() => setShow(!show)}
                    aria-label={show ? "Ẩn mật khẩu" : "Hiện mật khẩu"} title={show ? "Ẩn mật khẩu" : "Hiện mật khẩu"}>
              <Icon name={show ? "eyeOff" : "eye"} size={18} />
            </button>
          </span>
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="btn accent wide" disabled={busy}>
          {busy ? <span className="spinner dark" /> : null}{login ? "Đăng nhập" : "Tạo tài khoản"}
        </button>
        <p className="switch-mode">
          {login ? "Chưa có tài khoản?" : "Đã có tài khoản?"}{" "}
          <button type="button" className="link" onClick={() => { setMode(login ? "register" : "login"); setError(""); }}>
            {login ? "Tạo tài khoản" : "Đăng nhập"}
          </button>
        </p>
      </form>
    </div>
  );
}
