import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, getToken, setToken } from "./api.js";
import { BRAND } from "./brand.js";
import Toaster, { Loading } from "./components/Toaster.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import LibraryPage from "./pages/LibraryPage.jsx";
import VideoPage from "./pages/VideoPage.jsx";

export default function App() {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(!getToken());
  const nav = useNavigate();
  const onLogin = useLocation().pathname === "/login";

  useEffect(() => {
    if (getToken()) {
      api.me().then(setUser).catch(() => setToken(null)).finally(() => setReady(true));
    }
    const onLogout = () => { setUser(null); nav("/login"); };
    window.addEventListener("bct-logout", onLogout);
    return () => window.removeEventListener("bct-logout", onLogout);
  }, [nav]);

  const logout = () => { setToken(null); setUser(null); nav("/login"); };
  if (!ready) return <Loading />;

  return (
    <>
      {!onLogin && (
        <div className="page-bg" aria-hidden="true"
             style={{ "--page-bg": `url("${BRAND.pageBackground}")`, "--page-bg-strength": BRAND.pageBackgroundStrength }} />
      )}
      {!onLogin && (
        <header className="topbar">
          <Link to="/" className="brand">
            <img src={BRAND.logo} alt="" className="logo" />
            {BRAND.name}
          </Link>
          {user && (
            <span className="who">
              <span className="avatar" aria-hidden="true">{user.username[0].toUpperCase()}</span>
              {user.username}
              <button className="btn secondary sm" onClick={logout}>Đăng xuất</button>
            </span>
          )}
        </header>
      )}
      <main className={onLogin ? "bleed" : ""}>
        <Routes>
          <Route path="/login" element={user ? <Navigate to="/" /> : <AuthPage onAuth={setUser} />} />
          <Route path="/" element={user ? <LibraryPage user={user} /> : <Navigate to="/login" />} />
          <Route path="/videos/:id" element={user ? <VideoPage /> : <Navigate to="/login" />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
      <Toaster />
    </>
  );
}
