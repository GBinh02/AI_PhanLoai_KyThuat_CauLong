import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api, getToken, setToken } from "./api.js";
import { Shuttle } from "./components/Icons.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import LibraryPage from "./pages/LibraryPage.jsx";
import VideoPage from "./pages/VideoPage.jsx";

export default function App() {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(!getToken());
  const nav = useNavigate();
  const bleed = useLocation().pathname === "/login";

  useEffect(() => {
    if (getToken()) {
      api.me().then(setUser).catch(() => setToken(null)).finally(() => setReady(true));
    }
    const onLogout = () => { setUser(null); nav("/login"); };
    window.addEventListener("bct-logout", onLogout);
    return () => window.removeEventListener("bct-logout", onLogout);
  }, [nav]);

  const logout = () => { setToken(null); setUser(null); nav("/login"); };
  if (!ready) return <p className="center muted">Đang tải…</p>;

  return (
    <>
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="logo"><Shuttle size={22} /></span>
          Phân loại kỹ thuật cầu lông
        </Link>
        {user && (
          <span className="who">
            <span className="avatar" aria-hidden="true">{user.username[0].toUpperCase()}</span>
            {user.username}
            <button className="link" onClick={logout}>Đăng xuất</button>
          </span>
        )}
      </header>
      <main className={bleed ? "bleed" : ""}>
        <Routes>
          <Route path="/login" element={user ? <Navigate to="/" /> : <AuthPage onAuth={setUser} />} />
          <Route path="/" element={user ? <LibraryPage /> : <Navigate to="/login" />} />
          <Route path="/videos/:id" element={user ? <VideoPage /> : <Navigate to="/login" />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </>
  );
}
