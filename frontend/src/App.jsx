import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { api, getToken, setToken } from "./api.js";
import AuthPage from "./pages/AuthPage.jsx";
import LibraryPage from "./pages/LibraryPage.jsx";
import VideoPage from "./pages/VideoPage.jsx";

export default function App() {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(!getToken());
  const nav = useNavigate();

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
        <Link to="/" className="brand">🏸 Phân loại kỹ thuật cầu lông</Link>
        {user && (
          <span className="who">
            {user.username} <button className="link" onClick={logout}>Đăng xuất</button>
          </span>
        )}
      </header>
      <main>
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
