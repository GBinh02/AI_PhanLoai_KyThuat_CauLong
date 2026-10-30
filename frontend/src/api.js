// Lop goi API mong. Token luu o localStorage (try/catch phong truong hop bi chan).
const KEY = "bct_token";

export const getToken = () => {
  try { return localStorage.getItem(KEY); } catch { return null; }
};
export const setToken = (t) => {
  try { t ? localStorage.setItem(KEY, t) : localStorage.removeItem(KEY); } catch { /* bo qua */ }
};

async function request(path, { method = "GET", json, form } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  if (form) body = form;
  const res = await fetch(`/api${path}`, { method, headers, body });
  if (res.status === 401 && token) { setToken(null); window.dispatchEvent(new Event("bct-logout")); }
  if (!res.ok) {
    let msg = `Lỗi ${res.status}`;
    try {
      const d = await res.json();
      msg = typeof d.detail === "string" ? d.detail : d.detail?.[0]?.msg || msg;
    } catch { /* khong co body JSON */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  login: (username, password) => request("/auth/login", { method: "POST", json: { username, password } }),
  register: (username, password) => request("/auth/register", { method: "POST", json: { username, password } }),
  me: () => request("/auth/me"),
  videos: () => request("/videos"),
  video: (id) => request(`/videos/${id}`),
  upload: (file, title) => {
    const form = new FormData();
    form.append("file", file);
    if (title) form.append("title", title);
    return request("/videos", { method: "POST", form });
  },
  rename: (id, title) => request(`/videos/${id}`, { method: "PATCH", json: { title } }),
  remove: (id) => request(`/videos/${id}`, { method: "DELETE" }),
  streamUrl: (id) => `/api/videos/${id}/stream?token=${encodeURIComponent(getToken() || "")}`,
  labels: (id) => request(`/videos/${id}/labels`),
  addLabel: (id, body) => request(`/videos/${id}/labels`, { method: "POST", json: body }),
  deleteLabel: (id, lid) => request(`/videos/${id}/labels/${lid}`, { method: "DELETE" }),
  aiStatus: () => request("/ai/status"),
  predictions: (id) => request(`/videos/${id}/predictions`),
  analyze: (id, body) => request(`/videos/${id}/analyze`, { method: "POST", json: body }),
  clipUrl: (id, pid, win) => `/api/videos/${id}/predictions/${pid}/clip/${win}?token=${encodeURIComponent(getToken() || "")}`,
  deletePrediction: (id, pid) => request(`/videos/${id}/predictions/${pid}`, { method: "DELETE" }),
};

export const LABELS = {
  clear: "Phông cầu",
  net_shot: "Bỏ nhỏ lưới",
  short_service: "Giao cầu ngắn",
  smash: "Đập cầu",
};

export const fmtTime = (s) => {
  if (s == null || Number.isNaN(s)) return "--:--";
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
};
export const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.ceil(b / 1e3)} KB`);
