import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: giao dien chay o :5173, goi API qua proxy toi uvicorn :8000
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
