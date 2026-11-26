import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // VITE_API_TARGET lets you point the dev proxy at a non-default backend.
  const env = process.env as Record<string, string | undefined>
  void mode
  const target = env.VITE_API_TARGET ?? 'http://127.0.0.1:8000'
  return {
    plugins: [react()],
    server: {
      port: 5173,
      // Proxying keeps the browser on a single origin during development, so
      // relative /api URLs behave identically in dev and in the built app.
      proxy: { '/api': { target, changeOrigin: true } },
    },
    build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 900 },
  }
})
