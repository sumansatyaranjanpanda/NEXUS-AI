import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND_URL = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // SSE streams (/v1/reports/stream/*) need configure: to disable
      // response buffering so events arrive immediately in the browser.
      '/v1': {
        target: BACKEND_URL,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyRes', (_proxyRes, _req, res) => {
            // Disable Vite's internal response buffering for SSE
            res.setHeader('X-Accel-Buffering', 'no')
          })
        },
      },
      '/health': { target: BACKEND_URL, changeOrigin: true },
    },
  },
})
