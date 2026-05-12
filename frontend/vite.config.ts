import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const BACKEND_URL = env.BACKEND_URL ?? 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true,  // listen on 0.0.0.0 so Docker can expose the port
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
  }
})
