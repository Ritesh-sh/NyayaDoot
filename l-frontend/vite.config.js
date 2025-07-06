import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    open: true,
  },
  build: {
    outDir: 'dist',
  },
  // Enable fallback for React Router (SPA)
  resolve: {
    alias: {},
  },
  // For SPA fallback
  preview: {
    open: true,
  },
  // Vite handles SPA fallback automatically, but if needed:
  // Uncomment below if you use a custom server
  // server: {
  //   historyApiFallback: true,
  // },
})
