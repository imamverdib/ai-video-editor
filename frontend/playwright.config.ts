import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:8877', headless: true },
  webServer: {
    command: 'cd .. && PYTHONPATH=. .venv/bin/python tests/browser_server.py',
    url: 'http://127.0.0.1:8877/api/health',
    reuseExistingServer: false,
    timeout: 30000,
  },
})
