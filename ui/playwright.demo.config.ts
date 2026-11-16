import { defineConfig, devices } from "@playwright/test";

// Smoke tests for the static console demo (docs/07-deployment.md#the-console-demo): no gateway, only the demo build
// served as a static host would. Build it first (`npm run build:demo`). With DEMO_URL set, the same tests run against a
// deployed copy instead (DEMO_URL=https://dev.meritorder.dev/ npm run e2e:demo), which also checks its headers' CSP.
const PORT = 4174;
const live = process.env.DEMO_URL;

export default defineConfig({
  testDir: "./e2e",
  testMatch: /demo\.spec\.ts/,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never", outputFolder: "playwright-report-demo" }]] : "list",
  timeout: 30_000,
  use: { baseURL: live ?? `http://127.0.0.1:${PORT}/`, trace: "retain-on-failure", screenshot: "only-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: live ? undefined : {
    command: `npx vite preview --mode demo --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
