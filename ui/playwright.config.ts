import { defineConfig, devices } from "@playwright/test";
import { fileURLToPath } from "node:url";

// Browser smoke tests for the console. The gateway is started here, from ui/e2e (so the repo's .env
// isn't loaded), against mock backends and an emptied Postgres database: key management and saving
// configuration need Postgres. Build the UI first (`npm run build`); the gateway serves it at /ui/.
const PORT = 8111;
const e2eDir = fileURLToPath(new URL("./e2e/", import.meta.url));

export const KEYS = {
  admin: process.env.E2E_ADMIN_KEY ?? "mo-e2e-admin-key",
  lead: process.env.E2E_LEAD_KEY ?? "mo-e2e-lead-key",
  user: process.env.E2E_USER_KEY ?? "mo-e2e-user-key",
};
export const DATABASE_URL = process.env.E2E_DATABASE_URL ?? "postgresql://postgres:e2e@localhost:5432/e2etest";
export const SERVER_ENV = {
  MERITORDER_CONFIG: "meritorder.yaml",
  MERITORDER_DATABASE_URL: DATABASE_URL,
  MERITORDER_REDIS_URL: "", // in-memory state: one gateway process
  MERITORDER_DEV_KEY: "",
  MERITORDER_BATCH_KEY: "e2e-batch-key", // uploaded batch files are held encrypted until they're submitted
  MERITORDER_RESPONSES_KEY: "e2e-responses-key", // eng's stored responses are sealed with it
  E2E_ADMIN_KEY: KEYS.admin,
  E2E_LEAD_KEY: KEYS.lead,
  E2E_USER_KEY: KEYS.user,
};
export const E2E_DIR = e2eDir;

export default defineConfig({
  testDir: "./e2e",
  testIgnore: /demo\.spec\.ts/, // the static demo's own tests: playwright.demo.config.ts
  fullyParallel: false,
  workers: 1, // the tests share one gateway and change its configuration
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  timeout: 30_000,
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "seed", testMatch: /seed\.setup\.ts/ },
    { name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } }, dependencies: ["seed"] },
  ],
  webServer: {
    command: `uv run python reset_db.py && uv run meritorder migrate && uv run meritorder --log-level warning serve --host 127.0.0.1 --port ${PORT}`,
    cwd: e2eDir,
    url: `http://127.0.0.1:${PORT}/readyz`,
    env: SERVER_ENV,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
