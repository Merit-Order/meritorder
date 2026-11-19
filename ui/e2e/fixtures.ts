import { test as base, expect, type Page } from "@playwright/test";
import { KEYS } from "../playwright.config";

export { expect, KEYS };

/** Every test fails on an uncaught exception, a console error, or a page that couldn't load its data. */
export const test = base.extend<{ problems: string[] }>({
  problems: [async ({ page }, use) => {
    const problems: string[] = [];
    page.on("pageerror", (e) => problems.push(`uncaught: ${e.message}`));
    page.on("console", (msg) => {
      if (msg.type() === "error") problems.push(`console: ${msg.text()} (${msg.location().url})`);
    });
    await use(problems);
    expect(problems, "console errors or uncaught exceptions").toEqual([]);
  }, { auto: true }],
});

/** Drop console errors a test expects (e.g. the 401 from a wrong key); fails if none matched. */
export function allow(problems: string[], pattern: RegExp) {
  const before = problems.length;
  for (let i = problems.length - 1; i >= 0; i--) if (pattern.test(problems[i])) problems.splice(i, 1);
  expect(before - problems.length, `expected a console error matching ${pattern}`).toBeGreaterThan(0);
}

/** Signed in as `who` without the sign-in form (that's tested on its own): the console's own call, which opens a session
 * tied to the key. The page's requests share the browser's cookies. */
export async function signInAs(page: Page, who: keyof typeof KEYS) {
  const r = await page.request.post("/auth/login", { data: { api_key: KEYS[who] }, headers: { "x-meritorder-console": "1" } });
  expect(r.ok(), await r.text()).toBeTruthy();
}

/** The sign-in page's key form (it opens on email and password). */
export async function signInWithKey(page: Page, key: string) {
  await page.getByRole("button", { name: "Sign in with an API key instead" }).click();
  await page.getByLabel("API key").fill(key);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
}

export async function open(page: Page, hash: string) {
  await page.goto(`/ui/#${hash}`);
}

/** No "Couldn't load this" alert on the page (Loading shows one when an API call fails). */
export async function expectLoaded(page: Page) {
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("Couldn't load this")).toHaveCount(0);
}
