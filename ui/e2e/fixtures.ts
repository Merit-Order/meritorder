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

/** Signed in as `who` without the sign-in form (that's tested on its own). */
export async function signInAs(page: Page, who: keyof typeof KEYS) {
  await page.addInitScript((key) => {
    try { localStorage.setItem("meritorder.key", key); } catch { /* ignore */ }
  }, KEYS[who]);
}

export async function open(page: Page, hash: string) {
  await page.goto(`/ui/#${hash}`);
}

/** No "Couldn't load this" alert on the page (Loading shows one when an API call fails). */
export async function expectLoaded(page: Page) {
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("Couldn't load this")).toHaveCount(0);
}
