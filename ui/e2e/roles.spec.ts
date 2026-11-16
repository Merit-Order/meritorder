import { expect, expectLoaded, open, signInAs, test } from "./fixtures";

const ADMIN_ONLY = ["Dispatch", "GPUs", "Controls", "Backends", "Models", "Teams", "People", "Keys", "Config"];

test("a regular user sees only their own usage and reports", async ({ page }) => {
  await signInAs(page, "user");
  await open(page, "");
  const header = page.locator("header");
  await expect(header.getByRole("button", { name: "Reports", exact: true })).toBeVisible();
  await expect(header.getByRole("button", { name: "My usage", exact: true })).toBeVisible();
  for (const label of ADMIN_ONLY) {
    await expect(header.getByRole("button", { name: label, exact: true })).toHaveCount(0);
  }
  await expectLoaded(page);
  // Other people's names never show up (lee and root also sent traffic).
  for (const other of ["lee", "root"]) await expect(page.getByText(other, { exact: true })).toHaveCount(0);
});

test("a regular user's Requests tab is their own, whatever the URL asks for", async ({ page }) => {
  await signInAs(page, "user");
  await open(page, "reports/requests?user=lee&team=default");
  await expectLoaded(page);
  await expect(page.getByLabel("Team", { exact: true })).toHaveCount(0);  // nothing to pick: they see only their own
  const panel = page.locator(".panel").filter({ has: page.getByRole("button", { name: "Next" }) });
  await expect(panel.getByText("alice", { exact: true }).first()).toBeVisible();
  for (const other of ["lee", "root"]) await expect(panel.getByText(other, { exact: true })).toHaveCount(0);
});

test("a regular user can't open admin pages by URL", async ({ page }) => {
  await signInAs(page, "user");
  for (const hash of ["config", "keys", "people", "controls", "gpus", "dispatch", "backends", "models"]) {
    await open(page, hash);
    // Unknown or forbidden pages fall back to the first page the role may see.
    await expect(page.getByText("Reports", { exact: true }).first()).toBeVisible();
    await expect(page.getByText("Configuration", { exact: true })).toHaveCount(0);
    await expect(page.getByText("API keys", { exact: true })).toHaveCount(0);
    await expectLoaded(page);
  }
});

test("a regular user's key is refused by admin APIs", async ({ request }) => {
  const { KEYS } = await import("./fixtures");
  for (const path of ["/v1/admin/config/source", "/v1/admin/keys", "/v1/admin/users", "/v1/admin/controls"]) {
    const r = await request.get(path, { headers: { authorization: `Bearer ${KEYS.user}` } });
    expect(r.status(), path).toBe(403);
  }
});

test("a team lead sees their team's pages but not the config or controls", async ({ page }) => {
  await signInAs(page, "lead");
  await open(page, "dispatch");
  const header = page.locator("header");
  for (const label of ["Dispatch", "Teams", "People", "Keys"]) {
    await expect(header.getByRole("button", { name: label, exact: true })).toBeVisible();
  }
  for (const label of ["Config", "Controls", "Models"]) {
    await expect(header.getByRole("button", { name: label, exact: true })).toHaveCount(0);
  }
  await open(page, "keys");
  await expect(page.getByText("Keys for eng")).toBeVisible();
  await expectLoaded(page);
  await open(page, "reports");
  await expect(page.getByRole("tab", { name: "Breakdown" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Bills" })).toHaveCount(0); // bills are for finance and admins
  // The learned router's comparison for their team, without the model versions (admins choose those).
  await page.getByRole("tab", { name: "Learned router" }).click();
  await expectLoaded(page);
  await expect(page.getByText("Shadow: compares, changes nothing")).toBeVisible();
  await expect(page.getByText("Model versions")).toHaveCount(0);
  // Whether each level is right for the work, for their team (recommendations are the admins').
  await page.getByRole("tab", { name: "Levels" }).click();
  await expectLoaded(page);
  await expect(page.getByText("Is each level right for the work?")).toBeVisible();
  await expect(page.getByText("Spend per day, with catalog and score changes")).toBeVisible();
});

test("everyone can see how requests get a level, and which models each level allows", async ({ page }) => {
  await signInAs(page, "user");
  await open(page, "reports");
  await page.getByRole("tab", { name: "Levels" }).click();
  await expectLoaded(page);
  await expect(page.getByText("Coding with simple wording and nothing harder")).toBeVisible();
  await expect(page.getByText(/How long the conversation is never raises the level/)).toBeVisible();
  await expect(page.getByText("the hardest: system design, FrontierSWE-hard work")).toBeVisible();
  await expect(page.getByText(/big \(5\)/).first()).toBeVisible();
  await expect(page.getByText("Is each level right for the work?")).toHaveCount(0); // outcomes are for team leads and admins
});
