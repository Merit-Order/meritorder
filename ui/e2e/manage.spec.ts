import { allow, expect, expectLoaded, KEYS, open, signInAs, test } from "./fixtures";

test.describe("as an admin", () => {
  test.beforeEach(async ({ page }) => signInAs(page, "admin"));

  test("issue a key, use it, revoke it", async ({ page, request }) => {
    await open(page, "keys");
    await expectLoaded(page);
    await page.getByLabel("User").fill("carol@example.com");
    await page.getByLabel("Team").fill("eng");
    await page.getByLabel("Label").fill("e2e laptop");
    await page.getByRole("button", { name: "Issue key" }).click();
    const alert = page.getByText("Key for carol@example.com (eng, user)");
    await expect(alert).toBeVisible();
    const secret = (await page.locator("code").filter({ hasText: /^mo-/ }).first().textContent())!.trim();
    expect(secret).toMatch(/^mo-/);

    const use = () => request.post("/v1/chat/completions", { headers: { authorization: `Bearer ${secret}` },
      data: { model: "small", messages: [{ role: "user", content: "hi" }] } });
    expect((await use()).status()).toBe(200);

    const row = page.getByRole("row").filter({ hasText: "carol@example.com" });
    await expect(row.getByText("active")).toBeVisible();
    await row.getByRole("button", { name: "Revoke" }).click();
    await expect(page.getByText("Key revoked")).toBeVisible();
    await expect(row.getByText("revoked")).toBeVisible();
    expect((await use()).status()).toBe(401); // stops working at once on this gateway
  });

  test("change a team budget on Controls; it's saved as a config version and survives a reload", async ({ page }) => {
    await open(page, "controls");
    await expectLoaded(page);
    const row = page.getByRole("row").filter({ hasText: /^eng/ });
    const budget = row.getByRole("textbox").first(); // team budget / month
    await expect(budget).toHaveValue("$1,000");
    // What a failed attempt costs a person on the team: the organization's $1 until set.
    await expect(page.getByText("A failure costs a person")).toBeVisible();
    await expect(row.getByPlaceholder("org's $1.00")).toBeVisible();
    await expect(row.getByText(/^not counted/)).toHaveCount(0);  // eng routes by expected total cost, the default
    await budget.fill("1234");
    await page.getByPlaceholder("What changed and why (optional)").fill("e2e: raise eng budget");
    await page.getByRole("button", { name: "Save as new config version" }).click();
    await expect(page.getByText(/Saved as version \d+/)).toBeVisible();

    await page.reload();
    await expectLoaded(page);
    await expect(page.getByRole("row").filter({ hasText: /^eng/ }).getByRole("textbox").first()).toHaveValue("$1,234");

    await open(page, "config");
    await expectLoaded(page);
    await expect(page.getByText(/version \d+/).first()).toBeVisible(); // now running from the database
    await page.getByRole("button", { name: "History" }).click();
    await expect(page.getByText("e2e: raise eng budget")).toBeVisible();
  });

  test("saving Controls doesn't write back a safeguard another admin changed meanwhile", async ({ page, request }) => {
    const h = { authorization: `Bearer ${KEYS.admin}` };
    const original = (await (await request.get("/v1/admin/config/source", { headers: h })).json()).document;
    await open(page, "controls");
    await expectLoaded(page);
    // Another admin sets the per-request cost limit while this page is open.
    const theirs = { ...original, safeguards: { ...(original.safeguards ?? {}), max_request_usd: 7 } };
    expect((await request.put("/v1/admin/config", { headers: h, data: { document: theirs, comment: "e2e: another admin" } })).ok()).toBeTruthy();
    // This admin raises the organization budget only, and saves.
    await page.getByLabel("Organization budget per month").fill("12000");
    await page.getByRole("button", { name: "Save as new config version" }).click();
    await expect(page.getByText(/Saved as version \d+/)).toBeVisible();
    const saved = (await (await request.get("/v1/admin/config/source", { headers: h })).json()).document;
    expect(saved.safeguards.max_request_usd).toBe(7);
    expect(saved.safeguards.org_budget_usd_month).toBe(12000);
    expect((await request.put("/v1/admin/config", { headers: h, data: { document: original, comment: "e2e: put back" } })).ok()).toBeTruthy();
  });

  test("the config editor validates before saving", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "YAML" }).click();
    const editor = page.getByLabel("Configuration YAML");
    const yaml = await editor.inputValue();
    expect(yaml).toContain("baseline_model");

    // A valid change: the forms pick it up, and Validate says it's fine.
    await editor.fill(yaml.replace(/max_session_usd: 20(\.0)?/, "max_session_usd: 25"));
    await page.getByRole("button", { name: "Apply YAML to the forms" }).click();
    await expect(page.getByText("Forms updated from YAML")).toBeVisible();
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();

    // An invalid one: the baseline model doesn't exist.
    await editor.fill(yaml.replace(/baseline_model: \S+/, "baseline_model: no-such-model"));
    await page.getByRole("button", { name: "Apply YAML to the forms" }).click();
    // Applying checks it too, and says what's wrong.
    await expect(page.getByText(/baseline_model is 'no-such-model'/).first()).toBeVisible();
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText("Not valid yet")).toBeVisible();
    await expect(page.getByText(/no-such-model/).first()).toBeVisible();

    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0); // back to the running configuration
  });

  test("a save never undoes one someone else made after the page opened", async ({ page, request, problems }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "YAML" }).click();
    const editor = page.getByLabel("Configuration YAML");
    const yaml = await editor.inputValue();
    // Someone else saves meanwhile (a script, without base_version: saved whatever the latest is).
    const src = await (await request.get("/v1/admin/config/source", { headers: { authorization: `Bearer ${KEYS.admin}` } })).json();
    const other = await request.put("/v1/admin/config", { headers: { authorization: `Bearer ${KEYS.admin}` },
      data: { document: src.document, comment: "e2e: someone else's save" } });
    expect(other.status()).toBe(200);
    await editor.fill(yaml.replace(/max_session_usd: 20(\.0)?/, "max_session_usd: 26"));
    await page.getByRole("button", { name: "Apply YAML to the forms" }).click();
    await expect(page.getByText("Forms updated from YAML")).toBeVisible();
    await page.getByRole("button", { name: "Save and apply" }).click();
    await expect(page.getByText("Saved by someone else meanwhile")).toBeVisible();
    await expect(page.getByText(/Reload to see their change/)).toBeVisible();
    allow(problems, /status of 409/); // the browser logs the refused save
    await page.getByRole("button", { name: "Discard" }).click();
  });

  test("the level rules are edited on the Routing tab, and a pattern that can't work is refused", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Routing" }).click();
    await expect(page.getByText("How requests get a level", { exact: true })).toBeVisible();
    await expect(page.getByLabel("Tool count sets an agent's level", { exact: true })).toBeVisible();
    await expect(page.getByLabel("Simple wording", { exact: true })).toHaveValue(/write \(unit \)\?tests/);
    const hard = page.getByLabel("Hard wording", { exact: true });
    await expect(hard).toHaveValue(/^\\brace condition/); // the defaults, from the gateway
    await hard.fill("(\n\\bdeadlock");
    await expect(page.getByText(/^Unsaved: Routing/)).toBeVisible();
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText("Not valid yet")).toBeVisible();
    await expect(page.getByText(/isn't a valid regular expression/).first()).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
    await expect(hard).toHaveValue(/^\\brace condition/);
  });

  test("every config setting explains itself; the help never changes the setting", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);

    // Keyboard: tabbing to a "?" opens its explanation, and screen readers get it as the description.
    await page.getByRole("tab", { name: "Routing" }).click();
    const about = page.getByRole("button", { name: "About Priority" });
    await about.focus();
    await page.keyboard.press("Shift+Tab"); // arrive by keyboard, as a keyboard user would
    await page.keyboard.press("Tab");
    await expect(about).toBeFocused();
    await expect(page.getByRole("tooltip")).toContainText("a faster route may cost up to 20% more");
    await expect(about).toHaveAttribute("aria-describedby", /.+/);
    // The field keeps its plain name.
    await expect(page.getByLabel("Priority", { exact: true })).toBeVisible();

    // Mouse: hovering opens it. Clicking a "?" inside a switch's label doesn't flip the switch.
    await page.getByRole("tab", { name: "Teams" }).click();
    await page.getByRole("row").filter({ hasText: /^eng/ }).click();
    const cache = page.getByRole("switch", { name: "Response cache", exact: true });
    const before = await cache.isChecked();
    const help = page.getByRole("button", { name: "About Response cache" });
    await help.hover();
    await expect(page.getByRole("tooltip")).toContainText("exact repeat");
    await help.click();
    expect(await cache.isChecked()).toBe(before);
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);

    // A list with only exclusions allows nothing, and the form says so.
    const models = page.getByRole("dialog").getByRole("combobox", { name: "Models", exact: true });
    await models.fill("!claude-*");
    await models.press("Enter");
    await expect(page.getByText("Models has only exclusions")).toBeVisible();
    await page.getByRole("button", { name: "Done" }).click();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("routing by expected total cost: a person's cost of a failure, for the organization and a team, and escalations", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Routing" }).click();
    await page.getByRole("combobox", { name: "Choose models by" }).click();
    await page.getByRole("option", { name: /^Expected total cost/ }).click();
    await page.getByRole("textbox", { name: "A failed level-5 attempt costs a person", exact: true }).fill("2.5");
    await page.getByRole("button", { name: "About A failed level-5 attempt costs a person" }).hover();
    await expect(page.getByRole("tooltip")).toContainText("the default $1 is one minute at $60 an hour. A request puts a share of it at stake");
    await page.getByRole("textbox", { name: "…for team eng", exact: true }).fill("4");
    await page.getByRole("textbox", { name: "At most this many escalations per task", exact: true }).fill("2");
    await page.getByRole("tab", { name: "YAML" }).click();
    const yaml = page.getByLabel("Configuration YAML");
    await expect(yaml).not.toHaveValue(/model_choice:/);  // expected_cost is the default: nothing to write
    await expect(yaml).toHaveValue(/human_cost_usd: 2\.5/);
    await expect(yaml).toHaveValue(/human_cost_usd: 4/);  // eng's own
    await expect(yaml).toHaveValue(/max_escalations: 2/);
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("routing by cost per success is set for the organization and for a team", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Routing" }).click();
    await page.getByRole("combobox", { name: "Choose models by" }).click();
    await page.getByRole("option", { name: /^Cost per success/ }).click();
    await page.getByRole("textbox", { name: "Never below this chance of success (0–1)", exact: true }).fill("0.7");
    await page.getByRole("button", { name: "About Never below this chance of success (0–1)" }).hover();
    await expect(page.getByRole("tooltip")).toContainText("A model exactly at the level has the chance a 5 has");

    await page.getByRole("tab", { name: "Teams" }).click();
    await page.getByRole("row").filter({ hasText: /^eng/ }).click();
    const drawer = page.getByRole("dialog");
    await drawer.getByRole("combobox", { name: "Choose models by" }).click();
    await page.getByRole("option", { name: /^Threshold/ }).click();
    await drawer.getByRole("textbox", { name: "Never below this chance of success (0–1)", exact: true }).fill("0.5");
    await page.getByRole("button", { name: "Done" }).click();

    await page.getByRole("tab", { name: "YAML" }).click();
    const yaml = page.getByLabel("Configuration YAML");
    await expect(yaml).toHaveValue(/model_choice: cost_per_success/);
    await expect(yaml).toHaveValue(/min_success_chance: 0\.7/);
    await expect(yaml).toHaveValue(/model_choice: threshold/);  // eng's own
    await expect(yaml).toHaveValue(/min_success_chance: 0\.5/);
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("model-based compression is turned on per team and needs the compressor service", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Teams" }).click();
    await page.getByRole("row").filter({ hasText: /^eng/ }).click();
    const drawer = page.getByRole("dialog");
    await drawer.getByRole("switch", { name: "Model-based compression (LLMLingua-2)", exact: true }).click();
    await expect(drawer.getByText("No compressor service is set")).toBeVisible();
    await drawer.getByRole("textbox", { name: "Keep this share of the prose (0–1)", exact: true }).fill("0.4");
    await drawer.getByRole("textbox", { name: "Wait for the model at most (ms)", exact: true }).fill("2500");
    await page.getByRole("button", { name: "About Never for these tools" }).hover();
    await expect(page.getByRole("tooltip")).toContainText("must see it exactly");
    await page.getByRole("button", { name: "Done" }).click();
    await expect(page.getByRole("row").filter({ hasText: /^eng/ }).getByText("model compression")).toBeVisible();

    // Without a service it doesn't validate; with one it does.
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/compressor\.url isn.t set/).first()).toBeVisible();
    await page.getByRole("textbox", { name: "URL", exact: true }).fill("http://compressor:8090");
    await page.getByRole("tab", { name: "YAML" }).click();
    const yaml = page.getByLabel("Configuration YAML");
    await expect(yaml).toHaveValue(/url: http:\/\/compressor:8090/);
    await expect(yaml).toHaveValue(/rate: 0\.4/);
    await expect(yaml).toHaveValue(/deadline_ms: 2500/);
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("stored responses are turned on per team with a retention, and the console shows what each team holds", async ({ page }) => {
    // The seed chained two responses for alice on eng, which has stored responses on for 30 days.
    await open(page, "teams");
    const eng = page.locator(".mantine-Paper-root").filter({ hasText: /^eng/ });
    await expect(eng.getByText("on, kept 30 days after the last turn")).toBeVisible();
    await expect(eng.getByText(/^2 responses, .* sealed, for 1 person$/)).toBeVisible();
    await expect(page.getByText("remember the word aubergine")).toHaveCount(0); // sizes, never content

    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Teams" }).click();
    await expect(page.getByRole("row").filter({ hasText: /^eng/ }).getByText("30 days")).toBeVisible();
    await expect(page.getByText(/^key [0-9a-f]{16}$/)).toBeVisible(); // the key's public id, never the key
    await page.getByRole("row").filter({ hasText: /^default/ }).click();
    const drawer = page.getByRole("dialog");
    await drawer.getByRole("switch", { name: "Stored responses (OpenAI Responses API)", exact: true }).click();
    await expect(drawer.getByText(/^Required: how long the gateway keeps/)).toBeVisible();
    await page.getByRole("button", { name: "Done" }).click();
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/needs retention_days when enabled/).first()).toBeVisible();
    await page.getByRole("row").filter({ hasText: /^default/ }).click();
    await drawer.getByRole("textbox", { name: "Keep for (days after the last turn)", exact: true }).fill("14");
    await page.getByRole("button", { name: "Done" }).click();
    await page.getByRole("tab", { name: "YAML" }).click();
    await expect(page.getByLabel("Configuration YAML")).toHaveValue(/retention_days: 14/);
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("the learned router is set on the Routing tab", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Routing" }).click();
    await expect(page.getByText("Learned router", { exact: true })).toBeVisible();
    const mode = page.getByRole("combobox", { name: "Learned router mode" });
    await expect(mode).toHaveValue(/^Shadow/);
    await mode.click();
    await page.getByRole("option", { name: /^Route: use its level when sure/ }).click();
    await page.getByRole("textbox", { name: "Use its level when at least this sure (0–1)", exact: true }).fill("0.85");
    await page.getByRole("textbox", { name: "Retrain every (hours)", exact: true }).fill("24");
    await page.getByRole("button", { name: "About Learned router mode" }).hover();
    await expect(page.getByRole("tooltip")).toContainText("the rules decide");

    await page.getByRole("tab", { name: "YAML" }).click();
    const yaml = page.getByLabel("Configuration YAML");
    await expect(yaml).toHaveValue(/learned:\s+mode: route\s+min_confidence: 0\.85\s+retrain_hours: 24/);
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("judged labels are turned on per team, after the estimate, its warning and a cap", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Routing" }).click();
    await expect(page.getByText("Judged labels", { exact: true })).toBeVisible();
    await expect(page.getByText(/sends a sample of its prompts and answers to the stronger model and a judge/)).toBeVisible();
    await page.getByRole("button", { name: "Turn on judged labels for eng" }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByText("Sampled prompts and answers are sent to two more models")).toBeVisible();
    // The estimate comes from eng's recorded traffic (the seeded requests, in shadow mode), with the evals' warning.
    await expect(dialog.getByText(/^This costs real money: about \$[\d.,]+ a month$/)).toBeVisible();
    await expect(dialog.getByText(/real costs have run up to three times the estimate/)).toBeVisible();
    const turnOn = dialog.getByRole("button", { name: /^Turn on/ });
    await expect(turnOn).toBeDisabled(); // no cap (there's no default) and no judge (none of the default ones is configured here)
    await dialog.getByRole("textbox", { name: "Monthly spending cap", exact: true }).fill("2");
    await dialog.getByRole("combobox", { name: "Judges" }).click();
    await page.getByRole("option", { name: "coder-oss", exact: true }).click();
    await dialog.getByText("Sampled prompts and answers are sent to two more models").click(); // closes the dropdown
    await expect(dialog.getByText(/Spending cap: \$2\.00 a month/)).toBeVisible();
    await expect(turnOn).toBeEnabled();
    await turnOn.click();
    await expect(page.getByRole("button", { name: "Change judged labels for eng" })).toBeVisible();
    await expect(page.getByText("on, 1.0% sampled")).toBeVisible();

    await page.getByRole("tab", { name: "YAML" }).click();
    const yaml = page.getByLabel("Configuration YAML");
    await expect(yaml).toHaveValue(/judged_labels:\s+enabled: true\s+sample_rate: 0\.01\s+stronger: auto\s+stronger_level: 5\s+judges:\s+- coder-oss\s+max_usd_month: 2/);
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("evals: every task can be opened to see what it asks and how it's graded", async ({ page }) => {
    await open(page, "models/evals");
    await expectLoaded(page);
    await page.locator("label", { hasText: /^Tasks$/ }).click();
    await expect(page.getByText("32 tasks")).toBeVisible();
    await page.getByRole("textbox", { name: "Search tasks" }).fill("lisp");
    await page.getByRole("row", { name: "Task c5-lisp" }).click();
    const drawer = page.getByRole("dialog", { name: "Task c5-lisp" });
    await expect(drawer.getByText("How it's graded: 40 hidden tests, run in a sandbox")).toBeVisible();
    await drawer.getByRole("button", { name: "Hidden tests", exact: true }).click();
    await expect(drawer.getByText(/def test_/).first()).toBeVisible();
    await page.keyboard.press("Escape");

    await page.getByRole("textbox", { name: "Search tasks" }).fill("oncall");
    await page.getByRole("row", { name: "Task a5-oncall-handoff" }).click();
    const agent = page.getByRole("dialog", { name: "Task a5-oncall-handoff" });
    await expect(agent.getByText("Files it can reach through the tools (", { exact: false })).toBeVisible();
    await expect(agent.getByText("The grader's data (hidden tests, expected answer)")).toBeVisible();
    await expect(agent.getByRole("cell", { name: "write_file" })).toBeVisible();
  });

  test("evals: estimate, start and cancel a run; see a finished run's scores and apply them", async ({ page, request }) => {
    const h = { authorization: `Bearer ${KEYS.admin}` };
    await open(page, "models/evals");
    await expectLoaded(page);
    await expect(page.getByRole("tab", { name: "Evals" })).toHaveAttribute("aria-selected", "true");

    // Start: pick a model and a slice of the starter suite; the estimate comes first.
    await page.getByRole("button", { name: "Run an eval…" }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("combobox", { name: "Models" }).click();
    await page.getByRole("option", { name: "small", exact: true }).click();
    await dialog.getByLabel("Only tasks matching").click(); // closes the dropdown
    const frontier = dialog.getByRole("checkbox", { name: /^frontier/ });
    if (await frontier.count()) await frontier.uncheck();
    await dialog.getByLabel("Only tasks matching").fill("g1-*");
    await expect(dialog.getByText(/^This run costs real money: about \$[\d.,]+ for 2 task runs$/)).toBeVisible();
    await expect(dialog.getByText(/real runs have cost up to three times the estimate/)).toBeVisible();
    await expect(dialog.getByLabel("Spending cap")).toHaveValue(/^\$1(\.00)?$/);  // 1.5x the estimate, at least a dollar
    await dialog.getByRole("button", { name: /^Start \(about/ }).click();
    await expect(page.getByText("Eval queued")).toBeVisible();
    // No worker runs in these tests, so it waits in the queue: cancel it.
    const queued = page.getByRole("row").filter({ has: page.getByText("queued", { exact: true }) }).first();
    await queued.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByRole("row").filter({ has: page.getByText("cancelled", { exact: true }) }).first()).toBeVisible();

    // A finished run, recorded the way `meritorder eval` records one.
    const created = await request.post("/v1/admin/evals/runs", { headers: h, data: { models: ["small"], suite: "starter", only: ["g1-*"], executor: "cli", max_cost_usd: 1 } });
    expect(created.status()).toBe(201);
    const id = (await created.json()).id as string;
    // Two easy tasks (one cut off), and ten hard ones, eight passed: enough hard tasks for a measured score.
    const result = (task_id: string, passed: boolean, level = 1) => ({ model: "small", task_id, task: "general", level, passed,
      reasons: passed ? [] : [task_id === "g1-echo" ? "expected 'OK'" : "wrong answer"], cost: 0.0005, latency_ms: 5, model_ms: 4200, error: null, kind: "answer", turns: 1, detail: "",
      skipped: false, effort: "low", backend: "mock-openai", output_tokens: task_id === "g1-echo" ? 128000 : 12, truncated: task_id === "g1-echo",
      answer: passed ? "Lisbon" : "Thinking about what to echo back" });
    const results = [result("g1-capital", true), result("g1-echo", false), ...Array.from({ length: 10 }, (_, i) => result(`g5-hard-${i}`, i < 8, 5))];
    const scores = { small: { tasks: { general: { levels: { "1": 0.5, "5": 0.8 }, counts: { "1": 2, "5": 10 }, hard: { passed: 8, runs: 10 }, score: 5,
      low: 3.4, high: 5, pass_rate: 0.75 } }, pass_rate: 0.75, cost: 0.006 } };
    const done = await request.patch(`/v1/admin/evals/runs/${id}`, { headers: h,
      data: { status: "done", scores, results, progress: { total: 12, done: 12 }, cost: 0.006 } });
    expect(done.ok()).toBeTruthy();
    await page.reload();
    await expectLoaded(page);
    // In use 3.0; measured 5.0 (the only model measured is the best), with its 90% range.
    const row = page.getByRole("row").filter({ hasText: /^small/ }).first();
    await expect(row).toContainText("3.0 →");
    await expect(row).toContainText(/5\.0\s*\d\.\d–5\.0/);
    await expect(row).toContainText("4.2 s");  // what the work took
    await expect(row).toContainText("10,678");  // average output tokens: (128,000 + 11 x 12) / 12

    // Its details: the failure and why.
    await page.getByRole("row").filter({ has: page.getByText("done", { exact: true }) }).first().click();
    const drawer = page.getByRole("dialog");
    await expect(drawer.getByText("g1-echo")).toBeVisible();
    await expect(drawer.getByText("expected 'OK'")).toBeVisible();

    // Each result in full: what the model wrote, how it ran, and that it was cut off.
    await drawer.getByRole("row", { name: "g1-echo on small: details" }).click();
    const detail = page.getByRole("dialog", { name: "g1-echo · small" });
    await expect(detail.getByText("Thinking about what to echo back")).toBeVisible();
    await expect(detail.getByText("cut off")).toBeVisible();
    await expect(detail.getByText(/effort low · on mock-openai · 128,000 output tokens/)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(detail).toBeHidden();
    await page.keyboard.press("Escape");

    // Applying writes the measured score into the configuration as a new version.
    await page.getByRole("button", { name: "Apply scores…" }).first().click();
    await expect(page.getByRole("dialog")).toContainText("general");
    await page.getByRole("button", { name: "Apply 1 change" }).click();
    await expect(page.getByText(/^Saved as config version \d+$/)).toBeVisible();
    await expect(page.getByText(/applied as v\d+/).first()).toBeVisible();
  });

  test("GPU break-even: what a replica sized to each model needs to pay for itself, by quality level and by model", async ({ page }) => {
    await open(page, "models/breakeven");
    await expectLoaded(page);
    await expect(page.getByRole("tab", { name: "GPU break-even" })).toHaveAttribute("aria-selected", "true");
    // The first report starts fetching GPU prices (through SkyPilot, where it's installed) and model sizes in the
    // background, and the page asks again until they're in.
    await expect(page.getByText("Levels where a GPU can pay")).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText("Getting the latest GPU prices and model sizes")).toBeHidden({ timeout: 180_000 });
    await expect(page.getByRole("img", { name: /Traffic needed for one GPU replica to pay for itself/ })).toBeVisible();
    await expect(page.getByText("By open-weight model")).toBeVisible();
    await expect(page.getByRole("button", { name: "Get latest prices" })).toBeVisible();
    await expect(page.getByText(/^Best \w+ level you can host$/).first()).toBeVisible();
    // Switching to on-demand prices recalculates.
    await expect(page.getByText("spot", { exact: true })).toBeVisible();
    await page.getByLabel("Spot prices").click();
    await expect(page.getByText("on-demand", { exact: true })).toBeVisible({ timeout: 30_000 });
    // Serving the weights in 4 bits: what fits changes, and it warns that quality scores are for the published weights.
    await page.getByText("4-bit", { exact: true }).click();
    await expect(page.getByText("Serving in 4 bits")).toBeVisible();
    await expect(page.getByText(/Quality scores are for the weights as published/)).toBeVisible();
  });

  test("People lists everyone who signed in or sent traffic", async ({ page }) => {
    await open(page, "people");
    await expectLoaded(page);
    await expect(page.getByText("People", { exact: true }).first()).toBeVisible();
  });

  test("Google Groups are switched on and mapped on Config → Access", async ({ page }) => {
    await open(page, "config");
    await expectLoaded(page);
    await page.getByRole("tab", { name: "Access" }).click();
    await page.locator("label", { hasText: /^Enabled$/ }).click(); // single sign-on (the browser tests' config has none)
    await page.getByRole("textbox", { name: "Issuer", exact: true }).fill("https://accounts.google.com");
    await page.getByRole("textbox", { name: "Client ID", exact: true }).fill("123456789012-abc.apps.googleusercontent.com");
    await page.getByRole("textbox", { name: "Google Workspace domains", exact: true }).fill("example.com");
    await page.locator("label", { hasText: /^Read Google Groups$/ }).click();
    await expect(page.getByRole("textbox", { name: "Key in environment variable", exact: true })).toHaveValue("MERITORDER_GOOGLE_GROUPS_CREDENTIALS");
    await page.getByRole("button", { name: "About Keep last known groups for (hours)" }).hover();
    await expect(page.getByRole("tooltip")).toContainText("the email and domain rules alone");
    const groupKey = "Entra group object ID, or Google group address";
    await page.getByRole("button", { name: `Add ${groupKey}` }).first().click();
    await page.getByRole("textbox", { name: groupKey, exact: true }).first().fill("eng@example.com");

    await page.getByRole("tab", { name: "YAML" }).click();
    const yaml = page.getByLabel("Configuration YAML");
    await expect(yaml).toHaveValue(/google_groups:\s+credentials_env: MERITORDER_GOOGLE_GROUPS_CREDENTIALS/);
    await expect(yaml).toHaveValue(/eng@example\.com: /);
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(page.getByText(/^Valid: \d+ chat models/)).toBeVisible();
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText(/^Unsaved:/)).toHaveCount(0);
  });

  test("the People page shows each person's Google groups and which one set their team", async ({ page }) => {
    // The browser tests can't sign in with Google, so the people list gets one Google person added on its way to the page.
    await page.route("**/v1/admin/users", async (route) => {
      const real = await route.fetch();
      const body = await real.json();
      const at = new Date(Date.now() - 120_000).toISOString();
      body.users.push({ id: "104857600000000000001", email: "grace@example.com", display_name: "Grace Example", team: "eng", role: "admin",
        managed: "idp", disabled: false, active_keys: 1, last_seen: at, spend_month: 0, requests_month: 0, budget_usd_month: null,
        default_budget_usd_month: 100,
        google: { groups: [{ email: "all@example.com", name: "All", direct: true, mapped: false },
          { email: "eng@example.com", name: "Eng", direct: true, mapped: true },
          { email: "gateway-admins@example.com", name: "Gateway admins", direct: false, mapped: true }], hidden: 0,
          using: "last_known", error: "couldn't reach the Cloud Identity API (ConnectError)", fetched_at: at, checked_at: new Date().toISOString(),
          failing_since: at, team: "eng", role: "admin", team_from: { by: "group", key: "eng@example.com" },
          role_from: { by: "group", key: "gateway-admins@example.com" } } });
      await route.fulfill({ response: real, json: { ...body, oidc: true, idp: "google", google_groups: true, google_groups_refresh_seconds: 60 } });
    });
    await open(page, "people");
    await expectLoaded(page);
    await expect(page.getByRole("columnheader", { name: "Google groups" })).toBeVisible();
    await expect(page.getByText(/Their Google groups are read at sign-in and again every 60 seconds/)).toBeVisible();
    const row = page.getByRole("row").filter({ hasText: "grace@example.com" });
    await expect(row.getByText("eng@example.com", { exact: true })).toBeVisible();
    await expect(row.getByText("gateway-admins@example.com", { exact: true })).toBeVisible();
    await expect(row.getByText("nested", { exact: true })).toBeVisible(); // in through staff@, not directly
    await expect(row.getByText("+1 more", { exact: true })).toBeVisible();
    await expect(row.getByText("Team from group eng@example.com; admin from group gateway-admins@example.com")).toBeVisible();
    await expect(row.getByText(/^Google didn't answer at .+: using the groups read at /)).toBeVisible();
    await page.unroute("**/v1/admin/users");
  });

  test("give one person their own monthly limit; their key follows it", async ({ page, request }) => {
    const issued = await request.post("/v1/admin/keys", { headers: { authorization: `Bearer ${KEYS.admin}` },
      data: { user: "dave@example.com", team: "eng" } });
    const secret = (await issued.json()).api_key as string;
    const use = () => request.post("/v1/chat/completions", { headers: { authorization: `Bearer ${secret}` },
      data: { model: "small", messages: [{ role: "user", content: "hi" }] } });

    await open(page, "people");
    await expectLoaded(page);
    const limit = page.getByLabel("Monthly limit for dave@example.com");
    await expect(limit).toHaveAttribute("placeholder", "default $100"); // eng's per-person default
    await limit.fill("0");
    await limit.press("Enter");
    await expect(page.getByText("Limit set to $0: their requests are refused.")).toBeVisible();
    expect((await use()).status()).toBe(429);

    await limit.fill("25");
    await limit.press("Enter");
    await expect(page.getByText("Monthly limit set to $25.00.")).toBeVisible();
    expect((await use()).status()).toBe(200);
    await page.reload();
    await expect(page.getByLabel("Monthly limit for dave@example.com")).toHaveValue("$25");

    await page.getByLabel("Monthly limit for dave@example.com").fill("");
    await page.getByLabel("Monthly limit for dave@example.com").press("Enter");
    await expect(page.getByText("Own limit removed: the default applies.")).toBeVisible();
  });
});
