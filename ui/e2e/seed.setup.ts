import { execFileSync } from "node:child_process";
import { test as setup, expect } from "@playwright/test";
import { E2E_DIR, KEYS, SERVER_ENV } from "../playwright.config";

// Some traffic so the dashboards and reports have something to show, then one worker cycle so the
// month-long reports (which read the hourly rollup on Postgres) include it.
setup("seed traffic", async ({ request }) => {
  const chat = (key: string, body: object) =>
    request.post("/v1/chat/completions", { headers: { authorization: `Bearer ${key}` }, data: body });
  const msg = (content: string) => [{ role: "user", content }];

  for (const [key, body] of [
    [KEYS.user, { model: "auto", messages: msg("write a python function that parses ISO 8601 dates and add tests") }],
    [KEYS.user, { model: "auto", messages: msg("what is 2+2?") }],
    [KEYS.lead, { model: "small", messages: msg("summarize this paragraph in one line") }],
    [KEYS.admin, { model: "big", messages: msg("design a migration plan for our billing database") }],
    [KEYS.user, { model: "auto", stream: true, messages: msg("hello there") }],
  ] as const) {
    const r = await chat(key, body);
    expect(r.status(), await r.text()).toBe(200);
  }
  const anthropic = await request.post("/v1/messages", {
    headers: { "x-api-key": KEYS.user, "anthropic-version": "2023-06-01" },
    data: { model: "big", max_tokens: 100, messages: msg("review this diff") },
  });
  expect(anthropic.status()).toBe(200);

  // Two chained turns in the Responses API's stored mode (on for eng, where alice is), so Teams shows what eng holds.
  const responses = (data: object) => request.post("/v1/responses", { headers: { authorization: `Bearer ${KEYS.user}` }, data });
  const first = await responses({ model: "small", input: "remember the word aubergine" });
  expect(first.status(), await first.text()).toBe(200);
  const second = await responses({ model: "small", input: "which word?", previous_response_id: (await first.json()).id });
  expect(second.status(), await second.text()).toBe(200);

  // Two batches on the mock backends' batch API: Anthropic's Message Batches, and OpenAI's (a file, then the batch).
  const batch = await request.post("/v1/messages/batches", {
    headers: { "x-api-key": KEYS.user, "anthropic-version": "2023-06-01" },
    data: { requests: [
      { custom_id: "first", params: { model: "big", max_tokens: 100, messages: msg("summarize the release notes") } },
      { custom_id: "second", params: { model: "auto", max_tokens: 100, messages: msg("what is 3+3?") } },
    ] },
  });
  expect(batch.status(), await batch.text()).toBe(200);
  const line = (id: string, content: string) =>
    JSON.stringify({ custom_id: id, method: "POST", url: "/v1/chat/completions", body: { model: "small", messages: msg(content) } });
  const file = await request.post("/v1/files", {
    headers: { authorization: `Bearer ${KEYS.lead}` },
    multipart: { purpose: "batch", file: { name: "in.jsonl", mimeType: "application/jsonl", buffer: Buffer.from(`${line("a", "hi")}\n${line("b", "yo")}\n`) } },
  });
  expect(file.status(), await file.text()).toBe(200);
  const openaiBatch = await request.post("/v1/batches", {
    headers: { authorization: `Bearer ${KEYS.lead}` },
    data: { input_file_id: (await file.json()).id, endpoint: "/v1/chat/completions", completion_window: "24h" },
  });
  expect(openaiBatch.status(), await openaiBatch.text()).toBe(200);

  await new Promise((r) => setTimeout(r, 500)); // ledger rows are written just after each response
  execFileSync("uv", ["run", "meritorder", "--log-level", "warning", "worker", "--once"],
    { cwd: E2E_DIR, env: { ...process.env, ...SERVER_ENV }, stdio: "inherit" });
  // An hour of GPU time no request used, as the controller would accrue it: the console counts it in spend.
  execFileSync("uv", ["run", "python", "seed_gpu.py"], { cwd: E2E_DIR, env: { ...process.env, ...SERVER_ENV }, stdio: "inherit" });
});
