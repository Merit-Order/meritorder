/** The console demo (docs/07-deployment.md#the-console-demo): the real console with no gateway behind it. Only the demo
 * build loads this (main.tsx, behind __DEMO__): it sets the clock to when the sample data was recorded, answers the
 * console's calls to the gateway in the browser (server.ts), and starts the sample traffic. */
import { startClock } from "./clock";
import data from "./data.json";

export async function install(): Promise<void> {
  startClock(data.now);
  const { answer, startTraffic } = await import("./server");  // after the clock: its answers are dated by it
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(input instanceof Request ? input.url : String(input), location.href);
    if (url.origin !== location.origin || !(url.pathname.startsWith("/v1/") || url.pathname === "/about")) return realFetch(input, init);
    const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
    let body: unknown = null;
    if (typeof init?.body === "string") {
      try { body = JSON.parse(init.body); } catch { body = init.body; }
    }
    await new Promise((r) => setTimeout(r, 40 + Math.random() * 160));  // a gateway takes a moment
    const a = answer(method, url.pathname, url.searchParams, body);
    const text = typeof a.body === "string" ? a.body : JSON.stringify(a.body);
    return new Response(text, { status: a.status, statusText: a.status < 400 ? "OK" : "Not Found",
      headers: { "content-type": a.type ?? "application/json" } });
  };
  startTraffic();
}
