import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import license from "rollup-plugin-license";

const HEADER = [
  "Third-party software in the Merit Order console",
  "",
  "The console's bundle (its JavaScript and its fonts) includes the packages below. Each keeps its own license, reproduced here with its",
  "copyright notice. Merit Order itself is under the GNU AGPL-3.0 (LICENSE in its source).",
].join("\n");

// The self-hosted fonts (Geist and Geist Mono, SIL OFL 1.1) come in through CSS, which rollup-plugin-license doesn't see,
// so their notices are added here in the same format.
const FONTS = ["@fontsource-variable/geist", "@fontsource-variable/geist-mono"];
const fontNotices = () => FONTS.map((name) => {
  const dir = new URL(`./node_modules/${name}/`, import.meta.url);
  const pkg = JSON.parse(readFileSync(new URL("package.json", dir), "utf8"));
  const repo = typeof pkg.repository === "string" ? pkg.repository : pkg.repository?.url ?? "";
  return [`Name: ${pkg.name}`, `Version: ${pkg.version}`, `License: ${pkg.license}`, "Private: false", `Description: ${pkg.description ?? ""}`,
    `Repository: ${repo}`, `Homepage: ${pkg.homepage ?? ""}`, "License Text:", "===", "", readFileSync(new URL("LICENSE", dir), "utf8").trim()].join("\n");
});

/** The demo's 404.html: the static host (CloudFront, GitHub Pages) serves it for any path it doesn't have, so a link to
 * /dispatch (or a refresh on one) lands on the console's page of that name (#dispatch) instead of an error. Its script is
 * a file of its own, not inline, so the site's Content-Security-Policy can allow scripts from 'self' only. */
function demoFallback(base: string): Plugin {
  return {
    name: "meritorder-demo-404",
    generateBundle() {
      const script = `var b=${JSON.stringify(base)},p=location.pathname;if(p.indexOf(b)===0)p=p.slice(b.length);`
        + `location.replace(b+(p?"#"+p.replace(/\\/+$/,""):"")+location.hash);\n`;
      this.emitFile({ type: "asset", fileName: "404.js", source: script });
      this.emitFile({ type: "asset", fileName: "404.html",
        source: `<!doctype html><meta charset="utf-8"><title>Merit Order demo</title><script src="${base}404.js"></script>\n` });
      this.emitFile({ type: "asset", fileName: ".nojekyll", source: "" });
    },
    transformIndexHtml: (html) => html.replace("<title>Merit Order</title>", "<title>Merit Order demo</title>\n    "
      + '<meta name="description" content="The Merit Order console with sample data: an LLM gateway that routes each request '
      + 'to the cheapest capacity good enough for it." />'),
  };
}

// The Content-Security-Policy dev.meritorder.dev sends (deploy/aws/demo-site.yaml, which a test keeps the same). `vite
// preview --mode demo` sends it too, so the demo's smoke test (npm run e2e:demo) fails on anything it would block.
const DEMO_CSP = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
  + "font-src 'self'; connect-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; "
  + "frame-ancestors 'none'; object-src 'none'; upgrade-insecure-requests";

// The gateway build goes into the Python package, which serves it at /ui. The minified bundle drops the packages' license
// comments, so rollup-plugin-license writes every bundled package's license and NOTICE text to third-party-notices.txt
// beside it (served at /ui/third-party-notices.txt, linked from the console's footer).
//
// `--mode demo` builds the static demo instead (docs/07-deployment.md#the-console-demo): the console with sample data and
// no gateway, into ui/dist-demo, served from DEMO_BASE (default /, as on its own domain; /meritorder/ for a project page).
export default defineConfig(({ mode }) => {
  const demo = mode === "demo";
  const outDir = fileURLToPath(new URL(demo ? "./dist-demo" : "../src/meritorder/ui_dist", import.meta.url));
  const gateway = process.env.MERITORDER_URL ?? "http://localhost:8080";  // `npm run dev` sends API calls there
  const base = demo ? (process.env.DEMO_BASE ?? "/").replace(/\/?$/, "/") : "/ui/";
  return {
    base,
    define: { __DEMO__: JSON.stringify(demo) },
    plugins: [
      react(),
      license({
        thirdParty: {
          includePrivate: false,
          output: {
            file: `${outDir}/third-party-notices.txt`,
            template: (deps) => `${HEADER}\n\n===\n\n${[...deps.map((d) => d.text()), ...fontNotices()].join("\n\n---\n\n")}\n`,
          },
        },
      }),
      ...(demo ? [demoFallback(base)] : []),
    ],
    build: { outDir, emptyOutDir: true },
    preview: { port: 4174, strictPort: true, headers: demo ? { "Content-Security-Policy": DEMO_CSP } : undefined },
    server: { proxy: demo ? undefined : { "/v1": gateway, "/auth": gateway, "/about": gateway } },
  };
});
