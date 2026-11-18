"""Measure model-based compression (LLMLingua-2) on real tool outputs (docs/04-caching-and-sessions.md).

Start the service (the ``compressor`` extra), then point this at it:

    meritorder compressor --port 8090 --device cpu
    uv run python scripts/bench_compressor.py --url http://127.0.0.1:8090 --pid <the service's pid> \\
        [--rates 0.33,0.5,0.7] [--pytest-log FILE] [--out results.json]

It builds tool outputs from this repository and the machine (pytest output, git log and diff, file reads, listings,
JSON, web pages as text, search results, prose, stack traces), with the repository's path, the home directory and
commit authors replaced. Each goes through the gateway's own guard and client (meritorder.compressor), and the script
reports, per kind of output:

* how much the model removed at each rate, of the whole result and of its prose;
* how long the service took (p50, p95) at 4k, 16k and 48k characters, and how long the gateway's guard took;
* whether every identifier, path, number, quoted string and error line came through byte for byte, found with
  patterns of its own rather than the guard's;
* how many names, numbers and paths survived, and how many sentences kept all of theirs: a proxy for whether an
  agent could still answer a question about the result. Only a proxy: it doesn't show that agents still succeed;
* the service's memory (RSS), with --pid.

Nothing here calls a provider: only the compressor service you started.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import glob
import html.parser
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import httpx

from meritorder.compression import ModelCompressionConfig
from meritorder.compressor import CompressorConfig, ModelCompressor, apply_cuts, cuts_for, segments
from meritorder.state import MemoryState
from meritorder.tokens import text_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
SIZES = (4_000, 16_000, 48_000)
WINDOWS = {4_000: 20, 16_000: 12, 48_000: 6}  # samples per kind and size for the latency percentiles


# ------------------------------------------------------------------ the corpus


def run(cmd: list[str], cwd: str = ROOT, check: bool = False) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600, env={**os.environ, "NO_COLOR": "1"})
    if check and p.returncode:
        raise SystemExit(f"{' '.join(cmd)} failed: {p.stderr[:500]}")
    return p.stdout + p.stderr


def clean(text: str) -> str:
    """Nothing private: the repository's path, the home directory and commit authors are replaced."""
    text = text.replace(ROOT, "/work/meritorder").replace(HOME, "/home/user")
    text = re.sub(r"^(Author|Commit):(\s+).*$", r"\1:\2A Developer <dev@example.com>", text, flags=re.MULTILINE)
    return re.sub(r"\b" + re.escape(os.path.basename(HOME)) + r"\b", "user", text)


class _Text(html.parser.HTMLParser):
    """An HTML page as a fetch tool returns it: text with headings, list items and paragraphs on lines of their own."""

    BLOCK = {"p", "div", "section", "article", "header", "footer", "table", "tr", "pre", "blockquote", "figure", "br", "hr"}

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "svg", "nav"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3", "h4"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "svg", "nav"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK or tag in ("h1", "h2", "h3", "h4", "li"):
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data if self.out and not self.out[-1].endswith("\n") else data.lstrip())

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", "".join(self.out))).strip() + "\n"


def html_text(path: str) -> str:
    p = _Text()
    p.feed(open(path, encoding="utf-8").read())
    return p.text()


def web_search(paths: list[str]) -> str:
    """Search results the way web-search tools list them: title, address, snippet."""
    out, n = [], 0
    for path in paths:
        doc = os.path.basename(path)
        body = open(path, encoding="utf-8").read()
        for m in re.finditer(r"^#{2,3} (.+)\n+((?:(?!#).*\n){1,8})", body, re.MULTILINE):
            title = m.group(1).strip()
            anchor = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            snippet = " ".join(line.strip() for line in m.group(2).splitlines() if line.strip())[:280]
            n += 1
            out.append(f"{n}. {title} - Merit Order docs\n   https://github.com/therick/meritorder/blob/main/docs/{doc}#{anchor}\n"
                       f"   {snippet}...\n")
    return "\n".join(out)


def json_responses() -> str:
    """JSON API responses: the test fixtures' provider replies, pretty-printed like `curl | jq`, then the price list."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_billing
    import test_gemini_upstream
    import test_google_groups
    import test_http_backend
    import test_live_prices
    import test_oidc_google

    fixtures = [test_http_backend.OPENAI_OK, test_http_backend.ANTHROPIC_OK, test_gemini_upstream.CALL_REPLY,
                test_gemini_upstream.TEXT_REPLY, test_google_groups.DIRECTORY, test_oidc_google.DISCOVERY,
                test_live_prices.OPENROUTER, test_live_prices.LITELLM, test_billing.RAW]
    prices = json.load(open(os.path.join(ROOT, "config", "prices.json")))
    return json.dumps({"fixtures": fixtures, "prices": prices}, indent=2, default=str) + "\n"


def stack_traces() -> str:
    """Python tracebacks (a failing pytest run, a refused connection, a config error) and a Node one, as a CI log
    with several failures would show them."""
    with tempfile.TemporaryDirectory() as d:
        test = os.path.join(d, "test_failing.py")
        open(test, "w").write(
            "import json, httpx, pytest\n"
            "from meritorder.config import build_config\n\n"
            "def parse(body):\n    return json.loads(body)['choices'][0]['message']\n\n"
            "def test_reply_shape():\n    assert parse('{\"choices\": []}')\n\n"
            "def test_config():\n    build_config({'backends': {'x': {'kind': 'nope'}}, 'models': 3})\n\n"
            "def test_upstream():\n    httpx.get('http://127.0.0.1:9/v1/models', timeout=1)\n\n"
            "@pytest.mark.parametrize('n', [1, 2, 3])\ndef test_ratio(n):\n    assert 10 / (n - 2) > 0\n")
        out = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--tb=long", "-q", test], cwd=d)
        if shutil.which("node"):
            out += run(["node", "-e", "function load(p){return JSON.parse(p)} function main(){load('{bad json')} main()"], cwd=d)
        out += run([sys.executable, "-c", "import httpx; httpx.get('http://127.0.0.1:9/', timeout=1)"], cwd=d)
    return out


EDGE = [
    "To install the service, run pip install 'meritorder[compressor]' and then start it with --device cpu, or with -d cuda on a "
    "machine that has a GPU. The flag --batch-size sets how many chunks the model takes at once, and the setting called "
    "\u201cdeadline\u201d decides how long a gateway waits. The value 'off' turns it off, and the word \"never\" stays as it is.",
    "The service listens on localhost:8090 by default. The X-Session-Id header tells the gateway which session a request "
    "belongs to, and the Retry-After header says how long to wait. If the model's files can't be read you will see the message "
    "No such file or directory in the log, followed by a hint about where the cache is. Permission denied means the volume is "
    "mounted read-only.",
    "Set the environment variable MERITORDER_COMPRESSOR_URL to the address of the service, for example http://compressor:8090, "
    "and keep api.anthropic.com in the allow list. The team called docs-team uses claude-code and kimi-k3.1 most of the time, "
    "while the platform team prefers gpt-oss on their own hardware and runs llm-d on a small cluster in the basement.",
    "Our office in Z\u00fcrich and the one in S\u00e3o Paulo both use the same settings, and the caf\u00e9 downstairs has nothing to do "
    "with any of it. Mount C:\\Users\\Public\\models on Windows, or ~/models on a Mac, so the weights are not downloaded again "
    "after every restart of the container.",
    "The price went from $3.00 to \u20ac2.50 per million tokens on 13 November, a cut of 17%, and the p95 latency dropped from "
    "1.2 s to 850 ms over the same week. Write to ops@example.com if the key that starts with sk-test-abc stops working. The "
    "commit 3f2a9c1 fixed the problem, and version v0.2.2 is the one to use from now on.",
    "Emoji like \U0001f680 and text in other scripts, such as \u6771\u4eac or \u041c\u043e\u0441\u043a\u0432\u0430, should come "
    "through unchanged, and so should a ligature like \ufb01le or a full-width \uff21\uff22\uff23. Only the words the model "
    "drops may go missing; every character it keeps must be exactly the one that was there before.",
]


def edge_cases() -> str:
    """Prose with the things the guard must keep: flags, header names, kebab-case names, quotes of every kind, error
    messages without the word error, Windows paths, money, accents, emoji and other scripts. Wrapped like a page."""
    import textwrap

    return "\n\n".join(textwrap.fill(p, 100) for p in EDGE) + "\n"


def corpus(pytest_log: str | None) -> dict[str, str]:
    """Each kind of tool output, long enough for the biggest size (repeated with fresh content where it runs out)."""
    docs = sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
    pages = sorted(glob.glob(os.path.join(ROOT, "docs", "html", "*.html")))
    if pytest_log:
        pytest_out = open(pytest_log, encoding="utf-8").read()
    else:
        pytest_out = run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "tests/test_compression_model.py",
                          "tests/test_levels.py", "tests/test_tokens.py", "tests/test_affinity.py", "tests/test_router.py"])
    kinds = {
        "pytest output": pytest_out,
        "git log --stat": run(["git", "log", "--stat", "-n", "120"]),
        "git diff": run(["git", "diff", "HEAD~8", "HEAD", "--", "src", "docs", "tests"]),
        "file read: Python (cat)": "".join(open(os.path.join(ROOT, "src", "meritorder", f)).read()
                                           for f in ("proxy.py", "router.py", "compressor.py")),
        "file read: Markdown (cat)": "".join(open(p).read() for p in [os.path.join(ROOT, "README.md")] + docs[7:9]),
        "ls -R / find": run(["ls", "-R", "src"]) + run(["find", ".", "-path", "./.git", "-prune", "-o", "-path", "./ui/node_modules",
                                                         "-prune", "-o", "-path", "./.*", "-prune", "-o", "-print"])
        + run(["ls", "-la", "/usr/bin"]),
        "JSON API responses": json_responses(),
        "web page as text": "".join(html_text(p) for p in pages if not p.endswith("index.html")),
        "search results: web": web_search(docs),
        "search results: grep": run(["git", "grep", "-n", "-I", "-e", "compress", "-e", "deadline", "-e", "cache", "--", "src", "docs"]),
        "long prose (docs)": "".join(open(p).read() for p in (docs[9], docs[0])),
        "stack traces": stack_traces(),
        "guard edge cases (prose)": edge_cases(),
    }
    out = {}
    for name, text in kinds.items():
        text = clean(text)
        while len(text) < SIZES[-1] * 2 and text:
            text += "\n" + text  # a few kinds are short: a longer log of the same kind
        out[name] = text
    return out


def window(text: str, start: int, size: int) -> str:
    """``size`` characters of ``text`` from the line at or after ``start``, ending at a line break."""
    if start:
        start = text.find("\n", start) + 1
    piece = text[start:start + size]
    return piece[:piece.rfind("\n") + 1] or piece


def all_prose(texts: list[str]) -> str:
    """Only the prose the guard would send, run together: the worst case for latency (every character goes to the model)."""
    return "\n\n".join(t[s:e] for t in texts for s, e in segments(t))


# ------------------------------------------------------------------ what must survive, found independently of the guard

CHECKS = {
    "path or URL": re.compile(r"https?://\S+|(?:[A-Za-z]:\\|~/|\.{1,2}/|/)?(?:[\w.@+-]+/)+[\w.@+-]*"),
    "number": re.compile(r"\d+(?:[.,:_]\d+)*"),
    "identifier": re.compile(r"\b[A-Za-z_][A-Za-z0-9]*_\w*|\b[a-z]+[A-Z]\w*|\b[A-Z][a-z]+[A-Z]\w*|\b[A-Z][A-Z0-9_]{2,}\b"
                             r"|\b\w+(?:\.\w+)+\b|\b\w+\(\)|\b\w+::\w+"),
    "command-line flag": re.compile(r"(?<![\w-])--?[A-Za-z][\w-]*"),
    # Names like claude-code, X-Session-Id and gpt-oss, and English compounds like opt-in, alike.
    "hyphenated word": re.compile(r"\b\w+(?:-\w+)+\b"),
    "quoted string": re.compile("\"[^\"\\n]{1,200}\"|`[^`\\n]{1,200}`|(?<!\\w)'[^'\\n]{1,80}'(?!\\w)|\u201c[^\u201d\\n]{1,200}\u201d"),
}
ERROR_LINE = re.compile(r"(?i)\b(?:errors?|exceptions?|traceback|failed|failures?|fatal|panic|warnings?|denied|refused|cannot|unable"
                        r"|invalid|timed out|timeout|undefined|abort(?:ed)?|not found|no such)\b|[a-z](?:Error|Exception)\b")
WORD = re.compile(r"[^\W\d_]+")
MEANING = re.compile(r"\b(?:only|all|every|each|always|any|most|some|few|many|more|less|least|both|either|first|last|same)\b", re.I)
UNIT = re.compile(r"\d+(?:[.,]\d+)*%? ([^\W\d_]+)")  # "6 hours", "512 MB", "3 is": the word after a number
SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def names(text: str, lo: int, hi: int) -> list[tuple[int, int, str]]:
    """Names: capitalized words that don't start a sentence, line or clause (Zürich, Redis, Москва), and words in
    scripts without case (東京)."""
    out = []
    for m in WORD.finditer(text, lo, hi):
        w = m.group()
        before = text[lo:m.start()].rstrip()
        if not before or before[-1] in ".!?:;*#>-(\"'“|":
            continue
        if len(w) > 1 and (w[0].isupper() or (not w[0].islower() and not w[0].isascii())):
            out.append((m.start(), m.end(), w))
    return out


def spans(text: str, rx: re.Pattern, lo: int = 0, hi: int | None = None) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in rx.finditer(text, lo, len(text) if hi is None else hi)]


def cut_index(cuts: list[list[int]]):
    starts = [s for s, _ in cuts]

    def hit(s: int, e: int) -> bool:
        i = bisect.bisect_right(starts, e - 1) - 1  # the last cut starting before e
        return i >= 0 and cuts[i][1] > s
    return hit


def guard_check(text: str, cuts: list[list[int]], out: str) -> dict:
    """Every protected item found in ``text``: how many, and which were cut (by the cuts, and as a substring check)."""
    hit = cut_index(cuts)
    res = {}
    for name, rx in CHECKS.items():
        found = spans(text, rx)
        damaged = [w for s, e, w in found if hit(s, e)]
        missing = sorted({w for _, _, w in found if w not in out} - set(damaged))
        res[name] = {"checked": len(found), "damaged": damaged, "missing": missing}
    lines, pos = [], 0
    for line in text.split("\n"):
        if ERROR_LINE.search(line):
            lines.append((pos, pos + len(line), line))
        pos += len(line) + 1
    res["error line"] = {"checked": len(lines), "damaged": [w for s, e, w in lines if hit(s, e)], "missing": []}
    return res


def facts(text: str, segs: list[tuple[int, int]], cuts: list[list[int]]) -> dict:
    """In the prose sent to the model: names, numbers and paths (what a question would ask for), meaning words, and
    sentences that kept all their facts."""
    hit = cut_index(cuts)
    count = {k: [0, 0] for k in ("names", "numbers", "words after numbers", "paths", "meaning words",
                                 "sentences with facts kept whole")}
    lost: dict[str, dict[str, int]] = {"names": {}, "words after numbers": {}, "meaning words": {}}

    def see(kind: str, s: int, e: int, w: str) -> bool:
        ok = not hit(s, e)
        count[kind][0] += ok
        count[kind][1] += 1
        if not ok and kind in lost:
            lost[kind][w] = lost[kind].get(w, 0) + 1
        return ok

    for lo, hi in segs:
        for m in SENTENCE.finditer(text, lo, hi):
            s0, s1 = m.start(), m.end()
            items = [("names", *x) for x in names(text, s0, s1)] + [("numbers", *x) for x in spans(text, CHECKS["number"], s0, s1)]
            items += [("paths", *x) for x in spans(text, CHECKS["path or URL"], s0, s1)]
            if items:
                ok = all([see(*x) for x in items])
                count["sentences with facts kept whole"][0] += ok
                count["sentences with facts kept whole"][1] += 1
            for u in UNIT.finditer(text, s0, s1):  # mostly "is", "and": counted apart, not as a fact
                see("words after numbers", u.start(1), u.end(1), u.group(1))
        for s, e, w in spans(text, MEANING, lo, hi):
            see("meaning words", s, e, w.lower())
    return {**count, **{f"lost {k}": sorted(v.items(), key=lambda kv: -kv[1])[:12] for k, v in lost.items()}}


# ------------------------------------------------------------------ memory


class RSS(threading.Thread):
    """The service's resident memory, sampled every 50 ms (ps works on macOS and Linux)."""

    def __init__(self, pid: int | None) -> None:
        super().__init__(daemon=True)
        self.pid, self.peak, self.idle, self.stop = pid, 0, self.sample(pid), False

    @staticmethod
    def sample(pid: int | None) -> int:
        if not pid:
            return 0
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return int(out or 0) * 1024

    def run(self) -> None:
        while not self.stop:
            self.peak = max(self.peak, self.sample(self.pid))
            time.sleep(0.05)

    def reset(self) -> int:
        peak, self.peak = self.peak, 0
        return peak


# ------------------------------------------------------------------ measuring


def prose_change(text: str, segs: list[tuple[int, int]], cuts: list[list[int]]) -> tuple[float, float]:
    """Estimated tokens in the prose sent to the model, before and after its cuts."""
    before = after = 0.0
    for s, e in segs:
        piece = text[s:e]
        before += text_tokens(piece)
        after += text_tokens(apply_cuts(piece, [[a - s, b - s] for a, b in cuts if s <= a and b <= e]))
    return before, after


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--pid", type=int, help="the service's process id, to sample its memory")
    ap.add_argument("--rates", default="0.33,0.5,0.7", help="share of the prose's tokens to keep; the first non-default is "
                    "measured for rates only (latency doesn't depend on it)")
    ap.add_argument("--pytest-log", help="a `pytest -v` log to use as pytest output (default: run a few test files)")
    ap.add_argument("--latency", action=argparse.BooleanOptionalAction, default=True, help="measure latency percentiles")
    ap.add_argument("--rates-check", action=argparse.BooleanOptionalAction, default=True,
                    help="measure rates and check the guard (they don't depend on the device: --no-rates-check on a second one)")
    ap.add_argument("--windows", type=float, default=1.0, help="scale the latency samples (1: 20, 12 and 6 per kind and size)")
    ap.add_argument("--out", help="write every number here as JSON")
    args = ap.parse_args()
    rates = [float(r) for r in args.rates.split(",")]
    default_rate = ModelCompressionConfig().rate

    health = httpx.get(f"{args.url}/healthz", timeout=10).json()
    cfg = CompressorConfig(url=args.url, model=health["model"])
    client = ModelCompressor(cfg, MemoryState())
    client.client.timeout = httpx.Timeout(600.0, connect=5.0)
    rss = RSS(args.pid)
    rss.start()
    print(f"service: {health['model']} on {health['device']} ({health['version']}); idle RSS {rss.idle / 2**20:.0f} MiB\n")

    kinds = corpus(args.pytest_log)
    await client._call(["A short warm-up sentence for the model, so the first timing isn't the one that loads it all."], 0.5)
    results: dict = {"service": health, "idle_rss_bytes": rss.idle, "kinds": {}, "latency": {}, "guard_ms": {}, "samples": []}

    # Rates, the guard check and the fact proxy: the start of each kind, at each size and rate.
    for kind, raw in kinds.items() if args.rates_check else ():
        per = results["kinds"][kind] = {}
        for size in SIZES:
            text = window(raw, 0, size)
            segs = segments(text)
            prose = sum(e - s for s, e in segs)
            row = per[str(size)] = {"chars": len(text), "prose_chars": prose, "segments": len(segs), "rates": {}}
            for rate in rates:
                if not segs:
                    row["rates"][str(rate)] = {"removed_chars": 0, "removed_tokens": 0.0, "tokens": text_tokens(text),
                                               "prose_tokens": 0.0, "removed_prose_tokens": 0.0, "ms": 0.0, "guard": guard_check(text, [], text)}
                    continue
                t0 = time.perf_counter()
                drops = await client._call([text[s:e] for s, e in segs], rate)
                ms = (time.perf_counter() - t0) * 1000
                cuts = cuts_for(text, segs, drops)
                out = apply_cuts(text, cuts)
                prose_tokens, prose_left = prose_change(text, segs, cuts)
                row["rates"][str(rate)] = {
                    "removed_chars": len(text) - len(out), "tokens": text_tokens(text), "removed_tokens": text_tokens(text) - text_tokens(out),
                    "prose_tokens": prose_tokens, "removed_prose_tokens": prose_tokens - prose_left,
                    "ms": ms, "guard": guard_check(text, cuts, out), "facts": facts(text, segs, cuts), "peak_rss_bytes": rss.reset(),
                    **({"sample": out[:1500]} if size == SIZES[0] and rate == default_rate else {}),
                }
        r = per[str(SIZES[1])]["rates"][str(default_rate)]
        print(f"{kind:<28} {per[str(SIZES[1])]['prose_chars'] / per[str(SIZES[1])]['chars']:>5.0%} prose  "
              f"removed {r['removed_tokens'] / max(1, r['tokens']):>5.1%} of tokens at rate {default_rate}  {r['ms']:>7.0f} ms")

    # The gateway's own time: the guard (first sight of a result) and applying stored cuts (every later turn).
    for size in SIZES:
        g, a = [], []
        for raw in kinds.values():
            text = window(raw, 0, size)
            t0 = time.perf_counter()
            segs = segments(text)
            g.append((time.perf_counter() - t0) * 1000)
            fake = [[[0, (e - s) // 3]] for s, e in segs]
            cuts = cuts_for(text, segs, fake)
            t0 = time.perf_counter()
            apply_cuts(text, cuts)
            a.append((time.perf_counter() - t0) * 1000)
        results["guard_ms"][str(size)] = {"segments_p50": statistics.median(g), "segments_max": max(g),
                                          "apply_p50": statistics.median(a), "apply_max": max(a)}

    # Latency at the default rate: several windows of each kind per size, and prose only (the worst case).
    if args.latency:
        prose_only = all_prose(list(kinds.values()))
        for size in SIZES:
            lat: dict[str, list[float]] = {}
            for kind, raw in [*kinds.items(), ("prose only (worst case)", prose_only)]:
                n = max(2, round(WINDOWS[size] * args.windows))
                step = max(1, (len(raw) - size) // n)
                for i in range(n):
                    text = window(raw, i * step, size)
                    segs = segments(text)
                    if not segs:
                        continue
                    t0 = time.perf_counter()
                    await client._call([text[s:e] for s, e in segs], default_rate)
                    lat.setdefault(kind, []).append((time.perf_counter() - t0) * 1000)
                    results["samples"].append({"kind": kind, "size": size, "segments": len(segs), "ms": lat[kind][-1],
                                               "prose_chars": sum(e - s for s, e in segs)})
            peak = rss.reset()
            results["latency"][str(size)] = {k: {"n": len(v), "p50": pct(v, 0.5), "p95": pct(v, 0.95), "max": max(v)} for k, v in lat.items()}
            results["latency"][str(size)]["_peak_rss_bytes"] = peak
            every = [x for k, v in lat.items() if k != "prose only (worst case)" for x in v]
            within = sum(x <= 1000 for x in every) / max(1, len(every))
            po = lat.get("prose only (worst case)", [0])
            print(f"\n{size:>6} chars: tool results p50 {pct(every, .5):6.0f} ms  p95 {pct(every, .95):6.0f} ms  "
                  f"({within:.0%} within 1 s);  prose only p50 {pct(po, .5):6.0f}  p95 {pct(po, .95):6.0f};  peak RSS {peak / 2**20:.0f} MiB")
    rss.stop = True
    results["peak_rss_bytes"] = max([results["latency"].get(str(s), {}).get("_peak_rss_bytes", 0) for s in SIZES]
                                    + [r.get("peak_rss_bytes", 0) for k in results["kinds"].values() for z in k.values() for r in z["rates"].values()])

    print("\nGuard check (items found by this script's own patterns; damaged = a cut touched it):")
    for kind, per in results["kinds"].items():
        bad = {}
        for z in per.values():
            for r in z["rates"].values():
                for cat, v in r["guard"].items():
                    if v["damaged"] or v["missing"]:
                        bad.setdefault(cat, set()).update(v["damaged"] + v["missing"])
        print(f"  {kind:<28} " + ("all intact" if not bad else "; ".join(f"{c}: {sorted(w)[:8]}" for c, w in bad.items())))
    await client.close()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
