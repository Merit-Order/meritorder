"""Model-based compression of tool results (LLMLingua-2), gateway side (docs/04-caching-and-sessions.md).

A model isn't deterministic across versions or hardware, so compressing the same tool result again on the next turn,
or on another gateway, could give different text and every later turn would miss the provider's prompt cache. So:

* Each tool result is decided once, the first time a gateway sees it: compressed, or left as it is. The decision is
  kept in the shared store (Redis) under a hash of the text, the compressor version and the settings, and every
  gateway applies the same one on every later turn. The first gateway to store a decision wins; a gateway that loses
  the race uses the winner's.
* What's stored is where the text was cut (character ranges to drop), never the text: the harness resends the
  original every turn, and cutting it the same way gives the same bytes.
* Only results new on this turn are compressed. One already sent in an earlier turn without a decision (compression
  was just turned on, or its entry expired) went upstream as it is, so it stays that way.
* A new result the service can't compress by the team's deadline, or at all, is stored as left alone.

The guard: only prose goes to the model. Code, file paths, identifiers, numbers, error and stack-trace lines, tables,
log lines, numbered file listings and JSON are kept as they are, and words with digits, paths or identifiers in them
are kept wherever the model would have dropped them. llmlingua's own way of keeping spans
(``structured_compress_prompt``) only works with the original LLMLingua, so the gateway keeps them itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, OrderedDict

import httpx
from pydantic import BaseModel, Field, field_validator

from .compression import ModelCompressionConfig
from .sessions import SESSION_TTL

log = logging.getLogger("meritorder.compressor")

# The guard, the request to the service and how cuts are applied. Part of every cache key: changing any of them
# (here, or in what the service is asked) must not reuse decisions made the old way.
# 2: search results, test ids, diffs, flags, kebab-case names, names, quantities and every kind of quote kept (docs/04).
VERSION = "llmlingua2-cuts-2"
FORCE_TOKENS = ["\n", ".", "?", "!"]  # llmlingua keeps these whatever the model says: line structure and sentence ends
MIN_SEGMENT = 300  # prose runs shorter than this stay as they are: too little for the model, and too little to gain
RETRY_SECONDS = 30.0  # after the service fails, don't call it again for this long (requests aren't slowed meanwhile)
SEEN = 20_000  # tool results whose prose ranges a gateway process remembers, so history isn't re-read every turn


class CompressorConfig(BaseModel):
    """The service model-based compression calls: ``meritorder compressor``, or anything that speaks its API
    (POST /v1/compress, docs/04-caching-and-sessions.md)."""

    url: str = ""  # e.g. http://compressor:8090; empty: model-based compression is off for everyone
    # The model the service must have loaded. Part of every cache key, so switching models never mixes decisions.
    model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    # How long a decision is kept after it was last used. At least as long as a session stays sticky (6 hours), so a
    # session never outlives the decisions its history was sent with.
    ttl_seconds: int = Field(default=SESSION_TTL, ge=SESSION_TTL)
    api_key_env: str | None = None  # an env var with a shared secret the service checks (MERITORDER_COMPRESSOR_KEY there)

    @field_validator("url", mode="before")
    @classmethod
    def _blank(cls, v):
        return (v or "").strip()


# ------------------------------------------------------------------ the guard

_FENCE = re.compile(r"^\s*(?:```|~~~)")
_LINE_NUMBER = re.compile(r"^\s*\d+(?:\t|→|:|\|)")  # cat -n and editors' file listings (Claude Code's Read)
# A first word with a path or a location in it: search results (grep -n, ripgrep, Claude Code's Grep: path:line:text),
# test ids (tests/test_x.py::test_y[...] PASSED), compiler and linter output (src/x.c:12:5: ...). An agent finds and
# edits lines by what these say, so they're quoted text, not prose.
_REFERENCE = re.compile(r"^\s*\S*(?:/|\\|::|\w:\d)")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")  # a unified diff's hunk; its lines start with " ", "+", "-" or "\"
# A result with a diff in it: there, lines starting like a hunk's are taken for one even without its @@ line (the rules
# keep the head and tail of a huge diff, so the tail starts in the middle of a hunk).
_DIFF = re.compile(r"^(?:diff --git |@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@|--- a/|\+\+\+ b/)", re.MULTILINE)
_STACK = re.compile(r"^\s*(?:at\s+\S|File \"|#\d+\s|Caused by:|\.\.\. \d+ more|\^)")
# Anywhere in a line, any case (TypeError, FAILED, NullPointerException): the line is an error message.
_ERROR = ("error", "exception", "traceback", "fail", "fatal", "panic", "warn", "denied", "refused", "not found", "undefined",
          "segmentation", "assert", "abort", "timed out", "timeout", "invalid", "cannot", "unable", "deprecat", "no such", "errno",
          "exit code", "exit status", "out of memory", "unauthorized", "forbidden")
_STRUCTURE = re.compile(r"^\s*(?:[#>{}\[\]<\"'|$%@/\\]|[\w.-]+:\s|[\w.-]+=|\[?\d{2,4}[-/:.]\d{2})")  # headings, quotes, YAML, logs
_CODE_WORD = re.compile(r"^\s*(?:def|class|import|from|return|function|const|let|var|fn|pub|impl|struct|package|func|public|private|"
                        r"static|SELECT|INSERT|UPDATE|DELETE|CREATE)\s")
# Counted on the line's UTF-8 bytes (bytes.translate is fast: this runs on every result the first time it's seen).
_SPACE_BYTES = b" \t\r\f\v"
_NOT_LETTER_BYTES = bytes(i for i in range(128) if not chr(i).isalpha())  # bytes of non-ASCII characters count as letters
_SYMBOL_BYTES = b"{}[]()<>;=|\\$`^~"
# Brackets count inside a word (configure(), list[str]), not around one: "(see below)" is prose. So do hyphens: names
# like claude-code, X-Session-Id and gpt-oss lose their hyphen or half of themselves otherwise (opt-in is kept too), and
# command-line flags (--device, -d).
_TECHNICAL = re.compile(r"[\d/\\_=@#$%<>{}`|~^*&+]|\w[(\[]|[)\]]\w|\(\)|::|->|\w\.\w|[a-z][A-Z]|[A-Z]{2,}|\w-\w|^\W?--?[^\W\d_]")
# Straight and typographic quotes; a single quote only around a whole word or more, so apostrophes aren't quotes.
_QUOTED = re.compile("\"[^\"\\n]{0,200}\"|`[^`\\n]{0,200}`|(?<!\\w)'[^'\\n]{1,200}'(?!\\w)|\u201c[^\u201d\\n]{0,200}\u201d|\u2018[^\u2019\\n]{1,200}\u2019")
_RUN = re.compile(r"\S+")
# Words whose loss turns a sentence around ("does not support" -> "does support") or changes how much it claims ("only
# admins can" -> "admins can", "the same model" -> "model"; LLMLingua-2 drops these three times in four): never cut.
_NEGATION = re.compile(r"(?:not|no|never|none|nor|neither|nothing|nobody|nowhere|without|cannot|unless|except|\w*n['’]t"
                       r"|only|all|every|each|always|any|most|some|few|many|more|less|least|both|either|first|last|same)\W*",
                       re.IGNORECASE)
_SENTENCE_END = frozenset(".!?:")  # a capitalized word after one of these (or a list marker) isn't taken for a name
_BULLETS = frozenset("-*+•")
_CLOSING = frozenset(".,;:!?)]}")


def _prose(line: str) -> bool:
    """A line of plain sentences, safe to shorten. Anything that might be code, data or an error is not."""
    if len(line) < 20 or line[:1] == "\t" or line.startswith("    ") or line.rstrip()[-1:] in "{};\\":
        return False
    if (line.count("|") >= 2 or _LINE_NUMBER.match(line) or _STACK.match(line) or _STRUCTURE.match(line) or _CODE_WORD.match(line)
            or _REFERENCE.match(line) or len(line.split(None, 4)) < 5):
        return False
    low = line.lower()
    if any(e in low for e in _ERROR):
        return False
    raw = line.encode("utf-8", "surrogatepass")
    chars = len(raw.translate(None, _SPACE_BYTES))
    letters = len(raw.translate(None, _NOT_LETTER_BYTES))
    symbols = len(raw) - len(raw.translate(None, _SYMBOL_BYTES))
    return letters >= 0.75 * chars and symbols <= 0.03 * chars


def _is_json(text: str) -> bool:
    s = text.strip()
    if s[:1] not in ("{", "["):
        return False
    try:
        json.loads(s)
    except RecursionError:
        return True  # nested deeper than the parser goes: data, not prose (and the error must not fail the request)
    except ValueError:
        return False
    return True


def segments(text: str) -> list[tuple[int, int]]:
    """The prose in a tool result, as (start, end) character ranges of whole lines: what may go to the model.
    Blank lines may sit inside a range (they're kept); fenced code blocks, diff hunks and everything ``_prose`` rejects
    never."""
    if _is_json(text):
        return []
    out: list[tuple[int, int]] = []
    start = end = -1  # the current run of prose lines
    pos = 0
    fenced = hunk = False
    diff = _DIFF.search(text) is not None
    for line in text.split("\n"):
        line_end = pos + len(line)
        if _HUNK.match(line) or ((hunk or diff) and line[:1] in ("", " ", "+", "-", "\\")):
            hunk, prose = True, False  # a diff's lines are the file's, as they are
        elif _FENCE.match(line):
            hunk, fenced, prose = False, not fenced, False
        else:
            hunk = False
            prose = not fenced and _prose(line)
        if prose:
            if start < 0:
                start = pos
            end = line_end
        elif line.strip() or fenced:
            if start >= 0 and end - start >= MIN_SEGMENT:
                out.append((start, end))
            start = end = -1
        pos = line_end + 1
    if start >= 0 and end - start >= MIN_SEGMENT:
        out.append((start, end))
    return out


def _name(word: str, before: str) -> bool:
    """A capitalized word that doesn't start a sentence or a list item (``before`` is the run before it): a name, like
    Redis, Zürich or both halves of Merit Order. LLMLingua-2 drops about one in ten of them."""
    if len(word) < 2 or not word[0].isupper() or not word[1].isalpha() or not before or before in _BULLETS:
        return False
    return before.rstrip(")]\"'”’*_")[-1:] not in _SENTENCE_END


def protected(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Ranges inside a prose range the model may never cut: quoted and backticked strings, negations and words that say
    how much (only, all, same), names, and every word with digits, a path, an identifier (snake_case, camelCase,
    CONSTANT, a.b, kebab-case, --flag), an operator or brackets in it."""
    spans = [(m.start(), m.end()) for m in _QUOTED.finditer(text, start, end)]
    before = ""
    for m in _RUN.finditer(text, start, end):
        run = m.group()
        word = run.lstrip("([\"'“‘")
        if _TECHNICAL.search(run) or _NEGATION.fullmatch(word) or _name(word, before):
            spans.append((m.start(), m.end()))
        before = run
    return sorted(spans)


def _subtract(cuts: list[tuple[int, int]], keep: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """``cuts`` without anything in ``keep`` (both sorted)."""
    out: list[tuple[int, int]] = []
    k = 0
    for s, e in cuts:
        while k < len(keep) and keep[k][1] <= s:
            k += 1
        j = k
        while s < e:
            if j < len(keep) and keep[j][0] < e:
                ks, ke = keep[j]
                if ks > s:
                    out.append((s, ks))
                s = max(s, ke)
                j += 1
            else:
                out.append((s, e))
                break
    return out


def _join(text: str, cuts) -> list[list[int]]:
    """Cuts separated only by whitespace, as one cut."""
    out: list[list[int]] = []
    for s, e in cuts:
        if out and not text[out[-1][1]:s].strip():
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def cuts_for(text: str, segs: list[tuple[int, int]], drops: list[list[list[int]]]) -> list[list[int]]:
    """Where to cut ``text``, from what the service said to drop in each prose range (offsets within the range), with
    the guard's protected words taken back out."""
    cuts: list[tuple[int, int]] = []
    for (start, end), drop in zip(segs, drops):
        within = sorted((start + max(0, s), start + min(e, end - start)) for s, e in drop if e > s)
        cuts += _subtract(within, protected(text, start, end))
    return _join(text, cuts)


def apply_cuts(text: str, cuts: list[list[int]]) -> str:
    """``text`` with the cut ranges removed. A cut keeps the line breaks it spanned (so line numbers still point at the
    same lines), and the words around a cut are left one space apart. Part of VERSION: stored cuts are applied with it
    on every later turn, so it must not change without a new version."""
    out: list[str] = []
    last = ""  # the last character written
    pos = 0
    n = len(text)
    for s, e in _join(text, cuts):
        piece = text[pos:s]
        breaks = text.count("\n", s, e)
        after = e
        while after < n and text[after] in " \t":
            after += 1
        following = text[after] if after < n else "\n"
        if breaks or following == "\n" or following in _CLOSING:
            piece = piece.rstrip(" \t") + "\n" * breaks  # no space before a line break or "." "," ")"
        elif (piece[-1:] or last) not in ("", " ", "\t", "\n"):
            piece += " "
        if piece:
            out.append(piece)
            last = piece[-1]
        pos = after
    out.append(text[pos:])
    return "".join(out)


def valid_cuts(cuts, n: int) -> bool:
    last = 0
    for c in cuts if isinstance(cuts, list) else [None]:
        if not (isinstance(c, list) and len(c) == 2 and all(isinstance(x, int) for x in c) and last <= c[0] < c[1] <= n):
            return False
        last = c[1]
    return True


def cache_key(team: str, model: str, rate: float, text: str) -> str:
    h = hashlib.sha256(f"{VERSION}\x00{team}\x00{model}\x00{rate:.4f}\x00".encode())
    h.update(text.encode("utf-8", "surrogatepass"))
    return f"cmp:{h.hexdigest()}"


# ------------------------------------------------------------------ the service client


class CompressorError(Exception):
    pass


class ModelCompressor:
    """Talks to the compressor service and keeps each tool result's decision in the shared store."""

    def __init__(self, cfg: CompressorConfig, state, metrics=None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.cfg = cfg
        self.state = state
        self.metrics = metrics
        key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        self.client = httpx.AsyncClient(base_url=cfg.url.rstrip("/"), transport=transport, timeout=httpx.Timeout(120.0, connect=5.0),
                                        headers={"authorization": f"Bearer {key}"} if key else {})
        self.retry_at = 0.0
        self.last_error: str | None = None
        self.seen: OrderedDict[str, list[tuple[int, int]]] = OrderedDict()  # cache key -> the text's prose ranges

    def segments(self, key: str, text: str) -> list[tuple[int, int]]:
        """``segments(text)``, remembered by the text's cache key: a session's history comes back on every turn."""
        segs = self.seen.get(key)
        if segs is None:
            segs = self.seen[key] = segments(text)
            if len(self.seen) > SEEN:
                self.seen.popitem(last=False)
        else:
            self.seen.move_to_end(key)
        return segs

    async def close(self) -> None:
        await self.client.aclose()

    async def status(self) -> dict:
        """What the console shows: whether the service answers, and with which model."""
        out = {"url": self.cfg.url, "model": self.cfg.model, "ttl_seconds": self.cfg.ttl_seconds, "reachable": False,
               "last_error": self.last_error}
        try:
            r = await self.client.get("/healthz", timeout=3.0)
            r.raise_for_status()
            data = r.json()
            out.update(reachable=True, loaded=data.get("model"), version=data.get("version"), device=data.get("device"),
                       matches=data.get("model") == self.cfg.model)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    async def _call(self, texts: list[str], rate: float) -> list[list[list[int]]]:
        t0 = time.monotonic()
        # ASCII JSON: httpx's own encoding (UTF-8) fails on a lone surrogate, which JSON can carry (half an emoji).
        body = json.dumps({"model": self.cfg.model, "rate": rate, "force_tokens": FORCE_TOKENS, "force_digits": True, "texts": texts})
        try:
            r = await self.client.post("/v1/compress", content=body, headers={"content-type": "application/json"})
        finally:
            if self.metrics is not None:
                self.metrics.observe("meritorder_compressor_duration_seconds", {}, time.monotonic() - t0)
        if r.status_code != 200:
            raise CompressorError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        if data.get("model") != self.cfg.model:
            raise CompressorError(f"the service runs {data.get('model')!r}, not {self.cfg.model!r}")
        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(texts):
            raise CompressorError("the service's reply doesn't match the request")
        drops = [res.get("drop") if isinstance(res, dict) else None for res in results]
        if not all(valid_cuts(d, len(t)) for d, t in zip(drops, texts)):
            raise CompressorError("the service returned ranges outside its texts")
        return drops

    async def compress(self, team: str, settings: ModelCompressionConfig, items: list[tuple[str, bool]]) -> tuple[dict[str, str], Counter]:
        """For each (tool result text, new on this turn): what to send instead. Texts left as they are aren't in the
        result. Also returns how many results ended up which way, for metrics."""
        outcomes: Counter = Counter()
        work = []
        for text, new in items:
            key = cache_key(team, self.cfg.model, settings.rate, text)
            segs = self.segments(key, text)
            if segs:
                work.append((text, new, segs, key))
            else:
                outcomes["no_prose"] += 1
        if not work:
            return {}, outcomes
        ttl = self.cfg.ttl_seconds
        try:
            stored = await self.state.get_json_many([w[3] for w in work], ttl)
        except Exception:
            log.exception("could not read compression decisions; sending tool results as they are")
            outcomes["store_error"] += len(work)
            return {}, outcomes
        decided: dict[str, dict] = {}
        fresh = [w for w, v in zip(work, stored) if v is None and w[1]]
        for w, v in zip(work, stored):
            if v is not None:
                decided[w[3]] = v
                outcomes["reused"] += 1
        new_entries: dict[str, dict] = {w[3]: {"s": "old"} for w, v in zip(work, stored) if v is None and not w[1]}
        if new_entries:
            outcomes["old"] += len(new_entries)
        if fresh:
            new_entries.update(await self._decide(fresh, settings, outcomes))
        if new_entries:
            try:
                decided.update(await self.state.set_json_once(new_entries, ttl))
            except Exception:
                log.exception("could not store compression decisions; sending those tool results as they are")
                outcomes["store_error"] += len(new_entries)
        out: dict[str, str] = {}
        for text, _, _, key in work:
            entry = decided.get(key)
            cuts = entry.get("d") if isinstance(entry, dict) else None
            if cuts and valid_cuts(cuts, len(text)):
                out[text] = apply_cuts(text, cuts)
        return out, outcomes

    async def _decide(self, fresh: list, settings: ModelCompressionConfig, outcomes: Counter) -> dict[str, dict]:
        """Compress the results new on this turn, by the deadline. Every one gets a decision: its cuts, or left alone."""
        if time.monotonic() < self.retry_at:
            outcomes["failed"] += len(fresh)
            return {w[3]: {"s": "failed"} for w in fresh}
        texts = [w[0][s:e] for w in fresh for s, e in w[2]]
        try:
            drops = await asyncio.wait_for(self._call(texts, settings.rate), settings.deadline_ms / 1000)
        except TimeoutError:
            outcomes["late"] += len(fresh)
            return {w[3]: {"s": "late"} for w in fresh}
        except Exception as e:
            self.retry_at = time.monotonic() + RETRY_SECONDS
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            log.warning("compressor service failed (%s); not calling it for %.0fs", self.last_error, RETRY_SECONDS)
            outcomes["failed"] += len(fresh)
            return {w[3]: {"s": "failed"} for w in fresh}
        self.last_error = None
        out: dict[str, dict] = {}
        i = 0
        for text, _, segs, key in fresh:
            cuts = cuts_for(text, segs, drops[i:i + len(segs)])
            i += len(segs)
            out[key] = {"d": cuts} if cuts else {"s": "kept"}
            outcomes["compressed" if cuts else "kept"] += 1
        return out
