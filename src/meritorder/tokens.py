"""Prompt token counts for routing: an estimate by content, corrected per model by what providers report.

The gateway needs a prompt's size before it picks a model (context-window checks, cost estimates, the output
allowance, the per-request cost limit), and it can't run every provider's tokenizer on the request path. So it
estimates, then learns how far off its estimates are for each model:

* **The estimate** (``estimate``, ``text_tokens``) counts what text is made of, not just its length: letters,
  digits, punctuation and symbols, spaces, newlines and indented lines, camelCase humps and changes between letters
  and digits (both split words in every tokenizer), and characters outside ASCII (a Chinese, Japanese or Korean
  character is about one token). Prose comes out near 4 characters a token, code and JSON nearer 3, hashes and
  base64 nearer 2, CJK text about 1. Message framing and the instructions providers add when tools are offered are
  counted too. Images, PDFs and audio aren't text: each is counted the way the model's provider counts it, from the
  image's size when the request carries its bytes (``Estimate.media``).
* **Calibration** (``Calibration``): every reply reports the real prompt token acc. Each model keeps a running
  ratio of reported to estimated tokens, so its estimates converge on its tokenizer and on how its provider frames
  messages and tools. A model without enough replies of its own leans on its family's ratio (``claude-*``,
  ``gpt-*``, ...), and a family without any on the plain estimate. Ratios outlive configuration reloads (the router
  hands them on, as it does measured speeds) and restarts (saved in the store as the setting ``token_calibration``).
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# ------------------------------------------------------------------ text


def _class_table() -> bytes:
    """Maps each UTF-8 byte to its class, so one translate and a few counts (all in C) measure a whole text."""
    t = bytearray(b"." * 256)  # ASCII punctuation, symbols and control characters
    t[ord("a"):ord("z") + 1] = b"a" * 26
    t[ord("A"):ord("Z") + 1] = b"A" * 26
    t[ord("0"):ord("9") + 1] = b"0" * 10
    t[ord(" ")] = t[ord("\t")] = t[ord("\r")] = ord(" ")  # "\r\n" is one line break, as tokenizers see it
    t[ord("\n")] = ord("\n")
    t[0x80:0xC0] = b"c" * 64  # continuation bytes
    t[0xC0:0xE0] = b"2" * 32  # starts a 2-byte character: accented Latin, Greek, Cyrillic, Hebrew, Arabic
    t[0xE0:0xF0] = b"3" * 16  # 3-byte: CJK, kana, Hangul, Indic and Thai scripts, curly quotes, dashes
    t[0xF0:0x100] = b"4" * 16  # 4-byte: emoji, rare CJK
    return bytes(t)


_CLASSES = _class_table()

# Tokens per occurrence. Set so that typical text lands near what current tokenizers produce (OpenAI's o200k and
# cl100k, Claude's, Llama's): calibration then corrects each model's overall level, and these keep the estimate
# from swinging with the mix of prose, code, data and languages in a conversation.
W_LETTER = 0.28  # ASCII letters: common words are one token, so prose is about 4 characters a token
W_DIGIT = 0.4  # digits go in groups of up to three, or one at a time
W_PUNCT = 0.5  # symbols often merge with a neighbour ("):", '{"')
W_SPACE = 0.05  # a space usually belongs to the next word's token
W_NEWLINE = 1.0
W_INDENT = 1.0  # an indented line's leading spaces are a token of their own
W_HUMP = 0.5  # a lower-to-upper case change (camelCase, random mixed case) starts a new token
W_SWITCH = 0.4  # so does a change between letters and digits (hashes, ids, base64)
W_CHAR2 = 0.4  # per 2-byte character
W_CHAR3 = 1.0  # per 3-byte character (a CJK character is about one token)
W_CHAR4 = 1.5  # per 4-byte character

# A long text is measured on evenly spaced slices of it, then scaled to its length: counting is C-speed but still
# costs about ten nanoseconds a byte, and a conversation near a million-token window is megabytes. With 512 slices,
# any part of the text longer than about 1/512th of it (8 KB of a 4 MB conversation) is sampled in proportion.
SAMPLE_CHARS = 256 * 1024
SAMPLE_SLICES = 512


# Counting costs about a microsecond per kilobyte per count, and an agent resends its whole history every turn: a
# 100 KB turn spent 1.2 ms of the gateway's CPU here, more than anything else it does. So each longer text's counts are
# kept, per process, by the text itself (a dict lookup hashes it, much cheaper than counting), and a request's texts are
# counted one by one: the counts add up to exactly the joined text's (``_join``), so the estimate doesn't change.
COUNTED_MIN_CHARS = 1_000  # shorter texts are counted each time: cheaper than remembering them
COUNTED_CHARS = 16_000_000  # how much text the counts are kept for (the oldest go first)
_counted: dict[str, tuple[int, ...]] = {}
_counted_chars = 0


def _count(s: str) -> tuple[int, ...]:
    """What the estimate weighs in a text: (bytes, lower, upper, digits, spaces, newlines, 2-, 3- and 4-byte characters,
    humps, switches, indents). Every count is of single classes or of pairs of different classes, so the counts of a
    joined text are the sum of its parts' plus what the joins add."""
    plain = s.isascii()  # O(1): Python knows
    c = (s.encode("ascii") if plain else s.encode("utf-8", "surrogatepass")).translate(_CLASSES)
    lower, upper, digits = c.count(b"a"), c.count(b"A"), c.count(b"0")
    spaces, newlines = c.count(b" "), c.count(b"\n")
    char2, char3, char4 = (0, 0, 0) if plain else (c.count(b"2"), c.count(b"3"), c.count(b"4"))
    humps = c.count(b"aA")
    switches = c.count(b"a0") + c.count(b"A0") + c.count(b"0a") + c.count(b"0A") if digits else 0
    indents = c.count(b"\n ") if newlines else 0
    return (len(c), lower, upper, digits, spaces, newlines, char2, char3, char4, humps, switches, indents)


def _counts(s: str) -> tuple[int, ...]:
    """``_count(s)``, remembered for longer texts."""
    global _counted_chars
    if len(s) < COUNTED_MIN_CHARS:
        return _count(s)
    got = _counted.get(s)
    if got is None:
        got = _counted[s] = _count(s)
        _counted_chars += len(s)
        while _counted_chars > COUNTED_CHARS and _counted:
            oldest = next(iter(_counted))
            del _counted[oldest]
            _counted_chars -= len(oldest)
    return got


def _weigh(n: int, lower: int, upper: int, digits: int, spaces: int, newlines: int, char2: int, char3: int, char4: int,
           humps: int, switches: int, indents: int) -> float:
    punct = n - lower - upper - digits - spaces - newlines - 2 * char2 - 3 * char3 - 4 * char4
    return ((lower + upper) * W_LETTER + digits * W_DIGIT + max(punct, 0) * W_PUNCT + spaces * W_SPACE
            + newlines * W_NEWLINE + indents * W_INDENT + humps * W_HUMP + switches * W_SWITCH
            + char2 * W_CHAR2 + char3 * W_CHAR3 + char4 * W_CHAR4)


def _measure(s: str) -> float:
    return _weigh(*_counts(s))


def _sampled(s: str) -> float:
    """A long text measured on evenly spaced slices of it, scaled to its length."""
    n = len(s)
    width = SAMPLE_CHARS // SAMPLE_SLICES
    starts = (i * (n - width) // (SAMPLE_SLICES - 1) for i in range(SAMPLE_SLICES))
    sample = "".join(s[start:start + width] for start in starts)
    return _measure(sample) * n / len(sample)


def text_tokens(s: str) -> float:
    """Estimated tokens in a text, by what it's made of."""
    if len(s) <= SAMPLE_CHARS:
        return _measure(s) if s else 0.0
    return _sampled(s)


def _join(texts: list[str]) -> float:
    """``text_tokens("\\n".join(texts))``, counting each text on its own (remembered when it's long). Each join adds a
    line break, and an indent when the next text starts with a space; no other pair a count looks for can span one."""
    if sum(len(t) for t in texts) + len(texts) - 1 > SAMPLE_CHARS:
        return _sampled("\n".join(texts))  # sampled, as one text
    if all(len(t) < COUNTED_MIN_CHARS for t in texts):
        return text_tokens("\n".join(texts))  # nothing worth remembering: one pass is cheapest
    parts: list[tuple[str, bool]] = []  # the long texts on their own (remembered), the short ones between them joined
    run: list[str] = []
    for t in texts:
        if len(t) >= COUNTED_MIN_CHARS:
            if run:
                parts.append(("\n".join(run), False))
                run = []
            parts.append((t, True))
        else:
            run.append(t)
    if run:
        parts.append(("\n".join(run), False))
    total = [0] * 12
    for i, (t, long) in enumerate(parts):
        for k, v in enumerate(_counts(t) if long else _count(t)):
            total[k] += v
        if i and t[:1] in (" ", "\t", "\r"):
            total[11] += 1
    total[0] += len(parts) - 1
    total[5] += len(parts) - 1
    return _weigh(*total)


# ------------------------------------------------------------------ images, documents, audio

SCHEMES = ("claude", "openai", "gemini")  # how each provider family counts an image (see image_scheme)
PER_MESSAGE = 4  # role and turn markers around each message
PER_PART = 3  # a tool call or tool result's own framing
TOOLS_PREAMBLE = 300  # what providers add when tools are offered (Claude's tool-use instructions are 300-350)
UNKNOWN_IMAGE = (1092, 1092)  # an image whose size isn't in the request (a URL): about the largest Claude takes
PDF_PAGE = {"claude": 2000, "openai": 2000, "gemini": 560}  # a page's text plus its image
PDF_BYTES_PER_PAGE = 75_000  # when the page objects can't be counted (compressed object streams)
PDF_SCAN_CHARS = 8_000_000  # of a bigger PDF's base64, pages are counted in this much and scaled to its size
AUDIO_PER_SECOND = {"claude": 10, "openai": 10, "gemini": 32}
_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def _claude_image(w: int, h: int) -> int:
    """Anthropic: scaled to at most 1568 px on the long edge and 1092 × 1092 pixels' worth, then width × height / 750
    (about 1,600 tokens at most)."""
    scale = min(1.0, 1568 / max(w, h), math.sqrt(1092 * 1092 / (w * h)))
    return max(1, math.ceil(w * scale * h * scale / 750))


def _openai_image(w: int, h: int, detail: str | None) -> int:
    """OpenAI: 85 at low detail; otherwise fit in 2048 × 2048, shorter side to 768, then 170 per 512-px tile plus 85."""
    if detail == "low":
        return 85
    s = min(1.0, 2048 / max(w, h))
    w2, h2 = w * s, h * s
    s = min(1.0, 768 / min(w2, h2))
    return 85 + 170 * math.ceil(w2 * s / 512) * math.ceil(h2 * s / 512)


def _gemini_image(w: int, h: int) -> int:
    """Gemini: 258 tokens for a small image, else 258 per 768-px tile (at most 1,120: Gemini 3's default resolution)."""
    if w <= 384 and h <= 384:
        return 258
    return min(1120, 258 * math.ceil(w / 768) * math.ceil(h / 768))


def image_dims(data: bytes) -> tuple[int, int] | None:
    """Width and height from the start of a PNG, JPEG, GIF or WebP file (no decoding). Slices past the end are
    empty, so a truncated header reads as zeros (an unknown size), never an error."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8 ":
            return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    if data[:2] == b"\xff\xd8":  # JPEG: walk the segments to the start of frame, which holds the size
        i = 2
        for _ in range(64):  # real files reach it within a few segments; a hostile one can't make this loop long
            if i + 9 > len(data) or data[i] != 0xFF:
                return None
            marker = data[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
            if marker == 0xFF:  # fill byte
                i += 1
            elif marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # markers without a length
                i += 2
            else:
                i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def _b64(data: str, limit: int | None = None) -> bytes:
    """Decode base64 (the first ``limit`` characters, when given), or b"" if it isn't."""
    if limit is not None:
        data = data[: limit - limit % 4]
    try:
        return base64.b64decode(data.strip() + "=" * (-len(data.strip()) % 4))
    except (binascii.Error, ValueError):
        return b""


def _data_url(url: str) -> tuple[str, str] | None:
    """(media type, base64 data) of a data: URL."""
    if not url.startswith("data:"):
        return None
    head, _, data = url.partition(",")
    return head[5:].split(";")[0], data


# ------------------------------------------------------------------ the estimate


@dataclass(frozen=True)
class Estimate:
    """A request's estimated prompt tokens: text (what calibration corrects) and media, per provider scheme."""

    text: int = 0  # text, code, JSON and message framing
    media: tuple[int, ...] = (0, 0, 0)  # images, PDFs and audio, counted as each of SCHEMES counts them

    @property
    def tokens(self) -> int:
        """Model-neutral: the text plus media as Claude counts it (for levels, rate limits, session sizes)."""
        return self.text + self.media[0]

    def media_for(self, scheme: str) -> int:
        return self.media[SCHEMES.index(scheme)] if scheme in SCHEMES else self.media[0]

    def less_text(self, tokens: float) -> "Estimate":
        """The same request with ``tokens`` of text taken out (what tool-output compression removed)."""
        return Estimate(text=max(0, self.text - int(tokens)), media=self.media)

    @classmethod
    def of(cls, value: "Estimate | int") -> "Estimate":
        return value if isinstance(value, Estimate) else cls(text=int(value))


@dataclass
class _Count:
    texts: list[str] = field(default_factory=list)
    framing: int = 0
    media: list[int] = field(default_factory=lambda: [0, 0, 0])

    def json(self, value: Any) -> None:
        if value is not None:
            self.texts.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))

    def image(self, dims: tuple[int, int] | None, detail: str | None = None) -> None:
        w, h = dims if dims and dims[0] > 0 and dims[1] > 0 else UNKNOWN_IMAGE
        for i, n in enumerate((_claude_image(w, h), _openai_image(w, h, detail), _gemini_image(w, h))):
            self.media[i] += n

    def pdf(self, data: str) -> None:
        raw = _b64(data, PDF_SCAN_CHARS)
        found = len(_PDF_PAGE.findall(raw)) * max(1.0, len(data) / PDF_SCAN_CHARS)
        pages = round(found) or max(1, round(len(data) * 3 / 4 / PDF_BYTES_PER_PAGE))
        for i, scheme in enumerate(SCHEMES):
            self.media[i] += pages * PDF_PAGE[scheme]

    def audio(self, data: str, fmt: str | None) -> None:
        size = len(data) * 3 // 4
        seconds = size / (32_000 if (fmt or "wav") == "wav" else 16_000)  # 16 kHz 16-bit mono, or about 128 kbit/s
        for i, scheme in enumerate(SCHEMES):
            self.media[i] += math.ceil(seconds * AUDIO_PER_SECOND[scheme])

    def result(self) -> Estimate:
        text = _join(self.texts) if self.texts else 0.0
        return Estimate(text=max(1, round(text) + self.framing), media=tuple(self.media))


def _image_source(acc: _Count, source: dict) -> None:
    """An Anthropic image block's source: base64 bytes (the size is in their header) or a URL."""
    data = source.get("data") if source.get("type") == "base64" else None
    acc.image(image_dims(_b64(data, 88_000)) if isinstance(data, str) else None)


def _url_image(acc: _Count, image: Any, detail: str | None = None) -> None:
    """An OpenAI image_url part: a data: URL carries the bytes, any other URL doesn't."""
    if isinstance(image, dict):
        image, detail = image.get("url"), image.get("detail") or detail
    parsed = _data_url(image) if isinstance(image, str) else None
    acc.image(image_dims(_b64(parsed[1], 88_000)) if parsed else None, detail)


def _document(acc: _Count, block: dict) -> None:
    source = block.get("source") or {}
    acc.json(block.get("title"))
    acc.json(block.get("context"))
    kind = source.get("type")
    if kind == "text":
        acc.json(source.get("data"))
    elif kind == "content":
        _content(acc, source.get("content"))
    elif kind == "base64" and isinstance(source.get("data"), str):
        acc.pdf(source["data"])
    else:  # a URL or an uploaded file: its size isn't in the request
        for i, scheme in enumerate(SCHEMES):
            acc.media[i] += PDF_PAGE[scheme]


def _block(acc: _Count, b: Any) -> None:
    if not isinstance(b, dict):
        acc.json(b)
        return
    kind = b.get("type")
    if kind in ("text", "input_text", "output_text"):
        acc.json(b.get("text"))
    elif kind == "image":
        _image_source(acc, b.get("source") or {})
    elif kind in ("image_url", "input_image"):
        _url_image(acc, b.get("image_url"), b.get("detail"))
    elif kind == "document":
        _document(acc, b)
    elif kind in ("file", "input_file"):  # Chat Completions' file part, the Responses API's input_file: a PDF as a data: URL
        f = (b.get("file") or {}) if kind == "file" else b
        parsed = _data_url(str(f.get("file_data") or ""))
        if parsed:
            acc.pdf(parsed[1])
        else:
            _document(acc, {})
    elif kind == "input_audio":
        audio = b.get("input_audio") or {}
        acc.audio(str(audio.get("data") or ""), audio.get("format"))
    elif kind in ("tool_use", "server_tool_use"):
        acc.framing += PER_PART
        acc.json(b.get("name"))
        acc.json(b.get("input"))
    elif kind == "tool_result":
        acc.framing += PER_PART
        _content(acc, b.get("content"))
    elif kind == "thinking":
        acc.json(b.get("thinking"))  # the signature is opaque and isn't read as text
    elif kind == "redacted_thinking":
        return
    else:  # anything else (search results, server tool results): its JSON
        acc.json({k: v for k, v in b.items() if k not in ("cache_control", "citations")})


def _content(acc: _Count, content: Any) -> None:
    if content is None:
        return
    if isinstance(content, str):
        acc.texts.append(content)
    elif isinstance(content, list):
        for b in content:
            _block(acc, b)
    else:
        _block(acc, content)


def estimate(value: Any) -> Estimate:
    """Estimated prompt tokens of a request body in either API format, or of any JSON value (embeddings input)."""
    acc = _Count()
    if isinstance(value, dict) and any(k in value for k in ("messages", "system", "tools")):
        _content(acc, value.get("system"))
        for m in value.get("messages") or []:
            if not isinstance(m, dict):
                acc.json(m)
                continue
            acc.framing += PER_MESSAGE
            _content(acc, m.get("content"))
            for call in m.get("tool_calls") or []:  # OpenAI: the call's name and its arguments (a JSON string)
                fn = (call.get("function") or {}) if isinstance(call, dict) else {}
                acc.framing += PER_PART
                acc.json(fn.get("name"))
                acc.json(fn.get("arguments"))
        if value.get("tools"):
            acc.framing += TOOLS_PREAMBLE
            acc.json(value["tools"])
        for k in ("tool_choice", "response_format"):
            if isinstance(value.get(k), dict):
                acc.json(value[k])
    else:
        _content(acc, value)
    return acc.result()


def count(value: Any) -> int:
    """Model-neutral estimated tokens (``estimate(value).tokens``)."""
    return estimate(value).tokens


# ------------------------------------------------------------------ calibration

_FAMILY = re.compile(r"[a-z]+")


def family(model: str) -> str:
    """A model's tokenizer family, from its name: claude-sonnet -> claude, gpt-sol -> gpt, qwen3-coder -> qwen."""
    m = _FAMILY.match(model.lower())
    return m.group(0) if m else model.lower()


def image_scheme(model: str) -> str:
    """Which of SCHEMES counts this model's images: OpenAI's tiles, Gemini's, or (Claude and most others) pixels."""
    fam = family(model)
    if fam in ("gpt", "o", "openai", "chatgpt"):
        return "openai"
    if fam in ("gemini", "gemma"):
        return "gemini"
    return "claude"


MIN_SAMPLE_TOKENS = 500  # below this, overheads the estimate can't see (a provider's own system text) dominate
FULL_WEIGHT_TOKENS = 20_000  # a prompt this big is a full sample; a smaller one counts in proportion
ALPHA = 0.1  # weight of the newest full sample once a ratio has history: about the last ten long prompts
PRIOR_WEIGHT = 1.0  # until a model has replies of its own, its ratio leans on its family's (or on 1)
TRUST_WEIGHT = 3.0  # from here on, outliers are capped and the spread sets the safety margin
BOUNDS = (0.4, 2.5)  # a sample outside this is a broken count (usage that left something out), not a tokenizer
OUTLIER = 0.25  # once trusted, one sample counts as at most 25% away from the current ratio
MAX_MEDIA_SHARE = 0.25  # requests that are mostly images or documents say little about the text ratio
UNCALIBRATED_MARGIN = 1 / 3  # how far the plain estimate may run low (dense code), allowed for in output limits
MIN_MARGIN = 0.03
SPREAD_MARGIN = 2.5  # the margin, in spreads (moving mean absolute deviation) of a calibrated ratio
CLOSE_SHARE = 0.85  # a prompt over this share of a window is reported in the routing reason
SETTING = "token_calibration"


@dataclass
class Ratio:
    """Reported prompt tokens per estimated token for one model, from its replies."""

    ratio: float = 1.0
    weight: float = 0.0  # samples so far, each counting up to 1 by its size
    spread: float = 0.0  # moving mean of |sample - ratio| / ratio
    samples: int = 0
    updated: float = 0.0  # wall-clock time of the latest sample

    def effective(self, prior: float) -> float:
        """The ratio to use: its own, leaning on ``prior`` while it has few samples."""
        return (self.ratio * self.weight + prior * PRIOR_WEIGHT) / (self.weight + PRIOR_WEIGHT)

    @property
    def trusted(self) -> bool:
        return self.weight >= TRUST_WEIGHT

    def update(self, sample: float, weight: float, prior: float, now: float) -> None:
        current = self.effective(prior)
        if self.trusted:
            sample = min(max(sample, current / (1 + OUTLIER)), current * (1 + OUTLIER))
        self.weight += weight
        step = max(weight / self.weight, ALPHA * weight)  # a plain mean at first, then a moving one
        self.spread += step * (abs(sample - current) / current - self.spread)
        self.ratio += step * (sample - self.ratio)
        self.samples += 1
        self.updated = now


class Calibration:
    """Per-model correction of the estimate, learned from the prompt token counts providers report.

    Kept per gateway process, handed to each new router on a configuration reload, and saved to the store so a
    restart starts from what was learned. A model with few replies of its own leans on the other models of its
    family (their ratios, weighted by how much each has seen), and a family with none on 1."""

    def __init__(self) -> None:
        self.models: dict[str, Ratio] = {}
        self.dirty: set[str] = set()  # models learned from since the last save
        self._priors: dict[str, float] = {}

    def prior(self, model: str) -> float:
        """What the other models of this model's family say, leaning on 1 while they've seen little."""
        if model not in self._priors:
            fam = family(model)
            others = [r for name, r in self.models.items() if name != model and family(name) == fam]
            weight = sum(r.weight for r in others)
            self._priors[model] = (sum(r.ratio * r.weight for r in others) + PRIOR_WEIGHT) / (weight + PRIOR_WEIGHT)
        return self._priors[model]

    def ratio(self, model: str) -> float:
        own = self.models.get(model)
        return own.effective(self.prior(model)) if own else self.prior(model)

    def count(self, model: str, est: Estimate | int) -> int:
        """Estimated prompt tokens for this model: the text corrected by its ratio, plus media as it counts them."""
        est = Estimate.of(est)
        return round(est.text * self.ratio(model)) + est.media_for(image_scheme(model))

    def margin(self, model: str) -> float:
        """How far this model's count may still run low: a third uncalibrated, down to 3% once it's measured."""
        own = self.models.get(model)
        if own is None or not own.trusted:
            return UNCALIBRATED_MARGIN
        return min(UNCALIBRATED_MARGIN, max(MIN_MARGIN, SPREAD_MARGIN * own.spread))

    def record(self, model: str, est: Estimate | int, reported: int, now: float | None = None) -> bool:
        """Learn from a reply: ``reported`` is the provider's prompt token count for a request estimated at ``est``.
        Returns whether the sample was used."""
        est = Estimate.of(est)
        media = est.media_for(image_scheme(model))
        if est.text < MIN_SAMPLE_TOKENS or media > MAX_MEDIA_SHARE * est.text or reported <= media:
            return False
        sample = (reported - media) / est.text
        if not BOUNDS[0] <= sample <= BOUNDS[1]:
            return False
        weight = min(1.0, est.text / FULL_WEIGHT_TOKENS)
        self.models.setdefault(model, Ratio()).update(sample, weight, self.prior(model), time.time() if now is None else now)
        self._priors.clear()
        self.dirty.add(model)
        return True

    def _from_family(self, model: str) -> bool:
        fam = family(model)
        return any(name != model and family(name) == fam and r.samples for name, r in self.models.items())

    def describe(self, model: str) -> str:
        """For routing reasons: how this model's count was made."""
        own = self.models.get(model)
        if own and own.samples:
            return f"estimate ×{self.ratio(model):.2f}, calibrated from {own.samples:,} {'reply' if own.samples == 1 else 'replies'}"
        if self._from_family(model):
            return f"estimate ×{self.ratio(model):.2f}, from other {family(model)} models"
        return "estimate, not calibrated yet"

    def view(self, model: str) -> dict:
        """For the console: the ratio in use and what it's based on."""
        own = self.models.get(model)
        source = "model" if own and own.samples else "family" if self._from_family(model) else "none"
        return {"ratio": round(self.ratio(model), 3), "source": source, "family": family(model),
                "samples": own.samples if own else 0, "spread": round(own.spread, 3) if own and own.trusted else None,
                "margin": round(self.margin(model), 3), "updated": own.updated if own else None}

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {"version": 1, "models": {k: asdict(v) for k, v in self.models.items()}}

    def load(self, doc: Any) -> None:
        """Take saved ratios for models this process hasn't learned anything about yet."""
        if not isinstance(doc, dict) or doc.get("version") != 1:
            return
        for name, rec in (doc.get("models") or {}).items():
            if name in self.models or not isinstance(rec, dict):
                continue
            try:
                r = Ratio(**{k: float(rec[k]) for k in ("ratio", "weight", "spread", "updated")}, samples=int(rec["samples"]))
            except (KeyError, TypeError, ValueError):
                continue
            if BOUNDS[0] <= r.ratio <= BOUNDS[1] and r.weight >= 0:
                self.models[name] = r
        self._priors.clear()

    def merged(self, stored: Any) -> dict:
        """What to save: the stored ratios, with the ones this process learned from since its last save replacing
        theirs (other gateways learn too; each keeps the latest it measured)."""
        doc = stored if isinstance(stored, dict) and stored.get("version") == 1 else {}
        mine = {k: asdict(v) for k, v in self.models.items() if k in self.dirty}
        return {"version": 1, "models": {**(doc.get("models") or {}), **mine}}


# ------------------------------------------------------------------ counts in provider errors

_REPORTED = [re.compile(p, re.IGNORECASE) for p in (
    r"prompt is too long:\s*([\d,]+)\s*tokens",  # Anthropic
    r"\(([\d,]+) in the messages",  # OpenAI, vLLM: "you requested N tokens (M in the messages, K in the completion)"
    r"messages resulted in ([\d,]+) tokens",  # OpenAI
    r"request has ([\d,]+) input tokens",  # vLLM
    r"input token count \(?([\d,]+)\)?",  # Gemini
)]


def reported_prompt_tokens(message: str | None) -> int | None:
    """The prompt's real size, when a provider's "too long" error says it."""
    for pattern in _REPORTED:
        m = pattern.search(message or "")
        if m:
            n = int(m.group(1).replace(",", ""))
            if n > 0:
                return n
    return None
