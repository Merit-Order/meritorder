"""Prompt token counts: the estimate by content (text, code, JSON, CJK, images, PDFs, audio), calibration per model
from what providers report (converging, used for window checks, capped against outliers, kept across reloads and
restarts), and where it shows (routing reasons, the Models page, metrics)."""

import asyncio
import base64
import copy
import json
import logging
import random
import struct
from types import SimpleNamespace

import httpx
import pytest
from conftest import ADMIN_KEY, TEST_CONFIG, auth
from test_count_tokens_policy import Counting
from test_http_backend import OPENAI_OK, Upstream

from meritorder import app as app_module
from meritorder import tokens as T
from meritorder.app import build_gateway, create_app
from meritorder.backends.base import Backend
from meritorder.config import BackendConfig, Config, ModelConfig, Price, Route, build_config
from meritorder.proxy import Gateway
from meritorder.router import Candidate, Router
from meritorder.sessions import Session
from meritorder.state import MemoryState
from meritorder.store import MemoryStore
from meritorder.tokens import Calibration, Estimate, estimate, image_dims, reported_prompt_tokens, text_tokens

PROSE = ("The quick brown fox jumps over the lazy dog. It was a sunny day, and everyone in the office was happy to see "
         "the new release go out on time. ") * 40
CODE = '''    def decide(self, requested: str, team: TeamConfig) -> Decision:
        """Pick and rank candidates."""
        floor = parse_floor(headers.get("x-route-floor", ""))
        for m in self.cfg.models:
            if m.q(floor.task) >= floor.level and team.allows_model(m.name):
                models.append(m)
        return Decision(requested=requested, auto=True, floor=floor)
''' * 20
JSON_TEXT = json.dumps([{"id": i, "name": f"item-{i}", "price": round(i * 1.37, 2), "tags": ["a", "b"], "ok": i % 2 == 0}
                        for i in range(300)])
CJK = "这是一个测试。我们正在检查令牌估计是否正确。日本語のテキストも含まれています。" * 20 + "한국어 문장도 있습니다." * 30


def chars_per_token(s: str) -> float:
    return len(s) / text_tokens(s)


# ------------------------------------------------------------------ the estimate


def test_text_is_counted_by_what_its_made_of():
    assert text_tokens("") == 0
    assert 3.6 < chars_per_token(PROSE) < 4.6  # common words are about a token each
    assert 2.8 < chars_per_token(CODE) < 4.0
    assert 2.0 < chars_per_token(JSON_TEXT) < 3.4  # symbols and short keys
    assert 0.8 < chars_per_token(CJK) < 1.6  # about one token a character, four times what length/4 would say
    random.seed(1)
    blob = base64.b64encode(random.randbytes(30_000)).decode()
    assert 1.6 < chars_per_token(blob) < 2.6  # case changes and digits split it
    assert 1.5 < chars_per_token(random.randbytes(10_000).hex()) < 2.5
    assert text_tokens("getUserName") > text_tokens("getusername")  # camelCase humps start tokens
    assert text_tokens("café naïve") > text_tokens("cafe naive")


def test_a_long_text_is_measured_on_slices_and_comes_out_the_same():
    text = (PROSE + CODE + JSON_TEXT + CJK) * 40  # well past SAMPLE_CHARS, mixed content
    assert len(text) > 2 * T.SAMPLE_CHARS
    assert text_tokens(text) == pytest.approx(T._measure(text), rel=0.02)


def _measure_joined(s: str) -> float:
    """The estimate as it was computed before counts were kept per text: one pass over the joined text."""
    plain = s.isascii()
    c = (s.encode("ascii") if plain else s.encode("utf-8", "surrogatepass")).translate(T._CLASSES)
    lower, upper, digits = c.count(b"a"), c.count(b"A"), c.count(b"0")
    spaces, newlines = c.count(b" "), c.count(b"\n")
    char2, char3, char4 = (0, 0, 0) if plain else (c.count(b"2"), c.count(b"3"), c.count(b"4"))
    punct = len(c) - lower - upper - digits - spaces - newlines - 2 * char2 - 3 * char3 - 4 * char4
    humps = c.count(b"aA")
    switches = c.count(b"a0") + c.count(b"A0") + c.count(b"0a") + c.count(b"0A") if digits else 0
    indents = c.count(b"\n ") if newlines else 0
    return ((lower + upper) * T.W_LETTER + digits * T.W_DIGIT + max(punct, 0) * T.W_PUNCT + spaces * T.W_SPACE
            + newlines * T.W_NEWLINE + indents * T.W_INDENT + humps * T.W_HUMP + switches * T.W_SWITCH
            + char2 * T.W_CHAR2 + char3 * T.W_CHAR3 + char4 * T.W_CHAR4)


def test_counting_texts_one_by_one_gives_exactly_the_joined_estimate(monkeypatch):
    """A request's texts are counted on their own and remembered (an agent resends its history every turn): the
    estimate must be the one counting the joined text gives, to the last bit, on every turn."""
    rng = random.Random(7)
    pieces = [PROSE[:3000], CODE[:2500], JSON_TEXT[:4000], CJK[:1500], "", " indented", "\tTabbed\n", "\rcarriage", "\n",
              " ", "aA0", "end0\n", "x" * 1200, "😀 emoji\ud83d half", "getUserName v2.1 a0A0"]
    for _ in range(300):
        texts = [rng.choice(pieces)[rng.randrange(3):] for _ in range(rng.randrange(1, 9))]
        want = _measure_joined("\n".join(texts)) if "\n".join(texts) else 0.0
        assert T._join(texts) == want == T._join(texts)  # the second time from the remembered counts
        assert text_tokens("\n".join(texts)) == want
    big = [PROSE + CODE] * 120  # past SAMPLE_CHARS: sampled as one text, as before
    assert T._join(big) == text_tokens("\n".join(big)) == T._sampled("\n".join(big))
    # The remembered counts stay within their bound, oldest out first.
    monkeypatch.setattr(T, "COUNTED_CHARS", 5000)
    T._counted.clear()
    monkeypatch.setattr(T, "_counted_chars", 0)
    for i in range(10):
        T._counts(f"{i} " + "y" * 1500)
    assert T._counted_chars <= 5000 and len(T._counted) == 3 and next(iter(T._counted)).startswith("7 ")


def test_both_api_formats_estimate_the_same_conversation_alike():
    tools_openai = [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object"}}}]
    tools_anthropic = [{"name": "read", "description": "Read a file", "input_schema": {"type": "object"}}]
    openai = {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": PROSE},
                           {"role": "assistant", "content": None, "tool_calls": [
                               {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.py"}'}}]},
                           {"role": "tool", "tool_call_id": "c1", "content": CODE}], "tools": tools_openai}
    anthropic = {"system": [{"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}],
                 "messages": [{"role": "user", "content": PROSE},
                              {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "a.py"}}]},
                              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1",
                                                            "content": [{"type": "text", "text": CODE}]}]}],
                 "tools": tools_anthropic}
    a, b = estimate(openai), estimate(anthropic)
    assert a.text == pytest.approx(b.text, rel=0.05) and a.media == b.media == (0, 0, 0)
    assert a.text > text_tokens(PROSE) + text_tokens(CODE) + T.TOOLS_PREAMBLE  # framing and tools are counted
    assert T.count(PROSE) == estimate(PROSE).tokens and estimate(["a", "b", 3]).tokens >= 1


def png(w: int, h: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00" + b"\x00" * 200_000


def jpeg(w: int, h: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof = b"\xff\xc0" + struct.pack(">HBHH", 17, 8, h, w) + b"\x00" * 12
    return b"\xff\xd8\xff" + app0[1:] + b"\xff\xff" + sof + b"\x00" * 1000  # a fill byte between segments is allowed


def test_image_sizes_come_from_the_file_header():
    assert image_dims(png(1000, 500)) == (1000, 500)
    assert image_dims(jpeg(640, 480)) == (640, 480)
    assert image_dims(b"GIF89a" + struct.pack("<HH", 320, 200)) == (320, 200)
    vp8 = b"RIFF" + b"\x00" * 4 + b"WEBPVP8 " + b"\x00" * 4 + b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", 800, 600)
    vp8l = b"RIFF" + b"\x00" * 4 + b"WEBPVP8L" + b"\x00" * 4 + b"\x2f" + struct.pack("<I", (799) | (599 << 14))
    vp8x = b"RIFF" + b"\x00" * 4 + b"WEBPVP8X" + b"\x00" * 8 + (799).to_bytes(3, "little") + (599).to_bytes(3, "little")
    assert image_dims(vp8) == image_dims(vp8l) == image_dims(vp8x) == (800, 600)
    assert image_dims(b"\xff\xd8\x00\x00" + b"\x00" * 20) is None and image_dims(b"not an image") is None
    restart = b"\xff\xd8" + b"\xff\xd0" + jpeg(64, 32)[2:]  # a marker without a length before the frame
    assert image_dims(restart) == (64, 32)
    assert image_dims(b"\xff\xd8" + b"\xff" * 5000) is None  # fill bytes forever: gives up quickly
    assert T._b64("abcde") == b"" and T._b64(b64(b"hello"), 4) == b"hel"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def user(*blocks) -> dict:
    """A request with one user message made of these content blocks."""
    return {"messages": [{"role": "user", "content": list(blocks)}]}


def test_images_count_the_way_each_provider_counts_them_not_as_text():
    pic = png(1000, 1000)
    claude = {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(pic)}},
                                                         {"type": "text", "text": "what is this?"}]}]}
    e = estimate(claude)
    assert e.text < 50  # 270 KB of base64 is not 67,000 tokens of text
    assert e.media == (1334, 765, 1032)  # Claude: w×h/750; OpenAI: 4 tiles of 512 + 85; Gemini: 2×2 tiles of 768
    url = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(pic)}"}}]}]}
    assert estimate(url).media == e.media
    low = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "low"}}]}]}
    assert estimate(low).media[1] == 85
    unknown = estimate({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]}]})
    assert unknown.media == (1590, 765, 1032)  # size unknown: about the largest image Claude takes
    huge = estimate(user({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64(jpeg(4000, 3000))}"}))
    assert huge.media[0] <= 1600 and huge.media[2] == 1120  # scaled down first; Gemini at its default resolution
    gif = b"GIF89a" + struct.pack("<HH", 100, 100)
    small = estimate(user({"type": "image_url", "image_url": f"data:image/gif;base64,{b64(gif)}"}))
    assert small.media[2] == 258 and small.media[0] == 14
    broken = estimate({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "%%%"}}]}]})
    assert broken.media == unknown.media
    assert e.tokens == e.text + 1334 and e.media_for("openai") == 765 and e.media_for("other") == 1334


def test_documents_and_audio(monkeypatch):
    pdf = b"%PDF-1.7\n" + b"1 0 obj << /Type /Pages /Count 3 >>\n" + b"<< /Type /Page >>\n" * 3 + b"x" * 1000
    claude = user({"type": "document", "title": "Report", "source": {"type": "base64", "media_type": "application/pdf", "data": b64(pdf)}})
    assert estimate(claude).media == (6000, 6000, 1680)  # three pages
    openai = {"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_data": f"data:application/pdf;base64,{b64(pdf)}"}}]}]}
    assert estimate(openai).media == (6000, 6000, 1680)
    packed = b"%PDF-1.7\n" + b"x" * 300_000  # page objects inside compressed streams: guessed from the size
    many = b"%PDF-1.7\n" + (b"<< /Type /Page >>" + b"x" * 1000) * 40
    assert estimate(user({"type": "document", "source": {"type": "base64", "data": b64(packed)}})).media[0] == 8000
    for doc in ({"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}}, {"type": "file", "file": {"file_id": "f1"}}):
        assert estimate({"messages": [{"role": "user", "content": [doc]}]}).media[0] == 2000  # size unknown: a page
    monkeypatch.setattr(T, "PDF_SCAN_CHARS", 20_000)  # a big PDF: pages counted in the start, scaled to its size
    assert 36 <= estimate(user({"type": "document", "source": {"type": "base64", "data": b64(many)}})).media[0] / 2000 <= 44
    text_doc = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": PROSE}, "context": "a note"}
    assert estimate({"messages": [{"role": "user", "content": [text_doc]}]}).text > text_tokens(PROSE)
    nested = {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": PROSE}]}}
    assert estimate({"messages": [{"role": "user", "content": [nested]}]}).text > text_tokens(PROSE)
    wav = "A" * 64_000  # 48,000 bytes of 16 kHz 16-bit audio: 1.5 seconds
    audio = estimate({"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}}]}]})
    assert audio.media == (15, 15, 48)


def test_thinking_signatures_and_unknown_blocks():
    sig = "S" * 40_000
    with_sig = estimate({"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": sig},
                                                                         {"type": "redacted_thinking", "data": sig}]}]})
    assert with_sig.text < 20  # the signature and redacted data aren't text the model reads
    other = estimate({"messages": [{"role": "user", "content": [{"type": "search_result", "title": "t", "content": PROSE, "cache_control": {}}]}]})
    assert other.text > text_tokens(PROSE)
    odd = estimate({"messages": ["not a message", {"role": "user", "content": {"type": "text", "text": "hi"}}, {"role": "user", "content": 7}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "x"}}})
    assert odd.text > 4


# ------------------------------------------------------------------ calibration


def test_calibration_converges_on_what_replies_report():
    c = Calibration()
    random.seed(7)
    for _ in range(30):
        est = Estimate(text=random.randint(20_000, 200_000))
        c.record("claude-sonnet", est, round(est.text * 1.3 * random.uniform(0.95, 1.05)))
    assert c.ratio("claude-sonnet") == pytest.approx(1.3, abs=0.03)
    assert c.count("claude-sonnet", 100_000) == pytest.approx(130_000, rel=0.03)
    assert c.margin("claude-sonnet") < 0.1  # measured, so less room is needed for error
    # Another model of the same family starts from the family's ratio, not from 1.
    assert c.ratio("claude-opus") == pytest.approx(1.3, abs=0.05) and "from other claude models" in c.describe("claude-opus")
    assert c.margin("claude-opus") == T.UNCALIBRATED_MARGIN
    assert c.ratio("gpt-sol") == 1.0 and c.describe("gpt-sol") == "estimate, not calibrated yet"
    assert "calibrated from 30 replies" in c.describe("claude-sonnet")


def test_few_samples_lean_on_the_prior_and_small_prompts_barely_count():
    c = Calibration()
    c.record("m", Estimate(text=20_000), 30_000)
    assert c.ratio("m") == pytest.approx(1.25)  # one full sample: halfway from 1 to 1.5
    small = Calibration()
    small.record("m", Estimate(text=1_000), 1_500)
    assert 1.0 < small.ratio("m") < 1.03  # a 1,000-token prompt is a twentieth of a sample
    assert not small.record("m", Estimate(text=300), 900)  # too small: fixed overheads dominate
    assert small.models["m"].samples == 1


def test_outliers_are_capped():
    c = Calibration()
    for _ in range(20):
        c.record("m", Estimate(text=50_000), 55_000)
    before = c.ratio("m")
    assert not c.record("m", Estimate(text=50_000), 500_000)  # ten times: a broken count, ignored
    assert not c.record("m", Estimate(text=50_000), 10_000)
    assert c.record("m", Estimate(text=50_000), 120_000)  # in bounds but far off: counted as at most 25% away
    assert c.ratio("m") - before <= T.ALPHA * before * T.OUTLIER * 1.01
    assert c.ratio("m") < 1.15
    # Requests that are mostly images say little about text, and a count below the media alone is wrong.
    assert not c.record("m", Estimate(text=5_000, media=(4_000, 4_000, 4_000)), 9_000)
    assert not c.record("m", Estimate(text=50_000, media=(2_000, 2_000, 2_000)), 1_500)


def test_counts_add_media_the_models_way():
    c = Calibration()
    est = Estimate(text=1000, media=(1334, 765, 1032))
    assert c.count("claude-x", est) == 2334 and c.count("gpt-x", est) == 1765 and c.count("gemini-x", est) == 2032
    assert c.count("qwen-vl", est) == 2334  # pixel-based, like Claude
    assert T.family("qwen3-coder") == "qwen" and T.family("GPT-Sol") == "gpt" and T.family("123") == "123"


def test_calibration_round_trips_and_merges_with_other_gateways():
    c = Calibration()
    for _ in range(5):
        c.record("claude-a", Estimate(text=20_000), 24_000, now=100.0)
    doc = json.loads(json.dumps(c.to_dict()))
    restored = Calibration()
    restored.load(doc)
    assert restored.ratio("claude-a") == pytest.approx(c.ratio("claude-a")) and restored.models["claude-a"].samples == 5
    assert not restored.dirty  # loaded, not learned
    # What this process learned wins over what was saved; others' models are kept.
    other = {"version": 1, "models": {"gpt-b": {"ratio": 0.9, "weight": 4, "spread": 0.02, "samples": 9, "updated": 5}}}
    merged = c.merged(other)
    assert set(merged["models"]) == {"claude-a", "gpt-b"}
    restored.load(merged)  # claude-a is already known here: kept; gpt-b is new
    assert restored.ratio("gpt-b") == pytest.approx((0.9 * 4 + 1) / 5)
    for junk in (None, [], {"version": 2}, {"version": 1, "models": {"x": {"ratio": "nan?"}, "y": "z", "w": {"ratio": 99, "weight": 1,
                                                                                                         "spread": 0, "samples": 1, "updated": 0}}}):
        fresh = Calibration()
        fresh.load(junk)
        assert not fresh.models
    assert Calibration().merged(None) == {"version": 1, "models": {}}


def test_provider_errors_that_say_the_real_size():
    assert reported_prompt_tokens("prompt is too long: 1,052,341 tokens > 1000000 maximum") == 1_052_341
    openai = "This model's maximum context length is 128000 tokens. However, your messages resulted in 130567 tokens."
    assert reported_prompt_tokens(openai) == 130_567
    assert reported_prompt_tokens("However, you requested 140000 tokens (130000 in the messages, 10000 in the completion).") == 130_000
    assert reported_prompt_tokens("This model's maximum context length is 32768 tokens. However, your request has 40000 input tokens.") == 40_000
    assert reported_prompt_tokens("The input token count (1234567) exceeds the maximum number of tokens allowed (1048576).") == 1_234_567
    assert reported_prompt_tokens("prompt is too long") is None and reported_prompt_tokens(None) is None
    assert reported_prompt_tokens("prompt is too long: 0 tokens") is None


# ------------------------------------------------------------------ routing


def two_models(small_window=40_000):
    bc = BackendConfig(name="up", format="openai", base_url="https://up.example/v1")
    small = ModelConfig(name="tight-small", quality={"coding": 3, "general": 3}, context_window=small_window,
                        routes=[Route(backend="up", model="s-1", price=Price(input=0.1, output=0.2))])
    big = ModelConfig(name="roomy-big", quality={"coding": 3, "general": 3}, context_window=1_000_000,
                      routes=[Route(backend="up", model="b-1", price=Price(input=1, output=2))])
    cfg = Config(backends=[bc], models=[small, big])
    return Router(cfg, {"up": Backend(bc)}), cfg


def decide(router, cfg, est, session=None, body=None):
    return router.decide(requested="auto", team=cfg.team("default"), body=body or {"messages": [{"role": "user", "content": "hi"}]},
                         headers={"x-route-floor": "general:3"}, session=session, prompt_tokens=est, boundary=session is None,
                         utilization={}, now=0.0)


def test_window_checks_use_the_calibrated_count_and_say_so(caplog):
    router, cfg = two_models()
    est = Estimate(text=30_000)
    d = decide(router, cfg, est)
    assert d.candidates[0].model.name == "tight-small" and d.candidates[0].prompt_tokens == 30_000  # fits uncalibrated
    for _ in range(10):
        router.tokens.record("tight-small", Estimate(text=20_000), 29_000)  # it runs 1.45 times the estimate
    ratio = router.tokens.ratio("tight-small")
    assert ratio == pytest.approx((1.45 * 10 + 1) / 11)  # ten full samples, leaning a little on 1
    d = decide(router, cfg, est)
    assert [c.model.name for c in d.candidates] == ["roomy-big"]
    assert (f"ruled out tight-small (40,000): window too small for about {round(30_000 * ratio):,} tokens "
            f"(estimate ×{ratio:.2f}, calibrated from 10 replies)") in d.reason
    asked = decide(router, cfg, Estimate(text=20_000), body={"max_tokens": 12_000, "messages": [{"role": "user", "content": "hi"}]})
    assert [c.model.name for c in asked.candidates] == ["roomy-big"]  # 28,000 in its tokens plus the reply: 40,000 (32,000 uncalibrated)
    assert f"about {round(20_000 * ratio):,} tokens plus a 12,000-token reply (estimate ×{ratio:.2f}" in asked.reason

    # Close but fitting: the reason (and the log) say how full the window is and how the count was made.
    with caplog.at_level(logging.INFO, logger="meritorder.router"):
        d = decide(router, cfg, Estimate(text=25_000))
    assert d.candidates[0].model.name == "tight-small"
    assert f"context window check close: tight-small ({round(25_000 * ratio):,} of 40,000 tokens, 88%;" in d.reason
    assert "near the context window" in caplog.text
    assert "close" not in decide(router, cfg, Estimate(text=5_000)).reason


def test_session_growth_and_long_prompt_prices_are_per_model():
    router, cfg = two_models()
    for _ in range(10):
        router.tokens.record("roomy-big", Estimate(text=20_000), 40_000)  # twice the estimate
    session = Session(id="s", model="roomy-big", backend="up", floor="general:3", turns=3, prefix_tokens=10_000)
    d = decide(router, cfg, Estimate(text=12_000), session)
    big = next(c for c in d.candidates if c.model.name == "roomy-big")
    ratio = router.tokens.ratio("roomy-big")
    turns = router._expected_turns(session)
    expected = router.estimate(Price(input=1, output=2), big.backend, round(12_000 * ratio), session, False, turns, "b-1",
                               new_tokens=round(2_000 * ratio), pause=router.pauses.share("default"))
    assert big.cost == pytest.approx(expected) and big.prompt_tokens == round(12_000 * ratio)


def test_output_allowance_leaves_less_room_once_the_model_is_measured():
    bc = BackendConfig(name="b", format="anthropic", kind="api", base_url="https://b.example")
    m = ModelConfig(name="m", max_output_tokens=128_000, context_window=1_000_000,
                    routes=[Route(backend="b", model="m-1", price=Price(input=8, output=40))])
    cand = Candidate(model=m, route=m.routes[0], backend=Backend(bc), cost=0, warm=False, quality=5)
    gw = Gateway(Config(backends=[bc], models=[m]), None, None, {}, None, None)
    assert gw.output_allowance({}, cand, 700_000) == (1_000_000 - int(700_000 * (1 + 1 / 3)), "context")
    for _ in range(10):
        gw.tokens.record("m", Estimate(text=100_000), 100_000)  # the estimate is right, every time
    assert gw.tokens.margin("m") == T.MIN_MARGIN
    assert gw.output_allowance({}, cand, 700_000) == (128_000, "model")  # 3% is enough room for error now


# ------------------------------------------------------------------ through the gateway


def gateway_with(upstream, **models):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {"up": {"kind": "api", "format": "openai", "base_url": "https://up.example/v1",
                              "auth": {"type": "api_key", "value": "k"}}}
    raw["models"] = {
        "tight-small": {"quality": {"coding": 3, "general": 3}, "context_window": 40_000,
                        "routes": [{"backend": "up", "model": "s-1", "price": {"input": 0.1, "output": 0.2}}]},
        "roomy-big": {"quality": {"coding": 3, "general": 3}, "context_window": 1_000_000,
                      "routes": [{"backend": "up", "model": "b-1", "price": {"input": 1, "output": 2}}]},
        **models,
    }
    raw["baseline_model"] = "roomy-big"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    gw.backends["up"].client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return gw


def reporting(ratio: float):
    """A provider whose tokenizer counts ``ratio`` times the gateway's estimate of what it was sent."""
    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = round(estimate(body).text * ratio)
        usage = {"prompt_tokens": n, "completion_tokens": 2, "total_tokens": n + 2}
        return httpx.Response(200, json={**OPENAI_OK, "model": body["model"], "usage": usage})
    return answer


async def send(gw, text: str, **headers):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://t") as c:
        return await c.post("/v1/chat/completions", headers={**auth(), "x-route-floor": "general:3", **headers},
                            json={"model": "auto", "messages": [{"role": "user", "content": text}]})


def words(tokens: int) -> str:
    """Prose of about this many estimated tokens."""
    return PROSE * max(1, round(tokens / text_tokens(PROSE)))


async def test_replies_calibrate_the_model_and_later_window_checks_use_it():
    up = Upstream(reporting(1.5))
    gw = gateway_with(up)
    for _ in range(9):
        r = await send(gw, words(20_000))
        assert r.status_code == 200 and r.headers["x-meritorder-model"] == "tight-small"
    await gw.drain()
    assert gw.tokens.models["tight-small"].samples == 9 and gw.tokens.ratio("tight-small") == pytest.approx(1.45, abs=0.02)
    assert "roomy-big" not in gw.tokens.models
    r = await send(gw, words(30_000))  # fits in 40,000 by the plain estimate; not at 1.45 times it
    assert r.headers["x-meritorder-model"] == "roomy-big"
    assert 'meritorder_token_ratio{model="tight-small"} 1.4' in gw.metrics.render()
    assert 'meritorder_token_calibration_samples_total{model="tight-small"} 9' in gw.metrics.render()

    # A configuration reload keeps what was learned (the new router gets the same calibration).
    ratio = gw.tokens.ratio("tight-small")
    await gw.apply_config(gw.cfg, None)
    assert gw.router.tokens is gw.tokens and gw.tokens.ratio("tight-small") == ratio
    # So does a restart: saved to the store, loaded by the next process.
    await gw.save_token_calibration()
    assert not gw.tokens.dirty and await gw.store.get_setting(T.SETTING)
    await gw.save_token_calibration()  # nothing new: no write
    again = build_gateway(gw.cfg, store=gw.store, state=MemoryState())
    await again.load_token_calibration()
    assert again.tokens.ratio("tight-small") == pytest.approx(ratio)
    await gw.drain()


async def test_compressed_tool_output_and_mock_replies_and_web_search():
    up = Upstream(reporting(1.0))
    gw = gateway_with(up)
    gw.cfg.team("default").compression.enabled = True
    log = "progress: copying files\n" * 4000
    body = {"model": "auto", "messages": [
        {"role": "user", "content": words(3_000)},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "sh", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": log}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", headers={**auth(), "x-route-floor": "general:3"}, json=body)
    assert r.status_code == 200
    await gw.drain()
    # The provider counted the compressed prompt; the estimate it's compared with is of the compressed prompt too.
    assert gw.tokens.models["tight-small"].ratio == pytest.approx(1.0, abs=0.02)

    # Mock backends make their counts up: they never calibrate anything.
    mock = build_gateway(build_config(copy.deepcopy(TEST_CONFIG), quiet=True), store=MemoryStore(), state=MemoryState())
    assert (await send(mock, words(5_000))).status_code == 200
    await mock.drain()
    assert not mock.tokens.models


async def test_web_search_results_dont_count_as_the_prompt():
    from test_http_backend import ANTHROPIC_OK

    def answer(request):
        return httpx.Response(200, json={**ANTHROPIC_OK, "usage": {"input_tokens": 90_000, "output_tokens": 2,
                                                                   "server_tool_use": {"web_search_requests": 1}}})
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {"ant": {"kind": "api", "format": "anthropic", "base_url": "https://ant.example", "auth": {"type": "api_key", "value": "k"}}}
    raw["models"] = {"claude-x": {"quality": {"general": 4}, "routes": [{"backend": "ant", "model": "cx", "price": {"input": 1, "output": 2}}]}}
    raw["baseline_model"] = "claude-x"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    gw.backends["ant"].client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://t") as c:
        r = await c.post("/v1/messages", headers=auth(), json={"model": "claude-x", "max_tokens": 10,
                                                               "messages": [{"role": "user", "content": words(20_000)}]})
    assert r.status_code == 200
    await gw.drain()
    assert not gw.tokens.models


async def test_a_too_long_error_that_gives_the_size_calibrates_and_moves_on():
    def answer(request):
        body = json.loads(request.content)
        if body["model"] == "s-1":
            return httpx.Response(400, json={"error": {"message": f"prompt is too long: {round(estimate(body).text * 1.6)} tokens > 40000 maximum"}})
        return reporting(1.0)(request)
    gw = gateway_with(Upstream(answer))
    r = await send(gw, words(30_000))
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "roomy-big"
    assert "prompt too long" in r.headers["x-meritorder-failed-over"]
    await gw.drain()
    assert gw.tokens.models["tight-small"].samples == 1 and gw.tokens.models["tight-small"].ratio == pytest.approx(1.6, abs=0.01)


async def test_count_tokens_learns_from_exact_counts_and_falls_back_to_the_calibrated_estimate(gateway, client):
    us = Counting("anthropic-us")
    body = {"model": "claude-x", "messages": [{"role": "user", "content": words(20_000)}]}
    exact = round(estimate({k: body.get(k) for k in ("system", "messages", "tools")}).text * 1.2)

    async def count_tokens(b, headers):
        us.calls += 1
        return {"input_tokens": exact}
    us.count_tokens = count_tokens
    gateway.backends["anthropic-us"] = us
    gateway.cfg.models.append(ModelConfig(name="claude-x", quality={"general": 4},
                                          routes=[Route(backend="anthropic-us", model="claude-x-1", price={"input": 3, "output": 15})]))
    r = await client.post("/v1/messages/count_tokens", json=body, headers=auth())
    assert r.json() == {"input_tokens": exact} and gateway.tokens.models["claude-x"].samples == 1
    gateway.cfg.team("default").allowed_backends = ["elsewhere"]  # not sent upstream: estimated here, calibrated
    r = await client.post("/v1/messages/count_tokens", json=body, headers=auth())
    est = estimate({k: body.get(k) for k in ("system", "messages", "tools")})
    assert r.json()["input_tokens"] == gateway.tokens.count("claude-x", est) > est.tokens


async def test_the_models_page_shows_each_models_ratio(gateway, client):
    for _ in range(4):
        gateway.tokens.record("big", Estimate(text=20_000), 22_000)
    models = {m["name"]: m for m in (await client.get("/v1/admin/config", headers=auth(ADMIN_KEY))).json()["models"]}
    big, small = models["big"]["tokens"], models["small"]["tokens"]
    assert big["source"] == "model" and big["samples"] == 4 and big["ratio"] == pytest.approx(1.08, abs=0.01) and big["spread"] is not None
    assert small == {"ratio": 1.0, "source": "none", "family": "small", "samples": 0, "spread": None, "margin": round(1 / 3, 3), "updated": None}


async def test_startup_loads_and_shutdown_saves_the_calibration(cfg, monkeypatch, caplog):
    real_sleep = asyncio.sleep

    async def fast(seconds):
        await real_sleep(0.005)

    monkeypatch.setattr(app_module, "asyncio", SimpleNamespace(create_task=asyncio.create_task, sleep=fast))
    store = MemoryStore()
    saved = Calibration()
    for _ in range(3):
        saved.record("small", Estimate(text=20_000), 26_000)
    await store.set_setting(T.SETTING, saved.to_dict())
    gw = build_gateway(cfg, store=store, state=MemoryState())
    app = create_app(gateway=gw)
    async with app.router.lifespan_context(app):
        assert gw.tokens.ratio("small") == pytest.approx(saved.ratio("small"))
        gw.tokens.record("big", Estimate(text=20_000), 21_000)
        for _ in range(100):
            if "big" in ((await store.get_setting(T.SETTING)) or {}).get("models", {}):
                break
            await real_sleep(0.005)
        assert "big" in (await store.get_setting(T.SETTING))["models"]  # the periodic save
        gw.tokens.record("big", Estimate(text=20_000), 21_000)
    assert (await store.get_setting(T.SETTING))["models"]["big"]["samples"] == 2  # saved on the way out

    class Broken(MemoryStore):
        async def get_setting(self, key):
            raise ConnectionError("postgres down")

    gw = build_gateway(cfg, store=Broken(), state=MemoryState())
    gw.tokens.record("big", Estimate(text=20_000), 21_000)
    app = create_app(gateway=gw)
    async with app.router.lifespan_context(app):
        await real_sleep(0.03)
    assert "could not load the token calibration" in caplog.text and "could not save the token calibration" in caplog.text
