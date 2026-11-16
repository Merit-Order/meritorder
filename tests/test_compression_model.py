"""Model-based compression of tool results (LLMLingua-2): the guard, the service, and the gateway's stored decisions,
which keep every later turn byte-identical on every gateway. A fake model stands in for LLMLingua-2: no weights."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
import sys
import time
import types

import httpx
import pytest
from conftest import ADMIN_KEY, auth, last_row
from test_affinity import add_gpu, publish, rep

from meritorder import compressor as cmp
from meritorder import compressor_service
from meritorder.app import build_gateway, create_app
from meritorder.compression import CompressionConfig, ModelCompressionConfig, compress_tool_results, model_candidates
from meritorder.compressor import (CompressorConfig, ModelCompressor, apply_cuts, cache_key, cuts_for, protected, segments,
                                   valid_cuts)
from meritorder.compressor_service import LLMLingua2, create_app as service_app, drops, parse_labels
from meritorder.config import build_config
from meritorder.state import MemoryState
from meritorder.store import SAVINGS_CAUSES

MODEL = "fake/llmlingua-2-xlm-roberta-large-test"
STOP = {"the", "a", "an", "of", "to", "that", "so", "and", "for", "it", "is", "in", "its", "them", "then", "every", "each"}

PROSE = (
    "The service reads its settings when it starts and keeps them for the whole run of the process.\n"
    "Most people never change them, because the defaults suit a small team that works in one place.\n"
    "When the team grows, the settings can move into the shared store so that every copy agrees.\n"
    "Each copy then checks the store every few seconds and picks up a change without a restart.\n"
    "This keeps the copies in step and means nobody has to log in to each machine to update them.\n"
)
DOC = PROSE + "def load(path):\n    return yaml.safe_load(open(path))\nError: config file not found at /etc/app/config.yaml\n"


class FakeEngine:
    """Splits words the way llmlingua reports them (words and punctuation; line breaks and digits forced) and drops
    the words ``drop`` picks. ``offset`` makes a second engine choose differently, like another model version."""

    version = "fake 1"
    threads = 1
    device = "cpu"

    def __init__(self, model: str = MODEL, offset: int = 0, delay: float = 0.0):
        self.model, self.offset, self.delay, self.calls = model, offset, delay, 0

    def labels(self, text, rate, force_tokens, force_digits):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        out = []
        for i, w in enumerate(re.findall(r"\w+|[^\w\s]|\n", text)):
            forced = w in force_tokens or (force_digits and any(c.isdigit() for c in w))
            dropped = w.lower() in STOP if not self.offset else (i + self.offset) % 3 == 0 and w.isalpha()
            out.append((w, 0 if dropped and not forced else 1))
        return out


def wire(gateway, engine: FakeEngine | None = None, transport=None, **model) -> ModelCompressor:
    """Turn on model-based compression for the default team, served by ``engine`` (in process) or ``transport``."""
    gateway.cfg.compressor = CompressorConfig(url="http://compressor.test", model=MODEL)
    gateway.cfg.team("default").compression = CompressionConfig(model=ModelCompressionConfig(enabled=True, min_chars=200, **model))
    transport = transport or httpx.ASGITransport(app=service_app(engine or FakeEngine()))
    gateway.compressor = ModelCompressor(gateway.cfg.compressor, gateway.state, gateway.metrics, transport=transport)
    return gateway.compressor


def spy(gateway, backend: str = "mock-anthropic") -> list[dict]:
    """Every body the backend is sent."""
    sent: list[dict] = []
    b = gateway.backends[backend]
    original = b.send

    async def send(body, stream, headers, **kw):
        sent.append(body)
        return await original(body, stream, headers, **kw)

    b.send = send
    return sent


def tool_result(text: str, tid: str, name: str = "WebFetch") -> list[dict]:
    return [{"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}]


def results_in(body: dict) -> list[str]:
    return [b["content"] for m in body["messages"] if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]


async def turn(client, messages, session="s1", model="big"):
    r = await client.post("/v1/messages", headers={**auth(), "x-session-id": session},
                          json={"model": model, "max_tokens": 50, "messages": messages})
    assert r.status_code == 200, r.text
    return r


FIRST = [{"role": "user", "content": "Read the docs and tell me how settings work. " * 20}, *tool_result(DOC, "t1")]
SECOND = FIRST + tool_result(PROSE.replace("settings", "options"), "t2")


# ------------------------------------------------------------------ the guard


def test_only_prose_goes_to_the_model():
    assert segments(DOC) == [(0, len(PROSE) - 1)]  # code, the error line and the path stay out
    fenced = PROSE + "```\n" + PROSE + "```\n"
    assert segments(fenced) == [(0, len(PROSE) - 1)]
    assert segments('{"text": "' + PROSE.replace("\n", " ") + '"}') == []  # JSON is left whole
    assert segments(PROSE[:200]) == []  # too short to bother
    assert segments("{ not json\n" + PROSE) == [(11, 11 + len(PROSE) - 1)]
    listing = "".join(f"{i:>6}\t{line}\n" for i, line in enumerate(PROSE.splitlines(), 1))
    assert segments(listing) == []  # a numbered file listing an agent may edit from
    table = "".join(f"| {line} | x |\n" for line in PROSE.splitlines())
    logs = "".join(f"2026-11-13 10:00:0{i} {line}\n" for i, line in enumerate(PROSE.splitlines()))
    yaml_ = "".join(f"key_{i}: {line}\n" for i, line in enumerate(PROSE.splitlines()))
    assert segments(table) == segments(logs) == segments(yaml_) == []
    # A blank line inside prose stays inside one range; a code line splits it.
    two = PROSE + "\n" + PROSE
    assert segments(two) == [(0, len(two) - 1)]
    error = "The build then raised a TypeError when it loaded the settings file again\n"
    split = PROSE + error + PROSE
    assert segments(split) == [(0, len(PROSE) - 1), (len(PROSE) + len(error), len(split) - 1)]  # an error line, any case


def test_protected_words_survive_whatever_the_model_says():
    text = 'Run the foo_bar job in /srv/app with fooBar and HTTP_PROXY set to 3 and "keep this quote" or `x y` now.'
    kept = {text[s:e] for s, e in protected(text, 0, len(text))}
    assert {"foo_bar", "/srv/app", "fooBar", "HTTP_PROXY", "3", '"keep this quote"', "`x y`"} <= kept
    assert "Run" not in kept and "job" not in kept
    negations = "It does not work, never did, and won't (without help); no."
    assert {negations[s:e] for s, e in protected(negations, 0, len(negations))} == {"not", "never", "won't", "(without", "no."}
    # Cut everything: only the protected words are left.
    words = [[m.start() - 0, m.end()] for m in re.finditer(r"\S+", text)]
    out = apply_cuts(text, cuts_for(text, [(0, len(text))], [words]))
    assert out.split() == ["foo_bar", "/srv/app", "fooBar", "HTTP_PROXY", "3", '"keep', "this", 'quote"', "`x", "y`"]
    # One cut over everything comes out the same: the protected words are taken back out of it.
    assert apply_cuts(text, cuts_for(text, [(0, len(text))], [[[0, len(text)]]])).split() == out.split()


def test_the_guard_keeps_what_the_real_models_cut():
    """What LLMLingua-2 cut on real tool outputs before the guard kept it (scripts/bench_compressor.py, docs/04)."""
    text = ("Start it with --device cpu or -d cuda; the flag --batch-size sets the chunks. The X-Session-Id header and "
            "claude-code use gpt-oss on llm-d, and the setting called “deadline” or 'off' (not a quote: the teams' "
            "settings, don't) applies to all of them, only once, with the same model at Merit Order and Redis.")
    kept = {text[s:e] for s, e in protected(text, 0, len(text))}
    assert {"--device", "-d", "--batch-size", "X-Session-Id", "claude-code", "gpt-oss", "“deadline”", "'off'", "all",
            "only", "same", "Merit", "Order", "Redis.", "don't)"} <= kept
    assert not {"Start", "The", "flag", "setting", "teams'", "settings,"} & kept  # a sentence's first word isn't a name
    assert not {text[s:e] for s, e in protected("- The list item. Then: Another one", 0, 34)} & {"The", "Then:", "Another"}
    # Lines an agent quotes back or edits from aren't prose: search results, test ids, compiler output, diffs, errors.
    line = "explains how the gateway keeps each session on the model it started with, so caches stay warm\n"
    for prefix in ("docs/03-routing.md:141:", "src\\router.py:12: ", "tests/test_x.py::test_y[a b c] PASSED ", "router.py:12:5: "):
        assert segments((prefix + line) * 6) == []
    diff = "diff --git a/d.md b/d.md\n@@ -1,6 +1,6 @@\n" + "".join(f"{c}{line}" for c in " -+ +-")
    assert segments(diff) == [] and segments(diff + "\n" + line * 4) == [(len(diff) + 1, len(diff) + len(line) * 4)]
    # The rules keep a huge diff's head and tail: the tail starts inside a hunk, with no @@ line before it.
    tail = diff[:60] + "[... 900 lines, 52000 characters omitted by Merit Order ...]\n" + "".join(f"{c}{line}" for c in "+-+ +-")
    assert segments(tail) == []
    assert segments(line * 3 + "No such file or directory while the gateway reads its settings\n" + line * 3) == []
    text = "alpha beta gamma.\ndelta epsilon, zeta\neta theta"
    at = lambda w: [text.index(w), text.index(w) + len(w)]  # noqa: E731
    assert apply_cuts(text, [at("beta")]) == "alpha gamma.\ndelta epsilon, zeta\neta theta"
    assert apply_cuts(text, [at("gamma")]) == "alpha beta.\ndelta epsilon, zeta\neta theta"  # no space before "."
    assert apply_cuts(text, [at("epsilon")]) == "alpha beta gamma.\ndelta, zeta\neta theta"
    assert apply_cuts(text, [[text.index("gamma"), text.index("epsilon") + 7]]) == "alpha beta\n, zeta\neta theta"
    assert apply_cuts(text, [at("alpha")]) == "beta gamma.\ndelta epsilon, zeta\neta theta"
    assert apply_cuts(text, [at("zeta")]) == "alpha beta gamma.\ndelta epsilon,\neta theta"
    assert apply_cuts(text, [at("theta")]) == "alpha beta gamma.\ndelta epsilon, zeta\neta"
    assert apply_cuts(text, [at("beta"), at("gamma")]) == "alpha.\ndelta epsilon, zeta\neta theta"  # joined into one cut
    assert apply_cuts(text, [at("delta"), at("epsilon"), at(", zeta")]) .count("\n") == text.count("\n")
    assert apply_cuts(text, []) == text
    assert apply_cuts("one-two three", [[3, 4]]) == "one two three"  # words a cut joined stay apart


def test_valid_cuts_and_keys():
    assert valid_cuts([], 5) and valid_cuts([[0, 2], [3, 5]], 5)
    assert not valid_cuts([[0, 6]], 5) and not valid_cuts([[2, 2]], 5) and not valid_cuts([[3, 4], [1, 2]], 5)
    assert not valid_cuts([[0, "1"]], 5) and not valid_cuts({"a": 1}, 5) and not valid_cuts([[0, 1, 2]], 5)
    k = cache_key("t", "m", 0.5, "x")
    assert k.startswith("cmp:") and k == cache_key("t", "m", 0.5, "x")
    assert len({k, cache_key("u", "m", 0.5, "x"), cache_key("t", "n", 0.5, "x"), cache_key("t", "m", 0.4, "x"),
                cache_key("t", "m", 0.5, "y")}) == 5
    assert cache_key("t", "m", 0.5, "\ud800") != k  # lone surrogates (valid JSON) hash too


def test_skipped_tools_and_short_results_are_not_candidates():
    cfg = CompressionConfig(model=ModelCompressionConfig(enabled=True, min_chars=200))
    body = {"messages": [{"role": "user", "content": PROSE}, *tool_result(PROSE, "a", "Read"), *tool_result("short", "b"),
                         *tool_result(PROSE, "c", "WebFetch")]}
    items, _ = model_candidates(body, "anthropic", cfg)
    assert items == [(PROSE, True)]  # the WebFetch result; Read is skipped, user text never looked at
    # Seen earlier in the history too: it went upstream as it is, so it counts as old.
    body["messages"] = [*tool_result(PROSE, "x"), *body["messages"]]
    assert model_candidates(body, "anthropic", cfg)[0] == [(PROSE, False)]
    assert model_candidates(body, "anthropic", CompressionConfig())[0] == []
    openai = {"messages": [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "fetch"}}]},
                           {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": PROSE}]}]}
    assert model_candidates(openai, "openai", cfg)[0] == [(PROSE, True)]
    out, res = compress_tool_results(openai, "openai", cfg, {PROSE: "short now"})
    assert out["messages"][1]["content"][0]["text"] == "short now" and res.model_chars_new == len(PROSE) - 9
    assert res.method == "model" and res.removed_tokens == res.model_tokens_new > 0


# ------------------------------------------------------------------ the service


def test_word_labels_are_matched_back_to_the_original_text():
    text = "Ｔhe ﬁne café 😀 naïve étude works... OK"
    # As a tokenizer reports them: normalized, lower case, the emoji left out, "..." as dots.
    words = [("the", 0), ("fine", 1), ("cafe", 0), ("naive", 0), ("etude", 0), ("works", 1), (".", 0), (".", 1), (".", 1),
             ("ok", 0), ("\n", 1), ("", 0)]
    got = [text[s:e] for s, e in drops(text, words)]
    assert got == ["Ｔhe", "café", "naïve", "étude", ".", "OK"]
    # A word that covers part of one character (… is three dots) is never dropped.
    assert drops("so… yes", [("so", 1), (".", 0), (".", 0), (".", 0), ("yes", 0)]) == [[4, 7]]
    # Lost: a word that isn't there stops the matching, and the rest stays as it is.
    assert drops("one two three", [("one", 0), ("zzz", 0), ("three", 0)]) == [[0, 3]]
    # Skipping plain ASCII words to find one is never allowed; skipping what a tokenizer couldn't read is.
    assert drops("alpha beta gamma", [("alpha", 1), ("gamma", 0)]) == []
    assert drops("alpha 😀x beta", [("alpha", 1), ("beta", 0)]) == [[9, 13]]
    assert drops("cafe\u0301 ok", [("cafe", 0), ("ok", 1)]) == [[0, 5]]  # the combining accent goes with its word


def test_parse_labels_reads_llmlingua_format():
    labelled = "\t\t|\t\t".join(["The 0", "fox 1", "\n 1", " 0", "end"])
    assert parse_labels(labelled) == [("The", 0), ("fox", 1), ("\n", 1), ("", 0), ("", 1)]
    assert parse_labels("") == []


async def test_service_api():
    engine = FakeEngine()
    app = service_app(engine, api_key="sekret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://c") as c:
        assert (await c.get("/healthz")).json() == {"ok": True, "model": MODEL, "version": "fake 1", "device": "cpu"}
        body = {"model": MODEL, "rate": 0.5, "force_tokens": ["\n", "."], "texts": [PROSE, "  "]}
        assert (await c.post("/v1/compress", json=body)).status_code == 401
        h = {"authorization": "Bearer sekret"}
        r = await c.post("/v1/compress", json=body, headers=h)
        assert r.status_code == 200 and r.json()["model"] == MODEL
        d = r.json()["results"]
        assert d[1] == {"drop": []} and {PROSE[s:e].lower() for s, e in d[0]["drop"]} <= STOP
        assert (await c.post("/v1/compress", json={**body, "model": "other"}, headers=h)).status_code == 409
        assert (await c.post("/v1/compress", json={**body, "rate": 0}, headers=h)).status_code == 422


async def test_service_sends_all_texts_through_the_model_at_once():
    class Batched(FakeEngine):
        def labels_many(self, texts, rate, force_tokens, force_digits):
            self.batches = getattr(self, "batches", []) + [list(texts)]
            return [self.labels(t, rate, force_tokens, force_digits) for t in texts]

    engine = Batched()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service_app(engine)), base_url="http://c") as c:
        r = await c.post("/v1/compress", json={"model": MODEL, "rate": 0.5, "texts": [PROSE, " ", DOC]})
    one = [drops(t, FakeEngine().labels(t, 0.5, ["\n"], True)) for t in (PROSE, DOC)]  # one text at a time
    assert engine.batches == [[PROSE, DOC]]
    assert [x["drop"] for x in r.json()["results"]] == [one[0], [], one[1]]


def fake_llmlingua(monkeypatch, calls: list):
    """llmlingua, torch and transformers as stand-ins with llmlingua 0.2.2's signatures, so the engine's calls are
    checked against them without the packages."""

    class PromptCompressor:
        def __init__(self, model_name="NousResearch/Llama-2-7b-hf", device_map="cuda", model_config={}, open_api_config={},
                     use_llmlingua2=False, llmlingua2_config={}):
            calls.append(("init", model_name, device_map, model_config, use_llmlingua2, llmlingua2_config))
            self.added_tokens = [f"[NEW{i}]" for i in range(100)]  # init_llmlingua2's
            # XLM-RoBERTa's tokenizer makes two pieces of a line break, so llmlingua swaps it for an added token.
            self.tokenizer = types.SimpleNamespace(tokenize=lambda t: ["▁", t] if t == "\n" else [t])

        def __chunk_context(self, origin_text, chunk_end_tokens):  # llmlingua's private method, name-mangled the same way
            calls.append(("chunk", origin_text, chunk_end_tokens))
            return origin_text.split("|")

        def __compress(self, context_list, reduce_rate=0.5, token_to_word="mean", force_tokens=[], token_map={},
                       force_reserve_digit=False, drop_consecutive=False):
            calls.append(("batch", context_list, reduce_rate, token_to_word, force_tokens, token_map, force_reserve_digit,
                          drop_consecutive))
            words = [[w.replace("[NEW0]", "\n") for chunk in chunks for w in chunk.split()] for chunks in context_list]
            return [], words, [[i % 2 for i in range(len(ws))] for ws in words]

        def compress_prompt(self, context, instruction="", question="", rate=0.5, target_token=-1, use_context_level_filter=True,
                            use_token_level_filter=True, return_word_label=False, word_sep="\t\t|\t\t", label_sep=" ",
                            token_to_word="mean", force_tokens=[], force_reserve_digit=False, drop_consecutive=False,
                            chunk_end_tokens=[".", "\n"], strict_preserve_uncompressed=True):
            calls.append(("compress", context, rate, force_tokens, force_reserve_digit, use_context_level_filter, return_word_label))
            words = context[0].split()
            return {"compressed_prompt": "", "fn_labeled_original_prompt": word_sep.join(f"{w}{label_sep}{i % 2}" for i, w in enumerate(words))}

    threads = [16]
    torch = types.SimpleNamespace(__version__="2.15.0", cuda=types.SimpleNamespace(is_available=lambda: False),
                                  backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
                                  get_num_threads=lambda: threads[0], set_num_threads=lambda n: threads.__setitem__(0, n))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.22.1"))
    monkeypatch.setitem(sys.modules, "llmlingua", types.SimpleNamespace(PromptCompressor=PromptCompressor, __version__="0.2.2"))


def test_engine_calls_llmlingua_the_way_its_api_expects(monkeypatch):
    calls: list = []
    fake_llmlingua(monkeypatch, calls)
    monkeypatch.setattr(compressor_service, "cpu_limit", lambda: None)
    engine = LLMLingua2("microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", revision="abc123")
    assert engine.device == "cpu" and engine.version == "llmlingua 0.2.2, transformers 5.22.1, torch 2.15.0, revision abc123"
    assert engine.threads == 16  # nothing limits it: torch's own choice
    monkeypatch.setattr(compressor_service, "cpu_limit", lambda: 2)
    assert LLMLingua2("microsoft/llmlingua-2-xlm-roberta-large-meetingbank").threads == 2  # a container's limit
    assert LLMLingua2("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", threads=3).threads == 3
    del calls[1:]
    assert calls[0] == ("init", engine.model, "cpu", {"trust_remote_code": False, "revision": "abc123"}, True,
                        {"max_batch_size": 50, "max_force_token": 100})
    assert engine.labels("one two three", 0.4, ["\n"], True) == [("one", 0), ("two", 1), ("three", 0)]
    assert calls[1] == ("compress", ["one two three"], 0.4, ["\n"], True, False, True)
    # Several texts in one pass: llmlingua's own steps, with the line break mapped to its added token for chunking.
    calls.clear()
    got = engine.labels_many(["one two|three", "four \n five"], 0.4, ["\n", "."], True)
    assert got == [[("one", 0), ("two", 1), ("three", 0)], [("four", 0), ("\n", 1), ("five", 0)]]
    ends = {".", "\n", "[NEW0]"}
    assert calls[:2] == [("chunk", "one two|three", ends), ("chunk", "four [NEW0] five", ends)]
    _, chunks, reduce_rate, *rest = calls[2]
    assert chunks == [["one two", "three"], ["four [NEW0] five"]] and reduce_rate == pytest.approx(0.6)
    assert rest == ["mean", ["\n", "."], {"\n": "[NEW0]"}, True, False]
    with pytest.raises(SystemExit, match="xlm-roberta-large"):
        LLMLingua2("some/other-model")


def test_cpu_limit_follows_the_container(tmp_path, monkeypatch):
    monkeypatch.setattr(compressor_service.os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
    monkeypatch.setattr(compressor_service.os, "cpu_count", lambda: 64)
    assert compressor_service.cpu_limit(str(tmp_path)) is None  # no cgroup files: nothing limits it
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert compressor_service.cpu_limit(str(tmp_path)) is None
    (tmp_path / "cpu.max").write_text("150000 100000\n")  # cgroup v2: 1.5 CPUs
    assert compressor_service.cpu_limit(str(tmp_path)) == 2
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("400000\n")  # cgroup v1: 4 CPUs
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert compressor_service.cpu_limit(str(tmp_path)) == 4
    monkeypatch.setattr(compressor_service.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)  # taskset, cpuset
    assert compressor_service.cpu_limit(str(tmp_path)) == 3
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    assert compressor_service.cpu_limit(str(tmp_path)) == 3


def test_engine_says_which_extra_it_needs(monkeypatch):
    monkeypatch.setitem(sys.modules, "llmlingua", None)
    with pytest.raises(SystemExit, match="compressor extra"):
        LLMLingua2("microsoft/llmlingua-2-xlm-roberta-large-meetingbank")


def test_serve_loads_the_model_then_listens(monkeypatch):
    import uvicorn

    from meritorder import cli, compressor_service

    seen = {}
    monkeypatch.setattr(compressor_service, "LLMLingua2", lambda *a: (seen.setdefault("engine", a), FakeEngine())[1])
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    monkeypatch.setenv("MERITORDER_COMPRESSOR_KEY", "k")
    cli.main(["compressor", "--port", "9999", "--device", "cpu", "--model", MODEL])
    assert seen["engine"] == (MODEL, "cpu", None, 50, 0) and seen["port"] == 9999 and seen["host"] == "127.0.0.1"


# ------------------------------------------------------------------ the gateway


async def test_compressed_once_then_byte_identical_on_every_later_turn(client, gateway):
    engine = FakeEngine()
    wire(gateway, engine)
    sent = spy(gateway)
    await turn(client, FIRST)
    first = results_in(sent[-1])[0]
    assert first != DOC and len(first) < len(DOC)
    assert "def load(path):\n    return yaml.safe_load(open(path))\nError: config file not found at /etc/app/config.yaml" in first
    assert FIRST[0]["content"] in [m["content"] for m in sent[-1]["messages"]]  # user text untouched
    await gateway.drain()
    row = await last_row(gateway)
    assert row["tags"]["compression"] == "model" and row["savings_compression_model"] > 0 and row["savings_compression"] == 0
    calls = engine.calls
    await turn(client, SECOND)
    assert results_in(sent[-1])[0] == first  # the same bytes: the provider's cache still matches
    assert engine.calls == calls + 1  # only the new result went to the model
    await turn(client, SECOND)
    assert results_in(sent[-1]) == results_in(sent[-2]) and engine.calls == calls + 1
    text = (await client.get("/metrics")).text
    assert 'meritorder_savings_usd_total{cause="compression_model"}' in text
    assert 'meritorder_compression_model_results_total{result="reused"} 3' in text
    assert "meritorder_compressor_duration_seconds_count" in text


async def test_two_gateways_send_the_same_bytes(client, gateway):
    """The second gateway's model would cut differently; it uses the stored decision instead."""
    wire(gateway, FakeEngine())
    other = build_gateway(gateway.cfg, store=gateway.store, state=gateway.state)
    other_engine = FakeEngine(offset=1)
    other.compressor = ModelCompressor(gateway.cfg.compressor, other.state, other.metrics,
                                       transport=httpx.ASGITransport(app=service_app(other_engine)))
    sent_a, sent_b = spy(gateway), spy(other)
    await turn(client, FIRST)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=other)), base_url="http://test") as c2:
        await turn(c2, SECOND)
        await turn(client, SECOND + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "thanks"}])
    a1, b2, a3 = results_in(sent_a[0]), results_in(sent_b[0]), results_in(sent_a[1])
    assert b2[0] == a1[0]  # gateway B applied A's cuts to the first result
    assert a3 == b2  # and A applies B's to the second
    # B's own model really does cut differently (so the match isn't luck).
    alone = ModelCompressor(gateway.cfg.compressor, MemoryState(), transport=httpx.ASGITransport(app=service_app(FakeEngine(offset=1))))
    mine, _ = await alone.compress("default", gateway.cfg.team("default").compression.model, [(DOC, True)])
    assert mine[DOC] != a1[0]
    await other.drain()
    await other.compressor.close()
    await alone.close()


async def test_racing_gateways_both_send_the_winners_cuts(gateway):
    a = wire(gateway, FakeEngine())
    b = ModelCompressor(gateway.cfg.compressor, gateway.state, transport=httpx.ASGITransport(app=service_app(FakeEngine(offset=2))))
    settings = gateway.cfg.team("default").compression.model
    (ra, _), (rb, _) = await asyncio.gather(a.compress("default", settings, [(DOC, True)]), b.compress("default", settings, [(DOC, True)]))
    assert ra[DOC] == rb[DOC]
    await b.close()


async def test_late_results_stay_uncompressed_everywhere(client, gateway):
    async def slow(request):
        await asyncio.sleep(0.5)
        return httpx.Response(500)

    wire(gateway, transport=httpx.MockTransport(slow), deadline_ms=20)
    sent = spy(gateway)
    await turn(client, FIRST)
    assert results_in(sent[-1])[0] == DOC  # not ready in time: sent as it is
    # Next turn, the service is quick again (or it's another gateway's): the late result still goes as it was.
    wire(gateway, FakeEngine())
    await turn(client, SECOND)
    first, second = results_in(sent[-1])
    assert first == DOC and second != PROSE.replace("settings", "options")
    text = (await client.get("/metrics")).text
    assert 'meritorder_compression_model_results_total{result="late"} 1' in text


async def test_old_results_seen_first_are_left_as_sent(client, gateway):
    """Compression turned on mid-session: what earlier turns sent as it is stays that way."""
    sent = spy(gateway)
    await turn(client, FIRST)
    wire(gateway, FakeEngine())
    await turn(client, SECOND)
    first, second = results_in(sent[-1])
    assert first == DOC and second != PROSE.replace("settings", "options")
    await turn(client, SECOND)
    assert results_in(sent[-1]) == [first, second]


async def test_a_failing_service_is_left_alone_for_a_while(client, gateway):
    calls = []

    def broken(request):
        calls.append(request)
        return httpx.Response(503, text="loading")

    comp = wire(gateway, transport=httpx.MockTransport(broken))
    sent = spy(gateway)
    await turn(client, FIRST)
    assert results_in(sent[-1])[0] == DOC and comp.last_error.startswith("CompressorError: HTTP 503")
    await turn(client, SECOND)  # within RETRY_SECONDS: not called again
    assert len(calls) == 1 and results_in(sent[-1]) == [DOC, PROSE.replace("settings", "options")]
    comp.retry_at = 0
    calls.clear()
    wrong = lambda r: httpx.Response(200, json={"model": "other", "results": []})  # noqa: E731
    comp.client._transport = httpx.MockTransport(wrong)
    third = SECOND + tool_result(PROSE.replace("settings", "values"), "t3")
    await turn(client, third)
    assert "runs 'other'" in comp.last_error
    for reply in ({"model": MODEL, "results": []}, {"model": MODEL, "results": [{"drop": [[0, 10_000]]}]}):
        comp.retry_at = 0
        comp.client._transport = httpx.MockTransport(lambda r, reply=reply: httpx.Response(200, json=reply))
        third = third + tool_result(PROSE + str(len(third)), f"t{len(third)}")
        await turn(client, third)
        assert results_in(sent[-1])[-1] == PROSE + str(len(third) - 2)
    text = (await client.get("/metrics")).text
    assert 'meritorder_compression_model_results_total{result="failed"}' in text


async def test_json_nested_too_deep_to_parse_is_left_whole(client, gateway):
    """A tool result nested deeper than Python's JSON parser goes (a minified dump, or a page written to break agents)
    is data, not prose: it's sent as it is, and the request goes through (it used to fail with a RecursionError, and so
    did every later turn that carried it in its history)."""
    wire(gateway, FakeEngine())
    sent = spy(gateway)
    deep = "[" * 3000 + "]" * 3000
    assert segments(deep) == segments("{" * 3000) == []
    await turn(client, [{"role": "user", "content": "go"}, *tool_result(deep, "t1")])
    assert results_in(sent[-1]) == [deep]


async def test_half_an_emoji_is_compressed_and_never_blamed_on_the_service(gateway):
    """JSON can carry a lone surrogate (a harness cut a string between the two UTF-16 halves of an emoji). It reaches the
    service escaped, so the text is compressed. It used to fail to encode before the request was sent, which counted as
    the service failing: model-based compression then stopped for everyone on the gateway for RETRY_SECONDS."""
    comp = wire(gateway, FakeEngine())
    settings = gateway.cfg.team("default").compression.model
    text = PROSE.replace("small team", "small \ud83d team")
    out, outcomes = await comp.compress("default", settings, [(text, True)])
    assert outcomes["compressed"] == 1 and comp.retry_at == 0 and comp.last_error is None
    assert len(out[text]) < len(text) and "\ud83d" in out[text]


async def test_store_errors_send_results_as_they_are(gateway):
    comp = wire(gateway, FakeEngine())
    settings = gateway.cfg.team("default").compression.model

    async def boom(*a, **k):
        raise ConnectionError("redis is down")

    comp.state = types.SimpleNamespace(get_json_many=boom)
    out, outcomes = await comp.compress("default", settings, [(DOC, True)])
    assert out == {} and outcomes["store_error"] == 1
    comp.state = types.SimpleNamespace(get_json_many=gateway.state.get_json_many, set_json_once=boom)
    out, outcomes = await comp.compress("default", settings, [(DOC, True), ("no prose here", True)])
    assert out == {} and outcomes["store_error"] == 1 and outcomes["no_prose"] == 1
    assert await comp.compress("default", settings, [("{}", True)]) == ({}, {"no_prose": 1})


async def test_nothing_to_cut_is_remembered_too(gateway):
    comp = wire(gateway, FakeEngine(offset=-1))  # drops nothing: every word is kept
    settings = gateway.cfg.team("default").compression.model
    comp_engine = FakeEngine()
    comp_engine.labels = lambda text, *a: [(w, 1) for w in text.split()]
    comp.client._transport = httpx.ASGITransport(app=service_app(comp_engine))
    out, outcomes = await comp.compress("default", settings, [(DOC, True)])
    assert out == {} and outcomes["kept"] == 1
    assert (await gateway.state.get_json(cache_key("default", MODEL, 0.5, DOC))) == {"s": "kept"}


async def test_gpu_served_models_get_it_too(client, gateway):
    add_gpu(gateway, replicas=1)
    await publish(gateway, rep(1))
    wire(gateway, FakeEngine())
    sent = spy(gateway, "gpu")
    messages = [{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "fetch", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": DOC}]
    r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "gpu-only", "messages": messages})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "gpu"
    assert len(sent[-1]["messages"][-1]["content"]) < len(DOC)
    row = await last_row(gateway)
    assert row["tags"]["compression"] == "model" and row["savings_compression_model"] > 0


async def test_rules_and_model_savings_are_counted_apart(client, gateway):
    """Each method is its own cause: the rules' cut, then the model's cut of what they left, never one in the other."""
    wire(gateway, FakeEngine())
    gateway.cfg.team("default").compression.enabled = True
    gateway.cfg.team("default").compression.min_chars = 200
    noisy = "\n".join(line + "   " for line in PROSE.splitlines()) + "\n" + "progress 50%\n" * 20  # trailing spaces, repeats
    await turn(client, [{"role": "user", "content": "go"}, *tool_result(noisy, "t1")])
    row = await last_row(gateway)
    assert row["savings_compression"] > 0 and row["savings_compression_model"] > 0 and row["tags"]["compression"] == "model"
    assert row["savings_total"] == pytest.approx(sum(row[f"savings_{c}"] for c in SAVINGS_CAUSES))
    assert row["baseline_cost"] - row["actual_cost"] == pytest.approx(row["savings_total"], abs=1e-9)
    # Reports, CSV and rollups carry the split.
    await gateway.store.rollup(full=True)
    for days in (1, 40):  # a short range reads the ledger, a long one the hourly rollup (Postgres)
        now = datetime.now(UTC)
        total = (await gateway.store.summary(now - timedelta(days=days), now + timedelta(days=1), "none"))[0]
        assert total["savings_compression_model"] == pytest.approx(row["savings_compression_model"])
        assert total["savings_compression"] == pytest.approx(row["savings_compression"])
    csv_ = await client.get("/v1/reports/summary.csv?group_by=team", headers=auth(ADMIN_KEY))
    assert "savings_compression_model" in csv_.text.splitlines()[0]
    text = (await client.get("/metrics")).text
    metric = {line.split()[0]: float(line.split()[1]) for line in text.splitlines() if line.startswith("meritorder_savings_usd_total")}
    assert metric['meritorder_savings_usd_total{cause="compression_model"}'] == pytest.approx(row["savings_compression_model"], abs=1e-6)
    assert metric['meritorder_savings_usd_total{cause="compression"}'] == pytest.approx(row["savings_compression"], abs=1e-6)


async def test_clients_cannot_set_the_compression_label(client, gateway):
    r = await client.post("/v1/chat/completions", headers={**auth(), "x-meritorder-tags": "compression=model,project=x"},
                          json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert (await last_row(gateway))["tags"] == {"project": "x"}


async def test_rules_alone_are_labelled_rules(client, gateway):
    gateway.cfg.team("default").compression = CompressionConfig(enabled=True, min_chars=200, max_chars=1000)
    messages = [{"role": "user", "content": "go"}, *tool_result("result line\n" * 500, "t1")]
    await turn(client, messages)
    row = await last_row(gateway)
    assert row["tags"]["compression"] == "rules" and row["savings_compression"] > 0


async def test_admin_sees_whether_the_service_answers(client, gateway):
    h = auth(ADMIN_KEY)
    assert (await client.get("/v1/admin/compressor", headers=h)).json() == {"configured": False, "teams": []}
    wire(gateway, FakeEngine())
    got = (await client.get("/v1/admin/compressor", headers=h)).json()
    assert got["configured"] and got["reachable"] and got["matches"] and got["teams"] == ["default"] and got["loaded"] == MODEL
    gateway.compressor.client._transport = httpx.MockTransport(lambda r: httpx.Response(502))
    got = (await client.get("/v1/admin/compressor", headers=h)).json()
    assert got["reachable"] is False and "502" in got["error"]
    assert (await client.get("/v1/admin/compressor", headers=auth())).status_code == 403


async def test_a_new_compressor_section_replaces_the_client(gateway):
    from meritorder.config import Config

    raw = gateway.cfg.model_dump()
    raw["compressor"] = {"url": "http://elsewhere:8090", "api_key_env": "MO_TEST_KEY"}
    old = gateway.compressor
    await gateway.apply_config(Config.model_validate(raw), None)
    assert gateway.compressor is not old and gateway.compressor.cfg.url == "http://elsewhere:8090"
    await gateway.compressor.close()


def test_config_needs_a_service_for_model_compression():
    raw = {"backends": {"m": {"kind": "mock", "format": "openai"}},
           "models": {"x": {"routes": [{"backend": "m", "model": "x", "price": {"input": 1}}]}},
           "teams": {"default": {"compression": {"model": {"enabled": True}}}}}
    with pytest.raises(ValueError, match="compressor.url"):
        build_config(raw, quiet=True)
    cfg = build_config({**raw, "compressor": {"url": " http://c:8090 "}}, quiet=True)
    assert cfg.compressor.url == "http://c:8090" and cfg.compressor.ttl_seconds == 6 * 3600
    with pytest.raises(ValueError, match="ttl_seconds"):
        build_config({**raw, "compressor": {"url": "http://c", "ttl_seconds": 60}}, quiet=True)


async def test_state_stores_once_and_renews(state):
    assert await state.get_json_many(["cmp:a", "cmp:b"], 60) == [None, None]
    assert await state.set_json_once({"cmp:a": {"d": [[0, 1]]}}, 60) == {"cmp:a": {"d": [[0, 1]]}}
    assert await state.set_json_once({"cmp:a": {"s": "late"}, "cmp:b": {"s": "old"}}, 60) == {"cmp:a": {"d": [[0, 1]]}, "cmp:b": {"s": "old"}}
    assert await state.get_json_many(["cmp:a", "cmp:b", "cmp:c"], 60) == [{"d": [[0, 1]]}, {"s": "old"}, None]


def test_prose_ranges_are_remembered_per_process(monkeypatch):
    monkeypatch.setattr(cmp, "SEEN", 2)
    comp = ModelCompressor(CompressorConfig(url="http://c", model=MODEL), MemoryState())
    assert comp.segments("a", PROSE) == comp.segments("a", "ignored: remembered") == [(0, len(PROSE) - 1)]
    comp.segments("b", "x")
    comp.segments("a", PROSE)  # used again: kept over b
    comp.segments("c", "y")
    assert list(comp.seen) == ["a", "c"]
    assert cmp.FORCE_TOKENS[0] == "\n" and cmp.MIN_SEGMENT > 0 and cmp.VERSION
