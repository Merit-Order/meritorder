"""``meritorder compressor``: LLMLingua-2 behind a small HTTP API, for model-based compression (docs/04).

Gateways call it with the prose of new tool results; it answers with the character ranges LLMLingua-2 would drop.
It never sends text back, and the gateways do the cutting, so what they store (where to cut) is enough to send the
same bytes on every later turn.

API (any server that speaks it can stand in):

    POST /v1/compress  {"model": "...", "rate": 0.5, "force_tokens": ["\\n", "."], "force_digits": true, "texts": ["..."]}
                    -> {"model": "...", "version": "...", "results": [{"drop": [[start, end], ...]}, ...]}
    GET  /healthz   -> {"ok": true, "model": "...", "version": "...", "device": "cpu"}

Offsets are Python string indices (code points) into each text, sorted and not overlapping. A request for another
model than the one loaded gets 409, so a gateway never stores one model's cuts under another's name.

Needs the ``compressor`` extra (llmlingua, torch, transformers). It follows llmlingua 0.2.2 (April 2024, the latest
release in November 2026): ``PromptCompressor(model_name, device_map, use_llmlingua2=True)`` and
``compress_prompt(..., return_word_label=True)``, whose ``fn_labeled_original_prompt`` lists each word the model saw
with 1 (keep) or 0 (drop). Its own ``compressed_prompt`` is rebuilt from the tokenizer's pieces (it puts spaces into
``1.2.3`` and loses characters the tokenizer doesn't know), so the words are matched back to the original text here.
A request's texts go through the model together (``LLMLingua2.labels_many``), the same steps one text at a time takes.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

log = logging.getLogger("meritorder.compressor_service")

WINDOW = 48  # how far ahead a word may be found when characters the tokenizer couldn't represent sit before it
WORD_SEP = "\t\t|\t\t"  # llmlingua's defaults for fn_labeled_original_prompt
LABEL_SEP = " "
# llmlingua 0.2.2 picks its token rules by these substrings of the model name (utils.is_begin_of_new_word), so a local
# path must keep them, e.g. /models/llmlingua-2-xlm-roberta-large-meetingbank.
FAMILIES = ("xlm-roberta-large", "bert-base-multilingual-cased")


class Engine(Protocol):
    model: str
    version: str
    device: str

    def labels(self, text: str, rate: float, force_tokens: list[str], force_digits: bool) -> list[tuple[str, int]]:
        """Each word the model saw in ``text``, in order, with 1 (keep) or 0 (drop). An engine may also have
        ``labels_many(texts, ...)``, the same for several texts in one pass, which the app then uses."""


def parse_labels(labelled: str) -> list[tuple[str, int]]:
    """llmlingua's ``fn_labeled_original_prompt``: ``word label`` pairs joined by WORD_SEP."""
    out = []
    for part in labelled.split(WORD_SEP) if labelled else []:
        word, _, label = part.rpartition(LABEL_SEP)
        out.append((word, 1 if label.strip() != "0" else 0))
    return out


def _loose(s: str) -> str:
    """Text as the tokenizers may have changed it: compatibility forms (ﬁ, full-width), no accents, any case."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def drops(text: str, words: list[tuple[str, int]]) -> list[list[int]]:
    """Where in ``text`` the words labelled 0 are, as [start, end) ranges. The words come from the tokenizer, which may
    have normalized characters, changed case, or skipped characters it can't represent, so they're matched loosely,
    in order. A word that can't be found stops the matching: the rest of the text stays as it is. A word that covers
    only part of a character (… is three dots) is never dropped."""
    loose = [_loose(c) for c in text]
    back: list[int] = []  # for each character of the normalized text, the original character it came from
    for i, piece in enumerate(loose):
        back.extend([i] * len(piece))
    norm = "".join(loose)
    n, size = len(text), len(norm)
    out: list[list[int]] = []
    j = 0
    for word, label in words:
        w = _loose(word)
        if not w.strip():
            continue  # forced line breaks, and the empty words llmlingua makes of some markers
        while j < size and norm[j].isspace():
            j += 1
        k = j if norm.startswith(w, j) else norm.find(w, j + 1, min(size, j + WINDOW + len(w)))
        # What was skipped to get there must be what a tokenizer leaves out: characters it doesn't know (XLM-R skips
        # them), or whole words with one in them (BERT's [UNK]). Plain ASCII words are never skipped.
        if k < 0 or (k > j and any(run.isascii() for run in text[back[j]:back[k]].split())):
            break
        e = k + len(w)
        whole = (k == 0 or back[k - 1] != back[k]) and (e == size or back[e] != back[e - 1])
        if label == 0 and whole:
            start, end = back[k], back[e - 1] + 1
            while end < n and not loose[end] and not text[end].isspace():
                end += 1  # combining marks and zero-width characters go with the word before them
            out.append([start, end])
        j = e
    return out


def cpu_limit(cgroup: str = "/sys/fs/cgroup") -> int | None:
    """How many CPUs a container lets this process use (cgroup v2's cpu.max, else v1's CFS quota), or the CPUs it's
    pinned to when fewer than the machine has; None when nothing limits it. torch sizes its thread pool by the machine's
    cores, so in a pod limited to 2 CPUs on a 32-core node it runs 32 threads and is throttled most of every period."""
    limits = []
    try:
        quota, period = open(os.path.join(cgroup, "cpu.max")).read().split()[:2]
        if quota != "max":
            limits.append(math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        try:
            quota_us = int(open(os.path.join(cgroup, "cpu", "cpu.cfs_quota_us")).read())
            period_us = int(open(os.path.join(cgroup, "cpu", "cpu.cfs_period_us")).read())
            if quota_us > 0:
                limits.append(math.ceil(quota_us / period_us))
        except (OSError, ValueError):
            pass
    pinned = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if pinned and pinned < (os.cpu_count() or pinned + 1):
        limits.append(pinned)
    return max(1, min(limits)) if limits else None


class LLMLingua2:
    """LLMLingua-2 through llmlingua's PromptCompressor, one request at a time (its tokenizer isn't safe to share across
    threads, so the app runs it on one worker thread)."""

    def __init__(self, model: str, device: str = "auto", revision: str | None = None, batch_size: int = 50, threads: int = 0) -> None:
        if not any(f in model for f in FAMILIES):
            raise SystemExit(f"llmlingua only knows LLMLingua-2 models whose name contains {' or '.join(FAMILIES)}: {model!r}")
        try:
            import torch
            import transformers
            from llmlingua import PromptCompressor
            from llmlingua import __version__ as llmlingua_version
        except ImportError as e:
            raise SystemExit("the compressor needs the compressor extra (the -compressor image, or `uv sync --extra compressor` "
                             f"in a clone): {e}") from None
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cpu" and (threads or cpu_limit()):
            torch.set_num_threads(threads or cpu_limit())
        self.threads = torch.get_num_threads()
        config = {"trust_remote_code": False, **({"revision": revision} if revision else {})}  # the published models need no code
        self.compressor = PromptCompressor(model_name=model, device_map=device, model_config=config, use_llmlingua2=True,
                                           llmlingua2_config={"max_batch_size": batch_size, "max_force_token": 100})
        self.model, self.device = model, device
        self.version = (f"llmlingua {llmlingua_version}, transformers {transformers.__version__}, torch {torch.__version__}"
                        + (f", revision {revision}" if revision else ""))

    def labels(self, text: str, rate: float, force_tokens: list[str], force_digits: bool) -> list[tuple[str, int]]:
        res = self.compressor.compress_prompt([text], rate=rate, force_tokens=force_tokens, force_reserve_digit=force_digits,
                                              drop_consecutive=False, use_context_level_filter=False, return_word_label=True,
                                              word_sep=WORD_SEP, label_sep=LABEL_SEP)
        return parse_labels(res.get("fn_labeled_original_prompt") or "")

    def labels_many(self, texts: list[str], rate: float, force_tokens: list[str], force_digits: bool) -> list[list[tuple[str, int]]]:
        """``labels`` for several texts in one pass through the model. A gateway sends each prose range of a tool result
        as its own text, and each becomes at least one 512-token chunk; one call per text runs the model once per chunk,
        while this puts every text's chunks in the same batches (up to --batch-size at once): 1.7 times faster on a CPU
        for 24 short ranges, more on a GPU. The labels are the same, word for word: each chunk keeps its own threshold.

        These are the steps of llmlingua 0.2.2's ``compress_prompt_llmlingua2`` (force tokens mapped, text chunked, then
        ``__compress``), which returns each text's words and labels; its public result joins every text's words into one
        list, which can't be split back. Private methods, so this follows the pinned version only (the extra pins it)."""
        pc = self.compressor
        token_map = {t: pc.added_tokens[i] for i, t in enumerate(force_tokens) if len(pc.tokenizer.tokenize(t)) != 1}
        ends = {".", "\n"} | {token_map[c] for c in (".", "\n") if c in token_map}
        chunked = []
        for text in texts:
            for original, added in token_map.items():
                text = text.replace(original, added)
            chunked.append(pc._PromptCompressor__chunk_context(text, chunk_end_tokens=ends))
        _, words, labels = pc._PromptCompressor__compress(chunked, reduce_rate=max(0, 1 - rate), token_to_word="mean",
                                                           force_tokens=force_tokens, token_map=token_map,
                                                           force_reserve_digit=force_digits, drop_consecutive=False)
        return [[(w, 1 if label else 0) for w, label in zip(ws, ls)] for ws, ls in zip(words, labels)]


class CompressRequest(BaseModel):
    model: str | None = None
    rate: float = Field(gt=0, le=1)
    force_tokens: list[str] = Field(default=["\n"], max_length=100)  # llmlingua keeps at most 100
    force_digits: bool = True
    texts: list[str]


def create_app(engine: Engine, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="Merit Order compressor")
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmlingua")  # one at a time; run replicas to scale

    def check(request: Request) -> None:
        if api_key and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {api_key}"):
            raise HTTPException(401, "a valid bearer token is required")

    def run(req: CompressRequest) -> list[dict]:
        texts = [t for t in req.texts if t.strip()]
        many = getattr(engine, "labels_many", None)  # all texts at once, when the engine can (LLMLingua2 can)
        if many is not None and texts:
            labelled = iter(many(texts, req.rate, req.force_tokens, req.force_digits))
        else:
            labelled = iter([engine.labels(t, req.rate, req.force_tokens, req.force_digits) for t in texts])
        return [{"drop": drops(text, next(labelled)) if text.strip() else []} for text in req.texts]

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "model": engine.model, "version": engine.version, "device": engine.device}

    @app.post("/v1/compress")
    async def compress(req: CompressRequest, request: Request):
        check(request)
        if req.model and req.model != engine.model:
            raise HTTPException(409, f"this compressor runs {engine.model}, not {req.model}")
        results = await asyncio.get_running_loop().run_in_executor(worker, run, req)
        return {"model": engine.model, "version": engine.version, "results": results}

    return app


def serve(host: str, port: int, model: str, device: str, revision: str | None, batch_size: int, log_level: str,
          threads: int = 0) -> None:
    """Load the model, then answer gateways until stopped."""
    import uvicorn

    engine = LLMLingua2(model, device, revision, batch_size, threads)
    log.info("compressor ready: %s on %s, %d threads (%s)", engine.model, engine.device, engine.threads, engine.version)
    uvicorn.run(create_app(engine, os.environ.get("MERITORDER_COMPRESSOR_KEY") or None), host=host, port=port,
                log_level=log_level.lower())
