"""Compression of large tool outputs (docs/04-caching-and-sessions.md).

Rules:
* Only tool results, only when they're large. Never user or assistant text.
* The rules here are deterministic: the same input always produces the same output, so applying them
  to every tool result in the history on every turn keeps the prompt prefix byte-stable and provider
  caches keep hitting. Nothing is stored.
* Model-based compression (LLMLingua-2, compressor.py) runs on top of the rules. A model isn't
  deterministic across versions, so each result is compressed once and what to drop is kept in the
  shared store by content hash; this module applies it the same way on every turn.
* Changing the settings mid-session changes the history once (one cache miss).
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .tokens import text_tokens

# Linear time on hostile input (these run on the event loop): an OSC sequence stops at the next ESC, and
# trailing whitespace is stripped per line rather than with a backtracking regex.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_BLANKS = re.compile(r"\n{3,}")

# Tools whose output an agent quotes back exactly (to edit a file it read), so a model never shortens it. Claude Code's
# Read prints line numbers, which the guard keeps anyway; these cover harnesses whose file reads don't.
SKIP_TOOLS = ["read", "read_file", "readfile", "view", "view_file", "open_file", "notebookread", "str_replace_based_edit_tool"]


class ModelCompressionConfig(BaseModel):
    """Model-based compression (LLMLingua-2) of tool results, on top of the rules. Needs the compressor service
    (the top-level ``compressor`` section). Only prose is sent: code, paths, identifiers, numbers, error lines,
    tables and JSON stay as they are (compressor.py)."""

    enabled: bool = False
    rate: float = Field(default=0.5, gt=0, lt=1)  # share of the prose's tokens to keep (llmlingua's own default)
    min_chars: int = Field(default=4000, ge=0)  # tool results shorter than this (after the rules) aren't sent to the model
    # A new tool result that isn't compressed by then goes uncompressed, and stays that way on every later turn.
    deadline_ms: int = Field(default=1000, gt=0)
    skip_tools: list[str] = SKIP_TOOLS  # results of these tools are never sent (glob patterns, any case)

    def skips(self, tool: str | None) -> bool:
        name = (tool or "").lower()
        return bool(name) and any(fnmatch.fnmatchcase(name, p.lower()) for p in self.skip_tools)


class CompressionConfig(BaseModel):
    enabled: bool = False  # the deterministic rules below
    min_chars: int = 4000  # leave anything shorter alone
    max_chars: int = 24000  # beyond this, keep the head and tail
    head_fraction: float = 0.6
    model: ModelCompressionConfig = ModelCompressionConfig()


@dataclass
class Result:
    """What compression took out of one request, split into the newest messages and older history, and by method."""

    removed_chars_new: int = 0  # by the rules, from tool results since the last assistant message (billed as fresh input)
    removed_chars_history: int = 0  # by the rules, from earlier tool results (would mostly be cache reads)
    removed_tokens_new: int = 0  # the same, as estimated tokens (meritorder.tokens)
    removed_tokens_history: int = 0
    model_chars_new: int = 0  # by model-based compression, on top of the rules
    model_chars_history: int = 0
    model_tokens_new: int = 0
    model_tokens_history: int = 0

    @property
    def removed_tokens(self) -> int:
        """Every token compression took out, by either method: what the upstream prompt is smaller by."""
        return self.removed_tokens_new + self.removed_tokens_history + self.model_tokens_new + self.model_tokens_history

    @property
    def method(self) -> str | None:
        """How the request was compressed, for the ledger's ``compression`` tag: model when the model removed anything."""
        if self.model_chars_new or self.model_chars_history:
            return "model"
        if self.removed_chars_new or self.removed_chars_history:
            return "rules"
        return None

    def add(self, newest: bool, chars: int, tokens: float, model: bool = False) -> None:
        if model:
            if newest:
                self.model_chars_new += chars
                self.model_tokens_new += round(tokens)
            else:
                self.model_chars_history += chars
                self.model_tokens_history += round(tokens)
        elif newest:
            self.removed_chars_new += chars
            self.removed_tokens_new += round(tokens)
        else:
            self.removed_chars_history += chars
            self.removed_tokens_history += round(tokens)


# The rules are deterministic and an agent resends its history every turn, so each long result's output is kept per
# process (about 0.15 ms saved per 10 KB result and turn), by the text and the settings that shaped it.
REMEMBERED_CHARS = 16_000_000  # how much text (in and out) is kept; the oldest goes first
_remembered: dict[tuple[str, int, int, float], str] = {}
_remembered_chars = 0


def compress_text(text: str, cfg: CompressionConfig) -> str:
    global _remembered_chars
    if len(text) < cfg.min_chars:
        return text
    key = (text, cfg.min_chars, cfg.max_chars, cfg.head_fraction)
    out = _remembered.get(key)
    if out is None:
        out = _remembered[key] = _apply_rules(text, cfg)
        _remembered_chars += len(text) + len(out)
        while _remembered_chars > REMEMBERED_CHARS and _remembered:
            oldest = next(iter(_remembered))
            _remembered_chars -= len(oldest[0]) + len(_remembered.pop(oldest))
    return out


def _apply_rules(text: str, cfg: CompressionConfig) -> str:
    out = _ANSI.sub("", text)
    out = "\n".join(line.rstrip(" \t") for line in out.split("\n"))
    out = _BLANKS.sub("\n\n", out)
    # Collapse runs of identical lines (progress bars, repeated log lines).
    lines = out.split("\n")
    collapsed: list[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        run = j - i + 1
        collapsed.append(lines[i] if run < 3 else f"{lines[i]}\n[previous line repeated {run - 1} more times]")
        i = j + 1
    out = "\n".join(collapsed)
    if len(out) > cfg.max_chars:
        head = int(cfg.max_chars * cfg.head_fraction)
        tail = cfg.max_chars - head
        omitted = out[head : len(out) - tail]
        out = (
            out[:head]
            + f"\n[... {omitted.count(chr(10)) + 1} lines, {len(omitted)} characters omitted by Merit Order ...]\n"
            + out[len(out) - tail :]
        )
    return out


# ------------------------------------------------------------------ walking a request's tool results

TextFn = Callable[[str, bool, str | None], str]  # (text, in the newest messages, tool name) -> new text


def newest_from(messages: list) -> int:
    """Index of the first message after the last assistant message: what's new on this turn (several tool results
    after parallel tool calls, in either format). Everything before it was sent on an earlier turn."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
            return i + 1
    return 0


def tool_names(messages: list, fmt: str) -> dict[str, str]:
    """Tool call id -> the tool's name, from the assistant messages that made the calls."""
    out: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        if fmt == "openai":
            for call in m.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    out[str(call["id"])] = str((call.get("function") or {}).get("name") or "")
        elif isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"):
                    out[str(b["id"])] = str(b.get("name") or "")
    return out


def _map_value(value, fn: Callable[[str], str]) -> tuple[object, bool]:
    """Apply ``fn`` to a tool result's content (a string or a list of text blocks)."""
    if isinstance(value, str):
        new = fn(value)
        return new, new != value
    if isinstance(value, list):
        changed = False
        out = []
        for block in value:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                new = fn(block["text"])
                if new != block["text"]:
                    block, changed = {**block, "text": new}, True
            out.append(block)
        return out, changed
    return value, False


def map_tool_results(body: dict, fmt: str, fn: TextFn) -> dict:
    """The body with ``fn`` applied to the text of every tool result, shallow-copied where something changed. User
    and assistant text never reach ``fn``."""
    messages = body.get("messages") or []
    first_new = newest_from(messages)
    names = tool_names(messages, fmt)
    new_messages = []
    any_change = False
    for idx, m in enumerate(messages):
        changed = m
        newest = idx >= first_new
        if fmt == "openai" and isinstance(m, dict) and m.get("role") == "tool":
            tool = names.get(str(m.get("tool_call_id")))
            content, did = _map_value(m.get("content"), lambda t: fn(t, newest, tool))
            if did:
                changed = {**m, "content": content}
        elif fmt == "anthropic" and isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), list):
            blocks = []
            did_any = False
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tool = names.get(str(b.get("tool_use_id")))
                    content, did = _map_value(b.get("content"), lambda t: fn(t, newest, tool))
                    if did:
                        b, did_any = {**b, "content": content}, True
                blocks.append(b)
            if did_any:
                changed = {**m, "content": blocks}
        any_change = any_change or changed is not m
        new_messages.append(changed)
    return {**body, "messages": new_messages} if any_change else body


class _Rules:
    """The rules for one request, remembering each text's output (a request is walked twice with model compression)."""

    def __init__(self, cfg: CompressionConfig) -> None:
        self.cfg = cfg
        self.seen: dict[str, str] = {}

    def __call__(self, text: str) -> str:
        if not self.cfg.enabled:
            return text
        out = self.seen.get(text)
        if out is None:
            out = self.seen[text] = compress_text(text, self.cfg)
        return out


def model_candidates(body: dict, fmt: str, cfg: CompressionConfig) -> tuple[list[tuple[str, bool]], _Rules]:
    """Tool results model-based compression may take (after the rules): (text, new on this turn). A text that also
    appears earlier in the history counts as old, since it was sent before as it is."""
    rules = _Rules(cfg)
    found: dict[str, bool] = {}
    if not cfg.model.enabled:
        return [], rules

    def see(text: str, newest: bool, tool: str | None) -> str:
        out = rules(text)
        if len(out) >= cfg.model.min_chars and not cfg.model.skips(tool):
            found[out] = found.get(out, True) and newest
        return text

    map_tool_results(body, fmt, see)
    return list(found.items()), rules


def compress_tool_results(body: dict, fmt: str, cfg: CompressionConfig, compressed: dict[str, str] | None = None,
                          rules: _Rules | None = None) -> tuple[dict, Result]:
    """Returns a (shallow-copied where changed) body with large tool results compressed: by the rules when they're on,
    then, for texts in ``compressed`` (post-rules text -> what model-based compression made of it), by the model."""
    result = Result()
    if not cfg.enabled and not compressed:
        return body, result
    rules = rules or _Rules(cfg)

    def shrink(text: str, newest: bool, tool: str | None) -> str:
        out = rules(text)
        if out != text:
            result.add(newest, len(text) - len(out), text_tokens(text) - text_tokens(out))
        if compressed and not cfg.model.skips(tool) and len(out) >= cfg.model.min_chars:
            model_out = compressed.get(out)
            if model_out is not None and model_out != out:
                result.add(newest, len(out) - len(model_out), text_tokens(out) - text_tokens(model_out), model=True)
                out = model_out
        return out

    return map_tool_results(body, fmt, shrink), result
