"""Rule-based task classifier: request -> quality floor (task type + minimum level).

Deliberately simple and explainable. The level comes from the work the request asks for: its wording, its tools,
and the client's own floor or hint. Tools say a request is coding; the wording of the task sets how hard (decision 61):
a coding agent sends the same dozens of tools whether it's asked to fix a typo or to redesign a service. How long the
context is never raises it. A long conversation isn't harder work; it only needs a model with room for it, and the
router rules out models whose context window can't hold the prompt plus the reply (router.fits). The rules and their defaults are ``routing.levels`` (config.LevelRules).

The harness can always override it with ``X-Route-Floor: coding:4`` or nudge it with ``X-Route-Hint: cheap|smart``.
An opt-in learned router (learned.py, decision 44) can correct the level these rules give, from outcomes; the rules
stay the default and the fallback, and their reason (which rules fired) is one of its features.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from .config import LevelRules
from .formats.translate import person_text
from .outcomes import correction
from .tokens import count

DEFAULT_RULES = LevelRules()
# What each level asks of a model: one scoring at least the level for the task (1-5; on the absolute scale a 5 solves
# FrontierSWE-hard work as often as the best model did in November 2026, a fixed bar: decision 58).
LEVEL_MEANINGS = {1: "trivial: a short question on its own", 2: "simple: general work, tests, renames, docstrings",
                  3: "ordinary coding or analysis", 4: "hard coding: multi-file, concurrency, security, performance",
                  5: "the hardest: system design, FrontierSWE-hard work"}


@dataclass
class Floor:
    task: str
    level: int
    reason: str
    explicit: bool = False  # set by the client's X-Route-Floor: never capped

    def __str__(self) -> str:
        return f"{self.task}:{self.level}"


def parse_floor(value: str, reason: str = "header") -> Floor | None:
    m = re.fullmatch(r"\s*([a-z_]+)\s*:\s*([1-5])\s*", value or "")
    return Floor(m.group(1), int(m.group(2)), reason) if m else None


@functools.lru_cache(maxsize=32)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern | None, tuple[re.Pattern, ...]]:
    """All the patterns as one (one pass over the text, as before they were configurable), and each on its own (to say
    which one matched)."""
    if not patterns:
        return None, ()
    each = tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE), each


def label(pattern: str) -> str:
    """A pattern as people read it: `\\brefactor` is "refactor", `\\.py\\b` is ".py", `\\bdef\\s` is "def" (escaped
    punctuation unescaped, `\\s` a space, other classes such as `\\w` kept)."""
    return re.sub(r"\\([^A-Za-z0-9])", r"\1", pattern.replace(r"\b", "").replace(r"\s", " ")).strip() or pattern


def matched(patterns: list[str], text: str) -> str | None:
    """The label of the pattern that matches first in the text, or None. Only the pattern is named, never the text
    it matched: the ledger holds no prompt content."""
    combined, each = _compiled(tuple(patterns))
    m = combined.search(text) if combined else None
    if m is None:
        return None
    return next((label(p) for p, rx in zip(patterns, each) if rx.match(text, m.start())), label(patterns[0]))


def matched_all(patterns: list[str], text: str) -> list[str]:
    """The labels of every pattern that matches the text, in the list's order (none: an empty list). Only searched one
    by one when the combined pattern found something, so text that matches none costs one pass."""
    combined, each = _compiled(tuple(patterns))
    if combined is None or combined.search(text) is None:
        return []
    return [label(p) for p, rx in zip(patterns, each) if rx.search(text)]


def _recent_user_text(body: dict, n: int = 2) -> str:
    """The text of the last ``n`` user messages a person wrote. In an Anthropic-format agent loop most user messages are
    only tool results, which have no text of their own (text_of skips tool_result blocks): those are passed over, so the
    wording of the prompt that started the task is still read mid-task. OpenAI's tool results are ``tool`` messages, so
    they never count either. What the harness adds (Claude Code's system reminders, Codex's AGENTS.md and environment
    context: translate.HARNESS_TEXT) is left out, and a message that's only that is passed over too. Only text parts
    are read, so a scan back through a long loop stays cheap."""
    texts = []
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            text = person_text(m.get("content"))
            if not text.strip():
                continue  # only tool results, images or the harness's text: nothing a person wrote
            texts.append(text)
            if len(texts) >= n:
                break
    return "\n".join(texts)[:20000]


def _coding_level(rules: LevelRules, tools: list, text: str, body: dict) -> tuple[int, list[str]]:
    """A coding request's level and why. The task's wording sets it: the hardest wording (system design, proofs) puts it
    at ``hardest``; hard wording (concurrency, security, multi-file) raises it by ``hard_raise``; simple wording (tests,
    renames, docstrings) with nothing harder puts it at ``simple``; anything else is ``coding``. A new prompt that says
    the last answer was wrong raises it by ``correction_raise``. With ``agent_by_tools``, a request with at least
    ``agent_tools`` tools starts at ``agent`` and simple wording doesn't lower it (the rule before decision 61)."""
    n = len(tools)
    level = rules.coding
    why = [f"coding: {n} tool{'' if n == 1 else 's'}" if tools else f"coding: code-like text ({matched(rules.coding_words, text)})"]
    by_tools = rules.agent_by_tools and n >= rules.agent_tools
    if by_tools:
        level, why = rules.agent, [f"agentic coding: {n} tools (at least {rules.agent_tools})"]
    hardest = matched(rules.hardest_words, text)
    hard = [] if hardest or not rules.hard_raise else matched_all(rules.hard_words, text)
    simple = None if hardest or hard or by_tools else matched(rules.simple_words, text)
    if hardest and rules.hardest > level:
        level = rules.hardest
        why.append(f"hardest wording ({hardest}): level {level}")
    elif hard:
        # Each sign of hard work goes a level up: "the race condition when two workers pick the same job" is two.
        up = rules.hard_raise * min(len(hard), rules.hard_signals)
        level = min(level + up, 5)
        why.append(f"hard wording ({', '.join(hard[:rules.hard_signals])}): +{up}")
    elif simple and rules.simple < level:
        level = rules.simple
        why.append(f"simple task wording ({simple}): level {level}")
    fixed = correction(body) if rules.correction_raise else None
    if fixed:
        level = min(level + rules.correction_raise, 5)
        why.append(f"the new prompt corrects the last answer ({fixed}): +{rules.correction_raise}")
    return level, why


def classify(body: dict, headers: dict[str, str], size: int | None = None, rules: LevelRules | None = None) -> Floor:
    """The request's floor and, in its reason, every rule that set it. ``size`` (the request's estimated tokens) is
    only used to recognize a short question on its own: it never raises a level."""
    explicit = parse_floor(headers.get("x-route-floor", ""), "set by X-Route-Floor")
    if explicit:
        explicit.explicit = True
        return explicit

    rules = rules or DEFAULT_RULES
    tools = body.get("tools") or []
    text = _recent_user_text(body)

    if tools or matched(rules.coding_words, text):
        task = "coding"
        level, why = _coding_level(rules, tools, text, body)
    else:
        task, level, why = "general", rules.general, ["general"]
        size = size if size is not None else count(body)
        if size < rules.short_tokens:
            level, why = rules.short, [f"short question: under {rules.short_tokens:,} tokens, no tools"]
        analysis = matched(rules.analysis_words, text)
        if analysis:
            level, why = rules.analysis, [f"analysis wording ({analysis})"]

    hint = (headers.get("x-route-hint") or "").lower()
    if hint == "cheap":
        level -= 1
        why.append("X-Route-Hint: cheap, -1")
    elif hint == "smart":
        level += 1
        why.append("X-Route-Hint: smart, +1")
    return Floor(task, max(1, min(5, level)), "; ".join(why))
