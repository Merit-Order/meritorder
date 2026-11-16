"""The Responses API's tools beyond plain functions, on backends that only take function tools (decision 57).

Followed: OpenAI's Responses API reference (custom tools, namespaces, tool search, local shell, hosted tools; read 15
November 2026) and Codex CLI's source (``codex-rs``, ``main`` of 15 November 2026: ``tools/src/tool_spec.rs``,
``protocol/src/models.rs``, ``core/src/tools/spec_plan.rs``). What Codex sends to a custom provider such as the gateway:

- function tools (``exec_command``, ``write_stdin``, ``view_image``, ``request_user_input``, MCP tools), some in
  ``namespace`` tools (``collaboration``, MCP servers); the model's calls name the namespace;
- ``custom`` (freeform) tools with a ``format`` (a Lark grammar, or plain text): ``apply_patch``, and Code Mode's ``exec``.
  Codex sends ``apply_patch`` only for models its catalog knows (the GPT models), never for ``auto``;
- ``tool_search`` (``execution: "client"``: Codex runs the search and answers with the tools it found);
- ``web_search``, ``external_web_access: false`` by default (OpenAI's cached index only). ``image_generation`` only with
  OpenAI's own sign-in, so never to the gateway;
- for a model in "Responses lite" mode (GPT-5.6 and GPT-6 in Codex's catalog), the tools as ``additional_tools`` items
  at the start of ``input`` instead of in ``tools``, and more of them later in the conversation.

``local_shell`` (the older built-in shell tool) is no longer sent, but its call items can still be in a conversation's
history, so it's translated too.

Every tool the model can call becomes a function tool, and its calls go back as the item the client asked for: a custom
tool takes one string argument, ``input``, with the format in its description; a namespaced tool is named
``namespace__name``. Hosted tools (web search, image generation, file search, ...) run at OpenAI: a backend that speaks
the Responses API gets them as they were, Claude gets live web search as its own server tool, and every other backend
goes without them, said in ``x-meritorder-dropped-tools``. What the reply converters need to map calls back travels in
the Chat Completions body under ``TOOLS_KEY``, which no backend is sent.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .translate import tool_media

TOOLS_KEY = "_meritorder_responses_tools"  # in a Chat Completions body made from a Responses request; never sent upstream
DEFAULT_NAMESPACE = "functions"  # Codex's namespace for top-level tools: its calls are named as they are
INPUT = "input"  # a custom tool's one argument
LOCAL_SHELL = "local_shell"
TOOL_SEARCH = "tool_search"
ANTHROPIC_WEB_SEARCH = "web_search_20250305"  # the basic version: every Claude model has it, and Vertex AI has only it
WEB_SEARCH_TYPES = ("web_search", "web_search_preview", "web_search_2025_08_26", "web_search_preview_2025_03_11")

_INPUT_SCHEMA = {"type": "object", "properties": {INPUT: {"type": "string", "description": "The tool's input, as raw text."}},
                 "required": [INPUT], "additionalProperties": False}
_LOCAL_SHELL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "array", "items": {"type": "string"}, "description": "The program and its arguments, e.g. [\"bash\", \"-lc\", \"ls\"]."},
        "timeout_ms": {"type": "integer", "description": "How long the command may run, in milliseconds."},
        "working_directory": {"type": "string", "description": "The directory to run it in."},
        "env": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Environment variables to set."},
        "user": {"type": "string", "description": "The user to run it as."},
    },
    "required": ["command"],
    "additionalProperties": False,
}
_LOCAL_SHELL_DESCRIPTION = "Runs a command on the user's machine and returns its output."


def dumps(value: Any) -> str:
    """JSON the same way every time, so a conversation's history converts to the same bytes on every turn."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def chat_name(namespace: str | None, name: str) -> str:
    """The function name a tool goes by on a backend without namespaces: its own in Codex's default namespace, else
    ``namespace__name`` (or the two joined, for a namespace that ends in ``_`` like ``mcp__github__``), with characters
    no provider takes replaced and the result kept to 64 characters (OpenAI's limit)."""
    if not namespace or namespace == DEFAULT_NAMESPACE:
        return name
    if namespace.endswith(("_", "-")):
        full = namespace + name
    else:
        full = f"{namespace}__{name}"
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", full) or "tool"
    if len(clean) > 64:
        clean = clean[:55] + "_" + hashlib.sha256(full.encode()).hexdigest()[:8]
    return clean


def custom_description(spec: dict) -> str:
    """A custom tool's description with what the model needs to call it as a function: the raw text in ``input``,
    and the grammar the text must match."""
    own = (spec.get("description") or "").strip()
    how = ("Call this tool with one argument, `input`: the tool's raw input as a single string. The text in it is the "
           "tool's own format, not JSON.")
    fmt = spec.get("format") or {}
    if fmt.get("type") == "grammar" and fmt.get("definition"):
        syntax = f" ({fmt['syntax']})" if fmt.get("syntax") else ""
        how += f" It must match this grammar{syntax}:\n\n{fmt['definition'].strip()}"
    return f"{own}\n\n{how}" if own else how


def _prefixed(description: str | None, ns_description: str | None) -> str | None:
    if not ns_description:
        return description
    return f"{ns_description.strip()}\n\n{description}" if description else ns_description.strip()


class ToolSet:
    """The tools of one Responses request, gathered from ``tools``, ``additional_tools`` items and tool search results,
    as Chat Completions function tools plus what maps their calls back."""

    def __init__(self) -> None:
        self.chat: dict[str, dict] = {}  # chat name -> Chat Completions tool; insertion order is the request's
        self.entries: dict[str, dict] = {}  # chat name -> {"kind", "name", "namespace"} for a tool that isn't a plain function
        self.native: list[dict] = []  # the Responses tools as sent, for a backend that speaks the Responses API
        self.hosted: list[dict] = []  # tools that run at OpenAI (web search, ...)

    def add(self, spec: Any, native: bool = True) -> None:
        if not isinstance(spec, dict):
            return
        if native:
            key = (spec.get("type"), spec.get("name"))
            for i, old in enumerate(self.native):
                if (old.get("type"), old.get("name")) == key:
                    self.native[i] = spec  # a later definition replaces an earlier one (Codex's incremental tools)
                    break
            else:
                self.native.append(spec)
        t = spec.get("type") or "function"
        if t == "namespace":
            for inner in spec.get("tools") or []:
                self._callable(inner, spec.get("name"), spec.get("description"))
        elif t in ("function", "custom", LOCAL_SHELL) or (t == TOOL_SEARCH and spec.get("execution") == "client"):
            self._callable(spec, None, None)
        else:
            key = (t, spec.get("name"))
            self.hosted = [h for h in self.hosted if (h.get("type"), h.get("name")) != key] + [spec]

    def _callable(self, spec: Any, namespace: str | None, ns_description: str | None) -> None:
        if not isinstance(spec, dict):
            return
        t = spec.get("type") or "function"
        if t == LOCAL_SHELL:
            name, fn = LOCAL_SHELL, {"name": LOCAL_SHELL, "description": _LOCAL_SHELL_DESCRIPTION, "parameters": _LOCAL_SHELL_SCHEMA}
        elif t == TOOL_SEARCH:
            name = TOOL_SEARCH
            fn = {"name": name, "description": spec.get("description"),
                  "parameters": spec.get("parameters") or {"type": "object", "properties": {}}}
        elif t == "custom":
            name = spec.get("name") or ""
            fn = {"name": chat_name(namespace, name), "description": _prefixed(custom_description(spec), ns_description),
                  "parameters": _INPUT_SCHEMA}
        elif t == "function":
            name = spec.get("name") or ""
            fn = {"name": chat_name(namespace, name), "description": _prefixed(spec.get("description"), ns_description),
                  "parameters": spec.get("parameters"), "strict": spec.get("strict")}
        else:
            return
        fn = {k: v for k, v in fn.items() if v is not None}
        cn = fn["name"]
        self.chat[cn] = {"type": "function", "function": fn}
        kind = "function" if t == "function" else t
        if kind != "function" or (namespace and namespace != DEFAULT_NAMESPACE):
            self.entries[cn] = {"kind": kind, "name": name, "namespace": namespace}
        else:
            self.entries.pop(cn, None)

    def tools(self) -> list[dict]:
        return list(self.chat.values())

    def registry(self) -> dict | None:
        """What travels under TOOLS_KEY: None when every tool is a plain function and none is hosted (nothing to map)."""
        if not self.entries and not self.hosted:
            return None
        return {"entries": self.entries, "native": self.native, "hosted": self.hosted}


def entry_for(registry: dict | None, name: str) -> dict | None:
    return ((registry or {}).get("entries") or {}).get(name)


# ---------------------------------------------------------------- history items, as Chat Completions tool calls


def history_call(item: dict) -> dict | None:
    """A call item from a conversation's history as a Chat Completions tool call; None for one that isn't a call."""
    t = item.get("type")
    call_id = item.get("call_id") or item.get("id") or ""
    if t == "function_call":
        name, args = chat_name(item.get("namespace"), item.get("name") or ""), item.get("arguments") or "{}"
    elif t == "custom_tool_call":
        name, args = chat_name(item.get("namespace"), item.get("name") or ""), dumps({INPUT: item.get("input") or ""})
    elif t == "local_shell_call":
        action = item.get("action") or {}
        name, args = LOCAL_SHELL, dumps({k: v for k, v in action.items() if k != "type"})
    elif t == "tool_search_call":
        a = item.get("arguments")
        name, args = TOOL_SEARCH, a if isinstance(a, str) else dumps(a if a is not None else {})
    else:
        return None
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def tool_output(output: Any, call_id: str) -> tuple[Any, dict | None]:
    """A tool's output as a Chat Completions tool message's content, and a user message holding its images, if any.

    Chat Completions tool messages take text only, and Claude wants its tool results before anything else in a turn,
    so images (Codex's ``view_image``) follow the run of tool results as a user message of their own."""
    if isinstance(output, str):
        return output, None
    if not isinstance(output, list):
        return json.dumps(output), None
    texts, media = [], []
    for p in output:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t in ("input_text", "output_text", "text"):
            texts.append(p.get("text") or "")
        elif t == "input_image":
            url = p.get("image_url") or (p.get("file_id") and f"file:{p['file_id']}")
            if url:
                part: dict = {"type": "image_url", "image_url": {"url": url}}
                if p.get("detail") in ("low", "high", "auto"):
                    part["image_url"]["detail"] = p["detail"]
                media.append(part)
        elif t == "input_file" and (file := input_file_part(p)):
            media.append(file)
    return tool_media("\n".join(texts), call_id, media)


def input_file_part(p: dict) -> dict | None:
    """A Responses ``input_file`` as a Chat Completions file part. Chat Completions has no file URL, so ``file_url`` goes
    as ``file_data`` (OpenRouter fetches it; Claude backends get it as a URL document)."""
    f: dict[str, Any] = {k: p[k] for k in ("filename", "file_data", "file_id") if p.get(k)}
    if p.get("file_url") and "file_data" not in f:
        f["file_data"] = p["file_url"]
    return {"type": "file", "file": f} if f.get("file_data") or f.get("file_id") else None


# ---------------------------------------------------------------- replies: a Chat Completions tool call as the client's item


def custom_input(arguments: str) -> str:
    """A custom tool's raw input from the function arguments the model wrote. A model that wrote the raw text itself
    (no JSON) gets it as it is."""
    try:
        value = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments or ""
    if isinstance(value, dict):
        if isinstance(value.get(INPUT), str):
            return value[INPUT]
        strings = [v for v in value.values() if isinstance(v, str)]
        if len(value) == 1 and strings:
            return strings[0]
    return value if isinstance(value, str) else arguments


def _loads(arguments: str) -> Any:
    try:
        return json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def local_shell_action(arguments: str) -> dict:
    args = _loads(arguments)
    args = args if isinstance(args, dict) else {}
    command = args.get("command")
    if isinstance(command, str):
        command = [command]
    return {"type": "exec", "command": command if isinstance(command, list) else [],
            **{k: args.get(k) for k in ("timeout_ms", "working_directory", "env", "user")}}


def output_item(entry: dict | None, item_id: str, call_id: str, name: str, arguments: str, status: str = "completed") -> dict:
    """The Responses output item for one call the model made through a function tool."""
    kind = (entry or {}).get("kind") or "function"
    ns = (entry or {}).get("namespace")
    if kind == "custom":
        item = {"type": "custom_tool_call", "id": item_id, "call_id": call_id, "name": entry["name"], "input": custom_input(arguments),
                "status": status}
    elif kind == LOCAL_SHELL:
        item = {"type": "local_shell_call", "id": item_id, "call_id": call_id, "action": local_shell_action(arguments), "status": status}
    elif kind == TOOL_SEARCH:
        item = {"type": "tool_search_call", "id": item_id, "call_id": call_id, "execution": "client", "arguments": _loads(arguments),
                "status": status}
    else:
        item = {"type": "function_call", "id": item_id, "call_id": call_id, "name": (entry or {}).get("name") or name,
                "arguments": arguments or "{}", "status": status}
    if ns and ns != DEFAULT_NAMESPACE and kind in ("custom", "function"):
        item["namespace"] = ns
    return item


ITEM_PREFIX = {"custom": "ctc", LOCAL_SHELL: "lsh", TOOL_SEARCH: "tsc", "function": "fc"}


class InputStream:
    """Decodes ``{"input": "..."}`` as it streams in, so a custom tool's input streams as text. Arguments of any
    other shape aren't streamed: the whole input comes with the call's last event instead."""

    _HEAD = re.compile(r'\s*\{\s*"input"\s*:\s*"')

    def __init__(self) -> None:
        self.raw = ""
        self.pos: int | None = None  # where the string's undecoded rest starts in raw, once the head is read
        self.state = "head"  # head -> text -> end, or off
        self.sent = ""

    def feed(self, chunk: str) -> str:
        self.raw += chunk
        if self.state == "head":
            m = self._HEAD.match(self.raw)
            if m:
                self.state, self.pos = "text", m.end()
            elif not _could_start(self.raw):
                self.state = "off"
        if self.state != "text":
            return ""
        out, i, raw = [], self.pos or 0, self.raw
        while i < len(raw):
            c = raw[i]
            if c == '"':
                self.state = "end"
                i += 1
                break
            if c != "\\":
                out.append(c)
                i += 1
                continue
            if i + 1 >= len(raw):
                break  # the escape continues in the next chunk
            if raw[i + 1] != "u":
                out.append(json.loads(f'"{raw[i:i + 2]}"'))
                i += 2
                continue
            if i + 6 > len(raw):
                break
            code = int(raw[i + 2:i + 6], 16)
            if 0xD800 <= code < 0xDC00:  # a surrogate pair: decode both halves at once
                if i + 12 > len(raw):
                    break
                out.append(json.loads(f'"{raw[i:i + 12]}"'))
                i += 12
            else:
                out.append(chr(code))
                i += 6
        self.pos = i
        text = "".join(out)
        self.sent += text
        return text


def _could_start(raw: str) -> bool:
    """Whether raw could still become the head ``{"input": "``."""
    s = re.sub(r"\s+", "", raw)
    return '{"input":"'.startswith(s) or s.startswith('{"input":"')


# ---------------------------------------------------------------- hosted tools, per backend


def _bedrock(base_url: str) -> bool:
    return "bedrock" in (base_url or "")


def anthropic_web_search(spec: dict) -> dict:
    """OpenAI's web search as Claude's: the domains it may use and the user's location carry over."""
    tool: dict[str, Any] = {"type": ANTHROPIC_WEB_SEARCH, "name": "web_search"}
    domains = (spec.get("filters") or {}).get("allowed_domains")
    if domains:
        tool["allowed_domains"] = list(domains)
    loc = spec.get("user_location") or {}
    where = {k: loc[k] for k in ("city", "region", "country", "timezone") if loc.get(k)}
    if where:
        tool["user_location"] = {"type": "approximate", **where}
    return tool


def place_hosted(registry: dict | None, fmt: str, api: str = "chat", base_url: str = "",
                 names: set[str] | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """Hosted tools on one backend: the Anthropic server tools that stand in for them, and each one left out, with why.

    A backend speaking the Responses API takes them all as they are. Claude (not on Bedrock, which has no web search)
    gets live web search as its own; OpenAI's cached-only search (Codex's default) has no equivalent there, and live
    search would read pages the user chose not to. ``names``: the request's function tools, which a server tool's name
    mustn't clash with."""
    hosted = (registry or {}).get("hosted") or []
    if not hosted or (fmt == "openai" and api == "responses"):
        return [], []
    added: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for spec in hosted:
        t = spec.get("type") or "?"
        label = spec.get("name") or t
        if t in WEB_SEARCH_TYPES and fmt == "anthropic" and not _bedrock(base_url):
            if spec.get("external_web_access") is False:
                dropped.append((label, "cached-only web search runs at OpenAI alone"))
            elif "web_search" in (names or set()) or added:
                dropped.append((label, "a function tool is already named web_search"))
            else:
                added.append(anthropic_web_search(spec))
            continue
        dropped.append((label, "runs at OpenAI, and this backend has no equivalent"))
    return added, dropped


def fit_upstream(up: dict, body: dict, fmt: str, api: str = "chat", base_url: str = "") -> list[tuple[str, str]]:
    """Finishes a backend's request made from a Responses turn: TOOLS_KEY comes out (only a backend speaking the
    Responses API reads it, to send the tools as they came), Claude gets web search as its own server tool, and what
    the backend goes without is returned, with why."""
    registry = body.get(TOOLS_KEY)
    if not registry:
        return []
    if not (fmt == "openai" and api == "responses"):
        up.pop(TOOLS_KEY, None)
    names = {(t.get("function") or {}).get("name") for t in body.get("tools") or []}
    added, dropped = place_hosted(registry, fmt, api, base_url, names)
    if added:
        up["tools"] = [*(up.get("tools") or []), *added]
    return dropped


def dropped_tools(body: dict, fmt: str, api: str = "chat", base_url: str = "") -> list[tuple[str, str]]:
    """The hosted tools a turn goes without on this backend, and why (``x-meritorder-dropped-tools``)."""
    registry = body.get(TOOLS_KEY)
    if not registry:
        return []
    names = {(t.get("function") or {}).get("name") for t in body.get("tools") or []}
    return place_hosted(registry, fmt, api, base_url, names)[1]
