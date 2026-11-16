"""Point coding harnesses (pi, Claude Code, Codex CLI) at Merit Order.

The API key lives in one file (written by `meritorder login`, mode 600). Harness
configs read it from there at request time, so logging in again is all it takes
to rotate the key.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "meritorder"
KEY_FILE = CONFIG_DIR / "api_key"
CREDENTIALS = CONFIG_DIR / "credentials.json"


def save_key(gateway: str, key: str, extra: dict | None = None) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in ((KEY_FILE, key + "\n"), (CREDENTIALS, json.dumps({"gateway": gateway, "api_key": key, **(extra or {})}, indent=2))):
        path.write_text(text)
        path.chmod(0o600)


def saved() -> dict:
    try:
        return json.loads(CREDENTIALS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def key_command() -> str:
    return f"!cat {shlex.quote(str(KEY_FILE))}"


def pi_provider(gateway: str, models: list[dict]) -> dict:
    """A pi `models.json` provider entry for Merit Order (OpenAI-compatible API)."""
    entries = []
    for m in models:
        if m.get("type", "chat") != "chat":
            continue  # embedding models aren't chat models
        entry: dict = {"id": m["id"], "name": f"{m['id']} (Merit Order)", "reasoning": True, "input": ["text", "image"]}
        if m.get("context_window"):
            entry["contextWindow"] = int(m["context_window"])
        entries.append(entry)
    return {
        "name": "Merit Order",
        "baseUrl": f"{gateway.rstrip('/')}/v1",
        "api": "openai-completions",
        "apiKey": key_command(),
        "authHeader": True,
        "compat": {
            # Send pi's session id as x-session-id: Merit Order keeps each session on the
            # backend that holds its prompt cache.
            "sendSessionAffinityHeaders": True,
            "sessionAffinityFormat": "openrouter",
            # Put Anthropic-style cache breakpoints in requests; they survive translation.
            "cacheControlFormat": "anthropic",
            "maxTokensField": "max_tokens",
            "supportsReasoningEffort": True,
            "thinkingFormat": "openai",
        },
        "models": entries,
    }


def write_pi(gateway: str, models: list[dict], agent_dir: Path | None = None) -> Path:
    agent_dir = agent_dir or Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))
    path = agent_dir / "models.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        data = {}
    data.setdefault("$schema", "https://pi.dev/schemas/models.schema.json")
    data.setdefault("providers", {})["meritorder"] = pi_provider(gateway, models)
    agent_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.with_suffix(".json.bak").write_text(path.read_text())
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def codex_home() -> Path:
    """Where Codex CLI keeps config.toml: CODEX_HOME when set, else ~/.codex."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()


CODEX_PROVIDER = "meritorder"
_CODEX_TOP = ("model_provider", "model", "model_context_window")  # the top-level keys setup sets


def codex_config(gateway: str, existing: str = "", model: str | None = "auto", context_tokens: int | None = None) -> str:
    """Codex CLI's config.toml with Merit Order as its model provider (Codex's config reference, read 15 November 2026).

    The provider speaks the Responses API over WebSocket (``supports_websockets``), so Codex chains each turn on its
    connection instead of resending the conversation, and reads the key from the key file at request time (``auth``).
    Codex summarizes its own history when it compacts (custom providers default to that). Everything else in the file is
    kept; only the provider's table and the top-level keys set here are replaced."""
    top: list[str] = []
    rest: list[str] = []
    in_table = skip = False
    replaced = {"model_provider", *(("model", "model_context_window") if model else ())}
    for line in existing.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_table = True
            name = s.strip("[]").strip().strip('"')
            skip = name == f"model_providers.{CODEX_PROVIDER}" or name.startswith(f"model_providers.{CODEX_PROVIDER}.")
        if skip:
            continue
        if not in_table:
            key = s.split("=", 1)[0].strip() if "=" in s and not s.startswith("#") else None
            if key not in replaced:
                top.append(line)
        else:
            rest.append(line)
    q = json.dumps  # a TOML basic string is a JSON string
    head = [f"model_provider = {q(CODEX_PROVIDER)}"]
    if model:
        head.append(f"model = {q(model)}")
        if context_tokens:
            head.append(f"model_context_window = {int(context_tokens)}")
    provider = [
        f"[model_providers.{CODEX_PROVIDER}]",
        f"name = {q('Merit Order')}",
        f"base_url = {q(gateway.rstrip('/') + '/v1')}",
        f"wire_api = {q('responses')}",
        "supports_websockets = true",
        "",
        f"[model_providers.{CODEX_PROVIDER}.auth]",
        f"command = {q('cat')}",
        f"args = [{q(str(KEY_FILE))}]",
    ]
    while top and not top[-1].strip():
        top.pop()
    while rest and not rest[-1].strip():
        rest.pop()
    parts = ["\n".join(head + top).strip("\n")]
    if rest:
        parts.append("\n".join(rest).strip("\n"))
    parts.append("\n".join(provider))
    return "\n\n".join(parts) + "\n"


def write_codex(gateway: str, config_path: Path | None = None, model: str | None = "auto", context_tokens: int | None = None) -> Path:
    import tomllib

    path = config_path or codex_home() / "config.toml"
    existing = path.read_text() if path.exists() else ""
    text = codex_config(gateway, existing, model, context_tokens)
    tomllib.loads(text)  # never leave Codex a file it can't read
    if path.exists():
        path.with_suffix(".toml.bak").write_text(existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def claude_config_dir() -> Path:
    """Where Claude Code keeps its settings: CLAUDE_CONFIG_DIR when set, else ~/.claude."""
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()


def write_claude_code(gateway: str, settings_path: Path | None = None, model: str | None = "auto",
                      context_tokens: int | None = None) -> Path:
    """Merge Merit Order into a Claude Code settings file: the gateway URL, the key helper, and the model.

    ``model`` is what new sessions start on (``auto`` lets Merit Order choose; None leaves Claude Code's choice).
    Claude Code doesn't know a name like ``auto``: it would assume a 200K context window, so ``context_tokens``
    (the window the gateway reports for that model) is passed on, and the name is added to the /model picker.
    """
    path = settings_path or claude_config_dir() / "settings.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        data = {}
    if path.exists():
        path.with_suffix(".json.bak").write_text(path.read_text())
    env = data.setdefault("env", {})
    env["ANTHROPIC_BASE_URL"] = gateway.rstrip("/")
    # Behind a custom URL Claude Code leaves out its request-class header unless asked: Merit Order uses it to give
    # side requests and compaction summaries a lower level (routing.request_class_levels).
    env["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] = "1"
    data["apiKeyHelper"] = f"cat {shlex.quote(str(KEY_FILE))}"
    if model:
        data["model"] = model
        if model.lower().startswith("claude-"):
            # Claude Code's own model names: it knows their windows and features already.
            for k in ("ANTHROPIC_CUSTOM_MODEL_OPTION", "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME", "CLAUDE_CODE_MAX_CONTEXT_TOKENS"):
                env.pop(k, None)
        else:
            env["ANTHROPIC_CUSTOM_MODEL_OPTION"] = model
            env["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] = f"{model} (Merit Order)"
            if context_tokens:
                env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(int(context_tokens))
            else:
                env.pop("CLAUDE_CODE_MAX_CONTEXT_TOKENS", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
