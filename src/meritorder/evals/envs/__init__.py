"""Scripted environments for multi-step (agent) eval tasks.

An agent task gives the model tools; the eval plays the environment behind them and grades the end state.
Each environment is a class built fresh for every run, from the task's ``env_args``:

    class MyEnv(Env):
        def __init__(self, sandbox, **args): ...
        tools = [...]                         # OpenAI function tools the model sees
        def call(self, name, arguments): ...  # -> the tool's result as text; never raises
        def grade(self, final_answer): ...    # -> (passed, reasons it failed)

Environments are trusted code. Anything the model writes that has to run (a repository's tests) goes through
``self.sandbox`` (see sandbox.py), never this process.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from ..sandbox import Sandbox


class Env:
    tools: list[dict] = []
    needs_sandbox = False  # True when grading runs model-written code: the task is skipped without a sandbox
    grading = "The end state of the environment"  # how it's graded, in a sentence, for the console's Tasks tab

    def __init__(self, sandbox: Sandbox | None = None, **args: Any):
        self.sandbox = sandbox
        self.args = args
        self.calls: list[tuple[str, dict]] = []  # every tool call, in order (for grading)

    def call(self, name: str, arguments: dict) -> str:
        raise NotImplementedError

    def grade(self, final_answer: str) -> tuple[bool, list[str]]:
        raise NotImplementedError

    def handle(self, name: str, raw_arguments: str | dict | None) -> str:
        """Record and run one tool call from the model. Bad JSON or an unknown tool comes back as an error message."""
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError:
                self.calls.append((name, {}))
                return json.dumps({"error": "arguments are not valid JSON"})
        if not isinstance(arguments, dict):
            arguments = {}
        self.calls.append((name, arguments))
        if name not in {t["function"]["name"] for t in self.tools}:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return self.call(name, arguments)
        except Exception as e:  # noqa: BLE001 (an environment bug must not crash the eval)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})


def tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    """An OpenAI function tool definition. ``required`` defaults to every property; pass [] for all optional."""
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties,
                                                            "required": list(properties) if required is None else required}}}


def load_env(path: str) -> type[Env]:
    """``module:Class``, e.g. ``meritorder.evals.envs.flights:FlightsEnv``."""
    module, _, cls = path.partition(":")
    env = getattr(importlib.import_module(module), cls)
    if not (isinstance(env, type) and issubclass(env, Env)):
        raise ValueError(f"{path} isn't an Env")
    return env
