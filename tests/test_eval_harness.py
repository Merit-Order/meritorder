"""The eval harness: task kinds (answer, code, agent, judged), checks, suites on disk, scoring, the sandbox,
eval runs as jobs (worker and CLI), and the admin API behind the console's Evals page."""

import copy
import json
from datetime import UTC, datetime

import httpx
import pytest
import yaml
from conftest import ADMIN_KEY, TEST_CONFIG, TEST_KEY, auth

from meritorder.app import create_app
from meritorder.evals import jobs, pyrunner
from meritorder.evals.envs import Env, load_env, tool
from meritorder.evals.runner import (RunContext, TaskResult, scale, extract_code, final_answer, grade, load_suite, parse_number, parse_verdict,
                                     run_suite, run_task, score, wilson)
from meritorder.evals.sandbox import DockerSandbox, LocalSandbox, SandboxResult, run_tests


def reply(text="", calls=()):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}


def gateway_of(answer):
    """A fake gateway: ``answer(body)`` returns a reply dict (or an httpx.Response)."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        out = answer(body)
        if isinstance(out, httpx.Response):
            return out
        return httpx.Response(200, json=out, headers={"x-meritorder-cost-usd": "0.001"})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


# ------------------------------------------------------------------ checks

def test_final_answers_numbers_and_constraints():
    assert final_answer("work...\nFinal answer: **Lisbon**.") == "Lisbon"
    assert final_answer("Final answer: 3\nthen\nFINAL ANSWER: `7`") == "7"
    assert final_answer("no marker") is None
    assert grade({"final": "lisbon"}, reply("Final answer: Lisbon."))[0]
    assert not grade({"final": "Porto"}, reply("Final answer: Lisbon"))[0]
    assert grade({"final_one_of": ["NYC", "New York"]}, reply("Final answer: new  york"))[0]
    assert grade({"final_number": "3/8"}, reply("Final answer: 0.375"))[0]
    assert grade({"final_number": 1234.5}, reply("Final answer: $1,234.50"))[0]
    assert grade({"final_number": 0.1, "tolerance": 0.01}, reply("Final answer: 0.105"))[0]
    assert grade({"final_number": 42}, reply("Final answer: forty-two")) == (False, "final answer 'forty-two' isn't a number")
    assert grade({"final_number": 0.32098765432098764, "tolerance": 1e-9}, reply("Final answer: 26/81 (about 0.321)"))[0]
    assert not grade({"final_number": 66}, reply("Final answer: about 0.66 or 66"))[0]  # the first number given is the answer
    assert grade({"final": "x"}, reply("x")) == (False, "no 'Final answer:' line")
    assert parse_number("12%") == 12 and parse_number("1/0") is None and parse_number("2e-3") == parse_number("0.002") and parse_number("") is None

    text = "Launch Notes\n\n- fast\n- cheap\n- simple\n\nwhat do you think?"
    ok = {"words": [5, 20], "bullets": 3, "paragraphs": 3, "must_include": ["cheap"], "must_not_include": ["very"], "max_chars": 200,
          "starts_with": "Launch", "ends_with": "?"}
    assert grade({"constraints": ok}, reply(text)) == (True, "")
    passed, why = grade({"constraints": {"words": [50, 60], "bullets": 2, "paragraphs": 1, "must_include": ["blazing"], "must_not_include": ["fast"],
                                         "max_chars": 10, "starts_with": "x", "ends_with": "!", "all_lowercase": True, "json_keys": ["a"], "rhymes": True}},
                         reply(text))
    assert not passed
    for bit in ("words, wanted 50-60", "3 bullet points", "3 paragraphs", "doesn't include 'blazing'", "includes 'fast'", "characters", "start with",
                "end with", "lowercase", "isn't a JSON object", "unknown constraints ['rhymes']"):
        assert bit in why
    assert grade({"constraints": {"json_keys": ["a", "b"]}}, reply('{"a": 1, "b": 2}'))[0]
    assert "JSON keys are ['a']" in grade({"constraints": {"json_keys": ["a", "b"]}}, reply('{"a": 1}'))[1]


def test_extract_code_prefers_the_longest_python_block():
    text = "Here:\n```python\ndef f():\n    return 1\n```\nUsage:\n```python\nf()\n```"
    assert extract_code(text).startswith("def f():")
    assert extract_code("```\nx = 1\n```") == "x = 1\n"
    assert extract_code("```js\nlet x\n```") == "let x\n"
    assert extract_code("def g(): pass") == "def g(): pass"


# ------------------------------------------------------------------ suites on disk

def write_task(root, tid, **fields):
    d = root / tid
    d.mkdir(parents=True)
    files = fields.pop("files", {})
    (d / "task.yaml").write_text(yaml.safe_dump({"id": tid, "task": "coding", "level": 3, **fields}))
    for name, text in files.items():
        (d / name).write_text(text)
    return d


def make_suite(tmp_path):
    root = tmp_path / "mysuite"
    write_task(root, "c3-add", kind="code", prompt_file="prompt.md", code={"tests": "tests.py", "timeout": 10},
               files={"prompt.md": "Write add(a, b).", "tests.py": "from solution import add\ndef test_add():\n    assert add(2, 3) == 5\n"})
    write_task(root, "g3-gen", task="general", generate="test_eval_harness:generated", generate_args={"n": 7})
    return root


def generated(n):
    return {"prompt": f"What is {n} squared? End with 'Final answer: <n>'.", "checks": [{"final_number": n * n}]}


def test_suites_from_directories_files_and_names(tmp_path):
    root = make_suite(tmp_path)
    s = load_suite(root)
    by_id = {t["id"]: t for t in s["tasks"]}
    assert s["name"] == "mysuite" and set(by_id) == {"c3-add", "g3-gen"}
    assert by_id["c3-add"]["prompt"] == "Write add(a, b)." and "def test_add" in by_id["c3-add"]["code"]["tests_source"]
    assert by_id["g3-gen"]["prompt"].startswith("What is 7 squared") and by_id["g3-gen"]["checks"] == [{"final_number": 49}]
    both = load_suite(f"starter,{root}")
    assert both["name"] == "starter+mysuite" and len(both["tasks"]) == 26
    with pytest.raises(ValueError, match="more than one suite: c3-add"):
        load_suite(f"{root},{root}")
    default = load_suite()  # the same default as the CLI, the console and the API
    assert default["name"] == "starter+frontier" and len(default["tasks"]) == 24 + len(load_suite("frontier")["tasks"])

    bad = tmp_path / "bad"
    write_task(bad, "x1", kind="telepathy", prompt="hi")
    with pytest.raises(ValueError, match="unknown kind 'telepathy'"):
        load_suite(bad)
    bad2 = tmp_path / "bad2"
    write_task(bad2, "x2", kind="agent", prompt="hi")
    with pytest.raises(ValueError, match=r"x2 \(agent\) is missing env"):
        load_suite(bad2)


def test_selecting_tasks_and_listing_suites():
    s = load_suite("starter")
    assert {t["id"] for t in jobs.select(s, ["g1-*", "c2-tool"])["tasks"]} == {"g1-capital", "g1-echo", "c2-tool"}
    assert jobs.select(s, None) is s
    listed = {x["name"]: x for x in jobs.suites()}
    assert listed["starter"]["tasks"] == 24 and listed["starter"]["kinds"]["answer"] == 24 and listed["starter"]["levels"]["5"] == 5


# ------------------------------------------------------------------ the sandbox and the test runner

def test_pyrunner_runs_each_test_and_enforces_forbidden_imports(tmp_path, monkeypatch):
    import sys
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("tests", "solution"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / "solution.py").write_text("import re as _r\nfrom os import path\ndef add(a, b):\n    return a + b\n")
    (tmp_path / "tests.py").write_text("from solution import add\ndef test_ok():\n    assert add(1, 1) == 2\ndef test_bad():\n    assert add(1, 1) == 3\n")
    out = pyrunner.main([])
    assert out["total"] == 2 and out["passed"] == 1 and out["failed"][0]["name"] == "test_bad"
    assert pyrunner.main(["--forbid", "re,json"])["error"] == "solution imports what the task forbids: re"
    assert pyrunner.forbidden_imports("x = __import__('re')\nfrom re import sub", {"re"}) == ["re", "re"]
    (tmp_path / "solution.py").write_text("def broken(:\n")
    assert pyrunner.main(["--forbid", "re"])["error"].startswith("solution doesn't parse")
    (tmp_path / "solution.py").write_text("raise SystemExit(3)\n")
    assert pyrunner.main([])["error"].startswith("loading the solution failed: SystemExit")


def test_sandbox_runs_tests_and_reports_crashes_and_timeouts(monkeypatch):
    tests = "from solution import add\ndef test_add():\n    assert add(2, 2) == 4\n"
    assert run_tests(LocalSandbox(), "def add(a, b):\n    return a + b\n", tests) == {"total": 1, "passed": 1, "failed": [], "error": None}
    assert run_tests(LocalSandbox(), "import time\ndef add(a, b):\n    time.sleep(5)\n", tests, timeout=0.5)["error"] == "timed out after 0s"

    class Broken:
        def run(self, files, argv, timeout=30.0):
            assert "_run.py" in files and argv[:3] == ["python", "-I", "-B"]
            return SandboxResult(1, "", "Traceback: boom")
    assert "the test run crashed (exit 1)" in run_tests(Broken(), "x", tests)["error"]
    with pytest.raises(ValueError, match="outside the sandbox"):
        LocalSandbox().run({"../escape.py": "x"}, ["python", "-c", "pass"])

    monkeypatch.setattr("meritorder.evals.sandbox.shutil.which", lambda name: None)
    assert "Docker isn't installed" in DockerSandbox().available()


# ------------------------------------------------------------------ task kinds

async def test_code_tasks_run_hidden_tests():
    t = {"id": "c3-add", "task": "coding", "level": 3, "kind": "code", "prompt": "add", "code": {"tests_source":
         "from solution import add\ndef test_a():\n    assert add(1, 2) == 3\ndef test_b():\n    assert add(-1, 1) == 0\n", "timeout": 10}}
    good, _ = gateway_of(lambda b: reply("```python\ndef add(a, b):\n    return a + b\n```"))
    bad, _ = gateway_of(lambda b: reply("```python\ndef add(a, b):\n    return a - b\n```"))
    ctx = RunContext(sandbox=LocalSandbox())
    async with good, bad:
        ok = await run_task(good, "http://gw", "k", "m", t, "s", ctx)
        no = await run_task(bad, "http://gw", "k", "m", t, "s", ctx)
        skipped = await run_task(good, "http://gw", "k", "m", t, "s", RunContext())
    assert ok.passed and ok.detail == "2/2 tests" and ok.kind == "code" and ok.cost == 0.001
    assert not no.passed and no.detail == "0/2 tests" and no.reasons[0].startswith("test_a: AssertionError")
    assert skipped.skipped and skipped.error == "skipped: no sandbox"


class CountEnv(Env):
    """Count to a target with an `add` tool, then report the total."""

    tools = [tool("add", "Add n to the counter", {"n": {"type": "integer"}})]

    def __init__(self, sandbox=None, target=3, **args):
        super().__init__(sandbox, **args)
        self.total, self.target = 0, target

    def call(self, name, arguments):
        if not isinstance(arguments.get("n"), int):
            return json.dumps({"error": "n must be an integer"})
        self.total += arguments["n"]
        return json.dumps({"total": self.total, "padding": "x" * 30_000})

    def grade(self, final_answer):
        ok = self.total == self.target and str(self.target) in final_answer
        return ok, [] if ok else [f"total {self.total}, wanted {self.target}"]


async def test_agent_tasks_play_the_environment_until_a_final_answer():
    t = {"id": "a3-count", "task": "general", "level": 3, "kind": "agent", "prompt": "Count to 3.", "env": "test_eval_harness:CountEnv",
         "env_args": {"target": 3}, "max_turns": 5}

    def agent(body):
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        if len(tool_msgs) < 3:
            return reply(calls=[("add", json.dumps({"n": 1}))])
        return reply("The total is 3.")
    c, seen = gateway_of(agent)
    async with c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.passed and res.turns == 4 and res.detail == "4 turns, 3 tool calls"
    assert res.steps == ['add({"n": 1})'] * 3 and res.answer == "The total is 3."  # kept for reviewing the run
    assert res.tool_calls == 3 and res.model_ms <= res.latency_ms
    last = seen[-1]["messages"]
    assert last[0]["role"] == "system" and "final answer and no tool calls" in last[0]["content"]
    assert last[2]["role"] == "assistant" and last[2]["tool_calls"][0]["function"]["name"] == "add"  # sent back as received
    assert last[3]["role"] == "tool" and last[3]["tool_call_id"] == "c0" and "[truncated:" in last[3]["content"]
    assert seen[0]["tools"][0]["function"]["name"] == "add"

    loop, _ = gateway_of(lambda b: reply(calls=[("add", "{bad json")]))
    async with loop:
        res = await run_task(loop, "http://gw", "k", "m", t, "s")
    assert not res.passed and res.reasons == ["no final answer within 5 turns"]

    down, _ = gateway_of(lambda b: httpx.Response(529, text="overloaded"))
    async with down:
        res = await run_task(down, "http://gw", "k", "m", t, "s")
    assert not res.passed and res.error == "HTTP 529: overloaded" and res.skipped  # an outage isn't the model's fault


async def test_claude_agents_keep_their_thinking_between_tool_calls():
    """Over Anthropic's Messages API, each reply (thinking and its signature included) goes back exactly as it came."""
    t = {"id": "a4-count", "task": "general", "level": 4, "kind": "agent", "prompt": "Count to 2.", "env": "test_eval_harness:CountEnv",
         "env_args": {"target": 2}, "max_turns": 5}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers, body))
        done = sum(1 for m in body["messages"] if m["role"] == "user" and isinstance(m["content"], list))
        usage = {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 50}
        if done < 2:
            content = [{"type": "thinking", "thinking": "", "signature": f"sig{done}"},
                       {"type": "tool_use", "id": f"tu{done}", "name": "add", "input": {"n": 1}}]
            return httpx.Response(200, json={"content": content, "stop_reason": "tool_use", "usage": usage})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "The total is 2."}], "stop_reason": "end_turn", "usage": usage},
                              headers={"x-meritorder-cost-usd": "0.01"})
    ctx = RunContext(formats={"claude": "anthropic"}, effort={"claude": {"4": "high"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        res = await run_task(c, "http://gw", "k", "claude", t, "s", ctx)
    assert res.passed and res.turns == 3 and res.tool_calls == 2 and res.steps == ['add({"n": 1})'] * 2 and res.answer == "The total is 2."
    assert res.input_tokens == 300 and res.output_tokens == 150
    path, headers, last = seen[-1]
    assert path == "/v1/messages" and headers["anthropic-version"] == "2023-06-01" and last["output_config"] == {"effort": "high"}
    assert "final answer and no tool calls" in last["system"] and last["tools"][0]["input_schema"]["properties"] == {"n": {"type": "integer"}}
    assert last["messages"][1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig0"}  # sent back unchanged
    assert last["messages"][2]["content"][0]["type"] == "tool_result" and last["messages"][2]["content"][0]["tool_use_id"] == "tu0"
    assert "[truncated:" in last["messages"][2]["content"][0]["content"] and "max_tokens" not in last
    # It asks for Claude's 5-minute cache itself, so the gateway adds no marker (and no 1-hour cache) of its own.
    assert all(body["cache_control"] == {"type": "ephemeral"} for _, _, body in seen)

    def cut(request):
        return httpx.Response(200, json={"content": [{"type": "thinking", "thinking": "", "signature": "s"}], "stop_reason": "max_tokens",
                                         "usage": {"output_tokens": 128_000}}, headers={"x-meritorder-max-tokens": "128000", "x-meritorder-max-tokens-source": "model"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(cut)) as c:
        res = await run_task(c, "http://gw", "k", "claude", t, "s", ctx)
    assert not res.passed and res.truncated and res.reasons[0].startswith("ran out of output: wrote all 128,000 tokens")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(529, text="busy"))) as c:
        res = await run_task(c, "http://gw", "k", "claude", t, "s", ctx)
    assert res.skipped and res.error == "HTTP 529: busy"

    view = {"claude": {"routes": [{"backend": "anthropic", "format": "anthropic"}]}, "gpt": {"routes": [{"backend": "openai", "format": "openai"}]}}
    assert jobs.formats(view, {"claude": "anthropic", "gpt": "openai", "gone": "x"}) == {"claude": "anthropic", "gpt": "openai"}


def test_environments_handle_bad_calls():
    env = load_env("test_eval_harness:CountEnv")()
    assert json.loads(env.handle("nope", "{}")) == {"error": "unknown tool nope"}
    assert json.loads(env.handle("add", "{bad")) == {"error": "arguments are not valid JSON"}
    assert json.loads(env.handle("add", "[1]")) == {"error": "n must be an integer"}
    assert json.loads(env.handle("add", {"n": 2}))["total"] == 2
    env.call = lambda name, arguments: 1 / 0
    assert json.loads(env.handle("add", {"n": 1})) == {"error": "ZeroDivisionError: division by zero"}
    with pytest.raises(ValueError, match="isn't an Env"):
        load_env("test_eval_harness:generated")
    with pytest.raises(NotImplementedError):
        Env().call("x", {})
    with pytest.raises(NotImplementedError):
        Env().grade("x")


async def test_judged_tasks_need_both_judges():
    t = {"id": "j4", "task": "general", "level": 4, "kind": "judged", "prompt": "Summarize the outage.", "rubric": ["Says 47 minutes", "Names the cause"],
         "reference_answer": "47 minutes; expired certificate.", "checks": [{"constraints": {"words": [1, 50]}}]}

    def verdicts(a, b):
        def answer(body):
            if body["model"] == "candidate":
                return reply("It lasted 47 minutes because a certificate expired.")
            assert "You don't know which model wrote it" in body["messages"][0]["content"]
            assert "<reference_answer>" in body["messages"][1]["content"]
            met = a if body["model"] == "judge-a" else b
            return reply(json.dumps({"criteria": [{"n": 1, "met": True, "why": "ok"}, {"n": 2, "met": met, "why": "?"}]}))
        return answer
    ctx = RunContext(judges=["judge-a", "judge-b"])
    for a, b, passed in ((True, True, True), (True, False, False)):
        c, seen = gateway_of(verdicts(a, b))
        async with c:
            res = await run_task(c, "http://gw", "k", "candidate", t, "s", ctx)
        assert res.passed is passed and res.cost == pytest.approx(0.003)  # the answer plus two judges
        assert res.detail == f"judges: judge-a 2/2, judge-b {2 if b else 1}/2"
    assert res.reasons == ["judge-b: not met: Names the cause"]
    assert res.answer == "It lasted 47 minutes because a certificate expired."
    assert [j["judge"] for j in res.judging] == ["judge-a", "judge-b"]  # every verdict and reason, to audit the judges
    assert res.judging[1]["criteria"][1] == {"n": 2, "met": False, "why": "?", "criterion": "Names the cause"}

    garbled, _ = gateway_of(lambda body: reply("I think it's fine") if body["model"] != "candidate" else reply("47 minutes"))
    async with garbled:
        res = await run_task(garbled, "http://gw", "k", "candidate", t, "s", ctx)
    assert res.error == "not graded: judge judge-a didn't return a usable verdict" and res.skipped
    failing, _ = gateway_of(lambda body: httpx.Response(500, text="nope") if body["model"] == "judge-a" else reply("47 minutes"))
    async with failing:
        res = await run_task(failing, "http://gw", "k", "candidate", t, "s", ctx)
    assert res.error.startswith("not graded: judge judge-a: HTTP 500") and res.skipped
    async with gateway_of(lambda b: reply("x"))[0] as c:
        assert (await run_task(c, "http://gw", "k", "m", t, "s")).skipped
    assert parse_verdict('{"criteria": [{"n": 1, "met": true}]}', 2) is None and parse_verdict("{not json}", 1) is None


async def test_eval_requests_run_uncapped_at_the_levels_effort_on_one_route():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content), request.headers))
        return httpx.Response(200, json={**reply("6 times 7.\nFinal answer: 42"), "usage": {"completion_tokens": 77}},
                              headers={"x-meritorder-cost-usd": "0.002", "x-meritorder-backend": "direct"})
    t = {"id": "g4-x", "task": "general", "level": 4, "prompt": "What is 6*7?", "checks": [{"final_number": 42}]}
    ctx = RunContext(effort={"m": {"4": "high"}}, backends={"m": "direct"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s", ctx)
        body, headers = seen[-1]
        assert res.passed and "max_tokens" not in body and body["reasoning_effort"] == "high" and headers["x-meritorder-backend"] == "direct"
        assert res.effort == "high" and res.backend == "direct" and res.output_tokens == 77 and res.answer.endswith("Final answer: 42")
        assert res.model_ms >= 0 and res.tool_calls == 0 and res.input_tokens == 0

        await run_task(c, "http://gw", "k", "other", {**t, "max_tokens": 5000}, "s", ctx)  # a task may still set a limit
        body, headers = seen[-1]
        assert body["max_tokens"] == 5000 and "reasoning_effort" not in body and "x-meritorder-backend" not in headers
        await run_task(c, "http://gw", "k", "m", {**t, "effort": "max"}, "s", ctx)
        assert seen[-1][0]["reasoning_effort"] == "max"


async def test_rate_limits_are_waited_out_and_billing_failures_arent_graded(monkeypatch):
    import meritorder.evals.runner as runner

    slept = []

    async def fake_sleep(s):
        slept.append(s)
    monkeypatch.setattr(runner, "RETRY_FOR", 600.0)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    answers = iter([httpx.Response(429, text="slow down", headers={"retry-after": "30"}), httpx.Response(529, text="busy"),
                    httpx.Response(200, json=reply("Final answer: 42"))])
    t = {"id": "g4-x", "task": "general", "level": 4, "prompt": "?", "checks": [{"final_number": 42}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: next(answers))) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.passed and not res.skipped and slept == [30.0, 5.0]  # as long as asked, then backing off

    monkeypatch.setattr(runner, "RETRY_FOR", 12.0)
    slept.clear()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429, text="slow down"))) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.skipped and res.error == "HTTP 429: slow down" and slept == [5.0, 7.0]  # gives up after RETRY_FOR seconds

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(402, text="requires more credits"))) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.skipped and res.error.startswith("not graded: the provider's account can't pay for it")

    def credit(request):
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "length"}]},
                              headers={"x-meritorder-max-tokens": "538612", "x-meritorder-max-tokens-source": "provider credit"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(credit)) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.skipped and "could pay for only 538,612 output tokens" in res.error


async def test_a_spending_limit_is_skipped_at_once_not_waited_out(monkeypatch):
    import meritorder.evals.runner as runner

    slept = []

    async def fake_sleep(s):
        slept.append(s)
    monkeypatch.setattr(runner, "RETRY_FOR", 600.0)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    t = {"id": "g4-x", "task": "general", "level": 4, "prompt": "?", "checks": [{"final_number": 42}]}
    budget = httpx.Response(429, text="the organization's monthly LLM budget of $50.00 is used up", headers={"x-meritorder-limit": "budget"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: budget)) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.skipped and res.error.startswith("HTTP 429: the organization's monthly LLM budget") and slept == []


async def test_server_errors_and_lost_connections_are_tried_again(monkeypatch):
    import meritorder.evals.runner as runner

    slept = []

    async def fake_sleep(s):
        slept.append(s)
    monkeypatch.setattr(runner, "RETRY_FOR", 600.0)
    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    t = {"id": "g4-x", "task": "general", "level": 4, "prompt": "?", "checks": [{"final_number": 42}]}

    def lost():
        raise httpx.ReadError("connection reset")
    answers = iter([lambda: httpx.Response(502, text="the stream broke off", headers={"x-meritorder-cost-usd": "0.01"}), lost,
                    lambda: httpx.Response(200, json=reply("Final answer: 42"), headers={"x-meritorder-cost-usd": "0.02"})])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: next(answers)())) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.passed and not res.skipped and slept == [5.0, 10.0] and res.cost == pytest.approx(0.03)  # the broken try cost too

    slept.clear()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="oops"))) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.skipped and res.error == "HTTP 500: oops" and slept == [5.0, 10.0]  # three tries in all, then skipped

    slept.clear()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: lost())) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.skipped and res.error.startswith("not graded: lost the connection to the gateway (after 3 tries): ReadError")

    slept.clear()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply("Final answer: 41")))) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert not res.passed and not res.skipped and slept == []  # a wrong answer isn't tried again


async def test_running_skipped_tasks_again_fills_them_in():
    from meritorder.store import MemoryStore

    store = MemoryStore()

    def res(task_id, passed=True, **kw):
        return {"model": "kimi", "task_id": task_id, "task": "coding", "level": 5, "passed": passed, "reasons": [], "cost": 0.01, **kw}
    first = [res(f"c5-{i}") for i in range(10)] + [res("c5-x", False, error="HTTP 402: requires more credits"),
                                                  res("c5-y", False, skipped=True, error="not graded: no credit")]
    for rid, results, models in [("ev_1", first, ["kimi"]), ("ev_2", [res("c5-x", False, reasons=["wrong"]), res("c5-0", False)], ["kimi"]),
                                 ("ev_3", [{**res("c5-0"), "model": "other"}], ["other"])]:
        await store.create_eval_run({"id": rid, "created_by": "root", "team": "ops", "status": "done", "executor": "cli",
                                     "request": {"models": models, "suite": "frontier"}})
        await store.update_eval_run(rid, results=results)
        await store.set_setting("evals", jobs.merge_latest(await store.get_setting("evals"), rid, "frontier", {m: {} for m in models}))
    await store.create_eval_run({"id": "ev_4", "created_by": "root", "team": "ops", "status": "cancelled", "executor": "cli",
                                 "request": {"models": ["kimi"]}, "results": [res("c5-y")]})

    view = await jobs.latest_view(store)
    hard = view["kimi"]["tasks"]["coding"]["hard"]
    # 10 from the first run (c5-0 replaced by the second run's fail), c5-x from the second; c5-y never graded (a
    # cancelled run doesn't count), so it's still skipped.
    assert hard == {"passed": 9, "runs": 11} and view["kimi"]["skipped"] == 1 and view["kimi"]["tasks"]["coding"]["score"] == 5.0
    assert view["kimi"]["run_id"] == "ev_2" and view["kimi"]["runs"] == ["ev_2", "ev_1"]
    assert {f["id"] for f in view["kimi"]["failed"]} == {"c5-0", "c5-x"}

    pairs = dict((d["task_id"], rid) for rid, d in (await jobs.latest_results(store))["kimi"][1])
    assert pairs["c5-0"] == "ev_2" and pairs["c5-1"] == "ev_1" and pairs["c5-y"] == "ev_1"


async def test_a_reply_that_stops_short_says_why():
    def cut(source):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "def f(:"}, "finish_reason": "length"}],
                                             "usage": {"completion_tokens": 128_000}},
                                  headers={"x-meritorder-max-tokens": "128000", "x-meritorder-max-tokens-source": source})
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = {"id": "g5-x", "task": "general", "level": 5, "prompt": "?", "checks": [{"final": "x"}]}
    async with cut("model") as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert not res.passed and not res.skipped and res.truncated and res.output_tokens == 128_000
    assert res.reasons[0] == "ran out of output: wrote all 128,000 tokens the model allows (thinking included) without finishing"
    async with cut("cost limit") as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s")
    assert res.skipped and res.truncated and "the per-request cost limit allowed 128,000 output tokens" in res.error  # the gateway's limit
    judged = {"id": "j4", "task": "general", "level": 4, "kind": "judged", "prompt": "?", "rubric": ["a"]}
    async with cut("model") as c:
        res = await run_task(c, "http://gw", "k", "m", judged, "s", RunContext(judges=["j1"]))
    assert not res.passed and res.judging == [] and res.reasons[0].startswith("ran out of output")  # no judges paid for it
    agent = {"id": "a4-x", "task": "general", "level": 4, "kind": "agent", "prompt": "Count.", "env": "test_eval_harness:CountEnv",
             "env_args": {"target": 3}, "max_turns": 5}
    async with cut("context") as c:
        res = await run_task(c, "http://gw", "k", "m", agent, "s")
    assert res.turns == 1 and res.reasons == ["ran out of context: the reply filled the 128,000 tokens left in the context window"]
    async with cut("client") as c:
        res = await run_task(c, "http://gw", "k", "m", {**t, "max_tokens": 128_000}, "s")
    assert res.reasons[0] == "ran out of output: hit the task's limit of 128,000 tokens"


def test_runs_record_each_models_effort_and_route():
    view = {"opus": {"effort_levels": ["low", "medium", "high", "xhigh", "max"], "routes": [{"backend": "bedrock", "live": False}, {"backend": "anthropic", "live": True},
                                                                                         {"backend": "openrouter", "live": True}]},
            "plain": {"effort_levels": [], "routes": [{"backend": "gpu", "live": True}]}, "gone": {"routes": [{"backend": "x", "live": False}]}}
    by_floor = {1: "low", 2: "low", 3: "medium", 4: "high", 5: "high"}
    assert jobs.efforts(view, ["opus", "plain"], by_floor) == {"opus": {"1": "low", "2": "low", "3": "medium", "4": "high", "5": "high"}, "plain": {}}
    assert jobs.routes(view, ["opus", "plain", "gone"]) == {"opus": "anthropic", "plain": "gpu"}


async def test_answers_are_judged_by_two_judges_from_other_families():
    from meritorder.evals.runner import family, judges_for

    panel = ["claude-opus", "gpt-sol", "gemini-carbon"]
    assert family("claude-haiku") == "claude" and family("openai/gpt-6") == "gpt" and family("gemini-flash-lite") == "gemini"
    assert judges_for("claude-haiku", panel) == ["gpt-sol", "gemini-carbon"]
    assert judges_for("gpt-luna", panel) == ["claude-opus", "gemini-carbon"]
    assert judges_for("gemini-flash", panel) == ["claude-opus", "gpt-sol"]
    assert judges_for("deepseek-v4-pro", panel) == ["claude-opus", "gpt-sol"]
    assert judges_for("claude-sonnet", ["claude-opus", "gpt-sol"]) == ["gpt-sol", "claude-opus"]  # only one other family: it still takes two
    assert judges_for("x", ["claude-opus"]) == ["claude-opus"]

    t = {"id": "j4", "task": "general", "level": 4, "kind": "judged", "prompt": "Summarize.", "rubric": ["Says 47 minutes"]}
    seen = []

    def answer(body):
        seen.append(body)
        if body["model"] == "claude-haiku":
            return reply("47 minutes.")
        return reply(json.dumps({"criteria": [{"n": 1, "met": True, "why": "ok"}]}))
    c, _ = gateway_of(answer)
    async with c:
        res = await run_task(c, "http://gw", "k", "claude-haiku", t, "s", RunContext(judges=panel))
    assert res.passed and [j["judge"] for j in res.judging] == ["gpt-sol", "gemini-carbon"]
    assert [b["model"] for b in seen] == ["claude-haiku", "gpt-sol", "gemini-carbon"]
    assert [b.get("reasoning_effort") for b in seen[1:]] == ["high", "high"]  # judges read carefully

    view = {m: {"routes": [{"live": True, "price": {"input": 1, "output": 1}}]} for m in panel + ["claude-haiku"]}
    suite = {"tasks": [t]}
    three = jobs.estimate(view, suite, ["claude-haiku"], 1, panel)["judging_usd"]
    two = jobs.estimate(view, suite, ["claude-haiku"], 1, panel[:2])["judging_usd"]
    assert three == two > 0  # two judges per answer, however many are on the panel


async def test_run_suite_reports_progress_and_stops_when_asked():
    suite = {"name": "mini", "tasks": [{"id": f"t{i}", "task": "general", "level": 1, "prompt": "hi", "checks": [{"contains": "hi"}]} for i in range(6)]}
    seen = []

    async def progress(r):
        seen.append(r.task_id)

    calls = {"n": 0}

    async def stop():
        calls["n"] += 1
        return calls["n"] > 2

    c, _ = gateway_of(lambda b: reply("hi"))
    async with c:
        res = await run_suite("http://gw", "k", ["m"], suite, concurrency=1, progress=progress, client=c, should_stop=stop)
    assert len(res) == 2 and seen == ["t0", "t1"]


# ------------------------------------------------------------------ scoring

def test_scores_rest_on_hard_tasks_on_the_routing_scale():
    def res(model, level, passed, n=4, skipped=0):
        return [TaskResult(model=model, task_id=f"t{level}-{i}", task="coding", level=level, passed=i < passed) for i in range(n)] + \
               [TaskResult(model=model, task_id="sk", task="coding", level=level, passed=False, skipped=True) for _ in range(skipped)]
    top = res("top", 3, 5, n=5) + res("top", 4, 5, n=5) + res("top", 5, 4, n=5)
    mid = res("mid", 3, 5, n=5) + res("mid", 4, 3, n=5) + res("mid", 5, 2, n=5, skipped=2)
    s = score(top + mid)
    t, m = s["top"]["tasks"]["coding"], s["mid"]["tasks"]["coding"]
    assert t["hard"] == {"passed": 9, "runs": 10} and m["hard"] == {"passed": 5, "runs": 10}  # levels 4-5; skipped runs left out
    assert t["score"] == 5.0 and m["score"] == 2.8  # 5 x 0.5 / 0.9
    assert m["low"] < 2.8 < m["high"] <= 5.0 and t["low"] < 5.0 == t["high"]
    assert s["mid"]["skipped"] == 2 and m["counts"] == {3: 5, 4: 5, 5: 5}
    lo, hi = wilson(30, 40)
    assert lo < 0.75 < hi and wilson(0, 0) == (0.0, 1.0)
    assert score([TaskResult(model="x", task_id="a", task="coding", level=1, passed=False, skipped=True)]) == {}
    none = score(res("a", 4, 0, n=10) + res("b", 5, 0, n=10))  # nobody passed a hard task: nothing to scale against
    assert none["a"]["tasks"]["coding"]["score"] is None and "nothing to compare" in none["b"]["tasks"]["coding"]["why"]
    thin = score(res("c", 5, 9, n=9))["c"]["tasks"]["coding"]
    assert thin["score"] is None and thin["why"] == "9 hard tasks graded; a score needs 10"
    assert scale({"old": {"tasks": {"coding": {"suggested": 4}}}})["old"]["tasks"]["coding"]["score"] is None  # recorded before scores


def test_stored_results_are_rescored_with_provider_errors_left_out():
    rows = [{"model": "m", "task_id": f"h{i}", "task": "coding", "level": 5, "passed": i < 3, "unknown_field": 1} for i in range(6)]
    rows += [{"model": "m", "task_id": f"x{i}", "task": "coding", "level": 5, "passed": False, "error": "HTTP 402: no credit"} for i in range(4)]
    rows += [{"model": "m", "task_id": "y", "task": "coding", "level": 4, "passed": False, "error": "HTTP 400: bad request"}]
    rs = jobs.results_from(rows)
    assert sum(r.skipped for r in rs) == 4  # an account that couldn't pay says nothing about the model; a 400 does
    s = jobs.rescore({"results": rows})
    assert s["m"]["tasks"]["coding"]["hard"] == {"passed": 3, "runs": 7} and s["m"]["skipped"] == 4
    assert jobs.rescore({"results": None, "scores": {"k": 1}}) == {"k": 1}
    assert "efficiency" not in s["m"] and "efficiency" not in s["m"]["tasks"]["coding"]  # not timed back then: not recorded, not 0 s
    timed = jobs.rescore({"results": [{**rows[0], "model_ms": 2000}]})
    assert timed["m"]["efficiency"]["median_s"] == 2.0


def test_scores_say_what_the_work_took():
    rs = [TaskResult(model="m", task_id=f"t{i}", task="coding", level=4, passed=i < 2, cost=0.10, model_ms=1000 * (i + 1), turns=i + 1,
                     tool_calls=2 * i, output_tokens=1000 * (i + 1), input_tokens=500, truncated=i == 3) for i in range(4)]
    e = score(rs)["m"]["efficiency"]
    assert e == {"median_s": 2.5, "p90_s": 3.7, "turns": 2.5, "tool_calls": 3.0, "output_tokens": 2500, "input_tokens": 500,
                 "cost_per_task": 0.1, "cost_per_pass": 0.2, "cut_off": 1}  # $0.40 spent for 2 solved
    assert score(rs)["m"]["tasks"]["coding"]["efficiency"] == e
    assert score([TaskResult(model="z", task_id="a", task="coding", level=1, passed=False, cost=0.1)])["z"]["efficiency"]["cost_per_pass"] is None


def test_cost_estimates():
    view = {"m": {"routes": [{"live": True, "price": {"input": 2, "output": 10}}, {"live": True, "price": {"input": 5, "output": 20}}]},
            "judge": {"routes": [{"live": True, "price": {"input": 4, "output": 20}}]}, "dead": {"routes": [{"live": False, "price": None}]}}
    suite = {"tasks": [{"id": "a", "prompt": "x" * 4000, "level": 1}, {"id": "j", "kind": "judged", "prompt": "y" * 400, "rubric": ["r"], "level": 4},
                       {"id": "g", "kind": "agent", "prompt": "z", "max_turns": 10, "env": "test_eval_harness:CountEnv", "level": 4}]}
    est = jobs.estimate(view, suite, ["m", "dead"], 2, ["judge"])
    assert est["unpriced"] == ["dead"] and est["by_model"]["dead"] == 0 and est["by_model"]["m"] > 0 and est["judging_usd"] > 0
    assert est["tasks"] == 3 and est["runs"] == 12 and est["judged"] == 1 and est["needs_sandbox"] == 0
    assert jobs.pick_judges(view) == [] and jobs.pick_judges(view, ["judge", "dead"]) == ["judge"]


# ------------------------------------------------------------------ eval runs: store, jobs, worker

async def test_eval_runs_in_the_store(store):
    for i in range(3):
        await store.create_eval_run({"id": f"ev_{i}", "created_by": "root", "team": "ops", "status": "queued", "executor": "worker",
                                     "request": {"models": ["small"], "suite": "starter"}})
    await store.create_eval_run({"id": "ev_cli", "created_by": "root", "team": "ops", "status": "running", "executor": "cli",
                                 "request": {"models": ["small"]}, "started_at": datetime.now(UTC)})
    first = await store.claim_eval_run()
    assert first["id"] == "ev_0" and first["status"] == "running" and first["started_at"]
    assert (await store.update_eval_run("ev_0", status="done", scores={"small": {}}, results=[{"x": 1}], cost=0.5, bogus=1))["cost"] == 0.5
    assert (await store.get_eval_run("ev_0"))["results"] == [{"x": 1}]
    listed = await store.list_eval_runs()
    assert len(listed) == 4 and "results" not in listed[0]
    assert await store.get_eval_run("nope") is None and await store.update_eval_run("nope", status="done") is None
    assert (await store.update_eval_run("ev_1"))["status"] == "queued"  # nothing to change
    await store.claim_eval_run()
    await store.claim_eval_run()
    assert await store.claim_eval_run() is None  # the CLI's run is never claimed


async def test_worker_runs_queued_evals_with_a_temporary_key(gateway, client):
    store = gateway.store
    await store.create_eval_run({"id": "ev_w", "created_by": "root", "team": "ops", "status": "queued", "executor": "worker",
                                 "request": {"models": ["small"], "suite": "starter", "only": ["g1-*"], "repeat": 1}})
    await jobs.worker_loop(store, "http://test", once=True, sandbox_factory=lambda: None, client=client)
    run = await store.get_eval_run("ev_w")
    assert run["status"] == "done" and run["progress"]["done"] == 2 and len(run["results"]) == 2 and run["cost"] > 0
    assert set(run["scores"]) == {"small"}
    latest = await store.get_setting("evals")
    assert latest["models"]["small"]["run_id"] == "ev_w"
    keys = [k for k in await store.list_keys(user_id="root") if (k.get("tags") or {}).get("eval_run") == "ev_w"]
    assert len(keys) == 1 and keys[0]["revoked_at"] is not None  # used for the run, then revoked
    runner = await store.get_setting("eval_runner")
    assert runner["sandbox"] is False and runner["why"] == "no sandbox configured"

    await store.create_eval_run({"id": "ev_bad", "created_by": "root", "team": "ops", "status": "queued", "executor": "worker",
                                 "request": {"models": ["small"], "suite": "/no/such/suite"}})
    await jobs.worker_loop(store, "http://test", once=True, sandbox_factory=lambda: None, client=client)
    assert (await store.get_eval_run("ev_bad"))["status"] == "failed"

    await store.create_eval_run({"id": "ev_stuck", "created_by": "root", "team": "ops", "status": "running", "executor": "worker",
                                 "request": {"models": ["small"]}})
    assert await jobs.interrupted(store) == 1
    assert "interrupted" in (await store.get_eval_run("ev_stuck"))["error"]


async def test_a_cancelled_run_stops_early(gateway, client):
    store = gateway.store
    run = await store.create_eval_run({"id": "ev_c", "created_by": "root", "team": "ops", "status": "running", "executor": "worker",
                                       "request": {"models": ["small"], "suite": "starter", "only": ["g*"], "concurrency": 1}})
    await store.update_eval_run("ev_c", status="cancelling")
    row = await jobs.execute(store, run, "http://test", ADMIN_KEY, concurrency=1, client=client)
    assert row["status"] == "cancelled" and row["progress"]["done"] < 11


# ------------------------------------------------------------------ the Tasks tab: every task, open

def test_tasks_say_what_they_ask_and_how_theyre_graded():
    frontier = {t["id"]: t for t in jobs.task_list("frontier")}
    assert len(frontier) == 32 and len(jobs.task_list("starter")) == 24
    assert frontier["c5-lisp"]["graded_by"] == "40 hidden tests, run in a sandbox" and frontier["c5-lisp"]["kind"] == "code"
    assert frontier["a5-oncall-handoff"]["graded_by"] == "21 hidden tests on the edited repository, in a sandbox"
    assert frontier["a4-charger-order"]["graded_by"].startswith("Exactly one order placed")  # the environment's own description
    assert frontier["g3-incident-summary"]["graded_by"] == "2 judges, 10 rubric points" and frontier["g5-filing-log-4hop"]["generated"]
    assert all(t["summary"] and t["graded_by"] for t in frontier.values())

    code = jobs.task_detail("frontier", "c5-lisp")["code"]
    assert code["tests_count"] == 40 and "def test_" in code["tests"] and code["reference"] and code["wrong"]
    agent = jobs.task_detail("frontier", "a5-oncall-handoff")
    assert "files/ISSUE.md" in agent["agent"]["visible"] and list(agent["agent"]["grader"]) == ["hidden/test_hidden_oncall.py"]
    assert [x["name"] for x in agent["tools"]] == ["list_files", "read_file", "search", "write_file", "run_tests"] and agent["agent"]["max_turns"] == 50
    assert jobs.task_detail("frontier", "a4-charger-order")["agent"]["grader"].keys() == {"procurement/chargers.yaml"}
    judged = jobs.task_detail("frontier", "g3-incident-summary")["judged"]
    assert len(judged["rubric"]) == 10 and "47 minutes" in judged["reference_answer"]
    gen = jobs.task_detail("frontier", "g5-filing-log-4hop")
    assert gen["generated"]["by"].endswith(":multi_hop") and len(gen["prompt"].split()) > 40_000
    assert jobs.task_detail("starter", "c2-tool")["tools"][0]["name"] == "get_weather" and jobs.task_detail("starter", "nope") is None
    assert jobs.graded_by({"kind": "answer", "checks": [{"final": "x"}]}) == "1 check on the reply"


async def test_tasks_api_shows_each_models_latest_result(gateway, client):
    admin = auth(ADMIN_KEY)
    listed = (await client.get("/v1/admin/evals/tasks?suite=starter", headers=admin)).json()
    assert listed["suite"] == "starter" and any(t["id"] == "g1-capital" for t in listed["tasks"])
    assert (await client.get("/v1/admin/evals/tasks?suite=/etc", headers=admin)).status_code == 404
    assert (await client.get("/v1/admin/evals/tasks", headers=auth(TEST_KEY))).status_code == 403
    assert (await client.get("/v1/admin/evals/tasks/nope?suite=starter", headers=admin)).status_code == 404
    assert (await client.get("/v1/admin/evals/tasks/g1-capital?suite=/etc", headers=admin)).status_code == 404

    run = {"id": "ev_t", "created_by": "root", "team": "ops", "status": "done", "executor": "cli", "request": {"models": ["small"]}}
    await gateway.store.create_eval_run(run)
    hit = {"model": "small", "task_id": "g1-capital", "task": "general", "level": 1, "passed": True, "reasons": [], "cost": 0.001, "answer": "Paris"}
    await gateway.store.update_eval_run("ev_t", results=[hit, {**hit, "task_id": "g1-echo"}])
    await gateway.store.set_setting("evals", jobs.merge_latest(None, "ev_t", "starter", {"small": {"pass_rate": 1.0}}))
    detail = (await client.get("/v1/admin/evals/tasks/g1-capital?suite=starter", headers=admin)).json()
    assert detail["prompt"] and detail["checks"] and [(r["model"], r["answer"], r["run_id"]) for r in detail["results"]] == [("small", "Paris", "ev_t")]


# ------------------------------------------------------------------ the admin API behind the Evals page

async def test_eval_api_estimate_start_cancel_and_report(gateway, client):
    admin = auth(ADMIN_KEY)
    suites = (await client.get("/v1/admin/evals/suites", headers=admin)).json()
    assert any(s["name"] == "starter" for s in suites["suites"]) and suites["default_judges"] == list(jobs.DEFAULT_JUDGES)
    assert (await client.get("/v1/admin/evals/suites", headers=auth(TEST_KEY))).status_code == 403

    req = {"models": ["small"], "suite": "starter", "only": ["g1-*", "c1-*"], "repeat": 2}
    est = (await client.post("/v1/admin/evals/estimate", headers=admin, json=req)).json()
    assert est["tasks"] == 4 and est["runs"] == 8 and est["usd"] > 0 and est["judges"] == []
    assert est["default_max_cost_usd"] == 1.0 and est["confirm_above_usd"] == jobs.CONFIRM_ABOVE_USD and est["budget_left_usd"] is None
    assert est["warning"][0].startswith("Estimated cost: about $") and "up to three times" in est["warning"][0]
    assert any(line.startswith("Spending cap: $1.00.") for line in est["warning"])
    assert (await client.post("/v1/admin/evals/estimate", headers=admin, json={**req, "models": ["nope"]})).status_code == 404
    assert (await client.post("/v1/admin/evals/estimate", headers=admin, json={**req, "suite": "/missing"})).status_code == 422
    assert (await client.post("/v1/admin/evals/estimate", headers=admin, json={**req, "only": ["zzz*"]})).status_code == 422
    gateway.backends["mock-openai"].available = False
    assert (await client.post("/v1/admin/evals/estimate", headers=admin, json=req)).status_code == 409
    gateway.backends["mock-openai"].available = True

    uncapped = await client.post("/v1/admin/evals/runs", headers=admin, json=req)  # nobody starts a run without seeing its cost
    assert uncapped.status_code == 422 and "estimated at about $" in uncapped.json()["detail"] and "max_cost_usd" in uncapped.json()["detail"]
    req = {**req, "max_cost_usd": 1.0}
    queued = await client.post("/v1/admin/evals/runs", headers=admin, json=req)
    assert queued.json()["request"]["max_cost_usd"] == 1.0
    assert queued.status_code == 201 and queued.json()["status"] == "queued" and queued.json()["request"]["estimate_usd"] == est["usd"]
    assert queued.json()["request"]["backends"] == {"small": "mock-openai"} and queued.json()["request"]["effort"] == {"small": {}}
    assert queued.json()["request"]["formats"] == {"small": "openai"}
    bad = await client.post("/v1/admin/evals/runs", headers=admin, json={**req, "backends": {"small": "mock-anthropic"}})
    assert bad.status_code == 409 and "small on mock-anthropic" in bad.json()["detail"]
    rid = queued.json()["id"]
    runs = (await client.get("/v1/admin/evals/runs", headers=admin)).json()
    assert runs["runs"][0]["id"] == rid and "results" not in runs["runs"][0]
    assert (await client.get(f"/v1/admin/evals/runs/{rid}", headers=admin)).json()["results"] is None
    assert (await client.get("/v1/admin/evals/runs/nope", headers=admin)).status_code == 404
    assert (await client.post(f"/v1/admin/evals/runs/{rid}/cancel", headers=admin)).json()["status"] == "cancelled"
    assert (await client.post(f"/v1/admin/evals/runs/{rid}/cancel", headers=admin)).status_code == 409
    assert (await client.post("/v1/admin/evals/runs/nope/cancel", headers=admin)).status_code == 404
    assert (await client.patch(f"/v1/admin/evals/runs/{rid}", headers=admin, json={"status": "done"})).status_code == 409  # not a CLI run

    cli = (await client.post("/v1/admin/evals/runs", headers=admin, json={**req, "executor": "cli"})).json()
    assert cli["status"] == "running"
    progress = await client.patch(f"/v1/admin/evals/runs/{cli['id']}", headers=admin, json={"progress": {"done": 1, "total": 8}, "cost": 0.01})
    assert progress.json() == {"id": cli["id"], "status": "running"}
    await client.post(f"/v1/admin/evals/runs/{cli['id']}/cancel", headers=admin)
    seen = await client.patch(f"/v1/admin/evals/runs/{cli['id']}", headers=admin, json={"progress": {"done": 2, "total": 8}})
    assert seen.json()["status"] == "cancelling"  # the CLI learns the console cancelled it
    scores = {"small": {"tasks": {"general": {"suggested": 1, "levels": {"1": 1.0}, "pass_rate": 1.0}}, "pass_rate": 1.0, "cost": 0.01}}
    done = await client.patch(f"/v1/admin/evals/runs/{cli['id']}", headers=admin, json={"status": "done", "scores": scores, "results": []})
    assert done.json()["status"] == "done"
    assert (await client.get("/v1/admin/evals", headers=admin)).json()["models"]["small"]["run_id"] == cli["id"]
    assert (await client.patch("/v1/admin/evals/runs/nope", headers=admin, json={})).status_code == 404


async def test_applying_a_runs_scores(gateway, client, monkeypatch):
    admin = auth(ADMIN_KEY)
    # Stored configurations refuse literal keys, so the running one takes them from the environment.
    monkeypatch.setenv("T_USER", TEST_KEY)
    monkeypatch.setenv("T_ADMIN", ADMIN_KEY)
    doc = copy.deepcopy(TEST_CONFIG)
    doc["auth"]["static_keys"] = [{"key_env": "T_USER", "user": "alice", "team": "default"},
                                  {"key_env": "T_ADMIN", "user": "root", "team": "ops", "role": "admin"}]
    gateway.file_raw = doc
    await gateway.store.create_eval_run({"id": "ev_a", "created_by": "root", "team": "ops", "status": "running", "executor": "cli",
                                         "request": {"models": ["small", "big"], "suite": "starter"}})
    assert (await client.post("/v1/admin/evals/runs/ev_a/apply", headers=admin, json={})).status_code in (409,)  # not finished (or no database)
    def hard(model, task, passed, n=10):
        return [{"model": model, "task_id": f"{task}{i}", "task": task, "level": 4 + i % 2, "passed": i < passed} for i in range(n)]
    # small passes 4 in 10 hard coding tasks to big's 10 in 10: 2.0 against big's 5.0. Only small ran general tasks: 5.0.
    results = hard("small", "coding", 4) + hard("small", "general", 6) + hard("big", "coding", 10)
    await gateway.store.update_eval_run("ev_a", status="done", results=results)
    shown = (await client.get("/v1/admin/evals/runs/ev_a", headers=admin)).json()["scores"]
    assert shown["small"]["tasks"]["coding"]["score"] == 2.0 and shown["big"]["tasks"]["coding"]["score"] == 5.0
    preview = (await client.post("/v1/admin/evals/runs/ev_a/apply", headers=admin, json={"models": ["small"], "dry_run": True})).json()
    assert preview["version"] is None and preview["changed"] == ["small coding 3→2.0", "small general 3→5.0"]
    assert preview["changes"][0] | {"low": None, "high": None} == {"model": "small", "task": "coding", "from": 3, "to": 2.0, "in_use": 3,
                                                                    "low": None, "high": None}
    r = await client.post("/v1/admin/evals/runs/ev_a/apply", headers=admin, json={"models": ["small"]})
    if gateway.store.kind != "postgres":
        assert r.status_code == 409  # saving configuration needs the database
        return
    assert r.status_code == 200 and r.json()["changed"] == preview["changed"] and r.json()["version"]
    assert gateway.cfg.model("small").quality["coding"] == 2.0 and gateway.cfg.model("big").quality["coding"] == 5
    # What a measured 5 means (big's 10 of 10 hard coding tasks, small's 6 of 10 general ones), for routing by cost per success.
    assert gateway.cfg.quality.measured_best_rate == {"coding": 1.0, "general": 0.6}
    assert (await gateway.store.get_eval_run("ev_a"))["applied_version"] == r.json()["version"]
    again = await client.post("/v1/admin/evals/runs/ev_a/apply", headers=admin, json={"models": ["small"]})
    assert again.json()["version"] is None and again.json()["changed"] == []
    over = await client.post("/v1/admin/evals/runs/ev_a/apply", headers=admin, json={"models": ["big"], "target": "quality_override"})
    assert over.json()["changed"] == ["big coding –→5.0"] and gateway.cfg.model("big").quality_sources["coding"] == "override"
    assert (await client.post("/v1/admin/evals/runs/nope/apply", headers=admin, json={})).status_code == 404


def test_app_still_builds():
    assert create_app  # imported for the in-process client fixtures above


# ------------------------------------------------------------------ the real Docker sandbox (where Docker exists, as on CI)

@pytest.mark.skipif(DockerSandbox().available() is not None, reason="needs Docker")
def test_docker_sandbox_isolates_model_code():
    box = DockerSandbox()
    tests = "from solution import f\ndef test_f():\n    assert f() == 1\n"
    assert run_tests(box, "def f():\n    return 1\n", tests)["passed"] == 1
    net = "import urllib.request\ndef f():\n    urllib.request.urlopen('http://example.com', timeout=3)\n    return 1\n"
    assert run_tests(box, net, tests)["passed"] == 0  # no network
    write = "def f():\n    open('/work/x', 'w').write('x')\n    return 1\n"
    assert run_tests(box, write, tests)["passed"] == 0  # read-only
    assert run_tests(box, "def f():\n    while True: pass\n", tests, timeout=3)["error"] == "timed out after 3s"


async def test_worker_loop_survives_errors_and_keeps_a_working_sandbox(caplog):
    class FlakyStore:
        def __init__(self):
            self.settings = {}

        async def set_setting(self, key, value):
            self.settings[key] = value

        async def list_eval_runs(self, limit=50):
            return []

        async def claim_eval_run(self):
            raise RuntimeError("database went away")

    class Ready:
        def available(self):
            return None

    store = FlakyStore()
    await jobs.worker_loop(store, "http://gw", once=True, sandbox_factory=Ready)
    assert store.settings["eval_runner"]["sandbox"] is True
    assert "eval runner cycle failed" in caplog.text


async def test_execute_records_a_crash(gateway, monkeypatch):
    run = await gateway.store.create_eval_run({"id": "ev_x", "created_by": "root", "team": "ops", "status": "running", "executor": "worker",
                                               "request": {"models": ["small"], "suite": "starter", "only": ["g1-*"]}})

    async def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(jobs, "run_suite", boom)
    row = await jobs.execute(gateway.store, run, "http://test", ADMIN_KEY)
    assert row["status"] == "failed" and row["error"] == "RuntimeError: kaboom"


def test_one_runaway_test_fails_alone():
    tests = ("from solution import f, g\n"
             "def test_fast():\n    assert f() == 1\n"
             "def test_stuck():\n    assert g() == 2\n"
             "def test_after():\n    assert f() == 1\n")
    out = run_tests(LocalSandbox(), "def f():\n    return 1\ndef g():\n    while True:\n        pass\n", tests, timeout=20, per_test=0.5)
    assert out["total"] == 3 and out["passed"] == 2
    assert out["failed"][0]["name"] == "test_stuck" and "took too long" in out["failed"][0]["error"]


async def test_eval_traffic_counts_as_spend_but_not_as_savings(gateway, client):
    from conftest import ledger

    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    await client.post("/v1/chat/completions", json=body, headers={**auth(), "x-meritorder-tags": "eval=starter-1"})
    await client.post("/v1/chat/completions", json=body, headers=auth())
    rows = {bool((r.get("tags") or {}).get("eval")): r for r in await ledger(gateway)}
    assert rows[True]["actual_cost"] > 0 and rows[True]["savings_routing"] == 0  # same model either way
    assert rows[False]["savings_routing"] > 0  # small instead of the baseline, big
    summary = await gateway.store.summary(datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC), "tag:eval")
    assert {r["key"] for r in summary} == {"starter-1", "(untagged)"}  # Reports → Breakdown → Tag: eval


async def test_a_run_stops_starting_tasks_at_its_spending_cap(monkeypatch):
    """No new task starts once the run has spent its cap; what ran still counts, and the run says why it stopped."""
    from meritorder.store import MemoryStore

    store = MemoryStore()
    tasks = [{"id": f"g1-{i}", "task": "general", "level": 1, "prompt": "?", "checks": [{"contains": "ok"}]} for i in range(10)]
    monkeypatch.setattr(jobs, "plan", lambda req: {"name": "custom", "tasks": tasks})
    await store.create_eval_run({"id": "ev_c", "created_by": "root", "team": "ops", "status": "queued", "executor": "worker",
                                 "request": {"models": ["m"], "suite": "custom", "max_cost_usd": 1.0}})
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=reply("ok"), headers={"x-meritorder-cost-usd": "0.4"}))
    async with httpx.AsyncClient(transport=transport) as c:
        row = await jobs.execute(store, await store.get_eval_run("ev_c"), "http://gw", "k", concurrency=1, client=c)
    assert row["status"] == "done" and len(row["results"]) == 3  # $0.40 each: the cap is reached on the third
    assert row["progress"]["stopped"] == "spending cap" and row["progress"]["cap"] == 1.0
    assert row["error"] == "stopped at its spending cap of $1.00 (spent $1.20); 7 task runs didn't start"
    assert jobs.default_cap(0.004) == 1.0 and jobs.default_cap(40) == 60.0 and jobs.default_cap(66.6) == 100.0
    lines = jobs.cost_warning({"usd": 900.0, "by_model": {"a": 600.0, "b": 300.0}, "judging_usd": 0, "unpriced": ["c"]}, 1350.0, 20.0)
    assert lines[1] == "By model: a $600.00, b $300.00." and "No known price for c" in lines[2]
    assert lines[-1].startswith("The organization has $20.00 of its monthly budget left")
