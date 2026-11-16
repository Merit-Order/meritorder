import json

from conftest import ADMIN_KEY, ledger

from meritorder.evals.runner import TaskResult, grade, load_suite, run_suite, score


def reply(text="", tool=None):
    msg = {"role": "assistant", "content": text}
    if tool:
        msg["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": tool[0], "arguments": json.dumps(tool[1])}}]
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}


def test_checks():
    assert grade({"contains": "canberra"}, reply("Canberra."))[0]
    assert not grade({"not_contains": "Dan"}, reply("Dan"))[0]
    assert grade({"equals": "OK"}, reply("OK."))[0] and not grade({"equals": "OK"}, reply("OK, sure"))[0]
    assert grade({"number": 155}, reply("2 hours 35 minutes = 155 minutes"))[0]
    assert grade({"number": 1200}, reply("1,200"))[0]
    assert grade({"json": {"name": "Maria Lopez", "age": 34}}, reply('```json\n{"name": "Maria Lopez", "age": 34}\n```'))[0]
    assert not grade({"json": {"age": 34}}, reply('{"age": "34"}'))[0]
    assert grade({"tool_call": {"name": "get_weather", "arguments": {"city": "Oslo"}}}, reply(tool=("get_weather", {"city": "oslo"})))[0]
    assert not grade({"tool_call": {"name": "get_weather"}}, reply("It's sunny"))[0]
    zip_check = {"regex_behaves": {"match": ["12345", "98765-4321"], "reject": ["1234", "12345-678"]}}
    assert grade(zip_check, reply(r"`^\d{5}(-\d{4})?$`".strip("`")))[0]
    assert not grade(zip_check, reply(r"\d{5}.*"))[0]


def test_starter_suite_and_scoring():
    suite = load_suite("starter")
    assert len(suite["tasks"]) == 24 and {t["task"] for t in suite["tasks"]} == {"coding", "general"}

    def results(model, hard_passed, hard=10):
        easy = [TaskResult(model=model, task_id=f"e{i}", task="coding", level=2, passed=True) for i in range(3)]
        return easy + [TaskResult(model=model, task_id=f"h{i}", task="coding", level=4 + i % 2, passed=i < hard_passed) for i in range(hard)]

    s = score(results("strong", 8) + results("mid", 7) + results("weak", 1) + results("thin", 3, hard=3))
    coding = {m: s[m]["tasks"]["coding"] for m in s}
    assert coding["strong"]["score"] == 5.0  # the best model measured
    assert coding["mid"]["score"] == 4.4  # 5 x 0.7 / 0.8: in proportion to the hard tasks it passes, easy ones don't count
    assert coding["weak"]["score"] == 1.0  # never below 1
    assert coding["thin"]["score"] is None and coding["thin"]["hard"] == {"passed": 3, "runs": 3}  # too few hard tasks to say
    assert coding["mid"]["low"] < 4.4 < coding["mid"]["high"]


async def test_run_suite_through_gateway(client, gateway):
    suite = {"name": "mini", "tasks": [
        {"id": "a", "task": "general", "level": 1, "prompt": "hello there", "checks": [{"contains": "hello"}]},
        {"id": "b", "task": "general", "level": 2, "prompt": "say banana", "checks": [{"contains": "cherry"}]}]}
    res = await run_suite("http://test", ADMIN_KEY, ["small"], suite, client=client)
    assert {r.task_id: r.passed for r in res} == {"a": True, "b": False}  # the mock echoes the prompt
    assert all(r.cost > 0 for r in res)
    await gateway.drain()
    assert all(row["tags"].get("eval", "").startswith("mini-") for row in await ledger(gateway))
