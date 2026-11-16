"""Batches in the gateway and the worker, request by request: routing at batch prices, refusals, budgets and the per-request
limit, splitting and fallback, reservations, polling with backoff and leases, settling into the ledger, retention, access."""

import copy
import io
import json
import time
from datetime import UTC, datetime, timedelta

import httpx
import openai
import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth

from meritorder.app import build_gateway, create_app
from meritorder.batches.content import Keyring, SealError
from meritorder.batches.poller import BatchPoller, WorkerContext, batch_loop, line_costs
from meritorder.batches.service import month_key
from meritorder.config import build_config
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

KEYS = [{"key": TEST_KEY, "user": "alice", "team": "default", "role": "user"},
        {"key": LEAD_KEY, "user": "lee", "team": "default", "role": "team_lead"},
        {"key": ADMIN_KEY, "user": "root", "team": "ops", "role": "admin"}]
CONFIG = {
    "baseline_model": "big",
    "backends": {
        "anthropic": {"format": "anthropic", "base_url": "https://api.anthropic.com", "auth": {"type": "api_key", "header": "x-api-key", "value": "sk-x"}},
        "mock-anthropic": {"kind": "mock", "format": "anthropic", "mock_latency_ms": 0},
        "mock-openai": {"kind": "mock", "format": "openai", "mock_latency_ms": 0},
        "openrouter": {"format": "openai", "base_url": "https://openrouter.ai/api/v1", "auth": {"type": "api_key", "value": "k"},
                       "batch": {"enabled": False}},  # a backend without a batch API, for these tests
    },
    "models": {
        "big": {"quality": {"coding": 5, "general": 5}, "routes": [
            {"backend": "anthropic", "model": "big-1", "price": {"input": 3, "output": 15}},
            {"backend": "mock-anthropic", "model": "big-m", "price": {"input": 4, "output": 20}}]},
        "small": {"quality": {"coding": 3, "general": 3}, "routes": [
            {"backend": "openrouter", "model": "small-or", "price": {"input": 0.01, "output": 0.05}},  # cheaper, but no batch API
            {"backend": "mock-openai", "model": "small-1", "price": {"input": 0.2, "output": 1.0, "batch": {"input": 0.08, "output": 0.4}}}]},
        "embed-small": {"kind": "embedding", "routes": [{"backend": "mock-openai", "model": "embed-1", "price": {"input": 0.02, "output": 0}}]},
    },
    "teams": {"default": {"budget_usd_month": 100, "user_budget_usd_month": 10}, "ops": {}},
    "auth": {"static_keys": KEYS},
}
MSG = {"id": "msg_1", "type": "message", "role": "assistant", "model": "big-1", "content": [{"type": "text", "text": "hi"}],
       "stop_reason": "end_turn", "usage": {"input_tokens": 1000, "output_tokens": 100}}


class Anthropic:
    """A fake of Anthropic's Message Batches: submit, then "in_progress" until told to end."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.status, self.submit_status, self.poll_status = "in_progress", 200, 200
        self.results: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            if self.submit_status != 200:
                return httpx.Response(self.submit_status, json={"error": {"message": "overloaded"}})
            self.batch = json.loads(request.content)["requests"]
            return httpx.Response(200, json={"id": "msgbatch_up", "processing_status": "in_progress"})
        if path.endswith("/cancel"):
            self.status = "canceling"
            return httpx.Response(200, json={"processing_status": "canceling"})
        if path.endswith("/results"):
            return httpx.Response(200, content=b"".join(json.dumps(r).encode() + b"\n" for r in self.results))
        if request.method == "DELETE":
            return httpx.Response(200, json={"type": "message_batch_deleted"})
        if self.poll_status != 200:
            return httpx.Response(self.poll_status, json={"error": {"message": "nope"}})
        return httpx.Response(200, json={"id": "msgbatch_up", "processing_status": self.status,
                                         "request_counts": {"processing": 0 if self.status == "ended" else len(self.batch), "succeeded": 0}})

    def succeed_all(self):
        self.status = "ended"
        self.results = [{"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": MSG}} for r in self.batch]


@pytest.fixture
def setup(tmp_path):
    def make(**changes):
        raw = copy.deepcopy(CONFIG)
        for path, value in changes.items():
            *head, last = path.split("__")
            node = raw
            for k in head:
                node = node.setdefault(k, {})
            node[last] = value
        cfg = build_config(raw, tmp_path)
        gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
        fake = Anthropic()
        gw.backends["anthropic"].client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test")
        return gw, fake, client
    return make


def lines(*rows) -> list[dict]:
    return [{"custom_id": cid, "params": {"model": model, "max_tokens": 50, "messages": [{"role": "user", "content": text}]}}
            for cid, model, text in rows]


async def due(gw) -> None:
    for j in gw.store._jobs.values():
        j["next_poll_at"] = datetime.now(UTC) - timedelta(seconds=1)


async def poll(gw) -> int:
    await due(gw)
    return await BatchPoller(gw).run_once()


async def test_batch_requests_route_only_to_batch_apis_at_batch_prices(setup):
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"), ("b", "small", "yo"))})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["meritorder"]["providers"] == {"anthropic": 1, "mock-openai": 1}  # not OpenRouter: it has no batch API
    assert fake.batch[0]["params"]["model"] == "big-1" and "cache_control" not in json.dumps(fake.batch[0])
    est = float(r.headers["x-meritorder-estimated-cost-usd"])
    assert est == pytest.approx(b["meritorder"]["estimated_cost_usd"], abs=1e-6) and est > 0
    month = month_key()
    assert gw.state._spend[f"{month}:user:alice"] == pytest.approx(est, abs=1e-6)  # reserved against the budgets
    lines_ = await gw.store.batch_lines(b["id"])
    assert [ln["backend"] for ln in lines_] == ["anthropic", "mock-openai"] and lines_[0]["upstream_model"] == "big-1"
    # The mock's job ends at once; Anthropic's is still running.
    assert await poll(gw) == 2
    assert (await gw.store.get_batch(b["id"]))["status"] == "in_progress"
    fake.succeed_all()
    await poll(gw)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["outcome"] == "completed" and done["counts"]["succeeded"] == 2 and done["reserved_cost"] == 0
    rows = {r["backend"]: r for r in gw.store.ledger}
    big = rows["anthropic"]
    assert big["service_tier"] == "batch" and big["batch_id"] == b["id"] and big["status"] == 200 and big["upstream_model"] == "big-1"
    assert big["actual_cost"] == pytest.approx((1000 * 1.5 + 100 * 7.5) / 1e6)  # half of $3/$15
    assert big["savings_batch"] == pytest.approx(big["actual_cost"]) and big["savings_routing"] == pytest.approx(0)
    small = rows["mock-openai"]
    assert small["actual_cost"] == pytest.approx((small["input_tokens"] * 0.08 + small["output_tokens"] * 0.4) / 1e6)  # the listed batch price
    assert small["savings_routing"] > 0  # against the baseline (big) at live prices
    total = sum(r["actual_cost"] for r in gw.store.ledger)
    assert gw.state._spend[f"{month}:user:alice"] == pytest.approx(total)  # the reservation was settled to what it cost
    assert done["actual_cost"] == pytest.approx(total) and done["savings_batch"] > 0
    results = [json.loads(x) for x in (await c.get(f"/v1/messages/batches/{b['id']}/results", headers=auth())).text.splitlines()]
    assert {x["custom_id"]: x["result"]["type"] for x in results} == {"a": "succeeded", "b": "succeeded"}


async def test_requests_the_policies_refuse_fail_on_their_own(setup):
    gw, _, c = setup(teams__default__allowed_models=["*", "!big"])
    client = openai.AsyncOpenAI(api_key=TEST_KEY, base_url="http://test/v1", http_client=c, max_retries=0)
    rows = [{"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": {"model": m, "messages": [{"role": "user", "content": "x"}]}}
            for cid, m in (("ok", "small"), ("denied", "big"), ("unknown", "nope"), ("embed", "embed-small"))]
    f = await client.files.create(file=("in.jsonl", io.BytesIO("\n".join(json.dumps(r) for r in rows).encode())), purpose="batch")
    b = await client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
    assert b.request_counts.total == 4 and b.request_counts.failed == 3 and b.model_extra["meritorder"]["providers"] == {"mock-openai": 1}
    await poll(gw)
    b = await client.batches.retrieve(b.id)
    assert b.status == "completed" and b.request_counts.completed == 1 and b.request_counts.failed == 3
    errors = {json.loads(x)["custom_id"]: json.loads(x)["response"] for x in (await client.files.content(b.error_file_id)).text.splitlines()}
    assert errors["denied"]["status_code"] == 403 and "not allowed for team" in errors["denied"]["body"]["error"]["message"]
    assert errors["unknown"]["status_code"] == 404 and "embedding model" in errors["embed"]["body"]["error"]["message"]
    assert len(gw.store.ledger) == 1  # refused before any provider saw them, like a live refusal: not in the ledger
    # Nothing at all can go: refused outright.
    f = await client.files.create(file=("in.jsonl", io.BytesIO(json.dumps(rows[1]).encode())), purpose="batch")
    with pytest.raises(openai.PermissionDeniedError, match="no request in this batch can go"):
        await client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")


async def test_a_batch_the_budget_cant_cover_is_refused_saying_which(setup):
    gw, _, c = setup(teams__default__user_budget_usd_month=0.001)
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "x" * 4000))})
    assert r.status_code == 429 and r.headers["x-meritorder-limit"] == "budget"
    msg = r.json()["error"]["message"]
    assert "estimated cost of $" in msg and "doesn't fit in alice's monthly budget of $0.00" in msg and "1 requests at batch prices" in msg
    gw.cfg.team("default").user_budget_usd_month = None
    gw.cfg.team("default").budget_usd_month = 0.0
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "x"))})
    assert r.status_code == 429 and "team 'default''s monthly budget" in r.json()["error"]["message"]
    gw.cfg.team("default").budget_usd_month = None
    gw.cfg.safeguards.org_budget_usd_month = 0.0
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "x"))})
    assert "the organization's monthly LLM budget" in r.json()["error"]["message"]


async def test_the_per_request_limit_applies_at_batch_prices(setup):
    gw, _, c = setup(safeguards__max_request_usd=0.01)
    reqs = lines(("fits", "big", "hi"))
    reqs.append({"custom_id": "huge", "params": {"model": "big", "max_tokens": 100_000, "messages": [{"role": "user", "content": "hi"}]}})
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": reqs})
    assert r.status_code == 200 and r.json()["request_counts"]["errored"] == 1
    refused = [ln for ln in await gw.store.batch_lines(r.json()["id"]) if ln["status"] == "refused"]
    assert refused[0]["custom_id"] == "huge" and "above the $0.01 per-request limit" in refused[0]["error"] and "at batch prices" in refused[0]["error"]


async def test_a_provider_that_refuses_the_job_hands_its_requests_to_the_next_route(setup):
    gw, fake, c = setup()
    fake.submit_status = 529
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"), ("b", "big", "yo"))})
    assert r.status_code == 200 and r.json()["meritorder"]["providers"] == {"mock-anthropic": 2}
    jobs = await gw.store.batch_jobs(r.json()["id"])
    assert sorted((j["backend"], j["status"]) for j in jobs) == [("anthropic", "failed"), ("mock-anthropic", "running")]
    await poll(gw)
    assert (await gw.store.get_batch(r.json()["id"]))["outcome"] == "completed"
    # With nowhere else to go, Anthropic's API says the batch failed, and why.
    gw, fake, c = setup(models__big__routes=[{"backend": "anthropic", "model": "big-1", "price": {"input": 3, "output": 15}}])
    fake.submit_status = 500
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    assert r.status_code == 502 and "every route that could take it failed: anthropic: anthropic: creating a message batch: overloaded" in r.text
    assert gw.state._spend[f"{month_key()}:user:alice"] == pytest.approx(0)  # nothing stays reserved


async def test_jobs_split_by_provider_limits_and_minimums(setup):
    gw, _, c = setup()
    gw.backends["mock-openai"].cfg.batch.max_requests = 2
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(*[(f"r{i}", "small", "x") for i in range(5)])})
    assert len(await gw.store.batch_jobs(r.json()["id"])) == 3
    # Too few requests for a provider's minimum: they move to their next route.
    gw.backends["anthropic"].cfg.batch.min_requests = 3
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "x"), ("b", "big", "y"))})
    assert r.json()["meritorder"]["providers"] == {"mock-anthropic": 2}
    gw.backends["mock-anthropic"].cfg.batch.min_requests = 3
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "x"), ("b", "small", "y"))})
    assert r.json()["meritorder"]["providers"] == {"mock-openai": 1}
    errs = [ln["error"] for ln in await gw.store.batch_lines(r.json()["id"]) if ln["status"] == "refused"]
    assert errs and "takes at least 3 requests" in errs[0]


async def test_cancelling_stops_what_hasnt_run_and_keeps_what_has(setup):
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"), ("b", "big", "yo"))})
    bid = r.json()["id"]
    r = await c.post(f"/v1/messages/batches/{bid}/cancel", headers=auth())
    assert r.json()["processing_status"] == "canceling" and fake.requests[-1].url.path.endswith("/cancel")
    assert (await c.post(f"/v1/messages/batches/{bid}/cancel", headers=auth())).status_code == 200  # twice is fine
    fake.status = "ended"
    fake.results = [{"custom_id": fake.batch[0]["custom_id"], "result": {"type": "succeeded", "message": MSG}},
                    {"custom_id": fake.batch[1]["custom_id"], "result": {"type": "canceled"}}]
    await poll(gw)
    b = (await c.get(f"/v1/messages/batches/{bid}", headers=auth())).json()
    assert b["processing_status"] == "ended" and b["request_counts"] == {"processing": 0, "succeeded": 1, "errored": 0, "canceled": 1, "expired": 0}
    assert (await gw.store.get_batch(bid))["outcome"] == "cancelled"
    statuses = sorted(r["status"] for r in gw.store.ledger)
    assert statuses == [200, 499] and sum(r["actual_cost"] for r in gw.store.ledger if r["status"] == 499) == 0
    out = [json.loads(x) for x in (await c.get(f"/v1/messages/batches/{bid}/results", headers=auth())).text.splitlines()]
    assert sorted(x["result"]["type"] for x in out) == ["canceled", "succeeded"]


async def test_polling_backs_off_and_survives_provider_trouble(setup, monkeypatch):
    gw, fake, c = setup()
    gw.cfg.batches.poll_seconds = 10
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    job = (await gw.store.batch_jobs(bid))[0]
    assert job["next_poll_at"] > datetime.now(UTC)  # not checked straight away
    assert await BatchPoller(gw).run_once() == 0
    waits = []
    for _ in range(3):
        before = datetime.now(UTC)
        await poll(gw)
        job = (await gw.store.batch_jobs(bid))[0]
        waits.append(round((job["next_poll_at"] - before).total_seconds()))
    assert waits == [10, 20, 40] and job["lease_owner"] is None  # nothing changed: the wait doubles
    fake.poll_status = 503
    alerts = []
    monkeypatch.setattr("meritorder.batches.poller.alerts.notify", lambda text, **kw: alerts.append(text) or _done())
    for _ in range(5):
        await poll(gw)
    job = (await gw.store.batch_jobs(bid))[0]
    assert job["failures"] == 5 and job["status"] == "running" and alerts and "can't be checked" in alerts[0]
    fake.poll_status = 404  # the provider lost it: its requests fail, at no cost
    await poll(gw)
    b = await gw.store.get_batch(bid)
    assert b["status"] == "ended" and b["outcome"] == "failed" and gw.store.ledger[0]["status"] == 502
    assert "no longer has this job" in gw.store.ledger[0]["error"]


async def _done():
    return None


async def test_leases_keep_two_workers_off_the_same_job(setup):
    gw, fake, c = setup()
    await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    await due(gw)
    first = await gw.store.claim_batch_jobs("w1")
    assert len(first) == 1 and await gw.store.claim_batch_jobs("w2") == []
    assert await gw.store.update_batch_job(first[0]["id"], "w2", polls=9) is None  # not w2's to change
    await gw.store.update_batch_job(first[0]["id"], "w1", release=True)
    assert len(await gw.store.claim_batch_jobs("w2")) == 1


async def test_a_job_the_gateway_never_finished_submitting_fails_after_a_while(setup):
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    job = (await gw.store.batch_jobs(r.json()["id"]))[0]
    gw.store._jobs[job["id"]].update(status="submitting")
    await poll(gw)
    assert gw.store._jobs[job["id"]]["status"] == "submitting"  # might still be on its way
    gw.store._jobs[job["id"]]["created_at"] -= timedelta(hours=1)
    await poll(gw)
    assert gw.store._jobs[job["id"]]["status"] == "failed" and (await gw.store.get_batch(r.json()["id"]))["outcome"] == "failed"


async def test_retention_deletes_the_providers_copies_and_unused_uploads(setup):
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    fake.succeed_all()
    await poll(gw)
    up = await c.post("/v1/files", headers=auth(), files={"file": ("x.jsonl", b"{}\n")}, data={"purpose": "batch"})
    gw.store._batches[bid]["ended_at"] -= timedelta(days=8)
    gw.store._files[up.json()["id"]]["created_at"] -= timedelta(days=8)
    poller = BatchPoller(gw)
    assert await poller.purge() == 1
    assert fake.requests[-1].method == "DELETE" and (await gw.store.get_batch(bid))["purged_at"]
    assert (await c.get(f"/v1/files/{up.json()['id']}/content", headers=auth())).status_code == 404
    r = await c.get(f"/v1/messages/batches/{bid}", headers=auth())
    assert r.json()["archived_at"] and r.json()["results_url"] is None
    r = await c.get(f"/v1/messages/batches/{bid}/results", headers=auth())
    assert r.status_code == 410 and "deleted 7 days after it ended" in r.text


async def test_only_the_owner_reads_results_and_admins_may_cancel(setup):
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    assert (await c.get(f"/v1/messages/batches/{bid}", headers=auth(LEAD_KEY))).status_code == 404
    assert (await c.get(f"/v1/messages/batches/{bid}", headers=auth(ADMIN_KEY))).status_code == 200
    assert (await c.get(f"/v1/batches/{bid}", headers=auth())).status_code == 404  # an Anthropic batch, not an OpenAI one
    fake.succeed_all()
    await poll(gw)
    r = await c.get(f"/v1/messages/batches/{bid}/results", headers=auth(ADMIN_KEY))
    assert r.status_code == 403 and "only the person who made a batch" in r.text
    assert (await c.post("/v1/messages/batches", headers={"x-api-key": "nope"}, json={})).status_code == 401
    rep = (await c.get("/v1/reports/batches", headers=auth(LEAD_KEY))).json()
    assert [x["id"] for x in rep["rows"]] == [bid] and rep["rows"][0]["can_cancel"] is False
    assert rep["totals"]["batches"] == 1 and rep["rows"][0]["status"] == "completed" and rep["rows"][0]["providers"] == {"anthropic": 1}
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("b", "big", "hi"))})
    rep = (await c.get("/v1/reports/batches", headers=auth(ADMIN_KEY))).json()
    assert rep["rows"][0]["can_cancel"] and rep["totals"]["running"] == 1
    assert (await c.post(f"/v1/messages/batches/{r.json()['id']}/cancel", headers=auth(ADMIN_KEY))).json()["processing_status"] == "canceling"
    assert (await gw.store.get_batch(r.json()["id"]))["cancelled_by"] == "root"


@pytest.mark.parametrize(("body", "message"), [
    ({"requests": []}, "non-empty list"),
    ({"requests": [{"custom_id": "a b", "params": {"messages": [], "max_tokens": 1}}]}, "1 to 64 letters"),
    ({"requests": [{"custom_id": "a", "params": {"messages": []}}]}, "max_tokens: required"),
    ({"requests": [{"custom_id": "a", "params": {"max_tokens": 1}}]}, "messages: required"),
    ({"requests": [{"custom_id": "a", "params": {"messages": [], "max_tokens": 1, "stream": True}}]}, "streaming"),
    ({"requests": lines(("a", "big", "x"), ("a", "big", "y"))}, "used twice"),
    ({"requests": ["x"]}, "needs custom_id and params"),
])
async def test_anthropic_batches_are_checked_as_anthropic_checks_them(setup, body, message):
    _, _, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json=body)
    assert r.status_code == 400 and message in r.json()["error"]["message"] and r.json()["type"] == "error"


def line(**kw) -> str:
    return json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions",
                       "body": {"model": "small", "messages": [{"role": "user", "content": "x"}]}, **kw})


@pytest.mark.parametrize(("content", "endpoint", "message"), [
    ("not json", "/v1/chat/completions", "isn't valid JSON"),
    (line(url="/v1/responses"), "/v1/chat/completions", "every line must call it"),
    (line() + "\n" + line(), "/v1/chat/completions", "used twice"),
    (line(method="GET"), "/v1/chat/completions", "method must be POST"),
    (line(body={"model": "small", "messages": [], "stream": True}), "/v1/chat/completions", "streaming"),
    (line(body={"model": "small"}), "/v1/chat/completions", "needs a messages list"),
    (line(url="/v1/embeddings", body={"model": "embed-small"}), "/v1/embeddings", "needs input"),
    (json.dumps({"custom_id": "", "body": {}}), "/v1/chat/completions", "has no custom_id"),
    (json.dumps({"custom_id": "a"}), "/v1/chat/completions", "needs custom_id, method, url and a body"),
    ("\n", "/v1/chat/completions", "has no requests"),
    (line(), "/v1/completions", "endpoint must be one of"),
    (line(url="/v1/responses", body={"model": "small", "input": "hi", "previous_response_id": "resp_1"}), "/v1/responses", "previous_response_id"),
])
async def test_openai_batches_are_checked_as_openai_checks_them(setup, content, endpoint, message):
    _, _, c = setup()
    f = (await c.post("/v1/files", headers=auth(), files={"file": ("in.jsonl", content.encode())}, data={"purpose": "batch"})).json()
    r = await c.post("/v1/batches", headers=auth(), json={"input_file_id": f["id"], "endpoint": endpoint, "completion_window": "24h"})
    assert r.status_code == 400 and message in r.json()["error"]["message"]


async def test_files_api_edges(setup, monkeypatch):
    gw, _, c = setup()
    assert (await c.post("/v1/files", headers=auth(), files={"file": ("a", b"x")}, data={"purpose": "fine-tune"})).status_code == 400
    r = await c.post("/v1/files", headers={**auth(), "content-type": "application/json"}, content=b"{}")
    assert r.status_code == 400 and "multipart/form-data" in r.text
    r = await c.post("/v1/files", headers=auth(), files={"other": ("a", b"x")}, data={"purpose": "batch"})
    assert r.status_code == 400 and "file field" in r.text
    f = (await c.post("/v1/files", headers=auth(), files={"file": ("in.jsonl", line().encode())}, data={"purpose": "batch"})).json()
    assert (await c.get(f"/v1/files/{f['id']}", headers=auth())).json()["filename"] == "in.jsonl"
    assert (await c.get(f"/v1/files/{f['id']}", headers=auth(LEAD_KEY))).status_code == 404  # someone else's
    assert (await c.get(f"/v1/files/{f['id']}/content", headers=auth())).content == line().encode()
    listed = (await c.get("/v1/files", headers=auth())).json()["data"]
    assert [x["id"] for x in listed] == [f["id"]]
    r = await c.post("/v1/batches", headers=auth(), json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions", "completion_window": "1h"})
    assert r.status_code == 400 and "24h" in r.text
    b = (await c.post("/v1/batches", headers=auth(), json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions"})).json()
    r = await c.get(f"/v1/files/{gw.store._batches[b['id']]['output_file_id']}/content", headers=auth())
    assert r.status_code == 400 and "hasn't finished" in r.text
    assert (await c.post("/v1/batches", headers=auth(), json={"input_file_id": gw.store._batches[b["id"]]["output_file_id"],
                                                               "endpoint": "/v1/chat/completions"})).status_code == 400
    await poll(gw)
    b = (await c.get(f"/v1/batches/{b['id']}", headers=auth())).json()
    assert [x["purpose"] for x in (await c.get("/v1/files?purpose=batch_output", headers=auth())).json()["data"]] == ["batch_output"] * 2
    assert (await c.get(f"/v1/files/{b['output_file_id']}", headers=auth())).json()["purpose"] == "batch_output"
    # Deleting the output file deletes the results at the providers too.
    assert (await c.delete(f"/v1/files/{b['output_file_id']}", headers=auth())).json()["deleted"] is True
    assert (await c.get(f"/v1/files/{b['output_file_id']}/content", headers=auth())).status_code == 404
    assert (await c.get(f"/v1/files/{b['output_file_id']}", headers=auth())).status_code == 404
    r = await c.get(f"/v1/files/{b['error_file_id']}/content", headers=auth())
    assert r.status_code == 410
    assert (await c.delete(f"/v1/files/{f['id']}", headers=auth())).json()["deleted"] is True
    r = await c.post("/v1/batches", headers=auth(), json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions"})
    assert r.status_code == 404 and "was deleted" in r.text
    r = await c.post("/v1/batches", headers={**auth(), "content-type": "application/json"}, content=b"[1]")
    assert r.status_code == 400 and "JSON object" in r.text
    r = await c.post("/v1/batches", headers={**auth(), "content-type": "application/json"}, content=b"{")
    assert r.status_code == 400 and "valid JSON" in r.text
    # A database-backed gateway without a key holds no uploads: they'd be in the clear.
    monkeypatch.setattr(gw.store, "kind", "postgres")
    r = await c.post("/v1/files", headers=auth(), files={"file": ("in.jsonl", b"x")}, data={"purpose": "batch"})
    assert r.status_code == 503 and "MERITORDER_BATCH_KEY" in r.text
    gw.cfg.batches.enabled = False
    assert (await c.post("/v1/files", headers=auth(), files={"file": ("in.jsonl", b"x")}, data={"purpose": "batch"})).status_code == 403
    assert (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "x"))})).status_code == 403


def test_uploads_are_sealed_and_a_wrong_key_says_so():
    s = Keyring("one key", memory=False)
    chunks = s.seal("file-1", b"a" * (5 * 1024 * 1024))
    assert len(chunks) == 2 and b"aaaa" not in chunks[0] and s.sealed_with(chunks) == s.id
    assert s.open("file-1", chunks) == b"a" * (5 * 1024 * 1024)
    with pytest.raises(SealError, match="MERITORDER_BATCH_KEY"):
        Keyring("another key", memory=False).open("file-1", chunks)
    with pytest.raises(SealError):
        s.open("file-2", chunks)  # a chunk moved to another file doesn't open
    assert Keyring(None, memory=False).key is None and Keyring(None, memory=True).key is not None
    with pytest.raises(SealError, match="no batch key"):
        Keyring(None).seal("x", b"y")


async def test_responses_and_embeddings_batches(setup):
    gw, _, c = setup()
    client = openai.AsyncOpenAI(api_key=TEST_KEY, base_url="http://test/v1", http_client=c, max_retries=0)
    rows = [{"custom_id": "r", "method": "POST", "url": "/v1/responses", "body": {"model": "small", "input": "hello there"}}]
    f = await client.files.create(file=("in.jsonl", io.BytesIO(json.dumps(rows[0]).encode())), purpose="batch")
    b = await client.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h")
    emb = [{"custom_id": "e", "method": "POST", "url": "/v1/embeddings", "body": {"model": "embed-small", "input": ["a", "b"]}},
           {"custom_id": "x", "method": "POST", "url": "/v1/embeddings", "body": {"model": "small", "input": "a"}}]
    f2 = await client.files.create(file=("e.jsonl", io.BytesIO("\n".join(json.dumps(r) for r in emb).encode())), purpose="batch")
    b2 = await client.batches.create(input_file_id=f2.id, endpoint="/v1/embeddings", completion_window="24h")
    await poll(gw)
    out = json.loads((await client.files.content((await client.batches.retrieve(b.id)).output_file_id)).text)
    assert out["response"]["body"]["object"] == "response" and out["response"]["body"]["output"][0]["content"][0]["text"].startswith("[small-1]")
    b2 = await client.batches.retrieve(b2.id)
    out = json.loads((await client.files.content(b2.output_file_id)).text)
    assert out["custom_id"] == "e" and len(out["response"]["body"]["data"]) == 2 and out["response"]["body"]["model"] == "embed-small"
    err = json.loads((await client.files.content(b2.error_file_id)).text)
    assert err["custom_id"] == "x" and "not configured" in err["response"]["body"]["error"]["message"]
    rows = [r for r in gw.store.ledger if r["client_format"] == "embeddings"]
    assert rows and rows[0]["service_tier"] == "batch" and rows[0]["savings_routing"] == 0 and rows[0]["savings_batch"] > 0


async def test_lists_page_like_the_providers(setup):
    gw, _, c = setup()
    ids = []
    for i in range(3):
        ids.append((await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines((f"a{i}", "small", "x"))})).json()["id"])
    for i, bid in enumerate(ids):
        gw.store._batches[bid]["created_at"] -= timedelta(minutes=10 - i)
    newest = ids[::-1]
    page = (await c.get("/v1/messages/batches?limit=2", headers=auth())).json()
    assert [x["id"] for x in page["data"]] == newest[:2] and page["has_more"] and page["last_id"] == newest[1]
    page = (await c.get(f"/v1/messages/batches?limit=2&after_id={newest[1]}", headers=auth())).json()
    assert [x["id"] for x in page["data"]] == newest[2:] and not page["has_more"]
    page = (await c.get(f"/v1/messages/batches?limit=1&before_id={newest[2]}", headers=auth())).json()
    assert [x["id"] for x in page["data"]] == [newest[1]] and page["has_more"]
    f = (await c.post("/v1/files", headers=auth(), files={"file": ("in.jsonl", line().encode())}, data={"purpose": "batch"})).json()
    first = (await c.post("/v1/batches", headers=auth(), json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions"})).json()
    page = (await c.get("/v1/batches?limit=1", headers=auth())).json()
    assert page["data"][0]["id"] == first["id"] and not page["has_more"]
    assert (await c.get(f"/v1/batches?after={first['id']}", headers=auth())).json()["data"] == []
    assert (await c.delete(f"/v1/messages/batches/{ids[0]}", headers=auth())).status_code == 400  # not ended yet


async def test_spend_counters_seed_from_the_ledger_and_running_reservations(setup):
    gw, _, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    est = r.json()["meritorder"]["estimated_cost_usd"]
    gw.state._spend.clear()  # Redis flushed
    assert await gw.spend("user", "alice") == pytest.approx(est, abs=1e-6)
    assert await gw.spend("team", "default") == pytest.approx(est, abs=1e-6) and await gw.spend("org", "all") == pytest.approx(est, abs=1e-6)
    assert await gw.store.batch_reserved(month_key(), "user_id", "bob") == 0


async def test_results_say_when_a_provider_cant_be_read(setup):
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    fake.succeed_all()
    await poll(gw)
    fake.results = [{"custom_id": "r0", "result": {"type": "succeeded", "message": {"content": "not a list"}}}]
    gw.backends["anthropic"].client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500, content=b"down")))
    out = json.loads((await c.get(f"/v1/messages/batches/{bid}/results", headers=auth())).text)
    assert out["result"]["type"] == "errored" and "couldn't be read from anthropic" in out["result"]["error"]["error"]["message"]


async def test_settled_costs_use_the_prices_the_job_was_submitted_at(setup):
    gw, _, _ = setup()
    from meritorder.pricing import Usage

    b = {"team": "default", "tags": {}}
    line = {"model": "big", "upstream_model": "big-1", "kind": "messages"}
    prices = {"big|big-1": {"standard": {"input": 3, "output": 15}}}  # no batch price stored: the published discount
    costs = line_costs(gw.cfg, b, line, prices, Usage(input_tokens=1_000_000))
    assert costs.actual == pytest.approx(1.5) and costs.savings_batch == pytest.approx(1.5)
    costs = line_costs(gw.cfg, {**b, "tags": {"eval": "x"}}, line, prices, Usage(input_tokens=1_000_000))
    assert costs.savings_routing == 0


async def test_the_worker_loop_follows_the_config_and_keeps_going(setup, monkeypatch):
    gw, _, c = setup()
    await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "small", "hi"))})
    await due(gw)

    class Follower:
        calls = 0

        async def current(self):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("database hiccup")
            if self.calls > 2:
                raise SystemExit
            return gw.cfg

    async def no_wait(_):
        return None

    monkeypatch.setattr("meritorder.batches.poller.asyncio.sleep", no_wait)
    with pytest.raises(SystemExit):
        await batch_loop(gw.store, gw.state, Follower())
    assert (await gw.store.list_batches())[0]["status"] == "ended"
    ctx = WorkerContext(gw.store, gw.state)
    await ctx.use(gw.cfg)
    backends = ctx.backends
    await ctx.use(gw.cfg)
    assert ctx.backends is backends
    await ctx.close()


async def test_a_model_without_a_batch_route_says_why_route_by_route(setup):
    gw, _, c = setup(models__live_only={"quality": {"coding": 3, "general": 3}, "routes": [
        {"backend": "openrouter", "model": "lo-1", "price": {"input": 1, "output": 1}}]})
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "live_only", "hi"), ("b", "small", "hi"))})
    refused = [ln for ln in await gw.store.batch_lines(r.json()["id"]) if ln["status"] == "refused"][0]
    assert "no route for 'live_only' has a batch API" in refused["error"] and "openrouter: batch.enabled is false" in refused["error"]
    gw.cfg.team("default").allowed_backends = ["*", "!openrouter"]
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "live_only", "hi"), ("b", "small", "hi"))})
    refused = [ln for ln in await gw.store.batch_lines(r.json()["id"]) if ln["status"] == "refused"][0]
    assert "openrouter: not allowed for team 'default'" in refused["error"]
    gw.cfg.team("default").allowed_backends = ["*"]
    gw.backends["mock-openai"].record_failure(time.time(), None, 401)  # its circuit is open: nothing can take "small" now
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "small", "hi"), ("b", "big", "hi"))})
    refused = [ln for ln in await gw.store.batch_lines(r.json()["id"]) if ln["status"] == "refused"][0]
    assert "mock-openai: unavailable right now" in refused["error"]


async def test_poller_edges(setup, monkeypatch):
    gw, fake, c = setup()
    alerts = []

    async def notify(text, **kw):
        alerts.append(text)

    monkeypatch.setattr("meritorder.batches.poller.alerts.notify", notify)
    # A job whose batch is gone.
    await gw.store.add_batch_jobs([{"id": "bj_orphan", "batch_id": "batch_gone", "backend": "anthropic", "api": "anthropic", "status": "running"}])
    await poll(gw)
    assert gw.store._jobs["bj_orphan"]["status"] == "failed"
    # A batch on a backend that's since been switched off: kept, and said.
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    live = gw.backends.pop("anthropic")
    await poll(gw)
    job = (await gw.store.batch_jobs(bid))[0]
    assert job["status"] == "running" and job["failures"] == 1 and "isn't enabled now" in alerts[-1]
    gw.store._batches[bid]["status"] = "ended"
    gw.store._batches[bid]["ended_at"] = datetime.now(UTC) - timedelta(days=30)
    assert await BatchPoller(gw).purge() == 0  # can't delete it there yet: tried again later
    gw.store._batches[bid].update(status="in_progress", ended_at=None)
    gw.backends["anthropic"] = live
    # A cancel the gateway couldn't send is sent on the next check.
    gw.store._batches[bid]["cancel_requested_at"] = datetime.now(UTC)
    await poll(gw)
    assert any(r.url.path.endswith("/cancel") for r in fake.requests) and (await gw.store.batch_jobs(bid))[0]["cancel_sent"]
    # One job failing to process doesn't stop the pass; it's tried again later.
    await gw.store.add_batch_jobs([{"id": "bj_x", "batch_id": bid, "backend": "anthropic", "api": "anthropic", "status": "running"}])

    async def boom(job):
        raise RuntimeError("bad row")

    poller = BatchPoller(gw)
    monkeypatch.setattr(poller, "process", boom)
    await due(gw)
    assert await poller.run_once() >= 1
    assert gw.store._jobs["bj_x"]["failures"] == 1 and gw.store._jobs["bj_x"]["lease_owner"] is None


async def test_a_provider_that_lost_the_mock_batch_fails_it(setup):
    gw, _, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "small", "hi"))})
    job = (await gw.store.batch_jobs(r.json()["id"]))[0]
    await gw.store.set_setting(f"mockbatch:{job['provider_id']}", {})
    await poll(gw)
    b = await gw.store.get_batch(r.json()["id"])
    assert b["outcome"] == "failed" and gw.store.ledger[0]["error"] == "the mock batch is gone"
    poller = BatchPoller(gw)
    assert (await poller.finish(b["id"]))["status"] == "ended"  # already ended: left as it is
    assert await poller.finish("nope") is None


async def test_finalizing_while_results_are_settled(setup):
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    job = (await gw.store.batch_jobs(bid))[0]
    gw.store._jobs[job["id"]]["status"] = "ended"
    b = await BatchPoller(gw).finish(bid)
    assert b["status"] == "finalizing" and b["finalizing_at"]
    assert (await c.get(f"/v1/messages/batches/{bid}", headers=auth())).json()["processing_status"] == "ended"
    fake.succeed_all()
    await poll(gw)  # the ended job is settled on its next claim
    assert (await gw.store.get_batch(bid))["status"] == "ended"


class OpenAIFake:
    """A fake of OpenAI's Files and Batch APIs: one batch, completed at once, one request succeeded and one failed."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.lines: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/files":
            data = request.content.split(b"Content-Type: application/jsonl\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
            self.lines = [json.loads(x) for x in data.splitlines() if x.strip()]
            return httpx.Response(200, json={"id": "file-in"})
        if request.method == "POST" and path == "/v1/batches":
            return httpx.Response(200, json={"id": "batch_up", "status": "validating"})
        if path == "/v1/batches/batch_up":
            return httpx.Response(200, json={"id": "batch_up", "status": "completed", "output_file_id": "file-out", "error_file_id": "file-err",
                                             "request_counts": {"total": 2, "completed": 1, "failed": 1}})
        ok, bad = self.lines[0]["custom_id"], self.lines[1]["custom_id"]
        if path == "/v1/files/file-out/content":
            body = {"id": "c", "object": "chat.completion", "created": 1, "model": "small-oa", "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
            return httpx.Response(200, content=json.dumps({"custom_id": ok, "response": {"status_code": 200, "body": body}}).encode())
        if path == "/v1/files/file-err/content":
            return httpx.Response(200, content=json.dumps({"custom_id": bad, "response": {"status_code": 400, "body": {
                "error": {"message": "bad request", "type": "invalid_request_error"}}}}).encode())
        return httpx.Response(200, json={"deleted": True})


async def test_openai_providers_output_and_error_files_go_to_the_right_client_files(setup):
    gw, _, c = setup(backends__openai={"format": "openai", "base_url": "https://api.openai.com/v1", "auth": {"type": "api_key", "value": "k"}},
                     models__small__routes=[{"backend": "openai", "model": "small-oa", "price": {"input": 0.1, "output": 0.4}}])
    fake = OpenAIFake()
    gw.backends["openai"].client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("ok", "small", "hi"), ("bad", "small", "yo"))})
    assert r.status_code == 200 and r.json()["meritorder"]["providers"] == {"openai": 2}
    assert fake.lines[0]["url"] == "/v1/chat/completions" and fake.lines[0]["body"]["model"] == "small-oa"
    await poll(gw)
    out = {json.loads(x)["custom_id"]: json.loads(x)["result"] for x in (await c.get(f"/v1/messages/batches/{r.json()['id']}/results",
                                                                                         headers=auth())).text.splitlines()}
    assert out["ok"]["type"] == "succeeded" and out["ok"]["message"]["content"][0]["text"] == "done"  # translated to Anthropic's format
    assert out["bad"]["error"]["error"] == {"type": "invalid_request_error", "message": "bad request"}
    # An OpenAI client's error file has only the failures, though the gateway reads just the provider's error file for it.
    f = (await c.post("/v1/files", headers=auth(), files={"file": ("in.jsonl", (line(custom_id="ok") + "\n" + line(custom_id="bad")).encode())},
                      data={"purpose": "batch"})).json()
    b = (await c.post("/v1/batches", headers=auth(), json={"input_file_id": f["id"], "endpoint": "/v1/chat/completions"})).json()
    await poll(gw)
    b = (await c.get(f"/v1/batches/{b['id']}", headers=auth())).json()
    errs = [json.loads(x) for x in (await c.get(f"/v1/files/{b['error_file_id']}/content", headers=auth())).text.splitlines()]
    assert [e["custom_id"] for e in errs] == ["bad"] and errs[0]["response"]["status_code"] == 400
    oks = [json.loads(x) for x in (await c.get(f"/v1/files/{b['output_file_id']}/content", headers=auth())).text.splitlines()]
    assert [o["custom_id"] for o in oks] == ["ok"]
    statuses = sorted(r["status"] for r in gw.store.ledger if r["batch_id"] == b["id"])
    assert statuses == [200, 400]


async def test_embeddings_batches_skip_batch_apis_that_take_none(setup):
    """Azure's batch deployments take no embedding model; Mistral's batch API does (as do OpenAI's, OpenRouter's, Gemini's)."""
    sent = []

    def mistral(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-m"})
        return httpx.Response(200, json={"id": "job-m", "status": "QUEUED"})

    gw, _, c = setup(backends__azure={"format": "openai", "base_url": "https://res.openai.azure.com/openai/v1",
                                      "auth": {"type": "api_key", "header": "api-key", "value": "k"}},
                     backends__mistral={"format": "openai", "base_url": "https://api.mistral.ai/v1", "auth": {"type": "api_key", "value": "k"}},
                     models__memb={"kind": "embedding", "routes": [
                         {"backend": "azure", "model": "emb-dep", "price": {"input": 0.01, "output": 0}},
                         {"backend": "mistral", "model": "mistral-embed", "price": {"input": 0.1, "output": 0}}]})
    gw.backends["mistral"].client = httpx.AsyncClient(transport=httpx.MockTransport(mistral))
    row = {"custom_id": "e", "method": "POST", "url": "/v1/embeddings", "body": {"model": "memb", "input": ["a"]}}
    up = (await c.post("/v1/files", headers=auth(), files={"file": ("e.jsonl", json.dumps(row).encode())}, data={"purpose": "batch"})).json()
    r = await c.post("/v1/batches", headers=auth(), json={"input_file_id": up["id"], "endpoint": "/v1/embeddings"})
    assert r.status_code == 200, r.text
    assert r.json()["meritorder"]["providers"] == {"mistral": 1}  # not Azure, though cheaper
    assert json.loads(sent[-1].content)["endpoint"] == "/v1/embeddings"
    gw.backends["mistral"].batch_api = None
    up = (await c.post("/v1/files", headers=auth(), files={"file": ("e.jsonl", json.dumps(row).encode())}, data={"purpose": "batch"})).json()
    r = await c.post("/v1/batches", headers=auth(), json={"input_file_id": up["id"], "endpoint": "/v1/embeddings"})
    assert r.status_code == 400 and "Azure" in r.text and "Mistral" in r.text
