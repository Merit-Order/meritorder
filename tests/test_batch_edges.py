"""Batches where timing, failures and access matter: a worker settling a job while the gateway is still submitting the
batch, batches checked against one budget at once, who may delete a batch, jobs that wait or hang on their way to a
provider, settling late, a cancel once the providers are done, an interrupted settle, and a provider reply that isn't JSON."""

import asyncio
from datetime import timedelta

import httpx
import pytest
from conftest import ADMIN_KEY, auth
from test_batch_service import lines, poll, setup  # noqa: F401  (setup is the fixture)

from meritorder.batches.service import month_key


async def test_a_job_settled_before_the_create_call_returns_is_released_once(setup):  # noqa: F811
    """The worker may settle a quick job while the gateway is still handing the batch's other jobs to their providers. The
    gateway's last update must not put that job's estimate back on the batch's reservation, or finishing the batch takes it
    off the spend counters a second time."""
    gw, fake, c = setup()

    async def slow_submit(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/messages/batches":
            # While this upload is under way, the mock provider's job has ended and the worker settles it.
            for _ in range(100):
                if any(j["backend"] == "mock-openai" and j["status"] == "running" for j in gw.store._jobs.values()):
                    break
                await asyncio.sleep(0)
            await poll(gw)
        return fake(request)

    gw.backends["anthropic"].client = httpx.AsyncClient(transport=httpx.MockTransport(slow_submit))
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "small", "hi"), ("b", "big", "yo"))})
    assert r.status_code == 200 and r.json()["meritorder"]["providers"] == {"anthropic": 1, "mock-openai": 1}
    bid = r.json()["id"]
    assert len([row for row in gw.store.ledger if row["batch_id"] == bid]) == 1  # the mock's job settled during the create call
    fake.succeed_all()
    await poll(gw)
    done = await gw.store.get_batch(bid)
    assert done["status"] == "ended" and done["reserved_cost"] == 0
    total = sum(row["actual_cost"] for row in gw.store.ledger)
    assert gw.state._spend[f"{month_key()}:user:alice"] == pytest.approx(total)  # what the batch cost, no less


async def test_batches_made_at_the_same_time_cant_both_fit_one_budget(setup, monkeypatch):  # noqa: F811
    """Two batches checked against the same budget at once: each fits on its own, both together don't. The budget binds:
    one of them is refused (or both, and the client tries again), never both accepted."""
    gw, _, c = setup(teams__default__user_budget_usd_month=None)
    one = lines(("a", "big", "x" * 4000))
    est = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": one})).json()["meritorder"]["estimated_cost_usd"]
    gw.state._spend.clear()
    gw.store._batches.clear()
    gw.cfg.team("default").user_budget_usd_month = est * 1.5
    real = gw.state.get_spend

    async def slow_get_spend(key):  # Redis is a round trip away: other requests run meanwhile
        await asyncio.sleep(0)
        return await real(key)

    monkeypatch.setattr(gw.state, "get_spend", slow_get_spend)
    a, b = await asyncio.gather(c.post("/v1/messages/batches", headers=auth(), json={"requests": one}),
                                c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("b", "big", "x" * 4000))}))
    accepted = [r for r in (a, b) if r.status_code == 200]
    assert len(accepted) <= 1, [r.status_code for r in (a, b)]
    assert all(r.status_code == 429 and r.headers["x-meritorder-limit"] == "budget" for r in (a, b) if r.status_code != 200)
    assert gw.state._spend[f"{month_key()}:user:alice"] == pytest.approx(est * len(accepted), abs=1e-6)  # refused ones hold nothing


async def test_an_admin_cant_delete_someone_elses_batch(setup):  # noqa: F811
    """Admins may see and cancel anyone's batch, but its results are its owner's: deleting them is the owner's call."""
    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    fake.succeed_all()
    await poll(gw)
    r = await c.delete(f"/v1/messages/batches/{bid}", headers=auth(ADMIN_KEY))
    assert r.status_code == 403 and "only the person who made a batch can delete it" in r.text
    assert not (await gw.store.get_batch(bid))["purged_at"] and not any(q.method == "DELETE" for q in fake.requests)
    assert (await c.get(f"/v1/messages/batches/{bid}/results", headers=auth())).status_code == 200
    assert (await c.delete(f"/v1/messages/batches/{bid}", headers=auth())).status_code == 200


async def test_a_job_waiting_its_turn_to_be_submitted_isnt_taken_for_a_stopped_gateways(setup, monkeypatch):  # noqa: F811
    """The worker fails a job still "submitting" 30 minutes after it was made: its gateway must have stopped. A job that
    waited that long for its turn (four go to the providers at a time) hasn't: if the worker failed it, the gateway would
    then submit it anyway, and its requests would run with nothing left to bill them to."""
    gw, fake, c = setup()
    made = gw.store.create_batch

    async def queued_an_hour(b, lines_, jobs):
        out = await made(b, lines_, jobs)
        for j in jobs:
            gw.store._jobs[j["id"]]["created_at"] -= timedelta(hours=1)
        return out

    async def watched_submit(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/messages/batches":
            await poll(gw)  # the worker looks while the upload is under way
        return fake(request)

    monkeypatch.setattr(gw.store, "create_batch", queued_an_hour)
    gw.backends["anthropic"].client = httpx.AsyncClient(transport=httpx.MockTransport(watched_submit))
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    assert r.status_code == 200, r.text  # not "no provider accepted the batch": Anthropic's did
    bid = r.json()["id"]
    assert (await gw.store.batch_jobs(bid))[0]["status"] == "running"
    fake.succeed_all()
    await poll(gw)
    rows = [row for row in gw.store.ledger if row["batch_id"] == bid]
    assert [(row["status"], row["actual_cost"] > 0) for row in rows] == [(200, True)]  # billed as it ran
    assert (await gw.store.get_batch(bid))["outcome"] == "completed"


async def test_a_provider_too_slow_to_take_a_job_hands_it_to_the_next_route(setup, monkeypatch):  # noqa: F811
    """A submission is cut off well before the worker would take the job for a stopped gateway's, so the two can't both
    decide what happened to it. The provider counts as having refused it."""
    from meritorder.batches import poller, service

    assert service.SUBMIT_LIMIT < poller.STUCK
    gw, fake, c = setup()
    monkeypatch.setattr(service, "SUBMIT_LIMIT", timedelta(seconds=0.05))

    async def hangs(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/messages/batches":
            await asyncio.sleep(1)
        return fake(request)

    gw.backends["anthropic"].client = httpx.AsyncClient(transport=httpx.MockTransport(hangs))
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    assert r.status_code == 200 and r.json()["meritorder"]["providers"] == {"mock-anthropic": 1}
    failed = [j for j in await gw.store.batch_jobs(r.json()["id"]) if j["backend"] == "anthropic"][0]
    assert failed["status"] == "failed" and "didn't take the job within" in failed["error"]


async def test_a_job_settled_long_after_it_ended_lands_in_the_ledger_when_it_is_counted(setup):  # noqa: F811
    """Settling can come long after the provider ended the job: its results couldn't be read for a while, or its backend
    was switched off. Its ledger rows are stamped when they're written, as the spend counters count them, so the two
    agree in every month (and the rows fall in the hours the rollups and the ClickHouse copy look at again)."""
    from datetime import UTC, datetime

    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    job = (await gw.store.batch_jobs(bid))[0]
    gw.store._jobs[job["id"]].update(status="ended", ended_at=datetime.now(UTC) - timedelta(days=40))  # last month
    fake.succeed_all()
    await poll(gw)
    assert (await gw.store.get_batch(bid))["status"] == "ended"
    row = next(row for row in gw.store.ledger if row["batch_id"] == bid)
    assert row["actual_cost"] > 0 and datetime.now(UTC) - row["ts"] < timedelta(hours=1)
    counted = gw.state._spend[f"{month_key()}:user:alice"]
    gw.state._spend.clear()  # Redis flushed: the counter is seeded again from the ledger
    assert await gw.spend("user", "alice") == pytest.approx(counted) == pytest.approx(row["actual_cost"])


async def test_cancelling_a_batch_the_providers_have_finished_changes_nothing(setup):  # noqa: F811
    """Once every job has ended (the batch is finalizing: its results are being settled), there's nothing left to stop. A
    cancel then leaves it as it is: it completes, and its requests aren't called cancelled."""
    from meritorder.batches.poller import BatchPoller

    gw, fake, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "big", "hi"))})
    bid = r.json()["id"]
    job = (await gw.store.batch_jobs(bid))[0]
    gw.store._jobs[job["id"]]["status"] = "ended"
    assert (await BatchPoller(gw).finish(bid))["status"] == "finalizing"
    assert (await c.get("/v1/reports/batches", headers=auth())).json()["rows"][0]["can_cancel"] is False  # the console offers none
    r = await c.post(f"/v1/messages/batches/{bid}/cancel", headers=auth())
    assert r.status_code == 200 and r.json()["cancel_initiated_at"] is None
    fake.succeed_all()
    await poll(gw)
    done = await gw.store.get_batch(bid)
    assert done["outcome"] == "completed" and done["cancelled_by"] is None and not any(q.url.path.endswith("/cancel") for q in fake.requests)


@pytest.mark.parametrize("how", ["lost", "stuck", "failed"])
async def test_a_failed_job_whose_settling_was_interrupted_is_settled_on_the_next_pass(setup, monkeypatch, how):  # noqa: F811
    """A job fails three ways: the provider no longer knows it (404), the gateway stopped while submitting it, or the
    provider failed it. Its requests are then settled as errors and the batch ends. If settling is interrupted (a database
    or Redis hiccup), the next pass must finish it, as it does for a job that ended, or the batch never ends and its
    reservation is held for good."""
    from datetime import UTC, datetime

    gw, fake, c = setup()
    model = "small" if how == "failed" else "big"  # small goes to the mock provider, big to Anthropic's (the fake)
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", model, "hi"))})
    bid = r.json()["id"]
    job = (await gw.store.batch_jobs(bid))[0]
    if how == "lost":
        fake.poll_status = 404
    elif how == "stuck":
        gw.store._jobs[job["id"]].update(status="submitting", created_at=datetime.now(UTC) - timedelta(hours=1))
    else:
        await gw.store.set_setting(f"mockbatch:{job['provider_id']}", {})  # the mock provider says the job failed
    settle = gw.store.settle_batch_lines
    calls = []

    async def hiccup(batch_id, settled):
        calls.append(len(settled))
        if len(calls) == 1:
            raise ConnectionError("the database went away")
        return await settle(batch_id, settled)

    monkeypatch.setattr(gw.store, "settle_batch_lines", hiccup)
    await poll(gw)
    assert (await gw.store.get_batch(bid))["status"] != "ended"
    await poll(gw)  # the next pass
    b = await gw.store.get_batch(bid)
    assert b["status"] == "ended" and b["outcome"] == "failed" and b["reserved_cost"] == 0
    assert gw.state._spend[f"{month_key()}:user:alice"] == pytest.approx(0)  # nothing stays reserved
    assert [row["status"] for row in gw.store.ledger if row["batch_id"] == bid] == [502]
    assert (await gw.store.batch_jobs(bid))[0]["status"] == "failed"


async def test_a_gemini_upload_reply_that_isnt_json_is_the_providers_refusal(setup):  # noqa: F811
    """The Files API's answer to the upload names the file the batch reads. One that isn't JSON (a proxy's error page with
    a 200) is the provider refusing the job, as any other bad reply is: the request moves on or is refused saying why. It
    isn't an exception that fails the create call while the batch it made goes on holding its reservation."""
    gw, _, c = setup(backends__gemini={"format": "openai", "api": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta",
                                       "auth": {"type": "api_key", "header": "x-goog-api-key", "value": "k"}},
                     models__small__routes=[{"backend": "gemini", "model": "gemini-x", "price": {"input": 0.1, "output": 0.4}}])

    def files_api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/v1beta/files":
            return httpx.Response(200, headers={"x-goog-upload-url": "https://upload.example/u1"}, json={})
        return httpx.Response(200, content=b"<html>upstream proxy</html>")

    gw.backends["gemini"].client = httpx.AsyncClient(transport=httpx.MockTransport(files_api))
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(("a", "small", "hi"))})
    assert r.status_code == 502 and "no provider accepted the batch" in r.text and "isn't JSON" in r.text
    assert [b["status"] for b in gw.store._batches.values()] == ["ended"]
    assert gw.state._spend[f"{month_key()}:user:alice"] == pytest.approx(0)  # nothing stays reserved
