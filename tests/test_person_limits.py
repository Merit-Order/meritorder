"""Each person can have their own monthly limit. It replaces their team's per-person default (higher or lower);
team and organization budgets still apply on top. Set on the People page (admin API), or on a static key."""

import copy

import pytest
import yaml
from conftest import ADMIN_KEY, LEAD_KEY, TEST_CONFIG, auth, ledger_row

from meritorder.alerts import check_budgets
from meritorder.app import build_gateway
from meritorder.config import load_config
from meritorder.proxy import _month_key
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

MSG = {"model": "big", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def durable(gateway):
    kind = gateway.store.kind
    gateway.store.kind = "postgres"
    yield gateway
    gateway.store.kind = kind


async def spent(gateway, user: str, usd: float) -> None:
    await gateway.state.seed_spend(f"{_month_key()}:user:{user}", usd)


async def ask(client, key=None):
    return await client.post("/v1/messages", headers=auth(key) if key else auth(), json=MSG)


def patch(client, uid, body, key=ADMIN_KEY):
    return client.patch(f"/v1/admin/users/{uid}", json=body, headers=auth(key))


async def test_own_limit_replaces_the_team_default_both_ways(client, durable):
    """alice signs in with a static key; the team's per-person default is $10."""
    await spent(durable, "alice", 12.0)
    assert (await ask(client)).status_code == 429  # over the $10 default
    await durable.store.upsert_user("alice", "default", "user")
    r = await patch(client, "alice", {"budget_usd_month": 50})
    assert r.status_code == 200 and r.json()["budget_usd_month"] == 50 and r.json()["default_budget_usd_month"] == 10
    assert (await ask(client)).status_code == 200  # her own $50 applies at once, not after the cache expires
    assert (await patch(client, "alice", {"budget_usd_month": 5})).status_code == 200
    r = await ask(client)
    assert r.status_code == 429 and "$5.00" in r.json()["error"]["message"]
    me = (await client.get("/v1/reports/me", headers=auth())).json()
    assert me["budget_usd"] == 5
    r = await patch(client, "alice", {"budget_usd_month": None})  # cleared: back to the team default
    assert r.json()["budget_usd_month"] is None
    assert "$10.00" in (await ask(client)).json()["error"]["message"]
    assert (await patch(client, "alice", {"role": "user"})).json()["budget_usd_month"] is None  # other edits leave it alone


async def test_a_limit_of_zero_stops_requests_and_the_team_budget_still_applies(client, durable):
    await durable.store.upsert_user("alice", "default", "user")
    await patch(client, "alice", {"budget_usd_month": 0})
    assert (await ask(client)).status_code == 429
    lee = (await client.get("/v1/reports/me", headers=auth(LEAD_KEY))).json()
    assert lee["budget_usd"] == 10  # everyone else keeps the team default
    await patch(client, "alice", {"budget_usd_month": 1000})
    await durable.state.add_spend([f"{_month_key()}:team:default"], 100.0)  # team budget is $100
    r = await ask(client)
    assert r.status_code == 429 and "team 'default'" in r.json()["error"]["message"]


async def test_people_signed_in_with_issued_keys(client, durable):
    _, secret = await durable.store.create_key("zoe", "default", "user")
    await durable.store.upsert_user("zoe", "default", "user")
    await spent(durable, "zoe", 20.0)
    assert (await ask(client, secret)).status_code == 429
    await patch(client, "zoe", {"budget_usd_month": 25})
    assert (await ask(client, secret)).status_code == 200
    users = {u["id"]: u for u in (await client.get("/v1/admin/users", headers=auth(ADMIN_KEY))).json()["users"]}
    assert users["zoe"]["budget_usd_month"] == 25 and users["zoe"]["default_budget_usd_month"] == 10


async def test_only_admins_set_limits_and_values_are_checked(client, durable):
    await durable.store.upsert_user("alice", "default", "user")
    assert (await patch(client, "alice", {"budget_usd_month": 99}, LEAD_KEY)).status_code == 403
    assert (await patch(client, "alice", {"budget_usd_month": -1})).status_code == 422
    assert (await patch(client, "alice", {"budget_usd_month": "lots"})).status_code == 422
    assert (await patch(client, "ghost", {"budget_usd_month": 5})).status_code == 404


def gateway_with_static_limit(tmp_path, limit):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["auth"]["static_keys"][0]["budget_usd_month"] = limit
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    return build_gateway(load_config(path), store=MemoryStore(), state=MemoryState())


async def test_static_key_limit_in_the_config_file(tmp_path):
    gw = gateway_with_static_limit(tmp_path, 40)
    p = await gw.authn.authenticate(auth())
    assert p.budget_usd_month == 40 and p.monthly_limit(10) == 40
    await gw.store.upsert_user("alice", "default", "user")
    await gw.store.update_user("alice", budget_usd_month=3)
    gw.authn.forget_user("alice")
    assert (await gw.authn.authenticate(auth())).budget_usd_month == 3  # the People page wins over the file
    with pytest.raises(ValueError):
        gateway_with_static_limit(tmp_path, -5)


async def test_alerts_use_each_persons_own_limit(gateway):
    for user, team in (("alice", "default"), ("bo", "default"), ("cy", "default")):
        await gateway.store.upsert_user(user, team, "user")
    await gateway.store.update_user("alice", budget_usd_month=100)  # $9 of $100: no alert, though over 80% of the $10 default
    await gateway.store.update_user("bo", budget_usd_month=4)  # $4.50 of $4: 100% alert
    await gateway.store.update_user("cy", budget_usd_month=0)  # $0 limit: refused outright, nothing to warn about
    for user, cost, team in (("alice", 9.0, "default"), ("bo", 2.0, "default"), ("bo", 2.5, "ops"), ("cy", 1.0, "default")):
        await gateway.store.record(ledger_row(user_id=user, team=team, actual_cost=cost))
    await gateway.store.rollup()
    sent = await check_budgets(gateway.cfg, gateway.store, gateway.state)
    people = [m for m in sent if "of their" in m]
    assert all("alice" not in m and "cy" not in m for m in people), people
    assert any(m.startswith("bo (default) has used 112% of their $4.00") for m in people), people  # spend across both teams
    assert await check_budgets(gateway.cfg, gateway.store, gateway.state) == []  # once per threshold per month


async def test_static_keys_still_work_when_the_database_is_down(tmp_path):
    gw = gateway_with_static_limit(tmp_path, 40)

    async def down(user_id):
        raise ConnectionError("database unavailable")

    gw.store.get_user = down
    p = await gw.authn.authenticate(auth())
    assert p.user_id == "alice" and p.budget_usd_month == 40
