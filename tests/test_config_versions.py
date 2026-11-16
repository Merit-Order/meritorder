"""Config saves name the version they were made to (base_version): a save made to an older version than the latest is
refused, so two admins (or an admin and the worker's auto-apply) never silently undo each other's change."""

import asyncio
import copy
import random

import pytest
import yaml
from conftest import ADMIN_KEY, TEST_CONFIG, auth
from test_benchmark import NOW, cfg_with, rows

from meritorder import benchmark as bm
from meritorder.state import MemoryState
from meritorder.store import ANY_VERSION, ConfigConflict, MemoryStore


@pytest.fixture
def durable(gateway, monkeypatch):
    """Config versions need Postgres; the in-memory store keeps them the same way."""
    monkeypatch.setenv("T_USER", "mo-test-key")
    monkeypatch.setenv("T_ADMIN", ADMIN_KEY)
    kind, gateway.store.kind = gateway.store.kind, "postgres"
    yield
    gateway.store.kind = kind


def _doc(**team):
    doc = copy.deepcopy(TEST_CONFIG)
    doc["auth"]["static_keys"] = [{"key_env": "T_USER", "user": "alice", "team": "default"},
                                  {"key_env": "T_ADMIN", "user": "root", "team": "ops", "role": "admin"}]
    doc["teams"]["default"].update(team)
    return doc


async def test_a_save_made_to_an_older_version_is_refused_and_the_newer_one_kept(client, gateway, durable):
    src = (await client.get("/v1/admin/config/source", headers=auth(ADMIN_KEY))).json()
    assert src["version"] is None  # from the file
    first = await client.put("/v1/admin/config", headers=auth(ADMIN_KEY),
                             json={"document": _doc(budget_usd_month=111), "comment": "first", "base_version": None})
    assert first.status_code == 200 and first.json()["version"] == 1
    # A second admin opened the editor at the same time (from the file) and saves after the first did.
    second = await client.put("/v1/admin/config", headers=auth(ADMIN_KEY),
                              json={"document": _doc(max_session_usd=5), "comment": "second", "base_version": None})
    assert second.status_code == 409
    assert "someone else saved the configuration (version 1)" in second.json()["detail"] and "reload" in second.json()["detail"]
    assert gateway.cfg.team("default").budget_usd_month == 111 and len(await gateway.store.config_versions()) == 1
    # Made to the latest version, it's saved.
    again = await client.put("/v1/admin/config", headers=auth(ADMIN_KEY),
                             json={"yaml": yaml.safe_dump(_doc(budget_usd_month=111, max_session_usd=5)), "base_version": 1})
    assert again.status_code == 200 and again.json()["version"] == 2
    stale = await client.put("/v1/admin/config", headers=auth(ADMIN_KEY), json={"document": _doc(), "base_version": 1})
    assert stale.status_code == 409 and "since you opened version 1" in stale.json()["detail"]
    # Left out (scripts written before base_version): saved over whatever is latest, as before.
    blind = await client.put("/v1/admin/config", headers=auth(ADMIN_KEY), json={"document": _doc(budget_usd_month=333)})
    assert blind.status_code == 200 and blind.json()["version"] == 3


async def test_restoring_from_an_older_version_than_the_latest_is_refused(client, gateway, durable):
    for n, base in ((1, None), (2, 1)):
        r = await client.put("/v1/admin/config", headers=auth(ADMIN_KEY), json={"document": _doc(budget_usd_month=n), "base_version": base})
        assert r.status_code == 200
    r = await client.post("/v1/admin/config/versions/1/restore", headers=auth(ADMIN_KEY), json={"base_version": 1})
    assert r.status_code == 409 and "version 2" in r.json()["detail"]
    r = await client.post("/v1/admin/config/versions/1/restore", headers=auth(ADMIN_KEY), json={"base_version": 2})
    assert r.status_code == 200 and r.json()["version"] == 3 and gateway.cfg.team("default").budget_usd_month == 1
    r = await client.post("/v1/admin/config/versions/2/restore", headers=auth(ADMIN_KEY))  # no body: no check, as before
    assert r.status_code == 200 and r.json()["version"] == 4


async def test_the_store_checks_the_base_version():
    store = MemoryStore()
    assert await store.save_config({"a": 1}, "x", None, base_version=None) == 1
    with pytest.raises(ConfigConflict) as e:
        await store.save_config({"a": 2}, "y", None, base_version=None)
    assert (e.value.base, e.value.current) == (None, 1)
    assert await store.save_config({"a": 2}, "y", None, base_version=1) == 2
    assert await store.save_config({"a": 3}, "z", None, base_version=ANY_VERSION) == 3


async def _ready():
    """A watcher with a recommendation (coding level 3 to 2) that the learned router backs, so auto_apply applies it."""
    store, state = MemoryStore(), MemoryState()
    cfg, raw = cfg_with(routing={"benchmark": {"auto_apply": True}})
    for r in rows("served", 3000, 0.1, 0.96, learned_level=2) + rows("down", 400, 0.1, 0.02, rng=random.Random(2)):
        await store.record_route_features({**r, "fv": "x", "text_features": [], "tags": []})
    await store.save_router_model({"members": []}, {"better": True}, "test", active=True)
    return store, state, cfg, raw


async def test_the_workers_auto_apply_builds_on_a_save_made_after_its_cycle_read_the_config():
    store, state, cfg, raw = await _ready()
    # The cycle read ``raw`` (the file); an admin then saves a change of their own.
    mine = copy.deepcopy(raw)
    mine["teams"]["default"]["budget_usd_month"] = 777
    await store.save_config(mine, "admin", "admin's change", base_version=None)
    await bm.Watcher().run(cfg, store, state, NOW.timestamp(), raw)
    saved = await store.active_config()
    assert saved["id"] == 2 and saved["body"]["routing"]["level_map"] == {"coding:3": 2}
    assert saved["body"]["teams"]["default"]["budget_usd_month"] == 777, "the auto-apply undid the admin's save"


async def test_the_workers_auto_apply_skips_a_pass_when_someone_saves_while_it_writes(monkeypatch):
    store, state, cfg, raw = await _ready()
    real = store.save_config

    async def racing(body, author, comment, base_version=ANY_VERSION):
        if author == "outcome-report":  # an admin's save lands between the watcher's read and its write
            await real({**raw, "baseline_model": raw.get("baseline_model")}, "admin", "meanwhile")
        return await real(body, author, comment, base_version)

    monkeypatch.setattr(store, "save_config", racing)
    out = await bm.Watcher().run(cfg, store, state, NOW.timestamp(), raw)
    assert out["recommendations"] and (await store.active_config())["author"] == "admin"
    assert not any(a["title"] == "Level moved by the outcome report" for a in await state.recent_alerts())


async def test_of_two_saves_made_to_the_same_version_at_once_exactly_one_wins(store):
    """On Postgres the check and the insert are one step (an advisory lock), so two gateways saving together can't both
    pass the check."""
    res = await asyncio.gather(*(store.save_config({"n": i}, f"admin{i}", None, base_version=None) for i in range(4)),
                               return_exceptions=True)
    assert sum(isinstance(r, int) for r in res) == 1 and sum(isinstance(r, ConfigConflict) for r in res) == 3
    assert len(await store.config_versions()) == 1
