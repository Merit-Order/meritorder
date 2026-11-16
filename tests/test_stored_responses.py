"""The Responses API's stored mode (decision 51) through the official OpenAI SDK: chaining with previous_response_id and
conversations, reading responses back, deleting, background responses, scoping to the person and team, retention, removing
people, and keys that rotate. On the mock backends; test_stored_providers.py drives fakes of real providers' APIs."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import openai
import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth, ledger

from meritorder.app import create_app
from meritorder.config import StoredResponsesConfig, build_config
from meritorder.sealing import Keyring, SealError, key_id
from meritorder.stored.service import StoredResponses, normalize_input


@pytest.fixture(autouse=True)
def responses_key(monkeypatch):
    """What's stored is sealed: a gateway on Postgres keeps it only with a key."""
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "test-responses-key")
    monkeypatch.delenv("MERITORDER_RESPONSES_OLD_KEYS", raising=False)


def stored_mode(gateway, team: str = "default", **kw) -> None:
    gateway.cfg.team(team).stored_responses = StoredResponsesConfig(enabled=True, retention_days=30, **kw)


@pytest.fixture
async def http(gateway):
    app = create_app(gateway=gateway)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def oai(http: httpx.AsyncClient, key: str = TEST_KEY) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=key, base_url="http://test/v1", http_client=http, max_retries=0)


def spy(gateway, backend: str) -> list[dict]:
    """Every body the backend is sent."""
    sent: list[dict] = []
    b = gateway.backends[backend]
    original = b.send

    async def send(body, stream, headers, **kw):
        sent.append(body)
        return await original(body, stream, headers, **kw)

    b.send = send
    return sent


def texts(messages: list[dict]) -> list[tuple[str, str]]:
    """(role, text) of each message, whichever format they're in."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append((m["role"], c or ""))
    return out


async def test_off_by_default(http, gateway):
    client = oai(http)
    r = await client.responses.create(model="small", input="hello")
    assert r.model_extra["store"] is False
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve(r.id)
    with pytest.raises(openai.BadRequestError, match="stored responses, which are off for team 'default'"):
        await client.responses.create(model="small", input="again", previous_response_id=r.id)
    with pytest.raises(openai.BadRequestError, match="stored responses"):
        await client.conversations.create()
    with pytest.raises(openai.BadRequestError, match="background"):
        await client.responses.create(model="small", input="later", background=True)
    with pytest.raises(openai.BadRequestError, match="item_reference"):
        await client.responses.create(model="small", input=[{"type": "item_reference", "id": "msg_1"}])
    assert await gateway.stored.sizes() == []


async def test_chain_rebuilds_the_whole_input_across_providers(http, gateway):
    """Turn 1 on an Anthropic-format backend, turn 2 on an OpenAI-format one, turn 3 back: each gets the whole conversation,
    built from what the gateway stored, and the instructions of its own request only."""
    stored_mode(gateway)
    to_anthropic, to_openai = spy(gateway, "mock-anthropic"), spy(gateway, "mock-openai")
    client = oai(http)
    r1 = await client.responses.create(model="big", input="My name is Ada.", instructions="Be brief.")
    assert r1.model_extra["store"] is True and r1.instructions == "Be brief." and r1.status == "completed"
    r2 = await client.responses.create(model="small", input="What is my name?", previous_response_id=r1.id)
    assert r2.previous_response_id == r1.id and r2.instructions is None
    assert texts(to_openai[-1]["messages"]) == [("user", "My name is Ada."), ("assistant", r1.output_text), ("user", "What is my name?")]
    r3 = await client.responses.create(model="big", input=[{"role": "user", "content": "Thanks."}], previous_response_id=r2.id,
                                       instructions="Be kind.")
    sent = to_anthropic[-1]
    assert sent["system"][0]["text"] == "Be kind." if isinstance(sent["system"], list) else sent["system"] == "Be kind."
    assert [t for _, t in texts(sent["messages"])] == ["My name is Ada.", r1.output_text, "What is my name?", r2.output_text, "Thanks."]

    # Read back as OpenAI returns them: the response, and the items it was made from (newest first by default).
    got = await client.responses.retrieve(r3.id)
    assert got.id == r3.id and got.output_text == r3.output_text and got.previous_response_id == r2.id
    items = await client.responses.input_items.list(r3.id)
    assert [i.type for i in items.data] == ["message"] * 5 and items.data[0].content[0].text == "Thanks."
    asc = [i async for i in client.responses.input_items.list(r3.id, order="asc", limit=2)]  # pages through all of them
    assert [i.role for i in asc] == ["user", "assistant", "user", "assistant", "user"] and len({i.id for i in asc}) == 5

    # Each turn is a ledger row like any other, all in the chain's one session.
    rows = await ledger(gateway)
    assert [r["served_model"] for r in rows] == ["big", "small", "big"] and len({r["session_id"] for r in rows}) == 1

    # Deleting is real: gone for reading, and for chaining anything that builds on it.
    assert await client.responses.delete(r2.id) is None
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve(r2.id)
    with pytest.raises(openai.NotFoundError, match="earlier response in this chain"):
        await client.responses.create(model="big", input="more", previous_response_id=r3.id)
    raw = await http.delete(f"/v1/responses/{r1.id}", headers=auth())
    assert raw.json() == {"id": r1.id, "object": "response.deleted", "deleted": True}


async def test_streamed_turns_are_kept_before_the_stream_ends(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    seen = []
    async with client.responses.stream(model="small", input="stream please") as s:
        async for ev in s:
            seen.append(ev.type)
            if ev.type == "response.completed":  # already kept: a client may chain the instant it sees this
                assert (await gateway.store.get_stored_response(ev.response.id))["status"] == "completed"
        final = await s.get_final_response()
    assert seen[0] == "response.created" and seen[-1] == "response.completed"
    r2 = await client.responses.create(model="small", input="and then?", previous_response_id=final.id, stream=False)
    assert r2.previous_response_id == final.id
    assert (await client.responses.retrieve(final.id)).output_text == final.output_text


async def test_store_false_is_stateless_even_in_stored_mode(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    r1 = await client.responses.create(model="small", input="keep this")
    r2 = await client.responses.create(model="small", input="not this", previous_response_id=r1.id, store=False)
    assert r2.model_extra["store"] is False
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve(r2.id)
    with pytest.raises(openai.BadRequestError, match="together"):
        await client.responses.create(model="small", input="x", previous_response_id=r1.id, conversation="conv_x")


async def test_conversations(http, gateway):
    stored_mode(gateway)
    to_openai = spy(gateway, "mock-openai")
    client = oai(http)
    conv = await client.conversations.create(metadata={"topic": "names"}, items=[{"role": "user", "content": "I'm Grace."}])
    assert conv.id.startswith("conv_") and conv.metadata == {"topic": "names"}
    r1 = await client.responses.create(model="small", input="Who am I?", conversation=conv.id)
    assert r1.conversation.id == conv.id
    assert texts(to_openai[-1]["messages"]) == [("user", "I'm Grace."), ("user", "Who am I?")]
    r2 = await client.responses.create(model="small", input="Say it again.", conversation={"id": conv.id}, store=False)
    assert [t for _, t in texts(to_openai[-1]["messages"])] == ["I'm Grace.", "Who am I?", r1.output_text, "Say it again."]

    items = [i async for i in client.conversations.items.list(conv.id, order="asc")]
    assert [getattr(i, "role", None) for i in items] == ["user", "user", "assistant", "user", "assistant"]
    assert items[-1].content[0].text == r2.output_text  # a store: false turn still adds to its conversation
    # input_items: the conversation as it stood, then the response's own input
    assert [i.content[0].text for i in (await client.responses.input_items.list(r1.id, order="asc")).data] == ["I'm Grace.", "Who am I?"]

    added = await client.conversations.items.create(conv.id, items=[{"type": "message", "role": "user", "content": "Note this."}])
    assert added.data[0].id.startswith("msg_")
    one = await client.conversations.items.retrieve(added.data[0].id, conversation_id=conv.id)
    assert one.content[0].text == "Note this."
    after = await client.conversations.items.delete(added.data[0].id, conversation_id=conv.id)
    assert after.id == conv.id and after.object == "conversation"
    assert (await client.conversations.update(conv.id, metadata={"topic": "done"})).metadata == {"topic": "done"}
    assert (await client.conversations.retrieve(conv.id)).metadata == {"topic": "done"}

    gone = await client.conversations.delete(conv.id)
    assert gone.deleted and gone.object == "conversation.deleted"
    with pytest.raises(openai.NotFoundError):
        await client.conversations.retrieve(conv.id)
    with pytest.raises(openai.NotFoundError):  # the items went with it
        await client.conversations.items.list(conv.id)
    assert (await client.responses.retrieve(r1.id)).id == r1.id  # its responses stay, as OpenAI keeps them


async def test_item_references_resolve_to_stored_items(http, gateway):
    """The Vercel AI SDK sends earlier turns' items as item_reference when store is on."""
    stored_mode(gateway)
    to_openai = spy(gateway, "mock-openai")
    client = oai(http)
    r1 = await client.responses.create(model="small", input="first")
    msg = r1.output[0].id
    await client.responses.create(model="small", input=[{"role": "user", "content": "first"}, {"type": "item_reference", "id": msg},
                                                        {"role": "user", "content": "second"}])
    assert [t for _, t in texts(to_openai[-1]["messages"])] == ["first", r1.output_text, "second"]
    with pytest.raises(openai.NotFoundError, match="msg_nope"):
        await client.responses.create(model="small", input=[{"type": "item_reference", "id": "msg_nope"}])
    other = oai(http, LEAD_KEY)  # someone else can't reach into alice's items
    with pytest.raises(openai.NotFoundError):
        await other.responses.create(model="small", input=[{"type": "item_reference", "id": msg}])


async def test_only_the_owner_reads_or_chains_unless_a_role_is_given(http, gateway):
    stored_mode(gateway)
    alice, lead, admin = oai(http), oai(http, LEAD_KEY), oai(http, ADMIN_KEY)
    r = await alice.responses.create(model="small", input="private")
    conv = await alice.conversations.create()
    for who in (lead, admin):  # admins included
        with pytest.raises(openai.NotFoundError):
            await who.responses.retrieve(r.id)
        with pytest.raises(openai.NotFoundError):
            await who.conversations.retrieve(conv.id)
    with pytest.raises(openai.NotFoundError):
        await lead.responses.create(model="small", input="peek", previous_response_id=r.id)
    with pytest.raises(openai.NotFoundError):
        await lead.responses.delete(r.id)

    stored_mode(gateway, readers=["team_lead"])
    assert (await lead.responses.retrieve(r.id)).id == r.id
    r2 = await lead.responses.create(model="small", input="continue", previous_response_id=r.id)
    assert (await gateway.store.get_stored_response(r2.id))["user_id"] == "lee"  # the new turn is the reader's own
    with pytest.raises(openai.NotFoundError):  # reading isn't owning
        await lead.responses.delete(r.id)
    with pytest.raises(openai.NotFoundError):  # the admin is on another team, and admin isn't listed
        await admin.responses.retrieve(r.id)
    stored_mode(gateway, readers=["admin"])
    assert (await admin.responses.retrieve(r.id)).id == r.id
    assert (await admin.conversations.retrieve(conv.id)).id == conv.id


async def test_a_person_who_moves_team_leaves_their_responses_behind(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    r = await client.responses.create(model="small", input="on default")
    row = await gateway.store.get_stored_response(r.id)
    assert row["team"] == "default"
    p = await gateway.authn.authenticate(auth())
    p.team = "research"  # what a move on the People page does to the key's principal
    assert not gateway.stored.may_read(p, row)


async def test_background_responses_are_polled_and_cancelled(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    r = await client.responses.create(model="small", input="take your time", background=True)
    assert r.status == "queued" and r.background is True
    for _ in range(100):
        got = await client.responses.retrieve(r.id)
        if got.status not in ("queued", "in_progress"):
            break
        await asyncio.sleep(0.01)
    assert got.status == "completed" and got.output_text and got.background is True
    assert (await client.responses.cancel(r.id)).status == "completed"  # nothing to stop any more
    plain = await client.responses.create(model="small", input="now")
    with pytest.raises(openai.BadRequestError, match="Only background responses"):
        await client.responses.cancel(plain.id)
    with pytest.raises(openai.BadRequestError, match="streamed"):
        await client.responses.create(model="small", input="x", background=True, stream=True)

    # Cancelled while it runs: it ends cancelled, and the answer that comes later isn't kept.
    gate = asyncio.Event()
    original = gateway.backends["mock-openai"].send

    async def slow(body, stream, headers, **kw):
        await gate.wait()
        return await original(body, stream, headers, **kw)

    gateway.backends["mock-openai"].send = slow
    r2 = await client.responses.create(model="small", input="slow one", background=True)
    await asyncio.sleep(0.05)
    assert (await client.responses.cancel(r2.id)).status == "cancelled"
    gate.set()
    await gateway.drain()
    assert (await client.responses.retrieve(r2.id)).status == "cancelled"


async def test_background_with_store_false_is_kept_only_to_be_polled(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    r = await client.responses.create(model="small", input="quick", background=True, store=False)
    await gateway.drain()
    got = await client.responses.retrieve(r.id)
    assert got.status == "completed"
    row = await gateway.store.get_stored_response(r.id)
    assert row["expires_at"] is not None and row["expires_at"] < datetime.now(UTC) + timedelta(minutes=11)
    with pytest.raises(openai.NotFoundError):  # not stored: nothing to chain from
        await client.responses.create(model="small", input="then", previous_response_id=r.id)
    await gateway.store.update_stored_response(r.id, expires_at=datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve(r.id)
    assert (await gateway.stored.purge())[0] == 1


async def test_background_failures_are_reported_when_polled(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    gateway.cfg.team("default").budget_usd_month = 0  # refused before any provider is called
    r = await client.responses.create(model="small", input="no money", background=True)
    await gateway.drain()
    got = await client.responses.retrieve(r.id)
    assert got.status == "failed" and "budget" in got.error.message

    # A gateway that stops mid-response leaves it in progress; the worker marks it failed after the timeout.
    gateway.cfg.team("default").budget_usd_month = None
    gate = asyncio.Event()
    original = gateway.backends["mock-openai"].send

    async def stuck(body, stream, headers, **kw):
        await gate.wait()
        return await original(body, stream, headers, **kw)

    gateway.backends["mock-openai"].send = stuck
    r = await client.responses.create(model="small", input="this gateway will stop", background=True)
    await asyncio.sleep(0.05)
    assert [x["id"] for x in await gateway.store.stale_background(datetime.now(UTC) + timedelta(hours=1))] == [r.id]
    got = await client.responses.retrieve(r.id)
    assert got.status == "failed" and "stopped" in got.error.message
    gate.set()
    await gateway.drain()
    assert (await client.responses.retrieve(r.id)).status == "failed"  # a late answer doesn't undo what clients were told


async def test_retention_counts_from_the_chains_last_turn(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    r1 = await client.responses.create(model="small", input="one")
    old = datetime.now(UTC) - timedelta(days=20)
    await gateway.store.update_stored_response(r1.id, touched_at=old)
    r2 = await client.responses.create(model="small", input="two", previous_response_id=r1.id)
    assert (await gateway.store.get_stored_response(r1.id))["touched_at"] > old  # continuing a chain keeps all of it
    for rid in (r1.id, r2.id):
        await gateway.store.update_stored_response(rid, touched_at=datetime.now(UTC) - timedelta(days=31))
    with pytest.raises(openai.NotFoundError):  # past retention: gone for reading before the purge runs
        await client.responses.retrieve(r2.id)
    conv = await client.conversations.create()
    await gateway.store.update_conversation(conv.id, touched_at=datetime.now(UTC) - timedelta(days=31))
    assert await gateway.stored.purge() == (2, 1)
    assert await gateway.store.get_stored_response(r1.id) is None


async def test_turning_stored_mode_off_deletes_what_the_team_kept(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    r = await client.responses.create(model="small", input="soon gone")
    gateway.cfg.team("default").stored_responses = StoredResponsesConfig()
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve(r.id)
    assert await gateway.stored.purge() == (1, 0)


async def test_removing_a_person_deletes_what_they_stored(http, gateway, monkeypatch):
    stored_mode(gateway)
    client = oai(http)
    r = await client.responses.create(model="small", input="mine")
    await client.conversations.create(items=[{"role": "user", "content": "also mine"}])
    lead_r = await oai(http, LEAD_KEY).responses.create(model="small", input="lee's")
    await gateway.store.upsert_user("alice", "default", "user")
    if gateway.store.kind == "postgres":  # the People page needs Postgres
        resp = await http.patch("/v1/admin/users/alice", headers=auth(ADMIN_KEY), json={"disabled": True})
        assert resp.status_code == 200 and resp.json()["stored_responses_deleted"] == 1
    else:
        assert await gateway.stored.forget_user("alice") == 1
    assert await gateway.store.get_stored_response(r.id) is None
    assert not [s for s in await gateway.stored.sizes() if s["conversations"]]
    assert await gateway.store.get_stored_response(lead_r.id) is not None  # nobody else's

    # Through SCIM: deactivating or deleting someone in the IdP.
    monkeypatch.setenv("MERITORDER_SCIM_TOKEN", "scim-token")
    scim = {"authorization": "Bearer scim-token"}
    await gateway.store.upsert_user("lee", "default", "team_lead")
    r = await http.patch("/scim/v2/Users/lee", headers=scim, json={"Operations": [{"op": "replace", "path": "active", "value": False}]})
    assert r.status_code == 200 and await gateway.store.get_stored_response(lead_r.id) is None
    await gateway.store.upsert_user("root", "ops", "admin")
    stored_mode(gateway, team="ops")
    root_r = await oai(http, ADMIN_KEY).responses.create(model="small", input="root's")
    assert (await http.delete("/scim/v2/Users/root", headers=scim)).status_code == 204
    assert await gateway.store.get_stored_response(root_r.id) is None


async def test_content_is_sealed_and_the_key_rotates(http, gateway, monkeypatch):
    stored_mode(gateway)
    client = oai(http)
    r = await client.responses.create(model="small", input="the secret word is aubergine")
    conv = await client.conversations.create(metadata={"k": "v"}, items=[{"role": "user", "content": "aubergine again"}])
    row = await gateway.store.get_stored_response(r.id)
    assert b"aubergine" not in bytes(row["content"]) and row["seal_key"] == key_id(__import__("hashlib").sha256(b"test-responses-key").digest())
    assert row["bytes"] == len(row["content"])

    # Rotate: the new key seals, the old one still opens, and the worker seals everything again with the new one.
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "the-new-key")
    monkeypatch.setenv("MERITORDER_RESPONSES_OLD_KEYS", "test-responses-key")
    assert (await client.responses.retrieve(r.id)).output_text == r.output_text
    before = await gateway.stored.sizes()
    assert before[0]["old_key"] == 3  # the response, the conversation and its item
    assert (await gateway.stored.maintain())["resealed"] == 3
    assert (await gateway.stored.sizes())[0]["old_key"] == 0
    monkeypatch.delenv("MERITORDER_RESPONSES_OLD_KEYS")
    assert (await client.responses.retrieve(r.id)).output_text == r.output_text
    assert (await client.conversations.items.list(conv.id)).data[0].content[0].text == "aubergine again"

    # A key that's gone can't open what it sealed, and a gateway on Postgres without any key keeps nothing.
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "unrelated")
    with pytest.raises(openai.InternalServerError, match="isn't configured"):
        await client.responses.retrieve(r.id)
    if gateway.store.kind != "memory":
        monkeypatch.delenv("MERITORDER_RESPONSES_KEY")
        with pytest.raises(openai.APIStatusError, match="MERITORDER_RESPONSES_KEY") as e:
            await client.responses.create(model="small", input="no key")
        assert e.value.status_code == 503


def test_keyring():
    ring = Keyring("one", ["zero"])
    blob = ring.seal(b"hello", b"here")
    assert ring.open(blob, b"here") == b"hello" and ring.sealed_with(blob) == ring.current
    with pytest.raises(SealError, match="belongs somewhere else"):
        ring.open(blob, b"there")
    with pytest.raises(SealError, match="isn't configured"):
        Keyring("two").open(blob, b"here")
    assert Keyring("zero", ["one"]).open(blob, b"here") == b"hello"
    with pytest.raises(SealError, match="not a sealed value"):
        ring.open(b"plain", b"here")
    assert not Keyring(None).usable and Keyring(None, memory=True).usable
    with pytest.raises(SealError):
        Keyring(None).seal(b"x", b"y")


def test_input_is_normalized_as_openai_stores_it():
    items = normalize_input("hi")
    assert items[0]["type"] == "message" and items[0]["content"] == [{"type": "input_text", "text": "hi"}] and items[0]["id"].startswith("msg_")
    out = normalize_input([{"role": "assistant", "content": "ok"}, {"type": "function_call_output", "call_id": "c", "output": "1"},
                           {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}], "id": "msg_keep"}])
    assert out[0]["content"][0]["type"] == "output_text" and out[1]["id"].startswith("fco_") and out[2]["id"] == "msg_keep"


def test_config_needs_a_retention(tmp_path):
    with pytest.raises(ValueError, match="retention_days"):
        StoredResponsesConfig(enabled=True)
    assert StoredResponsesConfig(enabled=False).retention_days is None
    assert StoredResponsesConfig(enabled=True, retention_days="30").retention_days == 30
    with pytest.raises(ValueError, match="api: responses"):
        build_config({"backends": {"x": {"format": "openai", "base_url": "https://x.example/v1", "responses_store": True}},
                      "models": {}}, tmp_path, quiet=True)


async def test_console_shows_sizes_never_content(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    await client.responses.create(model="small", input="count me, don't read me")
    await client.conversations.create(items=[{"role": "user", "content": "me too"}])
    r = await http.get("/v1/console/teams", headers=auth(ADMIN_KEY))
    team = next(t for t in r.json()["teams"] if t["name"] == "default")
    s = team["stored_responses"]
    assert s["enabled"] and s["retention_days"] == 30 and s["responses"] == 1 and s["conversations"] == 1 and s["items"] == 1
    assert s["bytes"] > 0 and s["people"] == 1 and s["oldest"]
    assert "count me" not in r.text and "me too" not in r.text
    key = r.json()["stored_key"]
    assert key["configured"] and key["env"] == "MERITORDER_RESPONSES_KEY" and len(key["id"]) == 16 and "test-responses-key" not in r.text


async def test_compression_decisions_hold_across_chained_turns(http, gateway):
    """A chained turn's history is rebuilt from the client's own input (stored before compression), so the rules cut the
    same tool result to the same bytes on every turn, as for a client that resends it."""
    from meritorder.compression import CompressionConfig

    stored_mode(gateway)
    gateway.cfg.team("default").compression = CompressionConfig(enabled=True, min_chars=100, max_chars=400)
    sent = spy(gateway, "mock-openai")
    client = oai(http)
    tool = [{"type": "function", "name": "run", "parameters": {"type": "object"}}]
    big = "\n".join(f"line {i % 3}" for i in range(400))
    r1 = await client.responses.create(model="small", tools=tool, input=[
        {"role": "user", "content": "run it"}, {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": big}])
    r2 = await client.responses.create(model="small", tools=tool, input="and now?", previous_response_id=r1.id)
    first, second = ([m for m in s["messages"] if m["role"] == "tool"][0]["content"] for s in sent)
    assert first == second and len(first) < len(big)
    row = await gateway.store.get_stored_response(r1.id)
    assert gateway.stored.open(row["content"], gateway.stored.response_aad(row))["input"][2]["output"] == big  # as the client sent it
    assert r2.previous_response_id == r1.id


async def test_what_the_api_refuses(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    for bad in ({"model": "small", "input": [1]}, {"model": "small", "input": "x", "previous_response_id": "resp_nope"}):
        r = await http.post("/v1/responses", headers=auth(), json=bad)
        assert r.status_code in (400, 404), r.text
    assert "Previous response with id 'resp_nope' not found." in r.json()["error"]["message"]
    r1 = await client.responses.create(model="small", input="one")
    assert (await http.get(f"/v1/responses/{r1.id}?stream=true", headers=auth())).status_code == 400
    assert (await http.get(f"/v1/responses/{r1.id}", headers={"authorization": "Bearer nope"})).status_code == 401
    with pytest.raises(openai.BadRequestError, match="metadata"):
        await client.conversations.create(metadata={"k": "v" * 600})
    conv = await client.conversations.create(items=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                                                    {"role": "user", "content": "c"}])
    for body in (b"not json", b"[]"):
        r = await http.post(f"/v1/conversations/{conv.id}", headers={**auth(), "content-type": "application/json"}, content=body)
        assert r.status_code == 400
    with pytest.raises(openai.BadRequestError, match="non-empty"):
        await client.conversations.items.create(conv.id, items=[])
    first = (await client.conversations.items.list(conv.id, order="asc")).data[0]
    with pytest.raises(openai.BadRequestError, match="already in this conversation"):
        await client.conversations.items.create(conv.id, items=[{"type": "message", "role": "user", "content": "again", "id": first.id}])
    page = await client.conversations.items.list(conv.id, order="asc", after=first.id, limit=1)
    assert [i.content[0].text for i in page.data] == ["b"] and page.has_more
    assert (await client.conversations.items.list(conv.id, after="msg_unknown")).data == []
    with pytest.raises(openai.NotFoundError):
        await client.conversations.items.retrieve("msg_unknown", conversation_id=conv.id)
    with pytest.raises(openai.NotFoundError):
        await client.conversations.items.delete("msg_unknown", conversation_id=conv.id)

    # A background response still running can't be chained from yet.
    gate = asyncio.Event()
    original = gateway.backends["mock-openai"].send

    async def slow(body, stream, headers, **kw):
        await gate.wait()
        return await original(body, stream, headers, **kw)

    gateway.backends["mock-openai"].send = slow
    bg = await client.responses.create(model="small", input="wait", background=True)
    with pytest.raises(openai.BadRequestError, match="chain from a completed response"):
        await client.responses.create(model="big", input="next", previous_response_id=bg.id)
    assert await client.responses.delete(bg.id) is None  # deleting a running one stops it
    gate.set()
    await gateway.drain()
    assert await gateway.store.get_stored_response(bg.id) is None


async def test_input_items_skip_what_was_deleted(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    r1 = await client.responses.create(model="small", input="one")
    r2 = await client.responses.create(model="small", input="two", previous_response_id=r1.id)
    await client.responses.delete(r1.id)
    assert [i.content[0].text for i in (await client.responses.input_items.list(r2.id)).data] == ["two"]


async def test_references_reach_conversation_items_and_skip_expired_responses(http, gateway):
    stored_mode(gateway)
    to_openai = spy(gateway, "mock-openai")
    client = oai(http)
    conv = await client.conversations.create(items=[{"role": "user", "content": "kept in a conversation"}])
    item = (await client.conversations.items.list(conv.id)).data[0].id
    await client.responses.create(model="small", input=[{"type": "item_reference", "id": item}, {"role": "user", "content": "go"}])
    assert texts(to_openai[-1]["messages"])[0] == ("user", "kept in a conversation")
    old = await client.responses.create(model="small", input="old")
    await gateway.store.update_stored_response(old.id, touched_at=datetime.now(UTC) - timedelta(days=40))
    with pytest.raises(openai.NotFoundError):
        await client.responses.create(model="small", input=[{"type": "item_reference", "id": old.output[0].id}])


async def test_keeping_a_response_can_fail_without_losing_the_answer(http, gateway, monkeypatch):
    stored_mode(gateway)
    client = oai(http)

    async def broken(row):
        raise RuntimeError("disk full")

    monkeypatch.setattr(gateway.store, "save_stored_response", broken)
    r = await client.responses.create(model="small", input="answer me anyway")
    assert r.output_text and r.model_extra["store"] is False
    assert gateway.metrics.counters["meritorder_stored_responses_total"][(("event", "failed"),)] == 1


async def test_a_background_response_that_breaks_is_failed(http, gateway, monkeypatch):
    stored_mode(gateway)
    client = oai(http)

    async def boom(*a, **kw):
        raise RuntimeError("something broke")

    monkeypatch.setattr(gateway, "_handle", boom)
    r = await client.responses.create(model="small", input="x", background=True)
    await gateway.drain()
    got = await client.responses.retrieve(r.id)
    assert got.status == "failed" and "something broke" in got.error.message


async def test_provider_copies_that_cant_be_deleted_are_left_to_the_provider(gateway):
    class Unreachable:
        async def delete_response(self, rid):
            raise RuntimeError("no route to host")

    rows = [{"id": "resp_a", "backend": "oai", "upstream_id": "resp_up"}, {"id": "resp_b", "backend": "gone", "upstream_id": "x"},
            {"id": "resp_c", "backend": "oai", "upstream_id": None}]
    assert await gateway.stored.delete_provider_copies(rows, {"oai": Unreachable()}) == 0


async def test_forgetting_someone_stops_their_running_background_responses(http, gateway):
    stored_mode(gateway)
    client = oai(http)
    gate = asyncio.Event()
    original = gateway.backends["mock-openai"].send

    async def slow(body, stream, headers, **kw):
        await gate.wait()
        return await original(body, stream, headers, **kw)

    gateway.backends["mock-openai"].send = slow
    r = await client.responses.create(model="small", input="slow", background=True)
    await asyncio.sleep(0.05)
    assert await gateway.stored.forget_user("alice") == 1
    gate.set()
    await gateway.drain()
    assert await gateway.store.get_stored_response(r.id) is None


async def test_values_no_key_opens_are_left_for_the_operator(http, gateway, monkeypatch):
    stored_mode(gateway)
    client = oai(http)
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "lost-key")
    await client.responses.create(model="small", input="sealed with a key that will be lost")
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "new-key")
    monkeypatch.setenv("MERITORDER_RESPONSES_OLD_KEYS", "test-responses-key")
    assert await gateway.stored.reseal() == 0  # it can't be opened, so it's left as it is (and logged)


def test_a_turn_that_merges_with_its_history_is_sent_whole(gateway):
    """A new input that starts with a tool call would merge into the history's last assistant message: the provider's
    chain can't be used, since its messages wouldn't start where the provider's stop."""
    from meritorder.stored.service import StoredTurn

    p = type("P", (), {"user_id": "alice", "team": "default"})()
    call = {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}", "id": "fc_1"}
    turn = StoredTurn(p, gateway.cfg.team("default"), "resp_x", True, False, normalize_input([{**call, "call_id": "c2", "id": "fc_2"}]),
                      history=normalize_input([{"role": "user", "content": "go"}, call]),
                      provider_previous={"backend": "oai", "model": "m", "id": "resp_up"})
    gateway.stored.rebuilt(turn, {"model": "small", "input": []})
    assert turn.provider_previous is None and turn.held == 0


def test_a_worker_purges_without_a_gateway(cfg):
    class Ctx:  # what the worker hands it: configuration, store, backends
        def __init__(self):
            from meritorder.store import MemoryStore

            self.cfg, self.store, self.backends = cfg, MemoryStore(), {}

    s = StoredResponses(Ctx())
    assert asyncio.run(s.maintain({})) == {"deleted_responses": 0, "deleted_conversations": 0, "failed_background": 0, "resealed": 0}


def test_built_config_reads_the_team_section(tmp_path):
    from conftest import TEST_CONFIG
    import copy

    raw = copy.deepcopy(TEST_CONFIG)
    raw["teams"]["default"]["stored_responses"] = {"enabled": True, "retention_days": 14, "readers": ["team_lead"]}
    raw["response_store"] = {"key_file": str(tmp_path / "key"), "old_keys_env": "OLD"}
    (tmp_path / "key").write_text("from-a-file\n")
    cfg = build_config(raw, tmp_path, quiet=True)
    assert cfg.team("default").stored_responses.readers == ["team_lead"]
    assert cfg.response_store.key() == "from-a-file" and cfg.response_store.old_keys() == []
