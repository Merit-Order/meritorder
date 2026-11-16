"""Quality scores from published benchmarks: Epoch AI's ability scale (or the Artificial Analysis index), each model's
chance on a reference task, the best-anchored or worst-to-best scale, overrides, and the worker keeping them live."""

import copy
import io
import json
import math
import zipfile

import httpx
import pytest
from conftest import ADMIN_KEY, TEST_CONFIG, auth

from meritorder import catalog, cli, live_prices, quality_data as qd
from meritorder.config import build_config
from meritorder.state import MemoryState
from meritorder.store import MemoryStore


@pytest.fixture
def durable(gateway):
    """The admin refresh needs a durable store; the in-memory one behaves the same here."""
    kind = gateway.store.kind
    gateway.store.kind = "postgres"
    yield gateway
    gateway.store.kind = kind


def epoch_zip(**extra) -> bytes:
    files = {
        "epoch_capabilities_index/eci_scores.csv": "Model,Display name,eci,eci_ci_low,eci_ci_high\nx,Claude Top 1,170,167,173\nx,Claude Mid 1,165,162,168\n"
                                                   "x,Cheap Model,150,148,152\n",
        "epoch_capabilities_index/edi_scores.csv": "benchmark_name,is_anchor,benchmark_release_date,edi,estimated_slope_scaled\nFrontierSWE,False,2026-09-02,165,0.16\n"
                                                   "FrontierCode,False,2026-06-08,164,0.086\nTerminal Bench,False,2025-05-19,149,0.164\n",
        "frontierswe_external.csv": "Model version,Harness,Score\nclaude-top-1_max,h,0.70\nclaude-top-1_high,h,0.66\nclaude-mid-1_max,h,0.62\n",
        "frontiercode_external.csv": "Model version,Main score,Harness\nclaude-mid-1_medium,0.40,h\nnot a number,x,h\n",
        # an older coding benchmark: "Accuracy mean", one row per agent
        "terminalbench_external.csv": "Model version,Agent,Accuracy mean\nopen-coder-480b_unknown,a,0.27\nopen-coder-480b_unknown,b,0.31\n"
                                      "cheap-model_unknown,a,0.90\n",
        **extra,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def config(**quality):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["models"] = {
        **raw["models"],
        "top": {"quality": {"coding": 3, "general": 3}, "routes": [{"backend": "mock-anthropic", "model": "claude-top-1"}]},
        "mid": {"aliases": ["Claude Mid 1"], "routes": [{"backend": "mock-anthropic", "model": "claude-mid-1-20260101"}]},
        "cheap": {"benchmark_name": "Cheap Model", "quality": {"coding": 2}, "quality_override": {"general": 4.5}, "routes": [{"backend": "mock-openai", "model": "c-1"}]},
        "guess": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "vendor", "model": "unknown-1"}]},
        "trusted": {"quality": {"coding": 4}, "quality_override": {"coding": 3.5}, "routes": [{"backend": "vendor", "model": "unknown-2"}]},
        "coder": {"hf_id": "Org/Open-Coder-480B", "routes": [{"backend": "vendor", "model": "org/open-coder"}]},  # Epoch knows its repo name
    }
    raw["backends"] = {**raw["backends"], "vendor": {"kind": "api", "format": "openai", "base_url": "https://vendor.example/v1"}}
    raw["quality"] = {"source": "epoch", "scale": "best", **quality}
    return build_config(raw, prices_data={}, quiet=True)


def test_epoch_data_becomes_chances_on_the_reference_task():
    data = qd.parse_epoch(epoch_zip())
    assert data["eci"]["claude-top-1"] == 170 and data["difficulty"]["FrontierSWE"] == (165.0, 0.16)
    assert sorted(data["results"]["FrontierSWE"]["claude-top-1"]) == [("high", 0.66), ("max", 0.70)]
    assert data["results"]["FrontierCode"] == {"claude-mid-1": [("medium", 0.40)]}  # "Main score"; junk rows skipped

    cfg = config()
    m = qd.epoch_measures(cfg, data, "high")
    top = m["top"]["coding"]
    assert top["evidence"]["FrontierSWE"]["variant"] == "high"  # the effort it's routed with, when published
    assert top["ability"] == pytest.approx(165 + math.log(0.66 / 0.34) / 0.16, abs=0.01) and top["basis"] == "results on FrontierSWE"
    assert m["top"]["general"]["basis"] == "Epoch Capabilities Index" and m["top"]["general"]["chance"] == pytest.approx(1 / (1 + math.exp(-0.16 * 5)), abs=1e-3)
    mid = m["mid"]["coding"]  # two benchmarks of different difficulty: averaged on the ability scale
    assert set(mid["evidence"]) == {"FrontierSWE", "FrontierCode"}
    assert mid["ability"] == pytest.approx((mid["evidence"]["FrontierSWE"]["ability"] + mid["evidence"]["FrontierCode"]["ability"]) / 2, abs=0.01)
    assert "cheap" in m and "small" not in m  # matched by benchmark_name; unknown models aren't guessed
    assert set(qd.model_keys(cfg.model("mid"))) >= {"claude-mid-1-20260101", "claude-mid-1"}
    assert qd.model_keys(cfg.model("coder")) == ["open-coder", "open-coder-480b"]  # route id first, then the Hugging Face name

    coder = m["coder"]["coding"]  # no coding evidence and no index: the older coding benchmark, matched by its hf_id
    assert coder["basis"] == "results on Terminal Bench (no index or newer results)" and coder["evidence"]["Terminal Bench"]["score"] == 0.31
    assert coder["ability"] == pytest.approx(149 + math.log(0.31 / 0.69) / 0.164, abs=0.01) and "general" not in m["coder"]
    assert m["cheap"]["coding"]["basis"] == "Epoch Capabilities Index"  # the index comes before older benchmarks
    assert qd.epoch_measures(config(fallback={}), data, "high").get("coder") is None  # switched off: not measured

    best = qd.scores(m, "best")
    assert best["top"]["general"] == 5.0 and best["cheap"]["general"] == 1.0  # floor of 1: any model may take level-1 work
    assert 1.0 < best["mid"]["general"] < 5.0
    spread = qd.scores(m, "range")
    assert spread["top"]["general"] == 5.0 and spread["cheap"]["general"] == 1.0
    assert qd.scores({"a": {"coding": {"chance": 0.3}}}, "range") == {"a": {"coding": 5.0}}  # one model: it's the best


def test_scores_reach_the_models_with_overrides_and_fallbacks():
    cfg = config()
    q = qd.refresh_quality(cfg, epoch=qd.parse_epoch(epoch_zip()))
    assert q["status"]["ok"] and q["attribution"].startswith("Epoch AI") and "big" in q["missing"] and "mock-large" not in q["missing"]
    loaded = config()
    from meritorder.config import apply_quality
    apply_quality(loaded, {"quality": q})
    top, cheap, big = loaded.model("top"), loaded.model("cheap"), loaded.model("big")
    assert top.quality["general"] == 5.0 and top.quality_sources == {"coding": "epoch", "general": "epoch"}
    assert cheap.quality["general"] == 4.5 and cheap.quality_sources["general"] == "override"  # set by hand: always wins
    assert big.quality == {"coding": 5, "general": 5} and big.quality_sources == {"coding": "config", "general": "config"}  # a mock: exempt
    guess, trusted = loaded.model("guess"), loaded.model("trusted")
    assert guess.quality == {"coding": 0.0, "general": 0.0} and guess.quality_sources == {"coding": "unmeasured", "general": "unmeasured"}
    assert trusted.quality["coding"] == 3.5 and trusted.quality_sources["coding"] == "override"  # an override puts it back
    never = config()
    apply_quality(never, {})  # nothing fetched yet: the written scores stand
    assert never.model("guess").quality == {"coding": 4, "general": 4}

    other = config(scale="range")  # scores stored on another scale are worked out again from their chances, at once
    apply_quality(other, {"quality": q})
    assert other.model("top").quality == {"coding": 5.0, "general": 5.0} and other.model("top").quality_sources["coding"] == "epoch"
    assert other.model("mid").quality["coding"] == qd.scores(q["measures"], "range")["mid"]["coding"]
    fixed = config(scale="absolute", anchor={"coding": 0.9, "general": 0.9})  # and on another anchor
    apply_quality(fixed, {"quality": q})
    assert fixed.model("top").quality["coding"] == round(5 * q["measures"]["top"]["coding"]["chance"] / 0.9, 1) < 5
    by_hand = config(source="config")
    apply_quality(by_hand, {"quality": q})
    assert by_hand.model("top").quality == {"coding": 3, "general": 3}


def test_artificial_analysis_and_failures(monkeypatch):
    cfg = config(source="artificial_analysis")
    assert qd.refresh_quality(cfg)["status"]["error"].startswith("ValueError: set ARTIFICIAL_ANALYSIS_API_KEY")
    pages = {1: {"data": [{"slug": "claude-top-1", "name": "Claude Top 1", "evaluations": {"artificial_analysis_intelligence_index": 70}},
                          {"slug": "x", "name": "X", "evaluations": {"artificial_analysis_intelligence_index": None}}], "pagination": {"has_more": True}},
             2: {"data": [{"slug": "cheap-model", "name": "Cheap Model", "evaluations": {"artificial_analysis_intelligence_index": 35}}], "pagination": {"has_more": False}}}
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None, **kw):
        seen.append((url, params, headers))
        return httpx.Response(200, json=pages[params["page"]], request=httpx.Request("GET", url))
    monkeypatch.setattr(qd.httpx, "get", fake_get)
    q = qd.refresh_quality(cfg, aa_key="k")
    assert seen[0][2] == {"x-api-key": "k"} and len(seen) == 2
    assert q["scores"]["top"] == {"coding": 5.0, "general": 5.0} and q["scores"]["cheap"]["coding"] == 2.5  # half the best's index
    assert q["measures"]["top"]["general"]["basis"] == "Artificial Analysis Intelligence Index"

    def down(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(qd.httpx, "get", down)
    failed = qd.refresh_quality(config())
    assert not failed["status"]["ok"] and "offline" in failed["status"]["error"] and failed["scores"] == {}
    assert qd.refresh_quality(config(source="config"))["missing"] == []


async def test_the_worker_keeps_scores_live():
    cfg, store, state = config(), MemoryStore(), MemoryState()
    data = {"zip": epoch_zip()}

    def fetch(c, epoch, aa, key):
        return qd.refresh_quality(c, epoch=qd.parse_epoch(data["zip"]))
    assert (await qd.refresh_quality_once(cfg, store, state, fetch))["stored"] is False  # no live catalog yet: prices first
    await store.set_setting("prices", {"updated": "p1", "prices": {"mock-openai:small-1": {"input": 1, "output": 2}}})
    first = await qd.refresh_quality_once(cfg, store, state, fetch)
    doc = await store.get_setting("prices")
    assert first["stored"] and doc["prices"] and doc["quality"]["scores"]["top"]["general"] == 5.0  # prices kept alongside
    assert await state.get_config_token() == f"file:p1:{doc['quality']['updated']}"

    data["zip"] = epoch_zip(**{"epoch_capabilities_index/eci_scores.csv": "Model,Display name,eci\nx,Claude Top 1,170\nx,Claude Mid 1,169\nx,Cheap Model,150\n"})
    moved = await qd.refresh_quality_once(cfg, store, state, fetch)
    assert any(c.startswith("mid general") for c in moved["changed"]) and any(a["title"] == "Quality scores changed" for a in await state.recent_alerts())

    def broken(c, epoch, aa, key):
        return {**qd.refresh_quality(config(source="config")), "source": "epoch", "status": {"ok": False, "error": "HTTP 503"}}
    kept = await qd.refresh_quality_once(cfg, store, state, broken)
    assert not kept["ok"] and (await store.get_setting("prices"))["quality"]["scores"]["top"]["general"] == 5.0  # the last good ones
    assert any(a["title"] == "Quality scores not refreshed" for a in await state.recent_alerts())

    refresher, calls = qd.QualityRefresher(), []

    def counting(c, epoch, aa, key):
        calls.append(1)
        return fetch(c, epoch, aa, key)
    await refresher.maybe(cfg, store, state, 100.0, counting)
    assert await refresher.maybe(cfg, store, state, 200.0, counting) is None  # not due for a day
    await refresher.maybe(config(scale="range"), store, state, 300.0, counting)  # the settings changed: right away
    assert len(calls) == 2

    def boom(c, epoch, aa, key):
        raise RuntimeError("bug")
    assert await qd.QualityRefresher().maybe(cfg, store, state, 1.0, boom) is None

    def prices_fetch(c, path, write):
        return catalog.refresh_prices(c, {"data": []}, write=False, litellm={})
    await live_prices.refresh_once(cfg, store, state, prices_fetch)
    assert (await store.get_setting("prices"))["quality"]["scores"]  # a price refresh keeps the scores


def test_the_cli_writes_scores_into_prices_json(tmp_path, monkeypatch, capsys):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["models"]["top"] = {"routes": [{"backend": "mock-anthropic", "model": "claude-top-1"}]}
    raw["quality"] = {"scale": "best"}  # the best model is 5, whatever its chance
    (tmp_path / "meritorder.yaml").write_text(json.dumps(raw))
    (tmp_path / "prices.json").write_text(json.dumps({"quality": {"source": "epoch", "scale": "best", "scores": {"top": {"coding": 1.0}}}}))
    monkeypatch.setattr(catalog.httpx, "get", lambda url, **k: httpx.Response(200, json={"data": []}, request=httpx.Request("GET", url)))
    monkeypatch.setattr(qd, "fetch_epoch", lambda: qd.parse_epoch(epoch_zip()))
    cli.main(["prices", "refresh", "--config", str(tmp_path / "meritorder.yaml")])
    out = capsys.readouterr().out
    data = json.loads((tmp_path / "prices.json").read_text())
    assert "Quality scores for 1 models from epoch" in out and data["quality"]["scores"]["top"]["coding"] == 5.0
    assert build_config(raw, tmp_path, quiet=True).model("top").quality_sources == {"general": "epoch", "coding": "epoch"}

    monkeypatch.setattr(qd, "fetch_epoch", lambda: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    cli.main(["prices", "refresh", "--config", str(tmp_path / "meritorder.yaml")])
    assert "Quality scores not refreshed" in capsys.readouterr().out
    assert json.loads((tmp_path / "prices.json").read_text())["quality"]["scores"]["top"]["coding"] == 5.0  # kept


async def test_admin_api_shows_and_refreshes_scores(durable, client, monkeypatch):
    def fake(c, epoch=None, aa=None, aa_key=None):
        return {"updated": "u1", "source": "epoch", "scale": c.quality.scale, "anchor": dict(c.quality.anchor), "reference": dict(c.quality.reference), "status": {"ok": True},
                "attribution": qd.ATTRIBUTION["epoch"], "scores": {"small": {"coding": 2.4}}, "measures": {"small": {"coding": {"chance": 0.3, "basis": "b", "evidence": {}}}},
                "missing": ["big"]}
    monkeypatch.setattr(qd, "refresh_quality", fake)
    assert (await client.post("/v1/admin/quality/refresh", headers=auth(ADMIN_KEY))).json()["stored"] is False  # no prices yet
    await durable.store.set_setting("prices", {"updated": "p", "prices": {}})
    r = await client.post("/v1/admin/quality/refresh", headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["stored"]
    cfg = (await client.get("/v1/admin/config", headers=auth(ADMIN_KEY))).json()
    small = next(m for m in cfg["models"] if m["name"] == "small")
    assert cfg["quality"]["source"] == "epoch" and cfg["quality"]["missing"] == ["big"] and not cfg["quality"]["stale"]
    assert small["quality"]["coding"] == 2.4 and small["quality_sources"]["coding"] == "epoch" and small["quality_detail"]["coding"]["chance"] == 0.3
