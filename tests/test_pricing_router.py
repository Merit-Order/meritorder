import time

import pytest

from meritorder.backends.base import Backend
from meritorder.config import BackendConfig, Config, ModelConfig, Price, Route, TeamConfig
from meritorder.pricing import Usage, attribute
from meritorder.router import Router, RoutingError, shadow_multiplier
from meritorder.sessions import Session

SONNET = Price(input=3, output=15, cache_read=0.3, cache_write=3.75)
CHEAP = Price(input=0.2, output=1.0, cache_read=0.02)


def test_savings_add_up():
    u = Usage(input_tokens=1_000, cache_read_tokens=90_000, cache_write_tokens=2_000, output_tokens=500)
    c = attribute(u, served_price=CHEAP, baseline_price=SONNET, proxy_cache=True)
    assert c.savings_total == pytest.approx(c.baseline - c.actual)
    assert c.savings_routing > 0 and c.savings_caching > 0 and c.savings_self_hosting == 0


def test_client_caching_is_not_claimed():
    u = Usage(input_tokens=1_000, cache_read_tokens=90_000, output_tokens=500)
    c = attribute(u, served_price=SONNET, baseline_price=SONNET, proxy_cache=False)
    assert c.savings_caching == 0
    assert c.savings_total == pytest.approx(0)


def test_self_hosting_savings():
    u = Usage(input_tokens=10_000, output_tokens=1_000)
    c = attribute(u, served_price=CHEAP, baseline_price=CHEAP, proxy_cache=False, self_hosted_cost=0.0001)
    assert c.actual == 0.0001
    assert c.savings_self_hosting == pytest.approx(c.served_api - 0.0001)


def test_shadow_price_curve():
    assert shadow_multiplier(0.1, 0.6, 0.02) == 0.02
    assert 0.02 < shadow_multiplier(0.8, 0.6, 0.02) < 1.0
    assert shadow_multiplier(1.0, 0.6, 0.02) is None


def _router(extra_backends=()):
    backends_cfg = [
        BackendConfig(name="api-a", format="anthropic", base_url="https://a"),
        BackendConfig(name="api-b", format="anthropic", base_url="https://b"),
        BackendConfig(name="api-o", format="openai", base_url="https://o"),
        BackendConfig(name="gpu", format="openai", kind="self_hosted", base_url="http://gpu"),
        *extra_backends,
    ]
    models = [
        ModelConfig(name="frontier", quality={"coding": 5, "general": 5},
                    routes=[Route(backend="api-a", model="f", price=SONNET), Route(backend="api-b", model="f", price=SONNET)]),
        ModelConfig(name="cheap", quality={"coding": 3, "general": 3}, routes=[Route(backend="api-o", model="c", price=CHEAP)]),
        ModelConfig(name="openweight", quality={"coding": 4, "general": 4}, routes=[Route(backend="gpu", model="ow", price=Price(input=1, output=4))]),
    ]
    cfg = Config(backends=backends_cfg, models=models, teams=[TeamConfig(name="default")])
    backends = {b.name: Backend(b) for b in backends_cfg}
    return Router(cfg, backends), cfg


def coding_body(n_tools=1):
    return {"messages": [{"role": "user", "content": "fix this bug in app.py"}], "tools": [{"name": f"t{i}"} for i in range(n_tools)]}


def test_auto_prefers_idle_gpu_even_for_better_model():
    router, cfg = _router()
    d = router.decide("auto", cfg.team("default"), coding_body(), {}, None, 5_000, True, {"gpu": 0.1}, time.time())
    assert d.floor.level == 3
    assert d.candidates[0].model.name == "openweight"  # quality 4 at ~zero marginal cost
    assert d.candidates[0].upgraded


def test_saturated_gpu_spills_to_api():
    router, cfg = _router()
    d = router.decide("auto", cfg.team("default"), coding_body(), {}, None, 5_000, True, {"gpu": 1.0}, time.time())
    assert d.candidates[0].model.name == "cheap"
    assert all(c.backend.name != "gpu" for c in d.candidates)


def test_explicit_model_is_never_substituted():
    router, cfg = _router()
    d = router.decide("frontier", cfg.team("default"), coding_body(), {}, None, 5_000, True, {}, time.time())
    assert {c.model.name for c in d.candidates} == {"frontier"}
    with pytest.raises(RoutingError):
        router.decide("nope", cfg.team("default"), coding_body(), {}, None, 5_000, True, {}, time.time())


def test_warm_cache_keeps_session_on_its_backend():
    router, cfg = _router()
    now = time.time()
    sess = Session(id="s", model="frontier", backend="api-b", turns=10, prefix_tokens=100_000, warm_until={"api-b": now + 200})
    d = router.decide("frontier", cfg.team("default"), coding_body(), {}, sess, 102_000, False, {}, now)
    assert d.candidates[0].backend.name == "api-b"
    assert d.candidates[0].cost < d.candidates[1].cost  # cold api-a pays to rebuild 100k tokens of cache


def test_floor_header_and_team_policy():
    router, cfg = _router()
    d = router.decide("auto", cfg.team("default"), coding_body(), {"x-route-floor": "coding:5"}, None, 5_000, True, {}, time.time())
    assert {c.model.name for c in d.candidates} == {"frontier"}
    team = TeamConfig(name="locked", allowed_models=["cheap"])
    d = router.decide("auto", team, coding_body(), {}, None, 5_000, True, {}, time.time())
    assert {c.model.name for c in d.candidates} == {"cheap"}


def test_normalize_base_url():
    from meritorder.config import normalize_base_url

    assert normalize_base_url("https://r.services.ai.azure.com/api/projects/defaultProject/anthropic") == "https://r.services.ai.azure.com/anthropic"
    assert normalize_base_url("https://r.openai.azure.com/openai/v1/openai/v1") == "https://r.openai.azure.com/openai/v1"
    assert normalize_base_url("https://api.anthropic.com/") == "https://api.anthropic.com"


def test_policy_exclusions():
    team = TeamConfig(name="t", allowed_backends=["*", "!openrouter-free", "!deep*"], allowed_models=["claude-*", "kimi*"])
    assert team.allows_backend("bedrock")
    assert not team.allows_backend("openrouter-free") and not team.allows_backend("deepseek")
    assert team.allows_model("claude-sonnet") and not team.allows_model("glm-5.3")


def test_effort_formats():
    from meritorder.config import effort_format

    assert effort_format(BackendConfig(name="a", format="anthropic", base_url="https://bedrock-mantle.x/anthropic")) == "anthropic"
    assert effort_format(BackendConfig(name="o", format="openai", base_url="https://openrouter.ai/api/v1")) == "openrouter"
    assert effort_format(BackendConfig(name="p", format="openai", base_url="https://api.openai.com/v1")) == "openai"
    assert effort_format(BackendConfig(name="v", format="openai", base_url="http://vllm:8000/v1")) == "none"
    assert effort_format(BackendConfig(name="v", format="openai", base_url="http://vllm:8000/v1", effort_format="openai")) == "openai"


def test_openrouter_effort_shape():
    from meritorder.classifier import Floor
    from meritorder.proxy import Gateway
    from meritorder.router import Candidate, Decision
    from meritorder.sessions import Session

    bc = BackendConfig(name="openrouter", format="openai", base_url="https://openrouter.ai/api/v1")
    model = ModelConfig(name="glm", effort_levels=["low", "medium", "high"], routes=[Route(backend="openrouter", model="z-ai/glm-5.3")])
    cfg = Config(backends=[bc], models=[model])
    gw = Gateway(cfg, None, None, {}, None, None)
    cand = Candidate(model=model, route=model.routes[0], backend=Backend(bc), cost=0, warm=False, quality=3)
    decision = Decision(requested="auto", auto=True, floor=Floor("general", 1, "test"), candidates=[cand])
    up, _ = gw.build_upstream({"messages": [{"role": "user", "content": "hi"}]}, "openai", cand, "s", decision, Session(id="s"))
    assert up["reasoning"] == {"effort": "low"} and "reasoning_effort" not in up
    up, _ = gw.build_upstream({"messages": [], "reasoning": {"effort": "high"}}, "openai", cand, "s", decision, Session(id="s"))
    assert up["reasoning"] == {"effort": "high"}  # client's choice kept
