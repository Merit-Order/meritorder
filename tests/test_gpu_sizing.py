"""Sizing a GPU replica to a model (Hugging Face sizes) and pricing it from SkyPilot's catalog."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from meritorder import gpu_sizing as gs
from meritorder.config import BackendConfig, Config, ModelConfig, Route, ServeConfig
from meritorder.store import MemoryStore

GPT_OSS_120B = {"hidden_size": 2880, "num_hidden_layers": 36, "num_local_experts": 128, "num_experts_per_tok": 4, "intermediate_size": 2880,
                "num_attention_heads": 64, "head_dim": 64, "num_key_value_heads": 8, "vocab_size": 201088}


def test_parameters_from_names():
    assert gs._name_params("mistralai/Mistral-Large-4-1T-A52B") == (1e12, 52e9)
    assert gs._name_params("meta-llama/Llama-4-Maverick-17B-128E-Instruct") == (None, 17e9)  # 17B is what each token uses
    assert gs._name_params("Qwen/Qwen3-Coder-30B-A3B-Instruct") == (30e9, 3e9)
    assert gs._name_params("openai/gpt-oss-120b") == (120e9, None)
    assert gs._name_params("org/no-numbers") == (None, None)


def test_active_parameters_from_layer_shapes():
    # gpt-oss-120b uses 5.1B parameters per token, by OpenAI's count.
    assert gs.active_from_config(GPT_OSS_120B) == pytest.approx(5.12e9, rel=0.01)
    assert gs.active_from_config({"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008}) is None  # dense
    nested = {"text_config": {**GPT_OSS_120B, "num_attention_heads": None, "moe_layer_freq": [1] * 18 + [0] * 18,
                              "dense_intermediate_size": 5760, "n_shared_experts": 1}}
    # 36 layers of attention at 4h², 18 MoE layers with 4 experts and a shared one, 18 dense ones 5,760 wide, the output head.
    h = 2880
    assert gs.active_from_config(nested) == 36 * 4 * h * h + 18 * 5 * h * 3 * h + 18 * 5760 * 3 * h + 201088 * h


def _client(routes: dict) -> httpx.AsyncClient:
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls[url] = calls.get(url, 0) + 1
        for prefix, answer in routes.items():
            if url.startswith(prefix):
                status, body = answer(calls[url]) if callable(answer) else answer
                return httpx.Response(status, json=body)
        return httpx.Response(404, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _files(*sizes: tuple[str, int]) -> list[dict]:
    return [{"rfilename": n, "size": b} for n, b in sizes]


async def test_model_size_from_hugging_face(monkeypatch):
    monkeypatch.setattr(gs.asyncio, "sleep", lambda s: _done())
    api = "https://huggingface.co/api/models/"
    conf = "https://huggingface.co/"
    async with _client({
        api + "org/moe": lambda n: (429, {}) if n == 1 else (200, {"safetensors": {"parameters": {"BF16": 1_000_000_000, "F8_E4M3": 3_000_000_000},
                                                                                       "total": 4_000_000_000}}),
        # gpt-oss: MXFP4 experts stored as bytes, half a byte per parameter with their scales; a copy under original/.
        api + "org/packed": (200, {"safetensors": {"parameters": {"BF16": 2_000_000_000, "U8": 114_000_000_000}, "total": 116_000_000_000},
                                   "siblings": _files(("model-00001-of-00002.safetensors", 33_000_000_000),
                                                      ("model-00002-of-00002.safetensors", 33_000_000_000),
                                                      ("original/model.safetensors", 66_000_000_000), ("config.json", 2_000))}),
        conf + "org/packed/resolve": (200, GPT_OSS_120B),
        # Real 8-bit integers: a byte each.
        api + "org/int8": (200, {"safetensors": {"parameters": {"BF16": 1_000_000_000, "I8": 9_000_000_000}, "total": 10_000_000_000},
                                 "siblings": _files(("model.safetensors", 11_000_000_000))}),
        # Mistral ships the weights twice: consolidated-* and model-*.
        api + "org/twice": (200, {"safetensors": {"parameters": {"F8_E4M3": 10_000_000_000}, "total": 10_000_000_000},
                                  "siblings": _files(("consolidated-00001-of-00001.safetensors", 10_000_000_000),
                                                     ("model-00001-of-00001.safetensors", 10_000_000_000))}),
        # No metadata Hugging Face can read, but files, and the parameters in its name: 1.06 bytes a parameter is FP8.
        api + "org/Large-1T-A50B": (200, {"siblings": _files(("consolidated-00001-of-00002.safetensors", 530_000_000_000),
                                                             ("consolidated-00002-of-00002.safetensors", 530_000_000_000))}),
        # The files listed hold less than the parameters need: not the whole model, so the parameter counts stand.
        api + "org/partial": (200, {"safetensors": {"parameters": {"BF16": 8_000_000_000}, "total": 8_000_000_000},
                                    "siblings": _files(("model_mtp.safetensors", 1_000_000_000))}),
        conf + "org/moe/resolve": (200, GPT_OSS_120B),
        api + "org/dense": (200, {"safetensors": {"parameters": {"BF16": 8_000_000_000}, "total": 8_000_000_000}}),
        conf + "org/dense/resolve": (200, {"hidden_size": 4096, "num_hidden_layers": 32}),
        api + "org/Big-1T-A50B": (401, {}),
        api + "org/gated-dense": (200, {"safetensors": {"parameters": {"BF16": 1_000_000_000}, "total": 1_000_000_000}}),
        conf + "org/gated-dense/resolve": (401, {}),
    }) as c:
        moe = await gs.fetch_model_size("org/moe", c)  # rate-limited once, then answered
        assert moe["weight_bytes"] == 2_000_000_000 + 3_000_000_000 and moe["bits"] == 8  # most parameters in FP8
        assert moe["bytes_by_bits"] == {"16": 2_000_000_000, "8": 3_000_000_000, "4": 0} and "parameter counts" in moe["basis"]
        assert moe["active_params"] == 4_000_000_000  # capped at the total the weights hold
        packed = await gs.fetch_model_size("org/packed", c)
        assert packed["weight_bytes"] == 66_000_000_000 and packed["bits"] == 4 and packed["published_bits"] == 4
        assert packed["bytes_by_bits"] == {"16": 4_000_000_000, "8": 0, "4": 62_000_000_000} and "packed 4-bit" in packed["basis"]
        assert packed["params_by_bits"]["4"] == 114_000_000_000 and packed["total_params"] == 116_000_000_000
        int8 = await gs.fetch_model_size("org/int8", c)
        assert int8["bits"] == 8 and int8["bytes_by_bits"] == {"16": 2_000_000_000, "8": 9_000_000_000, "4": 0}
        assert (await gs.fetch_model_size("org/twice", c))["weight_bytes"] == 10_000_000_000  # one copy
        large = await gs.fetch_model_size("org/Large-1T-A50B", c)
        assert large["weight_bytes"] == 1_060_000_000_000 and large["bits"] == 8 and large["total_params"] == 1e12 and "(FP8)" in large["basis"]
        assert (await gs.fetch_model_size("org/partial", c))["weight_bytes"] == 16_000_000_000
        assert "estimated from its config" in moe["basis"]
        dense = await gs.fetch_model_size("org/dense", c)
        assert dense["active_params"] == dense["total_params"] == 8_000_000_000 and dense["basis"].endswith("dense")
        named = await gs.fetch_model_size("org/Big-1T-A50B", c)  # gated: from its name, at 16 bits
        assert named["weight_bytes"] == 2e12 and named["active_params"] == 50e9 and "from its name" in named["basis"] and named["bits"] == 16
        gated = await gs.fetch_model_size("org/gated-dense", c)
        assert gated["active_params"] == 1_000_000_000 and "assumed dense" in gated["basis"]
        assert "error" in await gs.fetch_model_size("org/nothing", c)
        both = await gs.fetch_model_sizes(["org/dense", "org/dense", "org/moe"], c)
        assert set(both) == {"org/dense", "org/moe"}


async def _done():
    return None


OFFERS = [
    {"cloud": "aws", "accelerator": "L4", "count": 1, "on_demand": {"usd_per_hour": 0.8, "region": "us-east-1", "instance_type": "g6.xlarge"},
     "spot": {"usd_per_hour": 0.2, "region": "us-west-2", "instance_type": "g6.xlarge"}},
    {"cloud": "aws", "accelerator": "H100", "count": 1, "on_demand": {"usd_per_hour": 6.9, "region": "us-east-1", "instance_type": "p5.4xlarge"},
     "spot": {"usd_per_hour": 2.7, "region": "us-west-2", "instance_type": "p5.4xlarge"}},
    {"cloud": "aws", "accelerator": "H100", "count": 8, "on_demand": {"usd_per_hour": 55.0, "region": "us-east-1", "instance_type": "p5.48xlarge"},
     "spot": {"usd_per_hour": 5.9, "region": "eu-north-1", "instance_type": "p5.48xlarge"}},
    {"cloud": "runpod", "accelerator": "H100", "count": 2, "on_demand": {"usd_per_hour": 5.0, "region": "AU", "instance_type": "2x_H100"}, "spot": None},
    {"cloud": "runpod", "accelerator": "L40S", "count": 3, "on_demand": {"usd_per_hour": 0.1, "region": "AU", "instance_type": "3x_L40S"}, "spot": None},
    {"cloud": "aws", "accelerator": "Trainium", "count": 1, "on_demand": {"usd_per_hour": 0.01, "region": "x", "instance_type": "trn1"}, "spot": None},
]


def test_sizing_a_replica_to_a_model():
    small = {"weight_bytes": 16e9, "active_params": 14e9, "bits": 16}
    r = gs.size_replica(small, OFFERS, {"aws", "runpod"}, spot=True)
    # 16 GB of weights at 70% of memory needs 23 GB: one L4, at its spot price. Three L40S are cheaper but vLLM splits 1, 2, 4 or 8 ways.
    assert (r["accelerator"], r["count"], r["usd_per_hour"], r["spot"], r["region"]) == ("L4", 1, 0.2, True, "us-west-2")
    assert r["on_demand_usd_per_hour"] == 0.8 and r["tflops"] == 121 and r["computes_in"] == "16-bit"
    mid = {"weight_bytes": 100e9, "active_params": 10e9, "bits": 8}
    on_demand = gs.size_replica(mid, OFFERS, {"aws", "runpod"}, spot=False)
    assert (on_demand["accelerator"], on_demand["count"], on_demand["cloud"], on_demand["usd_per_hour"]) == ("H100", 2, "runpod", 5.0)
    assert on_demand["tflops"] == 2 * 1979 and on_demand["computes_in"] == "FP8"  # FP8 weights on a GPU with FP8
    assert gs.size_replica(mid, OFFERS, {"aws"}, spot=True)["count"] == 8  # aws only has 1 or 8
    too_big = gs.size_replica({"weight_bytes": 600e9, "active_params": 1e9, "bits": 16}, OFFERS, {"aws"}, spot=True)
    assert too_big == {"error": "needs 857 GB of GPU memory; the largest machine on offer has 640 GB"}
    assert gs.size_replica(small, OFFERS, {"gcp"}, spot=True) == {"error": "no offer in the chosen clouds"}
    fixed = gs.size_replica(small, OFFERS, {"aws"}, spot=False, accelerators="H100:8")
    assert (fixed["accelerator"], fixed["count"], fixed["usd_per_hour"]) == ("H100", 8, 55.0)
    assert gs.size_replica({"error": "x"}, OFFERS, {"aws"}, True, accelerators="H100:1")["count"] == 1  # a deployment needs no size
    assert gs.size_replica({"error": "unknown"}, OFFERS, {"aws"}, True) == {"error": "unknown"}
    assert "no H100:4 offer" in gs.size_replica(small, OFFERS, {"aws"}, True, accelerators="H100:4")["error"]
    assert gs.size_replica(small, OFFERS, {"aws"}, True, regions=["eu-north-1"])["count"] == 8  # the only spot offer there
    # The machines worth considering, cheapest first, each faster than the cheaper ones: two H100s on demand at RunPod,
    # then eight on spot at AWS. One H100 can't hold it; the L4 and L40S neither.
    options = gs.replica_options(mid, OFFERS, {"aws", "runpod"}, spot=True)
    assert [(o["cloud"], o["count"], o["usd_per_hour"]) for o in options] == [("runpod", 2, 5.0), ("aws", 8, 5.9)]
    small_options = gs.replica_options(small, OFFERS, {"aws", "runpod"}, spot=True)
    assert [(o["accelerator"], o["count"]) for o in small_options] == [("L4", 1), ("H100", 1), ("H100", 2), ("H100", 8)]
    assert gs.replica_options({"weight_bytes": 600e9, "active_params": 1e9}, OFFERS, {"aws"}, True)["error"].startswith("needs 857 GB")


def _size(b16: float, b8: float = 0, b4: float = 0, **kw) -> dict:
    """Sizes as fetch_model_size returns them: bytes by precision, with parameters at 2, 1 and 1/2 bytes each."""
    params = {"16": b16 / 2, "8": b8, "4": b4 * 2}
    bits = gs._dominant(params)
    return {"weight_bytes": b16 + b8 + b4, "bytes_by_bits": {"16": b16, "8": b8, "4": b4}, "params_by_bits": params, "bits": bits,
            "published_bits": bits, "basis": "b", "active_params": 1e9, "total_params": sum(params.values()), **kw}


def test_quantizing():
    dense = _size(80e9)  # a 40B-parameter model in 16 bits
    assert gs.quantized(dense, "published") is dense
    fp8 = gs.quantized(dense, "fp8")
    assert fp8["weight_bytes"] == 40e9 and fp8["bits"] == 8 and fp8["published_bits"] == 16 and fp8["basis"].endswith("served at FP8")
    four = gs.quantized(dense, "4bit")  # 4.25 bits a weight with its scales
    assert four["weight_bytes"] == 21.25e9 and four["bits"] == 4 and four["bytes_by_bits"] == {"16": 0, "8": 0, "4": 21_250_000_000}
    assert "about 4.25 bits" in four["basis"]
    # FP8 published, with some 16-bit layers: vLLM's --quantization fp8 doesn't apply to a quantized checkpoint; 4 bits does.
    glm = _size(4e9, 750e9)
    assert gs.quantized(glm, "fp8")["weight_bytes"] == glm["weight_bytes"] and "served as published" in gs.quantized(glm, "fp8")["basis"]
    assert gs.quantized(glm, "4bit")["weight_bytes"] == pytest.approx(4e9 * 4.25 / 16 + 750e9 * 4.25 / 8)
    # Published in 4 bits (Kimi): the 16-bit layers are all that's left to shrink; nothing at all when there are none.
    kimi = _size(115e9, 0, 1446e9)
    assert kimi["bits"] == 4 and gs.quantized(kimi, "4bit")["weight_bytes"] == pytest.approx(115e9 * 4.25 / 16 + 1446e9)
    only4 = _size(0, 0, 10e9)
    assert gs.quantized(only4, "4bit")["weight_bytes"] == 10e9 and "already 4-bit as published" in gs.quantized(only4, "4bit")["basis"]
    assert gs.quantized({"error": "x"}, "4bit") == {"error": "x"}
    assert gs.quantized({"weight_bytes": 1, "basis": "old"}, "4bit") == {"weight_bytes": 1, "basis": "old"}  # stored before this version


def test_compute_precision():
    b200, h100, a100 = gs.ACCELERATORS["B200"], gs.ACCELERATORS["H100"], gs.ACCELERATORS["A100-80GB"]
    assert gs.peak_tflops(b200, 4) == 4500 and gs.compute_precision(b200, 4) == "FP8"  # Blackwell: 4-bit weights, 8-bit activations
    assert gs.peak_tflops(h100, 4) == 989 and gs.compute_precision(h100, 4) == "16-bit"  # Hopper dequantizes 4-bit weights
    assert gs.peak_tflops(h100, 8) == 1979 and gs.peak_tflops(a100, 8) == 312 and gs.compute_precision(a100, 8) == "16-bit"
    assert gs.peak_tflops(b200, 16) == 2250


def test_capacity():
    size = {"active_params": 1e9}
    # 10% of 1,000 TFLOPS at 2 operations per parameter for 1B active parameters: 50,000 tok/s computed; with half the
    # input read from the prompt cache, twice as many tokens.
    assert gs.capacity_tps({"tflops": 1000}, size, 0.1, 0.0) == pytest.approx(50_000)
    assert gs.capacity_tps({"tflops": 1000}, size, 0.1, 0.5) == pytest.approx(100_000)
    assert gs.capacity_tps({"tflops": 1000}, size, 0.1, 0.99) == pytest.approx(500_000)  # capped at 10x


def _cfg():
    return Config(backends=[
        BackendConfig(name="api", format="openai"),
        BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=16.0, capacity_tokens_per_second=4000,
                      serve=ServeConfig(model="deployed", accelerators="H100:8", clouds=["aws"], use_spot=True)),
        BackendConfig(name="box", kind="self_hosted", format="openai", gpu_cost_per_hour=3.0, capacity_tokens_per_second=900,
                      serve=ServeConfig(model="boxed", provisioner="static", static_url="http://box.test/v1")),
    ], models=[
        ModelConfig(name="deployed", hf_id="org/deployed", quality={"coding": 3}, routes=[Route(backend="gpu", model="d")]),
        ModelConfig(name="boxed", hf_id="org/boxed", quality={"coding": 3}, routes=[Route(backend="box", model="b")]),
        ModelConfig(name="small", hf_id="org/small", quality={"coding": 2}, routes=[Route(backend="api", model="s")]),
        ModelConfig(name="unsized", hf_id="org/unsized", quality={"coding": 2}, routes=[Route(backend="api", model="u")]),
        ModelConfig(name="closed", quality={"coding": 5}, routes=[Route(backend="api", model="c")]),
    ])


def test_replica_plans():
    sizes = {"models": {"org/deployed": _size(160e9, active_params=3e9), "org/small": _size(16e9, active_params=14e9),
                        "org/unsized": {"error": "no size published"}}}
    plans = gs.replica_plans(_cfg(), {"offers": OFFERS}, sizes, {"aws", "runpod"}, spot=True)
    assert set(plans) == {"deployed", "boxed", "small", "unsized"}  # open-weight models only
    d = plans["deployed"]  # its deployment's GPUs, clouds and spot setting, priced from the catalog; its measured capacity
    assert d["deployment"] == "gpu" and (d["replica"]["accelerator"], d["replica"]["count"], d["replica"]["usd_per_hour"]) == ("H100", 8, 5.9)
    assert d["replica"]["source"] == "catalog" and d["capacity_tps"] == 4000
    assert plans["boxed"]["replica"] == {"usd_per_hour": 3.0, "source": "configured"} and plans["boxed"]["capacity_tps"] == 900
    assert plans["small"]["replica"]["accelerator"] == "L4" and plans["small"]["capacity_tps"] is None
    assert plans["small"]["options"][0] is plans["small"]["replica"] and "options" not in plans["deployed"]  # a deployment's GPUs are fixed
    assert plans["unsized"] == {"error": "no size published", "size": None}
    priced = gs.replica_plans(_cfg(), {"offers": OFFERS}, sizes, {"aws"}, spot=True, replica_prices={"gpu": 7.5})
    assert priced["deployed"]["replica"]["usd_per_hour"] == 7.5 and priced["deployed"]["replica"]["source"] == "controller"
    # Not in the catalog: the configured price.
    none = gs.replica_plans(_cfg(), {"offers": []}, sizes, {"aws"}, spot=True)
    assert none["deployed"]["replica"] == {"accelerator": "H100", "count": 8, "usd_per_hour": 16.0, "spot": True, "source": "configured"}
    assert none["small"]["error"] == "no offer in the chosen clouds"
    fp8 = gs.replica_plans(_cfg(), {"offers": OFFERS}, sizes, {"aws", "runpod"}, spot=True, weights="fp8")
    assert fp8["small"]["size"]["weight_bytes"] == 8e9 and fp8["small"]["replica"]["computes_in"] == "FP8"  # an L4 has FP8
    four = gs.replica_plans(_cfg(), {"offers": OFFERS}, sizes, {"aws", "runpod"}, spot=True, weights="4bit")
    assert four["small"]["size"]["weight_bytes"] == 4.25e9 and four["small"]["replica"]["computes_in"] == "16-bit"
    cfg = _cfg()
    cfg.backend("box").capacity_tokens_per_second = 0
    assert gs.replica_plans(cfg, {"offers": OFFERS}, sizes, {"aws"}, spot=True)["boxed"]["error"] == "no price or capacity for its deployment"


async def test_stale_prices_are_fetched_in_the_background_while_the_report_answers(monkeypatch):
    from meritorder import gpu

    monkeypatch.setattr(gpu, "sky_available", lambda: True)
    store = MemoryStore()
    calls = []

    async def catalog(clouds, refresh=False):
        calls.append(("catalog", tuple(clouds), refresh))
        return {"offers": OFFERS, "catalog_updated": {"aws": "2026-11-11T17:00:00+00:00"}, "skypilot": "0.14.2"}

    async def sizes(ids, client):
        calls.append(("sizes", tuple(ids)))
        return {i: {"error": "x"} for i in ids}

    monkeypatch.setattr(gs, "fetch_catalog", catalog)
    monkeypatch.setattr(gs, "fetch_model_sizes", sizes)
    cfg = _cfg()
    cat, found = await gs.current(cfg, store)  # nothing stored: the report answers at once, and says prices are coming
    assert cat == {"refreshing": True} and found == {}
    await gs.settled(store)
    cat, found = await gs.current(cfg, store)
    assert cat["offers"] == OFFERS and cat["skypilot"] == "0.14.2" and cat["refreshing"] is False
    assert set(found["models"]) == {"org/deployed", "org/boxed", "org/small", "org/unsized"}
    assert calls[0] == ("catalog", ("aws", "gcp", "azure", "lambda", "runpod", "nebius"), False)  # its deployment's clouds first
    calls.clear()
    await gs.current(cfg, store)
    assert calls == []  # fresh: as stored
    old = (datetime.now(UTC) - timedelta(hours=7)).isoformat(timespec="seconds")
    await store.set_setting("gpu_catalog", {**cat, "fetched_at": old})
    stale, _ = await gs.current(cfg, store)
    assert stale["offers"] == OFFERS and stale["fetched_at"] == old and stale["refreshing"]  # the old prices, while new ones come
    await gs.settled(store)
    assert [c[0] for c in calls] == ["catalog"]  # prices again; sizes are good for a week
    calls.clear()
    stored = await store.get_setting("model_sizes")
    assert stored["version"] == gs.SIZES_VERSION
    await store.set_setting("model_sizes", {**stored, "version": 1})
    await gs.current(cfg, store)
    await gs.settled(store)
    assert [c[0] for c in calls] == ["sizes"]  # sized by an older version: again
    out = await gs.refresh_once(cfg, store, download=True)
    assert out["offers"] == len(OFFERS) and calls[-2][2] is True and out["unsized"] == sorted(found["models"])

    async def broken(clouds, refresh=False):
        raise gs.SkyPilotMissing("SkyPilot isn't installed")

    monkeypatch.setattr(gs, "fetch_catalog", broken)
    await store.set_setting("gpu_catalog", {**cat, "fetched_at": old})
    await store.set_setting("model_sizes", {})
    calls.clear()
    await gs.current(cfg, store)
    await gs.settled(store)
    cat, found = await gs.current(cfg, store)
    assert cat["problem"] == "SkyPilot isn't installed" and cat["offers"] == OFFERS  # the last prices stand
    assert calls[0][0] == "sizes" and found["models"]  # sizes are fetched all the same


async def test_one_refresh_at_a_time_and_everyone_waits_for_it(monkeypatch):
    """Lookups at once slowed each other past SkyPilot's time limit and starved the gateway (every report load and
    Get latest prices started its own): now they share the one running."""
    import asyncio

    from meritorder import gpu

    monkeypatch.setattr(gpu, "sky_available", lambda: True)
    store, calls, release = MemoryStore(), [], asyncio.Event()

    async def catalog(clouds, refresh=False):
        calls.append(refresh)
        await release.wait()
        return {"offers": OFFERS, "skypilot": "0.14.2"}

    async def sizes(ids, client):
        return {i: {"error": "x"} for i in ids}

    monkeypatch.setattr(gs, "fetch_catalog", catalog)
    monkeypatch.setattr(gs, "fetch_model_sizes", sizes)
    cfg = _cfg()
    for _ in range(5):  # five report loads while prices are missing
        cat, _ = await gs.current(cfg, store)
        assert cat["refreshing"]
    joined = asyncio.create_task(gs.refresh_once(cfg, store))  # asks for nothing more: joins it
    later = asyncio.create_task(gs.refresh_once(cfg, store, download=True))  # wants catalogs downloaded: runs after it
    await asyncio.sleep(0)
    assert calls == [False] and gs.refreshing(store)
    release.set()
    assert (await joined)["offers"] == len(OFFERS)
    assert (await later)["offers"] == len(OFFERS) and calls == [False, True]
    assert not gs.refreshing(store)
    # A caller who goes away doesn't stop it.
    release.clear()
    gone = asyncio.create_task(gs.refresh_once(cfg, store, download=True))
    await asyncio.sleep(0)
    gone.cancel()
    await asyncio.sleep(0)
    assert gs.refreshing(store)
    release.set()
    await gs.settled(store)
    assert calls == [False, True, True] and not gs.refreshing(store)


async def test_a_failed_refresh_is_reported_and_retried_after_a_pause(monkeypatch):
    from meritorder import gpu

    monkeypatch.setattr(gpu, "sky_available", lambda: True)
    store, calls = MemoryStore(), []

    async def catalog(clouds, refresh=False):
        calls.append(refresh)
        raise RuntimeError("SkyPilot's catalog lookup failed: it took longer than 10 minutes")

    async def sizes(ids, client):
        return {i: {"error": "x"} for i in ids}

    monkeypatch.setattr(gs, "fetch_catalog", catalog)
    monkeypatch.setattr(gs, "fetch_model_sizes", sizes)
    cfg = _cfg()
    await gs.current(cfg, store)
    await gs.settled(store)
    cat, _ = await gs.current(cfg, store)  # the report says why, and doesn't start another right away
    assert cat["problem"] == "SkyPilot's catalog lookup failed: it took longer than 10 minutes" and not cat["refreshing"]
    assert calls == [False]
    with pytest.raises(RuntimeError, match="10 minutes"):  # Get latest prices tries again now
        await gs.refresh_once(cfg, store, download=True)
    assert calls == [False, True]
    key = gs._key(store)
    gs._failed[key] = (gs._failed[key][0], datetime.now(UTC) - timedelta(minutes=gs.RETRY_MINUTES + 1))
    await gs.current(cfg, store)  # a while later, a report starts one
    await gs.settled(store)
    assert calls == [False, True, False]

    async def works(clouds, refresh=False):
        return {"offers": OFFERS}

    monkeypatch.setattr(gs, "fetch_catalog", works)
    await gs.refresh_once(cfg, store)
    cat, _ = await gs.current(cfg, store)
    assert "problem" not in cat and cat["offers"] == OFFERS  # one that works clears it


def test_a_failed_catalog_lookup_says_why(monkeypatch, caplog):
    import subprocess

    from meritorder import gpu

    monkeypatch.setattr(gpu, "sky_python", lambda: "/opt/skypilot/bin/python")

    def run(result):
        def fake(*a, **kw):
            assert kw["timeout"] == gpu.CATALOG_TIMEOUT_SECONDS
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(subprocess, "run", fake)

    run(subprocess.TimeoutExpired("python", gpu.CATALOG_TIMEOUT_SECONDS))
    with caplog.at_level("WARNING", logger="meritorder.gpu"):
        assert gpu.catalog_offers(["aws"]) is None
    assert gpu.catalog_error() == "it took longer than 10 minutes" and "took longer than 10 minutes" in caplog.text
    run(subprocess.CompletedProcess([], 1, "", "Traceback (most recent call last):\n  ...\nValueError: bad catalog\n\n"))
    assert gpu.catalog_offers(["aws"]) is None and gpu.catalog_error() == "SkyPilot exited with status 1: ValueError: bad catalog"
    run(OSError("no such file"))
    assert gpu.catalog_offers(["aws"]) is None and "couldn't run SkyPilot's Python (/opt/skypilot/bin/python)" in gpu.catalog_error()
    run(subprocess.CompletedProcess([], 0, '{"offers": []}\n', ""))
    assert gpu.catalog_offers(["aws"]) == {"offers": []} and gpu.catalog_error() is None


async def test_fetch_catalog_goes_through_skypilot(monkeypatch):
    from meritorder import gpu

    monkeypatch.setattr(gpu, "sky_available", lambda: False)
    with pytest.raises(gs.SkyPilotMissing):
        await gs.fetch_catalog(["aws"])
    monkeypatch.setattr(gpu, "sky_available", lambda: True)
    monkeypatch.setattr(gpu, "catalog_offers", lambda clouds, refresh: {"offers": [], "clouds": clouds, "refresh": refresh})
    assert await gs.fetch_catalog(["aws"], True) == {"offers": [], "clouds": ["aws"], "refresh": True}
    monkeypatch.setattr(gpu, "catalog_offers", lambda clouds, refresh: None)
    monkeypatch.setattr(gpu, "_last_catalog_error", "it took longer than 10 minutes")
    with pytest.raises(RuntimeError, match="lookup failed: it took longer than 10 minutes"):
        await gs.fetch_catalog(["aws"])


async def test_without_skypilot_prices_come_from_the_controller(monkeypatch):
    from meritorder import gpu
    from meritorder.controller import PRICES_REQUESTED, publish_gpu_prices
    from meritorder.state import MemoryState

    store, state = MemoryStore(), MemoryState()
    calls = []

    async def catalog(clouds, refresh=False):
        calls.append(("catalog", refresh))
        return {"offers": OFFERS, "skypilot": "0.14.2"}

    async def sizes(ids, client):
        calls.append(("sizes",))
        return {i: {"error": "x"} for i in ids}

    monkeypatch.setattr(gs, "fetch_catalog", catalog)
    monkeypatch.setattr(gs, "fetch_model_sizes", sizes)
    monkeypatch.setattr(gpu, "sky_available", lambda: False)
    cfg = _cfg()
    await gs.current(cfg, store)  # the gateway has no SkyPilot: sizes only (in the background), and it says where prices come from
    await gs.settled(store)
    cat, found = await gs.current(cfg, store)
    assert calls == [("sizes",)] and found["models"] and "hasn't published any yet" in cat["problem"]
    # The controller (which has SkyPilot) publishes them when they're stale...
    calls.clear()
    await publish_gpu_prices(cfg, store, state)
    assert calls == [("catalog", False)]
    cat, _ = await gs.current(cfg, store)
    assert cat["offers"] == OFFERS and "problem" not in cat
    calls.clear()
    await publish_gpu_prices(cfg, store, state)
    assert calls == []  # fresh
    # ...and right away, catalogs downloaded again, when someone asks for the latest.
    await state.set_json(PRICES_REQUESTED, {"by": "root"})
    await publish_gpu_prices(cfg, store, state)
    assert calls == [("catalog", True)] and await state.get_json(PRICES_REQUESTED) is None
    old = (datetime.now(UTC) - timedelta(hours=7)).isoformat(timespec="seconds")
    await store.set_setting("gpu_catalog", {**cat, "fetched_at": old})
    cat, _ = await gs.current(cfg, store)
    assert "hasn't published any in the last 6 hours" in cat["problem"] and cat["offers"] == OFFERS

    async def broken(clouds, refresh=False):
        raise RuntimeError("catalog down")

    monkeypatch.setattr(gs, "fetch_catalog", broken)
    await publish_gpu_prices(cfg, store, state)  # logged; the controller carries on


async def test_a_controller_without_skypilot_says_so_once(monkeypatch, caplog):
    """A controller for GPUs on Kubernetes alone runs the plain image, without SkyPilot (docs/07). Its prices come from
    kubernetes.node_types, so catalog prices can't be fetched: found on kind, it warned about that every cycle. It warns
    once, tries again every few minutes, and at once when someone asks for the latest."""
    import logging

    from meritorder import controller
    from meritorder.controller import PRICES_REQUESTED, publish_gpu_prices
    from meritorder.state import MemoryState

    store, state, calls, clock = MemoryStore(), MemoryState(), [], [1000.0]

    async def missing(clouds, refresh=False):
        calls.append(refresh)
        raise RuntimeError("SkyPilot isn't installed where the gateway runs")

    monkeypatch.setattr(gs, "fetch_catalog", missing)
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(controller, "_prices_failed", {"at": None, "error": None})
    cfg = _cfg()
    with caplog.at_level(logging.INFO, logger="meritorder.controller"):
        for _ in range(3):  # three cycles
            await publish_gpu_prices(cfg, store, state)
        clock[0] += controller.PRICES_RETRY_SECONDS
        await publish_gpu_prices(cfg, store, state)
        await state.set_json(PRICES_REQUESTED, {"by": "root"})
        await publish_gpu_prices(cfg, store, state)
    assert calls == [False, False, True]
    assert caplog.text.count("GPU prices couldn't be fetched") == 1 and await state.get_json(PRICES_REQUESTED) is None

