"""Sizing a GPU replica to a model, priced from SkyPilot's catalog.

- **What a model needs** comes from Hugging Face: its weights' size in bytes as published (the safetensors files'
  sizes; their metadata says how many parameters are in 16 bits, 8 bits, or packed 4 bits), and for
  mixture-of-experts models the parameters active per token (its config.json, or a name such as "A3B"), which is
  what sets throughput.
- **Quantizing** (``quantized``) serves weights published in 16 bits at 8 (FP8), or anything above 4 bits at 4
  (AWQ, GPTQ, NVFP4, MXFP4: about 4.25 bits a weight with their scales). It changes what fits and what the GPUs
  compute in, not the quality scores, which are for the weights as published.
- **What GPUs cost** comes from SkyPilot's catalog, through SkyPilot itself (in its own environment, as the
  controller uses it): on-demand and spot prices for every instance with GPUs, by cloud and region. SkyPilot keeps
  its catalogs current on its own schedule; a refresh downloads them again first. Lookups never use cloud
  credentials (none are needed for prices).
- **A replica** is a setup (GPU type and count, 1-8 on one machine) where the weights fill at most ``FIT`` of the
  GPUs' memory, leaving the rest for the KV cache. ``replica_options`` lists the ones worth considering (cheapest per
  hour first, each faster than the cheaper ones); the break-even picks the one that pays from the least traffic.
- **Its throughput** is an estimate: a share of the GPUs' peak dense compute at the precision they compute the
  weights in (``compute_precision``), at two operations per active parameter per token. Input read from the prompt
  cache isn't computed again, so it doesn't count against it. A deployment's configured
  ``capacity_tokens_per_second`` (measured) replaces the estimate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

log = logging.getLogger("meritorder.gpu_sizing")

DEFAULT_CLOUDS = ("aws", "gcp", "azure", "lambda", "runpod", "nebius")
HF_API = "https://huggingface.co/api/models/{id}?blobs=true&expand[]=safetensors&expand[]=siblings"
HF_CONFIG = "https://huggingface.co/{id}/resolve/main/config.json"
FIT = 0.7  # the weights may fill this share of the GPUs' memory; the rest is the KV cache's
CATALOG_MAX_AGE_HOURS = 6  # the report fetches prices again when they're older than this
RETRY_MINUTES = 5  # after a refresh fails, reports wait this long before starting another (Get latest prices doesn't)
SIZES_MAX_AGE_HOURS = 24 * 7
SIZES_VERSION = 2  # stored sizes from an older version are fetched again
FOUR_BIT = 4.25  # bits a 4-bit weight takes with its scales (one 8-bit scale per 32 weights, as MXFP4 and Kimi's checkpoints)
WEIGHTS = ("published", "fp8", "4bit")  # how the weights are served: as published, or quantized to 8 or 4 bits


@dataclass(frozen=True)
class Accelerator:
    memory_gb: float
    bf16_tflops: float  # dense, from the vendor's spec sheet
    fp8_tflops: float | None = None  # GPUs with FP8 tensor cores (Ada, Hopper, Blackwell, MI300)
    fp4: bool = False  # FP4 tensor cores (Blackwell): 4-bit weights are computed with 8-bit activations, at the FP8 rate


# Datacenter GPUs vLLM serves on, by SkyPilot's names for them. Peak dense tensor throughput from the spec sheets.
ACCELERATORS = {
    "L4": Accelerator(24, 121, 242),
    "A10G": Accelerator(24, 70),
    "A10": Accelerator(24, 125),
    "L40S": Accelerator(48, 362, 733),
    "A100": Accelerator(40, 312),
    "A100-80GB": Accelerator(80, 312),
    "A100-80GB-SXM": Accelerator(80, 312),
    "H100": Accelerator(80, 989, 1979),
    "H100-SXM": Accelerator(80, 989, 1979),
    "H100-MEGA": Accelerator(80, 989, 1979),
    "H100-NVL": Accelerator(94, 835, 1671),
    "GH200": Accelerator(96, 989, 1979),
    "H200": Accelerator(141, 989, 1979),
    "H200-SXM": Accelerator(141, 989, 1979),
    "B200": Accelerator(180, 2250, 4500, fp4=True),
    "B300": Accelerator(288, 2250, 4500, fp4=True),
    "MI300X": Accelerator(192, 1307, 2615),
}

DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
               "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}
BITS_NAME = {16: "16-bit", 8: "FP8", 4: "4-bit"}


# ------------------------------------------------------------------ what a model needs

def _name_params(hf_id: str) -> tuple[float | None, float | None]:
    """(total, active) parameters from a name like "Mistral-Large-4-1T-A52B" or "Llama-4-Maverick-17B-128E"."""
    name = hf_id.rsplit("/", 1)[-1]
    active = (re.search(r"(?:^|[-_])A(\d+(?:\.\d+)?)B(?![a-z])", name, re.I)  # "...-A52B"
              or re.search(r"(?:^|[-_])(\d+(?:\.\d+)?)B-\d+E(?![a-z])", name, re.I))  # "...-17B-128E": 17B active
    totals = [float(m.group(1)) * (1e12 if m.group(2).upper() == "T" else 1e9)
              for m in re.finditer(r"(?:^|[-_])(\d+(?:\.\d+)?)([BT])(?![a-z])(?!-\d+E)", name, re.I)]
    return (totals[-1] if totals else None), (float(active.group(1)) * 1e9 if active else None)


def active_from_config(config: dict) -> float | None:
    """Parameters each token goes through in a mixture-of-experts model, from its layer sizes: attention, the experts
    it's routed to plus any shared expert, dense layers, and the output head. None when the config describes no
    experts (a dense model) or not enough to work it out. Counted from the shapes, not the stored weights, which
    for 4-bit experts hold two parameters per byte."""
    c = {**config, **(config.get("text_config") if isinstance(config.get("text_config"), dict) else {})}
    experts = c.get("n_routed_experts") or c.get("num_experts") or c.get("num_local_experts") or c.get("moe_num_experts")
    per_token = c.get("num_experts_per_tok") or c.get("moe_topk") or c.get("num_experts_per_token")
    h, layers = c.get("hidden_size"), c.get("num_hidden_layers")
    width = c.get("moe_intermediate_size") or c.get("intermediate_size")
    if not (experts and per_token and h and layers and width) or experts <= per_token:
        return None
    heads = c.get("num_attention_heads")
    head_dim = c.get("head_dim") or (h // heads if heads else None)
    if heads and head_dim:
        kv = c.get("num_key_value_heads") or heads
        attention = 2 * h * heads * head_dim + 2 * h * kv * head_dim  # q and o, k and v
    else:
        attention = 4 * h * h
    freq = c.get("moe_layer_freq")
    moe_layers = sum(1 for x in freq if x) if isinstance(freq, list) else layers - int(c.get("first_k_dense_replace") or 0)
    dense_width = c.get("dense_intermediate_size") or c.get("intermediate_size") or width
    shared = (c.get("shared_expert_intermediate_size") or c.get("shared_intermediate_size")
              or (int(c.get("n_shared_experts") or 0) * width))
    ffn = moe_layers * (per_token * width + shared) * 3 * h + (layers - moe_layers) * dense_width * 3 * h
    return float(layers * attention + ffn + (c.get("vocab_size") or 0) * h)


def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """GET from Hugging Face, waiting out a rate limit or a hiccup twice before giving up."""
    for attempt in range(3):
        r = await client.get(url, headers=_hf_headers(), follow_redirects=True)
        if r.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
            return r
        await asyncio.sleep(1.0 * (attempt + 1))
    return r


def _root_safetensors_bytes(siblings: list[dict]) -> int | None:
    """The model's weights on disk: its top-level .safetensors files (not copies such as gpt-oss's original/), of one
    format where a repository has two (Mistral's ship consolidated-* for mistral-inference and model-* for
    transformers and vLLM: the same weights twice)."""
    files = [f for f in siblings or [] if str(f.get("rfilename", "")).endswith(".safetensors") and "/" not in f["rfilename"]]
    if not files or any(f.get("size") is None for f in files):
        return None
    hf_format = [f for f in files if not f["rfilename"].startswith("consolidated")]
    return sum(int(f["size"]) for f in hf_format or files)


def _bucket(bytes_per_param: float) -> int:
    return 16 if bytes_per_param >= 1.5 else 8 if bytes_per_param >= 0.75 else 4


def _dominant(params_by_bits: dict) -> int:
    """The precision most of the parameters are in, which is what the GPUs compute in."""
    return max((16, 8, 4), key=lambda b: (params_by_bits.get(str(b)) or 0, -b))


def _breakdown(params: dict, files: int | None) -> tuple[int, dict, dict, str]:
    """(weight bytes, bytes by bits, parameters by bits, how) from Hugging Face's parameter counts by dtype and the
    files' size. Packed 4-bit weights (gpt-oss's and Kimi's MXFP4, DeepSeek's FP4 experts) are stored as U8 or I8
    with about half a byte per parameter; the files' size tells them apart from real 8-bit ones."""
    p16 = sum(n for k, n in params.items() if k in ("BF16", "F16"))
    p8 = sum(n for k, n in params.items() if k.startswith("F8"))
    packed = sum(n for k, n in params.items() if k in ("I8", "U8"))
    other = sum(n * DTYPE_BYTES.get(k, 2) for k, n in params.items() if k not in ("BF16", "F16", "I8", "U8") and not k.startswith("F8"))
    known = 2 * p16 + p8 + other
    if files and files < known:  # fewer bytes than the parameters need: the file list isn't the whole model
        files = None
    four = 0
    if files and packed and files > known and (files - known) / packed < 0.75:
        four, how = files - known, "weights from Hugging Face (file sizes; packed 4-bit where stored as bytes)"
    else:
        p8 += packed
        how = "weights from Hugging Face (file sizes)" if files else "weights from Hugging Face (parameter counts)"
    weight = files or known + four
    return (weight, {"16": 2 * p16, "8": p8, "4": four}, {"16": p16, "8": p8, "4": packed if four else 0}, how)


async def fetch_model_size(hf_id: str, client: httpx.AsyncClient) -> dict:
    """{weight_bytes, bytes_by_bits, params_by_bits, bits, total_params, active_params, basis}: the weights as
    published, by precision (16, 8 or 4 bits); ``error`` instead when nothing is known."""
    total = weight = None
    by_bits = params_by_bits = None
    basis = []
    try:
        r = await _get(client, HF_API.format(id=hf_id))
        info = r.json() if r.status_code == 200 else {}
        st = info.get("safetensors") or {}
        params = st.get("parameters") or {}
        files = _root_safetensors_bytes(info.get("siblings") or [])
        if params:
            weight, by_bits, params_by_bits, how = _breakdown(params, files)
            total = float(st.get("total") or sum(params.values()))
            basis.append(how)
        elif files:
            weight = files  # no metadata (a format Hugging Face doesn't parse): the parameters from its name, below
    except (httpx.HTTPError, ValueError) as e:
        log.debug("%s: model info failed: %s", hf_id, e)
    name_total, name_active = _name_params(hf_id)
    if total is None and name_total:
        total = name_total
        if weight:  # file sizes and a parameter count in the name: their ratio says the precision
            bits = _bucket(weight / total)
            basis.append(f"weights from Hugging Face (file sizes), parameters from its name ({BITS_NAME[bits]})")
        else:
            bits, weight = 16, name_total * 2
            basis.append("size from its name, at 16 bits")
        by_bits = {str(b): (weight if b == bits else 0) for b in (16, 8, 4)}
        params_by_bits = {str(b): (total if b == bits else 0) for b in (16, 8, 4)}
    if total is None:
        return {"error": "no size published (gated, or no weights on Hugging Face)"}
    active = name_active
    if active:
        basis.append("active parameters from its name")
    else:
        try:
            r = await _get(client, HF_CONFIG.format(id=hf_id))
            if r.status_code == 200:
                moe = active_from_config(r.json())
                active = min(moe, total) if moe else total
                basis.append("active parameters estimated from its config" if moe else "dense")
        except (httpx.HTTPError, ValueError) as e:
            log.debug("%s: config failed: %s", hf_id, e)
    if not active:
        active = total
        basis.append("assumed dense (no config to read)")
    bits = _dominant(params_by_bits)
    return {"weight_bytes": int(weight), "bytes_by_bits": {k: int(v) for k, v in by_bits.items()},
            "params_by_bits": {k: int(v) for k, v in params_by_bits.items()}, "bits": bits, "published_bits": bits,
            "total_params": int(total), "active_params": int(active), "basis": "; ".join(basis)}


def quantized(size: dict, weights: str) -> dict:
    """The model served with its weights quantized: ``fp8`` serves a 16-bit checkpoint in FP8 (vLLM's
    --quantization fp8, which doesn't apply to a checkpoint that's already quantized); ``4bit`` serves everything
    above 4 bits in 4 (a 4-bit checkpoint made from it: AWQ, GPTQ, NVFP4), at FOUR_BIT bits a weight. Weights
    already at or below the target stay as they are. ``published`` changes nothing."""
    if weights == "published" or "error" in size or "bytes_by_bits" not in size:
        return size
    target = 8 if weights == "fp8" else 4
    by_bits, params = dict(size["bytes_by_bits"]), dict(size["params_by_bits"])
    if target == 8 and (by_bits.get("8") or by_bits.get("4")):
        return {**size, "basis": size["basis"] + "; published already quantized, so served as published"}
    new_bits, saved = (8 if target == 8 else FOUR_BIT), 0.0
    for k in ("16", "8"):
        if int(k) <= target or not by_bits.get(k):
            continue
        shrunk = by_bits[k] * new_bits / int(k)
        saved += by_bits[k] - shrunk
        by_bits[str(target)] = by_bits.get(str(target), 0) + shrunk
        params[str(target)] = params.get(str(target), 0) + params.get(k, 0)
        by_bits[k] = params[k] = 0
    if not saved:
        return {**size, "basis": size["basis"] + f"; already {BITS_NAME[size['bits']]} as published"}
    return {**size, "weight_bytes": int(size["weight_bytes"] - saved), "bytes_by_bits": {k: int(v) for k, v in by_bits.items()},
            "params_by_bits": params, "bits": _dominant(params),
            "basis": size["basis"] + f"; served at {BITS_NAME[target]}" + (f" (about {FOUR_BIT} bits a weight with its scales)" if target == 4 else "")}


async def fetch_model_sizes(hf_ids: list[str], client: httpx.AsyncClient) -> dict:
    sem = asyncio.Semaphore(8)

    async def one(i: str):
        async with sem:
            return i, await fetch_model_size(i, client)

    return dict(await asyncio.gather(*(one(i) for i in sorted(set(hf_ids)))))


# ------------------------------------------------------------------ what GPUs cost

class SkyPilotMissing(RuntimeError):
    pass


async def fetch_catalog(clouds: list[str], refresh: bool = False) -> dict:
    """GPU offers in these clouds from SkyPilot's catalog (see ``gpu.catalog_offers``)."""
    from .gpu import catalog_error, catalog_offers, sky_available

    if not sky_available():
        raise SkyPilotMissing("SkyPilot isn't installed where the gateway runs: build its image with --build-arg SKYPILOT=0.14.2, "
                              "or point MERITORDER_SKY_PYTHON at SkyPilot's Python")
    found = await asyncio.to_thread(catalog_offers, clouds, refresh)
    if not found:
        raise RuntimeError(f"SkyPilot's catalog lookup failed: {catalog_error() or 'it returned nothing'}")
    return found


def catalog_clouds(cfg) -> list[str]:
    """The clouds to price: SkyPilot deployments' own clouds first, then the common GPU clouds."""
    own = [c.lower() for b in cfg.backends if b.serve and b.serve.provisioner == "skypilot" for c in b.serve.clouds]
    return list(dict.fromkeys(own + list(DEFAULT_CLOUDS)))


async def refresh(cfg, store, client: httpx.AsyncClient | None = None, sizes: bool = True, download: bool = False,
                  catalog: bool = True) -> dict:
    """Fetch GPU prices from SkyPilot (``download``: its catalogs too, rather than on its own schedule) and, with
    ``sizes``, every open-weight model's size, and keep them in the store."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=60, headers={"User-Agent": "merit-order"})
    try:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        clouds = catalog_clouds(cfg)
        failed: Exception | None = None
        out: dict = {}
        try:
            if catalog:
                found = {**await fetch_catalog(clouds, download), "fetched_at": now, "clouds": clouds}
                await store.set_setting("gpu_catalog", found)
                out = {"catalog": {k: found.get(k) for k in ("fetched_at", "clouds", "catalog_updated", "skypilot")}, "offers": len(found["offers"])}
        except Exception as e:  # noqa: BLE001 (sizes are still worth fetching; the caller hears about it below)
            failed = e
        if sizes:
            ids = [m.hf_id for m in cfg.models if m.kind == "chat" and m.hf_id]
            found = await fetch_model_sizes(ids, client)
            await store.set_setting("model_sizes", {"fetched_at": now, "version": SIZES_VERSION, "models": found})
            out["models"] = len(found)
            out["unsized"] = sorted(i for i, s in found.items() if "error" in s)
        if failed:
            raise failed
        return out
    finally:
        if own:
            await client.aclose()


def _age_hours(stamp: str | None) -> float:
    if not stamp:
        return float("inf")
    return (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds() / 3600


# One refresh at a time for each store in this process. A SkyPilot lookup takes about a minute alone, and several at
# once slow each other down and starve the gateway, so everyone who wants prices while one runs waits for it.
_running: dict[tuple[int, int], tuple[asyncio.Task, dict]] = {}
_failed: dict[tuple[int, int], tuple[str, datetime]] = {}  # the last refresh's error, until one works


def _key(store) -> tuple[int, int]:
    return id(asyncio.get_running_loop()), id(store)


def _covers(have: dict, want: dict) -> bool:
    """Whether a refresh already running gets everything this one asks for."""
    return all(have[k] or not want[k] for k in ("sizes", "catalog", "download"))


def _finished(key: tuple[int, int], task: asyncio.Task) -> None:
    if _running.get(key, (None,))[0] is task:
        del _running[key]
    if task.cancelled():
        return
    if (e := task.exception()) is not None:  # retrieved here, so a background refresh's error is never lost
        log.warning("GPU prices or model sizes couldn't be fetched: %s", e)
        _failed[key] = (str(e), datetime.now(UTC))
    else:
        _failed.pop(key, None)


async def refresh_once(cfg, store, sizes: bool = True, download: bool = False, catalog: bool = True,
                       wait: bool = True) -> dict | None:
    """``refresh``, joining the one already running for this store when it gets what this asks for (else after it).
    The refresh carries on if the caller goes away. ``wait=False`` starts it (or joins) and returns at once."""
    key, want = _key(store), {"sizes": sizes, "catalog": catalog, "download": download}
    while (cur := _running.get(key)) and not cur[0].done():
        if _covers(cur[1], want):
            return await asyncio.shield(cur[0]) if wait else None
        await asyncio.wait({cur[0]})
    task = asyncio.create_task(refresh(cfg, store, sizes=sizes, download=download, catalog=catalog))
    _running[key] = (task, want)
    task.add_done_callback(lambda t: _finished(key, t))
    return await asyncio.shield(task) if wait else None


def refreshing(store) -> bool:
    cur = _running.get(_key(store))
    return bool(cur and not cur[0].done())


async def settled(store) -> None:
    """Wait for this store's refresh, if one is running (tests, and anything that needs the result)."""
    if cur := _running.get(_key(store)):
        await asyncio.wait({cur[0]})


async def current(cfg, store) -> tuple[dict, dict]:
    """The stored catalog and model sizes. When they're stale or missing, a refresh starts in the background and the
    stored ones are returned at once with ``refreshing``; the report asks again once it's done. Prices come through
    SkyPilot where the gateway has it; otherwise the GPU controller publishes them (controller.publish_gpu_prices),
    and a stale catalog is reported as a problem. A refresh that failed is reported until one works, and reports
    start another only ``RETRY_MINUTES`` after it."""
    from .gpu import sky_available

    catalog = await store.get_setting("gpu_catalog") or {}
    sizes = await store.get_setting("model_sizes") or {}
    ids = {m.hf_id for m in cfg.models if m.kind == "chat" and m.hf_id}
    stale_sizes = (_age_hours(sizes.get("fetched_at")) > SIZES_MAX_AGE_HOURS or not ids <= set(sizes.get("models") or {})
                   or sizes.get("version") != SIZES_VERSION)
    stale_catalog = _age_hours(catalog.get("fetched_at")) > CATALOG_MAX_AGE_HOURS
    here = sky_available()
    failed = _failed.get(_key(store))
    just_failed = failed is not None and (datetime.now(UTC) - failed[1]).total_seconds() < RETRY_MINUTES * 60
    if ((stale_catalog and here) or stale_sizes) and not just_failed:
        await refresh_once(cfg, store, sizes=stale_sizes, catalog=stale_catalog and here, wait=False)
    problem = failed[0] if failed else None
    if stale_catalog and not here and not problem:
        problem = ("SkyPilot isn't installed where the gateway runs, so GPU prices come from the GPU controller, which "
                   + ("hasn't published any in the last 6 hours" if catalog.get("fetched_at") else "hasn't published any yet")
                   + ": run it (compose --profile gpu, or controller.enabled in the Helm chart), or install SkyPilot in the gateway's image")
    out = {**catalog, "refreshing": refreshing(store)}
    return ({**out, "problem": problem} if problem else out), sizes


# ------------------------------------------------------------------ a replica for a model

def compute_precision(acc: Accelerator, bits: int) -> str:
    """What the GPU computes weights mostly in ``bits`` in: FP8 weights at FP8 where it has FP8 tensor cores (Ada,
    Hopper, Blackwell, MI300); 4-bit weights at FP8 on Blackwell, whose FP4 kernels take 8-bit activations (vLLM's
    for MXFP4; full FP4 needs 4-bit activations too); everything else at 16 bits (Hopper's 4-bit kernels, and every
    GPU without FP8, dequantize the weights)."""
    if acc.fp8_tflops and (bits == 8 or (bits == 4 and acc.fp4)):
        return "FP8"
    return "16-bit"


def peak_tflops(acc: Accelerator, bits: int) -> float:
    """Dense compute per GPU for weights mostly in ``bits``, at the precision it computes them in."""
    return acc.fp8_tflops if compute_precision(acc, bits) == "FP8" else acc.bf16_tflops


def size_replica(size: dict, offers: list[dict], clouds: set[str], spot: bool, fit: float = FIT,
                 accelerators: str | None = None, regions: list[str] | None = None) -> dict:
    """The replica for a model: the cheapest per hour among the offers in ``clouds`` whose memory holds its weights at
    ``fit``, on 1, 2, 4 or 8 GPUs (how vLLM splits a model), the faster one on a tie. ``accelerators`` ("H100:4")
    fixes the GPU, as a deployment does. Spot prices where asked for and offered, else on-demand."""
    found = replica_options(size, offers, clouds, spot, fit, accelerators, regions)
    return found[0] if isinstance(found, list) else found


def replica_options(size: dict, offers: list[dict], clouds: set[str], spot: bool, fit: float = FIT,
                    accelerators: str | None = None, regions: list[str] | None = None) -> list[dict] | dict:
    """Every machine worth considering for a model, cheapest per hour first, each with more compute than the cheaper
    ones (a pricier machine that's no faster never helps): the cheapest takes the least traffic to pay for, a faster
    one may be the only one that pays at all. ``{error}`` when none holds the weights (see ``size_replica``)."""
    if "error" in size and not accelerators:
        return {"error": size["error"]}
    need_gb = size.get("weight_bytes", 0) / 1e9 / fit
    want = None
    if accelerators:
        name, _, n = accelerators.partition(":")
        want = (name.upper() if name.upper() in ACCELERATORS else name, int(n or 1))
    found = []
    for o in offers:
        acc = ACCELERATORS.get(o["accelerator"])
        if acc is None or o["cloud"] not in clouds or o["count"] not in (1, 2, 4, 8) \
                or (want and (o["accelerator"].upper(), o["count"]) != (want[0].upper(), want[1])):
            continue
        if not want and acc.memory_gb * o["count"] < need_gb:
            continue
        price, used_spot = (o["spot"], True) if spot and o["spot"] else (o["on_demand"], False)
        if price is None or (regions and price.get("region") not in regions):
            continue
        tflops = peak_tflops(acc, size.get("bits", 16)) * o["count"]
        found.append(((price["usd_per_hour"], -tflops), {
            "accelerator": o["accelerator"], "count": o["count"], "memory_gb": acc.memory_gb * o["count"], "tflops": tflops,
            "computes_in": compute_precision(acc, size.get("bits", 16)),
            "cloud": o["cloud"], "region": price.get("region"), "instance_type": price.get("instance_type"),
            "spot": used_spot, "usd_per_hour": price["usd_per_hour"],
            "on_demand_usd_per_hour": (o["on_demand"] or {}).get("usd_per_hour"),
            "spot_usd_per_hour": (o["spot"] or {}).get("usd_per_hour")}))
    if not found:
        largest = max((ACCELERATORS[o["accelerator"]].memory_gb * o["count"] for o in offers
                       if o["cloud"] in clouds and o["accelerator"] in ACCELERATORS and o["count"] in (1, 2, 4, 8)), default=0)
        if want:
            return {"error": f"no {accelerators} offer in {', '.join(sorted(clouds))}"}
        return {"error": f"needs {need_gb:,.0f} GB of GPU memory; the largest machine on offer has {largest:,.0f} GB"
                if largest and need_gb > largest else "no offer in the chosen clouds"}
    frontier, fastest = [], 0.0
    for _, replica in sorted(found, key=lambda x: x[0]):
        if replica["tflops"] > fastest:
            frontier.append(replica)
            fastest = replica["tflops"]
    return frontier


def capacity_tps(replica: dict, size: dict, efficiency: float, cache_read_share: float) -> float:
    """Estimated tokens per second one replica serves: ``efficiency`` of its peak compute at 2 operations per active
    parameter per token, for the tokens it has to compute (cache reads are served from the prefix cache)."""
    computed = efficiency * replica["tflops"] * 1e12 / (2 * size["active_params"])
    return computed / max(1.0 - cache_read_share, 0.1)


def replica_plans(cfg, catalog: dict, sizes: dict, clouds: set[str], spot: bool, weights: str = "published",
                  replica_prices: dict[str, float] | None = None) -> dict[str, dict]:
    """Per open-weight chat model, the replica its break-even is worked out for: its own deployment's (its GPUs,
    clouds and spot setting, priced from the catalog or at the controller's current price, and its configured
    capacity), else one sized to it, with its weights served as ``weights`` (see ``quantized``). Each is
    {deployment, replica, size, capacity_tps} or {error}; a sized one also has ``options`` (``replica_options``),
    for the break-even to pick the machine that pays from the least traffic."""
    from .gpu import deployment_for

    offers = catalog.get("offers") or []
    out: dict[str, dict] = {}
    for m in cfg.models:
        if m.kind != "chat" or not m.hf_id:
            continue
        options = None
        size = (sizes.get("models") or {}).get(m.hf_id) or {"error": "its size hasn't been fetched yet"}
        size = quantized(size, weights)
        dep = deployment_for(cfg, m)
        sized = "error" not in size
        if dep and dep.serve and dep.serve.provisioner == "skypilot":
            s = dep.serve
            replica = size_replica(size, offers, {c.lower() for c in s.clouds} or clouds, s.use_spot, accelerators=s.accelerators,
                                   regions=s.regions or None)
            if "error" in replica:  # not in the catalog: the configured price
                replica = {"accelerator": s.accelerators.partition(":")[0], "count": int(s.accelerators.partition(":")[2] or 1),
                           "usd_per_hour": dep.gpu_cost_per_hour, "spot": s.use_spot, "source": "configured"}
            else:
                replica["source"] = "catalog"
        elif dep and dep.serve and dep.serve.provisioner == "kubernetes":  # your cluster: its node type's configured price
            from .gpu import gpu_count
            from .kube import expected_node_price

            s, node = dep.serve, expected_node_price(cfg, dep)
            replica = {"accelerator": s.accelerators.partition(":")[0], "count": gpu_count(s.accelerators), "cloud": "kubernetes",
                       **({"usd_per_hour": node["usd_per_hour"], "instance_type": node["node_type"], "spot": node["spot"], "source": "node_type"}
                          if node else {"usd_per_hour": dep.gpu_cost_per_hour, "spot": s.use_spot, "source": "configured"})}
        elif dep:  # a server you run: its configured price
            replica = {"usd_per_hour": dep.gpu_cost_per_hour, "source": "configured"}
        else:
            found = replica_options(size, offers, clouds, spot)
            if isinstance(found, dict):
                out[m.name] = {"error": found["error"], "size": size if sized else None}
                continue
            for r in found:
                r["source"] = "catalog"
            replica, options = found[0], found
        if dep and (replica_prices or {}).get(dep.name):  # what the controller is paying, or would
            replica = {**replica, "usd_per_hour": replica_prices[dep.name], "source": "controller"}
        configured = dep.capacity_tokens_per_second if dep and dep.capacity_tokens_per_second else None
        if not replica.get("usd_per_hour") or not (configured or (sized and replica.get("tflops"))):
            out[m.name] = {"error": "no price or capacity for its deployment", "size": size if sized else None}
            continue
        out[m.name] = {"deployment": dep.name if dep else None, "replica": replica, "size": size if sized else None, "capacity_tps": configured,
                       **({"options": options} if options else {})}
    return out
