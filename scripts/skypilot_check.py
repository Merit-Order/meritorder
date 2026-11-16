"""A real cycle of the SkyPilot provisioner on a real cloud, through Merit Order's own API. This one spends money.

Against a running gateway and controller, with one small model (the shipped config's `gpu-small`: gpt-oss-20b) on the
cheapest GPU that holds it on the cloud you name (`--cloud`, any cloud a SkyPilot replica can run on; Lambda by default,
where that's one A10 at about $1.29 an hour). The GPU is picked from SkyPilot's catalog, as the controller sees it, for
the model's size on Hugging Face (`--accelerators` names one instead); the deployment must be set to that GPU and cloud,
and the script says how when it isn't. Then it:

1. starts one replica (an admin pin) and waits until it's warm; if the controller gives up on it, prints why each replica
   failed (the NVIDIA driver and CUDA versions, and vLLM's last output, from the GPU activity log) and stops there;
2. sends a few requests through the gateway, pinned to that deployment;
3. checks that the replica refuses a request without the deployment's key, and with a wrong one;
4. scales to zero and checks that the cluster is gone from SkyPilot, and from the cloud's own list of instances (asked
   through SkyPilot's provisioner for that cloud, where SkyPilot keeps what it needs for that: a local API server);
5. starts one again, then stops the controller (so nothing renews the lease), unpins it, and checks the same way that
   the machine deletes itself within the lease plus autostop;
6. always, in a `finally`: unpins it, starts the controller again if it stopped it, and tears down any cluster of the
   deployment that's still there, straight through SkyPilot.

It prints the time and an estimated cost at each step. It asks SkyPilot about the clusters as the controller sees them
(the same ~/.sky and credentials), so it can list them and, if it must, tear them down itself: by default in the
Compose controller's container (`docker compose exec`, or `docker compose run` of the same service while it's
stopped); `--sky local` uses this machine's SkyPilot, for a controller that runs here. From the checkout, with the
stack up (`docker compose --profile gpu up -d`), `MERITORDER_SKYPILOT_SMALL=true` in .env, and the controller's
SkyPilot volumes uncommented in compose.yaml:

    uv run python scripts/skypilot_check.py --gateway http://localhost:8080 --key "$ADMIN_KEY" --yes-spend

The controller is stopped and started with `docker compose stop/start controller` (change it with --stop-controller
and --start-controller, e.g. `pkill -STOP -f "meritorder controller"`, which also stands in for a laptop that sleeps).
Nothing runs without --yes-spend.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

import httpx

failed: list[str] = []
T0 = time.monotonic()


def step(ok: bool, what: str, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        failed.append(what)


class Spend:
    """What the run has cost so far: time with a machine up, at the deployment's price."""

    def __init__(self, usd_per_hour: float):
        self.usd_per_hour, self.up_since, self.seconds = usd_per_hour, None, 0.0

    def machine(self, up: bool) -> None:
        now = time.monotonic()
        if self.up_since is not None:
            self.seconds += now - self.up_since
        self.up_since = now if up else None

    def total(self) -> float:
        extra = time.monotonic() - self.up_since if self.up_since is not None else 0.0
        return (self.seconds + extra) / 3600 * self.usd_per_hour

    def line(self, what: str) -> None:
        print(f"[{(time.monotonic() - T0) / 60:5.1f} min, about ${self.total():.2f} so far] {what}", flush=True)


class Gateway:
    def __init__(self, url: str, key: str):
        self.c = httpx.Client(base_url=url, headers={"authorization": f"Bearer {key}"}, timeout=60)

    def control(self, backend: str, **body) -> None:
        self.c.post(f"/v1/admin/gpu/{backend}/control", json=body).raise_for_status()

    def report(self) -> dict:
        r = self.c.get("/v1/reports/gpu")
        r.raise_for_status()
        return r.json()

    def deployment(self, backend: str) -> dict:
        found = next((d for d in self.report()["deployments"] if d["backend"] == backend), None)
        if found is None:
            raise SystemExit(f"no deployment {backend!r} on this gateway")
        return found

    def wait(self, backend: str, done, timeout: float, what: str, spend: Spend) -> dict:
        t0, seen = time.monotonic(), []
        while True:
            d = self.deployment(backend)
            for msg in (d.get("status_message"), d.get("reason")):
                if msg and msg not in seen:
                    seen.append(msg)
                    spend.line(f"  {msg}")
            if done(d):
                return d
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"{backend}: {what} after {timeout / 60:.0f} min; state {d.get('state')}")
            time.sleep(15)

    def failures(self, backend: str, since: float) -> list[dict]:
        """The deployment's failed replicas since ``since`` (unix seconds), oldest first: why, the driver and CUDA versions,
        and vLLM's last output, from the GPU activity log."""
        r = self.c.get("/v1/admin/gpu/events", params={"limit": 50, "backend": backend})
        r.raise_for_status()
        events = [e for e in r.json().get("events") or [] if e.get("kind") == "replica_failed"]
        return [e for e in reversed(events) if datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00")).timestamp() >= since]

    def chat(self, backend: str, model: str) -> httpx.Response:
        return self.c.post("/v1/chat/completions", headers={"x-meritorder-backend": backend}, timeout=300,
                           json={"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": "Say hello in five words."}]})


def ready(d: dict) -> bool:
    return d.get("state") == "warm" and any(r.get("status") == "READY" for r in d.get("replicas_info") or [])


FAILURE_COOLDOWN = 1800  # seconds a failed start cools down (controller.py, gpu.py)


def warm_or_given_up(since: float):
    """Done when warm, or when the controller has given up on the start: back to API only after it was seen starting,
    or with a failure cooldown set since ``since`` (a start that failed within one poll)."""
    started = {"yes": False}

    def done(d: dict) -> bool:
        started["yes"] = started["yes"] or d.get("state") != "api_only"
        # A failed start cools down for FAILURE_COOLDOWN from when it failed: one that began before this start is an
        # earlier run's (whose cooldown a pin overrides), not this one's.
        new_failure = (d.get("cooldown_until") or 0) - FAILURE_COOLDOWN >= since - 5
        failed = d.get("state") == "api_only" and (started["yes"] or new_failure)
        return ready(d) or failed

    return done


def show_startup(rep: dict) -> None:
    """How long the replica's start took, phase by phase (the controller's and the gate's marks)."""
    s = rep.get("startup") or {}
    if not s.get("phases"):
        print("  start phases: not known (the controller didn't launch this replica itself, or the gate is older)", flush=True)
        return
    total = s.get("total_seconds")
    print(f"  start: {total / 60:.1f} min from the launch request to ready" if total is not None else "  start:", flush=True)
    for p in s["phases"]:
        print(f"    {p['phase']:<34} {p['seconds'] / 60:5.1f} min", flush=True)
    said = [f"downloaded in {s['download_seconds']:.0f}s" if s.get("download_seconds") is not None else None,
            f"loaded in {s['load_seconds']:.0f}s" if s.get("load_seconds") is not None else None]
    if any(said):
        print("    vLLM says the weights " + " and ".join(x for x in said if x), flush=True)


def show_failures(gw: Gateway, backend: str, since: float) -> None:
    """Print why each replica failed, with its driver and CUDA versions and vLLM's last output."""
    try:
        found = gw.failures(backend, since)
    except Exception as e:  # noqa: BLE001 (the failure itself is already reported)
        print(f"  ---- the GPU activity log can't be read: {e}", flush=True)
        return
    if not found:
        print("  ---- no failed replica in the GPU activity log (`sky logs <cluster>` may still have it)", flush=True)
    for e in found:
        x = e.get("detail") or {}
        print(f"  {x.get('cluster')}: {x.get('reason')}", flush=True)
        print(f"    NVIDIA driver {x.get('driver_version')}, CUDA {x.get('cuda_version')} at most; image CUDA {x.get('image_cuda')}"
              + (f"; GPUs {', '.join(x['gpus'])}" if x.get("gpus") else ""), flush=True)
        tail = x.get("tail") or []
        print(f"    its last {len(tail)} lines ({x.get('source')}):" if tail else f"    no output: {x.get('logs_error') or 'none read'}", flush=True)
        for line in tail:
            print(f"    | {line}", flush=True)


# Runs in SkyPilot's Python, like the controller's calls: remembers where a cluster runs on its cloud (in a file only
# SkyPilot's user can read, next to its state: a provider config can hold a registry password), then asks that cloud's
# own API, through SkyPilot's provisioner for it, for the cluster's instances that aren't terminated.
CLOUD_SCRIPT = r'''
import json, os, sys
REFS = os.path.expanduser("~/.sky/meritorder-check")


def remember(req):
    from sky import global_user_state
    handle = global_user_state.get_handle_from_cluster_name(req["cluster"])
    if handle is None:
        raise ValueError(f"SkyPilot here has no record of {req['cluster']} (a remote API server keeps it)")
    ref = {"cloud": repr(handle.launched_resources.cloud), "cluster": req["cluster"], "on_cloud": handle.cluster_name_on_cloud,
           "provider": global_user_state.get_cluster_yaml_dict(handle.cluster_yaml)["provider"]}
    os.makedirs(REFS, mode=0o700, exist_ok=True)
    with open(os.open(os.path.join(REFS, req["cluster"] + ".json"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(ref, f)
    return {"cloud": ref["cloud"], "on_cloud": ref["on_cloud"]}


def instances(req):
    from sky import provision
    with open(os.path.join(REFS, req["cluster"] + ".json")) as f:
        ref = json.load(f)
    found = provision.query_instances(ref["cloud"], ref["cluster"], ref["on_cloud"], ref["provider"], non_terminated_only=True)
    return {"cloud": ref["cloud"], "instances": {k: getattr(v[0], "value", str(v[0])) for k, v in found.items()}}


def forget(req):
    try:
        os.remove(os.path.join(REFS, req["cluster"] + ".json"))
    except FileNotFoundError:
        pass
    return {}


if __name__ == "__main__":
    req = json.loads(sys.argv[1])
    try:
        print("MERITORDER_RESULT " + json.dumps({"remember": remember, "instances": instances, "forget": forget}[req["op"]](req)))
    except Exception as e:
        print("MERITORDER_RESULT " + json.dumps({"error": f"{type(e).__name__}: {e}"[-1000:]}))
        sys.exit(1)
'''


class Sky:
    """SkyPilot as the controller sees it: the same ~/.sky and credentials. ``compose``: in the controller's container
    (`docker compose exec`), or, while it's stopped, in a one-off container of the same service with the same volumes
    (`docker compose run`). ``local``: this machine's SkyPilot, for a controller that runs here."""

    def __init__(self, where: str):
        self.where, self.controller_up = where, True

    def run(self, script: str, req: dict, marker: str | None = "MERITORDER_RESULT ", env: dict | None = None):
        """Run one of our scripts in SkyPilot's Python, where the controller's SkyPilot is, and read its answer: the last
        line starting with ``marker`` (None: the last line)."""
        import json

        from meritorder import gpu

        if self.where == "local":
            base = [gpu.sky_python() or sys.executable]
        else:
            base = ["docker", "compose", "exec", "-T", "controller"] if self.controller_up \
                else ["docker", "compose", "run", "--rm", "--no-deps", "-T", "controller"]
            base.append("/opt/skypilot/bin/python")
        out = subprocess.run([*base, "-c", script, json.dumps(req)], capture_output=True, text=True, timeout=900, env=env)
        lines = [x for x in out.stdout.splitlines() if marker is None or x.startswith(marker)]
        result = json.loads(lines[-1][len(marker or ""):]) if lines else None
        if out.returncode or not lines or (isinstance(result, dict) and "error" in result):
            raise RuntimeError((result or {}).get("error") if isinstance(result, dict) else out.stderr.strip()[-500:])
        return result

    def call(self, req: dict):
        from meritorder.skypilot import _SKY_SCRIPT

        return self.run(_SKY_SCRIPT, req)

    def mine(self, prefix: str) -> list[str]:
        return [c["name"] for c in self.call({"op": "clusters", "prefix": prefix, "refresh": True})]

    def down(self, name: str) -> None:
        self.call({"op": "down", "cluster": name})

    def offers(self, cloud: str) -> list[dict]:
        """Every GPU offer on this cloud from SkyPilot's catalog (no credentials needed)."""
        from meritorder import gpu

        env = gpu._catalog_env() if self.where == "local" else None
        return (self.run(gpu._CATALOG_SCRIPT, {"op": "offers", "clouds": [cloud]}, marker=None, env=env) or {}).get("offers") or []

    def remember(self, cluster: str) -> dict:
        return self.run(CLOUD_SCRIPT, {"op": "remember", "cluster": cluster})

    def instances(self, cluster: str) -> dict:
        return self.run(CLOUD_SCRIPT, {"op": "instances", "cluster": cluster})

    def forget(self, cluster: str) -> None:
        self.run(CLOUD_SCRIPT, {"op": "forget", "cluster": cluster})


def cheapest_fit(gw: Gateway, sky: Sky, model: str, cloud: str, spot: bool) -> dict:
    """The cheapest GPU on ``cloud`` that holds ``model``'s weights with room for its KV cache (gpu_sizing.size_replica),
    from its size on Hugging Face and SkyPilot's catalog. {accelerators, usd_per_hour, instance_type, region} or {error}."""
    from meritorder import gpu_sizing

    r = gw.c.get("/v1/reports/gpu/candidates")
    r.raise_for_status()
    hf_id = next((m["hf_id"] for m in r.json()["models"] if m["id"] == model), None)
    if not hf_id:
        return {"error": f"{model} has no hf_id on this gateway, so its size isn't known: name the GPU with --accelerators"}

    async def size() -> dict:
        async with httpx.AsyncClient(timeout=60) as c:
            return await gpu_sizing.fetch_model_size(hf_id, c)

    found = gpu_sizing.size_replica(asyncio.run(size()), sky.offers(cloud), {cloud}, spot)
    if "error" in found:
        return {"error": f"{model} on {cloud}: {found['error']}"}
    return {"accelerators": f"{found['accelerator']}:{found['count']}", "usd_per_hour": found["usd_per_hour"],
            "instance_type": found.get("instance_type"), "region": found.get("region"), "spot": found.get("spot")}


def gone_from_cloud(sky: Sky, cluster: str | None) -> None:
    """Step: the cloud's own list of instances has none of the cluster's left (where SkyPilot can ask it from here)."""
    if not cluster:
        return
    try:
        found = sky.instances(cluster)
    except Exception as e:  # noqa: BLE001 (a remote API server, or a cloud SkyPilot can't list: said, not failed)
        print(f"  ---- the cloud's own list of instances can't be checked from here: {e}", flush=True)
        return
    left = found.get("instances") or {}
    step(not left, f"no instance of {cluster} left in {found.get('cloud')}'s own list",
         ", ".join(f"{k} ({v})" for k, v in left.items()) or "none")


def run(cmd: str) -> None:
    print(f"  $ {cmd}", flush=True)
    subprocess.run(shlex.split(cmd), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default=os.environ.get("MERITORDER_URL", "http://localhost:8080"))
    ap.add_argument("--key", default=os.environ.get("MERITORDER_API_KEY"), help="an admin's API key")
    ap.add_argument("--backend", default="gpu-small", help="the SkyPilot deployment to cycle (default gpu-small)")
    ap.add_argument("--cloud", default="lambda", help="the cloud it must be set to launch on (default lambda): any SkyPilot cloud a "
                                                      "replica can run on (docs/05-gpu-autoscaling.md#which-clouds)")
    ap.add_argument("--accelerators", help="the GPU it must use (default: the cheapest that holds the model on that cloud, from "
                                           "SkyPilot's catalog, e.g. A10:1 on Lambda)")
    ap.add_argument("--stop-controller", default="docker compose stop controller")
    ap.add_argument("--start-controller", default="docker compose start controller")
    ap.add_argument("--sky", choices=["compose", "local"], default="compose",
                    help="where to ask SkyPilot: in the compose controller's container (default), or this machine's SkyPilot")
    ap.add_argument("--skip-death", action="store_true", help="skip step 5 (the lease running out)")
    ap.add_argument("--yes-spend", action="store_true", help="required: this starts real GPUs, twice")
    args = ap.parse_args()
    if not args.key:
        raise SystemExit("an admin API key is needed: --key or MERITORDER_API_KEY")
    from meritorder import sky_clouds

    found_cloud = sky_clouds.cloud(args.cloud)
    if found_cloud is None or found_cloud.unusable:
        raise SystemExit(f"{args.cloud}: {found_cloud.unusable if found_cloud else 'not a cloud SkyPilot 0.14.2 has'}")
    cloud = found_cloud.name
    gw = Gateway(args.gateway, args.key)
    d = gw.deployment(args.backend)
    spec = d.get("spec") or {}
    if d.get("provisioner") != "skypilot" or not d.get("enabled"):
        raise SystemExit(f"{args.backend} isn't an enabled SkyPilot deployment (set MERITORDER_SKYPILOT_SMALL=true for gpu-small)")
    sky = Sky(args.sky)
    model = d.get("model")
    pick = {"accelerators": args.accelerators} if args.accelerators else cheapest_fit(gw, sky, model, cloud, bool(spec.get("use_spot")))
    if "error" in pick:
        raise SystemExit(pick["error"])
    want = pick["accelerators"]
    if pick.get("usd_per_hour"):
        print(f"The cheapest GPU on {cloud} that holds {model}: {want} ({pick.get('instance_type')}, {pick.get('region')}) at "
              f"${pick['usd_per_hour']:.2f}/h{' spot' if pick.get('spot') else ''}, SkyPilot's catalog price.")
    if [c.lower() for c in spec.get("clouds") or []] != [cloud] or spec.get("accelerators") != want:
        price = pick.get("usd_per_hour")
        raise SystemExit(f"{args.backend} runs {spec.get('accelerators')} on {spec.get('clouds')}; for this check set its serve section to\n"
                         f"  accelerators: \"{want}\"\n  clouds: [{cloud}]\nand restart the stack"
                         + (f" (one replica: about ${price:.2f} an hour)" if price else ""))
    report = gw.report()
    if not report["controller"]["alive"]:
        raise SystemExit("no controller is running: start it first (docker compose --profile gpu up -d)")
    price = (d.get("price") or {}).get("usd_per_hour") or pick.get("usd_per_hour") or d.get("gpu_usd_per_hour") or 0.0
    lease, autostop = spec.get("lease_minutes") or 15, spec.get("autostop_minutes") or 5
    worst = (45 + 5 + 45 + lease + autostop + 10) / 60 * price
    print(f"{args.backend}: {model} on {want} in {cloud} at ${price:.2f}/h ({(d.get('price') or {}).get('source', 'catalog')}).")
    print(f"Two starts and a lease of {lease} min + autostop {autostop} min: usually 45 to 75 minutes of one machine, "
          f"about ${price * 0.75:.2f} to ${price * 1.25:.2f}; at most about ${worst:.2f} if every wait runs to its timeout.")
    for w in sky_clouds.warnings([cloud], bool(spec.get("use_spot"))):
        print(f"  note: {w}")
    if not args.yes_spend:
        print("Nothing started: run again with --yes-spend to spend it.")
        return 2

    spend = Spend(price)
    started = time.time()
    prefix = cluster = None
    controller_stopped = False
    try:
        print("--- 1. start one replica", flush=True)
        gw.control(args.backend, mode="on", replicas=1, reason="skypilot_check: up")
        spend.machine(True)
        d = gw.wait(args.backend, warm_or_given_up(started), 45 * 60, "not warm", spend)
        if not ready(d):
            raise RuntimeError(f"provisioning failed: {d.get('reason') or d.get('status_message')}")
        rep = next(r for r in d["replicas_info"] if r.get("status") == "READY")
        cluster = rep.get("cluster") or ""
        prefix = cluster.rsplit("-", 1)[0] + "-" if cluster else None
        step(bool(cluster), "warm", f"{cluster} in {rep.get('cloud')}/{rep.get('region')} ({rep.get('instance_type')}, "
                                    f"${rep.get('usd_per_hour')}/h), {rep.get('endpoint')}" + (" through an SSH tunnel" if rep.get("tunnel") else ""))
        remember(sky, cluster)
        show_startup(rep)
        spend.line("warm")

        print("--- 2. requests through the gateway", flush=True)
        replies = [gw.chat(args.backend, model) for _ in range(3)]
        r = replies[-1]
        step(all(x.status_code == 200 for x in replies) and r.headers.get("x-meritorder-backend") == args.backend,
             "requests pinned to the deployment reach vLLM", r.text[:160])
        spend.line("requests done")

        print("--- 3. the replica refuses callers without the key", flush=True)
        url = rep.get("endpoint") or ""
        try:
            bare = httpx.get(url.rstrip("/") + "/models", timeout=15)
            wrong = httpx.get(url.rstrip("/") + "/models", timeout=15, headers={"authorization": "Bearer wrong"})
            metrics = httpx.get(url[:-3] + "/metrics" if url.endswith("/v1") else url + "/metrics", timeout=15)
            step(bare.status_code == wrong.status_code == metrics.status_code == 401, "no key, a wrong key, /metrics: all 401",
                 f"{bare.status_code}, {wrong.status_code}, {metrics.status_code}")
        except httpx.HTTPError as e:
            step(True, "the replica isn't reachable from here at all", f"{type(e).__name__} (an SSH tunnel serves the controller)")

        print("--- 4. scale to zero", flush=True)
        gw.control(args.backend, mode="off", drain=False, reason="skypilot_check: down")
        gw.wait(args.backend, lambda d: d.get("state") == "api_only", 15 * 60, "not stopped", spend)
        t0 = time.monotonic()
        while prefix and sky.mine(prefix) and time.monotonic() - t0 < 600:
            time.sleep(15)
        left = sky.mine(prefix) if prefix else []
        spend.machine(bool(left))
        step(not left, "the cluster is gone from SkyPilot", ", ".join(left) or "none left")
        gone_from_cloud(sky, cluster)
        spend.line("stopped")

        if not args.skip_death:
            print("--- 5. the controller dies: the machine deletes itself", flush=True)
            gw.control(args.backend, mode="on", replicas=1, reason="skypilot_check: up again")
            spend.machine(True)
            again = time.time()
            d = gw.wait(args.backend, warm_or_given_up(again), 45 * 60, "not warm again", spend)
            if not ready(d):
                started = again  # only this start's failures
                raise RuntimeError(f"provisioning failed again: {d.get('reason') or d.get('status_message')}")
            rep = next((r for r in d["replicas_info"] if r.get("status") == "READY"), {})
            cluster = rep.get("cluster") or cluster
            remember(sky, cluster)
            show_startup(rep)
            spend.line("warm again; stopping the controller")
            run(args.stop_controller)
            controller_stopped, sky.controller_up = True, False
            died = time.monotonic()
            # Unpinned while the controller is away, so it doesn't start another one when it comes back.
            gw.control(args.backend, mode="off", drain=False, reason="skypilot_check: controller death")
            limit = (lease + autostop + 10) * 60
            while prefix and sky.mine(prefix) and time.monotonic() - died < limit:
                spend.line(f"  still there ({(time.monotonic() - died) / 60:.1f} min without the controller)")
                time.sleep(60)
            left = sky.mine(prefix) if prefix else []
            took = (time.monotonic() - died) / 60
            spend.machine(bool(left))
            step(not left, f"deleted itself {took:.1f} min after the controller stopped",
                 f"lease {lease} + autostop {autostop} min" + (f"; still there: {', '.join(left)}" if left else ""))
            gone_from_cloud(sky, cluster)
            spend.line("machine gone")
    except Exception as e:  # noqa: BLE001 (reported, then the finally cleans up)
        step(False, "the check stopped early", f"{type(e).__name__}: {e}")
        show_failures(gw, args.backend, started)
    finally:
        print("--- cleanup", flush=True)
        try:
            gw.control(args.backend, mode="off", drain=False, reason="skypilot_check: cleanup")
        except Exception as e:  # noqa: BLE001
            print(f"  couldn't unpin {args.backend}: {e}", flush=True)
        if controller_stopped:
            try:
                run(args.start_controller)
                sky.controller_up = True
            except Exception as e:  # noqa: BLE001
                print(f"  couldn't start the controller again: {e}", flush=True)
        if prefix:
            try:
                for name in sky.mine(prefix):
                    print(f"  tearing down {name} straight through SkyPilot", flush=True)
                    try:
                        sky.down(name)
                    except Exception as e:  # noqa: BLE001
                        print(f"  FAILED to tear down {name}: {e}. Run `sky down {name}` now.", flush=True)
                left = sky.mine(prefix)
                step(not left, "nothing of this deployment left in SkyPilot", ", ".join(left) or "none")
            except Exception as e:  # noqa: BLE001
                step(False, "listing the clusters to clean up", f"{e}: check `sky status` for {prefix}* yourself, now")
        if cluster:
            try:
                sky.forget(cluster)
            except Exception:  # noqa: BLE001 (only the reference file in ~/.sky/meritorder-check)
                pass
        spend.machine(False)
        spend.line(f"done: {len(failed)} failed" + (f" ({'; '.join(failed)})" if failed else ""))
    return 1 if failed else 0


def remember(sky: Sky, cluster: str | None) -> None:
    """Keep where the cluster runs on its cloud, to ask the cloud itself once SkyPilot says it's gone."""
    if not cluster:
        return
    try:
        ref = sky.remember(cluster)
        print(f"  {cluster} is {ref.get('on_cloud')} on {ref.get('cloud')}", flush=True)
    except Exception as e:  # noqa: BLE001 (the cloud's own list just won't be checked)
        print(f"  ---- can't keep where {cluster} runs on its cloud: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
