"""Checks the Kubernetes GPU provisioner on the kind cluster scripts/kind/setup.sh made, through Merit Order's own API: the
controller in the cluster starts, scales, drains and stops each deployment as an admin's GPU controls ask, and gives up on
the ones that can't start (a missing Secret, an image that isn't there, a crash loop, a quota) with the reason, and waits for a
node that doesn't come. Prints each step and what Kubernetes and Merit Order showed.

    KUBECONFIG=/tmp/meritorder-kind.kubeconfig uv run python scripts/kind/check.py [--only gpu-kserve]

Nothing here is mocked except the vLLM image (scripts/fake_vllm.py) and the GPUs (the node advertises nvidia.com/gpu).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time

import httpx

KEY = "mo-kind-check-key-not-a-secret"  # MERITORDER_DEV_KEY in scripts/kind/values.yaml
GPUS = "gpus"
FAILURES = {  # backend -> what its start should fail on
    "gpu-no-secret": "CreateContainerConfigError",
    "gpu-bad-image": "can't be pulled",
    "gpu-crash": "keeps crashing",
}
failed: list[str] = []


def kubectl(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kubectl", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout if r.returncode == 0 else ""


def kjson(*args: str) -> dict | None:
    out = kubectl(*args, "-o", "json", check=False)
    return json.loads(out) if out else None


def step(ok: bool, what: str, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        failed.append(what)


class Gateway:
    def __init__(self, url: str):
        self.c = httpx.Client(base_url=url, headers={"authorization": f"Bearer {KEY}"}, timeout=30)

    def control(self, backend: str, **body) -> None:
        r = self.c.post(f"/v1/admin/gpu/{backend}/control", json=body)
        r.raise_for_status()

    def deployment(self, backend: str) -> dict:
        r = self.c.get("/v1/reports/gpu")
        r.raise_for_status()
        return next(d for d in r.json()["deployments"] if d["backend"] == backend)

    def wait(self, backend: str, done, timeout: float, what: str) -> tuple[dict, float, list[str]]:
        """Poll the deployment until done(record) or the timeout; returns the record, the seconds it took and every
        distinct message on the way (what the console's card showed)."""
        t0, seen = time.monotonic(), []
        while True:
            d = self.deployment(backend)
            for msg in (d.get("status_message"), d.get("reason")):
                if msg and msg not in seen:
                    seen.append(msg)
            if done(d):
                return d, time.monotonic() - t0, seen
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"{backend}: {what} after {timeout:.0f}s; last state {d.get('state')}: {seen[-3:]}")
            time.sleep(2)

    def chat(self, backend: str) -> httpx.Response:
        """A request pinned to the deployment. Gateways see the controller's states within 5 seconds; until then it's a
        503 that says the deployment isn't serving yet."""
        for _ in range(10):
            r = self.c.post("/v1/chat/completions", headers={"x-meritorder-backend": backend},
                            json={"model": "tiny-coder", "messages": [{"role": "user", "content": f"hello {backend}"}]})
            if r.status_code != 503:
                return r
            time.sleep(1)
        return r


def ready(n: int):
    return lambda d: d.get("state") == "warm" and sum(1 for r in d.get("replicas_info") or [] if r.get("status") == "READY") == n


def resource(backend: str, api: str) -> dict | None:
    kind = "inferenceservice" if api == "kserve" else "deployment"
    return kjson("-n", GPUS, "get", kind, f"meritorder-{backend}")


def lifecycle(gw: Gateway, backend: str, api: str) -> None:
    print(f"--- {backend} ({api})", flush=True)
    gw.control(backend, mode="on", replicas=1, reason="kind check: up")
    d, secs, seen = gw.wait(backend, ready(1), 600, "not warm")
    step(True, f"up: warm in {secs:.0f}s", " -> ".join(seen))
    obj = resource(backend, api)
    step(obj is not None, f"the {('InferenceService' if api == 'kserve' else 'Deployment')} exists")
    if api == "kserve":
        ann = obj["metadata"].get("annotations") or {}
        pred = obj["spec"]["predictor"]
        step(ann.get("serving.kserve.io/autoscalerClass") == "none" and pred.get("minReplicas") == pred.get("maxReplicas") == 1,
             "Standard mode, KServe's autoscaler off, 1 replica", json.dumps({k: v for k, v in ann.items() if "kserve" in k}))
        url = (obj.get("status") or {}).get("address", {}).get("url")
        svc = kjson("-n", GPUS, "get", "service", f"meritorder-{backend}-predictor")
        ports = [(p.get("port"), p.get("targetPort")) for p in (svc or {}).get("spec", {}).get("ports", [])]
        step(bool(url) and bool(svc), "status.address.url and the predictor Service", f"{url}; ports {ports}")
        deps = kjson("-n", GPUS, "get", "deployments", "-l", f"serving.kserve.io/inferenceservice=meritorder-{backend}")
        step(bool(deps and deps["items"]), "the predictor Deployment carries serving.kserve.io/inferenceservice",
             ", ".join(i["metadata"]["name"] for i in (deps or {}).get("items", [])))
    pods = kjson("-n", GPUS, "get", "pods", "-l", f"meritorder.dev/deployment=meritorder-{backend}")
    step(bool(pods and pods["items"]), "pods carry Merit Order's labels" + (" (through spec.predictor.labels)" if api == "kserve" else ""),
         ", ".join(p["metadata"]["name"] for p in (pods or {}).get("items", [])))
    if api == "kserve":  # what scripts/kind/meritorder.yaml asks for, not KServe's defaults (1 CPU, 2 GiB)
        res = pods["items"][0]["spec"]["containers"][0]["resources"] if pods and pods["items"] else {}
        step(res.get("limits", {}).get("memory") == "128Mi" and res.get("limits", {}).get("cpu") == "100m",
             "the pod has the configured cpu and memory", json.dumps(res))
    if api == "llm-d":
        labels = pods["items"][0]["metadata"]["labels"] if pods and pods["items"] else {}
        step(labels.get("app") == "tiny-coder", "pods carry the InferencePool's selector labels", f"app={labels.get('app')}")
    rep = (d.get("replicas_info") or [{}])[0]
    step(bool(rep.get("endpoint")) and rep.get("instance_type") == "g6.12xlarge" and rep.get("zone") == "kind-a",
         "the replica's pod address, node type and zone", f"{rep.get('endpoint')}, {rep.get('instance_type')}, {rep.get('zone')}")
    step(abs((rep.get("usd_per_hour") or 0) - 1.15) < 1e-6, "priced at its share of the node: 1 of 4 GPUs of a $4.60 node",
         f"${rep.get('usd_per_hour')}/h")
    step(bool(d.get("endpoint")), "the endpoint gateways use", str(d.get("endpoint")))
    replies = [gw.chat(backend) for _ in range(3)]
    r = replies[-1]
    step(all(x.status_code == 200 for x in replies) and r.headers.get("x-meritorder-backend") == backend and "You said" in r.text,
         "requests through the gateway reach vLLM", r.text[:120])
    d, secs, _ = gw.wait(backend, lambda d: any(x.get("utilization") is not None for x in d.get("replicas_info") or []), 180,
                         "no load read from vLLM's metrics")
    step(True, f"utilization read from each pod's /metrics in {secs:.0f}s",
         json.dumps([{k: x.get(k) for k in ("id", "utilization", "running", "waiting")} for x in d.get("replicas_info") or []]))

    gw.control(backend, mode="on", replicas=2, reason="kind check: scale")
    d, secs, _ = gw.wait(backend, ready(2), 600, "not 2 replicas")
    obj = resource(backend, api)
    n = obj["spec"]["predictor"]["minReplicas"] if api == "kserve" else obj["spec"]["replicas"]
    step(n == 2, f"scale: 2 ready replicas in {secs:.0f}s", ", ".join(r.get("id") or "?" for r in d.get("replicas_info") or []))

    gw.control(backend, mode="on", replicas=1, reason="kind check: scale down")
    d, secs, _ = gw.wait(backend, ready(1), 300, "not back to 1 replica")
    step(True, f"scale down: 1 replica in {secs:.0f}s")

    gw.control(backend, mode="off", drain=True, reason="kind check: drain")
    d, secs, _ = gw.wait(backend, lambda d: d.get("state") == "draining", 120, "not draining")
    step(resource(backend, api) is not None, f"drain: draining in {secs:.0f}s, pods still serve", d.get("reason") or "")
    d, secs, _ = gw.wait(backend, lambda d: d.get("state") == "api_only", 300, "not stopped after draining")
    gone = _gone(backend, api)
    step(gone, f"down after draining: stopped in {secs:.0f}s and the resource deleted", d.get("reason") or "")

    gw.control(backend, mode="on", replicas=1, reason="kind check: up again")
    gw.wait(backend, ready(1), 600, "not warm again")
    gw.control(backend, mode="off", drain=False, reason="kind check: stop now")
    d, secs, _ = gw.wait(backend, lambda d: d.get("state") == "api_only", 120, "not stopped")
    step(_gone(backend, api), f"down now: stopped in {secs:.0f}s and the resource deleted", d.get("reason") or "")
    gw.control(backend, mode="auto")


def _gone(backend: str, api: str, timeout: float = 60) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if resource(backend, api) is None:
            return True
        time.sleep(2)
    return False


def failure(gw: Gateway, backend: str, expect: str) -> None:
    print(f"--- {backend}: should fail with {expect}", flush=True)
    before = gw.deployment(backend).get("since")  # a failure from an earlier run doesn't count
    gw.control(backend, mode="on", replicas=1, reason="kind check: a start that fails")
    d, secs, seen = gw.wait(backend, lambda d: d.get("since") != before and d.get("state") == "api_only"
                            and "failed" in (d.get("reason") or ""), 600, "didn't fail")
    step(expect in (d.get("reason") or ""), f"failed in {secs:.0f}s with the reason", d.get("reason") or "")
    api = "kserve" if backend != "gpu-crash" else "llm-d"
    step(_gone(backend, api), "the resource was deleted", " -> ".join(seen))
    time.sleep(25)  # still pinned: a few more controller cycles
    later = gw.deployment(backend)
    step(later.get("since") == d.get("since") and resource(backend, api) is None, "still pinned, it waits out its cooldown",
         later.get("note") or "")
    gw.control(backend, mode="auto")


def waits_for_a_node(gw: Gateway) -> None:
    """More GPUs than any node has: the pod stays Pending, and the controller waits (a cluster autoscaler may add a node)."""
    backend = "gpu-too-big"
    print(f"--- {backend}: should wait for a node", flush=True)
    gw.control(backend, mode="on", replicas=1, reason="kind check: no node fits")
    d, secs, seen = gw.wait(backend, lambda d: "waiting for a node" in (d.get("status_message") or ""), 120, "not waiting for a node")
    time.sleep(25)  # a few more controller cycles: still waiting, not failed
    d = gw.deployment(backend)
    step(d.get("state") == "provisioning", f"waits for a node (seen in {secs:.0f}s, still waiting 25s later)", d.get("status_message") or "")
    gw.control(backend, mode="off", drain=False, reason="kind check: stop waiting")
    gw.wait(backend, lambda d: d.get("state") == "api_only", 120, "not stopped")
    step(_gone(backend, "kserve"), "stopped, and the resource deleted", " -> ".join(seen))
    gw.control(backend, mode="auto")


QUOTA = {"apiVersion": "v1", "kind": "ResourceQuota", "metadata": {"name": "no-gpus", "namespace": GPUS},
         "spec": {"hard": {"requests.nvidia.com/gpu": "0"}}}


def refused_by_quota(gw: Gateway) -> None:
    """A ResourceQuota that allows no GPUs: the predictor Deployment reports ReplicaFailure, and the start fails at once."""
    backend = "gpu-kserve"
    print("--- a GPU quota of 0 in the namespace: the start should fail", flush=True)
    subprocess.run(["kubectl", "apply", "-f", "-"], input=json.dumps(QUOTA), text=True, check=True, capture_output=True)
    try:
        before = gw.deployment(backend).get("since")
        gw.control(backend, mode="on", replicas=1, reason="kind check: over quota")
        d, secs, seen = gw.wait(backend, lambda d: d.get("since") != before and d.get("state") == "api_only"
                                and "failed" in (d.get("reason") or ""), 300, "didn't fail")
        reason = d.get("reason") or ""
        step("pods can't be created" in reason and "quota" in reason, f"failed in {secs:.0f}s with the reason", reason)
        step(_gone(backend, "kserve"), "the resource was deleted", " -> ".join(seen))
    finally:
        kubectl("-n", GPUS, "delete", "resourcequota", "no-gpus", check=False)
        gw.control(backend, mode="auto")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", nargs="*", default=None, help="backends to check (default: all)")
    p.add_argument("--port", type=int, default=0, help="local port for the gateway (default: a free one)")
    args = p.parse_args()
    port = args.port or free_port()
    pf = subprocess.Popen(["kubectl", "-n", "meritorder", "port-forward", "svc/mo-meritorder", f"{port}:80"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                if httpx.get(f"{url}/healthz", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        gw = Gateway(url)
        want = lambda b: args.only is None or b in args.only  # noqa: E731
        for backend, api in (("gpu-kserve", "kserve"), ("gpu-llmd", "llm-d")):
            if want(backend):
                lifecycle(gw, backend, api)
        for backend, expect in FAILURES.items():
            if want(backend):
                failure(gw, backend, expect)
        if want("gpu-too-big"):
            waits_for_a_node(gw)
        if want("quota"):
            refused_by_quota(gw)
    finally:
        pf.terminate()
    print(f"--- {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
