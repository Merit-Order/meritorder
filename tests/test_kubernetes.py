"""The Kubernetes provisioner (KServe and llm-d) and its API access, against a Kubernetes API in memory: never a real cluster."""

from __future__ import annotations

import base64
import copy
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError

from meritorder import controller, gpu, gpu_sizing, kube
from meritorder.config import BackendConfig, Config, KubernetesConfig, KubernetesServe, ModelConfig, NodeType, Route, ServeConfig
from meritorder.gpu import API_ONLY, PROVISIONING, WARM, Deployment
from meritorder.kube import ExecToken, KubeAPI, KubeError, KubernetesProvisioner
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

ISVC = "/apis/serving.kserve.io/v1beta1/namespaces/gpus/inferenceservices"
DEPLOYMENTS = "/apis/apps/v1/namespaces/gpus/deployments"
POOL_V1 = "/apis/inference.networking.k8s.io/v1/namespaces/gpus/inferencepools/qwen-pool"
POOL_ALPHA = "/apis/inference.networking.x-k8s.io/v1alpha2/namespaces/gpus/inferencepools/qwen-pool"


def api_status(code: int, reason: str, message: str) -> httpx.Response:
    return httpx.Response(code, json={"kind": "Status", "apiVersion": "v1", "status": "Failure", "reason": reason, "message": message, "code": code})


def merge(base: dict, patch: dict) -> dict:
    """JSON merge patch (RFC 7386), as the API server applies application/merge-patch+json."""
    for k, v in patch.items():
        if v is None:
            base.pop(k, None)
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


class FakeCluster:
    """A Kubernetes API in memory: objects by path, label-selected lists of pods and deployments, nodes, and faults to inject."""

    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.pods: list[dict] = []
        self.predictors: list[dict] = []  # Deployments KServe would make for an InferenceService
        self.nodes: dict[str, dict] = {}
        self.nodes_forbidden = False
        self.served = {"inferenceservices", "deployments"}  # resource types this cluster serves (CRDs installed)
        self.calls: list[tuple[str, str]] = []
        self.auth: list[str | None] = []
        self.faults: list[list] = []
        self.lingering = 0  # pod lists a deleted resource's pods are still in (their grace period)

    def fault(self, method: str, part: str, result, times: int = 1) -> None:
        self.faults.append([method, part, result, times])

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        self.auth.append(request.headers.get("authorization"))
        for f in self.faults:
            if f[3] and f[0] == method and f[1] in path:
                f[3] -= 1
                if isinstance(f[2], Exception):
                    raise f[2]
                return f[2]
        selector = request.url.params.get("labelSelector")
        if method == "GET" and selector is not None:
            if path.endswith("/pods") and self.lingering:
                self.lingering -= 1
                if not self.lingering:
                    self.pods = [p for p in self.pods if not p.get("deleted")]
            if path.rsplit("/", 1)[1] not in self.served | {"pods"}:
                return httpx.Response(404, text="404 page not found")
            wanted = [part.partition("=")[::2] for part in selector.split(",")]
            items = self.pods if path.endswith("/pods") else [o for p, o in self.objects.items() if p.startswith(path + "/")] + self.predictors
            return httpx.Response(200, json={"items": [i for i in items
                                                       if all((i["metadata"].get("labels") or {}).get(k) == v for k, v in wanted)]})
        if path.startswith("/api/v1/nodes/"):
            if self.nodes_forbidden:
                return api_status(403, "Forbidden", 'nodes is forbidden: User "system:serviceaccount:mo:mo-controller" cannot get resource "nodes"')
            node = self.nodes.get(path.rsplit("/", 1)[1])
            return httpx.Response(200, json=node) if node else api_status(404, "NotFound", "node not found")
        if method == "POST":
            if path.rsplit("/", 1)[1] not in self.served:
                return httpx.Response(404, text="404 page not found")
            body = json.loads(request.content)
            target = f"{path}/{body['metadata']['name']}"
            if target in self.objects:
                return api_status(409, "AlreadyExists", f"{body['metadata']['name']} already exists")
            body["metadata"]["resourceVersion"] = "1"
            body.setdefault("status", {})
            self.objects[target] = body
            return httpx.Response(201, json=body)
        obj = self.objects.get(path)
        if obj is None:
            return api_status(404, "NotFound", f'"{path.rsplit("/", 1)[1]}" not found')
        if method == "GET":
            return httpx.Response(200, json=obj)
        if method == "PUT":
            body = json.loads(request.content)
            if body["metadata"].get("resourceVersion") != obj["metadata"]["resourceVersion"]:
                return api_status(409, "Conflict", "the object has been modified")
            body["metadata"]["resourceVersion"] = str(int(obj["metadata"]["resourceVersion"]) + 1)
            body["status"] = obj.get("status", {})
            self.objects[path] = body
            return httpx.Response(200, json=body)
        if method == "PATCH":
            assert request.headers["content-type"] == "application/merge-patch+json"
            return httpx.Response(200, json=merge(obj, json.loads(request.content)))
        assert method == "DELETE" and request.url.params.get("propagationPolicy") == "Background"
        del self.objects[path]
        name = path.rsplit("/", 1)[1]
        mine = [p for p in self.pods if name in ((p["metadata"].get("labels") or {}).get(kube.KSERVE_LABEL),
                                                (p["metadata"].get("labels") or {}).get(kube.MANAGED))]
        for p in mine:
            p["deleted"] = True
        if not self.lingering:
            self.pods = [p for p in self.pods if p not in mine]
        return httpx.Response(200, json={"kind": "Status", "status": "Success"})


def pod(name: str, labels: dict, node: str | None = None, ready: bool = False, ip: str | None = None, waiting: str | None = None,
        message: str = "", restarts: int = 0, last: dict | None = None, unschedulable: str | None = None, port: int = 8000) -> dict:
    conditions = []
    if unschedulable:
        conditions.append({"type": "PodScheduled", "status": "False", "reason": "Unschedulable", "message": unschedulable})
    if ready:
        conditions.append({"type": "Ready", "status": "True"})
    cs = {"name": "vllm", "restartCount": restarts, "state": {"waiting": {"reason": waiting, "message": message}} if waiting else {"running": {}}}
    if last:
        cs["lastState"] = {"terminated": last}
    return {"metadata": {"name": name, "labels": labels},
            "spec": {"nodeName": node, "containers": [{"name": "vllm", "ports": [{"containerPort": port}]}]},
            "status": {"phase": "Running" if node else "Pending", "podIP": ip, "startTime": "2026-11-12T10:00:00Z",
                       "conditions": conditions, "containerStatuses": [cs] if node else []}}


def node(name: str, instance: str = "a3-highgpu-8g", spot: bool = False) -> dict:
    labels = {"node.kubernetes.io/instance-type": instance, "topology.kubernetes.io/region": "us-central1",
              "topology.kubernetes.io/zone": "us-central1-a"}
    if spot:
        labels["cloud.google.com/gke-spot"] = "true"
    return {"metadata": {"name": name, "labels": labels}}


def k8s_config(**k) -> Config:
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="org/open-model", routes=[Route(backend="gpu", model="open-model")])
    serve = ServeConfig(model="ow", provisioner="kubernetes", accelerators="H100:4", vllm_args="--max-model-len 65536",
                        kubernetes=KubernetesServe(**{"node_type": "a3-highgpu-8g", "hf_token_secret": "hf", "cpu": "32", "memory": "256Gi", **k}))
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=30.0, serve=serve)
    node_types = {"a3-highgpu-8g": NodeType(usd_per_hour=88.0, gpus=8, spot_usd_per_hour=30.0)}
    return Config(backends=[backend], models=[model], kubernetes=KubernetesConfig(namespace="gpus", node_types=node_types))


def provisioner(cfg: Config, cluster: FakeCluster) -> KubernetesProvisioner:
    async def token():
        return "sa-token"

    return KubernetesProvisioner(cfg.backends[0], cfg, KubeAPI("https://k8s.test", "default", token, transport=httpx.MockTransport(cluster.handler)))


@pytest.fixture(autouse=True)
def quick_stops(monkeypatch):
    """A stop polls for its pods to go without waiting between polls."""
    monkeypatch.setattr(kube, "DOWN_POLL_SECONDS", 0)


@pytest.fixture
def probe(monkeypatch):
    """Which endpoints answer /v1/models."""
    answering: set[str] = set()

    async def fake(endpoint):
        return endpoint in answering

    monkeypatch.setattr(gpu, "_probe", fake)
    return answering


KSERVE_URL = "http://meritorder-gpu-predictor.gpus.svc.cluster.local"


def serving(cluster: FakeCluster, name: str = "meritorder-gpu", pods: int = 1, spot: bool = False) -> None:
    """KServe has made the predictor and published its address; ``pods`` replicas are ready on their own nodes."""
    cluster.objects[f"{ISVC}/{name}"]["status"] = {"address": {"url": KSERVE_URL}, "conditions": [{"type": "Ready", "status": "True"}]}
    labels = {kube.KSERVE_LABEL: name}
    cluster.pods = [pod(f"{name}-predictor-{i}", labels, node=f"gke-gpu-{i}", ready=True, ip=f"10.0.0.{i + 1}") for i in range(pods)]
    cluster.nodes = {f"gke-gpu-{i}": node(f"gke-gpu-{i}", spot=spot) for i in range(pods)}


# ------------------------------------------------------------------ KServe


async def test_kserve_lifecycle_create_ready_scale_load_and_delete(probe, monkeypatch):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    await prov.up(2)
    isvc = cluster.objects[f"{ISVC}/meritorder-gpu"]
    assert isvc["metadata"]["annotations"] == {"serving.kserve.io/deploymentMode": "Standard", "serving.kserve.io/autoscalerClass": "none"}
    pred = isvc["spec"]["predictor"]
    assert pred["minReplicas"] == pred["maxReplicas"] == 2 and pred["labels"][kube.MANAGED] == "meritorder-gpu"
    [c] = pred["containers"]
    assert c["name"] == "kserve-container" and c["image"] == "vllm/vllm-openai:v0.32.0-cu129" and c["command"] == ["vllm"]
    assert c["args"][:2] == ["serve", "org/open-model"] and "--enable-prefix-caching" in c["args"] and c["args"][-2:] == ["--max-model-len", "65536"]
    assert c["args"][c["args"].index("--tensor-parallel-size") + 1] == "4" and c["args"][c["args"].index("--served-model-name") + 1] == "open-model"
    assert c["resources"]["limits"] == c["resources"]["requests"] == {"nvidia.com/gpu": "4", "cpu": "32", "memory": "256Gi"}
    assert {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": "hf", "key": "HF_TOKEN"}}} in c["env"]
    assert pred["nodeSelector"] == {"node.kubernetes.io/instance-type": "a3-highgpu-8g"}
    assert pred["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    assert {"name": "dshm", "emptyDir": {"medium": "Memory"}} in pred["volumes"] and pred["terminationGracePeriodSeconds"] == 120
    assert set(cluster.auth) == {"Bearer sa-token"}

    # Created, no pods yet; then one waiting for a GPU node (a cluster autoscaler may add one: not a failure).
    st = await prov.status()
    assert not st.ready and not st.failed and st.message == "waiting for InferenceService pods"
    cluster.pods = [pod("meritorder-gpu-predictor-0", {kube.KSERVE_LABEL: "meritorder-gpu"},
                        unschedulable="0/3 nodes are available: 3 Insufficient nvidia.com/gpu.")]
    st = await prov.status()
    assert not st.ready and not st.failed and "waiting for a node: 0/3 nodes are available: 3 Insufficient nvidia.com/gpu" in st.message
    assert st.replicas_info[0]["status"] == "PENDING" and st.replicas_info[0]["usd_per_hour"] == 0.0  # on no node, it costs nothing

    # Ready: two pods on nodes, KServe's address published, and it answers.
    serving(cluster, pods=2)
    address = cluster.objects[f"{ISVC}/meritorder-gpu"].pop("status")
    st = await prov.status()
    assert not st.ready and st.message == "waiting for KServe to publish the service address"
    cluster.objects[f"{ISVC}/meritorder-gpu"]["status"] = address
    probe.add(KSERVE_URL + "/v1")
    st = await prov.status()
    assert st.ready and st.endpoint == KSERVE_URL + "/v1" and st.replicas == 2 and st.message is None
    r = st.replicas_info[0]
    assert r["status"] == "READY" and r["infra"] == "gke-gpu-0" and (r["region"], r["zone"]) == ("us-central1", "us-central1-a")
    assert r["instance_type"] == "a3-highgpu-8g" and r["resources"] == "4x nvidia.com/gpu, a3-highgpu-8g" and not r["spot"]
    assert r["usd_per_hour"] == 44.0 and r["endpoint"] == "http://10.0.0.1:8000"
    assert r["launched_at"] == datetime(2026, 11, 12, 10, tzinfo=UTC).timestamp()
    assert cluster.calls.count(("GET", "/api/v1/nodes/gke-gpu-0")) == 1  # node labels are read once
    assert await prov.status() and cluster.calls.count(("GET", "/api/v1/nodes/gke-gpu-0")) == 1

    # Each pod's address, which the controller reads its metrics from and gateways can keep a session on.
    assert [gpu.replica_url(r["endpoint"]) for r in st.replicas_info] == ["http://10.0.0.1:8000/v1", "http://10.0.0.2:8000/v1"]

    await prov.scale(3)
    assert (pred := cluster.objects[f"{ISVC}/meritorder-gpu"]["spec"]["predictor"])["minReplicas"] == pred["maxReplicas"] == 3

    await prov.down()
    assert f"{ISVC}/meritorder-gpu" not in cluster.objects and prov.endpoint is None
    st = await prov.status()
    assert st.failed and "isn't in namespace gpus" in st.message  # gone while it should run: deleted outside Merit Order
    await prov.down()  # already gone: fine


async def test_kserve_prices_from_node_types(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    assert await prov.expected_price() == {"usd_per_hour": 44.0, "source": "node_type", "node_type": "a3-highgpu-8g", "node_usd_per_hour": 88.0,
                                           "node_gpus": 8, "gpus": 4, "spot": False}
    cfg.backends[0].serve.use_spot = True
    assert (await prov.expected_price())["usd_per_hour"] == 15.0
    await prov.up(1)
    serving(cluster, spot=True)
    [r] = (await prov.status()).replicas_info
    assert r["spot"] and r["usd_per_hour"] == 15.0  # a spot node, at the spot rate
    cluster.nodes["gke-gpu-0"]["metadata"]["labels"]["node.kubernetes.io/instance-type"] = "unpriced-type"
    prov._nodes.clear()
    assert (await prov.status()).replicas_info[0]["usd_per_hour"] is None  # a node type with no configured price

    # No node type, or one without a price: the controller falls back to gpu_cost_per_hour.
    plain = k8s_config(node_type=None)
    assert await provisioner(plain, FakeCluster()).expected_price() is None
    assert "nodeSelector" not in provisioner(plain, FakeCluster()).pod_spec(8000, "vllm")


async def test_nodes_forbidden_prices_at_the_configured_node_type(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    await prov.up(1)
    serving(cluster, pods=2)
    cluster.nodes_forbidden = True
    st = await prov.status()
    assert [r["usd_per_hour"] for r in st.replicas_info] == [44.0, 44.0] and st.replicas_info[0]["zone"] is None
    assert sum(1 for c in cluster.calls if c[1].startswith("/api/v1/nodes/")) == 1  # asked once, then not again
    cluster.fault("GET", "/api/v1/nodes/", httpx.Response(500, text="etcd timeout"))
    prov._nodes_forbidden = False
    assert await prov._node_labels("other") == {}  # a passing error: no labels this time


async def test_existing_resources_are_taken_over_and_missing_ones_recreated(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    await prov.up(1)
    cluster.objects[f"{ISVC}/meritorder-gpu"]["spec"]["predictor"]["minReplicas"] = 7  # changed by hand, or an older config
    await prov.up(2)  # already there (a controller restarted mid-run): replaced with the current spec
    isvc = cluster.objects[f"{ISVC}/meritorder-gpu"]
    assert isvc["spec"]["predictor"]["minReplicas"] == 2 and isvc["metadata"]["resourceVersion"] == "2"
    assert ("PUT", f"{ISVC}/meritorder-gpu") in cluster.calls

    isvc["metadata"]["deletionTimestamp"] = "2026-11-12T10:00:00Z"
    with pytest.raises(KubeError, match="still being deleted"):
        await prov.up(1)

    del cluster.objects[f"{ISVC}/meritorder-gpu"]
    await prov.scale(2)  # deleted outside Merit Order while running: made again
    assert cluster.objects[f"{ISVC}/meritorder-gpu"]["spec"]["predictor"]["maxReplicas"] == 2

    # A 409 whose object is gone by the time it's read: create again.
    cluster.fault("POST", "inferenceservices", api_status(409, "AlreadyExists", "exists"))
    del cluster.objects[f"{ISVC}/meritorder-gpu"]
    await prov.up(1)
    assert f"{ISVC}/meritorder-gpu" in cluster.objects
    cluster.fault("PATCH", "inferenceservices", api_status(422, "Invalid", "spec.predictor.minReplicas: Invalid value"))
    with pytest.raises(KubeError, match="Invalid value"):
        await prov.scale(2)


# ------------------------------------------------------------------ llm-d


def install_pool(cluster: FakeCluster, path: str = POOL_V1, spec: dict | None = None) -> None:
    cluster.objects[path] = {"metadata": {"name": "qwen-pool"}, "spec": spec or {
        "selector": {"matchLabels": {"llm-d.ai/inferenceServing": "true", "llm-d.ai/model": "qwen"}},
        "targetPorts": [{"number": 8200}], "endpointPickerRef": {"name": "qwen-pool-epp", "port": {"number": 9002}}}}


LLMD = {"api": "llm-d", "inference_pool": "qwen-pool", "endpoint": "http://llm-d-gateway.gpus.svc.cluster.local/v1"}


async def test_llmd_lifecycle_behind_the_inference_pool(probe):
    cfg, cluster = k8s_config(**LLMD, labels={"team": "ml"}, annotations={"sidecar.istio.io/inject": "false"}, cpu="16", memory="200Gi",
                              cache_pvc="hf-cache", tolerations=[{"key": "spot", "operator": "Exists"}]), FakeCluster()
    install_pool(cluster)
    prov = provisioner(cfg, cluster)
    await prov.up(1)
    dep = cluster.objects[f"{DEPLOYMENTS}/meritorder-gpu"]
    tmpl = dep["spec"]["template"]
    assert dep["spec"]["replicas"] == 1 and dep["spec"]["selector"] == {"matchLabels": {kube.MANAGED: "meritorder-gpu"}}
    assert dep["spec"]["progressDeadlineSeconds"] == 45 * 60  # model loading takes long: don't let the Deployment give up first
    assert tmpl["metadata"]["labels"]["llm-d.ai/model"] == "qwen" and tmpl["metadata"]["labels"]["team"] == "ml"
    assert tmpl["metadata"]["annotations"] == {"sidecar.istio.io/inject": "false"}
    [c] = tmpl["spec"]["containers"]
    assert c["args"][c["args"].index("--port") + 1] == "8200" and c["ports"][0]["containerPort"] == 8200  # the pool's target port
    assert c["resources"]["limits"] == {"nvidia.com/gpu": "4", "cpu": "16", "memory": "200Gi"}
    assert {"name": "cache", "persistentVolumeClaim": {"claimName": "hf-cache"}} in tmpl["spec"]["volumes"]
    assert tmpl["spec"]["tolerations"][1] == {"key": "spot", "operator": "Exists"}

    st = await prov.status()
    assert not st.ready and st.message == "waiting for Deployment pods"
    labels = tmpl["metadata"]["labels"]
    cluster.pods = [pod("meritorder-gpu-abc", labels, node="gke-gpu-0", ip="10.1.0.5", port=8200)]
    cluster.nodes = {"gke-gpu-0": node("gke-gpu-0")}
    dep["status"] = {"conditions": [{"type": "Available", "status": "False", "reason": "MinimumReplicasUnavailable",
                                     "message": "Deployment does not have minimum availability."}]}
    st = await prov.status()
    assert not st.ready and st.message == "Deployment does not have minimum availability." and st.replicas_info[0]["status"] == "STARTING"
    cluster.pods[0]["status"]["conditions"].append({"type": "Ready", "status": "True"})
    st = await prov.status()  # the pod is ready but the Gateway doesn't answer (a wrong address, a NetworkPolicy)
    assert not st.ready and st.message == f"pods are ready, but {LLMD['endpoint']}/models doesn't answer"
    probe.add(LLMD["endpoint"])
    st = await prov.status()
    assert st.ready and st.endpoint == LLMD["endpoint"] and st.replicas_info[0]["endpoint"] == "http://10.1.0.5:8200"

    await prov.scale(3)
    assert cluster.objects[f"{DEPLOYMENTS}/meritorder-gpu"]["spec"]["replicas"] == 3
    await prov.down()
    assert f"{DEPLOYMENTS}/meritorder-gpu" not in cluster.objects and POOL_V1 in cluster.objects  # the router stays up


async def test_llmd_pool_versions_and_missing_pool(probe):
    cfg = k8s_config(**LLMD)
    cluster = FakeCluster()
    with pytest.raises(KubeError, match="InferencePool 'qwen-pool' not found in namespace gpus: install llm-d's router"):
        await provisioner(cfg, cluster).up(1)
    install_pool(cluster, POOL_ALPHA, {"selector": {"app": "qwen"}, "targetPortNumber": 8000})  # an older llm-d (v1alpha2)
    await provisioner(cfg, cluster).up(1)
    assert cluster.objects[f"{DEPLOYMENTS}/meritorder-gpu"]["spec"]["template"]["metadata"]["labels"]["app"] == "qwen"
    empty = FakeCluster()
    install_pool(empty, spec={"selector": {}})
    with pytest.raises(KubeError, match="selects no pods"):
        await provisioner(cfg, empty).up(1)


def test_llmd_needs_its_pool_and_gateway():
    with pytest.raises(ValidationError, match="needs inference_pool"):
        KubernetesServe(api="llm-d", inference_pool="p")
    assert KubernetesServe(**LLMD).api == "llm-d"


def test_kserve_needs_cpu_and_memory():
    """KServe fills in a predictor's missing cpu and memory, requests and limits, with its own defaults: 1 CPU and 2 GiB
    in KServe 0.21 (seen on kind). vLLM can't load a model in that, so a KServe deployment says what a replica may use."""
    for kubernetes in (None, KubernetesServe(), KubernetesServe(memory="200Gi")):
        with pytest.raises(ValidationError, match="KServe deployments need cpu and memory"):
            ServeConfig(model="ow", provisioner="kubernetes", kubernetes=kubernetes)
    ServeConfig(model="ow", provisioner="kubernetes", kubernetes=KubernetesServe(cpu="16", memory="200Gi"))
    ServeConfig(model="ow", provisioner="kubernetes", kubernetes=KubernetesServe(**LLMD))  # a plain Deployment: no defaults
    ServeConfig(model="ow", provisioner="skypilot")


# ------------------------------------------------------------------ failures


async def test_quota_refusal_is_a_failure(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    await prov.up(1)
    cluster.predictors = [{"metadata": {"name": "meritorder-gpu-predictor", "labels": {kube.KSERVE_LABEL: "meritorder-gpu"}},
                           "status": {"conditions": [{"type": "ReplicaFailure", "status": "True", "reason": "FailedCreate",
                                                      "message": 'pods "meritorder-gpu-predictor-x" is forbidden: exceeded quota: gpu-quota, '
                                                                 "requested: requests.nvidia.com/gpu=4, used: requests.nvidia.com/gpu=8, "
                                                                 "limited: requests.nvidia.com/gpu=8"}]}}]
    st = await prov.status()
    assert st.failed and not st.ready and st.message.startswith("pods can't be created: ") and "exceeded quota: gpu-quota" in st.message


@pytest.mark.parametrize("waiting,message,restarts,fatal,expect", [
    ("ErrImagePull", 'failed to pull image "vllm/vllm-openai:v9": manifest unknown', 0, True, "the image can't be pulled (ErrImagePull)"),
    ("ImagePullBackOff", 'Back-off pulling image "ghcr.io/x/vllm": unauthorized: authentication required', 0, True, "the image can't be pulled"),
    ("InvalidImageName", 'couldn\'t parse image name "vllm:"', 0, True, "InvalidImageName"),
    ("ErrImagePull", "dial tcp: lookup registry-1.docker.io: i/o timeout", 0, False, "pulling the image failed; Kubernetes is retrying"),
    ("CreateContainerConfigError", 'secret "hf" not found', 0, True, 'CreateContainerConfigError: secret "hf" not found'),
    ("CrashLoopBackOff", "back-off 40s restarting failed container", 3, True, "vLLM keeps crashing (OOMKilled, 3 restarts)"),
    ("CrashLoopBackOff", "back-off 10s restarting failed container", 1, False, "waiting for vLLM to load the model"),
])
async def test_pod_failures(probe, waiting, message, restarts, fatal, expect):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    await prov.up(1)
    cluster.pods = [pod("p-0", {kube.KSERVE_LABEL: "meritorder-gpu"}, node="n-0", waiting=waiting, message=message, restarts=restarts,
                        last={"reason": "OOMKilled", "exitCode": 137})]
    st = await prov.status()
    assert st.failed is fatal and not st.ready and expect in st.message
    assert st.replicas_info[0]["status"] == ("FAILED" if fatal else "STARTING")


def test_pod_states_and_problems():
    labels = {"a": "b"}
    gone = pod("p", labels, node="n", ready=True)
    gone["metadata"]["deletionTimestamp"] = "2026-11-12T10:00:00Z"
    done = pod("p", labels, node="n")
    done["status"]["phase"] = "Failed"
    assert [kube.pod_state(p) for p in (gone, done, pod("p", labels), pod("p", labels, node="n", ready=True))] == [
        "TERMINATING", "FAILED", "PENDING", "READY"]
    init = pod("p", labels, node="n")
    init["status"]["initContainerStatuses"] = [{"state": {"waiting": {"reason": "InvalidImageName", "message": "bad"}}}]
    assert kube.blocking_problem([init], [])[1]
    assert kube.blocking_problem([], [{"status": {"conditions": [{"type": "ReplicaFailure", "status": "False"}]}}]) == (None, False)
    assert kube.unready_message({"status": {"conditions": [{"type": "Ready", "status": "True", "message": "ok"}]}}) is None
    assert kube.unix("not a time") is None and kube.unix(None) is None


async def test_api_timeouts_and_outages(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    cluster.fault("POST", "inferenceservices", httpx.ReadTimeout("timed out"))
    with pytest.raises(KubeError, match=r"the Kubernetes API didn't answer within 15s \(POST inferenceservices.serving.kserve.io\)"):
        await prov.up(1)
    await prov.up(1)
    serving(cluster)

    # No endpoint known yet: an outage is "not ready", not a failure (the provisioning timeout still applies).
    cluster.fault("GET", "/inferenceservices/", httpx.ConnectError("connection refused"))
    st = await prov.status()
    assert not st.ready and not st.failed and "can't reach the Kubernetes API at https://k8s.test" in st.message

    probe.add(KSERVE_URL + "/v1")
    assert (await prov.status()).ready
    # Warm, then the API stops answering: the replicas still do, so it stays warm.
    cluster.fault("GET", "/inferenceservices/", httpx.ReadTimeout("timed out"))
    st = await prov.status()
    assert st.ready and st.endpoint == KSERVE_URL + "/v1" and st.replicas == 1 and "can't read it from Kubernetes, but it answers" in st.message
    probe.clear()
    cluster.fault("GET", "/inferenceservices/", api_status(503, "ServiceUnavailable", "apiserver is shutting down"))
    st = await prov.status()
    assert not st.ready and not st.failed and "apiserver is shutting down" in st.message


async def test_permissions_and_missing_kserve(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    forbidden = 'inferenceservices.serving.kserve.io is forbidden: User "x" cannot create'
    cluster.fault("POST", "inferenceservices", api_status(403, "Forbidden", forbidden))
    with pytest.raises(KubeError, match="may not create inferenceservices.serving.kserve.io: turn on controller.kubernetes") as e:
        await prov.up(1)
    assert e.value.status == 403 and e.value.reason == "Forbidden"
    cluster.fault("GET", "/inferenceservices/", api_status(403, "Forbidden", "forbidden"))
    assert (await prov.status()).failed  # no endpoint to fall back on, and it won't fix itself
    cluster.fault("GET", "/inferenceservices/", api_status(401, "Unauthorized", "Unauthorized"))
    st = await prov.status()
    assert st.failed and "refused the service account token" in st.message
    cluster.served.discard("inferenceservices")
    with pytest.raises(KubeError, match="doesn't serve inferenceservices.serving.kserve.io: is KServe installed"):
        await prov.up(1)


async def test_no_credentials_is_a_failure(monkeypatch, tmp_path):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.delenv("MERITORDER_GPU_NAMESPACE", raising=False)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "missing"))
    cfg = k8s_config()
    prov = gpu.make_provisioner(cfg.backends[0], cfg)
    assert isinstance(prov, KubernetesProvisioner)
    st = await prov.status()
    assert st.failed and "no Kubernetes credentials" in st.message
    with pytest.raises(KubeError, match="no Kubernetes credentials"):
        await prov.up(1)


def test_namespace_resolution(monkeypatch):
    cfg, cluster = k8s_config(), FakeCluster()
    assert provisioner(cfg, cluster).namespace == "gpus"
    cfg.kubernetes.namespace = None
    monkeypatch.setenv("MERITORDER_GPU_NAMESPACE", "from-chart")
    assert provisioner(cfg, cluster).namespace == "from-chart"
    monkeypatch.delenv("MERITORDER_GPU_NAMESPACE")
    assert provisioner(cfg, cluster).namespace == "default"  # the API's: the service account's or the kubeconfig context's
    assert provisioner(k8s_config(namespace="team-a"), cluster).namespace == "team-a"
    long = k8s_config()
    long.backends[0].name = "A_very_long_backend_name_that_goes_on_and_on_and_on"
    assert len(name := provisioner(long, cluster).name) <= 45 and name.startswith("meritorder-a-very-long") and not name.endswith("-")


# ------------------------------------------------------------------ the controller drives it


async def test_controller_drives_kubernetes_through_its_lifecycle(probe, monkeypatch):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    store, state = MemoryStore(), MemoryState()
    deps = [Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)]
    sampled = []

    async def fetch_load(endpoint, headers=None):
        sampled.append(endpoint)
        return {"utilization": 0.5, "waiting": 1, "running": 6}

    monkeypatch.setattr(gpu, "fetch_load", fetch_load)
    monkeypatch.setattr(controller, "fetch_load", fetch_load)
    t = time.time()

    async def control(**c):
        await store.set_setting("gpu_controls", {"deployments": {"gpu": {"by": "root", **c}}})

    await control(mode="on", replicas=2)
    [rec] = await controller.tick(cfg, store, state, deps, now=t)
    assert rec["state"] == PROVISIONING and ("POST", ISVC) in cluster.calls
    assert rec["price"] == {"usd_per_hour": 44.0, "source": "node_type", "node_type": "a3-highgpu-8g", "node_usd_per_hour": 88.0,
                            "node_gpus": 8, "gpus": 4, "spot": False, "configured_usd_per_hour": 30.0}
    assert rec["status_message"] == "waiting for InferenceService pods" and rec["burn_usd_per_hour"] == 88.0

    cluster.pods = [pod("meritorder-gpu-predictor-0", {kube.KSERVE_LABEL: "meritorder-gpu"}, unschedulable="0/2 nodes are available")]
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 60)
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 120)  # the same message isn't logged twice
    assert rec["state"] == PROVISIONING and "waiting for a node" in rec["status_message"]
    messages = [e["detail"]["message"] for e in await store.gpu_events() if e["kind"] == "status"]
    assert len(messages) == 2 and "waiting for a node: 0/2 nodes are available" in messages[0]

    serving(cluster, pods=2)
    probe.add(KSERVE_URL + "/v1")
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 180)
    assert rec["state"] == WARM and rec["endpoint"] == KSERVE_URL + "/v1" and rec["status_message"] is None
    assert [r["usd_per_hour"] for r in rec["replicas_info"]] == [44.0, 44.0] and rec["burn_usd_per_hour"] == 88.0
    # Load from each pod (not the Service): the mean utilization, and the requests waiting across them.
    assert rec["utilization"] == 0.5 and rec["waiting"] == 2 and sorted(sampled) == ["http://10.0.0.1:8000/v1", "http://10.0.0.2:8000/v1"]
    assert (await state.utilization())["gpu"] == 0.5 and rec["cache"]["affinity"]  # KServe pods: gateways keep sessions on theirs

    await control(mode="on", replicas=3)
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 240)
    assert rec["replicas"] == 3 and cluster.objects[f"{ISVC}/meritorder-gpu"]["spec"]["predictor"]["minReplicas"] == 3

    await control(mode="off", drain=True)
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 300)
    assert rec["state"] == "draining" and f"{ISVC}/meritorder-gpu" in cluster.objects  # draining: the replicas keep serving
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 300 + 15 * 60)
    assert rec["state"] == API_ONLY and f"{ISVC}/meritorder-gpu" not in cluster.objects and rec["last_run"]["cost"] > 0
    kinds = [(e["kind"], e["detail"].get("to")) for e in await store.gpu_events()]
    assert ("state", "warm") in kinds and ("state", "api_only") in kinds and ("scale", None) in kinds


async def test_controller_stops_and_cools_down_when_quota_refuses_the_pods(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    cfg.backends[0].serve.min_replicas = 1
    prov = provisioner(cfg, cluster)
    store, state = MemoryStore(), MemoryState()
    deps = [Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)]
    t = time.time()
    [rec] = await controller.tick(cfg, store, state, deps, now=t)
    assert rec["state"] == PROVISIONING
    cluster.predictors = [{"metadata": {"name": "p", "labels": {kube.KSERVE_LABEL: "meritorder-gpu"}},
                           "status": {"conditions": [{"type": "ReplicaFailure", "status": "True", "message": "exceeded quota: gpu-quota"}]}}]
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 60)
    assert rec["state"] == API_ONLY and rec["cooldown_until"] == t + 60 + 1800
    assert rec["reason"] == "provisioning failed (pods can't be created: exceeded quota: gpu-quota); cooling down 30 min"
    assert ("DELETE", f"{ISVC}/meritorder-gpu") in cluster.calls
    assert any(e["kind"] == "state" and "exceeded quota" in e["detail"]["reason"] for e in await store.gpu_events())


async def test_a_pinned_deployment_that_failed_waits_out_its_cooldown(probe):
    """Found on kind: pinned on, a start that failed for good (a missing Secret) was made again every other cycle, failing
    and alerting each time, while its card said it was cooling down. Pinned before the failure, it now waits out the
    cooldown; pinning it again (once the Secret is there) retries at once."""
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    store, state = MemoryStore(), MemoryState()
    deps = [Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)]
    t = time.time()
    pin = {"deployments": {"gpu": {"mode": "on", "replicas": 1, "by": "root", "at": t - 1}}}
    await store.set_setting("gpu_controls", pin)
    [rec] = await controller.tick(cfg, store, state, deps, now=t)
    assert rec["state"] == PROVISIONING
    cluster.pods = [pod("p-0", {kube.KSERVE_LABEL: "meritorder-gpu"}, node="n-0", waiting="CreateContainerConfigError",
                        message='secret "hf" not found')]
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 60)
    assert rec["state"] == API_ONLY and "cooling down 30 min" in rec["reason"]
    assert not (await store.get_setting("gpu_controls"))["deployments"]  # the failed start ended the pin
    creates = sum(1 for m, p in cluster.calls if m == "POST")
    cluster.pods = []
    await store.set_setting("gpu_controls", pin)  # written back by a control saved meanwhile, with its old time
    for minutes in (2, 3, 10):
        [rec] = await controller.tick(cfg, store, state, deps, now=t + minutes * 60)
        assert rec["state"] == API_ONLY and "pin it again to retry now" in rec["note"]
    assert sum(1 for m, p in cluster.calls if m == "POST") == creates  # not made again
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {**pin["deployments"]["gpu"], "at": t + 11 * 60}}})
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 11 * 60 + 1)
    assert rec["state"] == PROVISIONING  # pinned again: a new start, cooldown or not


async def test_controller_gives_up_when_no_node_comes(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    cfg.backends[0].serve.min_replicas = 1
    prov = provisioner(cfg, cluster)
    store, state = MemoryStore(), MemoryState()
    deps = [Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)]
    t = time.time()
    await controller.tick(cfg, store, state, deps, now=t)
    cluster.pods = [pod("p-0", {kube.KSERVE_LABEL: "meritorder-gpu"}, unschedulable="0/4 nodes are available: 4 Insufficient nvidia.com/gpu.")]
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 46 * 60)
    assert rec["state"] == API_ONLY and rec["reason"].startswith("provisioning timed out after 45 min (p-0: waiting for a node: 0/4 nodes")
    assert f"{ISVC}/meritorder-gpu" not in cluster.objects


def test_llmd_leaves_replica_choice_to_its_endpoint_picker():
    assert not k8s_config(**LLMD).backends[0].serve.replica_affinity  # llm-d's picker routes by prefix cache already
    assert k8s_config().backends[0].serve.replica_affinity  # KServe: gateways keep sessions on their pod
    kept = ServeConfig(model="ow", provisioner="kubernetes", replica_affinity=True, kubernetes=KubernetesServe(**LLMD))
    assert kept.replica_affinity  # set explicitly: honoured


# ------------------------------------------------------------------ the rest of the machinery


def test_deployment_for_and_breakeven_price_from_node_types():
    cfg = k8s_config()
    cfg.backends.append(BackendConfig(name="box", kind="self_hosted", format="openai",
                                      serve=ServeConfig(model="ow", provisioner="static", static_url="http://box/v1")))
    cfg.models[0].routes.append(Route(backend="box", model="m"))
    assert gpu.deployment_for(cfg, cfg.models[0]).name == "gpu"  # the one the controller starts, before a server you run
    plans = gpu_sizing.replica_plans(cfg, {"offers": []}, {"models": {}}, {"aws"}, spot=False)
    assert plans["ow"]["replica"] == {"accelerator": "H100", "count": 4, "cloud": "kubernetes", "usd_per_hour": 44.0,
                                      "instance_type": "a3-highgpu-8g", "spot": False, "source": "node_type"}
    cfg.backends[0].serve.kubernetes.node_type = None
    plans = gpu_sizing.replica_plans(cfg, {"offers": []}, {"models": {}}, {"aws"}, spot=False)
    assert plans["ow"]["replica"]["source"] == "configured" and plans["ow"]["replica"]["usd_per_hour"] == 30.0


def test_shared_vllm_command():
    cfg = k8s_config()
    assert gpu.vllm_command(cfg.backends[0], cfg, 9000)[:7] == ["vllm", "serve", "org/open-model", "--host", "0.0.0.0", "--port", "9000"]
    assert gpu.gpu_count("H100:8") == 8 and gpu.gpu_count("L4") == 1 and gpu.gpu_count("A100:x") == 1


def test_config_from_yaml():
    raw = yaml.safe_load("""
kubernetes:
  namespace: gpus
  node_types:
    g6.12xlarge: {usd_per_hour: 4.6, gpus: 4}
backends:
  gpu-k8s:
    kind: self_hosted
    format: openai
    serve:
      provisioner: kubernetes
      model: ow
      accelerators: "L4:4"
      kubernetes: {api: llm-d, inference_pool: p, endpoint: "http://gw/v1", node_type: g6.12xlarge}
models:
  ow: {hf_id: o/m, routes: [{backend: gpu-k8s, model: m}]}
""")
    from meritorder.config import build_config

    cfg = build_config(raw, prices_data={})
    assert cfg.kubernetes.node_types["g6.12xlarge"].gpus == 4 and cfg.backend("gpu-k8s").enabled
    assert kube.expected_node_price(cfg, cfg.backend("gpu-k8s"))["usd_per_hour"] == 4.6


# ------------------------------------------------------------------ API access


def test_resource_names():
    assert kube.resource(f"{ISVC}/x") == "inferenceservices.serving.kserve.io"
    assert kube.resource("/api/v1/namespaces/gpus/pods?labelSelector=a") == "pods"
    assert kube.resource("/api/v1/nodes/n") == "nodes" and kube.resource("/apis/apps/v1/namespaces/a/deployments") == "deployments.apps"


async def test_api_errors_and_retries_after_401():
    calls = []

    def handler(request):
        calls.append(request.headers.get("authorization"))
        if request.url.path == "/api/v1/namespaces/a/pods":
            return httpx.Response(500, text="boom")
        if len(calls) == 1:
            return api_status(401, "Unauthorized", "Unauthorized")
        return httpx.Response(200, json={"ok": True})

    class Token:
        def __init__(self):
            self.n = 0

        async def __call__(self):
            self.n += 1
            return f"t{self.n}"

        def reset(self):
            pass

    api = KubeAPI("https://k8s.test/", token=Token(), transport=httpx.MockTransport(handler))
    assert await api.call("GET", "/api/v1/namespaces") == {"ok": True} and calls == ["Bearer t1", "Bearer t2"]  # a fresh token, once
    with pytest.raises(KubeError, match="GET pods: boom") as e:
        await api.call("GET", "/api/v1/namespaces/a/pods")
    assert e.value.status == 500 and e.value.reason == ""
    await api.close()
    await api.close()
    plain = KubeAPI("https://k8s.test", transport=httpx.MockTransport(lambda r: httpx.Response(204)))
    assert await plain.call("DELETE", "/api/v1/x") == {}


async def test_in_cluster_service_account(tmp_path, monkeypatch, certs):
    (tmp_path / "token").write_text("sa-token\n")
    (tmp_path / "namespace").write_text("meritorder")
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    api = KubeAPI.from_environment(sa_dir=tmp_path)
    assert api.server == "https://10.96.0.1:443" and api.namespace == "meritorder" and api.verify is True
    assert await api.token() == "sa-token"
    (tmp_path / "ca.crt").write_bytes(certs["cert"])
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "fd00::1")
    api = KubeAPI.in_cluster(tmp_path)
    assert api.server == "https://[fd00::1]:443" and isinstance(api.verify, kube.ssl.SSLContext)


@pytest.fixture
def certs():
    """A self-signed certificate and its key, as a kubeconfig would carry them."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "kubernetes")])
    now = datetime.now(UTC)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(1)
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True).sign(key, hashes.SHA256()))
    return {"cert": cert.public_bytes(serialization.Encoding.PEM),
            "key": key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())}


def write_kubeconfig(tmp_path: Path, user: dict, cluster: dict | None = None, **extra) -> Path:
    doc = {"apiVersion": "v1", "kind": "Config", "current-context": "dev",
           "contexts": [{"name": "dev", "context": {"cluster": "kind", "user": "me", "namespace": "ml"}},
                        {"name": "nouser", "context": {"cluster": "kind"}}],
           "clusters": [{"name": "kind", "cluster": {"server": "https://127.0.0.1:6443", **(cluster or {})}}],
           "users": [{"name": "me", "user": user}], **extra}
    path = tmp_path / "config"
    path.write_text(yaml.safe_dump(doc))
    return path


async def test_kubeconfig_tokens_certificates_and_contexts(tmp_path, monkeypatch, certs):
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    path = write_kubeconfig(tmp_path, {"token": "dev-token"}, {"certificate-authority-data": b64(certs["cert"])})
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv("KUBECONFIG", f"{path}{os.pathsep}/elsewhere")
    api = KubeAPI.from_environment()
    assert api.server == "https://127.0.0.1:6443" and api.namespace == "ml" and await api.token() == "dev-token"
    assert KubeAPI.from_kubeconfig(path, "nouser").token is None and KubeAPI.from_kubeconfig(path, "nouser").namespace == "default"
    with pytest.raises(KubeError, match="no context named 'prod'"):
        KubeAPI.from_kubeconfig(path, "prod")

    (tmp_path / "token-file").write_text("file-token")
    (tmp_path / "ca.pem").write_bytes(certs["cert"])
    (tmp_path / "client.pem").write_bytes(certs["cert"])
    (tmp_path / "client.key").write_bytes(certs["key"])
    files = write_kubeconfig(tmp_path, {"tokenFile": "token-file", "client-certificate": "client.pem", "client-key": "client.key"},
                             {"certificate-authority": "ca.pem"})  # relative to the kubeconfig's directory
    assert await KubeAPI.from_environment(str(files)).token() == "file-token"
    inline = write_kubeconfig(tmp_path, {"client-certificate-data": b64(certs["cert"]), "client-key-data": b64(certs["key"])},
                              {"insecure-skip-tls-verify": True})
    ctx = KubeAPI.from_kubeconfig(inline).verify
    assert ctx.verify_mode == kube.ssl.CERT_NONE and not ctx.check_hostname

    (tmp_path / "broken").write_text("contexts: [")
    with pytest.raises(KubeError, match="can't read kubeconfig"):
        KubeAPI.from_kubeconfig(tmp_path / "broken")


EXEC_PLUGIN = """
import json, os, sys
count = os.path.join(os.path.dirname(sys.argv[0]), "count")
n = int(open(count).read()) + 1 if os.path.exists(count) else 1
open(count, "w").write(str(n))
info = json.loads(os.environ["KUBERNETES_EXEC_INFO"])
assert info["kind"] == "ExecCredential" and os.environ["CLUSTER"] == "dev"
mode = sys.argv[1]
if mode == "fail":
    sys.stderr.write("token expired; run aws sso login")
    sys.exit(1)
if mode == "garbage":
    print("not json")
elif mode == "nothing":
    print(json.dumps({"status": {}}))
else:
    exp = "2000-01-01T00:00:00Z" if mode == "expired" else None
    print(json.dumps({"apiVersion": info["apiVersion"], "kind": "ExecCredential",
                      "status": {"token": f"exec-{n}", **({"expirationTimestamp": exp} if exp else {})}}))
"""


async def test_exec_plugin_tokens(tmp_path):
    script = tmp_path / "plugin.py"
    script.write_text(EXEC_PLUGIN)

    def plugin(mode: str) -> ExecToken:
        return ExecToken({"command": sys.executable, "args": [str(script), mode], "env": [{"name": "CLUSTER", "value": "dev"}]}, tmp_path)

    ok = plugin("ok")
    assert await ok() == "exec-1" and await ok() == "exec-1"  # cached: no expiry given
    ok.reset()
    assert await ok() == "exec-2"
    expired = plugin("expired")
    assert await expired() == "exec-3" and await expired() == "exec-4"  # already expired: run again each time
    for mode, message in (("fail", "failed: token expired; run aws sso login"), ("garbage", "printed no ExecCredential"),
                          ("nothing", "gave no token")):
        with pytest.raises(KubeError, match=message):
            await plugin(mode)()
    with pytest.raises(KubeError, match="can't run"):
        await ExecToken({"command": str(tmp_path / "no-such-plugin")}, tmp_path)()
    path = write_kubeconfig(tmp_path, {"exec": {"apiVersion": "client.authentication.k8s.io/v1", "command": sys.executable,
                                                "args": [str(script), "ok"], "env": [{"name": "CLUSTER", "value": "dev"}]}})
    assert isinstance(KubeAPI.from_kubeconfig(path).token, ExecToken)


async def test_exec_plugin_timeout(tmp_path, monkeypatch):
    async def slow(aw, timeout):
        aw.close()
        raise TimeoutError

    monkeypatch.setattr(kube.asyncio, "wait_for", slow)
    with pytest.raises(KubeError, match="didn't finish in 60s"):
        await ExecToken({"command": sys.executable, "args": ["-c", "import time; time.sleep(5)"]}, tmp_path)()
