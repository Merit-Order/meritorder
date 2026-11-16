"""Kubernetes GPU provisioner: vLLM on your own cluster's GPUs, through KServe or llm-d (docs/05-gpu-autoscaling.md#kubernetes).

The controller drives it like the SkyPilot one: ``up``, ``scale``, ``down``, ``status`` (readiness, the endpoint, and
where each replica runs, with each pod's address, which the controller reads vLLM's metrics from and gateways can send
a session back to), and ``expected_price``. Two ways to run the vLLM pods:

- ``kserve``: an InferenceService (serving.kserve.io/v1beta1) in KServe's Standard mode whose predictor is the vLLM
  OpenAI server, with a fixed replica count and no KServe autoscaler: the controller decides how many.
- ``llm-d``: a Deployment of vLLM pods carrying the labels of an InferencePool installed with llm-d's router (the
  Gateway API inference extension). Its endpoint picker sends each request to the replica that has the most of its
  prompt cached. The router stays up while Merit Order starts and stops the model servers behind it.

It talks to the Kubernetes API over plain HTTP (httpx): with the pod's service account in the cluster, or a kubeconfig
(token, client certificate, or exec plugin) for development. No Kubernetes client library.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import ssl
import tempfile
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from . import gpu
from .config import BackendConfig, Config, KubernetesConfig, KubernetesServe, read_secret_file

log = logging.getLogger("meritorder.kube")

SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")
KSERVE = "/apis/serving.kserve.io/v1beta1"
POOL_APIS = ("/apis/inference.networking.k8s.io/v1", "/apis/inference.networking.x-k8s.io/v1alpha2")  # GA, then older llm-d
MANAGED = "meritorder.dev/deployment"
INSTALL = "meritorder.dev/install"  # which install made it (its id, as in SkyPilot cluster names): the sweep deletes only its own
OWNED = "app.kubernetes.io/managed-by=meritorder"
DOWN_WAIT_SECONDS = 60  # a stop waits this long beyond the pods' grace period for them to be gone, then fails (tried again)
DOWN_POLL_SECONDS = 2.0
KSERVE_LABEL = "serving.kserve.io/inferenceservice"
# Node labels that mark spot or preemptible capacity: Karpenter, EKS managed node groups, GKE, AKS.
SPOT_LABELS = {"karpenter.sh/capacity-type": "spot", "eks.amazonaws.com/capacityType": "spot", "cloud.google.com/gke-spot": "true",
               "cloud.google.com/gke-preemptible": "true", "kubernetes.azure.com/scalesetpriority": "spot"}
CRASH_LIMIT = 3  # restarts after which a crash-looping vLLM counts as failed (a bad argument, too little GPU memory)
# Image pull errors that won't go away by retrying: a wrong name or tag, or a registry that refuses the cluster.
IMAGE_GONE = re.compile(r"not found|manifest unknown|unauthorized|denied|forbidden|invalid reference|repository does not exist", re.I)


# ------------------------------------------------------------------ API access


class KubeError(RuntimeError):
    """A Kubernetes API call that failed. ``status`` is the HTTP status, 0 when the API didn't answer or there are no
    credentials; ``reason`` is the API's reason (NotFound, Forbidden, ...), or Timeout, Unreachable, Config."""

    def __init__(self, message: str, status: int = 0, reason: str = ""):
        super().__init__(message)
        self.status = status
        self.reason = reason


TokenSource = Callable[[], Awaitable[str | None]]


class KubeAPI:
    """Plain HTTPS calls to the Kubernetes API with a bearer token or client certificate."""

    def __init__(self, server: str, namespace: str = "default", token: TokenSource | None = None,
                 verify: ssl.SSLContext | bool = True, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 15.0):
        self.server = server.rstrip("/")
        self.namespace = namespace
        self.token = token
        self.verify = verify
        self.transport = transport
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_environment(cls, kubeconfig: str | None = None, context: str | None = None,
                         transport: httpx.AsyncBaseTransport | None = None, sa_dir: Path = SERVICE_ACCOUNT) -> "KubeAPI":
        """The service account when running in the cluster, else a kubeconfig (``kubeconfig``, $KUBECONFIG, ~/.kube/config)."""
        if not kubeconfig and os.environ.get("KUBERNETES_SERVICE_HOST") and (sa_dir / "token").exists():
            return cls.in_cluster(sa_dir, transport)
        path = kubeconfig or (os.environ.get("KUBECONFIG") or "").split(os.pathsep)[0] or str(Path.home() / ".kube" / "config")
        if not Path(path).expanduser().exists():
            raise KubeError("no Kubernetes credentials: run the controller in the cluster with a service account, "
                            "or set kubernetes.kubeconfig (or KUBECONFIG) for development", reason="Config")
        return cls.from_kubeconfig(path, context, transport)

    @classmethod
    def in_cluster(cls, sa_dir: Path = SERVICE_ACCOUNT, transport: httpx.AsyncBaseTransport | None = None) -> "KubeAPI":
        host, port = os.environ["KUBERNETES_SERVICE_HOST"], os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        host = f"[{host}]" if ":" in host else host  # IPv6
        token_file = str(sa_dir / "token")

        async def token() -> str | None:
            return read_secret_file(token_file)  # bound tokens rotate: the kubelet rewrites the file, read again on change

        ca = sa_dir / "ca.crt"
        verify = ssl.create_default_context(cafile=str(ca)) if ca.exists() else True
        return cls(f"https://{host}:{port}", read_secret_file(str(sa_dir / "namespace")) or "default", token, verify, transport)

    @classmethod
    def from_kubeconfig(cls, path: str | Path, context: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> "KubeAPI":
        path = Path(path).expanduser()
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            raise KubeError(f"can't read kubeconfig {path}: {e}", reason="Config") from e

        def pick(kind: str, name: str | None) -> dict:
            for item in doc.get(kind) or []:
                if item.get("name") == name:
                    return item.get(kind[:-1]) or {}
            raise KubeError(f"kubeconfig {path} has no {kind[:-1]} named {name!r}", reason="Config")

        ctx = pick("contexts", context or doc.get("current-context"))
        cluster = pick("clusters", ctx.get("cluster"))
        user = pick("users", ctx["user"]) if ctx.get("user") else {}

        def local(p: str | None) -> str | None:  # paths in a kubeconfig are relative to the file
            return str((path.parent / Path(p).expanduser())) if p else None

        verify = tls_context(ca_file=local(cluster.get("certificate-authority")), ca_data=cluster.get("certificate-authority-data"),
                             cert_file=local(user.get("client-certificate")), key_file=local(user.get("client-key")),
                             cert_data=user.get("client-certificate-data"), key_data=user.get("client-key-data"),
                             insecure=bool(cluster.get("insecure-skip-tls-verify")))
        return cls(cluster["server"], ctx.get("namespace") or "default", kubeconfig_token(user, path.parent), verify, transport)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.server, verify=self.verify, timeout=self.timeout, transport=self.transport)
        return self._client

    async def call(self, method: str, path: str, body: dict | None = None, params: dict | None = None, missing_ok: bool = False,
                   content_type: str = "application/json") -> dict | None:
        """One API call: the response object, or None for a 404 when ``missing_ok``. Errors raise KubeError."""
        for attempt in (0, 1):
            headers = {"accept": "application/json"}
            token = await self.token() if self.token else None
            if token:
                headers["authorization"] = f"Bearer {token}"
            if body is not None:
                headers["content-type"] = content_type
            try:
                r = await self.client.request(method, path, params=params, headers=headers,
                                              content=json.dumps(body) if body is not None else None)
            except httpx.TimeoutException as e:
                raise KubeError(f"the Kubernetes API didn't answer within {self.timeout:g}s ({method} {resource(path)})", reason="Timeout") from e
            except httpx.HTTPError as e:
                raise KubeError(f"can't reach the Kubernetes API at {self.server}: {e}", reason="Unreachable") from e
            if r.status_code == 401 and attempt == 0 and hasattr(self.token, "reset"):
                self.token.reset()  # an exec plugin's token expired early: get a new one and try once more
                continue
            if r.status_code == 404 and missing_ok:
                return None
            if r.status_code >= 400:
                raise api_error(r, method, path)
            return r.json() if r.content else {}
        raise AssertionError("unreachable")  # pragma: no cover

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


VERBS = {"GET": "read", "POST": "create", "PUT": "update", "PATCH": "patch", "DELETE": "delete"}


def resource(path: str) -> str:
    """'/apis/serving.kserve.io/v1beta1/namespaces/gpus/inferenceservices/x' -> 'inferenceservices.serving.kserve.io'."""
    parts = [p for p in path.split("?")[0].split("/") if p]
    group = parts[1] if parts[:1] == ["apis"] and len(parts) > 1 else ""
    rest = parts[3:] if group else parts[2:]
    if rest[:1] == ["namespaces"] and len(rest) > 2:
        rest = rest[2:]
    name = rest[0] if rest else "?"
    return f"{name}.{group}" if group else name


def api_error(r: httpx.Response, method: str, path: str) -> KubeError:
    try:
        body = r.json()
    except ValueError:
        body = None
    body = body if isinstance(body, dict) else {}
    message = body.get("message") or r.text[:300].strip() or r.reason_phrase
    what = resource(path)
    if r.status_code == 401:
        message += " (the API refused the service account token or kubeconfig credentials)"
    elif r.status_code == 403:
        message += (f" (the controller may not {VERBS.get(method, method)} {what}: turn on controller.kubernetes in the Helm chart "
                    "for this namespace, or grant the Role in docs/07-deployment.md)")
    return KubeError(f"{method} {what}: {message}", r.status_code, body.get("reason") or "")


def tls_context(ca_file: str | None = None, ca_data: str | None = None, cert_file: str | None = None, key_file: str | None = None,
                cert_data: str | None = None, key_data: str | None = None, insecure: bool = False) -> ssl.SSLContext:
    """TLS for a kubeconfig cluster: its CA (file or inline), and the user's client certificate if it has one."""
    ctx = ssl.create_default_context(cafile=ca_file, cadata=base64.b64decode(ca_data).decode() if ca_data else None)
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    if cert_data and key_data:
        with tempfile.TemporaryDirectory() as d:  # private (0700); load_cert_chain reads the files, which then go
            cert, key = Path(d) / "cert.pem", Path(d) / "key.pem"
            cert.write_bytes(base64.b64decode(cert_data))
            key.write_bytes(base64.b64decode(key_data))
            ctx.load_cert_chain(cert, key)
    elif cert_file and key_file:
        ctx.load_cert_chain(cert_file, key_file)
    return ctx


def kubeconfig_token(user: dict, base: Path) -> TokenSource | None:
    """A kubeconfig user's bearer token: written in, in a file, or from an exec plugin. None for client certificates."""
    if user.get("token"):
        static = user["token"]

        async def fixed() -> str | None:
            return static

        return fixed
    if user.get("tokenFile"):
        token_file = str(base / user["tokenFile"])

        async def from_file() -> str | None:
            return read_secret_file(token_file)

        return from_file
    if user.get("exec"):
        return ExecToken(user["exec"], base)
    return None


class ExecToken:
    """A kubeconfig exec credential plugin (aws eks get-token, gke-gcloud-auth-plugin, kubelogin): run it for a token,
    and again a minute before that token expires, or when the API refuses it."""

    def __init__(self, spec: dict, cwd: Path):
        self.spec = spec
        self.cwd = cwd
        self.value: str | None = None
        self.expires: float | None = None

    def reset(self) -> None:
        self.value = None

    async def __call__(self) -> str | None:
        if self.value and (self.expires is None or time.time() < self.expires - 60):
            return self.value
        command = self.spec.get("command") or ""
        env = {**os.environ, **{e["name"]: e["value"] for e in self.spec.get("env") or []}}
        env["KUBERNETES_EXEC_INFO"] = json.dumps({"apiVersion": self.spec.get("apiVersion") or "client.authentication.k8s.io/v1",
                                                  "kind": "ExecCredential", "spec": {"interactive": False}})
        try:
            proc = await asyncio.create_subprocess_exec(command, *(self.spec.get("args") or []), env=env, cwd=self.cwd,
                                                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except OSError as e:
            raise KubeError(f"kubeconfig exec plugin {command!r} can't run: {e}", reason="Config") from e
        try:
            out, err = await asyncio.wait_for(proc.communicate(), 60)
        except TimeoutError as e:
            proc.kill()
            raise KubeError(f"kubeconfig exec plugin {command!r} didn't finish in 60s", reason="Timeout") from e
        if proc.returncode:
            raise KubeError(f"kubeconfig exec plugin {command!r} failed: {err.decode(errors='replace')[-300:].strip()}", reason="Config")
        try:
            status = json.loads(out).get("status") or {}
        except (ValueError, AttributeError) as e:
            raise KubeError(f"kubeconfig exec plugin {command!r} printed no ExecCredential", reason="Config") from e
        if not status.get("token"):
            raise KubeError(f"kubeconfig exec plugin {command!r} gave no token (client certificates from plugins aren't supported)",
                            reason="Config")
        self.value = status["token"]
        exp = status.get("expirationTimestamp")
        self.expires = datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp() if exp else None
        return self.value


# ------------------------------------------------------------------ what pods are doing


def blocking_problem(pods: list[dict], workloads: list[dict]) -> tuple[str | None, bool]:
    """The first thing keeping replicas from serving, and whether it stays that way until a person acts: a quota or
    policy that refuses the pods, an image that doesn't exist or can't be pulled, a missing Secret, vLLM crashing on start.
    Waiting for a GPU node isn't fatal: a cluster autoscaler may be adding one (the provisioning timeout still applies)."""
    for w in workloads:
        for c in (w.get("status") or {}).get("conditions") or []:
            if c.get("type") == "ReplicaFailure" and c.get("status") == "True":  # exceeded quota, PodSecurity, a webhook
                return f"pods can't be created: {c.get('message') or c.get('reason') or 'refused'}", True
    waiting = None
    for p in pods:
        name = (p.get("metadata") or {}).get("name")
        st = p.get("status") or {}
        for cs in (st.get("initContainerStatuses") or []) + (st.get("containerStatuses") or []):
            w = (cs.get("state") or {}).get("waiting") or {}
            reason, msg = w.get("reason") or "", (w.get("message") or "").strip()
            if reason == "InvalidImageName" or (reason in ("ErrImagePull", "ImagePullBackOff") and IMAGE_GONE.search(msg)):
                return f"{name}: the image can't be pulled ({reason}): {msg}", True
            if reason in ("CreateContainerConfigError", "CreateContainerError"):
                return f"{name}: {reason}: {msg}", True
            if reason == "CrashLoopBackOff" and (cs.get("restartCount") or 0) >= CRASH_LIMIT:
                last = (cs.get("lastState") or {}).get("terminated") or {}
                why = last.get("reason") or f"exit code {last.get('exitCode')}"
                return f"{name}: vLLM keeps crashing ({why}, {cs.get('restartCount')} restarts)", True
            if reason in ("ErrImagePull", "ImagePullBackOff") and not waiting:
                # Without Kubernetes' own words, which alternate between the two reasons and would log anew each cycle.
                waiting = f"{name}: pulling the image failed; Kubernetes is retrying"
        for c in st.get("conditions") or []:
            if c.get("type") == "PodScheduled" and c.get("status") == "False" and not waiting:
                waiting = f"{name}: waiting for a node: {c.get('message') or c.get('reason')}"
    return waiting, False


def pod_state(pod: dict) -> str:
    """READY, STARTING (on a node, loading), PENDING (no node yet), FAILED, or TERMINATING."""
    meta, spec, st = pod.get("metadata") or {}, pod.get("spec") or {}, pod.get("status") or {}
    if meta.get("deletionTimestamp"):
        return "TERMINATING"
    if st.get("phase") in ("Failed", "Succeeded"):
        return "FAILED"
    if any(c.get("type") == "Ready" and c.get("status") == "True" for c in st.get("conditions") or []):
        return "READY"
    if blocking_problem([pod], [])[1]:
        return "FAILED"
    return "STARTING" if spec.get("nodeName") else "PENDING"


def unready_message(obj: dict) -> str | None:
    """What a resource's own Ready (KServe) or Available (Deployment) condition says while it isn't."""
    for c in (obj.get("status") or {}).get("conditions") or []:
        if c.get("type") in ("Ready", "Available") and c.get("status") != "True" and (c.get("message") or c.get("reason")):
            return c.get("message") or c.get("reason")
    return None


def unix(ts: str | None) -> float | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if ts else None
    except ValueError:
        return None


# ------------------------------------------------------------------ prices


def node_price(cluster: KubernetesConfig, node_type: str | None, gpus: int, spot: bool) -> float | None:
    """A replica's share of a node's configured price: its GPUs out of the node's. None for a node type not configured."""
    t = cluster.node_types.get(node_type) if node_type else None
    if t is None:
        return None
    rate = t.spot_usd_per_hour if spot and t.spot_usd_per_hour is not None else t.usd_per_hour
    return round(rate * gpus / t.gpus, 4)


def expected_node_price(cfg: Config, backend: BackendConfig) -> dict | None:
    """What one replica of this deployment costs on its configured node type, or None when it names none with a price."""
    serve = backend.serve
    assert serve is not None
    k = serve.kubernetes or KubernetesServe()
    t = cfg.kubernetes.node_types.get(k.node_type) if k.node_type else None
    if t is None:
        return None
    gpus = gpu.gpu_count(serve.accelerators)
    spot = serve.use_spot and t.spot_usd_per_hour is not None
    return {"usd_per_hour": node_price(cfg.kubernetes, k.node_type, gpus, spot), "source": "node_type", "node_type": k.node_type,
            "node_usd_per_hour": t.spot_usd_per_hour if spot else t.usd_per_hour, "node_gpus": t.gpus, "gpus": gpus, "spot": spot}


# ------------------------------------------------------------------ the provisioner


class KubernetesProvisioner:
    """Runs vLLM on a Kubernetes cluster's GPUs, as a KServe InferenceService or as model servers behind llm-d's router."""

    def __init__(self, backend: BackendConfig, cfg: Config, api: KubeAPI | None = None, install: str | None = None):
        assert backend.serve is not None
        self.install = install  # this install's id, in the INSTALL label
        self.backend = backend
        self.cfg = cfg
        self.serve = backend.serve
        self.k = self.serve.kubernetes or KubernetesServe()
        self.cluster = cfg.kubernetes
        # A DNS label short enough for KServe's derived names (<name>-predictor-...).
        self.name = re.sub(r"[^a-z0-9-]", "-", f"meritorder-{backend.name}".lower())[:45].strip("-")
        self._api = api
        self.endpoint: str | None = None  # the last endpoint that answered
        self.replicas_info: list[dict] = []
        self._nodes: dict[str, dict] = {}
        self._nodes_forbidden = False

    @property
    def api(self) -> KubeAPI:
        if self._api is None:
            self._api = KubeAPI.from_environment(self.cluster.kubeconfig, self.cluster.context)
        return self._api

    @property
    def namespace(self) -> str:
        return self.k.namespace or self.cluster.namespace or os.environ.get("MERITORDER_GPU_NAMESPACE") or self.api.namespace

    @property
    def kind(self) -> str:
        return "InferenceService" if self.k.api == "kserve" else "Deployment"

    def path(self, name: str | None = None) -> str:
        ns = self.namespace
        base = f"{KSERVE}/namespaces/{ns}/inferenceservices" if self.k.api == "kserve" else f"/apis/apps/v1/namespaces/{ns}/deployments"
        return f"{base}/{name}" if name else base

    def labels(self) -> dict[str, str]:
        return {**self.k.labels, "app.kubernetes.io/managed-by": "meritorder", "app.kubernetes.io/component": "gpu", MANAGED: self.name,
                **({INSTALL: self.install} if self.install else {})}

    def resource_path(self) -> str:
        """What it runs, without asking the cluster: the kind, the configured namespace and the name. Two provisioners with
        the same one run the same resource (controller.carry_over)."""
        ns = self.k.namespace or self.cluster.namespace or os.environ.get("MERITORDER_GPU_NAMESPACE") or ""
        return f"{self.k.api}:{ns}:{self.name}"

    # ---- manifests

    def pod_spec(self, port: int, container: str) -> dict:
        """The vLLM pod: the same `vllm serve` as on SkyPilot, the replica's GPUs, shared memory for tensor parallelism,
        a Hugging Face cache, and the node type's selector."""
        serve, k = self.serve, self.k
        command = gpu.vllm_command(self.backend, self.cfg, port) + shlex.split(serve.vllm_args or "")
        limits = {k.gpu_resource: str(gpu.gpu_count(serve.accelerators))}
        for key, value in (("cpu", k.cpu), ("memory", k.memory)):
            if value:
                limits[key] = value
        env: list[dict] = [{"name": "HF_HOME", "value": "/cache/huggingface"}]
        if k.hf_token_secret:
            env.append({"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": k.hf_token_secret, "key": "HF_TOKEN"}}})
        cache = {"persistentVolumeClaim": {"claimName": k.cache_pvc}} if k.cache_pvc else {"emptyDir": {}}
        spec: dict = {
            "containers": [{
                "name": container, "image": serve.image, "command": command[:1], "args": command[1:],
                "ports": [{"name": "http", "containerPort": port, "protocol": "TCP"}],
                "env": env, "resources": {"requests": dict(limits), "limits": limits},
                "readinessProbe": {"httpGet": {"path": "/health", "port": port}, "periodSeconds": 10, "failureThreshold": 3},
                "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}, {"name": "cache", "mountPath": "/cache"}],
            }],
            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}, {"name": "cache", **cache}],
            # GPU nodes are often tainted for their resource (GKE and AKS do it themselves).
            "tolerations": [{"key": k.gpu_resource, "operator": "Exists", "effect": "NoSchedule"}, *k.tolerations],
            "terminationGracePeriodSeconds": k.termination_grace_seconds,
        }
        selector = {**({self.cluster.node_type_label: k.node_type} if k.node_type else {}), **k.node_selector}
        if selector:
            spec["nodeSelector"] = selector
        return spec

    def inference_service(self, replicas: int) -> dict:
        """KServe Standard mode with the autoscaler off: exactly ``replicas`` predictor pods."""
        annotations = {"serving.kserve.io/deploymentMode": "Standard", "serving.kserve.io/autoscalerClass": "none", **self.k.annotations}
        return {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
                "metadata": {"name": self.name, "namespace": self.namespace, "labels": self.labels(), "annotations": annotations},
                "spec": {"predictor": {"minReplicas": replicas, "maxReplicas": replicas, "labels": self.labels(),
                                       **self.pod_spec(8000, "kserve-container")}}}

    def deployment(self, replicas: int, pool: dict) -> dict:
        """Model server pods for llm-d: the InferencePool's labels, listening on its target port."""
        return {"apiVersion": "apps/v1", "kind": "Deployment",
                "metadata": {"name": self.name, "namespace": self.namespace, "labels": self.labels()},
                "spec": {"replicas": replicas, "selector": {"matchLabels": {MANAGED: self.name}},
                         "progressDeadlineSeconds": max(600, self.serve.provision_timeout_minutes * 60),
                         "template": {"metadata": {"labels": {**pool["labels"], **self.labels()}, "annotations": dict(self.k.annotations)},
                                      "spec": self.pod_spec(pool["port"], "vllm")}}}

    async def pool(self) -> dict:
        """The InferencePool's pod labels and target port: the vLLM pods must carry the one and listen on the other."""
        for api in POOL_APIS:
            obj = await self.api.call("GET", f"{api}/namespaces/{self.namespace}/inferencepools/{self.k.inference_pool}", missing_ok=True)
            if obj is not None:
                break
        else:
            raise KubeError(f"InferencePool {self.k.inference_pool!r} not found in namespace {self.namespace}: install llm-d's router "
                            "(its Gateway, InferencePool and endpoint picker) there first", 404, "NotFound")
        spec = obj.get("spec") or {}
        sel = spec.get("selector") or {}
        labels = sel.get("matchLabels") if isinstance(sel.get("matchLabels"), dict) else {a: b for a, b in sel.items() if isinstance(b, str)}
        if not labels:
            raise KubeError(f"InferencePool {self.k.inference_pool!r} selects no pods (its spec.selector is empty)", reason="Invalid")
        ports = spec.get("targetPorts") or []
        port = (ports[0] or {}).get("number") if ports else spec.get("targetPortNumber")
        return {"labels": labels, "port": int(port or 8000)}

    # ---- lifecycle

    async def up(self, replicas: int) -> None:
        replicas = max(replicas, 1)
        obj = self.inference_service(replicas) if self.k.api == "kserve" else self.deployment(replicas, await self.pool())
        try:
            await self.api.call("POST", self.path(), obj)
            log.info("%s: created %s %s/%s with %d replica(s)", self.backend.name, self.kind, self.namespace, self.name, replicas)
            return
        except KubeError as e:
            if e.status == 404:
                raise KubeError(f"this cluster doesn't serve {resource(self.path())}: is KServe installed?", 404, "NotFound") from e
            if e.status != 409:
                raise
        # It's there already (a controller restarted mid-run, or someone made it): make it ours as configured now.
        current = await self.api.call("GET", self.path(self.name), missing_ok=True)
        if current is None:
            await self.api.call("POST", self.path(), obj)
            return
        theirs = ((current.get("metadata") or {}).get("labels") or {}).get(INSTALL)
        if theirs and self.install and theirs != self.install:
            raise KubeError(f"{self.kind} {self.namespace}/{self.name} belongs to another Merit Order install ({theirs}): give this "
                            "deployment another backend name or namespace", 409, "Conflict")
        if (current.get("metadata") or {}).get("deletionTimestamp"):
            raise KubeError(f"{self.kind} {self.name} is still being deleted after the last stop; it starts again once that's done", 409, "Conflict")
        obj["metadata"]["resourceVersion"] = current["metadata"]["resourceVersion"]
        await self.api.call("PUT", self.path(self.name), obj)
        log.info("%s: updated existing %s %s/%s to %d replica(s)", self.backend.name, self.kind, self.namespace, self.name, replicas)

    async def scale(self, replicas: int) -> None:
        replicas = max(replicas, 1)
        kserve = {"spec": {"predictor": {"minReplicas": replicas, "maxReplicas": replicas}}}
        patch = kserve if self.k.api == "kserve" else {"spec": {"replicas": replicas}}
        try:
            await self.api.call("PATCH", self.path(self.name), patch, content_type="application/merge-patch+json")
        except KubeError as e:
            if e.status != 404:
                raise
            log.warning("%s: %s %s is gone (deleted outside Merit Order); creating it again", self.backend.name, self.kind, self.name)
            await self.up(replicas)

    async def down(self) -> None:
        """Delete it, and wait until it and its pods are gone: they bill until then (their grace period), and a start
        right after would find it still being deleted. Raises when they're still there after the grace period and
        DOWN_WAIT_SECONDS, so the controller keeps it as running, and billed, and tries again next cycle."""
        await self.api.call("DELETE", self.path(self.name), params={"propagationPolicy": "Background"}, missing_ok=True)
        self.endpoint, self.replicas_info = None, []
        selector = f"{KSERVE_LABEL}={self.name}" if self.k.api == "kserve" else f"{MANAGED}={self.name}"
        deadline = time.monotonic() + self.k.termination_grace_seconds + DOWN_WAIT_SECONDS
        while True:
            obj = await self.api.call("GET", self.path(self.name), missing_ok=True)
            pods = (await self.api.call("GET", f"/api/v1/namespaces/{self.namespace}/pods", params={"labelSelector": selector})
                    or {}).get("items") or []
            if obj is None and not pods:
                return
            if time.monotonic() >= deadline:
                raise KubeError(f"{self.kind} {self.name} is still being deleted ({len(pods)} pod(s) left)", 409, "Conflict")
            await asyncio.sleep(DOWN_POLL_SECONDS)

    # ---- status

    async def status(self) -> gpu.Status:
        try:
            return await self._status()
        except KubeError as e:
            # The API is unreachable or refusing; the replicas may well still be serving, so judge by the endpoint.
            if self.endpoint and await gpu._probe(self.endpoint):
                ready = [r for r in self.replicas_info if r.get("status") == "READY"]
                return gpu.Status(ready=True, endpoint=self.endpoint, replicas=len(ready), replicas_info=self.replicas_info,
                                  message=f"can't read it from Kubernetes, but it answers: {e}")
            return gpu.Status(failed=e.status in (401, 403) or e.reason == "Config", message=str(e))

    async def _status(self) -> gpu.Status:
        obj = await self.api.call("GET", self.path(self.name), missing_ok=True)
        if obj is None:
            self.endpoint, self.replicas_info = None, []
            return gpu.Status(failed=True, message=f"{self.kind} {self.name} isn't in namespace {self.namespace} (deleted outside Merit Order?)")
        selector = f"{KSERVE_LABEL}={self.name}" if self.k.api == "kserve" else f"{MANAGED}={self.name}"
        ns = self.namespace
        pods = (await self.api.call("GET", f"/api/v1/namespaces/{ns}/pods", params={"labelSelector": selector}) or {}).get("items") or []
        if self.k.api == "kserve":  # KServe's predictor Deployment says when pods can't be created (quota, policy)
            workloads = (await self.api.call("GET", f"/apis/apps/v1/namespaces/{ns}/deployments", params={"labelSelector": selector})
                         or {}).get("items") or []
        else:
            workloads = [obj]
        problem, fatal = blocking_problem(pods, workloads)
        self.replicas_info = [await self._replica(p) for p in pods]
        ready = [r for r in self.replicas_info if r["status"] == "READY"]
        endpoint = self._endpoint(obj)
        if not ready and not problem:
            problem = unready_message(obj) or ("waiting for vLLM to load the model" if pods else f"waiting for {self.kind} pods")
        if ready and not endpoint:
            problem = problem or "waiting for KServe to publish the service address"
        answers = bool(ready and endpoint and await gpu._probe(endpoint))
        if answers:
            self.endpoint = endpoint
        elif ready and endpoint:  # e.g. a wrong Gateway address for llm-d, or a NetworkPolicy in the way
            problem = problem or f"pods are ready, but {endpoint}/models doesn't answer"
        return gpu.Status(ready=answers, endpoint=endpoint if answers else None, failed=fatal, replicas=len(ready),
                          replicas_info=self.replicas_info, message=problem)

    def _endpoint(self, obj: dict) -> str | None:
        if self.k.endpoint:
            return self.k.endpoint.rstrip("/")
        st = obj.get("status") or {}
        url = (st.get("address") or {}).get("url") or st.get("url")
        return url.rstrip("/") + "/v1" if url else None

    async def _replica(self, pod: dict) -> dict:
        meta, spec, st = pod.get("metadata") or {}, pod.get("spec") or {}, pod.get("status") or {}
        node = spec.get("nodeName")
        labels = await self._node_labels(node) if node else {}
        node_type = labels.get(self.cluster.node_type_label) or (self.k.node_type if node else None)
        spot = any(labels.get(key, "").lower() == value for key, value in SPOT_LABELS.items()) if labels else self.serve.use_spot
        gpus = gpu.gpu_count(self.serve.accelerators)
        ports = [p.get("containerPort") for c in spec.get("containers") or [] for p in c.get("ports") or [] if p.get("containerPort")]
        ip = st.get("podIP")
        host = f"[{ip}]" if ip and ":" in ip else ip
        return {"id": meta.get("name"), "status": pod_state(pod), "infra": node or "no node yet",
                "region": labels.get("topology.kubernetes.io/region"), "zone": labels.get("topology.kubernetes.io/zone"),
                "resources": f"{gpus}x {self.k.gpu_resource}" + (f", {node_type}" if node_type else ""),
                "instance_type": node_type, "spot": bool(spot), "launched_at": unix(st.get("startTime")),
                "endpoint": f"http://{host}:{ports[0] if ports else 8000}" if ip else None,
                # On no node yet, it costs nothing (a node being added for it bills once it exists and the pod lands there).
                "usd_per_hour": node_price(self.cluster, node_type, gpus, bool(spot)) if node else 0.0}

    async def _node_labels(self, node: str) -> dict:
        """A node's labels (its type, zone, spot), when the controller may read nodes: optional, a ClusterRole. Without it,
        replicas are priced at the configured node type."""
        if node in self._nodes:
            return self._nodes[node]
        if self._nodes_forbidden:
            return {}
        try:
            obj = await self.api.call("GET", f"/api/v1/nodes/{node}", missing_ok=True)
        except KubeError as e:
            if e.status in (401, 403):
                self._nodes_forbidden = True
                log.info("%s: can't read nodes (%s); pricing replicas at the configured node type", self.backend.name, e)
            return {}
        labels = ((obj or {}).get("metadata") or {}).get("labels") or {}
        self._nodes[node] = labels
        return labels

    # ---- prices

    async def expected_price(self) -> dict | None:
        """A replica's share of its node type's configured price: a cluster has no price catalog."""
        return expected_node_price(self.cfg, self.backend)


# ------------------------------------------------------------------ the sweep


class Sweeper:
    """Every cycle, delete this install's GPU resources on the cluster (InferenceServices and llm-d Deployments labelled
    managed-by meritorder and with this install's id) that no running deployment owns: one switched off, removed or
    renamed while the controller was down, or a stop that failed. Each is logged in the GPU activity log. Another
    install's are never touched, nor anything KServe made (it has an owner)."""

    def __init__(self, install: str, api: KubeAPI | None = None):
        self.install = install
        self._api = api
        self.leftovers = False  # the last sweep found some: keep sweeping even with no Kubernetes deployment configured

    async def sweep(self, cfg: Config, store, deps: list, states: dict[str, str]) -> list[str]:
        """``states``: each controlled deployment's state after this cycle. Returns the paths deleted."""
        configured = [b for b in cfg.backends if b.serve is not None and b.serve.provisioner == "kubernetes"]
        if not (configured or self.leftovers):
            return []
        provs = [d.provisioner for d in deps if isinstance(d.provisioner, KubernetesProvisioner)]
        try:
            api = self._api or (provs[0].api if provs else KubeAPI.from_environment(cfg.kubernetes.kubeconfig, cfg.kubernetes.context))
            self._api = api
            owned = {p.path(p.name) for p in provs if states.get(p.backend.name, gpu.API_ONLY) != gpu.API_ONLY}
            spaces = {p.namespace for p in provs} | {(b.serve.kubernetes or KubernetesServe()).namespace or cfg.kubernetes.namespace
                                                     or os.environ.get("MERITORDER_GPU_NAMESPACE") or api.namespace for b in configured}
        except Exception as e:  # noqa: BLE001 (tried again next cycle)
            log.warning("Kubernetes sweep: can't reach the cluster: %s", e)
            return []
        found, done = 0, []
        selector = f"{OWNED},{INSTALL}={self.install}"
        for ns in sorted(x for x in spaces if x):
            for base in (f"{KSERVE}/namespaces/{ns}/inferenceservices", f"/apis/apps/v1/namespaces/{ns}/deployments"):
                try:
                    items = (await api.call("GET", base, params={"labelSelector": selector}, missing_ok=True) or {}).get("items") or []
                except KubeError as e:
                    log.warning("Kubernetes sweep: can't list %s in %s: %s", resource(base), ns, e)
                    continue
                for obj in items:
                    meta = obj.get("metadata") or {}
                    path = f"{base}/{meta.get('name')}"
                    if path in owned or meta.get("ownerReferences") or meta.get("deletionTimestamp"):
                        continue  # wanted, KServe's own (its predictor's Deployment), or going already
                    found += 1
                    who = (meta.get("labels") or {}).get(MANAGED) or meta.get("name")
                    log.warning("Kubernetes sweep: %s %s/%s isn't wanted by any running deployment: deleting it", resource(base), ns,
                                meta.get("name"))
                    detail = {"resource": f"{resource(base)}/{meta.get('name')}", "namespace": ns}
                    try:
                        await api.call("DELETE", path, params={"propagationPolicy": "Background"}, missing_ok=True)
                    except KubeError as e:
                        log.error("Kubernetes sweep: couldn't delete %s: %s", path, e)
                        if store is not None:
                            await store.gpu_event("error", who, "controller", {**detail, "action": "sweep", "error": str(e)[:500]})
                        continue
                    done.append(path)
                    if store is not None:
                        await store.gpu_event("sweep", who, "controller", {**detail, "reason": "no running deployment owns it"})
        self.leftovers = bool(found)
        return done
