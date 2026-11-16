"""What differs between the clouds SkyPilot launches on, as far as Merit Order's replicas care (SkyPilot 0.14.2).

Every replica relies on the same chain (docs/05-gpu-autoscaling.md#which-clouds): the gate stops vLLM when its lease
runs out, the task ends, and SkyPilot's skylet on the machine deletes it (autostop with ``down``) using the cloud
credentials SkyPilot put there at launch. This table holds only what makes a cloud behave differently in that chain,
read from SkyPilot 0.14.2's source (``sky/clouds/<cloud>.py`` and ``sky/provision/<cloud>/``):

- **Clouds Merit Order can't use.** A replica needs Docker images (``image_id: docker:``, for the vLLM image) and
  autodown (the machine deleting itself). A cloud without either is refused when the config loads.
- **How a replica is reached.** Most clouds give the machine a public IP and open the gate's port (``direct``). On some
  the address is a mapped port SkyPilot looks up (``endpoint``: Kubernetes services, RunPod's port mappings). Where
  SkyPilot can't open a port at all, the task asks for none, and the controller reaches the gate over SSH (``tunnel``).
- **What the machine holds to delete itself.** On clouds without instance identities, SkyPilot copies an API key file
  to every machine, and the skylet deletes the machine with it: a key for the whole account, readable by anyone who gets
  onto the machine. The config load warns about each, so you can give SkyPilot a separate, revocable key.
- **Spot machines that stop.** A preempted spot machine on these clouds is stopped (its disk still billing), not
  deleted, and a stopped machine runs no skylet: only the controller deletes it.

SkyPilot's names are used throughout (``sky check``'s, lowercase), and the ``infra:`` values Merit Order writes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

SKYPILOT_VERSION = "0.14.2"

Reach = Literal["direct", "endpoint", "tunnel"]


@dataclass(frozen=True)
class Cloud:
    name: str  # SkyPilot's name for it (``sky check``, ``infra:``)
    extra: str | None = None  # its pip extra, skypilot[<extra>]
    unusable: str | None = None  # why a replica can't run there (None: it can)
    reach: Reach = "direct"
    # The credential file SkyPilot copies to the machine and the skylet deletes it with, when that's a long-lived key
    # (rather than an instance identity), and what it can do.
    key_on_vm: str | None = None
    spot: bool = False  # SkyPilot offers spot (preemptible) machines there
    spot_stops: bool = False  # a preempted spot machine is stopped (its disk billing) rather than deleted


_NO_DOCKER = "SkyPilot 0.14.2 can't run a Docker image there (image_id: docker:), which the vLLM image needs"
_NO_AUTODOWN = ("SkyPilot 0.14.2 can't make a machine there delete itself (autodown), so a replica whose controller is "
                "gone would bill until someone deleted it")

CLOUDS: dict[str, Cloud] = {c.name: c for c in (
    # The VM deletes itself with the skypilot-v1 IAM role SkyPilot attaches (EC2 and S3 full access). With static keys in
    # ~/.aws/credentials, SkyPilot also copies that file to the machine unless aws.remote_identity says otherwise.
    Cloud("aws", "aws", spot=True,
          key_on_vm="~/.aws/credentials, when the controller uses static keys (the machine deletes itself with the "
                    "skypilot-v1 IAM role; set aws.remote_identity: NO_UPLOAD in ~/.sky/config.yaml to keep the keys off it)"),
    # The skypilot-v1 service account is attached (compute and storage admin on the project), but SkyPilot also copies
    # your gcloud credentials, which the skylet uses first.
    Cloud("gcp", "gcp", spot=True,
          key_on_vm="~/.config/gcloud (your gcloud login or service-account key, which the machine deletes itself with; set "
                    "gcp.remote_identity: SERVICE_ACCOUNT in ~/.sky/config.yaml to use the skypilot-v1 service account instead)"),
    # SkyPilot's Azure client is the CLI credential only, so the machine deletes itself with your copied `az login`.
    Cloud("azure", "azure", spot=True, spot_stops=True,
          key_on_vm="~/.azure (your `az login` token cache: every tenant and subscription that login reaches; log the "
                    "controller in with a service principal limited to one subscription instead)"),
    Cloud("lambda", "lambda", key_on_vm="~/.lambda_cloud/lambda_keys (an API key for the whole Lambda account)"),
    Cloud("runpod", "runpod", reach="endpoint", spot=True, key_on_vm="~/.runpod/config.toml (a RunPod API key)"),
    Cloud("nebius", "nebius", spot=True, spot_stops=True,
          key_on_vm="~/.nebius/credentials.json (a Nebius service-account key)"),
    Cloud("kubernetes", "kubernetes", reach="endpoint"),  # deletes its pod with SkyPilot's service account in its namespace
    Cloud("ssh", "ssh", reach="endpoint"),  # SSH node pools: SkyPilot's Kubernetes on your machines
    Cloud("vast", "vast", reach="tunnel", spot=True, key_on_vm="~/.config/vastai/vast_api_key (a Vast.ai API key)"),
    Cloud("fluidstack", "fluidstack", key_on_vm="~/.fluidstack/api_key (a Fluidstack API key)"),
    Cloud("paperspace", "paperspace", key_on_vm="~/.paperspace/config.json (a Paperspace API key)"),
    Cloud("do", "do", key_on_vm="~/.config/doctl/config.yaml (a DigitalOcean API token)"),
    Cloud("mithril", "mithril", reach="tunnel", spot=True, key_on_vm="~/.config/mithril/config.yaml (a Mithril API key)"),
    Cloud("seeweb", "seeweb", key_on_vm="~/.seeweb_cloud/seeweb_keys (a Seeweb API key)"),
    Cloud("slurm", "slurm", reach="tunnel"),  # your own cluster: autodown needs Pyxis containers; no ports
    Cloud("oci", "oci", unusable=_NO_DOCKER),
    Cloud("cudo", "cudo", unusable=_NO_DOCKER),
    Cloud("shadeform", "shadeform", unusable=_NO_DOCKER),
    Cloud("scp", "scp", unusable=_NO_DOCKER),
    Cloud("vsphere", "vsphere", unusable=_NO_DOCKER),
    Cloud("ibm", "ibm", unusable=_NO_DOCKER),
    Cloud("hyperbolic", "hyperbolic", unusable=_NO_AUTODOWN),
    Cloud("primeintellect", "primeintellect", unusable=_NO_AUTODOWN),
    Cloud("verda", "verda", unusable=_NO_AUTODOWN),
    Cloud("yotta", "yotta", unusable=_NO_AUTODOWN),
    Cloud("coreweave", "coreweave", unusable="CoreWeave runs through SkyPilot's kubernetes cloud: use clouds: [kubernetes] "
                                             "with your CoreWeave cluster's kubeconfig context"),
)}
ALIASES = {"digitalocean": "do", "k8s": "kubernetes", "lambda_cloud": "lambda", "lambdalabs": "lambda"}
# What the -gpu image installs by default (the Dockerfile's SKYPILOT_EXTRAS): every cloud a replica can run on.
DEFAULT_EXTRAS = tuple(c.extra for c in CLOUDS.values() if not c.unusable and c.extra and c.name not in ("ssh", "slurm"))


def cloud(name: str | None) -> Cloud | None:
    """The table's entry for a cloud as SkyPilot or a config names it (any case), or None for one it doesn't list."""
    if not name:
        return None
    n = str(name).lower()
    return CLOUDS.get(ALIASES.get(n, n))


def unusable(clouds: list[str]) -> list[str]:
    """Why each of these clouds can't run a replica, one line per cloud that can't."""
    return [f"{name}: {c.unusable}" for name in clouds if (c := cloud(name)) and c.unusable]


def reach(name: str | None) -> Reach:
    c = cloud(name)
    return c.reach if c else "direct"


def opens_ports(name: str | None) -> bool:
    """Whether the task asks SkyPilot to open the gate's port there: not where it can't (asking would rule the cloud out)."""
    return reach(name) != "tunnel"


# ------------------------------------------------------------------ volumes (serve.weights_cache: volume)

# Where SkyPilot 0.14.2 has volumes (sky/utils/volume.py, VolumeType; sky/volumes/volume.py, VOLUME_TYPE_TO_CLOUD):
# a PersistentVolumeClaim on Kubernetes and a network volume on RunPod. Nowhere else: AWS, GCP and Azure have only the
# disks a machine is launched with (resources.volumes, deleted with it unless attached by hand), and SkyPilot can't
# attach a Lambda persistent filesystem (its Lambda launch sends none: sky/provision/lambda_cloud/instance.py).
VOLUME_TYPES = {"kubernetes": "k8s-pvc", "runpod": "runpod-network-volume"}
RUNPOD_MIN_VOLUME_GB = 10  # sky/utils/volume.py, MIN_RUNPOD_NETWORK_VOLUME_SIZE_GB
# RunPod's network volume price, per GB a month: $0.07 for the first TB, $0.05 beyond (docs.runpod.io/pods/storage/types,
# November 2026).
RUNPOD_USD_PER_GB_MONTH = (0.07, 0.05)


def volume_infra(infra: str | None) -> tuple[str, str | None, str | None]:
    """(cloud, region, zone) of a SkyPilot infra string: "runpod/CA/CA-MTL-1" -> ("runpod", "CA", "CA-MTL-1")."""
    parts = (infra or "").strip().strip("/").split("/", 2)
    c = cloud(parts[0])
    return (c.name if c else parts[0].lower(), parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None)


def volume_problem(provisioner: str, infra: str | None, size_gb: int | None, clouds: list[str], regions: list[str]) -> str | None:
    """Why a weights cache volume can't be had with these settings, or None when it can."""
    if provisioner != "skypilot":
        return "only SkyPilot deployments can have one (on Kubernetes through KServe or llm-d, cache the weights with your cluster's own storage)"
    if not infra:
        return "weights_cache_infra must say where the volume lives (kubernetes, kubernetes/<context>, or runpod/<region>/<data center>)"
    where, region, zone = volume_infra(infra)
    if where not in VOLUME_TYPES:
        return (f"SkyPilot {SKYPILOT_VERSION} has volumes only on {' and '.join(VOLUME_TYPES)}, not on {where}: on Lambda, AWS, GCP, "
                "Azure and the others, a start downloads the weights each time (docs/05, \"Faster starts\")")
    if where == "runpod" and not zone:
        return "a RunPod network volume lives in one data center: weights_cache_infra must be runpod/<region>/<data center>"
    if not size_gb:
        return "weights_cache_gb must say how big the volume is (the model's weights and some room; you pay for every GB each month)"
    if where == "runpod" and size_gb < RUNPOD_MIN_VOLUME_GB:
        return f"a RunPod network volume has at least {RUNPOD_MIN_VOLUME_GB} GB"
    if [cloud(c).name if cloud(c) else c.lower() for c in clouds] != [where]:
        return (f"its replicas can run only where the volume is: set clouds to [{where}] (SkyPilot refuses a task whose volume "
                "and resources are in different places)")
    if region and regions and regions != [region]:
        return f"its replicas can run only where the volume is: set regions to [{region}], or leave them out"
    return None


def volume_usd_per_month(infra: str | None, size_gb: int | None, usd_per_gb_month: float | None) -> float | None:
    """What the volume costs a month: at the configured price, else RunPod's published one; None when neither is known
    (a Kubernetes PVC costs what your storage class does)."""
    if not size_gb:
        return None
    if usd_per_gb_month is not None:
        return round(size_gb * usd_per_gb_month, 2)
    if volume_infra(infra)[0] == "runpod":
        first, beyond = RUNPOD_USD_PER_GB_MONTH
        return round(min(size_gb, 1000) * first + max(size_gb - 1000, 0) * beyond, 2)
    return None


def key_path(c: Cloud) -> str | None:
    """The credential file of ``key_on_vm``, as a path under ~."""
    return c.key_on_vm.split(" ")[0].rstrip(",") if c.key_on_vm else None


def _found(home: str, c: Cloud) -> bool:
    path = key_path(c)
    return bool(path) and os.path.exists(os.path.join(home, path.removeprefix("~/")))


def warnings(clouds: list[str], use_spot: bool, home: str | None = None) -> list[str]:
    """What to know before launching on these clouds, judged by the credential files in ``home`` (the controller's):

    - each long-lived key SkyPilot copies to the machines of a cloud they may run on (empty ``clouds``: any), which
      they delete themselves with;
    - the keys of other clouds, which SkyPilot 0.14.2 copies to every machine too, whatever cloud it runs on
      (``sky/check.py``, ``get_cloud_credential_file_mounts``), unless ~/.sky/config.yaml says not to;
    - with spot, the clouds where a preempted machine is stopped (its disk billing) instead of deleted.
    """
    home = home or os.path.expanduser("~")
    allowed = [c for n in clouds if (c := cloud(n)) and not c.unusable] if clouds else [c for c in CLOUDS.values() if not c.unusable]
    found = [c for c in CLOUDS.values() if c.key_on_vm and _found(home, c)]
    out = [f"{c.name}: SkyPilot copies {c.key_on_vm} to each replica, which deletes itself with it. Give SkyPilot a separate "
           "key you can revoke, for this alone" for c in found if c in allowed]
    others = [c for c in found if c not in allowed]
    if others:
        out.append("SkyPilot also copies " + ", ".join(key_path(c) for c in others) + " to every replica, whatever cloud it runs on: "
                   "set remote_identity: NO_UPLOAD for " + ", ".join(c.name for c in others) + " in ~/.sky/config.yaml to keep "
                   "them off")
    if use_spot:
        out += [f"{c.name}: a preempted spot machine is stopped, not deleted, and its disk bills until the controller tears it "
                "down (a stopped machine can't delete itself)" for c in allowed if c.spot_stops and c.spot]
    return out
