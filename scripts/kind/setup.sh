#!/usr/bin/env bash
# The Kubernetes GPU provisioner on a kind cluster without GPUs (docs/05-gpu-autoscaling.md#checked-on-kind):
# KServe in Standard mode, the Gateway API Inference Extension's InferencePool for llm-d, a vLLM stand-in
# (scripts/fake_vllm.py) on a node that advertises fake nvidia.com/gpu, and Merit Order from its Helm chart with the
# controller in the cluster. Then `python scripts/kind/check.py` runs the checks.
#
#   scripts/kind/setup.sh            # create the cluster and install everything
#   scripts/kind/setup.sh down       # delete the cluster and the images it built
#
# It writes its own kubeconfig ($KUBECONFIG_FILE) and never touches ~/.kube/config. Needs docker, kind, kubectl, helm.
set -euo pipefail
cd "$(dirname "$0")/../.."

CLUSTER=${CLUSTER:-meritorder-kind}
export KUBECONFIG=${KUBECONFIG_FILE:-${TMPDIR:-/tmp}/$CLUSTER.kubeconfig}
NODE_IMAGE=${NODE_IMAGE:-}                   # empty: kind's default
REPO=$CLUSTER-meritorder                     # Merit Order, built without SkyPilot, tagged by its image id
FAKE=$CLUSTER-fake-vllm:dev                  # the vLLM stand-in (FAKE_VLLM_IMAGE in scripts/kind/meritorder.yaml)
CERT_MANAGER=v1.21.3
KSERVE=v0.21.0
GIE=v1.6.2                                   # Gateway API Inference Extension: the InferencePool CRD llm-d uses
GPUS=8                                       # fake nvidia.com/gpu the node advertises

if [[ "${1:-}" == down ]]; then
  kind delete cluster --name "$CLUSTER" --kubeconfig "$KUBECONFIG"
  docker image rm $(docker images --format '{{.Repository}}:{{.Tag}}' "$REPO") "$FAKE" 2>/dev/null || true
  rm -f "$KUBECONFIG"
  exit 0
fi

echo "--- cluster $CLUSTER (kubeconfig $KUBECONFIG)"
kind get clusters | grep -qx "$CLUSTER" ||
  kind create cluster --name "$CLUSTER" --kubeconfig "$KUBECONFIG" ${NODE_IMAGE:+--image "$NODE_IMAGE"} --wait 120s
NODE=$CLUSTER-control-plane
# A node type with a price in scripts/kind/meritorder.yaml, a zone, and GPUs to schedule on (an extended resource: the
# kubelet keeps it, and the scheduler counts it like a device plugin's).
kubectl label node "$NODE" --overwrite node.kubernetes.io/instance-type=g6.12xlarge topology.kubernetes.io/zone=kind-a
kubectl patch node "$NODE" --subresource=status --type=json \
  -p "[{\"op\":\"add\",\"path\":\"/status/capacity/nvidia.com~1gpu\",\"value\":\"$GPUS\"}]"

echo "--- images: Merit Order (no SkyPilot) and the vLLM stand-in"
docker build -q --build-arg SKYPILOT= -t "$REPO:dev" .
# Its own tag per build, so the chart's pods roll when the code changed (the cluster never pulls these images).
IMAGE=$REPO:$(docker image inspect --format '{{.Id}}' "$REPO:dev" | cut -c8-19)
docker tag "$REPO:dev" "$IMAGE"
docker build -q -f scripts/kind/Dockerfile.fake-vllm -t "$FAKE" .
kind load docker-image --name "$CLUSTER" "$IMAGE" "$FAKE" postgres:16-alpine redis:7-alpine

echo "--- cert-manager $CERT_MANAGER, KServe $KSERVE (Standard mode), InferencePool CRDs $GIE"
kubectl apply -f "https://github.com/cert-manager/cert-manager/releases/download/$CERT_MANAGER/cert-manager.yaml"
kubectl -n cert-manager rollout status deploy/cert-manager-webhook --timeout=180s
# KServe's own charts: the minimal CRDs (kserve.yaml's full ones, with LLMInferenceService, need more memory than a small
# kind node has for its API server) and the InferenceService controller, Standard mode by default.
KSERVE_CHARTS=https://github.com/kserve/kserve/releases/download/$KSERVE
helm upgrade --install kserve-crd "$KSERVE_CHARTS/helm-chart-kserve-crd-minimal-$KSERVE.tgz" -n kserve --create-namespace --wait
helm upgrade --install kserve "$KSERVE_CHARTS/helm-chart-kserve-resources-$KSERVE.tgz" -n kserve \
  --set kserve.controller.deploymentMode=Standard --wait --timeout 5m
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$GIE/manifests.yaml"

echo "--- the GPU namespace: an InferencePool, and a Service standing in for llm-d's Gateway"
kubectl create namespace gpus --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f - <<'EOF'
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata: {name: tiny-pool, namespace: gpus}
spec:
  selector: {matchLabels: {app: tiny-coder}}
  targetPorts: [{number: 8000}]
  endpointPickerRef: {name: tiny-pool-epp, port: {number: 9002}}
---
# llm-d's Gateway and endpoint picker aren't installed: this Service sends the Gateway's traffic to the pool's pods.
apiVersion: v1
kind: Service
metadata: {name: tiny-gateway, namespace: gpus}
spec:
  selector: {app: tiny-coder}
  ports: [{port: 80, targetPort: 8000}]
EOF

echo "--- Postgres and Redis for Merit Order, then its chart"
kubectl create namespace meritorder --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n meritorder -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata: {name: postgres}
spec:
  selector: {matchLabels: {app: postgres}}
  template:
    metadata: {labels: {app: postgres}}
    spec:
      containers:
        - name: postgres
          image: postgres:16-alpine
          env: [{name: POSTGRES_PASSWORD, value: kind}, {name: POSTGRES_DB, value: meritorder}]
          resources: {limits: {memory: 256Mi}}
---
apiVersion: v1
kind: Service
metadata: {name: postgres}
spec: {selector: {app: postgres}, ports: [{port: 5432}]}
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: redis}
spec:
  selector: {matchLabels: {app: redis}}
  template:
    metadata: {labels: {app: redis}}
    spec:
      containers: [{name: redis, image: redis:7-alpine, resources: {limits: {memory: 64Mi}}}]
---
apiVersion: v1
kind: Service
metadata: {name: redis}
spec: {selector: {app: redis}, ports: [{port: 6379}]}
EOF
kubectl -n meritorder rollout status deploy/postgres deploy/redis --timeout=120s
# Helm 4 applies server-side: a second run takes the controller's arguments back from the patch below.
FORCE=$(helm version --template '{{.Version}}' | grep -q '^v3' || echo --force-conflicts)
helm upgrade --install mo charts/meritorder -n meritorder -f scripts/kind/values.yaml $FORCE \
  --set image.repository="${IMAGE%:*}" --set image.tag="${IMAGE##*:}" --set env.FAKE_VLLM_IMAGE="$FAKE" \
  --set-file 'configFiles.meritorder\.yaml=scripts/kind/meritorder.yaml' --wait --timeout 5m
# A 10-second controller cycle instead of 60, so the checks don't wait a minute for each step.
kubectl -n meritorder patch deploy mo-meritorder-controller --type=json \
  -p '[{"op":"replace","path":"/spec/template/spec/containers/0/args","value":["meritorder","controller","--interval","10"]}]'
kubectl -n meritorder rollout status deploy/mo-meritorder-controller --timeout=120s
kubectl -n meritorder get pods
echo "--- ready: python scripts/kind/check.py (KUBECONFIG=$KUBECONFIG)"
