# syntax=docker/dockerfile:1.7
FROM node:26.13.0-slim AS ui
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ui/.npmrc ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Extras: OpenTelemetry, plus Google and Azure identity so Vertex and Azure OpenAI/Foundry work keyless
# (Workload Identity / managed identity), and the KV-event subscriber the GPU controller uses (pyzmq, msgpack).
# Trim with e.g. --build-arg EXTRAS="--extra otel". The compressor image (`-compressor` tag) is
# --build-arg EXTRAS="--extra otel --extra compressor" --build-arg SKYPILOT="": llmlingua, torch and transformers, which
# stay out of every other image.
ARG EXTRAS="--extra otel --extra gcp --extra azure --extra kv-events"
# LICENSE and NOTICE go into the installed package's metadata (.dist-info/licenses), so every image carries them.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project ${EXTRAS}
COPY src ./src
COPY --from=ui /app/src/meritorder/ui_dist ./src/meritorder/ui_dist
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable ${EXTRAS}
# llmlingua loads tiktoken's cl100k_base encoding when it starts: fetch it now, so the compressor needs no internet for it.
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken
RUN mkdir -p /app/tiktoken && if /app/.venv/bin/python -c "import llmlingua" 2>/dev/null; then \
      /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"; fi
# The GPU controller image (`-gpu` tag) adds SkyPilot in its own environment, so its dependencies never touch the
# gateway's: --build-arg SKYPILOT=0.14.2. SKYPILOT_EXTRAS are the clouds it can launch on: by default every cloud a
# SkyPilot replica can run on (src/meritorder/sky_clouds.py, DEFAULT_EXTRAS; docs/05-gpu-autoscaling.md#which-clouds).
# That's about 2.0 GB installed, 1 GB of it Azure's CLI; --build-arg SKYPILOT_EXTRAS=aws,gcp,kubernetes,lambda is about
# 0.8 GB. pip rather than uv here: Azure's CLI pins pre-release Azure SDKs, which pip accepts when a pin names one.
ARG SKYPILOT=""
ARG SKYPILOT_EXTRAS="aws,gcp,azure,lambda,runpod,nebius,kubernetes,vast,fluidstack,paperspace,do,mithril,seeweb"
RUN --mount=type=cache,target=/root/.cache/pip mkdir -p /opt/skypilot && if [ -n "$SKYPILOT" ]; then \
      python -m venv /opt/skypilot && /opt/skypilot/bin/pip install --disable-pip-version-check \
        "skypilot[${SKYPILOT_EXTRAS}]==${SKYPILOT}"; fi

FROM python:3.12-slim
# The release workflow sets the same labels (and the version and revision) through docker/metadata-action.
LABEL org.opencontainers.image.title="meritorder" \
      org.opencontainers.image.source="https://github.com/Merit-Order/meritorder" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.vendor="The Merit Order Authors"
# The model cache and SkyPilot's state exist (owned by the user) so a volume mounted there, as Compose's compressor and
# controller do, is writable.
RUN useradd --create-home --uid 10001 meritorder \
    && install -d -o meritorder -g meritorder /home/meritorder/.cache/huggingface /home/meritorder/.sky
# With SkyPilot (the -gpu image): the ssh and rsync clients it sets its clusters up with, which the controller also opens
# an SSH tunnel with to a replica whose port it can't reach (serve.ssh_tunnel).
ARG SKYPILOT=""
RUN if [ -n "$SKYPILOT" ]; then apt-get update && apt-get install -y --no-install-recommends openssh-client rsync \
      && rm -rf /var/lib/apt/lists/*; fi
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/tiktoken /app/tiktoken
COPY --from=build /opt/skypilot /opt/skypilot
COPY config /app/config
ENV PATH="/app/.venv/bin:$PATH:/opt/skypilot/bin" \
    MERITORDER_SKY_PYTHON=/opt/skypilot/bin/python \
    TIKTOKEN_CACHE_DIR=/app/tiktoken \
    PYTHONUNBUFFERED=1 \
    MERITORDER_CONFIG=/app/config/meritorder.yaml
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"
CMD ["meritorder", "serve", "--host", "0.0.0.0", "--port", "8080"]
