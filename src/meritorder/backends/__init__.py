"""Backend adapters: pass-through first, translation only when formats differ."""

from __future__ import annotations

import logging

from ..config import Config
from .auth import ReplicaKeyAuth, make_auth
from .base import Backend, BackendError, Outage, UpstreamResponse, describe
from .http import AnthropicBackend, GeminiBackend, OpenAIBackend
from .mock import MockBackend

log = logging.getLogger(__name__)

__all__ = ["Backend", "BackendError", "Outage", "UpstreamResponse", "build_backends", "describe"]


def build_backends(cfg: Config) -> dict[str, Backend]:
    out: dict[str, Backend] = {}
    for bc in cfg.backends:
        if not bc.enabled:
            continue
        if bc.kind == "mock":
            out[bc.name] = MockBackend(bc)
            continue
        try:
            # A SkyPilot deployment's replicas let in only the deployment's key, which every gateway works out itself.
            replicas = bc.kind == "self_hosted" and bc.serve is not None and bc.serve.provisioner == "skypilot"
            auth = ReplicaKeyAuth(bc.name) if replicas else make_auth(bc.auth)
        except ImportError as e:
            log.warning("backend %s disabled: missing optional dependency (%s)", bc.name, e)
            bc.enabled = False
            bc.disabled_reason = f"missing optional dependency ({e})"
            continue
        except Exception as e:
            log.warning("backend %s disabled: credentials unavailable (%s)", bc.name, e)
            bc.enabled = False
            bc.disabled_reason = f"credentials unavailable ({type(e).__name__}: {e})"[:200]
            continue
        if not auth.available():
            log.info("backend %s disabled: no credentials found", bc.name)
            bc.enabled = False
            bc.disabled_reason = "no credentials found"
            continue
        cls = AnthropicBackend if bc.format == "anthropic" else GeminiBackend if bc.api == "gemini" else OpenAIBackend
        out[bc.name] = cls(bc, auth)
    return out
