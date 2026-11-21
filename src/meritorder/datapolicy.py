"""Where a request's data goes, and a caller's policy about it (decision 72, docs/06-auth-and-spend.md#data-policy).

Each backend has a provider (who receives what's sent: anthropic, bedrock, azure_openai, vertex...) and two facts the
operator states from their agreement with it: whether it keeps no prompts or outputs (zero retention) and whether it may
train on them. Unknown stays unknown: the gateway never guesses a retention term.

A caller (Blackstart's facade, or anyone) can say, per request, which providers may receive it and whether zero
retention is required:

    X-MeritOrder-Allowed-Providers: anthropic, bedrock
    X-MeritOrder-Zero-Retention: required

Routes to other providers, or (when required) to providers not stated as zero-retention, are left out, as a team's
allowed_backends would be. If nothing is left, the request is refused with 403 ``data_policy``, naming the rule.
GET /v1/routes lists each named route's candidates with these facts, so a caller can check its policy before sending.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

PROVIDERS_HEADER = "x-meritorder-allowed-providers"
ZERO_RETENTION_HEADER = "x-meritorder-zero-retention"

# The provider behind a backend, from its address, when the config doesn't name one.
_HOSTS = [
    ("api.anthropic.com", "anthropic"), ("bedrock", "bedrock"), (".openai.azure.com", "azure_openai"),
    (".services.ai.azure.com", "foundry"), ("aiplatform.googleapis.com", "vertex"), ("generativelanguage.googleapis.com", "gemini"),
    ("api.openai.com", "openai"), ("openrouter.ai", "openrouter"), ("api.mistral.ai", "mistral"), ("api.deepseek.com", "deepseek"),
    ("api.x.ai", "xai"), ("api.moonshot", "moonshot"), ("api.z.ai", "zai"), ("api.minimax", "minimax"),
]


def provider_of(cfg) -> str:
    """Who receives what's sent to this backend: the config's ``provider``, else known from its address; your own GPUs
    are ``self_hosted``; anything else is named after the backend."""
    if getattr(cfg, "provider", None):
        return str(cfg.provider).lower()
    if cfg.kind == "self_hosted":
        return "self_hosted"
    if cfg.kind == "mock":
        return "mock"
    if getattr(cfg, "flavor", "standard") == "vertex" or getattr(cfg, "api", None) == "gemini" and cfg.project:
        return "vertex"
    host = (urlsplit(cfg.base_url or "").hostname or "").lower()
    return next((name for part, name in _HOSTS if part in host), cfg.name)


def facts(cfg) -> dict:
    """The data facts for a backend, as GET /v1/routes shows them: provider, zero retention and training (None when
    unknown), and where they come from."""
    if cfg.kind in ("self_hosted", "mock"):
        return {"provider": provider_of(cfg), "zero_retention": True, "trains_on_data": False, "source": "self_hosted"}
    stated = cfg.zero_retention is not None or cfg.trains_on_data is not None
    return {"provider": provider_of(cfg), "zero_retention": cfg.zero_retention, "trains_on_data": cfg.trains_on_data,
            "source": "config" if stated else "unknown"}


@dataclass(frozen=True)
class Policy:
    providers: frozenset[str] | None  # None: any provider
    zero_retention: bool

    def refuses(self, cfg) -> str | None:
        """Why this backend can't take the request under the policy, or None."""
        f = facts(cfg)
        if self.providers is not None and f["provider"] not in self.providers:
            return f"provider {f['provider']} isn't in {PROVIDERS_HEADER}"
        if self.zero_retention and f["zero_retention"] is not True:
            known = "keeps data" if f["zero_retention"] is False else "isn't stated as zero-retention"
            return f"provider {f['provider']} {known} and {ZERO_RETENTION_HEADER} is required"
        return None


def from_headers(headers: dict) -> Policy | None:
    raw = headers.get(PROVIDERS_HEADER)
    providers = frozenset(p.strip().lower() for p in raw.split(",") if p.strip()) if raw is not None else None
    zero = (headers.get(ZERO_RETENTION_HEADER) or "").strip().lower() in ("required", "true", "1", "yes")
    if providers is None and not zero:
        return None
    return Policy(providers=providers, zero_retention=zero)


def refusal(policy: Policy, why: dict[str, str]) -> str:
    """The 403's message: the rule, and what each route was refused for."""
    detail = "; ".join(f"{b}: {w}" for b, w in list(why.items())[:8])
    return f"data_policy: no route for this request meets the caller's data policy ({detail})"
