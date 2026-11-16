"""How the gateway authenticates to each provider."""

from __future__ import annotations

import asyncio
import os
import time

from ..config import BackendAuth


class BackendAuthProvider:
    def available(self) -> bool:
        return True

    async def apply(self, method: str, url: str, headers: dict[str, str], body: bytes) -> dict[str, str]:
        return headers


class ApiKeyAuth(BackendAuthProvider):
    def __init__(self, spec: BackendAuth):
        self.header = spec.header.lower()
        self.prefix = spec.prefix if spec.prefix is not None else ("Bearer " if self.header == "authorization" else "")
        self.spec = spec

    def available(self) -> bool:
        return bool(self.spec.key())

    async def apply(self, method, url, headers, body):
        return {**headers, self.header: f"{self.prefix}{self.spec.key()}"}


class SigV4Auth(BackendAuthProvider):
    """AWS SigV4 request signing via botocore (used for Claude in Amazon Bedrock, Cost Explorer, and S3 for the CUR
    export: S3 also wants the payload's hash in a header, which botocore's S3 signer adds)."""

    def __init__(self, spec: BackendAuth):
        from botocore.session import Session

        self.service = spec.service or "bedrock-mantle"
        self.region = spec.region or "us-east-1"
        self.session = Session(profile=spec.profile or None)
        try:
            self.credentials = self.session.get_credentials()
        except Exception:
            self.credentials = None

    def available(self) -> bool:
        return self.credentials is not None

    async def apply(self, method, url, headers, body):
        from botocore.auth import S3SigV4Auth
        from botocore.auth import SigV4Auth as BotoSigV4
        from botocore.awsrequest import AWSRequest

        frozen = await asyncio.to_thread(self.credentials.get_frozen_credentials)
        request = AWSRequest(method=method, url=url, data=body, headers=dict(headers))
        (S3SigV4Auth if self.service == "s3" else BotoSigV4)(frozen, self.service, self.region).add_auth(request)
        return dict(request.headers.items())


class AzureADAuth(BackendAuthProvider):
    """Entra ID token for Azure OpenAI / Foundry (needs the ``azure`` extra)."""

    def __init__(self, spec: BackendAuth):
        from azure.identity.aio import DefaultAzureCredential

        self.scope = spec.scope or "https://cognitiveservices.azure.com/.default"
        self.credential = DefaultAzureCredential()
        self._token: str | None = None
        self._expires = 0.0

    async def apply(self, method, url, headers, body):
        if not self._token or time.time() > self._expires - 120:
            token = await self.credential.get_token(self.scope)
            self._token, self._expires = token.token, float(token.expires_on)
        return {**headers, "authorization": f"Bearer {self._token}"}


class GoogleAuth(BackendAuthProvider):
    """Application Default Credentials for Vertex AI (needs the ``gcp`` extra)."""

    def __init__(self, spec: BackendAuth):
        import google.auth

        self.credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    def _refresh(self) -> None:
        from google.auth.transport.requests import Request

        self.credentials.refresh(Request())

    async def apply(self, method, url, headers, body):
        if not self.credentials.valid:
            await asyncio.to_thread(self._refresh)
        return {**headers, "authorization": f"Bearer {self.credentials.token}"}


class ReplicaKeyAuth(BackendAuthProvider):
    """The deployment's key, for a SkyPilot deployment's replicas: worked out from this install's secret (skypilot.py),
    which the gateway loads from the store with the controller's records. Until it's loaded nothing is added, and the
    replicas refuse the request (401)."""

    def __init__(self, backend: str):
        self.backend = backend

    async def apply(self, method, url, headers, body):
        from ..skypilot import known_identity

        ident = known_identity()
        return {**headers, "authorization": f"Bearer {ident.api_key(self.backend)}"} if ident else headers


def make_auth(spec: BackendAuth) -> BackendAuthProvider:
    if spec.type == "api_key":
        return ApiKeyAuth(spec)
    if spec.type == "aws_sigv4":
        # Like the official SDKs: a Bedrock bearer token takes priority over the AWS credential chain.
        if (spec.service or "bedrock-mantle") == "bedrock-mantle" and os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            return ApiKeyAuth(BackendAuth(type="api_key", header="x-api-key", value_env="AWS_BEARER_TOKEN_BEDROCK"))
        return SigV4Auth(spec)
    if spec.type == "azure_ad":
        return AzureADAuth(spec)
    if spec.type == "google":
        return GoogleAuth(spec)
    return BackendAuthProvider()
