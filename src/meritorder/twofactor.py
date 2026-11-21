"""Two-factor sign-in for local accounts (decision 73): authenticator-app codes (TOTP, RFC 6238) and recovery codes.

* Codes are six digits for 30-second steps (SHA-1, as every authenticator app expects), accepted one step either side
  for clock drift, and each step's code works once per person (kept in Redis for two minutes).
* The authenticator's secret has to be read back to check a code, so it can't be hashed like a password. It's sealed
  (AES-256-GCM, sealing.py) with ``MERITORDER_AUTH_KEY`` when that's set, tied to its person, and stored as it is
  otherwise; then someone who can read the database also needs the person's password, which stays hashed.
* Ten recovery codes, each usable once, are shown when two-factor is turned on and stored only as hashes.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import struct
import time
from urllib.parse import quote

from .sealing import Keyring, SealError

DIGITS, STEP, DRIFT = 6, 30, 1
RECOVERY_CODES = 10
ISSUER = "Merit Order"


def new_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def code_at(secret: str, step: int) -> str:
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    digest = hmac.new(key, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(value % 10**DIGITS).zfill(DIGITS)


def matching_step(secret: str, code: str, now: float | None = None) -> int | None:
    """The 30-second step this code belongs to (within the drift allowed), or None."""
    code = "".join(ch for ch in code or "" if ch.isdigit())
    if len(code) != DIGITS:
        return None
    step = int((time.time() if now is None else now) // STEP)
    for s in range(step - DRIFT, step + DRIFT + 1):
        if hmac.compare_digest(code_at(secret, s), code):
            return s
    return None


def uri(secret: str, account: str) -> str:
    """The otpauth:// link authenticator apps read (from a QR code, or opened directly on a phone or a Mac)."""
    label = quote(f"{ISSUER}:{account}")
    return f"otpauth://totp/{label}?secret={secret}&issuer={quote(ISSUER)}&algorithm=SHA1&digits={DIGITS}&period={STEP}"


def _keyring(memory: bool) -> Keyring:
    return Keyring(os.environ.get("MERITORDER_AUTH_KEY"), memory=memory)


def seal(secret: str, user_id: str, memory: bool = False) -> str:
    ring = _keyring(memory)
    if not ring.usable:
        return "plain:" + secret
    return "sealed:" + base64.b64encode(ring.seal(secret.encode(), f"totp:{user_id}".encode())).decode()


def unseal(stored: str, user_id: str, memory: bool = False) -> str:
    if stored.startswith("plain:"):
        return stored[6:]
    if stored.startswith("sealed:"):
        return _keyring(memory).open(base64.b64decode(stored[7:]), f"totp:{user_id}".encode()).decode()
    raise SealError("not a stored authenticator secret")


def new_recovery_codes() -> list[str]:
    """Ten codes like ``k7qp-3mzd-x2vh``: 60 bits each, easy to read out and type."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    return ["-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3)) for _ in range(RECOVERY_CODES)]


def hash_recovery(code: str) -> str:
    normal = "".join(ch for ch in (code or "").lower() if ch.isalnum())
    return hashlib.sha256(b"meritorder recovery code\0" + normal.encode()).hexdigest()


def looks_like_recovery(code: str) -> bool:
    return sum(ch.isalpha() for ch in code or "") > 0
