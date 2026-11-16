"""AES-256-GCM for content the gateway holds at rest, with key rotation.

Each sealed value starts with a version byte and the id of the key that sealed it (8 bytes of a hash of the key, never
the key), then the nonce and the ciphertext. A keyring holds the current key, which seals, and older keys, which only
open, so values sealed before a rotation stay readable until they are sealed again with the current key. The caller
passes associated data that names where the value belongs (its kind, its id, its owner): a value copied into another
row, or another person's row, doesn't open.

Keys are any string, the same on every process; its SHA-256 is the AES key. An in-memory store, which keeps nothing past
the process, gets a random key for the life of the process when none is set.
"""

from __future__ import annotations

import hashlib
import os
import secrets

VERSION = 1
_HEAD = 1 + 8 + 12  # version, key id, nonce


class SealError(Exception):
    """A value that can't be opened: no key it was sealed with is configured, or it was changed or moved."""


def key_id(key: bytes) -> str:
    """A key's public name: 8 bytes of a hash of it, so sealed values say which key opens them without revealing it."""
    return hashlib.sha256(b"meritorder key id\0" + key).hexdigest()[:16]


def _derive(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()


class Keyring:
    _process_key = secrets.token_bytes(32)

    def __init__(self, current: str | None, old: list[str] | None = None, memory: bool = False):
        self.key = _derive(current) if current else (self._process_key if memory else None)
        keys = [self.key] if self.key else []
        keys += [_derive(s) for s in old or [] if s]
        self.keys = {key_id(k): k for k in keys}
        self.current = key_id(self.key) if self.key else None

    @property
    def usable(self) -> bool:
        return self.key is not None

    def seal(self, data: bytes, aad: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if self.key is None:
            raise SealError("no key to seal with")
        nonce = os.urandom(12)
        return bytes([VERSION]) + bytes.fromhex(self.current or "") + nonce + AESGCM(self.key).encrypt(nonce, data, aad)

    def sealed_with(self, blob: bytes) -> str | None:
        """The id of the key a value was sealed with."""
        return bytes(blob[1:9]).hex() if len(blob) > _HEAD and blob[0] == VERSION else None

    def open(self, blob: bytes, aad: bytes) -> bytes:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        kid = self.sealed_with(blob)
        if kid is None:
            raise SealError("not a sealed value")
        key = self.keys.get(kid)
        if key is None:
            raise SealError(f"sealed with key {kid}, which isn't configured (was the key rotated without keeping the old one?)")
        try:
            return AESGCM(key).decrypt(bytes(blob[9:21]), bytes(blob[21:]), aad)
        except InvalidTag:
            raise SealError("the value doesn't open with its key: it was changed, or belongs somewhere else") from None
