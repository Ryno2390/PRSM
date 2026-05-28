"""Sprint 851 — Coinbase CDP WaaS HTTP backend.

Implements the ``_WaasBackend`` Protocol used by
``CoinbaseWaaSClient`` against Coinbase Developer Platform's v2 EVM
smart-account creation endpoint
(https://docs.cdp.coinbase.com/api-v2/docs/welcome).

CDP v2 authentication uses an Ed25519-signed JWT per request:

  Header: {"alg": "EdDSA", "typ": "JWT",
           "kid": "<api_key_name>",
           "nonce": "<random 16-hex>"}
  Payload: {"iss": "cdp", "sub": "<api_key_name>",
            "nbf": <now>, "exp": <now+120>,
            "aud": ["<METHOD>", "<host>"],
            "uri": "<METHOD> <host><path>"}

Signed with the operator's Ed25519 PEM private key
(COINBASE_CDP_API_KEY_PRIVATE env var). PyJWT 2.x supports Ed25519
via algorithm="EdDSA" + a `cryptography` Ed25519PrivateKey object.

Constructor validates the PEM at instantiation — placeholder strings
(e.g., "REPLACE_WITH...") fail parsing and surface as ValueError so
``from_env()`` can return None + ``CoinbaseWaaSClient.from_env()``
falls back gracefully. Operators see adapter_wired=False until the
real PEM lands.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


_CDP_V2_BASE = "https://api.cdp.coinbase.com"
_CDP_TIMEOUT_SECONDS = 30.0
_JWT_EXPIRY_SECONDS = 120


def _load_ed25519_pem(pem: str):
    """Parse an Ed25519 private key in any format CDP issues.

    Supports three forms:
      1. Standard PEM block (-----BEGIN PRIVATE KEY-----...-----END...)
      2. Raw base64 64-byte libsodium format (32-byte seed +
         32-byte derived public key) — CDP v2's default for new keys
      3. Raw base64 32-byte seed only

    Raises ValueError on placeholder strings or unrecognized format.
    """
    if not pem or "REPLACE_WITH" in pem:
        raise ValueError(
            "COINBASE_CDP_API_KEY_PRIVATE is empty or still a "
            "placeholder — paste the Ed25519 key from CDP."
        )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    stripped = pem.strip()

    # Form 1: standard PEM block
    if stripped.startswith("-----BEGIN"):
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key,
        )
        try:
            key = load_pem_private_key(
                stripped.encode("utf-8"), password=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Ed25519 PEM parse failed: {exc}",
            ) from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(
                f"expected Ed25519 private key, got "
                f"{type(key).__name__}"
            )
        return key

    # Forms 2 + 3: raw base64. Strip surrounding whitespace, allow
    # accidental internal whitespace (paste mangling).
    import base64
    compact = "".join(stripped.split())
    try:
        raw = base64.b64decode(compact, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Ed25519 base64 decode failed: {exc}",
        ) from exc
    if len(raw) == 64:
        # libsodium format: first 32 bytes are the seed.
        seed = raw[:32]
    elif len(raw) == 32:
        seed = raw
    else:
        raise ValueError(
            f"Ed25519 raw key must be 32 or 64 bytes after base64 "
            f"decode, got {len(raw)} bytes (input "
            f"{len(compact)} base64 chars). PEM or base64 expected."
        )
    return Ed25519PrivateKey.from_private_bytes(seed)


class CdpWaaSBackend:
    """CDP v2 WaaS / smart-account backend for CoinbaseWaaSClient."""

    def __init__(
        self,
        api_key_name: str,
        api_key_private_pem: str,
        *,
        base_url: str = _CDP_V2_BASE,
        client: Any = None,  # injected httpx.Client for tests
    ) -> None:
        if not api_key_name:
            raise ValueError("api_key_name is required")
        # Parse PEM eagerly — surfaces placeholder/malformed errors
        # at construction so from_env can detect + return None.
        self._private_key = _load_ed25519_pem(api_key_private_pem)
        self._api_key_name = api_key_name
        self._base_url = base_url.rstrip("/")
        if client is None:
            import httpx
            self._client = httpx.Client(timeout=_CDP_TIMEOUT_SECONDS)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _build_jwt(self, method: str, path: str) -> str:
        """Build a per-request Ed25519-signed JWT for CDP v2."""
        import jwt as _jwt
        host = urlparse(self._base_url).netloc
        now = int(time.time())
        headers = {
            "alg": "EdDSA",
            "typ": "JWT",
            "kid": self._api_key_name,
            "nonce": secrets.token_hex(8),
        }
        payload = {
            "iss": "cdp",
            "sub": self._api_key_name,
            "nbf": now,
            "exp": now + _JWT_EXPIRY_SECONDS,
            "aud": [method, host],
            "uri": f"{method} {host}{path}",
        }
        return _jwt.encode(
            payload,
            self._private_key,
            algorithm="EdDSA",
            headers=headers,
        )

    def _post(
        self, path: str, body: Dict[str, Any],
    ) -> Dict[str, Any]:
        token = self._build_jwt("POST", path)
        resp = self._client.post(
            f"{self._base_url}{path}",
            json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def create_wallet(
        self, user_id: str, email: str,
    ) -> Dict[str, Any]:
        """Create a smart-account on Base for this user.

        Returns dict shaped for CoinbaseWaaSClient.provision_wallet
        consumption: wallet_id, address, network.
        """
        # CDP v2 EVM smart-account endpoint. The exact path and
        # payload mirror CDP's "Create an EVM account" call;
        # network is implied by the API surface (Base mainnet).
        body = {
            "name": f"prsm-{user_id}",
            # Persona's reference-id pattern — useful for ops
            # correlation if CDP exposes user-account lookups.
            "metadata": {
                "user_id": user_id,
                "email": email,
                "publisher": "prsm-foundation",
            },
        }
        payload = self._post("/platform/v2/evm/accounts", body)
        # Expected response: {"name": ..., "address": "0x...", ...}
        account = payload.get("data") or payload  # tolerate envelope
        address = account.get("address")
        if not address:
            raise RuntimeError(
                f"CDP account create returned no address: "
                f"{payload!r}"
            )
        return {
            "wallet_id": account.get("name") or f"prsm-{user_id}",
            "address": address,
            "network": "base-mainnet",
        }


def from_env(
    *,
    api_key_name: Optional[str] = None,
    api_key_private_pem: Optional[str] = None,
    client: Any = None,
) -> Optional["CdpWaaSBackend"]:
    """Construct a CdpWaaSBackend from env, or None when missing.

    Returns None when required env is absent OR when the PEM is a
    placeholder (so CoinbaseWaaSClient.from_env() shows
    adapter_wired=False with the honest signal until the operator
    pastes the real PEM).
    """
    api_key_name = (
        api_key_name or os.environ.get("COINBASE_CDP_API_KEY_NAME")
    )
    api_key_private_pem = (
        api_key_private_pem
        or os.environ.get("COINBASE_CDP_API_KEY_PRIVATE")
    )
    if not api_key_name or not api_key_private_pem:
        return None
    try:
        return CdpWaaSBackend(
            api_key_name=api_key_name,
            api_key_private_pem=api_key_private_pem,
            client=client,
        )
    except ValueError as exc:
        # Placeholder PEM or malformed key — honest signal +
        # callers can read the warning to know what to paste.
        logger.info(
            "CdpWaaSBackend not constructed: %s", exc,
        )
        return None
