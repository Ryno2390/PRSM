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


def _load_wallet_secret(secret: str):
    """Parse CDP v2 Wallet Secret — ECDSA P-256 PKCS8 DER base64.

    The Wallet Secret authenticates the X-Wallet-Auth header attached
    to /platform/v2/evm/* wallet-management calls. Distinct from the
    Bearer JWT (signed with the Ed25519 API key) — CDP v2 uses TWO
    separate signing identities:

      Bearer Authorization   → API key (Ed25519 EdDSA)
      X-Wallet-Auth header   → Wallet Secret (ECDSA P-256 ES256)

    Operator generates the Wallet Secret via CDP dashboard
    (Server Wallets section). The wire format is the PKCS8 DER body
    base64-encoded with no PEM markers — typically ~180 chars.
    """
    if not secret or "REPLACE_WITH" in secret:
        raise ValueError(
            "COINBASE_CDP_WALLET_SECRET is empty or still a "
            "placeholder — generate one via CDP dashboard "
            "Server Wallets section."
        )
    stripped = secret.strip()
    # Standard PEM block also accepted in case CDP ships it that way
    if stripped.startswith("-----BEGIN"):
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key,
        )
        try:
            return load_pem_private_key(
                stripped.encode("utf-8"), password=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Wallet Secret PEM parse failed: {exc}",
            ) from exc
    # Default path: base64 of PKCS8 DER
    import base64
    compact = "".join(stripped.split())
    try:
        raw = base64.b64decode(compact, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Wallet Secret base64 decode failed: {exc}",
        ) from exc
    from cryptography.hazmat.primitives.serialization import (
        load_der_private_key,
    )
    try:
        key = load_der_private_key(raw, password=None)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Wallet Secret DER parse failed: {exc}",
        ) from exc
    # Must be EC (P-256) — Ed25519 would mean someone pasted the
    # API key instead of the Wallet Secret.
    from cryptography.hazmat.primitives.asymmetric.ec import (
        EllipticCurvePrivateKey,
    )
    if not isinstance(key, EllipticCurvePrivateKey):
        raise ValueError(
            f"Wallet Secret must be ECDSA P-256, got "
            f"{type(key).__name__}. (Did you paste the API key "
            f"instead of the Wallet Secret?)"
        )
    return key


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
        wallet_secret: Optional[str] = None,
        base_url: str = _CDP_V2_BASE,
        client: Any = None,  # injected httpx.Client for tests
    ) -> None:
        if not api_key_name:
            raise ValueError("api_key_name is required")
        # Parse PEM eagerly — surfaces placeholder/malformed errors
        # at construction so from_env can detect + return None.
        self._private_key = _load_ed25519_pem(api_key_private_pem)
        # Wallet Secret optional at construction — only needed for
        # wallet-management calls (create_wallet etc). Read-only
        # calls work with just the API key.
        self._wallet_key = (
            _load_wallet_secret(wallet_secret)
            if wallet_secret else None
        )
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

    def _build_wallet_auth_jwt(
        self,
        method: str,
        path: str,
        body_obj: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the X-Wallet-Auth JWT (sp854 — F21, sp854b — F22).

        Signed with the operator's Wallet Secret (ECDSA P-256, ES256).
        Distinct from the Bearer Authorization JWT (Ed25519/EdDSA).
        Required by CDP v2 wallet-management endpoints.

        JWT shape per CDP v2 wallet-auth spec
        (docs.cdp.coinbase.com/api-reference/v2/authentication):

          header:
            alg: ES256
            typ: JWT
            (no kid — kid is Bearer-token only)

          payload:
            iat:    now (seconds)
            nbf:    now
            jti:    16-byte hex nonce
            uris:   ["METHOD host/path"]
            reqHash: SHA-256(canonical-sorted JSON) hex
                     — only if body present
            (no iss/sub/aud/exp — those are Bearer-token only;
             docs explicitly state they're absent here)

        ``reqHash`` covers canonically-sorted JSON, NOT the raw
        body bytes — this is the subtle wire-format detail that
        causes a 401 if missed (F22 root cause). JSON must be
        json.dumps(..., sort_keys=True, separators=(',', ':')).
        """
        import hashlib
        import json as _json
        import jwt as _jwt
        if self._wallet_key is None:
            raise RuntimeError(
                "wallet_secret not configured — call requires "
                "X-Wallet-Auth but COINBASE_CDP_WALLET_SECRET is "
                "unset."
            )
        host = urlparse(self._base_url).netloc
        now = int(time.time())
        payload: Dict[str, Any] = {
            "iat": now,
            "nbf": now,
            "jti": secrets.token_hex(16),
            "uris": [f"{method} {host}{path}"],
        }
        if body_obj is not None:
            canonical = _json.dumps(
                body_obj, sort_keys=True, separators=(",", ":"),
            )
            payload["reqHash"] = hashlib.sha256(
                canonical.encode("utf-8"),
            ).hexdigest()
        return _jwt.encode(
            payload,
            self._wallet_key,
            algorithm="ES256",
            headers={"alg": "ES256", "typ": "JWT"},
        )

    def _post(
        self,
        path: str,
        body: Dict[str, Any],
        *,
        wallet_auth_required: bool = False,
    ) -> Dict[str, Any]:
        import json as _json
        # Canonical-sorted JSON: matches what reqHash hashes so the
        # bytes we send + the bytes we sign are byte-identical. CDP
        # rejects mismatches with a generic 401 (F22 root cause).
        body_bytes = _json.dumps(
            body, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        token = self._build_jwt("POST", path)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if wallet_auth_required:
            # Sp854: attach X-Wallet-Auth JWT for CDP v2 wallet ops.
            headers["X-Wallet-Auth"] = self._build_wallet_auth_jwt(
                "POST", path, body,
            )
        resp = self._client.post(
            f"{self._base_url}{path}",
            content=body_bytes,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    def create_wallet(
        self, user_id: str, email: str,
    ) -> Dict[str, Any]:
        """Create an EVM account on Base for this user.

        Returns dict shaped for CoinbaseWaaSClient.provision_wallet
        consumption: wallet_id, address, network.

        CDP v2 body schema is strict — only `name` (2-36 alphanumeric
        chars + hyphens, globally unique) and `accountPolicy` (UUID)
        are accepted. Unknown fields cause silent 401 (F22 root
        cause was likely the rejected `metadata` field).
        """
        # Sanitize name to match the CDP regex
        # ^[A-Za-z0-9][A-Za-z0-9-]{0,34}[A-Za-z0-9]$
        import re as _re
        safe_user_id = _re.sub(r"[^A-Za-z0-9-]", "-", user_id)[:30]
        safe_user_id = safe_user_id.strip("-") or "user"
        # Append short suffix for global uniqueness (CDP requires
        # unique names across the project).
        name = f"prsm-{safe_user_id}-{secrets.token_hex(3)}"
        body = {"name": name}
        payload = self._post(
            "/platform/v2/evm/accounts", body,
            wallet_auth_required=True,
        )
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
    wallet_secret: Optional[str] = None,
    client: Any = None,
) -> Optional["CdpWaaSBackend"]:
    """Construct a CdpWaaSBackend from env, or None when missing.

    Returns None when required env is absent OR when the PEM is a
    placeholder (so CoinbaseWaaSClient.from_env() shows
    adapter_wired=False with the honest signal until the operator
    pastes the real PEM).

    Wallet secret is optional — read-only calls work without it;
    wallet-management calls (create_wallet etc) require it. When
    unset, the backend constructs but raises if a wallet-management
    call is attempted (honest signal F21 follow-on).
    """
    api_key_name = (
        api_key_name or os.environ.get("COINBASE_CDP_API_KEY_NAME")
    )
    api_key_private_pem = (
        api_key_private_pem
        or os.environ.get("COINBASE_CDP_API_KEY_PRIVATE")
    )
    wallet_secret = (
        wallet_secret
        or os.environ.get("COINBASE_CDP_WALLET_SECRET")
        or None
    )
    if not api_key_name or not api_key_private_pem:
        return None
    try:
        return CdpWaaSBackend(
            api_key_name=api_key_name,
            api_key_private_pem=api_key_private_pem,
            wallet_secret=wallet_secret,
            client=client,
        )
    except ValueError as exc:
        # Placeholder PEM or malformed key — honest signal +
        # callers can read the warning to know what to paste.
        logger.info(
            "CdpWaaSBackend not constructed: %s", exc,
        )
        return None
