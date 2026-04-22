"""Signed bootstrap-peer-list discovery for the PRSM P2P layer.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §3.1, §4.1, §6 Task 1.

Clients fetch the bootstrap list from:

  1. HTTPS primary (e.g., https://bootstrap.prsm.ai/bootstrap-nodes.json)
  2. DNS TXT fallback (e.g., `_prsm-bootstrap.prsm.ai`)

The list is signed by a Foundation-controlled Ed25519 key whose public
half is committed into the client binary. Any list the client acts on MUST
verify against that pubkey AND be unexpired.

Scope boundary — what this module does NOT do:

  * Actual libp2p dialing / connection — that's Task 3 (NAT traversal).
  * HTTPS / DNS transport — callers inject fetchers so the module stays
    testable and transport-agnostic.
  * Bootstrap-node operation / signing-key custody — Foundation ops runbook
    (Task 2).
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


logger = logging.getLogger(__name__)


__all__ = [
    "BootstrapError",
    "BootstrapExpiredError",
    "BootstrapFetcher",
    "BootstrapList",
    "BootstrapMalformedError",
    "BootstrapPeer",
    "BootstrapSignatureError",
    "BootstrapUnavailableError",
    "DnsBootstrapFetcher",
    "HttpsBootstrapFetcher",
    "discover_bootstrap_peers",
    "sign_bootstrap_list",
    "verify_bootstrap_list",
]


SUPPORTED_VERSION = 1


# ---- Errors -----------------------------------------------------------------


class BootstrapError(Exception):
    """Base class for bootstrap-list failures."""


class BootstrapMalformedError(BootstrapError):
    """Document did not match the expected schema."""


class BootstrapSignatureError(BootstrapError):
    """Signature did not verify against the Foundation pubkey."""


class BootstrapExpiredError(BootstrapError):
    """`expires_at` is in the past."""


class BootstrapUnavailableError(BootstrapError):
    """Both HTTPS primary and DNS fallback returned nothing usable."""


# ---- Model ------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapPeer:
    peer_id: str
    multiaddrs: tuple[str, ...]
    region: str
    operator: str

    @classmethod
    def from_dict(cls, d: dict) -> "BootstrapPeer":
        try:
            return cls(
                peer_id=d["peer_id"],
                multiaddrs=tuple(d["multiaddrs"]),
                region=d["region"],
                operator=d["operator"],
            )
        except (KeyError, TypeError) as exc:
            raise BootstrapMalformedError(f"peer: {exc}") from exc

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "multiaddrs": list(self.multiaddrs),
            "region": self.region,
            "operator": self.operator,
        }


@dataclass(frozen=True)
class BootstrapList:
    version: int
    expires_at: str  # ISO 8601, UTC
    bootstrap_peers: tuple[BootstrapPeer, ...]
    signature: str  # base64


# ---- Canonical JSON for signing/verification --------------------------------


def _canonical_body_bytes(body: dict) -> bytes:
    """Deterministic JSON encoding of the signable body.

    `sort_keys=True` + no-whitespace separators produces a byte sequence that
    does not depend on Python dict insertion order or on platform-specific
    JSON serialisers. Any signer / verifier reaching the same body yields
    byte-equal output.
    """
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _body_from_doc(doc: dict) -> dict:
    """Return the signable portion (everything except `signature`)."""
    return {k: v for k, v in doc.items() if k != "signature"}


# ---- Signing + verification -------------------------------------------------


def sign_bootstrap_list(
    peers: list[BootstrapPeer],
    expires_at_iso: str,
    private_key: Ed25519PrivateKey,
    *,
    version: int = SUPPORTED_VERSION,
) -> dict:
    """Produce the JSON document a verifier can validate.

    Returned dict is serialisable and can be written to disk, served over
    HTTPS, or split across DNS TXT chunks.
    """
    body = {
        "version": version,
        "expires_at": expires_at_iso,
        "bootstrap_peers": [p.to_dict() for p in peers],
    }
    signature = private_key.sign(_canonical_body_bytes(body))
    return {**body, "signature": base64.b64encode(signature).decode("ascii")}


def verify_bootstrap_list(
    doc: dict,
    public_key: Ed25519PublicKey,
    *,
    now: datetime,
) -> BootstrapList:
    """Verify + parse. Raises a BootstrapError subclass on any failure.

    Order is: shape → version → expiry → signature. Expiry is checked
    BEFORE signature so we do not burn a verification on a document we
    would reject anyway — and so the error message points at the
    user-actionable failure first (an expired list is fixed by a refresh
    pull, not by a crypto investigation).
    """
    # Shape check.
    if not isinstance(doc, dict):
        raise BootstrapMalformedError("document is not an object")
    for field in ("version", "expires_at", "bootstrap_peers", "signature"):
        if field not in doc:
            raise BootstrapMalformedError(f"missing field: {field}")

    # Version check — unknown future versions are safer to reject than to
    # treat as "version 1 with unknown fields."
    if doc["version"] != SUPPORTED_VERSION:
        raise BootstrapMalformedError(
            f"unsupported version: {doc['version']} "
            f"(this client supports {SUPPORTED_VERSION})"
        )

    # Expiry check.
    try:
        expires = datetime.fromisoformat(doc["expires_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BootstrapMalformedError(f"expires_at: {exc}") from exc
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if now >= expires:
        raise BootstrapExpiredError(
            f"list expired at {doc['expires_at']}; now is {now.isoformat()}"
        )

    # Signature check over the canonical body.
    try:
        signature = base64.b64decode(doc["signature"], validate=True)
    except Exception as exc:
        raise BootstrapMalformedError(f"signature: {exc}") from exc
    body = _body_from_doc(doc)
    try:
        public_key.verify(signature, _canonical_body_bytes(body))
    except InvalidSignature as exc:
        raise BootstrapSignatureError("signature does not verify") from exc

    # Parse peers (shape-check each).
    peers_raw = doc["bootstrap_peers"]
    if not isinstance(peers_raw, list):
        raise BootstrapMalformedError("bootstrap_peers must be a list")
    peers = tuple(BootstrapPeer.from_dict(p) for p in peers_raw)

    return BootstrapList(
        version=doc["version"],
        expires_at=doc["expires_at"],
        bootstrap_peers=peers,
        signature=doc["signature"],
    )


# ---- Fetchers ---------------------------------------------------------------


class BootstrapFetcher(Protocol):
    """Returns a parsed JSON dict on success, or None on any failure.

    Implementations should not raise — a failure (network error, parse
    error, timeout) must be signalled by returning None so the caller's
    fallback logic can take over cleanly.
    """

    def fetch(self) -> Optional[dict]: ...


@dataclass
class HttpsBootstrapFetcher:
    """Fetch the bootstrap list from an HTTPS URL.

    `get` is injected so tests can pass a stub and production uses
    requests/httpx without this module depending on either.
    """

    url: str
    get: Callable[[str], str]

    def fetch(self) -> Optional[dict]:
        try:
            body = self.get(self.url)
        except Exception as exc:
            logger.warning("HTTPS fetch failed (%s): %s", self.url, exc)
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("HTTPS body parse failed (%s): %s", self.url, exc)
            return None


@dataclass
class DnsBootstrapFetcher:
    """Fetch the bootstrap list from DNS TXT records.

    DNS TXT records are limited to 255-byte chunks; the published record
    can consist of multiple chunks which concatenate to the full JSON
    document. The `resolve_txt` callable returns a list of TXT strings
    from the given domain.
    """

    domain: str
    resolve_txt: Callable[[str], list[str]]

    def fetch(self) -> Optional[dict]:
        try:
            chunks = self.resolve_txt(self.domain)
        except Exception as exc:
            logger.warning("DNS TXT resolution failed (%s): %s", self.domain, exc)
            return None
        if not chunks:
            return None
        body = "".join(chunks)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("DNS body parse failed (%s): %s", self.domain, exc)
            return None


# ---- Discovery orchestration ------------------------------------------------


def discover_bootstrap_peers(
    public_key: Ed25519PublicKey,
    *,
    primary: BootstrapFetcher,
    fallback: BootstrapFetcher,
    now: datetime,
) -> BootstrapList:
    """Try primary → fallback. Raise BootstrapUnavailableError if neither
    yields a verifiable list.

    Primary-verification failures (signature / expiry / malformed) trigger
    the fallback, BUT a fallback-verification failure is re-raised as-is —
    we do not infinitely hunt for a valid list. If the fallback is
    successfully fetched but fails verification, that is a signal the
    signing key has been compromised or the DNS record is stale, and
    operators need to know via the original exception.
    """
    primary_doc = primary.fetch()
    if primary_doc is not None:
        try:
            return verify_bootstrap_list(primary_doc, public_key, now=now)
        except BootstrapError as exc:
            logger.warning(
                "primary bootstrap list failed verification (%s); falling back",
                exc,
            )

    fallback_doc = fallback.fetch()
    if fallback_doc is not None:
        return verify_bootstrap_list(fallback_doc, public_key, now=now)

    raise BootstrapUnavailableError(
        "primary and fallback bootstrap sources both unavailable"
    )
