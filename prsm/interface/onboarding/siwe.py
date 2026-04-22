"""EIP-4361 Sign-In With Ethereum backend verifier for PRSM wallet onboarding.

Per docs/2026-04-22-phase4-wallet-sdk-design-plan.md §6 Task 1.

The `siwe` PyPI package handles the EIP-4361 parsing + signature recovery +
domain/timing checks. This module wraps it with PRSM-specific invariants:

  - expected-chain-id enforcement (the message-level chain_id must match the
    chain we issued a nonce for),
  - single-use nonce storage (replay protection),
  - normalised PRSM exception types so callers can render friendly errors
    without branching on the upstream exception taxonomy.
"""

from __future__ import annotations

import secrets
import string
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Optional, Protocol

from siwe import (
    DomainMismatch,
    ExpiredMessage,
    InvalidSignature,
    MalformedSession,
    NotYetValidMessage,
    SiweMessage,
    VerificationError,
)


__all__ = [
    "InMemoryNonceStore",
    "NonceStore",
    "SiweChainIdError",
    "SiweDomainError",
    "SiweError",
    "SiweExpiredError",
    "SiweMalformedError",
    "SiweNonceError",
    "SiweNotYetValidError",
    "SiweSignatureError",
    "VerifiedSiwe",
    "verify",
]


class SiweError(Exception):
    """Base class for SIWE verification failures."""


class SiweMalformedError(SiweError):
    """Raw message did not parse as an EIP-4361 SIWE message."""


class SiweSignatureError(SiweError):
    """Signature did not recover to the address stated in the message."""


class SiweDomainError(SiweError):
    """Message's `domain` field did not match the server's expected domain."""


class SiweChainIdError(SiweError):
    """Message's `chain_id` did not match the server's expected chain."""


class SiweExpiredError(SiweError):
    """`now` is after the message's `expiration_time`."""


class SiweNotYetValidError(SiweError):
    """`now` is before the message's `not_before`."""


class SiweNonceError(SiweError):
    """Nonce was not registered by the server, or has already been consumed."""


@dataclass(frozen=True)
class VerifiedSiwe:
    """Result of a successful SIWE verification.

    `address` is EIP-55 checksummed — safe to use as a primary key for
    wallet-to-identity bindings.
    """

    address: str
    domain: str
    chain_id: int
    nonce: str
    issued_at: str
    statement: Optional[str]


class NonceStore(Protocol):
    """Single-use nonce store. Production deployments should use Redis or
    an equivalent TTL-capable store; see Phase 4 plan §8.4."""

    def issue(self, ttl_seconds: int = 300) -> str: ...
    def consume(self, nonce: str) -> bool: ...


_NONCE_ALPHABET = string.ascii_letters + string.digits


def _random_nonce(length: int = 17) -> str:
    # EIP-4361 requires alphanumeric, at least 8 chars. 17 → ~101 bits.
    return "".join(secrets.choice(_NONCE_ALPHABET) for _ in range(length))


@dataclass
class InMemoryNonceStore:
    """Process-local nonce store. Suitable for tests and single-process
    deployments; use a shared backend (Redis) for any multi-worker rollout."""

    _now: Callable[[], float] = field(default_factory=lambda: time.time)
    _nonces: Dict[str, float] = field(default_factory=dict)

    def issue(self, ttl_seconds: int = 300) -> str:
        self._purge_expired()
        nonce = _random_nonce()
        # In the extremely unlikely event of collision, regenerate.
        while nonce in self._nonces:
            nonce = _random_nonce()
        self._nonces[nonce] = self._now() + ttl_seconds
        return nonce

    def consume(self, nonce: str) -> bool:
        self._purge_expired()
        if nonce in self._nonces:
            del self._nonces[nonce]
            return True
        return False

    def _purge_expired(self) -> None:
        now = self._now()
        expired = [n for n, exp in self._nonces.items() if exp <= now]
        for n in expired:
            del self._nonces[n]


def verify(
    raw_message: str,
    signature: str,
    *,
    expected_domain: str,
    expected_chain_id: int,
    nonce_store: NonceStore,
    now: Optional[datetime] = None,
) -> VerifiedSiwe:
    """Verify an EIP-4361 message + signature.

    Order of checks matters: we validate every invariant BEFORE consuming the
    nonce, so a failed request (wrong domain, bad signature, etc.) does not
    invalidate a still-fresh nonce — the client can re-submit against the
    same nonce after fixing the failure.

    The final `nonce_store.consume` call is the commit point; after it
    returns True, the same (message, signature) pair cannot succeed again.
    """

    try:
        msg = SiweMessage.from_message(raw_message)
    except MalformedSession as exc:
        raise SiweMalformedError(str(exc)) from exc
    except Exception as exc:  # pydantic ValidationError, grammar errors, etc.
        raise SiweMalformedError(f"parse failed: {exc}") from exc

    if msg.chain_id != expected_chain_id:
        raise SiweChainIdError(
            f"chain_id mismatch: message={msg.chain_id}, expected={expected_chain_id}"
        )

    try:
        msg.verify(signature, domain=expected_domain, timestamp=now)
    except DomainMismatch as exc:
        raise SiweDomainError(str(exc)) from exc
    except ExpiredMessage as exc:
        raise SiweExpiredError(str(exc)) from exc
    except NotYetValidMessage as exc:
        raise SiweNotYetValidError(str(exc)) from exc
    except InvalidSignature as exc:
        raise SiweSignatureError(str(exc)) from exc
    except VerificationError as exc:
        # Any other siwe verification failure — surface as signature error.
        raise SiweSignatureError(str(exc)) from exc

    if not nonce_store.consume(msg.nonce):
        raise SiweNonceError(
            f"nonce not registered or already consumed: {msg.nonce}"
        )

    return VerifiedSiwe(
        address=msg.address,
        domain=msg.domain,
        chain_id=msg.chain_id,
        nonce=msg.nonce,
        issued_at=msg.issued_at,
        statement=msg.statement,
    )
