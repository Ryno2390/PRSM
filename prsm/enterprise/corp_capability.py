"""Sprint 306 — soulbound $CORP authorization capability.

Vision §7 Enterprise Confidentiality Mode layer 2:
ergonomics + accounting + audit. Not the security gate —
that's the encryption (sprint 304) + TEE policy (305 / 305a).

A $CORP capability is a signed grant: the enterprise issuer
binds (subject_pubkey, scope, quota_units, expires_at,
nonce) under an Ed25519 signature. The subject can then
redeem the capability by signing a fresh `RedemptionRequest`
with their device-bound private key. This dual-signature
design makes the capability **soulbound in practice**: a
leaked capability without the corresponding device key is
useless, because the subject signature on each redemption
won't verify. Phishing both the capability AND device key
is equivalent to existing baseline corporate-account-
takeover risk — no new attack surface.

The store tracks cumulative consumption + a redemption
ledger + nonce-replay defense. Quota over-spend is refused
WITHOUT consuming any quota.

This module is the primitive layer. Sprint 306a wires it
into the /compute/inference and /content/upload dispatch
paths via an `X-CORP-Capability` header.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

logger = logging.getLogger(__name__)


CAPABILITY_VERSION = "v1"


class CapabilityStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


# ── b64 helpers ──────────────────────────────────────


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(s: str) -> bytes:
    if not isinstance(s, str):
        raise ValueError(
            f"expected base64 string, got "
            f"{type(s).__name__}"
        )
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"invalid base64: {e}")


def _load_ed25519_priv(b64: str) -> Ed25519PrivateKey:
    raw = _b64d(b64)
    if len(raw) != 32:
        raise ValueError(
            f"Ed25519 privkey must be 32 bytes, got "
            f"{len(raw)}"
        )
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_ed25519_pub(b64: str) -> Ed25519PublicKey:
    raw = _b64d(b64)
    if len(raw) != 32:
        raise ValueError(
            f"Ed25519 pubkey must be 32 bytes, got "
            f"{len(raw)}"
        )
    return Ed25519PublicKey.from_public_bytes(raw)


# ── Dataclasses ──────────────────────────────────────


@dataclass
class CorpIssuer:
    issuer_id: str
    signing_pubkey_b64: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issuer_id": self.issuer_id,
            "signing_pubkey_b64": self.signing_pubkey_b64,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "CorpIssuer":
        return cls(
            issuer_id=d["issuer_id"],
            signing_pubkey_b64=d["signing_pubkey_b64"],
        )


@dataclass
class CorpCapability:
    capability_id: str
    issuer_id: str
    subject_id: str
    subject_pubkey_b64: str
    scope: List[str]
    quota_units: int
    issued_at: float
    expires_at: float
    nonce: str
    signature_b64: str
    version: str = CAPABILITY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "capability_id": self.capability_id,
            "issuer_id": self.issuer_id,
            "subject_id": self.subject_id,
            "subject_pubkey_b64": self.subject_pubkey_b64,
            "scope": list(self.scope),
            "quota_units": int(self.quota_units),
            "issued_at": float(self.issued_at),
            "expires_at": float(self.expires_at),
            "nonce": self.nonce,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "CorpCapability":
        v = d.get("version", "")
        if v != CAPABILITY_VERSION:
            raise ValueError(
                f"unknown capability version {v!r}; "
                f"expected {CAPABILITY_VERSION!r}"
            )
        return cls(
            capability_id=d["capability_id"],
            issuer_id=d["issuer_id"],
            subject_id=d["subject_id"],
            subject_pubkey_b64=d["subject_pubkey_b64"],
            scope=list(d.get("scope") or []),
            quota_units=int(d["quota_units"]),
            issued_at=float(d["issued_at"]),
            expires_at=float(d["expires_at"]),
            nonce=d["nonce"],
            signature_b64=d.get("signature_b64", ""),
            version=v,
        )


@dataclass
class RedemptionRequest:
    capability_id: str
    action: str
    units_requested: int
    nonce: str
    timestamp: float
    subject_signature_b64: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "action": self.action,
            "units_requested": int(self.units_requested),
            "nonce": self.nonce,
            "timestamp": float(self.timestamp),
            "subject_signature_b64": (
                self.subject_signature_b64
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "RedemptionRequest":
        return cls(
            capability_id=d["capability_id"],
            action=d["action"],
            units_requested=int(d["units_requested"]),
            nonce=d["nonce"],
            timestamp=float(d["timestamp"]),
            subject_signature_b64=d[
                "subject_signature_b64"
            ],
        )


@dataclass
class RedemptionResult:
    status: CapabilityStatus
    capability_id: str
    units_consumed_this_request: int = 0
    remaining_quota: int = 0
    diagnostic: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "capability_id": self.capability_id,
            "units_consumed_this_request": (
                self.units_consumed_this_request
            ),
            "remaining_quota": self.remaining_quota,
            "diagnostic": self.diagnostic,
        }


# ── Keypair generation ───────────────────────────────


def _generate_ed25519_keypair() -> tuple[str, str]:
    priv = Ed25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _b64e(priv_raw), _b64e(pub_raw)


def generate_issuer_keypair() -> tuple[str, str]:
    return _generate_ed25519_keypair()


def generate_subject_keypair() -> tuple[str, str]:
    return _generate_ed25519_keypair()


# ── Canonical encoding ───────────────────────────────


def canonical_capability_bytes(
    cap: CorpCapability,
) -> bytes:
    """JSON-encoded canonical form excluding the signature
    field. Sort keys to make field-order irrelevant."""
    payload = {
        "version": cap.version,
        "capability_id": cap.capability_id,
        "issuer_id": cap.issuer_id,
        "subject_id": cap.subject_id,
        "subject_pubkey_b64": cap.subject_pubkey_b64,
        "scope": sorted(cap.scope),
        "quota_units": int(cap.quota_units),
        "issued_at": float(cap.issued_at),
        "expires_at": float(cap.expires_at),
        "nonce": cap.nonce,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def canonical_redemption_bytes(
    req: RedemptionRequest,
) -> bytes:
    payload = {
        "capability_id": req.capability_id,
        "action": req.action,
        "units_requested": int(req.units_requested),
        "nonce": req.nonce,
        "timestamp": float(req.timestamp),
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


# ── Signing ──────────────────────────────────────────


def sign_capability(
    *,
    issuer_id: str,
    issuer_privkey_b64: str,
    subject_id: str,
    subject_pubkey_b64: str,
    scope: List[str],
    quota_units: int,
    issued_at: float,
    expires_at: float,
    capability_id: Optional[str] = None,
    nonce: Optional[str] = None,
) -> CorpCapability:
    import uuid
    cap = CorpCapability(
        capability_id=(
            capability_id or str(uuid.uuid4())
        ),
        issuer_id=issuer_id,
        subject_id=subject_id,
        subject_pubkey_b64=subject_pubkey_b64,
        scope=list(scope),
        quota_units=int(quota_units),
        issued_at=float(issued_at),
        expires_at=float(expires_at),
        nonce=nonce or _b64e(os.urandom(16)),
        signature_b64="",  # filled below
    )
    priv = _load_ed25519_priv(issuer_privkey_b64)
    sig = priv.sign(canonical_capability_bytes(cap))
    cap.signature_b64 = _b64e(sig)
    return cap


def sign_redemption(
    *,
    subject_privkey_b64: str,
    capability_id: str,
    action: str,
    units_requested: int,
    nonce: str,
    timestamp: float,
) -> RedemptionRequest:
    req = RedemptionRequest(
        capability_id=capability_id,
        action=action,
        units_requested=int(units_requested),
        nonce=nonce,
        timestamp=float(timestamp),
        subject_signature_b64="",
    )
    priv = _load_ed25519_priv(subject_privkey_b64)
    sig = priv.sign(canonical_redemption_bytes(req))
    req.subject_signature_b64 = _b64e(sig)
    return req


# ── Verification ─────────────────────────────────────


def verify_capability_signature(
    cap: CorpCapability, issuer: CorpIssuer,
) -> bool:
    if cap.issuer_id != issuer.issuer_id:
        return False
    try:
        pub = _load_ed25519_pub(issuer.signing_pubkey_b64)
        sig = _b64d(cap.signature_b64)
    except ValueError:
        return False
    try:
        pub.verify(sig, canonical_capability_bytes(cap))
        return True
    except InvalidSignature:
        return False


def verify_redemption_signature(
    req: RedemptionRequest, cap: CorpCapability,
) -> bool:
    if req.capability_id != cap.capability_id:
        return False
    try:
        pub = _load_ed25519_pub(cap.subject_pubkey_b64)
        sig = _b64d(req.subject_signature_b64)
    except ValueError:
        return False
    try:
        pub.verify(sig, canonical_redemption_bytes(req))
        return True
    except InvalidSignature:
        return False


def evaluate_redemption(
    *,
    capability: CorpCapability,
    request: RedemptionRequest,
    issuer: CorpIssuer,
    consumed_so_far: int,
    now: Optional[float] = None,
) -> RedemptionResult:
    now = now if now is not None else time.time()

    # Capability signature (issuer)
    if not verify_capability_signature(
        capability, issuer,
    ):
        return RedemptionResult(
            status=CapabilityStatus.FAIL,
            capability_id=capability.capability_id,
            diagnostic=(
                "capability signature does not verify "
                "against the provided issuer"
            ),
        )

    # Redemption signature (subject)
    if not verify_redemption_signature(
        request, capability,
    ):
        return RedemptionResult(
            status=CapabilityStatus.FAIL,
            capability_id=capability.capability_id,
            diagnostic=(
                "redemption signature does not verify "
                "against the capability's subject pubkey"
            ),
        )

    # Capability_id binding
    if request.capability_id != capability.capability_id:
        return RedemptionResult(
            status=CapabilityStatus.FAIL,
            capability_id=capability.capability_id,
            diagnostic=(
                f"redemption capability_id "
                f"{request.capability_id!r} does not "
                f"match capability "
                f"{capability.capability_id!r}"
            ),
        )

    # Expiry
    if now > capability.expires_at:
        return RedemptionResult(
            status=CapabilityStatus.FAIL,
            capability_id=capability.capability_id,
            diagnostic=(
                f"capability expired at "
                f"{capability.expires_at} (now={now})"
            ),
        )

    # Scope
    if request.action not in capability.scope:
        return RedemptionResult(
            status=CapabilityStatus.FAIL,
            capability_id=capability.capability_id,
            diagnostic=(
                f"action {request.action!r} not in "
                f"capability scope {capability.scope}"
            ),
        )

    # Quota — zero-unit redemption is operator confusion
    if request.units_requested <= 0:
        return RedemptionResult(
            status=CapabilityStatus.FAIL,
            capability_id=capability.capability_id,
            diagnostic=(
                f"units_requested must be > 0; got "
                f"{request.units_requested}"
            ),
        )
    proposed_total = (
        consumed_so_far + request.units_requested
    )
    if proposed_total > capability.quota_units:
        return RedemptionResult(
            status=CapabilityStatus.FAIL,
            capability_id=capability.capability_id,
            diagnostic=(
                f"over quota: consumed_so_far="
                f"{consumed_so_far} + requested="
                f"{request.units_requested} > "
                f"quota_units={capability.quota_units}"
            ),
        )

    return RedemptionResult(
        status=CapabilityStatus.PASS,
        capability_id=capability.capability_id,
        units_consumed_this_request=int(
            request.units_requested,
        ),
        remaining_quota=int(
            capability.quota_units - proposed_total,
        ),
        diagnostic=(
            f"redemption accepted; "
            f"remaining={capability.quota_units - proposed_total}"
        ),
    )


# ── Store ────────────────────────────────────────────


class CorpCapabilityStore:
    """In-memory + optional-filesystem store for $CORP
    issuers, redemption ledger, and per-capability
    consumption accounting.

    Persistence layout (when persist_dir is set):
      issuers.json — list of CorpIssuer dicts
      ledger/<capability_id>.json — list of redemption
                                    audit records
      consumed.json — capability_id → consumed_units
    """

    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
    ) -> None:
        self._issuers: Dict[str, CorpIssuer] = {}
        self._consumed: Dict[str, int] = {}
        self._ledger: Dict[str, List[Dict[str, Any]]] = {}
        # Replay defense — track (capability_id, nonce)
        # pairs already redeemed.
        self._seen_nonces: Dict[str, set[str]] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir)
            if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(
                parents=True, exist_ok=True,
            )
            self._load_from_disk()

    @classmethod
    def from_env(cls) -> "CorpCapabilityStore":
        raw = os.environ.get("PRSM_CORP_CAPABILITY_DIR")
        persist_dir = Path(raw) if raw else None
        return cls(persist_dir=persist_dir)

    # ── Issuers ───────────────────────────────────────

    def register_issuer(self, issuer: CorpIssuer) -> None:
        # Validate pubkey BEFORE persisting (loud-fail on
        # operator misconfig)
        try:
            _load_ed25519_pub(issuer.signing_pubkey_b64)
        except ValueError as e:
            raise ValueError(
                f"invalid signing_pubkey_b64 for "
                f"issuer {issuer.issuer_id!r}: {e}"
            )
        self._issuers[issuer.issuer_id] = issuer
        self._persist_issuers()

    def get_issuer(
        self, issuer_id: str,
    ) -> Optional[CorpIssuer]:
        return self._issuers.get(issuer_id)

    def list_issuers(self) -> List[CorpIssuer]:
        return list(self._issuers.values())

    # ── Redemption ────────────────────────────────────

    def get_consumed(self, capability_id: str) -> int:
        return self._consumed.get(capability_id, 0)

    def get_ledger(
        self, capability_id: str,
    ) -> List[Dict[str, Any]]:
        return list(self._ledger.get(capability_id, []))

    def redeem(
        self,
        capability: CorpCapability,
        request: RedemptionRequest,
        *,
        now: Optional[float] = None,
    ) -> RedemptionResult:
        issuer = self._issuers.get(capability.issuer_id)
        if issuer is None:
            return RedemptionResult(
                status=CapabilityStatus.FAIL,
                capability_id=capability.capability_id,
                diagnostic=(
                    f"issuer {capability.issuer_id!r} not "
                    f"registered with this PRSM operator"
                ),
            )

        # Replay defense — nonce per (capability_id) must
        # not have been seen before.
        seen = self._seen_nonces.setdefault(
            capability.capability_id, set(),
        )
        if request.nonce in seen:
            return RedemptionResult(
                status=CapabilityStatus.FAIL,
                capability_id=capability.capability_id,
                diagnostic=(
                    f"redemption nonce {request.nonce!r} "
                    f"already used for this capability "
                    f"(replay attack defense)"
                ),
            )

        result = evaluate_redemption(
            capability=capability,
            request=request,
            issuer=issuer,
            consumed_so_far=self.get_consumed(
                capability.capability_id,
            ),
            now=now,
        )
        if result.status == CapabilityStatus.PASS:
            self._consumed[capability.capability_id] = (
                self._consumed.get(
                    capability.capability_id, 0,
                )
                + request.units_requested
            )
            seen.add(request.nonce)
            self._ledger.setdefault(
                capability.capability_id, [],
            ).append({
                "timestamp": float(
                    now if now is not None else time.time()
                ),
                "action": request.action,
                "units_requested": int(
                    request.units_requested,
                ),
                "nonce": request.nonce,
                "subject_id": capability.subject_id,
            })
            self._persist_consumed()
            self._persist_ledger(capability.capability_id)
        return result

    # ── Persistence ───────────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        # Issuers
        ipath = self._persist_dir / "issuers.json"
        if ipath.exists():
            try:
                rows = json.loads(ipath.read_text())
                for r in rows:
                    iss = CorpIssuer.from_dict(r)
                    self._issuers[iss.issuer_id] = iss
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CorpCapabilityStore: skipping corrupt "
                    "issuers.json: %s", exc,
                )
        # Consumed counters
        cpath = self._persist_dir / "consumed.json"
        if cpath.exists():
            try:
                self._consumed = {
                    k: int(v) for k, v in json.loads(
                        cpath.read_text(),
                    ).items()
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CorpCapabilityStore: skipping corrupt "
                    "consumed.json: %s", exc,
                )
        # Ledger files
        ledger_dir = self._persist_dir / "ledger"
        if ledger_dir.exists():
            for path in ledger_dir.glob("*.json"):
                try:
                    rows = json.loads(path.read_text())
                    cap_id = path.stem
                    self._ledger[cap_id] = rows
                    # Rebuild nonce-seen index from ledger
                    seen = self._seen_nonces.setdefault(
                        cap_id, set(),
                    )
                    for r in rows:
                        n = r.get("nonce")
                        if n:
                            seen.add(n)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "CorpCapabilityStore: skipping "
                        "corrupt ledger %s: %s",
                        path, exc,
                    )

    def _persist_issuers(self) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / "issuers.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps([
                i.to_dict()
                for i in self._issuers.values()
            ]))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CorpCapabilityStore: issuers persist "
                "failed: %s", exc,
            )

    def _persist_consumed(self) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / "consumed.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self._consumed))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CorpCapabilityStore: consumed persist "
                "failed: %s", exc,
            )

    def _persist_ledger(self, capability_id: str) -> None:
        if self._persist_dir is None:
            return
        ledger_dir = self._persist_dir / "ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        safe = (
            capability_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        path = ledger_dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(
                self._ledger.get(capability_id, []),
            ))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CorpCapabilityStore: ledger persist "
                "failed for %s: %s",
                capability_id, exc,
            )
