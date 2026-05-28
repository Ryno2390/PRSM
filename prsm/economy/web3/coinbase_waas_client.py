"""Sprint 276 — Coinbase Wallet-as-a-Service (WaaS) adapter.

Per Vision §14 "Crypto-UX adoption barrier" mitigation: "Embedded
wallets by default. Privy, Web3Auth, Magic.link patterns provide
wallet functionality without requiring users to manage seed phrases.
Email-based recovery. Indistinguishable from web2 onboarding."

This adapter wraps the Coinbase Developer Platform (CDP) Wallet API,
which provisions MPC-secured wallets server-side. The PRSM node
acts as the project owner; each PRSM user gets a CDP-managed wallet
on first interaction. The user never sees a seed phrase, never
installs a wallet extension, never holds gas tokens explicitly
(future sprint: gasless txns via paymaster).

This v1 scaffold mirrors the offramp-quote PENDING_COMMISSION pattern
shipped 2026-05-08: when CDP API keys are absent the adapter returns
preview records without hitting any external API; when keys are
present the adapter delegates to a dependency-injected backend that
will wrap the real CDP SDK once commissioned.

Operator env:
  - COINBASE_CDP_API_KEY_NAME      — CDP project key id
  - COINBASE_CDP_API_KEY_PRIVATE   — CDP key private material
  - PRSM_WAAS_STORE_DIR            — opt-in JSON persistence dir
  - PRSM_WAAS_NETWORK              — defaults to "base-mainnet"

Per R-2026-05-08-1 (the offramp composer-only invariant): that rule
is specifically scoped to `coinbase_offramp_initiate`. WaaS wallet
PROVISIONING is not a money flow; it's user provisioning under
operator authorization (operator's CDP API key). The execute path is
allowed here once commissioned.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class _WaasBackend(Protocol):
    """Dependency-injected backend. Production backend wraps the
    Coinbase CDP SDK; tests use a fake.

    Method must return a dict with keys: wallet_id, address, network.
    """

    def create_wallet(
        self, user_id: str, email: str,
    ) -> Dict[str, Any]: ...


@dataclass
class WaasWalletRecord:
    user_id: str
    email: str
    wallet_id: Optional[str]
    address: Optional[str]
    network: Optional[str]
    status: str  # PROVISIONED | PENDING_COMMISSION | FAILED
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WaasWalletRecord":
        return cls(
            user_id=d["user_id"],
            email=d.get("email", ""),
            wallet_id=d.get("wallet_id"),
            address=d.get("address"),
            network=d.get("network"),
            status=d.get("status", "PENDING_COMMISSION"),
            created_at=d.get("created_at", 0.0),
        )


class CoinbaseWaaSClient:
    """In-process WaaS adapter.

    State:
      - In-memory dict[user_id -> WaasWalletRecord] is the primary
        index.
      - Disk persistence (one JSON file per user_id) when
        ``persist_dir`` is set or PRSM_WAAS_STORE_DIR env var
        resolves at construction.
      - Backend is dependency-injected; None when uncommissioned
        OR for tests that want to assert no-backend-call invariants.
    """

    def __init__(
        self,
        api_key_name: Optional[str] = None,
        api_key_private: Optional[str] = None,
        *,
        network: str = "base-mainnet",
        persist_dir: Optional[Path] = None,
        backend: Optional[_WaasBackend] = None,
    ) -> None:
        self._api_key_name = api_key_name
        self._api_key_private = api_key_private
        self._network = network
        self._backend = backend
        self._wallets: Dict[str, WaasWalletRecord] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir) if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    @classmethod
    def from_env(
        cls, *, backend: Optional[_WaasBackend] = None,
    ) -> Optional["CoinbaseWaaSClient"]:
        """Build a client from env vars. Returns a working client
        even when keys are absent — provision calls just return
        PENDING_COMMISSION records until keys land."""
        key_name = os.environ.get("COINBASE_CDP_API_KEY_NAME") or None
        key_priv = (
            os.environ.get("COINBASE_CDP_API_KEY_PRIVATE") or None
        )
        network = (
            os.environ.get("PRSM_WAAS_NETWORK") or "base-mainnet"
        )
        # Sp860 — persist by default so wallets survive daemon
        # restarts. Operators opt out by setting the env var to
        # the sentinel ":memory:" (Python sqlite-style); explicit
        # filesystem paths still take precedence as before.
        persist_raw = os.environ.get("PRSM_WAAS_STORE_DIR")
        if persist_raw == ":memory:":
            persist_dir = None
        elif persist_raw:
            persist_dir = Path(persist_raw)
        else:
            persist_dir = Path.home() / ".prsm" / "waas-wallets"
        # Sp851 — auto-wire CDP WaaS backend when caller didn't
        # supply one + both keys present + PEM parses cleanly.
        # Placeholder PEMs (e.g., "REPLACE_WITH...") fail parse
        # in CdpWaaSBackend's constructor; from_env catches and
        # returns None → adapter_wired stays False (honest signal).
        if backend is None and key_name and key_priv:
            try:
                from prsm.economy.web3.coinbase_waas_cdp_backend import (
                    from_env as _cdp_waas_from_env,
                )
                backend = _cdp_waas_from_env()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CoinbaseWaaSClient: CDP backend auto-wire "
                    "failed (falling back to PENDING_COMMISSION): "
                    "%s", exc,
                )
                backend = None
        return cls(
            api_key_name=key_name,
            api_key_private=key_priv,
            network=network,
            persist_dir=persist_dir,
            backend=backend,
        )

    def is_commissioned(self) -> bool:
        """True iff both CDP API keys are present. Backend is NOT
        required to be commissioned — it's required to actually
        execute, but the commission check is purely env-key based
        so an operator can verify their CDP credentials are wired
        even before plugging in the SDK backend."""
        return bool(self._api_key_name and self._api_key_private)

    def adapter_wired(self) -> bool:
        """True iff a CDP WaaS SDK backend has been injected.

        Orthogonal to ``is_commissioned``. Sp848 surfaces this as a
        second status flag so operators distinguish "env vars wired"
        from "ready to execute provision_wallet()" — without this
        signal, an env-commissioned-but-stubbed deployment looks
        identical to a fully-wired one via /wallet/waas/status.
        """
        return self._backend is not None

    def provision_wallet(
        self, user_id: str, email: str,
    ) -> WaasWalletRecord:
        """Provision an MPC wallet for the given user_id.

        Idempotent: re-calling for the same user_id returns the
        previously-provisioned record without invoking the backend.
        Pre-commission: returns a PENDING_COMMISSION record without
        invoking the backend.
        """
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")
        if not email or not isinstance(email, str):
            raise ValueError("email must be a non-empty string")

        existing = self._wallets.get(user_id)
        if existing is not None and existing.status == "PROVISIONED":
            return existing

        if not self.is_commissioned() or self._backend is None:
            record = WaasWalletRecord(
                user_id=user_id, email=email,
                wallet_id=None, address=None,
                network=self._network,
                status="PENDING_COMMISSION",
                created_at=time.time(),
            )
            self._wallets[user_id] = record
            self._write_to_disk(record)
            return record

        try:
            payload = self._backend.create_wallet(user_id, email)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CoinbaseWaaSClient: backend create_wallet failed "
                "for user_id=%s: %s",
                user_id, exc,
            )
            record = WaasWalletRecord(
                user_id=user_id, email=email,
                wallet_id=None, address=None,
                network=self._network,
                status="FAILED",
                created_at=time.time(),
            )
            self._wallets[user_id] = record
            return record

        record = WaasWalletRecord(
            user_id=user_id, email=email,
            wallet_id=payload.get("wallet_id"),
            address=payload.get("address"),
            network=payload.get("network", self._network),
            status="PROVISIONED",
            created_at=time.time(),
        )
        self._wallets[user_id] = record
        self._write_to_disk(record)
        return record

    def get_wallet(self, user_id: str) -> Optional[WaasWalletRecord]:
        return self._wallets.get(user_id)

    def list_wallets(self) -> List[WaasWalletRecord]:
        return list(self._wallets.values())

    # ── Persistence helpers ──────────────────────────────

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for path in self._persist_dir.glob("*.json"):
            try:
                d = json.loads(path.read_text())
                record = WaasWalletRecord.from_dict(d)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CoinbaseWaaSClient: skipping corrupt %s: %s",
                    path, exc,
                )
                continue
            self._wallets[record.user_id] = record

    def _write_to_disk(self, record: WaasWalletRecord) -> None:
        if self._persist_dir is None:
            return
        # Sanitize user_id to be filename-safe. Since user_id
        # values come from PRSM auth, they're not user-controlled
        # but defense-in-depth: strip path separators.
        safe = record.user_id.replace("/", "_").replace("\\", "_")
        path = self._persist_dir / f"{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(record.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CoinbaseWaaSClient: disk write failed for %s: %s",
                record.user_id, exc,
            )
