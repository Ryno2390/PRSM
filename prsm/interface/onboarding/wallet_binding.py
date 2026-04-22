"""Wallet-to-PRSM-node identity binding.

Per docs/2026-04-22-phase4-wallet-sdk-design-plan.md §4 + §6 Task 2.

A binding records that a given Ethereum wallet address controls a given PRSM
Ed25519 node identity. The binding is attested by an EIP-191 signature from
the wallet over a canonical message that includes both identifiers and an
issued-at timestamp; recovery of that signature to the stated wallet address
is the binding's authenticity proof.

Two storage backends ship here: `InMemoryWalletBindingStore` for tests and
single-process deployments, and `SqliteWalletBindingStore` for Foundation
deployments that need crash-durability but do not yet need Postgres.

Scope boundary — what this module does NOT do:

  - SIWE sign-in verification (that's `siwe.verify` in Task 1).
  - HTTP / REST endpoint wiring (that's `wallet_api.py` in a later task).
  - On-chain IdentityBinding.sol migration (Phase 4 plan §8.2 open issue).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import keccak, to_checksum_address

from prsm.node.identity import NodeIdentity, generate_node_identity


__all__ = [
    "BindingConflictError",
    "BindingError",
    "BindingSignatureError",
    "IdentityBinding",
    "InMemoryWalletBindingStore",
    "SqliteWalletBindingStore",
    "WalletBindingService",
    "WalletBindingStore",
    "build_binding_message",
]


class BindingError(Exception):
    """Base for identity-binding failures."""


class BindingSignatureError(BindingError):
    """The submitted signature did not recover to the claimed wallet address."""


class BindingConflictError(BindingError):
    """A conflicting binding exists — either the wallet is already bound to a
    different node, or the node is already bound to a different wallet."""


@dataclass(frozen=True)
class IdentityBinding:
    wallet_address: str         # EIP-55 checksummed
    node_id_hex: str            # 32-char hex (sha256(public_key)[:32])
    bound_at_unix: int
    wallet_signature: str       # 0x-prefixed hex, EIP-191 personal_sign format
    signing_message_hash: str   # 0x-prefixed keccak256 of the EIP-191-prefixed bytes


def build_binding_message(wallet_address: str, node_id_hex: str, issued_at_iso: str) -> str:
    """Canonical binding attestation message.

    Structure is deliberately human-readable so wallet UIs show the user
    exactly what they are authorising. Order and whitespace are fixed —
    verifiers re-build the message from (wallet, node, issued_at) tuples,
    so any drift here is a verification failure.
    """
    return (
        "PRSM Identity Binding\n"
        f"Wallet: {to_checksum_address(wallet_address)}\n"
        f"Node: {node_id_hex}\n"
        f"Issued-At: {issued_at_iso}"
    )


class WalletBindingStore(Protocol):
    def get_by_wallet(self, wallet_address: str) -> Optional[IdentityBinding]: ...
    def get_by_node_id(self, node_id_hex: str) -> Optional[IdentityBinding]: ...
    def insert(self, binding: IdentityBinding) -> None: ...


class InMemoryWalletBindingStore:
    """Process-local store keyed by EIP-55 wallet address and node_id."""

    def __init__(self) -> None:
        self._by_wallet: Dict[str, IdentityBinding] = {}
        self._by_node: Dict[str, IdentityBinding] = {}

    def get_by_wallet(self, wallet_address: str) -> Optional[IdentityBinding]:
        return self._by_wallet.get(to_checksum_address(wallet_address))

    def get_by_node_id(self, node_id_hex: str) -> Optional[IdentityBinding]:
        return self._by_node.get(node_id_hex)

    def insert(self, binding: IdentityBinding) -> None:
        self._by_wallet[binding.wallet_address] = binding
        self._by_node[binding.node_id_hex] = binding


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS identity_bindings (
    wallet_address        TEXT PRIMARY KEY,
    node_id_hex           TEXT NOT NULL UNIQUE,
    bound_at_unix         INTEGER NOT NULL,
    wallet_signature      TEXT NOT NULL,
    signing_message_hash  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_identity_bindings_node_id
    ON identity_bindings(node_id_hex);
"""


class SqliteWalletBindingStore:
    """SQLite-backed binding store for Foundation deployments.

    Migration to Postgres is straightforward — the schema has no SQLite-
    specific types or pragmas.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_by_wallet(self, wallet_address: str) -> Optional[IdentityBinding]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM identity_bindings WHERE wallet_address = ?",
                (to_checksum_address(wallet_address),),
            ).fetchone()
        return _row_to_binding(row) if row else None

    def get_by_node_id(self, node_id_hex: str) -> Optional[IdentityBinding]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM identity_bindings WHERE node_id_hex = ?",
                (node_id_hex,),
            ).fetchone()
        return _row_to_binding(row) if row else None

    def insert(self, binding: IdentityBinding) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO identity_bindings "
                "(wallet_address, node_id_hex, bound_at_unix, "
                " wallet_signature, signing_message_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    binding.wallet_address,
                    binding.node_id_hex,
                    binding.bound_at_unix,
                    binding.wallet_signature,
                    binding.signing_message_hash,
                ),
            )
            conn.commit()


def _row_to_binding(row: sqlite3.Row) -> IdentityBinding:
    return IdentityBinding(
        wallet_address=row["wallet_address"],
        node_id_hex=row["node_id_hex"],
        bound_at_unix=row["bound_at_unix"],
        wallet_signature=row["wallet_signature"],
        signing_message_hash=row["signing_message_hash"],
    )


class WalletBindingService:
    """Orchestrates the two-step onboarding:

      1. `sign_in(wallet)` — returns `(node_id_hex, is_new_user)`. For a new
         user, the node_id is freshly generated server-side; the service does
         NOT persist anything until `bind()` is called. The client is
         responsible for remembering the node_id across the sign-in →
         binding-attestation round-trip (plan §4.2).

      2. `bind(wallet, node_id, signature, issued_at)` — verifies the
         EIP-191 signature over the canonical binding message and records
         the binding. Idempotent: calling bind() again with the same
         (wallet, node_id) returns the stored record unchanged.
    """

    def __init__(
        self,
        store: WalletBindingStore,
        *,
        node_id_factory: Callable[[], NodeIdentity] = generate_node_identity,
    ) -> None:
        self._store = store
        self._node_id_factory = node_id_factory

    def sign_in(self, wallet_address: str) -> tuple[str, bool]:
        existing = self._store.get_by_wallet(wallet_address)
        if existing is not None:
            return existing.node_id_hex, False
        return self._node_id_factory().node_id, True

    def bind(
        self,
        wallet_address: str,
        node_id_hex: str,
        signature: str,
        issued_at_iso: str,
        *,
        now_unix: Optional[int] = None,
    ) -> IdentityBinding:
        checksum = to_checksum_address(wallet_address)

        # Idempotency + conflict checks before crypto verification — a stored
        # record with a different node_id is a permanent conflict, so we
        # refuse cheaply rather than verify a signature we're going to
        # discard.
        existing_by_wallet = self._store.get_by_wallet(checksum)
        if existing_by_wallet is not None:
            if existing_by_wallet.node_id_hex != node_id_hex:
                raise BindingConflictError(
                    f"wallet {checksum} already bound to node "
                    f"{existing_by_wallet.node_id_hex}"
                )
            return existing_by_wallet

        existing_by_node = self._store.get_by_node_id(node_id_hex)
        if existing_by_node is not None and existing_by_node.wallet_address != checksum:
            raise BindingConflictError(
                f"node {node_id_hex} already bound to wallet "
                f"{existing_by_node.wallet_address}"
            )

        message = build_binding_message(checksum, node_id_hex, issued_at_iso)
        encoded = encode_defunct(text=message)
        try:
            recovered = Account.recover_message(encoded, signature=signature)
        except Exception as exc:
            raise BindingSignatureError(f"signature recovery failed: {exc}") from exc

        if to_checksum_address(recovered) != checksum:
            raise BindingSignatureError(
                f"recovered {recovered} does not match claimed wallet {checksum}"
            )

        signing_hash = "0x" + keccak(encoded.body).hex()
        binding = IdentityBinding(
            wallet_address=checksum,
            node_id_hex=node_id_hex,
            bound_at_unix=now_unix if now_unix is not None else int(time.time()),
            wallet_signature=signature,
            signing_message_hash=signing_hash,
        )
        self._store.insert(binding)
        return binding

    def get_by_wallet(self, wallet_address: str) -> Optional[IdentityBinding]:
        return self._store.get_by_wallet(wallet_address)

    def get_by_node_id(self, node_id_hex: str) -> Optional[IdentityBinding]:
        return self._store.get_by_node_id(node_id_hex)
