"""Sprint 240 — source_agent_pubkey → FTNS wallet resolution.

Closes the LAST Vision §13 deferred sub-item: "source_agent_
pubkey → FTNS wallet address resolution. v1 uses hex pubkey as
wallet identifier; production needs a node-id → wallet mapping
registry."

Why
---

The settlement layer's ``compute_split_amounts`` emits splits
keyed by ``source_agent_pubkey_hex``. PaymentEscrow's
``release_escrow_split`` treats that key as a wallet_id and
credits it directly. Pre-fix this forced operators running N
compute agents to register N separate wallets — one per pubkey.
In practice an operator has ONE FTNS wallet that should receive
the compensation for ALL of their agents.

This module provides an opt-in remapping layer. Operators
populate a JSON file mapping ``pubkey_hex -> wallet_id`` and
point ``PRSM_COMPUTE_WALLET_MAP_FILE`` at it. Settlement calls
``.resolve()`` on each split's recipient_id before passing to
``release_escrow_split``.

Fail-soft throughout: missing/corrupt/wrong-shape file = empty
map = pure pass-through (v1 backward-compat preserved).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComputeWalletMap:
    """Immutable pubkey_hex → wallet_id mapping. Use
    ``from_env()`` for production loading, ``from_mapping()``
    for tests."""

    _map: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None,
    ) -> "ComputeWalletMap":
        """Build from ``PRSM_COMPUTE_WALLET_MAP_FILE`` env var.

        Empty env / missing file / corrupt JSON / wrong shape =
        empty map. Logged as warnings so operators can debug
        without a crash."""
        env = env if env is not None else os.environ
        path = (env.get("PRSM_COMPUTE_WALLET_MAP_FILE") or "").strip()
        if not path:
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            logger.info(
                "compute_wallet_map: file %r not found; using empty map",
                path,
            )
            return cls()
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "compute_wallet_map: failed to read %r (%s); "
                "using empty map", path, e,
            )
            return cls()
        if not isinstance(raw, dict):
            logger.warning(
                "compute_wallet_map: %r is not a JSON object "
                "(got %s); using empty map", path, type(raw).__name__,
            )
            return cls()
        # Filter to str → str only.
        clean: Dict[str, str] = {
            k: v for k, v in raw.items()
            if isinstance(k, str) and isinstance(v, str) and v
        }
        if len(clean) < len(raw):
            logger.warning(
                "compute_wallet_map: dropped %d malformed entries "
                "from %r", len(raw) - len(clean), path,
            )
        return cls(_map=clean)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, str],
    ) -> "ComputeWalletMap":
        """Build directly from a mapping (tests + programmatic)."""
        return cls(_map=dict(mapping))

    def resolve(self, recipient_id: str) -> str:
        """Return the mapped wallet_id, or ``recipient_id`` itself
        when no mapping exists. Pure pass-through preserves v1
        behavior for un-migrated operators."""
        return self._map.get(recipient_id, recipient_id)

    def __len__(self) -> int:
        return len(self._map)


def resolve_splits(
    splits: List[Tuple[str, float]],
    wallet_map: ComputeWalletMap,
) -> List[Tuple[str, float]]:
    """Apply ``wallet_map.resolve()`` to every recipient in a
    splits list. Amounts unchanged. Recipients not present in
    the map fall through to themselves."""
    return [(wallet_map.resolve(r), amt) for r, amt in splits]
