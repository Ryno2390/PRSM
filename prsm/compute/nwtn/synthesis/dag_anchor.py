"""
PRSM DAG Anchor
===============

For major project milestones (checkpoint merges), this module anchors the
Project Ledger entry's ``chain_hash`` to the PRSM DAG ledger.

This provides **external, decentralised verification**: the ledger's
hash chain is independent (tamper-evident without network access), but
DAG-anchoring makes the hash publicly verifiable by any PRSM node without
trusting the local machine.

When to anchor
--------------
Not every nightly synthesis warrants a DAG anchor — that would be
expensive in FTNS.  The recommended pattern (enforced by the caller) is:

- Anchor at every ``CheckpointReviewer`` merge (milestone reached).
- Optionally anchor at the end of major phases.
- Routine nightly syntheses are *not* anchored (saved to the Project
  Ledger locally only).

Graceful degradation
--------------------
If the PRSM network is unavailable (no DAG ledger connection, no FTNS
balance, or network error), the anchor silently fails and returns an
``AnchorReceipt`` with ``status='unavailable'``.  The Project Ledger
remains valid — the hash chain provides tamper-evidence regardless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Data models
# ======================================================================

@dataclass
class AnchorReceipt:
    """Result of a DAG anchoring attempt."""

    ledger_entry_index: int
    chain_hash: str
    status: str
    """'anchored' | 'unavailable' | 'error'"""

    dag_tx_id: Optional[str] = None
    """DAG transaction ID if status == 'anchored'."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == "anchored"


# ======================================================================
# DAGAnchor
# ======================================================================

class DAGAnchor:
    """
    Anchors a Project Ledger entry's chain hash to the PRSM DAG.

    Parameters
    ----------
    dag_ledger : optional
        A ``DAGLedger`` instance.  If None, all anchor calls return
        ``status='unavailable'`` (graceful no-op).
    wallet_id : str, optional
        FTNS wallet used to pay for the anchor transaction.
    ftns_per_anchor : float
        FTNS cost per anchor transaction (default: 0.01).
    """

    def __init__(
        self,
        dag_ledger=None,
        wallet_id: Optional[str] = None,
        ftns_per_anchor: float = 0.01,
    ) -> None:
        self._dag = dag_ledger
        self._wallet = wallet_id
        self._cost = ftns_per_anchor

    async def anchor(
        self,
        entry_index: int,
        chain_hash: str,
        session_id: str,
    ) -> AnchorReceipt:
        """
        Anchor *chain_hash* for ledger entry *entry_index* to the PRSM DAG.

        Parameters
        ----------
        entry_index : int
        chain_hash : str
            The ledger entry's chain_hash (hex SHA-256).
        session_id : str

        Returns
        -------
        AnchorReceipt
        """
        if self._dag is None:
            logger.debug(
                "DAGAnchor: no DAG ledger configured — skipping anchor for entry #%d",
                entry_index,
            )
            return AnchorReceipt(
                ledger_entry_index=entry_index,
                chain_hash=chain_hash,
                status="unavailable",
            )

        try:
            tx_id = await self._submit_anchor(entry_index, chain_hash, session_id)
            logger.info(
                "DAGAnchor: entry #%d anchored (tx=%s)", entry_index, tx_id
            )
            return AnchorReceipt(
                ledger_entry_index=entry_index,
                chain_hash=chain_hash,
                status="anchored",
                dag_tx_id=tx_id,
            )
        except Exception as exc:
            logger.warning(
                "DAGAnchor: anchor failed for entry #%d: %s", entry_index, exc
            )
            return AnchorReceipt(
                ledger_entry_index=entry_index,
                chain_hash=chain_hash,
                status="error",
                error=str(exc),
            )

    async def _submit_anchor(
        self, entry_index: int, chain_hash: str, session_id: str
    ) -> str:
        """Submit a DAG transaction containing the ledger hash as metadata."""
        from prsm.node.dag_ledger import TransactionType

        memo = (
            f"nwtn:ledger:anchor:"
            f"entry={entry_index}:"
            f"session={session_id}:"
            f"hash={chain_hash[:16]}"
        )

        tx = await self._dag.submit_transaction(
            tx_type=TransactionType.SYSTEM,
            amount=self._cost,
            from_wallet=self._wallet or "nwtn-system",
            to_wallet="nwtn-system",
            metadata={"ledger_chain_hash": chain_hash, "memo": memo},
        )

        # Different DAGLedger implementations return different types —
        # handle both string tx_id and object with .tx_id attribute
        if isinstance(tx, str):
            return tx
        return getattr(tx, "tx_id", str(tx))
