"""
Persistent privacy-budget tracker.

Phase 3.x.4 Task 5 — wraps the in-memory ``PrivacyBudgetTracker``
in a signed, append-only journal so:

- Restart-survival: replay reconstitutes ``total_spent`` on construction.
- Tamper-evidence: ``verify_chain`` runs before replay; corruption
  raises rather than silently producing wrong cumulative ε.
- Reset auditing: ``reset()`` becomes a signed RESET entry rather
  than a silent in-memory wipe.

API parity with the parent class — every existing call site
(``can_spend / record_spend / total_spent / remaining /
get_audit_report / reset``) continues to work unchanged. Persistence
is opt-in via the ``store=`` and ``identity=`` kwargs; with both
None the tracker behaves exactly like the in-memory parent.

Trust model: same as Phase 3.x.2 — open publishing + signature-verify
against a pubkey sidecar at the journal root. Cross-node chain-of-
custody requires the on-chain anchor planned for Phase 3.x.3.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from prsm.node.identity import NodeIdentity
from prsm.security.privacy_budget import PrivacyBudgetTracker, PrivacySpend
from prsm.security.privacy_budget_persistence.models import (
    PrivacyBudgetEntry,
    PrivacyBudgetEntryType,
)
from prsm.security.privacy_budget_persistence.signing import sign_entry
from prsm.security.privacy_budget_persistence.store import (
    JournalCorruptionError,
    PrivacyBudgetStore,
)

logger = logging.getLogger(__name__)


class PersistentPrivacyBudgetTracker(PrivacyBudgetTracker):
    """``PrivacyBudgetTracker`` with a signed, append-only journal.

    Construction with both ``store`` AND ``identity``:
      1. Verify the on-disk chain (signatures + chain hashes).
         Raise :class:`JournalCorruptionError` on mismatch.
      2. Replay each entry, populating ``self._spends`` directly.
         (Historical entries skip the parent's ``can_spend`` gate —
         they were valid at the time they were written.)
      3. RESET entries clear the in-memory ``_spends`` list, matching
         what the original ``reset()`` did at the time.

    Construction with neither: behaves exactly like the in-memory
    parent. No disk I/O, no signatures.

    Construction with one but not the other: programming error —
    raise ``ValueError``. Either both are wired or neither.

    Hot path (``record_spend``):
      1. Pre-flight ``can_spend`` (parent's check). False → return
         False without journal write. Rejected spends never pollute
         the chain.
      2. Build SPEND entry chained to ``store.latest_hash()``, sign,
         append to store. If the store rejects (sequence/chain/
         signature error), the exception propagates — in-memory
         state stays untouched.
      3. Call ``super().record_spend`` to update in-memory state.

    Hot path (``reset``):
      1. Build RESET entry, sign, append.
      2. Call ``super().reset`` to clear in-memory state.

    Both paths fail closed: if the journal write raises, in-memory
    state stays consistent with disk. The next replay will produce
    the same state.
    """

    def __init__(
        self,
        max_epsilon: float = 100.0,
        *,
        store: Optional[PrivacyBudgetStore] = None,
        identity: Optional[NodeIdentity] = None,
    ) -> None:
        super().__init__(max_epsilon=max_epsilon)

        # Both-or-neither rule: one without the other is a config bug.
        if (store is None) != (identity is None):
            raise ValueError(
                "PersistentPrivacyBudgetTracker requires both store AND "
                "identity, or neither. Got "
                f"store={'set' if store else 'None'}, "
                f"identity={'set' if identity else 'None'}."
            )

        self._store = store
        self._identity = identity

        if store is None or identity is None:
            return  # in-memory mode; nothing to replay

        # Verify the chain BEFORE replaying. A tampered journal must
        # not silently reconstitute wrong total_spent — the operator
        # needs to see the corruption.
        if not store.verify_chain(identity.public_key_b64):
            raise JournalCorruptionError(
                "privacy-budget journal failed verify_chain on construction; "
                "the on-disk journal is either corrupted or signed by a "
                "different key. Refusing to silently reconstitute possibly-"
                "wrong cumulative ε state."
            )

        # Replay: populate _spends directly, bypassing parent's
        # can_spend gate (historical events were valid at the time).
        for entry in store.replay():
            if entry.entry_type == PrivacyBudgetEntryType.RESET:
                self._spends.clear()
            elif entry.entry_type == PrivacyBudgetEntryType.SPEND:
                self._spends.append(
                    PrivacySpend(
                        epsilon=entry.epsilon,
                        operation=entry.operation,
                        model_id=entry.model_id,
                        timestamp=entry.timestamp,
                    )
                )

    def record_spend(
        self, epsilon: float, operation: str, model_id: str = ""
    ) -> bool:
        # Pre-flight: same behavior as parent. Rejection here means no
        # journal write — failed spends don't pollute the chain.
        if not self.can_spend(epsilon):
            logger.warning(
                f"Privacy budget exceeded (pre-flight): "
                f"{self.total_spent + epsilon:.1f} > {self.max_epsilon}"
            )
            return False

        if self._store is not None and self._identity is not None:
            self._append_entry(
                PrivacyBudgetEntryType.SPEND, epsilon, operation, model_id
            )

        return super().record_spend(epsilon, operation, model_id)

    def reset(self) -> None:
        # Reset is a journaled event, not a silent wipe. Auditors see
        # exactly when and at what cumulative ε an operator reset.
        if self._store is not None and self._identity is not None:
            self._append_entry(PrivacyBudgetEntryType.RESET, 0.0, "", "")
        super().reset()

    # -- internals --

    def _append_entry(
        self,
        entry_type: PrivacyBudgetEntryType,
        epsilon: float,
        operation: str,
        model_id: str,
    ) -> None:
        """Build, sign, and append the next chained entry."""
        assert self._store is not None
        assert self._identity is not None
        unsigned = PrivacyBudgetEntry(
            sequence_number=len(self._store),
            entry_type=entry_type,
            node_id=self._identity.node_id,
            epsilon=epsilon,
            operation=operation,
            model_id=model_id,
            timestamp=time.time(),
            prev_entry_hash=self._store.latest_hash(),
        )
        signed = sign_entry(unsigned, self._identity)
        # If the store rejects (chain/sequence/etc.), the exception
        # propagates and in-memory state is NOT updated by the caller.
        self._store.append(signed)
