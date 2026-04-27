"""
PRSM privacy-budget persistence — signed, append-only journal of DP-ε spends.

Phase 3.x.4 closes the in-memory caveat from Phase 3.x.1's privacy-budget
gating: ``PrivacyBudgetTracker`` was process-local; a node restart
silently dropped cumulative ε state, and ``reset()`` produced no audit
trail. This module wraps the existing tracker in a chained-per-entry
signed journal so:

- Restart-survival: replay reconstitutes ``total_spent`` on construction.
- Tamper-evidence: every entry carries its predecessor's hash; any
  historical edit invalidates every subsequent signature.
- Reset auditing: ``reset()`` becomes a signed RESET entry rather than
  a silent in-memory wipe. Auditors see exactly when and at what
  cumulative ε.

Trust model: same as Phase 3.x.2 model registry. Open publishing,
signature-verify against a pubkey sidecar at the journal root.
Cross-node chain-of-custody (the on-chain anchor) is deferred to
Phase 3.x.3.

See ``docs/2026-04-27-phase3.x.4-persistent-privacy-budget-design-plan.md``.
"""

from prsm.security.privacy_budget_persistence.models import (
    ENTRY_SCHEMA_VERSION,
    ENTRY_SIGNING_DOMAIN,
    GENESIS_PREV_HASH,
    PrivacyBudgetEntry,
    PrivacyBudgetEntryType,
)
from prsm.security.privacy_budget_persistence.signing import (
    is_signed,
    sign_entry,
    verify_entry,
)
from prsm.security.privacy_budget_persistence.store import (
    InMemoryPrivacyBudgetStore,
    JournalCorruptionError,
    OutOfOrderAppendError,
    PrivacyBudgetStore,
    PrivacyBudgetStoreError,
    hash_entry_payload,
)

__all__ = [
    # Models (Task 1)
    "ENTRY_SCHEMA_VERSION",
    "ENTRY_SIGNING_DOMAIN",
    "GENESIS_PREV_HASH",
    "PrivacyBudgetEntry",
    "PrivacyBudgetEntryType",
    # Signing (Task 2)
    "is_signed",
    "sign_entry",
    "verify_entry",
    # Store (Task 3)
    "PrivacyBudgetStore",
    "InMemoryPrivacyBudgetStore",
    "hash_entry_payload",
    # Exceptions (Task 3)
    "PrivacyBudgetStoreError",
    "OutOfOrderAppendError",
    "JournalCorruptionError",
]
