"""Sprint 572 — F24 fix: TransactionType.REWARD enum member.

Three production call sites credit FTNS via the LocalLedger using
``tx_type=TransactionType.REWARD``:

- ``prsm/node/bittorrent_provider.py:320`` — seeder reward (per-GB
  uploaded, reward_loop every reward_interval_secs)
- ``prsm/node/node.py:1587`` — generic mint_tokens (staking rewards,
  appeal refunds)
- ``prsm/node/bittorrent_proofs.py:401`` — BitTorrent storage-proof
  verification reward

But ``TransactionType`` in ``prsm/node/local_ledger.py`` and
``prsm/node/dag_ledger.py`` has no ``REWARD`` member — only
``WELCOME_GRANT``, ``COMPUTE_PAYMENT``, ``COMPUTE_EARNING``,
``STORAGE_REWARD``, ``CONTENT_ROYALTY``, ``TRANSFER``,
``BRIDGE_DEPOSIT``, ``BRIDGE_WITHDRAW`` (+ ``APPROVAL``, ``SYSTEM``,
``GENESIS`` in dag_ledger).

Result: every seeder reward_loop tick on the bootstrap-us droplet
logged ``Reward loop error: type object 'TransactionType' has no
attribute 'REWARD'`` (F24 — hourly spam since sprint 458/459 deploy
on 2026-05-15). Real reward credits never landed; staking_manager
+ bittorrent rewards silently no-op'd.

Sprint 572 adds ``REWARD = "reward"`` to BOTH TransactionType
enums (local_ledger + dag_ledger) so the three call sites resolve.
The call sites encode caller intent ("a generic reward credit");
the enum was missing the member, not the call sites being wrong.
"""
from __future__ import annotations

import pytest


def test_local_ledger_transaction_type_has_reward():
    """local_ledger.TransactionType MUST include REWARD = 'reward'."""
    from prsm.node.local_ledger import TransactionType
    assert hasattr(TransactionType, "REWARD"), (
        "TransactionType.REWARD missing — F24 root cause; "
        "3 call sites in node.py/bittorrent_provider.py/"
        "bittorrent_proofs.py rely on this member"
    )
    assert TransactionType.REWARD.value == "reward"


def test_dag_ledger_transaction_type_has_reward():
    """dag_ledger.TransactionType MUST also include REWARD for parity.

    Both enums share the same value-set across the node layer. If
    one has REWARD and the other doesn't, a code path that funnels
    through dag_ledger raises while the local_ledger path succeeds.
    """
    from prsm.node.dag_ledger import TransactionType
    assert hasattr(TransactionType, "REWARD"), (
        "dag_ledger.TransactionType.REWARD missing — parity with "
        "local_ledger required"
    )
    assert TransactionType.REWARD.value == "reward"


def test_bittorrent_provider_reward_loop_call_site_resolves():
    """sprint-572 invariant: the import + member-access at the
    bittorrent_provider reward_loop site doesn't raise.
    """
    # Mirror the call-site import + member access exactly.
    from prsm.node.local_ledger import TransactionType
    tx_type = TransactionType.REWARD  # would raise pre-572
    assert tx_type.value == "reward"


def test_mint_tokens_call_site_resolves():
    """node.mint_tokens uses TransactionType.REWARD for staking
    rewards + appeal refunds. Must resolve cleanly.
    """
    from prsm.node.local_ledger import TransactionType
    tx_type = TransactionType.REWARD
    assert tx_type.value == "reward"


def test_bittorrent_proofs_award_seeder_call_site_resolves():
    """award_seeder() in bittorrent_proofs.py uses
    TransactionType.REWARD for storage-proof verification rewards.
    """
    from prsm.node.local_ledger import TransactionType
    tx_type = TransactionType.REWARD
    assert tx_type.value == "reward"


def test_reward_distinct_from_storage_reward():
    """REWARD and STORAGE_REWARD are semantically different and must
    NOT collapse to the same value (would break ledger-history
    filtering).
    """
    from prsm.node.local_ledger import TransactionType
    assert TransactionType.REWARD.value != TransactionType.STORAGE_REWARD.value
