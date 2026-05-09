"""PaymentEscrow.list_escrows_by_requester — public surface for
aggregate-source balance quoting.

Closes the audit-prep §7.23 honest-scope deferred item: aggregate
prsm_balance_check across on-chain FTNS + claimable royalties +
escrowed-in-pending-jobs. This test pins the public API surface.

Backwards-compat: existing PaymentEscrow public surface unchanged.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.payment_escrow import (
    EscrowEntry,
    EscrowStatus,
    PaymentEscrow,
)


def _escrow(
    *,
    job_id: str,
    requester: str,
    amount: float = 1.0,
    status: EscrowStatus = EscrowStatus.PENDING,
) -> EscrowEntry:
    return EscrowEntry(
        escrow_id=f"esc-{job_id}",
        job_id=job_id,
        requester_id=requester,
        amount=amount,
        status=status,
    )


def _empty_escrow_manager() -> PaymentEscrow:
    """Build a PaymentEscrow with a stub ledger; tests inject
    EscrowEntry objects into _escrows directly to bypass the
    create_escrow ledger-side flow."""
    ledger = MagicMock()
    pe = PaymentEscrow(ledger=ledger, node_id="test-node")
    return pe


class TestListEscrowsByRequester:
    def test_empty_when_no_escrows(self):
        pe = _empty_escrow_manager()
        assert pe.list_escrows_by_requester("0x" + "11" * 20) == []

    def test_returns_escrows_for_matching_requester(self):
        pe = _empty_escrow_manager()
        addr = "0x" + "11" * 20
        e1 = _escrow(job_id="j1", requester=addr, amount=2.0)
        e2 = _escrow(job_id="j2", requester=addr, amount=3.0)
        pe._escrows[e1.escrow_id] = e1
        pe._escrows[e2.escrow_id] = e2
        result = pe.list_escrows_by_requester(addr)
        assert len(result) == 2
        # Order is not guaranteed (dict iteration); compare as set.
        amounts = sorted(r.amount for r in result)
        assert amounts == [2.0, 3.0]

    def test_excludes_escrows_for_other_requesters(self):
        pe = _empty_escrow_manager()
        my_addr = "0x" + "11" * 20
        other_addr = "0x" + "22" * 20
        pe._escrows["a"] = _escrow(
            job_id="j1", requester=my_addr, amount=1.0,
        )
        pe._escrows["b"] = _escrow(
            job_id="j2", requester=other_addr, amount=999.0,
        )
        result = pe.list_escrows_by_requester(my_addr)
        assert len(result) == 1
        assert result[0].requester_id == my_addr
        assert result[0].amount == 1.0

    def test_includes_pending_only_by_default(self):
        """Default: only PENDING escrows are returned (the funds
        actually at risk / locked-up). Released / refunded escrows
        are accounting history, not current liability."""
        pe = _empty_escrow_manager()
        addr = "0x" + "11" * 20
        pe._escrows["a"] = _escrow(
            job_id="j1", requester=addr, amount=1.0,
            status=EscrowStatus.PENDING,
        )
        pe._escrows["b"] = _escrow(
            job_id="j2", requester=addr, amount=2.0,
            status=EscrowStatus.RELEASED,
        )
        pe._escrows["c"] = _escrow(
            job_id="j3", requester=addr, amount=3.0,
            status=EscrowStatus.REFUNDED,
        )
        result = pe.list_escrows_by_requester(addr)
        assert len(result) == 1
        assert result[0].amount == 1.0
        assert result[0].status == EscrowStatus.PENDING

    def test_include_all_statuses_when_explicitly_requested(self):
        """Operators wanting full escrow history (e.g., for accounting
        reconciliation) can opt into all statuses."""
        pe = _empty_escrow_manager()
        addr = "0x" + "11" * 20
        pe._escrows["a"] = _escrow(
            job_id="j1", requester=addr, status=EscrowStatus.PENDING,
        )
        pe._escrows["b"] = _escrow(
            job_id="j2", requester=addr, status=EscrowStatus.RELEASED,
        )
        result = pe.list_escrows_by_requester(addr, pending_only=False)
        assert len(result) == 2

    def test_address_match_is_case_insensitive(self):
        """Addresses can be checksummed (0xAb...) or lowercased (0xab...);
        the same wallet must match regardless of casing."""
        pe = _empty_escrow_manager()
        stored_addr = "0x" + "Ab" * 20
        query_addr = "0x" + "ab" * 20
        pe._escrows["a"] = _escrow(
            job_id="j1", requester=stored_addr, amount=1.0,
        )
        result = pe.list_escrows_by_requester(query_addr)
        assert len(result) == 1
