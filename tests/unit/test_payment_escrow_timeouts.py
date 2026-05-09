"""PaymentEscrow timeout + cleanup-interval configurability.

v1 hardcoded:
  - default_timeout = 3600s (1h escrow expiry)
  - periodic_cleanup interval = 600s (10min)

v2 ships configurable via constructor params + env vars
(2026-05-09):
  - PRSM_ESCROW_TIMEOUT_SEC (default 3600)
  - PRSM_ESCROW_CLEANUP_INTERVAL_SEC (default 600)

Operators with high-throughput workloads (compute jobs settling
in seconds) want shorter timeouts so stuck escrows refund
promptly. Operators with long-running compute (multi-hour
inference) want longer timeouts so legitimate jobs aren't
auto-refunded out from under them.
"""
from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.payment_escrow import (
    EscrowEntry, EscrowStatus, PaymentEscrow,
)


def _ledger():
    """Stub ledger that always succeeds + tracks transfers."""
    led = MagicMock()
    led.get_balance = AsyncMock(return_value=100.0)
    tx = MagicMock()
    tx.tx_id = "tx-stub"
    led.transfer = AsyncMock(return_value=tx)
    led.create_wallet = AsyncMock(return_value=None)
    return led


# ──────────────────────────────────────────────────────────────────────
# Constructor param
# ──────────────────────────────────────────────────────────────────────


class TestEscrowTimeoutConstructor:
    def test_default_timeout_param_overrides_default(self):
        escrow = PaymentEscrow(
            ledger=_ledger(), node_id="test-node",
            default_timeout=120.0,
        )
        assert escrow.default_timeout == 120.0

    def test_cleanup_interval_param_overrides_default(self):
        escrow = PaymentEscrow(
            ledger=_ledger(), node_id="test-node",
            cleanup_interval=15.0,
        )
        assert escrow.cleanup_interval == 15.0

    def test_default_timeout_when_no_param_no_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_ESCROW_TIMEOUT_SEC", None)
            escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
        # v1 default preserved.
        assert escrow.default_timeout == 3600.0

    def test_default_cleanup_interval_when_no_param_no_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_ESCROW_CLEANUP_INTERVAL_SEC", None)
            escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
        assert escrow.cleanup_interval == 600.0


# ──────────────────────────────────────────────────────────────────────
# Env-var resolution
# ──────────────────────────────────────────────────────────────────────


class TestEscrowTimeoutEnv:
    def test_env_var_sets_default_timeout(self):
        with patch.dict(
            os.environ, {"PRSM_ESCROW_TIMEOUT_SEC": "300"},
        ):
            escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
        assert escrow.default_timeout == 300.0

    def test_env_var_sets_cleanup_interval(self):
        with patch.dict(
            os.environ, {"PRSM_ESCROW_CLEANUP_INTERVAL_SEC": "30"},
        ):
            escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
        assert escrow.cleanup_interval == 30.0

    def test_constructor_param_wins_over_env(self):
        """Explicit constructor param overrides env var."""
        with patch.dict(
            os.environ, {"PRSM_ESCROW_TIMEOUT_SEC": "9999"},
        ):
            escrow = PaymentEscrow(
                ledger=_ledger(), node_id="test-node",
                default_timeout=42.0,
            )
        assert escrow.default_timeout == 42.0

    def test_invalid_env_falls_back_to_default(self):
        """Non-numeric env value → log + fall back to default."""
        with patch.dict(
            os.environ, {"PRSM_ESCROW_TIMEOUT_SEC": "not_a_number"},
        ):
            escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
        assert escrow.default_timeout == 3600.0

    def test_zero_or_negative_env_falls_back_to_default(self):
        """Zero / negative timeout would be a footgun (every escrow
        instantly expires). Treat as invalid + use default."""
        with patch.dict(
            os.environ, {"PRSM_ESCROW_TIMEOUT_SEC": "-1"},
        ):
            escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
        assert escrow.default_timeout == 3600.0
        with patch.dict(
            os.environ, {"PRSM_ESCROW_TIMEOUT_SEC": "0"},
        ):
            escrow = PaymentEscrow(ledger=_ledger(), node_id="test-node")
        assert escrow.default_timeout == 3600.0


# ──────────────────────────────────────────────────────────────────────
# cleanup_expired_escrows uses configured timeout
# ──────────────────────────────────────────────────────────────────────


class TestCleanupUsesConfiguredTimeout:
    def test_short_timeout_marks_old_escrow_expired(self):
        escrow = PaymentEscrow(
            ledger=_ledger(), node_id="test-node",
            default_timeout=0.1,  # 100ms
        )
        # Seed an escrow with created_at far in the past.
        entry = EscrowEntry(
            escrow_id="esc-1", job_id="job-1",
            requester_id="req", amount=5.0,
            status=EscrowStatus.PENDING,
            created_at=time.time() - 60.0,  # 60s old
        )
        escrow._escrows[entry.escrow_id] = entry
        # cleanup_expired_escrows should refund (60s > 0.1s).
        cleaned = asyncio.run(escrow.cleanup_expired_escrows())
        assert cleaned == 1
        assert entry.status == EscrowStatus.REFUNDED

    def test_long_timeout_keeps_recent_escrow_pending(self):
        escrow = PaymentEscrow(
            ledger=_ledger(), node_id="test-node",
            default_timeout=86400.0,  # 1 day
        )
        entry = EscrowEntry(
            escrow_id="esc-1", job_id="job-1",
            requester_id="req", amount=5.0,
            status=EscrowStatus.PENDING,
            created_at=time.time() - 60.0,  # 60s old
        )
        escrow._escrows[entry.escrow_id] = entry
        cleaned = asyncio.run(escrow.cleanup_expired_escrows())
        assert cleaned == 0
        assert entry.status == EscrowStatus.PENDING
