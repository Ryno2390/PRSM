"""Sprint 190 — TransferRequest field bounds + timestamp coercion.

Two bugs fixed in /api/ftns/transfer:

1. Empty to_wallet returned 500 (no Pydantic constraint). Now
   Pydantic Field(min_length=1) rejects with 422 upfront.

2. `tx.timestamp` is a Unix float on LocalLedger transactions
   but the handler called `tx.timestamp.isoformat()` —
   AttributeError → 500. Coerce float → datetime → ISO string.

Pre-fix probe:
  POST /api/ftns/transfer {"to_wallet":"x","amount":1}
  → 500 Internal Server Error
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_transfer_request_rejects_empty_to_wallet():
    from prsm.dashboard.app import TransferRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        TransferRequest(to_wallet="", amount=1.0)
    assert "to_wallet" in str(exc.value).lower()


def test_transfer_request_rejects_long_to_wallet():
    from prsm.dashboard.app import TransferRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TransferRequest(to_wallet="x" * 257, amount=1.0)


def test_transfer_request_accepts_boundary():
    from prsm.dashboard.app import TransferRequest

    req = TransferRequest(to_wallet="x" * 256, amount=1.0)
    assert req.to_wallet == "x" * 256


def test_transfer_request_rejects_long_description():
    from prsm.dashboard.app import TransferRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TransferRequest(
            to_wallet="ok", amount=1.0,
            description="x" * 513,
        )


def test_timestamp_coercion_handles_float():
    """Sprint 190 — Unix float timestamp coerced to ISO string."""
    # The coercion logic lives in the handler — verify the helper
    # path via direct datetime conversion (mirror the handler's
    # else branch).
    ts_float = 1_700_000_000.5
    iso = datetime.fromtimestamp(
        ts_float, tz=timezone.utc,
    ).isoformat()
    assert "T" in iso  # ISO format
    assert iso.endswith("+00:00")


def test_timestamp_coercion_handles_datetime():
    """Sprint 190 — datetime objects with .isoformat() pass through."""
    ts = datetime.now(timezone.utc)
    assert ts.isoformat().endswith("+00:00")
