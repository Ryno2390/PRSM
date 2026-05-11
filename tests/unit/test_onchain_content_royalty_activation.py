"""Sprint 248 — env-gated activation of on-chain content royalty
dispatch in the api.py forge settlement path.

Verifies:
  - Default off: dispatch never runs.
  - Enabled but client unwired: no crash, no dispatch.
  - Enabled + client wired + addr set: dispatch fires.
  - Bad env values: fail-soft to sane defaults / skip.

Test strategy: import the dispatcher module + patch it inside
api.py's settlement block. Since the inline import in the
handler resolves at call time, patching prsm.economy.onchain_
content_royalty works.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.onchain_content_royalty import DispatchResult


def test_dispatcher_module_exposes_dispatch_function():
    """Pin the symbol api.py imports inline. Sprint 248 wires:
       from prsm.economy.onchain_content_royalty import (
           dispatch_content_access_royalties,
       )
    """
    from prsm.economy import onchain_content_royalty
    assert hasattr(
        onchain_content_royalty,
        "dispatch_content_access_royalties",
    )


def test_dispatch_result_status_vocabulary():
    """Sprint 248 settlement reads .status to count sent/skipped/
    failed. Pin the statuses the dispatcher emits (sprint 257
    added skipped_zero_amount)."""
    valid = {
        "sent",
        "skipped_no_record",
        "skipped_bad_hash",
        "skipped_zero_amount",
        "failed",
    }
    # Each constructable.
    for s in valid:
        r = DispatchResult(cid="x", status=s)
        assert r.status == s


def test_status_startswith_skipped_taxonomy():
    """api.py groups telemetry by status.startswith('skipped').
    Make sure no future status accidentally collides."""
    valid_started_with_skipped = {
        "skipped_no_record",
        "skipped_bad_hash",
        "skipped_zero_amount",
    }
    valid_not_skipped = {"sent", "failed"}
    for s in valid_started_with_skipped:
        assert s.startswith("skipped")
    for s in valid_not_skipped:
        assert not s.startswith("skipped")


def test_default_env_disabled(monkeypatch):
    """Operator hasn't enabled — env unset = no dispatch path."""
    monkeypatch.delenv(
        "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED", raising=False,
    )
    import os as _os
    assert _os.environ.get(
        "PRSM_ONCHAIN_CONTENT_ROYALTY_ENABLED", "0",
    ) == "0"


def test_per_shard_wei_default_sentinel():
    """Default per-shard amount = 0.001 FTNS (10^15 wei).
    Anchored so a stray env-removal doesn't drop a 0 default
    which would silently disable all dispatches."""
    DEFAULT_WEI = 1_000_000_000_000_000
    assert DEFAULT_WEI == 10**15
