"""Sprint 761 — operator-facing storage bandwidth cap env vars.

The `BandwidthLimiter` mechanism in `prsm.core.bandwidth_limiter`
has existed since before this session. `storage_provider.py`
constructs one but always with hardcoded 0 (unlimited). Operators
on metered/capped consumer ISPs (or wanting their gaming PC to
not saturate upload during work hours) had no way to set caps.

Sprint 761 wires two env vars into the constructor:
- `PRSM_STORAGE_UPLOAD_MBPS` (float, default 0 = unlimited)
- `PRSM_STORAGE_DOWNLOAD_MBPS` (float, default 0 = unlimited)

Backward-compat: unset env → 0 → unlimited → identical to pre-761
behavior. Existing operator fleet sees no behavior change.
"""
from __future__ import annotations

import os


def setup_function():
    """Ensure clean env state before each test."""
    os.environ.pop("PRSM_STORAGE_UPLOAD_MBPS", None)
    os.environ.pop("PRSM_STORAGE_DOWNLOAD_MBPS", None)


def teardown_function():
    os.environ.pop("PRSM_STORAGE_UPLOAD_MBPS", None)
    os.environ.pop("PRSM_STORAGE_DOWNLOAD_MBPS", None)


def _build_storage_provider():
    """Construct a minimal StorageProvider for inspecting its
    initial bandwidth-limit state."""
    from unittest.mock import MagicMock
    from prsm.node.storage_provider import StorageProvider
    identity = MagicMock()
    identity.node_id = "test-node"
    return StorageProvider(
        identity=identity,
        gossip=MagicMock(),
        ledger=MagicMock(),
    )


def test_default_unset_env_means_unlimited():
    """Backward-compat: PRSM_STORAGE_UPLOAD_MBPS unset → 0
    (unlimited). Operators who haven't set the env see the
    pre-761 behavior."""
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 0.0
    assert sp.download_mbps_limit == 0.0
    assert sp.bandwidth_limiter.upload_limit_mbps == 0.0
    assert sp.bandwidth_limiter.download_limit_mbps == 0.0


def test_upload_env_sets_limit():
    """PRSM_STORAGE_UPLOAD_MBPS=10 → 10 Mbps cap."""
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = "10"
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 10.0
    assert sp.bandwidth_limiter.upload_limit_mbps == 10.0


def test_download_env_sets_limit():
    """PRSM_STORAGE_DOWNLOAD_MBPS=50 → 50 Mbps cap."""
    os.environ["PRSM_STORAGE_DOWNLOAD_MBPS"] = "50"
    sp = _build_storage_provider()
    assert sp.download_mbps_limit == 50.0
    assert sp.bandwidth_limiter.download_limit_mbps == 50.0


def test_float_value_supported():
    """Float values like 2.5 are common for consumer ISPs."""
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = "2.5"
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 2.5


def test_non_float_value_safely_defaults_to_zero():
    """Operator typos in PRSM_STORAGE_UPLOAD_MBPS=foo shouldn't
    crash daemon-start; safe-default to 0 (unlimited) with a
    warning log."""
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = "ten-megabits"
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 0.0


def test_negative_value_safely_defaults_to_zero():
    """Negative Mbps is nonsensical; safe-default to 0."""
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = "-5"
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 0.0


def test_both_envs_independent():
    """Upload + download envs are separately tunable (asymmetric
    ISPs — common in cable / DSL)."""
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = "5"
    os.environ["PRSM_STORAGE_DOWNLOAD_MBPS"] = "100"
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 5.0
    assert sp.download_mbps_limit == 100.0


def test_whitespace_handled():
    """Operators copy-paste from configs with stray whitespace."""
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = "  10.0  "
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 10.0


def test_empty_string_treated_as_unset():
    """`Environment=PRSM_STORAGE_UPLOAD_MBPS=` (empty value) →
    treated same as unset → 0/unlimited."""
    os.environ["PRSM_STORAGE_UPLOAD_MBPS"] = ""
    sp = _build_storage_provider()
    assert sp.upload_mbps_limit == 0.0
