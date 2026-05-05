"""T6 (2026-05-05) — unit tests for ContentUploader._register_on_chain.

Tests the on-chain ProvenanceRegistry registration that fires after a
successful IPFS write. Uses a mock provenance client to avoid touching
the real Web3 stack.

Branches verified:
  - no provenance_client wired → returns None silently
  - no provenance_hash_hex → returns None silently
  - already-registered → returns None (idempotent)
  - happy path → returns the tx hash from the mock
  - exception during call → returns None + doesn't propagate
  - royalty rate clamped to [0, 0.98] then converted to bps
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.node.content_uploader import ContentUploader


def _make_uploader(provenance_client=None):
    """Build a minimal ContentUploader with mocks for non-relevant deps."""
    return ContentUploader(
        identity=MagicMock(),
        gossip=MagicMock(),
        ledger=MagicMock(),
        provenance_client=provenance_client,
    )


# A valid 32-byte hash, hex-encoded with 0x prefix
VALID_HASH = "0x" + "ab" * 32
ZERO_HASH = "0x" + "00" * 32


def test_no_client_returns_none():
    """No on-chain client wired → silent skip."""
    uploader = _make_uploader(provenance_client=None)
    result = uploader._register_on_chain(
        provenance_hash_hex=VALID_HASH,
        royalty_rate=0.05,
        cid="QmTest",
    )
    assert result is None


def test_no_hash_returns_none():
    """No hash computed (creator_address was None) → silent skip."""
    client = MagicMock()
    uploader = _make_uploader(provenance_client=client)
    result = uploader._register_on_chain(
        provenance_hash_hex=None,
        royalty_rate=0.05,
        cid="QmTest",
    )
    assert result is None
    client.register_content.assert_not_called()


def test_invalid_hash_length_returns_none():
    """Malformed hash (wrong byte length) → skip with warning."""
    client = MagicMock()
    uploader = _make_uploader(provenance_client=client)
    result = uploader._register_on_chain(
        provenance_hash_hex="0xdeadbeef",  # 4 bytes, not 32
        royalty_rate=0.05,
        cid="QmTest",
    )
    assert result is None
    client.register_content.assert_not_called()


def test_already_registered_skips():
    """Idempotent — already-registered hash returns None without retry."""
    client = MagicMock()
    client.is_registered.return_value = True
    uploader = _make_uploader(provenance_client=client)
    result = uploader._register_on_chain(
        provenance_hash_hex=VALID_HASH,
        royalty_rate=0.05,
        cid="QmTest",
    )
    assert result is None
    client.is_registered.assert_called_once()
    # Critical: register_content must NOT be called on duplicate
    client.register_content.assert_not_called()


def test_happy_path_returns_tx_hash():
    """Fresh registration → calls register_content + returns its tx hash."""
    client = MagicMock()
    client.is_registered.return_value = False
    client.register_content.return_value = ("0xabc123", "confirmed")
    uploader = _make_uploader(provenance_client=client)
    result = uploader._register_on_chain(
        provenance_hash_hex=VALID_HASH,
        royalty_rate=0.05,
        cid="QmTest",
    )
    assert result == "0xabc123"
    client.register_content.assert_called_once()
    # Check the args
    call_kwargs = client.register_content.call_args.kwargs
    assert len(call_kwargs["content_hash"]) == 32  # bytes, not hex string
    assert call_kwargs["royalty_rate_bps"] == 500  # 0.05 * 10000
    assert call_kwargs["metadata_uri"] == "ipfs://QmTest"


def test_exception_returns_none_without_propagating():
    """Any failure (broadcast, revert, RPC) → None, doesn't propagate."""
    client = MagicMock()
    client.is_registered.side_effect = RuntimeError("rpc down")
    uploader = _make_uploader(provenance_client=client)
    # Must NOT raise — the upload itself should never be blocked by
    # on-chain registration failure.
    result = uploader._register_on_chain(
        provenance_hash_hex=VALID_HASH,
        royalty_rate=0.05,
        cid="QmTest",
    )
    assert result is None


@pytest.mark.parametrize("rate,expected_bps", [
    (0.0, 0),         # zero rate
    (0.001, 10),      # 0.1%
    (0.01, 100),      # 1%
    (0.05, 500),      # 5%
    (0.10, 1000),     # 10%
    (0.98, 9800),     # max allowed
    (0.99, 9800),     # clamped to MAX
    (1.5, 9800),      # absurd → clamped
    (-0.1, 0),        # negative → clamped to 0
])
def test_royalty_rate_conversion_to_bps(rate, expected_bps):
    """Royalty rate float → bps int, clamped to [0, 9800]."""
    client = MagicMock()
    client.is_registered.return_value = False
    client.register_content.return_value = ("0x" + "f" * 64, "confirmed")
    uploader = _make_uploader(provenance_client=client)
    uploader._register_on_chain(
        provenance_hash_hex=VALID_HASH,
        royalty_rate=rate,
        cid="QmTest",
    )
    actual_bps = client.register_content.call_args.kwargs["royalty_rate_bps"]
    assert actual_bps == expected_bps, (
        f"rate={rate} expected_bps={expected_bps} got_bps={actual_bps}"
    )
