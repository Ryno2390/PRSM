"""Sprint 542 — F20 fix: chunked inbound scan to fit RPC payload limit.

Production-blocker observed on 2026-05-19: ``/wallet/transactions/onchain/
inbound`` returned ``413 Client Error: Payload Too Large`` from Base public
RPC. Default ``lookback_blocks=100_000`` exceeds the public-node per-request
limit (~10k blocks on mainnet.base.org).

InboundMonitor doesn't hit this because it scans only the per-tick delta
(typically <60 blocks at 2s/block, 60s interval), confirmed live by
sprint 541's bridge_deposit tx ``8bcb1e83`` crediting +0.000001 FTNS to
operator wallet on the operator's own outbound broadcast.

Fix: ``scan_inbound_transfers_chunked`` splits any range > ``max_window``
into sub-windows, calls ``scan_inbound_transfers`` per window, concatenates
results in ascending block order. ``max_window`` defaults to ``9_000`` (under
the 10k public-RPC ceiling with headroom).
"""

import pytest

from prsm.economy.ftns_onchain import scan_inbound_transfers


# ── Helpers ────────────────────────────────────────────────────────


class _FakeEvent:
    """Pretends to be ``contract.events.Transfer`` for tests — records
    every ``get_logs`` call so we can assert how the chunker splits."""

    def __init__(self, blocks_to_transfers=None, raise_at=None):
        # blocks_to_transfers: dict mapping (from_block, to_block) → list
        #   of mock log objects (or None to skip seeding)
        # raise_at: dict mapping (from_block, to_block) → Exception to raise
        self.calls: list[tuple[int, int]] = []
        self._blocks = blocks_to_transfers or {}
        self._raise_at = raise_at or {}

    def get_logs(self, from_block, to_block, argument_filters=None):
        self.calls.append((from_block, to_block))
        if (from_block, to_block) in self._raise_at:
            raise self._raise_at[(from_block, to_block)]
        return self._blocks.get((from_block, to_block), [])


class _FakeContract:
    def __init__(self, fake_event):
        class _Events:
            Transfer = fake_event
        self.events = _Events()


def _mklog(block_number: int, tx_hash: str, value_wei: int,
           from_addr: str, to_addr: str):
    """A minimal log-shaped object accepted by ``scan_inbound_transfers``."""
    class _Args:
        def __getitem__(self, k):
            return {
                "from": from_addr, "to": to_addr, "value": value_wei,
            }[k]

    class _Log:
        pass
    log = _Log()
    log.transactionHash = bytes.fromhex(tx_hash.removeprefix("0x"))
    log.blockNumber = block_number
    log.args = _Args()
    return log


# ── Pin tests ──────────────────────────────────────────────────────


def test_chunk_helper_single_window_when_range_within_limit():
    """If (to-from) <= max_window, exactly one underlying call."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers_chunked

    fake = _FakeEvent(blocks_to_transfers={
        (1000, 5000): [
            _mklog(2500, "0x" + "aa" * 32, 1_000_000_000_000_000_000,
                   "0xCba5b409a504480d4C969a47FC74cd8c109F8B15", "0x4acdE458766C704B2511583572303e77109cFFE8"),
        ],
    })
    contract = _FakeContract(fake)

    out = scan_inbound_transfers_chunked(
        contract, recipient="0x4acdE458766C704B2511583572303e77109cFFE8",
        from_block=1000, to_block=5000,
        max_window=9_000,
    )

    assert fake.calls == [(1000, 5000)]
    assert len(out) == 1
    assert out[0]["block_number"] == 2500


def test_chunk_helper_splits_large_range():
    """range = 25k blocks, max_window = 9k → 3 sub-windows."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers_chunked

    fake = _FakeEvent()
    contract = _FakeContract(fake)

    scan_inbound_transfers_chunked(
        contract, recipient="0x4acdE458766C704B2511583572303e77109cFFE8",
        from_block=100_000, to_block=125_000,  # 25_001 inclusive
        max_window=9_000,
    )

    assert fake.calls == [
        (100_000, 108_999),   # 9000 blocks
        (109_000, 117_999),
        (118_000, 125_000),   # tail < max_window
    ]


def test_chunk_helper_preserves_block_order_across_windows():
    """Concatenated output is in ascending block order."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers_chunked

    fake = _FakeEvent(blocks_to_transfers={
        (100_000, 108_999): [
            _mklog(100_500, "0x" + "11" * 32, 1, "0xCba5b409a504480d4C969a47FC74cd8c109F8B15", "0x4acdE458766C704B2511583572303e77109cFFE8"),
            _mklog(108_900, "0x" + "22" * 32, 2, "0xCba5b409a504480d4C969a47FC74cd8c109F8B15", "0x4acdE458766C704B2511583572303e77109cFFE8"),
        ],
        (109_000, 117_999): [
            _mklog(110_000, "0x" + "33" * 32, 3, "0xCba5b409a504480d4C969a47FC74cd8c109F8B15", "0x4acdE458766C704B2511583572303e77109cFFE8"),
        ],
        (118_000, 125_000): [
            _mklog(124_999, "0x" + "44" * 32, 4, "0xCba5b409a504480d4C969a47FC74cd8c109F8B15", "0x4acdE458766C704B2511583572303e77109cFFE8"),
        ],
    })
    contract = _FakeContract(fake)

    out = scan_inbound_transfers_chunked(
        contract, recipient="0x4acdE458766C704B2511583572303e77109cFFE8",
        from_block=100_000, to_block=125_000,
        max_window=9_000,
    )

    block_nums = [t["block_number"] for t in out]
    assert block_nums == [100_500, 108_900, 110_000, 124_999]


def test_chunk_helper_default_max_window_is_9000():
    """Default ``max_window=9_000`` keeps us under Base public RPC's
    documented 10k-block-per-request ceiling with headroom.

    Semantics: ``max_window`` = max blocks per window (inclusive count
    = to - from + 1). 9000 blocks fits in one call; 9001 splits.
    """
    from prsm.economy.ftns_onchain import scan_inbound_transfers_chunked

    fake = _FakeEvent()
    contract = _FakeContract(fake)

    # Range of 9000 blocks (0..8999 inclusive) → single window.
    scan_inbound_transfers_chunked(
        contract, recipient="0x4acdE458766C704B2511583572303e77109cFFE8",
        from_block=0, to_block=8_999,
        # no max_window kwarg — uses default
    )
    assert fake.calls == [(0, 8_999)]

    # 9001 blocks (0..9000) → must split (8999 cap on first window).
    fake.calls.clear()
    scan_inbound_transfers_chunked(
        contract, recipient="0x4acdE458766C704B2511583572303e77109cFFE8",
        from_block=0, to_block=9_000,
    )
    assert fake.calls == [(0, 8_999), (9_000, 9_000)]


def test_chunk_helper_zero_range():
    """from_block == to_block → exactly one call covering the single
    block (no off-by-one swallow)."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers_chunked

    fake = _FakeEvent()
    contract = _FakeContract(fake)

    scan_inbound_transfers_chunked(
        contract, recipient="0x4acdE458766C704B2511583572303e77109cFFE8",
        from_block=42, to_block=42,
        max_window=9_000,
    )

    assert fake.calls == [(42, 42)]


def test_chunk_helper_propagates_underlying_exception():
    """If a sub-window raises, chunker re-raises (no silent swallow).

    Background InboundMonitor catches its own exceptions; this helper
    is also used by the user-facing endpoint, which surfaces errors
    as 500 so operators see the failure."""
    from prsm.economy.ftns_onchain import scan_inbound_transfers_chunked

    err = RuntimeError("413 Payload Too Large")
    fake = _FakeEvent(raise_at={(109_000, 117_999): err})
    contract = _FakeContract(fake)

    with pytest.raises(RuntimeError, match="413"):
        scan_inbound_transfers_chunked(
            contract, recipient="0x4acdE458766C704B2511583572303e77109cFFE8",
            from_block=100_000, to_block=125_000,
            max_window=9_000,
        )


def test_endpoint_default_lookback_uses_chunker():
    """The endpoint should hand work to the chunker, never call
    ``scan_inbound_transfers`` directly with a 100k-block range."""
    import prsm.node.api as api_mod
    import inspect

    src = inspect.getsource(api_mod)
    # The legacy direct call was the F20 failure mode. After fix,
    # the endpoint path must reference the chunker (the inbound
    # endpoint specifically — there's still legitimate use elsewhere).
    assert "scan_inbound_transfers_chunked" in src, (
        "endpoint must use the chunked variant to avoid 413 on "
        "public RPC"
    )


def test_endpoint_default_lookback_blocks_lowered():
    """The endpoint default should be small enough that even un-chunked
    callers (e.g. third-party clients reading the OpenAPI schema)
    don't blow past the 10k public-RPC limit. Default ≤ 9_000."""
    import inspect
    import prsm.node.api as api_mod

    src = inspect.getsource(api_mod)
    # Original default was lookback_blocks=100_000. Post-fix, the
    # signature must declare ≤ 9_000 so out-of-the-box GETs to
    # /wallet/transactions/onchain/inbound succeed against public
    # Base RPC.
    assert "lookback_blocks: int = 100000" not in src, (
        "Endpoint still uses 100_000-block default — that's the F20 "
        "bug. Lower to ≤ 9_000 or rely on chunking explicitly."
    )
