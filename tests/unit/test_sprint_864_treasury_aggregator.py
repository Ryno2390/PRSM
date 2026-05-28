"""Sprint 864 — Treasury aggregator pin tests.

Defends the fleet-wide rollup of WaaS wallet balances. Composable
test design: inject a fake WaaS client + a fake balance reader to
exercise the aggregation logic in isolation.

Pin tests:
  - Empty WaaS → zero totals + empty wallet list
  - Multiple wallets sum correctly
  - PENDING_COMMISSION wallets (no address) excluded from balance reads
  - max_wallets caps per-call load
  - Funded count = wallets with any positive balance
  - block_number = max across reads
  - One bad RPC read doesn't kill the whole rollup
  - None waas_client returns clean empty envelope
"""
from __future__ import annotations

from prsm.economy.web3.treasury_aggregator import aggregate_treasury


class _FakeRecord:
    def __init__(self, user_id, address=None, status="PROVISIONED",
                 wallet_id=None):
        self.user_id = user_id
        self.address = address
        self.status = status
        self.wallet_id = wallet_id or f"prsm-{user_id}-w"


class _FakeWaas:
    def __init__(self, records):
        self._r = records

    def list_wallets(self):
        return list(self._r)


class _FakeBalances:
    def __init__(
        self, *, address, usdc=0.0, ftns=0.0, eth=0.0, block=1,
        rpc_url="https://mock-rpc",
    ):
        self.address = address
        self.usdc = usdc
        self.usdc_units = int(usdc * 10**6)
        self.ftns = ftns
        self.ftns_units = int(ftns * 10**18)
        self.native_eth = eth
        self.native_eth_wei = int(eth * 10**18)
        self.block_number = block
        self.rpc_url = rpc_url

    def to_dict(self):
        return {
            "address": self.address,
            "usdc": self.usdc, "usdc_units": self.usdc_units,
            "ftns": self.ftns, "ftns_units": self.ftns_units,
            "native_eth": self.native_eth,
            "native_eth_wei": self.native_eth_wei,
            "block_number": self.block_number,
            "rpc_url": self.rpc_url,
        }


class _FakeReader:
    def __init__(self, balances_by_addr=None, raises_for=None):
        self._b = balances_by_addr or {}
        self._raises = raises_for or set()

    def get_balances(self, address):
        if address in self._raises:
            raise RuntimeError(f"simulated RPC failure for {address}")
        if address in self._b:
            return self._b[address]
        return _FakeBalances(address=address)


# ── Empty / None paths ───────────────────────────────────────

def test_none_waas_client_returns_empty_envelope():
    r = aggregate_treasury(
        waas_client=None, balance_reader=_FakeReader(),
    )
    assert r["overall"]["total_usdc"] == 0
    assert r["overall"]["wallet_count_total"] == 0
    assert r["wallets"] == []
    assert "note" in r


def test_empty_waas_returns_zero_totals():
    r = aggregate_treasury(
        waas_client=_FakeWaas([]),
        balance_reader=_FakeReader(),
    )
    assert r["overall"]["wallet_count_total"] == 0
    assert r["overall"]["wallet_count_with_address"] == 0
    assert r["overall"]["wallet_count_funded"] == 0
    assert r["overall"]["total_usdc"] == 0
    assert r["wallets"] == []


# ── Aggregation math ─────────────────────────────────────────

def test_multiple_wallets_sum_correctly():
    records = [
        _FakeRecord("alice", address="0x" + "11" * 20),
        _FakeRecord("bob", address="0x" + "22" * 20),
    ]
    balances = {
        "0x" + "11" * 20: _FakeBalances(
            address="0x" + "11" * 20,
            usdc=5.0, ftns=100.0, eth=0.1, block=10,
        ),
        "0x" + "22" * 20: _FakeBalances(
            address="0x" + "22" * 20,
            usdc=3.0, ftns=50.0, eth=0.05, block=12,
        ),
    }
    r = aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=_FakeReader(balances),
    )
    assert r["overall"]["total_usdc"] == 8.0
    assert r["overall"]["total_ftns"] == 150.0
    assert abs(r["overall"]["total_native_eth"] - 0.15) < 1e-9
    assert r["overall"]["wallet_count_funded"] == 2
    assert r["overall"]["block_number"] == 12  # max across reads
    assert len(r["wallets"]) == 2


def test_unfunded_wallets_not_counted_as_funded():
    records = [
        _FakeRecord("alice", address="0x" + "11" * 20),
        _FakeRecord("bob", address="0x" + "22" * 20),
    ]
    # bob has balance, alice does not
    balances = {
        "0x" + "22" * 20: _FakeBalances(
            address="0x" + "22" * 20, usdc=10.0,
        ),
    }
    r = aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=_FakeReader(balances),
    )
    assert r["overall"]["wallet_count_funded"] == 1
    assert r["overall"]["wallet_count_with_address"] == 2


# ── PENDING_COMMISSION exclusion ─────────────────────────────

def test_pending_commission_wallets_excluded_from_balance_reads():
    """Wallets without an address shouldn't trigger RPC calls (no
    point — there's no address to query)."""
    queried = []

    class _TrackingReader(_FakeReader):
        def get_balances(self, address):
            queried.append(address)
            return _FakeBalances(address=address)

    records = [
        _FakeRecord("alice", address="0x" + "11" * 20),
        _FakeRecord("bob", address=None, status="PENDING_COMMISSION"),
    ]
    r = aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=_TrackingReader(),
    )
    # bob's address-less wallet not queried
    assert queried == ["0x" + "11" * 20]
    assert r["overall"]["wallet_count_total"] == 2
    assert r["overall"]["wallet_count_with_address"] == 1


# ── max_wallets cap ──────────────────────────────────────────

def test_max_wallets_caps_rpc_load():
    records = [
        _FakeRecord(f"user_{i}", address="0x" + f"{i:02x}" * 20)
        for i in range(10)
    ]
    queried = []

    class _TrackingReader(_FakeReader):
        def get_balances(self, address):
            queried.append(address)
            return _FakeBalances(address=address)

    r = aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=_TrackingReader(),
        max_wallets=3,
    )
    assert len(queried) == 3
    assert r["overall"]["wallet_count_total"] == 10
    assert r["overall"]["wallet_count_with_address"] == 3


def test_max_wallets_none_means_unbounded():
    records = [
        _FakeRecord(f"u{i}", address="0x" + f"{i:02x}" * 20)
        for i in range(5)
    ]
    queried = []

    class _TrackingReader(_FakeReader):
        def get_balances(self, address):
            queried.append(address)
            return _FakeBalances(address=address)

    aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=_TrackingReader(),
        max_wallets=None,
    )
    assert len(queried) == 5


# ── Fail-soft on bad RPC reads ───────────────────────────────

def test_one_bad_rpc_read_does_not_kill_rollup():
    """Critical operator-UX invariant: a single flaky RPC call
    must not blank the whole treasury view. Other wallets must
    still aggregate cleanly, and the failing wallet appears with
    an `error` field for forensics."""
    bad_addr = "0x" + "ee" * 20
    good_addr = "0x" + "11" * 20
    records = [
        _FakeRecord("alice", address=good_addr),
        _FakeRecord("bob", address=bad_addr),
    ]
    balances = {good_addr: _FakeBalances(
        address=good_addr, usdc=5.0,
    )}
    reader = _FakeReader(balances, raises_for={bad_addr})

    r = aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=reader,
    )
    # Alice's balance still counted
    assert r["overall"]["total_usdc"] == 5.0
    # bob entry present with error
    bob_entry = next(
        w for w in r["wallets"] if w["user_id"] == "bob"
    )
    assert bob_entry["balances"] is None
    assert "simulated RPC failure" in bob_entry["error"]


# ── rpc_url surfaced ─────────────────────────────────────────

def test_rpc_url_surfaced_when_any_read_succeeded():
    records = [_FakeRecord("alice", address="0x" + "11" * 20)]
    balances = {"0x" + "11" * 20: _FakeBalances(
        address="0x" + "11" * 20, rpc_url="https://example-rpc",
    )}
    r = aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=_FakeReader(balances),
    )
    assert r["overall"]["rpc_url"] == "https://example-rpc"


def test_rpc_url_none_when_no_addresses():
    records = [_FakeRecord(
        "bob", address=None, status="PENDING_COMMISSION",
    )]
    r = aggregate_treasury(
        waas_client=_FakeWaas(records),
        balance_reader=_FakeReader(),
    )
    assert r["overall"]["rpc_url"] is None
