"""Sprint 302 — formal-invariant harness for §14 item 4.

Vision §14 item 4: "Formal verification on highest-value
contracts. Payment escrow and royalty distribution
contracts undergo formal-methods verification, not just
standard audit."

This sprint ships the SPEC LAYER + RUNTIME PROBE: pinned
invariants in code (the formal spec), a checker that
verifies them against on-chain state via an injected
backend, and a public read surface so anyone can audit
what PRSM has committed to. Actual symbolic-execution runs
(halmos, Certora) consume the same registry on a follow-on
sprint.

Highest-value target this sprint: RoyaltyDistributor v2.
Five pinned invariants cover anti-confiscation (network fee
fixed at 2%), ownership integrity, treasury immutability,
solvency (THE money invariant), and pause-state observability.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.formal_invariants import (
    INVARIANT_REGISTRY,
    Invariant,
    InvariantChecker,
    InvariantKind,
    InvariantResult,
    InvariantSeverity,
    InvariantStatus,
    list_invariants_for_contract,
)


# ── Enums ────────────────────────────────────────────


def test_severity_values():
    assert InvariantSeverity.CRITICAL.value == "critical"
    assert InvariantSeverity.HIGH.value == "high"
    assert InvariantSeverity.MEDIUM.value == "medium"


def test_status_values():
    assert InvariantStatus.PASS.value == "pass"
    assert InvariantStatus.FAIL.value == "fail"
    assert InvariantStatus.SKIPPED.value == "skipped"


def test_kind_values():
    # Pinned check kinds — adding a new one requires updating
    # the dispatcher
    assert InvariantKind.UINT256_EQ.value == "uint256_eq"
    assert InvariantKind.UINT256_GTE.value == "uint256_gte"
    assert InvariantKind.ADDRESS_EQ.value == "address_eq"
    assert InvariantKind.BOOL_READ.value == "bool_read"
    assert (
        InvariantKind.BALANCE_GTE_CLAIMABLE.value
        == "balance_gte_claimable"
    )


# ── Pinned registry ──────────────────────────────────


def test_registry_has_royalty_distributor():
    assert "royalty_distributor" in INVARIANT_REGISTRY
    invariants = INVARIANT_REGISTRY["royalty_distributor"]
    assert len(invariants) >= 5


def test_registry_invariant_ids_unique():
    seen = set()
    for invs in INVARIANT_REGISTRY.values():
        for inv in invs:
            assert inv.id not in seen, (
                f"duplicate invariant id {inv.id}"
            )
            seen.add(inv.id)


def test_registry_has_network_fee_anti_tamper():
    invs = INVARIANT_REGISTRY["royalty_distributor"]
    ids = {i.id for i in invs}
    assert "INV-RD-1" in ids
    rd1 = next(i for i in invs if i.id == "INV-RD-1")
    assert rd1.kind == InvariantKind.UINT256_EQ
    assert rd1.severity == InvariantSeverity.CRITICAL
    assert rd1.expected == 200


def test_registry_has_solvency_invariant():
    """The single most important invariant —
    balance(this) >= totalClaimable. Failure = insolvency."""
    invs = INVARIANT_REGISTRY["royalty_distributor"]
    solvency = next(
        (i for i in invs
         if i.kind == InvariantKind.BALANCE_GTE_CLAIMABLE),
        None,
    )
    assert solvency is not None
    assert solvency.severity == InvariantSeverity.CRITICAL


def test_registry_has_owner_check():
    invs = INVARIANT_REGISTRY["royalty_distributor"]
    addr_eq = [
        i for i in invs
        if i.kind == InvariantKind.ADDRESS_EQ
    ]
    assert len(addr_eq) >= 2  # owner + networkTreasury


def test_list_invariants_for_unknown_contract_empty():
    assert list_invariants_for_contract("nonexistent") == []


# ── InvariantChecker — mock backend ──────────────────


class _MockBackend:
    """Returns scripted values per (addr, selector) tuple,
    or raises RuntimeError to simulate RPC failure."""

    def __init__(self):
        self.uint256: dict = {}
        self.address: dict = {}
        self.bool_v: dict = {}
        self.raise_for: set = set()

    def call_uint256(
        self, addr: str, selector: str,
    ):
        key = (addr.lower(), selector.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.uint256.get(key)

    def call_address(
        self, addr: str, selector: str,
    ):
        key = (addr.lower(), selector.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.address.get(key)

    def call_bool(
        self, addr: str, selector: str,
    ):
        key = (addr.lower(), selector.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.bool_v.get(key)

    def token_balance_of(
        self, token: str, holder: str,
    ):
        key = (token.lower(), holder.lower(), "balance")
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.uint256.get(key)


def _checker(backend) -> InvariantChecker:
    return InvariantChecker(backend=backend)


def test_check_uint256_eq_pass():
    inv = Invariant(
        id="X-1", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() == 42",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 42
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS
    assert result.value == 42


def test_check_uint256_eq_fail():
    inv = Invariant(
        id="X-2", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() == 42",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 100
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL
    assert result.value == 100


def test_check_uint256_eq_skipped_on_rpc_error():
    inv = Invariant(
        id="X-3", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() == 42",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.raise_for.add(("0xabc", "0xdead"))
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.SKIPPED
    assert "rpc" in (result.error or "").lower()


def test_check_uint256_gte_pass():
    inv = Invariant(
        id="X-4", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() >= 100",
        kind=InvariantKind.UINT256_GTE,
        selector="0xdead", expected=100,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 200
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS


def test_check_uint256_gte_fail():
    inv = Invariant(
        id="X-5", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() >= 100",
        kind=InvariantKind.UINT256_GTE,
        selector="0xdead", expected=100,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 50
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL


def test_check_address_eq_pass_case_insensitive():
    inv = Invariant(
        id="X-6", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="owner() == 0xab",
        kind=InvariantKind.ADDRESS_EQ,
        selector="0xowner",
        expected="0xABcdef0000000000000000000000000000000001",
    )
    backend = _MockBackend()
    backend.address[("0xabc", "0xowner")] = (
        "0xabcdef0000000000000000000000000000000001"
    )
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS


def test_check_address_eq_fail():
    inv = Invariant(
        id="X-7", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="owner() == ...",
        kind=InvariantKind.ADDRESS_EQ,
        selector="0xowner",
        expected="0x" + "11" * 20,
    )
    backend = _MockBackend()
    backend.address[("0xabc", "0xowner")] = "0x" + "22" * 20
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL


def test_check_bool_read_observable():
    """BOOL_READ is observability — never fails on value
    itself, just surfaces. Used for paused() etc."""
    inv = Invariant(
        id="X-8", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.MEDIUM,
        spec_text="paused() — operator-observable",
        kind=InvariantKind.BOOL_READ,
        selector="0xpaused", expected=None,
    )
    backend = _MockBackend()
    backend.bool_v[("0xabc", "0xpaused")] = True
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS
    assert result.value is True


def test_check_balance_gte_claimable_pass():
    """The solvency invariant — backend looks up
    balance(contract) and totalClaimable separately."""
    inv = Invariant(
        id="X-9", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalclaimable",
        # extra params for the ftns address (looked up via
        # a contract method too, or supplied in params)
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalclaimable",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xc0ffee"
    ftns_addr = "0x" + "ff" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[
        (contract_addr, "0xtotalclaimable")
    ] = 1_000
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 1_500
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.PASS
    assert "balance=1500" in (result.diagnostic or "")
    assert "totalClaimable=1000" in (result.diagnostic or "")


def test_check_balance_gte_claimable_fail():
    inv = Invariant(
        id="X-10", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalclaimable",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalclaimable",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xc0ffee"
    ftns_addr = "0x" + "ff" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[
        (contract_addr, "0xtotalclaimable")
    ] = 2_000
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 1_500
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.FAIL


def test_check_balance_gte_claimable_skipped_on_rpc_fail():
    inv = Invariant(
        id="X-11", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalclaimable",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalclaimable",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xc0ffee"
    backend.raise_for.add((contract_addr, "0xftnsaddr"))
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.SKIPPED


# ── check_contract aggregation ───────────────────────


def test_check_contract_returns_list():
    backend = _MockBackend()
    # No backend data — all RPC reads return None; checker
    # marks each None-return as SKIPPED (can't verify).
    results = _checker(backend).check_contract(
        "royalty_distributor", contract_address="0xabc",
    )
    assert (
        len(results)
        == len(INVARIANT_REGISTRY["royalty_distributor"])
    )
    # All skipped because backend returns None
    for r in results:
        assert r.status == InvariantStatus.SKIPPED


def test_check_contract_unknown_returns_empty():
    backend = _MockBackend()
    assert _checker(backend).check_contract(
        "nonexistent", contract_address="0xabc",
    ) == []


# ── Public surface ───────────────────────────────────


def test_invariant_to_dict_serializable():
    inv = INVARIANT_REGISTRY["royalty_distributor"][0]
    d = inv.to_dict()
    assert d["id"] == inv.id
    assert d["contract_name"] == inv.contract_name
    assert d["kind"] == inv.kind.value
    assert d["severity"] == inv.severity.value
    # Callable fields not present
    assert "check_fn" not in d


def test_result_to_dict_serializable():
    inv = Invariant(
        id="X-12", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 42
    result = _checker(backend).check_one(inv, "0xabc")
    d = result.to_dict()
    assert d["status"] == "pass"
    assert d["invariant_id"] == "X-12"
    assert d["value"] == 42
