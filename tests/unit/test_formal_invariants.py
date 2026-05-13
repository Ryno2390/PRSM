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


# ── Sprint 356 — FTNSToken + EscrowPool extension ────
#
# Background: while researching the §14 item 4 extension we
# discovered the existing `_SEL_FTNS` selector was wrong
# (keccak256("ftns()") first 4 bytes = 0xefa21b41, not the
# 0x9b03f021 that ship sprint 302 had committed). Result: on
# real mainnet RPC, INV-RD-4 — explicitly called "THE money
# invariant" in the module docstring — would SKIP rather
# than catch solvency drift. The mocked-backend tests above
# all pass with any selector, so this stayed invisible until
# we tried to extend the harness to additional contracts.
#
# This block adds: the selector correctness pin (regression
# test on the discovered bug), the new UINT256_LTE kind for
# supply-cap-style invariants, FTNSToken registry entries
# (supply ceiling), and EscrowPool registry entries
# (solvency mirror of INV-RD-4 against totalEscrowedBalance).


def test_ftns_selector_pinned_to_canonical_keccak():
    """Regression pin: the `ftns()` getter selector MUST be
    the keccak256("ftns()") first-4-bytes value 0xefa21b41.
    The original sprint 302 commit had 0x9b03f021 which is
    NOT the correct selector and would silently SKIP INV-RD-4
    on real RPC. Catching this is exactly what this harness
    was built to do — but the harness itself had a typo.
    """
    from prsm.economy.web3 import formal_invariants as fi
    assert fi._SEL_FTNS == "0xefa21b41", (
        f"_SEL_FTNS was {fi._SEL_FTNS}; canonical keccak256("
        f"'ftns()')[:4] is 0xefa21b41"
    )


def test_uint256_lte_kind_value():
    assert InvariantKind.UINT256_LTE.value == "uint256_lte"


def test_check_uint256_lte_pass():
    inv = Invariant(
        id="X-LTE-1", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead",
        expected=1_000_000_000 * 10**18,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = (
        100_000_000 * 10**18
    )
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS
    assert result.value == 100_000_000 * 10**18


def test_check_uint256_lte_pass_at_boundary():
    inv = Invariant(
        id="X-LTE-2", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead", expected=1000,
    )
    backend = _MockBackend()
    # Exactly at boundary — LTE means inclusive of equal
    backend.uint256[("0xabc", "0xdead")] = 1000
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS


def test_check_uint256_lte_fail_supply_breach():
    """If totalSupply ever exceeds MAX_SUPPLY, the contract
    has been compromised (MINTER_ROLE was supposed to enforce
    this on every mint). Failure here = monetary base attack.
    """
    inv = Invariant(
        id="X-LTE-3", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead",
        expected=1_000_000_000 * 10**18,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = (
        1_000_000_001 * 10**18
    )
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL
    assert "1000000001" in (result.diagnostic or "")


def test_check_uint256_lte_skipped_on_rpc_error():
    inv = Invariant(
        id="X-LTE-4", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead", expected=1000,
    )
    backend = _MockBackend()
    backend.raise_for.add(("0xabc", "0xdead"))
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.SKIPPED


def test_check_uint256_lte_skipped_on_none():
    inv = Invariant(
        id="X-LTE-5", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="x", kind=InvariantKind.UINT256_LTE,
        selector="0xdead", expected=1000,
    )
    backend = _MockBackend()  # no value set → returns None
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.SKIPPED


# ── FTNSToken registry ───────────────────────────────


def test_registry_has_ftns_token():
    assert "ftns_token" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["ftns_token"]
    assert len(invs) >= 2


def test_ftns_max_supply_invariant_pinned_to_1B():
    invs = INVARIANT_REGISTRY["ftns_token"]
    max_inv = next(
        (i for i in invs if i.id == "INV-FT-1"), None,
    )
    assert max_inv is not None
    assert max_inv.severity == InvariantSeverity.CRITICAL
    assert max_inv.kind == InvariantKind.UINT256_EQ
    # 1B FTNS in wei
    assert max_inv.expected == 1_000_000_000 * 10**18


def test_ftns_total_supply_lte_max_supply_invariant():
    invs = INVARIANT_REGISTRY["ftns_token"]
    sup = next(
        (i for i in invs if i.id == "INV-FT-2"), None,
    )
    assert sup is not None
    assert sup.severity == InvariantSeverity.CRITICAL
    assert sup.kind == InvariantKind.UINT256_LTE
    assert sup.expected == 1_000_000_000 * 10**18


# ── EscrowPool registry ──────────────────────────────


def test_registry_has_escrow_pool():
    assert "escrow_pool" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["escrow_pool"]
    assert len(invs) >= 1


def test_escrow_pool_solvency_invariant_critical():
    """Mirror of INV-RD-4 against totalEscrowedBalance.
    If ftns.balanceOf(EscrowPool) drops below the sum of
    requester escrow credits, some requester withdraw or
    batch-settlement will revert at the ERC-20 transfer
    boundary — operational impact is the same shape as
    RoyaltyDistributor insolvency.
    """
    invs = INVARIANT_REGISTRY["escrow_pool"]
    sol = next(
        (i for i in invs
         if i.kind == InvariantKind.BALANCE_GTE_CLAIMABLE),
        None,
    )
    assert sol is not None
    assert sol.severity == InvariantSeverity.CRITICAL
    # Reserve-label override should surface in the spec_text
    assert "totalEscrowedBalance" in sol.spec_text


def test_escrow_pool_solvency_diagnostic_uses_reserve_label():
    """When `reserve_label` param is set, the
    balance-gte-claimable handler MUST surface that label in
    the diagnostic. Without it, operators reading EscrowPool
    output would see 'totalClaimable=N' which is the wrong
    contract's variable name."""
    inv = Invariant(
        id="X-EP-DIAG", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalEscrowedBalance",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalescrowed",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalescrowed",
            "reserve_label": "totalEscrowedBalance",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xpool"
    ftns_addr = "0x" + "aa" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[
        (contract_addr, "0xtotalescrowed")
    ] = 5_000
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 5_500
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.PASS
    assert (
        "totalEscrowedBalance=5000"
        in (result.diagnostic or "")
    )
    # Old label MUST NOT appear when override is set
    assert "totalClaimable" not in (result.diagnostic or "")


def test_balance_gte_claimable_default_label_preserved():
    """Backward-compat — when `reserve_label` is NOT in
    params, the diagnostic still says 'totalClaimable' so
    INV-RD-4's output shape stays identical to sprint 302."""
    inv = Invariant(
        id="X-RD-DIAG", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtc",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtc",
            # NO reserve_label override
        },
    )
    backend = _MockBackend()
    contract_addr = "0xrd"
    ftns_addr = "0x" + "bb" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[(contract_addr, "0xtc")] = 100
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 200
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.PASS
    assert "totalClaimable=100" in (result.diagnostic or "")
