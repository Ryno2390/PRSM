"""Sprint 905 — formal invariants pin the deployed v1 monetary params.

The §10 tokenomics audit (2026-05-07) flagged two undetectable-drift
gaps: NO formal invariant pinned (a) the EmissionController emission
schedule constants (`baselineRatePerSecond`, `mintCap`) or (b) the
CompensationDistributor 50/30/20 reward split. A contract substitution
or a (90-day-gated) weight change would have gone unflagged by the
runtime invariant harness.

sp905 closes both. All expected values are pinned to the EXACT mainnet
deploy (`contracts/deployments/phase8-emission-base-1778164608198.json`,
chain 8453):
  - EmissionController.baselineRatePerSecond == 1e18  (1 FTNS/sec start)
  - EmissionController.mintCap                == 900M * 1e18
  - CompensationDistributor.currentWeights()  == (5000, 3000, 2000) bps

The split lives in a 3-word struct getter (`currentWeights()` returns
the `PoolWeights` tuple), so pinning each field needs word-offset
decoding — sp905 adds the `UINT256_AT_WORD_EQ` invariant kind for that.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.formal_invariants import (
    INVARIANT_REGISTRY,
    Invariant,
    InvariantChecker,
    InvariantKind,
    InvariantSeverity,
    InvariantStatus,
)

_EC = "emission_controller"
_CD = "compensation_distributor"

# Exact mainnet deploy values (phase8-emission-base-1778164608198.json).
_MINT_CAP = 900_000_000 * 10**18
_BASELINE_RATE = 10**18
_CREATOR_BPS = 5000
_OPERATOR_BPS = 3000
_GRANT_BPS = 2000

# Canonical selectors (first 4 bytes of keccak256 of the signature).
_SEL_MINT_CAP = "0x76c71ca1"           # mintCap()
_SEL_BASELINE = "0x968226be"           # baselineRatePerSecond()
_SEL_CURRENT_WEIGHTS = "0x322db68a"    # currentWeights()


def _inv(contract: str, inv_id: str) -> Invariant:
    for inv in INVARIANT_REGISTRY[contract]:
        if inv.id == inv_id:
            return inv
    raise AssertionError(f"{inv_id} not in {contract} registry")


# ── A word-offset-aware mock backend ─────────────────────


class _WordMockBackend:
    """Scripts (addr, selector[, word]) -> value; raises to sim RPC error."""

    def __init__(self):
        self.uint256: dict = {}
        self.words: dict = {}
        self.raise_for: set = set()

    def call_uint256(self, addr, selector):
        key = (addr.lower(), selector.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.uint256.get(key)

    def call_uint256_at_word(self, addr, selector, word_index):
        key = (addr.lower(), selector.lower(), word_index)
        if (addr.lower(), selector.lower()) in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.words.get(key)

    # Unused-by-these-tests protocol methods.
    def call_address(self, addr, selector):
        return None

    def call_bool(self, addr, selector):
        return None

    def token_balance_of(self, token, holder):
        return None

    def call_has_role(self, addr, role_hash, account):
        return None


# ── Registry presence + exact deployed values ────────────


def test_emission_mintcap_invariant_pins_900M():
    inv = _inv(_EC, "INV-EC-3")
    assert inv.kind == InvariantKind.UINT256_EQ
    assert inv.selector == _SEL_MINT_CAP
    assert inv.expected == _MINT_CAP
    assert inv.severity == InvariantSeverity.CRITICAL


def test_emission_baseline_rate_invariant_pins_1e18():
    inv = _inv(_EC, "INV-EC-4")
    assert inv.kind == InvariantKind.UINT256_EQ
    assert inv.selector == _SEL_BASELINE
    assert inv.expected == _BASELINE_RATE
    assert inv.severity == InvariantSeverity.CRITICAL


@pytest.mark.parametrize(
    "inv_id,word_index,expected_bps",
    [
        ("INV-CD-3", 0, _CREATOR_BPS),
        ("INV-CD-4", 1, _OPERATOR_BPS),
        ("INV-CD-5", 2, _GRANT_BPS),
    ],
)
def test_split_invariants_pin_50_30_20(inv_id, word_index, expected_bps):
    inv = _inv(_CD, inv_id)
    assert inv.kind == InvariantKind.UINT256_AT_WORD_EQ
    assert inv.selector == _SEL_CURRENT_WEIGHTS
    assert inv.params.get("word_index") == word_index
    assert inv.expected == expected_bps
    assert inv.severity == InvariantSeverity.CRITICAL


def test_split_invariants_sum_to_10000():
    total = sum(
        _inv(_CD, i).expected for i in ("INV-CD-3", "INV-CD-4", "INV-CD-5")
    )
    assert total == 10_000


@pytest.mark.parametrize(
    "signature,selector",
    [
        ("mintCap()", _SEL_MINT_CAP),
        ("baselineRatePerSecond()", _SEL_BASELINE),
        ("currentWeights()", _SEL_CURRENT_WEIGHTS),
    ],
)
def test_selectors_match_keccak(signature, selector):
    """A typo'd selector would silently SKIP against mainnet
    (eth_call to a wrong selector returns 0x -> None -> SKIPPED),
    giving false confidence. Pin the keccak derivation so the
    selector is provably the real ABI selector."""
    try:
        from web3 import Web3
        computed = Web3.keccak(text=signature)[:4].hex()
    except ImportError:  # pragma: no cover
        from eth_hash.auto import keccak
        computed = keccak(signature.encode())[:4].hex()
    assert selector.removeprefix("0x").lower() == computed.lower()


# ── New checker kind: UINT256_AT_WORD_EQ ──────────────────


def _word_inv(word_index: int, expected: int) -> Invariant:
    return Invariant(
        id="INV-TEST-WORD",
        contract_name=_CD,
        title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="currentWeights()[word] == expected",
        kind=InvariantKind.UINT256_AT_WORD_EQ,
        selector=_SEL_CURRENT_WEIGHTS,
        expected=expected,
        params={"word_index": word_index},
    )


def test_word_eq_pass():
    backend = _WordMockBackend()
    addr = "0x" + "ab" * 20
    backend.words[(addr.lower(), _SEL_CURRENT_WEIGHTS, 1)] = 3000
    res = InvariantChecker(backend=backend).check_one(
        _word_inv(1, 3000), addr,
    )
    assert res.status == InvariantStatus.PASS
    assert res.value == 3000


def test_word_eq_fail_surfaces_diagnostic():
    backend = _WordMockBackend()
    addr = "0x" + "ab" * 20
    backend.words[(addr.lower(), _SEL_CURRENT_WEIGHTS, 0)] = 4500
    res = InvariantChecker(backend=backend).check_one(
        _word_inv(0, 5000), addr,
    )
    assert res.status == InvariantStatus.FAIL
    assert "4500" in (res.diagnostic or "")
    assert "5000" in (res.diagnostic or "")


def test_word_eq_none_is_skipped():
    backend = _WordMockBackend()  # no value scripted -> None
    addr = "0x" + "ab" * 20
    res = InvariantChecker(backend=backend).check_one(
        _word_inv(2, 2000), addr,
    )
    assert res.status == InvariantStatus.SKIPPED


def test_word_eq_rpc_error_is_skipped():
    backend = _WordMockBackend()
    addr = "0x" + "ab" * 20
    backend.raise_for.add((addr.lower(), _SEL_CURRENT_WEIGHTS))
    res = InvariantChecker(backend=backend).check_one(
        _word_inv(0, 5000), addr,
    )
    assert res.status == InvariantStatus.SKIPPED
    assert "RPC error" in (res.error or "")


def test_full_split_checks_against_tuple_backend():
    """End-to-end: a backend returning the deployed (5000,3000,2000)
    tuple makes all three split invariants PASS."""
    backend = _WordMockBackend()
    addr = "0x" + "cd" * 20
    cd_addr = addr.lower()
    backend.words[(cd_addr, _SEL_CURRENT_WEIGHTS, 0)] = _CREATOR_BPS
    backend.words[(cd_addr, _SEL_CURRENT_WEIGHTS, 1)] = _OPERATOR_BPS
    backend.words[(cd_addr, _SEL_CURRENT_WEIGHTS, 2)] = _GRANT_BPS
    checker = InvariantChecker(backend=backend)
    for inv_id in ("INV-CD-3", "INV-CD-4", "INV-CD-5"):
        res = checker.check_one(_inv(_CD, inv_id), addr)
        assert res.status == InvariantStatus.PASS, inv_id
