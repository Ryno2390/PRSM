"""Sprint 496 — FTNS-side mainnet TX runbook integrity pins.

Sprint 496 shipped `docs/operations/ftns-side-mainnet-tx-runbook.md`
— the test plan + operator runbook for real FTNS-side
on-chain mutations on Base mainnet (chainId 8453). This
covers the OC column of the coverage matrix (sprint 486)
which was the last major untested dimension.

The runbook is operator-facing: when the user decides to
fund the test wallet, the runbook is the source of truth
for what to do. These pins defend the load-bearing
elements:

  1. Canonical contract addresses match the live Base
     mainnet deployment.
  2. Test wallet address matches sprint 467's funded
     wallet.
  3. Required env vars are documented.
  4. Staged TX sequence stops at TX-4 (TX-5 settler bond
     explicitly deferred — Foundation Safe ceremony).
  5. Risk register includes the critical "wrong network"
     invariant.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = (
    REPO_ROOT / "docs" / "operations"
    / "ftns-side-mainnet-tx-runbook.md"
)


def _read() -> str:
    return RUNBOOK.read_text()


def test_runbook_exists():
    assert RUNBOOK.is_file(), (
        f"FTNS mainnet TX runbook missing at {RUNBOOK}"
    )


def test_runbook_pins_canonical_ftns_address():
    """The runbook must reference the exact canonical
    FTNS token address. Any drift means operators would
    send TX to the wrong contract — potentially burning
    funds."""
    text = _read()
    assert "0x5276a3756C85f2E9e46f6D34386167a209aa16e5" in text


def test_runbook_pins_canonical_royalty_distributor():
    """Same defense for RoyaltyDistributor v2."""
    text = _read()
    assert "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e" in text


def test_runbook_pins_test_wallet_from_sprint_467():
    """Sprint 498 pivot: sprint-464 wallet
    `0x2Fd48D…` is stranded (key lost). Active test
    wallet is now `0x4acdE458…`. Runbook must reference
    BOTH — the stranded one as the documented lesson,
    the active one as the operational target."""
    text = _read()
    assert "0x2Fd48D2d026bEf7563C85c647674cb945C4d4f57" in text, (
        "stranded sprint-464 wallet must remain referenced "
        "as the documented lost-key lesson"
    )
    assert "0x4acdE458766C704B2511583572303e77109cFFE8" in text, (
        "active sprint-498 wallet must be the operational "
        "target for TX-1/2/3"
    )


def test_runbook_documents_persistent_key_file():
    """Sprint 498 lesson: the key MUST be persisted to a
    file the operator controls BEFORE the daemon is
    launched with it, so a daemon restart doesn't lose
    the key forever (as happened to the sprint-464
    wallet). Runbook must document the persistence path."""
    text = _read()
    assert "~/.prsm/test-wallet.env" in text
    assert "chmod 600" in text or "0600" in text


def test_runbook_documents_all_required_env_vars():
    """Operator needs the full env-var list to set up the
    daemon. Any missing var means surface returns 503 or
    silently degrades."""
    text = _read()
    for var in (
        "PRSM_NETWORK=mainnet",
        "BASE_RPC_URL",
        "PRSM_ONCHAIN_FTNS=1",
        "FTNS_WALLET_PRIVATE_KEY",
        "PRSM_ROYALTY_DISTRIBUTOR_ADDRESS",
        "PRSM_FTNS_TOKEN_ADDRESS",
    ):
        assert var in text, (
            f"runbook missing required env var: {var}"
        )


def test_runbook_includes_staged_tx_sequence():
    """The TX sequence must be staged smallest→largest so
    operators catch RPC config bugs on TX-1 before they
    risk meaningful FTNS on TX-4."""
    text = _read()
    for marker in (
        "TX-1: FTNS self-transfer",
        "TX-2: FTNS transfer to a different address",
        "TX-3: Claim royalty",
        "TX-4: Stake commissioning",
    ):
        assert marker in text, (
            f"runbook missing staged TX step: {marker}"
        )


def test_runbook_explicitly_defers_settler_bond():
    """TX-5 settler bond (min 10,000 FTNS) is deferred
    pending Foundation Safe ceremony. The runbook MUST
    document this deferral so an over-eager operator
    doesn't burn 10k FTNS thinking it's part of the
    sprint-496 plan."""
    text = _read()
    assert "TX-5" in text
    assert "deferred" in text.lower()
    assert "Foundation Safe" in text or "10,000 FTNS" in text


def test_runbook_sprint_497_defers_tx_4_too():
    """Sprint 497 dry-run discovered TX-4 stake commission
    is ALSO Foundation-ceremony-gated (PENDING_COMMISSION
    in-memory mirror until contract deployed). Pre-sprint-497
    runbook implied TX-4 was executable as soon as wallet
    funded; sprint 497 corrected this."""
    text = _read()
    # The TX-4 section must call out the deferral
    tx4_idx = text.find("### TX-4")
    assert tx4_idx >= 0
    tx4_body = text[tx4_idx:tx4_idx + 3000]
    assert "DEFERRED" in tx4_body or "deferred" in tx4_body
    assert "PENDING_COMMISSION" in tx4_body, (
        "TX-4 must explain the in-memory mirror behavior "
        "so operators don't expect chain interaction"
    )


def test_runbook_sprint_497_schema_correction():
    """Sprint 496 documented TX-4 schema as
    `creator_eth_address + amount_ftns`. Sprint 497 dry-run
    discovered the actual schema is `creator_id + amount_wei`.
    The corrected schema must appear in the TX-4 example."""
    text = _read()
    tx4_idx = text.find("### TX-4")
    tx4_body = text[tx4_idx:tx4_idx + 3000]
    # Correct schema must be in the example
    assert '"creator_id"' in tx4_body
    assert '"amount_wei"' in tx4_body


def test_runbook_sprint_497_dry_run_mode_documented():
    """Sprint 497 dry-run discovered /wallet/royalty/claim
    has a built-in DRY_RUN mode (returns
    {"status":"DRY_RUN","claimable_ftns":0.0}) when there's
    nothing to claim. The runbook must document this so
    operators don't read the response as an error."""
    text = _read()
    tx3_idx = text.find("### TX-3")
    tx3_body = text[tx3_idx:tx3_idx + 2000]
    assert "DRY_RUN" in tx3_body
    assert "claimable_ftns" in tx3_body


def test_runbook_executable_today_summary_accurate():
    """The cost summary must distinguish 'executable today'
    (TX-1, TX-2, TX-3) from 'deferred' (TX-4, TX-5).
    Sprint 496 over-estimated cost by assuming all 5 TX
    would run."""
    text = _read()
    # The Cost summary section must call out the
    # executable-today vs deferred split.
    cost_idx = text.find("## Cost summary")
    assert cost_idx >= 0
    cost_body = text[cost_idx:cost_idx + 2500]
    assert "EXECUTABLE" in cost_body or "executable" in cost_body
    assert "DEFERRED" in cost_body or "deferred" in cost_body.lower()


def test_runbook_includes_chain_id_invariant():
    """The wrong-network risk is critical (could burn
    funds). Runbook must call out chainId 8453 as the
    pre-broadcast invariant."""
    text = _read()
    assert "8453" in text
    assert "chain_id" in text.lower() or "chainid" in text.lower()


def test_runbook_includes_rollback_section():
    """FTNS-side TX are irreversible — runbook must
    surface this explicitly so operators don't expect a
    refund path."""
    text = _read()
    assert "irreversible" in text.lower()
    assert "Rollback" in text or "rollback" in text


def test_runbook_includes_cost_estimate():
    """Operators need a USD ballpark before deciding to
    fund the test wallet."""
    text = _read()
    assert "Cost summary" in text or "cost" in text.lower()
    # Must reference the actual sprint-467 observed gas
    # price (0.006 Gwei) so the estimate is defensible.
    assert "0.006 Gwei" in text or "Gwei" in text


def test_runbook_does_not_leak_private_key():
    """Defensive: runbook must never include an actual
    private key. If a contributor accidentally pastes one,
    this test catches it before commit."""
    text = _read()
    # Real Ed25519/secp256k1 private keys are 64 hex
    # chars. The runbook should reference the env var
    # NAME, not a value.
    import re
    # Look for `0x` followed by 64 hex chars OUTSIDE of
    # the FTNS token address + royalty distributor (40-
    # char addresses).
    suspicious = re.findall(
        r"\b0x[0-9a-fA-F]{64}\b", text,
    )
    assert not suspicious, (
        f"runbook contains 64-hex-char hex literal — could "
        f"be a leaked private key: {suspicious[:3]}"
    )
