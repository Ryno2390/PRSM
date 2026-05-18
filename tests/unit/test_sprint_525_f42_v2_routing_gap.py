"""Sprint 525 — F42 surfaced: V1 client used against V2 contract.

Sprint 525 attempted to prefer V2 ProvenanceRegistry in the
canonical fallback path (networks.py has both V1 and V2 wired
for mainnet; V2 is canonical post the A-08 RoyaltyDistributor
ceremony).

Live-verify on Base mainnet surfaced **F42**: the auto-register
code path (content_uploader._register_on_chain) uses V1
ProvenanceRegistryClient which has a 3-arg `registerContent`
signature. V2 contract `0xe0cedDA354…` has a 5-arg signature
(adds embedding_commitment + fingerprint_kind). When V1 client
calls 3-arg method on V2 contract:
  - No matching ABI element exists
  - EVM may fall through to a fallback function (or similar)
  - eth_estimateGas returns the block gas limit (~400M)
  - send_raw_transaction rejects with "insufficient funds for
    gas * price + value: have 494839935833377 want
    2400000000000000"

Fix (sprint 526 candidate): proper V2 client routing. Either:
  A) Have content_uploader._register_on_chain detect V2 contract
     and dispatch to ProvenanceRegistryV2Client.register_content_v2
     with the extra args (ZERO_BYTES32 defaults match the existing
     hash-only contract).
  B) Add a polymorphic ProvenanceRegistryClient.register() that
     resolves the right method by contract version detection.

For this sprint, REVERTED the prefer-V2 fallback to keep V1 path
stable. Operators wanting V2 must set
PRSM_PROVENANCE_REGISTRY_ADDRESS=0xe0cedDA354… explicitly AND
upgrade the client (sprint 526).

ResolvedEndpoints schema gained `provenance_registry_v2` field
+ env-var override `PRSM_PROVENANCE_REGISTRY_V2_ADDRESS` so
sprint 526 can wire it cleanly.
"""
from __future__ import annotations


def test_resolved_endpoints_exposes_provenance_registry_v2():
    """networks.py ResolvedEndpoints schema must include
    provenance_registry_v2 so sprint 526 can read it."""
    from prsm.config.networks import resolve_endpoints
    ep = resolve_endpoints("mainnet")
    assert hasattr(ep, "provenance_registry_v2")
    # Mainnet has the V2 address pinned
    assert ep.provenance_registry_v2 == (
        "0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf"
    )


def test_canonical_fallback_uses_v1_until_v2_client_wired():
    """Sprint-525 partial fix: the canonical fallback path in
    _build_provenance_client_or_none currently uses V1 ONLY,
    because the V1 client can't talk to V2 (F42).

    This pin pegs the current state. Sprint 526 will replace
    with a proper V2-aware routing test.
    """
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[2]
        / "prsm/node/node.py"
    ).read_text()
    assert "F42" in src
    assert "Sprint 526 candidate" in src
    # V1-only fallback locked in until F42 fixed
    assert (
        "ep.provenance_registry or" in src
        or "ep.provenance_registry\n" in src
    )


def test_f42_documented():
    """Defensive: surface text must capture F42 awareness so
    future ad-hoc readers find the gap quickly."""
    from pathlib import Path
    here = Path(__file__).read_text()
    assert "F42" in here
    assert "3-arg" in here
    assert "5-arg" in here
    assert "0xe0cedDA354" in here
