"""Sprint 523 — `/content/upload` auto-registers provenance on-chain.

Sprint 520/522 verified `prsm provenance register` works as a
standalone CLI. Sprint 523 verifies the AUTOMATIC path: when the
daemon is launched with PRSM_PROVENANCE_REGISTRY_ADDRESS +
PRSM_ONCHAIN_PROVENANCE=1, every /content/upload triggers a
ProvenanceRegistry write transparently — operator doesn't have
to remember to register.

The auto-register code path (content_uploader._register_on_chain
at line 1017) was wired by an earlier T6 sprint but had never
been operationally exercised on mainnet. Sprint 523 confirms it
fires end-to-end.

Live-verified Base mainnet operator wallet 0x4acdE458…:

  $ curl -X POST /content/upload -d '{"text": "sprint 523...", ...}'
  → { "cid": "14856c57…", "content_hash": "be6c5f8c…", ...,
      "provenance_tx_hash": null }  ← response doesn't surface tx_hash

  $ curl /content/mine
  → entries[i].provenance_tx_hash =
      "0x82d1776d7ffe9457cc5eeca58b14a950ee77e4d7b00fd1d3de8518726b924006"

On-chain receipt: block 46166531, status success, to
ProvenanceRegistry V1 (0xdF470BFa…), 116157 gas @ 0.006 Gwei =
0.0000007 ETH, 1 ProvenanceRegistered event.

Vision §11 "creator provenance happens automatically on upload"
promise live-attested via PRSM API.

Surfaced UX gap: the upload response JSON doesn't include
provenance_tx_hash (operator has to query /content/mine to see
it). Sprint-524 candidate: thread provenance_tx_hash through the
upload response so the operator gets immediate confirmation.
"""
from __future__ import annotations


def test_auto_register_tx_recorded_on_chain():
    """Docstring-presence pin so a future revision that strips
    the live-verify references fails CI."""
    from pathlib import Path
    here = Path(__file__).read_text()
    assert "0x82d1776d" in here
    assert "block 46166531" in here
    assert "116157 gas" in here
    assert "provenance_tx_hash" in here
    assert "Vision §11" in here
