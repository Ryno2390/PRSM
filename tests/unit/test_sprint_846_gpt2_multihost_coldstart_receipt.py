"""Sprint 846 — gpt2 multi-host cold-start receipt sample.

Adds `docs/sample-receipts/gpt2-multihost-coldstart-2026-05-25.json`
to the verifier-demo collection (joining sp706/707/708's 4 samples).

Captured live 2026-05-25 from the Mac probe → production fleet via
the sp836-845 cold-start gossip arc. Distinct from
`multi-host-2stage-2026-05-22.json` (sp708):

  sp708 sample provenance:
    Hand-configured operator addresses; static GPU pool; pre-sp836
    pool plumbing; pre-sp838 hw_profile relay.

  sp846 sample provenance:
    Mac probe with NO prior knowledge of operators; joined via
    `wss://bootstrap-us.prsm-network.com:8765`; bootstrap relayed
    NYC + SFO hardware_profile (sp838); sp843 producer-side overrides
    propagated; sp682/sp836 DHT pool admitted both peers; allocator
    distributed gpt2's 12 layers as a 2-stage chain across NYC + SFO;
    signed receipt returned with per-stage attestations.

Pin tests:
- Sample file exists + parses as JSON
- Has required InferenceReceipt fields
- output_hash matches the sp688 known-good gpt2 greedy reference
  (`fe0663fd13cdb08743dca3d2d5d1168762556ea36f77d8ab6009306d12a6f351`)
  proving multi-host = single-host semantic equivalence
- stage_count == 2 (the multi-host claim)
- Both stages are anchor-registered operator node_ids (NYC + SFO)
- settler_node_id == Mac's persistent identity (cdefb8e5…)
- README documents the new sample in the Files table
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_PATH = (
    REPO_ROOT / "docs" / "sample-receipts"
    / "gpt2-multihost-coldstart-2026-05-25.json"
)
README_PATH = REPO_ROOT / "docs" / "sample-receipts" / "README.md"

# Known-good gpt2 greedy reference from sp688 — bit-identical
# under proper position-embedding handling.
GPT2_KNOWN_OUTPUT_HASH = (
    "fe0663fd13cdb08743dca3d2d5d1168762556ea36f77d8ab6009306d12a6f351"
)
MAC_NODE_ID = "cdefb8e5214c72731aeb3fbe6833fc6f"
NYC_NODE_ID = "484f003c895ee02ac7ed01e570a6a51f"
SFO_NODE_ID = "d437aa67d99cff4a6a17179f5c731b77"


def _load_sample():
    return json.loads(SAMPLE_PATH.read_text())


def test_sample_exists_and_parses():
    assert SAMPLE_PATH.exists()
    r = _load_sample()
    assert isinstance(r, dict)


def test_sample_has_required_receipt_fields():
    r = _load_sample()
    required = {
        "job_id", "request_id", "model_id",
        "content_tier", "privacy_tier",
        "tee_type", "tee_attestation",
        "output_hash", "duration_seconds", "cost_ftns",
        "settler_signature", "settler_node_id",
        "topology_assignment",
    }
    missing = required - set(r.keys())
    assert not missing, f"sample missing required fields: {missing}"


def test_sample_model_is_gpt2():
    assert _load_sample()["model_id"] == "gpt2"


def test_sample_output_hash_matches_sp688_reference():
    """The load-bearing claim of this sample — multi-host
    distributed gpt2 inference produces bit-identical output to
    sp688's single-host HuggingFace greedy reference. If this
    breaks, either gpt2 weights drifted (extremely unlikely; HF
    pins them), OR the multi-host code diverged semantically
    (the actual risk we're guarding against)."""
    r = _load_sample()
    assert r["output_hash"] == GPT2_KNOWN_OUTPUT_HASH, (
        f"sp846 sample output_hash {r['output_hash']!r} no longer "
        f"matches sp688 known-good gpt2 greedy reference "
        f"{GPT2_KNOWN_OUTPUT_HASH!r}. Either reproduce + capture a "
        "new sample if intentional, OR debug why distributed "
        "inference diverged from local."
    )


def test_sample_is_multihost_2stage():
    r = _load_sample()
    topo = r["topology_assignment"]
    assert topo["stage_count"] == 2
    positions = topo["positions"]
    assert len(positions) == 2
    # Each position: [stage_index, slot_index, node_id]
    stage_nodes = {pos[2] for pos in positions}
    assert stage_nodes == {NYC_NODE_ID, SFO_NODE_ID}, (
        f"expected stages on NYC + SFO; got {stage_nodes}"
    )


def test_sample_settler_is_mac_identity():
    r = _load_sample()
    assert r["settler_node_id"] == MAC_NODE_ID


def test_sample_settler_signature_is_present():
    r = _load_sample()
    sig = r["settler_signature"]
    assert isinstance(sig, str)
    # Ed25519 sig is 64 bytes = 128 hex chars
    assert len(sig) == 128


def test_sample_attestation_envelope_present():
    """sp708's invariant: multi-host receipts carry per-stage
    attestation_hex for ALL stages in the topology_assignment."""
    r = _load_sample()
    att = r.get("tee_attestation", "")
    assert isinstance(att, str)
    assert len(att) > 0
    # tee_attestation is hex-encoded JSON of the envelope
    envelope_bytes = bytes.fromhex(att)
    # Sp448's PRSM-MS-ATT-V1 prefix
    assert envelope_bytes.startswith(b"PRSM-MS-ATT-V1:")


def test_readme_documents_new_sample():
    """The sample must be referenced in the README so external
    auditors discover it via the canonical entry-point."""
    txt = README_PATH.read_text()
    assert "gpt2-multihost-coldstart-2026-05-25.json" in txt
    assert "cold-start gossip" in txt.lower() or (
        "coldstart" in txt
    )


def test_readme_documents_output_hash_match_claim():
    """The README should call out the bit-identical-to-sp688
    claim because it's the load-bearing assertion for the
    sample."""
    txt = README_PATH.read_text()
    assert "bit-identical" in txt
    assert "fe0663fd" in txt
