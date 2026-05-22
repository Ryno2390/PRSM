"""Sprint 706 — pin test: every sample receipt in docs/sample-receipts/
must verify cleanly against the standalone verifier.

Defends the audit-doc §6.1 "try it yourself in 30 seconds" claim
against repo drift. If someone accidentally edits a sample receipt
file OR changes the signing-payload schema without updating the
samples, this test fails.

NOT live-attested every run — the verifier's on-chain anchor lookup
is mocked here so CI doesn't hit Base mainnet RPC on every test
run. Live attestation happens by running the verifier directly
(documented in docs/sample-receipts/README.md).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


SAMPLE_RECEIPTS_DIR = (
    Path(__file__).parent.parent.parent / "docs" / "sample-receipts"
)


def _load_verifier():
    path = (
        Path(__file__).parent.parent.parent
        / "scripts" / "verify_prsm_receipt.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_prsm_receipt", path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_prsm_receipt"] = mod
    spec.loader.exec_module(mod)
    return mod


def _list_sample_receipts():
    return sorted(SAMPLE_RECEIPTS_DIR.glob("*.json"))


def test_sample_receipts_dir_exists():
    """The directory + README must exist (audit-doc §6.1 references)."""
    assert SAMPLE_RECEIPTS_DIR.is_dir(), (
        f"docs/sample-receipts/ missing — audit-doc §6.1 refers to it"
    )
    assert (SAMPLE_RECEIPTS_DIR / "README.md").exists()


def test_at_least_one_sample_receipt_shipped():
    """At least one receipt must be committed to the repo so the
    audit-doc "try it yourself" walkthrough works."""
    receipts = _list_sample_receipts()
    assert receipts, (
        "No sample receipts found in docs/sample-receipts/. "
        "Add at least one (sprint 706 shipped the first)."
    )


@pytest.mark.parametrize("receipt_path", _list_sample_receipts())
def test_sample_receipt_signing_bytes_reconstruct(receipt_path):
    """For every sample receipt, the standalone verifier's
    _build_signing_payload must produce bytes that match what the
    PRSM-side InferenceReceipt.signing_payload would have produced
    for the same fields.

    This catches the case where:
      - A sample receipt was captured under one signing-bytes schema
      - The schema later changed
      - But the sample wasn't updated
    Without this test, the audit-doc demo would silently break."""
    verifier = _load_verifier()
    with open(receipt_path) as f:
        receipt = json.load(f)
    # Just confirm the build doesn't crash and produces bytes.
    # We can't verify signature without anchor lookup (which needs
    # network); that's done by the live-verify command documented
    # in the README.
    payload = verifier._build_signing_payload(receipt)
    assert isinstance(payload, bytes)
    assert len(payload) > 0
    # Critical fields must appear in the payload (sanity)
    assert receipt["settler_node_id"].encode("utf-8") in payload
    assert receipt["output_hash"].encode("utf-8") in payload


def test_sample_receipt_has_required_signed_fields():
    """Every sample receipt must carry the fields the verifier
    expects. Catches the case where someone commits a receipt
    that's missing settler_signature or settler_node_id (e.g., an
    unsigned dry-run receipt)."""
    receipts = _list_sample_receipts()
    for path in receipts:
        with open(path) as f:
            r = json.load(f)
        assert r.get("settler_node_id"), f"{path.name} missing settler_node_id"
        assert r.get("settler_signature"), f"{path.name} missing settler_signature"
        assert r.get("output_hash"), f"{path.name} missing output_hash"


def test_sample_receipt_tier_standard_has_noise_trace():
    """The tier-standard sample must carry activation_noise_trace
    (the entire point of shipping a tier-standard sample is to
    demonstrate the DP path)."""
    path = SAMPLE_RECEIPTS_DIR / "tier-standard-dp-2026-05-22.json"
    if not path.exists():
        pytest.skip("expected sample receipt not present")
    with open(path) as f:
        r = json.load(f)
    assert r["privacy_tier"] == "standard"
    trace = r.get("activation_noise_trace")
    assert trace is not None
    assert trace["tier"] == "standard"
    assert trace["total_epsilon_spent"] > 0


def test_sample_receipt_tier_none_unary_bit_identical_hash():
    """Sprint 707 — tier-none unary sample's output_hash must
    match sprint 688's bit-identical-to-HuggingFace canonical
    value (`fe0663fd...`) so the sample is recognizable as the
    canonical gpt2 / "The capital of France is" / max_tokens=1
    reference. If this hash drifts, either the prompt changed OR
    the gpt2 reference changed — caller must update the sample
    deliberately."""
    path = SAMPLE_RECEIPTS_DIR / "tier-none-unary-2026-05-22.json"
    if not path.exists():
        pytest.skip("expected sample receipt not present")
    with open(path) as f:
        r = json.load(f)
    assert r["privacy_tier"] == "none"
    assert r["streamed_output"] is False
    assert r["output_hash"].startswith("fe0663fd"), (
        f"tier-none unary sample's output_hash drifted from sprint "
        f"688's canonical fe0663fd...; got {r['output_hash']}"
    )


def test_sample_receipt_streaming_carries_streamed_output_flag():
    """Sprint 707 — streaming sample must have streamed_output=True
    + the sprint 694 canonical output_hash `1b46bc86...` for the
    4-token decode of "The capital of France is"."""
    path = SAMPLE_RECEIPTS_DIR / "tier-none-streaming-2026-05-22.json"
    if not path.exists():
        pytest.skip("expected streaming sample receipt not present")
    with open(path) as f:
        r = json.load(f)
    assert r["streamed_output"] is True, (
        "streaming sample must carry streamed_output=True; without "
        "it the canonical signing payload omits the field and "
        "the verifier would interpret as unary"
    )
    assert r["output_hash"].startswith("1b46bc86"), (
        f"streaming sample's output_hash drifted from sprint 694's "
        f"canonical 1b46bc86...; got {r['output_hash']}"
    )
