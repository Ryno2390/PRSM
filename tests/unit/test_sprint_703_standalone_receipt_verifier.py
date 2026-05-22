"""Sprint 703 — pin tests for the standalone receipt verifier.

`scripts/verify_prsm_receipt.py` is intentionally PRSM-import-free
so a third party can use it without the PRSM repo. These pin tests
defend the canonical-signing-bytes implementation against drift
from the PRSM-side `InferenceReceipt.signing_payload` source.

If `prsm/compute/inference/models.py:signing_payload` ever changes
its field order, conditional encoding, or formatting, these tests
will fail — alerting the developer to update the standalone
verifier in the SAME commit so external auditors don't read stale
verification logic.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_verifier():
    """Load scripts/verify_prsm_receipt.py as a module without
    running its CLI."""
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


def test_verifier_signing_bytes_match_prsm_side_no_streaming():
    """Canonical signing bytes from the standalone verifier must
    match prsm.compute.inference.models.InferenceReceipt.signing_payload
    for a minimal tier-none receipt (no streamed_output, no
    activation_noise_trace, no topology_assignment)."""
    from prsm.compute.inference.models import InferenceReceipt
    from prsm.compute.inference.models import ContentTier
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    from decimal import Decimal

    receipt = InferenceReceipt(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE,
        tee_attestation=b"PRSM-MS-ATT-V1:{\"stages\":[],\"version\":1}",
        output_hash=b"\xab" * 32, duration_seconds=1.0,
        cost_ftns=Decimal("0.12"), settler_node_id="a" * 32,
    )
    prsm_bytes = receipt.signing_payload()

    verifier = _load_verifier()
    receipt_dict = {
        "job_id": "job-1", "request_id": "req-1", "model_id": "gpt2",
        "content_tier": "A", "privacy_tier": "none",
        "epsilon_spent": 0.0, "tee_type": "software",
        "tee_attestation": (
            b"PRSM-MS-ATT-V1:{\"stages\":[],\"version\":1}".hex()
        ),
        "output_hash": (b"\xab" * 32).hex(),
        "duration_seconds": 1.0, "cost_ftns": "0.12",
        "settler_node_id": "a" * 32,
    }
    verifier_bytes = verifier._build_signing_payload(receipt_dict)
    assert prsm_bytes == verifier_bytes, (
        "standalone verifier signing-payload diverged from PRSM-side "
        "InferenceReceipt.signing_payload — update "
        "scripts/verify_prsm_receipt.py _build_signing_payload to match"
    )


def test_verifier_signing_bytes_match_prsm_side_with_streamed_output():
    """Same as above but with streamed_output=True. Confirms the
    conditional-append for the streaming flag matches PRSM side."""
    from prsm.compute.inference.models import InferenceReceipt
    from prsm.compute.inference.models import ContentTier
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    from decimal import Decimal

    receipt = InferenceReceipt(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0, tee_type=TEEType.SOFTWARE,
        tee_attestation=b"PRSM-MS-ATT-V1:{\"stages\":[],\"version\":1}",
        output_hash=b"\xab" * 32, duration_seconds=1.0,
        cost_ftns=Decimal("0.12"), settler_node_id="a" * 32,
        streamed_output=True,
    )
    prsm_bytes = receipt.signing_payload()

    verifier = _load_verifier()
    verifier_bytes = verifier._build_signing_payload({
        "job_id": "job-1", "request_id": "req-1", "model_id": "gpt2",
        "content_tier": "A", "privacy_tier": "none",
        "epsilon_spent": 0.0, "tee_type": "software",
        "tee_attestation": (
            b"PRSM-MS-ATT-V1:{\"stages\":[],\"version\":1}".hex()
        ),
        "output_hash": (b"\xab" * 32).hex(),
        "duration_seconds": 1.0, "cost_ftns": "0.12",
        "settler_node_id": "a" * 32,
        "streamed_output": True,
    })
    assert prsm_bytes == verifier_bytes


def test_verifier_signing_bytes_match_with_activation_noise_trace():
    """tier-standard receipt with activation_noise_trace populated.
    Confirms the canonical hash for the trace field matches PRSM."""
    from prsm.compute.inference.models import InferenceReceipt
    from prsm.compute.inference.activation_dp import (
        ActivationNoiseTrace,
    )
    from prsm.compute.inference.models import ContentTier
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    from decimal import Decimal

    trace = ActivationNoiseTrace(
        per_stage_epsilon=[8.0], total_epsilon_spent=8.0,
        clip_norm=1.0, stage_count=1, tier="standard",
    )
    receipt = InferenceReceipt(
        job_id="job-1", request_id="req-1", model_id="gpt2",
        content_tier=ContentTier.A, privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0, tee_type=TEEType.SOFTWARE,
        tee_attestation=b"PRSM-MS-ATT-V1:{\"stages\":[],\"version\":1}",
        output_hash=b"\xab" * 32, duration_seconds=1.0,
        cost_ftns=Decimal("0.132"), settler_node_id="a" * 32,
        activation_noise_trace=trace,
    )
    prsm_bytes = receipt.signing_payload()

    verifier = _load_verifier()
    verifier_bytes = verifier._build_signing_payload({
        "job_id": "job-1", "request_id": "req-1", "model_id": "gpt2",
        "content_tier": "A", "privacy_tier": "standard",
        "epsilon_spent": 8.0, "tee_type": "software",
        "tee_attestation": (
            b"PRSM-MS-ATT-V1:{\"stages\":[],\"version\":1}".hex()
        ),
        "output_hash": (b"\xab" * 32).hex(),
        "duration_seconds": 1.0, "cost_ftns": "0.132",
        "settler_node_id": "a" * 32,
        "activation_noise_trace": trace.to_dict(),
    })
    assert prsm_bytes == verifier_bytes


def test_verifier_zero_prsm_imports():
    """The standalone script must NOT import from `prsm.*`. If
    someone accidentally adds a PRSM import to the verifier, this
    breaks the "no PRSM repo needed" claim."""
    path = (
        Path(__file__).parent.parent.parent
        / "scripts" / "verify_prsm_receipt.py"
    )
    src = path.read_text()
    # No lines like "from prsm." or "import prsm"
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("from prsm.") or stripped.startswith("import prsm"):
            pytest.fail(
                f"standalone verifier imports from prsm.*: {stripped!r} "
                f"— breaks no-PRSM-repo claim"
            )


def test_verifier_tamper_rejection_logic():
    """Tampering output_hash invalidates the signature (sanity)."""
    verifier = _load_verifier()
    receipt = {
        "job_id": "j", "request_id": "r", "model_id": "gpt2",
        "content_tier": "A", "privacy_tier": "none",
        "epsilon_spent": 0.0, "tee_type": "software",
        "tee_attestation": "aa", "output_hash": "bb",
        "duration_seconds": 1.0, "cost_ftns": "0.1",
        "settler_node_id": "x",
    }
    bytes_a = verifier._build_signing_payload(receipt)
    receipt["output_hash"] = "cc"
    bytes_b = verifier._build_signing_payload(receipt)
    assert bytes_a != bytes_b, (
        "tamper detection broken — receipt with changed "
        "output_hash produced identical signing bytes"
    )
