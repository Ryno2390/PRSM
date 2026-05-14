"""Sprint 415 — End-to-end §7 topology-receipt pathway.

Demonstrates the complete chain works:

  RpcChainExecutor (mocked inner)
    → TopologyAwareChainExecutor       (sprint 414)
    → ParallaxScheduledExecutor._build_signed_receipt
                                       (sprint 413 wiring)
    → sign_receipt (sprint 297 conditional encoding)
    → verify_receipt_privacy_claim(require_topology_rotation=True)
                                       (sprint 292 verifier)

Pre-sprint-413, the verify path with
``require_topology_rotation=True`` would FAIL on every
live receipt because the field was always None. After
sprint 414 wires TopologyAwareChainExecutor into the
dispatch path, the receipt carries a verifiable topology
hash, and the verifier passes.

This is the load-bearing end-to-end claim for the §7
topology pathway: signed-receipt verification with
mandatory topology now works against real receipts.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest


def _request():
    from prsm.compute.inference.models import (
        InferenceRequest, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel
    return InferenceRequest(
        prompt="end-to-end",
        model_id="model-e2e",
        budget_ftns=Decimal("5.0"),
        privacy_tier=PrivacyLevel.STANDARD,
        content_tier=ContentTier.A,
    )


def _outcome_no_topology(*, output="e2e-output"):
    """What a raw RpcChainExecutor returns — no topology
    yet."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    return ChainExecutionResult(
        output=output,
        duration_seconds=2.5,
        tee_attestation=b"\x42" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.0,
    )


def _chain(stages):
    chain = MagicMock()
    chain.stages = list(stages)
    chain.layer_ranges = [(i, i + 1) for i in range(len(stages))]
    return chain


def _build_parallax_with_topology_aware(node_identity):
    """Construct ParallaxScheduledExecutor with a Topology-
    AwareChainExecutor wrapping a mock inner. Heavy
    constructor bypassed via __new__ since this test
    targets _build_signed_receipt directly."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    executor = ParallaxScheduledExecutor.__new__(
        ParallaxScheduledExecutor,
    )
    executor._identity = node_identity
    return executor


def test_end_to_end_topology_pathway_with_signed_receipt():
    """The headline end-to-end test for sprint 415.

    Build full chain → sign → verify with strict topology
    requirement → assert pass.
    """
    from prsm.node.identity import generate_node_identity
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    from prsm.compute.inference.privacy_verification import (
        verify_receipt_privacy_claim,
    )

    # 1. Real identity for signing + verification
    identity = generate_node_identity("test-settler")

    # 2. Mock inner executor that returns a topology-less
    #    outcome (simulating a raw RpcChainExecutor)
    chain = _chain(["node-stage-0", "node-stage-1", "node-stage-2"])
    inner = MagicMock()
    inner.execute_chain = MagicMock(
        return_value=_outcome_no_topology(),
    )

    # 3. Wrap with sprint-414 TopologyAwareChainExecutor
    decorated = TopologyAwareChainExecutor(inner=inner)

    # 4. Execute via the decorator — produces an outcome
    #    with topology_assignment populated
    outcome = decorated.execute_chain(
        request=_request(), chain=chain,
    )
    assert outcome.topology_assignment is not None
    assert outcome.topology_assignment.stage_count == 3

    # 5. Build the signed receipt via parallax executor's
    #    sprint-413 wiring
    parallax = _build_parallax_with_topology_aware(identity)
    receipt = parallax._build_signed_receipt(
        request=_request(),
        cost=Decimal("1.0"),
        outcome=outcome,
        streamed=False,
    )

    # 6. Receipt carries the topology
    assert receipt.topology_assignment is not None
    assert (
        receipt.topology_assignment.stable_hash()
        == outcome.topology_assignment.stable_hash()
    )

    # 7. STRICT verification: require_topology_rotation=True
    #    against the signed receipt
    result = verify_receipt_privacy_claim(
        receipt,
        require_topology_rotation=True,
        identity=identity,
    )
    assert result.ok is True, f"verify failed: {result.reasons}"
    assert result.signature_valid is True
    assert result.topology_structurally_valid is True


def test_pre_sprint_414_receipt_fails_strict_topology_verify():
    """Regression pin: a receipt WITHOUT
    topology_assignment populated (the pre-sprint-414
    behavior) MUST fail verify_receipt_privacy_claim when
    require_topology_rotation=True. This is the literal
    bug sprint 414 fixed.
    """
    from prsm.node.identity import generate_node_identity
    from prsm.compute.inference.privacy_verification import (
        verify_receipt_privacy_claim,
    )

    identity = generate_node_identity("test-settler")

    # Skip the TopologyAwareChainExecutor — outcome has no
    # topology
    raw_outcome = _outcome_no_topology()
    assert raw_outcome.topology_assignment is None

    parallax = _build_parallax_with_topology_aware(identity)
    receipt = parallax._build_signed_receipt(
        request=_request(),
        cost=Decimal("1.0"),
        outcome=raw_outcome,
        streamed=False,
    )
    assert receipt.topology_assignment is None

    result = verify_receipt_privacy_claim(
        receipt,
        require_topology_rotation=True,
        identity=identity,
    )
    assert result.ok is False
    assert any(
        "topology_assignment missing" in r
        for r in result.reasons
    )


def test_topology_history_rejection_on_repeat():
    """When operator supplies a topology_history, the
    verifier rejects receipts whose topology matches an
    entry in the history (rotation-violation detection).
    Pinned end-to-end across the sprint-414 pathway."""
    from prsm.node.identity import generate_node_identity
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    from prsm.compute.inference.topology_rotation import (
        TopologyHistory,
    )
    from prsm.compute.inference.privacy_verification import (
        verify_receipt_privacy_claim,
    )

    identity = generate_node_identity("test-settler")

    # Run the same chain twice — the topology will be
    # identical (it's a deterministic function of stages)
    chain = _chain(["a", "b"])
    inner = MagicMock()
    inner.execute_chain = MagicMock(
        return_value=_outcome_no_topology(),
    )
    decorated = TopologyAwareChainExecutor(inner=inner)

    outcome = decorated.execute_chain(
        request=_request(), chain=chain,
    )

    parallax = _build_parallax_with_topology_aware(identity)
    receipt = parallax._build_signed_receipt(
        request=_request(),
        cost=Decimal("1.0"),
        outcome=outcome,
        streamed=False,
    )

    # History pre-loaded with the same topology (simulating
    # a recent prior dispatch)
    history = TopologyHistory(max_entries=10)
    history.record(receipt.topology_assignment)

    result = verify_receipt_privacy_claim(
        receipt,
        require_topology_rotation=True,
        identity=identity,
        topology_history=history,
    )
    # Should fail: topology repeats history entry
    assert result.ok is False
    assert any(
        "repeats an entry in supplied history" in r
        for r in result.reasons
    )


def test_default_posture_passes_without_topology_requirement():
    """Default (permissive) posture: a receipt WITHOUT
    topology + WITHOUT require_topology_rotation=True
    still verifies. Sprint 415 doesn't change the
    backwards-compat default — operators opt-in to the
    strict topology check."""
    from prsm.node.identity import generate_node_identity
    from prsm.compute.inference.privacy_verification import (
        verify_receipt_privacy_claim,
    )

    identity = generate_node_identity("test-settler")

    raw_outcome = _outcome_no_topology()
    parallax = _build_parallax_with_topology_aware(identity)
    receipt = parallax._build_signed_receipt(
        request=_request(),
        cost=Decimal("1.0"),
        outcome=raw_outcome,
        streamed=False,
    )

    # Permissive verify (no require_* flags)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    assert result.ok is True
    assert result.signature_valid is True
