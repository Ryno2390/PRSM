"""Sprint 311 — cross-organization federated learning
bridge.

Hierarchical FL composition: each enterprise runs its own
`FederatedLearningOrchestrator` over its own worker pool +
encrypted shards. A "consortium" orchestrator at a third
location treats each enterprise as a "worker" whose
"gradient" is the enterprise-level aggregated_update.

The entire composition runs on the existing primitives —
the only new code is a bridge helper that takes an
enterprise's aggregated round and wraps it as a signed
GradientUpdate ready to submit to the consortium
orchestrator. The consortium then verifies the signature
(sprint 308a), optionally unseals encrypted transport
(sprint 308c), applies central DP (sprint 308a), and
aggregates via FedAvg or FedMedian (sprint 308).

The bridge thus closes the loop end-to-end without
introducing any new wire format or cryptographic
primitive — purely a structural composition.
"""
from __future__ import annotations

import base64
import time
from typing import Optional

from prsm.enterprise.federated_learning import (
    FederatedRound,
    GradientUpdate,
    RoundStatus,
    seal_gradient_for_orchestrator,
    sign_gradient_update,
)


def aggregated_round_to_gradient_update(
    *,
    local_round: FederatedRound,
    consortium_job_id: str,
    consortium_round_index: int,
    enterprise_node_id: str,
    enterprise_privkey_b64: str,
    sample_count: int,
    transport_pubkey_b64: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> GradientUpdate:
    """Wrap a local (enterprise-level) aggregated round as
    a signed GradientUpdate bound for a consortium
    orchestrator.

    Args:
      local_round: an aggregated FederatedRound from the
        enterprise's own orchestrator. The round MUST be
        in status AGGREGATED — bridging an unaggregated
        round publishes empty bytes, which is operator
        confusion.
      consortium_job_id / consortium_round_index: the
        consortium-level job and round that this submission
        is intended for.
      enterprise_node_id: identifier the consortium
        recognizes for this enterprise. Must match an
        entry in the consortium's worker registry (the
        consortium calls `register_worker_key` for each
        participating enterprise with the matching pubkey).
      enterprise_privkey_b64: Ed25519 privkey the
        enterprise uses to sign cross-org submissions.
        Different from individual worker privkeys —
        represents the enterprise's identity at the
        consortium level.
      sample_count: total samples used across all the
        enterprise's workers this round. Used by the
        consortium's FedAvg-style weighting.
      transport_pubkey_b64: if the consortium requires
        encrypted-gradient transport (sprint 308c), this
        is its transport pubkey. The aggregated update
        gets sealed to it before signing — the consortium
        unseals on aggregation.
      timestamp: defaults to time.time().

    Returns:
      A signed (and optionally sealed) GradientUpdate
      compatible with the consortium orchestrator's
      `accept_gradient_update` interface.
    """
    if local_round.status != RoundStatus.AGGREGATED:
        raise ValueError(
            f"local_round must be in status AGGREGATED to "
            f"bridge; got {local_round.status.value!r}. "
            f"Call orchestrator.aggregate_round() before "
            f"bridging."
        )
    if not local_round.aggregated_update:
        raise ValueError(
            "local_round.aggregated_update is empty — "
            "nothing to bridge"
        )

    aggregated_bytes = local_round.aggregated_update

    if transport_pubkey_b64 is not None:
        sealed_b64, envelope_b64 = (
            seal_gradient_for_orchestrator(
                aggregated_bytes, transport_pubkey_b64,
            )
        )
        gradient_b64 = sealed_b64
    else:
        gradient_b64 = base64.b64encode(
            aggregated_bytes,
        ).decode("ascii")
        envelope_b64 = None

    update = GradientUpdate(
        job_id=consortium_job_id,
        round_index=int(consortium_round_index),
        worker_node_id=enterprise_node_id,
        gradient_b64=gradient_b64,
        sample_count=int(sample_count),
        worker_attestation_b64="",
        worker_signature_b64="",
        timestamp=(
            timestamp if timestamp is not None
            else time.time()
        ),
        gradient_envelope_b64=envelope_b64,
    )
    return sign_gradient_update(
        update,
        worker_privkey_b64=enterprise_privkey_b64,
    )
