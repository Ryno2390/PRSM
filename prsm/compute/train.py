"""Sprint 308b — worker-side federated-training shim.

The orchestrator (sprint 308) dispatches a round; each
assigned worker calls /compute/train on its own node.
This module is the primitive that runs INSIDE that
endpoint: invoke a training strategy, produce a gradient
vector, wrap it in a GradientUpdate signed by the worker's
Ed25519 privkey (sprint 308a), and bind the worker's TEE
attestation blob into the signed payload (sprint 308b).

The result goes back to the caller (orchestrator,
enterprise, or whoever ran the round), who is responsible
for POSTing it to /admin/federated/job/{id}/update. This
factoring keeps the worker stateless w.r.t. orchestrator
URL — distribution is the orchestrator's concern, not the
worker's.

Training strategies are pluggable. v1 ships a
deterministic stub that produces a fixed-shape gradient
seeded by (job_id, round_index, dataset_cid). Real
training (PyTorch / JAX / pluggable backends) wires here
in a follow-on; the surface contract stays stable.
"""
from __future__ import annotations

import hashlib
import time
import struct
from enum import Enum
from typing import Callable, List, Optional, Protocol

from prsm.enterprise.federated_learning import (
    GradientUpdate,
    encode_gradient,
    seal_gradient_for_orchestrator,
    sign_gradient_update,
)

import base64 as _b64


_STUB_GRADIENT_DIM = 8


class TrainingStrategy(str, Enum):
    """Pluggable training strategy identifier. v1 ships
    'stub' only; real backends register additional values
    in a follow-on."""

    STUB = "stub"


class TrainingFn(Protocol):
    """Signature of a training function: takes the round's
    inputs, returns a flat float gradient vector. The
    actual training logic (PyTorch model + optimizer +
    epoch loop, etc.) lives inside concrete
    implementations."""

    def __call__(
        self,
        *,
        job_id: str,
        round_index: int,
        dataset_cid: str,
        sample_count: int,
    ) -> List[float]: ...


def deterministic_stub_train_fn(
    *,
    job_id: str,
    round_index: int,
    dataset_cid: str,
    sample_count: int,
) -> List[float]:
    """Sprint 308b default — a deterministic gradient
    function seeded by (job_id, round_index, dataset_cid).
    Useful for end-to-end testing the orchestration layer
    BEFORE a real training backend is wired.

    The output is reproducible given the same inputs and
    varies cleanly with any input change."""
    seed_material = (
        f"{job_id}|{round_index}|{dataset_cid}"
        .encode("utf-8")
    )
    digest = hashlib.sha256(seed_material).digest()
    out: List[float] = []
    # Produce STUB_GRADIENT_DIM floats from the digest by
    # unpacking 4 bytes at a time as little-endian uint32
    # and mapping to [-1, 1].
    for i in range(_STUB_GRADIENT_DIM):
        # digest is 32 bytes; we need 4 bytes per float.
        # Cycle through with wrap to allow dim > 8.
        offset = (i * 4) % 28  # keep last 4-byte window safe
        u32 = struct.unpack(
            "<I", digest[offset:offset + 4],
        )[0]
        # Map uint32 → [-1, 1]
        out.append((u32 / 0xFFFFFFFF) * 2.0 - 1.0)
    return out


def compute_signed_gradient_update(
    *,
    job_id: str,
    round_index: int,
    dataset_cid: str,
    sample_count: int,
    worker_node_id: str,
    worker_privkey_b64: str,
    worker_attestation_b64: str,
    train_fn: Optional[TrainingFn] = None,
    timestamp: Optional[float] = None,
    transport_pubkey_b64: Optional[str] = None,
) -> GradientUpdate:
    """Run the training strategy, wrap the gradient in a
    GradientUpdate, sign it under the worker's Ed25519
    privkey, and return it.

    Sprint 308b: the signed payload binds the
    worker_attestation_b64 so a tampered attestation breaks
    verification.

    Sprint 308c: when transport_pubkey_b64 is provided, the
    gradient bytes are sealed (X25519 ECDH +
    ChaCha20-Poly1305) to that pubkey BEFORE signing. The
    signed payload also binds gradient_envelope_b64 so a
    MITM can't strip or replace the envelope without
    breaking the signature. When transport_pubkey_b64 is
    None, the gradient is plaintext (sprint 308b
    backwards-compat)."""
    fn = train_fn or deterministic_stub_train_fn
    grad = fn(
        job_id=job_id,
        round_index=round_index,
        dataset_cid=dataset_cid,
        sample_count=sample_count,
    )
    plaintext = encode_gradient(grad)

    if transport_pubkey_b64 is not None:
        sealed_b64, envelope_b64 = (
            seal_gradient_for_orchestrator(
                plaintext, transport_pubkey_b64,
            )
        )
        gradient_b64 = sealed_b64
    else:
        gradient_b64 = _b64.b64encode(
            plaintext,
        ).decode("ascii")
        envelope_b64 = None

    update = GradientUpdate(
        job_id=job_id,
        round_index=int(round_index),
        worker_node_id=worker_node_id,
        gradient_b64=gradient_b64,
        sample_count=int(sample_count),
        worker_attestation_b64=worker_attestation_b64,
        worker_signature_b64="",
        timestamp=(
            timestamp if timestamp is not None
            else time.time()
        ),
        gradient_envelope_b64=envelope_b64,
    )
    return sign_gradient_update(
        update,
        worker_privkey_b64=worker_privkey_b64,
    )
