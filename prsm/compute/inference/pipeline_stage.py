"""Sprint 312 — pipeline stage runner Protocol + stub.

A `StageRunner` is a callable invoked by the
PipelineOrchestrator at each stage. It takes the previous
stage's output activations + this stage's metadata, runs
this stage's compute (in v1: a stub), and returns the
activations to pass to the next stage.

Real PyTorch per-stage forward pass lands later via the
same Protocol — sprint 314 will ship a
`pytorch_stage_runner(model_factory, partition)` factory
that constructs and runs the per-stage layers. Sprint 312
ships the orchestration layer + a deterministic stub
runner so the multi-stage coordination logic + receipt
chain can be tested in isolation.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Callable, List, Protocol


class StageRunner(Protocol):
    """Compute one stage's forward pass. v1 is sync;
    cross-node distribution (sprint 313) introduces an
    async sibling."""

    def __call__(
        self,
        *,
        input_activations: bytes,
        stage_id: int,
        layer_indices: List[int],
    ) -> bytes: ...


_DEFAULT_OUTPUT_SIZE = 64


def deterministic_stub_stage_runner(
    *,
    output_size: int = _DEFAULT_OUTPUT_SIZE,
) -> StageRunner:
    """Returns a StageRunner that produces fixed-size
    output bytes seeded by SHA-256 of the inputs.
    Deterministic per (input, stage_id, layer_indices) —
    suitable for testing orchestration + receipt chain
    without a real model."""

    def _run(
        *,
        input_activations: bytes,
        stage_id: int,
        layer_indices: List[int],
    ) -> bytes:
        seed = hashlib.sha256()
        seed.update(struct.pack("<I", stage_id))
        # Hash layer indices in sorted-canonical form to
        # avoid surprises if caller passes them in
        # non-sorted order
        for layer in sorted(layer_indices):
            seed.update(struct.pack("<I", layer))
        seed.update(input_activations)
        digest = seed.digest()
        # Stretch the 32-byte digest to output_size via
        # SHA-256 chaining (this is a fine PRG for
        # producing deterministic stub bytes; not a real
        # cryptographic stream)
        out = bytearray()
        current = digest
        while len(out) < output_size:
            current = hashlib.sha256(current).digest()
            out.extend(current)
        return bytes(out[:output_size])

    return _run
