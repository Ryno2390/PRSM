"""Sprint 312 — pipeline stage runner Protocol + stub.

A `StageRunner` is a callable that takes (input_activations,
stage_id, layer_indices) and returns the output activations
to pass to the next stage. v1 ships a deterministic stub
runner suitable for testing the orchestration layer
WITHOUT a real PyTorch model partitioning step (sprint 314
wires real model partitioning via the same Protocol —
pluggable surface).

Properties tested:
  - Stub is deterministic: same (input, stage_id,
    layer_indices) → same output bytes
  - Stub output varies cleanly with any input change
  - Stub output has a stable length (chosen v1 size)
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.pipeline_stage import (
    StageRunner,
    deterministic_stub_stage_runner,
)


# ── Stub determinism ───────────────────────────────


def test_stub_is_deterministic():
    fn = deterministic_stub_stage_runner()
    a = fn(
        input_activations=b"hello",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    b = fn(
        input_activations=b"hello",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    assert a == b


def test_stub_varies_with_input():
    fn = deterministic_stub_stage_runner()
    a = fn(
        input_activations=b"hello",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    b = fn(
        input_activations=b"world",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    assert a != b


def test_stub_varies_with_stage_id():
    fn = deterministic_stub_stage_runner()
    a = fn(
        input_activations=b"hello",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    b = fn(
        input_activations=b"hello",
        stage_id=1, layer_indices=[0, 1, 2],
    )
    assert a != b


def test_stub_varies_with_layer_indices():
    fn = deterministic_stub_stage_runner()
    a = fn(
        input_activations=b"hello",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    b = fn(
        input_activations=b"hello",
        stage_id=0, layer_indices=[3, 4, 5],
    )
    assert a != b


def test_stub_output_has_fixed_length():
    fn = deterministic_stub_stage_runner()
    out = fn(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    # Default v1 output size = 64 bytes (a reasonable
    # placeholder for activation tensors)
    assert len(out) == 64


def test_stub_with_custom_output_size():
    fn = deterministic_stub_stage_runner(output_size=128)
    out = fn(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    assert len(out) == 128


def test_stub_handles_empty_input():
    fn = deterministic_stub_stage_runner()
    out = fn(
        input_activations=b"",
        stage_id=0, layer_indices=[0],
    )
    assert len(out) == 64


def test_stub_handles_large_input():
    fn = deterministic_stub_stage_runner()
    big_input = b"x" * 10_000
    out = fn(
        input_activations=big_input,
        stage_id=0, layer_indices=[0, 1, 2],
    )
    assert len(out) == 64


# ── Protocol compliance ─────────────────────────────


def test_stub_satisfies_runtime_protocol():
    """Runtime-callable check — a StageRunner can be any
    callable with the right signature."""
    fn = deterministic_stub_stage_runner()
    # Direct call as expected
    out = fn(
        input_activations=b"x",
        stage_id=0, layer_indices=[0],
    )
    assert isinstance(out, bytes)
