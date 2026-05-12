"""Sprint 314 — real PyTorch per-stage forward pass.

Sprint 312 shipped the StageRunner Protocol with a
deterministic-stub default; sprint 313 added HTTP
transport. Sprint 314 wires a real PyTorch backend into
the same Protocol so workers actually run model layers
instead of returning stub bytes.

Design: the model factory returns an nn.Sequential. The
partition's layer_indices select which children of the
Sequential this stage owns. Per-stage forward pass slices
the Sequential by indices and runs the input tensor
through them.

Activations are serialized via a thin tensor envelope
(shape + dtype + data_b64) compatible with cross-stage
transport over the existing /compute/inference/pipeline/
stage endpoint.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from prsm.compute.inference.pytorch_stage_runner import (
    deserialize_activation,
    pytorch_stage_runner,
    serialize_activation,
)


def _tiny_model_factory():
    """6-layer Sequential. Per-stage runner can slice any
    contiguous range."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.Identity(),
    )


# ── Serialization helpers ──────────────────────────


def test_serialize_round_trip_float32():
    t = torch.randn(2, 4)
    out = deserialize_activation(serialize_activation(t))
    assert torch.equal(out, t)


def test_serialize_preserves_shape():
    t = torch.randn(1, 3, 5)
    out = deserialize_activation(serialize_activation(t))
    assert out.shape == (1, 3, 5)


def test_serialize_rejects_non_tensor():
    with pytest.raises(TypeError):
        serialize_activation([1.0, 2.0])


def test_deserialize_rejects_malformed():
    with pytest.raises(Exception):
        deserialize_activation(b"not a valid envelope")


# ── pytorch_stage_runner basic forward ─────────────


def test_runner_forwards_through_assigned_layers():
    """Stage 0 owns layers 0-2; running the runner on an
    input tensor produces the same output as forwarding
    that input through layers 0-2 of the full model."""
    runner = pytorch_stage_runner(
        model_factory=_tiny_model_factory,
    )
    input_t = torch.randn(2, 4)
    input_bytes = serialize_activation(input_t)
    out_bytes = runner(
        input_activations=input_bytes,
        stage_id=0, layer_indices=[0, 1, 2],
    )
    out_t = deserialize_activation(out_bytes)

    # Compare against the equivalent direct forward
    model = _tiny_model_factory()
    expected = nn.Sequential(*list(model.children())[0:3])(
        input_t,
    )
    assert torch.allclose(out_t, expected, atol=1e-6)


def test_runner_composes_across_stages():
    """Output of stage 0 fed into stage 1's runner
    reproduces what the full model would have computed
    for layers 0-5 directly. This is the load-bearing
    composition property — distributed pipeline forward
    pass equals monolithic forward pass."""
    runner = pytorch_stage_runner(
        model_factory=_tiny_model_factory,
    )
    input_t = torch.randn(2, 4)
    # Stage 0: layers 0-2
    stage0_out = deserialize_activation(runner(
        input_activations=serialize_activation(input_t),
        stage_id=0, layer_indices=[0, 1, 2],
    ))
    # Stage 1: layers 3-5
    stage1_out = deserialize_activation(runner(
        input_activations=serialize_activation(stage0_out),
        stage_id=1, layer_indices=[3, 4, 5],
    ))
    # Compare to full-model forward
    model = _tiny_model_factory()
    expected = model(input_t)
    assert torch.allclose(stage1_out, expected, atol=1e-6)


def test_runner_with_layer_index_out_of_range_raises():
    runner = pytorch_stage_runner(
        model_factory=_tiny_model_factory,
    )
    input_t = torch.randn(2, 4)
    with pytest.raises(IndexError):
        runner(
            input_activations=serialize_activation(input_t),
            stage_id=0, layer_indices=[0, 1, 99],
        )


def test_runner_with_dimension_mismatch_surfaces():
    """Feeding a wrong-shape tensor into the assigned
    layers must raise a clear PyTorch error (not silently
    produce wrong output)."""
    runner = pytorch_stage_runner(
        model_factory=_tiny_model_factory,
    )
    bad_input = torch.randn(2, 99)  # model expects (*, 4)
    with pytest.raises(Exception):
        runner(
            input_activations=serialize_activation(bad_input),
            stage_id=0, layer_indices=[0],
        )


def test_runner_deterministic_with_same_model_factory():
    """The model factory is called fresh each StageRunner
    invocation? No — the runner caches the model on first
    call (otherwise per-stage compute repeats model
    construction). Determinism comes from the factory
    being deterministic. Verify same input + same factory
    → same output."""
    runner_a = pytorch_stage_runner(
        model_factory=_tiny_model_factory,
    )
    runner_b = pytorch_stage_runner(
        model_factory=_tiny_model_factory,
    )
    input_t = torch.randn(2, 4)
    input_bytes = serialize_activation(input_t)
    out_a = runner_a(
        input_activations=input_bytes,
        stage_id=0, layer_indices=[0, 1, 2],
    )
    out_b = runner_b(
        input_activations=input_bytes,
        stage_id=0, layer_indices=[0, 1, 2],
    )
    assert out_a == out_b


# ── Integration with PipelineInferenceOrchestrator ─


def test_e2e_with_pipeline_orchestrator():
    """The whole loop: orchestrator proposes a 2-stage
    job, supplies pytorch_stage_runner instances per
    stage, executes, produces a verifiable receipt. End-
    to-end output equals the monolithic forward pass."""
    from prsm.compute.inference.pipeline_orchestrator import (
        PipelineInferenceOrchestrator,
        PipelineRoundStatus,
    )
    from prsm.compute.inference.pipeline_partition import (
        PipelinePartition,
    )
    from prsm.compute.inference.pipeline_receipt import (
        verify_pipeline_receipt,
    )
    from prsm.enterprise.federated_learning import (
        generate_worker_keypair,
    )

    priv, pub = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    # 6 layers split 3/3 across 2 nodes
    partition = PipelinePartition(
        total_layers=6,
        stage_layer_ranges=[[0, 1, 2], [3, 4, 5]],
        stage_node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="tiny-mlp", partition=partition,
    )

    # The prompt IS the input tensor (serialized)
    input_t = torch.randn(2, 4)
    prompt_bytes = serialize_activation(input_t)

    runner_factory = pytorch_stage_runner(
        model_factory=_tiny_model_factory,
    )
    rnd = orch.execute(
        job.job_id, prompt=prompt_bytes,
        stage_runners=[runner_factory, runner_factory],
    )
    assert rnd.status == PipelineRoundStatus.COMPLETED

    # Receipt verifies end-to-end
    result = verify_pipeline_receipt(
        rnd.receipt, orchestrator_pubkey_b64=pub,
    )
    assert result.ok, result.diagnostic

    # And the output equals the monolithic forward pass —
    # this is the key correctness property
    final_bytes = bytes()
    # Reconstruct: stage 0's input was prompt; stage 1's
    # output is the receipt's output_hash hashed-from.
    # Since we can't recover bytes from the hash, run the
    # pipeline manually one more time + compare hashes
    import hashlib
    expected_t = _tiny_model_factory()(input_t)
    expected_bytes = serialize_activation(expected_t)
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    assert rnd.receipt.output_hash == expected_hash
