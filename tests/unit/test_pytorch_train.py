"""Sprint 309 — real PyTorch backend in the TrainingFn
Protocol.

Sprint 308b shipped the FL `TrainingFn` Protocol with a
deterministic-stub default. This sprint wires a real
PyTorch training loop into the same surface so workers
can produce honest gradients instead of stubs. The
output is a flattened parameter-delta vector compatible
with the sprint 308 FedAvg / FedMedian aggregators.

Design choices for v1:
  - Flattened parameters (not per-tensor dicts). Aggregation
    is element-wise; structure is recovered locally via
    torch.nn.utils.vector_to_parameters.
  - Parameter DELTA as "gradient" (FedAvg-style — the
    difference (new_params - initial_params) IS what gets
    averaged across workers). This matches the FL literature
    and what most production FL systems ship.
  - Pluggable DataLoader Protocol — caller resolves
    dataset_cid → (features, labels) tensors. The default
    `synthetic_data_loader` generates reproducible toy data
    for testing without depending on a real dataset corpus.
  - Caller-supplied model factory and optimizer factory.
    Models are not serialized over the wire — code defines
    the model, only parameters travel.

Out of scope: GPU dispatch (CPU only in v1); multi-GPU
DDP; mixed-precision; dataset streaming from PRSM content
layer (operators wire that via the DataLoader Protocol).
"""
from __future__ import annotations

import base64

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from prsm.compute.pytorch_train import (
    TorchTrainConfig,
    pytorch_train_fn,
    synthetic_data_loader,
)
from prsm.enterprise.federated_learning import (
    decode_gradient,
)


# ── Synthetic data loader ───────────────────────────


def test_synthetic_data_loader_returns_tensor_pair():
    loader = synthetic_data_loader(
        input_dim=4, output_dim=2,
        n_samples=16,
    )
    x, y = loader("QmA")
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (16, 4)
    assert y.shape == (16,)


def test_synthetic_data_loader_deterministic():
    """Same cid → same data (for reproducibility in tests)."""
    loader = synthetic_data_loader(
        input_dim=4, output_dim=2, n_samples=8,
    )
    x1, y1 = loader("QmA")
    x2, y2 = loader("QmA")
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_synthetic_data_loader_varies_with_cid():
    loader = synthetic_data_loader(
        input_dim=4, output_dim=2, n_samples=8,
    )
    x_a, _ = loader("QmA")
    x_b, _ = loader("QmB")
    assert not torch.equal(x_a, x_b)


# ── pytorch_train_fn fixture ────────────────────────


def _tiny_mlp_factory():
    """4 → 8 → 2 MLP. ~50 parameters; tiny enough to train
    in milliseconds, big enough to verify aggregation
    works on a non-trivial parameter count."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _expected_param_count() -> int:
    model = _tiny_mlp_factory()
    return sum(p.numel() for p in model.parameters())


def _default_config():
    return TorchTrainConfig(
        model_factory=_tiny_mlp_factory,
        optimizer_factory=lambda params: torch.optim.SGD(
            params, lr=0.01,
        ),
        loss_fn=nn.CrossEntropyLoss(),
        epochs=2,
        batch_size=4,
    )


# ── train_fn output shape + nonzero gradient ────────


def test_train_fn_output_has_expected_param_count():
    fn = pytorch_train_fn(
        config=_default_config(),
        data_loader=synthetic_data_loader(
            input_dim=4, output_dim=2, n_samples=16,
        ),
    )
    gradient = fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=16,
    )
    assert isinstance(gradient, list)
    assert all(isinstance(v, float) for v in gradient)
    assert len(gradient) == _expected_param_count()


def test_train_fn_produces_nonzero_gradient():
    """Real training on real data must move the parameters
    — a non-trivial fraction of the gradient elements
    should be non-zero."""
    fn = pytorch_train_fn(
        config=_default_config(),
        data_loader=synthetic_data_loader(
            input_dim=4, output_dim=2, n_samples=16,
        ),
    )
    gradient = fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=16,
    )
    nonzero_count = sum(1 for v in gradient if v != 0.0)
    # Loose lower bound — we just want to confirm training
    # actually happened. A frozen-model bug would produce
    # ~all-zeros.
    assert nonzero_count > len(gradient) * 0.5


def test_train_fn_deterministic_with_seed():
    """Two runs with the same job_id + round_index +
    dataset_cid produce identical gradients. Reproducibility
    is critical for orchestrator audit + replay."""
    cfg = _default_config()
    cfg.seed = 42
    loader = synthetic_data_loader(
        input_dim=4, output_dim=2, n_samples=16,
    )
    fn = pytorch_train_fn(config=cfg, data_loader=loader)
    g1 = fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=16,
    )
    g2 = fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=16,
    )
    for a, b in zip(g1, g2):
        assert abs(a - b) < 1e-6


def test_train_fn_varies_with_dataset_cid():
    """Different dataset → different gradient (because the
    data the worker trains on is different)."""
    cfg = _default_config()
    cfg.seed = 42
    loader = synthetic_data_loader(
        input_dim=4, output_dim=2, n_samples=16,
    )
    fn = pytorch_train_fn(config=cfg, data_loader=loader)
    g_a = fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=16,
    )
    g_b = fn(
        job_id="j1", round_index=0,
        dataset_cid="QmB", sample_count=16,
    )
    assert g_a != g_b


# ── More-epochs → bigger delta sanity check ─────────


def test_more_epochs_produce_larger_delta_norm():
    """Training for more epochs should move the params
    further (in L2 norm) — a sanity check that the loop
    actually iterates."""
    short_cfg = _default_config()
    short_cfg.epochs = 1
    short_cfg.seed = 7
    long_cfg = _default_config()
    long_cfg.epochs = 5
    long_cfg.seed = 7
    loader = synthetic_data_loader(
        input_dim=4, output_dim=2, n_samples=32,
    )
    g_short = pytorch_train_fn(
        config=short_cfg, data_loader=loader,
    )(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=32,
    )
    g_long = pytorch_train_fn(
        config=long_cfg, data_loader=loader,
    )(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=32,
    )
    norm_short = sum(v * v for v in g_short) ** 0.5
    norm_long = sum(v * v for v in g_long) ** 0.5
    # 5 epochs should move parameters more than 1 epoch
    assert norm_long > norm_short


# ── Config validation ───────────────────────────────


def test_config_rejects_epochs_zero():
    with pytest.raises(ValueError, match="epochs"):
        TorchTrainConfig(
            model_factory=_tiny_mlp_factory,
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=0.01,
            ),
            loss_fn=nn.CrossEntropyLoss(),
            epochs=0,
            batch_size=4,
        )


def test_config_rejects_batch_size_zero():
    with pytest.raises(ValueError, match="batch_size"):
        TorchTrainConfig(
            model_factory=_tiny_mlp_factory,
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=0.01,
            ),
            loss_fn=nn.CrossEntropyLoss(),
            epochs=1,
            batch_size=0,
        )


# ── Integration with sprint 308 FL stack ────────────


def test_integrates_with_compute_signed_gradient_update():
    """End-to-end: pytorch_train_fn → wrapped in
    GradientUpdate → signed → verifiable. The whole point
    of the pluggable TrainingFn Protocol."""
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    from prsm.enterprise.federated_learning import (
        generate_worker_keypair,
        verify_gradient_update_signature,
    )
    priv, pub = generate_worker_keypair()
    fn = pytorch_train_fn(
        config=_default_config(),
        data_loader=synthetic_data_loader(
            input_dim=4, output_dim=2, n_samples=16,
        ),
    )
    update = compute_signed_gradient_update(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=16,
        worker_node_id="n1",
        worker_privkey_b64=priv,
        worker_attestation_b64="",
        train_fn=fn,
    )
    # Signature verifies
    assert verify_gradient_update_signature(update, pub)
    # Gradient has the right shape
    grad = decode_gradient(
        base64.b64decode(update.gradient_b64),
    )
    assert len(grad) == _expected_param_count()


def test_integrates_with_orchestrator_aggregate():
    """Two workers train independently with different
    dataset shards; the orchestrator aggregates via
    FedAvg. The aggregated output is the weighted average
    of the parameter deltas."""
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    from prsm.enterprise.federated_learning import (
        AggregationStrategy,
        FederatedLearningOrchestrator,
        WorkerKey,
        generate_worker_keypair,
    )
    orch = FederatedLearningOrchestrator()
    priv_a, pub_a = generate_worker_keypair()
    priv_b, pub_b = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pub_a))
    orch.register_worker_key(WorkerKey("n2", pub_b))
    job = orch.propose_job(
        model_id="tiny-mlp",
        dataset_cids=["QmA", "QmB"],
        worker_pool=["n1", "n2"],
        rounds_target=1,
        min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
    )
    orch.issue_round(job.job_id)

    fn = pytorch_train_fn(
        config=_default_config(),
        data_loader=synthetic_data_loader(
            input_dim=4, output_dim=2, n_samples=16,
        ),
    )
    for node_id, priv, dataset in (
        ("n1", priv_a, "QmA"),
        ("n2", priv_b, "QmB"),
    ):
        u = compute_signed_gradient_update(
            job_id=job.job_id, round_index=0,
            dataset_cid=dataset, sample_count=16,
            worker_node_id=node_id,
            worker_privkey_b64=priv,
            worker_attestation_b64="",
            train_fn=fn,
        )
        orch.accept_gradient_update(u)
    rnd = orch.aggregate_round(job.job_id, 0)
    aggregated = decode_gradient(rnd.aggregated_update)
    assert len(aggregated) == _expected_param_count()


# ── DataLoader error surfacing ──────────────────────


def test_data_loader_exception_surfaces():
    """If the DataLoader raises (corrupt shard,
    decryption failure, etc.), the error must propagate —
    not silently produce a zero gradient."""
    def failing_loader(cid):
        raise RuntimeError(f"failed to load {cid}")

    fn = pytorch_train_fn(
        config=_default_config(),
        data_loader=failing_loader,
    )
    with pytest.raises(RuntimeError, match="failed to load"):
        fn(
            job_id="j1", round_index=0,
            dataset_cid="QmA", sample_count=16,
        )


def test_data_loader_dimension_mismatch_surfaces():
    """If the DataLoader returns features with wrong
    input_dim for the model, training raises a clear
    error instead of producing garbage."""
    def wrong_dim_loader(cid):
        # Model expects 4 input features; supply 99
        return (
            torch.randn(16, 99),
            torch.randint(0, 2, (16,)),
        )

    fn = pytorch_train_fn(
        config=_default_config(),
        data_loader=wrong_dim_loader,
    )
    with pytest.raises(Exception):
        fn(
            job_id="j1", round_index=0,
            dataset_cid="QmA", sample_count=16,
        )
