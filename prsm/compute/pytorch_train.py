"""Sprint 309 — real PyTorch backend in the FL TrainingFn
Protocol.

Sprint 308b shipped the pluggable `TrainingFn` interface
with a deterministic-stub default. This module wires a
real PyTorch training loop into the same surface so
workers can produce honest parameter-delta gradients
instead of synthetic stubs.

The output is a flattened parameter-delta vector
compatible with sprint 308's FedAvg / FedMedian
aggregators. Sprint 309 v1 = FedAvg-style "gradient =
new_params - initial_params" — the parameter delta IS the
quantity averaged across workers (matches the standard FL
literature).

Pluggable surface:
  TorchTrainConfig — caller supplies model_factory,
                     optimizer_factory, loss_fn, epochs,
                     batch_size, optional seed
  DataLoader Protocol — caller resolves dataset_cid →
                        (features, labels) tensors;
                        defaults are synthetic for testing
  pytorch_train_fn(config, data_loader) → TrainingFn

Out of scope v1: GPU dispatch (CPU only); multi-GPU DDP;
mixed-precision; live PRSM content-layer integration
(operators wire that via the DataLoader Protocol).

This module imports torch at module level — only callers
who actually want a real backend pay the dependency cost.
Stub callers continue using sprint 308b's
deterministic_stub_train_fn.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data import TensorDataset


# ── DataLoader Protocol ──────────────────────────────


class DataLoader(Protocol):
    """A function that resolves a dataset_cid to a pair of
    tensors (features, labels). Operators integrating PRSM
    plug their own implementation here — typically one
    that fetches the encrypted shard via PRSM's content
    layer and decrypts inside the TEE."""

    def __call__(
        self, dataset_cid: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


def synthetic_data_loader(
    *,
    input_dim: int,
    output_dim: int,
    n_samples: int,
) -> DataLoader:
    """Default DataLoader for testing — produces
    reproducible-per-cid synthetic data. Features sampled
    from N(0, 1) seeded by SHA-256(cid); labels are
    derived from a fixed linear projection so the model
    has a real signal to learn."""

    def loader(
        dataset_cid: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Seed deterministically from cid
        seed_bytes = hashlib.sha256(
            dataset_cid.encode("utf-8"),
        ).digest()
        seed = int.from_bytes(seed_bytes[:8], "big")
        seed = seed & 0x7FFFFFFFFFFFFFFF  # keep positive
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(
            n_samples, input_dim, generator=gen,
        )
        # Hidden linear function for the label
        true_w = torch.randn(
            input_dim, output_dim, generator=gen,
        )
        logits = x @ true_w
        y = logits.argmax(dim=1)
        return x, y

    return loader


# ── Config ──────────────────────────────────────────


@dataclass
class TorchTrainConfig:
    """Caller-supplied training configuration.

    model_factory: () -> nn.Module — fresh model instance
        per training round (parameters loaded from the
        round's initial state inside pytorch_train_fn).
    optimizer_factory: (params) -> Optimizer — fresh
        optimizer wrapping the model's parameters.
    loss_fn: nn.Module — the loss to minimize.
    epochs: int >= 1 — local epochs per round.
    batch_size: int >= 1 — mini-batch size.
    seed: Optional[int] — set for deterministic training
        (useful for reproducibility tests + audit).
    """

    model_factory: Callable[[], nn.Module]
    optimizer_factory: Callable[
        [List[torch.nn.Parameter]], torch.optim.Optimizer,
    ]
    loss_fn: nn.Module
    epochs: int
    batch_size: int
    seed: Optional[int] = None

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(
                f"epochs must be >= 1, got {self.epochs}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got "
                f"{self.batch_size}"
            )


# ── Training fn factory ─────────────────────────────


def pytorch_train_fn(
    *,
    config: TorchTrainConfig,
    data_loader: DataLoader,
):
    """Returns a TrainingFn compatible with
    prsm.compute.train.TrainingFn. The returned callable
    is invoked by compute_signed_gradient_update with the
    round's (job_id, round_index, dataset_cid,
    sample_count); it produces a flattened
    parameter-delta gradient.

    The training loop:
      1. Instantiate model + optimizer via the factories
      2. Snapshot initial parameters (flat vector)
      3. Load data via data_loader(dataset_cid)
      4. Train for config.epochs over mini-batches
      5. Snapshot final parameters (flat vector)
      6. Return (final - initial) as a Python list of
         floats
    """

    def _train(
        *,
        job_id: str,
        round_index: int,
        dataset_cid: str,
        sample_count: int,
    ) -> List[float]:
        if config.seed is not None:
            torch.manual_seed(int(config.seed))

        model = config.model_factory()
        model.train()

        # Snapshot initial parameters
        initial_params = parameters_to_vector(
            model.parameters(),
        ).detach().clone()

        # Build optimizer AFTER snapshotting (optimizer
        # holds parameter references)
        optimizer = config.optimizer_factory(
            list(model.parameters()),
        )

        # Load training data
        features, labels = data_loader(dataset_cid)
        dataset = TensorDataset(features, labels)
        # Generator pinned for determinism when seed set
        if config.seed is not None:
            loader_gen = torch.Generator().manual_seed(
                int(config.seed),
            )
        else:
            loader_gen = None
        torch_loader = _TorchDataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=loader_gen,
        )

        # Training loop
        for _epoch in range(config.epochs):
            for x_batch, y_batch in torch_loader:
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = config.loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

        # Snapshot final parameters + compute delta
        final_params = parameters_to_vector(
            model.parameters(),
        ).detach()
        delta = (final_params - initial_params).tolist()
        return delta

    return _train
