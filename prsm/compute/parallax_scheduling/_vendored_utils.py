# Derived from GradientHQ/parallax (Apache-2.0)
# https://github.com/GradientHQ/parallax — commit c8c8ebdaaf2924b6d25e2d1caff61e27374cce0b
# Original copyright (c) 2025 Gradient and contributors
# Modifications copyright (c) 2026 Prismatica, Inc.
# See licenses/PARALLAX-APACHE-2.0.txt for the upstream license terms.
# See licenses/PARALLAX-NOTICE.txt + git history for the full diff vs. upstream.

"""Minimal port of the upstream ``parallax_utils`` helpers required by the
vendored ``scheduling/`` modules.

Only the symbols that the vendored scheduling files import are provided here.
PRSM does not vendor the full ``parallax_utils`` package — see
``licenses/PARALLAX-NOTICE.txt`` for the rationale.

Symbols ported (Apache-2.0 attribution applies):
  - ``get_logger`` (from ``parallax_utils.logging_config``)
  - ``bytes_per_element`` (from ``parallax_utils.utils``)
  - ``compute_max_batch_size`` (from ``parallax_utils.utils``)

The Apple Silicon (MLX) path is intentionally stubbed pending Phase 3.x.6
Task 2/3 adaptation; CUDA and CPU paths are fully functional. See
``compute_max_tokens_in_cache`` below for the gated branch.
"""

from __future__ import annotations

import logging
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# Logging shim — replaces parallax_utils.logging_config.get_logger
# ──────────────────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib ``logging.Logger``. Mirrors the upstream signature.

    Upstream's ``parallax_utils.logging_config.get_logger`` configures a
    rich logger; for PRSM we route through stdlib so log output composes
    with the existing ``structlog`` setup elsewhere in the tree.
    """
    return logging.getLogger(name)


# ──────────────────────────────────────────────────────────────────────────
# bytes_per_element — direct port from parallax_utils/utils.py
# ──────────────────────────────────────────────────────────────────────────


def bytes_per_element(dtype) -> int:
    """Return element size in bytes for supported torch / MLX dtypes.

    Direct port of the upstream function. The MLX branch is preserved
    behind a soft import so PRSM does not require ``mlx`` to be installed
    on every node — CUDA and CPU nodes hit the torch branch only.
    """
    try:
        import mlx.core as mx  # type: ignore
    except Exception:
        mx = None

    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore[assignment]

    if dtype is None:
        return 2
    if torch is not None and dtype in (
        getattr(torch, "float32", None),
        getattr(torch, "bfloat16", None),
        getattr(torch, "float16", None),
        getattr(torch, "half", None),
        getattr(torch, "int8", None),
    ):
        if dtype == torch.float32:
            return 4
        if dtype in (torch.bfloat16, torch.float16, torch.half):
            return 2
        if dtype == torch.int8:
            return 1
    if mx is not None and dtype in (
        getattr(mx, "float32", None),
        getattr(mx, "bfloat16", None),
        getattr(mx, "float16", None),
    ):
        if dtype == mx.float32:
            return 4
        return 2
    return 2


# ──────────────────────────────────────────────────────────────────────────
# compute_max_tokens_in_cache + derive_max_batch_size + compute_max_batch_size
# Direct ports from parallax_utils/utils.py with the upstream HardwareInfo
# branch stubbed pending Phase 3.x.6 Task 2/3.
# ──────────────────────────────────────────────────────────────────────────


def compute_max_tokens_in_cache(
    *,
    device: str,
    kv_cache_memory_fraction: float,
    num_shard_layers: int,
    num_key_value_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    elem_bytes: int,
    available_cache_bytes: Optional[int] = None,
) -> int:
    """Estimate max tokens storable in KV cache given current free memory and fraction.

    Direct port. The Apple-Silicon (MLX) branch raises NotImplementedError
    in this Task-1 vendor commit — Tasks 2/3 will substitute either
    PRSM's own hardware-info shim or remove the path entirely if the
    PRSM scheduler doesn't need it.
    """
    if available_cache_bytes is not None:
        available_cache_size = int(available_cache_bytes)
    elif device == "cuda":
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "torch required for the CUDA path but is not installed"
            ) from exc
        free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        available_cache_size = int(free_bytes * kv_cache_memory_fraction)
    else:
        # Upstream resolves this via parallax.server.server_info.HardwareInfo
        # + mlx.get_active_memory(). PRSM does not vendor that module path
        # in Task 1; Tasks 2/3 substitute or remove. Until then, callers
        # MUST pass available_cache_bytes explicitly on non-CUDA paths.
        raise NotImplementedError(
            "compute_max_tokens_in_cache: non-CUDA path requires "
            "available_cache_bytes to be passed explicitly until Phase "
            "3.x.6 Tasks 2/3 substitute the upstream HardwareInfo dep."
        )
    per_token_cache_size = (
        num_shard_layers * num_key_value_heads * (head_dim_k + head_dim_v) * elem_bytes
    )
    return max(0, available_cache_size // per_token_cache_size)


def derive_max_batch_size(
    *,
    requested_max_batch_size: Optional[int],
    max_sequence_len: Optional[int],
    max_tokens_in_cache: Optional[int],
) -> int:
    """Derive final max_batch_size by clamping requested size against KV capacity.

    Direct port. Preserves upstream's clamping logic.
    """
    candidates = []
    if requested_max_batch_size is not None:
        candidates.append(int(requested_max_batch_size))
    if (
        max_tokens_in_cache is not None
        and max_sequence_len is not None
        and max_sequence_len > 0
    ):
        candidates.append(max(1, int(max_tokens_in_cache) // int(max_sequence_len)))
    if not candidates:
        return 1
    return max(1, min(candidates))


def compute_max_batch_size(
    *,
    requested_max_batch_size: Optional[int],
    max_sequence_len: Optional[int],
    device: Optional[str],
    kv_cache_memory_fraction: float,
    num_shard_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype=None,
    elem_bytes: Optional[int] = None,
    memory_gb: Optional[float] = None,
    head_dim_k: Optional[int] = None,
    head_dim_v: Optional[int] = None,
) -> int:
    """Compute final max_batch_size by chaining dtype→elem_bytes, KV capacity, and clamping.

    Direct port. If ``memory_gb`` is provided, available_cache_bytes is
    computed from it (avoiding the device-heuristic path). Otherwise
    callers on non-CUDA devices must rely on Tasks 2/3 substituting
    a real hardware-info source — until then,
    ``compute_max_tokens_in_cache`` raises NotImplementedError on that
    path.
    """
    eb = elem_bytes if elem_bytes is not None else bytes_per_element(dtype)
    available_cache_bytes = None
    if memory_gb is not None:
        available_cache_bytes = int(memory_gb * 1024**3 * kv_cache_memory_fraction)
    max_tokens = compute_max_tokens_in_cache(
        device=device or "",  # empty means non-cuda path
        kv_cache_memory_fraction=kv_cache_memory_fraction,
        num_shard_layers=num_shard_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim_k=head_dim_k if head_dim_k is not None else head_dim,
        head_dim_v=head_dim_v if head_dim_v is not None else head_dim,
        elem_bytes=eb,
        available_cache_bytes=available_cache_bytes,
    )
    return derive_max_batch_size(
        requested_max_batch_size=requested_max_batch_size,
        max_sequence_len=max_sequence_len,
        max_tokens_in_cache=max_tokens,
    )
