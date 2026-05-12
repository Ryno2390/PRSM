"""Sprint 316a — distributed tensor-parallel forward over HTTP.

Sprint 316 shipped the math primitive
(`forward_column_parallel_sharded`) running in-process.
This module is the HTTP transport — each TP worker holds
its own weight shard W_k locally, exposes
`POST /compute/inference/tensor_parallel/shard`, and this
module's `http_tp_forward` dispatches the same input X to
all workers, gathers partials, concatenates along the
last dim.

Topology contrast vs sprint 313 (pipeline HTTP):
  - sprint 313: SEQUENTIAL chain (stage K's output feeds
    stage K+1's input)
  - sprint 316a: PARALLEL fan-out (same X to all shards;
    gather + concat)

v1 dispatches sequentially per-worker for simplicity (the
mathematical result is identical to true-parallel
dispatch; the perf delta matters only for production
deployment where round-trip latency dominates). Real
async-parallel dispatch using threading or asyncio = 316b
follow-on.
"""
from __future__ import annotations

import base64
from typing import Any, Callable, List, Optional


class HTTPTPRunnerError(Exception):
    """Raised when any TP worker returns a non-2xx
    response or the response shape is malformed."""


def _default_http_post(
    url: str, *, json: dict, timeout: float,
):
    import httpx
    return httpx.post(url, json=json, timeout=timeout)


def http_tp_forward(
    *,
    worker_urls: List[str],
    input_activations: bytes,
    http_post: Optional[Callable[..., Any]] = None,
    request_timeout: float = 60.0,
) -> bytes:
    """Distribute input X to N TP workers; each computes
    its partial matmul X @ W_local; gather + concatenate
    along last dim.

    Each worker_url is the base URL of a TP worker node
    that exposes /compute/inference/tensor_parallel/shard
    AND has been pre-configured with its weight shard
    (Node._tp_weight_shard set at startup).

    Returns: serialized concatenated activation bytes
    (sprint 314 wire format — JSON envelope with shape +
    dtype + data_b64).
    """
    if not worker_urls:
        raise ValueError(
            "worker_urls must be non-empty"
        )
    post = http_post if http_post is not None else (
        _default_http_post
    )
    input_b64 = base64.b64encode(
        input_activations,
    ).decode("ascii")

    partial_b64s: List[str] = []
    for shard_id, url in enumerate(worker_urls):
        endpoint = (
            url.rstrip("/")
            + "/compute/inference/tensor_parallel/shard"
        )
        body = {
            "shard_id": int(shard_id),
            "input_activations_b64": input_b64,
        }
        try:
            response = post(
                endpoint, json=body,
                timeout=request_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPTPRunnerError(
                f"HTTP POST to TP worker {url!r} "
                f"failed: {type(exc).__name__}: {exc}"
            )
        status = getattr(response, "status_code", None)
        if status is None or status >= 400:
            try:
                detail = response.json().get(
                    "detail", "(no detail)",
                )
            except Exception:  # noqa: BLE001
                detail = "(no detail)"
            raise HTTPTPRunnerError(
                f"TP worker {url!r} (shard_id="
                f"{shard_id}) returned {status}: "
                f"{detail}"
            )
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPTPRunnerError(
                f"TP worker {url!r} returned non-JSON "
                f"response: {type(exc).__name__}: {exc}"
            )
        partial = payload.get("output_partial_b64")
        if not isinstance(partial, str):
            raise HTTPTPRunnerError(
                f"TP worker {url!r} response missing "
                f"'output_partial_b64' field"
            )
        partial_b64s.append(partial)

    # Concatenate partial outputs along the last dim. We
    # deserialize each partial here + use the sprint 316
    # math primitive — keeps the wire format consistent.
    # PyTorch import is lazy to keep import cost off the
    # non-TP code paths.
    import torch
    from prsm.compute.inference.pytorch_stage_runner import (
        deserialize_activation,
        serialize_activation,
    )
    partials = [
        deserialize_activation(base64.b64decode(p))
        for p in partial_b64s
    ]
    concatenated = torch.cat(partials, dim=-1)
    return serialize_activation(concatenated)
