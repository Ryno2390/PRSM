"""Sprint 316a — distributed tensor-parallel forward over HTTP.

Sprint 316 shipped the math primitive (column-parallel
split + sharded forward) running in-process. This sprint
ships the HTTP transport: each TP worker holds its OWN
weight shard W_k, exposes
`POST /compute/inference/tensor_parallel/shard`, and the
orchestrator dispatches X to all workers IN PARALLEL,
gathers partials, concatenates.

Topology contrast vs sprint 313:
  - sprint 313 (pipeline TP): stages execute SEQUENTIALLY;
    each stage's output feeds the next stage's input
  - sprint 316a (tensor TP): all shards execute IN PARALLEL
    on the same X; outputs are gathered + concatenated
    along the last dim

The math correctness property (sharded forward equals
monolithic forward; sprint 316) is preserved end-to-end
through HTTP.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient

from prsm.compute.inference.http_tp_runner import (
    HTTPTPRunnerError,
    http_tp_forward,
)
from prsm.compute.inference.pytorch_stage_runner import (
    deserialize_activation,
    serialize_activation,
)
from prsm.compute.inference.tensor_parallel import (
    split_weight_column_parallel,
)
from prsm.node.api import create_api_app


def _tp_worker_client(
    *, node_id: str, weight_shard: torch.Tensor,
):
    """A FastAPI TestClient simulating a remote TP worker
    node holding its own weight shard W_k."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = node_id
    node.ftns_ledger = None
    node._tp_weight_shard = weight_shard
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _make_test_client_caller(client_map):
    """Build a function callable like httpx.post that
    dispatches to the right TestClient based on URL prefix.
    client_map maps URL prefix → TestClient."""

    def caller(url, *, json, timeout):
        for prefix, client in client_map.items():
            if url.startswith(prefix):
                path = url[len(prefix):]
                return client.post(path, json=json)
        raise RuntimeError(f"no client for url {url!r}")

    return caller


# ── /compute/inference/tensor_parallel/shard endpoint ──


def test_tp_shard_endpoint_503_no_weight():
    """Worker hasn't been configured with a weight shard
    — refuse loud."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "no-weight"
    node.ftns_ledger = None
    node._tp_weight_shard = None
    client = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    resp = client.post(
        "/compute/inference/tensor_parallel/shard",
        json={
            "shard_id": 0,
            "input_activations_b64": base64.b64encode(
                serialize_activation(torch.randn(2, 4)),
            ).decode(),
        },
    )
    assert resp.status_code == 503


def test_tp_shard_endpoint_happy_path():
    W_local = torch.randn(4, 3)
    client = _tp_worker_client(
        node_id="w0", weight_shard=W_local,
    )
    x = torch.randn(2, 4)
    resp = client.post(
        "/compute/inference/tensor_parallel/shard",
        json={
            "shard_id": 0,
            "input_activations_b64": base64.b64encode(
                serialize_activation(x),
            ).decode(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    partial_bytes = base64.b64decode(
        body["output_partial_b64"],
    )
    partial = deserialize_activation(partial_bytes)
    # Partial output equals local X @ W_local
    assert torch.allclose(partial, x @ W_local, atol=1e-6)


def test_tp_shard_endpoint_422_negative_shard_id():
    W_local = torch.randn(4, 3)
    client = _tp_worker_client(
        node_id="w0", weight_shard=W_local,
    )
    resp = client.post(
        "/compute/inference/tensor_parallel/shard",
        json={
            "shard_id": -1,
            "input_activations_b64": base64.b64encode(
                serialize_activation(torch.randn(2, 4)),
            ).decode(),
        },
    )
    assert resp.status_code == 422


def test_tp_shard_endpoint_422_bad_base64():
    W_local = torch.randn(4, 3)
    client = _tp_worker_client(
        node_id="w0", weight_shard=W_local,
    )
    resp = client.post(
        "/compute/inference/tensor_parallel/shard",
        json={
            "shard_id": 0,
            "input_activations_b64": "not-base64!",
        },
    )
    assert resp.status_code == 422


def test_tp_shard_endpoint_502_dimension_mismatch():
    """Worker has W_local of shape (4, 3) but input X has
    wrong in_features — torch.matmul raises, endpoint
    returns 502."""
    W_local = torch.randn(4, 3)
    client = _tp_worker_client(
        node_id="w0", weight_shard=W_local,
    )
    bad_x = torch.randn(2, 99)  # wrong shape
    resp = client.post(
        "/compute/inference/tensor_parallel/shard",
        json={
            "shard_id": 0,
            "input_activations_b64": base64.b64encode(
                serialize_activation(bad_x),
            ).decode(),
        },
    )
    assert resp.status_code == 502


# ── http_tp_forward ────────────────────────────────


def test_http_tp_forward_two_workers_equals_monolithic():
    """The whole point: 2-way distributed TP forward
    produces the same output as a single-node forward."""
    in_features, out_features = 4, 8
    W = torch.randn(in_features, out_features)
    x = torch.randn(3, in_features)
    monolithic = x @ W

    # Split + distribute
    shards = split_weight_column_parallel(W, n_parts=2)
    client_a = _tp_worker_client(
        node_id="w0", weight_shard=shards[0],
    )
    client_b = _tp_worker_client(
        node_id="w1", weight_shard=shards[1],
    )

    caller = _make_test_client_caller({
        "http://w0": client_a,
        "http://w1": client_b,
    })
    distributed = http_tp_forward(
        worker_urls=["http://w0", "http://w1"],
        input_activations=serialize_activation(x),
        http_post=caller,
    )
    distributed_tensor = deserialize_activation(distributed)
    assert torch.allclose(
        monolithic, distributed_tensor, atol=1e-6,
    )


def test_http_tp_forward_four_workers_equals_monolithic():
    in_features, out_features = 8, 16
    W = torch.randn(in_features, out_features)
    x = torch.randn(2, in_features)
    monolithic = x @ W

    shards = split_weight_column_parallel(W, n_parts=4)
    clients = [
        _tp_worker_client(
            node_id=f"w{i}", weight_shard=shards[i],
        )
        for i in range(4)
    ]
    caller = _make_test_client_caller({
        f"http://w{i}": clients[i] for i in range(4)
    })
    distributed = http_tp_forward(
        worker_urls=[f"http://w{i}" for i in range(4)],
        input_activations=serialize_activation(x),
        http_post=caller,
    )
    distributed_tensor = deserialize_activation(distributed)
    assert torch.allclose(
        monolithic, distributed_tensor, atol=1e-6,
    )


def test_http_tp_forward_3d_input_works():
    """Real transformer activations are 3D. Composition
    has to hold across that case too."""
    W = torch.randn(8, 16)
    x = torch.randn(2, 5, 8)
    monolithic = x @ W
    shards = split_weight_column_parallel(W, n_parts=2)
    client_a = _tp_worker_client(
        node_id="w0", weight_shard=shards[0],
    )
    client_b = _tp_worker_client(
        node_id="w1", weight_shard=shards[1],
    )
    caller = _make_test_client_caller({
        "http://w0": client_a, "http://w1": client_b,
    })
    distributed = http_tp_forward(
        worker_urls=["http://w0", "http://w1"],
        input_activations=serialize_activation(x),
        http_post=caller,
    )
    distributed_tensor = deserialize_activation(distributed)
    assert distributed_tensor.shape == (2, 5, 16)
    assert torch.allclose(
        monolithic, distributed_tensor, atol=1e-6,
    )


def test_http_tp_forward_propagates_worker_5xx():
    """If a worker's shard endpoint fails (5xx),
    http_tp_forward raises HTTPTPRunnerError — the
    operator above sees the failure clearly."""
    W = torch.randn(4, 8)
    x = torch.randn(2, 4)
    shards = split_weight_column_parallel(W, n_parts=2)
    # Worker 0 has no weight → 503
    client_a = _tp_worker_client(
        node_id="w0", weight_shard=None,
    )
    client_b = _tp_worker_client(
        node_id="w1", weight_shard=shards[1],
    )
    caller = _make_test_client_caller({
        "http://w0": client_a, "http://w1": client_b,
    })
    with pytest.raises(HTTPTPRunnerError):
        http_tp_forward(
            worker_urls=["http://w0", "http://w1"],
            input_activations=serialize_activation(x),
            http_post=caller,
        )


def test_http_tp_forward_rejects_empty_worker_list():
    with pytest.raises(ValueError, match="worker"):
        http_tp_forward(
            worker_urls=[],
            input_activations=serialize_activation(
                torch.randn(2, 4),
            ),
            http_post=lambda *a, **k: None,
        )


def test_http_tp_forward_deterministic():
    """Same input + same weight shards → same output bytes.
    HTTP transport doesn't introduce non-determinism."""
    W = torch.randn(4, 8)
    x = torch.randn(2, 4)
    shards = split_weight_column_parallel(W, n_parts=2)
    client_a = _tp_worker_client(
        node_id="w0", weight_shard=shards[0],
    )
    client_b = _tp_worker_client(
        node_id="w1", weight_shard=shards[1],
    )
    caller = _make_test_client_caller({
        "http://w0": client_a, "http://w1": client_b,
    })
    serialized_x = serialize_activation(x)
    out_a = http_tp_forward(
        worker_urls=["http://w0", "http://w1"],
        input_activations=serialized_x,
        http_post=caller,
    )
    out_b = http_tp_forward(
        worker_urls=["http://w0", "http://w1"],
        input_activations=serialized_x,
        http_post=caller,
    )
    assert out_a == out_b
