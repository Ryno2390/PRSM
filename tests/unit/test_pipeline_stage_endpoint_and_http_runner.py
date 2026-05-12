"""Sprint 313 — cross-node activation streaming.

Sprint 312's orchestrator ran all stages in-process. This
sprint converts that into a real distributed system: each
stage runs on a separate physical node, exposing
`POST /compute/inference/pipeline/stage`. The orchestrator
uses an `http_stage_runner(node_url)` factory to call each
stage in sequence, threading activations through HTTP
request/response.

v1 is orchestrator-driven (every stage's output round-trips
through the orchestrator). Worker-to-worker chaining where
stages directly call the next stage's URL is a follow-on —
saves bandwidth but adds coordination complexity not
warranted for v1.

Per-stage signatures = sprint 313a; v1 relies on the
activation hash chain (sprint 312) for tamper detection.
"""
from __future__ import annotations

import base64
import hashlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference.http_stage_runner import (
    HTTPStageRunnerError,
    http_stage_runner,
)
from prsm.compute.inference.pipeline_orchestrator import (
    PipelineInferenceOrchestrator,
    PipelineRoundStatus,
)
from prsm.compute.inference.pipeline_partition import (
    even_layer_partition,
)
from prsm.compute.inference.pipeline_receipt import (
    verify_pipeline_receipt,
)
from prsm.compute.inference.pipeline_stage import (
    deterministic_stub_stage_runner,
)
from prsm.enterprise.federated_learning import (
    generate_worker_keypair,
)
from prsm.node.api import create_api_app


def _stage_worker_client(
    *,
    node_id: str = "stage-worker",
    stage_runner=None,
):
    """A FastAPI TestClient simulating a remote stage
    worker node. The orchestrator's HTTP stage runner
    factory talks to this."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = node_id
    node.ftns_ledger = None
    node._pipeline_stage_runner = (
        stage_runner
        if stage_runner is not None
        else deterministic_stub_stage_runner()
    )
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── Stage worker endpoint ───────────────────────────


def test_stage_endpoint_happy_path():
    client = _stage_worker_client()
    input_bytes = b"hello stage"
    resp = client.post(
        "/compute/inference/pipeline/stage",
        json={
            "job_id": "j-1",
            "round_id": "r-1",
            "stage_id": 0,
            "layer_indices": [0, 1, 2],
            "input_activations_b64": base64.b64encode(
                input_bytes,
            ).decode(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage_id"] == 0
    assert body["worker_node_id"] == "stage-worker"
    out_bytes = base64.b64decode(
        body["output_activations_b64"],
    )
    # Same output as the local stub for the same inputs
    local = deterministic_stub_stage_runner()(
        input_activations=input_bytes,
        stage_id=0, layer_indices=[0, 1, 2],
    )
    assert out_bytes == local


def test_stage_endpoint_503_no_runner():
    """If the operator hasn't configured a stage runner,
    refuse loud."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "no-runner"
    node.ftns_ledger = None
    node._pipeline_stage_runner = None
    client = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    resp = client.post(
        "/compute/inference/pipeline/stage",
        json={
            "job_id": "j", "round_id": "r",
            "stage_id": 0, "layer_indices": [0],
            "input_activations_b64": base64.b64encode(
                b"x",
            ).decode(),
        },
    )
    assert resp.status_code == 503


def test_stage_endpoint_422_negative_stage_id():
    client = _stage_worker_client()
    resp = client.post(
        "/compute/inference/pipeline/stage",
        json={
            "job_id": "j", "round_id": "r",
            "stage_id": -1, "layer_indices": [0],
            "input_activations_b64": base64.b64encode(
                b"x",
            ).decode(),
        },
    )
    assert resp.status_code == 422


def test_stage_endpoint_422_empty_layer_indices():
    """A stage with no layers is operator confusion —
    refuse."""
    client = _stage_worker_client()
    resp = client.post(
        "/compute/inference/pipeline/stage",
        json={
            "job_id": "j", "round_id": "r",
            "stage_id": 0, "layer_indices": [],
            "input_activations_b64": base64.b64encode(
                b"x",
            ).decode(),
        },
    )
    assert resp.status_code == 422


def test_stage_endpoint_422_bad_base64():
    client = _stage_worker_client()
    resp = client.post(
        "/compute/inference/pipeline/stage",
        json={
            "job_id": "j", "round_id": "r",
            "stage_id": 0, "layer_indices": [0],
            "input_activations_b64": "not-base64!",
        },
    )
    assert resp.status_code == 422


def test_stage_endpoint_502_runner_raises():
    """If the configured stage runner raises during compute,
    surface as 502 with the underlying exception."""
    def failing_runner(*, input_activations, stage_id,
                       layer_indices):
        raise RuntimeError("stage compute simulated failure")

    client = _stage_worker_client(stage_runner=failing_runner)
    resp = client.post(
        "/compute/inference/pipeline/stage",
        json={
            "job_id": "j", "round_id": "r",
            "stage_id": 0, "layer_indices": [0],
            "input_activations_b64": base64.b64encode(
                b"x",
            ).decode(),
        },
    )
    assert resp.status_code == 502
    assert "simulated" in resp.json()["detail"].lower()


# ── http_stage_runner factory ───────────────────────


def _make_test_client_caller(stage_client):
    """Build a function callable like `requests.post` but
    routing through a FastAPI TestClient — lets us test
    the runner factory without an actual HTTP server."""

    def caller(url, *, json, timeout):
        # Strip the host prefix; TestClient takes the path
        path = url
        for prefix in (
            "http://localhost", "http://stage-worker",
        ):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        return stage_client.post(path, json=json)

    return caller


def test_http_stage_runner_calls_remote_endpoint():
    stage_client = _stage_worker_client()
    runner = http_stage_runner(
        stage_node_url="http://stage-worker",
        job_id="j-1", round_id="r-1",
        http_post=_make_test_client_caller(stage_client),
    )
    out = runner(
        input_activations=b"input",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    # Output matches what the stub stage runner produces
    local_stub = deterministic_stub_stage_runner()(
        input_activations=b"input",
        stage_id=0, layer_indices=[0, 1, 2],
    )
    assert out == local_stub


def test_http_stage_runner_raises_on_5xx():
    def failing_runner(*, input_activations, stage_id,
                       layer_indices):
        raise RuntimeError("stage failed")

    stage_client = _stage_worker_client(
        stage_runner=failing_runner,
    )
    runner = http_stage_runner(
        stage_node_url="http://stage-worker",
        job_id="j-1", round_id="r-1",
        http_post=_make_test_client_caller(stage_client),
    )
    with pytest.raises(
        HTTPStageRunnerError, match="502|stage failed",
    ):
        runner(
            input_activations=b"input",
            stage_id=0, layer_indices=[0, 1, 2],
        )


def test_http_stage_runner_raises_on_4xx():
    """Server returns 422 (e.g., bad input shape) — the
    runner must surface it, not silently produce
    plausible-but-wrong output."""
    stage_client = _stage_worker_client()
    # Override caller to send a malformed body that the
    # endpoint will reject
    def bad_caller(url, *, json, timeout):
        return stage_client.post(
            "/compute/inference/pipeline/stage",
            json={"junk": "fields"},
        )

    runner = http_stage_runner(
        stage_node_url="http://stage-worker",
        job_id="j-1", round_id="r-1",
        http_post=bad_caller,
    )
    with pytest.raises(HTTPStageRunnerError):
        runner(
            input_activations=b"x",
            stage_id=0, layer_indices=[0],
        )


def test_http_stage_runner_carries_job_and_round_id():
    """The factory bakes job_id + round_id into each call.
    Verify by capturing the outgoing body."""
    captured = {}

    def capturing_caller(url, *, json, timeout):
        captured.update(json)
        # Return a minimal-valid response
        stage_client = _stage_worker_client()
        return stage_client.post(
            "/compute/inference/pipeline/stage",
            json=json,
        )

    runner = http_stage_runner(
        stage_node_url="http://stage-worker",
        job_id="my-job-id", round_id="my-round-id",
        http_post=capturing_caller,
    )
    runner(
        input_activations=b"x",
        stage_id=5, layer_indices=[3, 4, 5],
    )
    assert captured["job_id"] == "my-job-id"
    assert captured["round_id"] == "my-round-id"
    assert captured["stage_id"] == 5
    assert captured["layer_indices"] == [3, 4, 5]


# ── End-to-end: orchestrator + HTTP stage workers ──


def test_e2e_pipeline_over_http_two_stages():
    """The whole loop: orchestrator proposes a 2-stage
    job, each stage is a SEPARATE FastAPI TestClient (a
    "remote node"), orchestrator uses http_stage_runner
    pointed at each, executes, produces a verifiable
    receipt with chain intact."""
    orch_priv, orch_pub = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=orch_priv,
    )
    partition = even_layer_partition(
        total_layers=6, node_ids=["worker-a", "worker-b"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )

    # Two separate "remote" stage worker test clients
    stage_a_client = _stage_worker_client(
        node_id="worker-a",
    )
    stage_b_client = _stage_worker_client(
        node_id="worker-b",
    )

    runner_a = http_stage_runner(
        stage_node_url="http://worker-a",
        job_id=job.job_id, round_id="r-1",
        http_post=_make_test_client_caller(stage_a_client),
    )
    runner_b = http_stage_runner(
        stage_node_url="http://worker-b",
        job_id=job.job_id, round_id="r-1",
        http_post=_make_test_client_caller(stage_b_client),
    )
    rnd = orch.execute(
        job.job_id, prompt=b"distributed inference",
        stage_runners=[runner_a, runner_b],
    )
    assert rnd.status == PipelineRoundStatus.COMPLETED

    # Receipt verifies end-to-end (signature + chain)
    result = verify_pipeline_receipt(
        rnd.receipt, orchestrator_pubkey_b64=orch_pub,
    )
    assert result.ok, result.diagnostic
    assert result.chain_valid


def test_e2e_pipeline_over_http_matches_in_process():
    """Distributed execution via HTTP and in-process
    execution produce IDENTICAL receipts (the stub
    runner is deterministic; the transport doesn't
    change the output). Verifies HTTP transport is
    transparent end-to-end."""
    priv, _ = generate_worker_keypair()
    partition = even_layer_partition(
        total_layers=4, node_ids=["a", "b"],
    )

    # In-process run
    orch_in = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    job_in = orch_in.propose_job(
        model_id="m1", partition=partition,
    )
    rnd_in = orch_in.execute(
        job_in.job_id, prompt=b"same prompt",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )

    # HTTP run
    orch_http = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    job_http = orch_http.propose_job(
        model_id="m1", partition=partition,
    )
    stage_a = _stage_worker_client(node_id="a")
    stage_b = _stage_worker_client(node_id="b")
    runner_a = http_stage_runner(
        stage_node_url="http://a",
        job_id=job_http.job_id, round_id="r",
        http_post=_make_test_client_caller(stage_a),
    )
    runner_b = http_stage_runner(
        stage_node_url="http://b",
        job_id=job_http.job_id, round_id="r",
        http_post=_make_test_client_caller(stage_b),
    )
    rnd_http = orch_http.execute(
        job_http.job_id, prompt=b"same prompt",
        stage_runners=[runner_a, runner_b],
    )

    # Output hashes should match — transport is transparent
    assert (
        rnd_in.receipt.output_hash
        == rnd_http.receipt.output_hash
    )
    for s_in, s_http in zip(
        rnd_in.receipt.stage_receipts,
        rnd_http.receipt.stage_receipts,
    ):
        assert (
            s_in.input_activation_hash
            == s_http.input_activation_hash
        )
        assert (
            s_in.output_activation_hash
            == s_http.output_activation_hash
        )


def test_e2e_remote_stage_failure_propagates():
    """If a remote stage worker's runner crashes (502 from
    the endpoint), the orchestrator marks the round
    FAILED + propagates."""
    priv, _ = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    partition = even_layer_partition(
        total_layers=4, node_ids=["good", "bad"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )

    good = _stage_worker_client(node_id="good")

    def crashing(*, input_activations, stage_id,
                 layer_indices):
        raise RuntimeError("stage runner crashed")

    bad = _stage_worker_client(
        node_id="bad", stage_runner=crashing,
    )

    runner_good = http_stage_runner(
        stage_node_url="http://good",
        job_id=job.job_id, round_id="r",
        http_post=_make_test_client_caller(good),
    )
    runner_bad = http_stage_runner(
        stage_node_url="http://bad",
        job_id=job.job_id, round_id="r",
        http_post=_make_test_client_caller(bad),
    )
    with pytest.raises(HTTPStageRunnerError):
        orch.execute(
            job.job_id, prompt=b"x",
            stage_runners=[runner_good, runner_bad],
        )
    rnd = orch.get_round(job.job_id)
    assert rnd.status == PipelineRoundStatus.FAILED


# ── MITM tamper detection (chain still works) ──────


def test_mitm_substitution_breaks_chain():
    """A MITM that swaps a stage's HTTP response for forged
    activations: orchestrator threads the forged
    activations into the next stage's input. The next
    stage's input_hash records what it ACTUALLY received
    (the forged bytes); receipt verification computes
    hashes from those recorded values and the chain
    holds — but the chain DOESN'T match what the verifier
    would expect if they're comparing to a known-good
    prompt run. Verify by running the same prompt twice,
    one MITMed, and showing the output hashes differ."""
    priv, _ = generate_worker_keypair()
    partition = even_layer_partition(
        total_layers=4, node_ids=["a", "b"],
    )

    # Clean baseline run
    orch_clean = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    job_clean = orch_clean.propose_job(
        model_id="m1", partition=partition,
    )
    stage_a = _stage_worker_client(node_id="a")
    stage_b = _stage_worker_client(node_id="b")
    runner_a_clean = http_stage_runner(
        stage_node_url="http://a",
        job_id=job_clean.job_id, round_id="r",
        http_post=_make_test_client_caller(stage_a),
    )
    runner_b_clean = http_stage_runner(
        stage_node_url="http://b",
        job_id=job_clean.job_id, round_id="r",
        http_post=_make_test_client_caller(stage_b),
    )
    rnd_clean = orch_clean.execute(
        job_clean.job_id, prompt=b"test",
        stage_runners=[runner_a_clean, runner_b_clean],
    )

    # MITMed run: caller for stage A returns FORGED bytes
    def mitm_caller(url, *, json, timeout):
        return type("R", (), {
            "status_code": 200,
            "json": lambda self=None: {
                "stage_id": 0,
                "worker_node_id": "a",
                "output_activations_b64": base64.b64encode(
                    b"FORGED" * 16,
                ).decode(),
            },
        })()

    orch_mitm = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    job_mitm = orch_mitm.propose_job(
        model_id="m1", partition=partition,
    )
    runner_a_mitm = http_stage_runner(
        stage_node_url="http://a",
        job_id=job_mitm.job_id, round_id="r",
        http_post=mitm_caller,
    )
    runner_b_mitm = http_stage_runner(
        stage_node_url="http://b",
        job_id=job_mitm.job_id, round_id="r",
        http_post=_make_test_client_caller(stage_b),
    )
    rnd_mitm = orch_mitm.execute(
        job_mitm.job_id, prompt=b"test",
        stage_runners=[runner_a_mitm, runner_b_mitm],
    )

    # The MITM produces a DIFFERENT output hash than clean
    # — a verifier comparing receipts can detect it
    assert (
        rnd_clean.receipt.output_hash
        != rnd_mitm.receipt.output_hash
    )
