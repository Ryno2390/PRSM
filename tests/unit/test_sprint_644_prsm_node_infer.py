"""Sprint 644 — unit tests for `prsm node infer` CLI.

Sprints 633-643 built up rich surface for the operator inference
command — peer selection, sampling, save-receipts, warm-up — but
the only coverage was live-attestation against the active Mac↔
droplet fleet. A regression in option handling or error-path code
wouldn't trip CI, only manifest when an operator hit it.

These tests use Click's CliRunner to invoke the command with
heavily mocked external surfaces (httpx, transformers, NodeConfig)
so CI can catch breakage without a running daemon + fleet.
"""
from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from prsm.cli import node


@pytest.fixture
def runner():
    return CliRunner()


def _peers_resp(connected_count=1):
    """Mock /peers response with N connected peers."""
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value={
        "connected": [
            {"peer_id": f"peer-{i}", "address": "1.2.3.4:9001"}
            for i in range(connected_count)
        ],
        "known": [],
        "connected_count": connected_count,
        "known_count": connected_count,
    })
    return r


def _signed_response_bytes(activation_shape=(1, 1, 50)):
    """Build a fake RunLayerSliceResponse-encoded bytes blob the
    CLI can parse_message() through to its argmax logic. We
    construct a real response signed by a throwaway identity so
    parse_message accepts it.
    """
    from prsm.compute.chain_rpc.protocol import (
        RunLayerSliceResponse, encode_message,
    )
    from prsm.compute.tee.models import TEEType
    from prsm.node.identity import generate_node_identity
    import numpy as np

    identity = generate_node_identity(display_name="test-stage")
    # Build logits where argmax = a known position so the CLI
    # decoded "next_token_id" is predictable.
    logits = np.zeros(activation_shape, dtype=np.float32)
    logits[0, 0, 7] = 99.0  # argmax → token 7
    resp = RunLayerSliceResponse.sign(
        identity=identity,
        request_id="test-req",
        activation_blob=logits.tobytes(),
        activation_shape=activation_shape,
        activation_dtype="float32",
        duration_seconds=0.1,
        tee_attestation=b"sw-attest",
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )
    return encode_message(resp)


def _chain_exec_ping_ok():
    """Mock the chain-exec-ping HTTP response (POST /admin/...)."""
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(return_value={
        "response_b64": base64.b64encode(
            _signed_response_bytes(),
        ).decode("ascii"),
    })
    return r


class _FakeTokenizer:
    """Stub HF tokenizer — encode returns shape-(1, N), decode
    returns a predictable token string."""

    def encode(self, text, return_tensors=None):
        import torch
        # 1 token per character for simplicity
        n = max(1, len(text))
        return torch.zeros((1, n), dtype=torch.long)

    def decode(self, ids):
        return f" tok{ids[0]}"


class _FakeHFModel:
    """Stub HF causal-LM — exposes transformer.wte/wpe + a config."""

    class _Config:
        num_hidden_layers = 12
        n_layer = 12

    config = _Config()

    class _Transformer:
        @staticmethod
        def wte(input_ids):
            import torch
            return torch.zeros(
                input_ids.shape + (8,), dtype=torch.float32,
            )

        @staticmethod
        def wpe(positions):
            import torch
            return torch.zeros(
                positions.shape + (8,), dtype=torch.float32,
            )

    transformer = _Transformer()

    def eval(self):
        return self


@pytest.fixture
def hf_stubs():
    """Patch HuggingFace imports to return our stubs without
    actually loading gpt2 weights."""
    fake_auto_tok = MagicMock(
        from_pretrained=MagicMock(return_value=_FakeTokenizer()),
    )
    fake_auto_model = MagicMock(
        from_pretrained=MagicMock(return_value=_FakeHFModel()),
    )
    # The CLI imports `transformers` inside the command body. Patch
    # the modules in sys.modules so the lazy import finds them.
    with patch.dict("sys.modules", {
        "transformers": MagicMock(
            AutoTokenizer=fake_auto_tok,
            AutoModelForCausalLM=fake_auto_model,
        ),
    }):
        yield


@pytest.fixture
def identity_stub(tmp_path):
    """Provide a real NodeIdentity to settler-sign HandoffTokens.
    The CLI calls `load_node_identity(cfg.identity_path)`.
    """
    from prsm.node.identity import generate_node_identity, save_node_identity
    identity = generate_node_identity(display_name="test-settler")
    id_path = tmp_path / "identity.json"
    save_node_identity(identity, id_path)

    fake_cfg = MagicMock()
    fake_cfg.identity_path = id_path
    with patch("prsm.node.config.NodeConfig.load", return_value=fake_cfg):
        yield identity


# --------------------------------------------------------------------------
# Failure-path coverage (no fleet needed)
# --------------------------------------------------------------------------


def test_unreachable_daemon_clean_error(runner):
    """Daemon down → CLI surfaces a clean error pointing at
    `prsm node start`, not a traceback.
    """
    with patch("httpx.get", side_effect=ConnectionError("refused")):
        result = runner.invoke(
            node, ["infer", "--prompt", "hi", "-n", "1"],
        )
    assert result.exit_code != 0
    assert "Failed to reach local daemon" in result.output
    assert "prsm node start" in result.output


def test_no_connected_peers_clean_error(runner):
    """No peers connected → clean error pointing at the discovery
    gap, not a tracebcack.
    """
    with patch("httpx.get", return_value=_peers_resp(connected_count=0)):
        result = runner.invoke(
            node, ["infer", "--prompt", "hi", "-n", "1"],
        )
    assert result.exit_code != 0
    assert "No connected peers" in result.output


def test_missing_hf_deps_clean_error(runner):
    """HF deps not installed → clean error with install hint.

    Simulated by making the `transformers` import raise ImportError.
    """
    with patch("httpx.get", return_value=_peers_resp(connected_count=1)):
        # Force the transformers import to fail
        import builtins
        orig_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named transformers")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_import):
            result = runner.invoke(
                node, ["infer", "--prompt", "hi", "-n", "1"],
            )
    assert result.exit_code != 0
    assert "HuggingFace deps missing" in result.output


def test_explicit_stage_peer_id_used(runner, hf_stubs, identity_stub):
    """--stage-peer-id overrides first-connected default. We can
    observe this by checking the chain-exec-ping call's peer_id
    arg lands as our custom value.
    """
    posted_payloads = []

    def post_capture(url, **kwargs):
        posted_payloads.append(kwargs.get("json", {}))
        return _chain_exec_ping_ok()

    with patch("httpx.get", return_value=_peers_resp(connected_count=2)), \
         patch("httpx.post", side_effect=post_capture):
        result = runner.invoke(node, [
            "infer", "--prompt", "hi", "-n", "1",
            "--stage-peer-id", "explicit-peer-x",
        ])
    assert result.exit_code == 0, result.output
    assert posted_payloads[0]["peer_id"] == "explicit-peer-x"


def test_save_receipts_writes_jsonl(runner, hf_stubs, identity_stub, tmp_path):
    """--save-receipts produces one JSON line per token with the
    expected fields. Defends sprint 634 against silent regressions.
    """
    receipts = tmp_path / "r.jsonl"
    with patch("httpx.get", return_value=_peers_resp()), \
         patch("httpx.post", return_value=_chain_exec_ping_ok()):
        result = runner.invoke(node, [
            "infer", "--prompt", "hi", "-n", "2",
            "--save-receipts", str(receipts),
        ])
    assert result.exit_code == 0, result.output
    assert receipts.exists()
    lines = receipts.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    # Sprint 634/635/639 fields must all be present
    for field in (
        "request_id", "settler_node_id", "stage_node_id",
        "stage_signature_b64", "model_id", "activation_blob_b64",
        "tee_attestation_b64", "next_token_id", "sampling_mode",
    ):
        assert field in rec, (
            f"receipt missing field {field!r}; got {rec.keys()}"
        )
    # sprint 639: default sampling = greedy
    assert rec["sampling_mode"] == "greedy"


def test_save_receipts_gzip_writes_compressed(
    runner, hf_stubs, identity_stub, tmp_path,
):
    """Sprint 642 — .gz extension triggers gzip stream."""
    import gzip
    receipts = tmp_path / "r.jsonl.gz"
    with patch("httpx.get", return_value=_peers_resp()), \
         patch("httpx.post", return_value=_chain_exec_ping_ok()):
        result = runner.invoke(node, [
            "infer", "--prompt", "h", "-n", "1",
            "--save-receipts", str(receipts),
        ])
    assert result.exit_code == 0, result.output
    # gzip header magic bytes
    raw = receipts.read_bytes()
    assert raw[:2] == b"\x1f\x8b", (
        "file is not gzip-compressed"
    )
    # Round-trip parse to confirm the JSON content is valid
    with gzip.open(receipts, "rt", encoding="utf-8") as f:
        rec = json.loads(f.read().strip())
    assert "stage_signature_b64" in rec


def test_temperature_changes_sampling_mode(
    runner, hf_stubs, identity_stub, tmp_path,
):
    """Sprint 639 — --temperature changes sampling_mode."""
    receipts = tmp_path / "r.jsonl"
    with patch("httpx.get", return_value=_peers_resp()), \
         patch("httpx.post", return_value=_chain_exec_ping_ok()):
        result = runner.invoke(node, [
            "infer", "--prompt", "h", "-n", "1",
            "--temperature", "1.0", "--top-k", "10", "--seed", "42",
            "--save-receipts", str(receipts),
        ])
    assert result.exit_code == 0, result.output
    rec = json.loads(receipts.read_text().strip().splitlines()[0])
    assert "temperature:1.000" in rec["sampling_mode"]
    assert "top_k:10" in rec["sampling_mode"]
    assert "seed:42" in rec["sampling_mode"]


def test_warm_up_fires_extra_request(
    runner, hf_stubs, identity_stub,
):
    """Sprint 643 — --warm-up adds one extra chain-exec-ping call."""
    post_count = [0]

    def post_count_capture(*a, **kw):
        post_count[0] += 1
        return _chain_exec_ping_ok()

    with patch("httpx.get", return_value=_peers_resp()), \
         patch("httpx.post", side_effect=post_count_capture):
        # Without warm-up: 1 token = 1 post
        runner.invoke(node, [
            "infer", "--prompt", "h", "-n", "1",
        ])
        baseline = post_count[0]
        post_count[0] = 0
        # With warm-up: should fire one extra
        runner.invoke(node, [
            "infer", "--prompt", "h", "-n", "1", "--warm-up",
        ])
        warm_up_count = post_count[0]
    assert warm_up_count == baseline + 1, (
        f"warm-up should add exactly 1 extra POST; "
        f"baseline={baseline} warm={warm_up_count}"
    )


def test_chain_exec_ping_http_error_clean_failure(
    runner, hf_stubs, identity_stub,
):
    """Stage returns 500 → CLI surfaces it without traceback."""
    err = MagicMock()
    err.status_code = 500
    err.text = "stage executor raised"
    with patch("httpx.get", return_value=_peers_resp()), \
         patch("httpx.post", return_value=err):
        result = runner.invoke(node, [
            "infer", "--prompt", "h", "-n", "1",
        ])
    assert result.exit_code != 0
    assert "HTTP 500" in result.output


def test_stage_error_response_clean_failure(
    runner, hf_stubs, identity_stub,
):
    """Stage returns a signed StageError → CLI reports code+message
    without traceback.
    """
    from prsm.compute.chain_rpc.protocol import (
        StageError, encode_message,
    )
    stage_err = StageError(
        request_id="x",
        code="MODEL_NOT_FOUND",
        message="model 'foo' not in registry",
    )
    err_bytes = encode_message(stage_err)
    err_resp = MagicMock()
    err_resp.status_code = 200
    err_resp.json = MagicMock(return_value={
        "response_b64": base64.b64encode(err_bytes).decode("ascii"),
    })
    with patch("httpx.get", return_value=_peers_resp()), \
         patch("httpx.post", return_value=err_resp):
        result = runner.invoke(node, [
            "infer", "--prompt", "h", "-n", "1",
        ])
    assert result.exit_code != 0
    assert "StageError" in result.output
    assert "MODEL_NOT_FOUND" in result.output


def test_json_format_emits_structured_output(
    runner, hf_stubs, identity_stub,
):
    """--format json → stdout is parseable JSON with the
    canonical fields. Sprint 633 contract.
    """
    with patch("httpx.get", return_value=_peers_resp()), \
         patch("httpx.post", return_value=_chain_exec_ping_ok()):
        result = runner.invoke(node, [
            "infer", "--prompt", "h", "-n", "1",
            "--format", "json",
        ])
    assert result.exit_code == 0, result.output
    # Strip Rich console output prefix if any — JSON is at the end
    out = result.output.strip()
    # Find the first '{' and parse from there
    idx = out.find("{")
    payload = json.loads(out[idx:])
    for field in (
        "prompt", "model", "stage_peer_id", "max_tokens",
        "generated_text", "elapsed_s", "per_token",
    ):
        assert field in payload, f"json output missing {field}"
