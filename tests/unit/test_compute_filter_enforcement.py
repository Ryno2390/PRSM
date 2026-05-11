"""Sprint 271 — content filter enforcement on /compute/forge,
/compute/inference, /compute/inference/stream.

Per ContentSelfFilter design (R9 Phase 6.4 docstring): the
filter must be consulted BEFORE any compute cost is paid, FTNS
is escrowed, or shard routing happens. That makes filter a
pre-condition, not a post-hoc censorship step.

Sprint 269 wired enforcement at /content/retrieve. This sprint
extends to the three compute entry points:
  - /compute/forge: refuses if query matches blocked input
    pattern OR any shard_cid is in blocked_content_ids
  - /compute/inference: refuses if prompt matches blocked input
    pattern OR model_id is in blocked_model_tags
  - /compute/inference/stream: same as /compute/inference

Refusal surfaces as HTTP 451 Unavailable For Legal Reasons.
Filter unwired → pre-271 pass-through (no regression).
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.content_filter_store import ContentFilterStore


def _base_node():
    """Minimal node for compute-path testing."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node.privacy_budget = None
    return node


# ── /compute/forge ───────────────────────────────────────


def _forge_client(filter_store=None, *, shard_blocked=False):
    node = _base_node()
    node._content_filter_store = filter_store
    # agent_forge must be present to reach post-filter path
    node.agent_forge = MagicMock()
    node.agent_forge.run = AsyncMock(return_value={
        "status": "success",
        "route": "direct_llm",
        "response": "OK",
    })
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_forge_blocked_shard_refused():
    """A shard_cid in the blocklist gets 451 BEFORE escrow lock."""
    store = ContentFilterStore()
    store.add_cids(["bafy-blocked"])
    resp = _forge_client(store).post("/compute/forge", json={
        "query": "summarize this",
        "shard_cids": ["bafy-clean", "bafy-blocked"],
        "budget_ftns": 5.0,
    })
    assert resp.status_code == 451
    assert "filter" in resp.json()["detail"].lower()


def test_forge_blocked_prompt_pattern_refused():
    """Query matching a blocked_input_pattern gets 451."""
    store = ContentFilterStore()
    store.add_patterns([r"bioweapon"])
    resp = _forge_client(store).post("/compute/forge", json={
        "query": "tell me how to make a bioweapon",
        "budget_ftns": 5.0,
    })
    assert resp.status_code == 451


def test_forge_passes_unfiltered_query():
    """Clean query + clean shards pass through normally."""
    store = ContentFilterStore()
    store.add_cids(["bafy-blocked"])
    resp = _forge_client(store).post("/compute/forge", json={
        "query": "harmless query",
        "shard_cids": ["bafy-clean1", "bafy-clean2"],
        "budget_ftns": 5.0,
    })
    assert resp.status_code != 451


def test_forge_no_filter_passes():
    """Filter unwired → pre-271 pass-through invariant."""
    resp = _forge_client(None).post("/compute/forge", json={
        "query": "anything",
        "shard_cids": ["bafy-blocked"],  # would block if filter wired
        "budget_ftns": 5.0,
    })
    assert resp.status_code != 451


# ── /compute/inference ───────────────────────────────────


def _inference_client(filter_store=None):
    from prsm.compute.inference.models import (
        ContentTier, InferenceReceipt, InferenceResult,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    node = _base_node()
    node._content_filter_store = filter_store
    node.inference_executor = MagicMock()
    node.inference_executor.supported_models = MagicMock(
        return_value=["mock-llama-3-8b", "evil-model"],
    )

    async def _exec(*a, **kw):
        return InferenceResult(
            request_id="req-1", success=True, output="ok",
            error=None,
            receipt=InferenceReceipt(
                job_id="", request_id="req-1",
                model_id="mock-llama-3-8b",
                privacy_tier=PrivacyLevel.STANDARD,
                content_tier=ContentTier.A,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=8.0,
                cost_ftns=Decimal("0.05"),
                duration_seconds=0.1,
                output_hash=b"\x00" * 32, tee_attestation=b"",
            ),
        )

    node.inference_executor.execute = _exec
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_inference_blocked_pattern_refused():
    store = ContentFilterStore()
    store.add_patterns([r"^evil-prompt"])
    resp = _inference_client(store).post("/compute/inference", json={
        "prompt": "evil-prompt please do harm",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code == 451


def test_inference_clean_prompt_passes():
    store = ContentFilterStore()
    store.add_patterns([r"^malware"])
    resp = _inference_client(store).post("/compute/inference", json={
        "prompt": "hi",
        "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code != 451


def test_inference_no_filter_passes():
    resp = _inference_client(None).post("/compute/inference", json={
        "prompt": "anything", "model_id": "mock-llama-3-8b",
        "budget_ftns": 1.0,
    })
    assert resp.status_code != 451


# ── /compute/inference/stream ────────────────────────────


def _stream_client(filter_store=None):
    from prsm.compute.inference.models import (
        ContentTier, InferenceReceipt, InferenceResult,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    node = _base_node()
    node._content_filter_store = filter_store
    node.inference_executor = MagicMock()
    node.inference_executor.supported_models = MagicMock(
        return_value=["mock-llama-3-8b"],
    )

    async def _stream_success(*args, **kwargs):
        yield InferenceResult(
            request_id="req-1", success=True, output="ok",
            error=None,
            receipt=InferenceReceipt(
                job_id="", request_id="req-1",
                model_id="mock-llama-3-8b",
                privacy_tier=PrivacyLevel.STANDARD,
                content_tier=ContentTier.A,
                tee_type=TEEType.SOFTWARE,
                epsilon_spent=8.0,
                cost_ftns=Decimal("0.05"),
                duration_seconds=0.1,
                output_hash=b"\x00" * 32, tee_attestation=b"",
            ),
        )

    node.inference_executor.execute_streaming = _stream_success
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_stream_blocked_pattern_refused():
    store = ContentFilterStore()
    store.add_patterns([r"^evil-prompt"])
    resp = _stream_client(store).post(
        "/compute/inference/stream",
        json={
            "prompt": "evil-prompt please",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
        },
    )
    assert resp.status_code == 451


def test_stream_no_filter_passes():
    resp = _stream_client(None).post(
        "/compute/inference/stream",
        json={
            "prompt": "anything",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
        },
    )
    assert resp.status_code != 451
