"""Phase 3.x.8.1 Task 2 — unit tests for the SSE streaming endpoint
(``POST /compute/inference/stream``) + its helper functions.

Two test surfaces:
1. Direct tests of the SSE encoder helpers (``_sse_event``,
   ``_token_event_to_dict``, ``_result_to_dict``) without spinning
   an HTTP client.
2. FastAPI TestClient tests of the endpoint against a mock node.
   Full E2E with a real executor is Task 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, AsyncIterator, List, Optional, Union

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
)
from prsm.compute.inference.parallax_executor import InferenceTokenEvent
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.api import (
    _result_to_dict,
    _sse_event,
    _token_event_to_dict,
    create_api_app,
)


# ──────────────────────────────────────────────────────────────────────────
# Helper-function tests (no HTTP)
# ──────────────────────────────────────────────────────────────────────────


class TestSSEEncoderHelpers:
    def test_sse_event_basic_string(self):
        out = _sse_event("token", "hello")
        assert out == b"event: token\ndata: hello\n\n"

    def test_sse_event_dict_serialized_as_json(self):
        out = _sse_event("token", {"sequence_index": 0, "text_delta": "hi"})
        assert out.startswith(b"event: token\ndata: ")
        assert out.endswith(b"\n\n")
        # Extract the JSON payload between "data: " and "\n\n".
        payload = out.split(b"data: ", 1)[1].split(b"\n\n", 1)[0]
        assert json.loads(payload) == {
            "sequence_index": 0, "text_delta": "hi",
        }

    def test_sse_event_decimal_via_default_str(self):
        # Decimal isn't JSON-native; the helper's default=str fallback
        # should serialize it as the decimal's string repr.
        out = _sse_event("result", {"cost_ftns": Decimal("0.5")})
        payload = out.split(b"data: ", 1)[1].split(b"\n\n", 1)[0]
        assert json.loads(payload) == {"cost_ftns": "0.5"}

    def test_sse_event_terminator_is_blank_line(self):
        # Per W3C SSE spec, frames are separated by a blank line.
        out = _sse_event("error", "oops")
        assert out.count(b"\n\n") == 1
        assert out.endswith(b"\n\n")

    def test_token_event_to_dict_preserves_optional_nones(self):
        event = InferenceTokenEvent(
            sequence_index=3, text_delta="hello",
        )
        d = _token_event_to_dict(event)
        # Stable schema: optional fields are emitted as null, not
        # omitted — keeps consumer parsers simple.
        assert d == {
            "sequence_index": 3,
            "text_delta": "hello",
            "token_id": None,
            "finish_reason": None,
        }

    def test_token_event_to_dict_full_fields(self):
        event = InferenceTokenEvent(
            sequence_index=0, text_delta="end",
            token_id=42, finish_reason="stop",
        )
        d = _token_event_to_dict(event)
        assert d["token_id"] == 42
        assert d["finish_reason"] == "stop"

    def test_result_to_dict_rebinds_job_id(self):
        # API-side job_id is authoritative; executor's internal job_id
        # gets overwritten in the response payload.
        receipt = _make_receipt(job_id="parallax-stream-job-deadbeef")
        result = InferenceResult(
            request_id="req-1", success=True,
            output="hello world",
            receipt=receipt,
        )
        d = _result_to_dict(result, job_id="infer-stream-cafef00d")
        assert d["job_id"] == "infer-stream-cafef00d"
        # The receipt sub-dict also has the rebound job_id.
        assert d["receipt"]["job_id"] == "infer-stream-cafef00d"

    def test_result_to_dict_with_no_receipt(self):
        result = InferenceResult(
            request_id="req-1", success=False,
            output="", error="something broke",
        )
        d = _result_to_dict(result, job_id="x")
        assert d["receipt"] is None
        assert d["success"] is False


# ──────────────────────────────────────────────────────────────────────────
# Endpoint integration tests (FastAPI TestClient + mock node)
# ──────────────────────────────────────────────────────────────────────────


def _make_receipt(*, job_id: str = "test-job") -> InferenceReceipt:
    return InferenceReceipt(
        job_id=job_id,
        request_id="req-1",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x01" * 32,
        output_hash=b"\xaa" * 32,
        duration_seconds=0.05,
        cost_ftns=Decimal("0.04"),
        streamed_output=True,
    )


class _FakeIdentity:
    node_id = "test-node-id"

    def sign(self, payload: bytes) -> str:
        # Returns a fake b64 signature — the API is supposed to use
        # node.identity for re-signing but tests just need a callable.
        import base64
        return base64.b64encode(b"\x00" * 64).decode()


@dataclass
class _FakeEscrow:
    """Minimal PaymentEscrow mock that records create/release/refund
    calls so tests can verify the settle-vs-refund branches."""
    create_count: int = 0
    release_count: int = 0
    refund_count: int = 0
    last_refund_reason: Optional[str] = None
    create_returns_none: bool = False

    async def create_escrow(self, *, job_id, amount, requester_id):
        self.create_count += 1
        if self.create_returns_none:
            return None
        return {"job_id": job_id, "amount": amount}

    async def release_escrow(self, *, job_id, provider_id):
        self.release_count += 1

    async def refund_escrow(self, job_id, reason):
        self.refund_count += 1
        self.last_refund_reason = reason

    def get_escrow(self, job_id):
        return None


class _FakeStreamingExecutor:
    """Implements ``execute_streaming`` only. Programmable to either
    succeed (yield N tokens + InferenceResult.success) or fail at
    a chosen point."""

    def __init__(
        self,
        *,
        token_deltas: Optional[List[str]] = None,
        terminal_failure_message: Optional[str] = None,
    ):
        self.token_deltas = token_deltas or ["hello", " ", "world"]
        self.terminal_failure_message = terminal_failure_message
        self.calls = 0

    async def execute_streaming(
        self, request: InferenceRequest,
    ) -> AsyncIterator[Union[InferenceTokenEvent, InferenceResult]]:
        self.calls += 1
        if self.terminal_failure_message is not None:
            # Pre-execute failure path — sole InferenceResult.failure.
            yield InferenceResult.failure(
                request.request_id, self.terminal_failure_message,
            )
            return
        last = len(self.token_deltas) - 1
        for i, delta in enumerate(self.token_deltas):
            yield InferenceTokenEvent(
                sequence_index=i,
                text_delta=delta,
                finish_reason="stop" if i == last else None,
            )
        joined = "".join(self.token_deltas)
        yield InferenceResult(
            request_id=request.request_id,
            success=True,
            output=joined,
            receipt=_make_receipt(
                job_id=f"parallax-stream-job-{request.request_id}",
            ),
        )


class _FakeNode:
    def __init__(
        self,
        *,
        executor: Optional[_FakeStreamingExecutor] = None,
        escrow: Optional[_FakeEscrow] = None,
        privacy_budget: Optional[Any] = None,
    ):
        self.identity = _FakeIdentity()
        self.inference_executor = executor or _FakeStreamingExecutor()
        self._payment_escrow = escrow
        self.privacy_budget = privacy_budget


def _parse_sse_events(body: bytes) -> List[tuple]:
    """Minimal SSE parser for tests. Returns [(event_type, data_dict)]
    in the order received."""
    events: List[tuple] = []
    current_event = "message"
    current_data: List[str] = []
    for line in body.decode("utf-8").split("\n"):
        if line == "":
            if current_data:
                payload = "\n".join(current_data)
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    parsed = payload
                events.append((current_event, parsed))
                current_data = []
                current_event = "message"
        elif line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data.append(line[len("data: "):])
    return events


def _make_test_client(
    *,
    executor: Optional[_FakeStreamingExecutor] = None,
    escrow: Optional[_FakeEscrow] = None,
) -> TestClient:
    node = _FakeNode(executor=executor, escrow=escrow)
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


class TestStreamEndpointHappyPath:
    def test_emits_token_events_then_terminal_result(self):
        executor = _FakeStreamingExecutor(
            token_deltas=["hello", " ", "world"],
        )
        client = _make_test_client(executor=executor)
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "anything",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
                "privacy_tier": "none",
                "content_tier": "A",
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(
                "text/event-stream"
            )
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)
        # Three token events + one result event.
        assert [e[0] for e in events] == [
            "token", "token", "token", "result",
        ]
        # Joined deltas equal the result's output.
        deltas = [e[1]["text_delta"] for e in events if e[0] == "token"]
        result_event = [e[1] for e in events if e[0] == "result"][0]
        assert "".join(deltas) == result_event["output"] == "hello world"
        # Last token's finish_reason is "stop".
        assert events[-2][1]["finish_reason"] == "stop"

    def test_buffering_disabled_via_headers(self):
        client = _make_test_client()
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["x-accel-buffering"] == "no"

    def test_escrow_settled_on_success(self):
        escrow = _FakeEscrow()
        executor = _FakeStreamingExecutor(token_deltas=["ok"])
        client = _make_test_client(executor=executor, escrow=escrow)
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            list(response.iter_bytes())
        assert escrow.create_count == 1
        assert escrow.release_count == 1
        assert escrow.refund_count == 0


class TestStreamEndpointFailurePaths:
    def test_pre_execute_failure_yields_error_event_and_refunds(self):
        escrow = _FakeEscrow()
        executor = _FakeStreamingExecutor(
            terminal_failure_message="Unknown model_id: bogus",
        )
        client = _make_test_client(executor=executor, escrow=escrow)
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)
        # NO token events; sole error event.
        assert [e[0] for e in events] == ["error"]
        assert "Unknown model_id" in events[0][1]["error"]
        # Escrow refunded.
        assert escrow.refund_count == 1
        assert escrow.release_count == 0

    def test_executor_exception_yields_error_event_and_refunds(self):
        escrow = _FakeEscrow()

        class _Crashing:
            async def execute_streaming(self, request):
                raise RuntimeError("executor crashed mid-pipeline")
                yield  # unreachable; needed for AsyncIterator conformance

        node = _FakeNode(executor=_Crashing(), escrow=escrow)
        app = create_api_app(node, enable_security=False)
        client = TestClient(app)
        with client.stream(
            "POST", "/compute/inference/stream",
            json={
                "prompt": "x",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        ) as response:
            body = b"".join(response.iter_bytes())
        events = _parse_sse_events(body)
        assert [e[0] for e in events] == ["error"]
        assert "executor crashed" in events[0][1]["error"]
        assert escrow.refund_count == 1


class TestStreamEndpointInputValidation:
    def test_missing_prompt_returns_400(self):
        client = _make_test_client()
        response = client.post(
            "/compute/inference/stream",
            json={"model_id": "mock-llama-3-8b", "budget_ftns": 1.0},
        )
        assert response.status_code == 400
        assert "prompt" in response.json()["detail"].lower()

    def test_missing_model_id_returns_400(self):
        client = _make_test_client()
        response = client.post(
            "/compute/inference/stream",
            json={"prompt": "hi", "budget_ftns": 1.0},
        )
        assert response.status_code == 400
        assert "model_id" in response.json()["detail"].lower()

    def test_zero_budget_returns_422(self):
        """Sprint 155 — semantic validation failure → 422 (was 400)."""
        client = _make_test_client()
        response = client.post(
            "/compute/inference/stream",
            json={
                "prompt": "hi",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 0,
            },
        )
        assert response.status_code == 422
        assert "budget" in response.json()["detail"].lower()


class TestStreamEndpointOperatorMisconfig:
    def test_no_executor_returns_503(self):
        node = _FakeNode()
        node.inference_executor = None
        app = create_api_app(node, enable_security=False)
        client = TestClient(app)
        response = client.post(
            "/compute/inference/stream",
            json={
                "prompt": "x",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        )
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()

    def test_unary_only_executor_returns_503(self):
        # Old TensorParallelInferenceExecutor doesn't have
        # execute_streaming — the streaming endpoint should reject
        # with 503 (operator misconfig) rather than crash.
        class _UnaryOnly:
            async def execute(self, request):
                pass

        node = _FakeNode()
        node.inference_executor = _UnaryOnly()
        app = create_api_app(node, enable_security=False)
        client = TestClient(app)
        response = client.post(
            "/compute/inference/stream",
            json={
                "prompt": "x",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
            },
        )
        assert response.status_code == 503
        assert "streaming" in response.json()["detail"].lower()
