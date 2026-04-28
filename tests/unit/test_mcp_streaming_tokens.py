"""Phase 3.x.8 Task 5 — unit tests for the MCP streaming-token
adapter (``prsm.compute.inference.mcp_streaming``).

Distinct from ``test_mcp_streaming.py`` (Phase 3.x.1 Task 8) which
covers the existing progress-event surface for non-token tools."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterator, List, Optional, Tuple, Union

import pytest

from prsm.compute.chain_rpc.client import StreamToken
from prsm.compute.inference.mcp_streaming import (
    MCPStreamingResult,
    stream_inference_to_mcp,
)
from prsm.compute.inference.models import (
    ContentTier,
    InferenceRequest,
)
from prsm.compute.inference.parallax_executor import ChainExecutionResult
from prsm.compute.inference.receipt import verify_receipt
from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeStreamingExecutor:
    """Implements only ``execute_chain_streaming`` — emits a fixed
    sequence of StreamTokens then a ChainExecutionResult."""
    deltas: List[str]
    duration_seconds: float = 0.05
    epsilon_spent: float = 0.0
    tee_type: TEEType = TEEType.SOFTWARE
    tee_attestation: bytes = b"\x01" * 32
    finish_reason: str = "stop"
    omit_terminal: bool = False
    yield_wrong_type: bool = False
    diverge_output: bool = False

    def execute_chain_streaming(
        self,
        *,
        request: InferenceRequest,
        chain: GPUChain,
    ) -> Iterator[Union[StreamToken, ChainExecutionResult]]:
        last = len(self.deltas) - 1
        for i, delta in enumerate(self.deltas):
            yield StreamToken(
                sequence_index=i,
                text_delta=delta,
                token_id=None,
                finish_reason=self.finish_reason if i == last else None,
            )
        if self.yield_wrong_type:
            yield "not-a-result"  # type: ignore[misc]
            return
        if self.omit_terminal:
            return
        joined = "".join(self.deltas)
        if self.diverge_output:
            joined = joined + "DIVERGED"
        yield ChainExecutionResult(
            output=joined,
            duration_seconds=self.duration_seconds,
            tee_attestation=self.tee_attestation,
            tee_type=self.tee_type,
            epsilon_spent=self.epsilon_spent,
        )


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def settler_identity():
    return generate_node_identity(display_name="settler-stream")


@pytest.fixture
def request_obj():
    return InferenceRequest(
        prompt="hello",
        model_id="mock-llama-3-8b",
        budget_ftns=Decimal("1.0"),
        privacy_tier=PrivacyLevel.STANDARD,
        content_tier=ContentTier.A,
        request_id="req-stream-1",
    )


@pytest.fixture
def chain_obj():
    return GPUChain(
        request_id="req-stream-1",
        region="us-east",
        stages=("alice",),
        layer_ranges=((0, 4),),
        total_latency_ms=10.0,
        stale_profile_count=0,
    )


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


class TestStreamingHappyPath:
    @pytest.mark.asyncio
    async def test_emits_one_progress_event_per_token(
        self, settler_identity, request_obj, chain_obj,
    ):
        executor = _FakeStreamingExecutor(deltas=["hello", " ", "world"])
        events: List[Tuple[str, float, Optional[float]]] = []

        async def emit(message: str, progress: float,
                       total: Optional[float]) -> None:
            events.append((message, progress, total))

        result = await stream_inference_to_mcp(
            executor=executor,
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
            emit_progress=emit,
        )
        # One event per StreamToken; counts are 1-indexed; total None.
        assert len(events) == 3
        assert events[0] == ("hello", 1.0, None)
        assert events[1] == (" ", 2.0, None)
        assert events[2] == ("world", 3.0, None)
        assert result.token_count == 3
        assert result.output == "hello world"

    @pytest.mark.asyncio
    async def test_no_emit_progress_still_completes(
        self, settler_identity, request_obj, chain_obj,
    ):
        # Non-streaming MCP client (emit_progress=None) still gets
        # the buffered output + receipt — the server-side runner
        # path is exercised either way.
        executor = _FakeStreamingExecutor(deltas=["a", "b", "c"])
        result = await stream_inference_to_mcp(
            executor=executor,
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
            emit_progress=None,
        )
        assert result.output == "abc"
        assert result.token_count == 3
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_signed_receipt_marks_streamed_output_true(
        self, settler_identity, request_obj, chain_obj,
    ):
        executor = _FakeStreamingExecutor(deltas=["hello world"])
        result = await stream_inference_to_mcp(
            executor=executor,
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
        )
        # Phase 3.x.8 Task 4 invariant: streamed receipts have
        # streamed_output=True AND the flag is part of the signed
        # payload (downgrade-resistant).
        assert result.receipt.streamed_output is True
        assert verify_receipt(result.receipt, identity=settler_identity)

    @pytest.mark.asyncio
    async def test_receipt_output_hash_commits_to_joined_text(
        self, settler_identity, request_obj, chain_obj,
    ):
        # The receipt's output_hash MUST be sha256 of the joined
        # output text — so a relay can't claim a different output
        # was produced.
        import hashlib
        executor = _FakeStreamingExecutor(deltas=["foo ", "bar ", "baz"])
        result = await stream_inference_to_mcp(
            executor=executor,
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
        )
        expected_hash = hashlib.sha256(b"foo bar baz").digest()
        assert result.receipt.output_hash == expected_hash

    @pytest.mark.asyncio
    async def test_finish_reason_propagates_from_terminal_token(
        self, settler_identity, request_obj, chain_obj,
    ):
        for reason in ("stop", "max_tokens", "cancelled"):
            executor = _FakeStreamingExecutor(
                deltas=["partial"], finish_reason=reason,
            )
            result = await stream_inference_to_mcp(
                executor=executor,
                request=request_obj,
                chain=chain_obj,
                cost_ftns=Decimal("0.5"),
                identity=settler_identity,
            )
            assert result.finish_reason == reason

    @pytest.mark.asyncio
    async def test_explicit_job_id_propagates_to_receipt(
        self, settler_identity, request_obj, chain_obj,
    ):
        executor = _FakeStreamingExecutor(deltas=["x"])
        result = await stream_inference_to_mcp(
            executor=executor,
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
            job_id="custom-job-99",
        )
        assert result.receipt.job_id == "custom-job-99"

    @pytest.mark.asyncio
    async def test_default_job_id_uses_streaming_prefix(
        self, settler_identity, request_obj, chain_obj,
    ):
        # Distinguishes streamed-vs-unary at audit-log filtering time.
        executor = _FakeStreamingExecutor(deltas=["y"])
        result = await stream_inference_to_mcp(
            executor=executor,
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
        )
        assert result.receipt.job_id.startswith("parallax-stream-job-")


class TestStreamingErrors:
    @pytest.mark.asyncio
    async def test_executor_yields_wrong_type_raises_runtime_error(
        self, settler_identity, request_obj, chain_obj,
    ):
        executor = _FakeStreamingExecutor(
            deltas=["a"], yield_wrong_type=True,
        )
        with pytest.raises(RuntimeError, match="unexpected type"):
            await stream_inference_to_mcp(
                executor=executor,
                request=request_obj,
                chain=chain_obj,
                cost_ftns=Decimal("0.5"),
                identity=settler_identity,
            )

    @pytest.mark.asyncio
    async def test_executor_omits_terminal_result_raises(
        self, settler_identity, request_obj, chain_obj,
    ):
        executor = _FakeStreamingExecutor(
            deltas=["a", "b"], omit_terminal=True,
        )
        with pytest.raises(RuntimeError, match="exhausted without"):
            await stream_inference_to_mcp(
                executor=executor,
                request=request_obj,
                chain=chain_obj,
                cost_ftns=Decimal("0.5"),
                identity=settler_identity,
            )

    @pytest.mark.asyncio
    async def test_diverging_aggregate_output_raises(
        self, settler_identity, request_obj, chain_obj,
    ):
        # If the executor yields tokens that DON'T concatenate to
        # ChainExecutionResult.output, the adapter catches the
        # divergence (defense-in-depth on top of executor's own
        # joined-text invariant).
        executor = _FakeStreamingExecutor(
            deltas=["a", "b"], diverge_output=True,
        )
        with pytest.raises(RuntimeError, match="diverges from"):
            await stream_inference_to_mcp(
                executor=executor,
                request=request_obj,
                chain=chain_obj,
                cost_ftns=Decimal("0.5"),
                identity=settler_identity,
            )

    @pytest.mark.asyncio
    async def test_executor_exception_propagates(
        self, settler_identity, request_obj, chain_obj,
    ):
        # An executor that raises mid-stream propagates the exception
        # through the adapter — caller maps it to MCP-friendly error.
        class _Raises:
            def execute_chain_streaming(self, *, request, chain):
                yield StreamToken(
                    sequence_index=0, text_delta="hi", finish_reason=None,
                )
                raise RuntimeError("upstream chain failed")

        with pytest.raises(RuntimeError, match="upstream chain failed"):
            await stream_inference_to_mcp(
                executor=_Raises(),
                request=request_obj,
                chain=chain_obj,
                cost_ftns=Decimal("0.5"),
                identity=settler_identity,
            )


class TestStreamingTamperResistance:
    @pytest.mark.asyncio
    async def test_streamed_flag_tamper_invalidates_signature(
        self, settler_identity, request_obj, chain_obj,
    ):
        """End-to-end downgrade-resistance: a relay that flips the
        streamed_output flag on a receipt produced by this adapter
        MUST be caught by signature verification."""
        import dataclasses
        executor = _FakeStreamingExecutor(deltas=["test"])
        result = await stream_inference_to_mcp(
            executor=executor,
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
        )
        assert verify_receipt(result.receipt, identity=settler_identity)
        # Adversary tries to claim it was non-streamed.
        downgraded = dataclasses.replace(
            result.receipt, streamed_output=False,
        )
        assert not verify_receipt(downgraded, identity=settler_identity)


class TestEmptyStream:
    @pytest.mark.asyncio
    async def test_zero_token_stream_with_terminal_result(
        self, settler_identity, request_obj, chain_obj,
    ):
        # Edge case: executor yields ZERO tokens (e.g., empty
        # output) followed directly by a terminal
        # ChainExecutionResult. Adapter should handle gracefully.
        class _EmptyStream:
            def execute_chain_streaming(self, *, request, chain):
                yield ChainExecutionResult(
                    output="",
                    duration_seconds=0.0,
                    tee_attestation=b"\x00" * 32,
                    tee_type=TEEType.SOFTWARE,
                    epsilon_spent=0.0,
                )

        result = await stream_inference_to_mcp(
            executor=_EmptyStream(),
            request=request_obj,
            chain=chain_obj,
            cost_ftns=Decimal("0.5"),
            identity=settler_identity,
        )
        assert result.token_count == 0
        assert result.output == ""
        assert result.finish_reason is None
        # Receipt still signed + verifiable.
        assert verify_receipt(result.receipt, identity=settler_identity)
