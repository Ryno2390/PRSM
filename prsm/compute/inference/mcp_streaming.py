"""Phase 3.x.8 Task 5 — MCP streaming-token adapter for the
``prsm_inference`` tool surface.

Bridges ``RpcChainExecutor.execute_chain_streaming`` to the MCP
progress-event surface (``emit_progress`` callback) added in
Phase 3.x.1 Task 8. On streaming-capable MCP clients, the adapter
emits one progress event per ``StreamToken`` and finalizes with a
signed ``InferenceReceipt`` whose ``streamed_output`` flag is True
(downgrade-resistant per Phase 3.x.8 Task 4). Non-streaming
clients (where ``emit_progress`` is None) get the same final
buffered text + receipt without intermediate events.

The adapter is intentionally a pure helper — it does NOT mutate
the MCP server's HTTP-layer code path. Production wiring of the
streaming HTTP endpoint (``POST /compute/inference/stream`` or
similar SSE-style surface) is a Phase 3.x.8.x follow-up; what
this module ships TODAY is the executor → MCP-progress bridge so
the rest of the surface can be tested + audited end-to-end.

Usage from a streaming-capable MCP handler:

    result = await stream_inference_to_mcp(
        executor=rpc_chain_executor,
        request=inference_request,
        chain=allocated_chain,
        cost_ftns=Decimal("0.5"),
        identity=node_identity,
        emit_progress=mcp_emit_progress,
    )
    # Returns MCPStreamingResult with text, signed receipt
    # (streamed_output=True), per-token count, finish_reason.

For a non-streaming caller, pass ``emit_progress=None`` — the
adapter still drives the streaming executor under the hood (so
the server-side runner path is exercised) and returns the same
result shape, but no progress events are emitted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Awaitable, Callable, List, Optional

from prsm.compute.inference.models import InferenceReceipt, InferenceRequest
from prsm.compute.inference.parallax_executor import ChainExecutionResult
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.parallax_scheduling.prsm_request_router import GPUChain
from prsm.node.identity import NodeIdentity


__all__ = [
    "MCPStreamingResult",
    "ProgressEmitter",
    "stream_inference_to_mcp",
]


logger = logging.getLogger(__name__)


# Mirrors the type alias in ``prsm.mcp_server`` (kept duplicated to
# avoid a hard import cycle: mcp_server is a top-level surface and
# this module is in the inference package). Signature is intentionally
# identical so production wiring in mcp_server.py can pass its
# ``emit_progress`` callable directly.
ProgressEmitter = Callable[[str, float, Optional[float]], Awaitable[None]]
"""``async (message, progress, total | None) -> None``.

The progress-event callback the MCP server hands to streaming-aware
tool handlers. ``progress`` is a count (here: 1-indexed token
count); ``total`` is the expected upper bound (None for
autoregressive output where length is variable)."""


@dataclass(frozen=True)
class MCPStreamingResult:
    """Aggregate result returned to the MCP tool handler after a
    streaming inference completes.

    Mirrors the shape an HTTP-layer streaming endpoint would expose
    once Phase 3.x.8.x lands the wire-protocol side. The handler
    formats this into the MCP-facing TextContent response (final
    receipt + cost footer) — same surface as the unary path.
    """

    output: str
    """Full joined output text, equal to the concatenation of every
    yielded ``StreamToken.text_delta`` and bit-identical to what
    ``execute_chain.output`` would have returned for the same
    chain + activation."""

    receipt: InferenceReceipt
    """Signed receipt with ``streamed_output=True``. The signing
    payload commits to the joined output bytes via
    ``output_hash`` AND to the streamed-mode flag via the
    Phase 3.x.8 Task 4 conditional encoding — flipping either
    invalidates the settler signature."""

    token_count: int
    """Number of ``StreamToken``s emitted (= number of progress
    events sent to ``emit_progress`` when one was provided)."""

    finish_reason: Optional[str]
    """Reason the tail stage stopped decoding. ``"stop"`` on a
    normal complete-output stream; ``"max_tokens"``,
    ``"cancelled"``, ``"error"`` on early termination. None only
    if the upstream protocol violated its own contract — surfaced
    so callers can detect that."""


async def stream_inference_to_mcp(
    *,
    executor: object,
    request: InferenceRequest,
    chain: GPUChain,
    cost_ftns: Decimal,
    identity: NodeIdentity,
    emit_progress: Optional[ProgressEmitter] = None,
    job_id: Optional[str] = None,
) -> MCPStreamingResult:
    """Drive ``executor.execute_chain_streaming`` and bridge it to
    the MCP progress-event surface.

    Args:
      executor: an ``RpcChainExecutor`` (or any object with an
        ``execute_chain_streaming(*, request, chain) ->
        Iterator[Union[StreamToken, ChainExecutionResult]]``
        method). The streaming generator is fully consumed inside
        this call; on completion the receipt is built + signed.
      request: the ``InferenceRequest`` to dispatch.
      chain: the ``GPUChain`` from the router/allocator.
      cost_ftns: the FTNS cost for this inference (computed by the
        ``ParallaxScheduledExecutor`` via ``estimate_cost``).
        Embedded in the signed receipt so callers can reconcile.
      identity: the settling node's ``NodeIdentity``. Signs the
        receipt under the same key used by the unary path.
      emit_progress: optional MCP progress callback. When provided,
        each ``StreamToken`` produces one
        ``emit_progress(text_delta, sequence_index + 1, None)``
        call. When None, the executor's streaming generator is
        still consumed end-to-end (so server-side runner is
        exercised + receipt is built) but no events are emitted —
        used for non-streaming MCP clients on a server that's
        otherwise streaming-capable.
      job_id: optional override for the receipt's ``job_id`` field.
        Defaults to a fresh ``"parallax-stream-job-<hex>"`` to
        match the unary path's ``"parallax-job-<hex>"`` convention
        (different prefix so log-level filtering can distinguish
        streamed vs. unary jobs at audit time).

    Returns:
      ``MCPStreamingResult`` carrying the joined output, signed
      receipt (with ``streamed_output=True``), per-token count,
      and finish_reason of the terminal token.

    Raises:
      ``ChainExecutionError`` (from the executor) for any failure
      mode of the streaming pipeline — caller is responsible for
      mapping these to MCP-friendly error responses (same pattern
      as the unary path).
      ``RuntimeError`` if the executor's generator yields neither
      tokens nor a terminal ``ChainExecutionResult`` (protocol
      violation by the executor).
    """
    # Lazy import inside the function body to avoid a hard import
    # cycle: client.py imports parallax_executor (for
    # ChainExecutionResult); this module imports both. Keeping
    # StreamToken behind a function-scope import means
    # ``mcp_streaming`` can be loaded by callers that don't have
    # the full chain_rpc surface available (e.g. lightweight MCP
    # frontends).
    from prsm.compute.chain_rpc.client import StreamToken

    final_outcome: Optional[ChainExecutionResult] = None
    text_deltas: List[str] = []
    token_count = 0
    finish_reason: Optional[str] = None

    for item in executor.execute_chain_streaming(  # type: ignore[attr-defined]
        request=request,
        chain=chain,
    ):
        if isinstance(item, StreamToken):
            token_count += 1
            text_deltas.append(item.text_delta)
            # Capture finish_reason from the terminal token (only
            # the LAST token has a non-None value per the 3.x.8
            # wire-format contract).
            if item.finish_reason is not None:
                finish_reason = item.finish_reason
            if emit_progress is not None:
                # Forward the incremental text to the MCP client.
                # ``message`` carries the delta text; ``progress``
                # is the 1-indexed token count; ``total`` is None
                # because autoregressive decode has variable
                # length.
                await emit_progress(
                    item.text_delta,
                    float(token_count),
                    None,
                )
        elif isinstance(item, ChainExecutionResult):
            # Terminal yield from the executor — capture and break.
            # Defense-in-depth: if the executor mistakenly yields
            # more after the result, the iteration's natural
            # exhaustion handles it.
            final_outcome = item
            break
        else:
            raise RuntimeError(
                f"executor.execute_chain_streaming yielded unexpected "
                f"type {type(item).__name__}; expected StreamToken or "
                f"ChainExecutionResult"
            )

    if final_outcome is None:
        raise RuntimeError(
            "executor.execute_chain_streaming exhausted without "
            "yielding a ChainExecutionResult (protocol violation)"
        )

    joined_output = "".join(text_deltas)
    if joined_output != final_outcome.output:
        # Defense-in-depth: the executor's own streaming-tail
        # dispatch already cross-checks this (joined deltas match
        # the signed activation_blob). Re-checking at this layer
        # catches any future divergence introduced by a refactor.
        raise RuntimeError(
            "stream-aggregated output diverges from "
            "ChainExecutionResult.output (executor invariant violated)"
        )

    receipt = _build_streamed_signed_receipt(
        request=request,
        cost=cost_ftns,
        outcome=final_outcome,
        identity=identity,
        job_id=job_id,
    )

    return MCPStreamingResult(
        output=joined_output,
        receipt=receipt,
        token_count=token_count,
        finish_reason=finish_reason,
    )


def _build_streamed_signed_receipt(
    *,
    request: InferenceRequest,
    cost: Decimal,
    outcome: ChainExecutionResult,
    identity: NodeIdentity,
    job_id: Optional[str],
) -> InferenceReceipt:
    """Mirror of ``ParallaxScheduledExecutor._build_signed_receipt``
    with ``streamed_output=True``. Kept here (rather than mutating
    the unary builder) so the unary path stays untouched —
    Phase 3.x.6's existing tests + audit notes don't shift.
    """
    output_hash = hashlib.sha256(outcome.output.encode("utf-8")).digest()
    unsigned = InferenceReceipt(
        job_id=job_id or f"parallax-stream-job-{uuid.uuid4().hex[:12]}",
        request_id=request.request_id,
        model_id=request.model_id,
        content_tier=request.content_tier,
        privacy_tier=request.privacy_tier,
        epsilon_spent=outcome.epsilon_spent,
        tee_type=outcome.tee_type,
        tee_attestation=outcome.tee_attestation,
        output_hash=output_hash,
        duration_seconds=outcome.duration_seconds,
        cost_ftns=cost,
        streamed_output=True,
    )
    return sign_receipt(unsigned, identity)
