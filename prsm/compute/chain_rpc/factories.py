"""Phase 3.x.7 Task 6 — production-wiring factories for chain-RPC.

One-call construction helpers so production callers (the orchestrator,
the operator's node bootstrapping path) don't need full-submodule
imports + deep parameter knowledge to wire an ``RpcChainExecutor``
into a ``ParallaxScheduledExecutor``.

Two factories:

  ``make_rpc_chain_executor(...)`` — RpcChainExecutor with sensible
  defaults. Production callers MUST override the prompt_encoder and
  output_decoder for real tokenization; the UTF-8 byte-passthrough
  defaults are sufficient for tests + integration but won't work
  with real LLMs.

  ``make_layer_stage_server(...)`` — LayerStageServer wired up with
  the canonical Phase 2 Ring 8 + 3.x.2 + 3.x.3 dependencies. Reduces
  the boilerplate at node-startup to a single call.

Default tokenizer adapter (``utf8_prompt_encoder`` /
``utf8_output_decoder``):
  - Encodes the prompt's UTF-8 bytes as an int32 numpy array.
  - Decodes by reversing the operation, stripping zero padding.
  - Round-trip safe for ASCII; UTF-8 boundaries preserved as long
    as the chain doesn't reorder bytes mid-tensor.
  - Production callers replace this with the model's real tokenizer
    + de-tokenizer (typically wraps the HuggingFace ``AutoTokenizer``
    bound to the model's vocab).

The factory makes the SDK-style "construct with one call" surface the
design plan §4 Task 6 acceptance criterion calls for.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from prsm.compute.chain_rpc.client import (
    AddressResolver,
    OutputDecoder,
    PromptEncoder,
    RpcChainExecutor,
    SendMessage,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceRunner,
    LayerStageServer,
)
from prsm.node.identity import NodeIdentity


# ──────────────────────────────────────────────────────────────────────────
# Default tokenizer adapter (UTF-8 byte-passthrough)
# ──────────────────────────────────────────────────────────────────────────


def utf8_prompt_encoder(prompt: str) -> np.ndarray:
    """Encode a prompt's UTF-8 bytes as a 1-D int32 numpy array.

    Pads to a 4-byte boundary with zeros so the byte buffer is a
    clean int32 view. Round-trips with ``utf8_output_decoder`` for
    ASCII + UTF-8 inputs.

    PRODUCTION USE WARNING: this is a transparent byte-passthrough
    suitable for tests and integration scenarios where the chain's
    layer runners just shuffle activations through. Real LLM
    inference requires tokenizing the prompt into the model's vocab
    space + an embedding lookup; production callers replace this
    with their model's tokenizer adapter.
    """
    raw = prompt.encode("utf-8")
    pad = (4 - len(raw) % 4) % 4
    raw = raw + b"\x00" * pad
    return np.frombuffer(raw, dtype=np.int32).copy()


def utf8_output_decoder(arr: np.ndarray) -> str:
    """Inverse of ``utf8_prompt_encoder``. Strips trailing zero bytes
    and decodes UTF-8 with replacement on invalid sequences (so a
    corrupted tail doesn't crash the executor)."""
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32, copy=False)
    return arr.tobytes().rstrip(b"\x00").decode("utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────
# RpcChainExecutor factory
# ──────────────────────────────────────────────────────────────────────────


def make_rpc_chain_executor(
    *,
    settler_identity: NodeIdentity,
    send_message: SendMessage,
    anchor: Any,
    prompt_encoder: Optional[PromptEncoder] = None,
    output_decoder: Optional[OutputDecoder] = None,
    address_resolver: Optional[AddressResolver] = None,
    default_deadline_seconds: float = 30.0,
) -> RpcChainExecutor:
    """Build an ``RpcChainExecutor`` with production-friendly defaults.

    Required:
      settler_identity   The settling node's NodeIdentity. Mints
                         per-stage HandoffTokens. Same identity that
                         signs the final InferenceReceipt at the
                         ParallaxScheduledExecutor layer.
      send_message       Phase 6 transport callable: (address, bytes)
                         → bytes.
      anchor             Phase 3.x.3 anchor for verifying each stage's
                         response signature.

    Optional (defaults are test-friendly, not production-ready):
      prompt_encoder     Default: ``utf8_prompt_encoder``. Replace
                         with the model's tokenizer adapter for real
                         LLM inference.
      output_decoder     Default: ``utf8_output_decoder``. Replace
                         with the model's de-tokenizer for real LLM
                         inference.
      address_resolver   Default: identity (node_id == address).
                         Suitable when the Phase 6 peer registry uses
                         node_ids directly.
      default_deadline_seconds  Default 30s.
    """
    return RpcChainExecutor(
        settler_identity=settler_identity,
        send_message=send_message,
        anchor=anchor,
        prompt_encoder=prompt_encoder or utf8_prompt_encoder,
        output_decoder=output_decoder or utf8_output_decoder,
        address_resolver=address_resolver,
        default_deadline_seconds=default_deadline_seconds,
    )


# ──────────────────────────────────────────────────────────────────────────
# LayerStageServer factory
# ──────────────────────────────────────────────────────────────────────────


def make_layer_stage_server(
    *,
    identity: NodeIdentity,
    registry: Any,
    runner: LayerSliceRunner,
    tee_runtime: Any,
    anchor: Any,
    clock: Optional[Callable[[], float]] = None,
) -> LayerStageServer:
    """Build a ``LayerStageServer`` for a node hosting one or more
    chain stages.

    Required:
      identity      This node's NodeIdentity. Signs RunLayerSliceResponse
                    payloads; the same identity callers register on the
                    Phase 3.x.3 anchor.
      registry      Phase 3.x.2 ModelRegistry (typically
                    FilesystemModelRegistry).
      runner        LayerSliceRunner — the actual layer-execution
                    adapter. Production wraps Phase 2 Ring 8's
                    TensorParallelExecutor; a tiny wrapper module
                    will land in a future task once the
                    TensorParallelExecutor.run_layer_range signature
                    is finalized.
      tee_runtime   Phase 2 Ring 8 TEERuntime (SoftwareTEERuntime
                    for dev; SgxTEERuntime / etc. for production).
      anchor        Phase 3.x.3 anchor for verifying upstream tokens.

    Optional:
      clock         Defaults to ``time.time``. Tests override.
    """
    kwargs = dict(
        identity=identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
    )
    if clock is not None:
        kwargs["clock"] = clock
    return LayerStageServer(**kwargs)
