"""Sprint 592 (Phase 2A) — chain-executor send-message adapter scaffolding.

Sprint 578 plumbed PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=rpc as a hook
for the real ``make_rpc_chain_executor`` factory. Phase 2 wiring
requires bridging the gap between:

- ``SendMessage = Callable[[str, bytes], bytes]`` (factory contract;
  SYNC; address → response bytes); see
  ``prsm/compute/chain_rpc/client.py``
- ``transport.send_to_peer(peer_id, P2PMessage) -> bool`` (async;
  awaits + does not return response bytes directly)

This module scaffolds the adapter contract. Phase 2A (sprint 592)
introduces the type signature + Protocol + a NotImplementedError
placeholder ``build_send_message_adapter()``. Phase 2B (sprint 593)
adds the address-resolver helper. Phase 2C (sprint 594) implements
the async-to-sync bridge using ``asyncio.run_coroutine_threadsafe``.
Phase 2D (sprint 595) wires it into ``_build_chain_executor``.

This staged approach keeps each sprint scoped + reviewable. The
scaffolding lets test code reference the eventual contract today.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable


# Re-export the canonical SendMessage signature from the factory's
# source of truth — adapters wired to make_rpc_chain_executor MUST
# conform to this Callable shape.
SendMessage = Callable[[str, bytes], bytes]


@runtime_checkable
class SendMessageAdapter(Protocol):
    """Phase 2 adapter contract.

    Implementations bridge between the daemon's async transport
    layer and the sync SendMessage contract required by
    ``make_rpc_chain_executor``. The adapter is responsible for:

    1. Resolving ``stage_address`` to a transport peer_id
       (delegates to a Phase-2B address resolver).
    2. Constructing a ``P2PMessage`` wrapping ``request_bytes``.
    3. Scheduling the async ``transport.send_to_peer`` on the
       daemon's running event loop (e.g., via
       ``asyncio.run_coroutine_threadsafe``).
    4. Blocking the calling thread until a response arrives or
       a deadline elapses; raising on transport failure.

    Phase 2A — this module. Protocol only; no implementation.
    """

    def __call__(self, stage_address: str, request_bytes: bytes) -> bytes:
        ...


class _Phase2AdapterNotReady(NotImplementedError):
    """Sprint 592 — raised by the scaffolding placeholder when
    callers try to USE the adapter before Phase 2C lands the real
    async-to-sync bridge.

    Distinct exception class so `_build_chain_executor`'s `rpc`
    branch can detect Phase 2 non-readiness cleanly + log the
    structured warning sprint 578 already established.
    """


def run_async_on_loop(
    loop: asyncio.AbstractEventLoop,
    coro: Coroutine[Any, Any, Any],
    timeout: float,
) -> Any:
    """Sprint 594 (Phase 2C) — async-to-sync bridge primitive.

    Schedules ``coro`` on a running event loop from a DIFFERENT
    thread + returns the result synchronously. Thin wrapper around
    ``asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)``.

    Thread-safety contract:
      - ``loop`` MUST be running on a different thread than the
        caller. Calling from the loop's own thread deadlocks
        because the loop cannot make progress while blocked in
        ``.result()``.

    Used by Phase 2D wiring (sprint 595+) to bridge the sync
    SendMessage contract over async transport calls. Exposed as a
    standalone helper so the threading primitive is unit-testable
    in isolation from the chain-executor protocol layer.

    Raises whatever the coroutine raises (passed through);
    ``concurrent.futures.TimeoutError`` on timeout expiry.
    """
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


class PeerNotFound(RuntimeError):
    """Sprint 593 (Phase 2B) — raised by the address resolver when
    a chain stage's node_id isn't currently in ``transport.peers``.

    Operators triaging chain-executor failures see the missing
    node_id in the exception message. Sprint-595 wiring will catch
    this at executor-startup time + report it via the trust-stack
    observability surfaces (sprint 579 CLI + 582 /health/detailed)
    so the failure is visible BEFORE a real inference request hits
    the dispatch.
    """


def build_address_resolver(node: Any) -> Callable[[str], str]:
    """Sprint 593 (Phase 2B) — map chain stage node_id → transport
    peer.address by looking up ``node.transport.peers[node_id]``.

    Raises ``PeerNotFound`` when the node isn't currently in
    transport.peers. The chain executor's dispatch loop catches
    this to surface "stage N unreachable" cleanly rather than
    propagating a KeyError up.

    Returns the canonical ``AddressResolver = Callable[[str], str]``
    shape from ``prsm/compute/chain_rpc/client.py``.
    """
    transport = node.transport

    def _resolve(node_id: str) -> str:
        peer = transport.peers.get(node_id)
        if peer is None:
            raise PeerNotFound(
                f"chain stage node_id {node_id!r} not currently "
                f"in transport.peers; cannot dispatch. Likely "
                f"causes: peer dropped connection, auto-dial sweep "
                f"hasn't reached this peer yet (sprint 573), or "
                f"peer never registered against the same bootstrap."
            )
        return peer.address

    return _resolve


# Sprint 596 — chain-executor wire-protocol identifiers.
# Chain-executor request messages set payload[CHAIN_REQ_KEY] = request_id;
# response messages set payload[CHAIN_RESP_KEY] = request_id + payload
# bytes (base64-encoded in the JSON envelope). Sprint 597 wires the
# response handler that reads CHAIN_RESP_KEY and resolves the pending
# Future identified by request_id.
CHAIN_MSG_TYPE = "chain_executor_rpc"
CHAIN_REQ_KEY = "chain_req_id"
CHAIN_RESP_KEY = "chain_resp_id"
CHAIN_PAYLOAD_KEY = "chain_payload_b64"
# Sprint 601 (Phase 2E-1) — error indicator in server-side response.
# Lets the requester distinguish "stage handler returned an error"
# from "no stage handler exists yet" (Phase 2E-1 ships scaffolding;
# Phase 2E-2+ replaces this with real stage execution).
CHAIN_ERROR_KEY = "chain_error"


class StageExecutionError(RuntimeError):
    """Sprint 602 (Phase 2E-2) — raised by StageExecutor.execute()
    when forward execution of a chain stage fails.

    Subclasses RuntimeError so generic operator error-handling
    paths catch it via isinstance(exc, RuntimeError) naturally.
    """


@runtime_checkable
class StageExecutor(Protocol):
    """Sprint 602 (Phase 2E-2) — server-side stage executor contract.

    Receives a chain stage's serialized request bytes (typically
    activation bytes from the previous stage) and returns the
    response bytes (output activation passed to the next stage).

    Phase 2E-4 (sprint 604) wires this Protocol into
    handle_chain_executor_request — request bytes flow in, response
    bytes flow back out via the existing wire format.

    Implementations may raise ``StageExecutionError`` on failure;
    the request handler converts to a CHAIN_ERROR_KEY response.
    """

    async def execute(self, request_bytes: bytes) -> bytes:
        ...


def build_stub_stage_executor() -> StageExecutor:
    """Sprint 602 (Phase 2E-2) — placeholder StageExecutor that
    raises StageExecutionError with an actionable message.

    Phase 2E-3 (sprint 603) ships a real implementation that
    decodes activation bytes + forwards through a model layer.
    Phase 2E-4 (sprint 604) wires this into the request handler
    so it's actually called.
    """

    class _StubStageExecutor:
        async def execute(self, request_bytes: bytes) -> bytes:
            raise StageExecutionError(
                "Sprint 602 Phase 2E-2 stub: StageExecutor.execute() "
                "is not yet implemented. Phase 2E-3 (sprint 603) "
                "will ship the real chain-stage forward execution. "
                "Until then, this node ACKs incoming requests via "
                "the sprint-601 request handler but cannot actually "
                "produce next-stage activations."
            )

    return _StubStageExecutor()


class LayerStageServerStageExecutor:
    """Sprint 606 (Phase 2F-1) — wraps a chain_rpc LayerStageServer
    as a StageExecutor.

    The existing
    ``prsm.compute.chain_rpc.server.LayerStageServer.handle(bytes) -> bytes``
    already has the exact signature our StageExecutor Protocol
    expects (sync, bytes-in/bytes-out, never raises by design).
    This adapter:
      - Provides the async ``execute()`` Protocol method
      - Dispatches the (sync, potentially CPU-heavy) handle()
        call via ``loop.run_in_executor`` so the event loop
        thread isn't blocked
      - Wraps any unexpected exception in ``StageExecutionError``
        (defense-in-depth — handle() shouldn't raise but if it
        does, the request handler surfaces it cleanly)

    Phase 2F-2 (sprint 607+) ships the factory that constructs
    the underlying LayerStageServer with the operator's identity +
    registry + runner + tee_runtime + anchor.
    """

    def __init__(self, *, server: Any) -> None:
        if server is None or not hasattr(server, "handle"):
            raise ValueError(
                "LayerStageServerStageExecutor requires a server "
                "with a .handle(bytes) -> bytes method"
            )
        self._server = server

    async def execute(self, request_bytes: bytes) -> bytes:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._server.handle, request_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"LayerStageServer.handle raised {type(exc).__name__}: "
                f"{exc}"
            ) from exc


def _resolve_hf_layers(hf_model: Any) -> Any:
    """Sprint 612 (Phase 2F-5c) — polymorphic HF layer extraction.

    Try known causal-LM layer-list paths in order:
      model.model.layers      LLaMA / Mistral / Qwen
      model.transformer.h     GPT-2 / Falcon / GPT-J
      model.gpt_neox.layers   GPT-NeoX / Pythia

    Returns the first path that resolves. Raises StageExecutionError
    listing all attempted paths if none match — operator triages.

    The "first that resolves" rule means LLaMA-style models with a
    populated ``model.model.layers`` always take that path even if
    other attributes are coincidentally present.
    """
    attempted = []
    # LLaMA / Mistral / Qwen style
    attempted.append(".model.layers")
    sub = getattr(hf_model, "model", None)
    if sub is not None:
        layers = getattr(sub, "layers", None)
        if layers is not None:
            return layers
    # GPT-2 / Falcon / GPT-J style
    attempted.append(".transformer.h")
    sub = getattr(hf_model, "transformer", None)
    if sub is not None:
        layers = getattr(sub, "h", None)
        if layers is not None:
            return layers
    # GPT-NeoX / Pythia style
    attempted.append(".gpt_neox.layers")
    sub = getattr(hf_model, "gpt_neox", None)
    if sub is not None:
        layers = getattr(sub, "layers", None)
        if layers is not None:
            return layers
    raise StageExecutionError(
        f"HuggingFaceLayerSliceRunner: model does not expose any "
        f"known layer-list path. Attempted: {', '.join(attempted)}. "
        f"Add support for this architecture in _resolve_hf_layers."
    )


class HuggingFaceLayerSliceRunner:
    """Sprint 610 (Phase 2F-5a) — first real-model LayerSliceRunner
    skeleton.

    Constructor accepts a HuggingFace ``model_id`` (e.g.,
    "meta-llama/Llama-3.2-1B") + target device (default "cpu" —
    safe on GPU-less machines).

    Phase 2F-5a is interface only. Phase 2F-5b ships:
      - Lazy ``transformers`` import + model loading from HF Hub
        or local checkpoint
      - Layer extraction + forward pass on the activation tensor
      - Layer-range validation against the model's actual depth
      - TEE attestation hook (Phase 2F-5c)

    Conforms to ``LayerSliceRunner`` Protocol. The ``model`` arg
    in ``run_layer_range`` (which the LayerStageServer passes from
    ``registry.get(model_id)``) is IGNORED by the HF runner — the
    HF runner owns its own model loading via the ``transformers``
    library. ShardedModel-based registry is for PRSM's own
    sharding scheme; HF runner is a parallel path.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
    ) -> None:
        if not model_id or not isinstance(model_id, str):
            raise ValueError(
                "HuggingFaceLayerSliceRunner requires model_id "
                "(HuggingFace model identifier, e.g., "
                "'meta-llama/Llama-3.2-1B')"
            )
        self.model_id = model_id
        self.device = device or "cpu"
        # Lazy-cached on first run_layer_range call.
        self._hf_model: Any = None

    def _ensure_model_loaded(self) -> Any:
        """Sprint 611 (Phase 2F-5b) — lazy model load + cache."""
        if self._hf_model is not None:
            return self._hf_model
        # Lazy imports keep transformers off the import path for
        # daemons not using the HF runner.
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner requires `transformers` "
                f"+ `torch` packages installed: {exc}. Install via "
                f"`pip install transformers torch`."
            ) from exc
        try:
            from transformers import AutoModelForCausalLM
            import torch as _torch
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=_torch.float32,
            )
            model = model.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner failed to load "
                f"{self.model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        self._hf_model = model
        return model

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Any,
        activation: Any,
        privacy_tier: Any,
        is_final_stage: bool,
    ) -> Any:
        """Sprint 611 (Phase 2F-5b) — real HF model forward pass.

        Workflow:
          1. Lazy-load HF model (cached after first call)
          2. Extract layer list (LLaMA-style ``.model.layers``)
          3. Convert numpy activation → torch [B, S, H]
          4. Build position_ids = arange(S)
          5. Forward through layers[start:end]
          6. Convert back → numpy + return LayerSliceResult

        The ``model`` arg (passed from registry.get) is ignored;
        HF runner owns its own model loading.

        Assumes LLaMA-style architecture (.model.layers). Future
        sprints handle GPT-style (.transformer.h) + other variants.
        """
        import time as _time
        hf_model = self._ensure_model_loaded()
        import torch as _torch
        from prsm.compute.chain_rpc.server import LayerSliceResult
        from prsm.compute.tee.models import TEEType

        start_t = _time.monotonic()
        # Sprint 612 — polymorphic layer extraction (LLaMA / GPT-2 /
        # GPT-NeoX). Raises StageExecutionError listing attempted
        # paths if unsupported architecture.
        layers = _resolve_hf_layers(hf_model)
        start, end = int(layer_range[0]), int(layer_range[1])
        if start < 0 or end > len(layers) or start >= end:
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner: invalid layer_range "
                f"({start}, {end}) for model with {len(layers)} layers"
            )
        original_ndim = activation.ndim
        hidden = _torch.from_numpy(activation).to(self.device)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        # position_ids = arange(S) broadcast over batch — sufficient
        # for non-cached forward; KV-cache + rolling positions land
        # in a later phase.
        seq_len = hidden.shape[-2] if hidden.dim() >= 2 else 1
        position_ids = _torch.arange(
            seq_len, device=self.device,
        ).unsqueeze(0)
        try:
            with _torch.no_grad():
                for i in range(start, end):
                    out = layers[i](hidden, position_ids=position_ids)
                    # LLaMA layer returns tuple (hidden_state, ...)
                    hidden = out[0] if isinstance(out, tuple) else out
        except Exception as exc:  # noqa: BLE001
            raise StageExecutionError(
                f"HuggingFaceLayerSliceRunner forward pass failed "
                f"at layer {i} of {self.model_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # Convert back to numpy, restore original ndim.
        if original_ndim == 2:
            hidden = hidden.squeeze(0)
        output_np = hidden.cpu().numpy()
        duration = _time.monotonic() - start_t
        return LayerSliceResult(
            output=output_np,
            duration_seconds=duration,
            tee_attestation=b"hf-runner-software-attestation",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )


class IdentityLayerSliceRunner:
    """Sprint 608 (Phase 2F-3) — smallest possible LayerSliceRunner.

    Returns the input activation UNCHANGED. Lets the FULL real-model
    server wire (chain_rpc.server.LayerStageServer parses the
    RunLayerSliceRequest, verifies upstream HandoffToken, signs the
    RunLayerSliceResponse) be exercised end-to-end without needing
    actual model weights loaded.

    Analogue to sprint-603 EchoStageExecutor but at the LayerSliceRunner
    layer (deeper in the stack — LayerStageServer.handle still runs
    full parse + sign + validate paths).

    Phase 2F-4+ ships real model-framework runners (HuggingFace
    transformers first probable target).
    """

    def run_layer_range(
        self,
        *,
        model: Any,
        layer_range: Any,
        activation: Any,
        privacy_tier: Any,
        is_final_stage: bool,
    ) -> Any:
        # Lazy imports — keep numpy off the import chain for daemons
        # that never enable the rpc/layer_stage kind.
        from prsm.compute.chain_rpc.server import LayerSliceResult
        from prsm.compute.tee.models import TEEType
        return LayerSliceResult(
            output=activation.copy(),  # defensive copy
            duration_seconds=0.0,
            tee_attestation=b"identity-runner-no-attestation",
            tee_type=TEEType.SOFTWARE,
            epsilon_spent=0.0,
        )


def build_layer_stage_server_executor(
    *,
    node: Any,
    runner: Any,
    model_registry_root_env: str = "PRSM_MODEL_REGISTRY_ROOT",
) -> "LayerStageServerStageExecutor":
    """Sprint 607 (Phase 2F-2) — factory for the real-model
    StageExecutor backed by chain_rpc.LayerStageServer.

    Required operator config:
      - ``$PRSM_MODEL_REGISTRY_ROOT`` env var (path to local
        model-registry directory)
      - ``$PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS`` env var
        (sprint 580 _build_anchor_or_none reads this)
      - ``runner`` argument — operator-supplied
        ``LayerSliceRunner`` (Phase 2F-3+ ships specific impls)
      - ``node.identity`` — present after PRSMNode.start

    Raises ``StageExecutionError`` with a structured message naming
    the missing piece when any prerequisite is absent, so operators
    flipping rpc kind on get actionable feedback.

    Lazy imports for the heavy ML deps (FilesystemModelRegistry,
    LayerStageServer, SoftwareTEERuntime) — default operators on
    PRSM_PARALLAX_STAGE_EXECUTOR_KIND=stub never trigger these.
    """
    import os as _os
    root = (_os.environ.get(model_registry_root_env, "") or "").strip()
    if not root:
        raise StageExecutionError(
            f"PRSM_MODEL_REGISTRY_ROOT unset; "
            f"build_layer_stage_server_executor needs the path to "
            f"the local FilesystemModelRegistry root directory."
        )
    from prsm.node.inference_wiring import _build_anchor_or_none
    anchor = _build_anchor_or_none()
    if anchor is None:
        raise StageExecutionError(
            "anchor unavailable (set PRSM_PUBLISHER_KEY_ANCHOR_ADDRESS); "
            "LayerStageServer requires it for upstream-token verification."
        )
    if runner is None or not hasattr(runner, "run_layer_range"):
        raise StageExecutionError(
            "runner is required (LayerSliceRunner with "
            ".run_layer_range method). Phase 2F-3+ ships specific "
            "runner implementations; pass one here."
        )
    if getattr(node, "identity", None) is None:
        raise StageExecutionError(
            "node.identity is None; daemon must be started "
            "(PRSMNode.start) before constructing the LayerStageServer "
            "executor."
        )

    # Lazy imports — only paid when this kind is actually selected.
    from prsm.compute.model_registry.registry import (
        FilesystemModelRegistry,
    )
    from prsm.compute.chain_rpc.server import LayerStageServer
    from prsm.compute.tee.runtime import SoftwareTEERuntime

    registry = FilesystemModelRegistry(root=root, anchor=anchor)
    tee_runtime = SoftwareTEERuntime()
    server = LayerStageServer(
        identity=node.identity,
        registry=registry,
        runner=runner,
        tee_runtime=tee_runtime,
        anchor=anchor,
    )
    return LayerStageServerStageExecutor(server=server)


def build_echo_stage_executor() -> StageExecutor:
    """Sprint 603 (Phase 2E-3) — diagnostic StageExecutor that
    echoes its input bytes back unchanged.

    NOT a real chain-stage forward pass (no model computation).
    Purpose: end-to-end wire testing of the Phase 2 client +
    server round-trip without requiring a model integration.

    Phase 2E-4 (sprint 604) wires this in behind an env var
    (PRSM_PARALLAX_STAGE_EXECUTOR_KIND=echo) for operator opt-in
    fleet diagnostics. Default stays at sprint-602 stub (raises)
    so production behavior isn't silently masked.

    Phase 2F+ ships real-model StageExecutor variants.
    """

    class _EchoStageExecutor:
        async def execute(self, request_bytes: bytes) -> bytes:
            return request_bytes

    return _EchoStageExecutor()


def _build_layer_slice_runner_from_env() -> Any:
    """Sprint 609 (Phase 2F-4) — env-driven LayerSliceRunner selection.

    Read PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND:
      - unset / "identity" → IdentityLayerSliceRunner (sprint 608,
        passthrough for testing). Default keeps testing-mode
        accessible without extra config.
      - anything else      → identity + WARNING (future: huggingface,
        torch, etc., shipped as Phase 2F-5+ runners).
    """
    import os as _os
    kind = (_os.environ.get(
        "PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND", "",
    ) or "").strip().lower() or "identity"
    if kind == "identity":
        return IdentityLayerSliceRunner()
    if kind == "huggingface":
        # Sprint 611 Phase 2F-5b — real HF transformer runner.
        # Operator MUST set PRSM_PARALLAX_HF_MODEL_ID to a valid HF
        # model identifier (e.g., "meta-llama/Llama-3.2-1B").
        import os as _os
        model_id = (_os.environ.get(
            "PRSM_PARALLAX_HF_MODEL_ID", "",
        ) or "").strip()
        if not model_id:
            import logging as _l
            _l.getLogger(__name__).warning(
                "Sprint 611 layer-slice-runner: huggingface kind "
                "requires PRSM_PARALLAX_HF_MODEL_ID env. Falling "
                "back to identity."
            )
            return IdentityLayerSliceRunner()
        device = (_os.environ.get(
            "PRSM_PARALLAX_HF_DEVICE", "",
        ) or "").strip() or "cpu"
        return HuggingFaceLayerSliceRunner(
            model_id=model_id, device=device,
        )
    import logging as _l
    _l.getLogger(__name__).warning(
        "Sprint 609/611 layer-slice-runner selector: "
        "PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND=%r unknown; falling "
        "back to identity. Valid: identity, huggingface.",
        kind,
    )
    return IdentityLayerSliceRunner()


def _build_stage_executor_from_env(node: Any = None) -> StageExecutor:
    """Sprint 604 (Phase 2E-4) — env-driven StageExecutor selection.
    Sprint 609 (Phase 2F-4) — extended with ``layer_stage`` kind.

    Read PRSM_PARALLAX_STAGE_EXECUTOR_KIND:
      - unset / "stub"   → build_stub_stage_executor (raises on
        execute(); production default — operators opt in explicitly).
      - "echo"           → build_echo_stage_executor (returns bytes
        unchanged; for end-to-end wire testing only).
      - "layer_stage"    → build_layer_stage_server_executor with
        runner from PRSM_PARALLAX_LAYER_SLICE_RUNNER_KIND. Real-model
        server-side path. Falls back to stub if factory raises
        StageExecutionError (missing env, missing anchor, etc.) with
        structured warning so operator sees the gap.
      - anything else    → stub fallback + structured warning.

    The optional ``node`` arg threads through for ``layer_stage``
    kind (needed for identity + downstream wiring). When None,
    layer_stage falls back to stub (sprint 604 callers pre-2F-4
    may not pass node).
    """
    import os as _os
    kind_raw = _os.environ.get(
        "PRSM_PARALLAX_STAGE_EXECUTOR_KIND", "",
    ).strip().lower()
    kind = kind_raw or "stub"
    if kind == "echo":
        return build_echo_stage_executor()
    if kind == "stub":
        return build_stub_stage_executor()
    if kind == "layer_stage":
        if node is None:
            import logging as _l
            _l.getLogger(__name__).warning(
                "Sprint 609 stage-executor selector: "
                "PRSM_PARALLAX_STAGE_EXECUTOR_KIND=layer_stage but "
                "the env-driven path is being called without a node "
                "arg (legacy caller pre-Phase 2F-4). Falling back "
                "to stub. Production callers in sprint-604 thread "
                "node through."
            )
            return build_stub_stage_executor()
        runner = _build_layer_slice_runner_from_env()
        try:
            return build_layer_stage_server_executor(
                node=node, runner=runner,
            )
        except StageExecutionError as exc:
            import logging as _l
            _l.getLogger(__name__).warning(
                "Sprint 609 stage-executor selector: "
                "build_layer_stage_server_executor raised %s. "
                "Falling back to stub.",
                exc,
            )
            return build_stub_stage_executor()
    # Unknown — warn + fall back to stub.
    import logging as _l
    _l.getLogger(__name__).warning(
        "Sprint 604/609 stage-executor selector: "
        "PRSM_PARALLAX_STAGE_EXECUTOR_KIND=%r unknown; falling "
        "back to stub. Valid: stub, echo, layer_stage.",
        kind_raw,
    )
    return build_stub_stage_executor()


async def handle_chain_executor_request(node: Any, msg: Any) -> bool:
    """Sprint 601 (Phase 2E-1) — server-side request handler.
    Sprint 604 (Phase 2E-4) — delegates execution to a StageExecutor.

    Receives a chain_executor_rpc REQUEST message (CHAIN_REQ_KEY set,
    CHAIN_RESP_KEY absent), decodes the base64 payload, hands the
    bytes to a ``StageExecutor`` (selected via
    ``PRSM_PARALLAX_STAGE_EXECUTOR_KIND`` env var), then ships the
    result back as a response message.

    Returns:
      True  — handled (response sent or attempted)
      False — not for us (wrong subtype, response message)

    Stage-executor selection:
      - ``node._chain_stage_executor`` attr if present (test
        injection / future production wiring)
      - else ``_build_stage_executor_from_env()`` (env-driven)

    Defensive semantics:
      - Wrong subtype → False
      - Has CHAIN_RESP_KEY → False (sprint 597 handles those)
      - Malformed base64 payload → CHAIN_ERROR_KEY response
      - StageExecutionError raised → CHAIN_ERROR_KEY response w/ msg
      - send_to_peer raises → log + return True (we tried)
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    if payload.get("subtype") != CHAIN_MSG_TYPE:
        return False
    request_id = payload.get(CHAIN_REQ_KEY)
    if not request_id:
        return False
    if payload.get(CHAIN_RESP_KEY):
        return False

    sender_id = getattr(msg, "sender_id", None)
    if not sender_id:
        return False

    # Decode incoming payload bytes.
    decode_error = None
    payload_bytes: bytes = b""
    try:
        payload_bytes = base64.b64decode(
            payload.get(CHAIN_PAYLOAD_KEY, ""),
        )
    except Exception as exc:  # noqa: BLE001
        decode_error = (
            f"chain-executor request payload base64-decode failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # Stage execution (skip if we already have a decode error).
    response_bytes: bytes = b""
    exec_error: str | None = decode_error
    if exec_error is None:
        executor: StageExecutor = getattr(
            node, "_chain_stage_executor", None,
        ) or _build_stage_executor_from_env(node=node)
        try:
            response_bytes = await executor.execute(payload_bytes)
        except StageExecutionError as exc:
            exec_error = (
                f"stage-executor raised StageExecutionError: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            exec_error = (
                f"stage-executor raised {type(exc).__name__}: {exc}"
            )

    # Build + ship response.
    try:
        from prsm.node.transport import P2PMessage, MSG_DIRECT
        if exec_error is not None:
            resp_payload = {
                "subtype": CHAIN_MSG_TYPE,
                CHAIN_REQ_KEY: request_id,
                CHAIN_RESP_KEY: request_id,
                CHAIN_ERROR_KEY: exec_error,
                CHAIN_PAYLOAD_KEY: "",
            }
        else:
            resp_payload = {
                "subtype": CHAIN_MSG_TYPE,
                CHAIN_REQ_KEY: request_id,
                CHAIN_RESP_KEY: request_id,
                CHAIN_PAYLOAD_KEY: base64.b64encode(
                    response_bytes,
                ).decode("ascii"),
            }
        response = P2PMessage(
            msg_type=MSG_DIRECT,
            sender_id=getattr(
                getattr(node, "identity", None), "node_id", "",
            ),
            payload=resp_payload,
        )
        await node.transport.send_to_peer(sender_id, response)
    except Exception as exc:  # noqa: BLE001
        import logging as _l
        _l.getLogger(__name__).warning(
            "Sprint 601/604 chain-executor request handler: "
            "failed to send response to %s: %s",
            sender_id, exc,
        )
    return True


def handle_chain_executor_response(node: Any, msg: Any) -> bool:
    """Sprint 597 (Phase 2D step 3) — resolve a pending chain-executor
    Future on receipt of a response message.

    Reads the response wire-protocol fields from ``msg.payload``:
      ``CHAIN_RESP_KEY``     → request_id (hex sha256 string)
      ``CHAIN_PAYLOAD_KEY``  → base64-encoded response bytes

    If the request_id is in ``node._chain_executor_pending``, sets
    the Future result. If not (request already timed out + popped,
    or unsolicited message), returns False without raising.

    Must be called from the loop thread (via the transport's normal
    inbound-message dispatch); the Future is on that loop. Use
    ``loop.call_soon_threadsafe`` if invoking from another thread.

    Returns ``True`` if the response was matched + Future resolved;
    ``False`` if no matching pending request (silent drop).
    """
    import base64
    payload = getattr(msg, "payload", None) or {}
    # Only handle our wire-protocol envelope; ignore other subtypes
    if payload.get("subtype") != CHAIN_MSG_TYPE:
        return False
    request_id = payload.get(CHAIN_RESP_KEY)
    if not request_id:
        return False
    pending = getattr(node, "_chain_executor_pending", None)
    if pending is None:
        return False
    future = pending.get(request_id)
    if future is None or future.done():
        # No pending request (timed out + cleaned up) OR already
        # resolved (duplicate response). Silent drop is correct.
        return False
    payload_b64 = payload.get(CHAIN_PAYLOAD_KEY, "")
    try:
        response_bytes = base64.b64decode(payload_b64)
    except Exception as exc:  # noqa: BLE001
        # Malformed payload → reject this response. Future stays
        # pending so the original SendMessage call still times out
        # cleanly rather than getting bogus bytes.
        future.set_exception(
            RuntimeError(
                f"chain-executor response payload base64-decode "
                f"failed for request_id={request_id}: {exc}"
            )
        )
        return True
    future.set_result(response_bytes)
    return True


def build_send_message_adapter(
    node: Any,
    timeout: float = 30.0,
) -> SendMessageAdapter:
    """Sprint 596 (Phase 2D step 2) — real SendMessage adapter.

    Bridges the sync ``SendMessage = Callable[[str, bytes], bytes]``
    contract over the daemon's async transport. Workflow:

      1. Hash request_bytes → request_id (sha256 hex).
      2. Resolve stage_address → peer_id via build_address_resolver-
         compatible lookup against ``node.transport.peers``.
      3. Create asyncio.Future, store in
         ``node._chain_executor_pending[request_id]``.
      4. Schedule coroutine on ``node._loop`` that sends a P2PMessage
         + awaits the future (resolved by sprint-597 response handler).
      5. Return response bytes synchronously.

    Phase 2D step 3 (sprint 597) wires the response handler that
    resolves the Future when a CHAIN_RESP_KEY message arrives.
    Until that lands, calls will time out — but the wire is sound.

    Raises:
      _Phase2AdapterNotReady — when node._loop is None (daemon
        not started or not running on asyncio context).
      PeerNotFound — when stage_address isn't in transport.peers.
      TimeoutError — when no response arrives within ``timeout``.
    """
    import base64
    import hashlib

    def _adapter(stage_address: str, request_bytes: bytes) -> bytes:
        loop = getattr(node, "_loop", None)
        if loop is None:
            raise _Phase2AdapterNotReady(
                "Sprint 596 SendMessage adapter: node._loop is None. "
                "Daemon must be started (sprint-595 captures the loop "
                "at PRSMNode.start). Until then, set "
                "PRSM_PARALLAX_CHAIN_EXECUTOR_KIND=stub for a working "
                "daemon."
            )

        request_id = hashlib.sha256(request_bytes).hexdigest()
        pending = node._chain_executor_pending

        async def _send_and_wait() -> bytes:
            # Import here to avoid circular: transport imports node
            # indirectly via __init__
            from prsm.node.transport import P2PMessage, MSG_DIRECT
            future = loop.create_future()
            pending[request_id] = future
            try:
                msg = P2PMessage(
                    msg_type=MSG_DIRECT,
                    sender_id=node.identity.node_id,
                    payload={
                        "subtype": CHAIN_MSG_TYPE,
                        CHAIN_REQ_KEY: request_id,
                        CHAIN_PAYLOAD_KEY: base64.b64encode(
                            request_bytes,
                        ).decode("ascii"),
                    },
                )
                # Resolve address → peer_id by lookup. The chain
                # executor passes stage_address that downstream code
                # treats as a peer_id (per sprint 593 resolver
                # contract). For Phase 2D the convention is:
                # stage_address IS the target peer_id.
                sent = await node.transport.send_to_peer(
                    stage_address, msg,
                )
                if not sent:
                    raise RuntimeError(
                        f"transport.send_to_peer returned False for "
                        f"peer_id={stage_address!r}; chain stage "
                        f"unreachable"
                    )
                import asyncio as _asyncio
                response_payload = await _asyncio.wait_for(
                    future, timeout=timeout,
                )
                return response_payload
            finally:
                pending.pop(request_id, None)

        return run_async_on_loop(loop, _send_and_wait(), timeout=timeout + 1.0)

    return _adapter
