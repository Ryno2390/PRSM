"""Phase 3.x.7 cross-host ChainExecutor RPC layer.

Wire protocol + LayerStageServer + RpcChainExecutor + activation
codec + production-wiring factories for running an inference chain
(a ``GPUChain`` decided by the Phase 3.x.6 scheduler) across PRSM
nodes on real Phase 6 transport.

Public surface (curated):
  - Wire protocol: ``HandoffToken``, ``RunLayerSliceRequest``,
    ``RunLayerSliceResponse``, ``StageError``, ``StageErrorCode``,
    ``encode_message`` / ``parse_message``, ``MAX_HANDSHAKE_BYTES``,
    ``CHAIN_RPC_PROTOCOL_VERSION``.
  - Server: ``LayerStageServer``, ``LayerSliceRunner`` Protocol,
    ``LayerSliceResult``.
  - Client: ``RpcChainExecutor``, ``ChainExecutionError``,
    ``ExecutorErrorCode``, ``StageOutcome``.
  - Activation codec: ``encode_activation`` / ``decode_activation``,
    ``chunk_activation`` / ``reassemble_chunked``, ``encode_for_wire``,
    ``ChunkedActivation``, ``ActivationCodecError``.
  - Production-wiring factories: ``make_rpc_chain_executor``,
    ``make_layer_stage_server``, ``utf8_prompt_encoder`` /
    ``utf8_output_decoder`` (test-friendly defaults).

Production callers typically use the inference-package re-exports
instead: ``from prsm.compute.inference import make_rpc_chain_executor``.
"""

from prsm.compute.chain_rpc.activation_codec import (
    ALLOWED_DTYPES,
    CHUNK_THRESHOLD_BYTES,
    DEFAULT_CHUNK_BYTES_ACTIVATION,
    ActivationCodecError,
    ChunkedActivation,
    chunk_activation,
    decode_activation,
    encode_activation,
    encode_for_wire,
    reassemble_chunked,
    should_chunk,
)
from prsm.compute.chain_rpc.protocol import (
    CHAIN_RPC_PROTOCOL_VERSION,
    MAX_HANDSHAKE_BYTES,
    ChainRpcMalformedError,
    ChainRpcMessageType,
    ChainRpcProtocolError,
    ChainRpcUnknownTypeError,
    ChainRpcVersionMismatchError,
    HandoffToken,
    RunLayerSliceRequest,
    RunLayerSliceResponse,
    StageError,
    StageErrorCode,
    encode_message,
    parse_message,
)
from prsm.compute.chain_rpc.client import (
    AddressResolver,
    ChainExecutionError,
    ExecutorErrorCode,
    OutputDecoder,
    PromptEncoder,
    RpcChainExecutor,
    SendMessage,
    StageOutcome,
)
from prsm.compute.chain_rpc.factories import (
    make_layer_stage_server,
    make_rpc_chain_executor,
    utf8_output_decoder,
    utf8_prompt_encoder,
)
from prsm.compute.chain_rpc.server import (
    LayerSliceResult,
    LayerSliceRunner,
    LayerStageServer,
)

__all__ = [
    # Protocol
    "CHAIN_RPC_PROTOCOL_VERSION",
    "MAX_HANDSHAKE_BYTES",
    "ChainRpcMalformedError",
    "ChainRpcMessageType",
    "ChainRpcProtocolError",
    "ChainRpcUnknownTypeError",
    "ChainRpcVersionMismatchError",
    "HandoffToken",
    "RunLayerSliceRequest",
    "RunLayerSliceResponse",
    "StageError",
    "StageErrorCode",
    "encode_message",
    "parse_message",
    # Activation codec
    "ALLOWED_DTYPES",
    "CHUNK_THRESHOLD_BYTES",
    "DEFAULT_CHUNK_BYTES_ACTIVATION",
    "ActivationCodecError",
    "ChunkedActivation",
    "chunk_activation",
    "decode_activation",
    "encode_activation",
    "encode_for_wire",
    "reassemble_chunked",
    "should_chunk",
    # Server
    "LayerSliceResult",
    "LayerSliceRunner",
    "LayerStageServer",
    # Client
    "AddressResolver",
    "ChainExecutionError",
    "ExecutorErrorCode",
    "OutputDecoder",
    "PromptEncoder",
    "RpcChainExecutor",
    "SendMessage",
    "StageOutcome",
    # Factories (production-wiring entry points)
    "make_layer_stage_server",
    "make_rpc_chain_executor",
    "utf8_output_decoder",
    "utf8_prompt_encoder",
]
