"""Phase 3.x.7 cross-host ChainExecutor RPC layer.

Wire protocol + LayerStageServer + RpcChainExecutor for running an
inference chain (a `GPUChain` decided by the Phase 3.x.6 scheduler)
across PRSM nodes on real Phase 6 transport.

Task 1 ships the wire protocol only — server and client land in Tasks
2 and 4. The public surface is curated as those tasks land; this
__init__ currently re-exports the protocol-layer types.
"""

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

__all__ = [
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
]
