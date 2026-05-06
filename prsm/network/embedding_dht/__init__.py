"""Embedding DHT — cross-node content embedding gossip.

PRSM-PROV-1 Item 3.

Public surface (re-exported from submodules as they land):
- protocol: wire-format messages + canonical signing payload
- local_index: RAM-backed (content_hash, model_id) → vector store
- dht_server: serves FIND_EMBEDDING / FETCH_EMBEDDING
- dht_client: queries peers via Phase 6 transport

For now, only ``protocol`` is wired up — Tasks 2-6 add the rest.
"""

from prsm.network.embedding_dht.dht_client import (
    EmbeddingDHTClient,
    EmbeddingDHTClientError,
    EmbeddingNotFoundError,
    SignatureVerificationError,
    TransportFailureError,
)
from prsm.network.embedding_dht.dht_server import EmbeddingDHTServer
from prsm.network.embedding_dht.local_index import (
    LocalEmbeddingIndex,
    LocalEmbeddingRecord,
)
from prsm.network.embedding_dht.protocol import (
    ALLOWED_DTYPES,
    EMBEDDING_DHT_PROTOCOL_VERSION,
    MAX_MESSAGE_BYTES,
    MAX_PROVIDERS_PER_RESPONSE,
    MAX_VECTOR_DIMENSION,
    SIGNING_DOMAIN_TAG,
    EmbeddingProvidersResponse,
    EmbeddingResponse,
    ErrorCode,
    ErrorResponse,
    FetchEmbeddingRequest,
    FindEmbeddingRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    MessageType,
    ProtocolError,
    ProviderInfo,
    UnknownMessageTypeError,
    canonical_signing_payload,
    encode_message,
    parse_message,
)

__all__ = [
    "EmbeddingDHTClient",
    "EmbeddingDHTClientError",
    "EmbeddingDHTServer",
    "EmbeddingNotFoundError",
    "SignatureVerificationError",
    "TransportFailureError",
    "LocalEmbeddingIndex",
    "LocalEmbeddingRecord",
    "ALLOWED_DTYPES",
    "EMBEDDING_DHT_PROTOCOL_VERSION",
    "MAX_MESSAGE_BYTES",
    "MAX_PROVIDERS_PER_RESPONSE",
    "MAX_VECTOR_DIMENSION",
    "SIGNING_DOMAIN_TAG",
    "EmbeddingProvidersResponse",
    "EmbeddingResponse",
    "ErrorCode",
    "ErrorResponse",
    "FetchEmbeddingRequest",
    "FindEmbeddingRequest",
    "IncompatibleProtocolVersionError",
    "MalformedMessageError",
    "MessageType",
    "ProtocolError",
    "ProviderInfo",
    "UnknownMessageTypeError",
    "canonical_signing_payload",
    "encode_message",
    "parse_message",
]
