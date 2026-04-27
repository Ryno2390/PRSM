"""
PRSM manifest DHT — opt-in cross-node distribution of model manifests.

Phase 3.x.5 closes the AVAILABILITY half of the cross-node trust gap
identified in Phase 3.x.3 ("DHT distribution NOT in scope: 3.x.3
covers TRUST"). 3.x.3 lets verifiers authenticate any manifest given
the on-chain anchor; 3.x.5 lets any node serve a verified copy.

Tightly scoped to MANIFEST distribution only. Privacy-budget journals
and inference receipts are explicitly NOT distributed via this DHT
(per design plan §1.2 — different artifact-type semantics).

Anchor verification is MANDATORY on every DHT-fetched manifest. There
is no "trust the network" mode — readers without an anchored
publisher key reject the bytes and try the next provider.

See ``docs/2026-04-27-phase3.x.5-manifest-dht-design-plan.md``.
"""

from prsm.network.manifest_dht.dht_client import (
    DEFAULT_K,
    DHTClientError,
    ManifestDHTClient,
    ManifestNotFoundError,
    RoutingTable,
    SendMessageFn,
    TransportFailureError,
)
from prsm.network.manifest_dht.dht_server import (
    UNKNOWN_REQUEST_ID,
    ManifestDHTServer,
)
from prsm.network.manifest_dht.local_index import LocalManifestIndex
from prsm.network.manifest_dht.protocol import (
    DHT_PROTOCOL_VERSION,
    ErrorCode,
    ErrorResponse,
    FetchManifestRequest,
    FindProvidersRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    ManifestResponse,
    MAX_MESSAGE_BYTES,
    MAX_PROVIDERS_PER_RESPONSE,
    MESSAGE_TYPE_REGISTRY,
    MessageType,
    ProtocolError,
    ProviderInfo,
    ProvidersResponse,
    UnknownMessageTypeError,
    encode_message,
    parse_message,
)

__all__ = [
    # Protocol constants (Task 1)
    "DHT_PROTOCOL_VERSION",
    "MAX_MESSAGE_BYTES",
    "MAX_PROVIDERS_PER_RESPONSE",
    "MESSAGE_TYPE_REGISTRY",
    # Enums
    "ErrorCode",
    "MessageType",
    # Message types
    "FindProvidersRequest",
    "FetchManifestRequest",
    "ProviderInfo",
    "ProvidersResponse",
    "ManifestResponse",
    "ErrorResponse",
    # Codec
    "encode_message",
    "parse_message",
    # Exceptions
    "ProtocolError",
    "MalformedMessageError",
    "UnknownMessageTypeError",
    "IncompatibleProtocolVersionError",
    # Local index (Task 2)
    "LocalManifestIndex",
    # DHT client (Task 3)
    "ManifestDHTClient",
    "RoutingTable",
    "SendMessageFn",
    "DEFAULT_K",
    "DHTClientError",
    "ManifestNotFoundError",
    "TransportFailureError",
    # DHT server (Task 4)
    "ManifestDHTServer",
    "UNKNOWN_REQUEST_ID",
]
