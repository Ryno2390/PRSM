"""
Phase 3.x.3 Task 3 — exception hierarchy for the publisher-key anchor.
"""


class PublisherKeyAnchorError(Exception):
    """Base error for any anchor-layer failure.

    Callers can ``except PublisherKeyAnchorError`` to catch any
    anchor-related failure without distinguishing the specific cause.
    """


class PublisherAlreadyRegisteredError(PublisherKeyAnchorError):
    """``register_self`` reverted because the publisher's node_id is
    already registered on-chain.

    Idempotency at the contract level: the write-once invariant
    rejects re-registration. Callers MAY treat this as a no-op
    (registration already happened in a previous run).
    """


class PublisherNotRegisteredError(PublisherKeyAnchorError):
    """Operation expected a registered publisher but the on-chain
    lookup returned empty bytes.

    Distinct from ``lookup`` which returns ``None`` on miss — this
    error is raised by call sites that need to ASSERT registration
    (e.g., a verifier that refuses to fall back).
    """


class AnchorRPCError(PublisherKeyAnchorError):
    """The RPC call to the anchor contract failed at the transport
    level (network, provider, malformed response).

    Distinct from contract reverts: an RPC failure is a transient
    operational error, while a revert is a deliberate contract-level
    rejection. Callers may want to retry RPC failures but not reverts.
    """
