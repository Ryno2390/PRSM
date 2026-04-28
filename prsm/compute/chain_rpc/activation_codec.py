"""Phase 3.x.7 Task 3 — Activation tensor encoding for cross-host handoff.

Two paths:

  Inline   For activations under ``CHUNK_THRESHOLD_BYTES`` (10 MiB by
           default), the bytes ride directly inside ``RunLayerSliceRequest``
           / ``RunLayerSliceResponse`` via the existing hex-encoded
           ``activation_blob`` field. No streaming overhead; minimal
           round trips.

  Streamed When the encoded blob exceeds the threshold, the wire layer
           splits it into chunks via Phase 6's ``ShardChunker`` (already
           shipped — it's the same chunker the >10 MB shard distribution
           path uses). The on-the-wire envelope carries the
           ``ChunkedPayloadEnvelope`` summary; the chunks themselves
           ride out-of-band on the gRPC streaming RPC.

The codec sits between the numpy world (where layer execution lives)
and the wire world (raw bytes the protocol layer carries). It does
NOT own the gRPC transport; that's the production wiring's
responsibility (Task 4 / Task 6).

Determinism guarantees:
  - Output bytes are byte-identical for byte-identical inputs.
  - Non-contiguous numpy arrays are forced C-contiguous via
    ``np.ascontiguousarray`` so striding doesn't leak into the wire.
  - Float dtypes preserved bit-for-bit; we don't quantize at this
    layer (R7 research-track concern).

Supported dtypes: float16, float32, float64, int8, int16, int32,
int64, uint8, bool. Other dtypes raise ``ActivationCodecError`` —
callers can extend the allow-list, but the default narrows the wire
surface to the dtypes typical inference paths produce.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from prsm.node.shard_streaming import (
    DEFAULT_CHUNK_BYTES,
    ShardAssembler,
    ShardChunk,
    ShardChunker,
    ShardManifest,
    StreamingError,
)


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────


CHUNK_THRESHOLD_BYTES = 10 * 1024 * 1024
"""Above this encoded-blob size, the wire layer splits into chunks via
Phase 6 ``ShardChunker``. Matches the design plan §3.3 + Phase 6 Task
6b's ``>10 MB`` cutoff for the gRPC streaming path. Operators tuning
the gRPC max-message cap may also want to adjust this in lockstep."""

DEFAULT_CHUNK_BYTES_ACTIVATION = DEFAULT_CHUNK_BYTES
"""Per-chunk size when streaming. 1 MiB matches Phase 6's default;
exposed here so Task 7 E2E tests can shrink it for coverage without
fabricating 10 MiB tensors."""

ALLOWED_DTYPES: frozenset = frozenset({
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "bool",
})
"""Supported numpy dtype names (the canonical str-form of ``np.dtype``).
Narrowed to the dtypes typical inference paths produce; structured
dtypes / object arrays / complex floats are out of scope (any of those
in an activation tensor signals a callsite bug rather than a codec
gap)."""


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────


class ActivationCodecError(Exception):
    """Base for activation-codec failures (encode + decode + chunked
    reassembly). The chain-RPC server maps these to
    ``StageError(ACTIVATION_INVALID)``; the client maps them to a
    structured ``ChainExecutionError``."""


# ──────────────────────────────────────────────────────────────────────────
# Inline path
# ──────────────────────────────────────────────────────────────────────────


def encode_activation(arr: np.ndarray) -> Tuple[bytes, Tuple[int, ...], str]:
    """Encode a numpy array as ``(raw_bytes, shape, dtype_str)``.

    Forces C-contiguous layout so output bytes are deterministic
    regardless of upstream striding. The caller decides whether to
    inline the bytes in the RPC handshake or stream them via
    ``chunk_activation`` based on the returned blob size.
    """
    if not isinstance(arr, np.ndarray):
        raise ActivationCodecError(
            f"encode_activation expects numpy.ndarray, got {type(arr).__name__}"
        )
    dtype_str = str(arr.dtype)
    if dtype_str not in ALLOWED_DTYPES:
        raise ActivationCodecError(
            f"unsupported dtype {dtype_str!r}; allowed: {sorted(ALLOWED_DTYPES)}"
        )
    contig = np.ascontiguousarray(arr)
    return contig.tobytes(), tuple(int(d) for d in contig.shape), dtype_str


def decode_activation(
    blob: bytes,
    shape: Tuple[int, ...],
    dtype_str: str,
) -> np.ndarray:
    """Reconstruct a numpy array from raw bytes + shape + dtype.

    Returns a fresh writable copy (not a view onto ``blob``) so callers
    can mutate without surprising the wire-layer's bookkeeping.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise ActivationCodecError(
            f"blob must be bytes, got {type(blob).__name__}"
        )
    if not isinstance(shape, tuple) or not all(
        isinstance(d, int) and d > 0 for d in shape
    ):
        raise ActivationCodecError(
            f"shape must be tuple of positive ints, got {shape!r}"
        )
    if dtype_str not in ALLOWED_DTYPES:
        raise ActivationCodecError(
            f"unsupported dtype {dtype_str!r}; allowed: {sorted(ALLOWED_DTYPES)}"
        )
    try:
        dtype = np.dtype(dtype_str)
    except TypeError as exc:
        raise ActivationCodecError(
            f"numpy rejected dtype {dtype_str!r}: {exc}"
        ) from exc
    expected_size = int(np.prod(shape)) * dtype.itemsize
    if expected_size != len(blob):
        raise ActivationCodecError(
            f"blob size {len(blob)} does not match shape {shape} × "
            f"dtype {dtype_str} (expected {expected_size} bytes)"
        )
    return np.frombuffer(blob, dtype=dtype).reshape(shape).copy()


def should_chunk(blob: bytes, *, threshold: int = CHUNK_THRESHOLD_BYTES) -> bool:
    """Return True if the encoded blob should ride the streaming path
    instead of inline. Pure decision function so callers can predicate
    on the same threshold used internally."""
    return len(blob) > threshold


# ──────────────────────────────────────────────────────────────────────────
# Streaming path — wraps Phase 6 ShardChunker / ShardAssembler
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkedActivation:
    """Output of ``chunk_activation`` — the manifest + ordered chunks.

    Production wiring (Task 4) sends ``manifest`` first as the gRPC
    stream's opening frame, then iterates ``chunks`` in sequence.
    Receivers reassemble via ``reassemble_chunked``.

    The ``shape`` and ``dtype`` are duplicated here (rather than only
    in ``manifest``) so the receiver can validate the post-reassembly
    bytes against the same expected size that the inline path uses.
    """

    manifest: ShardManifest
    chunks: List[ShardChunk]
    shape: Tuple[int, ...]
    dtype_str: str


def chunk_activation(
    arr: np.ndarray,
    *,
    activation_id: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES_ACTIVATION,
) -> ChunkedActivation:
    """Encode + chunk an activation for streamed handoff.

    ``activation_id`` is propagated as the chunker's ``shard_id``
    (Phase 6 ``ShardChunker`` is dual-use — it's the same logic used
    for >10 MB shard distribution; we reuse it here for activations).
    Production callers typically use ``f"{request_id}::{stage_index}"``
    so receivers can correlate the activation to the chain stage.
    """
    blob, shape, dtype_str = encode_activation(arr)
    chunker = ShardChunker(chunk_bytes=chunk_bytes)
    manifest, chunks = chunker.chunk(activation_id, blob)
    return ChunkedActivation(
        manifest=manifest,
        chunks=chunks,
        shape=shape,
        dtype_str=dtype_str,
    )


def reassemble_chunked(
    chunked: ChunkedActivation,
    *,
    chunks: Iterable[ShardChunk],
) -> np.ndarray:
    """Reverse of ``chunk_activation``: feed the inbound chunks into
    a ``ShardAssembler`` (Phase 6), finalize, then ``decode_activation``
    on the resulting bytes.

    ``chunks`` may be a generator (gRPC stream); the assembler enforces
    strict in-order delivery and per-chunk + overall sha256 checks
    before returning. Reassembly errors map to ``ActivationCodecError``
    so callers see a uniform error surface regardless of whether the
    failure was at the codec or the integrity layer.
    """
    assembler = ShardAssembler(chunked.manifest)
    try:
        for chunk in chunks:
            assembler.feed(chunk)
        payload = assembler.finalize()
    except StreamingError as exc:
        raise ActivationCodecError(
            f"chunked activation reassembly failed: {exc}"
        ) from exc
    return decode_activation(payload, chunked.shape, chunked.dtype_str)


# ──────────────────────────────────────────────────────────────────────────
# Convenience: round-trip for inline-or-streamed
# ──────────────────────────────────────────────────────────────────────────


def encode_for_wire(
    arr: np.ndarray,
    *,
    activation_id: str,
    threshold: int = CHUNK_THRESHOLD_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES_ACTIVATION,
) -> Tuple[bool, object]:
    """Pick inline vs streamed automatically.

    Returns ``(is_chunked, payload)`` where ``payload`` is either:
      - inline:   ``(blob, shape, dtype_str)`` from ``encode_activation``
      - streamed: a ``ChunkedActivation`` from ``chunk_activation``

    Convenience for production callers (Task 4) who want the threshold
    decision made at one place; tests can call ``encode_activation`` /
    ``chunk_activation`` directly to exercise either path.
    """
    blob, shape, dtype_str = encode_activation(arr)
    if not should_chunk(blob, threshold=threshold):
        return False, (blob, shape, dtype_str)
    chunker = ShardChunker(chunk_bytes=chunk_bytes)
    manifest, chunks = chunker.chunk(activation_id, blob)
    return True, ChunkedActivation(
        manifest=manifest,
        chunks=chunks,
        shape=shape,
        dtype_str=dtype_str,
    )
