"""Phase 3.x.7 Task 3 — activation_codec unit tests.

Coverage matches design plan §4 Task 3 acceptance:
  - Round-trip equality for every supported dtype
  - Shape preservation including non-contiguous + multi-dim arrays
  - Large-tensor chunking via Phase 6 ShardChunker
  - Reassembly integrity (per-chunk sha256 + overall sha256)
  - encode_for_wire auto-pick threshold behavior
  - Error mapping: unsupported dtype, size mismatch, integrity failures
"""

from __future__ import annotations

import numpy as np
import pytest

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
from prsm.node.shard_streaming import ShardChunk


# ──────────────────────────────────────────────────────────────────────────
# Inline encode / decode round-trip
# ──────────────────────────────────────────────────────────────────────────


SUPPORTED_DTYPES = [
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
]


class TestEncodeActivation:
    @pytest.mark.parametrize("dtype_str", SUPPORTED_DTYPES)
    def test_round_trip_preserves_dtype(self, dtype_str):
        rng = np.random.default_rng(seed=42)
        if dtype_str == "bool":
            arr = rng.integers(0, 2, size=(4, 8), dtype=np.uint8).astype(np.bool_)
        elif dtype_str.startswith("float"):
            arr = rng.standard_normal(size=(4, 8)).astype(np.dtype(dtype_str))
        else:
            arr = rng.integers(
                np.iinfo(np.dtype(dtype_str)).min // 2 + 1
                if np.iinfo(np.dtype(dtype_str)).min < 0
                else 0,
                np.iinfo(np.dtype(dtype_str)).max // 2,
                size=(4, 8),
                dtype=np.dtype(dtype_str),
            )

        blob, shape, returned_dtype = encode_activation(arr)
        recovered = decode_activation(blob, shape, returned_dtype)

        assert returned_dtype == dtype_str
        assert recovered.dtype == arr.dtype
        np.testing.assert_array_equal(recovered, arr)

    def test_preserves_shape(self):
        for shape in [(1,), (8,), (1, 4), (3, 5), (2, 3, 4), (2, 2, 2, 2)]:
            arr = np.ones(shape, dtype=np.float32)
            blob, returned_shape, _ = encode_activation(arr)
            assert returned_shape == shape
            recovered = decode_activation(blob, returned_shape, "float32")
            assert recovered.shape == shape

    def test_non_contiguous_input_forces_contiguous_output(self):
        # Create a non-contiguous view via slicing.
        original = np.arange(24, dtype=np.float32).reshape(4, 6)
        view = original[:, ::2]  # non-contiguous (stride > itemsize)
        assert not view.flags["C_CONTIGUOUS"]

        blob, shape, dtype_str = encode_activation(view)
        recovered = decode_activation(blob, shape, dtype_str)

        # Encoded bytes must match what a fresh contiguous copy would
        # produce — striding doesn't leak into the wire format.
        contig_view = np.ascontiguousarray(view)
        assert blob == contig_view.tobytes()
        np.testing.assert_array_equal(recovered, view)

    def test_rejects_non_ndarray(self):
        with pytest.raises(ActivationCodecError, match="numpy.ndarray"):
            encode_activation([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_unsupported_dtype(self):
        # Complex numbers are out of scope.
        arr = np.array([1 + 2j], dtype=np.complex64)
        with pytest.raises(ActivationCodecError, match="unsupported dtype"):
            encode_activation(arr)


class TestDecodeActivation:
    def test_rejects_size_mismatch(self):
        with pytest.raises(ActivationCodecError, match="does not match"):
            decode_activation(b"\x00" * 7, (1, 4), "float32")  # need 16 bytes

    def test_rejects_unsupported_dtype(self):
        with pytest.raises(ActivationCodecError, match="unsupported dtype"):
            decode_activation(b"", (1,), "complex64")

    def test_rejects_non_bytes_blob(self):
        with pytest.raises(ActivationCodecError, match="blob must be bytes"):
            decode_activation("a string", (1,), "float32")  # type: ignore[arg-type]

    def test_rejects_invalid_shape(self):
        with pytest.raises(ActivationCodecError, match="shape"):
            decode_activation(b"\x00" * 4, (0, 1), "float32")
        with pytest.raises(ActivationCodecError, match="shape"):
            decode_activation(b"\x00" * 4, (-1,), "float32")
        with pytest.raises(ActivationCodecError, match="shape"):
            decode_activation(b"\x00" * 4, [1], "float32")  # type: ignore[arg-type]

    def test_returns_writable_copy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        blob, shape, dtype_str = encode_activation(arr)
        recovered = decode_activation(blob, shape, dtype_str)
        # Mutating the recovered array must not affect the original blob.
        recovered[0] = 99.0
        recovered_again = decode_activation(blob, shape, dtype_str)
        assert recovered_again[0] == 1.0


# ──────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────


class TestChunkActivation:
    def test_round_trip_small_tensor(self):
        arr = np.arange(64, dtype=np.float32)
        chunked = chunk_activation(
            arr, activation_id="req-1::stage-0", chunk_bytes=64
        )
        recovered = reassemble_chunked(chunked, chunks=chunked.chunks)
        np.testing.assert_array_equal(recovered, arr)

    def test_round_trip_large_tensor(self):
        # 16 MiB float32 array (4M elements) — exceeds the default
        # 10 MiB threshold; explicitly chunked.
        arr = np.random.default_rng(seed=7).standard_normal(
            size=(2, 2_000_000)
        ).astype(np.float32)
        chunked = chunk_activation(arr, activation_id="big::0")
        # Ensure we actually got multiple chunks.
        assert chunked.manifest.total_chunks > 1
        recovered = reassemble_chunked(chunked, chunks=chunked.chunks)
        assert recovered.shape == arr.shape
        np.testing.assert_array_equal(recovered, arr)

    def test_chunked_activation_carries_shape_and_dtype(self):
        arr = np.ones((3, 5), dtype=np.float16)
        chunked = chunk_activation(arr, activation_id="x", chunk_bytes=8)
        assert chunked.shape == (3, 5)
        assert chunked.dtype_str == "float16"

    def test_manifest_payload_sha_matches_concatenation(self):
        import hashlib
        arr = np.arange(100, dtype=np.int32)
        chunked = chunk_activation(arr, activation_id="x", chunk_bytes=64)
        concat = b"".join(c.data for c in chunked.chunks)
        expected = hashlib.sha256(concat).hexdigest()
        assert chunked.manifest.payload_sha256 == expected

    def test_per_chunk_sha_individually_correct(self):
        import hashlib
        arr = np.arange(100, dtype=np.int32)
        chunked = chunk_activation(arr, activation_id="x", chunk_bytes=32)
        for chunk in chunked.chunks:
            assert chunk.chunk_sha256 == hashlib.sha256(chunk.data).hexdigest()


class TestReassembleChunked:
    def test_detects_corrupted_chunk(self):
        arr = np.arange(64, dtype=np.float32)
        chunked = chunk_activation(arr, activation_id="x", chunk_bytes=64)
        # Tamper one chunk's data without updating its sha.
        bad_chunks = list(chunked.chunks)
        bad_chunks[0] = ShardChunk(
            shard_id=bad_chunks[0].shard_id,
            sequence=bad_chunks[0].sequence,
            data=b"\x00" * len(bad_chunks[0].data),
            chunk_sha256=bad_chunks[0].chunk_sha256,  # original sha
        )
        with pytest.raises(ActivationCodecError, match="reassembly failed"):
            reassemble_chunked(chunked, chunks=bad_chunks)

    def test_detects_out_of_order_chunks(self):
        arr = np.arange(64, dtype=np.float32)
        chunked = chunk_activation(arr, activation_id="x", chunk_bytes=32)
        if len(chunked.chunks) < 2:
            pytest.skip("need at least two chunks for ordering test")
        reordered = list(chunked.chunks)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with pytest.raises(ActivationCodecError, match="reassembly failed"):
            reassemble_chunked(chunked, chunks=reordered)

    def test_detects_dropped_chunk(self):
        arr = np.arange(64, dtype=np.float32)
        chunked = chunk_activation(arr, activation_id="x", chunk_bytes=32)
        if len(chunked.chunks) < 2:
            pytest.skip("need at least two chunks for drop test")
        partial = chunked.chunks[:-1]
        with pytest.raises(ActivationCodecError, match="reassembly failed"):
            reassemble_chunked(chunked, chunks=partial)

    def test_chunks_can_arrive_via_generator(self):
        """The reassembly path accepts an iterable, including generators
        from a real gRPC stream."""
        arr = np.arange(20, dtype=np.float32)
        chunked = chunk_activation(arr, activation_id="x", chunk_bytes=16)

        def stream():
            for c in chunked.chunks:
                yield c

        recovered = reassemble_chunked(chunked, chunks=stream())
        np.testing.assert_array_equal(recovered, arr)


# ──────────────────────────────────────────────────────────────────────────
# Threshold + auto-pick
# ──────────────────────────────────────────────────────────────────────────


class TestShouldChunk:
    def test_below_threshold_returns_false(self):
        assert should_chunk(b"x" * 1024) is False

    def test_above_threshold_returns_true(self):
        assert should_chunk(b"x" * (CHUNK_THRESHOLD_BYTES + 1)) is True

    def test_boundary_exact_threshold_is_inline(self):
        assert should_chunk(b"x" * CHUNK_THRESHOLD_BYTES) is False

    def test_custom_threshold_overrides_default(self):
        assert should_chunk(b"x" * 100, threshold=50) is True
        assert should_chunk(b"x" * 100, threshold=200) is False


class TestEncodeForWire:
    def test_small_tensor_picks_inline(self):
        arr = np.zeros((4, 4), dtype=np.float32)
        is_chunked, payload = encode_for_wire(arr, activation_id="x")
        assert is_chunked is False
        assert isinstance(payload, tuple)
        assert len(payload) == 3
        blob, shape, dtype_str = payload
        assert shape == (4, 4)
        assert dtype_str == "float32"

    def test_above_threshold_picks_streamed(self):
        # Force chunking via low custom threshold.
        arr = np.zeros((1024,), dtype=np.float32)  # 4 KiB
        is_chunked, payload = encode_for_wire(
            arr, activation_id="x", threshold=1024, chunk_bytes=512
        )
        assert is_chunked is True
        assert isinstance(payload, ChunkedActivation)
        assert payload.shape == (1024,)
        assert payload.dtype_str == "float32"
        assert payload.manifest.total_chunks >= 2

    def test_streamed_round_trips(self):
        rng = np.random.default_rng(seed=11)
        arr = rng.standard_normal(size=(64,)).astype(np.float32)
        is_chunked, payload = encode_for_wire(
            arr, activation_id="x", threshold=64, chunk_bytes=32
        )
        assert is_chunked is True
        recovered = reassemble_chunked(payload, chunks=payload.chunks)
        np.testing.assert_array_equal(recovered, arr)


# ──────────────────────────────────────────────────────────────────────────
# Sanity / constants
# ──────────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_threshold_is_ten_mib(self):
        assert CHUNK_THRESHOLD_BYTES == 10 * 1024 * 1024

    def test_default_chunk_size_one_mib(self):
        assert DEFAULT_CHUNK_BYTES_ACTIVATION == 1 * 1024 * 1024

    def test_allowed_dtypes_includes_typical_inference_dtypes(self):
        assert "float32" in ALLOWED_DTYPES
        assert "float16" in ALLOWED_DTYPES
        assert "int8" in ALLOWED_DTYPES
        # Complex floats / objects must NOT be in the allowed set.
        assert "complex64" not in ALLOWED_DTYPES
        assert "object" not in ALLOWED_DTYPES
