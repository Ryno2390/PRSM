"""
PRSM-PROV-1 Item 4 T4.4 — VideoFingerprint backend tests.

Covers:
  - KIND attribute matches FingerprintKind.VIDEO_MULTIHASH.
  - compute() round-trip on a synthetic video (PyAV-encoded).
  - Different videos produce different fingerprints.
  - Identical video bytes produce identical fingerprints.
  - Truncated / non-video input returns None.
  - Empty bytes returns None.
  - Payload format: exactly 64 bytes (8 keyframes * 8 bytes each).
  - similarity() returns 1.0 for identical, 0.0 for malformed,
    fractional for partial-match.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Skip the entire module if PyAV / numpy aren't available.
av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from prsm.data.fingerprints import FingerprintKind  # noqa: E402
from prsm.data.fingerprints.video import VideoFingerprint  # noqa: E402
from prsm.data.fingerprints.base import FingerprintRecord  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Synthetic video helpers
# ──────────────────────────────────────────────────────────────────────


def _gradient_video_bytes(
    *,
    duration_sec: float = 4.0,
    fps: int = 24,
    width: int = 64,
    height: int = 64,
    seed: int = 0,
    container_format: str = "mp4",
    codec: str = "mpeg4",
) -> bytes:
    """Encode a synthetic video as bytes.

    Each frame is a 2-D gradient whose pattern shifts across time —
    enough variation that the 8-keyframe sample produces different
    pHashes per frame (so a different ``seed`` yields a different
    fingerprint). Default 4s @ 24fps = 96 frames, more than the 8
    keyframes the backend samples.

    Uses PyAV to encode in-memory via a temp file (PyAV's open() needs
    a path or seekable handle for muxing).
    """
    n_frames = int(duration_sec * fps)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{container_format}",
    )
    tmp.close()
    try:
        container = av.open(tmp.name, mode="w", format=container_format)
        try:
            stream = container.add_stream(codec, rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            for i in range(n_frames):
                # Time-varying gradient. seed shifts the spatial
                # pattern; i adds a temporal phase. uint8 to satisfy
                # av.VideoFrame.from_ndarray's gray8 contract.
                ys = np.arange(height, dtype=np.float32).reshape(-1, 1)
                xs = np.arange(width, dtype=np.float32).reshape(1, -1)
                phase = (i / n_frames) * 2.0 * np.pi
                pattern = (
                    128.0
                    + 80.0 * np.sin((xs + ys + seed * 7) * 0.1 + phase)
                    + 40.0 * np.cos((xs - ys + seed * 11) * 0.07 - phase)
                )
                gray = np.clip(pattern, 0, 255).astype(np.uint8)
                # Build an RGB frame (YUV420p needs colour data,
                # so replicate the gray channel into all three).
                rgb = np.stack([gray, gray, gray], axis=-1)
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush.
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestVideoFingerprint:
    def test_kind(self):
        assert VideoFingerprint.KIND == FingerprintKind.VIDEO_MULTIHASH

    def test_compute_round_trip(self):
        """Synthetic gradient video produces a 64-byte payload."""
        video_bytes = _gradient_video_bytes(seed=0)
        record = VideoFingerprint().compute(video_bytes, filename="clip.mp4")
        assert record is not None, (
            "expected fingerprint for a 4s synthetic clip — "
            "PyAV / FFmpeg may be misconfigured"
        )
        assert isinstance(record, FingerprintRecord)
        assert record.kind == FingerprintKind.VIDEO_MULTIHASH
        # 8 keyframes * 8 bytes per pHash = 64 bytes total.
        assert len(record.payload) == 64

    def test_compute_identical_bytes_identical_fingerprint(self):
        """Same MP4 bytes → identical 64-byte payload."""
        video_bytes = _gradient_video_bytes(seed=0)
        backend = VideoFingerprint()
        a = backend.compute(video_bytes, filename="a.mp4")
        b = backend.compute(video_bytes, filename="b.mp4")
        assert a is not None and b is not None
        assert a.payload == b.payload
        assert backend.similarity(a.payload, b.payload) == pytest.approx(1.0)

    def test_compute_different_videos_lower_similarity(self):
        """Different gradient seeds produce different multi-hashes."""
        video_a = _gradient_video_bytes(seed=0)
        video_b = _gradient_video_bytes(seed=42)
        backend = VideoFingerprint()
        a = backend.compute(video_a, filename="a.mp4")
        b = backend.compute(video_b, filename="b.mp4")
        assert a is not None and b is not None
        assert a.payload != b.payload
        sim = backend.similarity(a.payload, b.payload)
        # Different gradient patterns should not all match within the
        # per-frame 6/64 Hamming tolerance — well below the 0.875
        # duplicate threshold.
        assert sim < 0.875, (
            f"expected sim < 0.875 for distinct gradients, got {sim}"
        )

    def test_compute_empty_returns_none(self):
        assert VideoFingerprint().compute(b"") is None

    def test_compute_garbage_returns_none(self):
        """Non-video bytes can't be demuxed → None."""
        # 4KB of zeros — PyAV's demuxer rejects without entering an
        # audio-decode fallback (this is the safe path; we don't go
        # through audioread the way pyacoustid does).
        result = VideoFingerprint().compute(b"\x00" * 4096, filename="garbage.mp4")
        assert result is None

    def test_similarity_identical_payloads(self):
        backend = VideoFingerprint()
        # 8 frames * 8 bytes = 64 bytes.
        payload = bytes(range(64))
        assert backend.similarity(payload, payload) == pytest.approx(1.0)

    def test_similarity_all_frames_differ(self):
        """All 8 frames flip every bit → 0/8 match → similarity 0.0."""
        backend = VideoFingerprint()
        a = b"\x00" * 64
        b = b"\xff" * 64
        assert backend.similarity(a, b) == pytest.approx(0.0)

    def test_similarity_partial_frame_match(self):
        """Half the keyframes match, half don't → similarity 0.5."""
        backend = VideoFingerprint()
        # Frames 0-3 identical, 4-7 fully flipped.
        a = b"\x00" * 64
        b = b"\x00" * 32 + b"\xff" * 32
        assert backend.similarity(a, b) == pytest.approx(0.5)

    def test_similarity_within_per_frame_tolerance(self):
        """A single bit-flip per frame stays under the 6/64 tolerance →
        all 8 frames match → similarity 1.0."""
        backend = VideoFingerprint()
        a = b"\x00" * 64
        # Flip one bit in each 8-byte frame group. 1/64 distance per
        # frame is well under the 6/64 tolerance.
        b_arr = bytearray(a)
        for i in range(8):
            b_arr[i * 8] = 0x01
        assert backend.similarity(a, bytes(b_arr)) == pytest.approx(1.0)

    def test_similarity_malformed_returns_zero(self):
        backend = VideoFingerprint()
        # Wrong total length → 0.0.
        assert backend.similarity(b"\x00" * 32, b"\x00" * 64) == 0.0
        assert backend.similarity(b"", b"\x00" * 64) == 0.0
        assert backend.similarity(b"\x00" * 64, b"") == 0.0
