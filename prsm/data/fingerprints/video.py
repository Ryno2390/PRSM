"""
PRSM-PROV-1 Item 4 T4.4 — VideoFingerprint backend.

Multi-keyframe perceptual-hash backend. Decodes the video via PyAV
(``av``), samples 8 evenly-spaced frames across the duration, computes
an 8x8 pHash for each, and packs the 8 hashes into a fixed 64-byte
payload. Similarity is the fraction of keyframes whose pHash distance
is within the per-frame matching tolerance — i.e. a video matches if
≥ N of the 8 keyframes match.

Threshold (per ``prsm/data/dedup_thresholds.yaml::defaults.video-multihash``):
duplicate iff ≥ 7/8 keyframes match (similarity ≥ 0.875), derivative
iff ≥ 5/8 (≥ 0.625). Robust to re-encoding / bit-rate changes / minor
trims at the head or tail; intentionally NOT robust to re-cuts that
shuffle scenes (those are different content for PRSM dedup).

Implementation notes:
  - Sampling at fixed time fractions (1/9, 2/9, …, 8/9 of duration) is
    cheaper than detecting actual scene-cut keyframes and keeps the
    fingerprint stable across small trim edits at either end. PyAV
    seeks via container-level timestamps; non-seekable streams fall
    back to sequential decode-and-skip.
  - Each frame is converted to an 8x8 grayscale grid via PyAV's
    rescaler, then pHash'd via the same DCT-based algorithm as the
    image backend (without depending on imagehash here — we want this
    backend to work on hosts that don't have imagehash installed).
  - Per-frame pHash matching uses Hamming distance ≤ 6/64 (matching
    image-phash's per-frame tolerance). The video-level similarity is
    the fraction of frames that pass that bar.
  - Empty / single-frame / corrupt video returns None so callers fall
    through to BYTE_HASH.
"""

from __future__ import annotations

import logging
import os
import struct
import tempfile
from typing import List, Optional

from prsm.data.fingerprints.base import (
    BinaryFingerprint,
    FingerprintKind,
    FingerprintRecord,
)

logger = logging.getLogger(__name__)


# Keyframe count + per-frame pHash size. Total payload = 8 frames * 8
# bytes/frame = 64 bytes (the multihash convention is N pHash_bytes).
_NUM_KEYFRAMES = 8
_PHASH_GRID = 8  # 8x8 = 64-bit pHash per frame
_PHASH_BITS = _PHASH_GRID * _PHASH_GRID
_PHASH_BYTES = _PHASH_BITS // 8
_PAYLOAD_BYTES = _NUM_KEYFRAMES * _PHASH_BYTES  # 64 bytes total

# Per-frame Hamming-distance tolerance (matches image-phash duplicate
# threshold of ≤ 6 / 64 bits ≈ 0.094 distance ⇒ ≥ 0.906 similarity).
_PER_FRAME_HAMMING_TOLERANCE = 6


class VideoFingerprint(BinaryFingerprint):
    """8-keyframe pHash backend for video content."""

    KIND = FingerprintKind.VIDEO_MULTIHASH

    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        if not content:
            return None
        try:
            import av  # type: ignore[import-not-found]
            import numpy as np
        except ImportError as exc:
            logger.warning(
                "VideoFingerprint backend unavailable: %s. Install with: "
                "pip install av (PyAV — bundles its own FFmpeg)",
                exc,
            )
            return None

        # PyAV needs a filesystem path (or a seekable file-like). The
        # filename suffix carries container hints to FFmpeg's demuxer;
        # default to .video when absent.
        suffix = ""
        if filename:
            suffix = os.path.splitext(filename)[1] or ""
        if not suffix:
            suffix = ".video"

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                frames = _sample_keyframes(tmp_path, _NUM_KEYFRAMES, np)
            except av.FFmpegError as exc:
                logger.debug(
                    "VideoFingerprint.compute decode rejected (filename=%s): %s",
                    filename, exc,
                )
                return None
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "VideoFingerprint.compute unexpected failure "
                    "(filename=%s): %s",
                    filename, exc,
                )
                return None

            if frames is None or len(frames) != _NUM_KEYFRAMES:
                logger.debug(
                    "VideoFingerprint.compute insufficient keyframes "
                    "(filename=%s, count=%s)",
                    filename, len(frames) if frames is not None else None,
                )
                return None

            # Pack each frame's pHash bytes back-to-back; the total is
            # always exactly _PAYLOAD_BYTES.
            payload = b"".join(_phash_8x8(frame, np) for frame in frames)
            if len(payload) != _PAYLOAD_BYTES:
                return None
            return FingerprintRecord(kind=self.KIND, payload=payload)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def similarity(self, a: bytes, b: bytes) -> float:
        """Fraction of keyframes whose pHash matches within tolerance.

        Each side is 8 frames * 8 bytes = 64 bytes. Returns 0.0 on any
        malformed payload.
        """
        if len(a) != _PAYLOAD_BYTES or len(b) != _PAYLOAD_BYTES:
            return 0.0
        matches = 0
        for i in range(_NUM_KEYFRAMES):
            start = i * _PHASH_BYTES
            end = start + _PHASH_BYTES
            hamming = 0
            for x, y in zip(a[start:end], b[start:end]):
                hamming += (x ^ y).bit_count()
            if hamming <= _PER_FRAME_HAMMING_TOLERANCE:
                matches += 1
        return matches / _NUM_KEYFRAMES


# ──────────────────────────────────────────────────────────────────────
# Frame sampling + pHash
# ──────────────────────────────────────────────────────────────────────


def _sample_keyframes(path: str, count: int, np) -> Optional[List]:
    """Open the video at ``path`` and return ``count`` evenly-spaced
    frames as numpy uint8 grayscale 8x8 arrays.

    Sampling strategy: target time = (i+1)/(count+1) of duration for i
    in [0, count). When the container reports duration we seek directly;
    otherwise we fall back to sequential decode + skip with a frame-count
    estimate.
    """
    import av  # type: ignore[import-not-found]

    container = av.open(path)
    try:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]

        # Target sample timestamps in stream time-base units.
        if stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            # av.time_base is microseconds.
            duration_sec = container.duration / 1_000_000.0
        else:
            duration_sec = 0.0

        if duration_sec <= 0:
            return _sample_keyframes_sequential(container, stream, count, np)

        targets_sec = [
            duration_sec * (i + 1) / (count + 1) for i in range(count)
        ]

        sampled: List = []
        for target in targets_sec:
            try:
                # PyAV seeks to the nearest keyframe ≤ target, then we
                # decode forward until we find a frame at-or-past
                # target time.
                offset = int(target / float(stream.time_base))
                container.seek(offset, stream=stream, any_frame=False)

                frame = None
                for decoded in container.decode(stream):
                    if decoded.time is not None and decoded.time >= target:
                        frame = decoded
                        break
                    frame = decoded  # fallback to last decoded
                if frame is None:
                    return None
                sampled.append(_frame_to_8x8_gray(frame, np))
            except (StopIteration, av.FFmpegError):
                return None

        return sampled
    finally:
        container.close()


def _sample_keyframes_sequential(container, stream, count: int, np) -> Optional[List]:
    """Fallback when container has no usable duration.

    Decode every frame; pick ``count`` evenly-spaced ones from the full
    list. Memory cost is bounded by the count of decoded frames; only
    used for short or stream-only inputs.
    """
    frames_seen: List = []
    try:
        for decoded in container.decode(stream):
            frames_seen.append(decoded)
    except Exception:  # noqa: BLE001
        if not frames_seen:
            return None

    n = len(frames_seen)
    if n < count:
        return None

    indices = [int((i + 1) * n / (count + 1)) for i in range(count)]
    return [_frame_to_8x8_gray(frames_seen[i], np) for i in indices]


def _frame_to_8x8_gray(frame, np):
    """Convert an av.VideoFrame to a 32x32 uint8 grayscale ndarray.

    32x32 is the input grid for an 8x8 pHash (DCT keeps the
    low-frequency 8x8 block). PyAV's reformat handles colour-space
    conversion + scaling in a single pass.
    """
    reformatted = frame.reformat(width=32, height=32, format="gray")
    arr = reformatted.to_ndarray()  # shape (32, 32), dtype uint8
    return arr.astype(np.float32)


def _phash_8x8(arr32: "object", np) -> bytes:
    """Compute an 8x8 pHash from a 32x32 float ndarray.

    Standard DCT-based perceptual hash:
      1. 2-D DCT of the 32x32 grayscale image.
      2. Take the top-left 8x8 block (low frequencies).
      3. Drop the DC coefficient; threshold the remaining 63 against
         the median; pack 64 bits with the threshold of the DC bit
         set to the median itself.
    """
    # Fall back to a stable scipy/numpy DCT — pyav doesn't ship one,
    # so we use scipy.fftpack if present, else hand-roll via numpy fft.
    try:
        from scipy.fftpack import dct  # type: ignore[import-not-found]
        dct_x = dct(dct(arr32, axis=0, norm="ortho"), axis=1, norm="ortho")
    except ImportError:
        # Manual DCT-II via numpy. Slower but dependency-free; keeps
        # the backend usable on lean installs.
        dct_x = _dct2_numpy(arr32, np)

    block = dct_x[:_PHASH_GRID, :_PHASH_GRID]
    # Drop the DC coefficient when computing the median — including it
    # skews the threshold heavily on solid-colour frames.
    flat = block.flatten()
    median = float(np.median(flat[1:]))
    bits = (flat > median).astype(np.uint8)
    # Pack big-endian, 8 bits per byte.
    return np.packbits(bits, bitorder="big").tobytes()


def _dct2_numpy(arr, np):
    """2-D DCT-II via numpy. Used when scipy isn't installed."""
    N = arr.shape[0]
    # Build the DCT-II basis matrix once (cached implicitly per N via
    # Python's interpreter cache; this function is called on small N
    # only — 32 — so the cost is irrelevant).
    n = np.arange(N)
    k = n.reshape(-1, 1)
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    norm = np.sqrt(2.0 / N)
    basis_normed = basis * norm
    basis_normed[0] *= 1.0 / np.sqrt(2.0)
    # Apply along both axes.
    return basis_normed @ arr @ basis_normed.T


__all__ = ["VideoFingerprint"]
