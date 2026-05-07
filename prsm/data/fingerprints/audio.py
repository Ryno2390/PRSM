"""
PRSM-PROV-1 Item 4 T4.3 — AudioFingerprint backend.

Chromaprint-based acoustic-fingerprint backend via ``pyacoustid`` (the
Python binding for the ``fpcalc`` Chromaprint CLI). Chromaprint produces
a sequence of 32-bit integer "subfingerprints" sampled at ~8 Hz across
the audio's duration; the canonical similarity metric is the bit-error
rate (BER) of XOR'd 32-bit frames over the longest aligned window.

Threshold (per ``prsm/data/dedup_thresholds.yaml::defaults.audio-chromaprint``):
duplicate if similarity ≥ 0.92, derivative if ≥ 0.75. Robust to
encoding format (MP3 vs FLAC vs WAV), bit-rate changes, and minor
re-mastering; intentionally NOT robust to major edits, time-stretching
beyond ~5%, or pitch shifts (those are different content for PRSM
dedup).

Implementation notes:
  - The ``acoustid.fingerprint_file`` API requires a filesystem path,
    so we write the bytes to a temporary file scoped to the call. The
    ``fpcalc`` binary handles all decoding (MP3, FLAC, OGG, WAV, M4A,
    etc.) via FFmpeg internally.
  - The fingerprint payload is the *raw 32-bit-int sequence packed as
    big-endian uint32 bytes*, NOT the base64-encoded string that
    ``pyacoustid`` returns by default. The packed format is more
    compact and lets us do XOR/popcount directly without re-decoding
    on every similarity call.
  - Empty / corrupt audio returns None so callers fall through to
    BYTE_HASH.
"""

from __future__ import annotations

import logging
import os
import struct
import tempfile
from typing import List, Optional, Tuple

from prsm.data.fingerprints.base import (
    BinaryFingerprint,
    FingerprintKind,
    FingerprintRecord,
)

logger = logging.getLogger(__name__)


# Each Chromaprint subfingerprint is 32 bits.
_CHROMAPRINT_FRAME_BITS = 32
_CHROMAPRINT_FRAME_BYTES = _CHROMAPRINT_FRAME_BITS // 8
# Below this many frames the audio is too short to be a reliable
# fingerprint — Chromaprint needs ≥ ~7 seconds to produce stable hashes
# at the default 8 Hz frame rate. We surface that as None so the upload
# path falls through to BYTE_HASH for very short clips.
_MIN_FRAMES_FOR_RELIABLE_FP = 8


class AudioFingerprint(BinaryFingerprint):
    """Chromaprint acoustic-fingerprint backend for audio content."""

    KIND = FingerprintKind.AUDIO_CHROMAPRINT

    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        if not content:
            return None
        try:
            import acoustid  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning(
                "AudioFingerprint backend unavailable: %s. Install with: "
                "pip install pyacoustid (also requires the chromaprint "
                "fpcalc binary on PATH; brew install chromaprint or "
                "apt-get install libchromaprint-tools)",
                exc,
            )
            return None

        # acoustid.fingerprint_file() needs a filesystem path. Use the
        # filename's suffix as a hint to fpcalc/FFmpeg about the
        # container — without it, formats like .m4a sometimes fail to
        # demux. Default to .audio if no suffix is supplied.
        suffix = ""
        if filename:
            suffix = os.path.splitext(filename)[1] or ""
        if not suffix:
            suffix = ".audio"

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                duration, fp = acoustid.fingerprint_file(tmp_path)
            except acoustid.FingerprintGenerationError as exc:
                logger.debug(
                    "AudioFingerprint.compute fpcalc rejected (filename=%s): %s",
                    filename, exc,
                )
                return None
            except Exception as exc:  # noqa: BLE001
                # fpcalc binary missing, FFmpeg backend missing,
                # truncated/corrupt audio — log at DEBUG so a
                # busted-content upload doesn't spam the logs.
                logger.debug(
                    "AudioFingerprint.compute unexpected failure "
                    "(filename=%s): %s",
                    filename, exc,
                )
                return None

            frames = _decode_chromaprint(fp)
            if frames is None or len(frames) < _MIN_FRAMES_FOR_RELIABLE_FP:
                logger.debug(
                    "AudioFingerprint.compute insufficient frames "
                    "(filename=%s, frames=%s)",
                    filename, len(frames) if frames is not None else None,
                )
                return None

            # Pack as big-endian uint32 sequence.
            payload = b"".join(struct.pack(">I", f & 0xFFFFFFFF) for f in frames)
            return FingerprintRecord(kind=self.KIND, payload=payload)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def similarity(self, a: bytes, b: bytes) -> float:
        """Bit-error-rate similarity over the longest aligned window.

        Both payloads are sequences of 32-bit big-endian uint32 frames.
        Different-length payloads are aligned at offset 0 and compared
        over the shorter of the two — Chromaprint sequences naturally
        share a prefix when the audio shares a prefix.

        Returns 0.0 when:
          - Either payload is malformed (length not a multiple of 4)
          - Either payload is empty
          - The two payloads have zero comparable frames
        """
        if not a or not b:
            return 0.0
        if len(a) % _CHROMAPRINT_FRAME_BYTES != 0:
            return 0.0
        if len(b) % _CHROMAPRINT_FRAME_BYTES != 0:
            return 0.0

        frames_a = _unpack_frames(a)
        frames_b = _unpack_frames(b)
        n = min(len(frames_a), len(frames_b))
        if n == 0:
            return 0.0

        diff_bits = 0
        total_bits = n * _CHROMAPRINT_FRAME_BITS
        for x, y in zip(frames_a[:n], frames_b[:n]):
            diff_bits += (x ^ y).bit_count()
        return 1.0 - (diff_bits / total_bits)


# ──────────────────────────────────────────────────────────────────────
# Chromaprint payload helpers
# ──────────────────────────────────────────────────────────────────────


def _decode_chromaprint(fp) -> Optional[List[int]]:
    """Convert pyacoustid's chromaprint output to a list of 32-bit ints.

    pyacoustid returns either a base64-encoded compressed string OR a
    raw list of ints depending on which API the caller hit. We accept
    both shapes; ``fingerprint_file`` returns compressed bytes by
    default, which we decode via the chromaprint module.
    """
    if fp is None:
        return None
    if isinstance(fp, (list, tuple)):
        return [int(x) & 0xFFFFFFFF for x in fp]
    # Bytes / str — compressed Chromaprint base64. The chromaprint C
    # library exposes a decode function that pyacoustid binds to.
    try:
        import chromaprint  # type: ignore[import-not-found]
        raw = fp.encode("utf-8") if isinstance(fp, str) else fp
        decoded = chromaprint.decode_fingerprint(raw)
        if decoded is None:
            return None
        # decode_fingerprint returns (frames, version_int) on success.
        if isinstance(decoded, tuple) and len(decoded) >= 1:
            frames = decoded[0]
        else:
            frames = decoded
        return [int(x) & 0xFFFFFFFF for x in frames]
    except ImportError:
        # The chromaprint Python package wasn't shipped alongside
        # pyacoustid — uncommon but possible. Fall back to None and
        # the caller surfaces "fingerprint not available".
        logger.debug(
            "chromaprint module unavailable for decoding compressed "
            "fingerprint; install with: pip install chromaprint",
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "AudioFingerprint chromaprint decode failed: %s", exc,
        )
        return None


def _unpack_frames(payload: bytes) -> List[int]:
    """Inverse of the packing in compute(): big-endian uint32 sequence."""
    n = len(payload) // _CHROMAPRINT_FRAME_BYTES
    return list(struct.unpack(f">{n}I", payload))


__all__ = ["AudioFingerprint"]
