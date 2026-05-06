"""
PRSM-PROV-1 Item 4 T4.2 — ImageFingerprint backend.

Perceptual-hash (pHash) backend via the ``imagehash`` library at the
default 8x8 = 64-bit grid. The pHash is robust to JPEG re-encoding,
quality changes, and minor color-space drift; it is intentionally NOT
robust to large crops or rotations (those are different content for
PRSM dedup purposes — see plan §3.2).

Threshold (per ``prsm/data/dedup_thresholds.yaml::defaults.image-phash``):
duplicate if Hamming distance ≤ 6 / 64 ≈ 0.094, i.e. similarity score
≥ 0.906. Enforcement lives in ``ThresholdResolver``; this backend just
produces the score.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

from prsm.data.fingerprints.base import (
    BinaryFingerprint,
    FingerprintKind,
    FingerprintRecord,
)

logger = logging.getLogger(__name__)


_PHASH_SIZE = 8
_PHASH_TOTAL_BITS = _PHASH_SIZE * _PHASH_SIZE
_PHASH_PAYLOAD_BYTES = _PHASH_TOTAL_BITS // 8


class ImageFingerprint(BinaryFingerprint):
    """Perceptual-hash backend for image content."""

    KIND = FingerprintKind.IMAGE_PHASH

    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        if not content:
            return None
        try:
            import imagehash  # type: ignore[import-not-found]
            import numpy as np
            from PIL import Image, UnidentifiedImageError  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning(
                "ImageFingerprint backend unavailable: %s. Install with: "
                "pip install imagehash Pillow",
                exc,
            )
            return None

        try:
            with Image.open(BytesIO(content)) as img:
                img.load()
                hash_obj = imagehash.phash(img, hash_size=_PHASH_SIZE)
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            logger.debug(
                "ImageFingerprint.compute skipped (filename=%s): %s",
                filename, exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "ImageFingerprint.compute unexpected failure (filename=%s): %s",
                filename, exc,
            )
            return None

        flat = np.asarray(hash_obj.hash, dtype=bool).flatten()
        if flat.size != _PHASH_TOTAL_BITS:
            logger.debug(
                "ImageFingerprint.compute unexpected hash size %d", flat.size,
            )
            return None
        payload = np.packbits(flat).tobytes()
        if len(payload) != _PHASH_PAYLOAD_BYTES:
            return None
        return FingerprintRecord(kind=self.KIND, payload=payload)

    def similarity(self, a: bytes, b: bytes) -> float:
        """Hamming-distance-based similarity in [0, 1]; identical → 1.0.

        Returns 0.0 when either payload is malformed (wrong length),
        which keeps a poisoned/truncated record from coincidentally
        scoring high against a real one.
        """
        if (
            len(a) != _PHASH_PAYLOAD_BYTES
            or len(b) != _PHASH_PAYLOAD_BYTES
        ):
            return 0.0
        hamming = 0
        for x, y in zip(a, b):
            hamming += (x ^ y).bit_count()
        return 1.0 - (hamming / _PHASH_TOTAL_BITS)
