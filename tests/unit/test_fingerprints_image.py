"""
PRSM-PROV-1 Item 4 T4.2 — ImageFingerprint backend tests.

Covers:
  - compute() round-trip on real (synthesized) PIL images.
  - JPEG re-encoding tolerance: same image at quality=50 vs quality=95
    produces identical (or near-identical) hashes.
  - Different images produce different hashes (cross-image distance).
  - Truncated / corrupt input returns None instead of raising.
  - Empty bytes returns None.
  - similarity() returns 1.0 for identical payloads, < 1.0 for
    differing, 0.0 for malformed payloads.
  - Payload format: 8 bytes (64-bit hash), packed bigendian-style.
  - KIND attribute matches FingerprintKind.IMAGE_PHASH.
"""

from __future__ import annotations

import math
from io import BytesIO

import pytest

# Skip the entire module if imagehash / PIL aren't available — keeps
# CI green on hosts without the optional deps.
imagehash = pytest.importorskip("imagehash")
PIL_Image = pytest.importorskip("PIL.Image")

from prsm.data.fingerprints import FingerprintKind  # noqa: E402
from prsm.data.fingerprints.base import FingerprintRecord  # noqa: E402
from prsm.data.fingerprints.image import (  # noqa: E402
    ImageFingerprint,
    _PHASH_PAYLOAD_BYTES,
    _PHASH_TOTAL_BITS,
)


# ---- helpers -------------------------------------------------------


def _solid_color_png(color: tuple, size: tuple = (128, 128)) -> bytes:
    img = PIL_Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _textured_image(size: tuple = (256, 256)) -> "PIL_Image.Image":
    """Synthesize a textured image with non-trivial DCT energy.

    A pure linear gradient is a pathological pHash input — its DCT
    energy concentrates in a handful of coefficients sitting right at
    the median boundary, so JPEG quantization tips many bits and
    produces high Hamming distance for visually-identical content.
    Mixing a gradient with a checker pattern + sine waves spreads
    energy through the DCT spectrum and makes pHash stable under
    re-encoding (matches real-photo behavior).
    """
    img = PIL_Image.new("RGB", size)
    px = img.load()
    w, h = size
    for x in range(w):
        for y in range(h):
            check = ((x // 16) + (y // 16)) % 2
            sine = int(40 * (math.sin(x * 0.1) + math.cos(y * 0.1)))
            r = (x * 200 // w + check * 60 + sine) % 256
            g = (y * 200 // h + check * 30 - sine) % 256
            b = ((x * y) // 16 + check * 100) % 256
            px[x, y] = (r, g, b)
    return img


def _textured_png(size: tuple = (256, 256)) -> bytes:
    buf = BytesIO()
    _textured_image(size).save(buf, format="PNG")
    return buf.getvalue()


def _textured_jpeg(quality: int, size: tuple = (256, 256)) -> bytes:
    buf = BytesIO()
    _textured_image(size).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---- KIND attribute -----------------------------------------------


def test_kind_is_image_phash():
    assert ImageFingerprint.KIND == FingerprintKind.IMAGE_PHASH


def test_instance_kind_matches():
    fp = ImageFingerprint()
    assert fp.KIND == FingerprintKind.IMAGE_PHASH


# ---- compute() basic ----------------------------------------------


def test_compute_returns_fingerprint_record():
    fp = ImageFingerprint()
    blob = _textured_png()
    record = fp.compute(blob)
    assert record is not None
    assert isinstance(record, FingerprintRecord)
    assert record.kind == FingerprintKind.IMAGE_PHASH
    assert isinstance(record.payload, bytes)


def test_compute_payload_length_is_8_bytes():
    fp = ImageFingerprint()
    record = fp.compute(_textured_png())
    assert record is not None
    assert len(record.payload) == _PHASH_PAYLOAD_BYTES == 8
    assert _PHASH_TOTAL_BITS == 64


def test_compute_is_deterministic():
    fp = ImageFingerprint()
    blob = _textured_png()
    r1 = fp.compute(blob)
    r2 = fp.compute(blob)
    assert r1 is not None and r2 is not None
    assert r1.payload == r2.payload


def test_compute_different_images_different_payloads():
    fp = ImageFingerprint()
    grad = fp.compute(_textured_png())
    red = fp.compute(_solid_color_png((255, 0, 0)))
    assert grad is not None and red is not None
    assert grad.payload != red.payload


# ---- compute() failure paths --------------------------------------


def test_empty_content_returns_none():
    fp = ImageFingerprint()
    assert fp.compute(b"") is None


def test_corrupt_content_returns_none():
    fp = ImageFingerprint()
    # Bytes that look nothing like an image.
    assert fp.compute(b"this is not an image at all" * 10) is None


def test_truncated_png_returns_none():
    fp = ImageFingerprint()
    full = _textured_png()
    # Keep only the PNG signature + IHDR header — strip out the actual
    # image data. PIL will raise when forced to load.
    truncated = full[:30]
    result = fp.compute(truncated)
    assert result is None


def test_compute_does_not_raise_on_garbage_with_filename_hint():
    """Even with filename hint suggesting an image, garbage bytes must
    return None rather than propagate an exception."""
    fp = ImageFingerprint()
    result = fp.compute(b"\x00" * 200, filename="evil.png")
    assert result is None


# ---- compute() robustness: re-encoding ----------------------------


def test_jpeg_quality_change_preserves_phash():
    """The whole point of pHash: q=50 vs q=95 of same image should
    match (Hamming distance very small, similarity very high)."""
    fp = ImageFingerprint()
    q95 = fp.compute(_textured_jpeg(quality=95))
    q50 = fp.compute(_textured_jpeg(quality=50))
    assert q95 is not None and q50 is not None
    sim = fp.similarity(q95.payload, q50.payload)
    # In practice this is almost always 1.0 for synthetic gradients,
    # but we allow 1-bit drift for robustness.
    assert sim >= (1.0 - 1.0 / _PHASH_TOTAL_BITS), (
        f"JPEG q95 vs q50 pHash diverged unexpectedly (sim={sim})"
    )


def test_png_to_jpeg_reencode_high_similarity():
    """Same image content, different container format — should still
    score above the dedup threshold (≥ 0.90)."""
    fp = ImageFingerprint()
    png = fp.compute(_textured_png())
    jpeg = fp.compute(_textured_jpeg(quality=85))
    assert png is not None and jpeg is not None
    sim = fp.similarity(png.payload, jpeg.payload)
    assert sim >= 0.90, (
        f"PNG ↔ JPEG re-encode similarity {sim} below dedup threshold"
    )


def test_resize_preserves_high_similarity():
    """Same content, downscaled — pHash must remain stable. Render a
    fixed source then pass it through PIL.Image.resize so the visual
    content is the same; the procedural texture function would
    generate different patterns at different sizes."""
    fp = ImageFingerprint()
    src = _textured_image(size=(256, 256))
    buf_big = BytesIO()
    src.save(buf_big, format="PNG")
    buf_small = BytesIO()
    src.resize((64, 64)).save(buf_small, format="PNG")
    big = fp.compute(buf_big.getvalue())
    small = fp.compute(buf_small.getvalue())
    assert big is not None and small is not None
    sim = fp.similarity(big.payload, small.payload)
    assert sim >= 0.90, f"Resize similarity {sim} below dedup threshold"


# ---- similarity() -------------------------------------------------


def test_similarity_identical_payloads_is_one():
    fp = ImageFingerprint()
    payload = b"\xff" * _PHASH_PAYLOAD_BYTES
    assert fp.similarity(payload, payload) == 1.0


def test_similarity_completely_inverted_is_zero():
    fp = ImageFingerprint()
    a = b"\x00" * _PHASH_PAYLOAD_BYTES
    b = b"\xff" * _PHASH_PAYLOAD_BYTES
    assert fp.similarity(a, b) == 0.0


def test_similarity_one_bit_flip_is_one_over_total_bits():
    fp = ImageFingerprint()
    a = b"\x00" * _PHASH_PAYLOAD_BYTES
    b = b"\x01" + b"\x00" * (_PHASH_PAYLOAD_BYTES - 1)
    expected = 1.0 - (1.0 / _PHASH_TOTAL_BITS)
    assert fp.similarity(a, b) == pytest.approx(expected)


def test_similarity_returns_value_in_unit_interval():
    fp = ImageFingerprint()
    p1 = fp.compute(_textured_png())
    p2 = fp.compute(_solid_color_png((0, 255, 0)))
    assert p1 is not None and p2 is not None
    sim = fp.similarity(p1.payload, p2.payload)
    assert 0.0 <= sim <= 1.0


def test_similarity_malformed_payload_a_returns_zero():
    fp = ImageFingerprint()
    bad = b"\x00" * 4  # too short
    good = b"\x00" * _PHASH_PAYLOAD_BYTES
    assert fp.similarity(bad, good) == 0.0


def test_similarity_malformed_payload_b_returns_zero():
    fp = ImageFingerprint()
    good = b"\x00" * _PHASH_PAYLOAD_BYTES
    bad = b"\x00" * (_PHASH_PAYLOAD_BYTES + 3)  # too long
    assert fp.similarity(good, bad) == 0.0


def test_similarity_both_empty_returns_zero():
    """Two empty payloads are NOT trivially similar — they're
    both malformed, so the score is 0.0 (anti-grief gate)."""
    fp = ImageFingerprint()
    assert fp.similarity(b"", b"") == 0.0


def test_similarity_is_symmetric():
    fp = ImageFingerprint()
    a = fp.compute(_textured_png())
    b = fp.compute(_solid_color_png((0, 0, 255)))
    assert a is not None and b is not None
    assert fp.similarity(a.payload, b.payload) == fp.similarity(
        b.payload, a.payload
    )


# ---- behavior under ABC -------------------------------------------


def test_image_fingerprint_is_concrete():
    """Subclass of BinaryFingerprint must instantiate cleanly (T4.1
    rejected the ABC; this confirms our subclass overrides both
    abstract methods)."""
    fp = ImageFingerprint()
    assert callable(fp.compute)
    assert callable(fp.similarity)
