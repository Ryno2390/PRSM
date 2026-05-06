"""
PRSM-PROV-1 Item 4 T4.1 — content-type detection + ABC tests.

Covers:
  - detect_content_kind() priority order (mime_hint > magic > suffix
    > magic-bytes > text-decode > BYTE_HASH).
  - Real magic bytes for the common formats (PNG / JPEG / GIF / WebP /
    WAV / MP3 / MP4 / FLAC / HDF5 / Parquet / NumPy / ID3 / EBML).
  - Filename suffix lookup for image / audio / video / structural /
    text formats.
  - UTF-8 text fallback respects the 50-char threshold (matches
    ContentUploader's existing _MIN_EMBEDDING_CHARS gate).
  - BYTE_HASH terminal fallback for unidentifiable binary content.
  - FingerprintKind enum values match the threshold-resolver YAML keys.
"""

from __future__ import annotations

import pytest

from prsm.data.fingerprints import (
    BinaryFingerprint,
    FingerprintKind,
    detect_content_kind,
)


# ---- enum values match YAML keys ----------------------------------


def test_fingerprint_kind_values_match_yaml_keys():
    """The string values of FingerprintKind MUST match keys in
    dedup_thresholds.yaml::defaults so the resolver round-trips."""
    expected_yaml_keys = {
        "text-vector",
        "image-phash",
        "audio-chromaprint",
        "video-multihash",
        "structural",
        "byte-hash",
    }
    actual = {k.value for k in FingerprintKind}
    assert actual == expected_yaml_keys


# ---- mime_hint priority -------------------------------------------


def test_mime_hint_image_returns_image_phash():
    assert detect_content_kind(
        b"anything", mime_hint="image/jpeg",
    ) == FingerprintKind.IMAGE_PHASH


def test_mime_hint_audio_returns_audio_chromaprint():
    assert detect_content_kind(
        b"x", mime_hint="audio/mpeg",
    ) == FingerprintKind.AUDIO_CHROMAPRINT


def test_mime_hint_video_returns_video_multihash():
    assert detect_content_kind(
        b"x", mime_hint="video/mp4",
    ) == FingerprintKind.VIDEO_MULTIHASH


def test_mime_hint_hdf5_returns_structural():
    assert detect_content_kind(
        b"x", mime_hint="application/x-hdf5",
    ) == FingerprintKind.STRUCTURAL


def test_mime_hint_parquet_returns_structural():
    assert detect_content_kind(
        b"x", mime_hint="application/vnd.apache.parquet",
    ) == FingerprintKind.STRUCTURAL


def test_mime_hint_text_returns_text_vector():
    assert detect_content_kind(
        b"x", mime_hint="text/plain",
    ) == FingerprintKind.TEXT_VECTOR


def test_mime_hint_overrides_filename():
    """Caller-provided MIME wins over filename suffix."""
    assert detect_content_kind(
        b"random",
        filename="actually-image.jpg",
        mime_hint="text/plain",
    ) == FingerprintKind.TEXT_VECTOR


def test_mime_hint_case_insensitive():
    assert detect_content_kind(
        b"x", mime_hint="IMAGE/PNG",
    ) == FingerprintKind.IMAGE_PHASH


# ---- magic-bytes detection ----------------------------------------


def test_png_magic_bytes_detected():
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    assert detect_content_kind(png_header) == FingerprintKind.IMAGE_PHASH


def test_jpeg_magic_bytes_detected():
    jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    assert detect_content_kind(jpeg_header) == FingerprintKind.IMAGE_PHASH


def test_gif87_magic_bytes_detected():
    assert detect_content_kind(
        b"GIF87a" + b"\x00" * 100,
    ) == FingerprintKind.IMAGE_PHASH


def test_gif89_magic_bytes_detected():
    assert detect_content_kind(
        b"GIF89a" + b"\x00" * 100,
    ) == FingerprintKind.IMAGE_PHASH


def test_webp_riff_disambiguates_to_image():
    """RIFF + WEBP at offset 8 → image-phash."""
    webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 50
    assert detect_content_kind(webp) == FingerprintKind.IMAGE_PHASH


def test_wav_riff_disambiguates_to_audio():
    """RIFF + WAVE at offset 8 → audio-chromaprint."""
    wav = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 50
    assert detect_content_kind(wav) == FingerprintKind.AUDIO_CHROMAPRINT


def test_mp3_id3_magic_detected():
    mp3 = b"ID3" + b"\x00" * 100
    assert detect_content_kind(mp3) == FingerprintKind.AUDIO_CHROMAPRINT


def test_flac_magic_detected():
    flac = b"fLaC" + b"\x00" * 100
    assert detect_content_kind(flac) == FingerprintKind.AUDIO_CHROMAPRINT


def test_ogg_magic_detected():
    ogg = b"OggS" + b"\x00" * 100
    assert detect_content_kind(ogg) == FingerprintKind.AUDIO_CHROMAPRINT


def test_mp4_ftyp_at_offset_4_detected():
    """MP4 / MOV ftyp box starts at offset 4."""
    mp4 = b"\x00\x00\x00\x20" + b"ftyp" + b"isom" + b"\x00" * 100
    assert detect_content_kind(mp4) == FingerprintKind.VIDEO_MULTIHASH


def test_matroska_ebml_magic_detected():
    mkv = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
    assert detect_content_kind(mkv) == FingerprintKind.VIDEO_MULTIHASH


def test_hdf5_magic_detected():
    hdf5 = b"\x89HDF\r\n\x1a\n" + b"\x00" * 100
    assert detect_content_kind(hdf5) == FingerprintKind.STRUCTURAL


def test_parquet_magic_detected():
    parquet = b"PAR1" + b"\x00" * 100
    assert detect_content_kind(parquet) == FingerprintKind.STRUCTURAL


def test_arrow_magic_detected():
    arrow = b"ARROW1" + b"\x00" * 100
    assert detect_content_kind(arrow) == FingerprintKind.STRUCTURAL


def test_numpy_npy_magic_detected():
    npy = b"\x93NUMPY" + b"\x00" * 100
    assert detect_content_kind(npy) == FingerprintKind.STRUCTURAL


def test_pdf_returns_text_vector():
    """PDF gets text-vector — text extraction handles dedup."""
    pdf = b"%PDF-1.4" + b"\x00" * 100
    assert detect_content_kind(pdf) == FingerprintKind.TEXT_VECTOR


# ---- filename suffix detection ------------------------------------


def test_filename_suffix_image():
    # Provide non-magic-byte content so we exercise the suffix path
    # (not magic detection).
    blob = b"\x00" * 100
    assert detect_content_kind(
        blob, filename="photo.png",
    ) == FingerprintKind.IMAGE_PHASH
    assert detect_content_kind(
        blob, filename="photo.JPEG",
    ) == FingerprintKind.IMAGE_PHASH


def test_filename_suffix_audio():
    blob = b"\x00" * 100
    assert detect_content_kind(
        blob, filename="track.flac",
    ) == FingerprintKind.AUDIO_CHROMAPRINT


def test_filename_suffix_video():
    blob = b"\x00" * 100
    assert detect_content_kind(
        blob, filename="clip.mkv",
    ) == FingerprintKind.VIDEO_MULTIHASH


def test_filename_suffix_structural_h5():
    blob = b"\x00" * 100
    assert detect_content_kind(
        blob, filename="dataset.h5",
    ) == FingerprintKind.STRUCTURAL
    assert detect_content_kind(
        blob, filename="dataset.npz",
    ) == FingerprintKind.STRUCTURAL


def test_filename_suffix_text():
    blob = b"\x00" * 100
    assert detect_content_kind(
        blob, filename="notes.md",
    ) == FingerprintKind.TEXT_VECTOR
    assert detect_content_kind(
        blob, filename="script.py",
    ) == FingerprintKind.TEXT_VECTOR


def test_unknown_suffix_falls_through():
    """A weird suffix doesn't short-circuit the rest of the
    detection chain."""
    text = ("hello world " * 10).encode("utf-8")  # >= 50 chars
    assert detect_content_kind(
        text, filename="doc.qqq",  # not in suffix table
    ) == FingerprintKind.TEXT_VECTOR


# ---- UTF-8 text fallback ------------------------------------------


def test_long_utf8_text_returns_text_vector():
    text = ("This is some readable English text. " * 5).encode("utf-8")
    assert detect_content_kind(text) == FingerprintKind.TEXT_VECTOR


def test_short_utf8_text_falls_to_byte_hash():
    """Below the 50-char threshold, even text falls to BYTE_HASH —
    the embedding API would skip it anyway."""
    short = b"hi"
    assert detect_content_kind(short) == FingerprintKind.BYTE_HASH


def test_empty_content_returns_byte_hash():
    assert detect_content_kind(b"") == FingerprintKind.BYTE_HASH


def test_random_binary_returns_byte_hash():
    """Bytes that aren't a known magic and don't decode as text."""
    blob = bytes(range(256)) * 2  # 512 bytes of all byte values —
                                   # not valid UTF-8 because of bytes
                                   # 0x80+ that don't form valid
                                   # multi-byte sequences.
    assert detect_content_kind(blob) == FingerprintKind.BYTE_HASH


# ---- ABC contract -------------------------------------------------


def test_binary_fingerprint_is_abstract():
    """BinaryFingerprint cannot be instantiated directly — concrete
    backends MUST subclass and implement compute() + similarity()."""
    with pytest.raises(TypeError):
        BinaryFingerprint()  # type: ignore[abstract]


def test_subclass_can_be_instantiated_when_complete():
    class _Stub(BinaryFingerprint):
        KIND = FingerprintKind.IMAGE_PHASH

        def compute(self, content, *, filename=None):
            return None

        def similarity(self, a, b):
            return 0.0

    stub = _Stub()
    assert stub.compute(b"") is None
    assert stub.similarity(b"", b"") == 0.0
