"""
PRSM-PROV-1 Item 4 T4.1 — BinaryFingerprint ABC + content-type
detection helper.

Concrete fingerprint backends (image-pHash, audio-Chromaprint,
video-multihash, structural) implement ``BinaryFingerprint`` and
register against a ``FingerprintKind``.

The default implementation does NOT pull any of the fingerprint-library
deps (``imagehash`` / ``pyacoustid`` / ``pyav``) — those land in T4.2
through T4.5. T4.1 ships only the framework so the upload path can
dispatch to ``"text-vector"`` (the existing path) or ``"binary"``
(deferred to byte-hash) until a real backend is registered.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---- enum ----------------------------------------------------------


class FingerprintKind(str, Enum):
    """Identifies which fingerprint algorithm/backend to use.

    Values match the keys in
    ``prsm/data/dedup_thresholds.yaml::defaults`` so a kind resolves
    to a threshold without further translation.

    ``TEXT_VECTOR`` is the existing path (Items 1, 2, 5 — embedding
    via OpenAI / sentence-transformers). The other four are added by
    Item 4. ``BYTE_HASH`` is the always-available fallback for
    content types we can't fingerprint (random binary blobs, encrypted
    payloads, archives we don't unpack).
    """

    TEXT_VECTOR = "text-vector"
    IMAGE_PHASH = "image-phash"
    AUDIO_CHROMAPRINT = "audio-chromaprint"
    VIDEO_MULTIHASH = "video-multihash"
    STRUCTURAL = "structural"
    BYTE_HASH = "byte-hash"


# ---- ABC -----------------------------------------------------------


@dataclass(frozen=True)
class FingerprintRecord:
    """Output of ``BinaryFingerprint.compute()``.

    Each backend produces a single record per piece of content. The
    dedup query matches records of the same ``kind`` against each
    other using a kind-specific distance function (see plan §3.2).
    """

    kind: FingerprintKind
    payload: bytes  # backend-specific binary fingerprint


class BinaryFingerprint(ABC):
    """Pluggable fingerprint backend.

    Implementations should:
      - Be cheap to instantiate (deferred / lazy heavy imports inside
        ``compute()`` so a node missing a backend's deps can still
        construct the registry).
      - Return ``None`` from ``compute()`` when the content can't be
        fingerprinted by this backend (unsupported codec, truncated
        file, etc.) — callers fall through to the next backend or to
        ``BYTE_HASH``.
      - Implement ``similarity(a, b) -> float`` returning a score in
        ``[0, 1]`` with the SAME convention used by
        ``ThresholdResolver`` (1.0 = identical, lower = less similar).
    """

    KIND: FingerprintKind  # set by subclass

    @abstractmethod
    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        """Compute a fingerprint for ``content``. Return None when the
        backend can't process this input (callers fall through)."""

    @abstractmethod
    def similarity(self, a: bytes, b: bytes) -> float:
        """Score similarity between two fingerprint payloads of this
        backend's kind. MUST return in ``[0, 1]``; identical ⇒ 1.0."""


# ---- content-type detection ----------------------------------------


# MIME-type prefix → FingerprintKind mapping. Order matters — longer
# matches first so ``application/x-hdf5`` beats ``application/``.
_MIME_KIND_TABLE: tuple = (
    ("image/", FingerprintKind.IMAGE_PHASH),
    ("audio/", FingerprintKind.AUDIO_CHROMAPRINT),
    ("video/", FingerprintKind.VIDEO_MULTIHASH),
    # Structural scientific-data formats. Each requires the
    # StructuralFingerprint backend to be registered.
    ("application/x-hdf5", FingerprintKind.STRUCTURAL),
    ("application/x-parquet", FingerprintKind.STRUCTURAL),
    ("application/vnd.apache.parquet", FingerprintKind.STRUCTURAL),
    ("application/vnd.apache.arrow.file", FingerprintKind.STRUCTURAL),
    ("application/vnd.apache.arrow.stream", FingerprintKind.STRUCTURAL),
    ("application/x-numpy-data", FingerprintKind.STRUCTURAL),
    # Text falls through to TEXT_VECTOR (existing embedding path).
    ("text/", FingerprintKind.TEXT_VECTOR),
    ("application/json", FingerprintKind.TEXT_VECTOR),
    ("application/xml", FingerprintKind.TEXT_VECTOR),
    ("application/x-yaml", FingerprintKind.TEXT_VECTOR),
)


# Filename suffix → FingerprintKind. Used as fallback when MIME
# detection can't tell us anything (no python-magic installed and
# stdlib mimetypes returns None).
_SUFFIX_KIND_TABLE: dict = {
    # Image
    ".jpg": FingerprintKind.IMAGE_PHASH,
    ".jpeg": FingerprintKind.IMAGE_PHASH,
    ".png": FingerprintKind.IMAGE_PHASH,
    ".gif": FingerprintKind.IMAGE_PHASH,
    ".bmp": FingerprintKind.IMAGE_PHASH,
    ".webp": FingerprintKind.IMAGE_PHASH,
    ".tiff": FingerprintKind.IMAGE_PHASH,
    ".tif": FingerprintKind.IMAGE_PHASH,
    # Audio
    ".mp3": FingerprintKind.AUDIO_CHROMAPRINT,
    ".wav": FingerprintKind.AUDIO_CHROMAPRINT,
    ".flac": FingerprintKind.AUDIO_CHROMAPRINT,
    ".ogg": FingerprintKind.AUDIO_CHROMAPRINT,
    ".m4a": FingerprintKind.AUDIO_CHROMAPRINT,
    ".aac": FingerprintKind.AUDIO_CHROMAPRINT,
    # Video
    ".mp4": FingerprintKind.VIDEO_MULTIHASH,
    ".mov": FingerprintKind.VIDEO_MULTIHASH,
    ".webm": FingerprintKind.VIDEO_MULTIHASH,
    ".mkv": FingerprintKind.VIDEO_MULTIHASH,
    ".avi": FingerprintKind.VIDEO_MULTIHASH,
    # Structural scientific data
    ".h5": FingerprintKind.STRUCTURAL,
    ".hdf5": FingerprintKind.STRUCTURAL,
    ".parquet": FingerprintKind.STRUCTURAL,
    ".arrow": FingerprintKind.STRUCTURAL,
    ".npy": FingerprintKind.STRUCTURAL,
    ".npz": FingerprintKind.STRUCTURAL,
    # Text-like
    ".txt": FingerprintKind.TEXT_VECTOR,
    ".md": FingerprintKind.TEXT_VECTOR,
    ".rst": FingerprintKind.TEXT_VECTOR,
    ".py": FingerprintKind.TEXT_VECTOR,
    ".js": FingerprintKind.TEXT_VECTOR,
    ".ts": FingerprintKind.TEXT_VECTOR,
    ".json": FingerprintKind.TEXT_VECTOR,
    ".yaml": FingerprintKind.TEXT_VECTOR,
    ".yml": FingerprintKind.TEXT_VECTOR,
    ".xml": FingerprintKind.TEXT_VECTOR,
    ".csv": FingerprintKind.TEXT_VECTOR,
    ".tsv": FingerprintKind.TEXT_VECTOR,
    ".html": FingerprintKind.TEXT_VECTOR,
    ".htm": FingerprintKind.TEXT_VECTOR,
}


# Magic-byte sniff for the most common formats — used when neither
# python-magic nor a filename hint is available. Each entry is
# (offset, magic_bytes, kind). Tested in order; first match wins.
_MAGIC_BYTES_TABLE: tuple = (
    (0, b"\x89PNG\r\n\x1a\n", FingerprintKind.IMAGE_PHASH),
    (0, b"\xff\xd8\xff", FingerprintKind.IMAGE_PHASH),  # JPEG
    (0, b"GIF87a", FingerprintKind.IMAGE_PHASH),
    (0, b"GIF89a", FingerprintKind.IMAGE_PHASH),
    (0, b"BM", FingerprintKind.IMAGE_PHASH),  # BMP — short magic
    (0, b"RIFF", FingerprintKind.IMAGE_PHASH),  # WebP / WAV — see refinement below
    # PDF text
    (0, b"%PDF-", FingerprintKind.TEXT_VECTOR),
    # HDF5
    (0, b"\x89HDF\r\n\x1a\n", FingerprintKind.STRUCTURAL),
    # Parquet ("PAR1" sentinel — present at file head AND tail)
    (0, b"PAR1", FingerprintKind.STRUCTURAL),
    # Arrow IPC
    (0, b"ARROW1", FingerprintKind.STRUCTURAL),
    # NumPy .npy
    (0, b"\x93NUMPY", FingerprintKind.STRUCTURAL),
    # ID3v2 (MP3 tag) — most MP3s with tags
    (0, b"ID3", FingerprintKind.AUDIO_CHROMAPRINT),
    # FLAC
    (0, b"fLaC", FingerprintKind.AUDIO_CHROMAPRINT),
    # OGG
    (0, b"OggS", FingerprintKind.AUDIO_CHROMAPRINT),
    # MP4 / MOV (ftyp box)
    (4, b"ftyp", FingerprintKind.VIDEO_MULTIHASH),
    # Matroska / WebM (EBML)
    (0, b"\x1a\x45\xdf\xa3", FingerprintKind.VIDEO_MULTIHASH),
)


def _kind_from_mime(mime_type: Optional[str]) -> Optional[FingerprintKind]:
    if not mime_type:
        return None
    mime_type = mime_type.lower().strip()
    for prefix, kind in _MIME_KIND_TABLE:
        if mime_type.startswith(prefix):
            return kind
    return None


def _kind_from_filename(filename: Optional[str]) -> Optional[FingerprintKind]:
    if not filename:
        return None
    suffix = os.path.splitext(filename)[1].lower()
    return _SUFFIX_KIND_TABLE.get(suffix)


def _kind_from_magic_bytes(content: bytes) -> Optional[FingerprintKind]:
    if not content:
        return None
    for offset, magic, kind in _MAGIC_BYTES_TABLE:
        end = offset + len(magic)
        if len(content) >= end and content[offset:end] == magic:
            # RIFF disambiguation: WebP has "WEBP" at offset 8;
            # WAV has "WAVE". Default to image-pHash since WebP is
            # more common; WAV is also fine to byte-hash if no audio
            # backend is registered.
            if magic == b"RIFF":
                if len(content) >= 12 and content[8:12] == b"WAVE":
                    return FingerprintKind.AUDIO_CHROMAPRINT
                return FingerprintKind.IMAGE_PHASH
            return kind
    return None


def detect_content_kind(
    content: bytes,
    *,
    filename: Optional[str] = None,
    mime_hint: Optional[str] = None,
) -> FingerprintKind:
    """Map ``content`` (and optional filename / explicit MIME hint) to a
    ``FingerprintKind``.

    Resolution priority (most authoritative first):
      1. Explicit ``mime_hint`` (caller knows the type, e.g. from an
         IPFS pin's content-type header).
      2. ``python-magic`` content sniff (if installed).
      3. Stdlib ``mimetypes.guess_type(filename)`` fallback.
      4. Filename suffix lookup table.
      5. Raw magic-bytes sniff (subset of common formats).
      6. Fallback to ``TEXT_VECTOR`` if content decodes cleanly as
         UTF-8 of meaningful length, else ``BYTE_HASH``.

    Returns one of ``FingerprintKind``. NEVER raises — when nothing
    matches, falls back to ``BYTE_HASH`` (the always-safe option).
    """
    # 1. Explicit MIME hint from caller.
    kind = _kind_from_mime(mime_hint)
    if kind is not None:
        return kind

    # 2. python-magic if available (best-quality content sniffing).
    try:
        import magic  # type: ignore[import-not-found]
        try:
            sniffed = magic.from_buffer(content, mime=True)
            kind = _kind_from_mime(sniffed)
            if kind is not None:
                return kind
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"python-magic sniff failed: {exc}")
    except ImportError:
        pass  # falls through to stdlib

    # 3. Stdlib mimetypes from filename.
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        kind = _kind_from_mime(guessed)
        if kind is not None:
            return kind

    # 4. Filename suffix table.
    kind = _kind_from_filename(filename)
    if kind is not None:
        return kind

    # 5. Magic-bytes sniff.
    kind = _kind_from_magic_bytes(content)
    if kind is not None:
        return kind

    # 6. UTF-8 text fallback. Mirrors ContentUploader._get_embedding's
    # existing 50-char threshold so the dispatch agrees with the
    # downstream embedding-eligibility gate.
    try:
        text = content.decode("utf-8", errors="strict").strip()
        if len(text) >= 50:
            return FingerprintKind.TEXT_VECTOR
    except UnicodeDecodeError:
        pass

    return FingerprintKind.BYTE_HASH
