"""
PRSM-PROV-1 Item 4 T4.1 — BinaryFingerprint scaffold.

Public surface:
- BinaryFingerprint: ABC for kind-specific perceptual / structural
  fingerprint backends.
- FingerprintKind: enum of supported fingerprint kinds matching the
  threshold-resolver YAML keys.
- detect_content_kind(content, filename) -> FingerprintKind: maps raw
  bytes (and an optional filename hint) to the appropriate fingerprint
  kind via MIME-type detection. Uses ``python-magic`` when available;
  falls back to stdlib ``mimetypes`` + filename suffix.

Concrete backends (T4.2+ — image, audio, video, structural) plug in
behind the ABC and are wired by ContentUploader at upload time.
"""

from prsm.data.fingerprints.base import (
    BinaryFingerprint,
    FingerprintKind,
    FingerprintRecord,
    detect_content_kind,
)


def _import_image_fingerprint():
    """Lazy importer so a host without imagehash/Pillow can still
    construct the registry — the backend simply won't be available."""
    try:
        from prsm.data.fingerprints.image import ImageFingerprint
        return ImageFingerprint
    except ImportError:
        return None


def _import_audio_fingerprint():
    """Lazy importer for AudioFingerprint. Backend reports its own
    missing-dep diagnostics at compute() time; the module always
    imports cleanly because its heavy deps are inside compute()."""
    try:
        from prsm.data.fingerprints.audio import AudioFingerprint
        return AudioFingerprint
    except ImportError:
        return None


def _import_video_fingerprint():
    """Lazy importer for VideoFingerprint (PyAV + numpy)."""
    try:
        from prsm.data.fingerprints.video import VideoFingerprint
        return VideoFingerprint
    except ImportError:
        return None


def _import_structural_fingerprint():
    """Lazy importer for StructuralFingerprint (h5py + pyarrow)."""
    try:
        from prsm.data.fingerprints.structural import StructuralFingerprint
        return StructuralFingerprint
    except ImportError:
        return None


ImageFingerprint = _import_image_fingerprint()
AudioFingerprint = _import_audio_fingerprint()
VideoFingerprint = _import_video_fingerprint()
StructuralFingerprint = _import_structural_fingerprint()

__all__ = [
    "BinaryFingerprint",
    "FingerprintKind",
    "FingerprintRecord",
    "detect_content_kind",
    "ImageFingerprint",
    "AudioFingerprint",
    "VideoFingerprint",
    "StructuralFingerprint",
]
