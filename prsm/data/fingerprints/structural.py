"""
PRSM-PROV-1 Item 4 T4.5 — StructuralFingerprint backend.

Structural-signature backend for tabular and hierarchical scientific
data formats — HDF5 (``.h5`` / ``.hdf5``) and Apache Parquet
(``.parquet``). The fingerprint captures the dataset's *schema +
shape*: the set of column/key names, their dtypes, and the row count
(or per-key shape for HDF5). Two datasets with identical structure
fingerprint identically; one extra row, one renamed column, or one
dtype shift produces a different fingerprint.

Threshold (per ``prsm/data/dedup_thresholds.yaml::defaults.structural``):
duplicate iff payload bytes are exactly equal — there is no fuzzy
match for structural identity. The backend therefore returns 1.0 for
exact-match payloads and 0.0 for any difference; ``ThresholdResolver``
treats this as a hard binary check.

Why structural-only (not row-content) at this layer:
  - Row-content hashing turns into a generic byte-hash, which is the
    BYTE_HASH backend's job. Structural fingerprinting is the layer
    that catches "the same dataset re-uploaded with one column added"
    or "the same study re-derived after a re-run." That use-case
    requires schema-aware comparison, not byte equality.
  - Including row counts makes "same schema, different snapshot"
    distinguishable without paying for full-content hashing. Two
    snapshots of a clinical trial database with the same schema but
    different patient counts — the right answer is "not the same
    dataset" and the structural fingerprint produces it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from io import BytesIO
from typing import Optional

from prsm.data.fingerprints.base import (
    BinaryFingerprint,
    FingerprintKind,
    FingerprintRecord,
)

logger = logging.getLogger(__name__)


# Structural payload is a fixed 32-byte SHA-256 of the canonical
# structural-signature JSON. The framing matches the ImageFingerprint
# convention of fixed-size opaque payloads.
_STRUCTURAL_PAYLOAD_BYTES = 32

# Magic-byte prefixes for format detection. We never trust the
# filename extension alone — content sniffing wins because uploaders
# can rename files and the format detection has to work on bytes.
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_PARQUET_MAGIC = b"PAR1"


class StructuralFingerprint(BinaryFingerprint):
    """Schema + shape fingerprint for HDF5 + Parquet content.

    Backend lookup is keyed on content magic bytes, not filename
    extension. Unsupported formats return None (callers fall through
    to ``BYTE_HASH``).
    """

    KIND = FingerprintKind.STRUCTURAL

    def compute(
        self, content: bytes, *, filename: Optional[str] = None,
    ) -> Optional[FingerprintRecord]:
        if not content:
            return None

        format_kind = _detect_format(content)
        if format_kind is None:
            logger.debug(
                "StructuralFingerprint.compute skipped (filename=%s): "
                "no recognized HDF5 or Parquet magic at content start",
                filename,
            )
            return None

        try:
            if format_kind == "hdf5":
                signature = _hdf5_structural_signature(content)
            else:  # parquet
                signature = _parquet_structural_signature(content)
        except _BackendUnavailable as exc:
            logger.warning(
                "StructuralFingerprint backend unavailable for %s: %s",
                format_kind, exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            # Truncated / malformed file. Don't crash the upload path —
            # caller falls through to BYTE_HASH so the upload still
            # registers.
            logger.debug(
                "StructuralFingerprint.compute parse failure "
                "(format=%s, filename=%s): %s",
                format_kind, filename, exc,
            )
            return None

        if signature is None:
            return None

        # Canonical JSON encoding then SHA-256. JSON serialization is
        # stable when keys are sorted; the helpers below sort their
        # outputs so the same dataset hashes identically across
        # platforms / library versions.
        canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        if len(digest) != _STRUCTURAL_PAYLOAD_BYTES:  # defensive
            return None
        return FingerprintRecord(kind=self.KIND, payload=digest)

    def similarity(self, a: bytes, b: bytes) -> float:
        """Binary equality. Structural identity has no fuzzy notion —
        a single column rename or row-count difference is "different
        dataset" by design.

        Returns 0.0 on malformed payload (wrong length) so a corrupt
        record can't coincidentally score high against a real one,
        same defense as the image backend.
        """
        if (
            len(a) != _STRUCTURAL_PAYLOAD_BYTES
            or len(b) != _STRUCTURAL_PAYLOAD_BYTES
        ):
            return 0.0
        return 1.0 if a == b else 0.0


# ──────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────


def _detect_format(content: bytes) -> Optional[str]:
    """Sniff content type. Returns "hdf5", "parquet", or None.

    Parquet's magic byte sequence appears at BOTH the start and end of
    the file. We check the start for fast common-case detection; the
    start-of-file marker is a reliable hint for valid Parquet files
    even though the schema lives in the footer.
    """
    if content.startswith(_HDF5_MAGIC):
        return "hdf5"
    if content.startswith(_PARQUET_MAGIC):
        return "parquet"
    return None


class _BackendUnavailable(Exception):
    """Raised when the underlying h5py / pyarrow library import fails.
    The caller logs this at WARNING (not DEBUG) since it's a config
    issue, not a content-quality issue."""


# ──────────────────────────────────────────────────────────────────────
# HDF5 backend
# ──────────────────────────────────────────────────────────────────────


def _hdf5_structural_signature(content: bytes) -> Optional[dict]:
    """Build canonical structural signature for an HDF5 file.

    Signature shape:

        {
            "format": "hdf5",
            "datasets": [
                {"path": "/group/name", "shape": [rows, cols], "dtype": "float32"},
                ...
            ]
        }

    `datasets` is sorted by path so the signature is stable across
    files written with different group-creation orders. Group
    attributes and metadata are NOT included — they're often
    timestamp / process-id contaminated and would falsely
    discriminate identical scientific datasets.
    """
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise _BackendUnavailable(
            "h5py not installed; pip install h5py",
        ) from exc

    datasets: list[dict] = []
    bio = BytesIO(content)
    with h5py.File(bio, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append({
                    "path": "/" + name,  # h5py omits leading slash
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                })
        f.visititems(_visit)

    datasets.sort(key=lambda d: d["path"])
    return {"format": "hdf5", "datasets": datasets}


# ──────────────────────────────────────────────────────────────────────
# Parquet backend
# ──────────────────────────────────────────────────────────────────────


def _parquet_structural_signature(content: bytes) -> Optional[dict]:
    """Build canonical structural signature for a Parquet file.

    Signature shape:

        {
            "format": "parquet",
            "num_rows": <int>,
            "columns": [
                {"name": "col_a", "type": "int64"},
                {"name": "col_b", "type": "string"},
                ...
            ]
        }

    Column order is preserved (Parquet schemas are inherently ordered;
    reordering columns produces a logically different file). Row
    count is included so two snapshots of the same query at different
    times produce different fingerprints.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise _BackendUnavailable(
            "pyarrow not installed; pip install pyarrow",
        ) from exc

    bio = BytesIO(content)
    pf = pq.ParquetFile(bio)
    schema = pf.schema_arrow
    columns = [
        {"name": field.name, "type": str(field.type)}
        for field in schema
    ]
    num_rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0
    return {
        "format": "parquet",
        "num_rows": num_rows,
        "columns": columns,
    }


__all__ = ["StructuralFingerprint"]
