"""Unit tests for StructuralFingerprint (PRSM-PROV-1 Item 4 T4.5).

Covers HDF5 + Parquet structural-signature generation, the magic-byte
content sniffing, the deterministic-across-platforms canonicalization,
and the binary-equality similarity model.
"""
from __future__ import annotations

from io import BytesIO

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prsm.data.fingerprints.base import FingerprintKind
from prsm.data.fingerprints.structural import (
    StructuralFingerprint,
    _detect_format,
)


# ──────────────────────────────────────────────────────────────────────
# fixtures — synthesize HDF5 + Parquet bytes in-memory
# ──────────────────────────────────────────────────────────────────────


def _hdf5_bytes(datasets: dict[str, np.ndarray]) -> bytes:
    """Create an in-memory HDF5 file with the given datasets."""
    bio = BytesIO()
    with h5py.File(bio, "w") as f:
        for path, arr in datasets.items():
            # Strip leading slash; h5py adds groups automatically.
            f.create_dataset(path.lstrip("/"), data=arr)
    return bio.getvalue()


def _parquet_bytes(table: pa.Table) -> bytes:
    """Create an in-memory Parquet file with the given pyarrow Table."""
    bio = BytesIO()
    pq.write_table(table, bio)
    return bio.getvalue()


# ──────────────────────────────────────────────────────────────────────
# format detection
# ──────────────────────────────────────────────────────────────────────


class TestFormatDetection:
    def test_detects_hdf5(self):
        content = _hdf5_bytes({"data": np.zeros(5)})
        assert _detect_format(content) == "hdf5"

    def test_detects_parquet(self):
        table = pa.table({"x": [1, 2, 3]})
        content = _parquet_bytes(table)
        assert _detect_format(content) == "parquet"

    def test_returns_none_for_random_bytes(self):
        assert _detect_format(b"random text not a recognized format") is None

    def test_returns_none_for_empty(self):
        assert _detect_format(b"") is None

    def test_returns_none_for_truncated_magic(self):
        # First 4 bytes of HDF5 magic but not all 8.
        assert _detect_format(b"\x89HDF") is None


# ──────────────────────────────────────────────────────────────────────
# compute() — basic happy paths
# ──────────────────────────────────────────────────────────────────────


class TestComputeHDF5:
    def test_returns_record_for_hdf5(self):
        content = _hdf5_bytes({
            "/measurements": np.arange(100, dtype=np.float32),
            "/labels": np.array([0, 1, 2, 1, 0], dtype=np.int64),
        })
        out = StructuralFingerprint().compute(content)
        assert out is not None
        assert out.kind == FingerprintKind.STRUCTURAL
        assert len(out.payload) == 32

    def test_identical_hdf5_files_fingerprint_identically(self):
        datasets = {
            "/data": np.array([1, 2, 3], dtype=np.int32),
            "/meta": np.array([0.5, 1.5], dtype=np.float64),
        }
        a = StructuralFingerprint().compute(_hdf5_bytes(datasets))
        b = StructuralFingerprint().compute(_hdf5_bytes(datasets))
        assert a is not None and b is not None
        assert a.payload == b.payload

    def test_different_dtype_produces_different_fingerprint(self):
        a = StructuralFingerprint().compute(_hdf5_bytes({
            "/x": np.zeros(10, dtype=np.float32),
        }))
        b = StructuralFingerprint().compute(_hdf5_bytes({
            "/x": np.zeros(10, dtype=np.float64),
        }))
        assert a is not None and b is not None
        assert a.payload != b.payload

    def test_different_shape_produces_different_fingerprint(self):
        a = StructuralFingerprint().compute(_hdf5_bytes({
            "/x": np.zeros(10, dtype=np.float32),
        }))
        b = StructuralFingerprint().compute(_hdf5_bytes({
            "/x": np.zeros(20, dtype=np.float32),
        }))
        assert a is not None and b is not None
        assert a.payload != b.payload

    def test_different_dataset_paths_differ(self):
        a = StructuralFingerprint().compute(_hdf5_bytes({
            "/measurements": np.zeros(10),
        }))
        b = StructuralFingerprint().compute(_hdf5_bytes({
            "/observations": np.zeros(10),
        }))
        assert a is not None and b is not None
        assert a.payload != b.payload

    def test_dataset_order_does_not_affect_fingerprint(self):
        """The signature sorts by path so same datasets in different
        creation order fingerprint identically."""
        a = StructuralFingerprint().compute(_hdf5_bytes({
            "/aaa": np.zeros(5),
            "/zzz": np.ones(5),
        }))
        b = StructuralFingerprint().compute(_hdf5_bytes({
            "/zzz": np.ones(5),
            "/aaa": np.zeros(5),
        }))
        assert a is not None and b is not None
        assert a.payload == b.payload

    def test_data_content_does_not_affect_fingerprint(self):
        """Two HDF5 files with same schema + shape but different VALUES
        fingerprint identically — that's the structural-only design.
        Byte-level dedup is BYTE_HASH's job, not StructuralFingerprint's.
        """
        a = StructuralFingerprint().compute(_hdf5_bytes({
            "/x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }))
        b = StructuralFingerprint().compute(_hdf5_bytes({
            "/x": np.array([99.0, 99.0, 99.0], dtype=np.float32),
        }))
        assert a is not None and b is not None
        assert a.payload == b.payload

    def test_nested_groups_captured(self):
        """h5py.visititems walks nested groups; both /a/x and /b/x are
        captured even though they're under different group prefixes."""
        a = StructuralFingerprint().compute(_hdf5_bytes({
            "/group1/x": np.zeros(5),
            "/group2/y": np.zeros(5),
        }))
        # Different schema → different fingerprint than flat case
        b = StructuralFingerprint().compute(_hdf5_bytes({
            "/x": np.zeros(5),
            "/y": np.zeros(5),
        }))
        assert a is not None and b is not None
        assert a.payload != b.payload


class TestComputeParquet:
    def test_returns_record_for_parquet(self):
        table = pa.table({
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["a", "b", "c"], type=pa.string()),
        })
        out = StructuralFingerprint().compute(_parquet_bytes(table))
        assert out is not None
        assert out.kind == FingerprintKind.STRUCTURAL
        assert len(out.payload) == 32

    def test_identical_parquet_fingerprint_identically(self):
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        a = StructuralFingerprint().compute(_parquet_bytes(table))
        b = StructuralFingerprint().compute(_parquet_bytes(table))
        assert a is not None and b is not None
        assert a.payload == b.payload

    def test_different_column_count_differs(self):
        a = StructuralFingerprint().compute(_parquet_bytes(
            pa.table({"x": [1, 2]}),
        ))
        b = StructuralFingerprint().compute(_parquet_bytes(
            pa.table({"x": [1, 2], "y": [3, 4]}),
        ))
        assert a is not None and b is not None
        assert a.payload != b.payload

    def test_different_column_type_differs(self):
        a = StructuralFingerprint().compute(_parquet_bytes(pa.table({
            "x": pa.array([1, 2, 3], type=pa.int32()),
        })))
        b = StructuralFingerprint().compute(_parquet_bytes(pa.table({
            "x": pa.array([1, 2, 3], type=pa.int64()),
        })))
        assert a is not None and b is not None
        assert a.payload != b.payload

    def test_different_row_count_differs(self):
        """Two snapshots of same schema but different row counts must
        differ. This is the canonical 'different snapshot' use case."""
        a = StructuralFingerprint().compute(_parquet_bytes(
            pa.table({"x": [1, 2, 3]}),
        ))
        b = StructuralFingerprint().compute(_parquet_bytes(
            pa.table({"x": [1, 2, 3, 4]}),
        ))
        assert a is not None and b is not None
        assert a.payload != b.payload

    def test_column_order_matters(self):
        """Reordering columns is a logically different schema in
        Parquet; the fingerprints differ."""
        a = StructuralFingerprint().compute(_parquet_bytes(pa.table([
            pa.array([1, 2]), pa.array(["x", "y"]),
        ], names=["a", "b"])))
        b = StructuralFingerprint().compute(_parquet_bytes(pa.table([
            pa.array(["x", "y"]), pa.array([1, 2]),
        ], names=["b", "a"])))
        assert a is not None and b is not None
        assert a.payload != b.payload


# ──────────────────────────────────────────────────────────────────────
# compute() — fall-through cases
# ──────────────────────────────────────────────────────────────────────


class TestComputeFallthrough:
    def test_returns_none_for_unknown_format(self):
        out = StructuralFingerprint().compute(b"plain text content")
        assert out is None

    def test_returns_none_for_empty(self):
        out = StructuralFingerprint().compute(b"")
        assert out is None

    def test_returns_none_for_truncated_hdf5(self):
        """HDF5 magic at start but truncated payload → caller falls
        through to BYTE_HASH rather than crashing."""
        truncated = b"\x89HDF\r\n\x1a\n" + b"\x00" * 50
        out = StructuralFingerprint().compute(truncated)
        assert out is None

    def test_returns_none_for_truncated_parquet(self):
        out = StructuralFingerprint().compute(b"PAR1" + b"\x00" * 100)
        assert out is None


# ──────────────────────────────────────────────────────────────────────
# similarity()
# ──────────────────────────────────────────────────────────────────────


class TestSimilarity:
    def test_identical_payloads_return_1(self):
        payload = b"\x42" * 32
        assert StructuralFingerprint().similarity(payload, payload) == 1.0

    def test_different_payloads_return_0(self):
        a = b"\x42" * 32
        b = b"\x43" * 32
        assert StructuralFingerprint().similarity(a, b) == 0.0

    def test_one_byte_difference_returns_0(self):
        """No fuzzy match — even one-byte difference is binary-not-equal,
        which is the design (structural identity is a hard property)."""
        a = bytes(range(32))
        b = bytes([0] + list(range(1, 32)))
        # Wait — b == a here. Build a real difference.
        b = bytes([99]) + bytes(range(1, 32))
        assert StructuralFingerprint().similarity(a, b) == 0.0

    def test_malformed_payload_returns_0(self):
        """Defense: a payload of wrong length can't coincidentally
        match a real one."""
        good = b"\x42" * 32
        short = b"\x42" * 16
        long = b"\x42" * 64
        assert StructuralFingerprint().similarity(good, short) == 0.0
        assert StructuralFingerprint().similarity(good, long) == 0.0
        assert StructuralFingerprint().similarity(short, long) == 0.0


# ──────────────────────────────────────────────────────────────────────
# end-to-end: round-trip through compute+similarity
# ──────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_same_dataset_round_trips_to_1(self):
        content = _hdf5_bytes({"/x": np.arange(50, dtype=np.float32)})
        fp = StructuralFingerprint()
        a = fp.compute(content)
        b = fp.compute(content)
        assert a is not None and b is not None
        assert fp.similarity(a.payload, b.payload) == 1.0

    def test_different_datasets_round_trip_to_0(self):
        c1 = _hdf5_bytes({"/x": np.zeros(10, dtype=np.float32)})
        c2 = _hdf5_bytes({"/y": np.zeros(10, dtype=np.float32)})
        fp = StructuralFingerprint()
        a = fp.compute(c1)
        b = fp.compute(c2)
        assert a is not None and b is not None
        assert fp.similarity(a.payload, b.payload) == 0.0

    def test_hdf5_and_parquet_with_same_logical_data_differ(self):
        """An HDF5 file holding /x=[1,2,3] and a Parquet file holding
        column x=[1,2,3] are different formats and must fingerprint
        differently — a researcher uploading the same data in two
        formats has uploaded two distinct artifacts from PRSM's
        provenance perspective."""
        hdf5_content = _hdf5_bytes({
            "/x": np.array([1, 2, 3], dtype=np.int64),
        })
        parquet_content = _parquet_bytes(pa.table({
            "x": pa.array([1, 2, 3], type=pa.int64()),
        }))
        fp = StructuralFingerprint()
        a = fp.compute(hdf5_content)
        b = fp.compute(parquet_content)
        assert a is not None and b is not None
        assert fp.similarity(a.payload, b.payload) == 0.0
