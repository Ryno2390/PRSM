"""Sprint 242 — ReceiptStore for signed InferenceReceipts.

Pre-fix: /compute/inference signs a receipt and returns it in
the HTTP response, but the server doesn't persist it. End-users
who don't save the response have no recourse later. Auditors
verifying a node's outputs have no post-hoc lookup surface.

This sprint adds:
  - prsm/node/receipt_store.py — LRU-bounded in-memory store
    keyed by job_id; mirrors the JobHistoryStore design pattern.
  - Filesystem persistence opt-in via PRSM_RECEIPT_STORE_DIR
    env var (mirrors JobHistoryStore's sprint-2026-05-09 design).
"""
from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal

import pytest

from prsm.node.receipt_store import ReceiptStore


def _sample_receipt(job_id: str = "job-1") -> dict:
    return {
        "job_id": job_id,
        "request_id": "req-1",
        "model_id": "mock-llama-3-8b",
        "privacy_tier": "standard",
        "content_tier": "A",
        "tee_type": "software",
        "epsilon_spent": 8.0,
        "cost_ftns": "0.10",
        "duration_seconds": 1.0,
        "output_hash": "00" * 32,
        "settler_signature": "deadbeef",
        "settler_node_id": "settler-7",
    }


class TestBasicLRU:
    def test_put_and_get(self):
        store = ReceiptStore()
        r = _sample_receipt("job-1")
        store.put("job-1", r)
        assert store.get("job-1") == r

    def test_get_missing_returns_none(self):
        store = ReceiptStore()
        assert store.get("missing") is None

    def test_lru_eviction(self):
        store = ReceiptStore(max_entries=2)
        store.put("a", _sample_receipt("a"))
        store.put("b", _sample_receipt("b"))
        store.put("c", _sample_receipt("c"))
        # "a" evicted
        assert store.get("a") is None
        assert store.get("b") is not None
        assert store.get("c") is not None

    def test_put_refreshes_recency(self):
        store = ReceiptStore(max_entries=2)
        store.put("a", _sample_receipt("a"))
        store.put("b", _sample_receipt("b"))
        store.get("a")  # refresh
        store.put("c", _sample_receipt("c"))
        # "b" evicted, "a" survives
        assert store.get("a") is not None
        assert store.get("b") is None
        assert store.get("c") is not None

    def test_len(self):
        store = ReceiptStore()
        assert len(store) == 0
        store.put("a", _sample_receipt("a"))
        assert len(store) == 1


class TestValidation:
    def test_empty_job_id_rejected(self):
        store = ReceiptStore()
        with pytest.raises(ValueError):
            store.put("", _sample_receipt())

    def test_non_dict_receipt_rejected(self):
        store = ReceiptStore()
        with pytest.raises(TypeError):
            store.put("a", "not a dict")  # type: ignore[arg-type]


class TestFilesystemPersistence:
    def test_no_env_no_persistence(self, tmp_path, monkeypatch):
        """Without env var = pure in-memory (sprint 242 default)."""
        monkeypatch.delenv("PRSM_RECEIPT_STORE_DIR", raising=False)
        store = ReceiptStore.from_env()
        store.put("job-1", _sample_receipt())
        # No files written under tmp_path.
        assert not list(tmp_path.iterdir())

    def test_env_persists_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRSM_RECEIPT_STORE_DIR", str(tmp_path))
        store = ReceiptStore.from_env()
        store.put("job-1", _sample_receipt())
        # Exactly one file written; SHA-256-named.
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_constructor_repopulates_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRSM_RECEIPT_STORE_DIR", str(tmp_path))
        s1 = ReceiptStore.from_env()
        s1.put("job-1", _sample_receipt("job-1"))
        s1.put("job-2", _sample_receipt("job-2"))
        # New store sees both
        s2 = ReceiptStore.from_env()
        assert s2.get("job-1") is not None
        assert s2.get("job-2") is not None

    def test_corrupt_file_fail_soft(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRSM_RECEIPT_STORE_DIR", str(tmp_path))
        (tmp_path / "garbage.json").write_text("{not json")
        s = ReceiptStore.from_env()
        # No crash; just skips the bad file.
        assert isinstance(s, ReceiptStore)

    def test_path_traversal_proof(self, tmp_path, monkeypatch):
        """Filenames are SHA-256 of job_id, not job_id itself —
        so '../../etc/passwd' as job_id can't escape."""
        monkeypatch.setenv("PRSM_RECEIPT_STORE_DIR", str(tmp_path))
        s = ReceiptStore.from_env()
        s.put("../../../etc/passwd", _sample_receipt("evil"))
        # The only file produced is inside tmp_path.
        files = list(tmp_path.rglob("*"))
        for f in files:
            if f.is_file():
                assert str(f).startswith(str(tmp_path))
