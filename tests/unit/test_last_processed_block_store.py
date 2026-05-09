"""LastProcessedBlockStore — protocol + InMemory + Filesystem impls.

Closes the afternoon-arc deferred item from
project_phase78_afternoon_arc_2026_05_08.md: today's first-tick
watcher semantics reset baseline at every restart, losing startup-
window events. With persistence, restarts pick up where they left
off.

The store is keyed by a per-watcher identifier ("key_distribution"
/ "storage_slashing" / "compensation_distributor") so a single
filesystem store can serve all three watchers without contention.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.economy.web3.last_processed_block_store import (
    FilesystemLastProcessedBlockStore,
    InMemoryLastProcessedBlockStore,
    LastProcessedBlockStore,
)


# ──────────────────────────────────────────────────────────────────────
# Protocol shape
# ──────────────────────────────────────────────────────────────────────


class TestProtocolShape:
    def test_protocol_methods_present(self):
        # Verify both impls satisfy the Protocol surface.
        for impl_cls in (
            InMemoryLastProcessedBlockStore,
            FilesystemLastProcessedBlockStore,
        ):
            assert hasattr(impl_cls, "load")
            assert hasattr(impl_cls, "save")
            assert hasattr(impl_cls, "delete")


# ──────────────────────────────────────────────────────────────────────
# InMemoryLastProcessedBlockStore
# ──────────────────────────────────────────────────────────────────────


class TestInMemoryStore:
    def test_load_missing_key_returns_none(self):
        store = InMemoryLastProcessedBlockStore()
        assert store.load("key_distribution") is None

    def test_save_then_load_returns_value(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("key_distribution", 12345)
        assert store.load("key_distribution") == 12345

    def test_save_overwrites_existing(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("key_distribution", 100)
        store.save("key_distribution", 200)
        assert store.load("key_distribution") == 200

    def test_save_negative_value_rejected(self):
        store = InMemoryLastProcessedBlockStore()
        with pytest.raises(ValueError, match="non-negative"):
            store.save("key_distribution", -1)

    def test_delete_removes_key(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("key_distribution", 100)
        store.delete("key_distribution")
        assert store.load("key_distribution") is None

    def test_delete_missing_key_no_op(self):
        store = InMemoryLastProcessedBlockStore()
        # Must not raise.
        store.delete("nonexistent")

    def test_distinct_keys_independent(self):
        store = InMemoryLastProcessedBlockStore()
        store.save("key_distribution", 100)
        store.save("storage_slashing", 200)
        store.save("compensation_distributor", 300)
        assert store.load("key_distribution") == 100
        assert store.load("storage_slashing") == 200
        assert store.load("compensation_distributor") == 300


# ──────────────────────────────────────────────────────────────────────
# FilesystemLastProcessedBlockStore
# ──────────────────────────────────────────────────────────────────────


class TestFilesystemStore:
    def test_load_missing_key_returns_none(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        assert store.load("key_distribution") is None

    def test_save_then_load_round_trip(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        store.save("key_distribution", 12345)
        assert store.load("key_distribution") == 12345

    def test_save_writes_json_file(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        store.save("key_distribution", 12345)
        # File path is base_dir / "<key>.json"
        expected_path = tmp_path / "key_distribution.json"
        assert expected_path.exists()
        body = json.loads(expected_path.read_text())
        assert body["last_processed_block"] == 12345
        assert body["watcher_key"] == "key_distribution"

    def test_load_after_restart_returns_persisted_value(self, tmp_path):
        # Simulate process restart: write with one store instance,
        # read with a fresh instance pointing at the same dir.
        store_a = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        store_a.save("key_distribution", 99999)
        store_b = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        assert store_b.load("key_distribution") == 99999

    def test_save_overwrites_existing_file(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        store.save("key_distribution", 100)
        store.save("key_distribution", 200)
        assert store.load("key_distribution") == 200

    def test_save_negative_value_rejected(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        with pytest.raises(ValueError, match="non-negative"):
            store.save("key_distribution", -1)

    def test_delete_removes_file(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        store.save("key_distribution", 100)
        path = tmp_path / "key_distribution.json"
        assert path.exists()
        store.delete("key_distribution")
        assert not path.exists()
        assert store.load("key_distribution") is None

    def test_delete_missing_key_no_op(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        # Must not raise.
        store.delete("nonexistent")

    def test_distinct_keys_use_distinct_files(self, tmp_path):
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        store.save("key_distribution", 100)
        store.save("storage_slashing", 200)
        store.save("compensation_distributor", 300)

        files = sorted(p.name for p in tmp_path.glob("*.json"))
        assert files == [
            "compensation_distributor.json",
            "key_distribution.json",
            "storage_slashing.json",
        ]
        assert store.load("key_distribution") == 100
        assert store.load("storage_slashing") == 200
        assert store.load("compensation_distributor") == 300

    def test_load_corrupt_file_returns_none_and_logs(self, tmp_path, caplog):
        # If the JSON is corrupt or wrong-shaped, load() should
        # return None (treat as missing) + log at WARNING. Don't
        # raise — that would prevent a watcher from starting after
        # any disk corruption.
        path = tmp_path / "key_distribution.json"
        path.write_text("{not valid json")
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        import logging
        with caplog.at_level(logging.WARNING):
            result = store.load("key_distribution")
        assert result is None
        assert any(
            "corrupt" in r.message.lower() or "invalid" in r.message.lower()
            for r in caplog.records
        )

    def test_load_wrong_shape_returns_none(self, tmp_path):
        # Valid JSON but missing the expected fields → None.
        path = tmp_path / "key_distribution.json"
        path.write_text('{"unrelated_field": 42}')
        store = FilesystemLastProcessedBlockStore(base_dir=tmp_path)
        assert store.load("key_distribution") is None

    def test_base_dir_auto_created(self, tmp_path):
        # If base_dir doesn't exist yet, the store creates it on
        # first save (mkdir -p semantics).
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        store = FilesystemLastProcessedBlockStore(base_dir=nested)
        store.save("key_distribution", 100)
        assert nested.exists()
        assert store.load("key_distribution") == 100

    def test_default_base_dir_is_under_dot_prsm(self):
        # Default path: ~/.prsm/watchers/
        store = FilesystemLastProcessedBlockStore()
        # Must end with the canonical directory (don't assert exact
        # home path so the test is portable).
        assert str(store.base_dir).endswith(".prsm/watchers") or \
               str(store.base_dir).endswith(".prsm\\watchers")  # Windows
