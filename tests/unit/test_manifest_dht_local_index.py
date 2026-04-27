"""
Unit tests — Phase 3.x.5 Task 2 — LocalManifestIndex.

Acceptance per design plan §4 Task 2: index round-trips through
restart; model_id → path is unambiguous; missing-file rebuild walks
the filesystem; defense-in-depth validation rejects unsafe ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prsm.network.manifest_dht import LocalManifestIndex


# ──────────────────────────────────────────────────────────────────────────
# Fixtures — stand up a Phase 3.x.2-style registry tree without
# pulling in the actual FilesystemModelRegistry. This keeps the
# index tests focused on index behavior, not registry coupling.
# ──────────────────────────────────────────────────────────────────────────


def _seed_model_dir(root: Path, model_id: str) -> Path:
    """Mirror the Phase 3.x.2 layout: <root>/<model_id>/manifest.json"""
    model_dir = root / model_id
    model_dir.mkdir()
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(f'{{"model_id": "{model_id}"}}')
    return manifest_path


@pytest.fixture
def empty_root(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def populated_root(tmp_path) -> Path:
    """Three pre-existing models, no dht_index.json yet — exercises
    the rebuild-from-walk path."""
    _seed_model_dir(tmp_path, "alpha")
    _seed_model_dir(tmp_path, "beta")
    _seed_model_dir(tmp_path, "gamma")
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalManifestIndex(tmp_path / "does-not-exist")

    def test_root_is_file_raises(self, tmp_path):
        f = tmp_path / "im-a-file"
        f.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            LocalManifestIndex(f)

    def test_str_path_accepted(self, empty_root):
        idx = LocalManifestIndex(str(empty_root))
        assert len(idx) == 0

    def test_empty_root_starts_empty(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        assert len(idx) == 0
        assert idx.list_models() == []
        # Persistence step ran on an empty registry → empty file exists.
        assert (empty_root / "dht_index.json").exists()


# ──────────────────────────────────────────────────────────────────────────
# Register / lookup
# ──────────────────────────────────────────────────────────────────────────


class TestRegisterLookup:
    def test_register_then_lookup(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        manifest = _seed_model_dir(empty_root, "llama-3-8b")
        idx.register("llama-3-8b", manifest)
        result = idx.lookup("llama-3-8b")
        assert result == manifest.resolve()

    def test_lookup_missing_returns_none(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        assert idx.lookup("not-registered") is None

    def test_lookup_non_string_returns_none(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        assert idx.lookup(42) is None  # type: ignore[arg-type]
        assert idx.lookup(b"bytes") is None  # type: ignore[arg-type]

    def test_lookup_empty_string_returns_none(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        assert idx.lookup("") is None

    def test_lookup_unsafe_id_returns_none(self, empty_root):
        # Defense in depth: even if somehow a "../escape" ended up in
        # the in-memory dict, lookup wouldn't return it.
        idx = LocalManifestIndex(empty_root)
        idx._entries["../escape"] = "../escape/manifest.json"
        assert idx.lookup("../escape") is None

    def test_register_replaces_silently(self, empty_root):
        # Registry-level uniqueness is the gatekeeper. If a duplicate
        # somehow reaches the index, overwrite without error — the
        # bug is upstream.
        idx = LocalManifestIndex(empty_root)
        m1 = _seed_model_dir(empty_root, "model-1")
        idx.register("model-1", m1)
        # Register again with the same id → silent overwrite (path is
        # the same, so this is effectively a no-op).
        idx.register("model-1", m1)
        assert len(idx) == 1

    def test_register_multiple(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        m_a = _seed_model_dir(empty_root, "alpha")
        m_b = _seed_model_dir(empty_root, "beta")
        idx.register("alpha", m_a)
        idx.register("beta", m_b)
        assert len(idx) == 2
        assert idx.list_models() == ["alpha", "beta"]
        assert idx.lookup("alpha") == m_a.resolve()
        assert idx.lookup("beta") == m_b.resolve()

    def test_register_unsafe_model_id_rejected(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        m = _seed_model_dir(empty_root, "ok")
        for bad in ["..", ".", "has space", "with/slash", ""]:
            with pytest.raises(ValueError, match="model_id"):
                idx.register(bad, m)

    def test_register_path_outside_root_rejected(
        self, empty_root, tmp_path_factory
    ):
        idx = LocalManifestIndex(empty_root)
        outside = tmp_path_factory.mktemp("other_root")
        outside_manifest = outside / "manifest.json"
        outside_manifest.write_text("{}")
        with pytest.raises(ValueError, match="not under index root"):
            idx.register("evil", outside_manifest)

    def test_contains_operator(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        m = _seed_model_dir(empty_root, "model-1")
        idx.register("model-1", m)
        assert "model-1" in idx
        assert "model-2" not in idx
        assert 42 not in idx  # type: ignore[operator]


# ──────────────────────────────────────────────────────────────────────────
# Persistence across instances
# ──────────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_round_trip_through_restart(self, empty_root):
        idx_a = LocalManifestIndex(empty_root)
        m1 = _seed_model_dir(empty_root, "alpha")
        m2 = _seed_model_dir(empty_root, "beta")
        idx_a.register("alpha", m1)
        idx_a.register("beta", m2)

        # Fresh instance, same root — simulates Node restart.
        idx_b = LocalManifestIndex(empty_root)
        assert idx_b.list_models() == ["alpha", "beta"]
        assert idx_b.lookup("alpha") == m1.resolve()
        assert idx_b.lookup("beta") == m2.resolve()

    def test_dht_index_json_uses_relative_paths(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        m = _seed_model_dir(empty_root, "alpha")
        idx.register("alpha", m)

        data = json.loads((empty_root / "dht_index.json").read_text())
        # Stored as relative path — the index is portable if the
        # operator relocates the root.
        assert data == {"alpha": "alpha/manifest.json"}

    def test_unregister_persists(self, empty_root):
        import shutil
        idx_a = LocalManifestIndex(empty_root)
        m = _seed_model_dir(empty_root, "alpha")
        idx_a.register("alpha", m)
        assert idx_a.unregister("alpha") is True
        # To persistently unregister, the on-disk model dir must also
        # be removed — otherwise orphan reconciliation re-adds the
        # entry at next construction (Phase 3.x.5 round 1 review
        # MEDIUM-2).
        shutil.rmtree(empty_root / "alpha")
        idx_b = LocalManifestIndex(empty_root)
        assert "alpha" not in idx_b

    def test_unregister_in_process_works(self, empty_root):
        # In-process unregister() removes the entry. Calling lookup()
        # immediately afterward returns None. Reconciliation only fires
        # at construction, so within a single instance the unregister
        # is durable.
        idx = LocalManifestIndex(empty_root)
        m = _seed_model_dir(empty_root, "alpha")
        idx.register("alpha", m)
        assert idx.lookup("alpha") is not None
        assert idx.unregister("alpha") is True
        assert idx.lookup("alpha") is None
        assert "alpha" not in idx

    def test_unregister_returns_false_for_missing(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        assert idx.unregister("not-there") is False


# ──────────────────────────────────────────────────────────────────────────
# Rebuild from filesystem walk
# ──────────────────────────────────────────────────────────────────────────


class TestRebuildFromWalk:
    def test_missing_index_file_triggers_walk(self, populated_root):
        # Three model dirs already exist on disk; no dht_index.json.
        # First construction must walk and populate.
        assert not (populated_root / "dht_index.json").exists()
        idx = LocalManifestIndex(populated_root)
        assert idx.list_models() == ["alpha", "beta", "gamma"]
        # And the freshly-built index was persisted.
        assert (populated_root / "dht_index.json").exists()

    def test_walk_skips_dirs_without_manifest(self, empty_root):
        _seed_model_dir(empty_root, "real-model")
        # An empty dir with no manifest.json — must be skipped.
        (empty_root / "empty-dir").mkdir()
        idx = LocalManifestIndex(empty_root)
        assert idx.list_models() == ["real-model"]

    def test_walk_skips_unsafe_dir_names(self, empty_root):
        _seed_model_dir(empty_root, "real-model")
        # A dir whose name violates the safe-id regex (contains a space).
        (empty_root / "has space").mkdir()
        (empty_root / "has space" / "manifest.json").write_text("{}")
        idx = LocalManifestIndex(empty_root)
        assert idx.list_models() == ["real-model"]

    def test_walk_skips_files_at_root(self, empty_root):
        _seed_model_dir(empty_root, "real-model")
        (empty_root / "stray.txt").write_text("not a model")
        idx = LocalManifestIndex(empty_root)
        assert idx.list_models() == ["real-model"]

    def test_explicit_rebuild(self, populated_root):
        # First construction populates the index.
        idx = LocalManifestIndex(populated_root)
        assert len(idx) == 3
        # Manually delete one model dir + add a new one; the in-memory
        # index hasn't been told.
        import shutil
        shutil.rmtree(populated_root / "alpha")
        _seed_model_dir(populated_root, "delta")
        # Explicit rebuild picks up the changes.
        idx.rebuild()
        assert idx.list_models() == ["beta", "delta", "gamma"]


# ──────────────────────────────────────────────────────────────────────────
# Corruption handling — fall back to walk on bad index JSON
# ──────────────────────────────────────────────────────────────────────────


class TestCorruption:
    def test_corrupt_json_falls_back_to_walk(self, populated_root):
        # Plant a corrupt index file alongside the populated tree.
        (populated_root / "dht_index.json").write_text("not valid json {{")
        idx = LocalManifestIndex(populated_root)
        # Walk recovered the three models.
        assert idx.list_models() == ["alpha", "beta", "gamma"]
        # And the index was rewritten correctly.
        data = json.loads((populated_root / "dht_index.json").read_text())
        assert set(data.keys()) == {"alpha", "beta", "gamma"}

    def test_unexpected_shape_falls_back_to_walk(self, populated_root):
        # Index file is JSON but a list, not a dict.
        (populated_root / "dht_index.json").write_text("[1, 2, 3]")
        idx = LocalManifestIndex(populated_root)
        assert idx.list_models() == ["alpha", "beta", "gamma"]

    def test_drops_unsafe_entry_at_load_time(self, empty_root, caplog):
        # An on-disk index containing a malicious entry — must be
        # silently dropped (not used) and a warning logged.
        import logging
        caplog.set_level(logging.WARNING, logger="prsm.network.manifest_dht.local_index")
        # Seed a real entry alongside the malicious one
        _seed_model_dir(empty_root, "real-model")
        (empty_root / "dht_index.json").write_text(
            json.dumps({
                "real-model": "real-model/manifest.json",
                "..": "../escape/manifest.json",
                "with/slash": "anywhere",
            })
        )
        idx = LocalManifestIndex(empty_root)
        # Only the real entry survived
        assert idx.list_models() == ["real-model"]
        # And warnings were emitted for the dropped entries
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 2

    def test_drops_missing_target(self, empty_root, caplog):
        # Index references a manifest path that doesn't exist on disk.
        # Drop the entry rather than serve a 404 at fetch time.
        import logging
        caplog.set_level(logging.WARNING, logger="prsm.network.manifest_dht.local_index")
        (empty_root / "dht_index.json").write_text(
            json.dumps({"phantom": "phantom/manifest.json"})
        )
        idx = LocalManifestIndex(empty_root)
        assert "phantom" not in idx
        assert any(
            "missing-target" in r.message
            for r in caplog.records
            if r.levelname == "WARNING"
        )


# ──────────────────────────────────────────────────────────────────────────
# Atomic-write invariant — using the same .tmp + os.replace idiom
# as Phase 3.x.2 / 3.x.4
# ──────────────────────────────────────────────────────────────────────────


class TestAtomicWrite:
    def test_no_orphan_tmp_after_register(self, empty_root):
        idx = LocalManifestIndex(empty_root)
        m = _seed_model_dir(empty_root, "alpha")
        idx.register("alpha", m)
        # No leftover .tmp files at root
        tmp_files = list(empty_root.glob("*.tmp"))
        assert tmp_files == []

    def test_persisted_json_is_canonical(self, empty_root):
        # sort_keys + indent=2 → deterministic output. Two indices with
        # the same content produce byte-equal files.
        idx_a = LocalManifestIndex(empty_root)
        m_a = _seed_model_dir(empty_root, "alpha")
        m_b = _seed_model_dir(empty_root, "beta")
        idx_a.register("alpha", m_a)
        idx_a.register("beta", m_b)
        bytes_a = (empty_root / "dht_index.json").read_bytes()

        # Trigger a re-persist via no-op unregister-then-rebuild
        idx_a.rebuild()
        bytes_b = (empty_root / "dht_index.json").read_bytes()
        assert bytes_a == bytes_b


# ──────────────────────────────────────────────────────────────────────────
# Round 1 review — MEDIUM-2 — orphan reconciliation
# ──────────────────────────────────────────────────────────────────────────


class TestOrphanReconciliation:
    """MEDIUM-2 from Phase 3.x.5 round 1 review: when a writer (e.g.,
    FilesystemModelRegistry._fetch_manifest_via_dht) writes a manifest
    to disk and then has its dht.announce() fail, the cache exists on
    disk but the index doesn't know about it. Without reconciliation
    this divergence persists across process restarts and the node
    silently stops serving the cached model to peers. Construction
    must auto-detect and recover.
    """

    def test_orphan_manifest_picked_up_on_construction(self, empty_root):
        # First instance: register one model, persist index.
        idx_a = LocalManifestIndex(empty_root)
        m_a = _seed_model_dir(empty_root, "alpha")
        idx_a.register("alpha", m_a)

        # Now simulate the divergence: manifest.json on disk for "beta"
        # but no corresponding entry in dht_index.json (this is what
        # happens when the registry caches via DHT and the announce
        # step fails after the file is written).
        _seed_model_dir(empty_root, "beta")

        # Second instance loads the JSON (which only knows alpha) but
        # must auto-detect beta and recover.
        idx_b = LocalManifestIndex(empty_root)
        assert "alpha" in idx_b
        assert "beta" in idx_b
        assert idx_b.lookup("beta") is not None

    def test_orphan_recovery_persists(self, empty_root):
        # After reconciliation, the recovered entry must be persisted
        # so subsequent constructions don't repeatedly walk and
        # re-warn — they take the fast-path JSON load.
        idx_a = LocalManifestIndex(empty_root)
        m_a = _seed_model_dir(empty_root, "alpha")
        idx_a.register("alpha", m_a)
        _seed_model_dir(empty_root, "beta")

        # First reconciliation
        idx_b = LocalManifestIndex(empty_root)
        assert "beta" in idx_b

        # Confirm the persisted JSON now contains beta.
        import json as _json
        on_disk = _json.loads((empty_root / "dht_index.json").read_text())
        assert "beta" in on_disk

    def test_no_orphans_no_writes(self, empty_root):
        # Steady state: no orphans on disk → reconciliation is a no-op
        # and shouldn't trigger an unnecessary persist.
        idx_a = LocalManifestIndex(empty_root)
        m = _seed_model_dir(empty_root, "alpha")
        idx_a.register("alpha", m)

        # Snapshot the index file's mtime / contents
        index_path = empty_root / "dht_index.json"
        mtime_before = index_path.stat().st_mtime_ns
        bytes_before = index_path.read_bytes()

        # Sleep a tiny amount so any rewrite would change mtime; then
        # reconstruct.
        import time as _time
        _time.sleep(0.01)
        idx_b = LocalManifestIndex(empty_root)
        assert "alpha" in idx_b
        # Index file unchanged — no spurious rewrite.
        assert index_path.read_bytes() == bytes_before
