"""SlashEventRing filesystem persistence (sprint 91).

When persist_dir is set, slash events survive node restart.
Symmetric to JobHistoryStore's PRSM_JOB_HISTORY_DIR pattern.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prsm.node.slash_event_log import SlashEventRing


class TestPersistOnAppend:
    def test_writes_file_on_append(self, tmp_path: Path):
        ring = SlashEventRing(persist_dir=tmp_path)
        ring.append(
            kind="proof_failure_slashed",
            provider="0xPROV",
            challenger="0xCHAL",
            slash_id=b"\x01" * 32,
        )
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["provider"] == "0xPROV"
        assert data["kind"] == "proof_failure_slashed"

    def test_no_persist_when_dir_unset(self, tmp_path: Path):
        ring = SlashEventRing()
        ring.append(
            kind="proof_failure_slashed",
            provider="0xPROV",
            challenger="0xCHAL",
            slash_id=b"\x01" * 32,
        )
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 0

    def test_creates_dir_if_missing(self, tmp_path: Path):
        target = tmp_path / "nested" / "slash_log"
        ring = SlashEventRing(persist_dir=target)
        assert target.exists()
        ring.append(
            kind="heartbeat_missing_slashed",
            provider="0xP", challenger="0xC",
            slash_id=b"\x02" * 32,
        )
        assert len(list(target.glob("*.json"))) == 1


class TestRehydrateOnInit:
    def test_loads_prior_entries(self, tmp_path: Path):
        # First instance writes
        r1 = SlashEventRing(persist_dir=tmp_path)
        r1.append(
            kind="proof_failure_slashed",
            provider="0xP1", challenger="0xC",
            slash_id=b"\x01" * 32,
        )
        r1.append(
            kind="heartbeat_missing_slashed",
            provider="0xP2", challenger="0xC",
            slash_id=b"\x02" * 32,
        )
        # Second instance reads
        r2 = SlashEventRing(persist_dir=tmp_path)
        assert r2.count() == 2
        recent = r2.recent()
        # Most-recent first; both providers present
        providers = {e.provider for e in recent}
        assert providers == {"0xP1", "0xP2"}

    def test_rehydrate_skips_corrupt_files(self, tmp_path: Path):
        # Write a corrupt file alongside a valid one
        r1 = SlashEventRing(persist_dir=tmp_path)
        r1.append(
            kind="proof_failure_slashed",
            provider="0xP1", challenger="0xC",
            slash_id=b"\x01" * 32,
        )
        (tmp_path / "corrupt.json").write_text("not valid json")
        r2 = SlashEventRing(persist_dir=tmp_path)
        assert r2.count() == 1

    def test_rehydrate_skips_invalid_kind(self, tmp_path: Path):
        # Write a file with bad `kind` value
        bogus = {
            "timestamp": 1700000000.0,
            "kind": "not_a_real_kind",
            "provider": "0xP",
            "challenger": "0xC",
            "slash_id": "0x" + "ff" * 32,
            "extras": {},
        }
        (tmp_path / "bogus.json").write_text(json.dumps(bogus))
        # Should NOT crash
        r = SlashEventRing(persist_dir=tmp_path)
        assert r.count() == 0


class TestPersistFailureFailSoft:
    def test_disk_write_failure_does_not_raise(self, tmp_path: Path):
        ring = SlashEventRing(persist_dir=tmp_path)
        # Make the dir unwritable mid-flight by replacing it with
        # a regular file to force OSError
        target = tmp_path / "blocked"
        target.write_text("blocking")
        # Now try a ring whose persist_dir is set to a path that
        # already exists as a file (not a dir) — append should
        # still succeed (in-memory) and not raise
        ring2 = SlashEventRing.__new__(SlashEventRing)
        ring2._max_entries = 256
        from collections import deque
        ring2._entries = deque(maxlen=256)
        ring2._persist_dir = target  # Not a dir!
        # Should NOT raise
        ring2.append(
            kind="proof_failure_slashed",
            provider="0xP", challenger="0xC",
            slash_id=b"\x01" * 32,
        )
        # In-memory ring still has the entry
        assert ring2.count() == 1
