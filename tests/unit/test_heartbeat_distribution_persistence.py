"""HeartbeatRecordedRing + DistributedEventRing filesystem
persistence (sprint 92). Symmetric to SlashEventRing
persistence (sprint 91)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prsm.node.heartbeat_log import HeartbeatRecordedRing
from prsm.node.distribution_log import DistributedEventRing


class TestHeartbeatPersist:
    def test_writes_on_append(self, tmp_path: Path):
        ring = HeartbeatRecordedRing(persist_dir=tmp_path)
        ring.append(provider="0xPROV", onchain_timestamp=1700000000)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        d = json.loads(files[0].read_text())
        assert d["provider"] == "0xPROV"
        assert d["onchain_timestamp"] == 1700000000

    def test_rehydrate_loads_prior(self, tmp_path: Path):
        r1 = HeartbeatRecordedRing(persist_dir=tmp_path)
        r1.append(provider="0xP1", onchain_timestamp=1)
        r1.append(provider="0xP2", onchain_timestamp=2)
        r2 = HeartbeatRecordedRing(persist_dir=tmp_path)
        assert r2.count() == 2

    def test_corrupt_file_skipped(self, tmp_path: Path):
        (tmp_path / "junk.json").write_text("not json")
        r = HeartbeatRecordedRing(persist_dir=tmp_path)
        assert r.count() == 0


class TestDistributionPersist:
    def test_writes_on_append(self, tmp_path: Path):
        ring = DistributedEventRing(persist_dir=tmp_path)
        ring.append(to_creator=100, to_operator=50, to_grant=25)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        d = json.loads(files[0].read_text())
        assert d["to_creator"] == 100
        assert d["total_distributed"] == 175

    def test_rehydrate_loads_prior(self, tmp_path: Path):
        r1 = DistributedEventRing(persist_dir=tmp_path)
        r1.append(
            to_creator=100, to_operator=50, to_grant=25,
            timestamp=1700000000,
        )
        r1.append(
            to_creator=200, to_operator=100, to_grant=50,
            timestamp=1700000001,
        )
        r2 = DistributedEventRing(persist_dir=tmp_path)
        assert r2.count() == 2

    def test_corrupt_file_skipped(self, tmp_path: Path):
        (tmp_path / "junk.json").write_text("oops")
        r = DistributedEventRing(persist_dir=tmp_path)
        assert r.count() == 0


class TestNoOpWhenUnset:
    def test_heartbeat_no_persist_when_unset(self, tmp_path: Path):
        ring = HeartbeatRecordedRing()
        ring.append(provider="0xP", onchain_timestamp=1)
        # tmp_path was never set as persist_dir
        assert len(list(tmp_path.glob("*.json"))) == 0

    def test_distribution_no_persist_when_unset(self, tmp_path: Path):
        ring = DistributedEventRing()
        ring.append(to_creator=1, to_operator=2, to_grant=3)
        assert len(list(tmp_path.glob("*.json"))) == 0
