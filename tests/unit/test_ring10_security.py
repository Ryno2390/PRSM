"""Tests for Ring 10 security hardening modules."""

import pytest
import hashlib

from prsm.security.integrity import IntegrityVerifier
from prsm.security.privacy_budget import PrivacyBudgetTracker
from prsm.security.audit_log import PipelineAuditLog


class TestIntegrityVerifier:
    def test_compute_checksum(self):
        data = b"test data"
        checksum = IntegrityVerifier.compute_checksum(data)
        assert checksum == hashlib.sha256(data).hexdigest()

    def test_verify_shard_valid(self):
        data = b"shard content"
        checksum = hashlib.sha256(data).hexdigest()
        assert IntegrityVerifier.verify_shard(data, checksum)

    def test_verify_shard_tampered(self):
        data = b"shard content"
        assert not IntegrityVerifier.verify_shard(data, "wrong_checksum")

    def test_verify_sharded_model(self):
        verifier = IntegrityVerifier()
        shard_data = b"tensor bytes"
        shards = [
            {"shard_id": "s0", "tensor_data": shard_data, "checksum": hashlib.sha256(shard_data).hexdigest()},
            {"shard_id": "s1", "tensor_data": shard_data, "checksum": hashlib.sha256(shard_data).hexdigest()},
        ]
        valid, errors = verifier.verify_sharded_model(shards)
        assert valid
        assert len(errors) == 0

    def test_verify_sharded_model_tampered(self):
        verifier = IntegrityVerifier()
        shards = [
            {"shard_id": "s0", "tensor_data": b"original", "checksum": "bad_checksum"},
        ]
        valid, errors = verifier.verify_sharded_model(shards)
        assert not valid
        assert "checksum mismatch" in errors[0]

    def test_verify_sharded_model_missing_checksum(self):
        verifier = IntegrityVerifier()
        shards = [{"shard_id": "s0", "tensor_data": b"data", "checksum": ""}]
        valid, errors = verifier.verify_sharded_model(shards)
        assert not valid
        assert "no checksum" in errors[0]


class TestPrivacyBudgetTracker:
    def test_initial_state(self):
        tracker = PrivacyBudgetTracker(max_epsilon=100.0)
        assert tracker.total_spent == 0.0
        assert tracker.remaining == 100.0

    def test_record_spend(self):
        tracker = PrivacyBudgetTracker(max_epsilon=100.0)
        assert tracker.record_spend(8.0, "inference", "model-1")
        assert tracker.total_spent == 8.0
        assert tracker.remaining == 92.0

    def test_budget_exceeded(self):
        tracker = PrivacyBudgetTracker(max_epsilon=10.0)
        tracker.record_spend(8.0, "op1")
        assert not tracker.can_spend(5.0)
        assert not tracker.record_spend(5.0, "op2")

    def test_audit_report(self):
        tracker = PrivacyBudgetTracker(max_epsilon=50.0)
        tracker.record_spend(8.0, "inference", "m1")
        tracker.record_spend(4.0, "inference", "m2")
        report = tracker.get_audit_report()
        assert report["total_spent"] == 12.0
        assert report["remaining"] == 38.0
        assert report["num_operations"] == 2

    def test_reset(self):
        tracker = PrivacyBudgetTracker()
        tracker.record_spend(8.0, "op")
        tracker.reset()
        assert tracker.total_spent == 0.0


class TestPipelineAuditLog:
    def test_record_entry(self):
        log = PipelineAuditLog()
        entry = log.record(
            model_id="model-1",
            shard_count=4,
            node_assignments=[{"node_id": f"n{i}", "shard_index": str(i)} for i in range(4)],
            pool_size=20,
        )
        assert entry.entry_id == "audit-000000"
        assert len(entry.entry_hash) > 0
        assert log.entry_count == 1

    def test_hash_chain_integrity(self):
        log = PipelineAuditLog()
        log.record("m1", 2, [{"node_id": "a"}, {"node_id": "b"}], 10)
        log.record("m1", 2, [{"node_id": "c"}, {"node_id": "d"}], 10)
        log.record("m1", 2, [{"node_id": "e"}, {"node_id": "f"}], 10)
        assert log.verify_chain()

    def test_chain_detects_tampering(self):
        log = PipelineAuditLog()
        log.record("m1", 2, [{"node_id": "a"}], 10)
        log.record("m1", 2, [{"node_id": "b"}], 10)
        # Tamper with first entry's hash
        log._entries[0].entry_hash = "tampered"
        assert not log.verify_chain()

    def test_get_entries(self):
        log = PipelineAuditLog()
        for i in range(5):
            log.record("m1", 2, [{"node_id": f"n{i}"}], 10)
        entries = log.get_entries(limit=3)
        assert len(entries) == 3

    def test_entropy_report(self):
        log = PipelineAuditLog()
        for i in range(10):
            log.record("m1", 2, [{"node_id": f"n{i % 5}"}], 10)
        report = log.entropy_report()
        assert report["entries"] == 10
        assert report["unique_nodes"] == 5
        assert report["chain_valid"] is True
        assert report["entropy_score"] > 0
