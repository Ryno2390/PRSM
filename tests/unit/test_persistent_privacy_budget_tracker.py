"""
Unit tests — Phase 3.x.4 Task 5 — PersistentPrivacyBudgetTracker.

Acceptance per design plan §4 Task 5: drop-in parity with
PrivacyBudgetTracker when store=None; restart simulation restores
total_spent; reset journals as RESET entry; can_spend pre-flight
refuses overspend before journal write; tamper-after-write detected
on next replay.

Real Ed25519 keypairs, real signed journals, real filesystem stores.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.security.privacy_budget import PrivacyBudgetTracker
from prsm.security.privacy_budget_persistence import (
    FilesystemPrivacyBudgetStore,
    InMemoryPrivacyBudgetStore,
    JournalCorruptionError,
    PersistentPrivacyBudgetTracker,
    PrivacyBudgetEntryType,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.4-task5-tracker")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.4-task5-impostor")


@pytest.fixture
def memory_store() -> InMemoryPrivacyBudgetStore:
    return InMemoryPrivacyBudgetStore()


@pytest.fixture
def fs_store(tmp_path, identity) -> FilesystemPrivacyBudgetStore:
    return FilesystemPrivacyBudgetStore(tmp_path, identity.public_key_b64)


# ──────────────────────────────────────────────────────────────────────────
# Construction — both/neither rule + replay
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_no_store_no_identity_is_in_memory_mode(self):
        # No-args construction → behaves exactly like the parent class.
        t = PersistentPrivacyBudgetTracker(max_epsilon=10.0)
        assert t.max_epsilon == 10.0
        assert t.total_spent == 0.0
        # No journal at all
        assert t._store is None
        assert t._identity is None

    def test_store_without_identity_raises(self, memory_store):
        with pytest.raises(ValueError, match="both store AND identity"):
            PersistentPrivacyBudgetTracker(store=memory_store)

    def test_identity_without_store_raises(self, identity):
        with pytest.raises(ValueError, match="both store AND identity"):
            PersistentPrivacyBudgetTracker(identity=identity)

    def test_both_store_and_identity_succeeds(self, memory_store, identity):
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=50.0, store=memory_store, identity=identity
        )
        assert t.max_epsilon == 50.0
        assert t.total_spent == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Drop-in parity (store=None) — all existing PrivacyBudgetTracker
# semantics MUST be preserved when persistence is opt-out.
# ──────────────────────────────────────────────────────────────────────────


class TestParityWithParent:
    """When store=None, the persistent subclass must behave exactly
    like the in-memory parent. Existing call sites in api.py and
    observability code don't pass store=, so this is the
    back-compat path that protects them."""

    def test_record_spend_returns_true_when_under_budget(self):
        t = PersistentPrivacyBudgetTracker(max_epsilon=10.0)
        assert t.record_spend(8.0, "inference") is True
        assert t.total_spent == 8.0

    def test_record_spend_returns_false_when_over_budget(self):
        t = PersistentPrivacyBudgetTracker(max_epsilon=10.0)
        t.record_spend(8.0, "inference")
        assert t.record_spend(4.0, "inference") is False
        assert t.total_spent == 8.0  # second spend NOT recorded

    def test_can_spend_predicate(self):
        t = PersistentPrivacyBudgetTracker(max_epsilon=10.0)
        t.record_spend(7.0, "x")
        assert t.can_spend(3.0) is True
        assert t.can_spend(3.001) is False

    def test_remaining_property(self):
        t = PersistentPrivacyBudgetTracker(max_epsilon=10.0)
        t.record_spend(3.5, "x")
        assert t.remaining == 6.5

    def test_reset_clears_spends(self):
        t = PersistentPrivacyBudgetTracker(max_epsilon=10.0)
        t.record_spend(5.0, "x")
        t.reset()
        assert t.total_spent == 0.0
        assert t.remaining == 10.0

    def test_get_audit_report_shape(self):
        t = PersistentPrivacyBudgetTracker(max_epsilon=10.0)
        t.record_spend(2.0, "inference", model_id="llama-3-8b")
        report = t.get_audit_report()
        # Same shape as parent's report
        assert report["max_epsilon"] == 10.0
        assert report["total_spent"] == 2.0
        assert report["remaining"] == 8.0
        assert report["num_operations"] == 1


# ──────────────────────────────────────────────────────────────────────────
# Persistent mode — record_spend writes to journal
# ──────────────────────────────────────────────────────────────────────────


class TestPersistentRecordSpend:
    def test_basic_spend_writes_journal_entry(self, memory_store, identity):
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=memory_store, identity=identity
        )
        assert t.record_spend(8.0, "inference", model_id="llama-3-8b") is True
        assert len(memory_store) == 1
        assert t.total_spent == 8.0

    def test_journal_entry_carries_spend_metadata(self, memory_store, identity):
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=memory_store, identity=identity
        )
        t.record_spend(4.0, "forge_query", model_id="mistral-7b")
        entry = list(memory_store.replay())[0]
        assert entry.entry_type == PrivacyBudgetEntryType.SPEND
        assert entry.epsilon == 4.0
        assert entry.operation == "forge_query"
        assert entry.model_id == "mistral-7b"
        assert entry.node_id == identity.node_id
        assert len(entry.signature) == 64

    def test_journal_chain_grows_through_multiple_spends(
        self, memory_store, identity
    ):
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=memory_store, identity=identity
        )
        t.record_spend(8.0, "op1")
        t.record_spend(4.0, "op2")
        t.record_spend(1.0, "op3")
        assert len(memory_store) == 3
        assert memory_store.verify_chain(identity.public_key_b64) is True

    def test_pre_flight_refusal_does_not_write_journal(
        self, memory_store, identity
    ):
        # The KEY property: a refused spend must NOT pollute the chain.
        # Otherwise, a malicious / buggy caller could spam can_spend-
        # exceeding requests and fill the journal with rejected entries.
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=10.0, store=memory_store, identity=identity
        )
        t.record_spend(8.0, "ok")
        assert len(memory_store) == 1

        # This spend would push total to 12.0 > 10.0 max → must refuse
        # AND must not write.
        assert t.record_spend(4.0, "rejected") is False
        assert len(memory_store) == 1
        assert t.total_spent == 8.0


# ──────────────────────────────────────────────────────────────────────────
# Persistent mode — reset journals as RESET entry
# ──────────────────────────────────────────────────────────────────────────


class TestPersistentReset:
    def test_reset_writes_journal_entry(self, memory_store, identity):
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=memory_store, identity=identity
        )
        t.record_spend(8.0, "inference")
        t.reset()
        assert len(memory_store) == 2  # SPEND + RESET
        entries = list(memory_store.replay())
        assert entries[1].entry_type == PrivacyBudgetEntryType.RESET
        assert entries[1].epsilon == 0.0
        assert entries[1].operation == ""
        assert entries[1].model_id == ""

    def test_reset_clears_in_memory_state(self, memory_store, identity):
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=memory_store, identity=identity
        )
        t.record_spend(5.0, "x")
        t.reset()
        # In-memory state cleared
        assert t.total_spent == 0.0
        assert t.remaining == 100.0
        # But the journal preserves history (audit trail)
        assert len(memory_store) == 2

    def test_reset_followed_by_spend_chains_correctly(
        self, memory_store, identity
    ):
        t = PersistentPrivacyBudgetTracker(
            max_epsilon=10.0, store=memory_store, identity=identity
        )
        t.record_spend(8.0, "before")
        t.reset()
        # After reset, a new 8.0 spend is allowed (would have failed pre-reset)
        assert t.record_spend(8.0, "after") is True
        assert t.total_spent == 8.0
        assert len(memory_store) == 3  # SPEND + RESET + SPEND
        assert memory_store.verify_chain(identity.public_key_b64) is True


# ──────────────────────────────────────────────────────────────────────────
# Restart simulation — replay reconstitutes total_spent
# ──────────────────────────────────────────────────────────────────────────


class TestRestartSimulation:
    def test_replay_restores_total_spent_filesystem(
        self, tmp_path, identity
    ):
        # Original tracker writes 3 spends to disk
        store_a = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        a = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store_a, identity=identity
        )
        a.record_spend(8.0, "op1")
        a.record_spend(4.0, "op2")
        a.record_spend(1.0, "op3")
        assert a.total_spent == 13.0

        # Fresh process simulates restart — new store, new tracker
        store_b = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        b = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store_b, identity=identity
        )
        # total_spent reconstituted from disk
        assert b.total_spent == 13.0
        assert b.remaining == 87.0

    def test_replay_handles_resets_correctly(self, tmp_path, identity):
        store_a = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        a = PersistentPrivacyBudgetTracker(
            max_epsilon=10.0, store=store_a, identity=identity
        )
        a.record_spend(8.0, "before")
        a.reset()
        a.record_spend(3.0, "after")

        # Restart
        store_b = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        b = PersistentPrivacyBudgetTracker(
            max_epsilon=10.0, store=store_b, identity=identity
        )
        # Only the post-reset spend counts toward total_spent
        assert b.total_spent == 3.0

    def test_replay_skips_can_spend_gate(self, tmp_path, identity):
        # Build a journal whose total_spent (13.0) > tracker's max
        # (would normally fail can_spend during replay if naive).
        store_a = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        a = PersistentPrivacyBudgetTracker(
            max_epsilon=15.0, store=store_a, identity=identity
        )
        a.record_spend(8.0, "x")
        a.record_spend(5.0, "y")
        assert a.total_spent == 13.0

        # Restart with a LOWER max_epsilon — the historical spends
        # are still the ground truth and must not be silently dropped.
        store_b = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        b = PersistentPrivacyBudgetTracker(
            max_epsilon=10.0, store=store_b, identity=identity
        )
        # Total still 13.0 — replay populates _spends directly,
        # bypassing parent's can_spend
        assert b.total_spent == 13.0
        # New spends cannot proceed (over budget)
        assert b.can_spend(0.001) is False

    def test_audit_report_after_replay(self, tmp_path, identity):
        store_a = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        a = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store_a, identity=identity
        )
        a.record_spend(2.0, "inference", model_id="llama-3-8b")
        a.record_spend(2.0, "inference", model_id="mistral-7b")

        store_b = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        b = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store_b, identity=identity
        )
        report = b.get_audit_report()
        assert report["total_spent"] == 4.0
        assert report["num_operations"] == 2
        # Per-spend metadata round-trips: model_id preserved through replay
        spend_models = {s["model_id"] for s in report["spends"]}
        assert spend_models == {"llama-3-8b", "mistral-7b"}


# ──────────────────────────────────────────────────────────────────────────
# Tamper detection at construction — verify_chain runs before replay
# ──────────────────────────────────────────────────────────────────────────


class TestTamperAtConstruction:
    def test_post_write_signature_tamper_caught(
        self, tmp_path, identity
    ):
        # Build a journal, then corrupt one signature on disk.
        store_a = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        a = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store_a, identity=identity
        )
        a.record_spend(8.0, "x")
        a.record_spend(4.0, "y")

        # Tamper signature on entry 1
        entry_path = tmp_path / "entries" / "000001.json"
        data = json.loads(entry_path.read_text())
        sig = data["signature"]
        data["signature"] = f"{int(sig[:2], 16) ^ 0xFF:02x}" + sig[2:]
        entry_path.write_text(json.dumps(data, sort_keys=True, indent=2))

        # Restart MUST refuse rather than silently reconstitute
        # possibly-wrong total_spent
        store_b = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        with pytest.raises(JournalCorruptionError, match="verify_chain"):
            PersistentPrivacyBudgetTracker(
                max_epsilon=100.0, store=store_b, identity=identity
            )

    def test_post_write_field_tamper_caught(self, tmp_path, identity):
        store_a = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        a = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store_a, identity=identity
        )
        a.record_spend(8.0, "x")

        # Reduce ε on disk → operator dodges budget. Must be caught.
        entry_path = tmp_path / "entries" / "000000.json"
        data = json.loads(entry_path.read_text())
        data["epsilon"] = 1.0
        entry_path.write_text(json.dumps(data, sort_keys=True, indent=2))

        store_b = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        with pytest.raises(JournalCorruptionError):
            PersistentPrivacyBudgetTracker(
                max_epsilon=100.0, store=store_b, identity=identity
            )

    def test_wrong_identity_caught(
        self, tmp_path, identity, other_identity
    ):
        # Open with identity A, write spends, then try to open the
        # SAME journal as identity B. The pubkey-sidecar binding on
        # FilesystemPrivacyBudgetStore catches this at store
        # construction (not at tracker construction).
        store_a = FilesystemPrivacyBudgetStore(
            tmp_path, identity.public_key_b64
        )
        a = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store_a, identity=identity
        )
        a.record_spend(8.0, "x")

        # The sidecar binding rejects the wrong key at store init
        with pytest.raises(JournalCorruptionError, match="doesn't match"):
            FilesystemPrivacyBudgetStore(
                tmp_path, other_identity.public_key_b64
            )


# ──────────────────────────────────────────────────────────────────────────
# Cross-impl parity — same tracker semantics with InMemory + Filesystem stores
# ──────────────────────────────────────────────────────────────────────────


class TestCrossStoreParity:
    @pytest.fixture(params=["memory", "filesystem"])
    def parametrized_tracker(self, request, tmp_path, identity):
        if request.param == "memory":
            store = InMemoryPrivacyBudgetStore()
        else:
            store = FilesystemPrivacyBudgetStore(
                tmp_path, identity.public_key_b64
            )
        return PersistentPrivacyBudgetTracker(
            max_epsilon=20.0, store=store, identity=identity
        )

    def test_basic_spend(self, parametrized_tracker):
        assert parametrized_tracker.record_spend(8.0, "x") is True
        assert parametrized_tracker.total_spent == 8.0

    def test_overbudget_refused(self, parametrized_tracker):
        parametrized_tracker.record_spend(15.0, "x")
        assert parametrized_tracker.record_spend(10.0, "y") is False
        assert parametrized_tracker.total_spent == 15.0

    def test_reset_clears_state_but_journals_event(
        self, parametrized_tracker
    ):
        parametrized_tracker.record_spend(5.0, "x")
        parametrized_tracker.reset()
        assert parametrized_tracker.total_spent == 0.0
        assert len(parametrized_tracker._store) == 2  # SPEND + RESET


# ──────────────────────────────────────────────────────────────────────────
# Subclass relationship — must be a real PrivacyBudgetTracker
# ──────────────────────────────────────────────────────────────────────────


class TestInheritance:
    def test_is_a_privacy_budget_tracker(self, memory_store, identity):
        t = PersistentPrivacyBudgetTracker(
            store=memory_store, identity=identity
        )
        assert isinstance(t, PrivacyBudgetTracker)
        # Existing call sites doing isinstance(node.privacy_budget,
        # PrivacyBudgetTracker) keep working.

    def test_default_max_epsilon_matches_parent(self):
        t = PersistentPrivacyBudgetTracker()
        assert t.max_epsilon == 100.0  # parent default
