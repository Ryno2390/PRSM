"""
Integration test — Phase 3.x.4 Task 6 — node.py privacy-budget wiring.

Acceptance per design plan §4 Task 6: a node configured with a
``data_dir`` produces a ``PersistentPrivacyBudgetTracker`` whose
journal at ``<data_dir>/privacy_budget/`` survives a node restart.

This exercises the "config-to-tracker" seam without bringing up a
full ``Node`` (which spins up P2P transport, ledger, IPFS, etc.).
The wiring in ``prsm/node/node.py`` is a 4-line block; the slice
under test is the journal layout + restart-survival behavior that
block is responsible for producing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prsm.node.identity import (
    NodeIdentity,
    generate_node_identity,
    load_node_identity,
    save_node_identity,
)
# Phase 3.x.4 round-1 review: import the production wiring factory
# directly rather than re-implementing it in the test. Drift between
# production and test now becomes a compile-time error.
from prsm.node.node import build_persistent_privacy_budget as _wire_privacy_budget
from prsm.security.privacy_budget import PrivacyBudgetTracker
from prsm.security.privacy_budget_persistence import (
    JournalCorruptionError,
    PersistentPrivacyBudgetTracker,
)


def _saved_identity(data_dir: Path) -> NodeIdentity:
    """Generate + save an identity exactly as Node.__init__ does."""
    identity_path = data_dir / "identity.json"
    if identity_path.exists():
        loaded = load_node_identity(identity_path)
        assert loaded is not None
        return loaded
    identity = generate_node_identity("phase3.x.4-task6-test-node")
    save_node_identity(identity, identity_path)
    return identity


# ──────────────────────────────────────────────────────────────────────────
# Wiring shape — produces the right tracker type at the right path
# ──────────────────────────────────────────────────────────────────────────


class TestWiringShape:
    def test_produces_persistent_tracker(self, tmp_path):
        identity = _saved_identity(tmp_path)
        tracker = _wire_privacy_budget(tmp_path, identity)
        # Subclass of PrivacyBudgetTracker (existing call sites work)
        assert isinstance(tracker, PrivacyBudgetTracker)
        assert isinstance(tracker, PersistentPrivacyBudgetTracker)

    def test_journal_directory_created(self, tmp_path):
        identity = _saved_identity(tmp_path)
        _wire_privacy_budget(tmp_path, identity)
        assert (tmp_path / "privacy_budget").is_dir()
        assert (tmp_path / "privacy_budget" / "node.pubkey").exists()
        assert (tmp_path / "privacy_budget" / "entries").is_dir()

    def test_pubkey_sidecar_matches_node_identity(self, tmp_path):
        identity = _saved_identity(tmp_path)
        _wire_privacy_budget(tmp_path, identity)
        sidecar = (tmp_path / "privacy_budget" / "node.pubkey").read_text().strip()
        assert sidecar == identity.public_key_b64

    def test_default_max_epsilon_is_100(self, tmp_path):
        # The wiring hard-codes max_epsilon=100.0 to match what
        # PrivacyBudgetTracker(max_epsilon=100.0) used in 3.x.1.
        identity = _saved_identity(tmp_path)
        tracker = _wire_privacy_budget(tmp_path, identity)
        assert tracker.max_epsilon == 100.0


# ──────────────────────────────────────────────────────────────────────────
# Restart survival — the whole point of Phase 3.x.4
# ──────────────────────────────────────────────────────────────────────────


class TestRestartSurvival:
    def test_total_spent_persists_across_restart(self, tmp_path):
        # Process A: identity + tracker + spends
        identity_a = _saved_identity(tmp_path)
        tracker_a = _wire_privacy_budget(tmp_path, identity_a)
        tracker_a.record_spend(8.0, "inference", model_id="llama-3-8b")
        tracker_a.record_spend(4.0, "forge_query")
        assert tracker_a.total_spent == 12.0

        # Process B (simulates Node restart): re-load identity from disk,
        # re-construct tracker pointing at the same data_dir.
        identity_b = _saved_identity(tmp_path)
        # Same identity round-tripped from disk
        assert identity_b.node_id == identity_a.node_id
        tracker_b = _wire_privacy_budget(tmp_path, identity_b)
        # Total survived
        assert tracker_b.total_spent == 12.0
        assert tracker_b.remaining == 88.0

    def test_audit_report_preserves_per_spend_metadata_across_restart(
        self, tmp_path
    ):
        identity_a = _saved_identity(tmp_path)
        tracker_a = _wire_privacy_budget(tmp_path, identity_a)
        tracker_a.record_spend(2.0, "inference", model_id="llama-3-8b")
        tracker_a.record_spend(2.0, "forge_query", model_id="mistral-7b")

        identity_b = _saved_identity(tmp_path)
        tracker_b = _wire_privacy_budget(tmp_path, identity_b)
        report = tracker_b.get_audit_report()
        assert report["num_operations"] == 2
        ops = {s["operation"] for s in report["spends"]}
        assert ops == {"inference", "forge_query"}
        models = {s["model_id"] for s in report["spends"]}
        assert models == {"llama-3-8b", "mistral-7b"}

    def test_reset_history_preserved_across_restart(self, tmp_path):
        identity_a = _saved_identity(tmp_path)
        tracker_a = _wire_privacy_budget(tmp_path, identity_a)
        tracker_a.record_spend(8.0, "before")
        tracker_a.reset()
        tracker_a.record_spend(3.0, "after")
        assert tracker_a.total_spent == 3.0

        identity_b = _saved_identity(tmp_path)
        tracker_b = _wire_privacy_budget(tmp_path, identity_b)
        # In-memory total reflects post-reset state
        assert tracker_b.total_spent == 3.0
        # Journal preserves the full history (RESET entry survives)
        assert len(tracker_b._store) == 3  # SPEND + RESET + SPEND


# ──────────────────────────────────────────────────────────────────────────
# Tamper detection at restart — node refuses to start with corrupt journal
# ──────────────────────────────────────────────────────────────────────────


class TestTamperAtRestart:
    def test_disk_tamper_blocks_node_init(self, tmp_path):
        # First run: write a couple of spends.
        identity_a = _saved_identity(tmp_path)
        tracker_a = _wire_privacy_budget(tmp_path, identity_a)
        tracker_a.record_spend(8.0, "x")
        tracker_a.record_spend(4.0, "y")

        # Corrupt the second entry's epsilon on disk → operator dodges
        # budget. The next "Node restart" must REFUSE rather than
        # silently reconstitute.
        import json
        entry_path = tmp_path / "privacy_budget" / "entries" / "000001.json"
        data = json.loads(entry_path.read_text())
        data["epsilon"] = 1.0  # was 4.0
        entry_path.write_text(json.dumps(data, sort_keys=True, indent=2))

        identity_b = _saved_identity(tmp_path)
        with pytest.raises(JournalCorruptionError):
            _wire_privacy_budget(tmp_path, identity_b)

    def test_pubkey_swap_blocks_node_init(self, tmp_path):
        # First run: write a spend under identity A.
        identity_a = _saved_identity(tmp_path)
        tracker_a = _wire_privacy_budget(tmp_path, identity_a)
        tracker_a.record_spend(8.0, "x")

        # Adversary replaces identity.json with a different identity
        # AND keeps the journal. Wiring must detect this at the store
        # level (pubkey-sidecar binding) before the tracker is built.
        (tmp_path / "identity.json").unlink()
        identity_b = generate_node_identity("attacker")
        save_node_identity(identity_b, tmp_path / "identity.json")

        loaded = load_node_identity(tmp_path / "identity.json")
        assert loaded is not None
        assert loaded.node_id != identity_a.node_id
        with pytest.raises(JournalCorruptionError, match="doesn't match"):
            _wire_privacy_budget(tmp_path, loaded)


# ──────────────────────────────────────────────────────────────────────────
# Same-process consecutive constructions don't break (idempotence)
# ──────────────────────────────────────────────────────────────────────────


class TestIdempotentRewire:
    def test_multiple_rewires_do_not_corrupt_journal(self, tmp_path):
        # Two trackers using the same store → the second one verifies
        # the chain it inherits from the first. As long as no disk
        # tamper happened between, both work.
        identity = _saved_identity(tmp_path)
        a = _wire_privacy_budget(tmp_path, identity)
        a.record_spend(2.0, "first")

        b = _wire_privacy_budget(tmp_path, identity)
        assert b.total_spent == 2.0
        b.record_spend(3.0, "second")
        assert b.total_spent == 5.0

        c = _wire_privacy_budget(tmp_path, identity)
        assert c.total_spent == 5.0
