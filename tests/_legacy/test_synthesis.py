"""
Tests for Sub-phase 10.4: Nightly Synthesis & Project Ledger

Coverage:
  1. Signer — hash_content, chain_hash, sign/verify single entry, verify_chain
  2. Reconstructor — template synthesis, LLM synthesis (mocked), empty snapshot
  3. ProjectLedger — append, read, verify, onboarding context, JSON persistence
  4. DAGAnchor — no-op path, success path (mocked), error path
  5. End-to-end — snapshot → synthesise → append → verify → onboard
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.nwtn.synthesis import (
    AnchorReceipt,
    DAGAnchor,
    EntrySignature,
    GENESIS_HASH,
    LedgerEntry,
    LedgerSigner,
    NarrativeSynthesizer,
    ProjectLedger,
    SynthesisResult,
    VerificationResult,
    hash_content,
)
from prsm.compute.nwtn.synthesis.signer import _compute_chain_hash


# ======================================================================
# Helpers / fixtures
# ======================================================================

def _make_synthesis(
    session_id: str = "sess-test",
    narrative: str = "## Session\n\nAll tests pass.",
    agents: list = None,
    n_entries: int = 5,
) -> SynthesisResult:
    return SynthesisResult(
        session_id=session_id,
        narrative=narrative,
        timestamp=datetime.now(timezone.utc),
        whiteboard_entry_count=n_entries,
        agents_involved=agents or ["agent/coder-20260326"],
        llm_assisted=False,
    )


def _make_snapshot(session_id: str = "sess-test", n_entries: int = 5):
    """Create a minimal WhiteboardSnapshot-like object for testing."""
    from prsm.compute.nwtn.whiteboard.schema import WhiteboardEntry, WhiteboardSnapshot

    entries = [
        WhiteboardEntry(
            id=i,
            session_id=session_id,
            source_agent=f"agent/coder-20260326",
            chunk=f"Discovery {i}: important finding about the system.",
            surprise_score=min(0.99, 0.55 + i * 0.04),  # capped at 0.99 (Pydantic max=1)
            raw_perplexity=50.0 + i * 5,
            similarity_score=0.1,
            timestamp=datetime.now(timezone.utc),
        )
        for i in range(n_entries)
    ]
    return WhiteboardSnapshot(
        session_id=session_id,
        entries=entries,
        created_at=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc),
    )


@pytest.fixture
def signer():
    s = LedgerSigner()
    s.load_or_generate()
    return s


@pytest.fixture
def ledger(tmp_path):
    l = ProjectLedger(ledger_dir=tmp_path / "ledger", project_title="Test Project")
    l.load()
    return l


# ======================================================================
# 1. Signer
# ======================================================================

class TestHashAndChain:
    def test_hash_content_is_hex_sha256(self):
        h = hash_content("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_content_deterministic(self):
        assert hash_content("test") == hash_content("test")

    def test_hash_content_sensitive(self):
        assert hash_content("test") != hash_content("Test")

    def test_compute_chain_hash_combines_both(self):
        ch = _compute_chain_hash("content_hash_abc", "previous_hash_xyz")
        assert len(ch) == 64
        # Different inputs → different chain hash
        assert ch != _compute_chain_hash("content_hash_abc", "different_previous")
        assert ch != _compute_chain_hash("different_content", "previous_hash_xyz")

    def test_genesis_hash_is_64_zeros(self):
        assert GENESIS_HASH == "0" * 64


class TestLedgerSigner:
    def test_generates_keypair_on_first_use(self, signer):
        assert signer.public_key_b64
        assert len(signer.public_key_b64) > 0

    def test_sign_entry_returns_signature(self, signer):
        sig = signer.sign_entry("abc123", GENESIS_HASH)
        assert isinstance(sig, EntrySignature)
        assert sig.signature_b64
        assert sig.chain_hash
        assert sig.public_key_b64 == signer.public_key_b64

    def test_sign_entry_chain_hash_correct(self, signer):
        content_hash = hash_content("some content")
        sig = signer.sign_entry(content_hash, GENESIS_HASH)
        expected_chain = _compute_chain_hash(content_hash, GENESIS_HASH)
        assert sig.chain_hash == expected_chain

    def test_verify_entry_valid(self, signer):
        content = "This is valid content."
        content_hash = hash_content(content)
        sig = signer.sign_entry(content_hash, GENESIS_HASH)

        ok = LedgerSigner.verify_entry(
            content=content,
            content_hash=content_hash,
            previous_hash=GENESIS_HASH,
            entry_sig=sig,
        )
        assert ok

    def test_verify_entry_tampered_content(self, signer):
        content = "Original content."
        content_hash = hash_content(content)
        sig = signer.sign_entry(content_hash, GENESIS_HASH)

        ok = LedgerSigner.verify_entry(
            content="Tampered content!",  # ← modified
            content_hash=content_hash,
            previous_hash=GENESIS_HASH,
            entry_sig=sig,
        )
        assert not ok

    def test_verify_entry_wrong_previous_hash(self, signer):
        content = "Valid content."
        content_hash = hash_content(content)
        sig = signer.sign_entry(content_hash, GENESIS_HASH)

        ok = LedgerSigner.verify_entry(
            content=content,
            content_hash=content_hash,
            previous_hash="wrong_previous_hash",  # ← modified
            entry_sig=sig,
        )
        assert not ok

    def test_verify_chain_empty(self):
        result = LedgerSigner.verify_chain([])
        assert result.valid
        assert result.entry_count == 0

    def test_verify_chain_single_entry(self, signer, tmp_path):
        ledger = ProjectLedger(ledger_dir=tmp_path / "l")
        ledger.load()
        entry = ledger.append(_make_synthesis(), signer)
        result = LedgerSigner.verify_chain(ledger._entries)
        assert result.valid
        assert result.entry_count == 1

    def test_verify_chain_multiple_entries(self, signer, tmp_path):
        ledger = ProjectLedger(ledger_dir=tmp_path / "l")
        ledger.load()
        for i in range(5):
            ledger.append(_make_synthesis(session_id=f"sess-{i}", narrative=f"Entry {i}"), signer)
        result = LedgerSigner.verify_chain(ledger._entries)
        assert result.valid
        assert result.entry_count == 5

    def test_verify_chain_detects_tampering(self, signer, tmp_path):
        ledger = ProjectLedger(ledger_dir=tmp_path / "l")
        ledger.load()
        for i in range(3):
            ledger.append(_make_synthesis(narrative=f"Session {i}"), signer)

        # Tamper with middle entry's content
        e = ledger._entries[1]
        # Build a new entry with tampered content but same hashes (fraud attempt)
        tampered = LedgerEntry(
            entry_index=e.entry_index,
            session_id=e.session_id,
            content="TAMPERED CONTENT",  # ← changed
            timestamp=e.timestamp,
            content_hash=e.content_hash,    # ← original hash (mismatch!)
            chain_hash=e.chain_hash,
            previous_hash=e.previous_hash,
            signature_b64=e.signature_b64,
            public_key_b64=e.public_key_b64,
        )
        ledger._entries[1] = tampered

        result = LedgerSigner.verify_chain(ledger._entries)
        assert not result.valid
        assert result.first_bad_index == 1

    def test_key_persistence(self, tmp_path):
        keyfile = tmp_path / "test.key"
        s1 = LedgerSigner(keyfile_path=keyfile)
        s1.load_or_generate()
        pub1 = s1.public_key_b64

        # Load from same keyfile
        s2 = LedgerSigner(keyfile_path=keyfile)
        s2.load_or_generate()
        pub2 = s2.public_key_b64

        assert pub1 == pub2  # same key loaded


# ======================================================================
# 2. Reconstructor
# ======================================================================

class TestNarrativeSynthesizer:
    @pytest.mark.asyncio
    async def test_template_synthesis_from_snapshot(self):
        synthesizer = NarrativeSynthesizer(backend_registry=None)
        snapshot = _make_snapshot()
        result = await synthesizer.synthesise(snapshot)

        assert isinstance(result, SynthesisResult)
        assert result.session_id == "sess-test"
        assert len(result.narrative) > 0
        assert not result.llm_assisted
        assert result.whiteboard_entry_count == 5

    @pytest.mark.asyncio
    async def test_template_contains_agent_table(self):
        synthesizer = NarrativeSynthesizer(backend_registry=None)
        snapshot = _make_snapshot()
        result = await synthesizer.synthesise(snapshot)
        assert "Agent Contributions" in result.narrative

    @pytest.mark.asyncio
    async def test_template_highlights_pivots(self):
        synthesizer = NarrativeSynthesizer(backend_registry=None, pivot_threshold=0.60)
        snapshot = _make_snapshot(n_entries=10)
        result = await synthesizer.synthesise(snapshot)
        # Some entries have score > 0.60 (0.60, 0.65, ... 0.95)
        assert result.pivot_count > 0
        assert "Pivots" in result.narrative

    @pytest.mark.asyncio
    async def test_template_with_meta_plan(self):
        from prsm.compute.nwtn.team.planner import MetaPlan, Milestone, AgentRole

        meta_plan = MetaPlan(
            session_id="sess-test",
            title="Test Project",
            objective="Build the BSC system",
            milestones=[
                Milestone(title="BSC Core", description="Predictor + filters"),
            ],
            required_roles=[AgentRole(role="coder", description="impl")],
        )
        synthesizer = NarrativeSynthesizer(backend_registry=None)
        snapshot = _make_snapshot()
        result = await synthesizer.synthesise(snapshot, meta_plan=meta_plan)
        assert "Milestone" in result.narrative

    @pytest.mark.asyncio
    async def test_llm_path_uses_backend(self):
        backend = MagicMock()
        backend.generate = AsyncMock(return_value=MagicMock(
            text="### What was accomplished\n\nWe built the BSC system.\n\n"
                 "### Pivots\n\nNone.\n\n### Pending\n\nPhase 10.5."
        ))
        synthesizer = NarrativeSynthesizer(backend_registry=backend)
        snapshot = _make_snapshot()
        result = await synthesizer.synthesise(snapshot)

        assert result.llm_assisted
        assert "BSC system" in result.narrative
        backend.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self):
        backend = MagicMock()
        backend.generate = AsyncMock(side_effect=Exception("API error"))
        synthesizer = NarrativeSynthesizer(backend_registry=backend)
        snapshot = _make_snapshot()
        result = await synthesizer.synthesise(snapshot)
        # Should fall back to template
        assert not result.llm_assisted

    @pytest.mark.asyncio
    async def test_empty_snapshot(self):
        from prsm.compute.nwtn.whiteboard.schema import WhiteboardSnapshot

        empty_snap = WhiteboardSnapshot(session_id="empty-sess", entries=[])
        synthesizer = NarrativeSynthesizer(backend_registry=None)
        result = await synthesizer.synthesise(empty_snap)

        assert result.whiteboard_entry_count == 0
        assert "empty" in result.narrative.lower() or result.narrative

    @pytest.mark.asyncio
    async def test_agents_involved_populated(self):
        synthesizer = NarrativeSynthesizer(backend_registry=None)
        snapshot = _make_snapshot()
        result = await synthesizer.synthesise(snapshot)
        assert "agent/coder-20260326" in result.agents_involved


# ======================================================================
# 3. ProjectLedger
# ======================================================================

class TestProjectLedger:
    def test_load_creates_files(self, tmp_path):
        l = ProjectLedger(ledger_dir=tmp_path / "l")
        l.load()
        assert (tmp_path / "l" / "project_ledger.md").exists()

    def test_initial_entry_count_zero(self, ledger):
        assert ledger.entry_count == 0

    def test_append_increments_count(self, ledger, signer):
        ledger.append(_make_synthesis(), signer)
        assert ledger.entry_count == 1

    def test_append_returns_entry_with_id(self, ledger, signer):
        entry = ledger.append(_make_synthesis(), signer)
        assert entry.entry_index == 0
        assert entry.content_hash
        assert entry.chain_hash
        assert entry.signature_b64
        assert entry.public_key_b64

    def test_first_entry_previous_hash_is_genesis(self, ledger, signer):
        entry = ledger.append(_make_synthesis(), signer)
        assert entry.previous_hash == GENESIS_HASH

    def test_second_entry_previous_hash_links_to_first(self, ledger, signer):
        e1 = ledger.append(_make_synthesis(narrative="First"), signer)
        e2 = ledger.append(_make_synthesis(narrative="Second"), signer)
        assert e2.previous_hash == e1.chain_hash

    def test_chain_is_valid_after_multiple_appends(self, ledger, signer):
        for i in range(7):
            ledger.append(_make_synthesis(narrative=f"Session {i}"), signer)
        result = ledger.verify()
        assert result.valid
        assert result.entry_count == 7

    def test_get_entry_by_index(self, ledger, signer):
        ledger.append(_make_synthesis(narrative="first"), signer)
        ledger.append(_make_synthesis(narrative="second"), signer)
        e = ledger.get_entry(1)
        assert "second" in e.content

    def test_get_entry_out_of_range(self, ledger, signer):
        with pytest.raises(IndexError):
            ledger.get_entry(99)

    def test_latest_entry(self, ledger, signer):
        assert ledger.latest_entry() is None
        ledger.append(_make_synthesis(narrative="only"), signer)
        assert "only" in ledger.latest_entry().content

    def test_latest_chain_hash_genesis_when_empty(self, ledger):
        assert ledger.latest_chain_hash() == GENESIS_HASH

    def test_latest_chain_hash_after_append(self, ledger, signer):
        e = ledger.append(_make_synthesis(), signer)
        assert ledger.latest_chain_hash() == e.chain_hash

    def test_markdown_file_contains_content(self, ledger, signer):
        ledger.append(_make_synthesis(narrative="## Test\n\nImportant info."), signer)
        md = ledger.read_markdown()
        assert "Important info" in md

    def test_markdown_file_contains_metadata(self, ledger, signer):
        ledger.append(_make_synthesis(), signer)
        md = ledger.read_markdown()
        assert "Content Hash" in md
        assert "Chain Hash" in md
        assert "Signature" in md

    def test_json_persistence_across_instances(self, tmp_path, signer):
        ldir = tmp_path / "ledger"

        l1 = ProjectLedger(ledger_dir=ldir)
        l1.load()
        e = l1.append(_make_synthesis(narrative="persisted"), signer)
        assert (ldir / "project_ledger.json").exists()

        l2 = ProjectLedger(ledger_dir=ldir)
        l2.load()
        assert l2.entry_count == 1
        assert "persisted" in l2.get_entry(0).content

    def test_verify_detects_tampered_entry(self, ledger, signer):
        for i in range(3):
            ledger.append(_make_synthesis(narrative=f"Entry {i}"), signer)

        # Directly corrupt an entry in the list
        e = ledger._entries[0]
        ledger._entries[0] = LedgerEntry(
            entry_index=e.entry_index, session_id=e.session_id,
            content="TAMPERED", timestamp=e.timestamp,
            content_hash=e.content_hash,  # stale hash
            chain_hash=e.chain_hash, previous_hash=e.previous_hash,
            signature_b64=e.signature_b64, public_key_b64=e.public_key_b64,
        )

        result = ledger.verify()
        assert not result.valid
        assert result.first_bad_index == 0

    def test_update_dag_anchor(self, ledger, signer):
        ledger.append(_make_synthesis(), signer)
        ledger.update_dag_anchor(0, "dag-tx-abc123")
        assert ledger.get_entry(0).dag_anchor_tx == "dag-tx-abc123"

    def test_update_dag_anchor_persisted(self, tmp_path, signer):
        ldir = tmp_path / "ledger"
        l1 = ProjectLedger(ledger_dir=ldir)
        l1.load()
        l1.append(_make_synthesis(), signer)
        l1.update_dag_anchor(0, "dag-tx-xyz")

        l2 = ProjectLedger(ledger_dir=ldir)
        l2.load()
        assert l2.get_entry(0).dag_anchor_tx == "dag-tx-xyz"

    def test_onboarding_context_not_empty(self, ledger, signer):
        for i in range(5):
            ledger.append(_make_synthesis(narrative=f"Session {i} work"), signer)
        ctx = ledger.to_onboarding_context()
        assert len(ctx) > 0
        assert "Test Project" in ctx

    def test_onboarding_context_respects_max_entries(self, tmp_path, signer):
        ldir = tmp_path / "ledger"
        l = ProjectLedger(ledger_dir=ldir)
        l.load()
        for i in range(10):
            l.append(_make_synthesis(narrative=f"Entry {i} content"), signer)
        ctx = l.to_onboarding_context(max_entries=3)
        # Should only include entries 7, 8, 9
        assert "Entry 9" in ctx
        assert "Entry 0" not in ctx  # old entry excluded

    def test_entry_markdown_block_structure(self, ledger, signer):
        entry = ledger.append(_make_synthesis(narrative="## Test\n\nContent."), signer)
        block = entry.to_markdown_block()
        assert f"Entry #{entry.entry_index}" in block
        assert "Content Hash" in block
        assert "Signature" in block

    def test_onboarding_context_empty_ledger(self, ledger):
        ctx = ledger.to_onboarding_context()
        assert "empty" in ctx.lower() or "No history" in ctx


# ======================================================================
# 4. DAGAnchor
# ======================================================================

class TestDAGAnchor:
    @pytest.mark.asyncio
    async def test_no_dag_returns_unavailable(self):
        anchor = DAGAnchor(dag_ledger=None)
        receipt = await anchor.anchor(0, "abc123", "sess-test")
        assert receipt.status == "unavailable"
        assert not receipt.success

    @pytest.mark.asyncio
    async def test_success_path(self):
        dag = MagicMock()
        dag.submit_transaction = AsyncMock(return_value="dag-tx-001")

        anchor = DAGAnchor(dag_ledger=dag, wallet_id="test-wallet")
        # Patch TransactionType so the import doesn't fail in test env
        with patch("prsm.compute.nwtn.synthesis.dag_anchor.DAGAnchor._submit_anchor",
                   new=AsyncMock(return_value="dag-tx-001")):
            receipt = await anchor.anchor(0, "a" * 64, "sess-test")

        assert receipt.success
        assert receipt.dag_tx_id == "dag-tx-001"
        assert receipt.status == "anchored"

    @pytest.mark.asyncio
    async def test_error_returns_error_status(self):
        dag = MagicMock()

        anchor = DAGAnchor(dag_ledger=dag)
        with patch("prsm.compute.nwtn.synthesis.dag_anchor.DAGAnchor._submit_anchor",
                   new=AsyncMock(side_effect=Exception("network down"))):
            receipt = await anchor.anchor(0, "abc123", "sess-test")

        assert receipt.status == "error"
        assert not receipt.success
        assert "network down" in receipt.error

    @pytest.mark.asyncio
    async def test_receipt_contains_chain_hash(self):
        anchor = DAGAnchor(dag_ledger=None)
        receipt = await anchor.anchor(2, "deadbeef", "sess-x")
        assert receipt.chain_hash == "deadbeef"
        assert receipt.ledger_entry_index == 2


# ======================================================================
# 5. End-to-end: snapshot → synthesise → append → verify → onboard
# ======================================================================

class TestSynthesisEndToEnd:
    @pytest.mark.asyncio
    async def test_full_session_synthesis_flow(self, tmp_path):
        """
        Complete flow: build a snapshot, synthesise it, append to the ledger,
        verify the chain, use the ledger for agent onboarding.
        """
        # 1. Build a snapshot (simulating end of a real work session)
        snapshot = _make_snapshot(session_id="e2e-001", n_entries=8)

        # 2. Synthesise
        synthesizer = NarrativeSynthesizer(backend_registry=None)
        synthesis = await synthesizer.synthesise(snapshot)

        assert synthesis.whiteboard_entry_count == 8
        assert synthesis.session_id == "e2e-001"

        # 3. Sign and append
        signer = LedgerSigner()
        signer.load_or_generate()
        ledger = ProjectLedger(
            ledger_dir=tmp_path / "ledger",
            project_title="PRSM E2E Test",
        )
        ledger.load()
        entry = ledger.append(synthesis, signer)

        assert entry.entry_index == 0
        assert entry.chain_hash != GENESIS_HASH
        assert entry.previous_hash == GENESIS_HASH

        # 4. Verify chain
        result = ledger.verify()
        assert result.valid
        assert result.entry_count == 1

        # 5. Second session
        snapshot2 = _make_snapshot(session_id="e2e-002", n_entries=4)
        synthesis2 = await synthesizer.synthesise(snapshot2)
        entry2 = ledger.append(synthesis2, signer)

        assert entry2.previous_hash == entry.chain_hash
        assert ledger.verify().valid

        # 6. Agent onboarding
        ctx = ledger.to_onboarding_context()
        assert "PRSM E2E Test" in ctx
        assert ledger.entry_count == 2

        # 7. Markdown file exists and has content
        md = ledger.read_markdown()
        assert len(md) > 100
        assert "e2e-001" in md or "Session Summary" in md

    @pytest.mark.asyncio
    async def test_dag_anchor_integration(self, tmp_path):
        """Verify that DAG anchor tx IDs are persisted to the ledger."""
        dag = MagicMock()

        signer = LedgerSigner()
        signer.load_or_generate()
        ledger = ProjectLedger(ledger_dir=tmp_path / "ledger")
        ledger.load()

        synthesis = await NarrativeSynthesizer().synthesise(
            _make_snapshot()
        )
        entry = ledger.append(synthesis, signer)

        anchor = DAGAnchor(dag_ledger=dag)
        with patch("prsm.compute.nwtn.synthesis.dag_anchor.DAGAnchor._submit_anchor",
                   new=AsyncMock(return_value="tx-milestone-1")):
            receipt = await anchor.anchor(
                entry.entry_index, entry.chain_hash, synthesis.session_id
            )
        assert receipt.success
        ledger.update_dag_anchor(entry.entry_index, receipt.dag_tx_id)

        # Reload and confirm persistence
        ledger2 = ProjectLedger(ledger_dir=tmp_path / "ledger")
        ledger2.load()
        assert ledger2.get_entry(0).dag_anchor_tx == "tx-milestone-1"
