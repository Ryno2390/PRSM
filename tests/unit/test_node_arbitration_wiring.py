"""PRSM-PROV-1 Item 6 T6.5.gov.next2 — node-startup arbitration wiring.

Tests the three builder helpers in ``prsm/node/node.py`` that
construct the disputed-band components ContentUploader needs:
  - ThresholdResolver (T6.2)
  - FilesystemArbitrationQueue (T6.5)
  - TokenWeightedVotingProposalSink (T6.5.gov.next)

All three must degrade to None on failure so a misconfigured node
still serves uploads (legacy 2-band behavior).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.node.node import (
    _build_arbitration_proposal_sink_or_none,
    _build_arbitration_queue_or_none,
    _build_threshold_resolver_or_none,
)


# ──────────────────────────────────────────────────────────────────────
# ThresholdResolver
# ──────────────────────────────────────────────────────────────────────


class TestBuildThresholdResolver:
    def test_returns_resolver_for_canonical_yaml(self):
        r = _build_threshold_resolver_or_none()
        assert r is not None
        # Verify the resolver actually works on a known kind.
        eff = r.resolve("text-vector")
        assert eff.derivative > 0

    def test_returns_none_when_yaml_load_fails(self):
        # Patch from_default_path to simulate IO failure.
        with patch(
            "prsm.data.dedup.thresholds.ThresholdResolver.from_default_path",
            side_effect=OSError("yaml unreadable"),
        ):
            assert _build_threshold_resolver_or_none() is None


# ──────────────────────────────────────────────────────────────────────
# FilesystemArbitrationQueue
# ──────────────────────────────────────────────────────────────────────


class TestBuildArbitrationQueue:
    def test_returns_queue_with_default_path(self, tmp_path, monkeypatch):
        # Override Path.home() so we don't pollute the real ~/.prsm.
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        q = _build_arbitration_queue_or_none()
        assert q is not None
        # The queue dir was created.
        assert (tmp_path / ".prsm" / "arbitration_queue").exists()

    def test_returns_none_when_construction_fails(self):
        # Patch FilesystemArbitrationQueue to raise on construct.
        with patch(
            "prsm.data.dedup.arbitration.FilesystemArbitrationQueue",
            side_effect=PermissionError("read-only fs"),
        ):
            assert _build_arbitration_queue_or_none() is None


# ──────────────────────────────────────────────────────────────────────
# TokenWeightedVotingProposalSink
# ──────────────────────────────────────────────────────────────────────


class TestBuildArbitrationProposalSink:
    def test_returns_none_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PRSM_ARBITRATION_PROPOSER_ID", raising=False)
        assert _build_arbitration_proposal_sink_or_none() is None

    def test_returns_none_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("PRSM_ARBITRATION_PROPOSER_ID", "")
        assert _build_arbitration_proposal_sink_or_none() is None

    def test_returns_none_when_env_whitespace(self, monkeypatch):
        monkeypatch.setenv("PRSM_ARBITRATION_PROPOSER_ID", "   ")
        assert _build_arbitration_proposal_sink_or_none() is None

    def test_returns_sink_when_env_set(self, monkeypatch):
        monkeypatch.setenv(
            "PRSM_ARBITRATION_PROPOSER_ID", "0xfoundation",
        )
        # TokenWeightedVoting construction can pull in the FTNS
        # service. Patch it out so the test doesn't require live
        # FTNS plumbing.
        with patch(
            "prsm.economy.governance.voting.TokenWeightedVoting",
        ) as mock_voting:
            mock_voting.return_value = object()  # any non-None backend
            sink = _build_arbitration_proposal_sink_or_none()
        assert sink is not None
        # The sink stores the proposer_id we configured.
        assert sink._proposer_id == "0xfoundation"

    def test_returns_none_when_voting_construction_raises(
        self, monkeypatch,
    ):
        monkeypatch.setenv(
            "PRSM_ARBITRATION_PROPOSER_ID", "0xfoundation",
        )
        with patch(
            "prsm.economy.governance.voting.TokenWeightedVoting",
            side_effect=RuntimeError("FTNS service unavailable"),
        ):
            assert _build_arbitration_proposal_sink_or_none() is None
