"""Unit tests for the result_consensus module."""

import pytest
from prsm.node.result_consensus import ResultConsensus, ConsensusMode, ProviderResult


@pytest.fixture
def consensus():
    return ResultConsensus(epsilon=0.01)


def test_single_mode_first_wins(consensus):
    consensus.start_consensus("j1", ConsensusMode.SINGLE, 2)
    consensus.submit_result("j1", "node-a", {"answer": "42"}, "sig-a")
    state = consensus.get_state("j1")
    assert state.consensus_reached is True
    assert state.agreed_result["answer"] == "42"


def test_majority_mode_agree(consensus):
    consensus.start_consensus("j2", ConsensusMode.MAJORITY, 3)
    result = {"score": 0.95}
    consensus.submit_result("j2", "node-a", result, "sig-a")
    consensus.submit_result("j2", "node-b", result, "sig-b")
    state = consensus.get_state("j2")
    assert state.consensus_reached is True


def test_majority_mode_disagree(consensus):
    consensus.start_consensus("j3", ConsensusMode.MAJORITY, 3)
    consensus.submit_result("j3", "node-a", {"score": 0.95}, "sig-a")
    consensus.submit_result("j3", "node-b", {"score": 0.10}, "sig-b")
    consensus.submit_result("j3", "node-c", {"score": 0.10}, "sig-c")
    state = consensus.get_state("j3")
    # 2/3 agree on 0.10 — that's a majority
    assert state.consensus_reached is True
    assert state.agreed_result["score"] == 0.10


def test_unanimous_all_agree(consensus):
    consensus.start_consensus("j4", ConsensusMode.UNANIMOUS, 3)
    data = {"ok": True}
    consensus.submit_result("j4", "node-a", data, "sig-a")
    consensus.submit_result("j4", "node-b", data, "sig-b")
    consensus.submit_result("j4", "node-c", data, "sig-c")
    state = consensus.get_state("j4")
    assert state.consensus_reached is True


def test_unanimous_disagree_fails(consensus):
    consensus.start_consensus("j5", ConsensusMode.UNANIMOUS, 2)
    consensus.submit_result("j5", "node-a", {"x": 1}, "sig-a")
    consensus.submit_result("j5", "node-b", {"x": 2}, "sig-b")
    state = consensus.get_state("j5")
    assert state.consensus_reached is False
    assert state.error == "Results are not unanimous"


def test_duplicate_submission_ignored(consensus):
    consensus.start_consensus("j6", ConsensusMode.SINGLE, 1)
    consensus.submit_result("j6", "node-a", {"x": 1}, "sig-a")
    consensus.submit_result("j6", "node-a", {"x": 2}, "sig-b")  # same provider
    state = consensus.get_state("j6")
    assert len(state.results) == 1


def test_cancel_consensus(consensus):
    consensus.start_consensus("j7", ConsensusMode.UNANIMOUS, 2)
    assert consensus.cancel_consensus("j7") is True
    state = consensus.get_state("j7")
    assert state.error == "Cancelled"
    assert state.is_complete is True


def test_result_hash_deterministic(consensus):
    r1 = ProviderResult("a", "j", {"x": 1, "y": 2}, "s")
    r2 = ProviderResult("a", "j", {"y": 2, "x": 1}, "s")  # different order
    assert r1.result_hash == r2.result_hash
