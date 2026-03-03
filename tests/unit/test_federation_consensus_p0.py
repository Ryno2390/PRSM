import inspect
from decimal import Decimal

import pytest

from prsm.compute.federation import consensus as consensus_module
from prsm.compute.federation.consensus import ConsensusResult, ConsensusType, DistributedConsensus
from prsm.compute.federation.p2p_network import P2PModelNetwork


def test_consensus_type_has_single_zk_snark_constant_definition():
    """Guard against duplicate constant definitions that silently shadow prior values."""
    source = inspect.getsource(consensus_module.ConsensusType)
    assert source.count('ZK_SNARK = "zk_snark"') == 1


def test_distributed_consensus_has_single_zk_snark_verification_method_definition():
    """Guard against duplicate method definitions that shadow canonical logic."""
    source = inspect.getsource(consensus_module.DistributedConsensus)
    assert source.count("def _zk_snark_verification(") == 1


@pytest.mark.asyncio
async def test_handle_byzantine_failures_applies_ftns_penalty_with_decimal(monkeypatch):
    """Regression: FTNS penalty path must not fail due to missing Decimal import."""
    consensus = DistributedConsensus()

    class _FakeFtnsService:
        def __init__(self):
            self.calls = []

        def deduct_tokens(self, peer_id, amount, description=""):
            self.calls.append((peer_id, amount, description))

    fake_ftns = _FakeFtnsService()
    consensus.ftns_service = fake_ftns

    # Keep test narrow and deterministic (no external safety behavior involved).
    monkeypatch.setattr(consensus_module, "ENABLE_SAFETY_CONSENSUS", False)

    await consensus.handle_byzantine_failures(["peer_faulty_001"])

    assert len(fake_ftns.calls) == 1
    peer_id, amount, description = fake_ftns.calls[0]
    assert peer_id == "peer_faulty_001"
    assert amount == Decimal("1000")
    assert "Byzantine failure penalty" in description


@pytest.mark.asyncio
async def test_p2p_network_consensus_smoke_path_uses_consensus_engine(monkeypatch):
    """Narrow collaboration-critical smoke path through P2P -> consensus usage."""
    network = P2PModelNetwork()

    observed = {"called": False, "consensus_type": None}

    async def _fake_achieve_result_consensus(peer_results, consensus_type, session_id=None):
        observed["called"] = True
        observed["consensus_type"] = consensus_type
        return ConsensusResult(
            agreed_value="ok",
            consensus_achieved=True,
            consensus_type=consensus_type,
            agreement_ratio=1.0,
            participating_peers=[r.get("peer_id", "unknown") for r in peer_results],
        )

    monkeypatch.setattr(network.consensus_engine, "achieve_result_consensus", _fake_achieve_result_consensus)

    peer_results = [
        {"peer_id": "peer_1", "result": "ok"},
        {"peer_id": "peer_2", "result": "ok"},
    ]

    is_valid = await network.validate_peer_contributions(
        peer_results,
        consensus_type=ConsensusType.SIMPLE_MAJORITY,
    )

    assert is_valid is True
    assert observed["called"] is True
    assert observed["consensus_type"] == ConsensusType.SIMPLE_MAJORITY

