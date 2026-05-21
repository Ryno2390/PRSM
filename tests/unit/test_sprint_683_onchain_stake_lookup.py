"""Sprint 683 — on-chain stake reads layer atop sprint 682's
DHT-backed GpuPoolProvider.

Peers that include `operator_address` in their hardware_profile +
PRSM_STAKE_BOND_ADDRESS + PRSM_BASE_RPC_URL are configured →
provider calls StakeManagerClient.stake_of(addr).amount_wei and
populates ParallaxGPU.stake_amount accordingly.

Absent operator_address / unconfigured contract / chain error →
fall back to 0 (sprint 682 behavior). Adapter C (stake-weighted)
in the trust stack already filters unstaked nodes downstream, so
graceful degradation is the safe default.

A 60s TTL cache prevents the provider from hitting the RPC on
every gpu_pool_provider() invocation (called per request).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


def test_stake_lookup_returns_zero_when_no_operator_address(monkeypatch):
    """Peer omits operator_address → stake_amount stays 0."""
    monkeypatch.setenv(
        "PRSM_STAKE_BOND_ADDRESS",
        "0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed",
    )
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    from prsm.node.onchain_stake_reader import OnChainStakeReader
    reader = OnChainStakeReader()
    assert reader.stake_amount_for(None) == 0
    assert reader.stake_amount_for("") == 0


def test_stake_lookup_returns_zero_when_contract_unconfigured(monkeypatch):
    """No PRSM_STAKE_BOND_ADDRESS env → reader returns 0 without
    attempting any RPC call."""
    monkeypatch.delenv("PRSM_STAKE_BOND_ADDRESS", raising=False)
    from prsm.node.onchain_stake_reader import OnChainStakeReader
    reader = OnChainStakeReader()
    assert reader.stake_amount_for("0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2") == 0


def test_stake_lookup_returns_amount_wei_when_configured(monkeypatch):
    """Configured + address present → reader returns amount_wei
    from StakeManagerClient.stake_of."""
    monkeypatch.setenv(
        "PRSM_STAKE_BOND_ADDRESS",
        "0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed",
    )
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")

    fake_record = MagicMock()
    fake_record.amount_wei = 5000_000000000000000000  # 5000 FTNS wei

    fake_client = MagicMock()
    fake_client.stake_of.return_value = fake_record

    from prsm.node.onchain_stake_reader import OnChainStakeReader
    reader = OnChainStakeReader(client_factory=lambda: fake_client)
    addr = "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2"
    assert reader.stake_amount_for(addr) == 5000_000000000000000000
    fake_client.stake_of.assert_called_once_with(addr)


def test_stake_lookup_caches_results_within_ttl(monkeypatch):
    """Repeated calls within TTL → ONE RPC call. Defends against
    the gpu_pool_provider() being called per-request."""
    monkeypatch.setenv(
        "PRSM_STAKE_BOND_ADDRESS",
        "0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed",
    )
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    fake_record = MagicMock()
    fake_record.amount_wei = 100
    fake_client = MagicMock()
    fake_client.stake_of.return_value = fake_record
    from prsm.node.onchain_stake_reader import OnChainStakeReader
    reader = OnChainStakeReader(
        client_factory=lambda: fake_client, ttl_seconds=60.0,
    )
    addr = "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2"
    for _ in range(5):
        reader.stake_amount_for(addr)
    assert fake_client.stake_of.call_count == 1


def test_stake_lookup_refreshes_after_ttl_expiry(monkeypatch):
    monkeypatch.setenv(
        "PRSM_STAKE_BOND_ADDRESS",
        "0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed",
    )
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    fake_record = MagicMock()
    fake_record.amount_wei = 100
    fake_client = MagicMock()
    fake_client.stake_of.return_value = fake_record
    from prsm.node.onchain_stake_reader import OnChainStakeReader

    fake_now = [1000.0]

    def _clock():
        return fake_now[0]

    reader = OnChainStakeReader(
        client_factory=lambda: fake_client,
        ttl_seconds=60.0,
        clock=_clock,
    )
    addr = "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2"
    reader.stake_amount_for(addr)
    fake_now[0] += 61.0  # past TTL
    reader.stake_amount_for(addr)
    assert fake_client.stake_of.call_count == 2


def test_stake_lookup_returns_zero_on_rpc_exception(monkeypatch):
    """RPC raises (network down, contract reverted, etc.) → reader
    returns 0 + does NOT propagate the exception. The pool provider
    cannot afford to crash the daemon over a transient chain
    error."""
    monkeypatch.setenv(
        "PRSM_STAKE_BOND_ADDRESS",
        "0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed",
    )
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    fake_client = MagicMock()
    fake_client.stake_of.side_effect = RuntimeError("rpc down")
    from prsm.node.onchain_stake_reader import OnChainStakeReader
    reader = OnChainStakeReader(client_factory=lambda: fake_client)
    addr = "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2"
    assert reader.stake_amount_for(addr) == 0


def test_pool_provider_populates_stake_from_operator_address(monkeypatch):
    """End-to-end: hardware_profile carrying operator_address →
    ParallaxGPU.stake_amount populated via stake reader."""
    monkeypatch.setenv(
        "PRSM_STAKE_BOND_ADDRESS",
        "0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed",
    )
    monkeypatch.setenv("PRSM_BASE_RPC_URL", "https://mainnet.base.org")
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo

    class _StubReader:
        def stake_amount_for(self, addr):
            return 7777 if addr else 0

    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {
        "peerA": PeerInfo(
            node_id="peerA",
            address="1.2.3.4:9001",
            hardware_profile={
                "tflops_fp16": 4.6,
                "ram_total_gb": 16.0,
                "operator_address": "0xF7d88c943B048dAd2e5178E40DaaD545dB3311c2",
            },
        ),
    }
    provider = build_dht_backed_pool_provider(
        node, stake_reader=_StubReader(),
    )
    gpus = list(provider())
    assert len(gpus) == 1
    assert gpus[0].stake_amount == 7777


def test_pool_provider_keeps_zero_stake_when_no_operator_address(monkeypatch):
    """No operator_address in hardware_profile → stake_amount=0
    (sprint 682 behavior preserved)."""
    monkeypatch.setenv(
        "PRSM_STAKE_BOND_ADDRESS",
        "0xD4C6584BB69d1cc46B32502c57124Df12D8979Ed",
    )
    from prsm.node.dht_backed_pool_provider import (
        build_dht_backed_pool_provider,
    )
    from prsm.node.discovery import PeerInfo

    class _CountingReader:
        def __init__(self):
            self.calls = 0
        def stake_amount_for(self, addr):
            self.calls += 1
            return 9999

    node = MagicMock()
    node.identity.node_id = "a" * 32
    node.discovery._local_hardware_profile = None
    node.discovery.known_peers = {
        "peerA": PeerInfo(
            node_id="peerA",
            address="1.2.3.4:9001",
            hardware_profile={
                "tflops_fp16": 4.6, "ram_total_gb": 16.0,
                # no operator_address
            },
        ),
    }
    reader = _CountingReader()
    provider = build_dht_backed_pool_provider(node, stake_reader=reader)
    gpus = list(provider())
    assert gpus[0].stake_amount == 0
    # Don't call into the reader at all when no address present
    assert reader.calls == 0
