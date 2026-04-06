"""Tests for Ring 6 production hardening."""

import pytest
from unittest.mock import MagicMock


class TestDynamicGasPricing:
    def test_gas_estimator_returns_wei(self):
        from prsm.economy.ftns_onchain import estimate_gas_price
        mock_w3 = MagicMock()
        mock_w3.eth.gas_price = 3_000_000_000  # 3 gwei in wei
        gas = estimate_gas_price(mock_w3, multiplier=1.2)
        assert gas > 0
        assert gas == int(3_000_000_000 * 1.2)

    def test_gas_estimator_fallback_on_error(self):
        from prsm.economy.ftns_onchain import estimate_gas_price, DEFAULT_GAS_GWEI
        mock_w3 = MagicMock()
        type(mock_w3.eth).gas_price = property(lambda self: (_ for _ in ()).throw(Exception("RPC error")))
        gas = estimate_gas_price(mock_w3, multiplier=1.0)
        assert gas == DEFAULT_GAS_GWEI * 1_000_000_000

    def test_gas_estimator_caps_at_max(self):
        from prsm.economy.ftns_onchain import estimate_gas_price
        mock_w3 = MagicMock()
        mock_w3.eth.gas_price = 500_000_000_000  # 500 gwei
        gas = estimate_gas_price(mock_w3, multiplier=1.0, max_gwei=50)
        assert gas <= 50_000_000_000


class TestRPCFailover:
    def test_failover_config_fields(self):
        from prsm.node.config import NodeConfig
        config = NodeConfig()
        assert hasattr(config, 'base_rpc_urls')
        assert isinstance(config.base_rpc_urls, list)
        assert len(config.base_rpc_urls) >= 1

    def test_failover_rotates_on_failure(self):
        from prsm.economy.ftns_onchain import RPCFailover
        failover = RPCFailover(urls=[
            "https://mainnet.base.org",
            "https://base-rpc.example.com",
            "https://base-backup.example.com",
        ])
        assert failover.current_url == "https://mainnet.base.org"
        failover.mark_failed()
        assert failover.current_url == "https://base-rpc.example.com"
        failover.mark_failed()
        assert failover.current_url == "https://base-backup.example.com"
        failover.mark_failed()
        assert failover.current_url == "https://mainnet.base.org"


class TestSettlerSignatureVerification:
    def test_verify_function_exists(self):
        from prsm.node.settler_registry import verify_settler_signature
        assert callable(verify_settler_signature)

    def test_empty_signature_rejected(self):
        from prsm.node.settler_registry import verify_settler_signature
        assert not verify_settler_signature("key", b"msg", "")

    def test_invalid_signature_rejected(self):
        from prsm.node.settler_registry import verify_settler_signature
        # Random base64 that's not a valid Ed25519 signature
        assert not verify_settler_signature(
            "aW52YWxpZC1rZXk=",
            b"test message",
            "aW52YWxpZC1zaWduYXR1cmU=",
        )
