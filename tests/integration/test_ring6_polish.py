"""Ring 6 Smoke Test — production hardening verification."""

import pytest
from click.testing import CliRunner


class TestRing6Smoke:
    def test_hardware_benchmark_runs(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["node", "benchmark"])
        assert result.exit_code == 0
        assert "tflops" in result.output.lower() or "tier" in result.output.lower()

    def test_yield_estimate_runs(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["ftns", "yield-estimate", "--hours", "8"])
        assert result.exit_code == 0
        assert "daily" in result.output.lower() or "monthly" in result.output.lower()

    def test_gas_estimator_importable(self):
        from prsm.economy.ftns_onchain import estimate_gas_price, RPCFailover
        assert callable(estimate_gas_price)
        failover = RPCFailover(urls=["https://example.com"])
        assert failover.current_url == "https://example.com"

    def test_settler_verification_importable(self):
        from prsm.node.settler_registry import verify_settler_signature
        assert callable(verify_settler_signature)
        assert not verify_settler_signature("key", b"msg", "")

    def test_all_ring_imports(self):
        from prsm.compute.wasm import WASMRuntime, WasmtimeRuntime, HardwareProfiler
        from prsm.compute.agents import AgentDispatcher, AgentExecutor, MobileAgent
        from prsm.compute.swarm import SwarmCoordinator, SwarmJob, SwarmResult
        from prsm.economy.pricing import PricingEngine, CostQuote, ProsumerTier
        from prsm.economy.prosumer import ProsumerManager
        # Ring 5 AgentForge removed in v1.6.0 (legacy NWTN AGI framework pruned)
        assert all(x is not None for x in [
            WASMRuntime, AgentDispatcher, SwarmCoordinator,
            PricingEngine,
        ])
