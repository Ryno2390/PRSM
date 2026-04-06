# Ring 6 — "The Polish" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the Rings 1-5 infrastructure for production: replace hardcoded gas pricing with dynamic estimation, add RPC failover for on-chain resilience, add CLI commands for the new Ring 1-5 features, and implement real ECDSA signature verification in the settler registry.

**Architecture:** Targeted improvements to existing modules — no new packages. Dynamic gas pricing wraps `eth_gasPrice` RPC calls. RPC failover adds an endpoint rotation list. CLI commands expose the new Ring 1-5 capabilities. Settler signature verification replaces the placeholder with Ed25519 checks using the existing identity infrastructure.

**Tech Stack:** Existing PRSM infrastructure. No new external dependencies.

**Scope note:** This plan covers the highest-impact hardening items. Full API authentication, mDNS discovery, OpenTelemetry extensions, and comprehensive documentation updates are tracked as future work in the spec but not built here — each is its own project.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `prsm/economy/ftns_onchain.py:260` | Dynamic gas pricing + RPC failover |
| Modify | `prsm/node/settler_registry.py:430` | Real signature verification |
| Modify | `prsm/cli.py` | New CLI commands for Ring 1-5 features |
| Modify | `prsm/node/config.py` | RPC failover config fields |
| Create | `tests/unit/test_ring6_hardening.py` | Gas pricing, RPC failover, signature tests |
| Create | `tests/integration/test_ring6_polish.py` | End-to-end hardening smoke test |

---

### Task 1: Dynamic Gas Pricing + RPC Failover

**Files:**
- Modify: `prsm/economy/ftns_onchain.py`
- Modify: `prsm/node/config.py`
- Test: `tests/unit/test_ring6_hardening.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_ring6_hardening.py`:

```python
"""Tests for Ring 6 production hardening."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal


class TestDynamicGasPricing:
    def test_gas_estimator_returns_wei(self):
        from prsm.economy.ftns_onchain import estimate_gas_price
        # Mock web3 provider
        mock_w3 = MagicMock()
        mock_w3.eth.gas_price = 3_000_000_000  # 3 gwei
        mock_w3.to_wei = MagicMock(side_effect=lambda x, unit: int(float(x) * 1e9))

        gas = estimate_gas_price(mock_w3, multiplier=1.2)
        assert gas > 0

    def test_gas_estimator_fallback_on_error(self):
        from prsm.economy.ftns_onchain import estimate_gas_price
        mock_w3 = MagicMock()
        mock_w3.eth.gas_price = property(lambda s: (_ for _ in ()).throw(Exception("RPC error")))
        mock_w3.to_wei = MagicMock(return_value=5_000_000_000)

        # Should fall back to default (5 gwei)
        gas = estimate_gas_price(mock_w3, multiplier=1.0)
        assert gas == 5_000_000_000

    def test_gas_estimator_caps_at_max(self):
        from prsm.economy.ftns_onchain import estimate_gas_price
        mock_w3 = MagicMock()
        mock_w3.eth.gas_price = 500_000_000_000  # 500 gwei (very high)
        mock_w3.to_wei = MagicMock(return_value=50_000_000_000)  # 50 gwei cap

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
        # Wraps around
        assert failover.current_url == "https://mainnet.base.org"


class TestSettlerSignatureVerification:
    def test_valid_signature_accepted(self):
        from prsm.node.settler_registry import verify_settler_signature
        from prsm.node.identity import NodeIdentity

        identity = NodeIdentity.generate()
        message = b"PRSM:batch_hash_abc:settler_001"
        signature = identity.sign(message)

        assert verify_settler_signature(
            public_key_b64=identity.public_key_b64,
            message=message,
            signature_b64=signature,
        )

    def test_invalid_signature_rejected(self):
        from prsm.node.settler_registry import verify_settler_signature
        from prsm.node.identity import NodeIdentity

        identity = NodeIdentity.generate()
        message = b"PRSM:batch_hash_abc:settler_001"

        assert not verify_settler_signature(
            public_key_b64=identity.public_key_b64,
            message=message,
            signature_b64="aW52YWxpZC1zaWduYXR1cmU=",  # Wrong signature
        )

    def test_empty_signature_rejected(self):
        from prsm.node.settler_registry import verify_settler_signature

        assert not verify_settler_signature(
            public_key_b64="some_key",
            message=b"test",
            signature_b64="",
        )
```

- [ ] **Step 2: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_ring6_hardening.py::TestDynamicGasPricing -v`
Expected: FAIL

- [ ] **Step 3: Implement dynamic gas pricing + RPC failover in ftns_onchain.py**

In `prsm/economy/ftns_onchain.py`, add these functions near the top (after imports):

```python
DEFAULT_GAS_GWEI = 5
MAX_GAS_GWEI = 50


class RPCFailover:
    """Rotates through RPC endpoints on failure."""

    def __init__(self, urls: list):
        self._urls = urls if urls else ["https://mainnet.base.org"]
        self._index = 0

    @property
    def current_url(self) -> str:
        return self._urls[self._index % len(self._urls)]

    def mark_failed(self) -> None:
        self._index = (self._index + 1) % len(self._urls)

    def mark_success(self) -> None:
        pass  # Stay on current


def estimate_gas_price(w3, multiplier: float = 1.2, max_gwei: int = MAX_GAS_GWEI) -> int:
    """Get dynamic gas price from network, with fallback and cap.

    Returns gas price in wei.
    """
    try:
        network_gas = w3.eth.gas_price
        adjusted = int(network_gas * multiplier)
        cap = max_gwei * 1_000_000_000  # gwei to wei
        return min(adjusted, cap)
    except Exception:
        return DEFAULT_GAS_GWEI * 1_000_000_000
```

Then modify the transfer method's gas price line (around line 260). Replace:
```python
"gasPrice": self.w3.to_wei("5", "gwei"),
```
With:
```python
"gasPrice": estimate_gas_price(self.w3),
```

- [ ] **Step 4: Add RPC failover config to NodeConfig**

In `prsm/node/config.py`, after the existing WASM fields, add:

```python
    # On-chain resilience (Ring 6)
    base_rpc_urls: List[str] = field(default_factory=lambda: [
        "https://mainnet.base.org",
    ])
    gas_price_multiplier: float = 1.2
    max_gas_gwei: int = 50
```

- [ ] **Step 5: Implement settler signature verification**

In `prsm/node/settler_registry.py`, add a module-level function:

```python
def verify_settler_signature(public_key_b64: str, message: bytes, signature_b64: str) -> bool:
    """Verify a settler's Ed25519 signature."""
    if not signature_b64:
        return False
    try:
        from prsm.node.identity import verify_signature
        return verify_signature(public_key_b64, message, signature_b64)
    except Exception:
        return False
```

Then in the `sign_batch` method (around line 430), replace the simulated verification with real verification. Find the comment "In production: verify ECDSA signature" and the line "For now, accept any non-empty signature". Replace that block with:

```python
            # Verify signature using settler's public key
            settler = self._settlers.get(settler_id)
            if settler and hasattr(settler, 'public_key_b64') and settler.public_key_b64:
                expected_msg = f"PRSM:{batch.batch_hash}:{settler_id}".encode()
                if not verify_settler_signature(settler.public_key_b64, expected_msg, signature):
                    raise ValueError(f"Invalid signature from settler {settler_id}")
```

NOTE: The existing settlers may not have `public_key_b64` set. If not present, accept the signature (backward compatibility). Only verify when the key is available.

- [ ] **Step 6: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_ring6_hardening.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add prsm/economy/ftns_onchain.py prsm/node/config.py prsm/node/settler_registry.py tests/unit/test_ring6_hardening.py
git commit -m "feat(ring6): dynamic gas pricing, RPC failover, settler signature verification"
```

---

### Task 2: CLI Commands for Rings 1-5

**Files:**
- Modify: `prsm/cli.py`
- Test: `tests/unit/test_ring6_hardening.py` (append)

- [ ] **Step 1: Append CLI tests**

Append to `tests/unit/test_ring6_hardening.py`:

```python
from click.testing import CliRunner


class TestCLICommands:
    def test_node_benchmark_command_exists(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["node", "benchmark", "--help"])
        assert result.exit_code == 0
        assert "hardware" in result.output.lower() or "benchmark" in result.output.lower()

    def test_ftns_yield_estimate_command_exists(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["ftns", "yield-estimate", "--help"])
        assert result.exit_code == 0
        assert "yield" in result.output.lower() or "estimate" in result.output.lower()

    def test_agent_forge_command_exists(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["agent", "forge", "--help"])
        assert result.exit_code == 0

    def test_compute_quote_command_exists(self):
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["compute", "quote", "--help"])
        assert result.exit_code == 0
        assert "cost" in result.output.lower() or "quote" in result.output.lower()
```

- [ ] **Step 2: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_ring6_hardening.py::TestCLICommands -v`
Expected: FAIL (commands don't exist yet)

- [ ] **Step 3: Add CLI commands to cli.py**

Read `prsm/cli.py` and find the `node` command group. Add a `benchmark` subcommand:

```python
@node.command()
def benchmark():
    """Run hardware profiler and display compute tier classification."""
    from prsm.compute.wasm.profiler import HardwareProfiler
    profiler = HardwareProfiler()
    profile = profiler.detect()
    click.echo(f"Hardware Profile:")
    click.echo(f"  CPU: {profile.cpu_cores} cores @ {profile.cpu_freq_mhz:.0f} MHz")
    click.echo(f"  GPU: {profile.gpu_name or 'None detected'}")
    if profile.gpu_vram_gb > 0:
        click.echo(f"  VRAM: {profile.gpu_vram_gb:.1f} GB")
    click.echo(f"  RAM: {profile.ram_total_gb:.1f} GB total, {profile.ram_available_gb:.1f} GB available")
    click.echo(f"  TFLOPS: {profile.tflops_fp32:.2f} FP32")
    click.echo(f"  Compute Tier: {profile.compute_tier.value.upper()}")
    click.echo(f"  Thermal: {profile.thermal_class.value}")
```

Find the `ftns` command group. Add a `yield-estimate` subcommand:

```python
@ftns.command("yield-estimate")
@click.option("--hours", default=8, help="Hours per day available for compute")
@click.option("--stake", default=0, type=float, help="FTNS staked")
def ftns_yield_estimate(hours, stake):
    """Estimate daily/monthly FTNS earnings based on your hardware."""
    from prsm.compute.wasm.profiler import HardwareProfiler
    from prsm.economy.pricing.engine import PricingEngine
    from prsm.economy.pricing.models import ProsumerTier

    profiler = HardwareProfiler()
    profile = profiler.detect()
    tier = ProsumerTier.from_stake(stake)
    engine = PricingEngine()

    estimate = engine.yield_estimate(
        hardware_tier=profile.compute_tier.value,
        tflops=profile.tflops_fp32,
        hours_per_day=hours,
        prosumer_tier=tier,
    )

    click.echo(f"Yield Estimate:")
    click.echo(f"  Hardware: {profile.compute_tier.value.upper()} ({profile.tflops_fp32:.1f} TFLOPS)")
    click.echo(f"  Stake: {stake:.0f} FTNS ({tier.label})")
    click.echo(f"  Yield Boost: {estimate['yield_boost']}x")
    click.echo(f"  Daily: {estimate['daily_ftns']:.2f} FTNS")
    click.echo(f"  Monthly: {estimate['monthly_ftns']:.2f} FTNS")
```

Find or create an `agent` command group. Add a `forge` subcommand:

```python
@agent.command("forge")
@click.argument("query")
@click.option("--budget", default=10.0, help="FTNS budget")
def agent_forge_cmd(query, budget):
    """Decompose a query and show the execution plan."""
    import asyncio
    from prsm.compute.nwtn.agent_forge.forge import AgentForge

    async def _run():
        forge = AgentForge()
        decomp = await forge.decompose(query)
        click.echo(f"Query: {query}")
        click.echo(f"Route: {decomp.recommended_route.value}")
        click.echo(f"Datasets: {decomp.required_datasets or 'None (direct LLM)'}")
        click.echo(f"Operations: {decomp.operations or 'None'}")
        click.echo(f"Hardware: {decomp.min_hardware_tier}")
        click.echo(f"Complexity: {decomp.estimated_complexity:.1f}")

    asyncio.run(_run())
```

Find the `compute` command group. Add a `quote` subcommand:

```python
@compute.command("quote")
@click.argument("query")
@click.option("--shards", default=3, help="Estimated number of data shards")
@click.option("--tier", default="t2", help="Hardware tier (t1-t4)")
def compute_quote(query, shards, tier):
    """Get a cost estimate for a compute query."""
    from prsm.economy.pricing.engine import PricingEngine

    engine = PricingEngine()
    quote = engine.quote_swarm_job(
        shard_count=shards,
        hardware_tier=tier,
        estimated_pcu_per_shard=50.0,
    )

    click.echo(f"Cost Quote for: {query}")
    click.echo(f"  Compute: {quote.compute_cost} FTNS")
    click.echo(f"  Data: {quote.data_cost} FTNS")
    click.echo(f"  Network Fee: {quote.network_fee} FTNS")
    click.echo(f"  Total: {quote.total} FTNS")
```

NOTE: When adding these commands, read cli.py first to find the exact locations of the command groups (node, ftns, compute, agent). The `agent` group may not exist yet — if not, create it:

```python
@main.group()
def agent():
    """Manage PRSM mobile agents."""
    pass
```

- [ ] **Step 4: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_ring6_hardening.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/cli.py tests/unit/test_ring6_hardening.py
git commit -m "feat(ring6): CLI commands — node benchmark, ftns yield-estimate, agent forge, compute quote"
```

---

### Task 3: Integration Smoke Test + Version Bump + Publish

**Files:**
- Create: `tests/integration/test_ring6_polish.py`
- Modify: `prsm/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create integration test**

Create `tests/integration/test_ring6_polish.py`:

```python
"""Ring 6 Smoke Test — production hardening verification."""

import pytest
from click.testing import CliRunner


class TestRing6Smoke:
    def test_hardware_benchmark_runs(self):
        """node benchmark should detect hardware without errors."""
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["node", "benchmark"])
        assert result.exit_code == 0
        assert "compute tier" in result.output.lower() or "tflops" in result.output.lower()

    def test_yield_estimate_runs(self):
        """ftns yield-estimate should produce numbers."""
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["ftns", "yield-estimate", "--hours", "8"])
        assert result.exit_code == 0
        assert "daily" in result.output.lower() or "monthly" in result.output.lower()

    def test_gas_estimator_importable(self):
        """Dynamic gas pricing function should be available."""
        from prsm.economy.ftns_onchain import estimate_gas_price, RPCFailover
        assert callable(estimate_gas_price)
        failover = RPCFailover(urls=["https://example.com"])
        assert failover.current_url == "https://example.com"

    def test_settler_signature_verification_importable(self):
        """Real signature verification should be available."""
        from prsm.node.settler_registry import verify_settler_signature
        assert callable(verify_settler_signature)
        # Empty signature should be rejected
        assert not verify_settler_signature("key", b"msg", "")

    def test_all_ring_imports_work(self):
        """Verify all Rings 1-5 modules import correctly."""
        from prsm.compute.wasm import WASMRuntime, WasmtimeRuntime, HardwareProfiler
        from prsm.compute.agents import AgentDispatcher, AgentExecutor, MobileAgent
        from prsm.compute.swarm import SwarmCoordinator, SwarmJob, SwarmResult
        from prsm.economy.pricing import PricingEngine, CostQuote, ProsumerTier
        from prsm.economy.prosumer import ProsumerManager
        from prsm.compute.nwtn.agent_forge import AgentForge, TaskDecomposition

        assert WASMRuntime is not None
        assert AgentDispatcher is not None
        assert SwarmCoordinator is not None
        assert PricingEngine is not None
        assert AgentForge is not None
```

- [ ] **Step 2: Run all Ring 6 tests**

Run: `python -m pytest tests/unit/test_ring6_hardening.py tests/integration/test_ring6_polish.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Run FULL regression across all Rings**

Run: `python -m pytest tests/unit/test_wasm_runtime.py tests/unit/test_hardware_profiler.py tests/unit/test_wasm_compute_provider.py tests/unit/test_mobile_agent_models.py tests/unit/test_agent_executor.py tests/unit/test_agent_dispatcher.py tests/unit/test_semantic_shard.py tests/unit/test_swarm_models.py tests/unit/test_swarm_coordinator.py tests/unit/test_pricing_engine.py tests/unit/test_prosumer.py tests/unit/test_agent_forge_models.py tests/unit/test_agent_forge.py tests/unit/test_ring6_hardening.py tests/integration/test_ring1_smoke.py tests/integration/test_ring2_dispatch.py tests/integration/test_ring3_swarm.py tests/integration/test_ring4_economy.py tests/integration/test_ring5_forge.py tests/integration/test_ring6_polish.py -v --timeout=60`
Expected: ALL tests PASS

- [ ] **Step 4: Version bump**

Change `__version__` in `prsm/__init__.py` to `"0.31.0"`.
Change `version` in `pyproject.toml` to `"0.31.0"`.

- [ ] **Step 5: Commit, push, publish**

```bash
git add tests/integration/test_ring6_polish.py prsm/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.31.0 for Ring 6 — The Polish (production hardening)"
git push origin main
rm -rf build/ dist/ prsm_network.egg-info/
python3 -m build
python3 -m twine upload dist/prsm_network-0.31.0*
```
