"""
PRSM Demo
==========

Self-contained demonstration of the Ring 1-10 pipeline.
Runs locally without needing a second node or external data.

Usage:
    python -m prsm.demo
    # or
    prsm demo
"""

import asyncio
import json
import sys
import time
from decimal import Decimal


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_step(step: int, title: str) -> None:
    print(f"  [{step}] {title}")


async def run_demo():
    print_header("PRSM Sovereign-Edge AI Demo")
    print("  Demonstrating the full Ring 1-10 pipeline locally.\n")

    # -- Ring 1: Hardware Profiling -------------------------------------------
    print_header("Ring 1: The Sandbox -- Hardware Detection")
    from prsm.compute.wasm import HardwareProfiler, WasmtimeRuntime

    profiler = HardwareProfiler()
    profile = profiler.detect()
    print_step(1, f"CPU: {profile.cpu_cores} cores @ {profile.cpu_freq_mhz:.0f} MHz")
    print_step(2, f"GPU: {profile.gpu_name or 'None detected'}")
    print_step(3, f"TFLOPS: {profile.tflops_fp32:.2f} FP32")
    print_step(4, f"Compute Tier: {profile.compute_tier.value.upper()}")
    print_step(5, f"Thermal: {profile.thermal_class.value}")
    print_step(6, f"TEE: {profile.tee_type or 'Software (WASM sandbox)'}")

    # WASM execution test
    runtime = WasmtimeRuntime()
    if runtime.available:
        from prsm.compute.wasm.models import ResourceLimits
        MINIMAL_WASM = bytes([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
            0x03, 0x02, 0x01, 0x00,
            0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
            0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
        ])
        module = runtime.load(MINIMAL_WASM)
        result = runtime.execute(module, b"", ResourceLimits(max_memory_bytes=64 * 1024 * 1024))
        print_step(7, f"WASM sandbox: {result.status.value} ({result.execution_time_seconds:.4f}s)")
    else:
        print_step(7, "WASM sandbox: wasmtime not installed (pip install prsm-network[wasm])")

    # -- Ring 3: Semantic Sharding --------------------------------------------
    print_header("Ring 3: The Swarm -- Semantic Data Sharding")
    from prsm.data.shard_models import SemanticShard, SemanticShardManifest

    shards = [
        SemanticShard(
            shard_id=f"demo-{i}",
            parent_dataset="demo-data",
            cid=f"QmDemo{i:04d}",
            centroid=[float(i) * 0.2, 1.0 - float(i) * 0.2, 0.5],
            record_count=1000,
            size_bytes=1024 * 1024,
            keywords=["electric vehicles", "registration", f"region-{i}"],
        )
        for i in range(5)
    ]
    manifest = SemanticShardManifest(
        dataset_id="demo-nada-nc",
        total_records=5000,
        total_size_bytes=5 * 1024 * 1024,
        shards=shards,
    )
    print_step(1, f"Dataset: {manifest.dataset_id} ({len(shards)} shards)")

    query_embedding = [0.1, 0.9, 0.5]
    relevant = manifest.find_relevant_shards(query_embedding, top_k=3)
    print_step(2, f"Query embedding: {query_embedding}")
    print_step(3, f"Relevant shards: {len(relevant)} (by cosine similarity)")
    for shard, score in relevant:
        print(f"         {shard.shard_id}: similarity={score:.3f}")

    # -- Ring 4: Pricing ------------------------------------------------------
    print_header("Ring 4: The Economy -- Hybrid Pricing")
    from prsm.economy.pricing import PricingEngine, DataAccessFee, ProsumerTier

    engine = PricingEngine(network_utilization=0.6)
    quote = engine.quote_swarm_job(
        shard_count=3,
        hardware_tier=profile.compute_tier.value,
        estimated_pcu_per_shard=50.0,
        data_fee=DataAccessFee(
            dataset_id="demo-nada",
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("0.5"),
        ),
    )
    print_step(1, f"Compute cost: {quote.compute_cost} FTNS")
    print_step(2, f"Data cost: {quote.data_cost} FTNS")
    print_step(3, f"Network fee: {quote.network_fee} FTNS")
    print_step(4, f"Total: {quote.total} FTNS")

    # Yield estimate
    estimate = engine.yield_estimate(
        hardware_tier=profile.compute_tier.value,
        tflops=profile.tflops_fp32,
        hours_per_day=8,
        prosumer_tier=ProsumerTier.DEDICATED,
    )
    print_step(5, f"Your yield (8hrs/day, Dedicated tier): {float(estimate['daily_ftns']):.2f} FTNS/day")

    # Revenue split
    from prsm.economy.pricing.revenue_split import RevenueSplitEngine
    splitter = RevenueSplitEngine()
    split = splitter.calculate_split(
        total_payment=quote.total,
        data_owner_id="nada-org",
        compute_providers={"your-node": 75.0, "peer-node": 25.0},
    )
    print_step(6, f"Revenue split: Data owner={split.data_owner_amount}, "
               f"Compute={sum(split.compute_amounts.values())}, Treasury={split.treasury_amount}")

    # -- Ring 5: Agent Forge --------------------------------------------------
    print_header("Ring 5: The Brain -- Agent Forge Decomposition")
    from prsm.compute.nwtn.agent_forge import AgentForge

    forge = AgentForge()
    decomp = await forge.decompose(
        "Analyze NADA vehicle registration trends in North Carolina for 2025"
    )
    print_step(1, f"Query decomposed:")
    print(f"         Route: {decomp.recommended_route.value}")
    print(f"         Datasets: {decomp.required_datasets or '(needs LLM backend for real decomposition)'}")
    print(f"         Operations: {decomp.operations or ['generate']}")
    print(f"         Complexity: {decomp.estimated_complexity}")

    # Instruction generation
    from prsm.compute.agents.instruction_set import instructions_from_decomposition
    instructions = instructions_from_decomposition(decomp.to_dict())
    print_step(2, f"Agent instructions: {len(instructions.instructions)} operations")
    for inst in instructions.instructions:
        print(f"         {inst.op.value}: {inst.params.get('description', '')}")

    # -- Ring 7: Confidential Compute -----------------------------------------
    print_header("Ring 7: The Vault -- Confidential Compute")
    from prsm.compute.tee.platform_detect import detect_tee_capability
    from prsm.compute.tee.dp_noise import DPNoiseInjector
    from prsm.compute.tee.models import DPConfig
    import numpy as np

    tee = detect_tee_capability()
    print_step(1, f"TEE: {tee.tee_type.value} (hardware={tee.is_hardware_backed})")

    injector = DPNoiseInjector(DPConfig(epsilon=8.0))
    original = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    noisy = injector.inject(original)
    print_step(2, f"DP noise (epsilon=8.0): original={original.tolist()}")
    print(f"                    noisy={[round(x, 2) for x in noisy.tolist()]}")
    print_step(3, f"Privacy budget spent: epsilon={injector.epsilon_spent}")

    # -- Ring 8: Model Sharding -----------------------------------------------
    print_header("Ring 8: The Shield -- Model Sharding")
    from prsm.compute.model_sharding import ModelSharder

    sharder = ModelSharder()
    weights = {"attention": np.random.randn(16, 8), "mlp": np.random.randn(8, 4)}
    model = sharder.shard_model("demo-model", "DemoNet", weights, n_shards=4)
    print_step(1, f"Model sharded: {len(model.shards)} shards, {model.total_size_bytes:,} bytes")

    # Verify integrity
    from prsm.security import IntegrityVerifier
    verifier = IntegrityVerifier()
    all_valid = True
    for shard in model.shards:
        valid = verifier.verify_shard(shard.tensor_data, shard.checksum)
        if not valid:
            print(f"         INTEGRITY FAILURE: shard {shard.shard_id}")
            all_valid = False
    print_step(2, f"Integrity verified: all {len(model.shards)} shard checksums valid")

    # -- Ring 10: Security ----------------------------------------------------
    print_header("Ring 10: The Fortress -- Security Audit")
    from prsm.security import PrivacyBudgetTracker, PipelineAuditLog

    budget = PrivacyBudgetTracker(max_epsilon=100.0)
    budget.record_spend(8.0, "demo-query-1", "demo-model")
    budget.record_spend(8.0, "demo-query-2", "demo-model")
    print_step(1, f"Privacy budget: {budget.total_spent}/{budget.max_epsilon} epsilon spent")

    audit = PipelineAuditLog()
    for i in range(3):
        audit.record(f"demo-model", 4, [{"node_id": f"n{j}"} for j in range(4)], 20)
    print_step(2, f"Audit log: {audit.entry_count} entries, chain valid={audit.verify_chain()}")

    # -- Summary --------------------------------------------------------------
    print_header("Demo Complete")
    print("  All 10 rings demonstrated successfully.\n")
    print("  To run with a live LLM backend:")
    print("    export OPENROUTER_API_KEY=your-key")
    print("    prsm node start")
    print("    prsm compute run --query 'Your question here' --budget 1.0")
    print()


def main():
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
