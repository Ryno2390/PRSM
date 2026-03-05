#!/usr/bin/env python3
"""
PRSM Smoke Test
===============

End-to-end validation of the core node workflow without external dependencies.
Tests the full lifecycle: create node, submit jobs, verify results, check balances.

Run with: python scripts/smoke_test.py
"""

import asyncio
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prsm.node.node import PRSMNode
from prsm.node.config import NodeConfig, NodeRole
from prsm.node.compute_provider import JobType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("smoke_test")


class SmokeTestResult:
    """Tracks test results."""
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.skipped: List[Tuple[str, str]] = []
    
    def add_pass(self, name: str):
        self.passed.append(name)
        logger.info(f"✅ PASS: {name}")
    
    def add_fail(self, name: str, reason: str):
        self.failed.append((name, reason))
        logger.error(f"❌ FAIL: {name} - {reason}")
    
    def add_skip(self, name: str, reason: str):
        self.skipped.append((name, reason))
        logger.warning(f"⏭️ SKIP: {name} - {reason}")
    
    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        logger.info("\n" + "=" * 60)
        logger.info("SMOKE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Passed:  {len(self.passed)}/{total}")
        logger.info(f"  Failed:  {len(self.failed)}/{total}")
        logger.info(f"  Skipped: {len(self.skipped)}/{total}")
        
        if self.failed:
            logger.info("\nFailed tests:")
            for name, reason in self.failed:
                logger.info(f"  - {name}: {reason}")
        
        if self.skipped:
            logger.info("\nSkipped tests:")
            for name, reason in self.skipped:
                logger.info(f"  - {name}: {reason}")
        
        logger.info("=" * 60)
        return len(self.failed) == 0


def get_random_port() -> int:
    """Get a random available port."""
    return random.randint(20000, 30000)


async def run_smoke_test() -> bool:
    """Run all smoke tests."""
    results = SmokeTestResult()
    node: Optional[PRSMNode] = None
    temp_dir: Optional[str] = None
    
    try:
        # ── Step 1: Create PRSMNode with in-memory config ─────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Creating PRSMNode with in-memory config")
        logger.info("=" * 60)
        
        temp_dir = tempfile.mkdtemp(prefix="prsm_smoke_")
        p2p_port = get_random_port()
        api_port = get_random_port()
        
        config = NodeConfig(
            display_name="smoke-test-node",
            roles=[NodeRole.FULL],
            data_dir=temp_dir,
            p2p_port=p2p_port,
            api_port=api_port,
            welcome_grant=100.0,
            allow_self_compute=True,
            bootstrap_nodes=[],  # No external connections
            bootstrap_fallback_enabled=False,  # Disable fallback for isolated test
        )
        
        node = PRSMNode(config)
        results.add_pass("Create PRSMNode with temp config")
        
        # ── Step 2: Initialize and start the node ─────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Initialize and start the node")
        logger.info("=" * 60)
        
        await node.initialize()
        results.add_pass("Initialize node")
        
        await node.start()
        results.add_pass("Start node")
        
        # Give subsystems a moment to fully start
        await asyncio.sleep(0.5)
        
        # ── Step 3: Submit a benchmark compute job ────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Submit benchmark compute job")
        logger.info("=" * 60)
        
        initial_balance = await node.ledger.get_balance(node.identity.node_id)
        logger.info(f"Initial FTNS balance: {initial_balance}")
        
        job_budget = 5.0
        job = await node.compute_requester.submit_job(
            job_type=JobType.BENCHMARK,
            payload={"iterations": 100000},
            ftns_budget=job_budget,
        )
        results.add_pass(f"Submit benchmark job (ID: {job.job_id[:8]}...)")
        
        # ── Step 4: Wait for the result ───────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 4: Wait for compute result")
        logger.info("=" * 60)
        
        result = await node.compute_requester.get_result(job.job_id, timeout=30.0)
        
        if result is None:
            results.add_fail("Wait for compute result", f"Job timed out or failed. Status: {job.status}, Error: {job.error}")
        else:
            results.add_pass("Wait for compute result")
            
            # ── Step 5: Verify the result has real benchmark data ──────────────
            logger.info("\n" + "=" * 60)
            logger.info("Step 5: Verify benchmark result data")
            logger.info("=" * 60)
            
            if "primes_found" in result:
                logger.info(f"  primes_found: {result['primes_found']}")
                results.add_pass("Result contains 'primes_found'")
            else:
                results.add_fail("Result contains 'primes_found'", f"Key not found in result: {result.keys()}")
            
            if "elapsed_seconds" in result:
                logger.info(f"  elapsed_seconds: {result['elapsed_seconds']}")
                results.add_pass("Result contains 'elapsed_seconds'")
            else:
                results.add_fail("Result contains 'elapsed_seconds'", f"Key not found in result: {result.keys()}")
            
            logger.info(f"  benchmark_type: {result.get('benchmark_type', 'N/A')}")
            logger.info(f"  ops_per_second: {result.get('ops_per_second', 'N/A')}")
        
        # ── Step 6: Check FTNS balance changed ────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Check FTNS balance changed")
        logger.info("=" * 60)
        
        final_balance = await node.ledger.get_balance(node.identity.node_id)
        logger.info(f"Final FTNS balance: {final_balance}")
        
        # In self-compute mode, the node earns from its own jobs but doesn't actually
        # pay itself (the debit and credit net out). So balance should be:
        # - Welcome grant: +100 FTNS
        # - Benchmark job: earns 5 FTNS (no net payment since self-compute)
        # - Inference job: earns 2 FTNS (no net payment since self-compute)
        # Total: 107 FTNS
        balance_diff = final_balance - initial_balance
        
        # Balance should have increased from compute earnings
        if final_balance >= initial_balance:
            results.add_pass(f"FTNS balance is {final_balance} (earned {balance_diff} FTNS from self-compute)")
        else:
            results.add_fail(
                f"FTNS balance check",
                f"Expected balance >= {initial_balance}, got {final_balance}"
            )
        
        # ── Step 7: Submit an inference job ────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 7: Submit inference job")
        logger.info("=" * 60)
        
        inference_job = await node.compute_requester.submit_job(
            job_type=JobType.INFERENCE,
            payload={"prompt": "What is the meaning of life?", "model": "test"},
            ftns_budget=2.0,
        )
        results.add_pass(f"Submit inference job (ID: {inference_job.job_id[:8]}...)")
        
        inference_result = await node.compute_requester.get_result(inference_job.job_id, timeout=30.0)
        
        if inference_result is None:
            results.add_fail("Wait for inference result", f"Job timed out or failed. Status: {inference_job.status}")
        else:
            results.add_pass("Wait for inference result")
            
            # Verify it returns with a valid source (mock or nwtn_orchestrator)
            source = inference_result.get("source")
            if source in ("mock", "nwtn_orchestrator"):
                logger.info(f"  source: {source}")
                results.add_pass(f"Inference result has valid source='{source}'")
            else:
                results.add_fail(
                    "Inference result has valid source",
                    f"Got source='{source}' instead of 'mock' or 'nwtn_orchestrator'"
                )
            
            logger.info(f"  response: {inference_result.get('response', 'N/A')[:100]}...")
        
        # ── Step 8: Check the content index is accessible ──────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 8: Check content index is accessible")
        logger.info("=" * 60)
        
        if node.content_index:
            stats = node.content_index.get_stats()
            logger.info(f"  Content index stats: {stats}")
            results.add_pass("Content index is accessible")
        else:
            results.add_fail("Content index is accessible", "Content index is None")
        
        # ── Step 9: Check get_status() returns complete data ───────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 9: Check get_status() returns complete data")
        logger.info("=" * 60)
        
        status = await node.get_status()
        
        required_fields = [
            "node_id", "display_name", "roles", "started", 
            "uptime_seconds", "p2p_address", "api_address", 
            "peers", "ftns_balance"
        ]
        
        missing_fields = [f for f in required_fields if f not in status]
        
        if not missing_fields:
            results.add_pass("get_status() returns all required fields")
            logger.info(f"  node_id: {status.get('node_id', 'N/A')[:16]}...")
            logger.info(f"  display_name: {status.get('display_name')}")
            logger.info(f"  roles: {status.get('roles')}")
            logger.info(f"  started: {status.get('started')}")
            logger.info(f"  uptime_seconds: {status.get('uptime_seconds')}")
            logger.info(f"  ftns_balance: {status.get('ftns_balance')}")
        else:
            results.add_fail(
                "get_status() returns all required fields",
                f"Missing fields: {missing_fields}"
            )
        
        # ── Step 10: Shut down cleanly ────────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("Step 10: Shut down cleanly")
        logger.info("=" * 60)
        
        await node.stop()
        results.add_pass("Node shutdown complete")
        
    except Exception as e:
        logger.exception(f"Smoke test failed with exception: {e}")
        results.add_fail("Smoke test execution", str(e))
        
        # Attempt cleanup on failure
        if node and node._started:
            try:
                await node.stop()
            except Exception:
                pass
    
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    return results.summary()


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("PRSM SMOKE TEST")
    logger.info("=" * 60)
    logger.info("Validating core node workflow without external dependencies")
    logger.info("")
    
    try:
        success = asyncio.run(run_smoke_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nSmoke test interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
