#!/usr/bin/env python3
"""Two-daemon end-to-end test for P2P federation.

Starts a second PRSM node in a temp data dir, connects to the running
primary daemon, and verifies multi-node P2P communication.

Prerequisites:
  - Primary node running: `prsm daemon start` (API on 8000, P2P on 9001)

Usage:
  python tests/node/test_two_daemon_e2e.py
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e-test")

# Suppress noisy loggers
for name in [
    "prsm.core", "prsm.compute", "prsm.data", "prsm.node.transport",
    "prsm.node.discovery", "prsm.node.gossip", "prsm.node.node",
    "websockets",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


async def check_primary_node():
    """Verify the primary daemon is running."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://127.0.0.1:8000/status",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                return data
    except Exception as e:
        return None


async def start_node_b(tmpdir: str):
    """Start Node B as a subprocess with its own data dir."""
    # Create a minimal startup script for Node B
    script = f'''
import asyncio, sys, os, logging
sys.path.insert(0, "{Path(__file__).resolve().parents[2]}")
os.environ["HOME"] = "{tmpdir}"  # Force ~/.prsm to be inside tmpdir

# Suppress noise
for n in ["prsm.core", "prsm.compute", "prsm.data", "websockets"]:
    logging.getLogger(n).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

from prsm.node.config import NodeConfig
from prsm.node.node import PRSMNode

async def main():
    # Create .prsm dir in tmpdir
    prsm_dir = os.path.join("{tmpdir}", ".prsm")
    os.makedirs(prsm_dir, exist_ok=True)

    config = NodeConfig(
        display_name="NodeB-E2E",
        data_dir=prsm_dir,
        p2p_port=9002,
        api_port=8001,
        bootstrap_nodes=["ws://127.0.0.1:9001"],
        bootstrap_connect_timeout=10.0,
        bootstrap_retry_attempts=2,
        max_concurrent_jobs=3,
    )
    config.ensure_dirs()

    node = PRSMNode(config)
    await node.initialize()
    print("NODE_B_ID=" + node.identity.node_id, flush=True)
    print("NODE_B_READY", flush=True)

    await node.start()
    print("NODE_B_STARTED", flush=True)

    # Wait for peer connection
    for i in range(30):
        await asyncio.sleep(1)
        pc = node.transport.peer_count if node.transport else 0
        if pc > 0:
            print(f"NODE_B_PEERS={{pc}}", flush=True)
            break
    else:
        print("NODE_B_PEERS=0", flush=True)

    # Report Node B API
    print(f"NODE_B_API=http://127.0.0.1:8001", flush=True)

    # Keep running until killed
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await node.stop()

asyncio.run(main())
'''
    script_path = os.path.join(tmpdir, "run_node_b.py")
    with open(script_path, "w") as f:
        f.write(script)

    # Load .env for API keys
    env = os.environ.copy()
    env_file = Path.home() / ".prsm" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")

    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    return proc


async def run_test():
    """Run the full E2E test."""
    import aiohttp

    # ── Step 1: Check primary node ──
    logger.info("Step 1: Checking primary node ...")
    status = await check_primary_node()
    if not status:
        logger.error("  Primary node not running. Start with: prsm daemon start")
        return False
    primary_id = status.get("node_id", "?")
    logger.info(f"  Primary node: {primary_id[:12]}...")

    # ── Step 2: Start Node B ──
    logger.info("Step 2: Starting Node B subprocess ...")
    tmpdir = tempfile.mkdtemp(prefix="prsm-node-b-")
    proc = await start_node_b(tmpdir)

    node_b_id = None
    node_b_peers = 0
    node_b_ready = False

    # Wait for Node B to initialize
    logger.info("  Waiting for Node B to start (up to 60s) ...")
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                stderr = proc.stderr.read()
                logger.error(f"  Node B exited with code {proc.returncode}")
                if stderr:
                    # Show last few lines of stderr
                    for l in stderr.strip().split("\n")[-5:]:
                        logger.error(f"    {l}")
                return False
            await asyncio.sleep(0.1)
            continue

        line = line.strip()
        if line.startswith("NODE_B_ID="):
            node_b_id = line.split("=", 1)[1]
            logger.info(f"  Node B identity: {node_b_id[:12]}...")
        elif line == "NODE_B_READY":
            logger.info("  Node B initialized")
        elif line == "NODE_B_STARTED":
            logger.info("  Node B started")
            node_b_ready = True
        elif line.startswith("NODE_B_PEERS="):
            node_b_peers = int(line.split("=")[1])
            logger.info(f"  Node B peers: {node_b_peers}")
            break

    if not node_b_ready:
        logger.error("  Node B did not start in time")
        proc.kill()
        return False

    # ── Step 3: Check peer visibility ──
    logger.info("Step 3: Checking peer connectivity ...")

    # Check from Node A's perspective
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:8000/peers",
                                   timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    peers_data = await resp.json()
                    if isinstance(peers_data, list):
                        logger.info(f"  Node A sees {len(peers_data)} peer(s)")
                    elif isinstance(peers_data, dict):
                        logger.info(f"  Node A peers: {json.dumps(peers_data)[:200]}")
                else:
                    logger.info(f"  /peers returned {resp.status}")
    except Exception as e:
        logger.info(f"  Could not check /peers: {e}")

    # ── Step 4: Submit compute job ──
    logger.info("Step 4: Submitting compute job from Node A ...")
    result = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:8000/compute/query",
                json={
                    "prompt": "What is 2+2? Answer briefly.",
                    "model": "nwtn",
                    "timeout": 60,
                    "budget": 0.01,
                },
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                result = await resp.json()
                logger.info(f"  Job ID: {result.get('job_id', '?')}")
                response_text = result.get("response", str(result))
                logger.info(f"  Response: {response_text[:120]}")
    except asyncio.TimeoutError:
        logger.error("  Compute query timed out")
    except Exception as e:
        logger.error(f"  Compute query failed: {e}")

    # ── Step 5: Check settlement ──
    logger.info("Step 5: Checking settlement queue ...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:8000/settlement/stats",
                                   timeout=aiohttp.ClientTimeout(total=5)) as resp:
                settle = await resp.json()
                logger.info(f"  Queue: {settle.get('queue_size', 0)} transfers, "
                           f"{settle.get('pending_amount', 0):.4f} FTNS pending")
    except Exception:
        pass

    # ── Results ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    tests = [
        ("Primary node running", status is not None),
        ("Node B started", node_b_ready),
        ("Node B identity generated", node_b_id is not None),
        ("Peer connection established", node_b_peers > 0),
        ("Compute job returned result", result is not None and "response" in result),
    ]

    all_pass = True
    for name, passed in tests:
        icon = "PASS" if passed else "FAIL"
        logger.info(f"  [{icon}] {name}")
        if not passed:
            all_pass = False

    logger.info("")
    if all_pass:
        logger.info("ALL TESTS PASSED — Multi-node P2P federation verified!")
    else:
        logger.info("Some tests failed — see details above")

    # Cleanup
    logger.info("Stopping Node B ...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    logger.info("Done.")

    return all_pass


def main():
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
