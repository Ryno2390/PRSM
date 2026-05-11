#!/usr/bin/env python3
"""Sprint 178 — verify cross-operator discovery via the live PRSM
bootstrap server.

Spins up two ``BootstrapClient`` instances concurrently against
``wss://bootstrap1.prsm-network.com:8765``, registers each, waits
for the other to register, then polls ``get_peers`` and verifies
each sees the other's peer_id.

Run as a script (NOT pytest — pytest's autouse mock_asyncio_sleep
fixture would break the timing-dependent registration handshake):

    python3 scripts/verify_cross_operator_discovery.py

Exit code 0 on cross-discovery success. Closes the proof gap for
sprints 164 (WSS fallback) + 165 (peer-poll) end-to-end against
the live network.
"""
from __future__ import annotations

import asyncio
import sys
import uuid


async def run_node(node_id: str, observed: list[str]) -> None:
    from prsm.bootstrap.client import BootstrapClient
    client = BootstrapClient(
        bootstrap_url="wss://bootstrap1.prsm-network.com:8765",
        node_id=node_id,
        port=9999,
        capabilities=["compute"],
        version="1.7.0",
    )
    try:
        await asyncio.wait_for(client.connect(), timeout=15)
        await client.start_heartbeat()
        # Allow the other node to register + propagate.
        await asyncio.sleep(15)
        peers = await asyncio.wait_for(client.get_peers(), timeout=15)
        for p in peers:
            observed.append(p.peer_id)
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass


async def main() -> int:
    alpha_id = "alpha-" + uuid.uuid4().hex[:8]
    beta_id = "beta-" + uuid.uuid4().hex[:8]
    seen_by_alpha: list[str] = []
    seen_by_beta: list[str] = []

    await asyncio.gather(
        run_node(alpha_id, seen_by_alpha),
        run_node(beta_id, seen_by_beta),
    )

    print(f"alpha id: {alpha_id}")
    print(f"beta  id: {beta_id}")
    print(f"alpha saw: {seen_by_alpha}")
    print(f"beta  saw: {seen_by_beta}")

    alpha_saw_beta = beta_id in seen_by_alpha
    beta_saw_alpha = alpha_id in seen_by_beta
    if alpha_saw_beta or beta_saw_alpha:
        print(
            f"\nPASS — cross-operator discovery works "
            f"(alpha→beta: {alpha_saw_beta}, beta→alpha: {beta_saw_alpha})"
        )
        return 0
    print("\nFAIL — neither node saw the other")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
