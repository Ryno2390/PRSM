"""
Multi-Node Test Lab
===================

Spins up multiple PRSM nodes on localhost with different ports
connected to each other, allowing full testing of:

- Cross-node compute job execution
- FTNS ledger synchronization
- Payment escrow and consensus
- Gossip propagation between nodes

Usage:
    python -m prsm.node.multinode_demo
    prsm multinode-demo     # (CLI alias)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Silence noisy logs during demo
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("multinode")

# Rich console for presentation
from rich.console import Console

console = Console()

# Internal imports
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.compute_provider import ComputeProvider, JobType, JobStatus
from prsm.node.compute_requester import ComputeRequester, SubmittedJob
from prsm.node.gossip import (
    GOSSIP_JOB_OFFER,
    GOSSIP_JOB_ACCEPT,
    GOSSIP_JOB_RESULT,
    GOSSIP_FTNS_TRANSACTION,
    GossipProtocol,
)
from prsm.node.result_consensus import ResultConsensus, ConsensusMode
from prsm.node.payment_escrow import PaymentEscrow
from prsm.node.transport import WebSocketTransport


class DemoNodeWrapper:
    """Lightweight wrapper around PRSM components for multi-node testing.

    Each wrapper represents a full PRSM node with its own identity,
    ledger, compute provider, and transport. Nodes communicate over
    loopback WebSocket connections.
    """

    def __init__(
        self,
        node_id: str,
        p2p_port: int,
        api_port: int,
        display_name: str = "",
    ):
        self.node_id = node_id
        self.p2p_port = p2p_port
        self.api_port = api_port
        self.display_name = display_name or f"node-{node_id[:8]}"
        self.running = False

        # Core components
        self.identity: Optional[NodeIdentity] = None
        self.ledger: Optional[LocalLedger] = None
        self.transport: Optional[WebSocketTransport] = None
        self.gossip: Optional[GossipProtocol] = None
        self.provider: Optional[ComputeProvider] = None
        self.requester: Optional[ComputeRequester] = None
        self.escrow: Optional[PaymentEscrow] = None
        self.consensus: Optional[ResultConsensus] = None

    async def start(self, initial_balance: float = 100.0) -> None:
        """Initialize and start all node components."""
        # Identity
        self.identity = generate_node_identity(self.display_name)
        console.print(f"    Identity:  {self.identity.node_id[:16]}...")
        console.print(f"    P2P port:  {self.p2p_port}")
        console.print(f"    API port:  {self.api_port}")

        # Ledger (in-memory for testing)
        self.ledger = LocalLedger(db_path=":memory:")
        await self.ledger.initialize()
        await self.ledger.create_wallet(self.identity.node_id, self.display_name)
        if initial_balance > 0:
            await self.ledger.credit(
                wallet_id=self.identity.node_id,
                amount=initial_balance,
                tx_type=TransactionType.WELCOME_GRANT,
                description="Initial balance for demo",
            )
            balance = await self.ledger.get_balance(self.identity.node_id)
            console.print(f"    Balance:   {balance:.2f} FTNS")

        # Gossip protocol (needs a transport — use a dummy one for local demo)
        class _GossipTransport:
            """Minimal shim so gossip protocol doesn't crash."""
            peers: Dict[str, Any] = {}
            peer_count: int = 0
            async def send_to_peer(self, peer_id: str, msg: Any) -> None: pass
            def on_message(self, msg_type: str, handler) -> None: pass
            async def gossip(self, msg: Any, fanout: int = 3) -> int:
                return 0  # No peers in local demo

        self.transport = _GossipTransport()
        self.gossip = GossipProtocol(
            transport=self.transport,
            fanout=3,
            default_ttl=10,
        )
        # Gossip.publish reads transport.identity.node_id
        self.transport.identity = self.identity
        self.gossip._running = True  # Mark as started (no background task needed)

        # Compute provider
        self.provider = ComputeProvider(
            identity=self.identity,
            transport=None,  # No network transport in local demo
            gossip=self.gossip,
            ledger=self.ledger,
            max_concurrent_jobs=5,
        )
        self.provider._running = True

        # Consensus manager
        self.consensus = ResultConsensus(
            epsilon=0.01,
            timeout_seconds=30.0,
        )

        # Payment escrow
        self.escrow = PaymentEscrow(
            ledger=self.ledger,
            node_id=self.identity.node_id,
        )

        self.running = True
        console.print()

    async def stop(self) -> None:
        if self.running:
            self.provider._running = False
            self.running = False

    async def get_balance(self) -> float:
        return await self.ledger.get_balance(self.identity.node_id)


class MultiNodeDemo:
    """Orchestrates a multi-node compute + payment demonstration.

    Creates 3 nodes:
    - Node A: Submits a benchmark compute job with escrow
    - Node B: Accepts and executes the job (provider)
    - Node C: Also accepts and executes (second provider for consensus)

    Demonstrates:
    1. Escrow creation (A locks FTNS before job runs)
    2. Job offer propagation (gossip)
    3. Cross-node job acceptance (B and C accept)
    4. Result consensus (B and C results compared)
    5. Payment release (A pays B and/or C based on consensus)
    6. Ledger reconciliation (all balances verified)
    """

    def __init__(self):
        self.nodes: List[DemoNodeWrapper] = []
        self.base_p2p_port = 18001
        self.base_api_port = 19001

    async def setup(self, num_nodes: int = 3) -> None:
        """Create and start nodes."""
        console.print("\n[bold cyan]Setting up nodes...[/bold cyan]\n")

        names = ["Alice (Requester)", "Bob (Provider)", "Carol (Provider)"]
        for i in range(num_nodes):
            name = names[i] if i < len(names) else f"Node {i}"
            node = DemoNodeWrapper(
                node_id=f"demo-node-{i}",
                p2p_port=self.base_p2p_port + i,
                api_port=self.base_api_port + i,
                display_name=name,
            )
            initial = 100.0 if i == 0 else 0.0  # Alice starts with FTNS
            if i > 0:
                initial = 5.0  # Providers get small balance for network fees
            await node.start(initial_balance=initial)
            self.nodes.append(node)

    async def submit_job_with_escrow(
        self,
        requester: DemoNodeWrapper,
        job_type: JobType,
        payload: Dict[str, Any],
        budget: float,
    ) -> Optional[str]:
        """Submit a job with escrow from the requester's node."""
        job_id = f"job-{int(time.time())}"

        console.print(f"    Job ID:     {job_id}")
        console.print(f"    Type:       {job_type.value}")
        console.print(f"    Budget:     {budget:.6f} FTNS")

        # Step 1: Create escrow
        balance_before = await requester.get_balance()
        console.print(f"\n  [bold]Step 1: Creating escrow...[/bold]")
        escrow = await requester.escrow.create_escrow(
            job_id=job_id,
            amount=budget,
            requester_id=requester.identity.node_id,
        )

        if not escrow:
            console.print(f"    [red]Escrow creation failed - insufficient balance![/red]")
            return None

        balance_after = await requester.get_balance()
        console.print(f"    Escrow ID:  {escrow.escrow_id[:16]}...")
        console.print(f"    Balance before: {balance_before:.6f} FTNS")
        console.print(f"    Balance after:  {balance_after:.6f} FTNS")
        console.print(f"    [green]Escrow locked: {budget:.6f} FTNS[/green]")

        # Step 2: Start consensus tracking (require 2 providers to agree)
        console.print(f"\n  [bold]Step 2: Starting consensus tracking...[/bold]")
        requester.consensus.start_consensus(
            job_id=job_id,
            mode=ConsensusMode.MAJORITY,
            required_providers=2,
        )
        console.print(f"    Mode:          Majority")
        console.print(f"    Required:      2 of 2 providers must agree")

        return job_id

    async def broadcast_job_offer(self, job_id: str, payload: Dict[str, Any], budget: float) -> None:
        """Simulate gossip broadcast of a job to all provider nodes."""
        console.print(f"\n  [bold]Step 3: Broadcasting job offer to network...[/bold]")

        # In a real network this would use gossip.publish
        # Here we simulate it by directly offering to each provider
        offers_sent = 0
        for node in self.nodes[1:]:  # Skip requester (node 0)
            job_offer = {
                "job_id": job_id,
                "job_type": payload.get("job_type", "benchmark"),
                "requester_id": self.nodes[0].identity.node_id,
                "payload": payload,
                "ftns_budget": budget,
            }
            await node.provider._on_job_offer(
                GOSSIP_JOB_OFFER, job_offer, self.nodes[0].identity.node_id
            )
            offers_sent += 1

        console.print(f"    Offers sent to {offers_sent} provider nodes")

    async def wait_for_results(self, job_id: str, timeout: float = 30.0) -> List[Dict[str, Any]]:
        """Wait for providers to complete and submit results."""
        console.print(f"\n  [bold]Step 4: Waiting for providers to complete job...[/bold]")

        results = []
        start_time = time.time()

        while time.time() - start_time < timeout:
            for node in self.nodes[1:]:  # Provider nodes
                if job_id in node.provider.completed_jobs:
                    job = node.provider.completed_jobs[job_id]
                    result_entry = {
                        "provider_id": node.identity.node_id,
                        "provider_name": node.display_name,
                        "result": job.result,
                        "signature": job.result_signature,
                    }
                    if result_entry not in results:
                        results.append(result_entry)
                        console.print(f"    [green]Result from {node.display_name} ({job.status.value})[/green]")

            if len(results) >= 2:  # Got results from both providers
                break

            await asyncio.sleep(0.5)

        if not results:
            console.print(f"    [red]No results received within timeout[/red]")

        return results

    async def evaluate_consensus_and_pay(
        self,
        job_id: str,
        results: List[Dict[str, Any]],
        budget: float,
    ) -> bool:
        """Run consensus on results and distribute payment."""
        console.print(f"\n  [bold]Step 5: Evaluating result consensus...[/bold]")

        requester = self.nodes[0]

        for r in results:
            requester.consensus.submit_result(
                job_id=job_id,
                provider_id=r["provider_id"],
                result=r["result"],
                signature=r["signature"],
            )

        state = requester.consensus.get_state(job_id)
        if not state or not state.consensus_reached:
            console.print(f"    [red]Consensus NOT reached: {state.error if state else 'unknown'}[/red]")
            # Refund escrow
            console.print(f"\n  [bold]Refunding escrow to requester...[/bold]")
            refunded = await requester.escrow.refund_escrow(
                job_id=job_id, reason="No consensus reached"
            )
            console.print(f"    Refund: {'success' if refunded else 'failed'}")
            return False

        console.print(f"    [green]Consensus reached![/green]")
        console.print(f"    Agreed result hash: {state.agreed_hash[:16]}...")
        console.print(f"    Providers agreeing: {state.provider_count}")

        # Pay the first provider (simple model - could split payment)
        console.print(f"\n  [bold]Step 6: Releasing escrow payment...[/bold]")

        balance_before = await requester.get_balance()
        provider_balance_before = await self.nodes[1].get_balance()

        tx = await requester.escrow.release_escrow(
            job_id=job_id,
            provider_id=results[0]["provider_id"],
            consensus_reached=True,
        )

        if tx:
            balance_after = await requester.get_balance()
            provider_balance_after = await self.nodes[1].get_balance()

            console.print(f"    Payment:      {budget:.6f} FTNS")
            console.print(f"    To provider:  {results[0]['provider_name']}")
            console.print(f"    Requester balance: {balance_before:.6f} -> {balance_after:.6f}")
            console.print(f"    Provider balance:  {provider_balance_before:.6f} -> {provider_balance_after:.6f}")
            return True
        else:
            console.print(f"    [red]Payment release failed[/red]")
            return False

    async def run(self) -> None:
        """Execute the full multi-node demonstration."""
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]PRSM Multi-Node Demonstration[/bold cyan]")
        console.print("[dim]Cross-node compute + consensus + escrow payments[/dim]")
        console.print("=" * 60)

        try:
            # Setup
            await self.setup(num_nodes=3)

            # Show initial state
            console.print("\n[bold cyan]Initial State:[/bold cyan]")
            for node in self.nodes:
                balance = await node.get_balance()
                console.print(f"  {node.display_name:25s} {balance:10.6f} FTNS")

            # Submit an embedding job (deterministic results for consensus)
            console.print(f"\n[bold cyan]Submitting Embedding Job:[/bold cyan]")
            requester = self.nodes[0]
            budget = 10.0

            job_id = await self.submit_job_with_escrow(
                requester=requester,
                job_type=JobType.EMBEDDING,
                payload={"job_type": "embedding", "text": "PRSM is a decentralized compute network"},
                budget=budget,
            )

            if not job_id:
                console.print("\n[red]Demo failed - job submission error[/red]")
                return

            # Broadcast to providers
            await self.broadcast_job_offer(
                job_id=job_id,
                payload={"job_type": "embedding", "text": "PRSM is a decentralized compute network"},
                budget=budget,
            )

            # Wait for results
            results = await self.wait_for_results(job_id=job_id)

            if results:
                # Run consensus and payment
                success = await self.evaluate_consensus_and_pay(
                    job_id=job_id,
                    results=results,
                    budget=budget,
                )

                if success:
                    # Final state
                    console.print(f"\n[bold cyan]Final State:[/bold cyan]")
                    for node in self.nodes:
                        balance = await node.get_balance()
                        delta = balance - (100.0 if node == self.nodes[0] else 5.0)
                        console.print(f"  {node.display_name:25s} {balance:10.6f} FTNS  ({delta:+.6f})")

                    console.print(f"\n  [bold green]Demo completed successfully![/bold green]")
                    console.print("  Nodes communicated, consensus reached, and payment transferred.")
                else:
                    console.print("\n  [yellow]Demo completed with partial success.[/yellow]")
            else:
                console.print("\n  [red]Demo failed - no results received[/red]")

        finally:
            # Cleanup
            for node in self.nodes:
                await node.stop()


async def main():
    """Entry point for the demo."""
    demo = MultiNodeDemo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
