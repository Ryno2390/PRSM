"""
AgentDispatcher — Requester-Side Dispatch Lifecycle
====================================================

Orchestrates the full mobile-agent lifecycle from the requester's
perspective: create agent, dispatch manifest via gossip, collect bids,
transfer WASM binary to the winner via direct WebSocket, and settle
escrow on result.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from prsm.compute.agents.executor import TIER_ORDER
from prsm.compute.agents.models import (
    AgentManifest,
    DispatchRecord,
    DispatchStatus,
    MobileAgent,
)
from prsm.node.transport import P2PMessage

logger = logging.getLogger(__name__)


class AgentDispatcher:
    """Manages the dispatch-bid-transfer-settle lifecycle for mobile agents.

    Parameters
    ----------
    identity:
        NodeIdentity-like object with ``node_id``, ``sign()``, ``public_key_b64``.
    gossip:
        GossipProtocol for publishing dispatches and receiving bids/results.
    transport:
        WebSocketTransport for direct peer-to-peer binary transfer.
    escrow:
        Escrow service with ``create()``, ``release()``, ``refund()`` methods.
    bid_timeout:
        Seconds to wait for bids after dispatch.
    result_timeout:
        Seconds to wait for execution result after transfer.
    """

    def __init__(
        self,
        identity: Any,
        gossip: Any,
        transport: Any,
        escrow: Any,
        bid_timeout: float = 10.0,
        result_timeout: float = 120.0,
    ) -> None:
        self.identity = identity
        self.gossip = gossip
        self.transport = transport
        self.escrow = escrow
        self.bid_timeout = bid_timeout
        self.result_timeout = result_timeout

        # agent_id -> DispatchRecord
        self._records: Dict[str, DispatchRecord] = {}
        # agent_id -> MobileAgent (holds the binary for transfer)
        self._agents: Dict[str, MobileAgent] = {}

        # Subscribe to gossip channels
        self.gossip.subscribe("agent_accept", self._on_agent_accept)
        self.gossip.subscribe("agent_result", self._on_agent_result)

    # ── Agent Creation ───────────────────────────────────────────────

    def create_agent(
        self,
        wasm_binary: bytes,
        manifest: AgentManifest,
        ftns_budget: float,
        ttl: int = 120,
    ) -> MobileAgent:
        """Create a signed mobile agent ready for dispatch.

        Returns a MobileAgent with this node as origin and a fresh signature.
        """
        agent_id = str(uuid.uuid4())
        signature = self.identity.sign(agent_id)

        agent = MobileAgent(
            agent_id=agent_id,
            wasm_binary=wasm_binary,
            manifest=manifest,
            origin_node=self.identity.node_id,
            signature=signature,
            ftns_budget=ftns_budget,
            ttl=ttl,
        )
        return agent

    # ── Dispatch ─────────────────────────────────────────────────────

    async def dispatch(
        self,
        agent: MobileAgent,
        bid_timeout: Optional[float] = None,
    ) -> DispatchRecord:
        """Broadcast an agent's manifest and begin collecting bids.

        Creates an escrow hold, publishes the manifest (NOT the binary)
        via gossip, and returns a DispatchRecord in BIDDING state.
        """
        # Store agent for later transfer
        self._agents[agent.agent_id] = agent

        # Create escrow
        escrow_id = await self.escrow.create(agent.ftns_budget)

        # Build dispatch record
        record = DispatchRecord(
            agent_id=agent.agent_id,
            origin_node=self.identity.node_id,
            target_node="",  # unknown until bid selected
            ftns_budget=agent.ftns_budget,
            status=DispatchStatus.BIDDING,
            escrow_id=escrow_id,
        )
        self._records[agent.agent_id] = record

        # Publish manifest via gossip (NO binary)
        payload = {
            "agent_id": agent.agent_id,
            "origin_node": agent.origin_node,
            "manifest": agent.manifest.to_dict(),
            "ftns_budget": agent.ftns_budget,
            "ttl": agent.ttl,
            "binary_hash": agent.binary_hash(),
            "size_bytes": agent.size_bytes,
        }
        await self.gossip.publish("agent_dispatch", payload)

        return record

    # ── Bid Handling ─────────────────────────────────────────────────

    async def _on_agent_accept(
        self,
        subtype: str,
        data: Dict[str, Any],
        sender_id: str,
    ) -> None:
        """Gossip callback: a provider submitted a bid for one of our agents."""
        agent_id = data.get("agent_id")
        if not agent_id or agent_id not in self._records:
            return

        record = self._records[agent_id]
        if record.status != DispatchStatus.BIDDING:
            return

        record.bids.append(data)
        logger.info(
            "Received bid for agent %s from %s (price=%.2f)",
            agent_id,
            data.get("provider_id", sender_id),
            data.get("bid_price", 0),
        )

    async def select_and_transfer(self, agent_id: str) -> bool:
        """Select the best bid and send the WASM binary to the winner.

        Returns True if transfer was initiated, False otherwise.
        """
        record = self._records.get(agent_id)
        agent = self._agents.get(agent_id)
        if not record or not agent:
            return False

        best = self._select_best_bid(record.bids, record.ftns_budget)
        if not best:
            return False

        provider_id = best["provider_id"]
        record.target_node = provider_id
        record.status = DispatchStatus.TRANSFERRING

        # Send binary via direct WebSocket
        msg = P2PMessage(
            msg_type="direct",
            sender_id=self.identity.node_id,
            payload={
                "type": "agent_binary",
                "agent_id": agent_id,
                "wasm_binary_b64": __import__("base64").b64encode(
                    agent.wasm_binary
                ).decode(),
                "manifest": agent.manifest.to_dict(),
                "ftns_budget": agent.ftns_budget,
                "ttl": agent.ttl,
            },
        )
        await self.transport.send_to_peer(provider_id, msg)

        record.status = DispatchStatus.EXECUTING
        logger.info(
            "Transferred agent %s to provider %s", agent_id, provider_id
        )
        return True

    # ── Result Handling ──────────────────────────────────────────────

    async def _on_agent_result(
        self,
        subtype: str,
        data: Dict[str, Any],
        sender_id: str,
    ) -> None:
        """Gossip callback: a provider published an execution result."""
        agent_id = data.get("agent_id")
        if not agent_id or agent_id not in self._records:
            return

        record = self._records[agent_id]
        record.result = data
        status = data.get("status", "error")

        if status == "success":
            record.status = DispatchStatus.COMPLETED
            if record.escrow_id:
                await self.escrow.release(record.escrow_id)
        else:
            record.status = DispatchStatus.FAILED
            if record.escrow_id:
                await self.escrow.refund(record.escrow_id)

        record.completed_at = time.time()
        record.done_event.set()

    # ── Timeout ──────────────────────────────────────────────────────

    async def _check_bid_timeout(self, agent_id: str) -> None:
        """Handle bid timeout: if no bids, refund and fail; otherwise select."""
        record = self._records.get(agent_id)
        if not record:
            return

        if not record.bids:
            record.status = DispatchStatus.FAILED
            if record.escrow_id:
                await self.escrow.refund(record.escrow_id)
            record.completed_at = time.time()
            record.done_event.set()
            logger.info("No bids for agent %s — refunding", agent_id)
        else:
            await self.select_and_transfer(agent_id)

    # ── Wait ─────────────────────────────────────────────────────────

    async def wait_for_result(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Block until the agent's result arrives or timeout elapses.

        Returns the result dict, or None on timeout.
        """
        record = self._records.get(agent_id)
        if not record:
            return None

        wait_time = timeout if timeout is not None else self.result_timeout
        try:
            await asyncio.wait_for(record.done_event.wait(), timeout=wait_time)
            return record.result
        except asyncio.TimeoutError:
            return None

    # ── Bid Selection ────────────────────────────────────────────────

    def _select_best_bid(
        self,
        bids: List[Dict[str, Any]],
        max_budget: float,
    ) -> Optional[Dict[str, Any]]:
        """Score bids and return the best one within budget.

        Score = headroom*0.3 + tier*0.3 + reputation*0.4

        - headroom: (budget - bid_price) / budget  (higher = cheaper)
        - tier: tier_level / 4  (higher = more capable)
        - reputation: raw value 0-1
        """
        if not bids:
            return None

        best_bid = None
        best_score = -1.0

        for bid in bids:
            bid_price = bid.get("bid_price", max_budget)
            if bid_price > max_budget:
                continue

            headroom = (max_budget - bid_price) / max_budget if max_budget > 0 else 0
            tier_str = bid.get("hardware_tier", "t1")
            tier_score = TIER_ORDER.get(tier_str, 1) / 4.0
            reputation = bid.get("reputation", 0.5)

            score = headroom * 0.3 + tier_score * 0.3 + reputation * 0.4

            if score > best_score:
                best_score = score
                best_bid = bid

        return best_bid

    # ── Lookup ───────────────────────────────────────────────────────

    def get_record(self, agent_id: str) -> Optional[DispatchRecord]:
        """Return the dispatch record for an agent, or None if not found."""
        return self._records.get(agent_id)
