"""
Agent Registry
==============

Network-wide directory of AI agents, populated by gossip advertisements.

Similar to ContentIndex but for agent capabilities rather than content CIDs.
Nodes advertise their local agents via GOSSIP_AGENT_ADVERTISE; other nodes
build a local registry of known agents for capability-based discovery.
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import uuid

from prsm.node.agent_identity import AgentIdentity, verify_delegation
from prsm.node.gossip import (
    GOSSIP_AGENT_ADVERTISE,
    GOSSIP_AGENT_DEREGISTER,
    GossipProtocol,
)
from prsm.node.transport import MSG_DIRECT, P2PMessage, PeerConnection, WebSocketTransport

logger = logging.getLogger(__name__)

MAX_REGISTERED_AGENTS = 5_000

# Callback type for agent message dispatch
AgentMessageCallback = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, None]]
# (agent_id, message_payload) -> None


@dataclass
class AgentRecord:
    """A known agent on the network."""
    agent_id: str
    agent_name: str
    agent_type: str
    principal_id: str
    principal_public_key: str
    public_key_b64: str
    delegation_cert: str
    capabilities: List[str] = field(default_factory=list)
    max_spend_ftns: float = 10.0
    node_id: str = ""          # Which node hosts this agent
    status: str = "online"     # online, paused, offline
    last_seen: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "principal_id": self.principal_id,
            "capabilities": self.capabilities,
            "max_spend_ftns": self.max_spend_ftns,
            "node_id": self.node_id,
            "status": self.status,
            "last_seen": self.last_seen,
            "created_at": self.created_at,
        }


class AgentRegistry:
    """Network-wide agent directory built from gossip advertisements.

    Supports:
    - Registration of local agents (on this node)
    - Discovery of remote agents by ID, capability, or principal
    - Message dispatch to local agents via registered callbacks
    - Gossip-based advertisement and deregistration
    """

    def __init__(self, gossip: GossipProtocol, transport: WebSocketTransport, node_id: str):
        self.gossip = gossip
        self.transport = transport
        self.node_id = node_id

        # All known agents (local + remote), keyed by agent_id
        self._agents: OrderedDict[str, AgentRecord] = OrderedDict()
        # capability keyword → set of agent_ids
        self._capability_index: Dict[str, Set[str]] = {}
        # Local agents on this node (subset of _agents)
        self._local_agent_ids: Set[str] = set()
        # Message dispatch callbacks for local agents
        self._message_handlers: Dict[str, AgentMessageCallback] = {}
        # Conversation logs: conversation_id → list of messages
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}

    def start(self) -> None:
        """Subscribe to agent gossip subtypes and register message handler."""
        self.gossip.subscribe(GOSSIP_AGENT_ADVERTISE, self._on_agent_advertise)
        self.gossip.subscribe(GOSSIP_AGENT_DEREGISTER, self._on_agent_deregister)
        self.transport.on_message(MSG_DIRECT, self._on_direct_message)
        logger.info("Agent registry started — listening for agent advertisements")

    # ── Local agent management ────────────────────────────────────

    def register_local(
        self,
        agent: AgentIdentity,
        message_handler: Optional[AgentMessageCallback] = None,
    ) -> AgentRecord:
        """Register an agent that runs on this node.

        Args:
            agent: The agent identity to register
            message_handler: Async callback for dispatching messages to this agent

        Returns:
            The AgentRecord that was created
        """
        record = AgentRecord(
            agent_id=agent.agent_id,
            agent_name=agent.agent_name,
            agent_type=agent.agent_type,
            principal_id=agent.principal_id,
            principal_public_key=agent.principal_public_key,
            public_key_b64=agent.public_key_b64,
            delegation_cert=agent.delegation_cert,
            capabilities=agent.capabilities,
            max_spend_ftns=agent.max_spend_ftns,
            node_id=self.node_id,
            status="online",
            created_at=agent.created_at,
        )
        self._agents[agent.agent_id] = record
        self._local_agent_ids.add(agent.agent_id)
        self._index_capabilities(record)

        if message_handler:
            self._message_handlers[agent.agent_id] = message_handler

        logger.info(f"Registered local agent: {agent.agent_name} ({agent.agent_id[:12]}...)")
        return record

    async def advertise_local_agents(self) -> int:
        """Gossip advertisements for all local agents. Returns count."""
        count = 0
        for agent_id in self._local_agent_ids:
            record = self._agents.get(agent_id)
            if record and record.status != "offline":
                await self.gossip.publish(GOSSIP_AGENT_ADVERTISE, {
                    **record.to_dict(),
                    "public_key_b64": record.public_key_b64,
                    "principal_public_key": record.principal_public_key,
                    "delegation_cert": record.delegation_cert,
                })
                count += 1
        return count

    async def deregister_local(self, agent_id: str) -> None:
        """Remove a local agent and notify the network."""
        if agent_id in self._local_agent_ids:
            self._local_agent_ids.discard(agent_id)
            self._message_handlers.pop(agent_id, None)
            record = self._agents.get(agent_id)
            if record:
                record.status = "offline"
                await self.gossip.publish(GOSSIP_AGENT_DEREGISTER, {
                    "agent_id": agent_id,
                    "node_id": self.node_id,
                })
            logger.info(f"Deregistered local agent: {agent_id[:12]}...")

    def set_agent_status(self, agent_id: str, status: str) -> None:
        """Update status of a local agent (online, paused)."""
        if agent_id in self._local_agent_ids:
            record = self._agents.get(agent_id)
            if record:
                record.status = status

    # ── Discovery / queries ──────────────────────────────────────

    def lookup(self, agent_id: str) -> Optional[AgentRecord]:
        """Look up an agent by ID."""
        return self._agents.get(agent_id)

    def search(self, capability: str, limit: int = 20) -> List[AgentRecord]:
        """Find agents that declare a given capability."""
        cap_lower = capability.lower()
        matching_ids = self._capability_index.get(cap_lower, set())
        results = []
        for aid in matching_ids:
            record = self._agents.get(aid)
            if record and record.status == "online":
                results.append(record)
                if len(results) >= limit:
                    break
        return results

    def get_agents_for_principal(self, principal_id: str) -> List[AgentRecord]:
        """Find all agents belonging to a given principal."""
        return [
            r for r in self._agents.values()
            if r.principal_id == principal_id
        ]

    def get_local_agents(self) -> List[AgentRecord]:
        """Return all agents registered on this node."""
        return [
            self._agents[aid] for aid in self._local_agent_ids
            if aid in self._agents
        ]

    def get_all_agents(self, limit: int = 100) -> List[AgentRecord]:
        """Return all known agents (local + remote)."""
        return list(self._agents.values())[:limit]

    # ── Message dispatch ─────────────────────────────────────────

    async def dispatch_message(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """Dispatch a message to a local agent via its registered handler.

        Returns True if the agent exists and has a handler.
        """
        handler = self._message_handlers.get(agent_id)
        if handler:
            try:
                await handler(agent_id, message)
                return True
            except Exception as e:
                logger.error(f"Agent message dispatch error ({agent_id[:12]}...): {e}")
        return False

    def record_conversation(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Log a message in a conversation thread."""
        self._conversations.setdefault(conversation_id, []).append({
            **message,
            "recorded_at": time.time(),
        })
        # Keep last 100 messages per conversation
        if len(self._conversations[conversation_id]) > 100:
            self._conversations[conversation_id] = self._conversations[conversation_id][-100:]

    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages from a conversation thread."""
        return self._conversations.get(conversation_id, [])

    def get_agent_conversations(self, agent_id: str, limit: int = 10) -> List[str]:
        """Get conversation IDs involving an agent."""
        result = []
        for conv_id, messages in self._conversations.items():
            for msg in messages:
                if msg.get("from_agent") == agent_id or msg.get("to_agent") == agent_id:
                    result.append(conv_id)
                    break
            if len(result) >= limit:
                break
        return result

    # ── Agent-to-agent messaging ───────────────────────────────

    async def send_agent_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        content_type: str,
        content: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """Send a message from one agent to another.

        If the target agent is local, dispatches directly.
        If remote, sends via P2P direct message to the hosting node.

        Args:
            from_agent_id: Sending agent ID (must be local)
            to_agent_id: Receiving agent ID
            content_type: "text", "task", "result", or "query"
            content: Message payload
            conversation_id: Optional thread ID (generated if not provided)

        Returns:
            The conversation_id, or None on failure
        """
        if from_agent_id not in self._local_agent_ids:
            logger.warning(f"Cannot send from non-local agent {from_agent_id[:12]}...")
            return None

        conv_id = conversation_id or str(uuid.uuid4())

        message = {
            "from_agent": from_agent_id,
            "to_agent": to_agent_id,
            "conversation_id": conv_id,
            "content_type": content_type,
            "content": content,
            "timestamp": time.time(),
        }

        # Record in conversation log
        self.record_conversation(conv_id, message)

        target = self._agents.get(to_agent_id)
        if not target:
            logger.warning(f"Target agent {to_agent_id[:12]}... not found in registry")
            return None

        if to_agent_id in self._local_agent_ids:
            # Local dispatch
            await self.dispatch_message(to_agent_id, message)
        else:
            # Remote: send via P2P direct message to the target's node
            target_node = target.node_id
            if not target_node:
                logger.warning(f"No node_id for agent {to_agent_id[:12]}...")
                return None

            msg = P2PMessage(
                msg_type=MSG_DIRECT,
                sender_id=self.node_id,
                payload={
                    "subtype": "agent_message",
                    **message,
                },
            )
            await self.transport.send_to_peer(target_node, msg)

        return conv_id

    async def _on_direct_message(self, msg: P2PMessage, peer: PeerConnection) -> None:
        """Route incoming agent_message direct messages."""
        subtype = msg.payload.get("subtype", "")
        if subtype != "agent_message":
            return

        to_agent = msg.payload.get("to_agent", "")
        if to_agent not in self._local_agent_ids:
            return  # Not for an agent on this node

        conv_id = msg.payload.get("conversation_id", "")
        if conv_id:
            self.record_conversation(conv_id, msg.payload)

        await self.dispatch_message(to_agent, msg.payload)

    # ── Gossip handlers ──────────────────────────────────────────

    async def _on_agent_advertise(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Process an incoming agent advertisement."""
        agent_id = data.get("agent_id", "")
        if not agent_id:
            return

        # Skip our own local agents — we already have them
        if agent_id in self._local_agent_ids:
            return

        # Verify delegation certificate
        if not verify_delegation(data):
            logger.warning(f"Rejected agent {agent_id[:12]}...: invalid delegation certificate")
            return

        if agent_id in self._agents:
            # Update existing record
            record = self._agents[agent_id]
            record.status = data.get("status", "online")
            record.last_seen = time.time()
            record.node_id = data.get("node_id", origin)
            self._agents.move_to_end(agent_id)
        else:
            # New agent record
            record = AgentRecord(
                agent_id=agent_id,
                agent_name=data.get("agent_name", ""),
                agent_type=data.get("agent_type", "general"),
                principal_id=data.get("principal_id", ""),
                principal_public_key=data.get("principal_public_key", ""),
                public_key_b64=data.get("public_key_b64", ""),
                delegation_cert=data.get("delegation_cert", ""),
                capabilities=data.get("capabilities", []),
                max_spend_ftns=data.get("max_spend_ftns", 10.0),
                node_id=data.get("node_id", origin),
                status=data.get("status", "online"),
                created_at=data.get("created_at", time.time()),
            )
            self._agents[agent_id] = record
            self._index_capabilities(record)
            self._evict_if_needed()

        logger.debug(f"Agent registry: {data.get('agent_name', agent_id[:12])} on node {origin[:12]}...")

    async def _on_agent_deregister(
        self, subtype: str, data: Dict[str, Any], origin: str
    ) -> None:
        """Mark an agent as offline when deregistered."""
        agent_id = data.get("agent_id", "")
        if agent_id in self._agents and agent_id not in self._local_agent_ids:
            self._agents[agent_id].status = "offline"

    # ── Internal helpers ─────────────────────────────────────────

    def _index_capabilities(self, record: AgentRecord) -> None:
        """Add agent capabilities to the capability index."""
        for cap in record.capabilities:
            self._capability_index.setdefault(cap.lower(), set()).add(record.agent_id)

    def _evict_if_needed(self) -> None:
        """Remove oldest non-local agents when exceeding the cap."""
        while len(self._agents) > MAX_REGISTERED_AGENTS:
            for aid in list(self._agents.keys()):
                if aid not in self._local_agent_ids:
                    evicted = self._agents.pop(aid)
                    for cap in evicted.capabilities:
                        cap_set = self._capability_index.get(cap.lower())
                        if cap_set:
                            cap_set.discard(aid)
                            if not cap_set:
                                del self._capability_index[cap.lower()]
                    break
            else:
                break  # Only local agents left, can't evict

    def get_stats(self) -> Dict[str, Any]:
        """Registry statistics for the status endpoint."""
        online = sum(1 for r in self._agents.values() if r.status == "online")
        return {
            "total_agents": len(self._agents),
            "local_agents": len(self._local_agent_ids),
            "online_agents": online,
            "capability_keywords": len(self._capability_index),
            "active_conversations": len(self._conversations),
        }
