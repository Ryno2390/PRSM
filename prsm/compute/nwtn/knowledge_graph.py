"""
PRSM Distributed Vector Knowledge Graph
=======================================

Implements:
1. Knowledge Pub/Sub: Real-time discovery propagation.
2. Vector-Based Contextual Memory: Immediate indexing of new findings.
3. Recursive Context Synchronization: Allowing agents to 'subscribe' to domain updates.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class DiscoveryUpdate:
    cid: str
    domain: str
    content: str
    impact_level: int
    timestamp: datetime = datetime.now(timezone.utc)

@dataclass
class ReasoningIntervention:
    """An agentic intervention in a live reasoning branch"""
    trace_id: str
    intervening_agent_id: str
    suggestion: str
    target_step_index: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class RecursiveKnowledgeGraph:
    """
    A distributed, living map of PRSM knowledge.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {} # domain -> callback functions
        self.knowledge_index: List[DiscoveryUpdate] = []
        self.live_traces: Dict[str, List[ReasoningIntervention]] = {} # trace_id -> interventions

    def submit_intervention(self, intervention: ReasoningIntervention):
        """Allows an agent to suggest an improvement to a live reasoning branch"""
        if intervention.trace_id not in self.live_traces:
            self.live_traces[intervention.trace_id] = []
        self.live_traces[intervention.trace_id].append(intervention)
        logger.info(f"🤝 Intervention received for trace {intervention.trace_id} from {intervention.intervening_agent_id}")

    def subscribe(self, domain: str, callback: Callable):
        """Allows an agent/orchestrator to subscribe to domain updates"""
        if domain not in self.subscribers:
            self.subscribers[domain] = []
        self.subscribers[domain].append(callback)
        logger.info(f"New subscriber added for domain: {domain}")

    async def publish_discovery(self, update: DiscoveryUpdate):
        """Publishes a new finding to the global knowledge graph"""
        self.knowledge_index.append(update)
        
        # 1. Immediate Indexing (Simulated Vector DB insert)
        logger.info(f"Indexing new discovery: {update.cid} in domain {update.domain}")
        
        # 2. Notify Subscribers
        if update.domain in self.subscribers:
            notifications = [
                callback(update) for callback in self.subscribers[update.domain]
            ]
            if notifications:
                await asyncio.gather(*notifications) if asyncio.iscoroutine(notifications[0]) else None
        
        # Global update notification
        if "global" in self.subscribers:
            for callback in self.subscribers["global"]:
                callback(update)

class GlobalBrainSync:
    """
    Ensures Orchestrators maintain an updated 'World Model' context.
    """
    def __init__(self, orchestrator_id: str, kg: RecursiveKnowledgeGraph):
        self.id = orchestrator_id
        self.kg = kg
        self.local_context_buffer: List[DiscoveryUpdate] = []

    def start_sync(self, domains: List[str]):
        for domain in domains:
            self.kg.subscribe(domain, self._on_discovery_received)

    async def _on_discovery_received(self, update: DiscoveryUpdate):
        """Callback when new knowledge is published"""
        logger.info(f"🧠 Global Brain Sync [{self.id}]: Received update {update.cid}")
        self.local_context_buffer.append(update)
        # In a real implementation, this would trigger an embedding update 
        # in the orchestrator's local vector cache.
