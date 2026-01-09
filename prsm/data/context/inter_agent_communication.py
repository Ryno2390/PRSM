"""
Inter-Agent Communication Protocols

🧠 COGNITION.AI INSIGHTS INTEGRATION:
- Structured agent message passing with comprehensive metadata and context tracking
- Agent coordination consensus mechanisms for distributed decision-making
- Enhanced communication protocols that address coordination challenges in multi-agent systems
- Message routing, prioritization, and conflict resolution for optimal agent collaboration
- Communication pattern analysis and optimization for improved multi-agent performance

This module implements sophisticated inter-agent communication that addresses
Cognition.AI's insights about poor coordination in naive multi-agent frameworks
and the need for structured, context-aware agent communication.
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from uuid import UUID, uuid4
from collections import defaultdict, deque
import pickle
import zlib

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import (
    PRSMBaseModel, TimestampMixin, AgentType, TaskStatus, SafetyLevel
)
from prsm.data.context.enhanced_context_compression import (
    ContextSegment, ContextType, ContextImportance, EnhancedContextCompressionEngine
)
from prsm.data.context.reasoning_trace_sharing import (
    ReasoningTraceLevel, EnhancedReasoningStep, ReasoningTraceSharingEngine
)
from prsm.data.context.selective_parallelism_engine import (
    ExecutionStrategy, TaskDefinition, ParallelismDecision
)

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """Types of inter-agent messages"""
    TASK_REQUEST = "task_request"               # Request another agent to perform a task
    TASK_RESPONSE = "task_response"             # Response to a task request
    CONTEXT_SHARE = "context_share"             # Share context information
    REASONING_SHARE = "reasoning_share"         # Share reasoning trace
    COORDINATION_REQUEST = "coordination_request"  # Request coordination/consensus
    COORDINATION_RESPONSE = "coordination_response"  # Response to coordination request
    STATUS_UPDATE = "status_update"            # Agent status update
    ERROR_NOTIFICATION = "error_notification"   # Error or failure notification
    RESOURCE_REQUEST = "resource_request"       # Request for resources
    RESOURCE_RESPONSE = "resource_response"     # Response to resource request
    BROADCAST = "broadcast"                     # Broadcast message to all agents
    DIRECT_MESSAGE = "direct_message"          # Direct agent-to-agent message


class MessagePriority(str, Enum):
    """Message priority levels"""
    CRITICAL = "critical"    # Immediate attention required
    HIGH = "high"           # High priority, process soon
    NORMAL = "normal"       # Normal priority
    LOW = "low"            # Low priority, process when convenient
    BACKGROUND = "background"  # Background information, no urgency


class ConsensusType(str, Enum):
    """Types of consensus mechanisms"""
    UNANIMOUS = "unanimous"     # All agents must agree
    MAJORITY = "majority"       # Majority agreement required
    QUORUM = "quorum"          # Minimum number of agents must agree
    WEIGHTED = "weighted"       # Weighted voting based on agent expertise
    LEADER_BASED = "leader_based"  # Leader makes final decision
    DEMOCRATIC = "democratic"   # Simple majority vote


class CommunicationPattern(str, Enum):
    """Communication patterns between agents"""
    POINT_TO_POINT = "point_to_point"     # Direct agent-to-agent
    BROADCAST = "broadcast"               # One-to-many
    MULTICAST = "multicast"              # One-to-specific-group
    PUBLISH_SUBSCRIBE = "publish_subscribe"  # Pub/sub pattern
    REQUEST_RESPONSE = "request_response"    # Synchronous req/resp
    PIPELINE = "pipeline"                 # Sequential processing
    MESH = "mesh"                        # Full mesh communication


class AgentMessage(PRSMBaseModel):
    """Comprehensive inter-agent message"""
    message_id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID = Field(default_factory=uuid4)
    
    # Message Routing
    sender_agent_id: str
    sender_agent_type: AgentType
    recipient_agent_ids: List[str] = Field(default_factory=list)
    recipient_agent_types: List[AgentType] = Field(default_factory=list)
    
    # Message Classification
    message_type: MessageType
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    communication_pattern: CommunicationPattern = Field(default=CommunicationPattern.POINT_TO_POINT)
    
    # Message Content
    subject: str
    content: Dict[str, Any] = Field(default_factory=dict)
    payload_data: Optional[bytes] = None  # For large binary data
    
    # Context and Metadata
    context_segments: List[ContextSegment] = Field(default_factory=list)
    reasoning_trace_ids: List[UUID] = Field(default_factory=list)
    task_dependencies: List[UUID] = Field(default_factory=list)
    
    # Timing and Lifecycle
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    response_required: bool = Field(default=False)
    response_timeout: Optional[int] = None  # seconds
    
    # Message Threading
    in_reply_to: Optional[UUID] = None
    thread_id: Optional[UUID] = None
    message_sequence: int = Field(default=1)
    
    # Routing and Delivery
    routing_path: List[str] = Field(default_factory=list)
    delivery_attempts: int = Field(default=0)
    max_delivery_attempts: int = Field(default=3)
    delivery_confirmations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Security and Validation
    message_hash: Optional[str] = None
    digital_signature: Optional[str] = None
    encryption_key_id: Optional[str] = None
    
    # Performance Tracking
    size_bytes: int = Field(default=0)
    compression_ratio: float = Field(default=1.0)
    processing_time_ms: List[float] = Field(default_factory=list)
    
    def calculate_message_hash(self) -> str:
        """Calculate hash of message content"""
        message_content = {
            "sender": self.sender_agent_id,
            "type": self.message_type,
            "subject": self.subject,
            "content": json.dumps(self.content, sort_keys=True),
            "created_at": self.created_at.isoformat()
        }
        content_str = json.dumps(message_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def compress_payload(self) -> None:
        """Compress large payload data"""
        if self.payload_data and len(self.payload_data) > 1024:
            compressed = zlib.compress(self.payload_data)
            self.compression_ratio = len(compressed) / len(self.payload_data)
            self.payload_data = compressed
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class ConsensusRequest(PRSMBaseModel):
    """Request for agent consensus"""
    consensus_id: UUID = Field(default_factory=uuid4)
    initiator_agent_id: str
    initiator_agent_type: AgentType
    
    # Consensus Parameters
    consensus_type: ConsensusType
    topic: str
    proposal: Dict[str, Any]
    
    # Participants
    required_participants: List[str] = Field(default_factory=list)
    optional_participants: List[str] = Field(default_factory=list)
    minimum_participants: int = Field(default=1)
    
    # Timing
    voting_deadline: datetime
    decision_timeout: int = Field(default=300)  # seconds
    
    # Consensus Rules
    unanimous_required: bool = Field(default=False)
    majority_threshold: float = Field(ge=0.5, le=1.0, default=0.6)
    quorum_size: Optional[int] = None
    agent_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Context
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_trace_ids: List[UUID] = Field(default_factory=list)
    background_context: str = Field(default="")


class ConsensusResponse(PRSMBaseModel):
    """Response to consensus request"""
    consensus_id: UUID
    respondent_agent_id: str
    respondent_agent_type: AgentType
    
    # Vote
    vote: str  # "approve", "reject", "abstain", "conditional"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    
    # Reasoning
    rationale: str
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    alternative_proposals: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Conditions (if conditional vote)
    conditions: List[str] = Field(default_factory=list)
    
    # Timing
    response_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConsensusResult(PRSMBaseModel):
    """Result of consensus process"""
    consensus_id: UUID
    consensus_request: ConsensusRequest
    
    # Results
    consensus_reached: bool
    final_decision: str  # "approved", "rejected", "failed", "timeout"
    
    # Vote Tallies
    total_participants: int
    votes_received: int
    approve_votes: int
    reject_votes: int
    abstain_votes: int
    conditional_votes: int
    
    # Detailed Results
    vote_breakdown: Dict[str, ConsensusResponse] = Field(default_factory=dict)
    winning_margin: float = Field(default=0.0)
    consensus_strength: float = Field(ge=0.0, le=1.0, default=0.0)
    
    # Implementation
    implementation_plan: Optional[Dict[str, Any]] = None
    implementation_timeline: Optional[datetime] = None
    responsible_agents: List[str] = Field(default_factory=list)
    
    # Timestamps
    started_at: datetime
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CommunicationChannel(PRSMBaseModel):
    """Communication channel between agents"""
    channel_id: UUID = Field(default_factory=uuid4)
    channel_name: str
    
    # Participants
    participant_agents: List[str] = Field(default_factory=list)
    moderator_agent: Optional[str] = None
    
    # Channel Configuration
    pattern: CommunicationPattern
    max_message_size: int = Field(default=1024*1024)  # 1MB
    message_retention_days: int = Field(default=30)
    allow_broadcast: bool = Field(default=True)
    require_authentication: bool = Field(default=True)
    
    # Message Filtering
    allowed_message_types: List[MessageType] = Field(
        default_factory=lambda: list(MessageType)
    )
    priority_filtering: bool = Field(default=False)
    content_filtering_enabled: bool = Field(default=True)
    
    # Performance
    message_queue: deque = Field(default_factory=deque)
    max_queue_size: int = Field(default=1000)
    message_rate_limit: int = Field(default=100)  # messages per minute
    
    # Statistics
    total_messages_sent: int = Field(default=0)
    total_messages_received: int = Field(default=0)
    average_response_time: float = Field(default=0.0)
    
    class Config:
        arbitrary_types_allowed = True


class InterAgentCommunicationEngine(TimestampMixin):
    """
    Inter-Agent Communication Engine
    
    Sophisticated communication system for multi-agent coordination,
    addressing Cognition.AI's insights about communication challenges
    in naive multi-agent frameworks.
    """
    
    def __init__(self):
        super().__init__()
        
        # Communication Infrastructure
        self.channels: Dict[UUID, CommunicationChannel] = {}
        self.message_queue: deque = deque()
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        
        # Active Communications
        self.active_conversations: Dict[UUID, List[AgentMessage]] = defaultdict(list)
        self.pending_responses: Dict[UUID, AgentMessage] = {}
        self.active_consensus: Dict[UUID, ConsensusRequest] = {}
        
        # Performance Monitoring
        self.message_statistics: Dict[str, Any] = defaultdict(int)
        self.communication_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.performance_metrics: Dict[str, float] = {}
        
        # Context Integration
        self.context_engine: Optional[EnhancedContextCompressionEngine] = None
        self.reasoning_engine: Optional[ReasoningTraceSharingEngine] = None
        
        # Configuration
        self.max_concurrent_conversations = 100
        self.message_processing_batch_size = 50
        self.consensus_timeout_default = 300  # 5 minutes
        self.message_retention_days = 30
        
        logger.info("InterAgentCommunicationEngine initialized")
    
    def set_context_engine(self, engine: EnhancedContextCompressionEngine):
        """Set context compression engine for integration"""
        self.context_engine = engine
        logger.info("Context engine integrated with communication system")
    
    def set_reasoning_engine(self, engine: ReasoningTraceSharingEngine):
        """Set reasoning trace engine for integration"""
        self.reasoning_engine = engine
        logger.info("Reasoning engine integrated with communication system")
    
    async def create_communication_channel(
        self,
        channel_name: str,
        participant_agents: List[str],
        pattern: CommunicationPattern = CommunicationPattern.POINT_TO_POINT,
        moderator_agent: Optional[str] = None
    ) -> CommunicationChannel:
        """
        Create a new communication channel
        
        Args:
            channel_name: Name of the channel
            participant_agents: List of agent IDs that can use this channel
            pattern: Communication pattern for the channel
            moderator_agent: Optional moderator agent ID
            
        Returns:
            Created communication channel
        """
        try:
            channel = CommunicationChannel(
                channel_name=channel_name,
                participant_agents=participant_agents,
                pattern=pattern,
                moderator_agent=moderator_agent
            )
            
            self.channels[channel.channel_id] = channel
            
            # Update routing table
            for agent_id in participant_agents:
                self.routing_table[agent_id].append(str(channel.channel_id))
            
            logger.info(
                "Communication channel created",
                channel_id=str(channel.channel_id),
                channel_name=channel_name,
                participants=len(participant_agents),
                pattern=pattern
            )
            
            return channel
            
        except Exception as e:
            logger.error("Error creating communication channel", error=str(e))
            raise
    
    async def send_message(
        self,
        sender_agent_id: str,
        sender_agent_type: AgentType,
        message_type: MessageType,
        subject: str,
        content: Dict[str, Any],
        recipient_agent_ids: List[str],
        priority: MessagePriority = MessagePriority.NORMAL,
        response_required: bool = False,
        response_timeout: Optional[int] = None,
        context_segments: Optional[List[ContextSegment]] = None,
        reasoning_trace_ids: Optional[List[UUID]] = None
    ) -> AgentMessage:
        """
        Send message between agents
        
        Args:
            sender_agent_id: ID of sending agent
            sender_agent_type: Type of sending agent
            message_type: Type of message
            subject: Message subject
            content: Message content
            recipient_agent_ids: List of recipient agent IDs
            priority: Message priority
            response_required: Whether response is required
            response_timeout: Timeout for response in seconds
            context_segments: Optional context segments to include
            reasoning_trace_ids: Optional reasoning trace IDs
            
        Returns:
            Sent message
        """
        try:
            message = AgentMessage(
                sender_agent_id=sender_agent_id,
                sender_agent_type=sender_agent_type,
                recipient_agent_ids=recipient_agent_ids,
                message_type=message_type,
                subject=subject,
                content=content,
                priority=priority,
                response_required=response_required,
                response_timeout=response_timeout,
                context_segments=context_segments or [],
                reasoning_trace_ids=reasoning_trace_ids or []
            )
            
            # Calculate message hash
            message.message_hash = message.calculate_message_hash()
            
            # Compress if needed
            if message.payload_data:
                message.compress_payload()
            
            # Set expiration if response required
            if response_required and response_timeout:
                message.expires_at = datetime.now(timezone.utc) + timedelta(seconds=response_timeout)
            
            # Add to message queue
            self.message_queue.append(message)
            
            # Track in active conversations
            self.active_conversations[message.conversation_id].append(message)
            
            # Track pending response if required
            if response_required:
                self.pending_responses[message.message_id] = message
            
            # Update statistics
            self._update_message_statistics(message)
            
            logger.info(
                "Message sent",
                message_id=str(message.message_id),
                sender=sender_agent_id,
                recipients=len(recipient_agent_ids),
                type=message_type,
                priority=priority
            )
            
            return message
            
        except Exception as e:
            logger.error("Error sending message", error=str(e))
            raise
    
    async def receive_messages(
        self,
        agent_id: str,
        message_types: Optional[List[MessageType]] = None,
        priority_filter: Optional[MessagePriority] = None,
        limit: int = 10
    ) -> List[AgentMessage]:
        """
        Receive messages for an agent
        
        Args:
            agent_id: ID of receiving agent
            message_types: Optional filter for message types
            priority_filter: Optional minimum priority filter
            limit: Maximum number of messages to return
            
        Returns:
            List of messages for the agent
        """
        try:
            messages = []
            processed_count = 0
            
            # Process messages from queue
            temp_queue = deque()
            
            while self.message_queue and processed_count < limit:
                message = self.message_queue.popleft()
                
                # Check if message is for this agent
                if (agent_id in message.recipient_agent_ids or 
                    not message.recipient_agent_ids):  # Broadcast message
                    
                    # Apply filters
                    if message_types and message.message_type not in message_types:
                        temp_queue.append(message)
                        continue
                    
                    if (priority_filter and 
                        self._get_priority_value(message.priority) < self._get_priority_value(priority_filter)):
                        temp_queue.append(message)
                        continue
                    
                    # Check if message is expired
                    if message.is_expired():
                        logger.warning("Message expired", message_id=str(message.message_id))
                        continue
                    
                    messages.append(message)
                    processed_count += 1
                else:
                    # Message not for this agent, put back in queue
                    temp_queue.append(message)
            
            # Put unprocessed messages back in queue
            while temp_queue:
                self.message_queue.appendleft(temp_queue.pop())
            
            logger.info(
                "Messages received",
                agent_id=agent_id,
                message_count=len(messages),
                queue_size=len(self.message_queue)
            )
            
            return messages
            
        except Exception as e:
            logger.error("Error receiving messages", error=str(e))
            return []
    
    def _get_priority_value(self, priority: MessagePriority) -> int:
        """Get numeric value for priority comparison"""
        priority_values = {
            MessagePriority.CRITICAL: 5,
            MessagePriority.HIGH: 4,
            MessagePriority.NORMAL: 3,
            MessagePriority.LOW: 2,
            MessagePriority.BACKGROUND: 1
        }
        return priority_values.get(priority, 3)
    
    async def send_response(
        self,
        original_message: AgentMessage,
        sender_agent_id: str,
        sender_agent_type: AgentType,
        response_content: Dict[str, Any],
        message_type: MessageType = MessageType.TASK_RESPONSE
    ) -> AgentMessage:
        """
        Send response to a message
        
        Args:
            original_message: Original message being replied to
            sender_agent_id: ID of responding agent
            sender_agent_type: Type of responding agent
            response_content: Response content
            message_type: Type of response message
            
        Returns:
            Response message
        """
        try:
            response = await self.send_message(
                sender_agent_id=sender_agent_id,
                sender_agent_type=sender_agent_type,
                message_type=message_type,
                subject=f"Re: {original_message.subject}",
                content=response_content,
                recipient_agent_ids=[original_message.sender_agent_id],
                priority=original_message.priority
            )
            
            # Link messages in thread
            response.in_reply_to = original_message.message_id
            response.thread_id = original_message.thread_id or original_message.message_id
            response.conversation_id = original_message.conversation_id
            
            # Remove from pending responses
            if original_message.message_id in self.pending_responses:
                del self.pending_responses[original_message.message_id]
            
            logger.info(
                "Response sent",
                original_message_id=str(original_message.message_id),
                response_message_id=str(response.message_id),
                sender=sender_agent_id
            )
            
            return response
            
        except Exception as e:
            logger.error("Error sending response", error=str(e))
            raise
    
    async def initiate_consensus(
        self,
        initiator_agent_id: str,
        initiator_agent_type: AgentType,
        topic: str,
        proposal: Dict[str, Any],
        consensus_type: ConsensusType = ConsensusType.MAJORITY,
        required_participants: List[str] = None,
        voting_deadline_minutes: int = 10,
        supporting_evidence: List[Dict[str, Any]] = None
    ) -> ConsensusRequest:
        """
        Initiate consensus process among agents
        
        Args:
            initiator_agent_id: ID of agent initiating consensus
            initiator_agent_type: Type of initiating agent
            topic: Topic for consensus
            proposal: Proposal details
            consensus_type: Type of consensus mechanism
            required_participants: List of required participant agent IDs
            voting_deadline_minutes: Voting deadline in minutes
            supporting_evidence: Optional supporting evidence
            
        Returns:
            Consensus request
        """
        try:
            consensus_request = ConsensusRequest(
                initiator_agent_id=initiator_agent_id,
                initiator_agent_type=initiator_agent_type,
                consensus_type=consensus_type,
                topic=topic,
                proposal=proposal,
                required_participants=required_participants or [],
                voting_deadline=datetime.now(timezone.utc) + timedelta(minutes=voting_deadline_minutes),
                supporting_evidence=supporting_evidence or []
            )
            
            # Store active consensus
            self.active_consensus[consensus_request.consensus_id] = consensus_request
            
            # Send consensus request to participants
            await self.send_message(
                sender_agent_id=initiator_agent_id,
                sender_agent_type=initiator_agent_type,
                message_type=MessageType.COORDINATION_REQUEST,
                subject=f"Consensus Request: {topic}",
                content={
                    "consensus_id": str(consensus_request.consensus_id),
                    "consensus_type": consensus_type,
                    "topic": topic,
                    "proposal": proposal,
                    "voting_deadline": consensus_request.voting_deadline.isoformat(),
                    "supporting_evidence": supporting_evidence or []
                },
                recipient_agent_ids=required_participants or [],
                priority=MessagePriority.HIGH,
                response_required=True,
                response_timeout=voting_deadline_minutes * 60
            )
            
            logger.info(
                "Consensus initiated",
                consensus_id=str(consensus_request.consensus_id),
                initiator=initiator_agent_id,
                topic=topic,
                participants=len(required_participants or [])
            )
            
            return consensus_request
            
        except Exception as e:
            logger.error("Error initiating consensus", error=str(e))
            raise
    
    async def submit_consensus_response(
        self,
        consensus_id: UUID,
        respondent_agent_id: str,
        respondent_agent_type: AgentType,
        vote: str,
        rationale: str,
        confidence: float = 1.0,
        conditions: List[str] = None
    ) -> ConsensusResponse:
        """
        Submit response to consensus request
        
        Args:
            consensus_id: ID of consensus request
            respondent_agent_id: ID of responding agent
            respondent_agent_type: Type of responding agent
            vote: Vote ("approve", "reject", "abstain", "conditional")
            rationale: Reasoning for the vote
            confidence: Confidence in the vote (0.0-1.0)
            conditions: Optional conditions for conditional vote
            
        Returns:
            Consensus response
        """
        try:
            if consensus_id not in self.active_consensus:
                raise ValueError(f"Consensus {consensus_id} not found or not active")
            
            consensus_request = self.active_consensus[consensus_id]
            
            # Check if voting is still open
            if datetime.now(timezone.utc) > consensus_request.voting_deadline:
                raise ValueError("Voting deadline has passed")
            
            response = ConsensusResponse(
                consensus_id=consensus_id,
                respondent_agent_id=respondent_agent_id,
                respondent_agent_type=respondent_agent_type,
                vote=vote,
                confidence=confidence,
                rationale=rationale,
                conditions=conditions or []
            )
            
            # Send response message
            await self.send_message(
                sender_agent_id=respondent_agent_id,
                sender_agent_type=respondent_agent_type,
                message_type=MessageType.COORDINATION_RESPONSE,
                subject=f"Consensus Response: {consensus_request.topic}",
                content={
                    "consensus_id": str(consensus_id),
                    "vote": vote,
                    "confidence": confidence,
                    "rationale": rationale,
                    "conditions": conditions or []
                },
                recipient_agent_ids=[consensus_request.initiator_agent_id],
                priority=MessagePriority.HIGH
            )
            
            logger.info(
                "Consensus response submitted",
                consensus_id=str(consensus_id),
                respondent=respondent_agent_id,
                vote=vote,
                confidence=confidence
            )
            
            return response
            
        except Exception as e:
            logger.error("Error submitting consensus response", error=str(e))
            raise
    
    async def evaluate_consensus(
        self,
        consensus_id: UUID,
        responses: List[ConsensusResponse]
    ) -> ConsensusResult:
        """
        Evaluate consensus responses and determine result
        
        Args:
            consensus_id: ID of consensus request
            responses: List of consensus responses
            
        Returns:
            Consensus result
        """
        try:
            if consensus_id not in self.active_consensus:
                raise ValueError(f"Consensus {consensus_id} not found")
            
            consensus_request = self.active_consensus[consensus_id]
            
            # Count votes
            approve_votes = len([r for r in responses if r.vote == "approve"])
            reject_votes = len([r for r in responses if r.vote == "reject"]) 
            abstain_votes = len([r for r in responses if r.vote == "abstain"])
            conditional_votes = len([r for r in responses if r.vote == "conditional"])
            
            total_votes = len(responses)
            total_participants = len(consensus_request.required_participants)
            
            # Determine consensus result
            consensus_reached = False
            final_decision = "failed"
            winning_margin = 0.0
            consensus_strength = 0.0
            
            if consensus_request.consensus_type == ConsensusType.UNANIMOUS:
                if approve_votes == total_votes and total_votes == total_participants:
                    consensus_reached = True
                    final_decision = "approved" 
                    consensus_strength = 1.0
                elif reject_votes > 0:
                    final_decision = "rejected"
                
            elif consensus_request.consensus_type == ConsensusType.MAJORITY:
                if total_votes >= total_participants * 0.5:  # Quorum check
                    if approve_votes > reject_votes:
                        consensus_reached = True
                        final_decision = "approved"
                        winning_margin = (approve_votes - reject_votes) / total_votes
                        consensus_strength = approve_votes / total_votes
                    else:
                        final_decision = "rejected"
                        winning_margin = (reject_votes - approve_votes) / total_votes
            
            elif consensus_request.consensus_type == ConsensusType.QUORUM:
                quorum_size = consensus_request.quorum_size or max(1, total_participants // 2)
                if total_votes >= quorum_size:
                    if approve_votes > reject_votes:
                        consensus_reached = True
                        final_decision = "approved"
                        consensus_strength = approve_votes / total_votes
            
            # Create result
            result = ConsensusResult(
                consensus_id=consensus_id,
                consensus_request=consensus_request,
                consensus_reached=consensus_reached,
                final_decision=final_decision,
                total_participants=total_participants,
                votes_received=total_votes,
                approve_votes=approve_votes,
                reject_votes=reject_votes,
                abstain_votes=abstain_votes,
                conditional_votes=conditional_votes,
                vote_breakdown={r.respondent_agent_id: r for r in responses},
                winning_margin=winning_margin,
                consensus_strength=consensus_strength,
                started_at=consensus_request.voting_deadline - timedelta(
                    seconds=consensus_request.decision_timeout
                )
            )
            
            # Remove from active consensus
            if consensus_id in self.active_consensus:
                del self.active_consensus[consensus_id]
            
            logger.info(
                "Consensus evaluated",
                consensus_id=str(consensus_id),
                consensus_reached=consensus_reached,
                final_decision=final_decision,
                approve_votes=approve_votes,
                reject_votes=reject_votes
            )
            
            return result
            
        except Exception as e:
            logger.error("Error evaluating consensus", error=str(e))
            raise
    
    def _update_message_statistics(self, message: AgentMessage):
        """Update message statistics"""
        self.message_statistics["total_messages"] += 1
        self.message_statistics[f"type_{message.message_type}"] += 1
        self.message_statistics[f"priority_{message.priority}"] += 1
        self.message_statistics[f"sender_{message.sender_agent_id}"] += 1
        
        # Track communication patterns
        pattern_key = f"{message.sender_agent_id}_to_{'_'.join(message.recipient_agent_ids)}"
        self.communication_patterns[pattern_key].append(datetime.now(timezone.utc))
    
    async def analyze_communication_patterns(self) -> Dict[str, Any]:
        """
        Analyze communication patterns for optimization insights
        
        Returns:
            Communication pattern analysis
        """
        try:
            analysis = {
                "total_messages": self.message_statistics.get("total_messages", 0),
                "active_conversations": len(self.active_conversations),
                "pending_responses": len(self.pending_responses),
                "active_consensus": len(self.active_consensus),
                "communication_channels": len(self.channels),
                "message_type_distribution": {},
                "priority_distribution": {},
                "agent_activity": {},
                "peak_communication_times": [],
                "most_active_patterns": [],
                "optimization_suggestions": []
            }
            
            # Message type distribution
            for key, count in self.message_statistics.items():
                if key.startswith("type_"):
                    message_type = key.replace("type_", "")
                    analysis["message_type_distribution"][message_type] = count
            
            # Priority distribution  
            for key, count in self.message_statistics.items():
                if key.startswith("priority_"):
                    priority = key.replace("priority_", "")
                    analysis["priority_distribution"][priority] = count
            
            # Agent activity
            for key, count in self.message_statistics.items():
                if key.startswith("sender_"):
                    agent_id = key.replace("sender_", "")
                    analysis["agent_activity"][agent_id] = count
            
            # Communication patterns
            pattern_counts = {}
            for pattern, timestamps in self.communication_patterns.items():
                pattern_counts[pattern] = len(timestamps)
            
            # Most active patterns
            analysis["most_active_patterns"] = sorted(
                pattern_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            # Performance analysis
            queue_size = len(self.message_queue)
            if queue_size > 100:
                analysis["optimization_suggestions"].append(
                    "Message queue is large, consider increasing processing capacity"
                )
            
            pending_count = len(self.pending_responses)
            if pending_count > 20:
                analysis["optimization_suggestions"].append(
                    "Many pending responses, consider timeout adjustments"
                )
            
            critical_messages = self.message_statistics.get("priority_critical", 0)
            total_messages = self.message_statistics.get("total_messages", 1)
            if critical_messages / total_messages > 0.3:
                analysis["optimization_suggestions"].append(
                    "High proportion of critical messages, review priority assignment"
                )
            
            logger.info(
                "Communication pattern analysis completed",
                total_messages=analysis["total_messages"],
                active_conversations=analysis["active_conversations"],
                suggestions=len(analysis["optimization_suggestions"])
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Error analyzing communication patterns", error=str(e))
            return {"error": str(e)}
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get current communication statistics"""
        return {
            "message_statistics": dict(self.message_statistics),
            "active_conversations": len(self.active_conversations),
            "pending_responses": len(self.pending_responses),
            "active_consensus": len(self.active_consensus),
            "message_queue_size": len(self.message_queue),
            "communication_channels": len(self.channels),
            "performance_metrics": dict(self.performance_metrics)
        }


# Global instance for easy access
_inter_agent_communication_engine = None

def get_inter_agent_communication_engine() -> InterAgentCommunicationEngine:
    """Get global inter-agent communication engine instance"""
    global _inter_agent_communication_engine
    if _inter_agent_communication_engine is None:
        _inter_agent_communication_engine = InterAgentCommunicationEngine()
    return _inter_agent_communication_engine