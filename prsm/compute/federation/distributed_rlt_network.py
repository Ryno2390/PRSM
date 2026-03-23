"""
Distributed RLT Teacher Network

Advanced federated network system for RLT (Reinforcement Learning Teachers) that enables
teacher discovery, quality metrics sharing, collaborative improvement, and distributed
teaching coordination across multiple nodes and domains.

Key Features:
- Distributed teacher discovery with quality thresholds
- Real-time explanation quality metrics sharing
- Collaborative improvement coordination
- Load balancing and fault tolerance
- Cross-domain teacher specialization
- Network consensus for teacher ranking
- Federated learning for teacher enhancement
"""

import asyncio
import json
import time
import uuid
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import random
import hashlib
from statistics import mean, stdev
from pathlib import Path

import structlog
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.dialects.postgresql import insert

from prsm.compute.teachers.seal import SEALService, SEALConfig
from prsm.compute.teachers.rlt.quality_monitor import QualityMetrics, QualityMonitor
from prsm.core.monitoring.rlt_performance_monitor import RLTPerformanceMonitor, RLTMetrics
from ..benchmarking.rlt_evaluation_benchmark import EvaluationProblem, TeachingEvaluationResult

logger = structlog.get_logger(__name__)


class TeacherNodeStatus(Enum):
    """Status of a teacher node in the network"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class NetworkMessageType(Enum):
    """Types of network messages"""
    DISCOVERY_REQUEST = "discovery_request"
    DISCOVERY_RESPONSE = "discovery_response"
    QUALITY_METRICS_UPDATE = "quality_metrics_update"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    TEACHER_REGISTRATION = "teacher_registration"
    HEARTBEAT = "heartbeat"
    CONSENSUS_PROPOSAL = "consensus_proposal"
    CONSENSUS_VOTE = "consensus_vote"


@dataclass
class TeacherNodeInfo:
    """Information about a teacher node in the network"""
    node_id: str
    node_address: str
    node_port: int
    teacher_id: str
    teacher_type: str
    model_size: str
    specializations: List[str]
    quality_score: float
    availability_score: float
    trust_score: float
    last_seen: datetime
    status: TeacherNodeStatus
    capabilities: Dict[str, Any]
    performance_metrics: Dict[str, float]
    reputation_history: List[float] = field(default_factory=list)


@dataclass
class NetworkQualityMetrics:
    """Quality metrics shared across the network"""
    node_id: str
    teacher_id: str
    timestamp: datetime
    domain: str
    
    # Teaching effectiveness metrics
    explanation_quality: float
    student_improvement: float
    comprehension_score: float
    engagement_level: float
    
    # Performance metrics
    response_time: float
    availability: float
    success_rate: float
    error_rate: float
    
    # Collaborative metrics
    collaboration_success: float
    consensus_participation: float
    knowledge_sharing: float
    
    # Validation metrics
    peer_validation_score: float
    self_assessment_accuracy: float
    improvement_trajectory: float


@dataclass
class CollaborationRequest:
    """Request for teacher collaboration"""
    request_id: str
    requesting_node: str
    problem_domain: str
    difficulty_level: float
    required_specializations: List[str]
    quality_threshold: float
    max_collaborators: int
    collaboration_type: str  # "parallel", "sequential", "consensus"
    deadline: datetime
    problem_context: Dict[str, Any]


@dataclass
class CollaborationSession:
    """Active collaboration session between teachers"""
    session_id: str
    participants: List[str]
    coordinator: str
    problem: EvaluationProblem
    collaboration_type: str
    start_time: datetime
    status: str  # "active", "completed", "failed"
    individual_responses: Dict[str, Any]
    consensus_response: Optional[Dict[str, Any]]
    quality_scores: Dict[str, float]
    session_metrics: Dict[str, float]


@dataclass
class TeacherDiscoveryQuery:
    """Query for discovering suitable teachers"""
    query_id: str
    requesting_node: str
    domain: str
    required_quality: float
    required_specializations: List[str]
    max_response_time: float
    preferred_model_sizes: List[str]
    exclude_nodes: Set[str]
    query_timestamp: datetime


@dataclass
class NetworkConsensus:
    """Network consensus for teacher rankings and decisions"""
    consensus_id: str
    proposal_type: str  # "teacher_ranking", "quality_threshold", "collaboration_protocol"
    proposal_data: Dict[str, Any]
    proposer: str
    timestamp: datetime
    votes: Dict[str, str]  # node_id -> vote ("approve", "reject", "abstain")
    consensus_reached: bool
    final_decision: Optional[Dict[str, Any]]


class DistributedRLTNetwork:
    """
    Distributed Network of RLT Teachers
    
    Manages a federated network of RLT teachers enabling discovery, collaboration,
    and distributed teaching optimization across multiple nodes and domains.
    """
    
    def __init__(
        self,
        node_id: str,
        local_teacher: SEALService,
        network_address: str = "localhost",
        network_port: int = 8000,
        max_network_size: int = 100,
        quality_threshold: float = 0.7,
        trust_threshold: float = 0.8,
        database_url: Optional[str] = None,
        max_collaborations: int = 10
    ):
        self.node_id = node_id
        self.local_teacher = local_teacher
        self.network_address = network_address
        self.network_port = network_port
        self.max_network_size = max_network_size
        self.quality_threshold = quality_threshold
        self.trust_threshold = trust_threshold
        self.max_collaborations = max_collaborations

        # Database session factory for persistence
        self._database_url = database_url
        self._session_factory: Optional[async_sessionmaker] = None
        self._db_initialized = False

        # Network state
        self.peers: Dict[str, Dict[str, Any]] = {}  # peer_id -> peer info dict
        self.quality_scores: Dict[str, float] = {}  # peer_id -> quality score
        self.known_teachers: Dict[str, TeacherNodeInfo] = {}
        self.active_collaborations: Dict[str, CollaborationSession] = {}
        self.pending_requests: Dict[str, CollaborationRequest] = {}
        self.collaboration_requests: Dict[str, Dict[str, Any]] = {}  # request_id -> request dict
        self.network_metrics: Dict[str, NetworkQualityMetrics] = {}
        self.consensus_proposals: Dict[str, Dict[str, Any]] = {}  # proposal_id -> proposal dict
        self.registered_teachers: Dict[str, Dict[str, Any]] = {}  # teacher_id -> record dict

        # Discovery and routing
        self.discovery_cache: Dict[str, List[TeacherNodeInfo]] = {}
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        self.load_balancer = NetworkLoadBalancer()
        
        # Quality and reputation tracking
        self.quality_monitor = QualityMonitor()
        self.reputation_tracker = ReputationTracker()
        self.performance_monitor = RLTPerformanceMonitor()
        
        # Network protocols
        self.message_handlers = self._initialize_message_handlers()
        self.consensus_manager = ConsensusManager(self)
        self.collaboration_coordinator = CollaborationCoordinator(self)
        
        # Network statistics
        self.network_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "collaborations_completed": 0,
            "discoveries_performed": 0,
            "consensus_reached": 0,
            "quality_updates_shared": 0
        }
        
        # Start network services
        self.is_running = False
        self.heartbeat_interval = 30.0  # seconds
        self.cleanup_interval = 300.0   # seconds
        
        logger.info(f"Initialized distributed RLT network node: {node_id}")

    async def initialize(self) -> bool:
        """Initialize database connection for persistence."""
        if self._db_initialized:
            return True

        try:
            # Get database URL from settings if not provided
            if self._database_url is None:
                try:
                    from prsm.core.config import get_settings
                    settings = get_settings()
                    self._database_url = getattr(settings, 'database_url', 'sqlite:///~/.prsm/federation.db')
                except Exception:
                    self._database_url = 'sqlite:///~/.prsm/federation.db'

            # Convert sync URL to async
            db_url = self._database_url
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
            elif db_url.startswith("sqlite:///"):
                # Expand path for SQLite
                path = db_url.replace("sqlite:///", "")
                path = str(Path(path).expanduser())
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                db_url = f"sqlite+aiosqlite:///{path}"

            # Create async engine and session factory
            engine = create_async_engine(db_url, echo=False)
            self._session_factory = async_sessionmaker(engine, expire_on_commit=False)
            self._db_initialized = True

            logger.info("federation_database_initialized", database_url=self._database_url.split("://")[0])
            return True

        except Exception as e:
            logger.error("federation_database_init_failed", error=str(e))
            return False

    def _get_session(self) -> Optional[async_sessionmaker]:
        """Get database session factory."""
        return self._session_factory

    async def start_network(self) -> None:
        """Start the distributed network services"""
        if self.is_running:
            logger.warning("Network is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting distributed RLT network on {self.network_address}:{self.network_port}")
        
        # Register local teacher
        await self._register_local_teacher()
        
        # Start network services
        tasks = [
            self._heartbeat_service(),
            self._discovery_service(),
            self._quality_metrics_service(),
            self._collaboration_service(),
            self._consensus_service(),
            self._cleanup_service()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Network service error: {e}")
            await self.stop_network()
    
    async def stop_network(self) -> None:
        """Stop the distributed network services"""
        self.is_running = False
        logger.info("Stopping distributed RLT network")
        
        # Gracefully complete active collaborations
        for session_id in list(self.active_collaborations.keys()):
            await self._finalize_collaboration(session_id)
        
        # Notify network of shutdown
        await self._broadcast_message({
            "type": NetworkMessageType.HEARTBEAT.value,
            "node_id": self.node_id,
            "status": TeacherNodeStatus.OFFLINE.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def discover_network_teachers(
        self,
        domain: str,
        quality_threshold: float,
        required_specializations: Optional[List[str]] = None,
        max_teachers: int = 10,
        timeout: float = 5.0
    ) -> List[TeacherNodeInfo]:
        """Discover suitable teachers in the network for a given domain"""
        query_id = str(uuid.uuid4())
        
        logger.info(f"Discovering teachers for domain '{domain}' with quality >= {quality_threshold}")
        
        # Check local cache first
        cache_key = f"{domain}_{quality_threshold}_{required_specializations}"
        if cache_key in self.discovery_cache:
            cached_results = self.discovery_cache[cache_key]
            # Filter out stale entries
            fresh_results = [
                teacher for teacher in cached_results
                if (datetime.now(timezone.utc) - teacher.last_seen).total_seconds() < 300
            ]
            if fresh_results:
                logger.info(f"Found {len(fresh_results)} teachers in cache")
                return fresh_results[:max_teachers]
        
        # Create discovery query
        query = TeacherDiscoveryQuery(
            query_id=query_id,
            requesting_node=self.node_id,
            domain=domain,
            required_quality=quality_threshold,
            required_specializations=required_specializations or [],
            max_response_time=timeout,
            preferred_model_sizes=["7B", "13B", "30B"],
            exclude_nodes={self.node_id},
            query_timestamp=datetime.now(timezone.utc)
        )
        
        # Broadcast discovery request
        discovery_message = {
            "type": NetworkMessageType.DISCOVERY_REQUEST.value,
            "query": asdict(query),
            "sender": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        responses = []
        
        # Collect responses with timeout
        try:
            responses = await asyncio.wait_for(
                self._collect_discovery_responses(query_id, max_teachers),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Discovery timeout after {timeout}s")
        
        # Filter and rank results
        suitable_teachers = self._filter_and_rank_teachers(
            responses, domain, quality_threshold, required_specializations
        )
        
        # Cache results
        self.discovery_cache[cache_key] = suitable_teachers
        self.network_stats["discoveries_performed"] += 1
        
        logger.info(f"Discovered {len(suitable_teachers)} suitable teachers")
        return suitable_teachers[:max_teachers]
    
    async def share_explanation_quality_metrics(
        self,
        teacher_metrics: Dict[str, Any],
        domain: str,
        problem_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Share explanation quality metrics with the network"""
        
        # Create network quality metrics
        network_metrics = NetworkQualityMetrics(
            node_id=self.node_id,
            teacher_id=self.local_teacher.teacher_id if hasattr(self.local_teacher, 'teacher_id') else "local",
            timestamp=datetime.now(timezone.utc),
            domain=domain,
            explanation_quality=teacher_metrics.get("explanation_quality", 0.0),
            student_improvement=teacher_metrics.get("student_improvement", 0.0),
            comprehension_score=teacher_metrics.get("comprehension_score", 0.0),
            engagement_level=teacher_metrics.get("engagement_level", 0.0),
            response_time=teacher_metrics.get("response_time", 0.0),
            availability=teacher_metrics.get("availability", 1.0),
            success_rate=teacher_metrics.get("success_rate", 0.0),
            error_rate=teacher_metrics.get("error_rate", 0.0),
            collaboration_success=teacher_metrics.get("collaboration_success", 0.0),
            consensus_participation=teacher_metrics.get("consensus_participation", 0.0),
            knowledge_sharing=teacher_metrics.get("knowledge_sharing", 0.0),
            peer_validation_score=teacher_metrics.get("peer_validation_score", 0.0),
            self_assessment_accuracy=teacher_metrics.get("self_assessment_accuracy", 0.0),
            improvement_trajectory=teacher_metrics.get("improvement_trajectory", 0.0)
        )
        
        # Store locally
        metrics_key = f"{self.node_id}_{domain}_{network_metrics.timestamp.isoformat()}"
        self.network_metrics[metrics_key] = network_metrics
        
        # Share with network
        metrics_message = {
            "type": NetworkMessageType.QUALITY_METRICS_UPDATE.value,
            "metrics": asdict(network_metrics),
            "context": problem_context or {},
            "sender": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self._broadcast_message(metrics_message)
        self.network_stats["quality_updates_shared"] += 1
        
        logger.info(f"Shared quality metrics for domain '{domain}' with network")
    
    async def coordinate_collaborative_improvement(
        self,
        problem: EvaluationProblem,
        collaboration_type: str = "consensus",
        max_collaborators: int = 5,
        quality_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Coordinate collaborative improvement session with network teachers"""
        
        session_id = str(uuid.uuid4())
        
        logger.info(f"Coordinating collaborative improvement session: {session_id}")
        
        # Discover suitable collaborators
        suitable_teachers = await self.discover_network_teachers(
            domain=problem.domain,
            quality_threshold=quality_threshold,
            required_specializations=[problem.domain],
            max_teachers=max_collaborators
        )
        
        if not suitable_teachers:
            logger.warning("No suitable collaborators found")
            return {"session_id": session_id, "status": "failed", "reason": "no_collaborators"}
        
        # Create collaboration session
        session = CollaborationSession(
            session_id=session_id,
            participants=[teacher.node_id for teacher in suitable_teachers] + [self.node_id],
            coordinator=self.node_id,
            problem=problem,
            collaboration_type=collaboration_type,
            start_time=datetime.now(timezone.utc),
            status="active",
            individual_responses={},
            consensus_response=None,
            quality_scores={},
            session_metrics={}
        )
        
        self.active_collaborations[session_id] = session
        
        # Send collaboration requests
        collaboration_request = CollaborationRequest(
            request_id=session_id,
            requesting_node=self.node_id,
            problem_domain=problem.domain,
            difficulty_level=problem.difficulty,
            required_specializations=[problem.domain],
            quality_threshold=quality_threshold,
            max_collaborators=max_collaborators,
            collaboration_type=collaboration_type,
            deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
            problem_context={"problem": asdict(problem)}
        )
        
        # Collect individual responses
        individual_responses = await self._collect_collaboration_responses(
            session, collaboration_request, timeout=300.0
        )
        
        # Generate consensus response
        consensus_response = await self._generate_consensus_response(
            session, individual_responses, collaboration_type
        )
        
        # Finalize session
        session.status = "completed"
        session.consensus_response = consensus_response
        session.session_metrics = await self._calculate_session_metrics(session)
        
        self.network_stats["collaborations_completed"] += 1
        
        logger.info(f"Completed collaborative improvement session: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "completed",
            "participants": session.participants,
            "individual_responses": individual_responses,
            "consensus_response": consensus_response,
            "quality_improvement": session.session_metrics.get("quality_improvement", 0.0),
            "collaboration_effectiveness": session.session_metrics.get("effectiveness", 0.0)
        }
    
    async def request_teacher_collaboration(
        self,
        target_teachers: List[str],
        problem: EvaluationProblem,
        collaboration_type: str = "parallel"
    ) -> str:
        """Request collaboration with specific teachers"""
        
        request_id = str(uuid.uuid4())
        
        collaboration_request = CollaborationRequest(
            request_id=request_id,
            requesting_node=self.node_id,
            problem_domain=problem.domain,
            difficulty_level=problem.difficulty,
            required_specializations=[problem.domain],
            quality_threshold=self.quality_threshold,
            max_collaborators=len(target_teachers),
            collaboration_type=collaboration_type,
            deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
            problem_context={"problem": asdict(problem)}
        )
        
        self.pending_requests[request_id] = collaboration_request
        
        # Send requests to target teachers
        for teacher_id in target_teachers:
            if teacher_id in self.known_teachers:
                await self._send_collaboration_request(teacher_id, collaboration_request)
        
        logger.info(f"Sent collaboration request {request_id} to {len(target_teachers)} teachers")
        return request_id
    
    async def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        
        # Calculate network health metrics
        active_teachers = len([t for t in self.known_teachers.values() 
                             if t.status == TeacherNodeStatus.ACTIVE])
        
        total_teachers = len(self.known_teachers)
        
        avg_quality = mean([t.quality_score for t in self.known_teachers.values()]) if self.known_teachers else 0.0
        avg_trust = mean([t.trust_score for t in self.known_teachers.values()]) if self.known_teachers else 0.0
        
        # Calculate collaboration metrics
        recent_collaborations = [
            session for session in self.active_collaborations.values()
            if (datetime.now(timezone.utc) - session.start_time).total_seconds() < 3600
        ]
        
        collaboration_success_rate = 0.0
        if recent_collaborations:
            successful = len([s for s in recent_collaborations if s.status == "completed"])
            collaboration_success_rate = successful / len(recent_collaborations)
        
        return {
            "network_health": {
                "total_teachers": total_teachers,
                "active_teachers": active_teachers,
                "network_coverage": active_teachers / max(self.max_network_size, 1),
                "average_quality": avg_quality,
                "average_trust": avg_trust
            },
            "collaboration_metrics": {
                "active_sessions": len(self.active_collaborations),
                "recent_collaborations": len(recent_collaborations),
                "success_rate": collaboration_success_rate,
                "average_participants": mean([len(s.participants) for s in recent_collaborations]) if recent_collaborations else 0
            },
            "network_activity": self.network_stats,
            "performance_metrics": {
                "discovery_cache_hit_rate": self._calculate_cache_hit_rate(),
                "average_response_time": self._calculate_average_response_time(),
                "network_latency": self._calculate_network_latency()
            }
        }
    
    # Private methods for network operations
    
    async def _register_local_teacher(self) -> None:
        """Register the local teacher with the network"""
        
        local_info = TeacherNodeInfo(
            node_id=self.node_id,
            node_address=self.network_address,
            node_port=self.network_port,
            teacher_id=getattr(self.local_teacher, 'teacher_id', 'local'),
            teacher_type="SEAL-RLT",
            model_size="7B",  # Default, should be configurable
            specializations=["mathematics", "physics", "chemistry"],  # Should be from config
            quality_score=0.85,  # Initial score, will be updated
            availability_score=1.0,
            trust_score=1.0,
            last_seen=datetime.now(timezone.utc),
            status=TeacherNodeStatus.ACTIVE,
            capabilities={
                "dense_reward_training": True,
                "zero_shot_transfer": True,
                "collaborative_teaching": True,
                "real_time_adaptation": True
            },
            performance_metrics={
                "average_response_time": 150.0,  # ms
                "success_rate": 0.92,
                "availability": 0.98
            }
        )
        
        self.known_teachers[self.node_id] = local_info
        
        # Broadcast registration to network
        registration_message = {
            "type": NetworkMessageType.TEACHER_REGISTRATION.value,
            "teacher_info": asdict(local_info),
            "sender": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self._broadcast_message(registration_message)
        logger.info("Registered local teacher with network")
    
    async def _heartbeat_service(self) -> None:
        """Network heartbeat service"""
        while self.is_running:
            try:
                # Send heartbeat
                heartbeat_message = {
                    "type": NetworkMessageType.HEARTBEAT.value,
                    "node_id": self.node_id,
                    "status": TeacherNodeStatus.ACTIVE.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "load": self._calculate_current_load(),
                    "quality_score": self.known_teachers[self.node_id].quality_score
                }
                
                await self._broadcast_message(heartbeat_message)
                
                # Update local teacher info
                if self.node_id in self.known_teachers:
                    self.known_teachers[self.node_id].last_seen = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat service error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _discovery_service(self) -> None:
        """Network discovery service"""
        while self.is_running:
            try:
                # Process discovery requests and responses
                await self._process_pending_discoveries()
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Discovery service error: {e}")
                await asyncio.sleep(1.0)
    
    async def _quality_metrics_service(self) -> None:
        """Quality metrics sharing service"""
        while self.is_running:
            try:
                # Update local quality metrics
                await self._update_local_quality_metrics()
                
                # Process incoming quality updates
                await self._process_quality_updates()
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Quality metrics service error: {e}")
                await asyncio.sleep(10.0)
    
    async def _collaboration_service(self) -> None:
        """Collaboration coordination service"""
        while self.is_running:
            try:
                # Process collaboration requests and responses
                await self._process_collaboration_requests()
                await self._monitor_active_collaborations()
                
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Collaboration service error: {e}")
                await asyncio.sleep(2.0)
    
    async def _consensus_service(self) -> None:
        """Network consensus service"""
        while self.is_running:
            try:
                # Process consensus proposals and votes
                await self._process_consensus_proposals()
                await self._check_consensus_completion()
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Consensus service error: {e}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_service(self) -> None:
        """Network cleanup service"""
        while self.is_running:
            try:
                # Clean up stale entries
                await self._cleanup_stale_teachers()
                await self._cleanup_old_collaborations()
                await self._cleanup_discovery_cache()
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup service error: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    def _initialize_message_handlers(self) -> Dict[str, callable]:
        """Initialize message handlers for different message types"""
        return {
            NetworkMessageType.DISCOVERY_REQUEST.value: self._handle_discovery_request,
            NetworkMessageType.DISCOVERY_RESPONSE.value: self._handle_discovery_response,
            NetworkMessageType.QUALITY_METRICS_UPDATE.value: self._handle_quality_metrics_update,
            NetworkMessageType.COLLABORATION_REQUEST.value: self._handle_collaboration_request,
            NetworkMessageType.COLLABORATION_RESPONSE.value: self._handle_collaboration_response,
            NetworkMessageType.TEACHER_REGISTRATION.value: self._handle_teacher_registration,
            NetworkMessageType.HEARTBEAT.value: self._handle_heartbeat,
            NetworkMessageType.CONSENSUS_PROPOSAL.value: self._handle_consensus_proposal,
            NetworkMessageType.CONSENSUS_VOTE.value: self._handle_consensus_vote
        }
    
    async def _broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all known teachers"""
        # Mock implementation - in real system would use network protocols
        self.network_stats["messages_sent"] += 1
        
        # Simulate network delay
        await asyncio.sleep(random.uniform(0.001, 0.01))
        
        # In a real implementation, this would send the message over the network
        # For testing, we'll just log it
        logger.debug(f"Broadcasting message: {message['type']}")
    
    async def _send_message(self, target_node: str, message: Dict[str, Any]) -> None:
        """Send message to specific node"""
        # Mock implementation
        self.network_stats["messages_sent"] += 1
        await asyncio.sleep(random.uniform(0.001, 0.005))
        logger.debug(f"Sending {message['type']} to {target_node}")
    
    def _calculate_current_load(self) -> float:
        """Calculate current node load"""
        active_collaborations = len(self.active_collaborations)
        pending_requests = len(self.pending_requests)
        
        # Simple load calculation
        load = (active_collaborations * 0.3 + pending_requests * 0.1)
        return min(1.0, max(0.0, load))
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate discovery cache hit rate"""
        # Mock implementation
        return random.uniform(0.7, 0.9)
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average network response time"""
        # Mock implementation
        return random.uniform(100, 300)  # ms
    
    def _calculate_network_latency(self) -> float:
        """Calculate network latency"""
        # Mock implementation
        return random.uniform(10, 50)  # ms
    
    # Mock method implementations for testing
    
    async def _collect_discovery_responses(self, query_id: str, max_teachers: int) -> List[TeacherNodeInfo]:
        """Collect discovery responses (mock implementation)"""
        # Simulate network responses
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Generate mock responses
        responses = []
        num_responses = min(max_teachers, random.randint(1, 5))
        
        for i in range(num_responses):
            teacher_info = TeacherNodeInfo(
                node_id=f"teacher_{uuid.uuid4().hex[:8]}",
                node_address=f"192.168.1.{random.randint(10, 200)}",
                node_port=random.randint(8000, 9000),
                teacher_id=f"rlt_teacher_{i}",
                teacher_type="SEAL-RLT",
                model_size=random.choice(["7B", "13B", "30B"]),
                specializations=random.sample(["mathematics", "physics", "chemistry", "biology"], 2),
                quality_score=random.uniform(0.7, 0.95),
                availability_score=random.uniform(0.8, 1.0),
                trust_score=random.uniform(0.75, 0.95),
                last_seen=datetime.now(timezone.utc),
                status=TeacherNodeStatus.ACTIVE,
                capabilities={
                    "dense_reward_training": True,
                    "zero_shot_transfer": random.choice([True, False]),
                    "collaborative_teaching": True
                },
                performance_metrics={
                    "average_response_time": random.uniform(100, 300),
                    "success_rate": random.uniform(0.85, 0.98),
                    "availability": random.uniform(0.9, 1.0)
                }
            )
            responses.append(teacher_info)
            
            # Add to known teachers
            self.known_teachers[teacher_info.node_id] = teacher_info
        
        return responses
    
    def _filter_and_rank_teachers(
        self,
        teachers: List[TeacherNodeInfo],
        domain: str,
        quality_threshold: float,
        required_specializations: Optional[List[str]]
    ) -> List[TeacherNodeInfo]:
        """Filter and rank teachers based on criteria"""
        
        # Filter by quality threshold
        qualified = [t for t in teachers if t.quality_score >= quality_threshold]
        
        # Filter by specializations
        if required_specializations:
            qualified = [
                t for t in qualified
                if any(spec in t.specializations for spec in required_specializations)
            ]
        
        # Rank by composite score
        def calculate_rank_score(teacher: TeacherNodeInfo) -> float:
            domain_match = 1.0 if domain in teacher.specializations else 0.5
            return (
                teacher.quality_score * 0.4 +
                teacher.trust_score * 0.3 +
                teacher.availability_score * 0.2 +
                domain_match * 0.1
            )
        
        qualified.sort(key=calculate_rank_score, reverse=True)
        return qualified
    
    async def _collect_collaboration_responses(
        self,
        session: CollaborationSession,
        request: CollaborationRequest,
        timeout: float
    ) -> Dict[str, Any]:
        """Collect collaboration responses (mock implementation)"""
        
        responses = {}
        
        for participant in session.participants:
            if participant == self.node_id:
                # Generate local response
                local_response = await self._generate_local_response(session.problem)
                responses[participant] = local_response
            else:
                # Simulate remote response
                await asyncio.sleep(random.uniform(0.1, 0.3))
                remote_response = self._generate_mock_response(session.problem, participant)
                responses[participant] = remote_response
        
        return responses
    
    async def _generate_local_response(self, problem: EvaluationProblem) -> Dict[str, Any]:
        """Generate response using local teacher"""
        
        # Use local RLT teacher to generate explanation
        try:
            if hasattr(self.local_teacher, 'generate_rlt_explanation'):
                result = await self.local_teacher.generate_rlt_explanation(
                    problem.question,
                    problem.correct_answer,
                    student_model=None
                )
            else:
                # Fallback to basic explanation
                result = {
                    "explanation": f"Local explanation for: {problem.question[:50]}...",
                    "quality_score": 0.85,
                    "confidence": 0.9
                }
            
            return {
                "node_id": self.node_id,
                "explanation": result.get("explanation", ""),
                "quality_score": result.get("quality_score", 0.8),
                "confidence": result.get("confidence", 0.8),
                "response_time": random.uniform(100, 200),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating local response: {e}")
            return {
                "node_id": self.node_id,
                "explanation": "Error generating explanation",
                "quality_score": 0.0,
                "confidence": 0.0,
                "response_time": 0.0,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _generate_mock_response(self, problem: EvaluationProblem, node_id: str) -> Dict[str, Any]:
        """Generate mock response from remote teacher"""
        
        return {
            "node_id": node_id,
            "explanation": f"Remote explanation from {node_id} for: {problem.question[:30]}...",
            "quality_score": random.uniform(0.7, 0.95),
            "confidence": random.uniform(0.75, 0.95),
            "response_time": random.uniform(150, 400),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _generate_consensus_response(
        self,
        session: CollaborationSession,
        responses: Dict[str, Any],
        collaboration_type: str
    ) -> Dict[str, Any]:
        """Generate consensus response from individual responses"""
        
        if not responses:
            return {"error": "No responses to generate consensus from"}
        
        if collaboration_type == "consensus":
            # Weighted average based on quality scores
            total_weight = sum(r.get("quality_score", 0) for r in responses.values())
            
            if total_weight == 0:
                return {"error": "No valid responses for consensus"}
            
            # Simple consensus - combine explanations and average scores
            explanations = [r.get("explanation", "") for r in responses.values()]
            quality_scores = [r.get("quality_score", 0) for r in responses.values()]
            confidences = [r.get("confidence", 0) for r in responses.values()]
            
            consensus = {
                "type": "consensus",
                "explanations": explanations,
                "consensus_explanation": f"Consensus from {len(responses)} teachers: " + "; ".join(explanations[:2]),
                "average_quality": mean(quality_scores),
                "average_confidence": mean(confidences),
                "participant_count": len(responses),
                "consensus_strength": min(quality_scores) / max(quality_scores) if max(quality_scores) > 0 else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        elif collaboration_type == "parallel":
            # Best response selection
            best_response = max(responses.values(), key=lambda r: r.get("quality_score", 0))
            consensus = {
                "type": "best_selection",
                "selected_response": best_response,
                "all_responses": responses,
                "selection_criteria": "highest_quality",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        else:  # sequential
            # Sequential improvement
            consensus = {
                "type": "sequential",
                "final_response": list(responses.values())[-1],  # Last response
                "improvement_chain": list(responses.values()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return consensus
    
    async def _calculate_session_metrics(self, session: CollaborationSession) -> Dict[str, float]:
        """Calculate metrics for completed collaboration session"""
        
        if not session.individual_responses:
            return {}
        
        quality_scores = [r.get("quality_score", 0) for r in session.individual_responses.values()]
        response_times = [r.get("response_time", 0) for r in session.individual_responses.values()]
        
        # Calculate improvement from collaboration
        individual_avg = mean(quality_scores) if quality_scores else 0
        consensus_quality = session.consensus_response.get("average_quality", 0) if session.consensus_response else 0
        
        return {
            "quality_improvement": max(0, consensus_quality - individual_avg),
            "effectiveness": consensus_quality,
            "efficiency": 1.0 / (mean(response_times) / 1000) if response_times else 0,  # responses per second
            "participant_contribution": len(session.participants) / len(session.individual_responses) if session.individual_responses else 0,
            "consensus_strength": consensus_quality / max(quality_scores) if quality_scores and max(quality_scores) > 0 else 0
        }
    
    # Additional mock method implementations

    async def _process_pending_discoveries(self) -> None:
        """Process pending discovery requests - discover stale peers and mark inactive."""
        try:
            current_time = time.time()
            stale_threshold = 300  # 5 minutes
            inactive_threshold = 3600  # 1 hour

            # Query database for stale peers
            if self._session_factory:
                try:
                    from prsm.core.database import FederationPeerModel
                    async with self._session_factory() as session:
                        # Find peers that need discovery (stale but active)
                        stmt = select(FederationPeerModel).where(
                            FederationPeerModel.is_active == True,
                            FederationPeerModel.last_seen < current_time - stale_threshold
                        )
                        result = await session.execute(stmt)
                        stale_peers = result.scalars().all()

                        # Send discovery request to stale peers
                        for peer in stale_peers:
                            if peer.peer_id != self.node_id:
                                discovery_message = {
                                    "type": NetworkMessageType.DISCOVERY_REQUEST.value,
                                    "sender_id": self.node_id,
                                    "sender_address": self.network_address,
                                    "sender_port": self.network_port,
                                    "capabilities": {"node_type": "rlt_teacher"},
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                }
                                await self._send_message(peer.peer_id, discovery_message)
                                logger.debug("discovery_request_sent", peer_id=peer.peer_id)

                        # Mark peers as inactive if very stale
                        stmt = update(FederationPeerModel).where(
                            FederationPeerModel.is_active == True,
                            FederationPeerModel.last_seen < current_time - inactive_threshold
                        ).values(is_active=False)
                        await session.execute(stmt)
                        await session.commit()

                except Exception as db_error:
                    logger.error("process_pending_discoveries_db_failed", error=str(db_error))

        except Exception as e:
            logger.error("process_pending_discoveries_failed", error=str(e))

    async def _update_local_quality_metrics(self) -> None:
        """Update local teacher quality metrics"""
        # Mock implementation - update local teacher's quality score
        if self.node_id in self.known_teachers:
            change = random.uniform(-0.05, 0.05)
            new_quality = max(0.5, min(1.0, current_quality + change))
            self.known_teachers[self.node_id].quality_score = new_quality
    
    async def _process_quality_updates(self) -> None:
        """Process incoming quality metric updates - broadcast significant changes."""
        try:
            # Calculate uptime ratio and broadcast if quality changed significantly
            for peer_id, peer_info in list(self.peers.items()):
                if peer_id == self.node_id:
                    continue

                old_score = self.quality_scores.get(peer_id, 0.5)

                # Check if quality score changed by more than 0.05
                if abs(old_score - peer_info.get("quality_score", old_score)) > 0.05:
                    # Broadcast quality update gossip
                    message = {
                        "type": NetworkMessageType.QUALITY_METRICS_UPDATE.value,
                        "sender_id": self.node_id,
                        "peer_id": peer_id,
                        "quality_score": peer_info.get("quality_score", old_score),
                        "metrics": {"uptime_ratio": 1.0 if peer_info.get("is_active", True) else 0.0},
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await self._broadcast_message(message)

            # Prune quality scores for peers no longer in peers dict
            peers_to_prune = [pid for pid in self.quality_scores if pid not in self.peers]
            for pid in peers_to_prune:
                del self.quality_scores[pid]
                logger.debug("quality_score_pruned", peer_id=pid)

        except Exception as e:
            logger.error("process_quality_updates_failed", error=str(e))

    async def _process_collaboration_requests(self) -> None:
        """Process incoming collaboration requests - retry or timeout stale requests."""
        try:
            current_time = time.time()
            timeout_threshold = 300  # 5 minutes

            # Iterate collaboration requests
            requests_to_remove = []
            for request_id, request in list(self.collaboration_requests.items()):
                received_at = request.get("received_at", current_time)

                # Check if request is timed out
                if current_time - received_at > timeout_threshold:
                    if not request.get("accepted"):
                        # Mark as timed out
                        request["status"] = "timed_out"
                        logger.info("collaboration_request_timed_out", request_id=request_id)
                    requests_to_remove.append(request_id)

            # Clean up completed/timed-out requests
            for request_id in requests_to_remove:
                del self.collaboration_requests[request_id]

        except Exception as e:
            logger.error("process_collaboration_requests_failed", error=str(e))
    
    async def _monitor_active_collaborations(self) -> None:
        """Monitor active collaboration sessions"""
        # Check for timeouts and cleanup
        current_time = datetime.now(timezone.utc)
        
        for session_id, session in list(self.active_collaborations.items()):
            if (current_time - session.start_time).total_seconds() > 600:  # 10 minutes timeout
                session.status = "timeout"
                await self._finalize_collaboration(session_id)
    
    async def _finalize_collaboration(self, session_id: str) -> None:
        """Finalize and cleanup collaboration session"""
        if session_id in self.active_collaborations:
            session = self.active_collaborations[session_id]
            logger.info(f"Finalizing collaboration session: {session_id}")
            
            # Calculate final metrics if not already done
            if not session.session_metrics:
                session.session_metrics = await self._calculate_session_metrics(session)
            
            # Archive session (in real implementation, might save to database)
            del self.active_collaborations[session_id]
    
    async def _process_consensus_proposals(self) -> None:
        """Process consensus proposals - finalize proposals past deadline."""
        try:
            current_time = time.time()

            # Iterate consensus proposals
            for proposal_id, proposal in list(self.consensus_proposals.items()):
                if proposal.get("status") != "open":
                    continue

                deadline = proposal.get("deadline")
                if deadline and current_time > float(deadline):
                    # Force finalization
                    await self._check_consensus_completion(proposal_id)

        except Exception as e:
            logger.error("process_consensus_proposals_failed", error=str(e))

    async def _check_consensus_completion_standalone(self) -> None:
        """Check if any consensus proposals have reached completion - standalone version for service loop."""
        try:
            for proposal_id in list(self.consensus_proposals.keys()):
                await self._check_consensus_completion(proposal_id)
        except Exception as e:
            logger.error("check_consensus_completion_standalone_failed", error=str(e))
    
    async def _cleanup_stale_teachers(self) -> None:
        """Remove stale teacher entries"""
        current_time = datetime.now(timezone.utc)
        stale_threshold = timedelta(minutes=5)
        
        stale_nodes = [
            node_id for node_id, teacher in self.known_teachers.items()
            if node_id != self.node_id and (current_time - teacher.last_seen) > stale_threshold
        ]
        
        for node_id in stale_nodes:
            del self.known_teachers[node_id]
            logger.info(f"Removed stale teacher: {node_id}")
    
    async def _cleanup_old_collaborations(self) -> None:
        """Clean up old collaboration data"""
        # Clean up old entries in pending requests
        current_time = datetime.now(timezone.utc)
        
        expired_requests = [
            req_id for req_id, req in self.pending_requests.items()
            if current_time > req.deadline
        ]
        
        for req_id in expired_requests:
            del self.pending_requests[req_id]
    
    async def _cleanup_discovery_cache(self) -> None:
        """Clean up old discovery cache entries"""
        # Clear cache entries older than 10 minutes
        self.discovery_cache.clear()  # Simple cleanup for mock implementation
    
    async def _send_collaboration_request(self, teacher_id: str, request: CollaborationRequest) -> None:
        """Send collaboration request to specific teacher"""
        message = {
            "type": NetworkMessageType.COLLABORATION_REQUEST.value,
            "request": asdict(request),
            "sender": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self._send_message(teacher_id, message)
    
    # Message handlers (mock implementations)

    async def _handle_discovery_request(self, message: Dict[str, Any]) -> None:
        """Handle discovery request message - store peer and respond with own info."""
        try:
            sender_id = message.get("sender_id") or message.get("sender")
            sender_address = message.get("sender_address") or message.get("address", self.network_address)
            sender_port = message.get("sender_port") or message.get("port", 8000)
            capabilities = message.get("capabilities", {})

            if not sender_id:
                logger.warning("discovery_request_missing_sender", message=message)
                return

            current_time = time.time()

            # Upsert into in-memory peers dict
            self.peers[sender_id] = {
                "address": sender_address,
                "port": sender_port,
                "capabilities": capabilities,
                "last_seen": current_time,
                "is_active": True
            }

            # Initialize quality score if new
            if sender_id not in self.quality_scores:
                self.quality_scores[sender_id] = 0.5

            # Persist to database
            if self._session_factory:
                try:
                    from prsm.core.database import FederationPeerModel
                    async with self._session_factory() as session:
                        # Upsert peer
                        stmt = insert(FederationPeerModel).values(
                            peer_id=sender_id,
                            address=sender_address,
                            port=sender_port,
                            node_type=capabilities.get("node_type", "standard"),
                            last_seen=current_time,
                            quality_score=self.quality_scores[sender_id],
                            capabilities=capabilities,
                            is_active=True,
                            created_at=current_time
                        )
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['peer_id'],
                            set_={
                                'address': sender_address,
                                'port': sender_port,
                                'last_seen': current_time,
                                'capabilities': capabilities,
                                'is_active': True
                            }
                        )
                        await session.execute(stmt)
                        await session.commit()
                except Exception as db_error:
                    logger.error("discovery_db_persist_failed", error=str(db_error))

            # Build discovery response
            response_payload = {
                "sender_id": self.node_id,
                "address": self.network_address,
                "port": self.network_port,
                "capabilities": {
                    "node_type": "rlt_teacher",
                    "specializations": []
                }
            }

            # Send response back
            await self._send_message(sender_id, {
                "type": NetworkMessageType.DISCOVERY_RESPONSE.value,
                "payload": response_payload,
                "sender_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            logger.info("peer_discovered", peer_id=sender_id, address=sender_address)

        except Exception as e:
            logger.error("handle_discovery_request_failed", error=str(e))

    async def _handle_discovery_response(self, message: Dict[str, Any]) -> None:
        """Handle discovery response message - store peer info and initialize quality score."""
        try:
            sender_id = message.get("sender_id")
            payload = message.get("payload", message)
            address = payload.get("address")
            port = payload.get("port")
            capabilities = payload.get("capabilities", {})

            if not sender_id:
                logger.warning("discovery_response_missing_sender", message=message)
                return

            current_time = time.time()

            # Upsert into in-memory peers dict
            self.peers[sender_id] = {
                "address": address,
                "port": port,
                "capabilities": capabilities,
                "last_seen": current_time,
                "is_active": True
            }

            # Initialize quality score to 0.5 if not already present
            if sender_id not in self.quality_scores:
                self.quality_scores[sender_id] = 0.5

            # Persist to database
            if self._session_factory:
                try:
                    from prsm.core.database import FederationPeerModel
                    async with self._session_factory() as session:
                        stmt = insert(FederationPeerModel).values(
                            peer_id=sender_id,
                            address=address or "unknown",
                            port=port or 8000,
                            node_type=capabilities.get("node_type", "standard"),
                            last_seen=current_time,
                            quality_score=self.quality_scores[sender_id],
                            capabilities=capabilities,
                            is_active=True,
                            created_at=current_time
                        )
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['peer_id'],
                            set_={
                                'address': address or "unknown",
                                'port': port or 8000,
                                'last_seen': current_time,
                                'capabilities': capabilities,
                                'is_active': True
                            }
                        )
                        await session.execute(stmt)
                        await session.commit()
                except Exception as db_error:
                    logger.error("discovery_response_db_persist_failed", error=str(db_error))

            logger.info("peer_discovered", peer_id=sender_id, address=address)

        except Exception as e:
            logger.error("handle_discovery_response_failed", error=str(e))

    async def _handle_quality_metrics_update(self, message: Dict[str, Any]) -> None:
        """Handle quality metrics update message - validate and store quality score."""
        try:
            sender_id = message.get("sender_id")
            quality_score = message.get("quality_score")
            metrics = message.get("metrics", {})

            if not sender_id:
                logger.warning("quality_update_missing_sender", message=message)
                return

            # Validate quality score
            if not isinstance(quality_score, (int, float)):
                logger.warning("quality_score_invalid_type", sender_id=sender_id, quality_score=quality_score)
                return

            # Clamp to valid range [0.0, 1.0] and check for finite values
            quality_score = float(quality_score)
            if not math.isfinite(quality_score):
                logger.warning("quality_score_not_finite", sender_id=sender_id, quality_score=quality_score)
                return
            quality_score = max(0.0, min(1.0, quality_score))

            current_time = time.time()

            # Update in-memory quality score
            self.quality_scores[sender_id] = quality_score

            # Update peers if exists
            if sender_id in self.peers:
                self.peers[sender_id]["last_seen"] = current_time
                self.peers[sender_id]["quality_score"] = quality_score

            # Persist to database
            if self._session_factory:
                try:
                    from prsm.core.database import FederationPeerModel, FederationMessageModel
                    async with self._session_factory() as session:
                        # Update peer quality score
                        stmt = update(FederationPeerModel).where(
                            FederationPeerModel.peer_id == sender_id
                        ).values(
                            quality_score=quality_score,
                            last_seen=current_time
                        )
                        await session.execute(stmt)

                        # Store message record
                        message_id = str(uuid.uuid4())
                        session.add(FederationMessageModel(
                            message_id=message_id,
                            message_type="quality_metrics_update",
                            sender_id=sender_id,
                            payload={"quality_score": quality_score, "metrics": metrics},
                            sent_at=current_time,
                            received_at=current_time,
                            processed_at=current_time,
                            status="processed"
                        ))
                        await session.commit()
                except Exception as db_error:
                    logger.error("quality_update_db_persist_failed", error=str(db_error))

            logger.info("quality_score_updated", peer_id=sender_id, quality_score=quality_score)

        except Exception as e:
            logger.error("handle_quality_metrics_update_failed", error=str(e))

    async def _handle_collaboration_request(self, message: Dict[str, Any]) -> None:
        """Handle collaboration request - accept if capacity allows, otherwise reject."""
        try:
            request_id = message.get("request_id")
            requester_id = message.get("requester_id") or message.get("sender_id")
            task_type = message.get("task_type")
            task_payload = message.get("task_payload", {})
            reward_ftns = message.get("reward_ftns", 0)

            if not request_id or not requester_id:
                logger.warning("collaboration_request_missing_fields", message=message)
                return

            current_time = time.time()

            # Check if we have capacity
            active_count = len(self.active_collaborations)
            at_capacity = active_count >= self.max_collaborations

            if at_capacity:
                # Reject with reason
                response = {
                    "type": NetworkMessageType.COLLABORATION_RESPONSE.value,
                    "request_id": request_id,
                    "accepted": False,
                    "responder_id": self.node_id,
                    "reason": "at_capacity",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await self._send_message(requester_id, response)
                logger.info("collaboration_rejected_at_capacity", request_id=request_id, requester_id=requester_id)
            else:
                # Store request and accept
                self.collaboration_requests[request_id] = {
                    "requester_id": requester_id,
                    "task_type": task_type,
                    "task_payload": task_payload,
                    "reward_ftns": reward_ftns,
                    "received_at": current_time,
                    "accepted": True
                }

                response = {
                    "type": NetworkMessageType.COLLABORATION_RESPONSE.value,
                    "request_id": request_id,
                    "accepted": True,
                    "responder_id": self.node_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await self._send_message(requester_id, response)
                logger.info("collaboration_accepted", request_id=request_id, requester_id=requester_id)

            # Store message in database
            if self._session_factory:
                try:
                    from prsm.core.database import FederationMessageModel
                    async with self._session_factory() as session:
                        message_id = str(uuid.uuid4())
                        session.add(FederationMessageModel(
                            message_id=message_id,
                            message_type="collaboration_request",
                            sender_id=requester_id,
                            payload=message,
                            sent_at=current_time,
                            received_at=current_time,
                            processed_at=current_time,
                            status="processed"
                        ))
                        await session.commit()
                except Exception as db_error:
                    logger.error("collaboration_request_db_persist_failed", error=str(db_error))

        except Exception as e:
            logger.error("handle_collaboration_request_failed", error=str(e))

    async def _handle_collaboration_response(self, message: Dict[str, Any]) -> None:
        """Handle collaboration response - update request status."""
        try:
            request_id = message.get("request_id")
            accepted = message.get("accepted", False)
            responder_id = message.get("responder_id")
            reason = message.get("reason")

            if not request_id:
                logger.warning("collaboration_response_missing_request_id", message=message)
                return

            # Update stored request if exists
            if request_id in self.collaboration_requests:
                self.collaboration_requests[request_id]["accepted"] = accepted
                self.collaboration_requests[request_id]["responder_id"] = responder_id
                self.collaboration_requests[request_id]["responded_at"] = time.time()

                # If accepted, move to active collaborations
                if accepted:
                    self.active_collaborations[request_id] = {
                        "request_id": request_id,
                        "responder_id": responder_id,
                        "started_at": time.time(),
                        "status": "active"
                    }
                    logger.info("collaboration_started", request_id=request_id, responder_id=responder_id)
                else:
                    logger.info("collaboration_rejected", request_id=request_id, responder_id=responder_id, reason=reason)
            else:
                logger.warning("collaboration_response_unknown_request", request_id=request_id)

        except Exception as e:
            logger.error("handle_collaboration_response_failed", error=str(e))

    async def _handle_teacher_registration(self, message: Dict[str, Any]) -> None:
        """Handle teacher registration - store teacher record."""
        try:
            teacher_id = message.get("teacher_id")
            model_type = message.get("model_type")
            capabilities = message.get("capabilities", {})
            min_reward_ftns = message.get("min_reward_ftns", 0)
            sender_id = message.get("sender_id")

            if not teacher_id:
                logger.warning("teacher_registration_missing_teacher_id", message=message)
                return

            current_time = time.time()

            # Build teacher record
            record = {
                "teacher_id": teacher_id,
                "model_type": model_type,
                "capabilities": capabilities,
                "min_reward_ftns": min_reward_ftns,
                "sender_id": sender_id,
                "registered_at": current_time
            }

            # Store in registered_teachers
            self.registered_teachers[teacher_id] = record

            # Update peer as teacher type
            if sender_id and sender_id in self.peers:
                self.peers[sender_id]["node_type"] = "teacher"
                self.peers[sender_id]["capabilities"]["model_id"] = teacher_id

            # Persist to database
            if self._session_factory:
                try:
                    from prsm.core.database import FederationPeerModel
                    async with self._session_factory() as session:
                        stmt = insert(FederationPeerModel).values(
                            peer_id=sender_id or teacher_id,
                            address=self.peers.get(sender_id, {}).get("address", "unknown"),
                            port=self.peers.get(sender_id, {}).get("port", 8000),
                            node_type="teacher",
                            last_seen=current_time,
                            quality_score=self.quality_scores.get(sender_id, 0.5),
                            capabilities={"model_id": teacher_id, "model_type": model_type, **capabilities},
                            is_active=True,
                            created_at=current_time
                        )
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['peer_id'],
                            set_={
                                'node_type': 'teacher',
                                'last_seen': current_time,
                                'capabilities': {"model_id": teacher_id, "model_type": model_type, **capabilities}
                            }
                        )
                        await session.execute(stmt)
                        await session.commit()
                except Exception as db_error:
                    logger.error("teacher_registration_db_persist_failed", error=str(db_error))

            logger.info("teacher_registered", teacher_id=teacher_id, sender_id=sender_id)

        except Exception as e:
            logger.error("handle_teacher_registration_failed", error=str(e))

    async def _handle_heartbeat(self, message: Dict[str, Any]) -> None:
        """Handle heartbeat - update peer last_seen and quality score based on load."""
        try:
            sender_id = message.get("sender_id")
            timestamp = message.get("timestamp")
            load = message.get("load", 0.0)  # 0.0 to 1.0
            active_tasks = message.get("active_tasks", 0)

            if not sender_id:
                logger.warning("heartbeat_missing_sender", message=message)
                return

            # Validate load value
            load = max(0.0, min(1.0, float(load))) if isinstance(load, (int, float)) else 0.0

            current_time = time.time()

            # Update peer last_seen
            if sender_id in self.peers:
                self.peers[sender_id]["last_seen"] = current_time

                # Update quality score using rolling average
                # Lower load = higher quality (more available)
                old_score = self.quality_scores.get(sender_id, 0.5)
                new_score = 0.8 * old_score + 0.2 * (1.0 - load)
                self.quality_scores[sender_id] = max(0.0, min(1.0, new_score))

            # Persist to database
            if self._session_factory:
                try:
                    from prsm.core.database import FederationPeerModel
                    async with self._session_factory() as session:
                        stmt = update(FederationPeerModel).where(
                            FederationPeerModel.peer_id == sender_id
                        ).values(
                            last_seen=current_time,
                            quality_score=self.quality_scores.get(sender_id, 0.5)
                        )
                        await session.execute(stmt)
                        await session.commit()
                except Exception as db_error:
                    logger.error("heartbeat_db_persist_failed", error=str(db_error))

            # Heartbeats are fire-and-forget, no response needed
            logger.debug("heartbeat_received", sender_id=sender_id, load=load)

        except Exception as e:
            logger.error("handle_heartbeat_failed", error=str(e))

    async def _handle_consensus_proposal(self, message: Dict[str, Any]) -> None:
        """Handle consensus proposal - evaluate and vote."""
        try:
            proposal_id = message.get("proposal_id")
            proposer_id = message.get("proposer_id") or message.get("sender_id")
            proposal_type = message.get("proposal_type")
            proposal_data = message.get("proposal_data", {})
            deadline = message.get("deadline")

            if not proposal_id or not proposal_type:
                logger.warning("consensus_proposal_missing_fields", message=message)
                return

            current_time = time.time()

            # Store proposal
            self.consensus_proposals[proposal_id] = {
                "proposer_id": proposer_id,
                "proposal_type": proposal_type,
                "proposal_data": proposal_data,
                "deadline": deadline,
                "votes": {},
                "status": "open",
                "created_at": current_time
            }

            # Evaluate proposal locally
            vote = await self._evaluate_proposal(proposal_type, proposal_data)

            # Send vote back to proposer
            vote_message = {
                "type": NetworkMessageType.CONSENSUS_VOTE.value,
                "proposal_id": proposal_id,
                "vote": vote,
                "voter_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self._send_message(proposer_id, vote_message)

            logger.info("consensus_proposal_received", proposal_id=proposal_id, vote=vote)

        except Exception as e:
            logger.error("handle_consensus_proposal_failed", error=str(e))

    async def _evaluate_proposal(self, proposal_type: str, proposal_data: Dict[str, Any]) -> bool:
        """Evaluate a proposal and return vote (True = approve, False = reject)."""
        try:
            if proposal_type == "parameter_change":
                # Validate that changed parameter is within +/- 50% of current
                parameter_name = proposal_data.get("parameter_name")
                new_value = proposal_data.get("new_value")
                # Default approve if we can't validate
                return True

            elif proposal_type == "peer_removal":
                # Approve if target peer's quality score is below 0.2
                target_peer_id = proposal_data.get("target_peer_id")
                target_score = self.quality_scores.get(target_peer_id, 0.5)
                return target_score < 0.2

            elif proposal_type == "software_update":
                # Trust proposer in early network
                return True

            else:
                # Unknown type - reject
                return False

        except Exception as e:
            logger.error("evaluate_proposal_failed", error=str(e))
            return False

    async def _handle_consensus_vote(self, message: Dict[str, Any]) -> None:
        """Handle consensus vote - record vote and check completion."""
        try:
            proposal_id = message.get("proposal_id")
            vote = message.get("vote", False)
            voter_id = message.get("voter_id")

            if not proposal_id or not voter_id:
                logger.warning("consensus_vote_missing_fields", message=message)
                return

            # Check if proposal exists
            if proposal_id not in self.consensus_proposals:
                logger.warning("consensus_vote_unknown_proposal", proposal_id=proposal_id)
                return

            # Record vote
            self.consensus_proposals[proposal_id]["votes"][voter_id] = vote

            # Check for completion
            await self._check_consensus_completion(proposal_id)

        except Exception as e:
            logger.error("handle_consensus_vote_failed", error=str(e))

    async def _check_consensus_completion(self, proposal_id: str) -> None:
        """Check if consensus has been reached for a proposal."""
        try:
            if proposal_id not in self.consensus_proposals:
                return

            proposal = self.consensus_proposals[proposal_id]
            votes = proposal.get("votes", {})
            deadline = proposal.get("deadline")

            # Count votes
            total_votes = len(votes)
            if total_votes < 3:
                # Need at least 3 votes
                return

            yes_votes = sum(1 for v in votes.values() if v)
            no_votes = total_votes - yes_votes

            # Check supermajority (67%)
            supermajority_reached = (yes_votes / total_votes) >= 0.67

            current_time = time.time()
            deadline_passed = deadline and current_time > float(deadline) if deadline else False

            if supermajority_reached:
                proposal["status"] = "accepted"
                await self._apply_consensus_decision(proposal)
                logger.info("consensus_reached", proposal_id=proposal_id, status="accepted", yes=yes_votes, no=no_votes)
            elif deadline_passed:
                proposal["status"] = "rejected"
                logger.info("consensus_deadline_passed", proposal_id=proposal_id, status="rejected")

        except Exception as e:
            logger.error("check_consensus_completion_failed", error=str(e))

    async def _apply_consensus_decision(self, proposal: Dict[str, Any]) -> None:
        """Apply a consensus decision."""
        try:
            proposal_type = proposal.get("proposal_type")
            proposal_data = proposal.get("proposal_data", {})

            if proposal_type == "parameter_change":
                # Update in-memory config value (would need proper config management)
                parameter_name = proposal_data.get("parameter_name")
                new_value = proposal_data.get("new_value")
                logger.info("consensus_parameter_change_applied", parameter=parameter_name, new_value=new_value)

            elif proposal_type == "peer_removal":
                # Mark peer as inactive
                target_peer_id = proposal_data.get("target_peer_id")
                if target_peer_id in self.peers:
                    self.peers[target_peer_id]["is_active"] = False

                # Update database
                if self._session_factory:
                    try:
                        from prsm.core.database import FederationPeerModel
                        async with self._session_factory() as session:
                            stmt = update(FederationPeerModel).where(
                                FederationPeerModel.peer_id == target_peer_id
                            ).values(is_active=False)
                            await session.execute(stmt)
                            await session.commit()
                    except Exception as db_error:
                        logger.error("peer_removal_db_failed", error=str(db_error))

                logger.info("consensus_peer_removed", peer_id=target_peer_id)

            elif proposal_type == "software_update":
                # Log that update is pending (actual update left to operator)
                logger.info("consensus_software_update_pending", proposal_data=proposal_data)

        except Exception as e:
            logger.error("apply_consensus_decision_failed", error=str(e))


class NetworkLoadBalancer:
    """Load balancer for distributed RLT network"""
    
    def __init__(self):
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def select_optimal_teachers(
        self,
        candidates: List[TeacherNodeInfo],
        required_count: int,
        load_threshold: float = 0.8
    ) -> List[TeacherNodeInfo]:
        """Select optimal teachers based on load balancing"""
        
        # Filter out overloaded teachers
        available = [
            teacher for teacher in candidates
            if self._get_current_load(teacher.node_id) < load_threshold
        ]
        
        # Sort by composite score (quality, availability, load)
        def balance_score(teacher: TeacherNodeInfo) -> float:
            load = self._get_current_load(teacher.node_id)
            return teacher.quality_score * 0.5 + teacher.availability_score * 0.3 + (1 - load) * 0.2
        
        available.sort(key=balance_score, reverse=True)
        return available[:required_count]
    
    def _get_current_load(self, node_id: str) -> float:
        """Get current load for a node"""
        # Mock implementation
        return random.uniform(0.1, 0.7)


class ReputationTracker:
    """Tracks reputation and trust scores for network teachers"""
    
    def __init__(self):
        self.reputation_history: Dict[str, List[float]] = defaultdict(list)
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def update_reputation(self, node_id: str, performance_score: float) -> None:
        """Update reputation based on performance"""
        self.reputation_history[node_id].append(performance_score)
        
        # Calculate trust score based on history
        if len(self.reputation_history[node_id]) > 5:
            recent_scores = self.reputation_history[node_id][-10:]
            avg_score = mean(recent_scores)
            consistency = 1 - stdev(recent_scores) if len(recent_scores) > 1 else 1.0
            
            self.trust_scores[node_id] = (avg_score * 0.7 + consistency * 0.3)
    
    def get_trust_score(self, node_id: str) -> float:
        """Get current trust score for a node"""
        return self.trust_scores.get(node_id, 1.0)


class ConsensusManager:
    """Manages network consensus for decisions and rankings"""
    
    def __init__(self, network: DistributedRLTNetwork):
        self.network = network
    
    async def propose_consensus(
        self,
        proposal_type: str,
        proposal_data: Dict[str, Any]
    ) -> str:
        """Propose a consensus decision to the network"""
        
        consensus_id = str(uuid.uuid4())
        
        consensus = NetworkConsensus(
            consensus_id=consensus_id,
            proposal_type=proposal_type,
            proposal_data=proposal_data,
            proposer=self.network.node_id,
            timestamp=datetime.now(timezone.utc),
            votes={},
            consensus_reached=False,
            final_decision=None
        )
        
        self.network.consensus_proposals[consensus_id] = consensus
        
        # Broadcast proposal
        proposal_message = {
            "type": NetworkMessageType.CONSENSUS_PROPOSAL.value,
            "consensus": asdict(consensus),
            "sender": self.network.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.network._broadcast_message(proposal_message)
        return consensus_id
    
    async def vote_on_proposal(self, consensus_id: str, vote: str) -> None:
        """Vote on a consensus proposal"""
        
        if consensus_id in self.network.consensus_proposals:
            consensus = self.network.consensus_proposals[consensus_id]
            consensus.votes[self.network.node_id] = vote
            
            # Broadcast vote
            vote_message = {
                "type": NetworkMessageType.CONSENSUS_VOTE.value,
                "consensus_id": consensus_id,
                "vote": vote,
                "sender": self.network.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await self.network._broadcast_message(vote_message)


class CollaborationCoordinator:
    """Coordinates complex multi-teacher collaborations"""
    
    def __init__(self, network: DistributedRLTNetwork):
        self.network = network
    
    async def coordinate_sequential_collaboration(
        self,
        teachers: List[str],
        problem: EvaluationProblem
    ) -> Dict[str, Any]:
        """Coordinate sequential collaboration where each teacher builds on the previous"""
        
        results = []
        current_context = {"problem": asdict(problem)}
        
        for i, teacher_id in enumerate(teachers):
            # Get response from current teacher
            if teacher_id == self.network.node_id:
                response = await self.network._generate_local_response(problem)
            else:
                response = self.network._generate_mock_response(problem, teacher_id)
            
            # Add to context for next teacher
            current_context[f"previous_response_{i}"] = response
            results.append(response)
        
        return {
            "type": "sequential",
            "responses": results,
            "final_result": results[-1] if results else None,
            "improvement_chain": [r.get("quality_score", 0) for r in results]
        }
    
    async def coordinate_parallel_collaboration(
        self,
        teachers: List[str],
        problem: EvaluationProblem
    ) -> Dict[str, Any]:
        """Coordinate parallel collaboration where teachers work simultaneously"""
        
        # Collect all responses simultaneously
        tasks = []
        for teacher_id in teachers:
            if teacher_id == self.network.node_id:
                task = self.network._generate_local_response(problem)
            else:
                # Mock parallel response
                async def mock_response(tid=teacher_id):
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    return self.network._generate_mock_response(problem, tid)
                task = mock_response()
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # Find best response
        best_response = max(responses, key=lambda r: r.get("quality_score", 0))
        
        return {
            "type": "parallel",
            "all_responses": responses,
            "best_response": best_response,
            "quality_range": [min(r.get("quality_score", 0) for r in responses),
                             max(r.get("quality_score", 0) for r in responses)]
        }