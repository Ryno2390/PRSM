"""
PRSM Intelligent Request Routing & Failover System
Advanced routing with machine learning, geographic optimization, and intelligent failover
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import logging
import uuid
import hashlib
from collections import defaultdict, deque
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# Machine learning for routing optimization
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

# GeoIP for geographic routing
try:
    import geoip2.database
    import geoip2.errors
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False

# HTTP client for health checks and routing
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    GEOGRAPHIC = "geographic"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


class FailoverMode(Enum):
    """Failover modes"""
    FAST_FAILOVER = "fast"         # Immediate failover on first failure
    GRACEFUL_FAILOVER = "graceful" # Wait for confirmation before failover
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker pattern
    ADAPTIVE = "adaptive"          # Machine learning based failover


class RequestType(Enum):
    """Types of requests for optimized routing"""
    API_CALL = "api_call"
    WEB_PAGE = "web_page"
    STATIC_CONTENT = "static_content"
    WEBSOCKET = "websocket"
    STREAMING = "streaming"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME = "real_time"


@dataclass
class GeographicLocation:
    """Geographic location information"""
    latitude: float
    longitude: float
    country: str
    region: str
    city: str
    continent: str
    timezone: str
    accuracy_radius: int = 0


@dataclass
class RequestContext:
    """Context information for routing decisions"""
    request_id: str
    source_ip: str
    user_agent: str
    path: str
    method: str
    headers: Dict[str, str]
    
    # Geographic information
    geographic_location: Optional[GeographicLocation] = None
    
    # Request characteristics
    request_type: RequestType = RequestType.API_CALL
    expected_response_size: int = 0
    priority: int = 100  # Lower number = higher priority
    timeout_seconds: int = 30
    
    # Session information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Performance requirements
    max_acceptable_latency_ms: int = 1000
    requires_sticky_session: bool = False
    cache_enabled: bool = True
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class NodePerformanceMetrics:
    """Performance metrics for a node"""
    node_id: str
    
    # Current metrics
    current_connections: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    network_utilization_percent: float = 0.0
    
    # Historical metrics (sliding windows)
    response_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    error_counts: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Geographic and network metrics
    geographic_latencies: Dict[str, float] = field(default_factory=dict)  # region -> avg latency
    
    # Health status
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_health_check: Optional[datetime] = None
    
    # Capacity metrics
    max_connections: int = 1000
    current_load_factor: float = 0.0  # 0.0 to 1.0
    
    # Cost metrics
    cost_per_request: float = 0.001
    
    def add_response_time(self, response_time_ms: float):
        """Add response time measurement"""
        self.response_times_ms.append(response_time_ms)
    
    def add_success_rate(self, success_rate: float):
        """Add success rate measurement"""
        self.success_rates.append(success_rate)
    
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times_ms:
            return 0.0
        return sum(self.response_times_ms) / len(self.response_times_ms)
    
    def get_avg_success_rate(self) -> float:
        """Get average success rate"""
        if not self.success_rates:
            return 1.0
        return sum(self.success_rates) / len(self.success_rates)
    
    def calculate_health_score(self) -> float:
        """Calculate composite health score (0.0-1.0)"""
        try:
            # Base health score
            health_score = 1.0 if self.is_healthy else 0.1
            
            # Adjust for resource usage
            resource_penalty = (self.cpu_usage_percent + self.memory_usage_percent) / 200
            health_score -= resource_penalty
            
            # Adjust for response time
            avg_response_time = self.get_avg_response_time()
            if avg_response_time > 500:  # >500ms penalty
                time_penalty = min(0.5, (avg_response_time - 500) / 1000)
                health_score -= time_penalty
            
            # Adjust for success rate
            success_rate = self.get_avg_success_rate()
            health_score *= success_rate
            
            # Adjust for load
            load_penalty = self.current_load_factor * 0.3
            health_score -= load_penalty
            
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating health score for {self.node_id}: {e}")
            return 0.5


@dataclass
class RoutingDecision:
    """Result of a routing decision"""
    target_node_id: str
    strategy_used: RoutingStrategy
    decision_confidence: float  # 0.0-1.0
    expected_response_time_ms: float
    backup_nodes: List[str] = field(default_factory=list)
    
    # Decision metadata
    decision_time_ms: float = 0.0
    geographic_match: bool = False
    load_balanced: bool = True
    
    # Reasoning
    decision_factors: Dict[str, float] = field(default_factory=dict)
    selection_reason: str = ""


class GeoIPService:
    """Geographic IP location service"""
    
    def __init__(self, database_path: Optional[str] = None):
        self.reader = None
        self.enabled = False
        
        if GEOIP_AVAILABLE and database_path:
            try:
                self.reader = geoip2.database.Reader(database_path)
                self.enabled = True
                logger.info("✅ GeoIP service initialized")
            except Exception as e:
                logger.warning(f"GeoIP initialization failed: {e}")
    
    def get_location(self, ip_address: str) -> Optional[GeographicLocation]:
        """Get geographic location for IP address"""
        if not self.enabled or not self.reader:
            return None
        
        try:
            response = self.reader.city(ip_address)
            
            return GeographicLocation(
                latitude=float(response.location.latitude or 0),
                longitude=float(response.location.longitude or 0),
                country=response.country.name or "",
                region=response.subdivisions.most_specific.name or "",
                city=response.city.name or "",
                continent=response.continent.name or "",
                timezone=response.location.time_zone or "",
                accuracy_radius=response.location.accuracy_radius or 0
            )
            
        except geoip2.errors.AddressNotFoundError:
            logger.debug(f"IP address not found in GeoIP database: {ip_address}")
            return None
        except Exception as e:
            logger.error(f"Error looking up IP {ip_address}: {e}")
            return None
    
    def calculate_distance(self, loc1: GeographicLocation, 
                         loc2: GeographicLocation) -> float:
        """Calculate distance between two locations in kilometers"""
        if not loc1 or not loc2:
            return float('inf')
        
        try:
            # Haversine formula
            from math import radians, cos, sin, asin, sqrt
            
            lat1, lon1 = radians(loc1.latitude), radians(loc1.longitude)
            lat2, lon2 = radians(loc2.latitude), radians(loc2.longitude)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Radius of earth in kilometers
            return c * 6371
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')


class MachineLearningRouter:
    """Machine learning based routing optimizer"""
    
    def __init__(self):
        self.enabled = ML_AVAILABLE
        self.model = None
        self.feature_scaler = None
        self.training_data = deque(maxlen=10000)
        self.last_training = None
        
        if self.enabled:
            self.model = LinearRegression()
            logger.info("✅ ML Router initialized")
    
    def add_training_sample(self, features: Dict[str, float], 
                          actual_response_time: float, success: bool):
        """Add training sample"""
        if not self.enabled:
            return
        
        try:
            sample = {
                **features,
                'response_time': actual_response_time,
                'success': 1.0 if success else 0.0,
                'timestamp': time.time()
            }
            
            self.training_data.append(sample)
            
            # Retrain periodically
            if (len(self.training_data) >= 100 and 
                (not self.last_training or 
                 time.time() - self.last_training > 3600)):  # Retrain hourly
                asyncio.create_task(self._retrain_model())
            
        except Exception as e:
            logger.error(f"Error adding training sample: {e}")
    
    async def _retrain_model(self):
        """Retrain the routing model"""
        if not self.enabled or len(self.training_data) < 50:
            return
        
        try:
            # Prepare training data
            samples = list(self.training_data)
            
            # Feature engineering
            feature_names = [
                'cpu_usage', 'memory_usage', 'current_connections', 
                'avg_response_time', 'success_rate', 'load_factor',
                'geographic_distance', 'time_of_day', 'day_of_week'
            ]
            
            X = []
            y = []
            
            for sample in samples:
                features = [sample.get(name, 0.0) for name in feature_names]
                X.append(features)
                y.append(sample['response_time'])
            
            if len(X) < 10:
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            self.model.fit(X, y)
            self.last_training = time.time()
            
            logger.info(f"✅ ML routing model retrained with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining ML model: {e}")
    
    def predict_performance(self, node_metrics: NodePerformanceMetrics,
                          request_context: RequestContext) -> Dict[str, float]:
        """Predict node performance for request"""
        if not self.enabled or not self.model:
            return {"predicted_response_time": 100.0, "confidence": 0.0}
        
        try:
            # Extract features
            current_time = datetime.now(timezone.utc)
            
            features = [
                node_metrics.cpu_usage_percent,
                node_metrics.memory_usage_percent,
                node_metrics.current_connections,
                node_metrics.get_avg_response_time(),
                node_metrics.get_avg_success_rate(),
                node_metrics.current_load_factor,
                0.0,  # geographic_distance - would be calculated
                current_time.hour,
                current_time.weekday()
            ]
            
            # Predict
            X = np.array([features])
            predicted_time = self.model.predict(X)[0]
            
            # Calculate confidence based on training data similarity
            confidence = min(1.0, len(self.training_data) / 1000.0)
            
            return {
                "predicted_response_time": max(1.0, predicted_time),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            return {"predicted_response_time": 100.0, "confidence": 0.0}


class CircuitBreaker:
    """Circuit breaker for node failover"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        
        # State tracking
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.last_failure_times: Dict[str, datetime] = {}
        self.circuit_states: Dict[str, str] = defaultdict(lambda: "closed")  # closed, open, half-open
    
    def can_route_to_node(self, node_id: str) -> bool:
        """Check if requests can be routed to node"""
        circuit_state = self.circuit_states[node_id]
        
        if circuit_state == "closed":
            return True
        elif circuit_state == "open":
            # Check if timeout has passed
            last_failure = self.last_failure_times.get(node_id)
            if last_failure:
                time_since_failure = datetime.now(timezone.utc) - last_failure
                if time_since_failure.total_seconds() > self.timeout_seconds:
                    # Move to half-open state
                    self.circuit_states[node_id] = "half-open"
                    return True
            return False
        elif circuit_state == "half-open":
            return True
        
        return False
    
    def record_success(self, node_id: str):
        """Record successful request"""
        if self.circuit_states[node_id] == "half-open":
            # Success in half-open state, close circuit
            self.circuit_states[node_id] = "closed"
            self.failure_counts[node_id] = 0
    
    def record_failure(self, node_id: str):
        """Record failed request"""
        self.failure_counts[node_id] += 1
        self.last_failure_times[node_id] = datetime.now(timezone.utc)
        
        if self.failure_counts[node_id] >= self.failure_threshold:
            # Open circuit
            self.circuit_states[node_id] = "open"
    
    def get_circuit_state(self, node_id: str) -> str:
        """Get current circuit state"""
        return self.circuit_states[node_id]


class IntelligentRouter:
    """Advanced intelligent routing system"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
        # Components
        self.geoip_service = GeoIPService()
        self.ml_router = MachineLearningRouter()
        self.circuit_breaker = CircuitBreaker()
        
        # Node registry and metrics
        self.node_metrics: Dict[str, NodePerformanceMetrics] = {}
        self.node_locations: Dict[str, GeographicLocation] = {}
        
        # Routing state
        self.routing_history = deque(maxlen=10000)
        self.session_affinity: Dict[str, str] = {}  # session_id -> node_id
        self.sticky_sessions: Dict[str, datetime] = {}  # session -> expiry
        
        # Configuration
        self.default_strategy = RoutingStrategy.HYBRID
        self.enable_ml_routing = True
        self.enable_geographic_routing = True
        self.session_timeout_minutes = 30
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "ml_predictions": 0,
            "geographic_optimizations": 0,
            "circuit_breaker_trips": 0,
            "session_affinities": 0
        }
    
    def register_node(self, node_id: str, location: Optional[GeographicLocation] = None):
        """Register a node with the router"""
        self.node_metrics[node_id] = NodePerformanceMetrics(node_id=node_id)
        
        if location:
            self.node_locations[node_id] = location
        
        logger.info(f"✅ Registered node {node_id} with intelligent router")
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update node performance metrics"""
        if node_id not in self.node_metrics:
            self.register_node(node_id)
        
        node_metric = self.node_metrics[node_id]
        
        # Update current metrics
        node_metric.current_connections = metrics.get('current_connections', 0)
        node_metric.cpu_usage_percent = metrics.get('cpu_usage_percent', 0.0)
        node_metric.memory_usage_percent = metrics.get('memory_usage_percent', 0.0)
        node_metric.network_utilization_percent = metrics.get('network_utilization_percent', 0.0)
        
        # Update historical metrics
        if 'response_time_ms' in metrics:
            node_metric.add_response_time(metrics['response_time_ms'])
        
        if 'success_rate' in metrics:
            node_metric.add_success_rate(metrics['success_rate'])
        
        # Update health status
        node_metric.is_healthy = metrics.get('is_healthy', True)
        node_metric.last_health_check = datetime.now(timezone.utc)
        
        # Calculate load factor
        if node_metric.max_connections > 0:
            node_metric.current_load_factor = node_metric.current_connections / node_metric.max_connections
    
    async def route_request(self, request_context: RequestContext) -> Optional[RoutingDecision]:
        """Route a request to the optimal node"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Enrich request context with geographic information
            if request_context.geographic_location is None:
                request_context.geographic_location = self.geoip_service.get_location(
                    request_context.source_ip
                )
            
            # Check for session affinity
            if request_context.session_id and request_context.requires_sticky_session:
                sticky_node = self._check_session_affinity(request_context.session_id)
                if sticky_node and self._is_node_available(sticky_node):
                    decision = RoutingDecision(
                        target_node_id=sticky_node,
                        strategy_used=RoutingStrategy.IP_HASH,
                        decision_confidence=1.0,
                        expected_response_time_ms=50.0,
                        geographic_match=True,
                        selection_reason="Session affinity"
                    )
                    self.stats["session_affinities"] += 1
                    return decision
            
            # Get available nodes
            available_nodes = self._get_available_nodes()
            
            if not available_nodes:
                logger.error("No available nodes for routing")
                self.stats["failed_routes"] += 1
                return None
            
            # Apply routing strategy
            decision = await self._apply_routing_strategy(
                request_context, available_nodes, self.default_strategy
            )
            
            if decision:
                # Set up session affinity if required
                if request_context.session_id and request_context.requires_sticky_session:
                    self._set_session_affinity(request_context.session_id, decision.target_node_id)
                
                # Record routing decision
                decision.decision_time_ms = (time.time() - start_time) * 1000
                self._record_routing_decision(request_context, decision)
                
                self.stats["successful_routes"] += 1
                return decision
            else:
                self.stats["failed_routes"] += 1
                return None
            
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            self.stats["failed_routes"] += 1
            return None
    
    def _get_available_nodes(self) -> List[str]:
        """Get list of available nodes"""
        available_nodes = []
        
        for node_id, metrics in self.node_metrics.items():
            # Check health
            if not metrics.is_healthy:
                continue
            
            # Check circuit breaker
            if not self.circuit_breaker.can_route_to_node(node_id):
                continue
            
            # Check capacity
            if metrics.current_load_factor >= 0.95:  # 95% capacity
                continue
            
            available_nodes.append(node_id)
        
        return available_nodes
    
    def _is_node_available(self, node_id: str) -> bool:
        """Check if a specific node is available"""
        return node_id in self._get_available_nodes()
    
    async def _apply_routing_strategy(self, request_context: RequestContext,
                                    available_nodes: List[str],
                                    strategy: RoutingStrategy) -> Optional[RoutingDecision]:
        """Apply the specified routing strategy"""
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_routing(available_nodes)
        
        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_routing(available_nodes)
        
        elif strategy == RoutingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_routing(available_nodes)
        
        elif strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_routing(available_nodes)
        
        elif strategy == RoutingStrategy.IP_HASH:
            return self._ip_hash_routing(request_context, available_nodes)
        
        elif strategy == RoutingStrategy.GEOGRAPHIC:
            return self._geographic_routing(request_context, available_nodes)
        
        elif strategy == RoutingStrategy.MACHINE_LEARNING:
            return await self._ml_routing(request_context, available_nodes)
        
        elif strategy == RoutingStrategy.HYBRID:
            return await self._hybrid_routing(request_context, available_nodes)
        
        else:
            # Fallback to round robin
            return self._round_robin_routing(available_nodes)
    
    def _round_robin_routing(self, available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Simple round robin routing"""
        if not available_nodes:
            return None
        
        # Simple counter-based round robin
        node_index = self.stats["total_requests"] % len(available_nodes)
        selected_node = available_nodes[node_index]
        
        return RoutingDecision(
            target_node_id=selected_node,
            strategy_used=RoutingStrategy.ROUND_ROBIN,
            decision_confidence=0.7,
            expected_response_time_ms=100.0,
            selection_reason="Round robin selection"
        )
    
    def _least_connections_routing(self, available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Route to node with least connections"""
        if not available_nodes:
            return None
        
        best_node = None
        min_connections = float('inf')
        
        for node_id in available_nodes:
            metrics = self.node_metrics[node_id]
            if metrics.current_connections < min_connections:
                min_connections = metrics.current_connections
                best_node = node_id
        
        if best_node:
            return RoutingDecision(
                target_node_id=best_node,
                strategy_used=RoutingStrategy.LEAST_CONNECTIONS,
                decision_confidence=0.8,
                expected_response_time_ms=80.0,
                selection_reason=f"Least connections ({min_connections})"
            )
        
        return None
    
    def _least_response_time_routing(self, available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Route to node with best response time"""
        if not available_nodes:
            return None
        
        best_node = None
        best_response_time = float('inf')
        
        for node_id in available_nodes:
            metrics = self.node_metrics[node_id]
            avg_response_time = metrics.get_avg_response_time()
            
            if avg_response_time < best_response_time:
                best_response_time = avg_response_time
                best_node = node_id
        
        if best_node:
            return RoutingDecision(
                target_node_id=best_node,
                strategy_used=RoutingStrategy.LEAST_RESPONSE_TIME,
                decision_confidence=0.9,
                expected_response_time_ms=best_response_time,
                selection_reason=f"Best response time ({best_response_time:.1f}ms)"
            )
        
        return None
    
    def _weighted_round_robin_routing(self, available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Weighted round robin based on node health scores"""
        if not available_nodes:
            return None
        
        # Calculate weights based on health scores
        weights = []
        for node_id in available_nodes:
            health_score = self.node_metrics[node_id].calculate_health_score()
            weights.append(health_score)
        
        # Weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return self._round_robin_routing(available_nodes)
        
        import random
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                selected_node = available_nodes[i]
                return RoutingDecision(
                    target_node_id=selected_node,
                    strategy_used=RoutingStrategy.WEIGHTED_ROUND_ROBIN,
                    decision_confidence=0.8,
                    expected_response_time_ms=90.0,
                    selection_reason=f"Weighted selection (weight: {weight:.2f})"
                )
        
        # Fallback
        return self._round_robin_routing(available_nodes)
    
    def _ip_hash_routing(self, request_context: RequestContext, 
                       available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Hash-based routing for session affinity"""
        if not available_nodes:
            return None
        
        # Create hash from IP and optional session ID
        hash_input = request_context.source_ip
        if request_context.session_id:
            hash_input += request_context.session_id
        
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        node_index = hash_value % len(available_nodes)
        selected_node = available_nodes[node_index]
        
        return RoutingDecision(
            target_node_id=selected_node,
            strategy_used=RoutingStrategy.IP_HASH,
            decision_confidence=1.0,
            expected_response_time_ms=70.0,
            selection_reason="IP hash consistency"
        )
    
    def _geographic_routing(self, request_context: RequestContext,
                          available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Geographic proximity based routing"""
        if not available_nodes or not request_context.geographic_location:
            return self._round_robin_routing(available_nodes)
        
        best_node = None
        min_distance = float('inf')
        
        for node_id in available_nodes:
            node_location = self.node_locations.get(node_id)
            if node_location:
                distance = self.geoip_service.calculate_distance(
                    request_context.geographic_location, node_location
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_node = node_id
        
        if best_node:
            self.stats["geographic_optimizations"] += 1
            return RoutingDecision(
                target_node_id=best_node,
                strategy_used=RoutingStrategy.GEOGRAPHIC,
                decision_confidence=0.9,
                expected_response_time_ms=60.0,
                geographic_match=True,
                selection_reason=f"Geographic proximity ({min_distance:.0f}km)"
            )
        
        # Fallback if no geographic data
        return self._round_robin_routing(available_nodes)
    
    async def _ml_routing(self, request_context: RequestContext,
                        available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Machine learning based routing"""
        if not available_nodes or not self.ml_router.enabled:
            return self._least_response_time_routing(available_nodes)
        
        best_node = None
        best_predicted_time = float('inf')
        best_confidence = 0.0
        
        for node_id in available_nodes:
            metrics = self.node_metrics[node_id]
            prediction = self.ml_router.predict_performance(metrics, request_context)
            
            predicted_time = prediction["predicted_response_time"]
            confidence = prediction["confidence"]
            
            # Weighted score considering both time and confidence
            score = predicted_time * (2.0 - confidence)  # Lower is better
            
            if score < best_predicted_time:
                best_predicted_time = predicted_time
                best_confidence = confidence
                best_node = node_id
        
        if best_node:
            self.stats["ml_predictions"] += 1
            return RoutingDecision(
                target_node_id=best_node,
                strategy_used=RoutingStrategy.MACHINE_LEARNING,
                decision_confidence=best_confidence,
                expected_response_time_ms=best_predicted_time,
                selection_reason=f"ML prediction (confidence: {best_confidence:.2f})"
            )
        
        return self._least_response_time_routing(available_nodes)
    
    async def _hybrid_routing(self, request_context: RequestContext,
                            available_nodes: List[str]) -> Optional[RoutingDecision]:
        """Hybrid routing combining multiple strategies"""
        if not available_nodes:
            return None
        
        # Score each node using multiple factors
        node_scores = {}
        
        for node_id in available_nodes:
            metrics = self.node_metrics[node_id]
            score = 0.0
            factors = {}
            
            # Health score (30% weight)
            health_score = metrics.calculate_health_score()
            score += health_score * 0.3
            factors["health"] = health_score
            
            # Response time score (25% weight)
            avg_response_time = metrics.get_avg_response_time()
            response_time_score = max(0, 1.0 - (avg_response_time / 1000))  # Normalize to 0-1
            score += response_time_score * 0.25
            factors["response_time"] = response_time_score
            
            # Load score (20% weight)
            load_score = max(0, 1.0 - metrics.current_load_factor)
            score += load_score * 0.2
            factors["load"] = load_score
            
            # Geographic score (15% weight)
            if (request_context.geographic_location and 
                node_id in self.node_locations):
                distance = self.geoip_service.calculate_distance(
                    request_context.geographic_location,
                    self.node_locations[node_id]
                )
                # Normalize distance (closer = higher score)
                geo_score = max(0, 1.0 - (distance / 10000))  # 10000km max
                score += geo_score * 0.15
                factors["geography"] = geo_score
            
            # ML prediction score (10% weight)
            if self.ml_router.enabled:
                prediction = self.ml_router.predict_performance(metrics, request_context)
                predicted_time = prediction["predicted_response_time"]
                ml_score = max(0, 1.0 - (predicted_time / 1000))
                score += ml_score * 0.1
                factors["ml_prediction"] = ml_score
            
            node_scores[node_id] = {
                "total_score": score,
                "factors": factors
            }
        
        # Select best node
        if node_scores:
            best_node = max(node_scores.keys(), key=lambda k: node_scores[k]["total_score"])
            best_score_data = node_scores[best_node]
            
            return RoutingDecision(
                target_node_id=best_node,
                strategy_used=RoutingStrategy.HYBRID,
                decision_confidence=best_score_data["total_score"],
                expected_response_time_ms=80.0,
                decision_factors=best_score_data["factors"],
                selection_reason=f"Hybrid scoring (score: {best_score_data['total_score']:.2f})"
            )
        
        return None
    
    def _check_session_affinity(self, session_id: str) -> Optional[str]:
        """Check if session has affinity to a node"""
        if session_id in self.session_affinity:
            # Check if session hasn't expired
            if session_id in self.sticky_sessions:
                if datetime.now(timezone.utc) < self.sticky_sessions[session_id]:
                    return self.session_affinity[session_id]
                else:
                    # Expired session
                    del self.session_affinity[session_id]
                    del self.sticky_sessions[session_id]
        
        return None
    
    def _set_session_affinity(self, session_id: str, node_id: str):
        """Set session affinity to a node"""
        self.session_affinity[session_id] = node_id
        expiry_time = datetime.now(timezone.utc) + timedelta(minutes=self.session_timeout_minutes)
        self.sticky_sessions[session_id] = expiry_time
    
    def _record_routing_decision(self, request_context: RequestContext, 
                               decision: RoutingDecision):
        """Record routing decision for analytics"""
        record = {
            "timestamp": request_context.timestamp.isoformat(),
            "request_id": request_context.request_id,
            "source_ip": request_context.source_ip,
            "target_node": decision.target_node_id,
            "strategy": decision.strategy_used.value,
            "confidence": decision.decision_confidence,
            "expected_response_time": decision.expected_response_time_ms,
            "decision_time_ms": decision.decision_time_ms,
            "geographic_match": decision.geographic_match,
            "selection_reason": decision.selection_reason
        }
        
        self.routing_history.append(record)
    
    def record_request_result(self, request_id: str, actual_response_time_ms: float,
                            success: bool, error_details: Optional[str] = None):
        """Record actual request results for learning"""
        try:
            # Find the routing decision
            routing_record = None
            for record in reversed(self.routing_history):
                if record["request_id"] == request_id:
                    routing_record = record
                    break
            
            if not routing_record:
                return
            
            node_id = routing_record["target_node"]
            
            # Update circuit breaker
            if success:
                self.circuit_breaker.record_success(node_id)
            else:
                self.circuit_breaker.record_failure(node_id)
                self.stats["circuit_breaker_trips"] += 1
            
            # Update node metrics
            if node_id in self.node_metrics:
                metrics = self.node_metrics[node_id]
                metrics.add_response_time(actual_response_time_ms)
                metrics.add_success_rate(1.0 if success else 0.0)
                
                if not success:
                    metrics.consecutive_failures += 1
                else:
                    metrics.consecutive_failures = 0
            
            # Add to ML training data
            if routing_record.get("strategy") in [RoutingStrategy.MACHINE_LEARNING.value, RoutingStrategy.HYBRID.value]:
                features = {
                    "cpu_usage": self.node_metrics[node_id].cpu_usage_percent,
                    "memory_usage": self.node_metrics[node_id].memory_usage_percent,
                    "current_connections": self.node_metrics[node_id].current_connections,
                    "load_factor": self.node_metrics[node_id].current_load_factor,
                    "time_of_day": datetime.now(timezone.utc).hour,
                    "day_of_week": datetime.now(timezone.utc).weekday()
                }
                
                self.ml_router.add_training_sample(features, actual_response_time_ms, success)
            
        except Exception as e:
            logger.error(f"Error recording request result: {e}")
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        try:
            # Calculate success rates
            if self.stats["total_requests"] > 0:
                success_rate = (self.stats["successful_routes"] / self.stats["total_requests"]) * 100
            else:
                success_rate = 0.0
            
            # Analyze recent routing history
            recent_history = list(self.routing_history)[-1000:]  # Last 1000 requests
            
            strategy_usage = defaultdict(int)
            node_usage = defaultdict(int)
            avg_decision_times = []
            
            for record in recent_history:
                strategy_usage[record["strategy"]] += 1
                node_usage[record["target_node"]] += 1
                avg_decision_times.append(record.get("decision_time_ms", 0))
            
            avg_decision_time = sum(avg_decision_times) / len(avg_decision_times) if avg_decision_times else 0
            
            # Node health summary
            healthy_nodes = sum(1 for metrics in self.node_metrics.values() if metrics.is_healthy)
            total_nodes = len(self.node_metrics)
            
            # Circuit breaker states
            circuit_states = defaultdict(int)
            for node_id in self.node_metrics.keys():
                state = self.circuit_breaker.get_circuit_state(node_id)
                circuit_states[state] += 1
            
            return {
                "routing_performance": {
                    "total_requests": self.stats["total_requests"],
                    "success_rate_percent": success_rate,
                    "avg_decision_time_ms": avg_decision_time
                },
                "node_health": {
                    "healthy_nodes": healthy_nodes,
                    "total_nodes": total_nodes,
                    "health_percentage": (healthy_nodes / total_nodes * 100) if total_nodes > 0 else 0
                },
                "strategy_usage": dict(strategy_usage),
                "node_usage": dict(node_usage),
                "circuit_breaker_states": dict(circuit_states),
                "session_affinity": {
                    "active_sessions": len(self.session_affinity),
                    "sticky_sessions": len(self.sticky_sessions)
                },
                "machine_learning": {
                    "enabled": self.ml_router.enabled,
                    "training_samples": len(self.ml_router.training_data),
                    "predictions_made": self.stats["ml_predictions"]
                },
                "geographic_routing": {
                    "enabled": self.enable_geographic_routing,
                    "optimizations": self.stats["geographic_optimizations"]
                },
                "detailed_stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error getting routing statistics: {e}")
            return {"error": str(e)}


# Global intelligent router instance
intelligent_router: Optional[IntelligentRouter] = None


def initialize_intelligent_router(redis_client: aioredis.Redis) -> IntelligentRouter:
    """Initialize the intelligent routing system"""
    global intelligent_router
    
    intelligent_router = IntelligentRouter(redis_client)
    logger.info("✅ Intelligent routing system initialized")
    return intelligent_router


def get_intelligent_router() -> IntelligentRouter:
    """Get the global intelligent router instance"""
    if intelligent_router is None:
        raise RuntimeError("Intelligent router not initialized")
    return intelligent_router


# Convenience functions for integration

async def route_request_intelligently(request_context: RequestContext) -> Optional[RoutingDecision]:
    """Route a request using intelligent routing"""
    if intelligent_router:
        return await intelligent_router.route_request(request_context)
    return None


def update_node_performance(node_id: str, metrics: Dict[str, Any]):
    """Update node performance metrics"""
    if intelligent_router:
        intelligent_router.update_node_metrics(node_id, metrics)


def record_request_outcome(request_id: str, response_time_ms: float, 
                         success: bool, error_details: Optional[str] = None):
    """Record request outcome for learning"""
    if intelligent_router:
        intelligent_router.record_request_result(request_id, response_time_ms, success, error_details)