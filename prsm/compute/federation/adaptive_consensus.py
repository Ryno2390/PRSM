"""
Adaptive Consensus Mechanisms for PRSM
Dynamically optimizes consensus strategy based on network conditions
"""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from enum import Enum
import math

from prsm.core.config import settings
from prsm.core.models import PeerNode, AgentResponse, SafetyFlag, SafetyLevel
from prsm.core.safety.circuit_breaker import CircuitBreakerNetwork, ThreatLevel
from prsm.core.safety.monitor import SafetyMonitor
from prsm.economy.tokenomics.ftns_service import ftns_service
from .consensus import DistributedConsensus, ConsensusResult, ConsensusType
from .hierarchical_consensus import HierarchicalConsensusNetwork


# === Adaptive Consensus Configuration ===

# Network condition thresholds
SMALL_NETWORK_THRESHOLD = int(getattr(settings, "PRSM_SMALL_NETWORK_THRESHOLD", 10))
MEDIUM_NETWORK_THRESHOLD = int(getattr(settings, "PRSM_MEDIUM_NETWORK_THRESHOLD", 25))
LARGE_NETWORK_THRESHOLD = int(getattr(settings, "PRSM_LARGE_NETWORK_THRESHOLD", 50))

# Performance thresholds
HIGH_LATENCY_THRESHOLD_MS = int(getattr(settings, "PRSM_HIGH_LATENCY_THRESHOLD", 200))
LOW_THROUGHPUT_THRESHOLD = float(getattr(settings, "PRSM_LOW_THROUGHPUT_THRESHOLD", 5.0))
HIGH_FAILURE_RATE_THRESHOLD = float(getattr(settings, "PRSM_HIGH_FAILURE_RATE", 0.1))

# Adaptation settings
NETWORK_MONITORING_WINDOW = int(getattr(settings, "PRSM_MONITORING_WINDOW", 60))  # seconds
ADAPTATION_COOLDOWN = int(getattr(settings, "PRSM_ADAPTATION_COOLDOWN", 30))     # seconds
MIN_SAMPLES_FOR_ADAPTATION = int(getattr(settings, "PRSM_MIN_ADAPTATION_SAMPLES", 5))

# Safety thresholds
BYZANTINE_TOLERANCE_STRICT = float(getattr(settings, "PRSM_BYZANTINE_STRICT", 0.20))  # 20%
BYZANTINE_TOLERANCE_RELAXED = float(getattr(settings, "PRSM_BYZANTINE_RELAXED", 0.33))  # 33%


class NetworkCondition(Enum):
    """Network condition classifications"""
    OPTIMAL = "optimal"           # Low latency, high throughput, low failures
    CONGESTED = "congested"       # High latency or low throughput
    UNRELIABLE = "unreliable"     # High failure rates or Byzantine behavior
    DEGRADED = "degraded"         # Multiple poor conditions
    RECOVERING = "recovering"     # Improving from poor conditions


class ConsensusStrategy(Enum):
    """Available consensus strategies"""
    FAST_MAJORITY = "fast_majority"           # Simple majority for optimal conditions
    WEIGHTED_CONSENSUS = "weighted_consensus" # Reputation-weighted for normal conditions
    HIERARCHICAL = "hierarchical"             # Multi-tier for large networks
    BYZANTINE_RESILIENT = "byzantine_resilient" # Maximum fault tolerance
    HYBRID_ADAPTIVE = "hybrid_adaptive"       # Dynamic combination


class NetworkMetrics:
    """Network performance metrics for adaptive decisions"""
    
    def __init__(self, window_size: int = NETWORK_MONITORING_WINDOW):
        self.window_size = window_size
        
        # Performance metrics (time-windowed)
        self.latency_samples = deque(maxlen=100)
        self.throughput_samples = deque(maxlen=100)
        self.consensus_times = deque(maxlen=50)
        self.failure_events = deque(maxlen=100)
        
        # Network composition
        self.active_nodes = set()
        self.byzantine_nodes = set()
        self.node_reputations = {}
        
        # Real-time state
        self.last_update = datetime.now(timezone.utc)
        self.current_condition = NetworkCondition.OPTIMAL
        self.recommended_strategy = ConsensusStrategy.FAST_MAJORITY
        
        # Adaptation history
        self.strategy_performance = defaultdict(list)
        self.last_adaptation = datetime.now(timezone.utc) - timedelta(seconds=ADAPTATION_COOLDOWN)
        
    def add_latency_sample(self, latency_ms: float):
        """Add network latency measurement"""
        self.latency_samples.append({
            'value': latency_ms,
            'timestamp': datetime.now(timezone.utc)
        })
        self._update_condition()
    
    def add_throughput_sample(self, ops_per_second: float):
        """Add throughput measurement"""
        self.throughput_samples.append({
            'value': ops_per_second,
            'timestamp': datetime.now(timezone.utc)
        })
        self._update_condition()
    
    def add_consensus_result(self, consensus_time: float, success: bool, strategy: ConsensusStrategy):
        """Add consensus performance result"""
        self.consensus_times.append({
            'time': consensus_time,
            'success': success,
            'strategy': strategy,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Track strategy performance
        self.strategy_performance[strategy].append({
            'time': consensus_time,
            'success': success,
            'timestamp': datetime.now(timezone.utc)
        })
        
        self._update_condition()
    
    def add_failure_event(self, failure_type: str, node_id: Optional[str] = None):
        """Add network failure event"""
        self.failure_events.append({
            'type': failure_type,
            'node_id': node_id,
            'timestamp': datetime.now(timezone.utc)
        })
        self._update_condition()
    
    def update_network_composition(self, active_nodes: Set[str], byzantine_nodes: Set[str], reputations: Dict[str, float]):
        """Update network composition information"""
        self.active_nodes = active_nodes.copy()
        self.byzantine_nodes = byzantine_nodes.copy()
        self.node_reputations = reputations.copy()
        self.last_update = datetime.now(timezone.utc)
        self._update_condition()
    
    def _update_condition(self):
        """Update current network condition based on recent metrics"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.window_size)
        
        # Filter recent samples
        recent_latency = [s['value'] for s in self.latency_samples if s['timestamp'] > cutoff]
        recent_throughput = [s['value'] for s in self.throughput_samples if s['timestamp'] > cutoff]
        recent_failures = [f for f in self.failure_events if f['timestamp'] > cutoff]
        recent_consensus = [c for c in self.consensus_times if c['timestamp'] > cutoff]
        
        # Calculate condition indicators
        avg_latency = statistics.mean(recent_latency) if recent_latency else 0
        avg_throughput = statistics.mean(recent_throughput) if recent_throughput else 0
        failure_rate = len(recent_failures) / max(1, len(recent_consensus))
        byzantine_ratio = len(self.byzantine_nodes) / max(1, len(self.active_nodes))
        
        # Determine network condition
        conditions = []
        
        if avg_latency > HIGH_LATENCY_THRESHOLD_MS:
            conditions.append("high_latency")
        
        if avg_throughput < LOW_THROUGHPUT_THRESHOLD:
            conditions.append("low_throughput")
            
        if failure_rate > HIGH_FAILURE_RATE_THRESHOLD:
            conditions.append("high_failures")
            
        if byzantine_ratio > BYZANTINE_TOLERANCE_STRICT:
            conditions.append("byzantine_threats")
        
        # Map conditions to network state
        if not conditions:
            self.current_condition = NetworkCondition.OPTIMAL
            self.recommended_strategy = self._get_optimal_strategy()
        elif len(conditions) == 1:
            if "high_latency" in conditions or "low_throughput" in conditions:
                self.current_condition = NetworkCondition.CONGESTED
                self.recommended_strategy = ConsensusStrategy.HIERARCHICAL
            else:
                self.current_condition = NetworkCondition.UNRELIABLE
                self.recommended_strategy = ConsensusStrategy.BYZANTINE_RESILIENT
        else:
            self.current_condition = NetworkCondition.DEGRADED
            self.recommended_strategy = ConsensusStrategy.HYBRID_ADAPTIVE
    
    def _get_optimal_strategy(self) -> ConsensusStrategy:
        """Get optimal strategy for current network size"""
        network_size = len(self.active_nodes)
        
        if network_size <= SMALL_NETWORK_THRESHOLD:
            return ConsensusStrategy.FAST_MAJORITY
        elif network_size <= MEDIUM_NETWORK_THRESHOLD:
            return ConsensusStrategy.WEIGHTED_CONSENSUS
        else:
            return ConsensusStrategy.HIERARCHICAL
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.window_size)
        
        recent_latency = [s['value'] for s in self.latency_samples if s['timestamp'] > cutoff]
        recent_throughput = [s['value'] for s in self.throughput_samples if s['timestamp'] > cutoff]
        recent_consensus = [c for c in self.consensus_times if c['timestamp'] > cutoff]
        recent_failures = [f for f in self.failure_events if f['timestamp'] > cutoff]
        
        return {
            'condition': self.current_condition.value,
            'recommended_strategy': self.recommended_strategy.value,
            'network_size': len(self.active_nodes),
            'byzantine_nodes': len(self.byzantine_nodes),
            'avg_latency_ms': statistics.mean(recent_latency) if recent_latency else 0,
            'avg_throughput': statistics.mean(recent_throughput) if recent_throughput else 0,
            'failure_rate': len(recent_failures) / max(1, len(recent_consensus)),
            'consensus_success_rate': (
                sum(1 for c in recent_consensus if c['success']) / max(1, len(recent_consensus))
            ),
            'samples_count': {
                'latency': len(recent_latency),
                'throughput': len(recent_throughput),
                'consensus': len(recent_consensus),
                'failures': len(recent_failures)
            }
        }


class AdaptiveConsensusEngine:
    """
    Adaptive consensus engine that dynamically selects optimal consensus mechanisms
    based on real-time network conditions and performance metrics
    """
    
    def __init__(self):
        # Core consensus mechanisms
        self.distributed_consensus = DistributedConsensus()
        self.hierarchical_consensus = HierarchicalConsensusNetwork()
        
        # Network monitoring
        self.network_metrics = NetworkMetrics()
        
        # Strategy implementations
        self.strategy_implementations = {
            ConsensusStrategy.FAST_MAJORITY: self._fast_majority_consensus,
            ConsensusStrategy.WEIGHTED_CONSENSUS: self._weighted_consensus,
            ConsensusStrategy.HIERARCHICAL: self._hierarchical_consensus,
            ConsensusStrategy.BYZANTINE_RESILIENT: self._byzantine_resilient_consensus,
            ConsensusStrategy.HYBRID_ADAPTIVE: self._hybrid_adaptive_consensus
        }
        
        # Safety integration
        self.circuit_breaker = CircuitBreakerNetwork()
        self.safety_monitor = SafetyMonitor()
        
        # Adaptation state
        self.current_strategy = ConsensusStrategy.FAST_MAJORITY
        self.adaptation_in_progress = False
        self.strategy_switch_count = 0
        
        # Performance tracking
        self.adaptive_metrics = {
            "total_adaptive_consensus": 0,
            "successful_adaptive_consensus": 0,
            "strategy_switches": 0,
            "adaptation_improvements": 0,
            "average_adaptation_time": 0.0,
            "strategy_usage": defaultdict(int),
            "strategy_performance": defaultdict(list)
        }
        
        # Synchronization
        self._adaptation_lock = asyncio.Lock()
    
    async def initialize_adaptive_consensus(self, peer_nodes: List[PeerNode]) -> bool:
        """Initialize adaptive consensus with network peers"""
        try:
            print(f"🧠 Initializing adaptive consensus with {len(peer_nodes)} peers")
            
            # Initialize hierarchical consensus for large network scenarios
            if len(peer_nodes) > SMALL_NETWORK_THRESHOLD:
                success = await self.hierarchical_consensus.initialize_hierarchical_network(peer_nodes)
                if not success:
                    print("⚠️ Hierarchical consensus initialization failed - using flat consensus")
            
            # Update network composition
            active_nodes = {peer.peer_id for peer in peer_nodes}
            byzantine_nodes = set()  # Will be populated as failures are detected
            reputations = {peer.peer_id: peer.reputation_score for peer in peer_nodes}
            
            self.network_metrics.update_network_composition(active_nodes, byzantine_nodes, reputations)
            
            # Select initial strategy
            self.current_strategy = self.network_metrics.recommended_strategy
            
            print(f"✅ Adaptive consensus initialized:")
            print(f"   - Initial strategy: {self.current_strategy.value}")
            print(f"   - Network condition: {self.network_metrics.current_condition.value}")
            print(f"   - Network size: {len(peer_nodes)} nodes")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing adaptive consensus: {e}")
            return False
    
    async def achieve_adaptive_consensus(self, 
                                       proposal: Dict[str, Any], 
                                       session_id: Optional[str] = None) -> ConsensusResult:
        """
        Achieve consensus using adaptive strategy selection
        
        Args:
            proposal: Consensus proposal
            session_id: Optional session identifier
            
        Returns:
            ConsensusResult with adaptive consensus details
        """
        start_time = datetime.now(timezone.utc)
        session_id = session_id or str(uuid4())
        
        async with self._adaptation_lock:
            try:
                self.adaptive_metrics["total_adaptive_consensus"] += 1
                
                print(f"🧠 Starting adaptive consensus (session: {session_id[:8]})")
                
                # Check if adaptation is needed
                await self._check_and_adapt_strategy()
                
                # Execute consensus with current strategy
                strategy_func = self.strategy_implementations[self.current_strategy]
                result = await strategy_func(proposal, session_id)
                
                # Update execution time
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                result.execution_time = execution_time
                
                # Record performance metrics
                self.network_metrics.add_consensus_result(
                    execution_time, 
                    result.consensus_achieved, 
                    self.current_strategy
                )
                
                # Update adaptive metrics
                if result.consensus_achieved:
                    self.adaptive_metrics["successful_adaptive_consensus"] += 1
                    
                    # Update average adaptation time
                    total_attempts = self.adaptive_metrics["total_adaptive_consensus"]
                    current_avg = self.adaptive_metrics["average_adaptation_time"]
                    self.adaptive_metrics["average_adaptation_time"] = (
                        (current_avg * (total_attempts - 1) + execution_time) / total_attempts
                    )
                
                # Track strategy usage
                self.adaptive_metrics["strategy_usage"][self.current_strategy] += 1
                self.adaptive_metrics["strategy_performance"][self.current_strategy].append({
                    'execution_time': execution_time,
                    'success': result.consensus_achieved,
                    'agreement_ratio': result.agreement_ratio
                })
                
                # Add adaptive information to result
                result.consensus_type = f"adaptive_{self.current_strategy.value}"
                
                print(f"🤝 Adaptive consensus {'achieved' if result.consensus_achieved else 'failed'}:")
                print(f"   - Strategy: {self.current_strategy.value}")
                print(f"   - Agreement: {result.agreement_ratio:.2%}")
                print(f"   - Execution time: {execution_time:.2f}s")
                print(f"   - Network condition: {self.network_metrics.current_condition.value}")
                
                return result
                
            except Exception as e:
                print(f"❌ Adaptive consensus error: {e}")
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_type=f"adaptive_{self.current_strategy.value}_error",
                    execution_time=execution_time
                )
    
    async def _check_and_adapt_strategy(self):
        """Check if strategy adaptation is needed and execute if appropriate"""
        try:
            # Check cooldown period
            now = datetime.now(timezone.utc)
            if (now - self.network_metrics.last_adaptation).total_seconds() < ADAPTATION_COOLDOWN:
                return
            
            # Check if we have sufficient samples
            summary = self.network_metrics.get_performance_summary()
            if summary['samples_count']['consensus'] < MIN_SAMPLES_FOR_ADAPTATION:
                return
            
            recommended_strategy = self.network_metrics.recommended_strategy
            
            # Check if strategy change is recommended
            if recommended_strategy != self.current_strategy:
                print(f"🔄 Adapting consensus strategy:")
                print(f"   - From: {self.current_strategy.value}")
                print(f"   - To: {recommended_strategy.value}")
                print(f"   - Reason: {self.network_metrics.current_condition.value} network condition")
                
                # Check if new strategy is likely to perform better
                if await self._should_switch_strategy(recommended_strategy):
                    self.current_strategy = recommended_strategy
                    self.network_metrics.last_adaptation = now
                    self.strategy_switch_count += 1
                    self.adaptive_metrics["strategy_switches"] += 1
                    
                    print(f"✅ Strategy adapted to {self.current_strategy.value}")
                else:
                    print(f"⚠️ Strategy adaptation deferred - insufficient performance improvement expected")
            
        except Exception as e:
            print(f"❌ Error checking strategy adaptation: {e}")
    
    async def _should_switch_strategy(self, new_strategy: ConsensusStrategy) -> bool:
        """Determine if switching to new strategy is beneficial"""
        try:
            # Get recent performance of current strategy
            current_performance = self.adaptive_metrics["strategy_performance"].get(self.current_strategy, [])
            
            if not current_performance:
                return True  # No data, worth trying new strategy
            
            # Calculate recent performance metrics for current strategy
            recent_current = current_performance[-10:]  # Last 10 attempts
            current_success_rate = sum(1 for p in recent_current if p['success']) / len(recent_current)
            current_avg_time = statistics.mean(p['execution_time'] for p in recent_current)
            
            # Check performance thresholds
            if current_success_rate < 0.8 or current_avg_time > 5.0:
                return True  # Current strategy performing poorly
            
            # Get historical performance of new strategy if available
            new_performance = self.adaptive_metrics["strategy_performance"].get(new_strategy, [])
            
            if not new_performance:
                return True  # No data for new strategy, worth trying
            
            # Compare historical performance
            recent_new = new_performance[-10:]
            new_success_rate = sum(1 for p in recent_new if p['success']) / len(recent_new)
            new_avg_time = statistics.mean(p['execution_time'] for p in recent_new)
            
            # Switch if new strategy shows better performance
            return (new_success_rate > current_success_rate * 1.1 or 
                    new_avg_time < current_avg_time * 0.9)
            
        except Exception as e:
            print(f"❌ Error evaluating strategy switch: {e}")
            return False
    
    async def _fast_majority_consensus(self, proposal: Dict[str, Any], session_id: str) -> ConsensusResult:
        """Fast majority consensus for optimal network conditions"""
        try:
            # Simulate peer results for fast consensus
            peer_results = []
            for node_id in list(self.network_metrics.active_nodes)[:10]:  # Limit for speed
                peer_result = {
                    "peer_id": node_id,
                    "result": proposal,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                peer_results.append(peer_result)
            
            # Use simple majority consensus
            result = await self.distributed_consensus.achieve_result_consensus(
                peer_results, 
                ConsensusType.SIMPLE_MAJORITY,
                session_id
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Fast majority consensus error: {e}")
            return ConsensusResult(consensus_achieved=False, consensus_type="fast_majority_error")
    
    async def _weighted_consensus(self, proposal: Dict[str, Any], session_id: str) -> ConsensusResult:
        """Weighted consensus using node reputations"""
        try:
            # Create peer results with reputation weighting
            peer_results = []
            for node_id in self.network_metrics.active_nodes:
                peer_result = {
                    "peer_id": node_id,
                    "result": proposal,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reputation": self.network_metrics.node_reputations.get(node_id, 0.5)
                }
                peer_results.append(peer_result)
            
            # Use weighted majority consensus
            result = await self.distributed_consensus.achieve_result_consensus(
                peer_results,
                ConsensusType.WEIGHTED_MAJORITY,
                session_id
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Weighted consensus error: {e}")
            return ConsensusResult(consensus_achieved=False, consensus_type="weighted_consensus_error")
    
    async def _hierarchical_consensus(self, proposal: Dict[str, Any], session_id: str) -> ConsensusResult:
        """Hierarchical consensus for large networks"""
        try:
            # Use hierarchical consensus implementation
            result = await self.hierarchical_consensus.achieve_hierarchical_consensus(proposal, session_id)
            return result
            
        except Exception as e:
            print(f"❌ Hierarchical consensus error: {e}")
            return ConsensusResult(consensus_achieved=False, consensus_type="hierarchical_error")
    
    async def _byzantine_resilient_consensus(self, proposal: Dict[str, Any], session_id: str) -> ConsensusResult:
        """Byzantine resilient consensus for unreliable conditions"""
        try:
            # Create peer results excluding known Byzantine nodes
            reliable_nodes = self.network_metrics.active_nodes - self.network_metrics.byzantine_nodes
            
            peer_results = []
            for node_id in reliable_nodes:
                peer_result = {
                    "peer_id": node_id,
                    "result": proposal,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                peer_results.append(peer_result)
            
            # Use Byzantine fault tolerant consensus
            result = await self.distributed_consensus.achieve_result_consensus(
                peer_results,
                ConsensusType.BYZANTINE_FAULT_TOLERANT,
                session_id
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Byzantine resilient consensus error: {e}")
            return ConsensusResult(consensus_achieved=False, consensus_type="byzantine_resilient_error")
    
    async def _hybrid_adaptive_consensus(self, proposal: Dict[str, Any], session_id: str) -> ConsensusResult:
        """Hybrid adaptive consensus combining multiple strategies"""
        try:
            # Execute multiple consensus strategies in parallel for comparison
            tasks = []
            
            # Fast majority for quick result
            fast_task = asyncio.create_task(self._fast_majority_consensus(proposal, f"{session_id}_fast"))
            tasks.append(("fast", fast_task))
            
            # Byzantine resilient for safety
            byzantine_task = asyncio.create_task(self._byzantine_resilient_consensus(proposal, f"{session_id}_byzantine"))
            tasks.append(("byzantine", byzantine_task))
            
            # Wait for first successful result or all to complete
            results = {}
            for name, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=10.0)
                    results[name] = result
                    
                    # Return first successful result
                    if result.consensus_achieved:
                        print(f"✅ Hybrid consensus achieved via {name} strategy")
                        result.consensus_type = f"hybrid_adaptive_{name}"
                        return result
                        
                except asyncio.TimeoutError:
                    print(f"⚠️ {name} strategy timed out")
                except Exception as e:
                    print(f"⚠️ {name} strategy failed: {e}")
            
            # If no strategy succeeded, return best available result
            if results:
                best_result = max(results.values(), key=lambda r: r.agreement_ratio)
                best_result.consensus_type = "hybrid_adaptive_best_effort"
                return best_result
            else:
                return ConsensusResult(consensus_achieved=False, consensus_type="hybrid_adaptive_failed")
                
        except Exception as e:
            print(f"❌ Hybrid adaptive consensus error: {e}")
            return ConsensusResult(consensus_achieved=False, consensus_type="hybrid_adaptive_error")
    
    async def report_network_event(self, event_type: str, metrics: Dict[str, Any]):
        """Report network events for adaptive decision making"""
        try:
            # Update network metrics based on event
            if event_type == "latency_measurement":
                self.network_metrics.add_latency_sample(metrics.get("latency_ms", 0))
                
            elif event_type == "throughput_measurement":
                self.network_metrics.add_throughput_sample(metrics.get("ops_per_second", 0))
                
            elif event_type == "node_failure":
                self.network_metrics.add_failure_event("node_failure", metrics.get("node_id"))
                
            elif event_type == "byzantine_detected":
                node_id = metrics.get("node_id")
                if node_id:
                    self.network_metrics.byzantine_nodes.add(node_id)
                    self.network_metrics.add_failure_event("byzantine_behavior", node_id)
                    
            elif event_type == "network_partition":
                self.network_metrics.add_failure_event("network_partition")
                
            print(f"📊 Network event reported: {event_type}")
            
        except Exception as e:
            print(f"❌ Error reporting network event: {e}")
    
    async def get_adaptive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive adaptive consensus metrics"""
        try:
            network_summary = self.network_metrics.get_performance_summary()
            
            # Calculate strategy performance statistics
            strategy_stats = {}
            for strategy, performances in self.adaptive_metrics["strategy_performance"].items():
                if performances:
                    success_rate = sum(1 for p in performances if p['success']) / len(performances)
                    avg_time = statistics.mean(p['execution_time'] for p in performances)
                    avg_agreement = statistics.mean(p.get('agreement_ratio', 0) for p in performances)
                    
                    strategy_stats[strategy.value] = {
                        'success_rate': success_rate,
                        'average_time': avg_time,
                        'average_agreement': avg_agreement,
                        'usage_count': len(performances)
                    }
            
            return {
                **self.adaptive_metrics,
                "current_strategy": self.current_strategy.value,
                "network_condition": network_summary['condition'],
                "recommended_strategy": network_summary['recommended_strategy'],
                "network_summary": network_summary,
                "strategy_statistics": strategy_stats,
                "adaptation_frequency": (
                    self.adaptive_metrics["strategy_switches"] / 
                    max(1, self.adaptive_metrics["total_adaptive_consensus"])
                ),
                "performance_improvement_rate": (
                    self.adaptive_metrics["adaptation_improvements"] /
                    max(1, self.adaptive_metrics["strategy_switches"])
                )
            }
            
        except Exception as e:
            print(f"❌ Error getting adaptive metrics: {e}")
            return self.adaptive_metrics
    
    async def get_strategy_recommendations(self) -> Dict[str, Any]:
        """Get current strategy recommendations and reasoning"""
        try:
            summary = self.network_metrics.get_performance_summary()
            
            recommendations = {
                'current_strategy': self.current_strategy.value,
                'recommended_strategy': summary['recommended_strategy'],
                'network_condition': summary['condition'],
                'reasoning': [],
                'confidence': 0.0
            }
            
            # Generate reasoning
            if summary['network_size'] <= SMALL_NETWORK_THRESHOLD:
                recommendations['reasoning'].append(f"Small network ({summary['network_size']} nodes) - fast consensus suitable")
                recommendations['confidence'] += 0.3
                
            elif summary['network_size'] >= LARGE_NETWORK_THRESHOLD:
                recommendations['reasoning'].append(f"Large network ({summary['network_size']} nodes) - hierarchical consensus recommended")
                recommendations['confidence'] += 0.4
                
            if summary['failure_rate'] > HIGH_FAILURE_RATE_THRESHOLD:
                recommendations['reasoning'].append(f"High failure rate ({summary['failure_rate']:.1%}) - Byzantine resilient consensus needed")
                recommendations['confidence'] += 0.4
                
            if summary['avg_latency_ms'] > HIGH_LATENCY_THRESHOLD_MS:
                recommendations['reasoning'].append(f"High latency ({summary['avg_latency_ms']:.1f}ms) - optimization needed")
                recommendations['confidence'] += 0.2
                
            if summary['avg_throughput'] < LOW_THROUGHPUT_THRESHOLD:
                recommendations['reasoning'].append(f"Low throughput ({summary['avg_throughput']:.1f} ops/s) - hierarchical or hybrid consensus recommended")
                recommendations['confidence'] += 0.3
            
            # Cap confidence at 1.0
            recommendations['confidence'] = min(1.0, recommendations['confidence'])
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting strategy recommendations: {e}")
            return {'error': str(e)}


# === Global Adaptive Consensus Instance ===

_adaptive_consensus_instance: Optional[AdaptiveConsensusEngine] = None

def get_adaptive_consensus() -> AdaptiveConsensusEngine:
    """Get or create the global adaptive consensus instance"""
    global _adaptive_consensus_instance
    if _adaptive_consensus_instance is None:
        _adaptive_consensus_instance = AdaptiveConsensusEngine()
    return _adaptive_consensus_instance