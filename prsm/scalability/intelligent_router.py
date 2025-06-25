#!/usr/bin/env python3
"""
PRSM Intelligent Routing System

High-performance routing system that directs traffic to optimal components
based on real-time performance metrics and load balancing algorithms.

Addresses the 300-user scalability breaking point with intelligent routing.
Expected performance improvement: 30%
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetrics:
    """Real-time metrics for a system component"""
    component_id: str
    timestamp: datetime
    throughput_ops_sec: float
    latency_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    success_rate: float
    active_connections: int
    queue_length: int = 0
    
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score for routing decisions"""
        # Weighted performance score
        throughput_weight = 0.4
        latency_weight = 0.3
        cpu_weight = 0.2
        success_weight = 0.1
        
        # Normalize metrics (higher is better)
        throughput_score = min(1.0, self.throughput_ops_sec / 10000)  # Target: 10k ops/sec
        latency_score = max(0.0, 1.0 - (self.latency_ms / 100))  # Target: <100ms
        cpu_score = max(0.0, 1.0 - (self.cpu_usage_percent / 100))  # Lower CPU is better
        success_score = self.success_rate  # Already 0-1
        
        return (throughput_score * throughput_weight + 
                latency_score * latency_weight + 
                cpu_score * cpu_weight + 
                success_score * success_weight)


@dataclass
class RoutingRule:
    """Routing rule configuration"""
    rule_id: str
    component_pattern: str
    weight: float = 1.0
    max_connections: int = 100
    health_threshold: float = 0.7
    enabled: bool = True


class IntelligentRouter:
    """
    Intelligent routing system that optimizes traffic distribution
    based on real-time component performance and availability
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.active_routes: Dict[str, str] = {}  # request_id -> component_id
        
        # Performance tracking
        self.total_requests_routed = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.routing_latency_history = deque(maxlen=1000)
        
        # Configuration
        self.health_check_interval = self.config.get("health_check_interval", 5)  # seconds
        self.performance_window = self.config.get("performance_window", 60)  # seconds
        self.load_balancing_algorithm = self.config.get("algorithm", "weighted_performance")
        
        # Component pool (RLT components from analysis)
        self.available_components = [
            "rlt_enhanced_compiler",
            "rlt_enhanced_router", 
            "rlt_enhanced_orchestrator",
            "rlt_performance_monitor",
            "rlt_claims_validator",
            "rlt_dense_reward_trainer",
            "rlt_quality_monitor",
            "distributed_rlt_network",
            "seal_rlt_enhanced_teacher"
        ]
        
        # Initialize routing rules
        self._initialize_routing_rules()
        
        logger.info(f"Intelligent router initialized with {len(self.available_components)} components")
    
    def _initialize_routing_rules(self):
        """Initialize default routing rules for RLT components"""
        
        # High-performance components get higher weights
        high_performance_components = [
            "rlt_enhanced_compiler",
            "rlt_enhanced_router",
            "rlt_enhanced_orchestrator"
        ]
        
        # CPU-constrained components get lower weights initially
        cpu_constrained_components = [
            "seal_rlt_enhanced_teacher",
            "distributed_rlt_network",
            "rlt_quality_monitor"
        ]
        
        for component in self.available_components:
            if component in high_performance_components:
                weight = 2.0  # Higher priority for high-performance components
                max_connections = 150
            elif component in cpu_constrained_components:
                weight = 0.7  # Lower priority for CPU-constrained components
                max_connections = 80
            else:
                weight = 1.0  # Default weight
                max_connections = 100
            
            rule = RoutingRule(
                rule_id=f"rule_{component}",
                component_pattern=component,
                weight=weight,
                max_connections=max_connections,
                health_threshold=0.7
            )
            self.routing_rules[component] = rule
    
    async def update_component_metrics(self, component_id: str, metrics_data: Dict[str, Any]):
        """Update real-time metrics for a component"""
        
        metrics = ComponentMetrics(
            component_id=component_id,
            timestamp=datetime.now(),
            throughput_ops_sec=metrics_data.get("throughput_ops_sec", 0),
            latency_ms=metrics_data.get("latency_ms", 0),
            cpu_usage_percent=metrics_data.get("cpu_usage_percent", 0),
            memory_usage_mb=metrics_data.get("memory_usage_mb", 0),
            success_rate=metrics_data.get("success_rate", 1.0),
            active_connections=metrics_data.get("active_connections", 0),
            queue_length=metrics_data.get("queue_length", 0)
        )
        
        self.component_metrics[component_id] = metrics
        self.metrics_history[component_id].append(metrics)
        
        # Auto-adjust routing weights based on performance
        await self._adjust_routing_weights(component_id, metrics)
    
    async def _adjust_routing_weights(self, component_id: str, metrics: ComponentMetrics):
        """Dynamically adjust routing weights based on component performance"""
        
        if component_id not in self.routing_rules:
            return
        
        rule = self.routing_rules[component_id]
        performance_score = metrics.performance_score
        
        # Adjust weight based on performance score
        if performance_score > 0.8:
            # High performance - increase weight
            new_weight = min(3.0, rule.weight * 1.1)
        elif performance_score < 0.5:
            # Poor performance - decrease weight
            new_weight = max(0.3, rule.weight * 0.9)
        else:
            # Moderate performance - gradual adjustment
            new_weight = rule.weight
        
        # Update rule if weight changed significantly
        if abs(new_weight - rule.weight) > 0.1:
            rule.weight = new_weight
            logger.info(f"Adjusted weight for {component_id}: {rule.weight:.2f} (performance: {performance_score:.2f})")
    
    async def route_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """
        Route a request to the optimal component based on current performance
        
        Args:
            request_data: Request information including type, priority, etc.
            
        Returns:
            Component ID to route to, or None if no suitable component available
        """
        start_time = time.time()
        
        try:
            # Get available healthy components
            healthy_components = await self._get_healthy_components()
            
            if not healthy_components:
                logger.warning("No healthy components available for routing")
                self.failed_routes += 1
                return None
            
            # Select best component based on algorithm
            selected_component = await self._select_optimal_component(
                healthy_components, 
                request_data
            )
            
            if selected_component:
                # Track the route
                request_id = request_data.get("request_id", f"req_{time.time()}")
                self.active_routes[request_id] = selected_component
                self.total_requests_routed += 1
                self.successful_routes += 1
                
                # Update component load
                if selected_component in self.component_metrics:
                    self.component_metrics[selected_component].active_connections += 1
                
                routing_time = (time.time() - start_time) * 1000  # ms
                self.routing_latency_history.append(routing_time)
                
                logger.debug(f"Routed request {request_id} to {selected_component} in {routing_time:.2f}ms")
                return selected_component
            else:
                self.failed_routes += 1
                return None
                
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            self.failed_routes += 1
            return None
    
    async def _get_healthy_components(self) -> List[str]:
        """Get list of healthy components available for routing"""
        
        healthy_components = []
        current_time = datetime.now()
        
        for component_id in self.available_components:
            # Check if we have recent metrics
            if component_id not in self.component_metrics:
                continue
            
            metrics = self.component_metrics[component_id]
            rule = self.routing_rules.get(component_id)
            
            if not rule or not rule.enabled:
                continue
            
            # Check metric freshness (within last 30 seconds)
            if (current_time - metrics.timestamp).total_seconds() > 30:
                continue
            
            # Check health thresholds
            performance_score = metrics.performance_score
            if performance_score < rule.health_threshold:
                continue
            
            # Check connection limits
            if metrics.active_connections >= rule.max_connections:
                continue
            
            healthy_components.append(component_id)
        
        return healthy_components
    
    async def _select_optimal_component(self, healthy_components: List[str], 
                                      request_data: Dict[str, Any]) -> Optional[str]:
        """Select the optimal component for the request"""
        
        if not healthy_components:
            return None
        
        if self.load_balancing_algorithm == "weighted_performance":
            return await self._weighted_performance_selection(healthy_components, request_data)
        elif self.load_balancing_algorithm == "least_connections":
            return await self._least_connections_selection(healthy_components)
        elif self.load_balancing_algorithm == "round_robin":
            return await self._round_robin_selection(healthy_components)
        else:
            # Default to weighted performance
            return await self._weighted_performance_selection(healthy_components, request_data)
    
    async def _weighted_performance_selection(self, components: List[str], 
                                           request_data: Dict[str, Any]) -> str:
        """Select component based on weighted performance scoring"""
        
        component_scores = []
        
        for component_id in components:
            metrics = self.component_metrics[component_id]
            rule = self.routing_rules[component_id]
            
            # Calculate weighted score
            performance_score = metrics.performance_score
            weight = rule.weight
            
            # Penalty for high load
            load_penalty = min(0.5, metrics.active_connections / rule.max_connections)
            
            # Request type affinity (if specified)
            affinity_bonus = 0.0
            request_type = request_data.get("type", "")
            if request_type and self._has_affinity(component_id, request_type):
                affinity_bonus = 0.2
            
            final_score = (performance_score * weight + affinity_bonus) * (1 - load_penalty)
            component_scores.append((component_id, final_score))
        
        # Sort by score (highest first)
        component_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best component
        return component_scores[0][0]
    
    async def _least_connections_selection(self, components: List[str]) -> str:
        """Select component with least active connections"""
        
        min_connections = float('inf')
        best_component = None
        
        for component_id in components:
            metrics = self.component_metrics[component_id]
            if metrics.active_connections < min_connections:
                min_connections = metrics.active_connections
                best_component = component_id
        
        return best_component
    
    async def _round_robin_selection(self, components: List[str]) -> str:
        """Simple round-robin selection"""
        
        # Simple round-robin based on total requests
        index = self.total_requests_routed % len(components)
        return components[index]
    
    def _has_affinity(self, component_id: str, request_type: str) -> bool:
        """Check if component has affinity for request type"""
        
        # Define component affinities
        affinities = {
            "rlt_enhanced_compiler": ["compilation", "code", "build"],
            "rlt_enhanced_router": ["routing", "dispatch", "coordination"],
            "rlt_enhanced_orchestrator": ["orchestration", "workflow", "coordination"],
            "rlt_performance_monitor": ["monitoring", "metrics", "performance"],
            "rlt_claims_validator": ["validation", "verification", "audit"],
            "rlt_dense_reward_trainer": ["training", "learning", "optimization"],
            "rlt_quality_monitor": ["quality", "assessment", "evaluation"],
            "distributed_rlt_network": ["networking", "distributed", "p2p"],
            "seal_rlt_enhanced_teacher": ["teaching", "education", "instruction"]
        }
        
        component_types = affinities.get(component_id, [])
        return any(req_type in request_type.lower() for req_type in component_types)
    
    async def complete_request(self, request_id: str, success: bool = True):
        """Mark a request as completed and update metrics"""
        
        if request_id in self.active_routes:
            component_id = self.active_routes[request_id]
            
            # Update component metrics
            if component_id in self.component_metrics:
                self.component_metrics[component_id].active_connections -= 1
                self.component_metrics[component_id].active_connections = max(
                    0, self.component_metrics[component_id].active_connections
                )
            
            # Clean up
            del self.active_routes[request_id]
            
            if not success:
                self.failed_routes += 1
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        
        total_requests = self.total_requests_routed
        success_rate = self.successful_routes / total_requests if total_requests > 0 else 0
        
        # Calculate average routing latency
        avg_routing_latency = (
            statistics.mean(self.routing_latency_history) 
            if self.routing_latency_history else 0
        )
        
        # Component distribution
        component_distribution = defaultdict(int)
        for component_id in self.active_routes.values():
            component_distribution[component_id] += 1
        
        # Performance by component
        component_performance = {}
        for component_id, metrics in self.component_metrics.items():
            component_performance[component_id] = {
                "performance_score": metrics.performance_score,
                "throughput_ops_sec": metrics.throughput_ops_sec,
                "latency_ms": metrics.latency_ms,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "active_connections": metrics.active_connections,
                "routing_weight": self.routing_rules.get(component_id, {}).weight if component_id in self.routing_rules else 1.0
            }
        
        return {
            "total_requests_routed": total_requests,
            "successful_routes": self.successful_routes,
            "failed_routes": self.failed_routes,
            "success_rate": success_rate,
            "avg_routing_latency_ms": avg_routing_latency,
            "active_routes_count": len(self.active_routes),
            "component_distribution": dict(component_distribution),
            "component_performance": component_performance,
            "routing_algorithm": self.load_balancing_algorithm,
            "healthy_components_count": len([
                c for c in self.available_components 
                if c in self.component_metrics and 
                self.component_metrics[c].performance_score >= 0.7
            ])
        }
    
    async def optimize_routing_configuration(self):
        """Automatically optimize routing configuration based on historical performance"""
        
        logger.info("🔧 Optimizing routing configuration based on performance history...")
        
        for component_id in self.available_components:
            if component_id not in self.metrics_history or len(self.metrics_history[component_id]) < 10:
                continue
            
            # Analyze historical performance
            recent_metrics = list(self.metrics_history[component_id])[-10:]
            
            avg_performance = statistics.mean(m.performance_score for m in recent_metrics)
            avg_throughput = statistics.mean(m.throughput_ops_sec for m in recent_metrics)
            avg_cpu = statistics.mean(m.cpu_usage_percent for m in recent_metrics)
            
            rule = self.routing_rules.get(component_id)
            if not rule:
                continue
            
            # Optimize max connections based on CPU usage
            if avg_cpu > 80:
                # High CPU - reduce max connections
                new_max_connections = max(50, int(rule.max_connections * 0.8))
            elif avg_cpu < 50:
                # Low CPU - can handle more connections
                new_max_connections = min(200, int(rule.max_connections * 1.2))
            else:
                new_max_connections = rule.max_connections
            
            # Optimize weight based on performance
            if avg_performance > 0.8:
                new_weight = min(3.0, rule.weight * 1.1)
            elif avg_performance < 0.5:
                new_weight = max(0.2, rule.weight * 0.9)
            else:
                new_weight = rule.weight
            
            # Apply optimizations
            if new_max_connections != rule.max_connections or abs(new_weight - rule.weight) > 0.1:
                logger.info(f"Optimized {component_id}: weight {rule.weight:.2f}→{new_weight:.2f}, "
                           f"max_conn {rule.max_connections}→{new_max_connections}")
                
                rule.max_connections = new_max_connections
                rule.weight = new_weight


# Example usage and testing
async def demo_intelligent_routing():
    """Demonstrate intelligent routing capabilities"""
    
    print("🚀 PRSM Intelligent Routing System Demo")
    print("=" * 60)
    
    router = IntelligentRouter({
        "algorithm": "weighted_performance",
        "health_check_interval": 5
    })
    
    # Simulate component metrics updates
    print("📊 Updating component metrics...")
    
    for i, component in enumerate(router.available_components):
        # Simulate realistic metrics (some components performing better than others)
        base_throughput = 6000 + (i * 200) + (i % 3 * 500)  # Varied performance
        cpu_usage = 40 + (i * 5) + (i % 2 * 20)  # Some CPU-constrained
        
        metrics = {
            "throughput_ops_sec": base_throughput,
            "latency_ms": 20 + (i * 2),
            "cpu_usage_percent": min(90, cpu_usage),
            "memory_usage_mb": 200 + (i * 30),
            "success_rate": 0.98 - (i * 0.01),
            "active_connections": i * 5,
            "queue_length": i * 2
        }
        
        await router.update_component_metrics(component, metrics)
    
    # Test routing decisions
    print("🎯 Testing routing decisions...")
    
    test_requests = [
        {"request_id": "req_1", "type": "compilation", "priority": "high"},
        {"request_id": "req_2", "type": "monitoring", "priority": "medium"},
        {"request_id": "req_3", "type": "validation", "priority": "low"},
        {"request_id": "req_4", "type": "orchestration", "priority": "high"},
        {"request_id": "req_5", "type": "training", "priority": "medium"}
    ]
    
    for request in test_requests:
        component = await router.route_request(request)
        print(f"Request {request['request_id']} ({request['type']}) → {component}")
    
    # Show routing statistics
    print("\n📈 Routing Statistics:")
    stats = router.get_routing_statistics()
    
    print(f"Total requests routed: {stats['total_requests_routed']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Average routing latency: {stats['avg_routing_latency_ms']:.2f}ms")
    print(f"Healthy components: {stats['healthy_components_count']}")
    
    print("\n🏆 Component Performance Ranking:")
    perf_ranking = sorted(
        stats['component_performance'].items(),
        key=lambda x: x[1]['performance_score'],
        reverse=True
    )
    
    for i, (component, perf) in enumerate(perf_ranking[:5], 1):
        print(f"{i}. {component}: {perf['performance_score']:.2f} "
              f"({perf['throughput_ops_sec']:.0f} ops/sec, "
              f"{perf['cpu_usage_percent']:.1f}% CPU)")
    
    # Test optimization
    print("\n🔧 Running routing optimization...")
    await router.optimize_routing_configuration()
    
    print("\n✅ Intelligent routing demo completed!")
    return router


if __name__ == "__main__":
    asyncio.run(demo_intelligent_routing())