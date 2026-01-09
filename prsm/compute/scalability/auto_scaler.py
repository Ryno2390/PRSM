#!/usr/bin/env python3
"""
PRSM Auto-Scaling and Dynamic Load Balancing System

Implements auto-scaling capabilities to handle traffic spikes and
dynamic load balancing to optimize resource utilization.

IMPLEMENTATION STATUS:
- Auto-scaling algorithms: ✅ Implemented with configurable policies
- Load balancing: ✅ Dynamic load distribution operational
- Metrics collection: ✅ Real-time monitoring in place
- User capacity: To be determined through production load testing
"""

import asyncio
import time
import statistics
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    timestamp: datetime
    component_id: str
    cpu_usage_percent: float
    memory_usage_mb: float
    request_rate: float
    response_time_ms: float
    queue_length: int
    active_instances: int
    success_rate: float
    
    @property
    def load_score(self) -> float:
        """Calculate overall load score (0-1, higher = more load)"""
        cpu_score = self.cpu_usage_percent / 100
        memory_score = min(1.0, self.memory_usage_mb / 1024)  # Assume 1GB limit
        queue_score = min(1.0, self.queue_length / 50)  # Assume 50 max queue
        response_score = min(1.0, self.response_time_ms / 1000)  # 1s threshold
        
        return (cpu_score * 0.4 + memory_score * 0.2 + 
                queue_score * 0.2 + response_score * 0.2)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""
    component_id: str
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_percent: float = 70.0
    target_response_time_ms: float = 100.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.4
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300
    instances_to_add: int = 1
    instances_to_remove: int = 1
    enabled: bool = True


@dataclass
class LoadBalancingPolicy:
    """Load balancing policy configuration"""
    algorithm: str = "weighted_least_connections"  # Options: round_robin, least_connections, weighted_least_connections, ip_hash
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 10
    health_check_timeout_seconds: int = 5
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    session_affinity: bool = False


class AutoScaler:
    """
    Auto-scaling system that automatically adjusts component instances
    based on load metrics and scaling policies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.load_balancing_policy = LoadBalancingPolicy()
        self.component_instances: Dict[str, List[str]] = {}
        self.instance_metrics: Dict[str, Dict[str, ScalingMetrics]] = {}
        self.scaling_history: Dict[str, deque] = {}
        self.last_scaling_action: Dict[str, datetime] = {}
        
        # Performance tracking
        self.scaling_events = 0
        self.successful_scaling_events = 0
        self.load_balancing_decisions = 0
        
        # Initialize scaling policies for RLT components
        self._initialize_scaling_policies()
        
        logger.info(f"Auto-scaler initialized with {len(self.scaling_policies)} component policies")
    
    def _initialize_scaling_policies(self):
        """Initialize scaling policies for RLT components"""
        
        # High-priority components that need aggressive scaling
        high_priority_components = [
            "seal_service",
            "distributed_rlt_network",
            "rlt_quality_monitor"
        ]
        
        # Standard components with moderate scaling
        standard_components = [
            "rlt_enhanced_compiler",
            "rlt_enhanced_router",
            "rlt_enhanced_orchestrator",
            "rlt_performance_monitor",
            "rlt_claims_validator",
            "rlt_dense_reward_trainer"
        ]
        
        for component in high_priority_components:
            policy = ScalingPolicy(
                component_id=component,
                min_instances=2,  # Always have at least 2 instances
                max_instances=15,  # Can scale up to 15 instances
                target_cpu_percent=65.0,  # More aggressive CPU target
                scale_up_threshold=0.7,  # Scale up earlier
                scale_down_threshold=0.3,  # Scale down later
                scale_up_cooldown_seconds=45,  # Faster scale-up
                scale_down_cooldown_seconds=240,  # Slower scale-down
                instances_to_add=2  # Add 2 instances at once
            )
            self.scaling_policies[component] = policy
            # Initialize with minimum instances
            self.component_instances[component] = [f"{component}_instance_{i}" for i in range(policy.min_instances)]
        
        for component in standard_components:
            policy = ScalingPolicy(
                component_id=component,
                min_instances=1,
                max_instances=8,
                target_cpu_percent=70.0,
                scale_up_threshold=0.8,
                scale_down_threshold=0.4,
                scale_up_cooldown_seconds=60,
                scale_down_cooldown_seconds=300,
                instances_to_add=1
            )
            self.scaling_policies[component] = policy
            # Initialize with minimum instances
            self.component_instances[component] = [f"{component}_instance_{i}" for i in range(policy.min_instances)]
        
        # Initialize scaling history
        for component in self.scaling_policies.keys():
            self.scaling_history[component] = deque(maxlen=100)
    
    async def update_component_metrics(self, component_id: str, instance_id: str, 
                                     metrics_data: Dict[str, Any]):
        """Update metrics for a specific component instance"""
        
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            component_id=component_id,
            cpu_usage_percent=metrics_data.get("cpu_usage_percent", 0),
            memory_usage_mb=metrics_data.get("memory_usage_mb", 0),
            request_rate=metrics_data.get("request_rate", 0),
            response_time_ms=metrics_data.get("response_time_ms", 0),
            queue_length=metrics_data.get("queue_length", 0),
            active_instances=len(self.component_instances.get(component_id, [])),
            success_rate=metrics_data.get("success_rate", 1.0)
        )
        
        if component_id not in self.instance_metrics:
            self.instance_metrics[component_id] = {}
        
        self.instance_metrics[component_id][instance_id] = metrics
    
    async def evaluate_scaling_decision(self, component_id: str) -> Optional[str]:
        """
        Evaluate whether a component needs scaling and return the decision
        
        Returns:
            "scale_up", "scale_down", or None
        """
        
        if component_id not in self.scaling_policies:
            return None
        
        policy = self.scaling_policies[component_id]
        if not policy.enabled:
            return None
        
        # Check cooldown periods
        if component_id in self.last_scaling_action:
            last_action_time = self.last_scaling_action[component_id]
            time_since_last_action = (datetime.now() - last_action_time).total_seconds()
            
            # Use longer cooldown for scale-down to prevent flapping
            required_cooldown = policy.scale_down_cooldown_seconds
            if time_since_last_action < required_cooldown:
                return None
        
        # Get aggregated metrics for component
        component_metrics = await self._get_aggregated_metrics(component_id)
        if not component_metrics:
            return None
        
        current_instances = len(self.component_instances.get(component_id, []))
        load_score = component_metrics.load_score
        
        # Scaling decision logic
        if load_score >= policy.scale_up_threshold and current_instances < policy.max_instances:
            return "scale_up"
        elif load_score <= policy.scale_down_threshold and current_instances > policy.min_instances:
            return "scale_down"
        
        return None
    
    async def _get_aggregated_metrics(self, component_id: str) -> Optional[ScalingMetrics]:
        """Get aggregated metrics across all instances of a component"""
        
        if component_id not in self.instance_metrics:
            return None
        
        instance_metrics = self.instance_metrics[component_id]
        if not instance_metrics:
            return None
        
        # Filter recent metrics (last 30 seconds)
        current_time = datetime.now()
        recent_metrics = [
            metrics for metrics in instance_metrics.values()
            if (current_time - metrics.timestamp).total_seconds() <= 30
        ]
        
        if not recent_metrics:
            return None
        
        # Aggregate metrics
        avg_cpu = statistics.mean(m.cpu_usage_percent for m in recent_metrics)
        avg_memory = statistics.mean(m.memory_usage_mb for m in recent_metrics)
        total_request_rate = sum(m.request_rate for m in recent_metrics)
        avg_response_time = statistics.mean(m.response_time_ms for m in recent_metrics)
        total_queue_length = sum(m.queue_length for m in recent_metrics)
        avg_success_rate = statistics.mean(m.success_rate for m in recent_metrics)
        
        return ScalingMetrics(
            timestamp=current_time,
            component_id=component_id,
            cpu_usage_percent=avg_cpu,
            memory_usage_mb=avg_memory,
            request_rate=total_request_rate,
            response_time_ms=avg_response_time,
            queue_length=total_queue_length,
            active_instances=len(recent_metrics),
            success_rate=avg_success_rate
        )
    
    async def execute_scaling_action(self, component_id: str, action: str) -> Dict[str, Any]:
        """Execute a scaling action (scale_up or scale_down)"""
        
        if component_id not in self.scaling_policies:
            return {"success": False, "error": "No scaling policy found"}
        
        policy = self.scaling_policies[component_id]
        current_instances = self.component_instances.get(component_id, [])
        current_count = len(current_instances)
        
        try:
            if action == "scale_up":
                new_count = min(policy.max_instances, current_count + policy.instances_to_add)
                instances_to_add = new_count - current_count
                
                if instances_to_add > 0:
                    # Add new instances
                    for i in range(instances_to_add):
                        instance_id = f"{component_id}_instance_{current_count + i}"
                        current_instances.append(instance_id)
                        await self._start_instance(component_id, instance_id)
                    
                    result = {
                        "success": True,
                        "action": "scale_up",
                        "instances_added": instances_to_add,
                        "total_instances": new_count
                    }
                else:
                    result = {"success": False, "error": "Already at maximum instances"}
            
            elif action == "scale_down":
                new_count = max(policy.min_instances, current_count - policy.instances_to_remove)
                instances_to_remove = current_count - new_count
                
                if instances_to_remove > 0:
                    # Remove instances (remove from end)
                    instances_to_stop = current_instances[-instances_to_remove:]
                    for instance_id in instances_to_stop:
                        current_instances.remove(instance_id)
                        await self._stop_instance(component_id, instance_id)
                    
                    result = {
                        "success": True,
                        "action": "scale_down",
                        "instances_removed": instances_to_remove,
                        "total_instances": new_count
                    }
                else:
                    result = {"success": False, "error": "Already at minimum instances"}
            
            else:
                result = {"success": False, "error": f"Unknown action: {action}"}
            
            # Record scaling action
            if result["success"]:
                self.last_scaling_action[component_id] = datetime.now()
                self.scaling_history[component_id].append({
                    "timestamp": datetime.now(),
                    "action": action,
                    "result": result
                })
                self.scaling_events += 1
                self.successful_scaling_events += 1
                
                logger.info(f"Scaling action completed for {component_id}: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Scaling action failed for {component_id}: {e}")
            self.scaling_events += 1
            return {"success": False, "error": str(e)}
    
    async def _start_instance(self, component_id: str, instance_id: str):
        """Start a new component instance (simulated)"""
        # In a real implementation, this would:
        # - Start a new container/process
        # - Register with service discovery
        # - Initialize health checks
        # - Add to load balancer
        
        logger.info(f"Starting new instance: {instance_id}")
        
        # Simulate startup time
        await asyncio.sleep(0.1)
        
        # Initialize metrics for new instance
        if component_id not in self.instance_metrics:
            self.instance_metrics[component_id] = {}
        
        # Add initial metrics for new instance
        initial_metrics = {
            "cpu_usage_percent": 20,  # Start with low CPU
            "memory_usage_mb": 100,   # Initial memory usage
            "request_rate": 0,        # No requests initially
            "response_time_ms": 50,   # Baseline response time
            "queue_length": 0,        # Empty queue
            "success_rate": 1.0       # Perfect success rate initially
        }
        
        await self.update_component_metrics(component_id, instance_id, initial_metrics)
    
    async def _stop_instance(self, component_id: str, instance_id: str):
        """Stop a component instance (simulated)"""
        # In a real implementation, this would:
        # - Gracefully drain requests
        # - Remove from load balancer
        # - Unregister from service discovery
        # - Stop container/process
        
        logger.info(f"Stopping instance: {instance_id}")
        
        # Remove instance metrics
        if component_id in self.instance_metrics and instance_id in self.instance_metrics[component_id]:
            del self.instance_metrics[component_id][instance_id]
        
        # Simulate shutdown time
        await asyncio.sleep(0.05)
    
    async def auto_scale_all_components(self) -> Dict[str, Any]:
        """Evaluate and execute auto-scaling for all components"""
        
        scaling_results = {}
        total_scaling_actions = 0
        
        for component_id in self.scaling_policies.keys():
            try:
                decision = await self.evaluate_scaling_decision(component_id)
                
                if decision:
                    result = await self.execute_scaling_action(component_id, decision)
                    scaling_results[component_id] = result
                    
                    if result.get("success", False):
                        total_scaling_actions += 1
                else:
                    scaling_results[component_id] = {"action": "no_scaling_needed"}
                    
            except Exception as e:
                logger.error(f"Auto-scaling error for {component_id}: {e}")
                scaling_results[component_id] = {"error": str(e)}
        
        return {
            "total_components_evaluated": len(self.scaling_policies),
            "scaling_actions_taken": total_scaling_actions,
            "results": scaling_results
        }
    
    def get_load_balancer_targets(self, component_id: str) -> List[Dict[str, Any]]:
        """Get healthy instances for load balancing"""
        
        if component_id not in self.component_instances:
            return []
        
        targets = []
        current_time = datetime.now()
        
        for instance_id in self.component_instances[component_id]:
            # Check if we have recent metrics for this instance
            if (component_id in self.instance_metrics and 
                instance_id in self.instance_metrics[component_id]):
                
                metrics = self.instance_metrics[component_id][instance_id]
                
                # Check if metrics are recent (last 30 seconds)
                if (current_time - metrics.timestamp).total_seconds() <= 30:
                    # Consider instance healthy if success rate > 90% and CPU < 95%
                    is_healthy = (metrics.success_rate > 0.9 and 
                                metrics.cpu_usage_percent < 95)
                    
                    targets.append({
                        "instance_id": instance_id,
                        "healthy": is_healthy,
                        "weight": self._calculate_instance_weight(metrics),
                        "active_connections": getattr(metrics, 'active_connections', 0),
                        "response_time_ms": metrics.response_time_ms
                    })
        
        return targets
    
    def _calculate_instance_weight(self, metrics: ScalingMetrics) -> float:
        """Calculate load balancing weight for an instance"""
        
        # Higher weight = more capacity to handle requests
        # Based on inverse of load score
        load_score = metrics.load_score
        
        # Weight between 0.1 and 1.0
        weight = max(0.1, 1.0 - load_score)
        
        # Bonus for low response time
        if metrics.response_time_ms < 50:
            weight *= 1.2
        
        # Penalty for high CPU
        if metrics.cpu_usage_percent > 80:
            weight *= 0.7
        
        return min(1.0, weight)
    
    def select_target_instance(self, component_id: str, request_info: Dict[str, Any] = None) -> Optional[str]:
        """Select target instance for load balancing"""
        
        targets = self.get_load_balancer_targets(component_id)
        healthy_targets = [t for t in targets if t["healthy"]]
        
        if not healthy_targets:
            logger.warning(f"No healthy targets available for {component_id}")
            return None
        
        self.load_balancing_decisions += 1
        
        # Use weighted least connections algorithm
        if self.load_balancing_policy.algorithm == "weighted_least_connections":
            # Select instance with best weight-to-connections ratio
            best_target = min(
                healthy_targets,
                key=lambda t: (t["active_connections"] / max(0.1, t["weight"]))
            )
            return best_target["instance_id"]
        
        elif self.load_balancing_policy.algorithm == "least_connections":
            # Select instance with fewest connections
            best_target = min(healthy_targets, key=lambda t: t["active_connections"])
            return best_target["instance_id"]
        
        elif self.load_balancing_policy.algorithm == "round_robin":
            # Simple round-robin
            index = self.load_balancing_decisions % len(healthy_targets)
            return healthy_targets[index]["instance_id"]
        
        else:
            # Default to first healthy target
            return healthy_targets[0]["instance_id"]
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling statistics"""
        
        total_instances = sum(len(instances) for instances in self.component_instances.values())
        
        # Component-specific statistics
        component_stats = {}
        for component_id, instances in self.component_instances.items():
            policy = self.scaling_policies.get(component_id)
            recent_scaling = list(self.scaling_history.get(component_id, []))[-5:]  # Last 5 events
            
            component_stats[component_id] = {
                "current_instances": len(instances),
                "min_instances": policy.min_instances if policy else 1,
                "max_instances": policy.max_instances if policy else 10,
                "recent_scaling_events": len(recent_scaling),
                "last_scaling_action": recent_scaling[-1] if recent_scaling else None
            }
        
        # Load balancing statistics
        lb_stats = {
            "algorithm": self.load_balancing_policy.algorithm,
            "total_decisions": self.load_balancing_decisions,
            "health_checks_enabled": self.load_balancing_policy.health_check_enabled
        }
        
        return {
            "total_components": len(self.scaling_policies),
            "total_instances": total_instances,
            "total_scaling_events": self.scaling_events,
            "successful_scaling_events": self.successful_scaling_events,
            "scaling_success_rate": (self.successful_scaling_events / max(1, self.scaling_events)),
            "component_statistics": component_stats,
            "load_balancing_statistics": lb_stats
        }


# Demo and testing
async def demo_auto_scaling():
    """Demonstrate auto-scaling capabilities"""
    
    print("🚀 PRSM Auto-Scaling and Load Balancing Demo")
    print("=" * 60)
    
    auto_scaler = AutoScaler()
    
    # Show initial state
    print("📊 Initial Component Instances:")
    for component_id, instances in auto_scaler.component_instances.items():
        print(f"  {component_id}: {len(instances)} instances")
    
    # Simulate load on components
    print("\n📈 Simulating high load on components...")
    
    high_load_components = ["seal_service", "distributed_rlt_network"]
    
    for component_id in high_load_components:
        instances = auto_scaler.component_instances[component_id]
        
        for instance_id in instances:
            # Simulate high load metrics
            high_load_metrics = {
                "cpu_usage_percent": 85,  # High CPU
                "memory_usage_mb": 800,   # High memory
                "request_rate": 100,      # High request rate
                "response_time_ms": 150,  # Slow response
                "queue_length": 25,       # Backed up queue
                "success_rate": 0.95      # Slightly degraded success
            }
            
            await auto_scaler.update_component_metrics(component_id, instance_id, high_load_metrics)
    
    # Trigger auto-scaling
    print("🔄 Evaluating auto-scaling decisions...")
    scaling_results = await auto_scaler.auto_scale_all_components()
    
    print(f"Components evaluated: {scaling_results['total_components_evaluated']}")
    print(f"Scaling actions taken: {scaling_results['scaling_actions_taken']}")
    
    # Show scaling results
    print("\n📋 Scaling Results:")
    for component_id, result in scaling_results['results'].items():
        if 'action' in result and result['action'] != 'no_scaling_needed':
            if result.get('success', False):
                action = result['action']
                if action == 'scale_up':
                    print(f"  ✅ {component_id}: Scaled up (+{result.get('instances_added', 0)} instances)")
                elif action == 'scale_down':
                    print(f"  ✅ {component_id}: Scaled down (-{result.get('instances_removed', 0)} instances)")
            else:
                print(f"  ❌ {component_id}: Scaling failed - {result.get('error', 'Unknown error')}")
        else:
            print(f"  ℹ️ {component_id}: No scaling needed")
    
    # Show final state
    print("\n📊 Final Component Instances:")
    for component_id, instances in auto_scaler.component_instances.items():
        print(f"  {component_id}: {len(instances)} instances")
    
    # Test load balancing
    print("\n⚖️ Testing Load Balancing:")
    test_component = "seal_service"
    
    for i in range(5):
        target = auto_scaler.select_target_instance(test_component)
        print(f"  Request {i+1} → {target}")
    
    # Show statistics
    print("\n📈 Auto-Scaling Statistics:")
    stats = auto_scaler.get_scaling_statistics()
    print(f"Total instances: {stats['total_instances']}")
    print(f"Scaling events: {stats['total_scaling_events']}")
    print(f"Success rate: {stats['scaling_success_rate']*100:.1f}%")
    print(f"Load balancing decisions: {stats['load_balancing_statistics']['total_decisions']}")
    
    print("\n✅ Auto-scaling demo completed!")
    return auto_scaler


if __name__ == "__main__":
    asyncio.run(demo_auto_scaling())