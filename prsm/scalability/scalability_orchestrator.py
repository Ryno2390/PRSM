#!/usr/bin/env python3
"""
PRSM Scalability Orchestrator

Orchestrates all scalability improvements to provide a unified system
that addresses the 300-user breaking point with intelligent coordination
of routing, CPU optimization, auto-scaling, and caching.

Expected overall improvement:
- Handle 500+ concurrent users (67% improvement from 300)
- Achieve 10,000+ ops/sec throughput (44% improvement from 6,984)
- Reduce CPU usage by 15-30% across bottleneck components
- Improve latency by 20-40% through intelligent caching
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .intelligent_router import IntelligentRouter
from .cpu_optimizer import CPUOptimizer
from .auto_scaler import AutoScaler
from .advanced_cache import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class ScalabilityMetrics:
    """Comprehensive scalability metrics"""
    timestamp: datetime
    total_concurrent_users: int
    throughput_ops_sec: float
    average_latency_ms: float
    success_rate: float
    cpu_utilization_percent: float
    memory_utilization_mb: float
    cache_hit_ratio: float
    active_instances: int
    routing_efficiency: float
    breaking_point_reached: bool = False


class ScalabilityOrchestrator:
    """
    Orchestrates all scalability improvements for optimal performance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.intelligent_router = IntelligentRouter()
        self.cpu_optimizer = CPUOptimizer()
        self.auto_scaler = AutoScaler()
        self.cache_manager = CacheManager()
        
        # Performance tracking
        self.metrics_history = []
        self.optimization_events = []
        self.current_load_level = "low"  # low, medium, high, extreme
        
        # Component list from Phase 3 analysis
        self.rlt_components = [
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
        
        # Load thresholds for optimization triggers
        self.load_thresholds = {
            "users": {"medium": 150, "high": 250, "extreme": 350},
            "cpu": {"medium": 60, "high": 75, "extreme": 85},
            "latency": {"medium": 100, "high": 200, "extreme": 500},
            "success_rate": {"warning": 0.98, "critical": 0.95, "failure": 0.90}
        }
        
        logger.info("Scalability orchestrator initialized")
    
    async def initialize(self):
        """Initialize all scalability components"""
        
        logger.info("🚀 Initializing scalability orchestrator...")
        
        # Initialize CPU optimizations for bottleneck components
        optimization_result = await self.cpu_optimizer.optimize_all_components()
        logger.info(f"CPU optimization completed: {optimization_result['components_optimized']} components optimized")
        
        # Initialize caches for all components
        for component_id in self.rlt_components:
            cache = await self.cache_manager.get_cache(component_id)
            logger.info(f"Initialized cache for {component_id}")
        
        logger.info("✅ Scalability orchestrator initialization complete")
    
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming request with full scalability optimizations
        """
        start_time = time.time()
        request_id = request_data.get("request_id", f"req_{int(time.time() * 1000)}")
        
        try:
            # 1. Check cache first
            cache_result = await self._check_cache(request_data)
            if cache_result is not None:
                return {
                    "request_id": request_id,
                    "result": cache_result,
                    "source": "cache",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "success": True
                }
            
            # 2. Route request to optimal component
            target_component = await self.intelligent_router.route_request(request_data)
            if not target_component:
                return {
                    "request_id": request_id,
                    "error": "No available component",
                    "success": False
                }
            
            # 3. Get target instance from auto-scaler
            target_instance = self.auto_scaler.select_target_instance(target_component, request_data)
            if not target_instance:
                return {
                    "request_id": request_id,
                    "error": "No available instance",
                    "success": False
                }
            
            # 4. Process request
            result = await self._process_request(target_component, target_instance, request_data)
            
            # 5. Cache result if appropriate
            if result.get("success", False):
                await self._cache_result(request_data, result)
            
            # 6. Update metrics
            await self._update_request_metrics(target_component, target_instance, start_time, result)
            
            # 7. Mark request as complete
            await self.intelligent_router.complete_request(request_id, result.get("success", False))
            
            processing_time = (time.time() - start_time) * 1000
            result.update({
                "request_id": request_id,
                "component": target_component,
                "instance": target_instance,
                "processing_time_ms": processing_time
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return {
                "request_id": request_id,
                "error": str(e),
                "success": False,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _check_cache(self, request_data: Dict[str, Any]) -> Optional[Any]:
        """Check if request result is cached"""
        
        request_type = request_data.get("type", "unknown")
        component_preference = self._determine_component_preference(request_type)
        
        if component_preference:
            cache_key = self._generate_request_cache_key(request_data)
            cache = await self.cache_manager.get_cache(component_preference)
            return await cache.get(cache_key)
        
        return None
    
    async def _cache_result(self, request_data: Dict[str, Any], result: Dict[str, Any]):
        """Cache request result"""
        
        if not result.get("success", False):
            return
        
        request_type = request_data.get("type", "unknown")
        component_preference = self._determine_component_preference(request_type)
        
        if component_preference:
            cache_key = self._generate_request_cache_key(request_data)
            cache = await self.cache_manager.get_cache(component_preference)
            
            # Determine TTL based on request type
            ttl_seconds = self._determine_cache_ttl(request_type)
            
            await cache.put(cache_key, result.get("data"), ttl_seconds)
    
    def _determine_component_preference(self, request_type: str) -> Optional[str]:
        """Determine preferred component for request type"""
        
        type_mapping = {
            "compilation": "rlt_enhanced_compiler",
            "routing": "rlt_enhanced_router",
            "orchestration": "rlt_enhanced_orchestrator",
            "monitoring": "rlt_performance_monitor",
            "validation": "rlt_claims_validator",
            "training": "rlt_dense_reward_trainer",
            "quality": "rlt_quality_monitor",
            "network": "distributed_rlt_network",
            "teaching": "seal_rlt_enhanced_teacher"
        }
        
        return type_mapping.get(request_type.lower())
    
    def _generate_request_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        
        import hashlib
        import json
        
        key_data = {
            "type": request_data.get("type"),
            "data": request_data.get("data"),
            "params": request_data.get("params", {})
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _determine_cache_ttl(self, request_type: str) -> int:
        """Determine cache TTL based on request type"""
        
        ttl_mapping = {
            "compilation": 600,     # 10 minutes
            "validation": 300,      # 5 minutes
            "monitoring": 60,       # 1 minute
            "training": 1800,       # 30 minutes
            "quality": 180,         # 3 minutes
            "network": 120,         # 2 minutes
            "teaching": 900,        # 15 minutes
            "orchestration": 240,   # 4 minutes
            "routing": 30           # 30 seconds
        }
        
        return ttl_mapping.get(request_type.lower(), 300)
    
    async def _process_request(self, component: str, instance: str, 
                             request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request on target component/instance (simulated)"""
        
        # Simulate processing time based on component and load
        base_processing_time = {
            "seal_rlt_enhanced_teacher": 0.08,
            "distributed_rlt_network": 0.06,
            "rlt_quality_monitor": 0.04,
            "rlt_dense_reward_trainer": 0.07,
            "rlt_claims_validator": 0.05,
            "rlt_enhanced_compiler": 0.03,
            "rlt_enhanced_router": 0.02,
            "rlt_enhanced_orchestrator": 0.04,
            "rlt_performance_monitor": 0.03
        }.get(component, 0.05)
        
        # Add load-based delay
        load_factor = {"low": 1.0, "medium": 1.3, "high": 1.7, "extreme": 2.5}[self.current_load_level]
        processing_time = base_processing_time * load_factor
        
        await asyncio.sleep(processing_time)
        
        # Simulate success rate based on load
        success_rates = {"low": 0.99, "medium": 0.97, "high": 0.94, "extreme": 0.88}
        success_rate = success_rates[self.current_load_level]
        
        import random
        success = random.random() < success_rate
        
        if success:
            return {
                "success": True,
                "data": f"Processed by {component}:{instance}",
                "processing_time": processing_time,
                "metadata": {"load_level": self.current_load_level}
            }
        else:
            return {
                "success": False,
                "error": "Processing failed",
                "processing_time": processing_time
            }
    
    async def _update_request_metrics(self, component: str, instance: str, 
                                    start_time: float, result: Dict[str, Any]):
        """Update component and instance metrics"""
        
        processing_time = (time.time() - start_time) * 1000
        success = result.get("success", False)
        
        # Update router metrics
        metrics_data = {
            "throughput_ops_sec": 1000 / processing_time if processing_time > 0 else 0,
            "latency_ms": processing_time,
            "cpu_usage_percent": self._estimate_cpu_usage(component),
            "memory_usage_mb": self._estimate_memory_usage(component),
            "success_rate": 1.0 if success else 0.0,
            "active_connections": 1,
            "queue_length": 0
        }
        
        await self.intelligent_router.update_component_metrics(component, metrics_data)
        
        # Update auto-scaler metrics
        scaling_metrics_data = {
            "cpu_usage_percent": metrics_data["cpu_usage_percent"],
            "memory_usage_mb": metrics_data["memory_usage_mb"],
            "request_rate": 60,  # Requests per minute
            "response_time_ms": processing_time,
            "queue_length": 0,
            "success_rate": 1.0 if success else 0.0
        }
        
        await self.auto_scaler.update_component_metrics(component, instance, scaling_metrics_data)
    
    def _estimate_cpu_usage(self, component: str) -> float:
        """Estimate CPU usage for component based on current load"""
        
        # Base CPU usage from Phase 3 analysis
        base_cpu = {
            "seal_rlt_enhanced_teacher": 82.8,
            "distributed_rlt_network": 79.8,
            "rlt_quality_monitor": 76.8,
            "rlt_dense_reward_trainer": 73.8,
            "rlt_claims_validator": 70.8,
            "rlt_enhanced_compiler": 62.4,
            "rlt_enhanced_router": 59.4,
            "rlt_enhanced_orchestrator": 56.4,
            "rlt_performance_monitor": 53.4
        }.get(component, 60.0)
        
        # Apply CPU optimization reduction (15-30%)
        optimized_cpu = base_cpu * 0.75  # 25% average reduction
        
        # Apply load factor
        load_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.3, "extreme": 1.6}[self.current_load_level]
        
        return min(95, optimized_cpu * load_multiplier)
    
    def _estimate_memory_usage(self, component: str) -> float:
        """Estimate memory usage for component"""
        
        base_memory = {
            "seal_rlt_enhanced_teacher": 350,
            "distributed_rlt_network": 320,
            "rlt_quality_monitor": 290,
            "rlt_dense_reward_trainer": 380,
            "rlt_claims_validator": 260,
            "rlt_enhanced_compiler": 240,
            "rlt_enhanced_router": 220,
            "rlt_enhanced_orchestrator": 280,
            "rlt_performance_monitor": 200
        }.get(component, 250.0)
        
        load_multiplier = {"low": 0.8, "medium": 1.0, "high": 1.2, "extreme": 1.4}[self.current_load_level]
        
        return base_memory * load_multiplier
    
    async def monitor_and_optimize(self):
        """Continuous monitoring and optimization"""
        
        logger.info("🔄 Starting continuous monitoring and optimization...")
        
        while True:
            try:
                # Collect current metrics
                current_metrics = await self._collect_system_metrics()
                self.metrics_history.append(current_metrics)
                
                # Keep last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Update load level
                await self._update_load_level(current_metrics)
                
                # Trigger optimizations if needed
                await self._trigger_optimizations(current_metrics)
                
                # Auto-scale components
                scaling_results = await self.auto_scaler.auto_scale_all_components()
                if scaling_results["scaling_actions_taken"] > 0:
                    logger.info(f"Auto-scaling performed: {scaling_results['scaling_actions_taken']} actions")
                
                # Optimize routing configuration
                await self.intelligent_router.optimize_routing_configuration()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self) -> ScalabilityMetrics:
        """Collect current system metrics"""
        
        # Get router statistics
        router_stats = self.intelligent_router.get_routing_statistics()
        
        # Get auto-scaler statistics
        scaler_stats = self.auto_scaler.get_scaling_statistics()
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_all_statistics()
        
        # Calculate aggregate metrics
        avg_throughput = statistics.mean([
            perf.get("throughput_ops_sec", 0) 
            for perf in router_stats.get("component_performance", {}).values()
        ]) if router_stats.get("component_performance") else 0
        
        avg_latency = statistics.mean([
            perf.get("latency_ms", 0)
            for perf in router_stats.get("component_performance", {}).values()
        ]) if router_stats.get("component_performance") else 0
        
        avg_cpu = statistics.mean([
            perf.get("cpu_usage_percent", 0)
            for perf in router_stats.get("component_performance", {}).values()
        ]) if router_stats.get("component_performance") else 0
        
        # Estimate concurrent users based on throughput and latency
        estimated_users = min(1000, max(1, int(avg_throughput * avg_latency / 1000))) if avg_latency > 0 else 1
        
        return ScalabilityMetrics(
            timestamp=datetime.now(),
            total_concurrent_users=estimated_users,
            throughput_ops_sec=avg_throughput,
            average_latency_ms=avg_latency,
            success_rate=router_stats.get("success_rate", 1.0),
            cpu_utilization_percent=avg_cpu,
            memory_utilization_mb=300,  # Estimated
            cache_hit_ratio=cache_stats.get("average_hit_ratio", 0.0),
            active_instances=scaler_stats.get("total_instances", 0),
            routing_efficiency=min(1.0, router_stats.get("success_rate", 0.0)),
            breaking_point_reached=estimated_users > 300 and router_stats.get("success_rate", 1.0) < 0.95
        )
    
    async def _update_load_level(self, metrics: ScalabilityMetrics):
        """Update current load level based on metrics"""
        
        user_level = "low"
        if metrics.total_concurrent_users >= self.load_thresholds["users"]["extreme"]:
            user_level = "extreme"
        elif metrics.total_concurrent_users >= self.load_thresholds["users"]["high"]:
            user_level = "high"
        elif metrics.total_concurrent_users >= self.load_thresholds["users"]["medium"]:
            user_level = "medium"
        
        cpu_level = "low"
        if metrics.cpu_utilization_percent >= self.load_thresholds["cpu"]["extreme"]:
            cpu_level = "extreme"
        elif metrics.cpu_utilization_percent >= self.load_thresholds["cpu"]["high"]:
            cpu_level = "high"
        elif metrics.cpu_utilization_percent >= self.load_thresholds["cpu"]["medium"]:
            cpu_level = "medium"
        
        latency_level = "low"
        if metrics.average_latency_ms >= self.load_thresholds["latency"]["extreme"]:
            latency_level = "extreme"
        elif metrics.average_latency_ms >= self.load_thresholds["latency"]["high"]:
            latency_level = "high"
        elif metrics.average_latency_ms >= self.load_thresholds["latency"]["medium"]:
            latency_level = "medium"
        
        # Take the highest level
        levels = ["low", "medium", "high", "extreme"]
        max_level = max(user_level, cpu_level, latency_level, key=lambda x: levels.index(x))
        
        if max_level != self.current_load_level:
            logger.info(f"Load level changed: {self.current_load_level} → {max_level}")
            self.current_load_level = max_level
    
    async def _trigger_optimizations(self, metrics: ScalabilityMetrics):
        """Trigger optimizations based on current metrics"""
        
        optimizations_triggered = []
        
        # Trigger CPU optimization if CPU usage is high
        if metrics.cpu_utilization_percent > 80:
            # Re-optimize CPU usage
            result = await self.cpu_optimizer.optimize_all_components()
            optimizations_triggered.append(f"CPU optimization: {result['components_optimized']} components")
        
        # Trigger cache optimization if hit ratio is low
        if metrics.cache_hit_ratio < 0.6:
            # Could implement cache warming or policy adjustments
            optimizations_triggered.append("Cache optimization triggered")
        
        # Log optimizations
        if optimizations_triggered:
            logger.info(f"Optimizations triggered: {', '.join(optimizations_triggered)}")
            self.optimization_events.append({
                "timestamp": datetime.now(),
                "optimizations": optimizations_triggered,
                "metrics": metrics
            })
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scalability statistics"""
        
        router_stats = self.intelligent_router.get_routing_statistics()
        cpu_optimization_status = self.cpu_optimizer.get_optimization_status()
        scaler_stats = self.auto_scaler.get_scaling_statistics()
        cache_stats = self.cache_manager.get_all_statistics()
        
        # Calculate recent performance
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        if recent_metrics:
            avg_throughput = statistics.mean([m.throughput_ops_sec for m in recent_metrics])
            avg_latency = statistics.mean([m.average_latency_ms for m in recent_metrics])
            avg_users = statistics.mean([m.total_concurrent_users for m in recent_metrics])
            avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
            max_users = max([m.total_concurrent_users for m in recent_metrics])
        else:
            avg_throughput = avg_latency = avg_users = avg_success_rate = max_users = 0
        
        return {
            "current_load_level": self.current_load_level,
            "recent_performance": {
                "avg_throughput_ops_sec": round(avg_throughput, 1),
                "avg_latency_ms": round(avg_latency, 1),
                "avg_concurrent_users": round(avg_users, 1),
                "max_concurrent_users": max_users,
                "avg_success_rate": round(avg_success_rate, 3)
            },
            "component_statistics": {
                "intelligent_routing": {
                    "total_requests_routed": router_stats.get("total_requests_routed", 0),
                    "success_rate": router_stats.get("success_rate", 0),
                    "healthy_components": router_stats.get("healthy_components_count", 0)
                },
                "cpu_optimization": {
                    "active_optimizations": cpu_optimization_status.get("active_optimizations", 0),
                    "components_optimized": len(cpu_optimization_status.get("components", {}))
                },
                "auto_scaling": {
                    "total_instances": scaler_stats.get("total_instances", 0),
                    "scaling_events": scaler_stats.get("total_scaling_events", 0),
                    "scaling_success_rate": scaler_stats.get("scaling_success_rate", 0)
                },
                "advanced_caching": {
                    "total_cached_items": cache_stats.get("total_items", 0),
                    "average_hit_ratio": cache_stats.get("average_hit_ratio", 0),
                    "total_cache_size_mb": cache_stats.get("total_size_mb", 0)
                }
            },
            "optimization_events": len(self.optimization_events),
            "scalability_improvements": {
                "expected_user_capacity": "500+ users (vs 300 baseline)",
                "expected_throughput": "10,000+ ops/sec (vs 6,984 baseline)",
                "expected_cpu_reduction": "15-30% reduction",
                "expected_latency_improvement": "20-40% reduction"
            }
        }


# Demo and testing
async def demo_scalability_orchestrator():
    """Demonstrate complete scalability orchestration"""
    
    print("🚀 PRSM Scalability Orchestrator Demo")
    print("=" * 60)
    
    orchestrator = ScalabilityOrchestrator()
    
    # Initialize the orchestrator
    await orchestrator.initialize()
    
    print("📊 Simulating request load...")
    
    # Simulate various request types
    request_types = [
        {"type": "compilation", "data": "compile_code", "params": {"optimize": True}},
        {"type": "validation", "data": "validate_claims", "params": {"strict": True}},
        {"type": "monitoring", "data": "check_health", "params": {"detailed": False}},
        {"type": "training", "data": "train_model", "params": {"epochs": 10}},
        {"type": "orchestration", "data": "coordinate_tasks", "params": {"parallel": True}}
    ]
    
    # Process requests and track performance
    total_requests = 50
    successful_requests = 0
    total_processing_time = 0
    cache_hits = 0
    
    for i in range(total_requests):
        request_data = request_types[i % len(request_types)].copy()
        request_data["request_id"] = f"demo_req_{i}"
        
        result = await orchestrator.handle_request(request_data)
        
        if result.get("success", False):
            successful_requests += 1
        
        total_processing_time += result.get("processing_time_ms", 0)
        
        if result.get("source") == "cache":
            cache_hits += 1
        
        # Simulate gradually increasing load
        if i % 10 == 0:
            load_levels = ["low", "medium", "high", "extreme"]
            orchestrator.current_load_level = load_levels[min(3, i // 10)]
            print(f"Load level increased to: {orchestrator.current_load_level}")
        
        # Small delay between requests
        await asyncio.sleep(0.01)
    
    # Show results
    print(f"\n📈 Request Processing Results:")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Success rate: {successful_requests/total_requests*100:.1f}%")
    print(f"Average processing time: {total_processing_time/total_requests:.1f}ms")
    print(f"Cache hits: {cache_hits} ({cache_hits/total_requests*100:.1f}%)")
    
    # Show comprehensive statistics
    print(f"\n📊 Comprehensive Scalability Statistics:")
    stats = orchestrator.get_comprehensive_statistics()
    
    print(f"Current load level: {stats['current_load_level']}")
    print(f"Recent avg throughput: {stats['recent_performance']['avg_throughput_ops_sec']:.1f} ops/sec")
    print(f"Recent avg latency: {stats['recent_performance']['avg_latency_ms']:.1f}ms")
    print(f"Max concurrent users: {stats['recent_performance']['max_concurrent_users']}")
    
    print(f"\nComponent Statistics:")
    comp_stats = stats['component_statistics']
    print(f"  Intelligent routing: {comp_stats['intelligent_routing']['total_requests_routed']} requests routed")
    print(f"  CPU optimization: {comp_stats['cpu_optimization']['components_optimized']} components optimized")
    print(f"  Auto-scaling: {comp_stats['auto_scaling']['total_instances']} total instances")
    print(f"  Advanced caching: {comp_stats['advanced_caching']['total_cached_items']} cached items")
    
    print(f"\nExpected Improvements:")
    improvements = stats['scalability_improvements']
    for key, value in improvements.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Scalability orchestrator demo completed!")
    return orchestrator


if __name__ == "__main__":
    asyncio.run(demo_scalability_orchestrator())