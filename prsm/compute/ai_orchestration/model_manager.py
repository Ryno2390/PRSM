#!/usr/bin/env python3
"""
Advanced AI Model Manager
=========================

Comprehensive system for managing multiple AI models, providers, and capabilities
with intelligent load balancing, health monitoring, and performance optimization.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Set
import uuid
from pathlib import Path
import math

from prsm.compute.plugins import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"
    LOCAL = "local"


class ModelStatus(Enum):
    """Model instance status"""
    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    INITIALIZING = "initializing"


class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MATH = "math"
    ANALYSIS = "analysis"
    CREATIVE_WRITING = "creative_writing"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SSM_NATIVE = "ssm_native"
    LNN_NATIVE = "lnn_native"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    success_rate: float = 100.0
    error_rate: float = 0.0
    
    # Usage metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    queue_length: int = 0
    
    # Cost metrics
    cost_per_token: float = 0.0
    total_cost: float = 0.0
    
    # Quality metrics
    quality_score: float = 0.0
    user_satisfaction: float = 0.0
    
    # Time tracking
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "avg_response_time_ms": self.avg_response_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "queue_length": self.queue_length,
            "cost_per_token": self.cost_per_token,
            "total_cost": self.total_cost,
            "quality_score": self.quality_score,
            "user_satisfaction": self.user_satisfaction,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class ModelInstance:
    """AI model instance configuration"""
    model_id: str
    name: str
    provider: ModelProvider
    model_name: str
    description: str = ""
    
    # Capabilities
    capabilities: Set[ModelCapability] = field(default_factory=set)
    
    # Configuration
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # API configuration
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    
    # Resource limits
    max_concurrent_requests: int = 10
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 90000
    
    # Status and health
    status: ModelStatus = ModelStatus.INITIALIZING
    health_score: float = 100.0
    
    # Load balancing
    weight: float = 1.0
    priority: int = 100
    
    # Performance tracking
    metrics: ModelMetrics = field(default_factory=lambda: ModelMetrics(""))
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if self.metrics.model_id == "":
            self.metrics.model_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "provider": self.provider.value,
            "model_name": self.model_name,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "max_concurrent_requests": self.max_concurrent_requests,
            "rate_limit_requests_per_minute": self.rate_limit_requests_per_minute,
            "rate_limit_tokens_per_minute": self.rate_limit_tokens_per_minute,
            "status": self.status.value,
            "health_score": self.health_score,
            "weight": self.weight,
            "priority": self.priority,
            "metrics": self.metrics.to_dict(),
            "config": self.config,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class SSMModelInstance(ModelInstance):
    """Native SSM model instance for edge-efficient inference"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.tokenizer = None
        self._lock = asyncio.Lock()
        
    async def initialize_local_model(self):
        """Initialize the local SSM model architecture"""
        from prsm.compute.nwtn.architectures.ssm_core import get_ssm_reasoner
        from prsm.core.utils.deterministic import force_determinism
        
        # Ensure identical weight initialization for consensus
        seed = self.config.get("seed", 42)
        force_determinism(seed)
        
        d_model = self.config.get("d_model", 512)
        layers = self.config.get("layers", 6)
        
        self.model = get_ssm_reasoner(d_model=d_model, layers=layers)
        self.model.eval()
        
        # In a real scenario, we'd load weights from IPFS/Torrent here
        logger.info(f"Initialized native SSM model: {self.name}")
        self.status = ModelStatus.AVAILABLE

    async def execute_ssm(self, input_ids: Any, states: Optional[List] = None):
        """Execute inference on the native SSM architecture"""
        import torch
        from prsm.core.utils.deterministic import generate_verification_hash
        
        # USE LOCAL GENERATOR TO PREVENT GLOBAL STATE POLLUTION
        # This is the "Gold Standard" for decentralized AI consensus
        seed = self.config.get("seed", 42)
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed)
        
        async with self._lock:
            with torch.no_grad():
                # In a more complex model, we would pass 'generator' to 
                # stochastic layers like Dropout. Our current SSM is 
                # purely deterministic in its math, so any variance 
                # comes from global state or library initialization.
                
                # Force deterministic algorithms for this scope
                torch.use_deterministic_algorithms(True, warn_only=True)
                
                logits, next_states = self.model(input_ids, states)
                
                # Generate a verification hash for the blockchain layer
                input_hash = hashlib.sha256(str(input_ids.tolist()).encode()).hexdigest()
                v_hash = generate_verification_hash(logits, self.model_id, input_hash)
                
                # --- NEW: GENERATE zk-SNARK PROOF ---
                from prsm.core.cryptography.zk_proofs import get_zk_proof_system, ZKProofRequest
                zk_system = await get_zk_proof_system()
                
                # Public inputs: inputs that anyone can see/verify
                public_inputs = [self.model_id, input_hash, v_hash]
                
                # Private inputs: inputs that stay hidden (like weights or intermediate states)
                # In a real SNARK, this would include the full weight tensor
                private_inputs = {
                    "raw_logits": logits.tolist(),
                    "seed": seed
                }
                
                zk_request = ZKProofRequest(
                    circuit_id="inference_verification",
                    proof_system="groth16",
                    statement=f"Prove inference for model {self.model_id}",
                    purpose="consensus_verification",
                    public_inputs=public_inputs,
                    private_inputs=private_inputs,
                    metadata={"prover_id": "local_node"}
                )
                
                zk_result = await zk_system.generate_proof(zk_request)
                
                return {
                    "logits": logits,
                    "next_states": next_states,
                    "verification_hash": v_hash,
                    "zk_proof_id": zk_result.proof_data if zk_result.success else None
                }

import hashlib
class LiquidModelInstance(ModelInstance):
    """Extreme edge ODE-based liquid model instance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self._lock = asyncio.Lock()
        
    async def initialize_local_model(self):
        """Initialize the local Liquid ODE architecture"""
        from prsm.compute.nwtn.architectures.liquid_core import get_liquid_reasoner
        from prsm.core.utils.deterministic import force_determinism
        
        seed = self.config.get("seed", 42)
        force_determinism(seed)
        
        self.model = get_liquid_reasoner(
            input_size=self.config.get("input_size", 64),
            hidden_size=self.config.get("hidden_size", 128)
        )
        self.model.eval()
        self.status = ModelStatus.AVAILABLE

    async def execute_liquid(self, input_data: Any, state: Optional[Any] = None):
        """Execute inference on the native Liquid architecture"""
        import torch
        from prsm.core.cryptography.zk_proofs import get_zk_proof_system, ZKProofRequest
        
        async with self._lock:
            with torch.no_grad():
                outputs, next_state = self.model(input_data, state)
                
                # ZK-Inference Proof
                zk_system = await get_zk_proof_system()
                v_hash = hashlib.sha256(str(outputs.tolist()).encode()).hexdigest()
                
                zk_request = ZKProofRequest(
                    circuit_id="inference_verification",
                    proof_system="groth16",
                    statement=f"Prove liquid inference for sensor node",
                    purpose="edge_integrity",
                    public_inputs=[self.model_id, v_hash],
                    private_inputs={"raw_state": next_state.tolist()},
                    metadata={"prover_id": self.model_id}
                )
                
                zk_result = await zk_system.generate_proof(zk_request)
                
                return {
                    "output": outputs,
                    "state": next_state,
                    "verification_hash": v_hash,
                    "zk_proof_id": zk_result.proof_data if zk_result.success else None
                }


class ModelHealthMonitor:
    """Health monitoring for model instances"""
    
    def __init__(self):
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.health_checks: Dict[str, datetime] = {}
        
        # Health check configuration
        self.check_interval = 60  # seconds
        self.timeout = 30  # seconds
        
        # Statistics
        self.stats = {
            "total_health_checks": 0,
            "healthy_models": 0,
            "unhealthy_models": 0,
            "avg_response_time_ms": 0.0
        }
    
    async def start_monitoring(self, model_instance: ModelInstance):
        """Start monitoring a model instance"""
        if model_instance.model_id in self.monitoring_tasks:
            logger.warning(f"Already monitoring model: {model_instance.name}")
            return
        
        task = asyncio.create_task(self._monitor_model(model_instance))
        self.monitoring_tasks[model_instance.model_id] = task
        
        logger.info(f"Started monitoring model: {model_instance.name}")
    
    async def stop_monitoring(self, model_id: str):
        """Stop monitoring a model instance"""
        if model_id not in self.monitoring_tasks:
            return
        
        task = self.monitoring_tasks[model_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self.monitoring_tasks[model_id]
        logger.info(f"Stopped monitoring model: {model_id}")
    
    async def _monitor_model(self, model_instance: ModelInstance):
        """Monitor model health"""
        while True:
            try:
                # Perform health check
                health_result = await self._perform_health_check(model_instance)
                
                # Update model health score
                model_instance.health_score = health_result["health_score"]
                
                # Update status based on health
                if health_result["health_score"] < 50:
                    model_instance.status = ModelStatus.ERROR
                elif health_result["health_score"] < 80:
                    model_instance.status = ModelStatus.OVERLOADED
                else:
                    if model_instance.status in [ModelStatus.ERROR, ModelStatus.OVERLOADED]:
                        model_instance.status = ModelStatus.AVAILABLE
                
                # Update statistics
                self.stats["total_health_checks"] += 1
                if health_result["health_score"] >= 80:
                    self.stats["healthy_models"] += 1
                else:
                    self.stats["unhealthy_models"] += 1
                
                # Store last check time
                self.health_checks[model_instance.model_id] = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error for {model_instance.name}: {e}")
                model_instance.status = ModelStatus.ERROR
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self, model_instance: ModelInstance) -> Dict[str, Any]:
        """Perform health check on model instance"""
        start_time = datetime.now()
        
        try:
            # Simple ping test (could be enhanced with actual API calls)
            health_score = 100.0
            
            # Check response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            if response_time > 5000:  # 5 seconds
                health_score -= 30
            elif response_time > 2000:  # 2 seconds
                health_score -= 15
            
            # Check error rate
            if model_instance.metrics.error_rate > 10:
                health_score -= 20
            elif model_instance.metrics.error_rate > 5:
                health_score -= 10
            
            # Check queue length
            if model_instance.metrics.queue_length > 100:
                health_score -= 20
            elif model_instance.metrics.queue_length > 50:
                health_score -= 10
            
            return {
                "health_score": max(0, health_score),
                "response_time_ms": response_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            return {
                "health_score": 0.0,
                "response_time_ms": float('inf'),
                "error": str(e),
                "timestamp": start_time.isoformat()
            }
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "monitored_models": len(self.monitoring_tasks),
            "statistics": self.stats,
            "last_health_checks": {
                model_id: timestamp.isoformat() 
                for model_id, timestamp in self.health_checks.items()
            }
        }


class LoadBalancer:
    """Intelligent load balancer for model instances"""
    
    def __init__(self):
        self.algorithm = "weighted_round_robin"  # weighted_round_robin, least_connections, response_time
        self.round_robin_counters: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            "total_selections": 0,
            "selections_by_model": {},
            "load_balancing_errors": 0
        }
    
    def select_model(self, models: List[ModelInstance], 
                    required_capabilities: Optional[Set[ModelCapability]] = None) -> Optional[ModelInstance]:
        """Select best model instance based on load balancing algorithm"""
        
        # Filter by capabilities
        available_models = []
        for model in models:
            if model.status not in [ModelStatus.AVAILABLE, ModelStatus.BUSY]:
                continue
            
            if required_capabilities:
                if not required_capabilities.issubset(model.capabilities):
                    continue
            
            available_models.append(model)
        
        if not available_models:
            return None
        
        try:
            if self.algorithm == "weighted_round_robin":
                selected = self._weighted_round_robin(available_models)
            elif self.algorithm == "least_connections":
                selected = self._least_connections(available_models)
            elif self.algorithm == "response_time":
                selected = self._fastest_response_time(available_models)
            else:
                selected = available_models[0]  # Fallback
            
            # Update statistics
            self.stats["total_selections"] += 1
            model_id = selected.model_id
            self.stats["selections_by_model"][model_id] = \
                self.stats["selections_by_model"].get(model_id, 0) + 1
            
            return selected
            
        except Exception as e:
            logger.error(f"Load balancing error: {e}")
            self.stats["load_balancing_errors"] += 1
            return available_models[0] if available_models else None
    
    def _weighted_round_robin(self, models: List[ModelInstance]) -> ModelInstance:
        """Weighted round-robin selection"""
        # Calculate total weight
        total_weight = sum(model.weight * model.health_score / 100 for model in models)
        
        if total_weight == 0:
            return models[0]
        
        # Simple weighted selection (could be optimized)
        import random
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for model in models:
            current_weight += model.weight * model.health_score / 100
            if rand_val <= current_weight:
                return model
        
        return models[-1]
    
    def _least_connections(self, models: List[ModelInstance]) -> ModelInstance:
        """Select model with least connections"""
        return min(models, key=lambda m: m.metrics.queue_length)
    
    def _fastest_response_time(self, models: List[ModelInstance]) -> ModelInstance:
        """Select model with fastest response time"""
        return min(models, key=lambda m: m.metrics.avg_response_time_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return self.stats


class ModelManager:
    """Main model management system"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./model_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Model registry
        self.models: Dict[str, ModelInstance] = {}
        
        # Management components
        self.health_monitor = ModelHealthMonitor()
        self.load_balancer = LoadBalancer()
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "total_models": 0,
            "active_models": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0.0
        }
        
        logger.info("Model Manager initialized")
    
    def register_model(self, model_instance: ModelInstance):
        """Register a new model instance"""
        self.models[model_instance.model_id] = model_instance
        self.stats["total_models"] += 1
        
        # Initialize rate limiter
        self.rate_limiters[model_instance.model_id] = {
            "requests": [],
            "tokens": []
        }
        
        # Start health monitoring
        asyncio.create_task(self.health_monitor.start_monitoring(model_instance))
        
        logger.info(f"Registered model: {model_instance.name}")
        
        # Emit event
        asyncio.create_task(self._emit_event("model_registered", model_instance.to_dict()))
    
    def unregister_model(self, model_id: str):
        """Unregister a model instance"""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Stop monitoring
        asyncio.create_task(self.health_monitor.stop_monitoring(model_id))
        
        # Remove from registry
        del self.models[model_id]
        del self.rate_limiters[model_id]
        
        self.stats["total_models"] -= 1
        if model.status == ModelStatus.AVAILABLE:
            self.stats["active_models"] -= 1
        
        logger.info(f"Unregistered model: {model.name}")
        
        # Emit event
        asyncio.create_task(self._emit_event("model_unregistered", {"model_id": model_id}))
    
    def get_model(self, model_id: str) -> Optional[ModelInstance]:
        """Get model instance by ID"""
        return self.models.get(model_id)
    
    def list_models(self, provider: Optional[ModelProvider] = None,
                   capabilities: Optional[Set[ModelCapability]] = None,
                   status: Optional[ModelStatus] = None) -> List[ModelInstance]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if capabilities:
            models = [m for m in models if capabilities.issubset(m.capabilities)]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return models
    
    def select_best_model(self, required_capabilities: Optional[Set[ModelCapability]] = None,
                         preferred_provider: Optional[ModelProvider] = None) -> Optional[ModelInstance]:
        """Select the best model for a task"""
        
        # Get available models
        available_models = self.list_models(
            provider=preferred_provider,
            capabilities=required_capabilities,
            status=None  # Load balancer will filter by status
        )
        
        if not available_models:
            return None
        
        # Use load balancer to select best model
        return self.load_balancer.select_model(available_models, required_capabilities)
    
    async def execute_request(self, model_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a request on a specific model"""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.models[model_id]
        
        # Check rate limits
        if not await self._check_rate_limits(model):
            raise Exception(f"Rate limit exceeded for model: {model_id}")
        
        # Update model status
        if model.status == ModelStatus.AVAILABLE:
            model.status = ModelStatus.BUSY
        
        start_time = datetime.now()
        
        try:
            # Execute request (placeholder - would integrate with actual model APIs)
            result = await self._execute_model_request(model, request_data)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._update_model_metrics(model, execution_time, True, result)
            
            # Update global stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            
            return result
            
        except Exception as e:
            # Update metrics for failure
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._update_model_metrics(model, execution_time, False, None)
            
            # Update global stats
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            
            raise
        
        finally:
            # Reset model status
            if model.status == ModelStatus.BUSY:
                model.status = ModelStatus.AVAILABLE
    
    async def _execute_model_request(self, model: ModelInstance, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request on model (placeholder for actual implementation)"""
        
        # Simulate API call
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "model_id": model.model_id,
            "response": "This is a simulated response",
            "tokens_used": 100,
            "execution_time_ms": 100
        }
    
    async def _check_rate_limits(self, model: ModelInstance) -> bool:
        """Check if model is within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        rate_limiter = self.rate_limiters[model.model_id]
        
        # Clean old entries
        rate_limiter["requests"] = [
            timestamp for timestamp in rate_limiter["requests"] 
            if timestamp > minute_ago
        ]
        
        # Check request rate limit
        if len(rate_limiter["requests"]) >= model.rate_limit_requests_per_minute:
            return False
        
        # Add current request
        rate_limiter["requests"].append(now)
        
        return True
    
    async def _update_model_metrics(self, model: ModelInstance, execution_time_ms: float, 
                                  success: bool, result: Optional[Dict[str, Any]]):
        """Update model performance metrics"""
        metrics = model.metrics
        
        # Update request counts
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update success/error rates
        metrics.success_rate = (metrics.successful_requests / metrics.total_requests) * 100
        metrics.error_rate = (metrics.failed_requests / metrics.total_requests) * 100
        
        # Update response time
        total_requests = metrics.total_requests
        current_avg = metrics.avg_response_time_ms
        metrics.avg_response_time_ms = \
            (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
        
        # Update token metrics if available
        if result and "tokens_used" in result:
            tokens_used = result["tokens_used"]
            metrics.total_tokens_generated += tokens_used
            metrics.tokens_per_second = tokens_used / (execution_time_ms / 1000)
        
        # Update global average response time
        self._update_global_response_time(execution_time_ms)
        
        metrics.last_updated = datetime.now(timezone.utc)
    
    def _update_global_response_time(self, execution_time_ms: float):
        """Update global average response time"""
        total_requests = self.stats["total_requests"] + 1
        current_avg = self.stats["avg_response_time_ms"]
        self.stats["avg_response_time_ms"] = \
            (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type}: {handler.__name__}")
    
    async def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit model event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(event_data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        active_models = len([m for m in self.models.values() 
                           if m.status == ModelStatus.AVAILABLE])
        
        # Model breakdown by provider
        provider_breakdown = {}
        for model in self.models.values():
            provider = model.provider.value
            provider_breakdown[provider] = provider_breakdown.get(provider, 0) + 1
        
        # Capability breakdown
        capability_breakdown = {}
        for model in self.models.values():
            for cap in model.capabilities:
                cap_name = cap.value
                capability_breakdown[cap_name] = capability_breakdown.get(cap_name, 0) + 1
        
        return {
            "model_statistics": {
                **self.stats,
                "active_models": active_models,
                "provider_breakdown": provider_breakdown,
                "capability_breakdown": capability_breakdown
            },
            "load_balancer_stats": self.load_balancer.get_stats(),
            "health_monitor_stats": self.health_monitor.get_monitoring_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_model_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for specific model"""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        return {
            "model_info": model.to_dict(),
            "detailed_metrics": model.metrics.to_dict(),
            "rate_limiter_status": self.rate_limiters.get(model_id, {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown of model manager"""
        logger.info("Shutting down Model Manager")
        
        # Stop all health monitoring
        for model_id in list(self.health_monitor.monitoring_tasks.keys()):
            await self.health_monitor.stop_monitoring(model_id)
        
        logger.info("Model Manager shutdown complete")


# Export main classes
__all__ = [
    'ModelProvider',
    'ModelStatus', 
    'ModelCapability',
    'ModelMetrics',
    'ModelInstance',
    'ModelHealthMonitor',
    'LoadBalancer',
    'ModelManager'
]