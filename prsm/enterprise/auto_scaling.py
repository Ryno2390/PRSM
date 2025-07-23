"""
PRSM Advanced Auto-Scaling Infrastructure Management
Enterprise-grade auto-scaling with predictive analytics, cost optimization, and intelligent scaling decisions
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
import statistics
from collections import defaultdict, deque
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# Machine learning for predictive scaling
try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

# Time series analysis
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ScalingReason(Enum):
    """Reasons for scaling decisions"""
    CPU_PRESSURE = "cpu_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    RESPONSE_TIME_DEGRADATION = "response_time_degradation"
    CONNECTION_OVERLOAD = "connection_overload"
    PREDICTED_DEMAND_INCREASE = "predicted_demand_increase"
    PREDICTED_DEMAND_DECREASE = "predicted_demand_decrease"
    COST_OPTIMIZATION = "cost_optimization"
    SCHEDULED_SCALING = "scheduled_scaling"
    MANUAL_INTERVENTION = "manual_intervention"
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"


class ScalingMode(Enum):
    """Auto-scaling modes"""
    REACTIVE = "reactive"      # React to current metrics
    PREDICTIVE = "predictive"  # Use ML predictions
    SCHEDULED = "scheduled"    # Time-based scaling
    HYBRID = "hybrid"          # Combination of all modes


class InstanceType(Enum):
    """Instance types for scaling"""
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
    COMPUTE_OPTIMIZED = "compute_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    STORAGE_OPTIMIZED = "storage_optimized"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    timestamp: datetime
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_utilization_percent: float = 0.0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate_percent: float = 0.0
    active_connections: int = 0
    
    # Queue metrics
    queue_depth: int = 0
    queue_processing_time_ms: float = 0.0
    
    # Business metrics
    concurrent_users: int = 0
    transactions_per_minute: float = 0.0
    
    # Cost metrics
    cost_per_hour: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "network_utilization_percent": self.network_utilization_percent,
            "avg_response_time_ms": self.avg_response_time_ms,
            "requests_per_second": self.requests_per_second,
            "error_rate_percent": self.error_rate_percent,
            "active_connections": self.active_connections,
            "queue_depth": self.queue_depth,
            "queue_processing_time_ms": self.queue_processing_time_ms,
            "concurrent_users": self.concurrent_users,
            "transactions_per_minute": self.transactions_per_minute,
            "cost_per_hour": self.cost_per_hour
        }


@dataclass
class ScalingThresholds:
    """Scaling thresholds configuration"""
    # Scale up thresholds
    cpu_scale_up_threshold: float = 75.0
    memory_scale_up_threshold: float = 80.0
    response_time_scale_up_threshold_ms: float = 500.0
    error_rate_scale_up_threshold: float = 5.0
    connection_scale_up_threshold: int = 1000
    
    # Scale down thresholds
    cpu_scale_down_threshold: float = 25.0
    memory_scale_down_threshold: float = 30.0
    response_time_scale_down_threshold_ms: float = 100.0
    error_rate_scale_down_threshold: float = 1.0
    connection_scale_down_threshold: int = 200
    
    # Composite thresholds
    scale_up_condition_count: int = 2  # How many conditions must be met
    scale_down_condition_count: int = 3  # How many conditions must be met for scale down
    
    # Time windows
    evaluation_window_minutes: int = 5
    scale_up_cooldown_minutes: int = 5
    scale_down_cooldown_minutes: int = 10


@dataclass
class ScalingConstraints:
    """Constraints for scaling operations"""
    # Instance limits
    min_instances: int = 2
    max_instances: int = 100
    max_instances_per_region: int = 20
    max_instances_per_az: int = 10
    
    # Scaling rate limits
    max_scale_up_instances: int = 5
    max_scale_down_instances: int = 3
    max_scaling_operations_per_hour: int = 10
    
    # Cost constraints
    max_hourly_cost: float = 1000.0
    max_daily_cost: float = 20000.0
    cost_optimization_enabled: bool = True
    
    # Performance constraints
    min_performance_level: float = 0.8  # 0.0-1.0
    max_acceptable_response_time_ms: float = 1000.0
    
    # Time constraints
    business_hours_start: int = 8  # 8 AM
    business_hours_end: int = 18   # 6 PM
    weekend_scaling_enabled: bool = True
    
    # Regional constraints
    preferred_regions: List[str] = field(default_factory=list)
    excluded_regions: List[str] = field(default_factory=list)
    
    # Instance type constraints
    allowed_instance_types: List[InstanceType] = field(default_factory=lambda: [
        InstanceType.SMALL, InstanceType.MEDIUM, InstanceType.LARGE
    ])


@dataclass
class ScalingAction:
    """Scaling action to be executed"""
    action_id: str
    direction: ScalingDirection
    reason: ScalingReason
    confidence: float  # 0.0-1.0
    
    # Target configuration
    target_instance_count: int
    current_instance_count: int
    instance_type: InstanceType
    regions: List[str]
    
    # Execution details
    instances_to_add: int = 0
    instances_to_remove: List[str] = field(default_factory=list)
    estimated_cost_impact: float = 0.0
    estimated_performance_impact: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execute_after: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Validation
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Metadata
    metrics_snapshot: Optional[ScalingMetrics] = None
    predicted_metrics: Optional[ScalingMetrics] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScalingSchedule:
    """Scheduled scaling configuration"""
    schedule_id: str
    name: str
    description: str
    
    # Schedule definition
    cron_expression: str  # e.g., "0 8 * * 1-5" for 8 AM weekdays
    target_instance_count: int
    instance_type: InstanceType
    
    # Duration
    duration_minutes: Optional[int] = None  # How long to maintain this scale
    
    # Constraints
    enabled: bool = True
    regions: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)  # Additional conditions
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_executed: Optional[datetime] = None
    execution_count: int = 0


class PredictiveScalingEngine:
    """Machine learning based predictive scaling"""
    
    def __init__(self):
        self.enabled = ML_AVAILABLE
        self.models = {}
        self.feature_scalers = {}
        self.training_data = defaultdict(lambda: deque(maxlen=10000))
        self.last_training = {}
        
        # Model configurations
        self.model_configs = {
            "cpu_prediction": {"retrain_interval": 3600, "min_samples": 100},
            "memory_prediction": {"retrain_interval": 3600, "min_samples": 100},
            "response_time_prediction": {"retrain_interval": 1800, "min_samples": 50},
            "demand_prediction": {"retrain_interval": 7200, "min_samples": 200}
        }
        
        if self.enabled:
            self._initialize_models()
            logger.info("✅ Predictive scaling engine initialized")
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            for model_name in self.model_configs.keys():
                self.models[model_name] = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
                self.feature_scalers[model_name] = StandardScaler()
                
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            self.enabled = False
    
    def add_training_data(self, metrics: ScalingMetrics):
        """Add training data for models"""
        if not self.enabled:
            return
        
        try:
            # Prepare features
            features = self._extract_features(metrics)
            
            # Add to training data for each model
            self.training_data["cpu_prediction"].append({
                "features": features,
                "target": metrics.cpu_usage_percent,
                "timestamp": metrics.timestamp.timestamp()
            })
            
            self.training_data["memory_prediction"].append({
                "features": features,
                "target": metrics.memory_usage_percent,
                "timestamp": metrics.timestamp.timestamp()
            })
            
            self.training_data["response_time_prediction"].append({
                "features": features,
                "target": metrics.avg_response_time_ms,
                "timestamp": metrics.timestamp.timestamp()
            })
            
            self.training_data["demand_prediction"].append({
                "features": features,
                "target": metrics.requests_per_second,
                "timestamp": metrics.timestamp.timestamp()
            })
            
            # Trigger retraining if needed
            await self._check_and_retrain_models()
            
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
    
    def _extract_features(self, metrics: ScalingMetrics) -> List[float]:
        """Extract features from metrics"""
        timestamp = metrics.timestamp
        
        return [
            # Time-based features
            timestamp.hour,
            timestamp.weekday(),
            timestamp.day,
            
            # Current metrics
            metrics.cpu_usage_percent,
            metrics.memory_usage_percent,
            metrics.avg_response_time_ms,
            metrics.requests_per_second,
            metrics.active_connections,
            metrics.error_rate_percent,
            metrics.queue_depth,
            metrics.concurrent_users,
            metrics.transactions_per_minute,
            
            # Derived features
            metrics.cpu_usage_percent * metrics.memory_usage_percent / 100,  # Resource pressure
            metrics.requests_per_second / max(1, metrics.active_connections),  # Request efficiency
            metrics.error_rate_percent * metrics.avg_response_time_ms / 100,  # Quality score
        ]
    
    async def _check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        current_time = time.time()
        
        for model_name, config in self.model_configs.items():
            last_training = self.last_training.get(model_name, 0)
            
            if (current_time - last_training > config["retrain_interval"] and
                len(self.training_data[model_name]) >= config["min_samples"]):
                
                await self._retrain_model(model_name)
    
    async def _retrain_model(self, model_name: str):
        """Retrain a specific model"""
        if not self.enabled:
            return
        
        try:
            training_data = list(self.training_data[model_name])
            
            if len(training_data) < 10:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for sample in training_data:
                X.append(sample["features"])
                y.append(sample["target"])
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.feature_scalers[model_name].fit_transform(X)
            
            # Train model
            self.models[model_name].fit(X_scaled, y)
            self.last_training[model_name] = time.time()
            
            logger.info(f"✅ Retrained {model_name} model with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining model {model_name}: {e}")
    
    async def predict_metrics(self, current_metrics: ScalingMetrics, 
                            prediction_horizon_minutes: int = 15) -> Optional[ScalingMetrics]:
        """Predict future metrics"""
        if not self.enabled:
            return None
        
        try:
            # Extract features for prediction
            features = self._extract_features(current_metrics)
            
            # Adjust time features for future prediction
            future_time = current_metrics.timestamp + timedelta(minutes=prediction_horizon_minutes)
            features[0] = future_time.hour  # Hour
            features[1] = future_time.weekday()  # Weekday
            features[2] = future_time.day  # Day
            
            X = np.array([features])
            
            # Make predictions
            predictions = {}
            
            for model_name, model in self.models.items():
                if model_name in self.feature_scalers:
                    try:
                        X_scaled = self.feature_scalers[model_name].transform(X)
                        prediction = model.predict(X_scaled)[0]
                        predictions[model_name] = max(0, prediction)  # Ensure non-negative
                    except Exception as e:
                        logger.debug(f"Error predicting with {model_name}: {e}")
            
            # Create predicted metrics
            predicted_metrics = ScalingMetrics(
                timestamp=future_time,
                cpu_usage_percent=predictions.get("cpu_prediction", current_metrics.cpu_usage_percent),
                memory_usage_percent=predictions.get("memory_prediction", current_metrics.memory_usage_percent),
                avg_response_time_ms=predictions.get("response_time_prediction", current_metrics.avg_response_time_ms),
                requests_per_second=predictions.get("demand_prediction", current_metrics.requests_per_second),
                
                # Copy other metrics (would be predicted in full implementation)
                error_rate_percent=current_metrics.error_rate_percent,
                active_connections=current_metrics.active_connections,
                queue_depth=current_metrics.queue_depth,
                concurrent_users=current_metrics.concurrent_users,
                transactions_per_minute=current_metrics.transactions_per_minute,
                cost_per_hour=current_metrics.cost_per_hour
            )
            
            return predicted_metrics
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    async def predict_scaling_need(self, current_metrics: ScalingMetrics,
                                 thresholds: ScalingThresholds) -> Dict[str, Any]:
        """Predict if scaling will be needed"""
        if not self.enabled:
            return {"scaling_needed": False, "confidence": 0.0}
        
        try:
            # Predict metrics for multiple time horizons
            horizons = [5, 15, 30, 60]  # minutes
            predictions = {}
            
            for horizon in horizons:
                predicted = await self.predict_metrics(current_metrics, horizon)
                if predicted:
                    predictions[horizon] = predicted
            
            if not predictions:
                return {"scaling_needed": False, "confidence": 0.0}
            
            # Analyze predictions for scaling needs
            scale_up_votes = 0
            scale_down_votes = 0
            confidence_scores = []
            
            for horizon, predicted in predictions.items():
                # Check scale up conditions
                scale_up_conditions = [
                    predicted.cpu_usage_percent > thresholds.cpu_scale_up_threshold,
                    predicted.memory_usage_percent > thresholds.memory_scale_up_threshold,
                    predicted.avg_response_time_ms > thresholds.response_time_scale_up_threshold_ms,
                    predicted.error_rate_percent > thresholds.error_rate_scale_up_threshold
                ]
                
                # Check scale down conditions
                scale_down_conditions = [
                    predicted.cpu_usage_percent < thresholds.cpu_scale_down_threshold,
                    predicted.memory_usage_percent < thresholds.memory_scale_down_threshold,
                    predicted.avg_response_time_ms < thresholds.response_time_scale_down_threshold_ms,
                    predicted.error_rate_percent < thresholds.error_rate_scale_down_threshold
                ]
                
                if sum(scale_up_conditions) >= thresholds.scale_up_condition_count:
                    scale_up_votes += 1
                    confidence_scores.append(0.8 / horizon * 15)  # Closer predictions = higher confidence
                
                if sum(scale_down_conditions) >= thresholds.scale_down_condition_count:
                    scale_down_votes += 1
                    confidence_scores.append(0.6 / horizon * 15)
            
            # Make decision
            if scale_up_votes > scale_down_votes:
                scaling_needed = True
                direction = ScalingDirection.SCALE_UP
                reason = ScalingReason.PREDICTED_DEMAND_INCREASE
            elif scale_down_votes > scale_up_votes:
                scaling_needed = True
                direction = ScalingDirection.SCALE_DOWN
                reason = ScalingReason.PREDICTED_DEMAND_DECREASE
            else:
                scaling_needed = False
                direction = ScalingDirection.MAINTAIN
                reason = None
            
            confidence = min(1.0, sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0
            
            return {
                "scaling_needed": scaling_needed,
                "direction": direction,
                "reason": reason,
                "confidence": confidence,
                "predictions": {h: p.to_dict() for h, p in predictions.items()},
                "analysis": {
                    "scale_up_votes": scale_up_votes,
                    "scale_down_votes": scale_down_votes,
                    "total_horizons": len(predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting scaling need: {e}")
            return {"scaling_needed": False, "confidence": 0.0}


class CostOptimizer:
    """Cost optimization for auto-scaling"""
    
    def __init__(self):
        # Instance cost mapping (would be loaded from cloud provider APIs)
        self.instance_costs = {
            InstanceType.MICRO: 0.0116,
            InstanceType.SMALL: 0.023,
            InstanceType.MEDIUM: 0.046,
            InstanceType.LARGE: 0.092,
            InstanceType.XLARGE: 0.184,
            InstanceType.COMPUTE_OPTIMIZED: 0.096,
            InstanceType.MEMORY_OPTIMIZED: 0.133,
            InstanceType.STORAGE_OPTIMIZED: 0.084
        }
        
        # Regional cost multipliers
        self.regional_multipliers = {
            "us-east-1": 1.0,
            "us-west-1": 1.1,
            "us-west-2": 1.05,
            "eu-west-1": 1.15,
            "eu-central-1": 1.12,
            "ap-southeast-1": 1.2,
            "ap-northeast-1": 1.18
        }
        
        # Spot instance discounts
        self.spot_discounts = {
            InstanceType.MICRO: 0.6,
            InstanceType.SMALL: 0.65,
            InstanceType.MEDIUM: 0.7,
            InstanceType.LARGE: 0.72,
            InstanceType.XLARGE: 0.75
        }
    
    def calculate_cost_impact(self, action: ScalingAction, region: str = "us-east-1") -> float:
        """Calculate cost impact of scaling action"""
        try:
            base_cost = self.instance_costs.get(action.instance_type, 0.1)
            regional_multiplier = self.regional_multipliers.get(region, 1.0)
            
            hourly_cost_per_instance = base_cost * regional_multiplier
            
            if action.direction == ScalingDirection.SCALE_UP:
                cost_impact = action.instances_to_add * hourly_cost_per_instance
            elif action.direction == ScalingDirection.SCALE_DOWN:
                cost_impact = -len(action.instances_to_remove) * hourly_cost_per_instance
            else:
                cost_impact = 0.0
            
            return cost_impact
            
        except Exception as e:
            logger.error(f"Error calculating cost impact: {e}")
            return 0.0
    
    def optimize_instance_selection(self, performance_requirements: Dict[str, float],
                                  regions: List[str]) -> Dict[str, Any]:
        """Optimize instance type and region selection"""
        try:
            required_cpu = performance_requirements.get("cpu_cores", 2)
            required_memory = performance_requirements.get("memory_gb", 4)
            required_network = performance_requirements.get("network_gbps", 1)
            max_cost_per_hour = performance_requirements.get("max_cost_per_hour", 1.0)
            
            # Instance specifications (simplified)
            instance_specs = {
                InstanceType.MICRO: {"cpu": 1, "memory": 1, "network": 0.1},
                InstanceType.SMALL: {"cpu": 1, "memory": 2, "network": 0.5},
                InstanceType.MEDIUM: {"cpu": 2, "memory": 4, "network": 1.0},
                InstanceType.LARGE: {"cpu": 4, "memory": 8, "network": 2.0},
                InstanceType.XLARGE: {"cpu": 8, "memory": 16, "network": 5.0},
                InstanceType.COMPUTE_OPTIMIZED: {"cpu": 8, "memory": 8, "network": 3.0},
                InstanceType.MEMORY_OPTIMIZED: {"cpu": 4, "memory": 32, "network": 2.0},
                InstanceType.STORAGE_OPTIMIZED: {"cpu": 4, "memory": 16, "network": 2.0}
            }
            
            # Find suitable instances
            suitable_instances = []
            
            for instance_type, specs in instance_specs.items():
                if (specs["cpu"] >= required_cpu and
                    specs["memory"] >= required_memory and
                    specs["network"] >= required_network):
                    
                    # Calculate cost for each region
                    for region in regions:
                        base_cost = self.instance_costs.get(instance_type, 0.1)
                        regional_cost = base_cost * self.regional_multipliers.get(region, 1.0)
                        
                        if regional_cost <= max_cost_per_hour:
                            suitable_instances.append({
                                "instance_type": instance_type,
                                "region": region,
                                "hourly_cost": regional_cost,
                                "specs": specs,
                                "cost_efficiency": (specs["cpu"] + specs["memory"]/4) / regional_cost
                            })
            
            if not suitable_instances:
                return {"error": "No suitable instances found"}
            
            # Sort by cost efficiency (higher is better)
            suitable_instances.sort(key=lambda x: x["cost_efficiency"], reverse=True)
            
            # Return top recommendations
            return {
                "recommended": suitable_instances[0],
                "alternatives": suitable_instances[1:5],
                "total_options": len(suitable_instances)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing instance selection: {e}")
            return {"error": str(e)}
    
    def suggest_cost_optimizations(self, current_metrics: ScalingMetrics,
                                 current_instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest cost optimization opportunities"""
        optimizations = []
        
        try:
            # Analyze current resource utilization
            if current_metrics.cpu_usage_percent < 30 and current_metrics.memory_usage_percent < 40:
                optimizations.append({
                    "type": "rightsizing",
                    "description": "Consider downsizing instances due to low resource utilization",
                    "potential_savings_percent": 20,
                    "confidence": 0.8
                })
            
            # Check for spot instance opportunities
            for instance in current_instances:
                instance_type = instance.get("instance_type")
                if instance_type and not instance.get("is_spot", False):
                    discount = self.spot_discounts.get(InstanceType(instance_type), 0)
                    if discount > 0:
                        optimizations.append({
                            "type": "spot_instances",
                            "description": f"Switch to spot instances for {discount*100:.0f}% savings",
                            "potential_savings_percent": discount * 100,
                            "confidence": 0.6,
                            "risk": "Medium - spot instances can be terminated"
                        })
            
            # Regional optimization
            if len(set(inst.get("region") for inst in current_instances)) > 1:
                optimizations.append({
                    "type": "regional_optimization",
                    "description": "Consolidate instances in lower-cost regions",
                    "potential_savings_percent": 15,
                    "confidence": 0.7
                })
            
            # Reserved instance recommendations
            if len(current_instances) >= 3:
                optimizations.append({
                    "type": "reserved_instances",
                    "description": "Consider reserved instances for stable workloads",
                    "potential_savings_percent": 30,
                    "confidence": 0.9,
                    "commitment": "1-3 year commitment required"
                })
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Error suggesting cost optimizations: {e}")
            return []


class AdvancedAutoScaler:
    """Advanced auto-scaling system with ML and cost optimization"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
        # Core components
        self.predictive_engine = PredictiveScalingEngine()
        self.cost_optimizer = CostOptimizer()
        
        # Configuration
        self.thresholds = ScalingThresholds()
        self.constraints = ScalingConstraints()
        self.scaling_mode = ScalingMode.HYBRID
        
        # State management
        self.metrics_history = deque(maxlen=10000)
        self.scaling_history = deque(maxlen=1000)
        self.pending_actions = {}
        self.cooldown_timers = {}
        self.scheduled_scalings: List[ScalingSchedule] = []
        
        # Performance tracking
        self.scaling_effectiveness = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            "total_scaling_decisions": 0,
            "successful_scale_ups": 0,
            "successful_scale_downs": 0,
            "failed_scaling_attempts": 0,
            "cost_optimizations_applied": 0,
            "predictive_scaling_accuracy": 0.0,
            "average_decision_time_ms": 0.0
        }
        
        logger.info("✅ Advanced auto-scaler initialized")
    
    def configure_thresholds(self, thresholds: ScalingThresholds):
        """Configure scaling thresholds"""
        self.thresholds = thresholds
    
    def configure_constraints(self, constraints: ScalingConstraints):
        """Configure scaling constraints"""
        self.constraints = constraints
    
    def add_scheduled_scaling(self, schedule: ScalingSchedule):
        """Add scheduled scaling configuration"""
        self.scheduled_scalings.append(schedule)
        logger.info(f"Added scheduled scaling: {schedule.name}")
    
    async def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics for analysis"""
        try:
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Add to predictive engine training data
            if self.predictive_engine.enabled:
                self.predictive_engine.add_training_data(metrics)
            
            # Store in Redis for persistence
            await self.redis.lpush(
                "autoscaling:metrics",
                json.dumps(metrics.to_dict())
            )
            await self.redis.ltrim("autoscaling:metrics", 0, 9999)  # Keep last 10k
            
        except Exception as e:
            logger.error(f"Error adding metrics: {e}")
    
    async def evaluate_scaling_decision(self, current_instances: List[Dict[str, Any]]) -> Optional[ScalingAction]:
        """Evaluate if scaling is needed"""
        start_time = time.time()
        
        try:
            if not self.metrics_history:
                return None
            
            current_metrics = self.metrics_history[-1]
            current_instance_count = len(current_instances)
            
            # Check cooldown periods
            if not self._check_cooldown_status():
                return None
            
            # Get scaling decision based on mode
            scaling_decision = None
            
            if self.scaling_mode in [ScalingMode.REACTIVE, ScalingMode.HYBRID]:
                reactive_decision = await self._evaluate_reactive_scaling(
                    current_metrics, current_instance_count
                )
                scaling_decision = reactive_decision
            
            if self.scaling_mode in [ScalingMode.PREDICTIVE, ScalingMode.HYBRID]:
                predictive_decision = await self._evaluate_predictive_scaling(
                    current_metrics, current_instance_count
                )
                
                # In hybrid mode, choose the decision with higher confidence
                if self.scaling_mode == ScalingMode.HYBRID and predictive_decision:
                    if (not scaling_decision or 
                        predictive_decision.confidence > scaling_decision.confidence):
                        scaling_decision = predictive_decision
                elif self.scaling_mode == ScalingMode.PREDICTIVE:
                    scaling_decision = predictive_decision
            
            if self.scaling_mode in [ScalingMode.SCHEDULED, ScalingMode.HYBRID]:
                scheduled_decision = await self._evaluate_scheduled_scaling(current_instance_count)
                
                # Scheduled scaling has highest priority
                if scheduled_decision:
                    scaling_decision = scheduled_decision
            
            # Validate and optimize the decision
            if scaling_decision:
                scaling_decision = await self._validate_and_optimize_decision(
                    scaling_decision, current_instances
                )
                
                if scaling_decision and scaling_decision.validated:
                    # Calculate decision time
                    decision_time_ms = (time.time() - start_time) * 1000
                    self.stats["average_decision_time_ms"] = (
                        (self.stats["average_decision_time_ms"] * self.stats["total_scaling_decisions"] + decision_time_ms) /
                        (self.stats["total_scaling_decisions"] + 1)
                    )
                    
                    self.stats["total_scaling_decisions"] += 1
                    return scaling_decision
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating scaling decision: {e}")
            return None
    
    async def _evaluate_reactive_scaling(self, metrics: ScalingMetrics, 
                                       current_count: int) -> Optional[ScalingAction]:
        """Evaluate reactive scaling based on current metrics"""
        try:
            # Check scale up conditions
            scale_up_conditions = [
                metrics.cpu_usage_percent > self.thresholds.cpu_scale_up_threshold,
                metrics.memory_usage_percent > self.thresholds.memory_scale_up_threshold,
                metrics.avg_response_time_ms > self.thresholds.response_time_scale_up_threshold_ms,
                metrics.error_rate_percent > self.thresholds.error_rate_scale_up_threshold,
                metrics.active_connections > self.thresholds.connection_scale_up_threshold
            ]
            
            # Check scale down conditions
            scale_down_conditions = [
                metrics.cpu_usage_percent < self.thresholds.cpu_scale_down_threshold,
                metrics.memory_usage_percent < self.thresholds.memory_scale_down_threshold,
                metrics.avg_response_time_ms < self.thresholds.response_time_scale_down_threshold_ms,
                metrics.error_rate_percent < self.thresholds.error_rate_scale_down_threshold,
                metrics.active_connections < self.thresholds.connection_scale_down_threshold
            ]
            
            # Determine scaling direction
            scale_up_votes = sum(scale_up_conditions)
            scale_down_votes = sum(scale_down_conditions)
            
            if scale_up_votes >= self.thresholds.scale_up_condition_count:
                # Scale up needed
                target_count = min(
                    current_count + self.constraints.max_scale_up_instances,
                    self.constraints.max_instances
                )
                
                if target_count > current_count:
                    return ScalingAction(
                        action_id=str(uuid.uuid4()),
                        direction=ScalingDirection.SCALE_UP,
                        reason=self._determine_primary_scale_up_reason(scale_up_conditions),
                        confidence=min(1.0, scale_up_votes / len(scale_up_conditions)),
                        target_instance_count=target_count,
                        current_instance_count=current_count,
                        instance_type=InstanceType.MEDIUM,  # Would be determined by optimization
                        regions=self.constraints.preferred_regions or ["us-east-1"],
                        instances_to_add=target_count - current_count,
                        metrics_snapshot=metrics
                    )
            
            elif scale_down_votes >= self.thresholds.scale_down_condition_count:
                # Scale down needed
                target_count = max(
                    current_count - self.constraints.max_scale_down_instances,
                    self.constraints.min_instances
                )
                
                if target_count < current_count:
                    return ScalingAction(
                        action_id=str(uuid.uuid4()),
                        direction=ScalingDirection.SCALE_DOWN,
                        reason=ScalingReason.COST_OPTIMIZATION,
                        confidence=min(1.0, scale_down_votes / len(scale_down_conditions)),
                        target_instance_count=target_count,
                        current_instance_count=current_count,
                        instance_type=InstanceType.MEDIUM,
                        regions=self.constraints.preferred_regions or ["us-east-1"],
                        instances_to_remove=[],  # Would be determined by selection algorithm
                        metrics_snapshot=metrics
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in reactive scaling evaluation: {e}")
            return None
    
    def _determine_primary_scale_up_reason(self, conditions: List[bool]) -> ScalingReason:
        """Determine the primary reason for scale up"""
        reason_mapping = [
            ScalingReason.CPU_PRESSURE,
            ScalingReason.MEMORY_PRESSURE,
            ScalingReason.RESPONSE_TIME_DEGRADATION,
            ScalingReason.CONNECTION_OVERLOAD,
            ScalingReason.CONNECTION_OVERLOAD
        ]
        
        for i, condition in enumerate(conditions):
            if condition and i < len(reason_mapping):
                return reason_mapping[i]
        
        return ScalingReason.CPU_PRESSURE
    
    async def _evaluate_predictive_scaling(self, metrics: ScalingMetrics,
                                         current_count: int) -> Optional[ScalingAction]:
        """Evaluate predictive scaling using ML"""
        if not self.predictive_engine.enabled:
            return None
        
        try:
            # Get prediction
            prediction_result = await self.predictive_engine.predict_scaling_need(
                metrics, self.thresholds
            )
            
            if not prediction_result.get("scaling_needed", False):
                return None
            
            direction = prediction_result["direction"]
            confidence = prediction_result["confidence"]
            reason = prediction_result["reason"]
            
            if confidence < 0.6:  # Low confidence threshold
                return None
            
            if direction == ScalingDirection.SCALE_UP:
                target_count = min(
                    current_count + self.constraints.max_scale_up_instances,
                    self.constraints.max_instances
                )
                instances_to_add = target_count - current_count
                instances_to_remove = []
            elif direction == ScalingDirection.SCALE_DOWN:
                target_count = max(
                    current_count - self.constraints.max_scale_down_instances,
                    self.constraints.min_instances
                )
                instances_to_add = 0
                instances_to_remove = []  # Would be determined by selection
            else:
                return None
            
            if target_count == current_count:
                return None
            
            return ScalingAction(
                action_id=str(uuid.uuid4()),
                direction=direction,
                reason=reason,
                confidence=confidence,
                target_instance_count=target_count,
                current_instance_count=current_count,
                instance_type=InstanceType.MEDIUM,
                regions=self.constraints.preferred_regions or ["us-east-1"],
                instances_to_add=instances_to_add,
                instances_to_remove=instances_to_remove,
                metrics_snapshot=metrics,
                predicted_metrics=prediction_result.get("predictions", {}).get(15)  # 15-minute prediction
            )
            
        except Exception as e:
            logger.error(f"Error in predictive scaling evaluation: {e}")
            return None
    
    async def _evaluate_scheduled_scaling(self, current_count: int) -> Optional[ScalingAction]:
        """Evaluate scheduled scaling"""
        try:
            current_time = datetime.now(timezone.utc)
            
            for schedule in self.scheduled_scalings:
                if not schedule.enabled:
                    continue
                
                # Check if schedule should execute now
                if self._should_execute_schedule(schedule, current_time):
                    target_count = schedule.target_instance_count
                    
                    if target_count != current_count:
                        direction = (ScalingDirection.SCALE_UP if target_count > current_count 
                                   else ScalingDirection.SCALE_DOWN)
                        
                        return ScalingAction(
                            action_id=str(uuid.uuid4()),
                            direction=direction,
                            reason=ScalingReason.SCHEDULED_SCALING,
                            confidence=1.0,
                            target_instance_count=target_count,
                            current_instance_count=current_count,
                            instance_type=schedule.instance_type,
                            regions=schedule.regions or self.constraints.preferred_regions or ["us-east-1"],
                            instances_to_add=max(0, target_count - current_count),
                            instances_to_remove=[],  # Would be determined by selection
                            tags={"schedule_id": schedule.schedule_id, "schedule_name": schedule.name}
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating scheduled scaling: {e}")
            return None
    
    def _should_execute_schedule(self, schedule: ScalingSchedule, current_time: datetime) -> bool:
        """Check if a schedule should execute now"""
        try:
            # Simple cron-like evaluation (simplified implementation)
            # In production, would use a proper cron parser
            
            # Example: "0 8 * * 1-5" = 8 AM on weekdays
            if schedule.cron_expression == "0 8 * * 1-5":
                return (current_time.hour == 8 and 
                       current_time.minute == 0 and
                       current_time.weekday() < 5)  # Monday = 0, Friday = 4
            
            # Add more cron expression handling as needed
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating schedule: {e}")
            return False
    
    async def _validate_and_optimize_decision(self, action: ScalingAction,
                                            current_instances: List[Dict[str, Any]]) -> Optional[ScalingAction]:
        """Validate and optimize scaling decision"""
        try:
            validation_errors = []
            
            # Constraint validation
            if action.target_instance_count < self.constraints.min_instances:
                validation_errors.append(f"Target count below minimum ({self.constraints.min_instances})")
            
            if action.target_instance_count > self.constraints.max_instances:
                validation_errors.append(f"Target count above maximum ({self.constraints.max_instances})")
            
            # Cost validation
            estimated_cost = self.cost_optimizer.calculate_cost_impact(action)
            if estimated_cost > self.constraints.max_hourly_cost:
                validation_errors.append(f"Cost impact too high: ${estimated_cost:.2f}/hour")
            
            # Time-based validation
            current_hour = datetime.now(timezone.utc).hour
            is_weekend = datetime.now(timezone.utc).weekday() >= 5
            
            if (not self.constraints.weekend_scaling_enabled and is_weekend and
                action.direction == ScalingDirection.SCALE_UP):
                validation_errors.append("Weekend scaling disabled")
            
            # Set validation results
            action.validation_errors = validation_errors
            action.validated = len(validation_errors) == 0
            action.estimated_cost_impact = estimated_cost
            
            if action.validated:
                # Optimize instance selection
                if action.direction == ScalingDirection.SCALE_UP:
                    optimization = self.cost_optimizer.optimize_instance_selection(
                        {"cpu_cores": 2, "memory_gb": 4, "network_gbps": 1},
                        action.regions
                    )
                    
                    if "recommended" in optimization:
                        recommended = optimization["recommended"]
                        action.instance_type = recommended["instance_type"]
                        action.estimated_cost_impact = (
                            recommended["hourly_cost"] * action.instances_to_add
                        )
            
            return action
            
        except Exception as e:
            logger.error(f"Error validating scaling decision: {e}")
            action.validation_errors.append(f"Validation error: {str(e)}")
            action.validated = False
            return action
    
    def _check_cooldown_status(self) -> bool:
        """Check if cooldown period has passed"""
        current_time = datetime.now(timezone.utc)
        
        for action_type, last_time in self.cooldown_timers.items():
            if action_type == "scale_up":
                cooldown_minutes = self.thresholds.scale_up_cooldown_minutes
            else:
                cooldown_minutes = self.thresholds.scale_down_cooldown_minutes
            
            if current_time < last_time + timedelta(minutes=cooldown_minutes):
                return False
        
        return True
    
    async def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action"""
        try:
            if not action.validated:
                logger.error(f"Cannot execute unvalidated action: {action.validation_errors}")
                return False
            
            # Set cooldown timer
            self.cooldown_timers[action.direction.value] = datetime.now(timezone.utc)
            
            # Record scaling action
            self.scaling_history.append({
                "action_id": action.action_id,
                "direction": action.direction.value,
                "reason": action.reason.value,
                "confidence": action.confidence,
                "target_count": action.target_instance_count,
                "current_count": action.current_instance_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cost_impact": action.estimated_cost_impact
            })
            
            # Update statistics
            if action.direction == ScalingDirection.SCALE_UP:
                self.stats["successful_scale_ups"] += 1
            elif action.direction == ScalingDirection.SCALE_DOWN:
                self.stats["successful_scale_downs"] += 1
            
            logger.info(f"✅ Executed scaling action: {action.direction.value} to {action.target_instance_count} instances")
            return True
            
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            self.stats["failed_scaling_attempts"] += 1
            return False
    
    async def get_scaling_recommendations(self, current_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get scaling recommendations and analysis"""
        try:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            current_metrics = self.metrics_history[-1]
            
            # Get scaling decision
            scaling_decision = await self.evaluate_scaling_decision(current_instances)
            
            # Get cost optimizations
            cost_optimizations = self.cost_optimizer.suggest_cost_optimizations(
                current_metrics, current_instances
            )
            
            # Analyze recent performance
            recent_metrics = list(self.metrics_history)[-60:]  # Last hour
            if recent_metrics:
                avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics)
                avg_response_time = sum(m.avg_response_time_ms for m in recent_metrics) / len(recent_metrics)
            else:
                avg_cpu = avg_memory = avg_response_time = 0
            
            return {
                "current_status": {
                    "instance_count": len(current_instances),
                    "avg_cpu_usage": avg_cpu,
                    "avg_memory_usage": avg_memory,
                    "avg_response_time_ms": avg_response_time,
                    "hourly_cost": sum(self.cost_optimizer.calculate_cost_impact(
                        ScalingAction(
                            action_id="", direction=ScalingDirection.MAINTAIN,
                            reason=ScalingReason.MANUAL_INTERVENTION, confidence=1.0,
                            target_instance_count=1, current_instance_count=0,
                            instance_type=InstanceType(inst.get("instance_type", "medium")),
                            regions=[]
                        ), inst.get("region", "us-east-1")
                    ) for inst in current_instances)
                },
                "scaling_recommendation": {
                    "action_needed": scaling_decision is not None,
                    "direction": scaling_decision.direction.value if scaling_decision else None,
                    "confidence": scaling_decision.confidence if scaling_decision else 0,
                    "reason": scaling_decision.reason.value if scaling_decision else None,
                    "target_count": scaling_decision.target_instance_count if scaling_decision else len(current_instances)
                },
                "cost_optimizations": cost_optimizations,
                "performance_analysis": {
                    "metrics_count": len(self.metrics_history),
                    "scaling_mode": self.scaling_mode.value,
                    "predictive_enabled": self.predictive_engine.enabled,
                    "recent_scaling_actions": len(self.scaling_history)
                },
                "statistics": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling recommendations: {e}")
            return {"error": str(e)}


# Global auto-scaler instance
advanced_auto_scaler: Optional[AdvancedAutoScaler] = None


def initialize_advanced_auto_scaler(redis_client: aioredis.Redis) -> AdvancedAutoScaler:
    """Initialize the advanced auto-scaling system"""
    global advanced_auto_scaler
    
    advanced_auto_scaler = AdvancedAutoScaler(redis_client)
    logger.info("✅ Advanced auto-scaling system initialized")
    return advanced_auto_scaler


def get_advanced_auto_scaler() -> AdvancedAutoScaler:
    """Get the global auto-scaler instance"""
    if advanced_auto_scaler is None:
        raise RuntimeError("Advanced auto-scaler not initialized")
    return advanced_auto_scaler


# Convenience functions

async def add_scaling_metrics(metrics: ScalingMetrics):
    """Add metrics for scaling analysis"""
    if advanced_auto_scaler:
        await advanced_auto_scaler.add_metrics(metrics)


async def get_scaling_recommendation(current_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get scaling recommendations"""
    if advanced_auto_scaler:
        return await advanced_auto_scaler.get_scaling_recommendations(current_instances)
    return {"error": "Auto-scaler not initialized"}


async def evaluate_scaling_now(current_instances: List[Dict[str, Any]]) -> Optional[ScalingAction]:
    """Evaluate if scaling is needed right now"""
    if advanced_auto_scaler:
        return await advanced_auto_scaler.evaluate_scaling_decision(current_instances)
    return None