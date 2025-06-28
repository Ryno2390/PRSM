"""
Evolution System Data Models

Core data structures for DGM-enhanced evolution system.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid


class SafetyStatus(str, Enum):
    """Safety validation status for solutions."""
    PENDING = "PENDING"
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    QUARANTINED = "QUARANTINED"


class RiskLevel(str, Enum):
    """Risk levels for modifications and solutions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ImpactLevel(str, Enum):
    """Impact levels for system modifications."""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SelectionStrategy(str, Enum):
    """Selection strategies for parent solutions."""
    PERFORMANCE_WEIGHTED = "PERFORMANCE_WEIGHTED"
    NOVELTY_WEIGHTED = "NOVELTY_WEIGHTED"
    QUALITY_DIVERSITY = "QUALITY_DIVERSITY"
    RANDOM = "RANDOM"
    STEPPING_STONE_FOCUSED = "STEPPING_STONE_FOCUSED"
    PURE_QUALITY = "PURE_QUALITY"
    PURE_DIVERSITY = "PURE_DIVERSITY"
    NOVELTY_FOCUSED = "NOVELTY_FOCUSED"


class ComponentType(str, Enum):
    """Types of components that can evolve."""
    TASK_ORCHESTRATOR = "TASK_ORCHESTRATOR"
    INTELLIGENT_ROUTER = "INTELLIGENT_ROUTER"
    GOVERNANCE_SYSTEM = "GOVERNANCE_SYSTEM"
    NETWORK_ECONOMICS = "NETWORK_ECONOMICS"
    CHRONOS_CLEARING = "CHRONOS_CLEARING"
    SAFETY_MONITOR = "SAFETY_MONITOR"
    CUSTOM = "CUSTOM"


@dataclass
class PerformanceStats:
    """Statistical analysis of performance results."""
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    confidence_interval: tuple[float, float]
    sample_size: int
    statistical_significance: float
    noise_level: float


class EvaluationResult(BaseModel):
    """Result from evaluating a solution."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    solution_id: str
    component_type: ComponentType
    
    # Performance metrics
    performance_score: float = Field(ge=0.0, le=1.0)
    task_success_rate: float = Field(ge=0.0, le=1.0)
    latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    resource_efficiency: Optional[float] = None
    
    # Evaluation details
    evaluation_tier: str  # quick, comprehensive, production
    tasks_evaluated: int
    tasks_successful: int
    evaluation_duration_seconds: float
    
    # Statistical analysis
    performance_stats: Optional[PerformanceStats] = None
    confidence_level: float = Field(default=0.95)
    
    # Metadata
    evaluator_version: str
    benchmark_suite: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModificationProposal(BaseModel):
    """Proposal for modifying a solution."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    solution_id: str
    component_type: ComponentType
    
    # Modification details
    modification_type: str  # code_change, config_update, architecture_change
    description: str
    rationale: str
    
    # Technical details
    code_changes: Dict[str, Any] = Field(default_factory=dict)
    config_changes: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    
    # Impact assessment
    estimated_performance_impact: float
    risk_level: RiskLevel
    impact_level: ImpactLevel
    
    # Resource requirements
    compute_requirements: Dict[str, Any] = Field(default_factory=dict)
    memory_requirements: Dict[str, Any] = Field(default_factory=dict)
    storage_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Safety considerations
    safety_considerations: List[str] = Field(default_factory=list)
    rollback_plan: str
    
    # Metadata
    proposer_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafetyValidationResult(BaseModel):
    """Result from safety validation of a modification."""
    
    modification_id: str
    passed: bool
    risk_level: RiskLevel
    
    # Validation checks
    capability_bounds_check: bool = True
    resource_limits_check: bool = True
    behavioral_constraints_check: bool = True
    impact_assessment_check: bool = True
    
    # Failed checks details
    failed_checks: List[str] = Field(default_factory=list)
    safety_violations: List[str] = Field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    required_mitigations: List[str] = Field(default_factory=list)
    
    # Metadata
    validator_id: str
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModificationResult(BaseModel):
    """Result from applying a modification."""
    
    modification_id: str
    success: bool
    
    # Result details
    performance_before: Optional[float] = None
    performance_after: Optional[float] = None
    performance_delta: Optional[float] = None
    
    # Execution details
    execution_time_seconds: float
    resources_used: Dict[str, Any] = Field(default_factory=dict)
    
    # Error information (if failed)
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Rollback information
    rollback_required: bool = False
    rollback_successful: Optional[bool] = None
    
    # Post-modification validation
    functionality_preserved: bool = True
    safety_status: SafetyStatus = SafetyStatus.PENDING
    
    # Metadata
    executor_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class Checkpoint:
    """System state checkpoint for rollback."""
    id: str
    component_id: str
    component_type: ComponentType
    state_snapshot: Dict[str, Any]
    configuration_snapshot: Dict[str, Any]
    timestamp: datetime
    storage_location: str


@dataclass
class GenealogyNode:
    """Node in the genealogy tree."""
    solution_id: str
    parent_ids: List[str]
    child_ids: List[str]
    generation: int
    creation_timestamp: datetime
    performance_score: float


class GenealogyTree(BaseModel):
    """Tree structure representing solution genealogy."""
    
    root_solution_id: str
    nodes: Dict[str, GenealogyNode] = Field(default_factory=dict)
    
    def add_node(self, node: GenealogyNode):
        """Add a node to the genealogy tree."""
        self.nodes[node.solution_id] = node
    
    def get_ancestors(self, solution_id: str, generations_back: int = 5) -> List[GenealogyNode]:
        """Get ancestors of a solution up to specified generations."""
        ancestors = []
        current_generation = [solution_id]
        
        for _ in range(generations_back):
            next_generation = []
            for node_id in current_generation:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    ancestors.append(node)
                    next_generation.extend(node.parent_ids)
            
            if not next_generation:
                break
            current_generation = next_generation
        
        return ancestors
    
    def get_descendants(self, solution_id: str, generations_forward: int = 5) -> List[GenealogyNode]:
        """Get descendants of a solution up to specified generations."""
        descendants = []
        current_generation = [solution_id]
        
        for _ in range(generations_forward):
            next_generation = []
            for node_id in current_generation:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    descendants.append(node)
                    next_generation.extend(node.child_ids)
            
            if not next_generation:
                break
            current_generation = next_generation
        
        return descendants
    
    def get_lineage_to_root(self, solution_id: str) -> List[GenealogyNode]:
        """Get complete lineage from solution to root."""
        lineage = []
        current_id = solution_id
        
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            lineage.append(node)
            
            # Follow the primary parent (first parent)
            if node.parent_ids:
                current_id = node.parent_ids[0]
            else:
                break
        
        return lineage


@dataclass
class ArchiveStats:
    """Statistics about the evolution archive."""
    total_solutions: int
    active_solutions: int
    generations: int
    average_performance: float
    best_performance: float
    performance_improvement_rate: float
    diversity_score: float
    stepping_stones_discovered: int
    breakthrough_solutions: int
    safety_violations: int
    last_updated: datetime


class SynchronizationResult(BaseModel):
    """Result from archive synchronization."""
    
    solutions_shared: int
    solutions_received: int
    conflicts_resolved: int
    synchronization_time_seconds: float
    bandwidth_used_mb: float
    
    # Error information
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NetworkEvolutionResult(BaseModel):
    """Result from network-wide evolution."""
    
    participating_nodes: int
    improvements_discovered: int
    consensus_achieved: bool
    deployment_successful: bool
    
    # Performance metrics
    network_performance_before: float
    network_performance_after: float
    network_performance_delta: float
    
    # Coordination metrics
    coordination_time_seconds: float
    consensus_time_seconds: float
    deployment_time_seconds: float
    
    # Resource usage
    total_compute_hours: float
    total_bandwidth_mb: float
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }