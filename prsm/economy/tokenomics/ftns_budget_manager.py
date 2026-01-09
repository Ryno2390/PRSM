#!/usr/bin/env python3
"""
FTNS Budget Manager

Manages computational budgets and resource allocation for FTNS token-based
AI training and inference operations within the PRSM ecosystem.

Core Functions:
- Budget allocation for training pipelines
- Resource usage tracking and limits
- Cost prediction for AI operations
- Token-based resource management
"""

import structlog
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum

logger = structlog.get_logger(__name__)


class BudgetType(Enum):
    """Types of computational budgets"""
    TRAINING = "training"
    INFERENCE = "inference"
    PIPELINE_EXECUTION = "pipeline_execution"
    DATA_PROCESSING = "data_processing"
    MODEL_EVALUATION = "model_evaluation"
    RESEARCH = "research"


class SpendingCategory(Enum):
    """Categories for budget spending tracking"""
    AGENT_EXECUTION = "agent_execution"
    AGENT_COORDINATION = "agent_coordination"
    MODEL_INFERENCE = "model_inference"
    CONTEXT_PROCESSING = "context_processing"
    REASONING_OPERATION = "reasoning_operation"
    BREAKTHROUGH_DETECTION = "breakthrough_detection"
    SYNTHESIS_GENERATION = "synthesis_generation"
    DATABASE_OPERATION = "database_operation"
    SYSTEM_OVERHEAD = "system_overhead"


class BudgetStatus(Enum):
    """Budget allocation status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    EXCEEDED = "exceeded"
    CANCELLED = "cancelled"
    PENDING = "pending"


class ResourceType(Enum):
    """Types of computational resources"""
    CPU_HOURS = "cpu_hours"
    GPU_HOURS = "gpu_hours"
    MEMORY_GB_HOURS = "memory_gb_hours"
    STORAGE_GB = "storage_gb"
    NETWORK_BANDWIDTH = "network_bandwidth"
    API_CALLS = "api_calls"
    TOKEN_OPERATIONS = "token_operations"


@dataclass
class ResourceUsage:
    """Resource usage record"""
    resource_type: ResourceType
    amount_used: Decimal
    cost_per_unit: Decimal
    total_cost: Decimal
    timestamp: datetime
    description: str = ""


@dataclass
class BudgetAllocation:
    """Budget allocation for a specific operation"""
    allocation_id: str
    user_id: str
    budget_type: BudgetType
    allocated_tokens: Decimal
    remaining_tokens: Decimal
    resource_limits: Dict[ResourceType, Decimal]
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_history: List[ResourceUsage] = field(default_factory=list)


class FTNSBudgetManager:
    """
    FTNS Budget Manager for computational resource allocation
    
    Manages token-based budgets for AI training, inference, and other
    computational operations within the PRSM ecosystem.
    """
    
    def __init__(self):
        """Initialize budget manager"""
        self.allocations: Dict[str, BudgetAllocation] = {}
        self.active_budgets: Dict[str, Any] = {}  # For orchestrator compatibility
        self.budget_history: Dict[str, Any] = {}  # For orchestrator compatibility
        self.resource_costs = {
            ResourceType.CPU_HOURS: Decimal('0.10'),
            ResourceType.GPU_HOURS: Decimal('2.50'),
            ResourceType.MEMORY_GB_HOURS: Decimal('0.05'),
            ResourceType.STORAGE_GB: Decimal('0.02'),
            ResourceType.NETWORK_BANDWIDTH: Decimal('0.01'),
            ResourceType.API_CALLS: Decimal('0.001'),
            ResourceType.TOKEN_OPERATIONS: Decimal('0.0001')
        }
        
        # Default resource limits for different budget types
        self.default_limits = {
            BudgetType.TRAINING: {
                ResourceType.CPU_HOURS: Decimal('10.0'),
                ResourceType.GPU_HOURS: Decimal('2.0'),
                ResourceType.MEMORY_GB_HOURS: Decimal('50.0'),
                ResourceType.API_CALLS: Decimal('1000')
            },
            BudgetType.INFERENCE: {
                ResourceType.CPU_HOURS: Decimal('1.0'),
                ResourceType.GPU_HOURS: Decimal('0.5'),
                ResourceType.MEMORY_GB_HOURS: Decimal('10.0'),
                ResourceType.API_CALLS: Decimal('100')
            },
            BudgetType.PIPELINE_EXECUTION: {
                ResourceType.CPU_HOURS: Decimal('5.0'),
                ResourceType.GPU_HOURS: Decimal('1.0'),
                ResourceType.MEMORY_GB_HOURS: Decimal('25.0'),
                ResourceType.API_CALLS: Decimal('500'),
                ResourceType.TOKEN_OPERATIONS: Decimal('1000000')  # 1M token operations
            },
            BudgetType.RESEARCH: {
                ResourceType.CPU_HOURS: Decimal('20.0'),
                ResourceType.GPU_HOURS: Decimal('5.0'),
                ResourceType.MEMORY_GB_HOURS: Decimal('100.0'),
                ResourceType.API_CALLS: Decimal('2000')
            }
        }
        
        logger.info("FTNSBudgetManager initialized", resource_types=len(self.resource_costs))
    
    def create_budget_allocation(self,
                               user_id: str,
                               budget_type: BudgetType,
                               token_amount: Decimal,
                               custom_limits: Optional[Dict[ResourceType, Decimal]] = None,
                               expiration_hours: Optional[int] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> BudgetAllocation:
        """Create new budget allocation for user"""
        try:
            allocation_id = f"budget_{len(self.allocations) + 1:06d}_{user_id}"
            
            # Use custom limits or defaults
            resource_limits = custom_limits or self.default_limits.get(budget_type, {})
            
            # Calculate expiration
            created_at = datetime.now(timezone.utc)
            expires_at = None
            if expiration_hours:
                from datetime import timedelta
                expires_at = created_at + timedelta(hours=expiration_hours)
            
            allocation = BudgetAllocation(
                allocation_id=allocation_id,
                user_id=user_id,
                budget_type=budget_type,
                allocated_tokens=token_amount,
                remaining_tokens=token_amount,
                resource_limits=resource_limits,
                created_at=created_at,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            self.allocations[allocation_id] = allocation
            
            logger.info("Budget allocation created",
                       allocation_id=allocation_id,
                       user_id=user_id,
                       budget_type=budget_type.value,
                       token_amount=float(token_amount))
            
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to create budget allocation: {e}", user_id=user_id)
            raise
    
    def get_allocation(self, allocation_id: str) -> Optional[BudgetAllocation]:
        """Get budget allocation by ID"""
        return self.allocations.get(allocation_id)
    
    def get_user_allocations(self, user_id: str) -> List[BudgetAllocation]:
        """Get all active budget allocations for user"""
        user_allocations = []
        current_time = datetime.now(timezone.utc)
        
        for allocation in self.allocations.values():
            if allocation.user_id == user_id:
                # Check if allocation is still valid (not expired)
                if allocation.expires_at is None or allocation.expires_at > current_time:
                    user_allocations.append(allocation)
        
        return user_allocations
    
    def record_resource_usage(self,
                             allocation_id: str,
                             resource_type: ResourceType,
                             amount_used: Decimal,
                             description: str = "") -> bool:
        """Record resource usage against budget allocation"""
        try:
            allocation = self.allocations.get(allocation_id)
            if not allocation:
                logger.warning("Budget allocation not found", allocation_id=allocation_id)
                return False
            
            # Check resource limits
            resource_limit = allocation.resource_limits.get(resource_type)
            if resource_limit is not None:
                current_usage = sum(
                    usage.amount_used for usage in allocation.usage_history
                    if usage.resource_type == resource_type
                )
                
                if current_usage + amount_used > resource_limit:
                    logger.warning("Resource limit would be exceeded",
                                  allocation_id=allocation_id,
                                  resource_type=resource_type.value,
                                  current_usage=float(current_usage),
                                  requested=float(amount_used),
                                  limit=float(resource_limit))
                    return False
            
            # Calculate cost
            cost_per_unit = self.resource_costs.get(resource_type, Decimal('0.01'))
            total_cost = amount_used * cost_per_unit
            
            # Check token budget
            if allocation.remaining_tokens < total_cost:
                logger.warning("Insufficient token budget",
                              allocation_id=allocation_id,
                              remaining_tokens=float(allocation.remaining_tokens),
                              required_tokens=float(total_cost))
                return False
            
            # Record usage
            usage = ResourceUsage(
                resource_type=resource_type,
                amount_used=amount_used,
                cost_per_unit=cost_per_unit,
                total_cost=total_cost,
                timestamp=datetime.now(timezone.utc),
                description=description
            )
            
            allocation.usage_history.append(usage)
            allocation.remaining_tokens -= total_cost
            
            logger.info("Resource usage recorded",
                       allocation_id=allocation_id,
                       resource_type=resource_type.value,
                       amount_used=float(amount_used),
                       cost=float(total_cost),
                       remaining_tokens=float(allocation.remaining_tokens))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record resource usage: {e}", allocation_id=allocation_id)
            return False
    
    def estimate_operation_cost(self,
                              budget_type: BudgetType,
                              estimated_resources: Dict[ResourceType, Decimal]) -> Decimal:
        """Estimate token cost for planned operation"""
        total_cost = Decimal('0')
        
        for resource_type, amount in estimated_resources.items():
            cost_per_unit = self.resource_costs.get(resource_type, Decimal('0.01'))
            resource_cost = amount * cost_per_unit
            total_cost += resource_cost
        
        # Add budget type multiplier
        type_multipliers = {
            BudgetType.TRAINING: Decimal('1.5'),      # Training is more expensive
            BudgetType.INFERENCE: Decimal('1.0'),     # Standard cost
            BudgetType.PIPELINE_EXECUTION: Decimal('1.2'),  # Slightly higher
            BudgetType.DATA_PROCESSING: Decimal('0.8'), # Lower cost
            BudgetType.RESEARCH: Decimal('1.3')       # Research premium
        }
        
        multiplier = type_multipliers.get(budget_type, Decimal('1.0'))
        total_cost *= multiplier
        
        return total_cost
    
    def check_budget_availability(self,
                                allocation_id: str,
                                planned_resources: Dict[ResourceType, Decimal]) -> Tuple[bool, str]:
        """Check if budget allocation can support planned resource usage"""
        allocation = self.allocations.get(allocation_id)
        if not allocation:
            return False, "Budget allocation not found"
        
        # Check expiration
        if allocation.expires_at and allocation.expires_at <= datetime.now(timezone.utc):
            return False, "Budget allocation expired"
        
        # Estimate cost
        estimated_cost = self.estimate_operation_cost(allocation.budget_type, planned_resources)
        
        # Check token budget
        if allocation.remaining_tokens < estimated_cost:
            return False, f"Insufficient tokens: {allocation.remaining_tokens} < {estimated_cost}"
        
        # Check resource limits
        for resource_type, planned_amount in planned_resources.items():
            resource_limit = allocation.resource_limits.get(resource_type)
            if resource_limit is not None:
                current_usage = sum(
                    usage.amount_used for usage in allocation.usage_history
                    if usage.resource_type == resource_type
                )
                
                if current_usage + planned_amount > resource_limit:
                    return False, f"Resource limit exceeded: {resource_type.value}"
        
        return True, "Budget available"
    
    def get_usage_summary(self, allocation_id: str) -> Dict[str, Any]:
        """Get usage summary for budget allocation"""
        allocation = self.allocations.get(allocation_id)
        if not allocation:
            return {}
        
        # Calculate resource usage by type
        resource_usage = {}
        total_cost = Decimal('0')
        
        for usage in allocation.usage_history:
            resource_type = usage.resource_type.value
            if resource_type not in resource_usage:
                resource_usage[resource_type] = {
                    'amount_used': Decimal('0'),
                    'total_cost': Decimal('0'),
                    'usage_count': 0
                }
            
            resource_usage[resource_type]['amount_used'] += usage.amount_used
            resource_usage[resource_type]['total_cost'] += usage.total_cost
            resource_usage[resource_type]['usage_count'] += 1
            total_cost += usage.total_cost
        
        return {
            'allocation_id': allocation_id,
            'user_id': allocation.user_id,
            'budget_type': allocation.budget_type.value,
            'allocated_tokens': float(allocation.allocated_tokens),
            'remaining_tokens': float(allocation.remaining_tokens),
            'tokens_used': float(allocation.allocated_tokens - allocation.remaining_tokens),
            'utilization_percentage': float((allocation.allocated_tokens - allocation.remaining_tokens) / allocation.allocated_tokens * 100),
            'resource_usage': {
                resource_type: {
                    'amount_used': float(data['amount_used']),
                    'total_cost': float(data['total_cost']),
                    'usage_count': data['usage_count']
                }
                for resource_type, data in resource_usage.items()
            },
            'total_operations': len(allocation.usage_history),
            'created_at': allocation.created_at.isoformat(),
            'expires_at': allocation.expires_at.isoformat() if allocation.expires_at else None
        }
    
    def cleanup_expired_allocations(self) -> int:
        """Clean up expired budget allocations"""
        current_time = datetime.now(timezone.utc)
        expired_count = 0
        
        expired_ids = []
        for allocation_id, allocation in self.allocations.items():
            if allocation.expires_at and allocation.expires_at <= current_time:
                expired_ids.append(allocation_id)
        
        for allocation_id in expired_ids:
            del self.allocations[allocation_id]
            expired_count += 1
        
        if expired_count > 0:
            logger.info("Expired budget allocations cleaned up", count=expired_count)
        
        return expired_count
    
    async def spend_budget_amount(self, budget_id: str, amount: Decimal, spending_category: str, description: str = "") -> bool:
        """
        Spend amount from budget - compatibility method for orchestrator
        
        Args:
            budget_id: Budget allocation ID
            amount: Amount to spend
            spending_category: Category of spending
            description: Description of spending
            
        Returns:
            True if spending was successful, False otherwise
        """
        try:
            # Record the spending as resource usage
            return self.record_resource_usage(
                allocation_id=budget_id,
                resource_type=ResourceType.TOKEN_OPERATIONS,
                amount_used=amount,
                description=f"{spending_category}: {description}"
            )
        except Exception as e:
            logger.error(f"Failed to spend budget amount: {e}", 
                        budget_id=budget_id, amount=float(amount))
            return False

    async def create_session_budget(self, session, user_input, budget_config: dict = None) -> 'BudgetAllocation':
        """
        Create a budget allocation for a NWTN session
        
        Args:
            session: PRSMSession object
            user_input: UserInput object
            budget_config: Budget configuration dict
            
        Returns:
            BudgetAllocation for the session
        """
        # Extract user_id and session_id from the provided objects
        user_id = session.user_id if hasattr(session, 'user_id') else str(session)
        session_id = str(session.session_id) if hasattr(session, 'session_id') else str(session)
        
        # Determine budget amount from config or use default
        base_budget = Decimal('50000')  # Large default for NWTN maximum capacity
        if budget_config and 'base_budget' in budget_config:
            base_budget = Decimal(str(budget_config['base_budget']))
        
        logger.info("Creating session budget", 
                   user_id=user_id, session_id=session_id, budget=base_budget)
        
        # Create budget allocation for the session
        allocation = self.create_budget_allocation(
            user_id=user_id,
            budget_type=BudgetType.PIPELINE_EXECUTION,
            token_amount=base_budget,
            metadata={"session_id": session_id, "description": f"NWTN session budget for {session_id}"}
        )
        
        logger.info("Session budget created successfully", 
                   allocation_id=allocation.allocation_id, budget=base_budget)
        
        # Create compatibility wrapper for orchestrator expectations
        class SessionBudget:
            def __init__(self, allocation):
                self.allocation = allocation
                self.budget_id = allocation.allocation_id
                self.total_budget = allocation.allocated_tokens
                self.initial_prediction = None
        
        return SessionBudget(allocation)


# Global budget manager instance
_global_budget_manager: Optional[FTNSBudgetManager] = None


def get_ftns_budget_manager() -> FTNSBudgetManager:
    """Get global FTNS budget manager instance"""
    global _global_budget_manager
    if _global_budget_manager is None:
        _global_budget_manager = FTNSBudgetManager()
    return _global_budget_manager


# Convenience functions
def create_training_budget(user_id: str, token_amount: Decimal,
                          expiration_hours: int = 24) -> BudgetAllocation:
    """Create budget allocation for AI training"""
    manager = get_ftns_budget_manager()
    return manager.create_budget_allocation(
        user_id=user_id,
        budget_type=BudgetType.TRAINING,
        token_amount=token_amount,
        expiration_hours=expiration_hours,
        metadata={'purpose': 'ai_training', 'auto_created': True}
    )


def create_pipeline_budget(user_id: str, token_amount: Decimal,
                          candidates_count: int = 5040,
                          operations_count: int = 177_000_000) -> BudgetAllocation:
    """Create budget allocation for NWTN pipeline execution"""
    manager = get_ftns_budget_manager()
    
    # Custom limits for pipeline execution
    custom_limits = {
        ResourceType.CPU_HOURS: Decimal('8.0'),  # Higher for complex processing
        ResourceType.MEMORY_GB_HOURS: Decimal('50.0'),
        ResourceType.API_CALLS: Decimal('1000'),
        ResourceType.TOKEN_OPERATIONS: Decimal(str(operations_count))
    }
    
    return manager.create_budget_allocation(
        user_id=user_id,
        budget_type=BudgetType.PIPELINE_EXECUTION,
        token_amount=token_amount,
        custom_limits=custom_limits,
        expiration_hours=12,  # 12 hour expiration for pipeline runs
        metadata={
            'purpose': 'nwtn_pipeline',
            'candidates_count': candidates_count,
            'operations_count': operations_count,
            'auto_created': True
        }
    )
