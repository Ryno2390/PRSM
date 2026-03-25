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
    TOOL_EXECUTION = "tool_execution"
    MARKETPLACE_TRADING = "marketplace_trading"


class BudgetStatus(Enum):
    """Budget allocation status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    EXCEEDED = "exceeded"
    CANCELLED = "cancelled"
    PENDING = "pending"
    SUSPENDED = "suspended"


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
        self.session_budgets: Dict[str, Any] = {}  # FTNSBudget objects by budget_id
        self.pending_expansions: Dict[str, Any] = {}  # Pending expansion requests
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
    
    async def predict_session_cost(self, user_input, session) -> 'BudgetPrediction':
        """
        Predict the cost of a session based on user input complexity.

        Args:
            user_input: UserInput object with prompt
            session: PRSMSession object

        Returns:
            BudgetPrediction with estimated costs
        """
        prompt = user_input.prompt if hasattr(user_input, 'prompt') else str(user_input)
        prompt_lower = prompt.lower()

        # Estimate complexity based on keywords
        complexity_keywords = ["analyze", "comprehensive", "simulation", "monte carlo",
                               "dimensional", "quantum", "photonic", "apm", "calculate",
                               "multi", "complex", "resonance"]
        simple_keywords = ["what", "how", "is", "define", "speed", "simple", "basic"]

        complexity_score = sum(1 for kw in complexity_keywords if kw in prompt_lower)
        simplicity_score = sum(1 for kw in simple_keywords if kw in prompt_lower)

        # Normalized complexity between 0 and 1
        query_complexity = min(1.0, complexity_score / max(len(complexity_keywords) * 0.3, 1))
        if simplicity_score > complexity_score:
            query_complexity = max(0.0, query_complexity - 0.3)

        confidence_score = 0.7 + (0.2 * (1 - query_complexity))  # Higher confidence for simpler queries

        # Base cost estimates by category
        base_cost = Decimal('10') + Decimal(str(int(query_complexity * 90)))

        category_estimates = {
            SpendingCategory.MODEL_INFERENCE: base_cost * Decimal('0.4'),
            SpendingCategory.AGENT_COORDINATION: base_cost * Decimal('0.25'),
            SpendingCategory.CONTEXT_PROCESSING: base_cost * Decimal('0.15'),
            SpendingCategory.REASONING_OPERATION: base_cost * Decimal('0.1'),
            SpendingCategory.SYNTHESIS_GENERATION: base_cost * Decimal('0.1'),
        }

        estimated_total = sum(category_estimates.values())

        return BudgetPrediction(
            estimated_total_cost=estimated_total,
            confidence_score=round(confidence_score, 2),
            query_complexity=round(query_complexity, 2),
            category_estimates=category_estimates,
            estimated_duration_seconds=float(10 + query_complexity * 290),
            risk_level="high" if query_complexity > 0.7 else "medium" if query_complexity > 0.4 else "low",
        )

    async def spend_budget_amount(self, budget_id: str, amount: Decimal, spending_category: Any, description: str = "", release_reserved: bool = False) -> bool:
        """
        Spend amount from an FTNSBudget or BudgetAllocation.

        Args:
            budget_id: Budget ID
            amount: Amount to spend
            spending_category: SpendingCategory enum or string
            description: Description of spending
            release_reserved: If True, also release reserved budget

        Returns:
            True if spending was successful, False otherwise
        """
        try:
            # Check session_budgets first (FTNSBudget objects)
            if budget_id in self.session_budgets:
                budget = self.session_budgets[budget_id]
                if budget.available_budget < amount and not budget.auto_expand_enabled:
                    return False

                # Check if auto-expansion needed
                if budget.available_budget < amount and budget.auto_expand_enabled:
                    # Auto-expand if within limits
                    needed = amount - budget.available_budget
                    if needed <= budget.max_auto_expand:
                        budget.total_budget += needed
                    else:
                        return False

                budget.total_spent += amount

                # Handle reservation release
                if release_reserved and budget.total_reserved > Decimal('0'):
                    release_amount = min(amount, budget.total_reserved)
                    budget.total_reserved -= release_amount

                # Check emergency threshold (95% usage)
                if budget.utilization_percentage >= 95 and not budget.auto_expand_enabled:
                    budget.status = BudgetStatus.SUSPENDED

                # Update category allocation
                cat_key = spending_category if isinstance(spending_category, SpendingCategory) else SpendingCategory(spending_category)
                if cat_key in budget.category_allocations:
                    budget.category_allocations[cat_key].spent_amount += amount

                # Record spending history
                budget.spending_history.append({
                    "action": "spend",
                    "amount": float(amount),
                    "category": cat_key.value if isinstance(cat_key, SpendingCategory) else str(cat_key),
                    "description": description,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                logger.info("Budget spent", budget_id=budget_id, amount=float(amount))
                return True

            # Fallback to BudgetAllocation logic
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

    async def reserve_budget_amount(self, budget_id: str, amount: Decimal, spending_category: Any, description: str = "") -> bool:
        """Reserve budget for a pending operation."""
        try:
            if budget_id in self.session_budgets:
                budget = self.session_budgets[budget_id]
                if budget.available_budget < amount:
                    return False
                budget.total_reserved += amount
                budget.spending_history.append({
                    "action": "reserve",
                    "amount": float(amount),
                    "category": spending_category.value if isinstance(spending_category, SpendingCategory) else str(spending_category),
                    "description": description,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reserve budget: {e}", budget_id=budget_id)
            return False

    async def request_budget_expansion(self, budget_id: str, amount: Decimal, reason: str) -> 'BudgetExpansionRequest':
        """Request a budget expansion, auto-approving if within auto-expand limits."""
        import uuid
        if budget_id not in self.session_budgets:
            raise ValueError(f"Budget {budget_id} not found")

        budget = self.session_budgets[budget_id]
        request_id = str(uuid.uuid4())

        request = BudgetExpansionRequest(
            request_id=request_id,
            budget_id=budget_id,
            user_id=budget.user_id,
            requested_amount=amount,
            reason=reason,
        )

        # Auto-approve if within auto-expand limits
        if budget.auto_expand_enabled and amount <= budget.max_auto_expand:
            request.approved = True
            request.approved_amount = amount
            request.auto_generated = True
            budget.total_budget += amount
            logger.info("Budget auto-expanded", budget_id=budget_id, amount=float(amount))
        else:
            # Queue for manual approval
            self.pending_expansions[request_id] = request
            logger.info("Budget expansion queued for approval", request_id=request_id)

        return request

    async def approve_budget_expansion(self, request_id: str, approved: bool, approved_amount: Decimal, reason: str = "") -> bool:
        """Approve or deny a pending budget expansion request."""
        if request_id not in self.pending_expansions:
            return False

        request = self.pending_expansions[request_id]
        request.approved = approved
        request.approved_amount = approved_amount
        request.approval_reason = reason

        if approved and request.budget_id in self.session_budgets:
            budget = self.session_budgets[request.budget_id]
            budget.total_budget += approved_amount

        del self.pending_expansions[request_id]
        return True

    async def get_budget_status(self, budget_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive budget status and analytics."""
        if budget_id not in self.session_budgets:
            return None

        budget = self.session_budgets[budget_id]

        # Build category breakdown
        category_breakdown = {}
        for cat, alloc in budget.category_allocations.items():
            category_breakdown[cat.value] = {
                "allocated": float(alloc.allocated_amount),
                "spent": float(alloc.spent_amount),
                "reserved": float(alloc.reserved_amount),
            }

        return {
            "budget_id": str(budget.budget_id),
            "session_id": str(budget.session_id),
            "user_id": budget.user_id,
            "total_budget": float(budget.total_budget),
            "total_spent": float(budget.total_spent),
            "total_reserved": float(budget.total_reserved),
            "available_budget": float(budget.available_budget),
            "utilization_percentage": budget.utilization_percentage,
            "status": budget.status.value,
            "category_breakdown": category_breakdown,
        }

    async def create_session_budget(self, session, user_input, budget_config: dict = None) -> 'FTNSBudget':
        """
        Create a session-level FTNSBudget with prediction, category allocations, and expansion config.

        Args:
            session: PRSMSession object
            user_input: UserInput object
            budget_config: Budget configuration dict

        Returns:
            FTNSBudget for the session
        """
        import uuid

        budget_config = budget_config or {}
        user_id = session.user_id if hasattr(session, 'user_id') else str(session)
        session_id = session.session_id if hasattr(session, 'session_id') else str(session)

        # Get prediction
        prediction = await self.predict_session_cost(user_input, session)

        # Determine total budget
        if 'total_budget' in budget_config:
            total_budget = Decimal(str(budget_config['total_budget']))
        else:
            total_budget = prediction.recommended_budget

        # Auto-expand config
        auto_expand = budget_config.get('auto_expand', False)
        max_auto_expand = Decimal(str(budget_config.get('max_auto_expand', 0)))
        expansion_increment = Decimal(str(budget_config.get('expansion_increment', 25)))

        # Build category allocations
        category_allocations: Dict[SpendingCategory, CategoryAllocation] = {}
        raw_cat_config = budget_config.get('category_allocations', {})

        if raw_cat_config:
            # Use user-provided allocations
            for cat_name, cat_cfg in raw_cat_config.items():
                try:
                    cat = SpendingCategory(cat_name)
                except ValueError:
                    continue
                if 'amount' in cat_cfg:
                    alloc_amount = Decimal(str(cat_cfg['amount']))
                elif 'percentage' in cat_cfg:
                    alloc_amount = total_budget * Decimal(str(cat_cfg['percentage'])) / Decimal('100')
                else:
                    alloc_amount = Decimal('0')
                category_allocations[cat] = CategoryAllocation(
                    category=cat, allocated_amount=alloc_amount
                )
        else:
            # Use prediction-based allocations
            for cat, est in prediction.category_estimates.items():
                if total_budget > Decimal('0'):
                    ratio = est / prediction.estimated_total_cost
                    alloc_amount = total_budget * ratio
                else:
                    alloc_amount = est
                category_allocations[cat] = CategoryAllocation(
                    category=cat, allocated_amount=alloc_amount
                )

        budget_id = str(uuid.uuid4())
        budget = FTNSBudget(
            budget_id=budget_id,
            session_id=session_id,
            user_id=user_id,
            total_budget=total_budget,
            status=BudgetStatus.ACTIVE,
            initial_prediction=prediction,
            category_allocations=category_allocations,
            auto_expand_enabled=auto_expand,
            max_auto_expand=max_auto_expand,
            expansion_increment=expansion_increment,
        )

        self.session_budgets[budget_id] = budget
        self.active_budgets[budget_id] = budget

        logger.info("Session FTNSBudget created",
                   budget_id=budget_id, user_id=user_id, total_budget=float(total_budget))

        return budget


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


@dataclass
class CategoryAllocation:
    """Budget allocation for a spending category"""
    category: SpendingCategory
    allocated_amount: Decimal
    spent_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    reserved_amount: Decimal = field(default_factory=lambda: Decimal('0'))


@dataclass
class BudgetExpansionRequest:
    """Request for budget expansion"""
    request_id: str
    budget_id: str
    user_id: str
    requested_amount: Decimal
    reason: str
    auto_generated: bool = False
    approved: Optional[bool] = None
    approved_amount: Optional[Decimal] = None
    approval_reason: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FTNSBudget:
    """Session budget with full tracking and expansion capabilities"""
    budget_id: str
    session_id: Any
    user_id: str
    total_budget: Decimal
    status: BudgetStatus
    initial_prediction: Optional['BudgetPrediction']
    category_allocations: Dict[SpendingCategory, CategoryAllocation]
    auto_expand_enabled: bool = False
    max_auto_expand: Decimal = field(default_factory=lambda: Decimal('0'))
    expansion_increment: Decimal = field(default_factory=lambda: Decimal('25'))
    total_spent: Decimal = field(default_factory=lambda: Decimal('0'))
    total_reserved: Decimal = field(default_factory=lambda: Decimal('0'))
    spending_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def available_budget(self) -> Decimal:
        return self.total_budget - self.total_spent - self.total_reserved

    @property
    def utilization_percentage(self) -> float:
        if self.total_budget == Decimal('0'):
            return 0.0
        return float((self.total_spent / self.total_budget) * 100)


# Aliases expected by budget_api.py
BudgetExpandRequest = BudgetExpansionRequest


@dataclass
class BudgetPrediction:
    """Budget prediction for a session or operation"""
    estimated_total_cost: Decimal
    confidence_score: float
    query_complexity: float
    category_estimates: Dict[SpendingCategory, Decimal]
    estimated_duration_seconds: float = 0.0
    recommended_budget: Decimal = Decimal('0')
    risk_level: str = "low"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.recommended_budget == Decimal('0'):
            self.recommended_budget = self.estimated_total_cost * Decimal('1.2')


@dataclass
class BudgetAlert:
    """Budget alert for monitoring and notifications"""
    alert_id: str
    user_id: str
    alert_type: str  # 'warning', 'critical', 'exceeded'
    message: str
    current_spend: Decimal
    budget_limit: Decimal
    percentage_used: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
