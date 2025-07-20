"""
PRSM FTNS Budget Management System

🎯 USER-CENTRIC BUDGET CONTROL:
- Per-prompt budget allocation with user-defined limits
- Real-time budget monitoring during query execution
- Dynamic budget expansion with user authorization
- Predictive cost estimation for complex workflows
- Multi-resource budget tracking (models, agents, tools, datasets)
- Budget overflow protection with graceful degradation

This module addresses the core UX challenge where users can't predict
FTNS costs for open-ended queries like Prismatica's Feynman tree analysis.
Users can set budgets upfront and get prompted when additional resources
are needed, preventing runaway costs while maintaining flexibility.

Key Features:
1. Budget Allocation: Set FTNS budgets per prompt/session with granular control
2. Real-Time Monitoring: Track spending as queries execute with live updates
3. Predictive Estimation: Estimate costs before execution based on complexity
4. Dynamic Expansion: Request additional budget when limits approached
5. Resource Breakdown: Track spending across models, agents, tools, datasets
6. User Authorization: Require explicit approval for budget overages
7. Marketplace Integration: Budget for U2U and A2A transactions
8. Emergency Controls: Circuit breakers for runaway spending
"""

import asyncio
import time
import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal, getcontext
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.models import PRSMSession, FTNSTransaction, UserInput
from ..tokenomics.ftns_service import FTNSService
# Legacy FTNS marketplace import removed - not used in current implementation

# Set high precision for financial calculations
getcontext().prec = 28

logger = structlog.get_logger(__name__)


class BudgetStatus(str, Enum):
    """Budget status states"""
    ACTIVE = "active"                    # Budget is active and tracking
    EXCEEDED = "exceeded"                # Budget limit exceeded, needs authorization
    DEPLETED = "depleted"                # Budget fully consumed
    SUSPENDED = "suspended"              # Budget suspended due to policy
    COMPLETED = "completed"              # Session completed within budget
    CANCELLED = "cancelled"              # Budget cancelled by user


class SpendingCategory(str, Enum):
    """Spending categories for budget tracking"""
    MODEL_INFERENCE = "model_inference"         # LLM API calls and inference
    AGENT_COORDINATION = "agent_coordination"   # Agent orchestration costs
    TOOL_EXECUTION = "tool_execution"           # MCP tool usage costs
    DATA_ACCESS = "data_access"                 # Dataset and knowledge access
    MARKETPLACE_TRADING = "marketplace_trading" # U2U and A2A transactions
    CONTEXT_PROCESSING = "context_processing"   # Context compression and management
    SAFETY_VALIDATION = "safety_validation"    # Safety and security checks
    STORAGE_OPERATIONS = "storage_operations"   # IPFS and data storage
    NETWORK_OPERATIONS = "network_operations"   # P2P network participation


class BudgetAlert(BaseModel):
    """Budget alert configuration"""
    alert_id: UUID = Field(default_factory=uuid4)
    threshold_percentage: float = Field(ge=0.0, le=100.0)  # Alert at % of budget
    alert_type: str = Field(default="warning")  # warning, critical, emergency
    notification_channels: List[str] = Field(default=["in_app"])  # in_app, email, webhook
    message_template: str = Field(default="Budget {percentage}% consumed")
    auto_actions: List[str] = Field(default_factory=list)  # pause, request_more, degrade
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BudgetAllocation(BaseModel):
    """Budget allocation for a specific spending category"""
    category: SpendingCategory
    allocated_amount: Decimal = Field(ge=0)
    spent_amount: Decimal = Field(default=Decimal('0'), ge=0)
    reserved_amount: Decimal = Field(default=Decimal('0'), ge=0)  # Reserved for pending ops
    
    @property
    def available_amount(self) -> Decimal:
        """Calculate available budget remaining"""
        return self.allocated_amount - self.spent_amount - self.reserved_amount
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate budget utilization percentage"""
        if self.allocated_amount == 0:
            return 0.0
        return float((self.spent_amount / self.allocated_amount) * 100)
    
    def can_spend(self, amount: Decimal) -> bool:
        """Check if amount can be spent within allocation"""
        return self.available_amount >= amount


class BudgetPrediction(BaseModel):
    """Predictive budget estimation for a query"""
    prediction_id: UUID = Field(default_factory=uuid4)
    query_complexity: float = Field(ge=0.0, le=1.0)
    estimated_total_cost: Decimal
    category_estimates: Dict[SpendingCategory, Decimal]
    confidence_score: float = Field(ge=0.0, le=1.0)
    prediction_factors: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_recommended_budget(self, safety_multiplier: float = 1.5) -> Decimal:
        """Get recommended budget with safety margin"""
        return self.estimated_total_cost * Decimal(str(safety_multiplier))


class FTNSBudget(BaseModel):
    """Comprehensive FTNS budget for a session or prompt"""
    budget_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: str
    
    # Budget configuration
    total_budget: Decimal = Field(gt=0)
    auto_expand_enabled: bool = Field(default=True)
    max_auto_expand: Decimal = Field(default=Decimal('0'))  # 0 = no limit
    expansion_increment: Decimal = Field(default=Decimal('50'))
    
    # Budget allocations by category
    category_allocations: Dict[SpendingCategory, BudgetAllocation] = Field(default_factory=dict)
    
    # Spending tracking
    total_spent: Decimal = Field(default=Decimal('0'), ge=0)
    total_reserved: Decimal = Field(default=Decimal('0'), ge=0)
    spending_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Alert configuration
    budget_alerts: List[BudgetAlert] = Field(default_factory=list)
    triggered_alerts: List[str] = Field(default_factory=list)
    
    # Status and metadata
    status: BudgetStatus = Field(default=BudgetStatus.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Prediction and estimation
    initial_prediction: Optional[BudgetPrediction] = None
    cost_estimates: Dict[str, Decimal] = Field(default_factory=dict)
    
    @property
    def available_budget(self) -> Decimal:
        """Calculate total available budget"""
        return self.total_budget - self.total_spent - self.total_reserved
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate overall budget utilization"""
        if self.total_budget == 0:
            return 0.0
        return float((self.total_spent / self.total_budget) * 100)
    
    def can_spend(self, amount: Decimal, category: SpendingCategory) -> bool:
        """Check if amount can be spent in category"""
        # Check total budget
        if self.available_budget < amount:
            return False
        
        # Check category allocation if exists
        if category in self.category_allocations:
            return self.category_allocations[category].can_spend(amount)
        
        return True
    
    def needs_authorization(self, amount: Decimal) -> bool:
        """Check if spending amount needs user authorization"""
        projected_total = self.total_spent + self.total_reserved + amount
        
        # Check if would exceed budget
        if projected_total > self.total_budget:
            return True
        
        # Check if would trigger high utilization threshold
        utilization_after = float((projected_total / self.total_budget) * 100)
        return utilization_after > 85.0  # Require auth at 85% budget usage


class BudgetExpandRequest(BaseModel):
    """Request for budget expansion"""
    request_id: UUID = Field(default_factory=uuid4)
    budget_id: UUID
    session_id: UUID
    user_id: str
    
    # Expansion details
    requested_amount: Decimal = Field(gt=0)
    expansion_reason: str
    cost_breakdown: Dict[SpendingCategory, Decimal] = Field(default_factory=dict)
    
    # Context
    current_utilization: float
    remaining_budget: Decimal
    estimated_completion_cost: Decimal
    
    # Request metadata
    auto_generated: bool = Field(default=False)  # True if auto-generated by system
    priority_level: str = Field(default="normal")  # low, normal, high, critical
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=30))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Response tracking
    approved: Optional[bool] = None
    approved_amount: Optional[Decimal] = None
    response_at: Optional[datetime] = None
    response_reason: Optional[str] = None


class FTNSBudgetManager:
    """
    Comprehensive FTNS Budget Management System
    
    Provides sophisticated budget management for PRSM sessions including:
    - Per-prompt budget allocation with predictive estimation
    - Real-time spending tracking across all resource categories
    - Dynamic budget expansion with user authorization workflows
    - Marketplace integration for U2U and A2A transaction budgets
    - Emergency controls and circuit breakers for runaway costs
    """
    
    def __init__(
        self,
        ftns_service: Optional[FTNSService] = None
    ):
        # Core services
        self.ftns_service = ftns_service or FTNSService()
        
        # Budget storage
        self.active_budgets: Dict[UUID, FTNSBudget] = {}
        self.budget_history: Dict[UUID, FTNSBudget] = {}
        self.pending_expansions: Dict[UUID, BudgetExpandRequest] = {}
        
        # Prediction and estimation
        self.cost_estimator = BudgetCostEstimator()
        self.prediction_cache: Dict[str, BudgetPrediction] = {}
        
        # Alert and notification handlers
        self.alert_handlers: Dict[str, Callable] = {}
        self.authorization_handlers: Dict[str, Callable] = {}
        
        # Configuration
        self.default_budget_amount = Decimal(str(settings.ftns_initial_grant or '100.0'))
        self.max_session_budget = Decimal(str(settings.ftns_max_session_budget or '10000.0'))
        self.emergency_stop_threshold = 0.95  # Stop at 95% budget consumption
        
        # Performance tracking
        self.budget_metrics = {
            "budgets_created": 0,
            "budgets_exceeded": 0,
            "auto_expansions": 0,
            "manual_expansions": 0,
            "emergency_stops": 0,
            "total_managed_spending": Decimal('0')
        }
        
        logger.info("FTNS Budget Manager initialized")
    
    async def create_session_budget(
        self,
        session: PRSMSession,
        user_input: UserInput,
        budget_config: Optional[Dict[str, Any]] = None
    ) -> FTNSBudget:
        """
        Create a comprehensive budget for a PRSM session
        
        Args:
            session: PRSM session to budget
            user_input: User input with potential budget preferences
            budget_config: Optional budget configuration override
            
        Returns:
            Created FTNS budget with predictive allocations
        """
        try:
            logger.info("Creating session budget",
                       session_id=session.session_id,
                       user_id=session.user_id)
            
            # Step 1: Generate cost prediction
            prediction = await self.predict_session_cost(user_input, session)
            
            # Step 2: Determine budget amount
            if budget_config and "total_budget" in budget_config:
                total_budget = Decimal(str(budget_config["total_budget"]))
            elif hasattr(user_input, 'budget_limit') and user_input.budget_limit:
                total_budget = Decimal(str(user_input.budget_limit))
            else:
                # Use prediction with safety margin
                total_budget = prediction.get_recommended_budget()
                
                # Ensure minimum budget
                minimum_budget = Decimal('100')  # Minimum 100 FTNS tokens
                total_budget = max(total_budget, minimum_budget)
                
                # Ensure within user balance and limits
                user_balance = await self.ftns_service.get_user_balance(session.user_id)
                max_affordable = min(Decimal(str(user_balance.balance)), Decimal(str(self.max_session_budget)))
                total_budget = min(total_budget, max_affordable)
            
            # Step 3: Create category allocations based on prediction
            category_allocations = self._create_category_allocations(
                prediction, total_budget, budget_config
            )
            
            # Step 4: Set up default alerts
            default_alerts = self._create_default_alerts(budget_config)
            
            # Step 5: Create budget object
            budget = FTNSBudget(
                session_id=session.session_id,
                user_id=session.user_id,
                total_budget=total_budget,
                category_allocations=category_allocations,
                budget_alerts=default_alerts,
                initial_prediction=prediction,
                auto_expand_enabled=budget_config.get("auto_expand", True) if budget_config else True,
                max_auto_expand=Decimal(str(budget_config.get("max_auto_expand", 500))) if budget_config else Decimal('500'),
                expansion_increment=Decimal(str(budget_config.get("expansion_increment", 50))) if budget_config else Decimal('50')
            )
            
            # Step 6: Store and track budget
            self.active_budgets[budget.budget_id] = budget
            self.budget_metrics["budgets_created"] += 1
            
            # Step 7: Reserve initial allocation
            await self.reserve_budget_amount(
                budget.budget_id,
                Decimal('10'),  # Reserve small amount for session startup
                SpendingCategory.AGENT_COORDINATION,
                "Session initialization"
            )
            
            logger.info("Session budget created",
                       session_id=session.session_id,
                       budget_id=budget.budget_id,
                       total_budget=float(total_budget),
                       predicted_cost=float(prediction.estimated_total_cost),
                       category_count=len(category_allocations))
            
            return budget
            
        except Exception as e:
            logger.error("Failed to create session budget",
                        session_id=session.session_id,
                        error=str(e))
            raise
    
    async def predict_session_cost(
        self,
        user_input: UserInput,
        session: PRSMSession
    ) -> BudgetPrediction:
        """
        Predict the cost of a session based on input analysis
        
        This uses ML-like analysis to estimate costs across all categories
        based on query complexity, historical patterns, and resource requirements.
        """
        try:
            # Generate cache key for prediction
            cache_key = f"{hash(user_input.prompt)}_{session.user_id}_{len(user_input.prompt)}"
            
            if cache_key in self.prediction_cache:
                logger.info("Using cached prediction", cache_key=cache_key)
                return self.prediction_cache[cache_key]
            
            # Analyze query complexity
            complexity_analysis = await self.cost_estimator.analyze_query_complexity(user_input.prompt)
            
            # Get user's historical spending patterns
            user_history = await self._get_user_spending_history(session.user_id)
            
            # Estimate costs by category
            category_estimates = {}
            
            # Model inference costs (usually largest component)
            model_cost = await self.cost_estimator.estimate_model_costs(
                complexity_analysis, user_history
            )
            category_estimates[SpendingCategory.MODEL_INFERENCE] = model_cost
            
            # Agent coordination costs
            agent_cost = await self.cost_estimator.estimate_agent_costs(
                complexity_analysis, user_history
            )
            category_estimates[SpendingCategory.AGENT_COORDINATION] = agent_cost
            
            # Tool execution costs (significant for technical queries)
            tool_cost = await self.cost_estimator.estimate_tool_costs(
                complexity_analysis, user_input.prompt
            )
            category_estimates[SpendingCategory.TOOL_EXECUTION] = tool_cost
            
            # Data access costs
            data_cost = await self.cost_estimator.estimate_data_costs(
                complexity_analysis, user_input.prompt
            )
            category_estimates[SpendingCategory.DATA_ACCESS] = data_cost
            
            # Context processing costs
            context_cost = await self.cost_estimator.estimate_context_costs(
                complexity_analysis, len(user_input.prompt)
            )
            category_estimates[SpendingCategory.CONTEXT_PROCESSING] = context_cost
            
            # Calculate smaller categories
            for category in [SpendingCategory.MARKETPLACE_TRADING, SpendingCategory.SAFETY_VALIDATION,
                           SpendingCategory.STORAGE_OPERATIONS, SpendingCategory.NETWORK_OPERATIONS]:
                category_estimates[category] = Decimal('2.0')  # Small fixed amounts
            
            # Calculate total and confidence
            total_estimated_cost = sum(category_estimates.values())
            confidence = self.cost_estimator.calculate_confidence(complexity_analysis, user_history)
            
            # Create prediction
            prediction = BudgetPrediction(
                query_complexity=complexity_analysis["complexity_score"],
                estimated_total_cost=total_estimated_cost,
                category_estimates=category_estimates,
                confidence_score=confidence,
                prediction_factors={
                    "complexity_analysis": complexity_analysis,
                    "user_avg_session_cost": user_history.get("avg_session_cost", 50.0),
                    "similar_query_costs": user_history.get("similar_costs", []),
                    "prompt_length": len(user_input.prompt)
                }
            )
            
            # Cache prediction
            self.prediction_cache[cache_key] = prediction
            
            logger.info("Session cost predicted",
                       total_cost=float(total_estimated_cost),
                       confidence=confidence,
                       complexity=complexity_analysis["complexity_score"])
            
            return prediction
            
        except Exception as e:
            logger.error("Cost prediction failed", error=str(e))
            # Return conservative fallback prediction
            return BudgetPrediction(
                query_complexity=0.5,
                estimated_total_cost=self.default_budget_amount,
                category_estimates={
                    SpendingCategory.MODEL_INFERENCE: self.default_budget_amount * Decimal('0.6'),
                    SpendingCategory.AGENT_COORDINATION: self.default_budget_amount * Decimal('0.2'),
                    SpendingCategory.TOOL_EXECUTION: self.default_budget_amount * Decimal('0.1'),
                    SpendingCategory.CONTEXT_PROCESSING: self.default_budget_amount * Decimal('0.1')
                },
                confidence_score=0.3
            )
    
    async def reserve_budget_amount(
        self,
        budget_id: UUID,
        amount: Decimal,
        category: SpendingCategory,
        description: str = ""
    ) -> bool:
        """
        Reserve budget amount for upcoming spending
        
        This prevents other operations from using budget that's already
        committed to pending operations.
        """
        try:
            if budget_id not in self.active_budgets:
                logger.warning("Budget not found for reservation", budget_id=budget_id)
                return False
            
            budget = self.active_budgets[budget_id]
            
            # Check if reservation is possible
            if not budget.can_spend(amount, category):
                logger.warning("Insufficient budget for reservation",
                             budget_id=budget_id,
                             amount=float(amount),
                             available=float(budget.available_budget))
                return False
            
            # Make reservation
            budget.total_reserved += amount
            
            # Update category allocation if exists
            if category in budget.category_allocations:
                budget.category_allocations[category].reserved_amount += amount
            
            # Track reservation
            budget.spending_history.append({
                "action": "reserve",
                "amount": float(amount),
                "category": category.value,
                "description": description,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "remaining_budget": float(budget.available_budget)
            })
            
            budget.updated_at = datetime.now(timezone.utc)
            
            logger.info("Budget amount reserved",
                       budget_id=budget_id,
                       amount=float(amount),
                       category=category.value,
                       remaining=float(budget.available_budget))
            
            return True
            
        except Exception as e:
            logger.error("Budget reservation failed",
                        budget_id=budget_id,
                        error=str(e))
            return False
    
    async def spend_budget_amount(
        self,
        budget_id: UUID,
        amount: Decimal,
        category: SpendingCategory,
        description: str = "",
        release_reserved: bool = True
    ) -> bool:
        """
        Spend budget amount and track the expense
        
        Args:
            budget_id: Budget to spend from
            amount: Amount to spend
            category: Spending category
            description: Spending description
            release_reserved: Whether to release reserved amount
            
        Returns:
            True if spending succeeded
        """
        try:
            if budget_id not in self.active_budgets:
                logger.warning("Budget not found for spending", budget_id=budget_id)
                return False
            
            budget = self.active_budgets[budget_id]
            
            # Check budget status
            if budget.status != BudgetStatus.ACTIVE:
                logger.warning("Cannot spend from inactive budget",
                             budget_id=budget_id,
                             status=budget.status.value)
                return False
            
            # Handle budget expansion if needed
            if not budget.can_spend(amount, category):
                expansion_approved = await self._handle_budget_expansion(
                    budget, amount, category, description
                )
                if not expansion_approved:
                    logger.warning("Budget expansion denied",
                                 budget_id=budget_id,
                                 amount=float(amount))
                    return False
            
            # Process spending
            budget.total_spent += amount
            
            # Update category allocation
            if category in budget.category_allocations:
                budget.category_allocations[category].spent_amount += amount
                if release_reserved:
                    # Release reserved amount up to spending amount
                    release_amount = min(amount, budget.category_allocations[category].reserved_amount)
                    budget.category_allocations[category].reserved_amount -= release_amount
                    budget.total_reserved -= release_amount
            
            # Track spending
            budget.spending_history.append({
                "action": "spend",
                "amount": float(amount),
                "category": category.value,
                "description": description,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "remaining_budget": float(budget.available_budget),
                "utilization": budget.utilization_percentage
            })
            
            budget.updated_at = datetime.now(timezone.utc)
            self.budget_metrics["total_managed_spending"] += amount
            
            # Check for alerts
            await self._check_budget_alerts(budget)
            
            # Check for emergency stop
            if budget.utilization_percentage >= (self.emergency_stop_threshold * 100):
                await self._trigger_emergency_stop(budget)
            
            logger.info("Budget amount spent",
                       budget_id=budget_id,
                       amount=float(amount),
                       category=category.value,
                       utilization=budget.utilization_percentage,
                       remaining=float(budget.available_budget))
            
            return True
            
        except Exception as e:
            logger.error("Budget spending failed",
                        budget_id=budget_id,
                        error=str(e))
            return False
    
    async def request_budget_expansion(
        self,
        budget_id: UUID,
        requested_amount: Decimal,
        reason: str,
        cost_breakdown: Optional[Dict[SpendingCategory, Decimal]] = None,
        auto_approve_threshold: Optional[Decimal] = None
    ) -> BudgetExpandRequest:
        """
        Request expansion of budget with user authorization workflow
        
        This creates a formal request that can be auto-approved for small amounts
        or require explicit user authorization for larger expansions.
        """
        try:
            if budget_id not in self.active_budgets:
                raise ValueError(f"Budget {budget_id} not found")
            
            budget = self.active_budgets[budget_id]
            
            # Create expansion request
            expand_request = BudgetExpandRequest(
                budget_id=budget_id,
                session_id=budget.session_id,
                user_id=budget.user_id,
                requested_amount=requested_amount,
                expansion_reason=reason,
                cost_breakdown=cost_breakdown or {},
                current_utilization=budget.utilization_percentage,
                remaining_budget=budget.available_budget,
                estimated_completion_cost=requested_amount
            )
            
            # Check for auto-approval
            auto_threshold = auto_approve_threshold or budget.expansion_increment
            if (budget.auto_expand_enabled and 
                requested_amount <= auto_threshold and 
                budget.max_auto_expand == 0 or 
                (budget.total_budget + requested_amount) <= (budget.category_allocations.get(SpendingCategory.MODEL_INFERENCE, BudgetAllocation(category=SpendingCategory.MODEL_INFERENCE, allocated_amount=Decimal('0'))).allocated_amount + budget.max_auto_expand)):
                
                # Auto-approve small expansions
                expand_request.approved = True
                expand_request.approved_amount = requested_amount
                expand_request.response_at = datetime.now(timezone.utc)
                expand_request.response_reason = "Auto-approved within threshold"
                expand_request.auto_generated = True
                
                # Apply expansion immediately
                await self._apply_budget_expansion(budget, expand_request)
                
                self.budget_metrics["auto_expansions"] += 1
                
                logger.info("Budget expansion auto-approved",
                           budget_id=budget_id,
                           amount=float(requested_amount),
                           new_total=float(budget.total_budget))
            else:
                # Require manual approval
                expand_request.approved = None  # Pending
                self.pending_expansions[expand_request.request_id] = expand_request
                
                # Trigger authorization handler if configured
                if "budget_expansion" in self.authorization_handlers:
                    await self.authorization_handlers["budget_expansion"](expand_request)
                
                logger.info("Budget expansion requires approval",
                           budget_id=budget_id,
                           request_id=expand_request.request_id,
                           amount=float(requested_amount))
            
            return expand_request
            
        except Exception as e:
            logger.error("Budget expansion request failed",
                        budget_id=budget_id,
                        error=str(e))
            raise
    
    async def approve_budget_expansion(
        self,
        request_id: UUID,
        approved: bool,
        approved_amount: Optional[Decimal] = None,
        reason: str = ""
    ) -> bool:
        """
        Approve or deny a budget expansion request
        
        Args:
            request_id: Expansion request ID
            approved: Whether to approve the request
            approved_amount: Amount approved (can be different from requested)
            reason: Approval/denial reason
            
        Returns:
            True if processing succeeded
        """
        try:
            if request_id not in self.pending_expansions:
                logger.warning("Expansion request not found", request_id=request_id)
                return False
            
            expand_request = self.pending_expansions[request_id]
            budget = self.active_budgets[expand_request.budget_id]
            
            # Update request
            expand_request.approved = approved
            expand_request.approved_amount = approved_amount if approved else Decimal('0')
            expand_request.response_at = datetime.now(timezone.utc)
            expand_request.response_reason = reason
            
            if approved and approved_amount and approved_amount > 0:
                # Apply expansion
                await self._apply_budget_expansion(budget, expand_request)
                self.budget_metrics["manual_expansions"] += 1
                
                logger.info("Budget expansion approved and applied",
                           request_id=request_id,
                           budget_id=expand_request.budget_id,
                           amount=float(approved_amount))
            else:
                # Mark budget as exceeded if expansion denied
                if budget.available_budget <= 0:
                    budget.status = BudgetStatus.EXCEEDED
                
                logger.info("Budget expansion denied",
                           request_id=request_id,
                           budget_id=expand_request.budget_id,
                           reason=reason)
            
            # Remove from pending
            del self.pending_expansions[request_id]
            
            return True
            
        except Exception as e:
            logger.error("Budget expansion approval failed",
                        request_id=request_id,
                        error=str(e))
            return False
    
    async def get_budget_status(self, budget_id: UUID) -> Optional[Dict[str, Any]]:
        """Get comprehensive budget status and analytics"""
        try:
            if budget_id not in self.active_budgets:
                # Check budget history
                if budget_id in self.budget_history:
                    budget = self.budget_history[budget_id]
                else:
                    return None
            else:
                budget = self.active_budgets[budget_id]
            
            # Calculate analytics
            category_breakdown = {}
            for category, allocation in budget.category_allocations.items():
                category_breakdown[category.value] = {
                    "allocated": float(allocation.allocated_amount),
                    "spent": float(allocation.spent_amount),
                    "reserved": float(allocation.reserved_amount),
                    "available": float(allocation.available_amount),
                    "utilization": allocation.utilization_percentage
                }
            
            # Calculate spending velocity
            recent_spending = [
                entry for entry in budget.spending_history[-10:]
                if entry["action"] == "spend"
            ]
            spending_velocity = sum(entry["amount"] for entry in recent_spending) / max(len(recent_spending), 1)
            
            # Estimate time to budget depletion
            if spending_velocity > 0 and budget.available_budget > 0:
                estimated_depletion_hours = float(budget.available_budget) / spending_velocity
            else:
                estimated_depletion_hours = None
            
            return {
                "budget_id": str(budget_id),
                "session_id": str(budget.session_id),
                "status": budget.status.value,
                "total_budget": float(budget.total_budget),
                "total_spent": float(budget.total_spent),
                "total_reserved": float(budget.total_reserved),
                "available_budget": float(budget.available_budget),
                "utilization_percentage": budget.utilization_percentage,
                "category_breakdown": category_breakdown,
                "spending_velocity": spending_velocity,
                "estimated_depletion_hours": estimated_depletion_hours,
                "triggered_alerts": budget.triggered_alerts,
                "pending_expansions": len([r for r in self.pending_expansions.values() 
                                          if r.budget_id == budget_id]),
                "spending_history_count": len(budget.spending_history),
                "created_at": budget.created_at.isoformat(),
                "updated_at": budget.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error("Get budget status failed", budget_id=budget_id, error=str(e))
            return None
    
    def _create_category_allocations(
        self,
        prediction: BudgetPrediction,
        total_budget: Decimal,
        config: Optional[Dict[str, Any]]
    ) -> Dict[SpendingCategory, BudgetAllocation]:
        """Create category allocations based on prediction and config"""
        allocations = {}
        
        # Use prediction as base, then adjust for total budget
        prediction_total = Decimal(str(prediction.estimated_total_cost))
        scale_factor = total_budget / prediction_total if prediction_total > 0 else Decimal('1')
        
        for category, estimated_cost in prediction.category_estimates.items():
            # Scale estimate to fit total budget
            allocated_amount = Decimal(str(estimated_cost)) * scale_factor
            
            # Apply config overrides if provided
            if config and "category_allocations" in config:
                category_config = config["category_allocations"].get(category.value)
                if category_config:
                    if "amount" in category_config:
                        allocated_amount = Decimal(str(category_config["amount"]))
                    elif "percentage" in category_config:
                        allocated_amount = total_budget * Decimal(str(category_config["percentage"] / 100))
            
            allocations[category] = BudgetAllocation(
                category=category,
                allocated_amount=allocated_amount
            )
        
        return allocations
    
    def _create_default_alerts(self, config: Optional[Dict[str, Any]]) -> List[BudgetAlert]:
        """Create default budget alerts"""
        alerts = []
        
        # Default alert thresholds
        default_thresholds = [
            (50.0, "warning", ["in_app"]),
            (75.0, "warning", ["in_app", "email"]),
            (90.0, "critical", ["in_app", "email"]),
            (95.0, "emergency", ["in_app", "email", "webhook"])
        ]
        
        # Override with config if provided
        if config and "alerts" in config:
            alert_configs = config["alerts"]
        else:
            alert_configs = default_thresholds
        
        for threshold, alert_type, channels in alert_configs:
            if isinstance(threshold, (int, float)):
                alerts.append(BudgetAlert(
                    threshold_percentage=float(threshold),
                    alert_type=alert_type,
                    notification_channels=channels,
                    message_template=f"Budget {threshold}% consumed - consider reviewing spending"
                ))
        
        return alerts
    
    async def _handle_budget_expansion(
        self,
        budget: FTNSBudget,
        needed_amount: Decimal,
        category: SpendingCategory,
        description: str
    ) -> bool:
        """Handle automatic budget expansion logic"""
        try:
            # Calculate expansion amount (round up to increment)
            expansion_needed = needed_amount - budget.available_budget
            expansion_increment = budget.expansion_increment
            expansion_amount = math.ceil(float(expansion_needed / expansion_increment)) * expansion_increment
            expansion_amount = Decimal(str(expansion_amount))
            
            # Create expansion request
            expand_request = await self.request_budget_expansion(
                budget.budget_id,
                expansion_amount,
                f"Auto-expansion for {category.value}: {description}",
                {category: expansion_needed}
            )
            
            # Return approval status
            return expand_request.approved is True
            
        except Exception as e:
            logger.error("Budget expansion handling failed",
                        budget_id=budget.budget_id,
                        error=str(e))
            return False
    
    async def _apply_budget_expansion(
        self,
        budget: FTNSBudget,
        expand_request: BudgetExpandRequest
    ):
        """Apply approved budget expansion"""
        expansion_amount = expand_request.approved_amount
        
        # Increase total budget
        budget.total_budget += expansion_amount
        
        # Distribute expansion across categories based on cost breakdown
        if expand_request.cost_breakdown:
            for category, amount in expand_request.cost_breakdown.items():
                if category in budget.category_allocations:
                    budget.category_allocations[category].allocated_amount += amount
        else:
            # Default: add to model inference category
            if SpendingCategory.MODEL_INFERENCE in budget.category_allocations:
                budget.category_allocations[SpendingCategory.MODEL_INFERENCE].allocated_amount += expansion_amount
        
        # Track expansion
        budget.spending_history.append({
            "action": "expand",
            "amount": float(expansion_amount),
            "category": "budget_expansion",
            "description": f"Budget expanded: {expand_request.expansion_reason}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": str(expand_request.request_id)
        })
        
        budget.updated_at = datetime.now(timezone.utc)
    
    async def _check_budget_alerts(self, budget: FTNSBudget):
        """Check and trigger budget alerts"""
        current_utilization = budget.utilization_percentage
        
        for alert in budget.budget_alerts:
            alert_key = f"{alert.alert_id}_{alert.threshold_percentage}"
            
            # Check if alert should trigger
            if (current_utilization >= alert.threshold_percentage and 
                alert_key not in budget.triggered_alerts):
                
                # Trigger alert
                budget.triggered_alerts.append(alert_key)
                
                # Call alert handler if configured
                if alert.alert_type in self.alert_handlers:
                    await self.alert_handlers[alert.alert_type](budget, alert, current_utilization)
                
                logger.warning("Budget alert triggered",
                             budget_id=budget.budget_id,
                             alert_type=alert.alert_type,
                             threshold=alert.threshold_percentage,
                             current_utilization=current_utilization)
    
    async def _trigger_emergency_stop(self, budget: FTNSBudget):
        """Trigger emergency stop for runaway spending"""
        budget.status = BudgetStatus.SUSPENDED
        self.budget_metrics["emergency_stops"] += 1
        
        # Track emergency stop
        budget.spending_history.append({
            "action": "emergency_stop",
            "amount": 0.0,
            "category": "system",
            "description": f"Emergency stop triggered at {budget.utilization_percentage:.1f}% utilization",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        logger.critical("Emergency budget stop triggered",
                       budget_id=budget.budget_id,
                       utilization=budget.utilization_percentage,
                       total_spent=float(budget.total_spent))
        
        # Call emergency handler if configured
        if "emergency_stop" in self.alert_handlers:
            await self.alert_handlers["emergency_stop"](budget, None, budget.utilization_percentage)
    
    async def _get_user_spending_history(self, user_id: str) -> Dict[str, Any]:
        """Get user's historical spending patterns for prediction"""
        try:
            # This would integrate with database service in production
            # For now, return mock data
            return {
                "avg_session_cost": 45.0,
                "session_count": 15,
                "similar_costs": [30.0, 60.0, 40.0, 55.0],
                "preferred_categories": [SpendingCategory.MODEL_INFERENCE, SpendingCategory.TOOL_EXECUTION]
            }
        except Exception as e:
            logger.error("Failed to get user spending history", user_id=user_id, error=str(e))
            return {"avg_session_cost": 50.0, "session_count": 0, "similar_costs": []}


class BudgetCostEstimator:
    """
    Sophisticated cost estimation for budget prediction
    
    Uses analysis of query complexity, historical patterns,
    and resource requirements to predict FTNS costs.
    """
    
    async def analyze_query_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze query complexity for cost estimation"""
        # Simple heuristic-based analysis
        # In production, this would use ML models
        
        complexity_indicators = {
            "length": len(prompt),
            "technical_terms": len([word for word in prompt.lower().split() 
                                  if word in ["analysis", "research", "calculate", "analyze", "optimize"]]),
            "question_marks": prompt.count("?"),
            "complexity_keywords": len([word for word in prompt.lower().split()
                                      if word in ["complex", "detailed", "comprehensive", "thorough"]])
        }
        
        # Calculate complexity score
        length_score = min(complexity_indicators["length"] / 1000, 1.0)
        technical_score = min(complexity_indicators["technical_terms"] / 5, 1.0)
        question_score = min(complexity_indicators["question_marks"] / 3, 1.0)
        keyword_score = min(complexity_indicators["complexity_keywords"] / 3, 1.0)
        
        complexity_score = (length_score + technical_score + question_score + keyword_score) / 4
        
        return {
            "complexity_score": complexity_score,
            "indicators": complexity_indicators,
            "estimated_tokens": len(prompt.split()) * 1.3,  # Rough token estimate
            "requires_tools": any(keyword in prompt.lower() 
                                for keyword in ["calculate", "compute", "search", "analyze", "code"]),
            "requires_research": any(keyword in prompt.lower()
                                   for keyword in ["research", "study", "investigate", "explore"])
        }
    
    async def estimate_model_costs(self, complexity_analysis: Dict[str, Any], 
                                 user_history: Dict[str, Any]) -> Decimal:
        """Estimate model inference costs"""
        base_cost = Decimal('20.0')  # Base model cost
        
        # Scale by complexity
        complexity_multiplier = 1 + complexity_analysis["complexity_score"]
        
        # Scale by estimated tokens
        token_multiplier = 1 + (complexity_analysis["estimated_tokens"] / 1000) * 0.5
        
        # Historical adjustment
        if user_history.get("avg_session_cost", 0) > 30:
            history_multiplier = 1.2
        else:
            history_multiplier = 0.8
        
        return base_cost * Decimal(str(complexity_multiplier * token_multiplier * history_multiplier))
    
    async def estimate_agent_costs(self, complexity_analysis: Dict[str, Any],
                                 user_history: Dict[str, Any]) -> Decimal:
        """Estimate agent coordination costs"""
        base_cost = Decimal('10.0')
        
        # More agents needed for complex queries
        if complexity_analysis["complexity_score"] > 0.7:
            return base_cost * Decimal('2.0')
        elif complexity_analysis["complexity_score"] > 0.4:
            return base_cost * Decimal('1.5')
        else:
            return base_cost
    
    async def estimate_tool_costs(self, complexity_analysis: Dict[str, Any], 
                                prompt: str) -> Decimal:
        """Estimate tool execution costs"""
        if not complexity_analysis.get("requires_tools", False):
            return Decimal('0.0')
        
        # Count potential tool usages
        tool_indicators = [
            "calculate", "compute", "search", "analyze", "code", "run", "execute"
        ]
        tool_count = sum(1 for indicator in tool_indicators if indicator in prompt.lower())
        
        # Base cost per tool usage
        cost_per_tool = Decimal('5.0')
        estimated_tools = max(tool_count, 1)
        
        return cost_per_tool * Decimal(str(estimated_tools))
    
    async def estimate_data_costs(self, complexity_analysis: Dict[str, Any],
                                prompt: str) -> Decimal:
        """Estimate data access costs"""
        if not complexity_analysis.get("requires_research", False):
            return Decimal('2.0')  # Minimal data access
        
        # Scale by research complexity
        base_cost = Decimal('5.0')
        research_multiplier = 1 + complexity_analysis["complexity_score"]
        
        return base_cost * Decimal(str(research_multiplier))
    
    async def estimate_context_costs(self, complexity_analysis: Dict[str, Any],
                                   prompt_length: int) -> Decimal:
        """Estimate context processing costs"""
        base_cost = Decimal('3.0')
        
        # Scale by prompt length and complexity
        length_factor = min(prompt_length / 500, 2.0)  # Cap at 2x for very long prompts
        complexity_factor = 1 + complexity_analysis["complexity_score"]
        
        return base_cost * Decimal(str(length_factor * complexity_factor))
    
    def calculate_confidence(self, complexity_analysis: Dict[str, Any],
                           user_history: Dict[str, Any]) -> float:
        """Calculate confidence in cost prediction"""
        base_confidence = 0.6
        
        # Higher confidence for simpler queries
        complexity_confidence = 1 - (complexity_analysis["complexity_score"] * 0.3)
        
        # Higher confidence with more user history
        history_confidence = min(user_history.get("session_count", 0) / 10, 0.3)
        
        total_confidence = min(base_confidence + complexity_confidence + history_confidence, 0.95)
        return max(total_confidence, 0.3)


# Global budget manager instance
_ftns_budget_manager = None

def get_ftns_budget_manager() -> FTNSBudgetManager:
    """Get global FTNS budget manager instance"""
    global _ftns_budget_manager
    if _ftns_budget_manager is None:
        _ftns_budget_manager = FTNSBudgetManager()
    return _ftns_budget_manager