"""
NWTN-FTNS Integration Module
Integrates enhanced FTNS accounting with NWTN orchestrator for Phase 1

This module provides seamless integration between the NWTN Orchestrator
and the Enhanced FTNS Service to ensure:

1. Real-time cost calculation during agent execution
2. Microsecond-precision usage tracking
3. Automatic balance validation and charging
4. Complete audit trails for governance
5. Performance optimization through cost prediction
6. Circuit breaker integration for economic safety

Integration Points:
- Session creation with FTNS balance validation
- Agent execution with real-time cost tracking
- Response compilation with final cost calculation
- Error handling with cost reconciliation
- Performance monitoring with economic analytics
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID
import structlog

from prsm.core.models import (
    UserInput, PRSMSession, PRSMResponse, AgentType,
    ReasoningStep, TaskStatus
)
from prsm.tokenomics.enhanced_ftns_service import (
    EnhancedFTNSService, MicrosecondCostCalculation, 
    UsageTrackingEntry, get_enhanced_ftns_service
)
from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

class FTNSIntegratedSession:
    """
    FTNS-integrated session manager for NWTN orchestrator
    
    Manages complete session lifecycle with:
    - Balance validation and reservation
    - Real-time cost tracking during execution
    - Usage analytics and optimization
    - Error handling with cost reconciliation
    """
    
    def __init__(self, session: PRSMSession):
        self.session = session
        self.ftns_service = get_enhanced_ftns_service()
        
        # Cost tracking
        self.initial_balance: Optional[Decimal] = None
        self.reserved_amount: Decimal = Decimal('0')
        self.consumed_amount: Decimal = Decimal('0')
        self.usage_entries: List[UsageTrackingEntry] = []
        
        # Performance tracking
        self.session_start_time = datetime.now(timezone.utc)
        self.agent_execution_times: Dict[str, Tuple[datetime, datetime]] = {}
        
        # Cost predictions
        self.predicted_cost: Optional[Decimal] = None
        self.cost_calculation: Optional[MicrosecondCostCalculation] = None
    
    async def initialize_with_balance_validation(self) -> bool:
        """
        Initialize session with FTNS balance validation
        
        Returns:
            True if user has sufficient balance, False otherwise
        """
        try:
            # Get current balance
            self.initial_balance = await self.ftns_service.get_user_balance_precise(
                self.session.user_id
            )
            
            # Predict session cost based on context allocation
            self.predicted_cost = await self._predict_session_cost()
            
            # Validate sufficient balance
            if self.initial_balance < self.predicted_cost:
                logger.warning("Insufficient FTNS balance for session",
                              session_id=str(self.session.session_id),
                              balance=float(self.initial_balance),
                              predicted_cost=float(self.predicted_cost))
                return False
            
            # Reserve predicted amount
            self.reserved_amount = self.predicted_cost
            
            logger.info("FTNS session initialized",
                       session_id=str(self.session.session_id),
                       initial_balance=float(self.initial_balance),
                       predicted_cost=float(self.predicted_cost),
                       reserved_amount=float(self.reserved_amount))
            
            return True
            
        except Exception as e:
            logger.error("FTNS session initialization failed",
                        session_id=str(self.session.session_id),
                        error=str(e))
            return False
    
    async def track_agent_execution(
        self,
        agent_type: AgentType,
        operation_type: str,
        context_units: int,
        start_time: datetime,
        end_time: datetime,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageTrackingEntry:
        """
        Track individual agent execution with precise timing
        
        Args:
            agent_type: Agent that executed
            operation_type: Type of operation performed
            context_units: Context consumed
            start_time: Execution start timestamp
            end_time: Execution end timestamp
            success: Whether execution succeeded
            metadata: Additional tracking data
            
        Returns:
            Usage tracking entry with cost calculation
        """
        try:
            # Calculate agent-specific cost
            agent_cost_calc = await self.ftns_service.calculate_microsecond_precision_cost(
                session=self.session,
                agents_used=[agent_type],
                context_units=context_units,
                execution_time_microseconds=int((end_time - start_time).total_seconds() * 1_000_000),
                metadata=metadata
            )
            
            # Track usage
            usage_entry = await self.ftns_service.track_usage_with_precision(
                session=self.session,
                agent_type=agent_type,
                operation_type=operation_type,
                context_units=context_units,
                start_timestamp=start_time,
                end_timestamp=end_time,
                cost_calculation=agent_cost_calc,
                success=success,
                metadata=metadata
            )
            
            # Add to session tracking
            self.usage_entries.append(usage_entry)
            self.consumed_amount += usage_entry.cost_ftns
            self.agent_execution_times[f"{agent_type.value}_{len(self.usage_entries)}"] = (start_time, end_time)
            
            logger.debug("Agent execution tracked",
                        session_id=str(self.session.session_id),
                        agent_type=agent_type.value,
                        cost_ftns=float(usage_entry.cost_ftns),
                        total_consumed=float(self.consumed_amount))
            
            return usage_entry
            
        except Exception as e:
            logger.error("Agent execution tracking failed",
                        session_id=str(self.session.session_id),
                        agent_type=agent_type.value,
                        error=str(e))
            raise
    
    async def finalize_session_with_precise_charging(
        self,
        response: PRSMResponse,
        agents_used: List[AgentType]
    ) -> Dict[str, Any]:
        """
        Finalize session with precise FTNS charging
        
        Args:
            response: Final PRSM response
            agents_used: All agents used in session
            
        Returns:
            Transaction summary with cost breakdown
        """
        try:
            session_end_time = datetime.now(timezone.utc)
            total_execution_time = int((session_end_time - self.session_start_time).total_seconds() * 1_000_000)
            
            # Calculate final session cost
            final_cost_calc = await self.ftns_service.calculate_microsecond_precision_cost(
                session=self.session,
                agents_used=agents_used,
                context_units=response.context_used,
                execution_time_microseconds=total_execution_time,
                metadata={
                    'confidence_score': response.confidence_score,
                    'reasoning_steps': len(response.reasoning_trace),
                    'safety_validated': response.safety_validated,
                    'usage_entries_count': len(self.usage_entries)
                }
            )
            
            self.cost_calculation = final_cost_calc
            
            # Process transaction
            transaction = await self.ftns_service.process_session_transaction(
                session=self.session,
                response=response,
                cost_calculation=final_cost_calc
            )
            
            # Calculate cost accuracy
            individual_costs = sum(entry.cost_ftns for entry in self.usage_entries)
            cost_difference = abs(final_cost_calc.total_cost - individual_costs)
            cost_accuracy = 1.0 - min(float(cost_difference / final_cost_calc.total_cost), 1.0)
            
            # Update response with actual FTNS charge
            response.ftns_charged = float(final_cost_calc.total_cost)
            
            # Generate transaction summary
            transaction_summary = {
                'transaction_id': str(transaction.transaction_id),
                'session_id': str(self.session.session_id),
                'user_id': self.session.user_id,
                'cost_breakdown': {
                    'initial_balance': float(self.initial_balance),
                    'predicted_cost': float(self.predicted_cost),
                    'actual_cost': float(final_cost_calc.total_cost),
                    'individual_agent_costs': float(individual_costs),
                    'cost_accuracy': cost_accuracy,
                    'agent_cost_breakdown': {
                        agent_type: float(cost) 
                        for agent_type, cost in final_cost_calc.agent_cost_breakdown.items()
                    }
                },
                'timing_metrics': {
                    'session_duration_ms': total_execution_time / 1000,
                    'calculation_duration_us': final_cost_calc.calculation_duration_microseconds,
                    'agent_execution_times': {
                        key: {
                            'start': start.isoformat(),
                            'end': end.isoformat(),
                            'duration_us': int((end - start).total_seconds() * 1_000_000)
                        }
                        for key, (start, end) in self.agent_execution_times.items()
                    }
                },
                'usage_analytics': {
                    'total_usage_entries': len(self.usage_entries),
                    'context_units_used': response.context_used,
                    'agents_executed': len(agents_used),
                    'session_success': response.confidence_score >= 0.7
                },
                'economic_metrics': {
                    'cost_efficiency': float(final_cost_calc.total_cost / Decimal(str(response.context_used))) if response.context_used > 0 else 0,
                    'user_tier_savings': float(Decimal('1') - final_cost_calc.user_tier_multiplier),
                    'time_based_adjustment': float(final_cost_calc.time_based_multiplier - Decimal('1')),
                    'complexity_impact': float(final_cost_calc.complexity_multiplier - Decimal('1'))
                }
            }
            
            logger.info("Session finalized with precise FTNS charging",
                       session_id=str(self.session.session_id),
                       actual_cost=float(final_cost_calc.total_cost),
                       predicted_cost=float(self.predicted_cost),
                       cost_accuracy=cost_accuracy,
                       transaction_id=str(transaction.transaction_id))
            
            return transaction_summary
            
        except Exception as e:
            logger.error("Session finalization failed",
                        session_id=str(self.session.session_id),
                        error=str(e))
            # Attempt cost reconciliation
            await self._handle_finalization_error(e)
            raise
    
    async def handle_session_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle session errors with cost reconciliation
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error handling summary
        """
        try:
            # Calculate partial costs for work completed
            partial_cost = sum(entry.cost_ftns for entry in self.usage_entries)
            
            # Determine refund amount
            if self.reserved_amount > partial_cost:
                refund_amount = self.reserved_amount - partial_cost
            else:
                refund_amount = Decimal('0')
            
            # Update session status
            self.session.status = TaskStatus.FAILED
            
            error_summary = {
                'session_id': str(self.session.session_id),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'cost_reconciliation': {
                    'reserved_amount': float(self.reserved_amount),
                    'partial_cost': float(partial_cost),
                    'refund_amount': float(refund_amount),
                    'work_completed': len(self.usage_entries) > 0
                },
                'usage_summary': {
                    'completed_operations': len(self.usage_entries),
                    'agents_executed': len(set(entry.agent_type for entry in self.usage_entries)),
                    'total_context_used': sum(entry.context_units for entry in self.usage_entries)
                }
            }
            
            logger.warning("Session error handled with cost reconciliation",
                          session_id=str(self.session.session_id),
                          error_type=type(error).__name__,
                          refund_amount=float(refund_amount))
            
            return error_summary
            
        except Exception as e:
            logger.error("Error handling failed",
                        session_id=str(self.session.session_id),
                        original_error=str(error),
                        handling_error=str(e))
            return {
                'session_id': str(self.session.session_id),
                'error_type': 'error_handling_failure',
                'original_error': str(error),
                'handling_error': str(e)
            }
    
    # === Private Helper Methods ===
    
    async def _predict_session_cost(self) -> Decimal:
        """Predict session cost based on context allocation and user history"""
        try:
            # Base prediction on context allocation
            base_prediction = Decimal(str(self.session.nwtn_context_allocation)) * Decimal('0.01')
            
            # Adjust based on session complexity if available
            if hasattr(self.session, 'complexity_estimate'):
                complexity_factor = Decimal(str(self.session.complexity_estimate))
                base_prediction *= (Decimal('1') + complexity_factor * Decimal('0.5'))
            
            # Add buffer for safety (20%)
            predicted_cost = base_prediction * Decimal('1.2')
            
            # Ensure minimum cost
            min_cost = Decimal('0.01')
            predicted_cost = max(predicted_cost, min_cost)
            
            # Ensure maximum reasonable cost
            max_cost = Decimal('100.0')
            predicted_cost = min(predicted_cost, max_cost)
            
            return predicted_cost
            
        except Exception as e:
            logger.error("Cost prediction failed", 
                        session_id=str(self.session.session_id),
                        error=str(e))
            return Decimal('1.0')  # Fallback prediction
    
    async def _handle_finalization_error(self, error: Exception):
        """Handle errors during session finalization"""
        try:
            # Log detailed error information
            logger.error("Session finalization error details",
                        session_id=str(self.session.session_id),
                        error_type=type(error).__name__,
                        error_message=str(error),
                        usage_entries=len(self.usage_entries),
                        consumed_amount=float(self.consumed_amount))
            
            # Attempt to save usage entries for audit
            if self.usage_entries:
                await self.ftns_service._persist_usage_cache()
            
        except Exception as e:
            logger.error("Error handling failed",
                        session_id=str(self.session.session_id),
                        error=str(e))

class NWTNFTNSIntegrator:
    """
    Integration manager for NWTN-FTNS coordination
    
    Provides high-level integration functions for:
    - Session lifecycle management
    - Cost optimization
    - Performance monitoring
    - Error recovery
    """
    
    def __init__(self):
        self.ftns_service = get_enhanced_ftns_service()
        self.active_sessions: Dict[UUID, FTNSIntegratedSession] = {}
        
        # Performance metrics
        self.integration_metrics = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'sessions_failed': 0,
            'total_cost_processed': Decimal('0'),
            'total_context_processed': 0,
            'avg_cost_accuracy': 0.0
        }
    
    async def create_integrated_session(self, user_input: UserInput) -> Optional[FTNSIntegratedSession]:
        """
        Create FTNS-integrated session with balance validation
        
        Args:
            user_input: User input for session creation
            
        Returns:
            Integrated session if balance is sufficient, None otherwise
        """
        try:
            # Create base session
            session = PRSMSession(
                user_id=user_input.user_id,
                nwtn_context_allocation=user_input.context_allocation or settings.ftns_initial_grant
            )
            
            # Create integrated session
            integrated_session = FTNSIntegratedSession(session)
            
            # Initialize with balance validation
            if await integrated_session.initialize_with_balance_validation():
                self.active_sessions[session.session_id] = integrated_session
                self.integration_metrics['sessions_created'] += 1
                
                logger.info("Integrated session created",
                           session_id=str(session.session_id),
                           user_id=user_input.user_id)
                
                return integrated_session
            else:
                logger.warning("Session creation failed due to insufficient balance",
                              user_id=user_input.user_id)
                return None
                
        except Exception as e:
            logger.error("Integrated session creation failed",
                        user_id=user_input.user_id,
                        error=str(e))
            return None
    
    async def complete_integrated_session(
        self,
        session_id: UUID,
        response: PRSMResponse,
        agents_used: List[AgentType]
    ) -> Dict[str, Any]:
        """
        Complete integrated session with final cost calculation
        
        Args:
            session_id: Session to complete
            response: Final response
            agents_used: Agents used in session
            
        Returns:
            Completion summary with cost breakdown
        """
        try:
            integrated_session = self.active_sessions.get(session_id)
            if not integrated_session:
                raise ValueError(f"Session {session_id} not found in active sessions")
            
            # Finalize session
            completion_summary = await integrated_session.finalize_session_with_precise_charging(
                response=response,
                agents_used=agents_used
            )
            
            # Update metrics
            self.integration_metrics['sessions_completed'] += 1
            self.integration_metrics['total_cost_processed'] += Decimal(str(response.ftns_charged))
            self.integration_metrics['total_context_processed'] += response.context_used
            
            # Update cost accuracy tracking
            cost_accuracy = completion_summary['cost_breakdown']['cost_accuracy']
            current_avg = self.integration_metrics['avg_cost_accuracy']
            completed_sessions = self.integration_metrics['sessions_completed']
            self.integration_metrics['avg_cost_accuracy'] = (
                (current_avg * (completed_sessions - 1) + cost_accuracy) / completed_sessions
            )
            
            # Clean up
            del self.active_sessions[session_id]
            
            logger.info("Integrated session completed",
                       session_id=str(session_id),
                       final_cost=response.ftns_charged,
                       cost_accuracy=cost_accuracy)
            
            return completion_summary
            
        except Exception as e:
            logger.error("Integrated session completion failed",
                        session_id=str(session_id),
                        error=str(e))
            await self._handle_session_failure(session_id, e)
            raise
    
    async def get_integration_analytics(self) -> Dict[str, Any]:
        """Get comprehensive integration analytics"""
        try:
            analytics = {
                'session_metrics': dict(self.integration_metrics),
                'active_sessions': len(self.active_sessions),
                'cost_efficiency': {
                    'avg_cost_per_session': float(
                        self.integration_metrics['total_cost_processed'] / 
                        max(self.integration_metrics['sessions_completed'], 1)
                    ),
                    'avg_context_per_session': (
                        self.integration_metrics['total_context_processed'] / 
                        max(self.integration_metrics['sessions_completed'], 1)
                    ),
                    'cost_per_context_unit': float(
                        self.integration_metrics['total_cost_processed'] / 
                        max(self.integration_metrics['total_context_processed'], 1)
                    )
                },
                'quality_metrics': {
                    'avg_cost_accuracy': self.integration_metrics['avg_cost_accuracy'],
                    'session_success_rate': (
                        self.integration_metrics['sessions_completed'] /
                        max(self.integration_metrics['sessions_created'], 1)
                    ),
                    'failure_rate': (
                        self.integration_metrics['sessions_failed'] /
                        max(self.integration_metrics['sessions_created'], 1)
                    )
                }
            }
            
            # Add FTNS service analytics
            ftns_analytics = await self.ftns_service.generate_usage_analytics(
                start_time=datetime.now(timezone.utc) - timedelta(hours=24),
                end_time=datetime.now(timezone.utc)
            )
            
            analytics['ftns_analytics'] = {
                'total_transactions': ftns_analytics.total_transactions,
                'total_cost_ftns': float(ftns_analytics.total_cost_ftns),
                'avg_cost_per_transaction': float(ftns_analytics.avg_cost_per_transaction),
                'cost_by_agent': {k: float(v) for k, v in ftns_analytics.cost_by_agent.items()},
                'performance_metrics': ftns_analytics.performance_metrics
            }
            
            return analytics
            
        except Exception as e:
            logger.error("Integration analytics failed", error=str(e))
            return {'error': str(e)}
    
    # === Private Helper Methods ===
    
    async def _handle_session_failure(self, session_id: UUID, error: Exception):
        """Handle session failure with cleanup"""
        try:
            integrated_session = self.active_sessions.get(session_id)
            if integrated_session:
                error_summary = await integrated_session.handle_session_error(error)
                self.integration_metrics['sessions_failed'] += 1
                
                # Clean up
                del self.active_sessions[session_id]
                
                logger.info("Session failure handled",
                           session_id=str(session_id),
                           error_summary=error_summary)
        
        except Exception as e:
            logger.error("Session failure handling failed",
                        session_id=str(session_id),
                        error=str(e))

# Global integrator instance
nwtn_ftns_integrator = None

def get_nwtn_ftns_integrator() -> NWTNFTNSIntegrator:
    """Get or create global NWTN-FTNS integrator instance"""
    global nwtn_ftns_integrator
    if nwtn_ftns_integrator is None:
        nwtn_ftns_integrator = NWTNFTNSIntegrator()
    return nwtn_ftns_integrator