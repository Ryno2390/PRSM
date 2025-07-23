"""
PRSM Exception Hierarchy
========================

Comprehensive exception hierarchy for all PRSM components with
rich context information and error codes.
"""

import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category classification"""
    SYSTEM = "system"
    USER_INPUT = "user_input"
    BUSINESS_LOGIC = "business_logic"
    NETWORK = "network"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"


class PRSMException(Exception):
    """
    Base exception class for all PRSM errors.
    
    Provides rich context information, error classification,
    and structured logging support.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        user_message: Optional[str] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_exception = original_exception
        self.user_message = user_message or message
        self.recovery_suggestions = recovery_suggestions or []
        
        # Metadata
        self.timestamp = datetime.now(timezone.utc)
        self.stack_trace = traceback.format_exc() if original_exception else None
        self.error_id = self._generate_error_id()
        
        super().__init__(message)
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking"""
        import uuid
        return f"PRSM-{uuid.uuid4().hex[:8].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "stack_trace": self.stack_trace
        }
    
    def add_context(self, key: str, value: Any) -> 'PRSMException':
        """Add context information (fluent interface)"""
        self.context[key] = value
        return self
    
    def with_recovery_suggestion(self, suggestion: str) -> 'PRSMException':
        """Add recovery suggestion (fluent interface)"""
        self.recovery_suggestions.append(suggestion)
        return self
    
    def is_retryable(self) -> bool:
        """Check if error is retryable (can be overridden by subclasses)"""
        return self.category in [ErrorCategory.NETWORK, ErrorCategory.RESOURCE]
    
    def get_user_friendly_message(self) -> str:
        """Get user-friendly error message"""
        return self.user_message


# NWTN-specific exceptions
class NWTNError(PRSMException):
    """Base class for NWTN-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        reasoning_engine: Optional[str] = None,
        query_id: Optional[str] = None,
        **kwargs
    ):
        self.reasoning_engine = reasoning_engine
        self.query_id = query_id
        
        context = kwargs.get('context', {})
        context.update({
            "component": "nwtn",
            "reasoning_engine": reasoning_engine,
            "query_id": query_id
        })
        kwargs['context'] = context
        
        super().__init__(message, error_code, **kwargs)


class ReasoningEngineError(NWTNError):
    """Errors from individual reasoning engines"""
    
    def __init__(
        self,
        message: str,
        reasoning_engine: str,
        processing_stage: str,
        query_id: Optional[str] = None,
        **kwargs
    ):
        self.processing_stage = processing_stage
        
        super().__init__(
            message=message,
            error_code=f"REASONING_ENGINE_{reasoning_engine.upper()}_ERROR",
            reasoning_engine=reasoning_engine,
            query_id=query_id,
            **kwargs
        )
        
        self.context["processing_stage"] = processing_stage


class MetaReasoningError(NWTNError):
    """Errors in meta-reasoning coordination"""
    
    def __init__(
        self,
        message: str,
        orchestration_stage: str,
        engines_involved: List[str],
        **kwargs
    ):
        self.orchestration_stage = orchestration_stage
        self.engines_involved = engines_involved
        
        super().__init__(
            message=message,
            error_code="META_REASONING_ERROR",
            **kwargs
        )
        
        self.context.update({
            "orchestration_stage": orchestration_stage,
            "engines_involved": engines_involved,
            "engine_count": len(engines_involved)
        })


class AnalogicalReasoningError(NWTNError):
    """Errors in analogical reasoning and chain discovery"""
    
    def __init__(
        self,
        message: str,
        source_domain: Optional[str] = None,
        target_domain: Optional[str] = None,
        chain_depth: Optional[int] = None,
        **kwargs
    ):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.chain_depth = chain_depth
        
        super().__init__(
            message=message,
            error_code="ANALOGICAL_REASONING_ERROR",
            **kwargs
        )
        
        self.context.update({
            "source_domain": source_domain,
            "target_domain": target_domain,
            "chain_depth": chain_depth
        })


# Tokenomics-specific exceptions
class TokenomicsError(PRSMException):
    """Base class for FTNS tokenomics errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.user_id = user_id
        self.operation = operation
        
        context = kwargs.get('context', {})
        context.update({
            "component": "tokenomics",
            "user_id": user_id,
            "operation": operation
        })
        kwargs['context'] = context
        
        super().__init__(message, error_code, **kwargs)


class InsufficientFundsError(TokenomicsError):
    """Insufficient FTNS balance error"""
    
    def __init__(
        self,
        user_id: str,
        required_amount: float,
        available_amount: float,
        operation: str,
        **kwargs
    ):
        self.required_amount = required_amount
        self.available_amount = available_amount
        
        message = f"Insufficient FTNS balance for {operation}: need {required_amount}, have {available_amount}"
        
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_FUNDS",
            user_id=user_id,
            operation=operation,
            category=ErrorCategory.BUSINESS_LOGIC,
            user_message=f"Insufficient FTNS balance. Need {required_amount} FTNS, but only have {available_amount} FTNS.",
            recovery_suggestions=[
                "Add more FTNS to your account",
                "Choose a less expensive processing option", 
                "Contact support for balance assistance"
            ],
            **kwargs
        )
        
        self.context.update({
            "required_amount": required_amount,
            "available_amount": available_amount,
            "deficit": required_amount - available_amount
        })


class PricingCalculationError(TokenomicsError):
    """Error in FTNS pricing calculation"""
    
    def __init__(
        self,
        message: str,
        calculation_stage: str,
        query_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.calculation_stage = calculation_stage
        self.query_parameters = query_parameters or {}
        
        super().__init__(
            message=message,
            error_code="PRICING_CALCULATION_ERROR",
            operation="pricing_calculation",
            **kwargs
        )
        
        self.context.update({
            "calculation_stage": calculation_stage,
            "query_parameters": query_parameters
        })


class MarketRateError(TokenomicsError):
    """Error in market rate calculation or retrieval"""
    
    def __init__(
        self,
        message: str,
        market_condition: Optional[str] = None,
        current_rate: Optional[float] = None,
        **kwargs
    ):
        self.market_condition = market_condition
        self.current_rate = current_rate
        
        super().__init__(
            message=message,
            error_code="MARKET_RATE_ERROR",
            operation="market_rate_calculation",
            **kwargs
        )
        
        self.context.update({
            "market_condition": market_condition,
            "current_rate": current_rate
        })


# Marketplace-specific exceptions
class MarketplaceError(PRSMException):
    """Base class for marketplace errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        asset_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.asset_id = asset_id
        self.user_id = user_id
        self.operation = operation
        
        context = kwargs.get('context', {})
        context.update({
            "component": "marketplace",
            "asset_id": asset_id,
            "user_id": user_id,
            "operation": operation
        })
        kwargs['context'] = context
        
        super().__init__(message, error_code, **kwargs)


class AssetNotFoundError(MarketplaceError):
    """Asset not found error"""
    
    def __init__(self, asset_id: str, **kwargs):
        super().__init__(
            message=f"Asset not found: {asset_id}",
            error_code="ASSET_NOT_FOUND",
            asset_id=asset_id,
            category=ErrorCategory.USER_INPUT,
            user_message=f"The requested asset (ID: {asset_id}) could not be found.",
            recovery_suggestions=[
                "Check the asset ID for typos",
                "Browse available assets in the marketplace",
                "Contact the asset creator"
            ],
            **kwargs
        )


class AssetPermissionError(MarketplaceError):
    """Asset permission/access error"""
    
    def __init__(
        self,
        asset_id: str,
        user_id: str,
        required_permission: str,
        **kwargs
    ):
        self.required_permission = required_permission
        
        super().__init__(
            message=f"Insufficient permissions for asset {asset_id}: need {required_permission}",
            error_code="ASSET_PERMISSION_ERROR",
            asset_id=asset_id,
            user_id=user_id,
            category=ErrorCategory.SECURITY,
            user_message="You don't have permission to access this asset.",
            recovery_suggestions=[
                "Purchase the asset if it's available for sale",
                "Contact the asset owner for access",
                "Check your subscription level"
            ],
            **kwargs
        )
        
        self.context["required_permission"] = required_permission


# Federation-specific exceptions
class FederationError(PRSMException):
    """Base class for federation/P2P network errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        node_id: Optional[str] = None,
        network_operation: Optional[str] = None,
        **kwargs
    ):
        self.node_id = node_id
        self.network_operation = network_operation
        
        context = kwargs.get('context', {})
        context.update({
            "component": "federation",
            "node_id": node_id,
            "network_operation": network_operation
        })
        kwargs['context'] = context
        
        super().__init__(message, error_code, **kwargs)


class NetworkConnectivityError(FederationError):
    """Network connectivity error"""
    
    def __init__(
        self,
        message: str,
        node_id: str,
        connection_type: str,
        **kwargs
    ):
        self.connection_type = connection_type
        
        super().__init__(
            message=message,
            error_code="NETWORK_CONNECTIVITY_ERROR",
            node_id=node_id,
            network_operation=f"connect_{connection_type}",
            category=ErrorCategory.NETWORK,
            **kwargs
        )
        
        self.context["connection_type"] = connection_type
    
    def is_retryable(self) -> bool:
        """Network errors are generally retryable"""
        return True


class ConsensusError(FederationError):
    """Consensus mechanism error"""
    
    def __init__(
        self,
        message: str,
        consensus_round: int,
        participating_nodes: List[str],
        **kwargs
    ):
        self.consensus_round = consensus_round
        self.participating_nodes = participating_nodes
        
        super().__init__(
            message=message,
            error_code="CONSENSUS_ERROR",
            network_operation="consensus",
            **kwargs
        )
        
        self.context.update({
            "consensus_round": consensus_round,
            "participating_nodes": participating_nodes,
            "node_count": len(participating_nodes)
        })


# System-level exceptions
class ProcessingError(PRSMException):
    """General processing error"""
    
    def __init__(
        self,
        message: str,
        component: str,
        operation: str,
        **kwargs
    ):
        self.component = component
        self.operation = operation
        
        super().__init__(
            message=message,
            error_code=f"{component.upper()}_PROCESSING_ERROR",
            category=ErrorCategory.SYSTEM,
            **kwargs
        )
        
        self.context.update({
            "component": component,
            "operation": operation
        })


class ResourceError(PRSMException):
    """Resource availability/exhaustion error"""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        requested: Union[int, float, str],
        available: Union[int, float, str],
        **kwargs
    ):
        self.resource_type = resource_type
        self.requested = requested
        self.available = available
        
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            category=ErrorCategory.RESOURCE,
            **kwargs
        )
        
        self.context.update({
            "resource_type": resource_type,
            "requested": requested,
            "available": available
        })
    
    def is_retryable(self) -> bool:
        """Resource errors may be retryable after some time"""
        return True


class ConfigurationError(PRSMException):
    """Configuration error"""
    
    def __init__(
        self,
        message: str,
        config_key: str,
        config_value: Any = None,
        **kwargs
    ):
        self.config_key = config_key
        self.config_value = config_value
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        self.context.update({
            "config_key": config_key,
            "config_value": config_value
        })


# Security-related exceptions
class SecurityError(PRSMException):
    """Base class for security errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        security_threat: str,
        **kwargs
    ):
        self.security_threat = security_threat
        
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        
        self.context["security_threat"] = security_threat


class AuthenticationError(SecurityError):
    """Authentication failure"""
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        auth_method: Optional[str] = None,
        **kwargs
    ):
        self.user_id = user_id
        self.auth_method = auth_method
        
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            security_threat="authentication_failure",
            **kwargs
        )
        
        self.context.update({
            "user_id": user_id,
            "auth_method": auth_method
        })


class AuthorizationError(SecurityError):
    """Authorization failure"""
    
    def __init__(
        self,
        message: str,
        user_id: str,
        required_permission: str,
        resource: Optional[str] = None,
        **kwargs
    ):
        self.user_id = user_id
        self.required_permission = required_permission
        self.resource = resource
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            security_threat="authorization_failure",
            **kwargs
        )
        
        self.context.update({
            "user_id": user_id,
            "required_permission": required_permission,
            "resource": resource
        })


# Utility function for error creation
def create_error(
    error_type: type,
    message: str,
    **kwargs
) -> PRSMException:
    """Factory function for creating errors with proper context"""
    if not issubclass(error_type, PRSMException):
        raise ValueError("error_type must be a subclass of PRSMException")
    
    return error_type(message, **kwargs)