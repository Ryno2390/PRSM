"""
Error Handlers and Decorators
=============================

Standardized error handling decorators and handlers for consistent
error processing across all PRSM components.
"""

import asyncio
import logging
import functools
from typing import Callable, Any, Dict, Optional, Union, Type, Awaitable
from contextlib import contextmanager, asynccontextmanager

from .exceptions import PRSMException, ProcessingError, ErrorSeverity, ErrorCategory
from .logging import ErrorLogger

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Synchronous error handler with recovery strategies"""
    
    def __init__(
        self,
        component_name: str,
        default_recovery: str = "raise",
        logger_instance: Optional[logging.Logger] = None
    ):
        self.component_name = component_name
        self.default_recovery = default_recovery
        self.error_logger = ErrorLogger(logger_instance or logger)
        self.error_counts = {}
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recovery_strategy: Optional[str] = None
    ) -> Any:
        """Handle error with specified or default recovery strategy"""
        
        # Convert to PRSM exception if needed
        if not isinstance(error, PRSMException):
            prsm_error = ProcessingError(
                message=str(error),
                component=self.component_name,
                operation="unknown",
                original_exception=error,
                context=context
            )
        else:
            prsm_error = error
            if context:
                prsm_error.context.update(context)
        
        # Log the error
        self.error_logger.log_error(prsm_error)
        
        # Update error counts
        error_key = f"{self.component_name}:{prsm_error.error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Apply recovery strategy
        strategy = recovery_strategy or self.default_recovery
        return self._apply_recovery_strategy(prsm_error, strategy)
    
    def _apply_recovery_strategy(self, error: PRSMException, strategy: str) -> Any:
        """Apply recovery strategy"""
        if strategy == "raise":
            raise error
        elif strategy == "return_none":
            return None
        elif strategy == "return_false":
            return False
        elif strategy == "return_empty":
            return {}
        elif strategy == "log_and_continue":
            logger.warning(f"Error handled, continuing: {error.message}")
            return None
        else:
            logger.error(f"Unknown recovery strategy: {strategy}")
            raise error


class AsyncErrorHandler:
    """Asynchronous error handler with recovery strategies"""
    
    def __init__(
        self,
        component_name: str,
        default_recovery: str = "raise",
        logger_instance: Optional[logging.Logger] = None
    ):
        self.component_name = component_name
        self.default_recovery = default_recovery
        self.error_logger = ErrorLogger(logger_instance or logger)
        self.error_counts = {}
    
    async def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recovery_strategy: Optional[str] = None
    ) -> Any:
        """Handle error asynchronously with specified recovery strategy"""
        
        # Convert to PRSM exception if needed
        if not isinstance(error, PRSMException):
            prsm_error = ProcessingError(
                message=str(error),
                component=self.component_name,
                operation="unknown",
                original_exception=error,
                context=context
            )
        else:
            prsm_error = error
            if context:
                prsm_error.context.update(context)
        
        # Log the error
        await self.error_logger.log_error_async(prsm_error)
        
        # Update error counts
        error_key = f"{self.component_name}:{prsm_error.error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Apply recovery strategy
        strategy = recovery_strategy or self.default_recovery
        return await self._apply_recovery_strategy(prsm_error, strategy)
    
    async def _apply_recovery_strategy(self, error: PRSMException, strategy: str) -> Any:
        """Apply recovery strategy asynchronously"""
        if strategy == "raise":
            raise error
        elif strategy == "return_none":
            return None
        elif strategy == "return_false":
            return False
        elif strategy == "return_empty":
            return {}
        elif strategy == "log_and_continue":
            logger.warning(f"Error handled, continuing: {error.message}")
            return None
        elif strategy == "retry_once":
            # This would need to be implemented by specific handlers
            logger.info(f"Retry requested for error: {error.error_code}")
            raise error
        else:
            logger.error(f"Unknown recovery strategy: {strategy}")
            raise error


# Synchronous error handling decorator
def handle_errors(
    component_name: str = "unknown",
    operation_name: str = "unknown",
    recovery_strategy: str = "raise",
    log_errors: bool = True,
    context_extractor: Optional[Callable[..., Dict[str, Any]]] = None
):
    """
    Decorator for standardized error handling in synchronous functions.
    
    Args:
        component_name: Name of the component for error context
        operation_name: Name of the operation for error context
        recovery_strategy: How to handle errors ("raise", "return_none", etc.)
        log_errors: Whether to log errors
        context_extractor: Function to extract context from function args
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler(
                component_name=component_name,
                default_recovery=recovery_strategy
            )
            
            try:
                return func(*args, **kwargs)
            
            except PRSMException as e:
                # Add operation context
                e.context.update({
                    "operation": operation_name,
                    "function": func.__name__
                })
                
                # Extract additional context if provided
                if context_extractor:
                    try:
                        additional_context = context_extractor(*args, **kwargs)
                        e.context.update(additional_context)
                    except Exception as ctx_error:
                        logger.warning(f"Context extraction failed: {ctx_error}")
                
                return error_handler.handle_error(e, recovery_strategy=recovery_strategy)
            
            except Exception as e:
                # Create context
                context = {
                    "operation": operation_name,
                    "function": func.__name__,
                    "component": component_name
                }
                
                # Extract additional context if provided
                if context_extractor:
                    try:
                        additional_context = context_extractor(*args, **kwargs)
                        context.update(additional_context)
                    except Exception as ctx_error:
                        logger.warning(f"Context extraction failed: {ctx_error}")
                
                return error_handler.handle_error(e, context=context, recovery_strategy=recovery_strategy)
        
        return wrapper
    return decorator


# Asynchronous error handling decorator
def handle_async_errors(
    component_name: str = "unknown",
    operation_name: str = "unknown", 
    recovery_strategy: str = "raise",
    log_errors: bool = True,
    context_extractor: Optional[Callable[..., Dict[str, Any]]] = None
):
    """
    Decorator for standardized error handling in asynchronous functions.
    
    Args:
        component_name: Name of the component for error context
        operation_name: Name of the operation for error context
        recovery_strategy: How to handle errors ("raise", "return_none", etc.)
        log_errors: Whether to log errors
        context_extractor: Function to extract context from function args
    """
    def decorator(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            error_handler = AsyncErrorHandler(
                component_name=component_name,
                default_recovery=recovery_strategy
            )
            
            try:
                return await func(*args, **kwargs)
            
            except PRSMException as e:
                # Add operation context
                e.context.update({
                    "operation": operation_name,
                    "function": func.__name__
                })
                
                # Extract additional context if provided
                if context_extractor:
                    try:
                        additional_context = context_extractor(*args, **kwargs)
                        e.context.update(additional_context)
                    except Exception as ctx_error:
                        logger.warning(f"Context extraction failed: {ctx_error}")
                
                return await error_handler.handle_error(e, recovery_strategy=recovery_strategy)
            
            except Exception as e:
                # Create context
                context = {
                    "operation": operation_name,
                    "function": func.__name__,
                    "component": component_name
                }
                
                # Extract additional context if provided  
                if context_extractor:
                    try:
                        additional_context = context_extractor(*args, **kwargs)
                        context.update(additional_context)
                    except Exception as ctx_error:
                        logger.warning(f"Context extraction failed: {ctx_error}")
                
                return await error_handler.handle_error(e, context=context, recovery_strategy=recovery_strategy)
        
        return wrapper
    return decorator


# Context managers for error handling
@contextmanager
def error_context(
    component_name: str,
    operation_name: str,
    recovery_strategy: str = "raise",
    context: Optional[Dict[str, Any]] = None
):
    """Context manager for error handling"""
    error_handler = ErrorHandler(component_name, recovery_strategy)
    
    try:
        yield error_handler
    except Exception as e:
        error_context = context or {}
        error_context.update({
            "operation": operation_name,
            "component": component_name
        })
        
        error_handler.handle_error(e, context=error_context, recovery_strategy=recovery_strategy)


@asynccontextmanager
async def async_error_context(
    component_name: str,
    operation_name: str,
    recovery_strategy: str = "raise",
    context: Optional[Dict[str, Any]] = None
):
    """Async context manager for error handling"""
    error_handler = AsyncErrorHandler(component_name, recovery_strategy)
    
    try:
        yield error_handler
    except Exception as e:
        error_context = context or {}
        error_context.update({
            "operation": operation_name,
            "component": component_name
        })
        
        await error_handler.handle_error(e, context=error_context, recovery_strategy=recovery_strategy)


# Specialized decorators for PRSM components
def handle_nwtn_errors(
    reasoning_engine: Optional[str] = None,
    operation: str = "reasoning",
    recovery_strategy: str = "raise"
):
    """Specialized decorator for NWTN errors"""
    def context_extractor(*args, **kwargs):
        context = {}
        if len(args) > 0 and hasattr(args[0], 'query_id'):
            context['query_id'] = args[0].query_id
        if 'query' in kwargs:
            context['query_length'] = len(kwargs['query'])
        if reasoning_engine:
            context['reasoning_engine'] = reasoning_engine
        return context
    
    return handle_async_errors(
        component_name="nwtn",
        operation_name=operation,
        recovery_strategy=recovery_strategy,
        context_extractor=context_extractor
    )


def handle_tokenomics_errors(
    operation: str = "tokenomics_operation",
    recovery_strategy: str = "raise"
):
    """Specialized decorator for tokenomics errors"""
    def context_extractor(*args, **kwargs):
        context = {}
        if 'user_id' in kwargs:
            context['user_id'] = kwargs['user_id']
        if 'amount' in kwargs:
            context['amount'] = kwargs['amount']
        return context
    
    return handle_async_errors(
        component_name="tokenomics",
        operation_name=operation,
        recovery_strategy=recovery_strategy,
        context_extractor=context_extractor
    )


def handle_marketplace_errors(
    operation: str = "marketplace_operation",
    recovery_strategy: str = "raise"
):
    """Specialized decorator for marketplace errors"""
    def context_extractor(*args, **kwargs):
        context = {}
        if 'user_id' in kwargs:
            context['user_id'] = kwargs['user_id']
        if 'asset_id' in kwargs:
            context['asset_id'] = kwargs['asset_id']
        return context
    
    return handle_async_errors(
        component_name="marketplace",
        operation_name=operation,
        recovery_strategy=recovery_strategy,
        context_extractor=context_extractor
    )


# Error aggregation utilities
class ErrorCollector:
    """Collect multiple errors for batch processing"""
    
    def __init__(self):
        self.errors: list[PRSMException] = []
    
    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the collection"""
        if isinstance(error, PRSMException):
            if context:
                error.context.update(context)
            self.errors.append(error)
        else:
            prsm_error = ProcessingError(
                message=str(error),
                component="error_collector",
                operation="collect",
                original_exception=error,
                context=context
            )
            self.errors.append(prsm_error)
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
    
    def get_error_count(self) -> int:
        """Get number of errors"""
        return len(self.errors)
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> list[PRSMException]:
        """Get errors by severity level"""
        return [e for e in self.errors if e.severity == severity]
    
    def get_critical_errors(self) -> list[PRSMException]:
        """Get critical errors"""
        return self.get_errors_by_severity(ErrorSeverity.CRITICAL)
    
    def raise_if_critical(self):
        """Raise exception if there are critical errors"""
        critical_errors = self.get_critical_errors()
        if critical_errors:
            # Raise the first critical error
            raise critical_errors[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error_count": len(self.errors),
            "errors": [error.to_dict() for error in self.errors],
            "severity_breakdown": {
                severity.value: len(self.get_errors_by_severity(severity))
                for severity in ErrorSeverity
            }
        }