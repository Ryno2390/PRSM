"""
PRSM Logging Configuration

Centralized logging configuration for all PRSM executor modules providing:
- Consistent structured logging with structlog
- Production-ready logging configuration
- Standardized log formats and metadata
- Environment-specific logging levels
- Comprehensive audit trail for compliance
"""

import structlog
import logging
import sys
from typing import Any, Dict
from pathlib import Path


def configure_structlog(
    environment: str = "production",
    log_level: str = "INFO",
    enable_json_logs: bool = True,
    log_file_path: str = None
) -> None:
    """
    Configure structlog for consistent logging across all executor modules
    
    Args:
        environment: Environment name (development, testing, production)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json_logs: Whether to output logs in JSON format for production
        log_file_path: Optional file path for log output
    """
    
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure processors based on environment
    processors = [
        # Add correlation IDs and timestamps
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        
        # Add stack info for errors
        structlog.processors.StackInfoRenderer(),
        
        # Add PRSM-specific metadata
        _add_prsm_context,
    ]
    
    if environment == "development":
        # Development: Human-readable colored output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    else:
        # Production: JSON output for log aggregation
        if enable_json_logs:
            processors.extend([
                structlog.processors.JSONRenderer()
            ])
        else:
            processors.extend([
                structlog.dev.ConsoleRenderer(colors=False)
            ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_prsm_context(logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add PRSM-specific context to log entries"""
    event_dict["system"] = "PRSM"
    event_dict["version"] = "1.0.0-production"
    
    # Add component information if available
    logger_name = event_dict.get("logger", "")
    if "executor" in logger_name:
        event_dict["component"] = "executor"
    elif "orchestrator" in logger_name:
        event_dict["component"] = "orchestrator"
    elif "router" in logger_name:
        event_dict["component"] = "router"
    elif "scalability" in logger_name:
        event_dict["component"] = "scalability"
    
    return event_dict


def get_executor_logger(name: str) -> Any:
    """
    Get a standardized logger for executor modules
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def log_execution_start(logger, operation: str, **kwargs) -> None:
    """Standardized execution start logging"""
    logger.info(
        "Execution started",
        operation=operation,
        **kwargs
    )


def log_execution_complete(logger, operation: str, duration: float, success: bool, **kwargs) -> None:
    """Standardized execution completion logging"""
    logger.info(
        "Execution completed",
        operation=operation,
        duration_seconds=duration,
        success=success,
        **kwargs
    )


def log_execution_error(logger, operation: str, error: Exception, duration: float = None, **kwargs) -> None:
    """Standardized execution error logging"""
    logger.error(
        "Execution failed",
        operation=operation,
        error_type=type(error).__name__,
        error_message=str(error),
        duration_seconds=duration,
        **kwargs,
        exc_info=True
    )


def log_security_event(logger, event_type: str, severity: str, **kwargs) -> None:
    """Standardized security event logging for compliance"""
    logger.warning(
        "Security event",
        event_type=event_type,
        severity=severity,
        audit_trail=True,
        **kwargs
    )


def log_performance_metrics(logger, operation: str, metrics: Dict[str, Any]) -> None:
    """Standardized performance metrics logging"""
    logger.info(
        "Performance metrics",
        operation=operation,
        metrics=metrics,
        category="performance"
    )


# Initialize logging on import for executor modules
def initialize_executor_logging(environment: str = None):
    """Initialize logging for executor modules"""
    import os
    
    # Detect environment if not specified
    if environment is None:
        environment = os.getenv("PRSM_ENVIRONMENT", "production")
    
    log_level = os.getenv("PRSM_LOG_LEVEL", "INFO")
    enable_json = os.getenv("PRSM_JSON_LOGS", "true").lower() == "true"
    
    configure_structlog(
        environment=environment,
        log_level=log_level,
        enable_json_logs=enable_json
    )


# Auto-initialize in production environments
import os
if os.getenv("PRSM_AUTO_INIT_LOGGING", "true").lower() == "true":
    initialize_executor_logging()