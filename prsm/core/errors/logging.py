"""
Structured Error Logging
========================

Enhanced logging system for PRSM errors with structured data,
contextual information, and integration with monitoring systems.
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

from .exceptions import PRSMException, ErrorSeverity, ErrorCategory

# Configure structured logger
logger = logging.getLogger(__name__)


@dataclass
class ErrorLogEntry:
    """Structured error log entry"""
    error_id: str
    timestamp: datetime
    error_code: str
    message: str
    component: str
    operation: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None


class ErrorLogger:
    """Enhanced error logger with structured output"""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger
        self.log_buffer: List[ErrorLogEntry] = []
        self.buffer_size = 100
        self.enable_buffering = False
        
        # Configure structured logging format
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure structured logging format"""
        if not self.logger.handlers:
            # Create console handler with JSON formatter
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(StructuredErrorFormatter())
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
    
    def log_error(
        self,
        error: Union[PRSMException, Exception],
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ):
        """Log error with structured data"""
        
        # Convert to PRSMException if needed
        if isinstance(error, PRSMException):
            prsm_error = error
        else:
            from .exceptions import ProcessingError
            prsm_error = ProcessingError(
                message=str(error),
                component="unknown",
                operation="unknown",
                original_exception=error,
                context=context
            )
        
        # Create log entry
        log_entry = ErrorLogEntry(
            error_id=prsm_error.error_id,
            timestamp=prsm_error.timestamp,
            error_code=prsm_error.error_code,
            message=prsm_error.message,
            component=prsm_error.context.get("component", "unknown"),
            operation=prsm_error.context.get("operation", "unknown"),
            severity=prsm_error.severity,
            category=prsm_error.category,
            context=prsm_error.context,
            stack_trace=prsm_error.stack_trace,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id
        )
        
        # Add to buffer if enabled
        if self.enable_buffering:
            self._add_to_buffer(log_entry)
        
        # Log immediately
        self._write_log_entry(log_entry)
    
    async def log_error_async(
        self,
        error: Union[PRSMException, Exception],
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ):
        """Asynchronous error logging"""
        # Run logging in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.log_error,
            error,
            context,
            user_id,
            request_id,
            correlation_id
        )
    
    def _write_log_entry(self, entry: ErrorLogEntry):
        """Write log entry using appropriate log level"""
        log_data = asdict(entry)
        
        # Convert datetime to ISO string
        log_data["timestamp"] = entry.timestamp.isoformat()
        log_data["severity"] = entry.severity.value
        log_data["category"] = entry.category.value
        
        # Choose log level based on severity
        if entry.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", extra={"error_data": log_data})
        elif entry.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error", extra={"error_data": log_data})
        elif entry.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error", extra={"error_data": log_data})
        else:
            self.logger.info("Low severity error", extra={"error_data": log_data})
    
    def _add_to_buffer(self, entry: ErrorLogEntry):
        """Add entry to buffer for batch processing"""
        self.log_buffer.append(entry)
        
        # Flush buffer if full
        if len(self.log_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Flush buffered log entries"""
        if not self.log_buffer:
            return
        
        for entry in self.log_buffer:
            self._write_log_entry(entry)
        
        self.log_buffer.clear()
    
    def enable_file_logging(self, log_file_path: str):
        """Enable file-based logging"""
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(StructuredErrorFormatter())
        self.logger.addHandler(file_handler)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error logging statistics"""
        return {
            "buffer_size": len(self.log_buffer),
            "buffer_enabled": self.enable_buffering,
            "handlers_count": len(self.logger.handlers),
            "logger_level": self.logger.level
        }


class StructuredErrorFormatter(logging.Formatter):
    """JSON formatter for structured error logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add error data if present
        if hasattr(record, "error_data"):
            log_data["error"] = record.error_data
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", "msecs",
                          "relativeCreated", "thread", "threadName", "processName", 
                          "process", "getMessage", "exc_info", "exc_text", "stack_info",
                          "error_data"]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str, separators=(',', ':'))


class ComponentErrorLogger(ErrorLogger):
    """Component-specific error logger"""
    
    def __init__(self, component_name: str, logger_instance: Optional[logging.Logger] = None):
        super().__init__(logger_instance)
        self.component_name = component_name
        self.component_stats = {
            "total_errors": 0,
            "errors_by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "errors_by_category": {category.value: 0 for category in ErrorCategory},
            "recent_errors": []
        }
    
    def log_error(
        self,
        error: Union[PRSMException, Exception],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log error with component context"""
        # Add component to context
        if context is None:
            context = {}
        context["component"] = self.component_name
        
        # Update component statistics
        self._update_component_stats(error)
        
        # Call parent method
        super().log_error(error, context, **kwargs)
    
    def _update_component_stats(self, error: Union[PRSMException, Exception]):
        """Update component error statistics"""
        self.component_stats["total_errors"] += 1
        
        if isinstance(error, PRSMException):
            self.component_stats["errors_by_severity"][error.severity.value] += 1
            self.component_stats["errors_by_category"][error.category.value] += 1
            
            # Keep recent errors (max 50)
            error_summary = {
                "error_id": error.error_id,
                "error_code": error.error_code,
                "timestamp": error.timestamp.isoformat(),
                "severity": error.severity.value
            }
            
            self.component_stats["recent_errors"].append(error_summary)
            if len(self.component_stats["recent_errors"]) > 50:
                self.component_stats["recent_errors"].pop(0)
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """Get component-specific error statistics"""
        return {
            "component": self.component_name,
            **self.component_stats,
            **self.get_error_statistics()
        }


class AggregatedErrorLogger:
    """Aggregate error logging across multiple components"""
    
    def __init__(self):
        self.component_loggers: Dict[str, ComponentErrorLogger] = {}
        self.global_stats = {
            "total_errors": 0,
            "errors_by_component": {},
            "errors_by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "errors_by_category": {category.value: 0 for category in ErrorCategory}
        }
    
    def get_component_logger(self, component_name: str) -> ComponentErrorLogger:
        """Get or create component logger"""
        if component_name not in self.component_loggers:
            self.component_loggers[component_name] = ComponentErrorLogger(component_name)
        return self.component_loggers[component_name]
    
    def log_error(
        self,
        component_name: str,
        error: Union[PRSMException, Exception],
        **kwargs
    ):
        """Log error for specific component"""
        component_logger = self.get_component_logger(component_name)
        component_logger.log_error(error, **kwargs)
        
        # Update global stats
        self._update_global_stats(component_name, error)
    
    def _update_global_stats(self, component_name: str, error: Union[PRSMException, Exception]):
        """Update global error statistics"""
        self.global_stats["total_errors"] += 1
        
        # Component stats
        if component_name not in self.global_stats["errors_by_component"]:
            self.global_stats["errors_by_component"][component_name] = 0
        self.global_stats["errors_by_component"][component_name] += 1
        
        # Severity and category stats
        if isinstance(error, PRSMException):
            self.global_stats["errors_by_severity"][error.severity.value] += 1
            self.global_stats["errors_by_category"][error.category.value] += 1
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global error statistics"""
        return {
            "global_stats": self.global_stats,
            "component_count": len(self.component_loggers),
            "component_stats": {
                name: logger.get_component_statistics()
                for name, logger in self.component_loggers.items()
            }
        }
    
    def flush_all_buffers(self):
        """Flush all component logger buffers"""
        for logger in self.component_loggers.values():
            logger.flush_buffer()


# Global error logger instance
_global_error_logger = AggregatedErrorLogger()


def log_error_with_context(
    component_name: str,
    error: Union[PRSMException, Exception],
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
):
    """Convenience function for logging errors with context"""
    _global_error_logger.log_error(
        component_name=component_name,
        error=error,
        context=context,
        user_id=user_id,
        request_id=request_id
    )


def get_error_logger(component_name: str) -> ComponentErrorLogger:
    """Get component-specific error logger"""
    return _global_error_logger.get_component_logger(component_name)


def get_global_error_statistics() -> Dict[str, Any]:
    """Get global error statistics"""
    return _global_error_logger.get_global_statistics()


def flush_all_error_logs():
    """Flush all error log buffers"""
    _global_error_logger.flush_all_buffers()