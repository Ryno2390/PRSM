"""
PRSM Structured Logging System
Advanced structured logging with correlation IDs, distributed tracing integration, and observability
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import json
import uuid
import logging
import traceback
import contextvars
from collections import defaultdict
import redis.asyncio as aioredis

# Import structured logging libraries
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogFormat(Enum):
    """Log output formats"""
    JSON = "json"
    HUMAN = "human"
    LOGFMT = "logfmt"


@dataclass
class LogContext:
    """Logging context with correlation tracking"""
    correlation_id: str
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service_name: str = "prsm-api"
    service_version: str = "1.0.0"
    environment: str = "production"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    context: LogContext
    fields: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[Dict[str, Any]] = None


@dataclass
class LoggingConfig:
    """Logging system configuration"""
    service_name: str = "prsm-api"
    service_version: str = "1.0.0"
    environment: str = "production"
    
    # Output configuration
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.JSON
    console_output: bool = True
    file_output: bool = False
    file_path: Optional[str] = None
    
    # Structured logging
    enable_correlation_ids: bool = True
    enable_trace_integration: bool = True
    include_caller_info: bool = False
    include_thread_info: bool = False
    
    # Storage configuration
    store_in_redis: bool = True
    redis_key_prefix: str = "logs"
    log_retention_hours: int = 24
    max_log_entries: int = 100000
    
    # Filtering and sampling
    enable_log_sampling: bool = False
    sampling_rate: float = 1.0
    filter_sensitive_data: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: [
        "password", "token", "secret", "key", "authorization"
    ])


class SensitiveDataFilter:
    """Filter for removing sensitive data from logs"""
    
    def __init__(self, sensitive_fields: List[str]):
        self.sensitive_fields = {field.lower() for field in sensitive_fields}
        self.replacement_value = "[REDACTED]"
    
    def filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively filter sensitive data from dictionary"""
        filtered = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                filtered[key] = self.replacement_value
            elif isinstance(value, dict):
                filtered[key] = self.filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self.filter_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        
        return filtered
    
    def filter_string(self, text: str) -> str:
        """Filter sensitive data from string (basic pattern matching)"""
        import re
        
        # Filter common patterns
        patterns = [
            (r'password["\s]*[:=]["\s]*[^"\s]+', f'password="[REDACTED]"'),
            (r'token["\s]*[:=]["\s]*[^"\s]+', f'token="[REDACTED]"'),
            (r'secret["\s]*[:=]["\s]*[^"\s]+', f'secret="[REDACTED]"'),
        ]
        
        filtered_text = text
        for pattern, replacement in patterns:
            filtered_text = re.sub(pattern, replacement, filtered_text, flags=re.IGNORECASE)
        
        return filtered_text


class LogStorage:
    """Redis-based log storage for centralized logging"""
    
    def __init__(self, redis_client: aioredis.Redis, config: LoggingConfig):
        self.redis = redis_client
        self.config = config
        self.retention_seconds = config.log_retention_hours * 3600
    
    async def store_log_entry(self, entry: LogEntry):
        """Store log entry in Redis"""
        try:
            # Create log data
            log_data = {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,
                "message": entry.message,
                "logger_name": entry.logger_name,
                "correlation_id": entry.context.correlation_id,
                "request_id": entry.context.request_id,
                "trace_id": entry.context.trace_id,
                "span_id": entry.context.span_id,
                "user_id": entry.context.user_id,
                "session_id": entry.context.session_id,
                "service_name": entry.context.service_name,
                "service_version": entry.context.service_version,
                "environment": entry.context.environment,
                "fields": entry.fields,
                "exception_info": entry.exception_info,
                "metadata": entry.context.metadata
            }
            
            # Store individual log entry
            log_key = f"{self.config.redis_key_prefix}:entry:{entry.context.correlation_id}:{int(entry.timestamp.timestamp() * 1000000)}"
            await self.redis.setex(log_key, self.retention_seconds, json.dumps(log_data))
            
            # Add to time-ordered index
            timestamp_score = entry.timestamp.timestamp()
            await self.redis.zadd(
                f"{self.config.redis_key_prefix}:timeline",
                {log_key: timestamp_score}
            )
            
            # Add to correlation ID index
            if entry.context.correlation_id:
                await self.redis.sadd(
                    f"{self.config.redis_key_prefix}:correlation:{entry.context.correlation_id}",
                    log_key
                )
                await self.redis.expire(
                    f"{self.config.redis_key_prefix}:correlation:{entry.context.correlation_id}",
                    self.retention_seconds
                )
            
            # Add to level index
            await self.redis.sadd(
                f"{self.config.redis_key_prefix}:level:{entry.level.value}",
                log_key
            )
            await self.redis.expire(
                f"{self.config.redis_key_prefix}:level:{entry.level.value}",
                self.retention_seconds
            )
            
            # Maintain index size
            await self._cleanup_old_entries()
            
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Error storing log entry: {e}")
    
    async def _cleanup_old_entries(self):
        """Clean up old log entries"""
        try:
            # Remove old entries from timeline
            cutoff_time = (datetime.now(timezone.utc).timestamp() - self.retention_seconds)
            await self.redis.zremrangebyscore(
                f"{self.config.redis_key_prefix}:timeline",
                0,
                cutoff_time
            )
            
            # Limit total entries
            current_count = await self.redis.zcard(f"{self.config.redis_key_prefix}:timeline")
            if current_count > self.config.max_log_entries:
                # Remove oldest entries
                excess = current_count - self.config.max_log_entries
                oldest_keys = await self.redis.zrange(
                    f"{self.config.redis_key_prefix}:timeline",
                    0,
                    excess - 1
                )
                
                if oldest_keys:
                    # Remove from timeline
                    await self.redis.zrem(
                        f"{self.config.redis_key_prefix}:timeline",
                        *oldest_keys
                    )
                    
                    # Remove individual entries
                    await self.redis.delete(*oldest_keys)
        
        except Exception as e:
            print(f"Error in log cleanup: {e}")
    
    async def get_logs_by_correlation_id(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a correlation ID"""
        try:
            log_keys = await self.redis.smembers(
                f"{self.config.redis_key_prefix}:correlation:{correlation_id}"
            )
            
            if not log_keys:
                return []
            
            # Get log entries
            logs = []
            for key in log_keys:
                log_data = await self.redis.get(key.decode())
                if log_data:
                    logs.append(json.loads(log_data))
            
            # Sort by timestamp
            logs.sort(key=lambda x: x["timestamp"])
            return logs
            
        except Exception as e:
            print(f"Error retrieving logs by correlation ID: {e}")
            return []
    
    async def get_recent_logs(self, limit: int = 100, 
                            level: Optional[LogLevel] = None) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        try:
            if level:
                # Get from level index
                log_keys = await self.redis.smembers(
                    f"{self.config.redis_key_prefix}:level:{level.value}"
                )
                log_keys = list(log_keys)[-limit:]  # Get most recent
            else:
                # Get from timeline
                log_keys = await self.redis.zrevrange(
                    f"{self.config.redis_key_prefix}:timeline",
                    0,
                    limit - 1
                )
            
            # Get log entries
            logs = []
            for key in log_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                log_data = await self.redis.get(key_str)
                if log_data:
                    logs.append(json.loads(log_data))
            
            return logs
            
        except Exception as e:
            print(f"Error retrieving recent logs: {e}")
            return []


class StructuredLogger:
    """Advanced structured logger with correlation tracking"""
    
    def __init__(self, config: LoggingConfig, redis_client: Optional[aioredis.Redis] = None):
        self.config = config
        self.redis = redis_client
        
        # Context tracking
        self.current_context: contextvars.ContextVar[Optional[LogContext]] = (
            contextvars.ContextVar('current_log_context', default=None)
        )
        
        # Storage
        self.log_storage = None
        if redis_client and config.store_in_redis:
            self.log_storage = LogStorage(redis_client, config)
        
        # Filtering
        self.sensitive_filter = None
        if config.filter_sensitive_data:
            self.sensitive_filter = SensitiveDataFilter(config.sensitive_fields)
        
        # Structured logging setup
        self.structlog_logger = None
        if STRUCTLOG_AVAILABLE:
            self._setup_structlog()
        
        # Statistics
        self.stats = {
            "logs_created": 0,
            "logs_filtered": 0,
            "logs_stored": 0,
            "correlation_ids_created": 0
        }
    
    def _setup_structlog(self):
        """Setup structlog configuration"""
        
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]
        
        if self.config.include_caller_info:
            processors.append(structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameterAdder.PATHNAME,
                    structlog.processors.CallsiteParameterAdder.FILENAME,
                    structlog.processors.CallsiteParameterAdder.MODULE,
                    structlog.processors.CallsiteParameterAdder.FUNC_NAME,
                    structlog.processors.CallsiteParameterAdder.LINENO,
                }
            ))
        
        if self.config.log_format == LogFormat.JSON:
            processors.append(structlog.processors.JSONRenderer())
        elif self.config.log_format == LogFormat.LOGFMT:
            processors.append(structlog.processors.KeyValueRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.config.log_level.value.upper())
            ),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.structlog_logger = structlog.get_logger()
    
    def create_correlation_id(self) -> str:
        """Create a new correlation ID"""
        correlation_id = str(uuid.uuid4())
        self.stats["correlation_ids_created"] += 1
        return correlation_id
    
    def create_context(self, 
                      correlation_id: Optional[str] = None,
                      request_id: Optional[str] = None,
                      trace_id: Optional[str] = None,
                      span_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      **metadata) -> LogContext:
        """Create a new logging context"""
        
        return LogContext(
            correlation_id=correlation_id or self.create_correlation_id(),
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            session_id=session_id,
            service_name=self.config.service_name,
            service_version=self.config.service_version,
            environment=self.config.environment,
            metadata=metadata
        )
    
    def set_context(self, context: LogContext) -> contextvars.Token:
        """Set the current logging context"""
        return self.current_context.set(context)
    
    def get_context(self) -> Optional[LogContext]:
        """Get the current logging context"""
        return self.current_context.get()
    
    def clear_context(self, token: contextvars.Token):
        """Clear the logging context"""
        self.current_context.reset(token)
    
    def _should_sample_log(self, level: LogLevel) -> bool:
        """Determine if log should be sampled"""
        if not self.config.enable_log_sampling:
            return True
        
        # Always log errors and critical messages
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            return True
        
        # Sample based on configured rate
        import random
        return random.random() < self.config.sampling_rate
    
    def _filter_sensitive_data(self, data: Any) -> Any:
        """Filter sensitive data from log data"""
        if not self.sensitive_filter:
            return data
        
        if isinstance(data, dict):
            return self.sensitive_filter.filter_dict(data)
        elif isinstance(data, str):
            return self.sensitive_filter.filter_string(data)
        else:
            return data
    
    async def _create_log_entry(self, 
                               level: LogLevel,
                               message: str,
                               logger_name: str,
                               fields: Optional[Dict[str, Any]] = None,
                               exception: Optional[Exception] = None) -> LogEntry:
        """Create a structured log entry"""
        
        # Get or create context
        context = self.get_context()
        if not context:
            context = self.create_context()
        
        # Get trace information if tracing is enabled
        if self.config.enable_trace_integration:
            try:
                from .tracing import get_current_trace_id, get_current_span_id
                if not context.trace_id:
                    context.trace_id = get_current_trace_id()
                if not context.span_id:
                    context.span_id = get_current_span_id()
            except (ImportError, RuntimeError):
                # Tracing not available or not initialized
                pass
        
        # Process fields
        processed_fields = {}
        if fields:
            processed_fields = self._filter_sensitive_data(fields)
        
        # Process exception info
        exception_info = None
        if exception:
            exception_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # Filter message
        filtered_message = self._filter_sensitive_data(message)
        
        return LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=filtered_message,
            logger_name=logger_name,
            context=context,
            fields=processed_fields,
            exception_info=exception_info
        )
    
    async def log(self, 
                 level: LogLevel,
                 message: str,
                 logger_name: str = "prsm",
                 fields: Optional[Dict[str, Any]] = None,
                 exception: Optional[Exception] = None):
        """Log a message with structured data"""
        
        # Check if log should be sampled
        if not self._should_sample_log(level):
            return
        
        # Create log entry
        entry = await self._create_log_entry(level, message, logger_name, fields, exception)
        
        # Store in Redis if configured
        if self.log_storage:
            try:
                await asyncio.create_task(self.log_storage.store_log_entry(entry))
                self.stats["logs_stored"] += 1
            except Exception as e:
                # Don't let storage errors break logging
                print(f"Error storing log: {e}")
        
        # Log with structlog if available
        if self.structlog_logger:
            try:
                log_data = {
                    "correlation_id": entry.context.correlation_id,
                    "request_id": entry.context.request_id,
                    "trace_id": entry.context.trace_id,
                    "span_id": entry.context.span_id,
                    "user_id": entry.context.user_id,
                    "session_id": entry.context.session_id,
                    "service_name": entry.context.service_name,
                    "service_version": entry.context.service_version,
                    "environment": entry.context.environment,
                    **entry.fields
                }
                
                # Remove None values
                log_data = {k: v for k, v in log_data.items() if v is not None}
                
                # Log with structlog
                getattr(self.structlog_logger, level.value)(
                    message,
                    **log_data,
                    exception_info=entry.exception_info
                )
            except Exception as e:
                print(f"Error with structlog: {e}")
        
        # Fallback to standard logging
        else:
            try:
                # Create standard log record
                python_logger = logging.getLogger(logger_name)
                
                # Format message with context
                formatted_message = f"[{entry.context.correlation_id}] {message}"
                if entry.fields:
                    formatted_message += f" | {json.dumps(entry.fields)}"
                
                # Log at appropriate level
                if level == LogLevel.DEBUG:
                    python_logger.debug(formatted_message, exc_info=exception)
                elif level == LogLevel.INFO:
                    python_logger.info(formatted_message, exc_info=exception)
                elif level == LogLevel.WARNING:
                    python_logger.warning(formatted_message, exc_info=exception)
                elif level == LogLevel.ERROR:
                    python_logger.error(formatted_message, exc_info=exception)
                elif level == LogLevel.CRITICAL:
                    python_logger.critical(formatted_message, exc_info=exception)
            except Exception as e:
                print(f"Error with standard logging: {e}")
        
        self.stats["logs_created"] += 1
    
    # Convenience methods
    
    async def debug(self, message: str, **fields):
        await self.log(LogLevel.DEBUG, message, fields=fields)
    
    async def info(self, message: str, **fields):
        await self.log(LogLevel.INFO, message, fields=fields)
    
    async def warning(self, message: str, **fields):
        await self.log(LogLevel.WARNING, message, fields=fields)
    
    async def error(self, message: str, exception: Optional[Exception] = None, **fields):
        await self.log(LogLevel.ERROR, message, fields=fields, exception=exception)
    
    async def critical(self, message: str, exception: Optional[Exception] = None, **fields):
        await self.log(LogLevel.CRITICAL, message, fields=fields, exception=exception)
    
    # Context managers
    
    @asynccontextmanager
    async def context_manager(self, context: LogContext):
        """Context manager for logging context"""
        token = self.set_context(context)
        try:
            yield self
        finally:
            self.clear_context(token)
    
    @asynccontextmanager
    async def correlation_context(self, correlation_id: Optional[str] = None, **metadata):
        """Context manager for correlation ID context"""
        context = self.create_context(correlation_id=correlation_id, **metadata)
        async with self.context_manager(context):
            yield context
    
    # Analytics methods
    
    async def get_correlation_logs(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a correlation ID"""
        if not self.log_storage:
            return []
        
        return await self.log_storage.get_logs_by_correlation_id(correlation_id)
    
    async def get_recent_logs(self, limit: int = 100, 
                            level: Optional[LogLevel] = None) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        if not self.log_storage:
            return []
        
        return await self.log_storage.get_recent_logs(limit, level)
    
    async def get_error_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        error_logs = await self.get_recent_logs(1000, LogLevel.ERROR)
        critical_logs = await self.get_recent_logs(1000, LogLevel.CRITICAL)
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Filter by time
        recent_errors = [
            log for log in error_logs
            if datetime.fromisoformat(log["timestamp"]) >= cutoff_time
        ]
        recent_critical = [
            log for log in critical_logs
            if datetime.fromisoformat(log["timestamp"]) >= cutoff_time
        ]
        
        # Analyze errors
        error_types = defaultdict(int)
        for log in recent_errors + recent_critical:
            if log.get("exception_info"):
                error_type = log["exception_info"].get("type", "Unknown")
                error_types[error_type] += 1
        
        return {
            "time_period_hours": hours,
            "total_errors": len(recent_errors),
            "total_critical": len(recent_critical),
            "error_types": dict(error_types),
            "error_rate_per_hour": (len(recent_errors) + len(recent_critical)) / hours
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging system statistics"""
        return {
            "logging_enabled": True,
            "service_name": self.config.service_name,
            "log_level": self.config.log_level.value,
            "log_format": self.config.log_format.value,
            "correlation_tracking": self.config.enable_correlation_ids,
            "trace_integration": self.config.enable_trace_integration,
            "redis_storage": self.config.store_in_redis,
            "statistics": self.stats.copy()
        }


# Global structured logger instance
structured_logger: Optional[StructuredLogger] = None


def initialize_logging(config: LoggingConfig, redis_client: Optional[aioredis.Redis] = None):
    """Initialize the structured logging system"""
    global structured_logger
    
    structured_logger = StructuredLogger(config, redis_client)
    logger.info("✅ Structured logging system initialized")


def get_logger() -> StructuredLogger:
    """Get the global structured logger instance"""
    if structured_logger is None:
        raise RuntimeError("Structured logging not initialized")
    return structured_logger


# Convenience functions

async def debug(message: str, **fields):
    """Log debug message"""
    if structured_logger:
        await structured_logger.debug(message, **fields)


async def info(message: str, **fields):
    """Log info message"""
    if structured_logger:
        await structured_logger.info(message, **fields)


async def warning(message: str, **fields):
    """Log warning message"""
    if structured_logger:
        await structured_logger.warning(message, **fields)


async def error(message: str, exception: Optional[Exception] = None, **fields):
    """Log error message"""
    if structured_logger:
        await structured_logger.error(message, exception=exception, **fields)


async def critical(message: str, exception: Optional[Exception] = None, **fields):
    """Log critical message"""
    if structured_logger:
        await structured_logger.critical(message, exception=exception, **fields)


@asynccontextmanager
async def correlation_context(correlation_id: Optional[str] = None, **metadata):
    """Context manager for correlation tracking"""
    if structured_logger:
        async with structured_logger.correlation_context(correlation_id, **metadata) as context:
            yield context
    else:
        # Fallback - create minimal context
        from dataclasses import dataclass
        
        @dataclass
        class MinimalContext:
            correlation_id: str = correlation_id or str(uuid.uuid4())
        
        yield MinimalContext()


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    if structured_logger:
        context = structured_logger.get_context()
        return context.correlation_id if context else None
    return None


# Decorator for automatic correlation tracking
def with_correlation(correlation_id: Optional[str] = None):
    """Decorator to automatically add correlation context"""
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with correlation_context(correlation_id) as context:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # For sync functions, just call directly
                # (correlation context is async-only)
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


# Decorator for automatic error logging
def log_errors(reraise: bool = True):
    """Decorator to automatically log exceptions"""
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    await error(
                        f"Exception in {func.__name__}",
                        exception=e,
                        function=func.__name__,
                        args=str(args),
                        kwargs=str(kwargs)
                    )
                    if reraise:
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # For sync functions, we can't use async logging
                    # Fall back to standard logging
                    import logging
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                    if reraise:
                        raise
            return sync_wrapper
    
    return decorator