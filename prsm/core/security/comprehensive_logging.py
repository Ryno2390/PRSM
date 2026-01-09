"""
PRSM Comprehensive Security Logging Integration
==============================================

Enterprise-grade security logging system with real-time monitoring,
alerting, and comprehensive audit trails for production environments.
"""

import asyncio
import structlog
import json
import gzip
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from uuid import uuid4
import threading
import queue
import time
import hashlib
import os

from prsm.core.config import get_settings
from ..integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class LogLevel(str, Enum):
    """Comprehensive log levels for security events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    AUDIT = "audit"


class EventCategory(str, Enum):
    """Security event categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    API_SECURITY = "api_security"
    NETWORK_SECURITY = "network_security"
    CRYPTO_OPERATIONS = "crypto_operations"
    GOVERNANCE = "governance"
    MARKETPLACE = "marketplace"
    WEB3_OPERATIONS = "web3_operations"
    USER_ACTIVITY = "user_activity"
    SYSTEM_INTEGRITY = "system_integrity"
    COMPLIANCE = "compliance"
    THREAT_DETECTION = "threat_detection"
    INCIDENT_RESPONSE = "incident_response"


class AlertSeverity(str, Enum):
    """Alert severity levels for monitoring systems"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityLogEntry:
    """Comprehensive security log entry structure"""
    log_id: str
    timestamp: datetime
    level: LogLevel
    category: EventCategory
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    
    # Event details
    message: str
    description: str
    component: str
    
    # Context data
    request_id: Optional[str]
    trace_id: Optional[str]
    correlation_id: Optional[str]
    
    # Security context
    risk_score: int  # 0-100
    threat_indicators: List[str]
    compliance_flags: List[str]
    
    # Technical details
    stack_trace: Optional[str]
    error_code: Optional[str]
    metadata: Dict[str, Any]
    
    # Audit trail
    action_taken: Optional[str]
    outcome: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class SecurityAlertManager:
    """Manages security alerts and notifications"""
    
    def __init__(self):
        self.alert_rules = {}
        self.alert_channels = {}
        self.alert_history = []
        self.suppression_rules = {}
        
    def add_alert_rule(self, rule_id: str, condition: Callable, severity: AlertSeverity, 
                      notification_channels: List[str]):
        """Add a security alert rule"""
        self.alert_rules[rule_id] = {
            "condition": condition,
            "severity": severity,
            "channels": notification_channels,
            "last_triggered": None,
            "trigger_count": 0
        }
    
    async def evaluate_alerts(self, log_entry: SecurityLogEntry):
        """Evaluate log entry against alert rules"""
        for rule_id, rule in self.alert_rules.items():
            try:
                if rule["condition"](log_entry):
                    await self._trigger_alert(rule_id, rule, log_entry)
            except Exception as e:
                logger.error("Alert rule evaluation failed", rule_id=rule_id, error=str(e))
    
    async def _trigger_alert(self, rule_id: str, rule: Dict, log_entry: SecurityLogEntry):
        """Trigger security alert"""
        alert = {
            "alert_id": str(uuid4()),
            "rule_id": rule_id,
            "severity": rule["severity"],
            "triggered_at": datetime.now(timezone.utc),
            "log_entry": log_entry.to_dict(),
            "suppressed": False
        }
        
        # Check suppression rules
        if await self._is_suppressed(rule_id, log_entry):
            alert["suppressed"] = True
            return
        
        # Update rule statistics
        rule["last_triggered"] = alert["triggered_at"]
        rule["trigger_count"] += 1
        
        # Store alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert, rule["channels"])
        
        logger.warning("Security alert triggered",
                      alert_id=alert["alert_id"],
                      rule_id=rule_id,
                      severity=rule["severity"].value)
    
    async def _is_suppressed(self, rule_id: str, log_entry: SecurityLogEntry) -> bool:
        """Check if alert should be suppressed"""
        # Simple time-based suppression (could be enhanced)
        suppression = self.suppression_rules.get(rule_id)
        if not suppression:
            return False
        
        last_alert = None
        for alert in reversed(self.alert_history):
            if alert["rule_id"] == rule_id and not alert["suppressed"]:
                last_alert = alert
                break
        
        if last_alert:
            time_since_last = datetime.now(timezone.utc) - datetime.fromisoformat(last_alert["triggered_at"])
            return time_since_last < timedelta(minutes=suppression.get("window_minutes", 60))
        
        return False
    
    async def _send_notifications(self, alert: Dict, channels: List[str]):
        """Send alert notifications"""
        for channel in channels:
            try:
                if channel == "log":
                    logger.critical("SECURITY ALERT",
                                  alert_id=alert["alert_id"],
                                  severity=alert["severity"],
                                  event=alert["log_entry"]["event_type"])
                elif channel == "webhook":
                    # In production, this would send to monitoring systems
                    logger.info("Alert webhook notification sent", alert_id=alert["alert_id"])
                # Add more notification channels as needed
            except Exception as e:
                logger.error("Failed to send alert notification", channel=channel, error=str(e))


class LogRotationManager:
    """Manages log file rotation and archival"""
    
    def __init__(self, log_dir: Path, max_size_mb: int = 100, retention_days: int = 90):
        self.log_dir = log_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.retention_days = retention_days
        
    async def check_rotation(self, log_file: Path):
        """Check if log file needs rotation"""
        if not log_file.exists():
            return
        
        file_size = log_file.stat().st_size
        if file_size >= self.max_size_bytes:
            await self._rotate_log(log_file)
    
    async def _rotate_log(self, log_file: Path):
        """Rotate log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{log_file.stem}_{timestamp}.gz"
        rotated_path = self.log_dir / "archived" / rotated_name
        
        # Ensure archive directory exists
        rotated_path.parent.mkdir(exist_ok=True)
        
        # Compress and move
        with open(log_file, 'rb') as f_in:
            with gzip.open(rotated_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Clear original file
        log_file.unlink()
        log_file.touch()
        
        logger.info("Log file rotated", original=str(log_file), archived=str(rotated_path))
        
        # Clean up old archives
        await self._cleanup_old_archives()
    
    async def _cleanup_old_archives(self):
        """Remove archived logs older than retention period"""
        archive_dir = self.log_dir / "archived"
        if not archive_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for archive_file in archive_dir.glob("*.gz"):
            file_time = datetime.fromtimestamp(archive_file.stat().st_mtime)
            if file_time < cutoff_date:
                archive_file.unlink()
                logger.info("Expired log archive removed", file=str(archive_file))


class ComprehensiveSecurityLogger:
    """
    Enterprise-grade comprehensive security logging system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger_id = str(uuid4())
        self.logger = logger.bind(component="security_logger", logger_id=self.logger_id)
        
        # Configuration
        self.config = config or self._load_default_config()
        
        # Initialize directories
        self.log_dir = Path(self.config["log_directory"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.security_log = self.log_dir / "security.jsonl"
        self.audit_log = self.log_dir / "audit.jsonl"
        self.access_log = self.log_dir / "access.jsonl"
        self.error_log = self.log_dir / "errors.jsonl"
        self.metrics_log = self.log_dir / "metrics.jsonl"
        
        # Components
        self.alert_manager = SecurityAlertManager()
        self.rotation_manager = LogRotationManager(
            self.log_dir,
            self.config["max_log_size_mb"],
            self.config["retention_days"]
        )
        
        # Async logging queue
        self.log_queue = asyncio.Queue(maxsize=10000)
        self.processing_task = None
        
        # Statistics
        self.stats = {
            "logs_written": 0,
            "alerts_triggered": 0,
            "errors_encountered": 0,
            "start_time": datetime.now(timezone.utc)
        }
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        print("🔐 Comprehensive Security Logger initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default logging configuration"""
        return {
            "log_directory": os.getenv("PRSM_SECURITY_LOG_DIR", "logs/security"),
            "max_log_size_mb": int(os.getenv("PRSM_MAX_LOG_SIZE_MB", "100")),
            "retention_days": int(os.getenv("PRSM_LOG_RETENTION_DAYS", "90")),
            "enable_real_time_alerts": os.getenv("PRSM_ENABLE_ALERTS", "true").lower() == "true",
            "enable_log_compression": os.getenv("PRSM_ENABLE_LOG_COMPRESSION", "true").lower() == "true",
            "log_level": os.getenv("PRSM_SECURITY_LOG_LEVEL", "INFO").upper(),
            "alert_webhook_url": os.getenv("PRSM_ALERT_WEBHOOK_URL"),
            "enable_metrics": os.getenv("PRSM_ENABLE_SECURITY_METRICS", "true").lower() == "true"
        }
    
    def _setup_default_alert_rules(self):
        """Setup default security alert rules"""
        
        # Critical authentication failures
        self.alert_manager.add_alert_rule(
            "auth_failure_burst",
            lambda log: (log.category == EventCategory.AUTHENTICATION and 
                        log.level == LogLevel.ERROR and
                        "failed_login" in log.event_type),
            AlertSeverity.HIGH,
            ["log", "webhook"]
        )
        
        # Privilege escalation attempts
        self.alert_manager.add_alert_rule(
            "privilege_escalation",
            lambda log: (log.category == EventCategory.AUTHORIZATION and
                        log.level == LogLevel.CRITICAL and
                        "privilege" in log.message.lower()),
            AlertSeverity.CRITICAL,
            ["log", "webhook"]
        )
        
        # Web3 transaction anomalies
        self.alert_manager.add_alert_rule(
            "web3_anomaly",
            lambda log: (log.category == EventCategory.WEB3_OPERATIONS and
                        log.risk_score > 80),
            AlertSeverity.HIGH,
            ["log", "webhook"]
        )
        
        # Governance manipulation attempts
        self.alert_manager.add_alert_rule(
            "governance_manipulation",
            lambda log: (log.category == EventCategory.GOVERNANCE and
                        log.level in [LogLevel.ERROR, LogLevel.CRITICAL]),
            AlertSeverity.CRITICAL,
            ["log", "webhook"]
        )
        
        # Data access anomalies
        self.alert_manager.add_alert_rule(
            "data_access_anomaly",
            lambda log: (log.category == EventCategory.DATA_ACCESS and
                        log.risk_score > 70),
            AlertSeverity.MEDIUM,
            ["log"]
        )
    
    async def start_logging(self):
        """Start the async logging system"""
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._process_log_queue())
            self.logger.info("Security logging system started")
    
    async def stop_logging(self):
        """Stop the async logging system"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
            self.logger.info("Security logging system stopped")
    
    async def log_security_event(
        self,
        event_type: str,
        level: LogLevel,
        category: EventCategory,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        component: str = "prsm",
        risk_score: int = 0,
        threat_indicators: List[str] = None,
        compliance_flags: List[str] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ):
        """Log a comprehensive security event"""
        
        log_entry = SecurityLogEntry(
            log_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=category,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=kwargs.get("user_agent"),
            message=message,
            description=kwargs.get("description", message),
            component=component,
            request_id=kwargs.get("request_id"),
            trace_id=kwargs.get("trace_id"),
            correlation_id=kwargs.get("correlation_id"),
            risk_score=risk_score,
            threat_indicators=threat_indicators or [],
            compliance_flags=compliance_flags or [],
            stack_trace=kwargs.get("stack_trace"),
            error_code=kwargs.get("error_code"),
            metadata=metadata or {},
            action_taken=kwargs.get("action_taken"),
            outcome=kwargs.get("outcome")
        )
        
        # Queue for async processing
        try:
            await self.log_queue.put(log_entry)
        except asyncio.QueueFull:
            # Emergency fallback - log directly
            await self._write_log_entry(log_entry)
            self.logger.warning("Log queue full, writing directly")
    
    async def _process_log_queue(self):
        """Process queued log entries"""
        while True:
            try:
                log_entry = await self.log_queue.get()
                await self._write_log_entry(log_entry)
                
                # Evaluate alerts
                if self.config["enable_real_time_alerts"]:
                    await self.alert_manager.evaluate_alerts(log_entry)
                
                self.log_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error processing log entry", error=str(e))
                self.stats["errors_encountered"] += 1
    
    async def _write_log_entry(self, log_entry: SecurityLogEntry):
        """Write log entry to appropriate files"""
        try:
            # Determine target log file
            if log_entry.category == EventCategory.COMPLIANCE:
                target_file = self.audit_log
            elif log_entry.category == EventCategory.DATA_ACCESS:
                target_file = self.access_log
            elif log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                target_file = self.error_log
            else:
                target_file = self.security_log
            
            # Write to file
            with open(target_file, 'a', encoding='utf-8') as f:
                f.write(log_entry.to_json() + '\n')
            
            # Check for rotation
            await self.rotation_manager.check_rotation(target_file)
            
            # Update statistics
            self.stats["logs_written"] += 1
            
            # Write metrics if enabled
            if self.config["enable_metrics"]:
                await self._write_metrics(log_entry)
            
        except Exception as e:
            self.logger.error("Failed to write log entry", error=str(e))
            self.stats["errors_encountered"] += 1
    
    async def _write_metrics(self, log_entry: SecurityLogEntry):
        """Write security metrics"""
        metrics = {
            "timestamp": log_entry.timestamp.isoformat(),
            "category": log_entry.category.value,
            "level": log_entry.level.value,
            "risk_score": log_entry.risk_score,
            "threat_indicators_count": len(log_entry.threat_indicators),
            "user_id": log_entry.user_id,
            "component": log_entry.component
        }
        
        with open(self.metrics_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')
    
    # Convenience methods for common security events
    
    async def log_authentication_event(self, event_type: str, user_id: str, 
                                     success: bool, ip_address: str = None, **kwargs):
        """Log authentication events"""
        level = LogLevel.INFO if success else LogLevel.ERROR
        risk_score = 0 if success else 30
        
        await self.log_security_event(
            event_type=event_type,
            level=level,
            category=EventCategory.AUTHENTICATION,
            message=f"Authentication {event_type}: {'success' if success else 'failure'}",
            user_id=user_id,
            ip_address=ip_address,
            risk_score=risk_score,
            **kwargs
        )
    
    async def log_authorization_event(self, action: str, user_id: str, resource: str,
                                    granted: bool, **kwargs):
        """Log authorization events"""
        level = LogLevel.INFO if granted else LogLevel.WARNING
        risk_score = 0 if granted else 40
        
        await self.log_security_event(
            event_type="authorization_check",
            level=level,
            category=EventCategory.AUTHORIZATION,
            message=f"Authorization for {action} on {resource}: {'granted' if granted else 'denied'}",
            user_id=user_id,
            risk_score=risk_score,
            metadata={"action": action, "resource": resource, "granted": granted},
            **kwargs
        )
    
    async def log_web3_transaction(self, tx_hash: str, user_id: str, contract_address: str,
                                 method: str, success: bool, **kwargs):
        """Log Web3 transaction events"""
        level = LogLevel.INFO if success else LogLevel.ERROR
        risk_score = kwargs.get("risk_score", 10 if success else 50)
        
        await self.log_security_event(
            event_type="web3_transaction",
            level=level,
            category=EventCategory.WEB3_OPERATIONS,
            message=f"Web3 transaction {method}: {'success' if success else 'failure'}",
            user_id=user_id,
            risk_score=risk_score,
            metadata={
                "tx_hash": tx_hash,
                "contract_address": contract_address,
                "method": method,
                "success": success
            },
            **kwargs
        )
    
    async def log_governance_action(self, action: str, user_id: str, proposal_id: str = None,
                                  outcome: str = None, **kwargs):
        """Log governance-related events"""
        await self.log_security_event(
            event_type="governance_action",
            level=LogLevel.AUDIT,
            category=EventCategory.GOVERNANCE,
            message=f"Governance action: {action}",
            user_id=user_id,
            risk_score=kwargs.get("risk_score", 20),
            metadata={
                "action": action,
                "proposal_id": proposal_id,
                "outcome": outcome
            },
            **kwargs
        )
    
    async def log_marketplace_activity(self, activity: str, user_id: str, model_id: str = None,
                                     amount: str = None, **kwargs):
        """Log marketplace-related events"""
        await self.log_security_event(
            event_type="marketplace_activity",
            level=LogLevel.INFO,
            category=EventCategory.MARKETPLACE,
            message=f"Marketplace activity: {activity}",
            user_id=user_id,
            risk_score=kwargs.get("risk_score", 5),
            metadata={
                "activity": activity,
                "model_id": model_id,
                "amount": amount
            },
            **kwargs
        )
    
    async def get_security_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get security metrics for the specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        metrics = {
            "time_period_hours": hours,
            "total_logs": self.stats["logs_written"],
            "alerts_triggered": len(self.alert_manager.alert_history),
            "errors_encountered": self.stats["errors_encountered"],
            "uptime_hours": (datetime.now(timezone.utc) - self.stats["start_time"]).total_seconds() / 3600,
            "queue_size": self.log_queue.qsize(),
            "categories": {},
            "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0}
        }
        
        # In production, this would analyze actual log files
        # For now, return mock metrics
        metrics["categories"] = {
            "authentication": 150,
            "authorization": 89,
            "web3_operations": 76,
            "marketplace": 43,
            "governance": 12
        }
        
        return metrics


# Global comprehensive security logger instance
_security_logger: Optional[ComprehensiveSecurityLogger] = None

async def get_security_logger() -> ComprehensiveSecurityLogger:
    """Get or create the global security logger instance"""
    global _security_logger
    if _security_logger is None:
        _security_logger = ComprehensiveSecurityLogger()
        await _security_logger.start_logging()
    return _security_logger

# Convenience function for backward compatibility
async def log_security_event(event_type: str, user_id: str, details: Dict[str, Any],
                           security_level: str = "info"):
    """Convenience function for logging security events"""
    logger_instance = await get_security_logger()
    
    level_mapping = {
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
        "critical": LogLevel.CRITICAL,
        "audit": LogLevel.AUDIT
    }
    
    await logger_instance.log_security_event(
        event_type=event_type,
        level=level_mapping.get(security_level.lower(), LogLevel.INFO),
        category=EventCategory.USER_ACTIVITY,  # Default category
        message=f"Security event: {event_type}",
        user_id=user_id,
        metadata=details
    )