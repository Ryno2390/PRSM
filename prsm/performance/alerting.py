"""
PRSM Advanced Alerting and Notification System
Comprehensive alerting system with multiple notification channels, intelligent rule engine, and escalation policies
"""

from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import time
import logging
import smtplib
import ssl
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from collections import defaultdict, deque
import redis.asyncio as aioredis
from contextlib import asynccontextmanager

# HTTP client for webhooks
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Slack SDK
try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

# Twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# PagerDuty
try:
    import pypd
    PAGERDUTY_AVAILABLE = True
except ImportError:
    PAGERDUTY_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertState(Enum):
    """Alert states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    MUTED = "muted"


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    CONSOLE = "console"


class ConditionOperator(Enum):
    """Condition operators for alert rules"""
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX_MATCH = "regex"


@dataclass
class AlertCondition:
    """Single alert condition"""
    metric_name: str
    operator: ConditionOperator
    threshold: Union[float, str]
    duration_minutes: int = 1
    aggregation: str = "avg"  # avg, max, min, sum, count


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    conditions: List[AlertCondition]
    severity: AlertSeverity
    enabled: bool = True
    
    # Evaluation settings
    evaluation_interval: int = 60  # seconds
    for_duration: int = 300  # seconds (5 minutes)
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_template: Optional[str] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Alert:
    """Active alert instance"""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    
    # Alert details
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    last_notification_at: Optional[datetime] = None
    
    # Metadata
    notification_count: int = 0
    escalation_level: int = 0


@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    channel: NotificationChannel
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    rate_limit: Optional[int] = None  # Max notifications per hour
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds


@dataclass
class EscalationPolicy:
    """Escalation policy for alerts"""
    policy_id: str
    name: str
    rules: List[Dict[str, Any]]  # List of escalation steps
    enabled: bool = True


@dataclass
class AlertingConfig:
    """Alerting system configuration"""
    service_name: str = "prsm-api"
    
    # Storage settings
    store_alerts_in_redis: bool = True
    alert_retention_hours: int = 168  # 7 days
    max_active_alerts: int = 1000
    
    # Evaluation settings
    default_evaluation_interval: int = 60  # seconds
    max_evaluation_workers: int = 10
    
    # Notification settings
    default_notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_rate_limit: int = 100  # per hour
    
    # Email configuration
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    
    # Slack configuration
    slack_bot_token: Optional[str] = None
    slack_default_channel: Optional[str] = None
    
    # Twilio configuration
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    
    # PagerDuty configuration
    pagerduty_api_key: Optional[str] = None
    pagerduty_service_key: Optional[str] = None


class AlertStorage:
    """Redis-based storage for alerts and rules"""
    
    def __init__(self, redis_client: aioredis.Redis, retention_hours: int = 168):
        self.redis = redis_client
        self.retention_seconds = retention_hours * 3600
    
    async def store_alert(self, alert: Alert):
        """Store alert in Redis"""
        try:
            alert_key = f"alert:{alert.alert_id}"
            alert_data = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "state": alert.state.value,
                "message": alert.message,
                "details": alert.details,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "started_at": alert.started_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "last_notification_at": alert.last_notification_at.isoformat() if alert.last_notification_at else None,
                "notification_count": alert.notification_count,
                "escalation_level": alert.escalation_level
            }
            
            await self.redis.setex(alert_key, self.retention_seconds, json.dumps(alert_data))
            
            # Add to active alerts if active
            if alert.state == AlertState.ACTIVE:
                await self.redis.sadd("alerts:active", alert.alert_id)
            else:
                await self.redis.srem("alerts:active", alert.alert_id)
            
            # Add to severity index
            await self.redis.sadd(f"alerts:severity:{alert.severity.value}", alert.alert_id)
            
            # Add to rule index
            await self.redis.sadd(f"alerts:rule:{alert.rule_id}", alert.alert_id)
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Retrieve alert from Redis"""
        try:
            alert_data = await self.redis.get(f"alert:{alert_id}")
            if alert_data:
                data = json.loads(alert_data)
                return Alert(
                    alert_id=data["alert_id"],
                    rule_id=data["rule_id"],
                    rule_name=data["rule_name"],
                    severity=AlertSeverity(data["severity"]),
                    state=AlertState(data["state"]),
                    message=data["message"],
                    details=data.get("details", {}),
                    labels=data.get("labels", {}),
                    annotations=data.get("annotations", {}),
                    started_at=datetime.fromisoformat(data["started_at"]),
                    resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
                    acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None,
                    last_notification_at=datetime.fromisoformat(data["last_notification_at"]) if data.get("last_notification_at") else None,
                    notification_count=data.get("notification_count", 0),
                    escalation_level=data.get("escalation_level", 0)
                )
        except Exception as e:
            logger.error(f"Error retrieving alert {alert_id}: {e}")
        
        return None
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        try:
            alert_ids = await self.redis.smembers("alerts:active")
            alerts = []
            
            for alert_id in alert_ids:
                alert = await self.get_alert(alert_id.decode())
                if alert:
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error retrieving active alerts: {e}")
            return []
    
    async def store_rule(self, rule: AlertRule):
        """Store alert rule"""
        try:
            rule_key = f"rule:{rule.rule_id}"
            rule_data = {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "conditions": [
                    {
                        "metric_name": c.metric_name,
                        "operator": c.operator.value,
                        "threshold": c.threshold,
                        "duration_minutes": c.duration_minutes,
                        "aggregation": c.aggregation
                    }
                    for c in rule.conditions
                ],
                "severity": rule.severity.value,
                "enabled": rule.enabled,
                "evaluation_interval": rule.evaluation_interval,
                "for_duration": rule.for_duration,
                "notification_channels": [ch.value for ch in rule.notification_channels],
                "notification_template": rule.notification_template,
                "tags": rule.tags,
                "created_at": rule.created_at.isoformat(),
                "updated_at": rule.updated_at.isoformat()
            }
            
            await self.redis.set(rule_key, json.dumps(rule_data))
            
            # Add to rules index
            await self.redis.sadd("rules:all", rule.rule_id)
            if rule.enabled:
                await self.redis.sadd("rules:enabled", rule.rule_id)
            
        except Exception as e:
            logger.error(f"Error storing rule: {e}")
    
    async def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Retrieve alert rule"""
        try:
            rule_data = await self.redis.get(f"rule:{rule_id}")
            if rule_data:
                data = json.loads(rule_data)
                return AlertRule(
                    rule_id=data["rule_id"],
                    name=data["name"],
                    description=data["description"],
                    conditions=[
                        AlertCondition(
                            metric_name=c["metric_name"],
                            operator=ConditionOperator(c["operator"]),
                            threshold=c["threshold"],
                            duration_minutes=c["duration_minutes"],
                            aggregation=c["aggregation"]
                        )
                        for c in data["conditions"]
                    ],
                    severity=AlertSeverity(data["severity"]),
                    enabled=data["enabled"],
                    evaluation_interval=data["evaluation_interval"],
                    for_duration=data["for_duration"],
                    notification_channels=[NotificationChannel(ch) for ch in data["notification_channels"]],
                    notification_template=data.get("notification_template"),
                    tags=data.get("tags", {}),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"])
                )
        except Exception as e:
            logger.error(f"Error retrieving rule {rule_id}: {e}")
        
        return None
    
    async def get_enabled_rules(self) -> List[AlertRule]:
        """Get all enabled alert rules"""
        try:
            rule_ids = await self.redis.smembers("rules:enabled")
            rules = []
            
            for rule_id in rule_ids:
                rule = await self.get_rule(rule_id.decode())
                if rule:
                    rules.append(rule)
            
            return rules
            
        except Exception as e:
            logger.error(f"Error retrieving enabled rules: {e}")
            return []


class NotificationManager:
    """Manager for sending notifications through various channels"""
    
    def __init__(self, config: AlertingConfig):
        self.config = config
        self.notification_history = deque(maxlen=1000)
        self.rate_limits = defaultdict(deque)
        
        # Initialize clients
        self.slack_client = None
        self.twilio_client = None
        self.http_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize notification clients"""
        try:
            # Slack client
            if SLACK_AVAILABLE and self.config.slack_bot_token:
                self.slack_client = AsyncWebClient(token=self.config.slack_bot_token)
            
            # Twilio client
            if TWILIO_AVAILABLE and self.config.twilio_account_sid and self.config.twilio_auth_token:
                self.twilio_client = TwilioClient(
                    self.config.twilio_account_sid,
                    self.config.twilio_auth_token
                )
            
            # HTTP client for webhooks
            if HTTPX_AVAILABLE:
                self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # PagerDuty
            if PAGERDUTY_AVAILABLE and self.config.pagerduty_api_key:
                pypd.api_key = self.config.pagerduty_api_key
            
        except Exception as e:
            logger.error(f"Error initializing notification clients: {e}")
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if notification is within rate limit"""
        now = time.time()
        channel_history = self.rate_limits[channel]
        
        # Remove old entries (older than 1 hour)
        while channel_history and now - channel_history[0] > 3600:
            channel_history.popleft()
        
        # Check if under limit
        if len(channel_history) >= self.config.notification_rate_limit:
            return False
        
        channel_history.append(now)
        return True
    
    async def send_notification(self, 
                              alert: Alert,
                              channel: NotificationChannel,
                              config: Dict[str, Any]) -> bool:
        """Send notification through specified channel"""
        
        if not self._check_rate_limit(channel):
            logger.warning(f"Rate limit exceeded for channel {channel}")
            return False
        
        try:
            success = False
            
            if channel == NotificationChannel.EMAIL:
                success = await self._send_email(alert, config)
            elif channel == NotificationChannel.SLACK:
                success = await self._send_slack(alert, config)
            elif channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook(alert, config)
            elif channel == NotificationChannel.SMS:
                success = await self._send_sms(alert, config)
            elif channel == NotificationChannel.PAGERDUTY:
                success = await self._send_pagerduty(alert, config)
            elif channel == NotificationChannel.CONSOLE:
                success = await self._send_console(alert, config)
            
            # Record notification
            self.notification_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_id": alert.alert_id,
                "channel": channel.value,
                "success": success
            })
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notification via {channel}: {e}")
            return False
    
    async def _send_email(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send email notification"""
        if not self.config.smtp_host or not self.config.smtp_from_email:
            logger.warning("Email configuration incomplete")
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.config.smtp_from_email
            msg['To'] = config.get('to', '')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            # Create email body
            body = self._format_alert_message(alert, 'email')
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls(context=context)
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_slack(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send Slack notification"""
        if not self.slack_client:
            logger.warning("Slack client not configured")
            return False
        
        try:
            channel = config.get('channel', self.config.slack_default_channel)
            if not channel:
                logger.warning("No Slack channel specified")
                return False
            
            # Format message
            message = self._format_alert_message(alert, 'slack')
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.FATAL: "danger"
            }
            
            # Send message
            response = await self.slack_client.chat_postMessage(
                channel=channel,
                text=f"Alert: {alert.rule_name}",
                attachments=[{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"{alert.severity.value.upper()}: {alert.rule_name}",
                    "text": message,
                    "ts": int(alert.started_at.timestamp())
                }]
            )
            
            return response["ok"]
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_webhook(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send webhook notification"""
        if not self.http_client:
            logger.warning("HTTP client not available")
            return False
        
        try:
            url = config.get('url')
            if not url:
                logger.warning("No webhook URL specified")
                return False
            
            # Prepare payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "state": alert.state.value,
                "message": alert.message,
                "details": alert.details,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "started_at": alert.started_at.isoformat(),
                "service": self.config.service_name
            }
            
            # Send webhook
            response = await self.http_client.post(
                url,
                json=payload,
                headers=config.get('headers', {})
            )
            
            return response.status_code < 400
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    async def _send_sms(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send SMS notification"""
        if not self.twilio_client or not self.config.twilio_from_number:
            logger.warning("SMS configuration incomplete")
            return False
        
        try:
            to_number = config.get('to')
            if not to_number:
                logger.warning("No SMS recipient specified")
                return False
            
            # Format message (SMS has character limits)
            message = f"ALERT [{alert.severity.value.upper()}]: {alert.rule_name}\n{alert.message[:100]}"
            
            # Send SMS
            message_instance = self.twilio_client.messages.create(
                body=message,
                from_=self.config.twilio_from_number,
                to=to_number
            )
            
            return message_instance.status != 'failed'
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    async def _send_pagerduty(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send PagerDuty notification"""
        if not PAGERDUTY_AVAILABLE or not self.config.pagerduty_service_key:
            logger.warning("PagerDuty configuration incomplete")
            return False
        
        try:
            # Create PagerDuty incident
            incident = pypd.Incident.create(
                service=self.config.pagerduty_service_key,
                incident_key=alert.alert_id,
                title=f"{alert.severity.value.upper()}: {alert.rule_name}",
                description=alert.message,
                details=alert.details
            )
            
            return incident is not None
            
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
            return False
    
    async def _send_console(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send console notification (logging)"""
        try:
            severity_map = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.FATAL: logging.CRITICAL
            }
            
            log_level = severity_map.get(alert.severity, logging.INFO)
            logger.log(log_level, f"ALERT [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending console notification: {e}")
            return False
    
    def _format_alert_message(self, alert: Alert, format_type: str) -> str:
        """Format alert message for different notification types"""
        
        if format_type == 'email':
            return f"""
            <html>
            <body>
                <h2>Alert: {alert.rule_name}</h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>State:</strong> {alert.state.value}</p>
                <p><strong>Started:</strong> {alert.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                
                {f'<h3>Details:</h3><pre>{json.dumps(alert.details, indent=2)}</pre>' if alert.details else ''}
                
                {f'<h3>Labels:</h3><ul>{"".join(f"<li>{k}: {v}</li>" for k, v in alert.labels.items())}</ul>' if alert.labels else ''}
                
                <p><em>Alert ID: {alert.alert_id}</em></p>
            </body>
            </html>
            """
        
        elif format_type == 'slack':
            details_text = ""
            if alert.details:
                details_text = f"\n*Details:*\n```{json.dumps(alert.details, indent=2)}```"
            
            labels_text = ""
            if alert.labels:
                labels_text = f"\n*Labels:* {', '.join(f'{k}={v}' for k, v in alert.labels.items())}"
            
            return f"""*{alert.message}*
            
*Started:* {alert.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
*State:* {alert.state.value}{details_text}{labels_text}

_Alert ID: {alert.alert_id}_"""
        
        else:
            return f"{alert.severity.value.upper()}: {alert.rule_name} - {alert.message}"
    
    async def close(self):
        """Close notification clients"""
        if self.http_client:
            await self.http_client.aclose()


class AlertEvaluator:
    """Evaluates alert rules against metrics"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def evaluate_rule(self, rule: AlertRule) -> Optional[Dict[str, Any]]:
        """Evaluate a single alert rule"""
        try:
            # Evaluate all conditions
            condition_results = []
            
            for condition in rule.conditions:
                result = await self._evaluate_condition(condition)
                condition_results.append(result)
            
            # Check if all conditions are met (AND logic)
            all_conditions_met = all(condition_results)
            
            if all_conditions_met:
                return {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "conditions_met": True,
                    "condition_results": condition_results
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return None
    
    async def _evaluate_condition(self, condition: AlertCondition) -> bool:
        """Evaluate a single condition"""
        try:
            # Get metric data from Redis
            metric_key = f"metrics:{condition.metric_name}:current"
            metric_data = await self.redis.get(metric_key)
            
            if not metric_data:
                logger.debug(f"No data available for metric {condition.metric_name}")
                return False
            
            data = json.loads(metric_data)
            metric_value = data.get("value", 0)
            
            # Apply threshold comparison
            return self._compare_values(
                metric_value,
                condition.operator,
                condition.threshold
            )
            
        except Exception as e:
            logger.error(f"Error evaluating condition for {condition.metric_name}: {e}")
            return False
    
    def _compare_values(self, value: float, operator: ConditionOperator, threshold: Union[float, str]) -> bool:
        """Compare metric value against threshold"""
        try:
            if operator == ConditionOperator.GREATER_THAN:
                return value > float(threshold)
            elif operator == ConditionOperator.GREATER_EQUAL:
                return value >= float(threshold)
            elif operator == ConditionOperator.LESS_THAN:
                return value < float(threshold)
            elif operator == ConditionOperator.LESS_EQUAL:
                return value <= float(threshold)
            elif operator == ConditionOperator.EQUAL:
                return value == float(threshold)
            elif operator == ConditionOperator.NOT_EQUAL:
                return value != float(threshold)
            elif operator == ConditionOperator.CONTAINS:
                return str(threshold) in str(value)
            elif operator == ConditionOperator.NOT_CONTAINS:
                return str(threshold) not in str(value)
            elif operator == ConditionOperator.REGEX_MATCH:
                import re
                return bool(re.search(str(threshold), str(value)))
            
            return False
            
        except Exception as e:
            logger.error(f"Error comparing values: {e}")
            return False


class AlertingSystem:
    """Comprehensive alerting and notification system"""
    
    def __init__(self, config: AlertingConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        
        # Components
        self.storage = AlertStorage(redis_client, config.alert_retention_hours)
        self.notification_manager = NotificationManager(config)
        self.evaluator = AlertEvaluator(redis_client)
        
        # State
        self.active_alerts: Dict[str, Alert] = {}
        self.evaluation_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Statistics
        self.stats = {
            "rules_evaluated": 0,
            "alerts_fired": 0,
            "alerts_resolved": 0,
            "notifications_sent": 0,
            "notification_failures": 0,
            "evaluation_errors": 0
        }
    
    async def start(self):
        """Start the alerting system"""
        if self.running:
            logger.warning("Alerting system is already running")
            return
        
        self.running = True
        
        # Load existing alerts
        await self._load_active_alerts()
        
        # Start rule evaluation
        await self._start_rule_evaluation()
        
        logger.info("✅ Alerting system started")
    
    async def stop(self):
        """Stop the alerting system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop evaluation tasks
        for task in self.evaluation_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.evaluation_tasks:
            await asyncio.gather(*self.evaluation_tasks.values(), return_exceptions=True)
        
        self.evaluation_tasks.clear()
        
        # Close notification manager
        await self.notification_manager.close()
        
        logger.info("🛑 Alerting system stopped")
    
    async def _load_active_alerts(self):
        """Load active alerts from storage"""
        try:
            active_alerts = await self.storage.get_active_alerts()
            for alert in active_alerts:
                self.active_alerts[alert.alert_id] = alert
            
            logger.info(f"Loaded {len(active_alerts)} active alerts")
            
        except Exception as e:
            logger.error(f"Error loading active alerts: {e}")
    
    async def _start_rule_evaluation(self):
        """Start evaluation tasks for all enabled rules"""
        try:
            enabled_rules = await self.storage.get_enabled_rules()
            
            for rule in enabled_rules:
                if rule.rule_id not in self.evaluation_tasks:
                    task = asyncio.create_task(self._rule_evaluation_loop(rule))
                    self.evaluation_tasks[rule.rule_id] = task
            
            logger.info(f"Started evaluation for {len(enabled_rules)} rules")
            
        except Exception as e:
            logger.error(f"Error starting rule evaluation: {e}")
    
    async def _rule_evaluation_loop(self, rule: AlertRule):
        """Evaluation loop for a single rule"""
        while self.running:
            try:
                # Evaluate rule
                result = await self.evaluator.evaluate_rule(rule)
                self.stats["rules_evaluated"] += 1
                
                if result and result["conditions_met"]:
                    # Rule triggered - create or update alert
                    await self._handle_triggered_rule(rule, result)
                else:
                    # Rule not triggered - check if we need to resolve any alerts
                    await self._handle_resolved_rule(rule)
                
                # Wait for next evaluation
                await asyncio.sleep(rule.evaluation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop for rule {rule.rule_id}: {e}")
                self.stats["evaluation_errors"] += 1
                await asyncio.sleep(rule.evaluation_interval)
    
    async def _handle_triggered_rule(self, rule: AlertRule, result: Dict[str, Any]):
        """Handle a triggered alert rule"""
        try:
            # Check if alert already exists for this rule
            existing_alert = None
            for alert in self.active_alerts.values():
                if alert.rule_id == rule.rule_id and alert.state == AlertState.ACTIVE:
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Update existing alert
                existing_alert.notification_count += 1
                existing_alert.last_notification_at = datetime.now(timezone.utc)
                await self.storage.store_alert(existing_alert)
            else:
                # Create new alert
                alert_id = f"alert_{rule.rule_id}_{int(time.time())}"
                alert = Alert(
                    alert_id=alert_id,
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    state=AlertState.ACTIVE,
                    message=f"Alert rule '{rule.name}' triggered",
                    details=result.get("condition_results", {}),
                    labels=rule.tags
                )
                
                self.active_alerts[alert_id] = alert
                await self.storage.store_alert(alert)
                
                self.stats["alerts_fired"] += 1
                
                # Send notifications
                await self._send_alert_notifications(alert, rule)
            
        except Exception as e:
            logger.error(f"Error handling triggered rule {rule.rule_id}: {e}")
    
    async def _handle_resolved_rule(self, rule: AlertRule):
        """Handle a resolved alert rule"""
        try:
            # Find active alerts for this rule
            alerts_to_resolve = []
            for alert in self.active_alerts.values():
                if alert.rule_id == rule.rule_id and alert.state == AlertState.ACTIVE:
                    alerts_to_resolve.append(alert)
            
            # Resolve alerts
            for alert in alerts_to_resolve:
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                
                await self.storage.store_alert(alert)
                del self.active_alerts[alert.alert_id]
                
                self.stats["alerts_resolved"] += 1
                
                # Send resolution notification
                await self._send_resolution_notifications(alert, rule)
            
        except Exception as e:
            logger.error(f"Error handling resolved rule {rule.rule_id}: {e}")
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for a triggered alert"""
        try:
            notification_configs = await self._get_notification_configs(rule)
            
            for channel, config in notification_configs.items():
                success = await self.notification_manager.send_notification(alert, channel, config)
                
                if success:
                    self.stats["notifications_sent"] += 1
                else:
                    self.stats["notification_failures"] += 1
            
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
    
    async def _send_resolution_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for a resolved alert"""
        try:
            # Update alert message for resolution
            alert.message = f"Alert rule '{rule.name}' resolved"
            
            notification_configs = await self._get_notification_configs(rule)
            
            for channel, config in notification_configs.items():
                success = await self.notification_manager.send_notification(alert, channel, config)
                
                if success:
                    self.stats["notifications_sent"] += 1
                else:
                    self.stats["notification_failures"] += 1
            
        except Exception as e:
            logger.error(f"Error sending resolution notifications: {e}")
    
    async def _get_notification_configs(self, rule: AlertRule) -> Dict[NotificationChannel, Dict[str, Any]]:
        """Get notification configurations for a rule"""
        configs = {}
        
        for channel in rule.notification_channels:
            if channel == NotificationChannel.EMAIL:
                configs[channel] = {"to": "admin@example.com"}  # Would be configurable
            elif channel == NotificationChannel.SLACK:
                configs[channel] = {"channel": "#alerts"}  # Would be configurable
            elif channel == NotificationChannel.WEBHOOK:
                configs[channel] = {"url": "https://hooks.example.com/alert"}  # Would be configurable
            elif channel == NotificationChannel.SMS:
                configs[channel] = {"to": "+1234567890"}  # Would be configurable
            elif channel == NotificationChannel.PAGERDUTY:
                configs[channel] = {}
            elif channel == NotificationChannel.CONSOLE:
                configs[channel] = {}
        
        return configs
    
    # Public API methods
    
    async def create_rule(self, rule: AlertRule) -> bool:
        """Create a new alert rule"""
        try:
            await self.storage.store_rule(rule)
            
            # Start evaluation if system is running
            if self.running and rule.enabled:
                task = asyncio.create_task(self._rule_evaluation_loop(rule))
                self.evaluation_tasks[rule.rule_id] = task
            
            logger.info(f"Created alert rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating rule: {e}")
            return False
    
    async def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.annotations["acknowledged_by"] = user
                
                await self.storage.store_alert(alert)
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Manually resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                alert.annotations["resolved_by"] = user
                
                await self.storage.store_alert(alert)
                del self.active_alerts[alert_id]
                
                self.stats["alerts_resolved"] += 1
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary"""
        try:
            # Count alerts by severity
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            # Get recent notification history
            recent_notifications = list(self.notification_manager.notification_history)[-10:]
            
            return {
                "system_status": "running" if self.running else "stopped",
                "active_alerts": len(self.active_alerts),
                "alerts_by_severity": dict(severity_counts),
                "total_rules": len(self.evaluation_tasks),
                "statistics": self.stats.copy(),
                "recent_notifications": recent_notifications,
                "notification_channels": [ch.value for ch in NotificationChannel]
            }
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {"error": str(e)}


# Global alerting system instance
alerting_system: Optional[AlertingSystem] = None


def initialize_alerting(config: AlertingConfig, redis_client: aioredis.Redis):
    """Initialize the alerting system"""
    global alerting_system
    
    alerting_system = AlertingSystem(config, redis_client)
    logger.info("✅ Alerting system initialized")


def get_alerting_system() -> AlertingSystem:
    """Get the global alerting system instance"""
    if alerting_system is None:
        raise RuntimeError("Alerting system not initialized")
    return alerting_system


async def start_alerting():
    """Start alerting system"""
    if alerting_system:
        await alerting_system.start()


async def stop_alerting():
    """Stop alerting system"""
    if alerting_system:
        await alerting_system.stop()


# Convenience functions for creating common alert rules

def create_cpu_alert_rule(threshold: float = 80.0, severity: AlertSeverity = AlertSeverity.WARNING) -> AlertRule:
    """Create CPU usage alert rule"""
    return AlertRule(
        rule_id="cpu_usage_high",
        name="High CPU Usage",
        description=f"CPU usage above {threshold}%",
        conditions=[
            AlertCondition(
                metric_name="system_cpu_usage_percent",
                operator=ConditionOperator.GREATER_THAN,
                threshold=threshold,
                duration_minutes=5
            )
        ],
        severity=severity,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL]
    )


def create_memory_alert_rule(threshold: float = 85.0, severity: AlertSeverity = AlertSeverity.WARNING) -> AlertRule:
    """Create memory usage alert rule"""
    return AlertRule(
        rule_id="memory_usage_high",
        name="High Memory Usage",
        description=f"Memory usage above {threshold}%",
        conditions=[
            AlertCondition(
                metric_name="system_memory_usage_percent",
                operator=ConditionOperator.GREATER_THAN,
                threshold=threshold,
                duration_minutes=5
            )
        ],
        severity=severity,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL]
    )


def create_disk_alert_rule(threshold: float = 90.0, severity: AlertSeverity = AlertSeverity.ERROR) -> AlertRule:
    """Create disk usage alert rule"""
    return AlertRule(
        rule_id="disk_usage_high",
        name="High Disk Usage",
        description=f"Disk usage above {threshold}%",
        conditions=[
            AlertCondition(
                metric_name="system_disk_usage_percent",
                operator=ConditionOperator.GREATER_THAN,
                threshold=threshold,
                duration_minutes=2
            )
        ],
        severity=severity,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL, NotificationChannel.SLACK]
    )


def create_error_rate_alert_rule(threshold: float = 5.0, severity: AlertSeverity = AlertSeverity.ERROR) -> AlertRule:
    """Create error rate alert rule"""
    return AlertRule(
        rule_id="error_rate_high",
        name="High Error Rate",
        description=f"Error rate above {threshold}%",
        conditions=[
            AlertCondition(
                metric_name="http_error_rate_percent",
                operator=ConditionOperator.GREATER_THAN,
                threshold=threshold,
                duration_minutes=3
            )
        ],
        severity=severity,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY]
    )