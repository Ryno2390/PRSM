"""
Notification System for Scheduled Workflows

📢 WORKFLOW NOTIFICATION SYSTEM:
- Real-time notifications for workflow state changes
- Multi-channel delivery (email, SMS, webhook, in-app)
- Smart notification filtering and aggregation
- Escalation policies for critical failures
- User preference management and quiet hours
- Historical notification tracking and analytics

This module implements comprehensive notification delivery that enables:
1. Instant alerts for workflow completion, failures, and state changes
2. Configurable notification channels and user preferences
3. Smart aggregation to prevent notification spam
4. Escalation workflows for critical system events
5. Analytics and delivery tracking for optimization
6. Integration with external notification services
"""

import asyncio
import json
import hashlib
import time
import smtplib
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from uuid import UUID, uuid4
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
import requests
import aiohttp

from prsm.core.models import PRSMBaseModel, TimestampMixin, AgentType
from prsm.compute.scheduling.workflow_scheduler import ScheduledWorkflow, WorkflowStatus, SchedulingPriority
from prsm.compute.scheduling.progress_tracker import ProgressStatus, ProgressTracker, ProgressAlert

logger = structlog.get_logger(__name__)


class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    EMAIL = "email"                    # Email notifications
    SMS = "sms"                       # SMS text messages
    WEBHOOK = "webhook"               # HTTP webhook calls
    IN_APP = "in_app"                # In-application notifications
    SLACK = "slack"                   # Slack workspace integration
    DISCORD = "discord"               # Discord server integration
    TEAMS = "teams"                   # Microsoft Teams integration
    PUSH = "push"                     # Mobile push notifications


class NotificationType(str, Enum):
    """Types of notifications"""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    RESOURCE_ALLOCATED = "resource_allocated"
    RESOURCE_DEALLOCATED = "resource_deallocated"
    COST_THRESHOLD_EXCEEDED = "cost_threshold_exceeded"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    SYSTEM_MAINTENANCE = "system_maintenance"
    SECURITY_ALERT = "security_alert"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"                       # Non-urgent updates
    NORMAL = "normal"                 # Standard notifications
    HIGH = "high"                     # Important alerts
    URGENT = "urgent"                 # Critical notifications requiring immediate attention
    EMERGENCY = "emergency"           # Emergency alerts with escalation


class NotificationStatus(str, Enum):
    """Notification delivery status"""
    PENDING = "pending"               # Waiting to be sent
    QUEUED = "queued"                # In delivery queue
    SENDING = "sending"               # Currently being sent
    DELIVERED = "delivered"           # Successfully delivered
    FAILED = "failed"                 # Delivery failed
    BOUNCED = "bounced"              # Email bounced
    READ = "read"                    # Notification was read by user
    DISMISSED = "dismissed"          # Notification was dismissed


class EscalationPolicy(PRSMBaseModel):
    """Escalation policy for critical notifications"""
    policy_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = Field(default="")
    
    # Trigger conditions
    applies_to_types: List[NotificationType] = Field(default_factory=list)
    min_priority: NotificationPriority = Field(default=NotificationPriority.HIGH)
    
    # Escalation steps
    escalation_steps: List[Dict[str, Any]] = Field(default_factory=list)  # [{"delay_minutes": 15, "channels": ["email"], "recipients": ["admin@example.com"]}]
    max_escalations: int = Field(default=3)
    
    # Conditions
    business_hours_only: bool = Field(default=False)
    enabled: bool = Field(default=True)


class NotificationPreferences(PRSMBaseModel):
    """User notification preferences"""
    user_id: str
    
    # Channel preferences
    enabled_channels: List[NotificationChannel] = Field(default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.IN_APP])
    
    # Type preferences
    notification_types: Dict[str, bool] = Field(default_factory=dict)  # notification_type -> enabled
    
    # Timing preferences
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    timezone: str = Field(default="UTC")
    
    # Aggregation preferences
    digest_mode: bool = Field(default=False)  # Send digest instead of individual notifications
    digest_frequency: str = Field(default="hourly")  # "immediate", "hourly", "daily", "weekly"
    
    # Priority filtering
    min_priority: NotificationPriority = Field(default=NotificationPriority.LOW)
    
    # Contact information
    email_addresses: List[str] = Field(default_factory=list)
    phone_numbers: List[str] = Field(default_factory=list)
    webhook_urls: List[str] = Field(default_factory=list)
    
    def is_notification_enabled(self, notification_type: NotificationType, channel: NotificationChannel) -> bool:
        """Check if notification type is enabled for user"""
        if channel not in self.enabled_channels:
            return False
        
        return self.notification_types.get(notification_type.value, True)
    
    def is_quiet_hours(self, check_time: Optional[datetime] = None) -> bool:
        """Check if current time is within quiet hours"""
        if not self.quiet_hours_start or not self.quiet_hours_end:
            return False
        
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        
        # Simple implementation - would need proper timezone handling in production
        current_hour = check_time.strftime("%H:%M")
        return self.quiet_hours_start <= current_hour <= self.quiet_hours_end


class NotificationTemplate(PRSMBaseModel):
    """Template for notification messages"""
    template_id: UUID = Field(default_factory=uuid4)
    notification_type: NotificationType
    channel: NotificationChannel
    
    # Template content
    subject_template: str = Field(default="")
    body_template: str = Field(default="")
    html_template: Optional[str] = None
    
    # Metadata
    template_version: str = Field(default="1.0")
    created_by: str = Field(default="system")
    is_active: bool = Field(default=True)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template with context variables"""
        try:
            subject = self.subject_template.format(**context)
            body = self.body_template.format(**context)
            html = self.html_template.format(**context) if self.html_template else None
            
            return {
                "subject": subject,
                "body": body,
                "html": html
            }
        except KeyError as e:
            logger.error("Template rendering failed - missing context variable", template_id=str(self.template_id), missing_var=str(e))
            return {
                "subject": f"Notification: {self.notification_type.value}",
                "body": f"Notification content could not be rendered. Type: {self.notification_type.value}",
                "html": None
            }


class NotificationMessage(PRSMBaseModel):
    """Individual notification message"""
    message_id: UUID = Field(default_factory=uuid4)
    
    # Target information
    user_id: str
    channel: NotificationChannel
    
    # Notification details
    notification_type: NotificationType
    priority: NotificationPriority = Field(default=NotificationPriority.NORMAL)
    
    # Content
    subject: str = Field(default="")
    body: str
    html_content: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    
    # Context
    workflow_id: Optional[UUID] = None
    step_id: Optional[UUID] = None
    related_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Delivery tracking
    status: NotificationStatus = Field(default=NotificationStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_for: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    # Retry information
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    last_error: Optional[str] = None
    
    # Delivery details
    delivery_details: Dict[str, Any] = Field(default_factory=dict)
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries and self.status in [NotificationStatus.FAILED, NotificationStatus.BOUNCED]


class NotificationDigest(PRSMBaseModel):
    """Aggregated notification digest"""
    digest_id: UUID = Field(default_factory=uuid4)
    user_id: str
    
    # Digest period
    start_time: datetime
    end_time: datetime
    frequency: str  # "hourly", "daily", "weekly"
    
    # Aggregated messages
    messages: List[NotificationMessage] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)  # notification_type -> count
    
    # Delivery
    channel: NotificationChannel = Field(default=NotificationChannel.EMAIL)
    status: NotificationStatus = Field(default=NotificationStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NotificationSystem(TimestampMixin):
    """
    Comprehensive Notification System for Scheduled Workflows
    
    Handles multi-channel notification delivery with smart aggregation,
    escalation policies, and user preference management.
    """
    
    def __init__(self, progress_tracker: Optional[ProgressTracker] = None):
        super().__init__()
        
        # Core components
        self.progress_tracker = progress_tracker
        
        # Configuration
        self.user_preferences: Dict[str, NotificationPreferences] = {}
        self.escalation_policies: List[EscalationPolicy] = []
        self.templates: Dict[Tuple[NotificationType, NotificationChannel], NotificationTemplate] = {}
        
        # Message management
        self.pending_messages: deque = deque()
        self.message_history: Dict[UUID, NotificationMessage] = {}
        self.active_escalations: Dict[UUID, Dict[str, Any]] = {}
        
        # Digest management
        self.digest_queues: Dict[str, List[NotificationMessage]] = defaultdict(list)
        self.last_digest_sent: Dict[str, datetime] = {}
        
        # Delivery tracking
        self.delivery_statistics: Dict[str, Any] = defaultdict(int)
        self.channel_health: Dict[NotificationChannel, Dict[str, Any]] = {}
        
        # External service configurations
        self.smtp_config: Dict[str, Any] = {}
        self.webhook_config: Dict[str, Any] = {}
        self.sms_config: Dict[str, Any] = {}
        
        # Background tasks
        self._notification_processor_task: Optional[asyncio.Task] = None
        self._digest_processor_task: Optional[asyncio.Task] = None
        
        self._initialize_default_templates()
        self._start_background_processors()
        
        logger.info("NotificationSystem initialized")
    
    def _initialize_default_templates(self):
        """Initialize default notification templates"""
        # Workflow completion templates
        email_completion_template = NotificationTemplate(
            notification_type=NotificationType.WORKFLOW_COMPLETED,
            channel=NotificationChannel.EMAIL,
            subject_template="✅ Workflow '{workflow_name}' Completed Successfully",
            body_template="""Hello {user_name},

Your workflow '{workflow_name}' has completed successfully!

📊 Execution Summary:
• Workflow ID: {workflow_id}
• Started: {start_time}
• Completed: {end_time}
• Duration: {duration}
• Steps Completed: {completed_steps}
• Total Cost: {total_cost} FTNS

🎯 Results:
{results_summary}

You can view detailed results in your dashboard: {dashboard_url}

Best regards,
PRSM Workflow System""",
            html_template="""<html><body>
<h2>✅ Workflow Completed Successfully</h2>
<p>Hello {user_name},</p>
<p>Your workflow '<strong>{workflow_name}</strong>' has completed successfully!</p>

<h3>📊 Execution Summary:</h3>
<ul>
<li><strong>Workflow ID:</strong> {workflow_id}</li>
<li><strong>Started:</strong> {start_time}</li>
<li><strong>Completed:</strong> {end_time}</li>
<li><strong>Duration:</strong> {duration}</li>
<li><strong>Steps Completed:</strong> {completed_steps}</li>
<li><strong>Total Cost:</strong> {total_cost} FTNS</li>
</ul>

<h3>🎯 Results:</h3>
<p>{results_summary}</p>

<p><a href="{dashboard_url}">View detailed results in your dashboard</a></p>

<p>Best regards,<br>PRSM Workflow System</p>
</body></html>"""
        )
        
        # Workflow failure templates
        email_failure_template = NotificationTemplate(
            notification_type=NotificationType.WORKFLOW_FAILED,
            channel=NotificationChannel.EMAIL,
            subject_template="❌ Workflow '{workflow_name}' Failed",
            body_template="""Hello {user_name},

Unfortunately, your workflow '{workflow_name}' has failed during execution.

🚨 Failure Details:
• Workflow ID: {workflow_id}
• Started: {start_time}
• Failed at: {failure_time}
• Failed Step: {failed_step}
• Error: {error_message}
• Cost Incurred: {cost_incurred} FTNS

🔧 Next Steps:
{next_steps}

You can retry the workflow or contact support for assistance: {support_url}

Best regards,
PRSM Workflow System""",
            html_template="""<html><body>
<h2>❌ Workflow Failed</h2>
<p>Hello {user_name},</p>
<p>Unfortunately, your workflow '<strong>{workflow_name}</strong>' has failed during execution.</p>

<h3>🚨 Failure Details:</h3>
<ul>
<li><strong>Workflow ID:</strong> {workflow_id}</li>
<li><strong>Started:</strong> {start_time}</li>
<li><strong>Failed at:</strong> {failure_time}</li>
<li><strong>Failed Step:</strong> {failed_step}</li>
<li><strong>Error:</strong> {error_message}</li>
<li><strong>Cost Incurred:</strong> {cost_incurred} FTNS</li>
</ul>

<h3>🔧 Next Steps:</h3>
<p>{next_steps}</p>

<p><a href="{support_url}">Contact support for assistance</a></p>

<p>Best regards,<br>PRSM Workflow System</p>
</body></html>"""
        )
        
        # Store templates
        self.templates[(NotificationType.WORKFLOW_COMPLETED, NotificationChannel.EMAIL)] = email_completion_template
        self.templates[(NotificationType.WORKFLOW_FAILED, NotificationChannel.EMAIL)] = email_failure_template
        
        # Add more default templates...
        self._add_slack_templates()
        self._add_webhook_templates()
        self._add_sms_templates()
    
    def _add_slack_templates(self):
        """Add Slack notification templates"""
        slack_completion_template = NotificationTemplate(
            notification_type=NotificationType.WORKFLOW_COMPLETED,
            channel=NotificationChannel.SLACK,
            subject_template="",
            body_template="""✅ *Workflow Completed*

*{workflow_name}* finished successfully!
• Duration: {duration}
• Cost: {total_cost} FTNS
• <{dashboard_url}|View Results>"""
        )
        
        slack_failure_template = NotificationTemplate(
            notification_type=NotificationType.WORKFLOW_FAILED,
            channel=NotificationChannel.SLACK,
            subject_template="",
            body_template="""❌ *Workflow Failed*

*{workflow_name}* failed during execution
• Error: {error_message}
• Failed Step: {failed_step}
• <{support_url}|Get Support>"""
        )
        
        self.templates[(NotificationType.WORKFLOW_COMPLETED, NotificationChannel.SLACK)] = slack_completion_template
        self.templates[(NotificationType.WORKFLOW_FAILED, NotificationChannel.SLACK)] = slack_failure_template
    
    def _add_webhook_templates(self):
        """Add webhook notification templates"""
        webhook_template = NotificationTemplate(
            notification_type=NotificationType.WORKFLOW_COMPLETED,
            channel=NotificationChannel.WEBHOOK,
            subject_template="workflow.completed",
            body_template=json.dumps({
                "event": "workflow.completed",
                "workflow_id": "{workflow_id}",
                "workflow_name": "{workflow_name}",
                "user_id": "{user_id}",
                "status": "completed",
                "timestamp": "{timestamp}",
                "duration_seconds": "{duration_seconds}",
                "total_cost": "{total_cost}",
                "results": "{results_json}"
            })
        )
        
        self.templates[(NotificationType.WORKFLOW_COMPLETED, NotificationChannel.WEBHOOK)] = webhook_template
    
    def _add_sms_templates(self):
        """Add SMS notification templates"""
        sms_completion_template = NotificationTemplate(
            notification_type=NotificationType.WORKFLOW_COMPLETED,
            channel=NotificationChannel.SMS,
            subject_template="",
            body_template="✅ PRSM: Workflow '{workflow_name}' completed in {duration}. Cost: {total_cost} FTNS"
        )
        
        sms_failure_template = NotificationTemplate(
            notification_type=NotificationType.WORKFLOW_FAILED,
            channel=NotificationChannel.SMS,
            subject_template="",
            body_template="❌ PRSM: Workflow '{workflow_name}' failed. Error: {error_message}"
        )
        
        self.templates[(NotificationType.WORKFLOW_COMPLETED, NotificationChannel.SMS)] = sms_completion_template
        self.templates[(NotificationType.WORKFLOW_FAILED, NotificationChannel.SMS)] = sms_failure_template
    
    def _start_background_processors(self):
        """Start background processing tasks"""
        self._notification_processor_task = asyncio.create_task(self._process_notifications())
        self._digest_processor_task = asyncio.create_task(self._process_digests())
    
    async def _process_notifications(self):
        """Background task to process pending notifications"""
        while True:
            try:
                if self.pending_messages:
                    message = self.pending_messages.popleft()
                    await self._deliver_message(message)
                else:
                    await asyncio.sleep(1)  # Short sleep when no messages
            except Exception as e:
                logger.error("Error in notification processor", error=str(e))
                await asyncio.sleep(5)  # Longer sleep on error
    
    async def _process_digests(self):
        """Background task to process digest notifications"""
        while True:
            try:
                await self._send_pending_digests()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error("Error in digest processor", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def send_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        context: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        workflow_id: Optional[UUID] = None,
        step_id: Optional[UUID] = None,
        channels: Optional[List[NotificationChannel]] = None
    ):
        """
        Send notification to user
        
        Args:
            user_id: Target user ID
            notification_type: Type of notification
            context: Template context variables
            priority: Notification priority
            workflow_id: Related workflow ID
            step_id: Related step ID
            channels: Specific channels to use (overrides user preferences)
        """
        try:
            # Get user preferences
            preferences = self.user_preferences.get(user_id)
            if not preferences:
                preferences = self._create_default_preferences(user_id)
            
            # Check if notification should be sent based on priority
            if priority.value < preferences.min_priority.value:
                logger.debug("Notification filtered by priority", user_id=user_id, priority=priority.value)
                return
            
            # Determine channels to use
            target_channels = channels or preferences.enabled_channels
            
            # Check quiet hours for non-urgent notifications
            if priority not in [NotificationPriority.URGENT, NotificationPriority.EMERGENCY]:
                if preferences.is_quiet_hours():
                    # Queue for later delivery
                    scheduled_time = self._calculate_next_active_time(preferences)
                    logger.info("Notification delayed due to quiet hours", user_id=user_id, scheduled_for=scheduled_time)
                else:
                    scheduled_time = None
            else:
                scheduled_time = None
            
            # Create messages for each enabled channel
            for channel in target_channels:
                if not preferences.is_notification_enabled(notification_type, channel):
                    continue
                
                # Check if user should receive digest instead
                if preferences.digest_mode and priority == NotificationPriority.NORMAL:
                    await self._add_to_digest(user_id, notification_type, context, channel, workflow_id, step_id)
                    continue
                
                message = await self._create_notification_message(
                    user_id=user_id,
                    channel=channel,
                    notification_type=notification_type,
                    priority=priority,
                    context=context,
                    workflow_id=workflow_id,
                    step_id=step_id,
                    scheduled_for=scheduled_time
                )
                
                if message:
                    self.pending_messages.append(message)
                    self.message_history[message.message_id] = message
            
            # Check for escalation policies
            await self._check_escalation_policies(notification_type, priority, context, workflow_id)
            
            logger.info(
                "Notification queued",
                user_id=user_id,
                notification_type=notification_type.value,
                priority=priority.value,
                channels=[c.value for c in target_channels]
            )
            
        except Exception as e:
            logger.error("Error sending notification", error=str(e), user_id=user_id, notification_type=notification_type.value)
    
    async def _create_notification_message(
        self,
        user_id: str,
        channel: NotificationChannel,
        notification_type: NotificationType,
        priority: NotificationPriority,
        context: Dict[str, Any],
        workflow_id: Optional[UUID] = None,
        step_id: Optional[UUID] = None,
        scheduled_for: Optional[datetime] = None
    ) -> Optional[NotificationMessage]:
        """Create notification message from template"""
        try:
            # Get template
            template = self.templates.get((notification_type, channel))
            if not template:
                logger.warning("No template found", notification_type=notification_type.value, channel=channel.value)
                return None
            
            # Render template
            rendered = template.render(context)
            
            # Create message
            message = NotificationMessage(
                user_id=user_id,
                channel=channel,
                notification_type=notification_type,
                priority=priority,
                subject=rendered["subject"],
                body=rendered["body"],
                html_content=rendered["html"],
                workflow_id=workflow_id,
                step_id=step_id,
                scheduled_for=scheduled_for,
                related_data=context
            )
            
            return message
            
        except Exception as e:
            logger.error("Error creating notification message", error=str(e))
            return None
    
    async def _deliver_message(self, message: NotificationMessage):
        """Deliver individual notification message"""
        try:
            # Check if scheduled for later
            if message.scheduled_for and datetime.now(timezone.utc) < message.scheduled_for:
                # Put back in queue for later
                self.pending_messages.append(message)
                return
            
            # Update status
            message.status = NotificationStatus.SENDING
            message.sent_at = datetime.now(timezone.utc)
            
            # Deliver based on channel
            if message.channel == NotificationChannel.EMAIL:
                success = await self._deliver_email(message)
            elif message.channel == NotificationChannel.SMS:
                success = await self._deliver_sms(message)
            elif message.channel == NotificationChannel.WEBHOOK:
                success = await self._deliver_webhook(message)
            elif message.channel == NotificationChannel.SLACK:
                success = await self._deliver_slack(message)
            elif message.channel == NotificationChannel.IN_APP:
                success = await self._deliver_in_app(message)
            else:
                logger.warning("Unsupported notification channel", channel=message.channel.value)
                success = False
            
            # Update message status
            if success:
                message.status = NotificationStatus.DELIVERED
                message.delivered_at = datetime.now(timezone.utc)
                self.delivery_statistics["delivered"] += 1
                self.delivery_statistics[f"delivered_{message.channel.value}"] += 1
            else:
                message.status = NotificationStatus.FAILED
                message.retry_count += 1
                self.delivery_statistics["failed"] += 1
                self.delivery_statistics[f"failed_{message.channel.value}"] += 1
                
                # Retry if possible
                if message.can_retry():
                    logger.info("Retrying notification delivery", message_id=str(message.message_id), retry_count=message.retry_count)
                    # Add delay before retry
                    message.scheduled_for = datetime.now(timezone.utc) + timedelta(minutes=5 * message.retry_count)
                    self.pending_messages.append(message)
            
            logger.info(
                "Notification delivery attempted",
                message_id=str(message.message_id),
                channel=message.channel.value,
                status=message.status.value,
                success=success
            )
            
        except Exception as e:
            logger.error("Error delivering notification", error=str(e), message_id=str(message.message_id))
            message.status = NotificationStatus.FAILED
            message.last_error = str(e)
    
    async def _deliver_email(self, message: NotificationMessage) -> bool:
        """Deliver email notification"""
        try:
            # Get user email addresses
            preferences = self.user_preferences.get(message.user_id)
            if not preferences or not preferences.email_addresses:
                logger.warning("No email addresses for user", user_id=message.user_id)
                return False
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message.subject
            msg['From'] = self.smtp_config.get('from_address', 'noreply@prsm.ai')
            msg['To'] = ', '.join(preferences.email_addresses)
            
            # Attach text and HTML parts
            text_part = MIMEText(message.body, 'plain')
            msg.attach(text_part)
            
            if message.html_content:
                html_part = MIMEText(message.html_content, 'html')
                msg.attach(html_part)
            
            # Send email (mock implementation)
            # In production, use proper SMTP server
            logger.info("Email sent", to=preferences.email_addresses, subject=message.subject)
            return True
            
        except Exception as e:
            logger.error("Error sending email", error=str(e))
            return False
    
    async def _deliver_sms(self, message: NotificationMessage) -> bool:
        """Deliver SMS notification"""
        try:
            # Get user phone numbers
            preferences = self.user_preferences.get(message.user_id)
            if not preferences or not preferences.phone_numbers:
                logger.warning("No phone numbers for user", user_id=message.user_id)
                return False
            
            # Mock SMS delivery
            # In production, integrate with SMS service (Twilio, AWS SNS, etc.)
            for phone_number in preferences.phone_numbers:
                logger.info("SMS sent", to=phone_number, message=message.body[:100])
            
            return True
            
        except Exception as e:
            logger.error("Error sending SMS", error=str(e))
            return False
    
    async def _deliver_webhook(self, message: NotificationMessage) -> bool:
        """Deliver webhook notification"""
        try:
            # Get user webhook URLs
            preferences = self.user_preferences.get(message.user_id)
            if not preferences or not preferences.webhook_urls:
                logger.warning("No webhook URLs for user", user_id=message.user_id)
                return False
            
            # Parse JSON body if applicable
            try:
                payload = json.loads(message.body)
            except json.JSONDecodeError:
                payload = {"message": message.body, "subject": message.subject}
            
            # Add metadata
            payload.update({
                "message_id": str(message.message_id),
                "user_id": message.user_id,
                "notification_type": message.notification_type.value,
                "priority": message.priority.value,
                "timestamp": message.created_at.isoformat()
            })
            
            # Send to all webhook URLs
            success = True
            async with aiohttp.ClientSession() as session:
                for webhook_url in preferences.webhook_urls:
                    try:
                        async with session.post(
                            webhook_url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status != 200:
                                logger.warning("Webhook delivery failed", url=webhook_url, status=response.status)
                                success = False
                            else:
                                logger.info("Webhook delivered", url=webhook_url)
                    except Exception as e:
                        logger.error("Webhook delivery error", url=webhook_url, error=str(e))
                        success = False
            
            return success
            
        except Exception as e:
            logger.error("Error sending webhook", error=str(e))
            return False
    
    async def _deliver_slack(self, message: NotificationMessage) -> bool:
        """Deliver Slack notification"""
        try:
            # Mock Slack delivery
            # In production, integrate with Slack API
            logger.info("Slack message sent", user_id=message.user_id, message=message.body)
            return True
            
        except Exception as e:
            logger.error("Error sending Slack message", error=str(e))
            return False
    
    async def _deliver_in_app(self, message: NotificationMessage) -> bool:
        """Deliver in-app notification"""
        try:
            # Store in in-app notification queue
            # In production, this would integrate with the web application's notification system
            logger.info("In-app notification queued", user_id=message.user_id, message=message.body)
            return True
            
        except Exception as e:
            logger.error("Error queueing in-app notification", error=str(e))
            return False
    
    def _create_default_preferences(self, user_id: str) -> NotificationPreferences:
        """Create default notification preferences for user"""
        preferences = NotificationPreferences(
            user_id=user_id,
            enabled_channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            notification_types={
                NotificationType.WORKFLOW_COMPLETED.value: True,
                NotificationType.WORKFLOW_FAILED.value: True,
                NotificationType.WORKFLOW_CANCELLED.value: True,
                NotificationType.COST_THRESHOLD_EXCEEDED.value: True,
                NotificationType.SECURITY_ALERT.value: True
            }
        )
        
        self.user_preferences[user_id] = preferences
        return preferences
    
    def _calculate_next_active_time(self, preferences: NotificationPreferences) -> datetime:
        """Calculate next time outside quiet hours"""
        # Simple implementation - would need proper timezone handling
        if preferences.quiet_hours_end:
            tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            end_hour = int(preferences.quiet_hours_end.split(':')[0])
            return tomorrow.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        else:
            return datetime.now(timezone.utc) + timedelta(hours=1)
    
    async def _add_to_digest(
        self,
        user_id: str,
        notification_type: NotificationType,
        context: Dict[str, Any],
        channel: NotificationChannel,
        workflow_id: Optional[UUID] = None,
        step_id: Optional[UUID] = None
    ):
        """Add notification to digest queue"""
        message = await self._create_notification_message(
            user_id=user_id,
            channel=channel,
            notification_type=notification_type,
            priority=NotificationPriority.NORMAL,
            context=context,
            workflow_id=workflow_id,
            step_id=step_id
        )
        
        if message:
            digest_key = f"{user_id}_{channel.value}"
            self.digest_queues[digest_key].append(message)
    
    async def _send_pending_digests(self):
        """Send pending digest notifications"""
        current_time = datetime.now(timezone.utc)
        
        for digest_key, messages in list(self.digest_queues.items()):
            if not messages:
                continue
            
            user_id, channel_name = digest_key.split('_', 1)
            channel = NotificationChannel(channel_name)
            
            preferences = self.user_preferences.get(user_id)
            if not preferences or not preferences.digest_mode:
                continue
            
            # Check if it's time to send digest
            last_sent = self.last_digest_sent.get(digest_key, current_time - timedelta(hours=25))
            
            should_send = False
            if preferences.digest_frequency == "immediate":
                should_send = True
            elif preferences.digest_frequency == "hourly":
                should_send = (current_time - last_sent) >= timedelta(hours=1)
            elif preferences.digest_frequency == "daily":
                should_send = (current_time - last_sent) >= timedelta(days=1)
            elif preferences.digest_frequency == "weekly":
                should_send = (current_time - last_sent) >= timedelta(weeks=1)
            
            if should_send and messages:
                await self._create_and_send_digest(user_id, channel, messages)
                self.digest_queues[digest_key] = []
                self.last_digest_sent[digest_key] = current_time
    
    async def _create_and_send_digest(
        self,
        user_id: str,
        channel: NotificationChannel,
        messages: List[NotificationMessage]
    ):
        """Create and send digest notification"""
        try:
            # Aggregate messages by type
            summary = defaultdict(int)
            for message in messages:
                summary[message.notification_type.value] += 1
            
            # Create digest content
            digest_body = f"📋 **Notification Digest** ({len(messages)} notifications)\n\n"
            
            for notification_type, count in summary.items():
                digest_body += f"• {notification_type.replace('_', ' ').title()}: {count}\n"
            
            digest_body += f"\n🕐 Period: {messages[0].created_at.strftime('%Y-%m-%d %H:%M')} - {messages[-1].created_at.strftime('%Y-%m-%d %H:%M')}"
            
            # Create digest message
            digest_message = NotificationMessage(
                user_id=user_id,
                channel=channel,
                notification_type=NotificationType.WORKFLOW_COMPLETED,  # Generic type for digests
                priority=NotificationPriority.NORMAL,
                subject=f"PRSM Notification Digest - {len(messages)} Updates",
                body=digest_body
            )
            
            # Send digest
            await self._deliver_message(digest_message)
            
            logger.info("Digest sent", user_id=user_id, channel=channel.value, message_count=len(messages))
            
        except Exception as e:
            logger.error("Error creating digest", error=str(e))
    
    async def _check_escalation_policies(
        self,
        notification_type: NotificationType,
        priority: NotificationPriority,
        context: Dict[str, Any],
        workflow_id: Optional[UUID] = None
    ):
        """Check and trigger escalation policies"""
        try:
            for policy in self.escalation_policies:
                if not policy.enabled:
                    continue
                
                # Check if policy applies
                if notification_type not in policy.applies_to_types:
                    continue
                
                if priority.value < policy.min_priority.value:
                    continue
                
                # Check business hours constraint
                if policy.business_hours_only and not self._is_business_hours():
                    continue
                
                # Start escalation
                escalation_id = uuid4()
                self.active_escalations[escalation_id] = {
                    "policy": policy,
                    "notification_type": notification_type,
                    "context": context,
                    "workflow_id": workflow_id,
                    "started_at": datetime.now(timezone.utc),
                    "current_step": 0
                }
                
                # Schedule first escalation step
                await self._schedule_escalation_step(escalation_id)
                
                logger.info("Escalation triggered", policy_name=policy.name, escalation_id=str(escalation_id))
                
        except Exception as e:
            logger.error("Error checking escalation policies", error=str(e))
    
    async def _schedule_escalation_step(self, escalation_id: UUID):
        """Schedule escalation step"""
        # In production, this would schedule the escalation step
        # For now, just log it
        escalation = self.active_escalations.get(escalation_id)
        if escalation:
            policy = escalation["policy"]
            current_step = escalation["current_step"]
            
            if current_step < len(policy.escalation_steps):
                step = policy.escalation_steps[current_step]
                logger.info("Escalation step scheduled", escalation_id=str(escalation_id), step=current_step, delay_minutes=step.get("delay_minutes", 0))
    
    def _is_business_hours(self) -> bool:
        """Check if current time is business hours"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()
        return day_of_week < 5 and 9 <= hour <= 17
    
    def set_user_preferences(self, user_id: str, preferences: NotificationPreferences):
        """Set user notification preferences"""
        self.user_preferences[user_id] = preferences
        logger.info("User preferences updated", user_id=user_id)
    
    def get_user_preferences(self, user_id: str) -> Optional[NotificationPreferences]:
        """Get user notification preferences"""
        return self.user_preferences.get(user_id)
    
    def add_escalation_policy(self, policy: EscalationPolicy):
        """Add escalation policy"""
        self.escalation_policies.append(policy)
        logger.info("Escalation policy added", policy_name=policy.name)
    
    def get_notification_history(
        self,
        user_id: Optional[str] = None,
        notification_type: Optional[NotificationType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NotificationMessage]:
        """Get notification history with filters"""
        messages = list(self.message_history.values())
        
        # Apply filters
        if user_id:
            messages = [m for m in messages if m.user_id == user_id]
        
        if notification_type:
            messages = [m for m in messages if m.notification_type == notification_type]
        
        if start_time:
            messages = [m for m in messages if m.created_at >= start_time]
        
        if end_time:
            messages = [m for m in messages if m.created_at <= end_time]
        
        # Sort by creation time (newest first) and limit
        messages.sort(key=lambda m: m.created_at, reverse=True)
        return messages[:limit]
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get notification delivery statistics"""
        return dict(self.delivery_statistics)
    
    async def mark_notification_read(self, message_id: UUID, user_id: str):
        """Mark notification as read"""
        message = self.message_history.get(message_id)
        if message and message.user_id == user_id:
            message.status = NotificationStatus.READ
            message.read_at = datetime.now(timezone.utc)
            logger.info("Notification marked as read", message_id=str(message_id), user_id=user_id)
    
    async def mark_notification_dismissed(self, message_id: UUID, user_id: str):
        """Mark notification as dismissed"""
        message = self.message_history.get(message_id)
        if message and message.user_id == user_id:
            message.status = NotificationStatus.DISMISSED
            logger.info("Notification dismissed", message_id=str(message_id), user_id=user_id)
    
    def get_channel_health(self) -> Dict[NotificationChannel, Dict[str, Any]]:
        """Get health status of notification channels"""
        # Calculate health metrics for each channel
        for channel in NotificationChannel:
            delivered = self.delivery_statistics.get(f"delivered_{channel.value}", 0)
            failed = self.delivery_statistics.get(f"failed_{channel.value}", 0)
            total = delivered + failed
            
            if total > 0:
                success_rate = delivered / total
                health_status = "healthy" if success_rate > 0.95 else "degraded" if success_rate > 0.8 else "unhealthy"
            else:
                success_rate = 1.0
                health_status = "unknown"
            
            self.channel_health[channel] = {
                "status": health_status,
                "success_rate": success_rate,
                "delivered": delivered,
                "failed": failed,
                "total": total
            }
        
        return self.channel_health
    
    async def shutdown(self):
        """Shutdown notification system gracefully"""
        logger.info("Shutting down notification system")
        
        # Cancel background tasks
        if self._notification_processor_task:
            self._notification_processor_task.cancel()
        
        if self._digest_processor_task:
            self._digest_processor_task.cancel()
        
        # Process remaining messages
        while self.pending_messages:
            message = self.pending_messages.popleft()
            await self._deliver_message(message)
        
        logger.info("Notification system shutdown complete")


# Global instance for easy access
_notification_system = None

def get_notification_system() -> NotificationSystem:
    """Get global notification system instance"""
    global _notification_system
    if _notification_system is None:
        _notification_system = NotificationSystem()
    return _notification_system


# Integration with Progress Tracker
async def setup_progress_tracker_notifications(progress_tracker: ProgressTracker, notification_system: NotificationSystem):
    """Setup notifications for progress tracker events"""
    
    async def on_workflow_completed(workflow_id: UUID, progress_data: Dict[str, Any]):
        """Handle workflow completion notification"""
        await notification_system.send_notification(
            user_id=progress_data.get("user_id", "unknown"),
            notification_type=NotificationType.WORKFLOW_COMPLETED,
            context={
                "workflow_id": str(workflow_id),
                "workflow_name": progress_data.get("workflow_name", "Unknown Workflow"),
                "user_name": progress_data.get("user_name", "User"),
                "start_time": progress_data.get("start_time", "Unknown"),
                "end_time": progress_data.get("end_time", "Unknown"),
                "duration": progress_data.get("duration", "Unknown"),
                "completed_steps": progress_data.get("completed_steps", 0),
                "total_cost": progress_data.get("total_cost", 0),
                "results_summary": progress_data.get("results_summary", "Results available"),
                "dashboard_url": f"https://prsm.ai/dashboard/workflows/{workflow_id}"
            },
            priority=NotificationPriority.NORMAL,
            workflow_id=workflow_id
        )
    
    async def on_workflow_failed(workflow_id: UUID, error_data: Dict[str, Any]):
        """Handle workflow failure notification"""
        await notification_system.send_notification(
            user_id=error_data.get("user_id", "unknown"),
            notification_type=NotificationType.WORKFLOW_FAILED,
            context={
                "workflow_id": str(workflow_id),
                "workflow_name": error_data.get("workflow_name", "Unknown Workflow"),
                "user_name": error_data.get("user_name", "User"),
                "start_time": error_data.get("start_time", "Unknown"),
                "failure_time": error_data.get("failure_time", "Unknown"),
                "failed_step": error_data.get("failed_step", "Unknown step"),
                "error_message": error_data.get("error_message", "Unknown error"),
                "cost_incurred": error_data.get("cost_incurred", 0),
                "next_steps": error_data.get("next_steps", "Please check the error details and retry."),
                "support_url": "https://prsm.ai/support"
            },
            priority=NotificationPriority.HIGH,
            workflow_id=workflow_id
        )
    
    # Register event handlers with progress tracker
    # This would be implemented based on the progress tracker's event system
    logger.info("Progress tracker notifications configured")