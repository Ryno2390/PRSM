"""
PRSM API Security Monitoring and Threat Detection System
Real-time security monitoring, threat detection, and incident response
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import hashlib
import ipaddress
from collections import defaultdict, deque
import redis.asyncio as aioredis
import logging
from fastapi import Request, BackgroundTasks
import geoip2.database
import geoip2.errors
import re
import statistics
import math

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of security alerts"""
    BRUTE_FORCE = "brute_force"
    ANOMALOUS_LOGIN = "anomalous_login"
    SUSPICIOUS_API_USAGE = "suspicious_api_usage"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    ACCOUNT_TAKEOVER = "account_takeover"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_PAYLOAD = "malicious_payload"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"


class IncidentStatus(Enum):
    """Security incident status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"


@dataclass
class ThreatIndicator:
    """Individual threat indicator"""
    indicator_type: str
    value: Any
    confidence: float  # 0.0 to 1.0
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "system"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    event_type: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    payload_size_bytes: int
    threat_indicators: List[ThreatIndicator] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAlert:
    """Security alert"""
    alert_id: str
    alert_type: AlertType
    threat_level: ThreatLevel
    title: str
    description: str
    timestamp: datetime
    affected_user_id: Optional[str]
    source_ip: str
    indicators: List[ThreatIndicator]
    recommended_actions: List[str]
    auto_resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIncident:
    """Security incident"""
    incident_id: str
    title: str
    description: str
    status: IncidentStatus
    threat_level: ThreatLevel
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str]
    alerts: List[str]  # Alert IDs
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)


class AnomalyDetector:
    """Machine learning-based anomaly detection"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.baseline_window_hours = 24
        self.anomaly_threshold = 2.5  # Standard deviations
        
    async def detect_usage_anomalies(self, user_id: str, current_metrics: Dict[str, float]) -> List[ThreatIndicator]:
        """Detect anomalous usage patterns"""
        indicators = []
        
        # Get historical baseline
        baseline = await self._get_user_baseline(user_id)
        
        for metric, current_value in current_metrics.items():
            if metric in baseline:
                mean = baseline[metric]["mean"]
                std_dev = baseline[metric]["std_dev"]
                
                if std_dev > 0:
                    z_score = abs((current_value - mean) / std_dev)
                    
                    if z_score > self.anomaly_threshold:
                        confidence = min(0.95, z_score / 5.0)  # Cap at 95%
                        
                        indicators.append(ThreatIndicator(
                            indicator_type="usage_anomaly",
                            value={
                                "metric": metric,
                                "current": current_value,
                                "baseline_mean": mean,
                                "z_score": z_score
                            },
                            confidence=confidence,
                            description=f"Anomalous {metric}: {current_value:.2f} (baseline: {mean:.2f} ±{std_dev:.2f})"
                        ))
        
        return indicators
    
    async def detect_login_anomalies(self, user_id: str, login_location: Dict[str, Any], 
                                   device_fingerprint: str) -> List[ThreatIndicator]:
        """Detect anomalous login patterns"""
        indicators = []
        
        # Check geographic anomalies
        user_locations = await self._get_user_locations(user_id)
        if user_locations:
            if self._is_geographic_anomaly(login_location, user_locations):
                indicators.append(ThreatIndicator(
                    indicator_type="geographic_anomaly",
                    value=login_location,
                    confidence=0.8,
                    description=f"Login from unusual location: {login_location.get('city', 'Unknown')}, {login_location.get('country', 'Unknown')}"
                ))
        
        # Check device anomalies
        user_devices = await self._get_user_devices(user_id)
        if device_fingerprint not in user_devices:
            indicators.append(ThreatIndicator(
                indicator_type="new_device",
                value=device_fingerprint,
                confidence=0.6,
                description="Login from new device"
            ))
        
        # Check time-based anomalies
        login_hour = datetime.now(timezone.utc).hour
        typical_hours = await self._get_typical_login_hours(user_id)
        if typical_hours and login_hour not in typical_hours:
            indicators.append(ThreatIndicator(
                indicator_type="unusual_time",
                value=login_hour,
                confidence=0.5,
                description=f"Login at unusual time: {login_hour}:00"
            ))
        
        return indicators
    
    async def _get_user_baseline(self, user_id: str) -> Dict[str, Dict[str, float]]:
        """Get user's behavioral baseline"""
        baseline_data = await self.redis.get(f"user_baseline:{user_id}")
        
        if not baseline_data:
            return {}
        
        return json.loads(baseline_data)
    
    async def _get_user_locations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's typical login locations"""
        locations_data = await self.redis.get(f"user_locations:{user_id}")
        
        if not locations_data:
            return []
        
        return json.loads(locations_data)
    
    async def _get_user_devices(self, user_id: str) -> Set[str]:
        """Get user's known device fingerprints"""
        devices_data = await self.redis.get(f"user_devices:{user_id}")
        
        if not devices_data:
            return set()
        
        return set(json.loads(devices_data))
    
    async def _get_typical_login_hours(self, user_id: str) -> Set[int]:
        """Get user's typical login hours"""
        hours_data = await self.redis.get(f"user_login_hours:{user_id}")
        
        if not hours_data:
            return set()
        
        return set(json.loads(hours_data))
    
    def _is_geographic_anomaly(self, login_location: Dict[str, Any], 
                             user_locations: List[Dict[str, Any]]) -> bool:
        """Check if login location is geographically anomalous"""
        login_country = login_location.get("country")
        login_lat = login_location.get("latitude")
        login_lon = login_location.get("longitude")
        
        if not all([login_country, login_lat, login_lon]):
            return False
        
        # Check if country is in user's typical countries
        typical_countries = {loc.get("country") for loc in user_locations}
        if login_country in typical_countries:
            return False
        
        # Check distance from typical locations
        min_distance = float('inf')
        for location in user_locations:
            if location.get("latitude") and location.get("longitude"):
                distance = self._calculate_distance(
                    login_lat, login_lon,
                    location["latitude"], location["longitude"]
                )
                min_distance = min(min_distance, distance)
        
        # Consider > 1000km anomalous
        return min_distance > 1000
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) * math.sin(delta_lon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


class ThreatIntelligence:
    """Threat intelligence and IOC checking"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.malicious_ips = set()
        self.malicious_user_agents = set()
        self.malicious_patterns = []
        self._load_threat_intel()
    
    def _load_threat_intel(self):
        """Load threat intelligence data"""
        # Known malicious IP ranges
        self.malicious_ip_ranges = [
            # Tor exit nodes, known botnets, etc.
            # This would be populated from threat intelligence feeds
        ]
        
        # Malicious user agents
        self.malicious_user_agents.update([
            "sqlmap",
            "nmap",
            "nikto",
            "havij",
            "masscan",
            "zap",
            "burpsuite"
        ])
        
        # Malicious patterns in requests
        self.malicious_patterns = [
            # SQL injection patterns
            re.compile(r'(union\s+select|select\s+.*\s+from|\'\s*or\s+1\s*=\s*1)', re.IGNORECASE),
            # XSS patterns
            re.compile(r'(<script|javascript:|onload=|onerror=)', re.IGNORECASE),
            # Command injection patterns
            re.compile(r'(;|\||\&)\s*(cat|ls|pwd|id|whoami|nc|netcat)', re.IGNORECASE),
            # Path traversal patterns
            re.compile(r'(\.\./|\.\.\\\|%2e%2e%2f|%2e%2e%5c)', re.IGNORECASE),
        ]
    
    async def check_ip_reputation(self, ip_address: str) -> List[ThreatIndicator]:
        """Check IP reputation"""
        indicators = []
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check against known malicious IPs
            if ip_address in self.malicious_ips:
                indicators.append(ThreatIndicator(
                    indicator_type="malicious_ip",
                    value=ip_address,
                    confidence=0.95,
                    description="IP address found in threat intelligence feeds",
                    source="threat_intel"
                ))
            
            # Check against malicious IP ranges
            for range_str in self.malicious_ip_ranges:
                try:
                    network = ipaddress.ip_network(range_str)
                    if ip in network:
                        indicators.append(ThreatIndicator(
                            indicator_type="malicious_ip_range",
                            value={"ip": ip_address, "range": range_str},
                            confidence=0.85,
                            description=f"IP address in malicious range: {range_str}",
                            source="threat_intel"
                        ))
                        break
                except ValueError:
                    continue
            
            # Check for Tor exit nodes (mock implementation)
            if await self._is_tor_exit_node(ip_address):
                indicators.append(ThreatIndicator(
                    indicator_type="tor_exit_node",
                    value=ip_address,
                    confidence=0.7,
                    description="Request from Tor exit node",
                    source="tor_detection"
                ))
            
        except ValueError:
            # Invalid IP address
            indicators.append(ThreatIndicator(
                indicator_type="invalid_ip",
                value=ip_address,
                confidence=0.8,
                description="Invalid IP address format",
                source="validation"
            ))
        
        return indicators
    
    def check_user_agent(self, user_agent: str) -> List[ThreatIndicator]:
        """Check user agent for malicious patterns"""
        indicators = []
        
        user_agent_lower = user_agent.lower()
        
        # Check against known malicious user agents
        for malicious_ua in self.malicious_user_agents:
            if malicious_ua in user_agent_lower:
                indicators.append(ThreatIndicator(
                    indicator_type="malicious_user_agent",
                    value=user_agent,
                    confidence=0.9,
                    description=f"User agent contains malicious pattern: {malicious_ua}",
                    source="user_agent_analysis"
                ))
        
        # Check for suspicious patterns
        if not user_agent or len(user_agent) < 10:
            indicators.append(ThreatIndicator(
                indicator_type="suspicious_user_agent",
                value=user_agent,
                confidence=0.6,
                description="Unusually short or missing user agent",
                source="user_agent_analysis"
            ))
        
        return indicators
    
    def check_request_payload(self, payload: str, content_type: str = "application/json") -> List[ThreatIndicator]:
        """Check request payload for malicious patterns"""
        indicators = []
        
        if not payload:
            return indicators
        
        # Check against malicious patterns
        for pattern in self.malicious_patterns:
            matches = pattern.findall(payload)
            if matches:
                indicators.append(ThreatIndicator(
                    indicator_type="malicious_payload",
                    value={"matches": matches, "pattern": pattern.pattern},
                    confidence=0.8,
                    description=f"Malicious pattern detected in payload: {matches[0]}",
                    source="payload_analysis"
                ))
        
        # Check payload size anomalies
        if len(payload) > 1000000:  # 1MB
            indicators.append(ThreatIndicator(
                indicator_type="large_payload",
                value=len(payload),
                confidence=0.5,
                description=f"Unusually large payload: {len(payload)} bytes",
                source="payload_analysis"
            ))
        
        return indicators
    
    async def _is_tor_exit_node(self, ip_address: str) -> bool:
        """Check if IP is a Tor exit node (mock implementation)"""
        # In production, this would check against Tor exit node lists
        # For now, return False
        return False


class SecurityMonitor:
    """Main security monitoring system"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.anomaly_detector = AnomalyDetector(redis_client)
        self.threat_intel = ThreatIntelligence(redis_client)
        
        # Event processing
        self.event_buffer: deque = deque(maxlen=10000)
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        
        # Thresholds
        self.brute_force_threshold = 10  # Failed attempts in 5 minutes
        self.rate_limit_alert_threshold = 5  # Rate limit violations in 10 minutes
        
        # Start background processing
        self.processing_task = None
    
    async def start_monitoring(self):
        """Start security monitoring background tasks"""
        self.processing_task = asyncio.create_task(self._process_events_loop())
        logger.info("✅ Security monitoring started")
    
    async def stop_monitoring(self):
        """Stop security monitoring"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Security monitoring stopped")
    
    async def process_security_event(self, event: SecurityEvent, background_tasks: BackgroundTasks):
        """Process a security event"""
        
        # Add to event buffer for real-time processing
        self.event_buffer.append(event)
        
        # Immediate threat analysis
        threats = await self._analyze_event_threats(event)
        event.threat_indicators.extend(threats)
        event.risk_score = self._calculate_risk_score(event)
        
        # Store event
        await self._store_security_event(event)
        
        # Generate alerts if needed
        alerts = await self._generate_alerts(event)
        for alert in alerts:
            background_tasks.add_task(self._handle_alert, alert)
        
        # Update baselines in background
        if event.user_id:
            background_tasks.add_task(self._update_user_baseline, event)
    
    async def _analyze_event_threats(self, event: SecurityEvent) -> List[ThreatIndicator]:
        """Analyze event for threat indicators"""
        indicators = []
        
        # IP reputation check
        ip_indicators = await self.threat_intel.check_ip_reputation(event.ip_address)
        indicators.extend(ip_indicators)
        
        # User agent analysis
        ua_indicators = self.threat_intel.check_user_agent(event.user_agent)
        indicators.extend(ua_indicators)
        
        # Payload analysis (if available)
        if "payload" in event.metadata:
            payload_indicators = self.threat_intel.check_request_payload(
                event.metadata["payload"],
                event.metadata.get("content_type", "application/json")
            )
            indicators.extend(payload_indicators)
        
        # Anomaly detection for authenticated users
        if event.user_id:
            # Usage anomalies
            current_metrics = {
                "requests_per_hour": await self._get_user_request_rate(event.user_id),
                "data_transferred_mb": await self._get_user_data_transfer(event.user_id),
                "unique_endpoints": await self._get_user_endpoint_diversity(event.user_id)
            }
            
            anomaly_indicators = await self.anomaly_detector.detect_usage_anomalies(
                event.user_id, current_metrics
            )
            indicators.extend(anomaly_indicators)
        
        return indicators
    
    def _calculate_risk_score(self, event: SecurityEvent) -> float:
        """Calculate risk score for event"""
        base_score = 0.0
        
        # Base score from response status
        if event.status_code >= 400:
            base_score += 0.2
        if event.status_code >= 500:
            base_score += 0.3
        
        # Score from threat indicators
        for indicator in event.threat_indicators:
            score_multiplier = {
                "malicious_ip": 0.8,
                "malicious_user_agent": 0.7,
                "malicious_payload": 0.9,
                "usage_anomaly": 0.5,
                "geographic_anomaly": 0.6,
                "tor_exit_node": 0.4
            }.get(indicator.indicator_type, 0.3)
            
            base_score += indicator.confidence * score_multiplier
        
        # Response time anomalies
        if event.response_time_ms > 10000:  # > 10 seconds
            base_score += 0.2
        
        # Large payload size
        if event.payload_size_bytes > 10000000:  # > 10MB
            base_score += 0.3
        
        return min(1.0, base_score)  # Cap at 1.0
    
    async def _generate_alerts(self, event: SecurityEvent) -> List[SecurityAlert]:
        """Generate security alerts based on event"""
        alerts = []
        
        # High-risk event alert
        if event.risk_score >= 0.8:
            alerts.append(SecurityAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.SUSPICIOUS_API_USAGE,
                threat_level=ThreatLevel.HIGH if event.risk_score >= 0.9 else ThreatLevel.MEDIUM,
                title="High-Risk API Request Detected",
                description=f"Request to {event.endpoint} from {event.ip_address} has high risk score: {event.risk_score:.2f}",
                timestamp=datetime.now(timezone.utc),
                affected_user_id=event.user_id,
                source_ip=event.ip_address,
                indicators=event.threat_indicators,
                recommended_actions=[
                    "Review request details",
                    "Check user account for compromise",
                    "Consider blocking IP if malicious"
                ]
            ))
        
        # Brute force detection
        if event.status_code == 401:  # Unauthorized
            recent_failures = await self._count_recent_failures(event.ip_address)
            if recent_failures >= self.brute_force_threshold:
                alerts.append(SecurityAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.BRUTE_FORCE,
                    threat_level=ThreatLevel.HIGH,
                    title="Brute Force Attack Detected",
                    description=f"IP {event.ip_address} has {recent_failures} failed login attempts in 5 minutes",
                    timestamp=datetime.now(timezone.utc),
                    affected_user_id=event.user_id,
                    source_ip=event.ip_address,
                    indicators=event.threat_indicators,
                    recommended_actions=[
                        "Block IP address",
                        "Notify affected users",
                        "Enable additional authentication"
                    ]
                ))
        
        # Rate limiting violations
        if "rate_limited" in event.metadata and event.metadata["rate_limited"]:
            recent_violations = await self._count_rate_limit_violations(event.ip_address)
            if recent_violations >= self.rate_limit_alert_threshold:
                alerts.append(SecurityAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.RATE_LIMIT_EXCEEDED,
                    threat_level=ThreatLevel.MEDIUM,
                    title="Excessive Rate Limiting Violations",
                    description=f"IP {event.ip_address} has exceeded rate limits {recent_violations} times in 10 minutes",
                    timestamp=datetime.now(timezone.utc),
                    affected_user_id=event.user_id,
                    source_ip=event.ip_address,
                    indicators=event.threat_indicators,
                    recommended_actions=[
                        "Review rate limit configuration",
                        "Consider stricter limits for this IP",
                        "Investigate if legitimate high-volume usage"
                    ]
                ))
        
        return alerts
    
    async def _handle_alert(self, alert: SecurityAlert):
        """Handle security alert"""
        
        # Store alert
        await self._store_security_alert(alert)
        
        # Log alert
        logger.warning(f"Security Alert: {alert.title} - {alert.description}")
        
        # Execute registered handlers
        handlers = self.alert_handlers.get(alert.alert_type, [])
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        # Auto-remediation for critical alerts
        if alert.threat_level == ThreatLevel.CRITICAL:
            await self._auto_remediate(alert)
    
    async def _auto_remediate(self, alert: SecurityAlert):
        """Automatic remediation for critical alerts"""
        
        # Block malicious IPs
        if alert.alert_type in [AlertType.BRUTE_FORCE, AlertType.MALICIOUS_PAYLOAD]:
            await self._block_ip_address(alert.source_ip, duration_hours=24)
            logger.info(f"Auto-blocked IP {alert.source_ip} for 24 hours")
        
        # Revoke compromised sessions
        if alert.alert_type == AlertType.ACCOUNT_TAKEOVER and alert.affected_user_id:
            # This would integrate with session manager
            logger.info(f"Would revoke sessions for user {alert.affected_user_id}")
    
    async def _process_events_loop(self):
        """Background event processing loop"""
        while True:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds
                
                if self.event_buffer:
                    events = list(self.event_buffer)
                    self.event_buffer.clear()
                    
                    # Batch processing
                    await self._batch_process_events(events)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
    
    async def _batch_process_events(self, events: List[SecurityEvent]):
        """Process events in batch for pattern detection"""
        
        # Group events by IP
        ip_groups = defaultdict(list)
        for event in events:
            ip_groups[event.ip_address].append(event)
        
        # Detect coordinated attacks
        for ip, ip_events in ip_groups.items():
            if len(ip_events) > 20:  # High volume from single IP
                # Check for distributed attack patterns
                endpoints = set(event.endpoint for event in ip_events)
                if len(endpoints) > 10:  # Scanning multiple endpoints
                    alert = SecurityAlert(
                        alert_id=self._generate_alert_id(),
                        alert_type=AlertType.SUSPICIOUS_API_USAGE,
                        threat_level=ThreatLevel.HIGH,
                        title="Potential API Scanning Detected",
                        description=f"IP {ip} accessed {len(endpoints)} different endpoints in batch",
                        timestamp=datetime.now(timezone.utc),
                        affected_user_id=None,
                        source_ip=ip,
                        indicators=[],
                        recommended_actions=[
                            "Block IP address",
                            "Review accessed endpoints",
                            "Check for data exfiltration"
                        ]
                    )
                    await self._handle_alert(alert)
    
    # Helper methods
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import secrets
        return f"alert_{secrets.token_hex(8)}"
    
    async def _store_security_event(self, event: SecurityEvent):
        """Store security event in Redis"""
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "session_id": event.session_id,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "endpoint": event.endpoint,
            "method": event.method,
            "status_code": event.status_code,
            "response_time_ms": event.response_time_ms,
            "payload_size_bytes": event.payload_size_bytes,
            "threat_indicators": [
                {
                    "indicator_type": ind.indicator_type,
                    "value": ind.value,
                    "confidence": ind.confidence,
                    "description": ind.description,
                    "timestamp": ind.timestamp.isoformat(),
                    "source": ind.source
                }
                for ind in event.threat_indicators
            ],
            "risk_score": event.risk_score,
            "metadata": event.metadata
        }
        
        await self.redis.setex(
            f"security_event:{event.event_id}",
            86400 * 30,  # 30 days retention
            json.dumps(event_data)
        )
    
    async def _store_security_alert(self, alert: SecurityAlert):
        """Store security alert in Redis"""
        alert_data = {
            "alert_id": alert.alert_id,
            "alert_type": alert.alert_type.value,
            "threat_level": alert.threat_level.value,
            "title": alert.title,
            "description": alert.description,
            "timestamp": alert.timestamp.isoformat(),
            "affected_user_id": alert.affected_user_id,
            "source_ip": alert.source_ip,
            "indicators": [
                {
                    "indicator_type": ind.indicator_type,
                    "value": ind.value,
                    "confidence": ind.confidence,
                    "description": ind.description,
                    "timestamp": ind.timestamp.isoformat(),
                    "source": ind.source
                }
                for ind in alert.indicators
            ],
            "recommended_actions": alert.recommended_actions,
            "auto_resolved": alert.auto_resolved,
            "metadata": alert.metadata
        }
        
        await self.redis.setex(
            f"security_alert:{alert.alert_id}",
            86400 * 90,  # 90 days retention
            json.dumps(alert_data)
        )
    
    async def _count_recent_failures(self, ip_address: str) -> int:
        """Count recent authentication failures from IP"""
        count_data = await self.redis.get(f"auth_failures:{ip_address}")
        return int(count_data) if count_data else 0
    
    async def _count_rate_limit_violations(self, ip_address: str) -> int:
        """Count recent rate limit violations from IP"""
        count_data = await self.redis.get(f"rate_violations:{ip_address}")
        return int(count_data) if count_data else 0
    
    async def _block_ip_address(self, ip_address: str, duration_hours: int = 24):
        """Block IP address for specified duration"""
        await self.redis.setex(
            f"blocked_ip:{ip_address}",
            duration_hours * 3600,
            json.dumps({
                "blocked_at": datetime.now(timezone.utc).isoformat(),
                "duration_hours": duration_hours,
                "reason": "automatic_security_block"
            })
        )
    
    async def _get_user_request_rate(self, user_id: str) -> float:
        """Get user's current request rate (requests per hour)"""
        # This would calculate based on recent events
        return 0.0  # Mock implementation
    
    async def _get_user_data_transfer(self, user_id: str) -> float:
        """Get user's current data transfer rate (MB per hour)"""
        # This would calculate based on recent events
        return 0.0  # Mock implementation
    
    async def _get_user_endpoint_diversity(self, user_id: str) -> int:
        """Get number of unique endpoints accessed by user recently"""
        # This would calculate based on recent events
        return 0  # Mock implementation
    
    async def _update_user_baseline(self, event: SecurityEvent):
        """Update user behavioral baseline"""
        # This would update the user's behavioral baseline
        pass  # Mock implementation
    
    def register_alert_handler(self, alert_type: AlertType, handler: Callable):
        """Register custom alert handler"""
        self.alert_handlers[alert_type].append(handler)


# Global security monitor instance
security_monitor: Optional[SecurityMonitor] = None


async def initialize_security_monitor(redis_client: aioredis.Redis):
    """Initialize the global security monitor"""
    global security_monitor
    security_monitor = SecurityMonitor(redis_client)
    await security_monitor.start_monitoring()
    logger.info("✅ Security monitoring system initialized")


def get_security_monitor() -> SecurityMonitor:
    """Get the global security monitor instance"""
    if security_monitor is None:
        raise RuntimeError("Security monitor not initialized. Call initialize_security_monitor() first.")
    return security_monitor


async def shutdown_security_monitor():
    """Shutdown the security monitor"""
    if security_monitor:
        await security_monitor.stop_monitoring()