"""
Security Sandbox & Circuit Breaker Integration
==============================================

Production-ready integration between the security sandbox and circuit breaker systems,
providing comprehensive automated safety protection for code execution environments.

Key Features:
- Real-time sandbox monitoring with circuit breaker integration
- Automated threat detection and response workflows
- Progressive safety escalation with emergency halt capabilities
- Comprehensive audit logging and compliance tracking
- Self-healing security controls with automated recovery
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
import structlog
from pathlib import Path

from prsm.core.integrations.security.enhanced_sandbox import (
    EnhancedSandboxManager, EnhancedSandboxResult
)
from prsm.core.safety.circuit_breaker import (
    CircuitBreakerNetwork, ThreatLevel, SafetyAssessment, CircuitState
)
from prsm.core.monitoring.enterprise_monitoring import get_monitoring, MonitoringComponent
from prsm.core.compliance.soc2_iso27001_framework import get_compliance_framework

logger = structlog.get_logger(__name__)


class SandboxThreatType(Enum):
    """Types of threats detected in sandbox execution"""
    RESOURCE_ABUSE = "resource_abuse"
    SUSPICIOUS_FILE_CREATION = "suspicious_file_creation"
    NETWORK_VIOLATION = "network_violation"
    EXECUTION_TIMEOUT = "execution_timeout"
    MALICIOUS_CODE_PATTERN = "malicious_code_pattern"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SYSTEM_MODIFICATION = "system_modification"


class ResponseAction(Enum):
    """Response actions for sandbox threats"""
    LOG_ONLY = "log_only"
    INCREASE_MONITORING = "increase_monitoring"
    SANDBOX_ISOLATION = "sandbox_isolation"
    CIRCUIT_BREAKER_TRIGGER = "circuit_breaker_trigger"
    EMERGENCY_HALT = "emergency_halt"
    USER_NOTIFICATION = "user_notification"


@dataclass
class SandboxThreatAssessment:
    """Assessment of sandbox execution threats"""
    assessment_id: str
    execution_id: str
    user_id: str
    threat_types: List[SandboxThreatType]
    severity_score: float  # 0.0 - 1.0
    threat_level: ThreatLevel
    recommended_actions: List[ResponseAction]
    security_events: List[Dict[str, Any]]
    resource_usage: Dict[str, Any]
    timestamp: datetime
    confidence: float = 0.9


@dataclass
class SafetyResponse:
    """Response to sandbox security threats"""
    response_id: str
    assessment_id: str
    actions_taken: List[ResponseAction]
    circuit_breaker_triggered: bool
    emergency_halt_activated: bool
    execution_terminated: bool
    recovery_plan: Optional[str]
    success: bool
    timestamp: datetime


class SandboxCircuitBreakerIntegration:
    """
    Integrated security system combining sandbox monitoring with circuit breaker safety
    
    Features:
    - Real-time sandbox execution monitoring with threat detection
    - Automated circuit breaker activation for dangerous code execution
    - Progressive threat escalation with configurable response thresholds
    - Emergency halt capabilities for critical security threats
    - Comprehensive audit logging and compliance evidence collection
    - Self-healing mechanisms with automatic recovery procedures
    """
    
    def __init__(self):
        self.sandbox_manager = EnhancedSandboxManager()
        self.circuit_breaker = CircuitBreakerNetwork()
        self.monitoring = get_monitoring()
        self.compliance_framework = get_compliance_framework()
        
        # Integration configuration
        self.threat_thresholds = {
            SandboxThreatType.RESOURCE_ABUSE: 0.6,
            SandboxThreatType.SUSPICIOUS_FILE_CREATION: 0.7,
            SandboxThreatType.NETWORK_VIOLATION: 0.8,
            SandboxThreatType.EXECUTION_TIMEOUT: 0.5,
            SandboxThreatType.MALICIOUS_CODE_PATTERN: 0.9,
            SandboxThreatType.PRIVILEGE_ESCALATION: 0.95,
            SandboxThreatType.DATA_EXFILTRATION: 0.95,
            SandboxThreatType.SYSTEM_MODIFICATION: 0.85
        }
        
        # Response configuration
        self.auto_response_enabled = True
        self.emergency_halt_threshold = 0.9
        self.circuit_breaker_threshold = 0.7
        self.max_failures_per_user = 5
        self.cooldown_period = timedelta(minutes=30)
        
        # Tracking state
        self.threat_assessments = {}
        self.user_failure_counts = {}
        self.active_responses = {}
        self.quarantined_executions = set()
        
        logger.info("Sandbox-Circuit Breaker integration initialized",
                   auto_response=self.auto_response_enabled,
                   threat_types=len(self.threat_thresholds))
    
    async def execute_with_safety_monitoring(
        self,
        content_path: str,
        execution_config: Dict[str, Any],
        user_id: str,
        platform: str = "unknown"
    ) -> Tuple[EnhancedSandboxResult, SandboxThreatAssessment]:
        """
        Execute code in sandbox with integrated safety monitoring and circuit breaker protection
        
        Args:
            content_path: Path to code/content to execute
            execution_config: Execution configuration parameters
            user_id: User requesting execution
            platform: Source platform or context
            
        Returns:
            Tuple of (sandbox result, threat assessment)
        """
        execution_id = str(uuid4())
        
        try:
            logger.info("Starting monitored sandbox execution",
                       execution_id=execution_id,
                       user_id=user_id,
                       platform=platform)
            
            # Start monitoring trace
            trace_id = self.monitoring.start_trace(
                operation="sandbox_execution_with_safety",
                component=MonitoringComponent.SECURITY
            )
            
            # Check if user is in cooldown or quarantined
            if await self._is_user_restricted(user_id):
                raise SecurityException(f"User {user_id} is currently restricted from code execution")
            
            # Check circuit breaker status
            if await self._should_block_execution(user_id, platform):
                raise SecurityException("Code execution blocked by circuit breaker safety system")
            
            # Execute in enhanced sandbox with monitoring
            sandbox_result = await self.sandbox_manager.execute_with_monitoring(
                content_path=content_path,
                execution_config=execution_config,
                user_id=user_id,
                platform=platform
            )
            
            # Assess threats from sandbox execution
            threat_assessment = await self._assess_sandbox_threats(
                execution_id, sandbox_result, user_id, platform
            )
            
            # Process threat assessment and determine response
            safety_response = await self._process_threat_assessment(threat_assessment)
            
            # Update circuit breaker based on assessment
            await self._update_circuit_breaker_from_assessment(threat_assessment)
            
            # Record compliance evidence
            await self._record_compliance_evidence(threat_assessment, safety_response)
            
            # Record monitoring metrics
            self._record_monitoring_metrics(threat_assessment, safety_response)
            
            # End monitoring trace
            self.monitoring.end_trace(
                trace_id, 
                success=sandbox_result.success and threat_assessment.threat_level.value < ThreatLevel.HIGH.value
            )
            
            logger.info("Monitored sandbox execution completed",
                       execution_id=execution_id,
                       threat_level=threat_assessment.threat_level.name,
                       safety_response=safety_response.success)
            
            return sandbox_result, threat_assessment
            
        except SecurityException as e:
            logger.warning("Security restriction applied",
                          execution_id=execution_id,
                          user_id=user_id,
                          restriction=str(e))
            
            # Create safe response
            restricted_result = EnhancedSandboxResult(
                success=False,
                output="",
                error_output=f"Security restriction: {str(e)}",
                exit_code=-3,
                execution_time=0.0,
                security_events=[{"type": "execution_blocked", "reason": str(e)}],
                resource_usage={"blocked": True}
            )
            
            threat_assessment = SandboxThreatAssessment(
                assessment_id=str(uuid4()),
                execution_id=execution_id,
                user_id=user_id,
                threat_types=[SandboxThreatType.PRIVILEGE_ESCALATION],
                severity_score=0.8,
                threat_level=ThreatLevel.HIGH,
                recommended_actions=[ResponseAction.USER_NOTIFICATION],
                security_events=[{"type": "security_restriction", "reason": str(e)}],
                resource_usage={},
                timestamp=datetime.now(timezone.utc)
            )
            
            return restricted_result, threat_assessment
            
        except Exception as e:
            logger.error("Failed sandbox execution with safety monitoring",
                        execution_id=execution_id,
                        error=str(e))
            raise
    
    async def _assess_sandbox_threats(
        self,
        execution_id: str,
        sandbox_result: EnhancedSandboxResult,
        user_id: str,
        platform: str
    ) -> SandboxThreatAssessment:
        """Assess threats from sandbox execution results"""
        assessment_id = str(uuid4())
        threat_types = []
        severity_score = 0.0
        
        # Analyze security events from sandbox
        for event in sandbox_result.security_events:
            event_type = event.get("type", "")
            
            if event_type == "execution_timeout":
                threat_types.append(SandboxThreatType.EXECUTION_TIMEOUT)
                severity_score += 0.3
            
            elif event_type == "disk_usage_exceeded":
                threat_types.append(SandboxThreatType.RESOURCE_ABUSE)
                severity_score += 0.4
            
            elif event_type == "suspicious_files_created":
                threat_types.append(SandboxThreatType.SUSPICIOUS_FILE_CREATION)
                severity_score += 0.5
            
            elif event_type == "network_violation":
                threat_types.append(SandboxThreatType.NETWORK_VIOLATION)
                severity_score += 0.6
        
        # Analyze output for malicious patterns
        malicious_patterns = await self._detect_malicious_patterns(
            sandbox_result.output, sandbox_result.error_output
        )
        
        if malicious_patterns:
            threat_types.append(SandboxThreatType.MALICIOUS_CODE_PATTERN)
            severity_score += len(malicious_patterns) * 0.2
        
        # Analyze resource usage for abuse
        if sandbox_result.resource_usage:
            if self._is_resource_abuse(sandbox_result.resource_usage):
                threat_types.append(SandboxThreatType.RESOURCE_ABUSE)
                severity_score += 0.3
        
        # Check for privilege escalation attempts
        if self._detect_privilege_escalation(sandbox_result.output, sandbox_result.error_output):
            threat_types.append(SandboxThreatType.PRIVILEGE_ESCALATION)
            severity_score += 0.7
        
        # Check for data exfiltration attempts
        if self._detect_data_exfiltration(sandbox_result.output):
            threat_types.append(SandboxThreatType.DATA_EXFILTRATION)
            severity_score += 0.8
        
        # Normalize severity score
        severity_score = min(1.0, severity_score)
        
        # Determine threat level
        threat_level = self._calculate_threat_level_from_score(severity_score, threat_types)
        
        # Determine recommended actions
        recommended_actions = self._determine_recommended_actions(threat_level, threat_types)
        
        assessment = SandboxThreatAssessment(
            assessment_id=assessment_id,
            execution_id=execution_id,
            user_id=user_id,
            threat_types=threat_types,
            severity_score=severity_score,
            threat_level=threat_level,
            recommended_actions=recommended_actions,
            security_events=sandbox_result.security_events,
            resource_usage=sandbox_result.resource_usage,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Store assessment
        self.threat_assessments[assessment_id] = assessment
        
        logger.info("Sandbox threat assessment completed",
                   assessment_id=assessment_id,
                   threat_level=threat_level.name,
                   severity_score=severity_score,
                   threat_count=len(threat_types))
        
        return assessment
    
    async def _process_threat_assessment(
        self, 
        assessment: SandboxThreatAssessment
    ) -> SafetyResponse:
        """Process threat assessment and execute appropriate responses"""
        response_id = str(uuid4())
        actions_taken = []
        circuit_breaker_triggered = False
        emergency_halt_activated = False
        execution_terminated = False
        recovery_plan = None
        
        try:
            # Always log the assessment
            actions_taken.append(ResponseAction.LOG_ONLY)
            
            if not self.auto_response_enabled:
                # Manual review mode
                actions_taken.append(ResponseAction.USER_NOTIFICATION)
                return SafetyResponse(
                    response_id=response_id,
                    assessment_id=assessment.assessment_id,
                    actions_taken=actions_taken,
                    circuit_breaker_triggered=False,
                    emergency_halt_activated=False,
                    execution_terminated=False,
                    recovery_plan="Manual review required",
                    success=True,
                    timestamp=datetime.now(timezone.utc)
                )
            
            # Automated response based on threat level
            if assessment.threat_level.value >= ThreatLevel.EMERGENCY.value:
                # Emergency response
                emergency_halt_activated = await self.circuit_breaker.trigger_emergency_halt(
                    threat_level=assessment.threat_level,
                    reason=f"Sandbox execution emergency: {', '.join([t.value for t in assessment.threat_types])}"
                )
                actions_taken.append(ResponseAction.EMERGENCY_HALT)
                execution_terminated = True
                
            elif assessment.threat_level.value >= ThreatLevel.CRITICAL.value:
                # Circuit breaker activation
                circuit_breaker_triggered = await self._trigger_circuit_breaker(assessment)
                actions_taken.append(ResponseAction.CIRCUIT_BREAKER_TRIGGER)
                execution_terminated = True
                
            elif assessment.threat_level.value >= ThreatLevel.HIGH.value:
                # Isolation and increased monitoring
                await self._isolate_execution(assessment)
                actions_taken.append(ResponseAction.SANDBOX_ISOLATION)
                actions_taken.append(ResponseAction.INCREASE_MONITORING)
                
            elif assessment.threat_level.value >= ThreatLevel.MODERATE.value:
                # Increased monitoring
                actions_taken.append(ResponseAction.INCREASE_MONITORING)
                actions_taken.append(ResponseAction.USER_NOTIFICATION)
            
            # Update user failure counts
            if assessment.threat_level.value >= ThreatLevel.HIGH.value:
                await self._update_user_failure_count(assessment.user_id)
            
            # Generate recovery plan if needed
            if circuit_breaker_triggered or emergency_halt_activated:
                recovery_plan = await self._generate_recovery_plan(assessment)
            
            response = SafetyResponse(
                response_id=response_id,
                assessment_id=assessment.assessment_id,
                actions_taken=actions_taken,
                circuit_breaker_triggered=circuit_breaker_triggered,
                emergency_halt_activated=emergency_halt_activated,
                execution_terminated=execution_terminated,
                recovery_plan=recovery_plan,
                success=True,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Store response
            self.active_responses[response_id] = response
            
            logger.info("Threat response executed",
                       response_id=response_id,
                       actions_count=len(actions_taken),
                       circuit_breaker_triggered=circuit_breaker_triggered,
                       emergency_halt=emergency_halt_activated)
            
            return response
            
        except Exception as e:
            logger.error("Failed to process threat assessment",
                        assessment_id=assessment.assessment_id,
                        error=str(e))
            
            return SafetyResponse(
                response_id=response_id,
                assessment_id=assessment.assessment_id,
                actions_taken=[ResponseAction.LOG_ONLY],
                circuit_breaker_triggered=False,
                emergency_halt_activated=False,
                execution_terminated=False,
                recovery_plan="Error in threat processing",
                success=False,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _update_circuit_breaker_from_assessment(
        self, 
        assessment: SandboxThreatAssessment
    ):
        """Update circuit breaker based on sandbox threat assessment"""
        # Create a mock output for circuit breaker assessment
        mock_output = {
            "execution_id": assessment.execution_id,
            "threat_types": [t.value for t in assessment.threat_types],
            "severity_score": assessment.severity_score,
            "security_events": assessment.security_events
        }
        
        # Feed assessment to circuit breaker
        cb_assessment = await self.circuit_breaker.monitor_model_behavior(
            model_id=f"sandbox_execution_{assessment.user_id}",
            output=mock_output
        )
        
        logger.debug("Circuit breaker updated from sandbox assessment",
                    circuit_breaker_threat=cb_assessment.threat_level.name,
                    sandbox_threat=assessment.threat_level.name)
    
    # Helper methods for threat detection
    async def _detect_malicious_patterns(self, output: str, error_output: str) -> List[str]:
        """Detect malicious code patterns in output"""
        patterns = []
        combined_output = f"{output} {error_output}".lower()
        
        malicious_indicators = [
            "shell injection", "code injection", "sql injection",
            "buffer overflow", "privilege escalation", "backdoor",
            "rootkit", "keylogger", "trojan", "virus", "malware",
            "reverse shell", "bind shell", "nc -e", "netcat",
            "/bin/sh", "/bin/bash", "cmd.exe", "powershell",
            "eval(", "exec(", "system(", "subprocess"
        ]
        
        for indicator in malicious_indicators:
            if indicator in combined_output:
                patterns.append(indicator)
        
        return patterns
    
    def _is_resource_abuse(self, resource_usage: Dict[str, Any]) -> bool:
        """Check if resource usage indicates abuse"""
        disk_usage = resource_usage.get("disk_usage_mb", 0)
        file_count = resource_usage.get("file_count", 0)
        
        # Thresholds for abuse detection
        return (disk_usage > 100 or  # More than 100MB
                file_count > 1000)      # More than 1000 files
    
    def _detect_privilege_escalation(self, output: str, error_output: str) -> bool:
        """Detect privilege escalation attempts"""
        combined = f"{output} {error_output}".lower()
        
        escalation_patterns = [
            "sudo", "su -", "chmod 777", "chown root",
            "setuid", "setgid", "/etc/passwd", "/etc/shadow",
            "privilege", "escalation", "root access"
        ]
        
        return any(pattern in combined for pattern in escalation_patterns)
    
    def _detect_data_exfiltration(self, output: str) -> bool:
        """Detect data exfiltration attempts"""
        output_lower = output.lower()
        
        exfiltration_patterns = [
            "curl", "wget", "http://", "https://",
            "ftp://", "scp", "rsync", "ssh",
            "base64", "gzip", "tar", "zip"
        ]
        
        # Check for suspicious network activity or data encoding
        return sum(1 for pattern in exfiltration_patterns if pattern in output_lower) >= 2
    
    def _calculate_threat_level_from_score(
        self, 
        severity_score: float, 
        threat_types: List[SandboxThreatType]
    ) -> ThreatLevel:
        """Calculate threat level from severity score and threat types"""
        # High-priority threats automatically escalate
        critical_threats = [
            SandboxThreatType.PRIVILEGE_ESCALATION,
            SandboxThreatType.DATA_EXFILTRATION,
            SandboxThreatType.SYSTEM_MODIFICATION
        ]
        
        if any(t in critical_threats for t in threat_types):
            if severity_score >= 0.9:
                return ThreatLevel.EMERGENCY
            elif severity_score >= 0.7:
                return ThreatLevel.CRITICAL
            else:
                return ThreatLevel.HIGH
        
        # Regular threat level calculation
        if severity_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif severity_score >= 0.7:
            return ThreatLevel.HIGH
        elif severity_score >= 0.4:
            return ThreatLevel.MODERATE
        elif severity_score >= 0.1:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE
    
    def _determine_recommended_actions(
        self, 
        threat_level: ThreatLevel, 
        threat_types: List[SandboxThreatType]
    ) -> List[ResponseAction]:
        """Determine recommended response actions"""
        actions = [ResponseAction.LOG_ONLY]
        
        if threat_level.value >= ThreatLevel.EMERGENCY.value:
            actions.extend([
                ResponseAction.EMERGENCY_HALT,
                ResponseAction.USER_NOTIFICATION
            ])
        elif threat_level.value >= ThreatLevel.CRITICAL.value:
            actions.extend([
                ResponseAction.CIRCUIT_BREAKER_TRIGGER,
                ResponseAction.SANDBOX_ISOLATION,
                ResponseAction.USER_NOTIFICATION
            ])
        elif threat_level.value >= ThreatLevel.HIGH.value:
            actions.extend([
                ResponseAction.SANDBOX_ISOLATION,
                ResponseAction.INCREASE_MONITORING,
                ResponseAction.USER_NOTIFICATION
            ])
        elif threat_level.value >= ThreatLevel.MODERATE.value:
            actions.extend([
                ResponseAction.INCREASE_MONITORING,
                ResponseAction.USER_NOTIFICATION
            ])
        
        return actions
    
    # Integration helper methods
    async def _is_user_restricted(self, user_id: str) -> bool:
        """Check if user is restricted from execution"""
        failure_count = self.user_failure_counts.get(user_id, 0)
        return failure_count >= self.max_failures_per_user
    
    async def _should_block_execution(self, user_id: str, platform: str) -> bool:
        """Check if execution should be blocked by circuit breaker"""
        # Check if there's an active emergency halt
        network_status = await self.circuit_breaker.get_network_status()
        return network_status.get("emergency_halt_active", False)
    
    async def _trigger_circuit_breaker(self, assessment: SandboxThreatAssessment) -> bool:
        """Trigger circuit breaker for sandbox threats"""
        return await self.circuit_breaker.trigger_emergency_halt(
            threat_level=assessment.threat_level,
            reason=f"Sandbox threat detected: {', '.join([t.value for t in assessment.threat_types])}"
        )
    
    async def _isolate_execution(self, assessment: SandboxThreatAssessment):
        """Isolate problematic execution"""
        self.quarantined_executions.add(assessment.execution_id)
        logger.warning("Execution quarantined",
                      execution_id=assessment.execution_id,
                      user_id=assessment.user_id)
    
    async def _update_user_failure_count(self, user_id: str):
        """Update user failure count"""
        self.user_failure_counts[user_id] = self.user_failure_counts.get(user_id, 0) + 1
        
        if self.user_failure_counts[user_id] >= self.max_failures_per_user:
            logger.warning("User reached maximum failure count",
                          user_id=user_id,
                          failure_count=self.user_failure_counts[user_id])
    
    async def _generate_recovery_plan(self, assessment: SandboxThreatAssessment) -> str:
        """Generate recovery plan for threat response"""
        if assessment.threat_level.value >= ThreatLevel.CRITICAL.value:
            return (
                f"Critical threat recovery plan:\n"
                f"1. Investigation required for execution {assessment.execution_id}\n"
                f"2. User {assessment.user_id} restricted pending review\n"
                f"3. Manual security team intervention needed\n"
                f"4. Threat types: {', '.join([t.value for t in assessment.threat_types])}"
            )
        else:
            return (
                f"Standard recovery plan:\n"
                f"1. Monitor user {assessment.user_id} for {self.cooldown_period}\n"
                f"2. Review security logs for patterns\n"
                f"3. Automatic recovery after cooldown period"
            )
    
    async def _record_compliance_evidence(
        self, 
        assessment: SandboxThreatAssessment, 
        response: SafetyResponse
    ):
        """Record compliance evidence for security events"""
        evidence_data = {
            "assessment": {
                "assessment_id": assessment.assessment_id,
                "execution_id": assessment.execution_id,
                "threat_level": assessment.threat_level.name,
                "severity_score": assessment.severity_score,
                "threat_types": [t.value for t in assessment.threat_types]
            },
            "response": {
                "response_id": response.response_id,
                "actions_taken": [a.value for a in response.actions_taken],
                "circuit_breaker_triggered": response.circuit_breaker_triggered,
                "emergency_halt_activated": response.emergency_halt_activated
            }
        }
        
        await self.compliance_framework.collect_evidence(
            control_id="CC6.2",  # System Access Monitoring
            evidence_type="security_incident_response",
            evidence_data=evidence_data,
            collected_by="sandbox_circuit_breaker_integration"
        )
    
    def _record_monitoring_metrics(
        self, 
        assessment: SandboxThreatAssessment, 
        response: SafetyResponse
    ):
        """Record monitoring metrics for threat assessment and response"""
        # Record threat assessment metrics
        self.monitoring.record_metric(
            name=f"sandbox_threat_assessment.{assessment.threat_level.name.lower()}",
            value=1,
            metric_type=self.monitoring.MetricType.COUNTER,
            component=MonitoringComponent.SECURITY,
            tags={
                "user_id": assessment.user_id,
                "severity_score": str(assessment.severity_score),
                "threat_count": str(len(assessment.threat_types))
            }
        )
        
        # Record response metrics
        for action in response.actions_taken:
            self.monitoring.record_metric(
                name=f"security_response.{action.value}",
                value=1,
                metric_type=self.monitoring.MetricType.COUNTER,
                component=MonitoringComponent.SECURITY,
                tags={
                    "threat_level": assessment.threat_level.name,
                    "success": str(response.success)
                }
            )
        
        # Record business metric for security incidents
        if assessment.threat_level.value >= ThreatLevel.HIGH.value:
            self.monitoring.record_business_metric(
                metric_name="security_incidents_high_severity",
                value=1,
                dimension="daily",
                metadata={
                    "threat_level": assessment.threat_level.name,
                    "auto_response": response.success
                }
            )


class SecurityException(Exception):
    """Exception raised for security-related execution restrictions"""
    pass


# Factory function
def get_sandbox_circuit_breaker_integration() -> SandboxCircuitBreakerIntegration:
    """Get the sandbox-circuit breaker integration instance"""
    return SandboxCircuitBreakerIntegration()