"""
Security Orchestrator
====================

Coordinates all security components for comprehensive content validation
before integration into PRSM. Provides a unified security pipeline.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from .vulnerability_scanner import VulnerabilityScanner, VulnerabilityResult
from .license_scanner import LicenseScanner, LicenseResult
from .threat_detector import ThreatDetector, ThreatResult, ThreatLevel
from .enhanced_sandbox import EnhancedSandboxManager, EnhancedSandboxResult
from .audit_logger import AuditLogger, SecurityEvent, EventLevel
from ..models.integration_models import SecurityRisk


class SecurityAssessment:
    """Comprehensive security assessment result"""
    
    def __init__(self, assessment_id: str, content_id: str, 
                 platform: str, user_id: str):
        self.assessment_id = assessment_id
        self.content_id = content_id
        self.platform = platform
        self.user_id = user_id
        self.timestamp = datetime.now(timezone.utc)
        
        # Individual scan results
        self.vulnerability_result: Optional[VulnerabilityResult] = None
        self.license_result: Optional[LicenseResult] = None
        self.threat_result: Optional[ThreatResult] = None
        self.sandbox_result: Optional[EnhancedSandboxResult] = None
        
        # Overall assessment
        self.overall_risk_level: SecurityRisk = SecurityRisk.UNKNOWN
        self.security_passed: bool = False
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        
        # Execution metadata
        self.scan_duration: float = 0.0
        self.scans_completed: List[str] = []
        self.scans_failed: List[str] = []


class SecurityOrchestrator:
    """Orchestrates comprehensive security scanning pipeline"""
    
    def __init__(self):
        """Initialize security orchestrator"""
        # Initialize security components
        self.vulnerability_scanner = VulnerabilityScanner()
        self.license_scanner = LicenseScanner()
        self.threat_detector = ThreatDetector()
        self.enhanced_sandbox = EnhancedSandboxManager()
        self.audit_logger = AuditLogger()
        
        # Security policies
        self.security_policies = {
            "require_license_compliance": True,
            "block_high_risk_vulnerabilities": True,
            "block_medium_risk_threats": True,
            "require_sandbox_validation": True,
            "max_acceptable_risk": SecurityRisk.LOW,
            "auto_quarantine_threats": True
        }
        
        print("🔍 Security Orchestrator initialized")
    
    async def comprehensive_security_assessment(
        self, 
        content_path: str, 
        metadata: Dict[str, Any],
        user_id: str,
        platform: str,
        content_id: str,
        enable_sandbox: bool = True
    ) -> SecurityAssessment:
        """
        Perform comprehensive security assessment of content
        
        Args:
            content_path: Path to content to assess
            metadata: Content metadata
            user_id: User requesting assessment
            platform: Source platform
            content_id: Unique content identifier
            enable_sandbox: Whether to enable sandbox execution
            
        Returns:
            SecurityAssessment with comprehensive results
        """
        assessment_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Create assessment object
        assessment = SecurityAssessment(
            assessment_id=assessment_id,
            content_id=content_id,
            platform=platform,
            user_id=user_id
        )
        
        # Log assessment start
        self.audit_logger.log_event(SecurityEvent(
            event_type="security_assessment_start",
            level=EventLevel.INFO,
            user_id=user_id,
            platform=platform,
            description=f"Starting comprehensive security assessment {assessment_id}",
            metadata={
                "assessment_id": assessment_id,
                "content_id": content_id,
                "content_path": content_path
            }
        ))
        
        try:
            # Run security scans in parallel for efficiency
            scan_tasks = []
            
            # 1. Vulnerability scanning
            scan_tasks.append(self._run_vulnerability_scan(content_path, metadata, assessment))
            
            # 2. License compliance
            scan_tasks.append(self._run_license_scan(metadata, assessment))
            
            # 3. Threat detection
            scan_tasks.append(self._run_threat_scan(content_path, metadata, assessment))
            
            # Execute scans concurrently
            await asyncio.gather(*scan_tasks, return_exceptions=True)
            
            # 4. Sandbox execution (if enabled and other scans pass)
            if enable_sandbox and self._should_run_sandbox(assessment):
                await self._run_sandbox_scan(content_path, metadata, assessment)
            
            # Analyze results and make final assessment
            self._finalize_assessment(assessment)
            
            # Calculate scan duration
            end_time = datetime.now(timezone.utc)
            assessment.scan_duration = (end_time - start_time).total_seconds()
            
            # Log final assessment
            self.audit_logger.log_import_security_check(
                user_id=user_id,
                platform=platform,
                content_id=content_id,
                passed=assessment.security_passed,
                issues=assessment.issues
            )
            
            return assessment
            
        except Exception as e:
            # Handle assessment error
            assessment.issues.append(f"Security assessment failed: {str(e)}")
            assessment.security_passed = False
            assessment.overall_risk_level = SecurityRisk.HIGH
            
            self.audit_logger.log_event(SecurityEvent(
                event_type="security_assessment_error",
                level=EventLevel.ERROR,
                user_id=user_id,
                platform=platform,
                description=f"Security assessment {assessment_id} failed: {str(e)}",
                metadata={
                    "assessment_id": assessment_id,
                    "error": str(e)
                }
            ))
            
            return assessment
    
    async def _run_vulnerability_scan(self, content_path: str, metadata: Dict[str, Any],
                                    assessment: SecurityAssessment) -> None:
        """Run vulnerability scanning"""
        try:
            assessment.vulnerability_result = await self.vulnerability_scanner.scan_vulnerabilities(
                content_path, metadata
            )
            assessment.scans_completed.append("vulnerability_scan")
            
            # Log results
            self.audit_logger.log_vulnerability_scan(
                user_id=assessment.user_id,
                platform=assessment.platform,
                scan_result=assessment.vulnerability_result,
                content_id=assessment.content_id
            )
            
        except Exception as e:
            assessment.scans_failed.append("vulnerability_scan")
            assessment.issues.append(f"Vulnerability scan failed: {str(e)}")
    
    async def _run_license_scan(self, metadata: Dict[str, Any],
                              assessment: SecurityAssessment) -> None:
        """Run license compliance scanning"""
        try:
            assessment.license_result = await self.license_scanner.scan_license(metadata)
            assessment.scans_completed.append("license_scan")
            
            # Log results
            self.audit_logger.log_license_check(
                user_id=assessment.user_id,
                platform=assessment.platform,
                license_result=assessment.license_result,
                content_id=assessment.content_id
            )
            
        except Exception as e:
            assessment.scans_failed.append("license_scan")
            assessment.issues.append(f"License scan failed: {str(e)}")
    
    async def _run_threat_scan(self, content_path: str, metadata: Dict[str, Any],
                             assessment: SecurityAssessment) -> None:
        """Run threat detection scanning"""
        try:
            assessment.threat_result = await self.threat_detector.scan_threats(
                content_path, metadata
            )
            assessment.scans_completed.append("threat_scan")
            
            # Log results
            self.audit_logger.log_threat_detection(
                user_id=assessment.user_id,
                platform=assessment.platform,
                threat_result=assessment.threat_result,
                content_id=assessment.content_id
            )
            
        except Exception as e:
            assessment.scans_failed.append("threat_scan")
            assessment.issues.append(f"Threat scan failed: {str(e)}")
    
    async def _run_sandbox_scan(self, content_path: str, metadata: Dict[str, Any],
                              assessment: SecurityAssessment) -> None:
        """Run sandbox execution if appropriate"""
        try:
            # Only run sandbox for executable content types
            if self._is_executable_content(metadata):
                execution_config = {
                    "timeout": 30,  # Limited execution time
                    "metadata": metadata
                }
                
                assessment.sandbox_result = await self.enhanced_sandbox.execute_with_monitoring(
                    content_path, execution_config, assessment.user_id, assessment.platform
                )
                assessment.scans_completed.append("sandbox_execution")
                
                # Log results
                self.audit_logger.log_sandbox_execution(
                    user_id=assessment.user_id,
                    platform=assessment.platform,
                    execution_result=assessment.sandbox_result,
                    content_id=assessment.content_id
                )
            else:
                assessment.warnings.append("Sandbox execution skipped for non-executable content")
            
        except Exception as e:
            assessment.scans_failed.append("sandbox_execution")
            assessment.issues.append(f"Sandbox execution failed: {str(e)}")
    
    def _should_run_sandbox(self, assessment: SecurityAssessment) -> bool:
        """Determine if sandbox execution should proceed based on initial scans"""
        # Don't run sandbox if critical threats detected
        if assessment.threat_result:
            if assessment.threat_result.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                return False
        
        # Don't run sandbox if critical vulnerabilities found
        if assessment.vulnerability_result:
            if assessment.vulnerability_result.risk_level == SecurityRisk.CRITICAL:
                return False
        
        return True
    
    def _is_executable_content(self, metadata: Dict[str, Any]) -> bool:
        """Check if content is executable and suitable for sandbox testing"""
        # Check file extensions and content types
        content_type = metadata.get("content_type", "").lower()
        filename = metadata.get("filename", "").lower()
        
        executable_indicators = [
            "python", "javascript", "java", "executable",
            ".py", ".js", ".jar", ".exe", ".sh", ".bat"
        ]
        
        return any(indicator in content_type or indicator in filename 
                  for indicator in executable_indicators)
    
    def _finalize_assessment(self, assessment: SecurityAssessment) -> None:
        """Analyze all scan results and make final security determination"""
        issues = []
        warnings = []
        recommendations = []
        risk_levels = []
        
        # Analyze vulnerability results
        if assessment.vulnerability_result:
            if assessment.vulnerability_result.vulnerabilities:
                issues.extend([f"Vulnerability: {v}" for v in assessment.vulnerability_result.vulnerabilities])
            risk_levels.append(assessment.vulnerability_result.risk_level)
        
        # Analyze license results
        if assessment.license_result:
            if not assessment.license_result.compliant:
                issues.extend([f"License: {issue}" for issue in assessment.license_result.issues])
                # License non-compliance is high risk
                risk_levels.append(SecurityRisk.HIGH)
            else:
                recommendations.append(f"License compliant: {assessment.license_result.license_type.value}")
        
        # Analyze threat results
        if assessment.threat_result:
            if assessment.threat_result.threats:
                for threat in assessment.threat_result.threats:
                    threat_desc = f"Threat: {threat.get('description', 'Unknown threat')}"
                    if threat.get('severity') in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                        issues.append(threat_desc)
                    else:
                        warnings.append(threat_desc)
            
            # Convert threat level to security risk
            threat_to_risk = {
                ThreatLevel.CRITICAL: SecurityRisk.CRITICAL,
                ThreatLevel.HIGH: SecurityRisk.HIGH,
                ThreatLevel.MEDIUM: SecurityRisk.MEDIUM,
                ThreatLevel.LOW: SecurityRisk.LOW,
                ThreatLevel.NONE: SecurityRisk.NONE
            }
            risk_levels.append(threat_to_risk.get(assessment.threat_result.threat_level, SecurityRisk.MEDIUM))
        
        # Analyze sandbox results
        if assessment.sandbox_result:
            if not assessment.sandbox_result.success:
                warnings.append(f"Sandbox execution failed: {assessment.sandbox_result.error_output}")
            
            if assessment.sandbox_result.security_events:
                for event in assessment.sandbox_result.security_events:
                    warnings.append(f"Sandbox event: {event.get('type', 'Unknown')}")
        
        # Determine overall risk level
        if risk_levels:
            # Use the highest risk level found
            risk_order = [SecurityRisk.NONE, SecurityRisk.LOW, SecurityRisk.MEDIUM, SecurityRisk.HIGH, SecurityRisk.CRITICAL]
            assessment.overall_risk_level = max(risk_levels, key=lambda x: risk_order.index(x))
        else:
            assessment.overall_risk_level = SecurityRisk.LOW
        
        # Apply security policies to determine if content passes
        assessment.security_passed = self._apply_security_policies(assessment, issues, warnings)
        
        # Set final results
        assessment.issues = issues
        assessment.warnings = warnings
        assessment.recommendations = recommendations
        
        # Add general recommendations
        if assessment.security_passed:
            recommendations.append("Content passed security validation")
        else:
            recommendations.append("Content blocked due to security concerns")
            
        if not assessment.scans_failed:
            recommendations.append("All security scans completed successfully")
    
    def _apply_security_policies(self, assessment: SecurityAssessment, 
                               issues: List[str], warnings: List[str]) -> bool:
        """Apply security policies to determine if content should be allowed"""
        policies = self.security_policies
        
        # Check license compliance policy
        if policies.get("require_license_compliance", True):
            if assessment.license_result and not assessment.license_result.compliant:
                return False
        
        # Check vulnerability policy
        if policies.get("block_high_risk_vulnerabilities", True):
            if assessment.vulnerability_result:
                if assessment.vulnerability_result.risk_level in [SecurityRisk.HIGH, SecurityRisk.CRITICAL]:
                    return False
        
        # Check threat policy
        if policies.get("block_medium_risk_threats", True):
            if assessment.threat_result:
                if assessment.threat_result.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    return False
        
        # Check overall risk policy
        max_risk = policies.get("max_acceptable_risk", SecurityRisk.LOW)
        risk_order = [SecurityRisk.NONE, SecurityRisk.LOW, SecurityRisk.MEDIUM, SecurityRisk.HIGH, SecurityRisk.CRITICAL]
        if risk_order.index(assessment.overall_risk_level) > risk_order.index(max_risk):
            return False
        
        # No policy violations found
        return True
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security system statistics"""
        return {
            "components": {
                "vulnerability_scanner": "active",
                "license_scanner": "active",
                "threat_detector": "active",
                "enhanced_sandbox": "active",
                "audit_logger": "active"
            },
            "policies": self.security_policies,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }


# Global security orchestrator instance
security_orchestrator = SecurityOrchestrator()