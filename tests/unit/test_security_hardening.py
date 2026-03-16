"""
Tests for Security Hardening Module

Tests for:
- SecurityAuditChecklist
- SecurityScanner
- PenetrationTestSuite
- SecretsManager
- SecureEnvironment
"""

import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Import security modules
from prsm.security.audit_checklist import (
    SecurityCheck,
    SecurityAuditChecklist,
    AuditReport,
    CheckResult,
    CheckStatus,
    CheckSeverity,
    CheckCategory,
)
from prsm.security.scanner import (
    SecurityScanner,
    Vulnerability,
    SecurityIssue,
    SecretLeak,
    ScanResult,
    VulnerabilitySeverity,
    IssueType,
)
from prsm.security.pentest import (
    PenetrationTestSuite,
    SecurityFinding,
    PenTestResult,
    PenTestReport,
    TestType,
    FindingSeverity,
)
from prsm.security.secrets import (
    SecretsManager,
    SecretBackend,
    SecretMetadata,
    SecretValue,
    EnvironmentBackend,
    FileBackend,
)
from prsm.security.env_config import (
    SecureEnvironment,
    SecretStrength,
    SecretStrengthResult,
    EnvironmentValidationResult,
)


# ===== SecurityAuditChecklist Tests =====

class TestSecurityCheck:
    """Tests for SecurityCheck"""
    
    def test_security_check_creation(self):
        """Test creating a security check"""
        check = SecurityCheck(
            check_id="test_check",
            name="Test Check",
            description="A test security check",
            category=CheckCategory.AUTHENTICATION,
            severity=CheckSeverity.HIGH,
            auto_check=True,
            remediation="Fix the issue",
            references=["CWE-123", "OWASP-A01"],
        )
        
        assert check.check_id == "test_check"
        assert check.name == "Test Check"
        assert check.category == CheckCategory.AUTHENTICATION
        assert check.severity == CheckSeverity.HIGH
        assert check.auto_check is True
    
    def test_security_check_defaults(self):
        """Test security check default values"""
        check = SecurityCheck(
            check_id="test",
            name="Test",
            description="Test",
            category=CheckCategory.NETWORK,
            severity=CheckSeverity.MEDIUM,
        )
        
        assert check.auto_check is True
        assert check.check_function is None
        assert check.remediation == ""
        assert check.references == []


class TestCheckResult:
    """Tests for CheckResult"""
    
    def test_check_result_creation(self):
        """Test creating a check result"""
        result = CheckResult(
            check_id="test_check",
            status=CheckStatus.PASS,
            message="Check passed",
            details={"key": "value"},
        )
        
        assert result.check_id == "test_check"
        assert result.status == CheckStatus.PASS
        assert result.message == "Check passed"
        assert result.details == {"key": "value"}
        assert isinstance(result.timestamp, datetime)
    
    def test_check_result_to_dict(self):
        """Test converting check result to dictionary"""
        result = CheckResult(
            check_id="test",
            status=CheckStatus.FAIL,
            message="Failed",
        )
        
        result_dict = result.to_dict() if hasattr(result, 'to_dict') else {
            "check_id": result.check_id,
            "status": result.status.value,
            "message": result.message,
        }
        
        assert result_dict["check_id"] == "test"
        assert result_dict["status"] == "fail"


class TestAuditReport:
    """Tests for AuditReport"""
    
    def test_audit_report_creation(self):
        """Test creating an audit report"""
        report = AuditReport(report_id="test_report")
        
        assert report.report_id == "test_report"
        assert len(report.results) == 0
        assert isinstance(report.timestamp, datetime)
    
    def test_add_result(self):
        """Test adding results to report"""
        report = AuditReport(report_id="test")
        result = CheckResult(
            check_id="check1",
            status=CheckStatus.PASS,
            message="Passed",
        )
        
        report.add_result(result)
        
        assert len(report.results) == 1
        assert report.results[0].check_id == "check1"
    
    def test_calculate_summary(self):
        """Test calculating summary"""
        report = AuditReport(report_id="test")
        
        # Add various results
        report.add_result(CheckResult("1", CheckStatus.PASS, "Pass"))
        report.add_result(CheckResult("2", CheckStatus.PASS, "Pass"))
        report.add_result(CheckResult("3", CheckStatus.FAIL, "Fail"))
        report.add_result(CheckResult("4", CheckStatus.WARNING, "Warning"))
        
        summary = report.calculate_summary()
        
        assert summary["total"] == 4
        assert summary["pass"] == 2
        assert summary["fail"] == 1
        assert summary["warning"] == 1
        assert summary["score"] == 50.0


class TestSecurityAuditChecklist:
    """Tests for SecurityAuditChecklist"""
    
    @pytest.fixture
    def checklist(self):
        """Create a security audit checklist"""
        return SecurityAuditChecklist()
    
    def test_checklist_initialization(self, checklist):
        """Test checklist initialization"""
        assert len(checklist.checks) > 0
        assert len(checklist._check_functions) > 0
    
    def test_checklist_has_default_checks(self, checklist):
        """Test that checklist has default checks"""
        # Check for authentication checks
        auth_checks = [c for c in checklist.checks if c.category == CheckCategory.AUTHENTICATION]
        assert len(auth_checks) > 0
        
        # Check for network checks
        network_checks = [c for c in checklist.checks if c.category == CheckCategory.NETWORK]
        assert len(network_checks) > 0
        
        # Check for data protection checks
        data_checks = [c for c in checklist.checks if c.category == CheckCategory.DATA_PROTECTION]
        assert len(data_checks) > 0
    
    def test_get_check_by_id(self, checklist):
        """Test getting a check by ID"""
        check = checklist.get_check_by_id("auth_jwt_expiry")
        
        assert check is not None
        assert check.check_id == "auth_jwt_expiry"
    
    def test_get_checks_by_category(self, checklist):
        """Test getting checks by category"""
        auth_checks = checklist.get_checks_by_category(CheckCategory.AUTHENTICATION)
        
        assert len(auth_checks) > 0
        for check in auth_checks:
            assert check.category == CheckCategory.AUTHENTICATION
    
    def test_get_checks_by_severity(self, checklist):
        """Test getting checks by severity"""
        critical_checks = checklist.get_checks_by_severity(CheckSeverity.CRITICAL)
        
        assert len(critical_checks) > 0
        for check in critical_checks:
            assert check.severity == CheckSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_run_audit(self, checklist):
        """Test running an audit"""
        report = await checklist.run_audit()
        
        assert report is not None
        assert isinstance(report, AuditReport)
        assert len(report.results) > 0
        assert "total" in report.summary
    
    @pytest.mark.asyncio
    async def test_run_audit_with_categories(self, checklist):
        """Test running audit with specific categories"""
        report = await checklist.run_audit(
            categories=[CheckCategory.AUTHENTICATION]
        )
        
        # All results should be from authentication category
        for result in report.results:
            check = checklist.get_check_by_id(result.check_id)
            if check:
                assert check.category == CheckCategory.AUTHENTICATION


# ===== SecurityScanner Tests =====

class TestVulnerability:
    """Tests for Vulnerability"""
    
    def test_vulnerability_creation(self):
        """Test creating a vulnerability"""
        vuln = Vulnerability(
            id="CVE-2024-1234",
            package="test-package",
            version="1.0.0",
            severity=VulnerabilitySeverity.HIGH,
            description="Test vulnerability",
            recommendation="Upgrade to version 2.0.0",
        )
        
        assert vuln.id == "CVE-2024-1234"
        assert vuln.package == "test-package"
        assert vuln.severity == VulnerabilitySeverity.HIGH
    
    def test_vulnerability_to_dict(self):
        """Test converting vulnerability to dictionary"""
        vuln = Vulnerability(
            id="CVE-2024-1234",
            package="test",
            version="1.0",
            severity=VulnerabilitySeverity.CRITICAL,
            description="Critical vulnerability",
        )
        
        vuln_dict = vuln.to_dict()
        
        assert vuln_dict["id"] == "CVE-2024-1234"
        assert vuln_dict["severity"] == "critical"


class TestSecurityIssue:
    """Tests for SecurityIssue"""
    
    def test_security_issue_creation(self):
        """Test creating a security issue"""
        issue = SecurityIssue(
            issue_id="issue1",
            issue_type=IssueType.CODE,
            file_path="/path/to/file.py",
            line_number=42,
            severity=VulnerabilitySeverity.MEDIUM,
            message="Potential security issue",
            cwe="CWE-123",
        )
        
        assert issue.issue_id == "issue1"
        assert issue.issue_type == IssueType.CODE
        assert issue.line_number == 42
        assert issue.cwe == "CWE-123"


class TestSecretLeak:
    """Tests for SecretLeak"""
    
    def test_secret_leak_creation(self):
        """Test creating a secret leak"""
        leak = SecretLeak(
            secret_id="secret1",
            secret_type="API Key",
            file_path="/path/to/config.py",
            line_number=10,
            matched_pattern="api_key = '...'",
            severity=VulnerabilitySeverity.HIGH,
        )
        
        assert leak.secret_id == "secret1"
        assert leak.secret_type == "API Key"
        assert leak.severity == VulnerabilitySeverity.HIGH


class TestScanResult:
    """Tests for ScanResult"""
    
    def test_scan_result_creation(self):
        """Test creating a scan result"""
        result = ScanResult(scan_id="scan1")
        
        assert result.scan_id == "scan1"
        assert len(result.vulnerabilities) == 0
        assert len(result.issues) == 0
        assert len(result.secrets) == 0
    
    def test_add_vulnerability(self):
        """Test adding a vulnerability"""
        result = ScanResult(scan_id="scan1")
        vuln = Vulnerability(
            id="CVE-2024-1234",
            package="test",
            version="1.0",
            severity=VulnerabilitySeverity.HIGH,
            description="Test",
        )
        
        result.add_vulnerability(vuln)
        
        assert len(result.vulnerabilities) == 1
    
    def test_calculate_summary(self):
        """Test calculating summary"""
        result = ScanResult(scan_id="scan1")
        
        result.add_vulnerability(Vulnerability(
            id="CVE-1", package="a", version="1.0",
            severity=VulnerabilitySeverity.HIGH, description="Test"
        ))
        result.add_vulnerability(Vulnerability(
            id="CVE-2", package="b", version="1.0",
            severity=VulnerabilitySeverity.MEDIUM, description="Test"
        ))
        result.add_issue(SecurityIssue(
            issue_id="issue1", issue_type=IssueType.CODE,
            file_path="test.py", line_number=1,
            severity=VulnerabilitySeverity.LOW, message="Test"
        ))
        
        summary = result.calculate_summary()
        
        assert summary["total_vulnerabilities"] == 2
        assert summary["total_issues"] == 1
        assert summary["vulnerabilities_by_severity"]["high"] == 1
        assert summary["vulnerabilities_by_severity"]["medium"] == 1


class TestSecurityScanner:
    """Tests for SecurityScanner"""
    
    @pytest.fixture
    def scanner(self, tmp_path):
        """Create a security scanner"""
        return SecurityScanner(project_root=str(tmp_path))
    
    def test_scanner_initialization(self, scanner):
        """Test scanner initialization"""
        assert scanner.project_root is not None
        assert len(scanner.SECRET_PATTERNS) > 0
        assert len(scanner.EXCLUDED_PATHS) > 0
    
    @pytest.mark.asyncio
    async def test_scan_dependencies(self, scanner):
        """Test dependency scanning"""
        vulns = await scanner.scan_dependencies()
        
        assert isinstance(vulns, list)
    
    @pytest.mark.asyncio
    async def test_scan_code(self, scanner):
        """Test code scanning"""
        issues = await scanner.scan_code()
        
        assert isinstance(issues, list)
    
    @pytest.mark.asyncio
    async def test_scan_secrets(self, scanner):
        """Test secret scanning"""
        secrets = await scanner.scan_secrets()
        
        assert isinstance(secrets, list)
    
    @pytest.mark.asyncio
    async def test_scan_all(self, scanner):
        """Test full scan"""
        result = await scanner.scan_all()
        
        assert isinstance(result, ScanResult)
        assert result.scan_id is not None
        assert result.duration_ms >= 0
    
    def test_export_results_json(self, scanner):
        """Test exporting results as JSON"""
        result = ScanResult(scan_id="test")
        json_output = scanner.export_results(result, "json")
        
        assert '"scan_id": "test"' in json_output
    
    def test_export_results_csv(self, scanner):
        """Test exporting results as CSV"""
        result = ScanResult(scan_id="test")
        csv_output = scanner.export_results(result, "csv")
        
        assert "type,id,severity" in csv_output


# ===== PenetrationTestSuite Tests =====

class TestSecurityFinding:
    """Tests for SecurityFinding"""
    
    def test_finding_creation(self):
        """Test creating a security finding"""
        finding = SecurityFinding(
            finding_id="finding1",
            test_type=TestType.AUTHENTICATION_BYPASS,
            severity=FindingSeverity.HIGH,
            title="Authentication Bypass",
            description="Potential authentication bypass vulnerability",
        )
        
        assert finding.finding_id == "finding1"
        assert finding.test_type == TestType.AUTHENTICATION_BYPASS
        assert finding.severity == FindingSeverity.HIGH
    
    def test_finding_to_dict(self):
        """Test converting finding to dictionary"""
        finding = SecurityFinding(
            finding_id="f1",
            test_type=TestType.INJECTION_ATTACK,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection",
            description="SQL injection vulnerability",
        )
        
        finding_dict = finding.to_dict()
        
        assert finding_dict["finding_id"] == "f1"
        assert finding_dict["test_type"] == "injection_attack"
        assert finding_dict["severity"] == "critical"


class TestPenTestResult:
    """Tests for PenTestResult"""
    
    def test_result_creation(self):
        """Test creating a penetration test result"""
        result = PenTestResult(
            test_id="test1",
            test_type=TestType.AUTHENTICATION_BYPASS,
        )
        
        assert result.test_id == "test1"
        assert result.test_type == TestType.AUTHENTICATION_BYPASS
        assert result.passed is True
        assert len(result.findings) == 0
    
    def test_add_finding(self):
        """Test adding a finding"""
        result = PenTestResult(
            test_id="test1",
            test_type=TestType.AUTHENTICATION_BYPASS,
        )
        
        # Add critical finding - should mark as failed
        result.add_finding(SecurityFinding(
            finding_id="f1",
            test_type=TestType.AUTHENTICATION_BYPASS,
            severity=FindingSeverity.CRITICAL,
            title="Critical Issue",
            description="Critical security issue",
        ))
        
        assert len(result.findings) == 1
        assert result.passed is False


class TestPenetrationTestSuite:
    """Tests for PenetrationTestSuite"""
    
    @pytest.fixture
    def pentest(self):
        """Create a penetration test suite"""
        return PenetrationTestSuite(base_url="http://localhost:8000")
    
    def test_pentest_initialization(self, pentest):
        """Test penetration test suite initialization"""
        assert pentest.base_url == "http://localhost:8000"
        assert len(pentest.get_test_types()) > 0
    
    def test_get_test_types(self, pentest):
        """Test getting available test types"""
        test_types = pentest.get_test_types()
        
        assert TestType.AUTHENTICATION_BYPASS in test_types
        assert TestType.INJECTION_ATTACK in test_types
        assert TestType.RATE_LIMITING in test_types
    
    @pytest.mark.asyncio
    async def test_test_authentication_bypass(self, pentest):
        """Test authentication bypass testing"""
        result = await pentest.test_authentication_bypass()
        
        assert isinstance(result, PenTestResult)
        assert result.test_type == TestType.AUTHENTICATION_BYPASS
    
    @pytest.mark.asyncio
    async def test_test_injection_attacks(self, pentest):
        """Test injection attack testing"""
        result = await pentest.test_injection_attacks()
        
        assert isinstance(result, PenTestResult)
        assert result.test_type == TestType.INJECTION_ATTACK
    
    @pytest.mark.asyncio
    async def test_test_rate_limiting(self, pentest):
        """Test rate limiting testing"""
        result = await pentest.test_rate_limiting()
        
        assert isinstance(result, PenTestResult)
        assert result.test_type == TestType.RATE_LIMITING
    
    @pytest.mark.asyncio
    async def test_test_access_control(self, pentest):
        """Test access control testing"""
        result = await pentest.test_access_control()
        
        assert isinstance(result, PenTestResult)
        assert result.test_type == TestType.ACCESS_CONTROL
    
    @pytest.mark.asyncio
    async def test_run_all_tests(self, pentest):
        """Test running all penetration tests"""
        report = await pentest.run_all_tests()
        
        assert isinstance(report, PenTestReport)
        assert report.report_id is not None
        assert len(report.results) > 0
        assert "total_tests" in report.summary
    
    def test_export_report_json(self, pentest):
        """Test exporting report as JSON"""
        report = PenTestReport(report_id="test")
        json_output = pentest.export_report(report, "json")
        
        assert '"report_id": "test"' in json_output
    
    def test_export_report_csv(self, pentest):
        """Test exporting report as CSV"""
        report = PenTestReport(report_id="test")
        csv_output = pentest.export_report(report, "csv")
        
        assert "test_type,finding_id" in csv_output


# ===== SecretsManager Tests =====

class TestSecretMetadata:
    """Tests for SecretMetadata"""
    
    def test_metadata_creation(self):
        """Test creating secret metadata"""
        metadata = SecretMetadata(
            key="test_secret",
            backend=SecretBackend.ENV,
        )
        
        assert metadata.key == "test_secret"
        assert metadata.backend == SecretBackend.ENV
        assert isinstance(metadata.created_at, datetime)


class TestSecretValue:
    """Tests for SecretValue"""
    
    def test_secret_value_creation(self):
        """Test creating a secret value"""
        metadata = SecretMetadata(key="test", backend=SecretBackend.ENV)
        value = SecretValue(value="secret123", metadata=metadata)
        
        assert value.value == "secret123"
        assert value.metadata.key == "test"
        assert len(value.checksum) > 0
    
    def test_is_expired(self):
        """Test checking if secret is expired"""
        from datetime import timedelta
        
        # Not expired
        metadata = SecretMetadata(key="test", backend=SecretBackend.ENV)
        value = SecretValue(value="secret", metadata=metadata)
        assert value.is_expired() is False
        
        # Expired
        metadata_expired = SecretMetadata(
            key="test",
            backend=SecretBackend.ENV,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        value_expired = SecretValue(value="secret", metadata=metadata_expired)
        assert value_expired.is_expired() is True


class TestEnvironmentBackend:
    """Tests for EnvironmentBackend"""
    
    @pytest.fixture
    def backend(self):
        """Create an environment backend"""
        return EnvironmentBackend()
    
    @pytest.mark.asyncio
    async def test_get_secret(self, backend):
        """Test getting a secret from environment"""
        os.environ["TEST_SECRET"] = "test_value"
        
        value = await backend.get_secret("TEST_SECRET")
        
        assert value == "test_value"
        
        del os.environ["TEST_SECRET"]
    
    @pytest.mark.asyncio
    async def test_get_secret_not_found(self, backend):
        """Test getting a non-existent secret"""
        value = await backend.get_secret("NONEXISTENT_SECRET")
        
        assert value is None
    
    @pytest.mark.asyncio
    async def test_set_secret(self, backend):
        """Test setting a secret in environment"""
        await backend.set_secret("NEW_SECRET", "new_value")
        
        assert os.environ.get("NEW_SECRET") == "new_value"
        
        del os.environ["NEW_SECRET"]
    
    @pytest.mark.asyncio
    async def test_delete_secret(self, backend):
        """Test deleting a secret"""
        os.environ["DELETE_SECRET"] = "value"
        
        result = await backend.delete_secret("DELETE_SECRET")
        
        assert result is True
        assert "DELETE_SECRET" not in os.environ
    
    @pytest.mark.asyncio
    async def test_list_secrets(self, backend):
        """Test listing secrets"""
        os.environ["LIST_SECRET_1"] = "value1"
        os.environ["LIST_SECRET_2"] = "value2"
        
        secrets = await backend.list_secrets()
        
        assert "LIST_SECRET_1" in secrets
        assert "LIST_SECRET_2" in secrets
        
        del os.environ["LIST_SECRET_1"]
        del os.environ["LIST_SECRET_2"]


class TestSecretsManager:
    """Tests for SecretsManager"""
    
    @pytest.fixture
    async def manager(self):
        """Create a secrets manager"""
        manager = SecretsManager(backend=SecretBackend.ENV)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test secrets manager initialization"""
        assert manager.backend == SecretBackend.ENV
        assert manager._backend_impl is not None
    
    @pytest.mark.asyncio
    async def test_get_secret(self, manager):
        """Test getting a secret"""
        os.environ["TEST_SECRET"] = "test_value"
        
        value = await manager.get_secret("TEST_SECRET")
        
        assert value == "test_value"
        
        del os.environ["TEST_SECRET"]
    
    @pytest.mark.asyncio
    async def test_get_secret_or_raise(self, manager):
        """Test getting a secret or raising exception"""
        os.environ["EXISTING_SECRET"] = "value"
        
        value = await manager.get_secret_or_raise("EXISTING_SECRET")
        assert value == "value"
        
        with pytest.raises(ValueError):
            await manager.get_secret_or_raise("NONEXISTENT_SECRET")
        
        del os.environ["EXISTING_SECRET"]
    
    @pytest.mark.asyncio
    async def test_validate_secrets(self, manager):
        """Test validating secrets"""
        # Set required secrets
        os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_12345678"
        os.environ["DATABASE_URL"] = "postgresql://localhost/test"
        os.environ["ENCRYPTION_KEY"] = "test_encryption_key_12345678"
        
        validation = await manager.validate_secrets()
        
        assert isinstance(validation, dict)
        assert "JWT_SECRET_KEY" in validation
        
        del os.environ["JWT_SECRET_KEY"]
        del os.environ["DATABASE_URL"]
        del os.environ["ENCRYPTION_KEY"]
    
    @pytest.mark.asyncio
    async def test_check_secret_strength(self, manager):
        """Test checking secret strength"""
        # Strong secret
        strong = await manager.check_secret_strength("test", "StrongP@ssw0rd123!")
        assert strong["strength"] in ["strong", "very_strong"]
        
        # Weak secret
        weak = await manager.check_secret_strength("test", "password")
        assert weak["strength"] in ["weak", "very_weak"]
    
    @pytest.mark.asyncio
    async def test_generate_secret(self, manager):
        """Test generating a secret"""
        secret = await manager.generate_secret(length=32)
        
        assert len(secret) == 32
    
    @pytest.mark.asyncio
    async def test_cache(self, manager):
        """Test secret caching"""
        os.environ["CACHED_SECRET"] = "cached_value"
        
        # First call - should cache
        value1 = await manager.get_secret("CACHED_SECRET")
        
        # Second call - should use cache
        value2 = manager.get_cached_secret("CACHED_SECRET")
        
        assert value1 == value2
        
        # Clear cache
        manager.clear_cache()
        cached = manager.get_cached_secret("CACHED_SECRET")
        assert cached is None
        
        del os.environ["CACHED_SECRET"]


# ===== SecureEnvironment Tests =====

class TestSecretStrengthResult:
    """Tests for SecretStrengthResult"""
    
    def test_result_creation(self):
        """Test creating a strength result"""
        result = SecretStrengthResult(
            key="test_key",
            exists=True,
            length=16,
            strength=SecretStrength.STRONG,
            score=75,
        )
        
        assert result.key == "test_key"
        assert result.exists is True
        assert result.length == 16
        assert result.strength == SecretStrength.STRONG
    
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = SecretStrengthResult(
            key="test",
            exists=True,
            length=20,
            strength=SecretStrength.MEDIUM,
            score=50,
            issues=["No special characters"],
            recommendations=["Add special characters"],
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["key"] == "test"
        assert result_dict["strength"] == "medium"
        assert "No special characters" in result_dict["issues"]


class TestEnvironmentValidationResult:
    """Tests for EnvironmentValidationResult"""
    
    def test_result_creation(self):
        """Test creating a validation result"""
        result = EnvironmentValidationResult()
        
        assert result.valid is True
        assert len(result.missing_required) == 0
        assert len(result.errors) == 0
    
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = EnvironmentValidationResult(
            valid=False,
            missing_required=["SECRET_KEY"],
            errors=["Missing required secret"],
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["valid"] is False
        assert "SECRET_KEY" in result_dict["missing_required"]


class TestSecureEnvironment:
    """Tests for SecureEnvironment"""
    
    @pytest.fixture
    def env(self):
        """Create a secure environment"""
        return SecureEnvironment()
    
    def test_initialization(self, env):
        """Test environment initialization"""
        assert len(env.REQUIRED_SECRETS) > 0
        assert len(env.RECOMMENDED_SECRETS) > 0
        assert len(env.SECURITY_CONFIG) > 0
    
    def test_validate_environment(self, env):
        """Test environment validation"""
        # Set required secrets
        os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_12345678"
        os.environ["DATABASE_URL"] = "postgresql://localhost/test"
        os.environ["ENCRYPTION_KEY"] = "test_encryption_key_12345678"
        
        result = env.validate_environment(check_secrets=False)
        
        assert isinstance(result, EnvironmentValidationResult)
        
        del os.environ["JWT_SECRET_KEY"]
        del os.environ["DATABASE_URL"]
        del os.environ["ENCRYPTION_KEY"]
    
    def test_check_secret_strength(self, env):
        """Test checking secret strength"""
        # Strong secret
        result = env.check_secret_strength("test", "StrongP@ssw0rd123!")
        assert result.strength in [SecretStrength.STRONG, SecretStrength.VERY_STRONG]
        
        # Weak secret
        result = env.check_secret_strength("test", "password")
        assert result.strength in [SecretStrength.WEAK, SecretStrength.VERY_WEAK]
        
        # Non-existent secret
        result = env.check_secret_strength("nonexistent")
        assert result.exists is False
    
    def test_is_production(self, env):
        """Test checking if production environment"""
        os.environ["APP_ENV"] = "production"
        assert env.is_production() is True
        
        os.environ["APP_ENV"] = "development"
        assert env.is_production() is False
    
    def test_get_config(self, env):
        """Test getting configuration values"""
        os.environ["TEST_CONFIG"] = "test_value"
        
        value = env.get("TEST_CONFIG")
        assert value == "test_value"
        
        value = env.get("NONEXISTENT", "default")
        assert value == "default"
        
        del os.environ["TEST_CONFIG"]
    
    def test_get_int(self, env):
        """Test getting integer configuration"""
        os.environ["INT_CONFIG"] = "42"
        
        value = env.get_int("INT_CONFIG")
        assert value == 42
        
        value = env.get_int("NONEXISTENT", 10)
        assert value == 10
        
        del os.environ["INT_CONFIG"]
    
    def test_get_bool(self, env):
        """Test getting boolean configuration"""
        os.environ["BOOL_CONFIG"] = "true"
        
        value = env.get_bool("BOOL_CONFIG")
        assert value is True
        
        os.environ["BOOL_CONFIG"] = "false"
        value = env.get_bool("BOOL_CONFIG")
        assert value is False
        
        del os.environ["BOOL_CONFIG"]
    
    def test_get_list(self, env):
        """Test getting list configuration"""
        os.environ["LIST_CONFIG"] = "a, b, c"
        
        value = env.get_list("LIST_CONFIG")
        assert value == ["a", "b", "c"]
        
        del os.environ["LIST_CONFIG"]
    
    def test_generate_env_template(self, env):
        """Test generating environment template"""
        template = env.generate_env_template()
        
        assert "JWT_SECRET_KEY" in template
        assert "DATABASE_URL" in template
        assert "ENCRYPTION_KEY" in template
    
    def test_export_validation_report(self, env):
        """Test exporting validation report"""
        # Validate first
        env.validate_environment(check_secrets=False)
        
        report = env.export_validation_report()
        
        assert "PRSM Environment Validation Report" in report


# ===== Integration Tests =====

class TestSecurityIntegration:
    """Integration tests for security modules"""
    
    @pytest.mark.asyncio
    async def test_full_security_audit(self):
        """Test running a full security audit"""
        # Create checklist
        checklist = SecurityAuditChecklist()
        
        # Run audit
        report = await checklist.run_audit()
        
        # Verify report
        assert len(report.results) > 0
        assert "total" in report.summary
        
        # Generate recommendations
        recommendations = report.generate_recommendations()
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_full_security_scan(self, tmp_path):
        """Test running a full security scan"""
        # Create scanner
        scanner = SecurityScanner(project_root=str(tmp_path))
        
        # Run scan
        result = await scanner.scan_all()
        
        # Verify result
        assert result.scan_id is not None
        assert result.duration_ms >= 0
        assert "risk_level" in result.summary
    
    @pytest.mark.asyncio
    async def test_full_penetration_test(self):
        """Test running a full penetration test"""
        # Create test suite
        pentest = PenetrationTestSuite()
        
        # Run all tests
        report = await pentest.run_all_tests()
        
        # Verify report
        assert report.report_id is not None
        assert len(report.results) > 0
        assert "total_tests" in report.summary
    
    @pytest.mark.asyncio
    async def test_secrets_management_flow(self):
        """Test secrets management flow"""
        # Create manager
        manager = SecretsManager(backend=SecretBackend.ENV)
        await manager.initialize()
        
        # Set secret
        await manager.set_secret("TEST_FLOW_SECRET", "test_value")
        
        # Get secret
        value = await manager.get_secret("TEST_FLOW_SECRET")
        assert value == "test_value"
        
        # Check strength
        strength = await manager.check_secret_strength("TEST_FLOW_SECRET")
        assert strength["exists"] is True
        
        # Delete secret
        await manager.delete_secret("TEST_FLOW_SECRET")
        
        # Verify deleted
        value = await manager.get_secret("TEST_FLOW_SECRET")
        assert value is None
    
    def test_environment_validation_flow(self):
        """Test environment validation flow"""
        # Create environment
        env = SecureEnvironment()
        
        # Set required secrets
        os.environ["JWT_SECRET_KEY"] = "test_jwt_secret_key_12345678"
        os.environ["DATABASE_URL"] = "postgresql://localhost/test"
        os.environ["ENCRYPTION_KEY"] = "test_encryption_key_12345678"
        
        # Validate
        result = env.validate_environment(check_secrets=True)
        
        # Check result
        assert isinstance(result, EnvironmentValidationResult)
        
        # Generate report
        report = env.export_validation_report()
        assert "PRSM Environment Validation Report" in report
        
        # Clean up
        del os.environ["JWT_SECRET_KEY"]
        del os.environ["DATABASE_URL"]
        del os.environ["ENCRYPTION_KEY"]


class TestCORSSecurityValidation:
    """Tests for CORS security validation"""

    def test_cors_wildcard_blocked_in_production(self):
        """Test that SecurityConfig with allowed_origins=['*'] raises ValueError in production"""
        from prsm.core.config.schemas import SecurityConfig
        
        # Save original env
        original_env = os.environ.get("PRSM_ENV")
        
        try:
            # Set production environment
            os.environ["PRSM_ENV"] = "production"
            
            # This should raise ValueError because wildcard is not allowed in production
            with pytest.raises(ValueError) as exc_info:
                SecurityConfig(allowed_origins=["*"])
            
            assert "Wildcard CORS origin" in str(exc_info.value)
            assert "not permitted in production" in str(exc_info.value)
        finally:
            # Restore original env
            if original_env is not None:
                os.environ["PRSM_ENV"] = original_env
            elif "PRSM_ENV" in os.environ:
                del os.environ["PRSM_ENV"]

    def test_configure_cors_uses_env_var(self):
        """Test that configure_cors() uses CORS_ORIGINS environment variable"""
        from prsm.core.security.middleware import configure_cors
        
        # Save original env
        original_cors = os.environ.get("CORS_ORIGINS")
        
        try:
            # Set CORS_ORIGINS env var
            os.environ["CORS_ORIGINS"] = "https://test.example.com"
            
            # Call configure_cors
            result = configure_cors()
            
            # Assert result contains only the env var origin
            assert result["allow_origins"] == ["https://test.example.com"]
            assert result["allow_credentials"] is True
            assert "GET" in result["allow_methods"]
            assert "Authorization" in result["allow_headers"]
        finally:
            # Restore original env
            if original_cors is not None:
                os.environ["CORS_ORIGINS"] = original_cors
            elif "CORS_ORIGINS" in os.environ:
                del os.environ["CORS_ORIGINS"]

    def test_configure_cors_no_wildcard_in_production(self):
        """Test that configure_cors() raises RuntimeError with wildcard in production"""
        from prsm.core.security.middleware import configure_cors
        
        # Save original env vars
        original_env = os.environ.get("PRSM_ENV")
        original_cors = os.environ.get("CORS_ORIGINS")
        
        try:
            # Set production environment and wildcard CORS
            os.environ["PRSM_ENV"] = "production"
            os.environ["CORS_ORIGINS"] = "*"
            
            # This should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                configure_cors()
            
            assert "Wildcard CORS origin is not permitted outside development" in str(exc_info.value)
        finally:
            # Restore original env vars
            if original_env is not None:
                os.environ["PRSM_ENV"] = original_env
            elif "PRSM_ENV" in os.environ:
                del os.environ["PRSM_ENV"]
            
            if original_cors is not None:
                os.environ["CORS_ORIGINS"] = original_cors
            elif "CORS_ORIGINS" in os.environ:
                del os.environ["CORS_ORIGINS"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])