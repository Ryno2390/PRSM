"""
Security Scanner

Automated security scanning for PRSM.
Scans dependencies, code, and secrets for security issues.
"""

import asyncio
import re
import subprocess
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger(__name__)


class VulnerabilitySeverity(Enum):
    """Severity levels for vulnerabilities"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(Enum):
    """Types of security issues"""
    DEPENDENCY = "dependency"
    CODE = "code"
    SECRET = "secret"
    CONFIGURATION = "configuration"
    BEST_PRACTICE = "best_practice"


@dataclass
class Vulnerability:
    """
    Represents a dependency vulnerability.
    
    Attributes:
        id: Vulnerability identifier (CVE, etc.)
        package: Affected package name
        version: Affected version
        severity: Vulnerability severity
        description: Vulnerability description
        recommendation: Fix recommendation
        references: External references
    """
    id: str
    package: str
    version: str
    severity: VulnerabilitySeverity
    description: str
    recommendation: str = ""
    references: List[str] = field(default_factory=list)
    cvss_score: Optional[float] = None
    fixed_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "package": self.package,
            "version": self.version,
            "severity": self.severity.value,
            "description": self.description,
            "recommendation": self.recommendation,
            "references": self.references,
            "cvss_score": self.cvss_score,
            "fixed_version": self.fixed_version,
        }


@dataclass
class SecurityIssue:
    """
    Represents a code security issue.
    
    Attributes:
        issue_id: Unique identifier
        issue_type: Type of issue
        file_path: File where issue was found
        line_number: Line number
        severity: Issue severity
        message: Issue description
        code_snippet: Relevant code snippet
        recommendation: Fix recommendation
        cwe: CWE identifier
        owasp: OWASP category
    """
    issue_id: str
    issue_type: IssueType
    file_path: str
    line_number: int
    severity: VulnerabilitySeverity
    message: str
    code_snippet: str = ""
    recommendation: str = ""
    cwe: Optional[str] = None
    owasp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "severity": self.severity.value,
            "message": self.message,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
            "cwe": self.cwe,
            "owasp": self.owasp,
        }


@dataclass
class SecretLeak:
    """
    Represents a leaked secret.
    
    Attributes:
        secret_id: Unique identifier
        secret_type: Type of secret
        file_path: File where secret was found
        line_number: Line number
        matched_pattern: Pattern that matched
        severity: Severity level
        recommendation: Fix recommendation
    """
    secret_id: str
    secret_type: str
    file_path: str
    line_number: int
    matched_pattern: str
    severity: VulnerabilitySeverity
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "secret_id": self.secret_id,
            "secret_type": self.secret_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "matched_pattern": self.matched_pattern,
            "severity": self.severity.value,
            "recommendation": self.recommendation,
        }


@dataclass
class ScanResult:
    """
    Result of a security scan.
    
    Attributes:
        scan_id: Unique identifier
        timestamp: When scan was run
        vulnerabilities: List of vulnerabilities found
        issues: List of security issues found
        secrets: List of secret leaks found
        summary: Summary statistics
        duration_ms: Scan duration in milliseconds
    """
    scan_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    issues: List[SecurityIssue] = field(default_factory=list)
    secrets: List[SecretLeak] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    
    def add_vulnerability(self, vuln: Vulnerability) -> None:
        """Add a vulnerability"""
        self.vulnerabilities.append(vuln)
    
    def add_issue(self, issue: SecurityIssue) -> None:
        """Add a security issue"""
        self.issues.append(issue)
    
    def add_secret(self, secret: SecretLeak) -> None:
        """Add a secret leak"""
        self.secrets.append(secret)
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        summary = {
            "total_vulnerabilities": len(self.vulnerabilities),
            "total_issues": len(self.issues),
            "total_secrets": len(self.secrets),
            "vulnerabilities_by_severity": {},
            "issues_by_severity": {},
            "secrets_by_type": {},
        }
        
        # Count vulnerabilities by severity
        for vuln in self.vulnerabilities:
            sev = vuln.severity.value
            summary["vulnerabilities_by_severity"][sev] = \
                summary["vulnerabilities_by_severity"].get(sev, 0) + 1
        
        # Count issues by severity
        for issue in self.issues:
            sev = issue.severity.value
            summary["issues_by_severity"][sev] = \
                summary["issues_by_severity"].get(sev, 0) + 1
        
        # Count secrets by type
        for secret in self.secrets:
            stype = secret.secret_type
            summary["secrets_by_type"][stype] = \
                summary["secrets_by_type"].get(stype, 0) + 1
        
        # Calculate overall risk score
        severity_weights = {
            VulnerabilitySeverity.CRITICAL: 10,
            VulnerabilitySeverity.HIGH: 7,
            VulnerabilitySeverity.MEDIUM: 4,
            VulnerabilitySeverity.LOW: 1,
            VulnerabilitySeverity.INFO: 0,
        }
        
        risk_score = 0
        for vuln in self.vulnerabilities:
            risk_score += severity_weights.get(vuln.severity, 0)
        for issue in self.issues:
            risk_score += severity_weights.get(issue.severity, 0)
        for secret in self.secrets:
            risk_score += severity_weights.get(secret.severity, 0)
        
        summary["risk_score"] = risk_score
        
        # Determine risk level
        if risk_score >= 50:
            summary["risk_level"] = "critical"
        elif risk_score >= 30:
            summary["risk_level"] = "high"
        elif risk_score >= 15:
            summary["risk_level"] = "medium"
        elif risk_score > 0:
            summary["risk_level"] = "low"
        else:
            summary["risk_level"] = "none"
        
        self.summary = summary
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scan_id": self.scan_id,
            "timestamp": self.timestamp.isoformat(),
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "issues": [i.to_dict() for i in self.issues],
            "secrets": [s.to_dict() for s in self.secrets],
            "summary": self.summary,
            "duration_ms": self.duration_ms,
        }


class SecurityScanner:
    """
    Automated security scanner for PRSM.
    
    Provides:
    - Dependency vulnerability scanning
    - Code security analysis
    - Secret leak detection
    - Configuration validation
    """
    
    # Secret patterns to detect
    SECRET_PATTERNS = [
        # API Keys
        (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9]{20,})["\']?', "API Key"),
        (r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?([a-zA-Z0-9]{20,})["\']?', "Secret Key"),
        (r'(?i)(access[_-]?token|accesstoken)\s*[=:]\s*["\']?([a-zA-Z0-9]{20,})["\']?', "Access Token"),
        
        # AWS
        (r'AKIA[0-9A-Z]{16}', "AWS Access Key"),
        (r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', "AWS Secret Key"),
        
        # Private Keys
        (r'-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----', "Private Key"),
        (r'-----BEGIN PGP PRIVATE KEY BLOCK-----', "PGP Private Key"),
        
        # Database URLs
        (r'(?:postgres|mysql|mongodb|redis)://[^\s\'"<>]+:[^\s\'"<>]+@[^\s\'"<>]+', "Database URL"),
        
        # JWT Secrets
        (r'(?i)jwt[_-]?secret\s*[=:]\s*["\']?([a-zA-Z0-9]{20,})["\']?', "JWT Secret"),
        
        # Passwords in config
        (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^\s\'"<>]{8,})["\']?', "Password"),
        
        # Generic secrets
        (r'(?i)(secret|token|key)\s*[=:]\s*["\']?([a-zA-Z0-9]{32,})["\']?', "Generic Secret"),
    ]
    
    # Files to exclude from scanning
    EXCLUDED_PATHS = {
        '.git',
        '__pycache__',
        'node_modules',
        '.venv',
        'venv',
        'env',
        '.env',
        'dist',
        'build',
        '.tox',
        '.pytest_cache',
        '.mypy_cache',
        'site-packages',
    }
    
    # File patterns to scan
    SCAN_PATTERNS = {
        '*.py',
        '*.yaml',
        '*.yml',
        '*.json',
        '*.env',
        '*.config',
        '*.ini',
        '*.toml',
    }
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the security scanner.
        
        Args:
            project_root: Root directory to scan
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._scan_id_counter = 0
    
    def _generate_scan_id(self) -> str:
        """Generate a unique scan ID"""
        self._scan_id_counter += 1
        return f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{self._scan_id_counter}"
    
    async def scan_all(self) -> ScanResult:
        """
        Run all security scans.
        
        Returns:
            ScanResult with all findings
        """
        
        scan_id = self._generate_scan_id()
        start_time = datetime.now(timezone.utc)
        
        result = ScanResult(scan_id=scan_id)
        
        # Run all scans in parallel
        vulns_task = self.scan_dependencies()
        issues_task = self.scan_code()
        secrets_task = self.scan_secrets()
        
        vulns, issues, secrets = await asyncio.gather(
            vulns_task, issues_task, secrets_task,
            return_exceptions=True
        )
        
        # Process results
        if isinstance(vulns, list):
            for v in vulns:
                result.add_vulnerability(v)
        elif isinstance(vulns, Exception):
            logger.error("Dependency scan failed", error=str(vulns))
        
        if isinstance(issues, list):
            for i in issues:
                result.add_issue(i)
        elif isinstance(issues, Exception):
            logger.error("Code scan failed", error=str(issues))
        
        if isinstance(secrets, list):
            for s in secrets:
                result.add_secret(s)
        elif isinstance(secrets, Exception):
            logger.error("Secret scan failed", error=str(secrets))
        
        # Calculate duration
        end_time = datetime.now(timezone.utc)
        result.duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Calculate summary
        result.calculate_summary()
        
        logger.info(
            "Security scan completed",
            scan_id=scan_id,
            vulnerabilities=len(result.vulnerabilities),
            issues=len(result.issues),
            secrets=len(result.secrets),
            risk_level=result.summary.get("risk_level", "unknown"),
        )
        
        return result
    
    async def scan_dependencies(self) -> List[Vulnerability]:
        """
        Scan dependencies for known vulnerabilities.
        
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        # Try using pip-audit if available
        try:
            vulns = await self._scan_with_pip_audit()
            vulnerabilities.extend(vulns)
        except Exception as e:
            logger.debug("pip-audit not available, trying safety", error=str(e))
            
            # Try using safety if available
            try:
                vulns = await self._scan_with_safety()
                vulnerabilities.extend(vulns)
            except Exception as e2:
                logger.debug("safety not available", error=str(e2))
        
        # Fallback: parse requirements files manually
        if not vulnerabilities:
            vulns = await self._scan_requirements_manually()
            vulnerabilities.extend(vulns)
        
        return vulnerabilities
    
    async def _scan_with_pip_audit(self) -> List[Vulnerability]:
        """Scan using pip-audit"""
        vulnerabilities = []
        
        try:
            # Run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format", "json", "--only-vuln"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=120,
            )
            
            if result.returncode == 0:
                return vulnerabilities
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data:
                        vuln = Vulnerability(
                            id=item.get("id", "unknown"),
                            package=item.get("name", "unknown"),
                            version=item.get("version", "unknown"),
                            severity=self._map_pip_audit_severity(item.get("severity", "")),
                            description=item.get("description", ""),
                            recommendation=f"Upgrade to version {item.get('fix_version', 'latest')}",
                            references=item.get("urls", []),
                            fixed_version=item.get("fix_version"),
                        )
                        vulnerabilities.append(vuln)
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            raise Exception("pip-audit not installed")
        except subprocess.TimeoutExpired:
            raise Exception("pip-audit timeout")
        
        return vulnerabilities
    
    async def _scan_with_safety(self) -> List[Vulnerability]:
        """Scan using safety"""
        vulnerabilities = []
        
        try:
            # Run safety
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=120,
            )
            
            if result.returncode == 0:
                return vulnerabilities
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data:
                        vuln = Vulnerability(
                            id=item.get("vulnerability_id", "unknown"),
                            package=item.get("package", "unknown"),
                            version=item.get("installed_version", "unknown"),
                            severity=VulnerabilitySeverity.HIGH,  # Safety doesn't provide severity
                            description=item.get("advisory", ""),
                            recommendation=item.get("more_info_url", ""),
                            references=[item.get("more_info_url", "")],
                        )
                        vulnerabilities.append(vuln)
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            raise Exception("safety not installed")
        except subprocess.TimeoutExpired:
            raise Exception("safety timeout")
        
        return vulnerabilities
    
    async def _scan_requirements_manually(self) -> List[Vulnerability]:
        """Manually check requirements for known vulnerable packages"""
        vulnerabilities = []
        
        # Known vulnerable packages (simplified check)
        known_vulnerabilities = {
            "requests": {"2.25.0": "CVE-2023-32681"},
            "urllib3": {"1.26.0": "CVE-2023-43804"},
            "pyyaml": {"5.0": "CVE-2020-14343"},
            "jinja2": {"2.10": "CVE-2020-28493"},
            "pillow": {"8.0.0": "CVE-2022-45498"},
        }
        
        requirements_files = ["requirements.txt", "requirements-dev.txt"]
        
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if not req_path.exists():
                continue
            
            try:
                content = req_path.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    # Parse package name and version
                    match = re.match(r'^([a-zA-Z0-9_-]+)\s*[=<>!~]+\s*([0-9.]+)', line)
                    if match:
                        package = match.group(1).lower()
                        version = match.group(2)
                        
                        if package in known_vulnerabilities:
                            if version in known_vulnerabilities[package]:
                                cve = known_vulnerabilities[package][version]
                                vuln = Vulnerability(
                                    id=cve,
                                    package=package,
                                    version=version,
                                    severity=VulnerabilitySeverity.HIGH,
                                    description=f"Known vulnerability in {package} {version}",
                                    recommendation=f"Upgrade {package} to latest version",
                                    references=[f"https://nvd.nist.gov/vuln/detail/{cve}"],
                                )
                                vulnerabilities.append(vuln)
            except Exception as e:
                logger.error(f"Failed to parse {req_file}", error=str(e))
        
        return vulnerabilities
    
    def _map_pip_audit_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map pip-audit severity to our severity levels"""
        severity_map = {
            "critical": VulnerabilitySeverity.CRITICAL,
            "high": VulnerabilitySeverity.HIGH,
            "medium": VulnerabilitySeverity.MEDIUM,
            "low": VulnerabilitySeverity.LOW,
            "info": VulnerabilitySeverity.INFO,
        }
        return severity_map.get(severity.lower(), VulnerabilitySeverity.MEDIUM)
    
    async def scan_code(self) -> List[SecurityIssue]:
        """
        Scan code for security issues.
        
        Returns:
            List of security issues found
        """
        issues = []
        
        # Try using bandit if available
        try:
            bandit_issues = await self._scan_with_bandit()
            issues.extend(bandit_issues)
        except Exception as e:
            logger.debug("bandit not available, using manual scan", error=str(e))
            
            # Fallback: manual code scan
            manual_issues = await self._scan_code_manually()
            issues.extend(manual_issues)
        
        return issues
    
    async def _scan_with_bandit(self) -> List[SecurityIssue]:
        """Scan using bandit"""
        issues = []
        
        try:
            # Run bandit
            result = subprocess.run(
                ["bandit", "-r", "-f", "json", str(self.project_root / "prsm")],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=300,
            )
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for item in data.get("results", []):
                        issue = SecurityIssue(
                            issue_id=f"bandit_{item.get('test_id', 'unknown')}",
                            issue_type=IssueType.CODE,
                            file_path=item.get("filename", ""),
                            line_number=item.get("line_number", 0),
                            severity=self._map_bandit_severity(item.get("issue_severity", "")),
                            message=item.get("issue_text", ""),
                            code_snippet=item.get("code", ""),
                            recommendation=item.get("more_info", ""),
                            cwe=item.get("cwe", {}).get("id") if isinstance(item.get("cwe"), dict) else None,
                        )
                        issues.append(issue)
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            raise Exception("bandit not installed")
        except subprocess.TimeoutExpired:
            raise Exception("bandit timeout")
        
        return issues
    
    async def _scan_code_manually(self) -> List[SecurityIssue]:
        """Manually scan code for common security issues"""
        issues = []
        
        # Patterns for common security issues
        security_patterns = [
            # SQL injection
            (r'execute\s*\(\s*[f]?["\'].*\+.*["\']', "SQL Injection", "CWE-89", "OWASP-A03"),
            (r'execute\s*\(\s*[f]?["\'].*%.*["\']', "SQL Injection", "CWE-89", "OWASP-A03"),
            
            # Hardcoded passwords
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded Password", "CWE-798", "OWASP-A07"),
            (r'passwd\s*=\s*["\'][^"\']+["\']', "Hardcoded Password", "CWE-798", "OWASP-A07"),
            
            # Debug mode
            (r'debug\s*=\s*True', "Debug Mode Enabled", "CWE-215", "OWASP-A05"),
            
            # Insecure random
            (r'random\.random\(\)', "Insecure Random Number", "CWE-338", None),
            
            # Pickle usage
            (r'pickle\.loads?\s*\(', "Pickle Deserialization", "CWE-502", "OWASP-A08"),
            
            # Eval usage
            (r'eval\s*\(', "Eval Usage", "CWE-95", "OWASP-A03"),
            
            # Shell injection
            (r'subprocess\..*shell\s*=\s*True', "Shell Injection", "CWE-78", "OWASP-A03"),
            
            # Insecure SSL
            (r'verify\s*=\s*False', "SSL Verification Disabled", "CWE-295", "OWASP-A07"),
            
            # Assert in production
            (r'assert\s+', "Assert Statement", "CWE-617", None),
        ]
        
        # Scan Python files
        for py_file in self.project_root.rglob("*.py"):
            # Skip excluded paths
            if any(excluded in py_file.parts for excluded in self.EXCLUDED_PATHS):
                continue
            
            try:
                content = py_file.read_text()
                lines = content.splitlines()
                
                for pattern, issue_name, cwe, owasp in security_patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            issue = SecurityIssue(
                                issue_id=f"manual_{py_file.name}_{i}",
                                issue_type=IssueType.CODE,
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=i,
                                severity=VulnerabilitySeverity.HIGH if "Injection" in issue_name or "Password" in issue_name else VulnerabilitySeverity.MEDIUM,
                                message=f"{issue_name} detected",
                                code_snippet=line.strip()[:100],
                                recommendation=f"Review and fix {issue_name.lower()}",
                                cwe=cwe,
                                owasp=owasp,
                            )
                            issues.append(issue)
            except Exception as e:
                logger.debug(f"Failed to scan {py_file}", error=str(e))
        
        return issues
    
    def _map_bandit_severity(self, severity: str) -> VulnerabilitySeverity:
        """Map bandit severity to our severity levels"""
        severity_map = {
            "HIGH": VulnerabilitySeverity.HIGH,
            "MEDIUM": VulnerabilitySeverity.MEDIUM,
            "LOW": VulnerabilitySeverity.LOW,
        }
        return severity_map.get(severity.upper(), VulnerabilitySeverity.MEDIUM)
    
    async def scan_secrets(self) -> List[SecretLeak]:
        """
        Scan for leaked secrets.
        
        Returns:
            List of secret leaks found
        """
        secrets = []
        
        # Try using detect-secrets if available
        try:
            detect_secrets = await self._scan_with_detect_secrets()
            secrets.extend(detect_secrets)
        except Exception as e:
            logger.debug("detect-secrets not available, using manual scan", error=str(e))
            
            # Fallback: manual secret scan
            manual_secrets = await self._scan_secrets_manually()
            secrets.extend(manual_secrets)
        
        return secrets
    
    async def _scan_with_detect_secrets(self) -> List[SecretLeak]:
        """Scan using detect-secrets"""
        secrets = []
        
        try:
            # Run detect-secrets
            result = subprocess.run(
                ["detect-secrets", "scan", str(self.project_root)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=300,
            )
            
            # Parse JSON output
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    for file_path, findings in data.get("results", {}).items():
                        for finding in findings:
                            secret = SecretLeak(
                                secret_id=f"secret_{finding.get('type', 'unknown')}_{finding.get('line_number', 0)}",
                                secret_type=finding.get("type", "unknown"),
                                file_path=file_path,
                                line_number=finding.get("line_number", 0),
                                matched_pattern=finding.get("hashed_secret", "")[:20],
                                severity=VulnerabilitySeverity.HIGH,
                                recommendation="Remove secret and use environment variables or secrets manager",
                            )
                            secrets.append(secret)
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            raise Exception("detect-secrets not installed")
        except subprocess.TimeoutExpired:
            raise Exception("detect-secrets timeout")
        
        return secrets
    
    async def _scan_secrets_manually(self) -> List[SecretLeak]:
        """Manually scan for secrets"""
        secrets = []
        
        # Scan all files
        for file_path in self.project_root.rglob("*"):
            # Skip excluded paths
            if any(excluded in file_path.parts for excluded in self.EXCLUDED_PATHS):
                continue
            
            # Only scan text files
            if not file_path.is_file():
                continue
            
            # Skip binary files
            try:
                content = file_path.read_text()
            except (UnicodeDecodeError, IOError):
                continue
            
            lines = content.splitlines()
            
            for pattern, secret_type in self.SECRET_PATTERNS:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        # Mask the actual secret
                        masked_line = re.sub(
                            pattern,
                            lambda m: m.group(0)[:10] + "..." + "[MASKED]",
                            line
                        )
                        
                        secret = SecretLeak(
                            secret_id=f"secret_{file_path.name}_{i}",
                            secret_type=secret_type,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=i,
                            matched_pattern=masked_line[:100],
                            severity=VulnerabilitySeverity.HIGH,
                            recommendation=f"Remove {secret_type} and use environment variables or secrets manager",
                        )
                        secrets.append(secret)
        
        return secrets
    
    async def scan_configuration(self) -> List[SecurityIssue]:
        """
        Scan configuration files for security issues.
        
        Returns:
            List of configuration security issues
        """
        issues = []
        
        config_files = [
            ".env",
            ".env.example",
            "config.yaml",
            "config.json",
            "docker-compose.yml",
            "Dockerfile",
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                continue
            
            try:
                content = config_path.read_text()
                lines = content.splitlines()
                
                # Check for common configuration issues
                for i, line in enumerate(lines, 1):
                    # Debug mode
                    if re.search(r'debug\s*[=:]\s*true', line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            issue_id=f"config_{config_file}_{i}",
                            issue_type=IssueType.CONFIGURATION,
                            file_path=config_file,
                            line_number=i,
                            severity=VulnerabilitySeverity.MEDIUM,
                            message="Debug mode enabled in configuration",
                            code_snippet=line.strip(),
                            recommendation="Disable debug mode in production",
                        ))
                    
                    # Insecure SSL
                    if re.search(r'verify\s*[=:]\s*false', line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            issue_id=f"config_{config_file}_{i}",
                            issue_type=IssueType.CONFIGURATION,
                            file_path=config_file,
                            line_number=i,
                            severity=VulnerabilitySeverity.HIGH,
                            message="SSL verification disabled",
                            code_snippet=line.strip(),
                            recommendation="Enable SSL verification",
                        ))
                    
                    # Exposed ports
                    if re.search(r'ports:\s*-\s*["\']?\d+:\d+', line):
                        issues.append(SecurityIssue(
                            issue_id=f"config_{config_file}_{i}",
                            issue_type=IssueType.CONFIGURATION,
                            file_path=config_file,
                            line_number=i,
                            severity=VulnerabilitySeverity.LOW,
                            message="Port exposed in Docker configuration",
                            code_snippet=line.strip(),
                            recommendation="Review exposed ports for security",
                        ))
            except Exception as e:
                logger.debug(f"Failed to scan {config_file}", error=str(e))
        
        return issues
    
    def get_scan_history(self) -> List[Dict[str, Any]]:
        """
        Get scan history (placeholder for future implementation).
        
        Returns:
            List of previous scan results
        """
        # This would be implemented with persistent storage
        return []
    
    def export_results(self, result: ScanResult, format: str = "json") -> str:
        """
        Export scan results in specified format.
        
        Args:
            result: Scan result to export
            format: Export format (json, csv, html)
            
        Returns:
            Formatted results
        """
        if format == "json":
            return json.dumps(result.to_dict(), indent=2)
        elif format == "csv":
            return self._export_csv(result)
        elif format == "html":
            return self._export_html(result)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_csv(self, result: ScanResult) -> str:
        """Export results as CSV"""
        lines = ["type,id,severity,description,file,line"]
        
        for vuln in result.vulnerabilities:
            lines.append(f"vulnerability,{vuln.id},{vuln.severity.value},{vuln.description},{vuln.package},{vuln.version}")
        
        for issue in result.issues:
            lines.append(f"issue,{issue.issue_id},{issue.severity.value},{issue.message},{issue.file_path},{issue.line_number}")
        
        for secret in result.secrets:
            lines.append(f"secret,{secret.secret_id},{secret.severity.value},{secret.secret_type},{secret.file_path},{secret.line_number}")
        
        return "\n".join(lines)
    
    def _export_html(self, result: ScanResult) -> str:
        """Export results as HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Scan Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .critical {{ color: #d32f2f; }}
                .high {{ color: #f57c00; }}
                .medium {{ color: #fbc02d; }}
                .low {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #333; color: white; }}
            </style>
        </head>
        <body>
            <h1>Security Scan Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Scan ID: {result.scan_id}</p>
                <p>Timestamp: {result.timestamp.isoformat()}</p>
                <p>Duration: {result.duration_ms}ms</p>
                <p>Risk Level: <span class="{result.summary.get('risk_level', 'unknown')}">{result.summary.get('risk_level', 'unknown').upper()}</span></p>
                <p>Total Vulnerabilities: {len(result.vulnerabilities)}</p>
                <p>Total Issues: {len(result.issues)}</p>
                <p>Total Secrets: {len(result.secrets)}</p>
            </div>
        """
        
        if result.vulnerabilities:
            html += """
            <h2>Vulnerabilities</h2>
            <table>
                <tr><th>ID</th><th>Package</th><th>Version</th><th>Severity</th><th>Description</th></tr>
            """
            for vuln in result.vulnerabilities:
                html += f'<tr><td>{vuln.id}</td><td>{vuln.package}</td><td>{vuln.version}</td><td class="{vuln.severity.value}">{vuln.severity.value}</td><td>{vuln.description}</td></tr>'
            html += "</table>"
        
        if result.issues:
            html += """
            <h2>Security Issues</h2>
            <table>
                <tr><th>ID</th><th>File</th><th>Line</th><th>Severity</th><th>Message</th></tr>
            """
            for issue in result.issues:
                html += f'<tr><td>{issue.issue_id}</td><td>{issue.file_path}</td><td>{issue.line_number}</td><td class="{issue.severity.value}">{issue.severity.value}</td><td>{issue.message}</td></tr>'
            html += "</table>"
        
        if result.secrets:
            html += """
            <h2>Secret Leaks</h2>
            <table>
                <tr><th>Type</th><th>File</th><th>Line</th><th>Severity</th></tr>
            """
            for secret in result.secrets:
                html += f'<tr><td>{secret.secret_type}</td><td>{secret.file_path}</td><td>{secret.line_number}</td><td class="{secret.severity.value}">{secret.severity.value}</td></tr>'
            html += "</table>"
            
            html += "</body></html>"
            return html
    
    
    def main():
        """CLI entry point for the security scanner."""
        import argparse
        
        parser = argparse.ArgumentParser(
            description="PRSM Security Scanner - Automated security scanning for PRSM"
        )
        parser.add_argument(
            "--output", "-o",
            default="security-report.json",
            help="Output file path for the security report (default: security-report.json)"
        )
        parser.add_argument(
            "--format", "-f",
            choices=["json", "csv", "html"],
            default="json",
            help="Output format (default: json)"
        )
        parser.add_argument(
            "--project-root", "-r",
            default=".",
            help="Project root directory to scan (default: current directory)"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        parser.add_argument(
            "--fail-on-critical",
            action="store_true",
            help="Exit with error code if critical vulnerabilities found"
        )
        
        args = parser.parse_args()
        
        print("🔍 PRSM Security Scanner")
        print("=" * 50)
        print(f"Project root: {args.project_root}")
        print(f"Output file: {args.output}")
        print(f"Output format: {args.format}")
        print("=" * 50)
        
        # Initialize scanner
        scanner = SecurityScanner(project_root=args.project_root)
        
        # Run full scan
        print("\n🔎 Running security scan...")
        result = scanner.scan_all()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Security Scan Results")
        print("=" * 50)
        print(f"Scan ID: {result.scan_id}")
        print(f"Timestamp: {result.timestamp.isoformat()}")
        print(f"Duration: {result.duration_ms}ms")
        print(f"Risk Level: {result.summary.get('risk_level', 'unknown').upper()}")
        print(f"Risk Score: {result.summary.get('risk_score', 0)}")
        print()
        print(f"Vulnerabilities: {len(result.vulnerabilities)}")
        print(f"  - Critical: {result.summary.get('vulnerabilities_by_severity', {}).get('critical', 0)}")
        print(f"  - High: {result.summary.get('vulnerabilities_by_severity', {}).get('high', 0)}")
        print(f"  - Medium: {result.summary.get('vulnerabilities_by_severity', {}).get('medium', 0)}")
        print(f"  - Low: {result.summary.get('vulnerabilities_by_severity', {}).get('low', 0)}")
        print()
        print(f"Security Issues: {len(result.issues)}")
        print(f"  - Critical: {result.summary.get('issues_by_severity', {}).get('critical', 0)}")
        print(f"  - High: {result.summary.get('issues_by_severity', {}).get('high', 0)}")
        print(f"  - Medium: {result.summary.get('issues_by_severity', {}).get('medium', 0)}")
        print(f"  - Low: {result.summary.get('issues_by_severity', {}).get('low', 0)}")
        print()
        print(f"Secret Leaks: {len(result.secrets)}")
        
        # Export results
        print(f"\n📄 Exporting results to {args.output}...")
        output_content = scanner.export_results(result, format=args.format)
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(output_content)
        
        print(f"✅ Results saved to {args.output}")
        
        # Print detailed findings if verbose
        if args.verbose:
            print("\n" + "=" * 50)
            print("📋 Detailed Findings")
            print("=" * 50)
            
            if result.vulnerabilities:
                print("\nVulnerabilities:")
                for vuln in result.vulnerabilities:
                    print(f"  - [{vuln.severity.value.upper()}] {vuln.id}: {vuln.package} {vuln.version}")
                    print(f"    {vuln.description[:100]}...")
            
            if result.issues:
                print("\nSecurity Issues:")
                for issue in result.issues:
                    print(f"  - [{issue.severity.value.upper()}] {issue.issue_id}")
                    print(f"    {issue.file_path}:{issue.line_number}")
                    print(f"    {issue.message[:100]}...")
            
            if result.secrets:
                print("\nSecret Leaks:")
                for secret in result.secrets:
                    print(f"  - [{secret.severity.value.upper()}] {secret.secret_type}")
                    print(f"    {secret.file_path}:{secret.line_number}")
        
        # Determine exit code
        if args.fail_on_critical:
            critical_count = (
                result.summary.get('vulnerabilities_by_severity', {}).get('critical', 0) +
                result.summary.get('issues_by_severity', {}).get('critical', 0) +
                len([s for s in result.secrets if s.severity == VulnerabilitySeverity.CRITICAL])
            )
            high_count = (
                result.summary.get('vulnerabilities_by_severity', {}).get('high', 0) +
                result.summary.get('issues_by_severity', {}).get('high', 0)
            )
            
            if critical_count > 0 or high_count > 0:
                print(f"\n❌ BUILD FAILED: Found {critical_count} critical and {high_count} high severity issues")
                sys.exit(1)
        
        print("\n✅ Security scan completed successfully")
        sys.exit(0)
    
    
    if __name__ == "__main__":
        main()