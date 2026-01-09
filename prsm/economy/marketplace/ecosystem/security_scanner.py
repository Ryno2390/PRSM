#!/usr/bin/env python3
"""
Security Scanner for Marketplace Ecosystem
==========================================

Advanced security scanning and vulnerability assessment system
for marketplace integrations with automated threat detection.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json
import hashlib
import re
import ast
import zipfile
import tempfile
import subprocess
from collections import defaultdict, Counter
import asyncio


class SecurityLevel(Enum):
    """Security clearance levels"""
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    CODE_INJECTION = "code_injection"
    XSS = "cross_site_scripting"
    SQL_INJECTION = "sql_injection"
    PATH_TRAVERSAL = "path_traversal"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    HARDCODED_SECRETS = "hardcoded_secrets"
    WEAK_CRYPTO = "weak_cryptography"
    UNSAFE_IMPORTS = "unsafe_imports"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INFORMATION_DISCLOSURE = "information_disclosure"
    MALICIOUS_CODE = "malicious_code"


class ScanType(Enum):
    """Types of security scans"""
    STATIC_ANALYSIS = "static_analysis"
    DEPENDENCY_CHECK = "dependency_check"
    MALWARE_SCAN = "malware_scan"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    COMPREHENSIVE = "comprehensive"


@dataclass
class SecurityVulnerability:
    """Individual security vulnerability"""
    id: str
    type: VulnerabilityType
    severity: SecurityLevel
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    cvss_score: Optional[float] = None  # Common Vulnerability Scoring System
    remediation: Optional[str] = None
    references: List[str] = None
    detected_at: datetime = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()


@dataclass
class SecurityReport:
    """Comprehensive security scan report"""
    id: str
    integration_id: str
    scan_type: ScanType
    status: str = "completed"
    started_at: datetime = None
    completed_at: datetime = None
    overall_score: int = 0  # 0-100 security score
    risk_level: SecurityLevel = SecurityLevel.UNKNOWN
    vulnerabilities: List[SecurityVulnerability] = None
    dependency_issues: List[Dict[str, Any]] = None
    malware_detected: bool = False
    suspicious_behaviors: List[str] = None
    recommendations: List[str] = None
    scan_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.vulnerabilities is None:
            self.vulnerabilities = []
        if self.dependency_issues is None:
            self.dependency_issues = []
        if self.suspicious_behaviors is None:
            self.suspicious_behaviors = []
        if self.recommendations is None:
            self.recommendations = []
        if self.scan_metadata is None:
            self.scan_metadata = {}
        if self.started_at is None:
            self.started_at = datetime.utcnow()
        if self.completed_at is None:
            self.completed_at = datetime.utcnow()


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    id: str
    name: str
    description: str
    enabled: bool = True
    severity_threshold: SecurityLevel = SecurityLevel.MEDIUM
    blocked_patterns: List[str] = None
    allowed_imports: Set[str] = None
    blocked_imports: Set[str] = None
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: Set[str] = None
    scan_timeout: int = 300  # 5 minutes
    auto_quarantine: bool = True
    notification_emails: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = []
        if self.allowed_imports is None:
            self.allowed_imports = set()
        if self.blocked_imports is None:
            self.blocked_imports = set()
        if self.allowed_file_types is None:
            self.allowed_file_types = {'.py', '.js', '.json', '.yaml', '.yml', '.md', '.txt'}
        if self.notification_emails is None:
            self.notification_emails = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at


class StaticCodeAnalyzer:
    """Static code analysis for security vulnerabilities"""
    
    def __init__(self):
        # Dangerous patterns and functions
        self.dangerous_patterns = {
            VulnerabilityType.CODE_INJECTION: [
                r'exec\s*\(',
                r'eval\s*\(',
                r'compile\s*\(',
                r'__import__\s*\(',
                r'subprocess\.call\s*\(',
                r'subprocess\.run\s*\(',
                r'os\.system\s*\(',
                r'os\.popen\s*\(',
                r'commands\.getoutput\s*\('
            ],
            VulnerabilityType.SQL_INJECTION: [
                r'cursor\.execute\s*\(\s*["\'].*%.*["\']',
                r'\.execute\s*\(\s*f["\'].*\{.*\}.*["\']',
                r'sql\s*=\s*["\'].*\+.*["\']',
                r'query\s*=\s*["\'].*%.*["\']'
            ],
            VulnerabilityType.PATH_TRAVERSAL: [
                r'open\s*\(\s*.*\+.*\)',
                r'file\s*\(\s*.*\+.*\)',
                r'os\.path\.join\s*\(.*input.*\)',
                r'\.\./',
                r'\.\.\\'
            ],
            VulnerabilityType.HARDCODED_SECRETS: [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][^"\']{20,}["\']',
                r'secret\s*=\s*["\'][^"\']{16,}["\']',
                r'token\s*=\s*["\'][^"\']{32,}["\']',
                r'BEGIN RSA PRIVATE KEY',
                r'BEGIN PRIVATE KEY'
            ],
            VulnerabilityType.WEAK_CRYPTO: [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'DES\.',
                r'RC4\.',
                r'random\.random\s*\('
            ],
            VulnerabilityType.UNSAFE_DESERIALIZATION: [
                r'pickle\.loads\s*\(',
                r'cPickle\.loads\s*\(',
                r'yaml\.load\s*\(',
                r'marshal\.loads\s*\('
            ]
        }
        
        # Suspicious imports
        self.dangerous_imports = {
            'os', 'subprocess', 'sys', 'eval', 'exec', 'compile',
            'importlib', '__builtin__', 'builtins', 'ctypes',
            'marshal', 'pickle', 'cPickle', 'dill'
        }
        
        # High-risk function calls
        self.risky_functions = {
            'exec', 'eval', 'compile', '__import__', 'getattr',
            'setattr', 'delattr', 'hasattr', 'callable', 'globals',
            'locals', 'vars', 'dir'
        }
    
    def analyze_python_code(self, code: str, file_path: str = None) -> List[SecurityVulnerability]:
        """Analyze Python code for security vulnerabilities"""
        vulnerabilities = []
        lines = code.split('\n')
        
        try:
            # Parse AST for structural analysis
            tree = ast.parse(code)
            ast_vulnerabilities = self._analyze_ast(tree, file_path)
            vulnerabilities.extend(ast_vulnerabilities)
        except SyntaxError as e:
            # Log syntax error but continue with pattern matching
            pass
        
        # Pattern-based analysis
        for line_num, line in enumerate(lines, 1):
            line_vulnerabilities = self._analyze_line(line, line_num, file_path)
            vulnerabilities.extend(line_vulnerabilities)
        
        return vulnerabilities
    
    def _analyze_ast(self, tree: ast.AST, file_path: str = None) -> List[SecurityVulnerability]:
        """Analyze AST for security issues"""
        vulnerabilities = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.risky_functions:
                        vuln = SecurityVulnerability(
                            id=self._generate_vuln_id(file_path, node.lineno, func_name),
                            type=VulnerabilityType.CODE_INJECTION,
                            severity=SecurityLevel.HIGH,
                            title=f"Dangerous function call: {func_name}",
                            description=f"Use of potentially dangerous function '{func_name}' detected",
                            file_path=file_path,
                            line_number=node.lineno,
                            cwe_id="CWE-94",
                            remediation=f"Avoid using '{func_name}' or ensure input validation"
                        )
                        vulnerabilities.append(vuln)
            
            # Check for dangerous imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_imports:
                        vuln = SecurityVulnerability(
                            id=self._generate_vuln_id(file_path, node.lineno, alias.name),
                            type=VulnerabilityType.UNSAFE_IMPORTS,
                            severity=SecurityLevel.MEDIUM,
                            title=f"Potentially dangerous import: {alias.name}",
                            description=f"Import of potentially dangerous module '{alias.name}'",
                            file_path=file_path,
                            line_number=node.lineno,
                            remediation=f"Review usage of '{alias.name}' module for security implications"
                        )
                        vulnerabilities.append(vuln)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.dangerous_imports:
                    vuln = SecurityVulnerability(
                        id=self._generate_vuln_id(file_path, node.lineno, node.module),
                        type=VulnerabilityType.UNSAFE_IMPORTS,
                        severity=SecurityLevel.MEDIUM,
                        title=f"Potentially dangerous import from: {node.module}",
                        description=f"Import from potentially dangerous module '{node.module}'",
                        file_path=file_path,
                        line_number=node.lineno,
                        remediation=f"Review usage of functions from '{node.module}' module"
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _analyze_line(self, line: str, line_num: int, file_path: str = None) -> List[SecurityVulnerability]:
        """Analyze individual line for security patterns"""
        vulnerabilities = []
        
        for vuln_type, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    severity = self._get_pattern_severity(vuln_type)
                    
                    vuln = SecurityVulnerability(
                        id=self._generate_vuln_id(file_path, line_num, pattern),
                        type=vuln_type,
                        severity=severity,
                        title=f"{vuln_type.value.replace('_', ' ').title()} detected",
                        description=f"Potentially dangerous pattern detected: {pattern}",
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        remediation=self._get_remediation(vuln_type)
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _generate_vuln_id(self, file_path: str, line_num: int, pattern: str) -> str:
        """Generate unique vulnerability ID"""
        content = f"{file_path}:{line_num}:{pattern}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_pattern_severity(self, vuln_type: VulnerabilityType) -> SecurityLevel:
        """Get severity level for vulnerability type"""
        severity_map = {
            VulnerabilityType.CODE_INJECTION: SecurityLevel.CRITICAL,
            VulnerabilityType.SQL_INJECTION: SecurityLevel.HIGH,
            VulnerabilityType.HARDCODED_SECRETS: SecurityLevel.HIGH,
            VulnerabilityType.PATH_TRAVERSAL: SecurityLevel.HIGH,
            VulnerabilityType.UNSAFE_DESERIALIZATION: SecurityLevel.HIGH,
            VulnerabilityType.WEAK_CRYPTO: SecurityLevel.MEDIUM,
            VulnerabilityType.UNSAFE_IMPORTS: SecurityLevel.MEDIUM
        }
        return severity_map.get(vuln_type, SecurityLevel.LOW)
    
    def _get_remediation(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation advice for vulnerability type"""
        remediation_map = {
            VulnerabilityType.CODE_INJECTION: "Use parameterized queries and input validation",
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries or ORM",
            VulnerabilityType.HARDCODED_SECRETS: "Use environment variables or secure key management",
            VulnerabilityType.PATH_TRAVERSAL: "Validate and sanitize file paths",
            VulnerabilityType.UNSAFE_DESERIALIZATION: "Use safe serialization formats like JSON",
            VulnerabilityType.WEAK_CRYPTO: "Use strong cryptographic algorithms (SHA-256, AES)",
            VulnerabilityType.UNSAFE_IMPORTS: "Review module usage and consider safer alternatives"
        }
        return remediation_map.get(vuln_type, "Review code for security implications")


class DependencyScanner:
    """Scanner for dependency vulnerabilities"""
    
    def __init__(self):
        # Known vulnerable packages (would be updated from security databases)
        self.vulnerable_packages = {
            'requests': {'<2.20.0': 'CVE-2018-18074'},
            'urllib3': {'<1.24.2': 'CVE-2019-11324'},
            'pyyaml': {'<5.1': 'CVE-2017-18342'},
            'jinja2': {'<2.10.1': 'CVE-2019-10906'},
            'flask': {'<1.0': 'CVE-2018-1000656'}
        }
    
    def scan_requirements(self, requirements_content: str) -> List[Dict[str, Any]]:
        """Scan requirements.txt for vulnerable dependencies"""
        issues = []
        lines = requirements_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse package and version
            if '==' in line:
                package, version = line.split('==', 1)
                package = package.strip()
                version = version.strip()
                
                if package.lower() in self.vulnerable_packages:
                    vulnerable_versions = self.vulnerable_packages[package.lower()]
                    for vuln_version, cve in vulnerable_versions.items():
                        if self._version_affected(version, vuln_version):
                            issues.append({
                                'package': package,
                                'version': version,
                                'vulnerability': cve,
                                'affected_versions': vuln_version,
                                'severity': 'high',
                                'description': f"Package {package} version {version} has known vulnerability {cve}"
                            })
        
        return issues
    
    def _version_affected(self, current_version: str, vulnerable_pattern: str) -> bool:
        """Check if current version is affected by vulnerability pattern"""
        # Simplified version comparison (would use proper semver in production)
        if vulnerable_pattern.startswith('<'):
            target_version = vulnerable_pattern[1:]
            try:
                current_parts = [int(x) for x in current_version.split('.')]
                target_parts = [int(x) for x in target_version.split('.')]
                
                # Pad shorter version with zeros
                max_len = max(len(current_parts), len(target_parts))
                current_parts.extend([0] * (max_len - len(current_parts)))
                target_parts.extend([0] * (max_len - len(target_parts)))
                
                return current_parts < target_parts
            except ValueError:
                return False
        
        return False


class MalwareScanner:
    """Scanner for malware and malicious code patterns"""
    
    def __init__(self):
        # Malicious patterns
        self.malware_patterns = [
            # Network communication
            r'socket\.socket\s*\(',
            r'urllib\.request\.urlopen\s*\(',
            r'requests\.(get|post|put|delete)\s*\(',
            r'http\.client\.',
            
            # File system access
            r'os\.remove\s*\(',
            r'os\.rmdir\s*\(',
            r'shutil\.rmtree\s*\(',
            r'os\.chmod\s*\(',
            
            # Process execution
            r'subprocess\.',
            r'os\.system\s*\(',
            r'os\.spawn\w*\s*\(',
            
            # Obfuscation
            r'\.decode\s*\(\s*["\']hex["\']',
            r'\.decode\s*\(\s*["\']base64["\']',
            r'codecs\.decode\s*\(',
            
            # Suspicious strings
            r'backdoor',
            r'keylogger',
            r'trojan',
            r'rootkit',
            r'botnet'
        ]
        
        # Entropy threshold for detecting obfuscated code
        self.entropy_threshold = 4.5
    
    def scan_code(self, code: str, file_path: str = None) -> Tuple[bool, List[str]]:
        """Scan code for malware patterns"""
        suspicious_behaviors = []
        malware_detected = False
        
        # Pattern matching
        for pattern in self.malware_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                suspicious_behaviors.append(f"Suspicious pattern detected: {pattern}")
                if any(keyword in pattern.lower() for keyword in ['backdoor', 'keylogger', 'trojan', 'rootkit', 'botnet']):
                    malware_detected = True
        
        # Check for high entropy (obfuscated code)
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            if len(line.strip()) > 50:  # Only check substantial lines
                entropy = self._calculate_entropy(line)
                if entropy > self.entropy_threshold:
                    suspicious_behaviors.append(f"High entropy line {line_num} (possible obfuscation): {entropy:.2f}")
        
        # Check for suspicious string concatenations
        if re.search(r'["\'][^"\']*["\']\s*\+\s*["\'][^"\']*["\'].*\+', code):
            suspicious_behaviors.append("Suspicious string concatenation detected")
        
        return malware_detected, suspicious_behaviors
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = Counter(text)
        text_len = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                probability = count / text_len
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy


class BehavioralAnalyzer:
    """Analyzer for suspicious behavioral patterns"""
    
    def __init__(self):
        self.suspicious_behaviors = [
            'excessive_file_operations',
            'network_communication',
            'process_execution',
            'system_modification',
            'data_exfiltration',
            'privilege_escalation',
            'anti_analysis',
            'persistence_mechanism'
        ]
    
    def analyze_behavior(self, code: str) -> List[str]:
        """Analyze code for suspicious behavioral patterns"""
        behaviors = []
        
        # File operations
        file_ops = len(re.findall(r'open\s*\(|file\s*\(|os\.path\.|shutil\.', code))
        if file_ops > 10:
            behaviors.append('excessive_file_operations')
        
        # Network communication
        if re.search(r'socket\.|urllib\.|requests\.|http\.|ftp\.', code):
            behaviors.append('network_communication')
        
        # Process execution
        if re.search(r'subprocess\.|os\.system|os\.spawn|os\.exec', code):
            behaviors.append('process_execution')
        
        # System modification
        if re.search(r'os\.chmod|os\.chown|registry\.|winreg\.', code):
            behaviors.append('system_modification')
        
        # Potential data exfiltration
        if re.search(r'base64\.encode|urllib\.parse\.quote|json\.dumps.*requests\.', code):
            behaviors.append('data_exfiltration')
        
        # Anti-analysis techniques
        if re.search(r'time\.sleep|threading\.Timer|random\.randint.*time', code):
            behaviors.append('anti_analysis')
        
        # Persistence mechanisms
        if re.search(r'crontab|startup|autostart|registry.*run', code, re.IGNORECASE):
            behaviors.append('persistence_mechanism')
        
        return behaviors


class SecurityScanner:
    """Main security scanner coordinator"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./security_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.reports: Dict[str, SecurityReport] = {}
        self.policies: Dict[str, SecurityPolicy] = {}
        
        # Scanner components
        self.static_analyzer = StaticCodeAnalyzer()
        self.dependency_scanner = DependencyScanner()
        self.malware_scanner = MalwareScanner()
        self.behavioral_analyzer = BehavioralAnalyzer()
        
        # Load existing data
        self._load_data()
        
        # Create default policy if none exists
        if not self.policies:
            self._create_default_policy()
    
    def _load_data(self):
        """Load security data from storage"""
        try:
            reports_file = self.storage_path / "reports.json"
            if reports_file.exists():
                with open(reports_file, 'r') as f:
                    data = json.load(f)
                    for report_data in data:
                        report = SecurityReport(**report_data)
                        # Convert datetime strings
                        report.started_at = datetime.fromisoformat(report_data['started_at'])
                        report.completed_at = datetime.fromisoformat(report_data['completed_at'])
                        
                        # Convert vulnerabilities
                        report.vulnerabilities = []
                        for vuln_data in report_data.get('vulnerabilities', []):
                            vuln = SecurityVulnerability(**vuln_data)
                            vuln.detected_at = datetime.fromisoformat(vuln_data['detected_at'])
                            report.vulnerabilities.append(vuln)
                        
                        self.reports[report.id] = report
            
            policies_file = self.storage_path / "policies.json"
            if policies_file.exists():
                with open(policies_file, 'r') as f:
                    data = json.load(f)
                    for policy_data in data:
                        policy = SecurityPolicy(**policy_data)
                        policy.created_at = datetime.fromisoformat(policy_data['created_at'])
                        policy.updated_at = datetime.fromisoformat(policy_data['updated_at'])
                        self.policies[policy.id] = policy
        
        except Exception as e:
            print(f"Error loading security data: {e}")
    
    def _save_data(self):
        """Save security data to storage"""
        try:
            # Save reports
            reports_data = []
            for report in self.reports.values():
                vulnerabilities_data = []
                for vuln in report.vulnerabilities:
                    vuln_dict = {
                        'id': vuln.id,
                        'type': vuln.type.value,
                        'severity': vuln.severity.value,
                        'title': vuln.title,
                        'description': vuln.description,
                        'file_path': vuln.file_path,
                        'line_number': vuln.line_number,
                        'code_snippet': vuln.code_snippet,
                        'cwe_id': vuln.cwe_id,
                        'cvss_score': vuln.cvss_score,
                        'remediation': vuln.remediation,
                        'references': vuln.references,
                        'detected_at': vuln.detected_at.isoformat()
                    }
                    vulnerabilities_data.append(vuln_dict)
                
                report_dict = {
                    'id': report.id,
                    'integration_id': report.integration_id,
                    'scan_type': report.scan_type.value,
                    'status': report.status,
                    'started_at': report.started_at.isoformat(),
                    'completed_at': report.completed_at.isoformat(),
                    'overall_score': report.overall_score,
                    'risk_level': report.risk_level.value,
                    'vulnerabilities': vulnerabilities_data,
                    'dependency_issues': report.dependency_issues,
                    'malware_detected': report.malware_detected,
                    'suspicious_behaviors': report.suspicious_behaviors,
                    'recommendations': report.recommendations,
                    'scan_metadata': report.scan_metadata
                }
                reports_data.append(report_dict)
            
            with open(self.storage_path / "reports.json", 'w') as f:
                json.dump(reports_data, f, indent=2)
            
            # Save policies
            policies_data = []
            for policy in self.policies.values():
                policy_dict = {
                    'id': policy.id,
                    'name': policy.name,
                    'description': policy.description,
                    'enabled': policy.enabled,
                    'severity_threshold': policy.severity_threshold.value,
                    'blocked_patterns': policy.blocked_patterns,
                    'allowed_imports': list(policy.allowed_imports),
                    'blocked_imports': list(policy.blocked_imports),
                    'max_file_size': policy.max_file_size,
                    'allowed_file_types': list(policy.allowed_file_types),
                    'scan_timeout': policy.scan_timeout,
                    'auto_quarantine': policy.auto_quarantine,
                    'notification_emails': policy.notification_emails,
                    'created_at': policy.created_at.isoformat(),
                    'updated_at': policy.updated_at.isoformat()
                }
                policies_data.append(policy_dict)
            
            with open(self.storage_path / "policies.json", 'w') as f:
                json.dump(policies_data, f, indent=2)
        
        except Exception as e:
            print(f"Error saving security data: {e}")
    
    def _create_default_policy(self):
        """Create default security policy"""
        default_policy = SecurityPolicy(
            id="default",
            name="Default Security Policy",
            description="Standard security policy for marketplace integrations",
            severity_threshold=SecurityLevel.MEDIUM,
            blocked_imports={'eval', 'exec', 'compile', '__import__'},
            max_file_size=10 * 1024 * 1024,  # 10MB
            scan_timeout=180,  # 3 minutes
            auto_quarantine=True
        )
        self.policies["default"] = default_policy
        self._save_data()
    
    async def scan_integration(self, integration_id: str, file_path: Union[str, Path],
                              scan_type: ScanType = ScanType.COMPREHENSIVE,
                              policy_id: str = "default") -> str:
        """Scan integration for security vulnerabilities"""
        # Generate scan ID
        scan_id = hashlib.md5(f"{integration_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
        
        # Create initial report
        report = SecurityReport(
            id=scan_id,
            integration_id=integration_id,
            scan_type=scan_type,
            status="running"
        )
        self.reports[scan_id] = report
        
        try:
            # Get policy
            policy = self.policies.get(policy_id, self.policies["default"])
            
            # Extract and scan files
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            if file_path.suffix == '.zip':
                await self._scan_zip_file(report, file_path, policy)
            else:
                await self._scan_single_file(report, file_path, policy)
            
            # Calculate overall security score
            report.overall_score = self._calculate_security_score(report)
            report.risk_level = self._determine_risk_level(report)
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            report.status = "completed"
            report.completed_at = datetime.utcnow()
        
        except Exception as e:
            report.status = "failed"
            report.scan_metadata['error'] = str(e)
            report.completed_at = datetime.utcnow()
        
        self._save_data()
        return scan_id
    
    async def _scan_zip_file(self, report: SecurityReport, zip_path: Path, policy: SecurityPolicy):
        """Scan ZIP file contents"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Check for zip bombs
                    total_size = sum(info.file_size for info in zip_ref.infolist())
                    if total_size > policy.max_file_size * 10:  # 10x max file size
                        report.suspicious_behaviors.append("Potential zip bomb detected")
                        return
                    
                    # Extract files
                    zip_ref.extractall(temp_path)
                
                # Scan extracted files
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        if file_path.suffix in policy.allowed_file_types:
                            await self._scan_single_file(report, file_path, policy)
                        else:
                            report.suspicious_behaviors.append(f"Disallowed file type: {file_path.suffix}")
            
            except zipfile.BadZipFile:
                report.suspicious_behaviors.append("Corrupted or invalid ZIP file")
    
    async def _scan_single_file(self, report: SecurityReport, file_path: Path, policy: SecurityPolicy):
        """Scan single file"""
        try:
            # Check file size
            if file_path.stat().st_size > policy.max_file_size:
                report.suspicious_behaviors.append(f"File exceeds size limit: {file_path}")
                return
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Static code analysis
            if file_path.suffix == '.py':
                vulnerabilities = self.static_analyzer.analyze_python_code(content, str(file_path))
                report.vulnerabilities.extend(vulnerabilities)
            
            # Malware scanning
            malware_detected, suspicious_behaviors = self.malware_scanner.scan_code(content, str(file_path))
            if malware_detected:
                report.malware_detected = True
            report.suspicious_behaviors.extend(suspicious_behaviors)
            
            # Behavioral analysis
            behaviors = self.behavioral_analyzer.analyze_behavior(content)
            report.suspicious_behaviors.extend([f"Suspicious behavior: {b}" for b in behaviors])
            
            # Dependency scanning for requirements files
            if file_path.name in ['requirements.txt', 'requirements-dev.txt']:
                dependency_issues = self.dependency_scanner.scan_requirements(content)
                report.dependency_issues.extend(dependency_issues)
            
            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
        
        except Exception as e:
            report.suspicious_behaviors.append(f"Error scanning file {file_path}: {str(e)}")
    
    def _calculate_security_score(self, report: SecurityReport) -> int:
        """Calculate overall security score (0-100)"""
        base_score = 100
        
        # Deduct points for vulnerabilities
        for vuln in report.vulnerabilities:
            if vuln.severity == SecurityLevel.CRITICAL:
                base_score -= 25
            elif vuln.severity == SecurityLevel.HIGH:
                base_score -= 15
            elif vuln.severity == SecurityLevel.MEDIUM:
                base_score -= 5
            elif vuln.severity == SecurityLevel.LOW:
                base_score -= 1
        
        # Deduct points for dependency issues
        for issue in report.dependency_issues:
            if issue.get('severity') == 'high':
                base_score -= 10
            elif issue.get('severity') == 'medium':
                base_score -= 5
            else:
                base_score -= 2
        
        # Deduct points for malware
        if report.malware_detected:
            base_score -= 50
        
        # Deduct points for suspicious behaviors
        base_score -= min(len(report.suspicious_behaviors) * 2, 30)
        
        return max(0, base_score)
    
    def _determine_risk_level(self, report: SecurityReport) -> SecurityLevel:
        """Determine overall risk level"""
        if report.overall_score >= 90:
            return SecurityLevel.LOW
        elif report.overall_score >= 70:
            return SecurityLevel.MEDIUM
        elif report.overall_score >= 40:
            return SecurityLevel.HIGH
        else:
            return SecurityLevel.CRITICAL
    
    def _generate_recommendations(self, report: SecurityReport) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Vulnerability-based recommendations
        vuln_types = set(vuln.type for vuln in report.vulnerabilities)
        for vuln_type in vuln_types:
            recommendations.append(f"Address {vuln_type.value.replace('_', ' ')} vulnerabilities")
        
        # Dependency recommendations
        if report.dependency_issues:
            recommendations.append("Update vulnerable dependencies to latest secure versions")
        
        # Malware recommendations
        if report.malware_detected:
            recommendations.append("CRITICAL: Remove or quarantine detected malware")
        
        # Behavioral recommendations
        if len(report.suspicious_behaviors) > 5:
            recommendations.append("Review code for suspicious behavioral patterns")
        
        # General recommendations
        if report.overall_score < 70:
            recommendations.append("Conduct thorough security review before deployment")
        
        return recommendations
    
    def get_security_report(self, scan_id: str) -> Optional[SecurityReport]:
        """Get security scan report"""
        return self.reports.get(scan_id)
    
    def get_integration_reports(self, integration_id: str) -> List[SecurityReport]:
        """Get all security reports for an integration"""
        return [report for report in self.reports.values() 
                if report.integration_id == integration_id]
    
    def create_security_policy(self, name: str, description: str, **kwargs) -> str:
        """Create new security policy"""
        policy_id = hashlib.md5(f"{name}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
        
        policy = SecurityPolicy(
            id=policy_id,
            name=name,
            description=description,
            **kwargs
        )
        
        self.policies[policy_id] = policy
        self._save_data()
        
        return policy_id
    
    def update_security_policy(self, policy_id: str, **updates) -> bool:
        """Update security policy"""
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        policy.updated_at = datetime.utcnow()
        self._save_data()
        
        return True
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary"""
        total_scans = len(self.reports)
        recent_scans = [r for r in self.reports.values() 
                       if r.started_at > datetime.utcnow() - timedelta(days=7)]
        
        risk_distribution = Counter(report.risk_level for report in self.reports.values())
        
        return {
            'total_scans': total_scans,
            'recent_scans': len(recent_scans),
            'active_policies': len([p for p in self.policies.values() if p.enabled]),
            'risk_distribution': {level.value: count for level, count in risk_distribution.items()},
            'average_security_score': sum(r.overall_score for r in self.reports.values()) / total_scans if total_scans > 0 else 0
        }
    
    def export_security_data(self, integration_id: Optional[str] = None) -> Dict[str, Any]:
        """Export security data for analysis"""
        reports_to_export = list(self.reports.values())
        
        if integration_id:
            reports_to_export = [r for r in reports_to_export if r.integration_id == integration_id]
        
        export_data = {
            'metadata': {
                'export_date': datetime.utcnow().isoformat(),
                'total_reports': len(reports_to_export),
                'integration_id': integration_id
            },
            'reports': [],
            'policies': []
        }
        
        # Export reports
        for report in reports_to_export:
            report_data = {
                'id': report.id,
                'integration_id': report.integration_id,
                'scan_type': report.scan_type.value,
                'overall_score': report.overall_score,
                'risk_level': report.risk_level.value,
                'vulnerability_count': len(report.vulnerabilities),
                'malware_detected': report.malware_detected,
                'completed_at': report.completed_at.isoformat()
            }
            export_data['reports'].append(report_data)
        
        # Export policies
        for policy in self.policies.values():
            policy_data = {
                'id': policy.id,
                'name': policy.name,
                'enabled': policy.enabled,
                'severity_threshold': policy.severity_threshold.value
            }
            export_data['policies'].append(policy_data)
        
        return export_data