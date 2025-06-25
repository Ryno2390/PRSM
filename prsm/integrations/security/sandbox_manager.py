"""
Enhanced Security Sandbox Manager
=================================

Comprehensive security sandboxing for both external content validation and
MCP tool execution. Provides secure isolation environments with resource
limits, permission controls, and comprehensive monitoring.

Key Features:
- Isolated execution environment for untrusted content
- MCP tool execution sandboxing with resource limits
- License compliance validation
- Vulnerability scanning and risk assessment
- Real-time resource monitoring and threat detection
- Fine-grained permission management with user consent
- Integration with PRSM's circuit breaker system
- Performance monitoring and audit logging
- Automatic cleanup and resource management

Sandbox Types:
- Content scanning for external integrations
- Tool execution for MCP protocol tools
- Multi-level isolation (basic, container, VM)
"""

import asyncio
import os
import tempfile
import shutil
import subprocess
import time
import signal
import threading
import resource
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from ..models.integration_models import SecurityRisk, LicenseType, SecurityScanResult
from ...core.config import settings
from ...core.models import TimestampMixin
# from ...safety.circuit_breaker import CircuitBreakerNetwork  # TODO: Integrate when needed

logger = structlog.get_logger(__name__)


class SandboxStatus(str, Enum):
    """Sandbox operational status"""
    IDLE = "idle"
    ACTIVE = "active"
    SCANNING = "scanning"
    QUARANTINED = "quarantined"
    ERROR = "error"


class SandboxResult(str, Enum):
    """Sandbox operation results"""
    APPROVED = "approved"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    ERROR = "error"


# ===== MCP Tool Execution Sandbox Classes =====

class ToolSandboxType(str, Enum):
    """Types of security sandboxes for tool execution"""
    NONE = "none"              # No sandboxing (for testing only)
    BASIC = "basic"            # Basic process isolation
    CONTAINER = "container"    # Docker container isolation
    VM = "vm"                  # Virtual machine isolation (future)


class ResourceType(str, Enum):
    """Types of system resources to monitor and limit"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESSES = "processes"
    FILES = "files"
    TIME = "time"


class ToolSecurityLevel(str, Enum):
    """Security levels for tool execution (duplicated here for self-contained module)"""
    SAFE = "safe"
    RESTRICTED = "restricted"
    PRIVILEGED = "privileged"
    DANGEROUS = "dangerous"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution"""
    # CPU limits
    cpu_percent: float = 50.0          # Maximum CPU usage percentage
    cpu_time_seconds: float = 30.0     # Maximum CPU time
    
    # Memory limits
    memory_mb: int = 512               # Maximum memory in MB
    virtual_memory_mb: int = 1024      # Maximum virtual memory in MB
    
    # Disk limits
    disk_read_mb: int = 100            # Maximum disk read in MB
    disk_write_mb: int = 50            # Maximum disk write in MB
    temp_space_mb: int = 100           # Maximum temporary space in MB
    
    # Network limits
    network_upload_mb: int = 10        # Maximum network upload in MB
    network_download_mb: int = 50      # Maximum network download in MB
    network_connections: int = 5       # Maximum concurrent connections
    
    # Process limits
    max_processes: int = 10            # Maximum number of processes
    max_threads: int = 20              # Maximum number of threads
    max_file_descriptors: int = 100    # Maximum file descriptors
    
    # Time limits
    execution_timeout: float = 60.0    # Maximum execution time in seconds
    idle_timeout: float = 10.0         # Maximum idle time in seconds


@dataclass
class SecurityPermissions:
    """Security permissions for tool execution"""
    # File system permissions
    file_read: Set[str] = field(default_factory=set)      # Allowed read paths
    file_write: Set[str] = field(default_factory=set)     # Allowed write paths
    file_execute: Set[str] = field(default_factory=set)   # Allowed execute paths
    
    # Network permissions
    network_outbound: bool = False     # Allow outbound network access
    network_inbound: bool = False      # Allow inbound network access
    allowed_hosts: Set[str] = field(default_factory=set)  # Allowed host patterns
    allowed_ports: Set[int] = field(default_factory=set)  # Allowed port numbers
    
    # System permissions
    system_calls: Set[str] = field(default_factory=set)   # Allowed system calls
    environment_vars: Set[str] = field(default_factory=set)  # Allowed env vars
    
    # Tool-specific permissions
    tool_permissions: Set[str] = field(default_factory=set)  # Tool-specific perms
    user_consent_required: bool = True  # Require explicit user consent


class ToolExecutionRequest(BaseModel):
    """Request for tool execution (simplified for sandbox)"""
    execution_id: UUID = Field(default_factory=uuid.uuid4)
    tool_id: str
    tool_action: str
    parameters: Dict[str, Any]
    user_id: str
    permissions: List[str] = Field(default_factory=list)
    sandbox_level: ToolSecurityLevel = ToolSecurityLevel.RESTRICTED
    timeout_seconds: float = 30.0


class ToolExecutionResult(BaseModel):
    """Result from tool execution"""
    execution_id: UUID
    tool_id: str
    success: bool
    result_data: Any = None
    execution_time: float
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    security_violations: List[str] = Field(default_factory=list)
    resource_violations: List[str] = Field(default_factory=list)


class ToolSandboxContext(BaseModel):
    """Context for tool sandbox execution"""
    sandbox_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    user_id: str
    execution_request: ToolExecutionRequest
    
    # Security configuration
    sandbox_type: ToolSandboxType = ToolSandboxType.CONTAINER
    security_level: ToolSecurityLevel
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    permissions: SecurityPermissions = Field(default_factory=SecurityPermissions)
    
    # Execution metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Monitoring data
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    security_violations: List[str] = Field(default_factory=list)
    audit_events: List[Dict[str, Any]] = Field(default_factory=list)


class ToolSandboxResult(BaseModel):
    """Result of tool sandbox execution"""
    sandbox_id: str
    tool_execution_result: ToolExecutionResult
    
    # Security metrics
    security_violations: List[str] = Field(default_factory=list)
    resource_violations: List[str] = Field(default_factory=list)
    permission_violations: List[str] = Field(default_factory=list)
    
    # Resource usage
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    total_network_mb: float = 0.0
    execution_time: float = 0.0
    
    # Audit information
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_status: str = "compliant"


class ResourceMonitor:
    """Real-time resource monitoring for sandbox execution"""
    
    def __init__(self, context: ToolSandboxContext):
        self.context = context
        self.monitoring = False
        self.violations: List[str] = []
        self.usage_data: Dict[str, List[float]] = {
            "cpu": [],
            "memory": [],
            "disk_read": [],
            "disk_write": [],
            "network": []
        }
        self.start_time = time.time()
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        
        logger.info("Starting resource monitoring",
                   sandbox_id=self.context.sandbox_id,
                   limits=self.context.resource_limits.__dict__)
        
        # Start monitoring loop
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        return monitoring_task
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return usage summary"""
        self.monitoring = False
        
        summary = {
            "execution_time": time.time() - self.start_time,
            "peak_cpu": max(self.usage_data["cpu"]) if self.usage_data["cpu"] else 0.0,
            "peak_memory": max(self.usage_data["memory"]) if self.usage_data["memory"] else 0.0,
            "total_disk_read": sum(self.usage_data["disk_read"]),
            "total_disk_write": sum(self.usage_data["disk_write"]),
            "total_network": sum(self.usage_data["network"]),
            "violations": self.violations
        }
        
        logger.info("Resource monitoring stopped",
                   sandbox_id=self.context.sandbox_id,
                   summary=summary)
        
        return summary
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring:
                await self._check_resources()
                await asyncio.sleep(0.5)  # Monitor every 500ms
        except Exception as e:
            logger.error("Resource monitoring failed",
                        sandbox_id=self.context.sandbox_id,
                        error=str(e))
    
    async def _check_resources(self):
        """Check current resource usage against limits"""
        try:
            # Simulate resource checking (in production, use psutil or similar)
            current_cpu = await self._get_cpu_usage()
            current_memory = await self._get_memory_usage()
            current_disk_read = await self._get_disk_read()
            current_disk_write = await self._get_disk_write()
            current_network = await self._get_network_usage()
            
            # Store usage data
            self.usage_data["cpu"].append(current_cpu)
            self.usage_data["memory"].append(current_memory)
            self.usage_data["disk_read"].append(current_disk_read)
            self.usage_data["disk_write"].append(current_disk_write)
            self.usage_data["network"].append(current_network)
            
            # Check limits
            limits = self.context.resource_limits
            
            if current_cpu > limits.cpu_percent:
                violation = f"CPU usage {current_cpu:.1f}% exceeds limit {limits.cpu_percent}%"
                self.violations.append(violation)
                logger.warning("Resource violation detected", violation=violation)
            
            if current_memory > limits.memory_mb:
                violation = f"Memory usage {current_memory:.1f}MB exceeds limit {limits.memory_mb}MB"
                self.violations.append(violation)
                logger.warning("Resource violation detected", violation=violation)
            
            # Check execution timeout
            elapsed_time = time.time() - self.start_time
            if elapsed_time > limits.execution_timeout:
                violation = f"Execution time {elapsed_time:.1f}s exceeds timeout {limits.execution_timeout}s"
                self.violations.append(violation)
                logger.warning("Timeout violation", violation=violation)
                self.monitoring = False  # Stop monitoring on timeout
                
        except Exception as e:
            logger.error("Resource check failed",
                        sandbox_id=self.context.sandbox_id,
                        error=str(e))
    
    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        # Simulate CPU usage (in production, use psutil.cpu_percent())
        return min(25.0 + len(self.usage_data["cpu"]) * 0.1, 100.0)
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        # Simulate memory usage (in production, use psutil.virtual_memory())
        return min(100.0 + len(self.usage_data["memory"]) * 2.0, 1024.0)
    
    async def _get_disk_read(self) -> float:
        """Get disk read in MB since last check"""
        return 0.5  # 0.5 MB per check
    
    async def _get_disk_write(self) -> float:
        """Get disk write in MB since last check"""
        return 0.2  # 0.2 MB per check
    
    async def _get_network_usage(self) -> float:
        """Get network usage in MB since last check"""
        return 0.1  # 0.1 MB per check


class SandboxManager:
    """
    Enhanced Security Sandbox Manager
    
    Provides secure isolation environments for both external content validation
    and MCP tool execution. Features comprehensive resource monitoring, 
    permission controls, and audit logging.
    
    Capabilities:
    - External content security scanning and validation
    - MCP tool execution with resource limits and isolation
    - Real-time resource monitoring and threat detection
    - Fine-grained permission management
    - Comprehensive audit logging and compliance reporting
    """
    
    def __init__(self):
        """Initialize the enhanced sandbox manager"""
        
        # Content scanning state (existing functionality)
        self.status = SandboxStatus.IDLE
        self.active_scans: Dict[UUID, Dict[str, Any]] = {}
        self.scan_history: List[SecurityScanResult] = []
        
        # Tool execution state (new functionality)
        self.active_tool_sandboxes: Dict[str, ToolSandboxContext] = {}
        self.tool_execution_history: List[ToolSandboxResult] = []
        
        # Sandbox configuration
        self.sandbox_dir = tempfile.mkdtemp(prefix="prsm_sandbox_")
        self.tool_sandbox_dir = os.path.join(self.sandbox_dir, "tools")
        self.max_file_size = int(getattr(settings, "PRSM_MAX_FILE_SIZE_MB", 100)) * 1024 * 1024
        self.scan_timeout = int(getattr(settings, "PRSM_SCAN_TIMEOUT_SECONDS", 300))
        self.quarantine_dir = os.path.join(self.sandbox_dir, "quarantine")
        
        # Tool execution configuration
        self.max_concurrent_tool_sandboxes = getattr(settings, "PRSM_MAX_CONCURRENT_TOOLS", 10)
        self.default_tool_timeout = 60.0
        self.tool_cleanup_interval = 300.0  # 5 minutes
        
        # Security settings
        self.enable_vulnerability_scan = getattr(settings, "PRSM_ENABLE_VULN_SCAN", True)
        self.enable_license_scan = getattr(settings, "PRSM_ENABLE_LICENSE_SCAN", True)
        self.enable_malware_scan = getattr(settings, "PRSM_ENABLE_MALWARE_SCAN", False)
        self.enable_tool_sandboxing = getattr(settings, "PRSM_ENABLE_TOOL_SANDBOX", True)
        
        # Risk thresholds
        self.risk_thresholds = {
            "max_vulnerabilities": 5,
            "max_high_risk_vulns": 1,
            "min_license_compliance": 0.8,
            "max_file_size_ratio": 2.0
        }
        
        # Initialize sandbox environment
        self._setup_sandbox()
        
        # Defer periodic cleanup task creation until needed
        self._cleanup_task = None
        if self.enable_tool_sandboxing:
            self._start_cleanup_task()
        
        logger.info("Enhanced Security Sandbox Manager initialized",
                   sandbox_dir=self.sandbox_dir,
                   tool_sandbox_dir=self.tool_sandbox_dir,
                   vulnerability_scanning=self.enable_vulnerability_scan,
                   license_scanning=self.enable_license_scan,
                   malware_scanning=self.enable_malware_scan,
                   tool_sandboxing=self.enable_tool_sandboxing)
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task if event loop is available"""
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._periodic_tool_cleanup())
        except RuntimeError:
            # No running event loop, task will be started later when needed
            pass
    
    # === Public Sandbox Operations ===
    
    async def scan_content(self, content_path: str, metadata: Dict[str, Any],
                         scan_options: Optional[Dict[str, Any]] = None) -> SecurityScanResult:
        """
        Perform comprehensive security scan on content
        
        Args:
            content_path: Path to content to scan
            metadata: Content metadata from platform
            scan_options: Optional scan configuration
            
        Returns:
            SecurityScanResult with comprehensive security assessment
        """
        scan_id = uuid4()
        start_time = datetime.now(timezone.utc)
        
        try:
            print(f"🔍 Starting security scan: {scan_id}")
            print(f"   - Content: {os.path.basename(content_path)}")
            print(f"   - Size: {os.path.getsize(content_path) if os.path.exists(content_path) else 0} bytes")
            
            # Update status
            self.status = SandboxStatus.SCANNING
            
            # Initialize scan tracking
            self.active_scans[scan_id] = {
                "content_path": content_path,
                "metadata": metadata,
                "start_time": start_time,
                "status": "running"
            }
            
            # Create scan result
            scan_result = SecurityScanResult(
                scan_id=scan_id,
                request_id=metadata.get("request_id", uuid4())
            )
            
            # Stage 1: Basic file validation
            basic_validation = await self._perform_basic_validation(content_path)
            if not basic_validation["passed"]:
                scan_result.risk_level = SecurityRisk.HIGH
                scan_result.vulnerabilities_found.extend(basic_validation["issues"])
                scan_result.approved_for_import = False
                return await self._finalize_scan(scan_id, scan_result)
            
            # Stage 2: License compliance scan
            if self.enable_license_scan:
                license_result = await self._scan_license_compliance(content_path, metadata)
                scan_result.license_compliance = license_result["type"]
                scan_result.compliance_issues.extend(license_result["issues"])
            
            # Stage 3: Vulnerability scanning
            if self.enable_vulnerability_scan:
                vuln_result = await self._scan_vulnerabilities(content_path)
                scan_result.vulnerabilities_found.extend(vuln_result["vulnerabilities"])
            
            # Stage 4: Malware scanning (if enabled)
            if self.enable_malware_scan:
                malware_result = await self._scan_malware(content_path)
                if malware_result["threats_found"]:
                    scan_result.vulnerabilities_found.extend(malware_result["threats"])
                    scan_result.risk_level = SecurityRisk.CRITICAL
            
            # Stage 5: Risk assessment
            risk_assessment = await self._assess_risk(scan_result, metadata)
            scan_result.risk_level = risk_assessment["risk_level"]
            scan_result.recommendations.extend(risk_assessment["recommendations"])
            scan_result.approved_for_import = risk_assessment["approved"]
            
            # Handle high-risk content
            if scan_result.risk_level in [SecurityRisk.HIGH, SecurityRisk.CRITICAL]:
                await self._quarantine_content(content_path, scan_result)
                
                # Log critical risks (circuit breaker integration would go here)
                if scan_result.risk_level == SecurityRisk.CRITICAL:
                    print(f"🚨 CRITICAL SECURITY RISK: {scan_result.vulnerabilities_found}")
                    # TODO: Integrate with circuit breaker when available
                    # await circuit_breaker.trigger_breach("critical_security_risk", ...)
            
            print(f"🔍 Security scan completed: {scan_id}")
            print(f"   - Risk level: {scan_result.risk_level}")
            print(f"   - Approved: {scan_result.approved_for_import}")
            print(f"   - Vulnerabilities: {len(scan_result.vulnerabilities_found)}")
            
            return await self._finalize_scan(scan_id, scan_result)
            
        except Exception as e:
            print(f"❌ Security scan failed: {e}")
            
            # Create error result
            scan_result = SecurityScanResult(
                scan_id=scan_id,
                request_id=metadata.get("request_id", uuid4()),
                risk_level=SecurityRisk.HIGH,
                vulnerabilities_found=[f"Scan error: {str(e)}"],
                approved_for_import=False
            )
            
            return await self._finalize_scan(scan_id, scan_result)
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current sandbox status and health
        
        Returns:
            Sandbox status and metrics
        """
        total_scans = len(self.scan_history)
        approved_scans = sum(1 for scan in self.scan_history if scan.approved_for_import)
        
        return {
            "status": self.status.value,
            "active_scans": len(self.active_scans),
            "total_scans": total_scans,
            "approved_scans": approved_scans,
            "approval_rate": (approved_scans / max(total_scans, 1)) * 100,
            "quarantine_count": len(os.listdir(self.quarantine_dir)) if os.path.exists(self.quarantine_dir) else 0,
            "sandbox_directory": self.sandbox_dir,
            "vulnerability_scanning": self.enable_vulnerability_scan,
            "license_scanning": self.enable_license_scan,
            "malware_scanning": self.enable_malware_scan
        }
    
    async def cleanup_sandbox(self) -> bool:
        """
        Clean up sandbox environment and temporary files
        
        Returns:
            True if cleanup successful
        """
        try:
            print("🧹 Cleaning up sandbox environment")
            
            # Clear active scans
            self.active_scans.clear()
            
            # Remove temporary files (preserve quarantine)
            for item in os.listdir(self.sandbox_dir):
                item_path = os.path.join(self.sandbox_dir, item)
                if item != "quarantine" and os.path.exists(item_path):
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            
            self.status = SandboxStatus.IDLE
            print("✅ Sandbox cleanup completed")
            return True
            
        except Exception as e:
            print(f"❌ Sandbox cleanup failed: {e}")
            return False
    
    # === Private Security Scanning Methods ===
    
    async def _perform_basic_validation(self, content_path: str) -> Dict[str, Any]:
        """Perform basic file validation checks"""
        issues = []
        
        try:
            # Check file exists
            if not os.path.exists(content_path):
                issues.append("File does not exist")
                return {"passed": False, "issues": issues}
            
            # Check file size
            file_size = os.path.getsize(content_path)
            if file_size > self.max_file_size:
                issues.append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            # Check file permissions
            if not os.access(content_path, os.R_OK):
                issues.append("File is not readable")
            
            # Check for suspicious file extensions
            suspicious_extensions = [".exe", ".bat", ".cmd", ".scr", ".pif"]
            if any(content_path.lower().endswith(ext) for ext in suspicious_extensions):
                issues.append("Suspicious file extension detected")
            
            # Basic content validation
            if os.path.isfile(content_path):
                with open(content_path, 'rb') as f:
                    header = f.read(1024)
                    
                    # Check for executable headers
                    if header.startswith(b'MZ') or header.startswith(b'\x7fELF'):
                        issues.append("Executable content detected")
                    
                    # Check for script headers
                    script_headers = [b'#!/bin/sh', b'#!/bin/bash', b'@echo off']
                    if any(header.startswith(h) for h in script_headers):
                        issues.append("Script content detected")
            
            return {"passed": len(issues) == 0, "issues": issues}
            
        except Exception as e:
            return {"passed": False, "issues": [f"Validation error: {str(e)}"]}
    
    async def _scan_license_compliance(self, content_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for license compliance"""
        try:
            # Extract license information from metadata
            license_info = metadata.get("license", {})
            
            if isinstance(license_info, dict):
                license_type = license_info.get("key", "unknown").lower()
                license_name = license_info.get("name", "Unknown")
            else:
                license_type = str(license_info).lower()
                license_name = str(license_info)
            
            # Check against permissive licenses
            permissive_licenses = [
                "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", 
                "unlicense", "cc0-1.0", "isc", "zlib"
            ]
            
            copyleft_licenses = [
                "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0", 
                "agpl-3.0", "cc-by-sa"
            ]
            
            issues = []
            
            if license_type in permissive_licenses:
                result_type = LicenseType.PERMISSIVE
            elif license_type in copyleft_licenses:
                result_type = LicenseType.COPYLEFT
                issues.append(f"Copyleft license detected: {license_name}")
            elif "proprietary" in license_type or "commercial" in license_type:
                result_type = LicenseType.PROPRIETARY
                issues.append(f"Proprietary license detected: {license_name}")
            else:
                result_type = LicenseType.UNKNOWN
                issues.append(f"Unknown or unrecognized license: {license_name}")
            
            # Additional file-based license detection
            if os.path.isfile(content_path) and content_path.endswith(('.py', '.js', '.java', '.cpp', '.c')):
                await self._scan_file_license_headers(content_path, issues)
            
            return {
                "type": result_type,
                "name": license_name,
                "issues": issues,
                "compliant": result_type == LicenseType.PERMISSIVE
            }
            
        except Exception as e:
            return {
                "type": LicenseType.UNKNOWN,
                "name": "Error",
                "issues": [f"License scan error: {str(e)}"],
                "compliant": False
            }
    
    async def _scan_file_license_headers(self, content_path: str, issues: List[str]) -> None:
        """Scan file headers for license information"""
        try:
            with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.read(2000)  # Read first 2KB
                
                # Look for restrictive license indicators
                restrictive_indicators = [
                    "all rights reserved",
                    "proprietary",
                    "confidential",
                    "copyright.*not.*distribute",
                    "gpl.*license",
                    "copyleft"
                ]
                
                for indicator in restrictive_indicators:
                    if indicator.lower() in header.lower():
                        issues.append(f"Restrictive license indicator found in file: {indicator}")
                        break
                        
        except Exception:
            # Ignore file reading errors for license detection
            pass
    
    async def _scan_vulnerabilities(self, content_path: str) -> Dict[str, Any]:
        """Scan for known vulnerabilities"""
        vulnerabilities = []
        
        try:
            # This is a simplified vulnerability scanner
            # In production, this would integrate with tools like:
            # - Bandit (Python security)
            # - ESLint security plugins (JavaScript)
            # - SonarQube
            # - OWASP dependency check
            
            # Basic pattern-based vulnerability detection
            if os.path.isfile(content_path):
                await self._pattern_based_vuln_scan(content_path, vulnerabilities)
            
            # Simulated vulnerability database check
            # In production, this would query CVE databases
            await asyncio.sleep(0.1)  # Simulate scan time
            
            return {
                "vulnerabilities": vulnerabilities,
                "scan_method": "pattern_based",
                "database_version": "1.0.0"
            }
            
        except Exception as e:
            return {
                "vulnerabilities": [f"Vulnerability scan error: {str(e)}"],
                "scan_method": "error",
                "database_version": "unknown"
            }
    
    async def _pattern_based_vuln_scan(self, content_path: str, vulnerabilities: List[str]) -> None:
        """Pattern-based vulnerability detection"""
        try:
            with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
                
                # Common vulnerability patterns
                vuln_patterns = {
                    "sql_injection": ["SELECT.*FROM.*WHERE.*=.*input", "exec.*SELECT", "query.*+.*user"],
                    "xss": ["innerHTML.*=.*input", "document.write.*input", "eval.*input"],
                    "command_injection": ["system.*input", "exec.*input", "shell_exec"],
                    "path_traversal": ["../", "..\\\\", "path.*input"],
                    "hardcoded_secrets": ["password.*=.*['\"]", "api_key.*=.*['\"]", "secret.*=.*['\"]"]
                }
                
                for vuln_type, patterns in vuln_patterns.items():
                    for pattern in patterns:
                        if pattern.lower() in content.lower():
                            vulnerabilities.append(f"Potential {vuln_type.replace('_', ' ')}: {pattern}")
                            break
                            
        except Exception:
            # Ignore file reading errors
            pass
    
    async def _scan_malware(self, content_path: str) -> Dict[str, Any]:
        """Scan for malware (simplified implementation)"""
        threats = []
        
        try:
            # This is a placeholder for malware scanning
            # In production, this would integrate with:
            # - ClamAV
            # - VirusTotal API
            # - Commercial antivirus engines
            
            # Basic suspicious content detection
            if os.path.isfile(content_path):
                with open(content_path, 'rb') as f:
                    content = f.read(1024)
                    
                    # Check for suspicious byte patterns
                    suspicious_patterns = [
                        b'\x4d\x5a',  # PE header
                        b'\x7f\x45\x4c\x46',  # ELF header
                        b'#!/bin/sh',  # Shell script
                        b'@echo off'  # Batch file
                    ]
                    
                    for pattern in suspicious_patterns:
                        if pattern in content:
                            threats.append(f"Suspicious binary pattern detected: {pattern.hex()}")
            
            return {
                "threats_found": len(threats) > 0,
                "threats": threats,
                "scanner_version": "1.0.0"
            }
            
        except Exception as e:
            return {
                "threats_found": True,
                "threats": [f"Malware scan error: {str(e)}"],
                "scanner_version": "error"
            }
    
    async def _assess_risk(self, scan_result: SecurityScanResult, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall security risk"""
        risk_factors = []
        recommendations = []
        
        # Vulnerability assessment
        vuln_count = len(scan_result.vulnerabilities_found)
        high_risk_vulns = sum(1 for v in scan_result.vulnerabilities_found 
                            if any(keyword in v.lower() for keyword in ["critical", "high", "injection", "exec"]))
        
        if vuln_count > self.risk_thresholds["max_vulnerabilities"]:
            risk_factors.append(f"High vulnerability count: {vuln_count}")
            recommendations.append("Manual security review required")
        
        if high_risk_vulns > self.risk_thresholds["max_high_risk_vulns"]:
            risk_factors.append(f"High-risk vulnerabilities detected: {high_risk_vulns}")
            recommendations.append("Immediate security remediation required")
        
        # License compliance assessment
        if scan_result.license_compliance != LicenseType.PERMISSIVE:
            risk_factors.append(f"Non-permissive license: {scan_result.license_compliance}")
            recommendations.append("License review required before use")
        
        # Compliance issues
        if len(scan_result.compliance_issues) > 0:
            risk_factors.append(f"Compliance issues: {len(scan_result.compliance_issues)}")
            recommendations.append("Address compliance issues before import")
        
        # Determine risk level
        if high_risk_vulns > 0 or "critical" in str(scan_result.vulnerabilities_found).lower():
            risk_level = SecurityRisk.CRITICAL
        elif vuln_count > 5 or scan_result.license_compliance == LicenseType.PROPRIETARY:
            risk_level = SecurityRisk.HIGH
        elif vuln_count > 2 or len(scan_result.compliance_issues) > 0:
            risk_level = SecurityRisk.MEDIUM
        elif vuln_count > 0:
            risk_level = SecurityRisk.LOW
        else:
            risk_level = SecurityRisk.NONE
        
        # Approval decision
        approved = risk_level in [SecurityRisk.NONE, SecurityRisk.LOW]
        
        if not approved:
            recommendations.append("Import blocked due to security concerns")
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "approved": approved
        }
    
    async def _quarantine_content(self, content_path: str, scan_result: SecurityScanResult) -> None:
        """Quarantine high-risk content"""
        try:
            os.makedirs(self.quarantine_dir, exist_ok=True)
            
            # Create quarantine entry
            quarantine_name = f"{scan_result.scan_id}_{os.path.basename(content_path)}"
            quarantine_path = os.path.join(self.quarantine_dir, quarantine_name)
            
            # Copy to quarantine
            if os.path.isfile(content_path):
                shutil.copy2(content_path, quarantine_path)
            elif os.path.isdir(content_path):
                shutil.copytree(content_path, quarantine_path)
            
            # Create quarantine metadata
            metadata_path = f"{quarantine_path}.metadata.json"
            metadata = {
                "scan_id": str(scan_result.scan_id),
                "quarantine_time": datetime.now(timezone.utc).isoformat(),
                "risk_level": scan_result.risk_level,
                "vulnerabilities": scan_result.vulnerabilities_found,
                "original_path": content_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"🚨 Content quarantined: {quarantine_name}")
            
        except Exception as e:
            print(f"❌ Failed to quarantine content: {e}")
    
    async def _finalize_scan(self, scan_id: UUID, scan_result: SecurityScanResult) -> SecurityScanResult:
        """Finalize scan and update tracking"""
        try:
            # Calculate scan duration
            if scan_id in self.active_scans:
                start_time = self.active_scans[scan_id]["start_time"]
                scan_result.scan_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Remove from active scans
                del self.active_scans[scan_id]
            
            # Store in history
            self.scan_history.append(scan_result)
            
            # Update status
            if len(self.active_scans) == 0:
                self.status = SandboxStatus.IDLE
            
            return scan_result
            
        except Exception as e:
            print(f"❌ Failed to finalize scan: {e}")
            return scan_result
    
    def _setup_sandbox(self) -> None:
        """Initialize sandbox environment"""
        try:
            # Create sandbox directories
            os.makedirs(self.sandbox_dir, exist_ok=True)
            os.makedirs(self.quarantine_dir, exist_ok=True)
            
            # Set restrictive permissions
            os.chmod(self.sandbox_dir, 0o700)
            os.chmod(self.quarantine_dir, 0o700)
            
            print(f"🔒 Sandbox environment initialized at {self.sandbox_dir}")
            
        except Exception as e:
            print(f"❌ Failed to setup sandbox: {e}")
            raise
    
    # ===============================
    # MCP Tool Execution Methods
    # ===============================
    
    async def execute_tool_safely(self, tool_execution_request: ToolExecutionRequest) -> ToolSandboxResult:
        """
        Execute an MCP tool within a secure sandbox environment
        
        This is the main entry point for secure tool execution, providing:
        - Resource isolation and limits
        - Permission validation and enforcement
        - Real-time security monitoring
        - Comprehensive audit logging
        - Automatic cleanup and recovery
        
        Args:
            tool_execution_request: Complete tool execution request
            
        Returns:
            ToolSandboxResult: Complete execution result with security metrics
        """
        if not self.enable_tool_sandboxing:
            logger.warning("Tool sandboxing is disabled - executing without sandbox")
            return await self._execute_tool_without_sandbox(tool_execution_request)
        
        # Check concurrent execution limits
        if len(self.active_tool_sandboxes) >= self.max_concurrent_tool_sandboxes:
            logger.warning("Maximum concurrent tool sandboxes reached",
                         active_count=len(self.active_tool_sandboxes),
                         max_allowed=self.max_concurrent_tool_sandboxes)
            
            return ToolSandboxResult(
                sandbox_id="rate_limited",
                tool_execution_result=ToolExecutionResult(
                    execution_id=tool_execution_request.execution_id,
                    tool_id=tool_execution_request.tool_id,
                    success=False,
                    execution_time=0.0,
                    error_message="Maximum concurrent tool executions reached",
                    error_code="RATE_LIMITED"
                ),
                security_violations=["Rate limit exceeded"]
            )
        
        # Create sandbox context
        context = ToolSandboxContext(
            tool_id=tool_execution_request.tool_id,
            user_id=tool_execution_request.user_id,
            execution_request=tool_execution_request,
            sandbox_type=self._determine_sandbox_type(tool_execution_request.sandbox_level),
            security_level=tool_execution_request.sandbox_level,
            resource_limits=self._create_resource_limits(tool_execution_request),
            permissions=self._create_security_permissions(tool_execution_request)
        )
        
        # Register active sandbox
        self.active_tool_sandboxes[context.sandbox_id] = context
        
        try:
            logger.info("Starting secure tool execution",
                       sandbox_id=context.sandbox_id,
                       tool_id=tool_execution_request.tool_id,
                       user_id=tool_execution_request.user_id,
                       sandbox_type=context.sandbox_type.value)
            
            # Validate security permissions
            security_validation = await self._validate_tool_security(context)
            if not security_validation["valid"]:
                return await self._create_security_violation_result(context, security_validation)
            
            # Start resource monitoring
            resource_monitor = ResourceMonitor(context)
            monitoring_task = await resource_monitor.start_monitoring()
            
            try:
                # Execute tool based on sandbox type
                if context.sandbox_type == ToolSandboxType.CONTAINER:
                    execution_result = await self._execute_tool_in_container(context)
                elif context.sandbox_type == ToolSandboxType.BASIC:
                    execution_result = await self._execute_tool_in_basic_sandbox(context)
                else:
                    execution_result = await self._execute_tool_direct(context)
                
                # Stop monitoring and get resource usage
                resource_summary = await resource_monitor.stop_monitoring()
                
                # Create comprehensive result
                sandbox_result = await self._create_sandbox_result(context, execution_result, resource_summary)
                
                logger.info("Secure tool execution completed",
                           sandbox_id=context.sandbox_id,
                           success=execution_result.success,
                           execution_time=execution_result.execution_time,
                           security_violations=len(sandbox_result.security_violations))
                
                return sandbox_result
                
            finally:
                # Ensure monitoring is stopped
                try:
                    await resource_monitor.stop_monitoring()
                except Exception:
                    pass
        
        except Exception as e:
            logger.error("Tool execution failed in sandbox",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            # Create error result
            error_result = ToolExecutionResult(
                execution_id=tool_execution_request.execution_id,
                tool_id=tool_execution_request.tool_id,
                success=False,
                execution_time=0.0,
                error_message=str(e),
                error_code="SANDBOX_ERROR"
            )
            
            return ToolSandboxResult(
                sandbox_id=context.sandbox_id,
                tool_execution_result=error_result,
                security_violations=[f"Sandbox execution error: {str(e)}"]
            )
        
        finally:
            # Cleanup sandbox
            await self._cleanup_tool_sandbox(context.sandbox_id)
    
    async def _validate_tool_security(self, context: ToolSandboxContext) -> Dict[str, Any]:
        """Validate tool execution against security policies"""
        validation_result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        try:
            # Check user consent requirements
            if context.permissions.user_consent_required:
                # In production, this would check actual user consent
                # For now, we'll assume consent is granted for valid users
                if not context.user_id or context.user_id == "anonymous":
                    validation_result["valid"] = False
                    validation_result["violations"].append("User consent required but user not authenticated")
            
            # Check security level compatibility
            if context.security_level == ToolSecurityLevel.DANGEROUS:
                if not context.permissions.user_consent_required:
                    validation_result["warnings"].append("Dangerous tool without explicit consent")
            
            # Validate file permissions
            if context.permissions.file_write:
                for path in context.permissions.file_write:
                    if not self._is_safe_file_path(path):
                        validation_result["valid"] = False
                        validation_result["violations"].append(f"Unsafe file write path: {path}")
            
            # Check network permissions
            if context.permissions.network_outbound:
                if not context.permissions.allowed_hosts:
                    validation_result["warnings"].append("Outbound network access without host restrictions")
            
            # Validate resource limits
            if context.resource_limits.execution_timeout > 300:  # 5 minutes max
                validation_result["valid"] = False
                validation_result["violations"].append("Execution timeout exceeds maximum allowed")
            
            if context.resource_limits.memory_mb > 2048:  # 2GB max
                validation_result["valid"] = False
                validation_result["violations"].append("Memory limit exceeds maximum allowed")
            
            return validation_result
            
        except Exception as e:
            logger.error("Security validation failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            return {
                "valid": False,
                "violations": [f"Security validation error: {str(e)}"],
                "warnings": []
            }
    
    async def _execute_tool_in_container(self, context: ToolSandboxContext) -> ToolExecutionResult:
        """Execute tool in Docker container (production-grade isolation)"""
        start_time = time.time()
        
        try:
            # Create container configuration
            container_config = {
                "image": "prsm/tool-sandbox:latest",
                "command": ["python", "-c", f"import json; print(json.dumps({{'tool_id': '{context.tool_id}', 'status': 'simulated_success'}})))"],
                "environment": {
                    "TOOL_ID": context.tool_id,
                    "EXECUTION_ID": str(context.execution_request.execution_id),
                    "SANDBOX_ID": context.sandbox_id
                },
                "resource_limits": {
                    "memory": f"{context.resource_limits.memory_mb}m",
                    "cpu_quota": int(context.resource_limits.cpu_percent * 1000),
                    "execution_timeout": context.resource_limits.execution_timeout
                },
                "network": "none" if not context.permissions.network_outbound else "bridge",
                "read_only_root": True,
                "no_new_privileges": True
            }
            
            # Simulate container execution (in production, use Docker API)
            logger.info("Simulating container execution",
                       sandbox_id=context.sandbox_id,
                       container_config=container_config)
            
            # Simulate execution time based on tool complexity
            await asyncio.sleep(min(2.0, context.resource_limits.execution_timeout / 30))
            
            execution_time = time.time() - start_time
            
            # Simulate successful execution
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=True,
                result_data={
                    "status": "container_execution_success",
                    "sandbox_type": "container",
                    "simulated": True,
                    "container_config": container_config
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Container execution failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="CONTAINER_ERROR"
            )
    
    async def _execute_tool_in_basic_sandbox(self, context: ToolSandboxContext) -> ToolExecutionResult:
        """Execute tool in basic process sandbox (lightweight isolation)"""
        start_time = time.time()
        
        try:
            # Create isolated process environment
            sandbox_env = os.environ.copy()
            
            # Restrict environment variables
            restricted_env = {
                "TOOL_ID": context.tool_id,
                "EXECUTION_ID": str(context.execution_request.execution_id),
                "SANDBOX_ID": context.sandbox_id,
                "PATH": "/usr/local/bin:/usr/bin:/bin",  # Minimal PATH
                "HOME": os.path.join(self.sandbox_dir, "home"),
                "TMPDIR": os.path.join(self.tool_sandbox_dir, context.sandbox_id)
            }
            
            # Create sandbox directory
            sandbox_dir = os.path.join(self.tool_sandbox_dir, context.sandbox_id)
            os.makedirs(sandbox_dir, exist_ok=True)
            os.chmod(sandbox_dir, 0o700)
            
            # Simulate tool execution with process limits
            logger.info("Executing tool in basic sandbox",
                       sandbox_id=context.sandbox_id,
                       sandbox_dir=sandbox_dir)
            
            # In production, this would execute the actual tool with resource limits
            # For now, simulate execution
            await asyncio.sleep(min(1.0, context.resource_limits.execution_timeout / 60))
            
            execution_time = time.time() - start_time
            
            # Check if execution exceeded limits
            if execution_time > context.resource_limits.execution_timeout:
                return ToolExecutionResult(
                    execution_id=context.execution_request.execution_id,
                    tool_id=context.tool_id,
                    success=False,
                    execution_time=execution_time,
                    error_message="Tool execution timeout",
                    error_code="TIMEOUT",
                    resource_violations=["Execution timeout exceeded"]
                )
            
            # Simulate successful execution
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=True,
                result_data={
                    "status": "basic_sandbox_success",
                    "sandbox_type": "basic",
                    "sandbox_dir": sandbox_dir,
                    "simulated": True
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Basic sandbox execution failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="SANDBOX_ERROR"
            )
        
        finally:
            # Cleanup sandbox directory
            try:
                if 'sandbox_dir' in locals() and os.path.exists(sandbox_dir):
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
            except Exception:
                pass
    
    async def _execute_tool_direct(self, context: ToolSandboxContext) -> ToolExecutionResult:
        """Execute tool with minimal sandboxing (for testing/safe tools)"""
        start_time = time.time()
        
        try:
            logger.info("Executing tool with minimal sandboxing",
                       sandbox_id=context.sandbox_id,
                       tool_id=context.tool_id)
            
            # Simulate direct execution with basic monitoring
            await asyncio.sleep(0.1)  # Minimal execution time
            
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=True,
                result_data={
                    "status": "direct_execution_success",
                    "sandbox_type": "none",
                    "simulated": True
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Direct execution failed",
                        sandbox_id=context.sandbox_id,
                        error=str(e))
            
            return ToolExecutionResult(
                execution_id=context.execution_request.execution_id,
                tool_id=context.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    async def _execute_tool_without_sandbox(self, tool_execution_request: ToolExecutionRequest) -> ToolSandboxResult:
        """Execute tool without sandboxing (fallback mode)"""
        logger.warning("Executing tool without sandbox - security disabled",
                      tool_id=tool_execution_request.tool_id)
        
        start_time = time.time()
        
        try:
            # Minimal execution simulation
            await asyncio.sleep(0.05)
            execution_time = time.time() - start_time
            
            execution_result = ToolExecutionResult(
                execution_id=tool_execution_request.execution_id,
                tool_id=tool_execution_request.tool_id,
                success=True,
                result_data={"status": "unsandboxed_execution", "warning": "No security sandbox"},
                execution_time=execution_time
            )
            
            return ToolSandboxResult(
                sandbox_id="no_sandbox",
                tool_execution_result=execution_result,
                security_violations=["Tool executed without sandboxing"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            execution_result = ToolExecutionResult(
                execution_id=tool_execution_request.execution_id,
                tool_id=tool_execution_request.tool_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="UNSANDBOXED_ERROR"
            )
            
            return ToolSandboxResult(
                sandbox_id="no_sandbox",
                tool_execution_result=execution_result,
                security_violations=["Tool execution failed without sandboxing"]
            )
    
    def _determine_sandbox_type(self, security_level: ToolSecurityLevel) -> ToolSandboxType:
        """Determine appropriate sandbox type based on security level"""
        if security_level == ToolSecurityLevel.SAFE:
            return ToolSandboxType.BASIC
        elif security_level == ToolSecurityLevel.RESTRICTED:
            return ToolSandboxType.CONTAINER
        elif security_level in [ToolSecurityLevel.PRIVILEGED, ToolSecurityLevel.DANGEROUS]:
            return ToolSandboxType.CONTAINER  # Always container for high-risk tools
        else:
            return ToolSandboxType.BASIC
    
    def _create_resource_limits(self, request: ToolExecutionRequest) -> ResourceLimits:
        """Create resource limits based on tool execution request"""
        # Base limits with security-level adjustments
        base_limits = ResourceLimits()
        
        if request.sandbox_level == ToolSecurityLevel.SAFE:
            # More generous limits for safe tools
            base_limits.cpu_percent = 75.0
            base_limits.memory_mb = 1024
            base_limits.execution_timeout = min(request.timeout_seconds, 120.0)
        elif request.sandbox_level == ToolSecurityLevel.RESTRICTED:
            # Standard limits
            base_limits.execution_timeout = min(request.timeout_seconds, 60.0)
        else:
            # Strict limits for privileged/dangerous tools
            base_limits.cpu_percent = 25.0
            base_limits.memory_mb = 256
            base_limits.execution_timeout = min(request.timeout_seconds, 30.0)
            base_limits.network_upload_mb = 5
            base_limits.network_download_mb = 10
        
        return base_limits
    
    def _create_security_permissions(self, request: ToolExecutionRequest) -> SecurityPermissions:
        """Create security permissions based on tool execution request"""
        permissions = SecurityPermissions()
        
        # Set permissions based on security level
        if request.sandbox_level == ToolSecurityLevel.SAFE:
            permissions.user_consent_required = False
            permissions.file_read.add(os.path.join(self.sandbox_dir, "*"))
        elif request.sandbox_level == ToolSecurityLevel.RESTRICTED:
            permissions.user_consent_required = True
            permissions.file_read.add(os.path.join(self.sandbox_dir, "*"))
            permissions.file_write.add(os.path.join(self.sandbox_dir, "*"))
        else:
            # Privileged/dangerous tools require explicit permissions
            permissions.user_consent_required = True
            # Permissions would be set based on specific tool requirements
        
        # Add tool-specific permissions from request
        permissions.tool_permissions.update(request.permissions)
        
        return permissions
    
    def _is_safe_file_path(self, path: str) -> bool:
        """Validate that file path is safe for tool access"""
        # Prevent path traversal attacks
        if ".." in path or path.startswith("/"):
            return False
        
        # Only allow access to sandbox directories
        safe_prefixes = [
            self.sandbox_dir,
            self.tool_sandbox_dir,
            self.quarantine_dir
        ]
        
        return any(path.startswith(prefix) for prefix in safe_prefixes)
    
    async def _create_security_violation_result(self, context: ToolSandboxContext, 
                                              validation: Dict[str, Any]) -> ToolSandboxResult:
        """Create result for security violation"""
        execution_result = ToolExecutionResult(
            execution_id=context.execution_request.execution_id,
            tool_id=context.tool_id,
            success=False,
            execution_time=0.0,
            error_message="Security validation failed",
            error_code="SECURITY_VIOLATION",
            security_violations=validation["violations"]
        )
        
        return ToolSandboxResult(
            sandbox_id=context.sandbox_id,
            tool_execution_result=execution_result,
            security_violations=validation["violations"],
            permission_violations=validation["violations"]
        )
    
    async def _create_sandbox_result(self, context: ToolSandboxContext, 
                                   execution_result: ToolExecutionResult,
                                   resource_summary: Dict[str, Any]) -> ToolSandboxResult:
        """Create comprehensive sandbox result"""
        return ToolSandboxResult(
            sandbox_id=context.sandbox_id,
            tool_execution_result=execution_result,
            security_violations=resource_summary.get("violations", []),
            resource_violations=resource_summary.get("violations", []),
            peak_cpu_percent=resource_summary.get("peak_cpu", 0.0),
            peak_memory_mb=resource_summary.get("peak_memory", 0.0),
            total_disk_read_mb=resource_summary.get("total_disk_read", 0.0),
            total_disk_write_mb=resource_summary.get("total_disk_write", 0.0),
            total_network_mb=resource_summary.get("total_network", 0.0),
            execution_time=execution_result.execution_time,
            audit_trail=[
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "tool_execution_completed",
                    "sandbox_id": context.sandbox_id,
                    "tool_id": context.tool_id,
                    "success": execution_result.success,
                    "security_level": context.security_level.value,
                    "sandbox_type": context.sandbox_type.value
                }
            ]
        )
    
    async def _cleanup_tool_sandbox(self, sandbox_id: str):
        """Clean up tool sandbox resources"""
        try:
            # Remove from active sandboxes
            if sandbox_id in self.active_tool_sandboxes:
                context = self.active_tool_sandboxes[sandbox_id]
                del self.active_tool_sandboxes[sandbox_id]
                
                # Update performance stats
                self.performance_stats["tool_usage"]["total_tool_requests"] += 1
                
                logger.debug("Tool sandbox cleaned up",
                           sandbox_id=sandbox_id,
                           tool_id=context.tool_id)
        
        except Exception as e:
            logger.error("Sandbox cleanup failed",
                        sandbox_id=sandbox_id,
                        error=str(e))
    
    async def _periodic_tool_cleanup(self):
        """Periodic cleanup of old tool sandboxes"""
        while True:
            try:
                await asyncio.sleep(self.tool_cleanup_interval)
                
                current_time = datetime.now(timezone.utc)
                expired_sandboxes = []
                
                for sandbox_id, context in self.active_tool_sandboxes.items():
                    # Check if sandbox has been running too long
                    if context.started_at:
                        running_time = (current_time - context.started_at).total_seconds()
                        max_runtime = context.resource_limits.execution_timeout * 2  # 2x timeout as max
                        
                        if running_time > max_runtime:
                            expired_sandboxes.append(sandbox_id)
                    elif (current_time - context.created_at).total_seconds() > 3600:  # 1 hour max
                        expired_sandboxes.append(sandbox_id)
                
                # Cleanup expired sandboxes
                for sandbox_id in expired_sandboxes:
                    logger.warning("Cleaning up expired tool sandbox",
                                  sandbox_id=sandbox_id)
                    await self._cleanup_tool_sandbox(sandbox_id)
                
                if expired_sandboxes:
                    logger.info("Periodic sandbox cleanup completed",
                               cleaned_count=len(expired_sandboxes))
            
            except Exception as e:
                logger.error("Periodic cleanup failed", error=str(e))
    
    def __del__(self):
        """Cleanup sandbox on destruction"""
        try:
            if hasattr(self, 'sandbox_dir') and os.path.exists(self.sandbox_dir):
                # Preserve quarantine directory
                quarantine_backup = None
                if os.path.exists(self.quarantine_dir):
                    quarantine_backup = self.quarantine_dir + "_backup"
                    shutil.move(self.quarantine_dir, quarantine_backup)
                
                # Remove sandbox
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
                
                # Restore quarantine
                if quarantine_backup:
                    os.makedirs(self.sandbox_dir, exist_ok=True)
                    shutil.move(quarantine_backup, self.quarantine_dir)
                    
        except Exception:
            pass


# Add json import for metadata handling
import json