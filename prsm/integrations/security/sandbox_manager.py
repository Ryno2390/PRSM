"""
Security Sandbox Manager
========================

Secure isolation environment for validating external content before
integration into PRSM's ecosystem. Provides comprehensive security
scanning, license compliance checking, and vulnerability assessment.

Key Features:
- Isolated execution environment for untrusted content
- License compliance validation
- Vulnerability scanning and risk assessment
- Integration with PRSM's circuit breaker system
- Performance monitoring and resource limits
"""

import asyncio
import os
import tempfile
import shutil
import subprocess
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from ..models.integration_models import SecurityRisk, LicenseType, SecurityScanResult
from ...core.config import settings
from ...safety.circuit_breaker import circuit_breaker


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


class SandboxManager:
    """
    Secure sandbox environment for external content validation
    
    Provides isolated execution and comprehensive security scanning
    for content imported from external platforms.
    """
    
    def __init__(self):
        """Initialize the sandbox manager"""
        
        # Sandbox state
        self.status = SandboxStatus.IDLE
        self.active_scans: Dict[UUID, Dict[str, Any]] = {}
        self.scan_history: List[SecurityScanResult] = []
        
        # Sandbox configuration
        self.sandbox_dir = tempfile.mkdtemp(prefix="prsm_sandbox_")
        self.max_file_size = int(getattr(settings, "PRSM_MAX_FILE_SIZE_MB", 100)) * 1024 * 1024
        self.scan_timeout = int(getattr(settings, "PRSM_SCAN_TIMEOUT_SECONDS", 300))
        self.quarantine_dir = os.path.join(self.sandbox_dir, "quarantine")
        
        # Security settings
        self.enable_vulnerability_scan = getattr(settings, "PRSM_ENABLE_VULN_SCAN", True)
        self.enable_license_scan = getattr(settings, "PRSM_ENABLE_LICENSE_SCAN", True)
        self.enable_malware_scan = getattr(settings, "PRSM_ENABLE_MALWARE_SCAN", False)
        
        # Risk thresholds
        self.risk_thresholds = {
            "max_vulnerabilities": 5,
            "max_high_risk_vulns": 1,
            "min_license_compliance": 0.8,
            "max_file_size_ratio": 2.0
        }
        
        # Initialize sandbox environment
        self._setup_sandbox()
        
        print(f"🔒 Security Sandbox Manager initialized")
        print(f"   - Sandbox directory: {self.sandbox_dir}")
        print(f"   - Vulnerability scanning: {self.enable_vulnerability_scan}")
        print(f"   - License scanning: {self.enable_license_scan}")
        print(f"   - Malware scanning: {self.enable_malware_scan}")
    
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
                
                # Trigger circuit breaker for critical risks
                if scan_result.risk_level == SecurityRisk.CRITICAL:
                    await circuit_breaker.trigger_breach(
                        "critical_security_risk",
                        f"Critical security risk detected in imported content: {scan_result.vulnerabilities_found}"
                    )
            
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