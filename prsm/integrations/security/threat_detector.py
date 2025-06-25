"""
Threat Detection System
======================

Advanced threat detection for PRSM integration layer.
Detects malicious content, suspicious patterns, and potential security threats.
"""

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4


class ThreatLevel(str, Enum):
    """Threat severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Types of threats that can be detected"""
    MALWARE = "malware"
    BACKDOOR = "backdoor"
    OBFUSCATED_CODE = "obfuscated_code"
    SUSPICIOUS_NETWORK = "suspicious_network"
    DATA_EXFILTRATION = "data_exfiltration"
    CRYPTOMINING = "cryptomining"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_IMPORTS = "suspicious_imports"


class ThreatResult:
    """Result from threat detection scan"""
    
    def __init__(self, threats: List[Dict[str, Any]], threat_level: ThreatLevel,
                 scan_method: str, details: Dict[str, Any]):
        self.threats = threats
        self.threat_level = threat_level
        self.scan_method = scan_method
        self.details = details
        self.scan_time = datetime.now(timezone.utc)
        self.scan_id = str(uuid4())


class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        """Initialize threat detector"""
        
        # Known malicious file hashes (in production this would be a larger database)
        self.malicious_hashes: Set[str] = {
            "d41d8cd98f00b204e9800998ecf8427e",  # Example hash
            "5d41402abc4b2a76b9719d911017c592",  # Example hash
        }
        
        # Suspicious patterns for different threat types
        self.threat_patterns = {
            ThreatType.BACKDOOR: {
                "patterns": [
                    r"eval\s*\(\s*base64_decode",
                    r"system\s*\(\s*\$_[A-Z]+",
                    r"shell_exec\s*\(\s*\$_",
                    r"exec\s*\(\s*\$_[GET|POST]",
                    r"\$_REQUEST\[.*\]\s*\(",
                    r"create_function.*eval"
                ],
                "severity": ThreatLevel.CRITICAL
            },
            ThreatType.OBFUSCATED_CODE: {
                "patterns": [
                    r"[A-Za-z0-9+/]{50,}={0,2}",  # Base64-like strings
                    r"\\x[0-9a-fA-F]{2}",  # Hex encoding
                    r"chr\s*\(\s*\d+\s*\)",  # Character encoding
                    r"String\.fromCharCode",  # JavaScript obfuscation
                    r"unescape\s*\(",  # URL decoding obfuscation
                    r"(?:var|let|const)\s+\w+\s*=\s*['\"][A-Za-z0-9+/]{100,}['\"]"
                ],
                "severity": ThreatLevel.HIGH
            },
            ThreatType.SUSPICIOUS_NETWORK: {
                "patterns": [
                    r"(?:http|https|ftp)://(?:[0-9]{1,3}\.){3}[0-9]{1,3}",  # IP addresses
                    r"socket\.connect\s*\(",
                    r"urllib\.request\.urlopen",
                    r"requests\.(?:get|post)\s*\(",
                    r"wget\s+http",
                    r"curl\s+.*http"
                ],
                "severity": ThreatLevel.MEDIUM
            },
            ThreatType.DATA_EXFILTRATION: {
                "patterns": [
                    r"(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]+['\"]",
                    r"(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]+['\"]",
                    r"smtp\.send",
                    r"email\.send",
                    r"keylog",
                    r"clipboard\.get"
                ],
                "severity": ThreatLevel.HIGH
            },
            ThreatType.CRYPTOMINING: {
                "patterns": [
                    r"stratum\+tcp://",
                    r"xmrig",
                    r"cryptonight",
                    r"monero",
                    r"mining[_-]pool",
                    r"hashrate"
                ],
                "severity": ThreatLevel.HIGH
            },
            ThreatType.PRIVILEGE_ESCALATION: {
                "patterns": [
                    r"sudo\s+.*-s",
                    r"setuid\s*\(",
                    r"setgid\s*\(",
                    r"/etc/passwd",
                    r"/etc/shadow",
                    r"chmod\s+[47]77"
                ],
                "severity": ThreatLevel.HIGH
            },
            ThreatType.SUSPICIOUS_IMPORTS: {
                "patterns": [
                    r"import\s+(?:os|sys|subprocess|socket|ctypes)",
                    r"from\s+(?:os|sys|subprocess|socket|ctypes)\s+import",
                    r"require\s*\(['\"](?:fs|child_process|net)['\"]",
                    r"#include\s*<(?:windows|winsock|sys/socket)\.h>"
                ],
                "severity": ThreatLevel.MEDIUM
            }
        }
        
        print("🛡️ Threat Detector initialized")
    
    async def scan_threats(self, content_path: str, 
                          metadata: Dict[str, Any]) -> ThreatResult:
        """
        Comprehensive threat detection scan
        
        Args:
            content_path: Path to content to scan
            metadata: Content metadata
            
        Returns:
            ThreatResult with threat assessment
        """
        threats = []
        details = {
            "files_scanned": 0,
            "patterns_checked": len(self.threat_patterns),
            "hash_checks": 0
        }
        
        try:
            # Hash-based detection
            hash_threats = await self._hash_based_detection(content_path)
            threats.extend(hash_threats["threats"])
            details.update(hash_threats["details"])
            
            # Pattern-based detection
            pattern_threats = await self._pattern_based_detection(content_path)
            threats.extend(pattern_threats["threats"])
            details.update(pattern_threats["details"])
            
            # Behavioral analysis
            behavior_threats = await self._behavioral_analysis(content_path, metadata)
            threats.extend(behavior_threats["threats"])
            details.update(behavior_threats["details"])
            
            # Determine overall threat level
            threat_level = self._assess_threat_level(threats)
            
            return ThreatResult(
                threats=threats,
                threat_level=threat_level,
                scan_method="comprehensive",
                details=details
            )
            
        except Exception as e:
            return ThreatResult(
                threats=[{"type": "scan_error", "description": f"Scan error: {str(e)}", "severity": ThreatLevel.HIGH}],
                threat_level=ThreatLevel.HIGH,
                scan_method="error",
                details={"error": str(e)}
            )
    
    async def _hash_based_detection(self, content_path: str) -> Dict[str, Any]:
        """Hash-based malware detection"""
        import os
        
        threats = []
        details = {"hash_checks": 0}
        
        try:
            if os.path.isfile(content_path):
                file_hash = self._calculate_file_hash(content_path)
                details["hash_checks"] = 1
                
                if file_hash in self.malicious_hashes:
                    threats.append({
                        "type": ThreatType.MALWARE,
                        "description": f"Known malicious file hash detected: {file_hash[:16]}...",
                        "severity": ThreatLevel.CRITICAL,
                        "file": os.path.basename(content_path)
                    })
                    
            elif os.path.isdir(content_path):
                for root, dirs, files in os.walk(content_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_hash = self._calculate_file_hash(file_path)
                        details["hash_checks"] += 1
                        
                        if file_hash in self.malicious_hashes:
                            threats.append({
                                "type": ThreatType.MALWARE,
                                "description": f"Known malicious file hash detected: {file_hash[:16]}...",
                                "severity": ThreatLevel.CRITICAL,
                                "file": file
                            })
            
            return {"threats": threats, "details": details}
            
        except Exception as e:
            return {
                "threats": [{"type": "hash_error", "description": f"Hash detection error: {str(e)}", "severity": ThreatLevel.MEDIUM}],
                "details": details
            }
    
    async def _pattern_based_detection(self, content_path: str) -> Dict[str, Any]:
        """Pattern-based threat detection"""
        import os
        
        threats = []
        details = {"files_scanned": 0}
        
        try:
            if os.path.isfile(content_path):
                file_threats = await self._scan_file_for_threats(content_path)
                threats.extend(file_threats)
                details["files_scanned"] = 1
                
            elif os.path.isdir(content_path):
                for root, dirs, files in os.walk(content_path):
                    for file in files:
                        if self._should_scan_file(file):
                            file_path = os.path.join(root, file)
                            file_threats = await self._scan_file_for_threats(file_path)
                            threats.extend(file_threats)
                            details["files_scanned"] += 1
            
            return {"threats": threats, "details": details}
            
        except Exception as e:
            return {
                "threats": [{"type": "pattern_error", "description": f"Pattern detection error: {str(e)}", "severity": ThreatLevel.MEDIUM}],
                "details": details
            }
    
    async def _scan_file_for_threats(self, file_path: str) -> List[Dict[str, Any]]:
        """Scan individual file for threat patterns"""
        import os
        
        threats = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(100000)  # Read first 100KB
                
                # Check each threat pattern
                for threat_type, threat_info in self.threat_patterns.items():
                    for pattern in threat_info["patterns"]:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                        if matches:
                            threats.append({
                                "type": threat_type,
                                "description": f"{threat_type.value.replace('_', ' ').title()} detected in {os.path.basename(file_path)}",
                                "severity": threat_info["severity"],
                                "file": os.path.basename(file_path),
                                "matches": len(matches),
                                "pattern": pattern[:50] + "..." if len(pattern) > 50 else pattern
                            })
                            break  # Only report once per threat type per file
                            
        except Exception:
            # Ignore file reading errors
            pass
        
        return threats
    
    async def _behavioral_analysis(self, content_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Behavioral analysis for suspicious activity"""
        threats = []
        details = {"behavioral_checks": 0}
        
        try:
            # Check metadata for suspicious indicators
            details["behavioral_checks"] += 1
            
            # Check for suspicious repository characteristics
            if "stars" in metadata and metadata["stars"] == 0:
                if "created_at" in metadata:
                    # New repository with no stars could be suspicious
                    threats.append({
                        "type": "suspicious_repo",
                        "description": "Repository with no stars and recent creation",
                        "severity": ThreatLevel.LOW
                    })
            
            # Check for suspicious naming patterns
            if "name" in metadata:
                name = metadata["name"].lower()
                suspicious_names = ["hack", "crack", "exploit", "backdoor", "keylog", "stealer"]
                for sus_name in suspicious_names:
                    if sus_name in name:
                        threats.append({
                            "type": "suspicious_naming",
                            "description": f"Suspicious name pattern detected: {sus_name}",
                            "severity": ThreatLevel.MEDIUM
                        })
                        break
            
            return {"threats": threats, "details": details}
            
        except Exception as e:
            return {
                "threats": [{"type": "behavior_error", "description": f"Behavioral analysis error: {str(e)}", "severity": ThreatLevel.LOW}],
                "details": details
            }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return "error_calculating_hash"
    
    def _should_scan_file(self, filename: str) -> bool:
        """Determine if file should be scanned for threats"""
        # Scan most file types, excluding obviously safe ones
        exclude_extensions = ['.txt', '.md', '.json', '.xml', '.yml', '.yaml', '.csv']
        
        return not any(filename.lower().endswith(ext) for ext in exclude_extensions)
    
    def _assess_threat_level(self, threats: List[Dict[str, Any]]) -> ThreatLevel:
        """Assess overall threat level based on detected threats"""
        if not threats:
            return ThreatLevel.NONE
        
        # Count threats by severity
        critical_count = sum(1 for t in threats if t.get("severity") == ThreatLevel.CRITICAL)
        high_count = sum(1 for t in threats if t.get("severity") == ThreatLevel.HIGH)
        medium_count = sum(1 for t in threats if t.get("severity") == ThreatLevel.MEDIUM)
        
        # Determine overall level
        if critical_count > 0:
            return ThreatLevel.CRITICAL
        elif high_count > 1:
            return ThreatLevel.HIGH
        elif high_count > 0 or medium_count > 2:
            return ThreatLevel.HIGH
        elif medium_count > 0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


# Global threat detector instance
threat_detector = ThreatDetector()