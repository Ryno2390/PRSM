"""
License Scanner
==============

Comprehensive license compliance scanner for external content
integration, ensuring PRSM only imports permissively licensed content.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from ..models.integration_models import LicenseType


class LicenseResult:
    """Result from license scanning operation"""
    
    def __init__(self, license_type: LicenseType, compliant: bool, 
                 details: Dict[str, Any], issues: List[str]):
        self.license_type = license_type
        self.compliant = compliant
        self.details = details
        self.issues = issues
        self.scan_time = datetime.now(timezone.utc)


class LicenseScanner:
    """
    License compliance scanner for external content
    """
    
    def __init__(self):
        """Initialize license scanner"""
        
        # License databases
        self.permissive_licenses = {
            "mit": "MIT License",
            "apache-2.0": "Apache License 2.0",
            "bsd-3-clause": "BSD 3-Clause License",
            "bsd-2-clause": "BSD 2-Clause License",
            "unlicense": "The Unlicense",
            "cc0-1.0": "Creative Commons Zero v1.0",
            "isc": "ISC License",
            "zlib": "zlib License"
        }
        
        self.copyleft_licenses = {
            "gpl-2.0": "GNU General Public License v2.0",
            "gpl-3.0": "GNU General Public License v3.0",
            "lgpl-2.1": "GNU Lesser General Public License v2.1",
            "lgpl-3.0": "GNU Lesser General Public License v3.0",
            "agpl-3.0": "GNU Affero General Public License v3.0",
            "cc-by-sa": "Creative Commons Attribution-ShareAlike"
        }
        
        print("📄 License Scanner initialized")
    
    async def scan_license(self, metadata: Dict[str, Any], 
                         content_path: Optional[str] = None) -> LicenseResult:
        """
        Scan content for license compliance
        
        Args:
            metadata: Content metadata containing license information
            content_path: Optional path to content for file-based scanning
            
        Returns:
            LicenseResult with compliance assessment
        """
        issues = []
        details = {}
        
        try:
            # Extract license from metadata
            license_info = metadata.get("license", {})
            
            if isinstance(license_info, dict):
                license_key = license_info.get("key", "unknown").lower()
                license_name = license_info.get("name", "Unknown")
                details = license_info.copy()
            else:
                license_key = str(license_info).lower()
                license_name = str(license_info)
                details = {"key": license_key, "name": license_name}
            
            # Determine license type
            if license_key in self.permissive_licenses:
                license_type = LicenseType.PERMISSIVE
                compliant = True
            elif license_key in self.copyleft_licenses:
                license_type = LicenseType.COPYLEFT
                compliant = False
                issues.append(f"Copyleft license not permitted: {license_name}")
            elif "proprietary" in license_key or "commercial" in license_key:
                license_type = LicenseType.PROPRIETARY
                compliant = False
                issues.append(f"Proprietary license not permitted: {license_name}")
            else:
                license_type = LicenseType.UNKNOWN
                compliant = False
                issues.append(f"Unknown or unrecognized license: {license_name}")
            
            # Additional file-based scanning if path provided
            if content_path:
                file_issues = await self._scan_file_licenses(content_path)
                issues.extend(file_issues)
                if file_issues:
                    compliant = False
            
            return LicenseResult(license_type, compliant, details, issues)
            
        except Exception as e:
            return LicenseResult(
                LicenseType.UNKNOWN, 
                False, 
                {"error": str(e)}, 
                [f"License scan error: {str(e)}"]
            )
    
    async def _scan_file_licenses(self, content_path: str) -> List[str]:
        """Scan files for embedded license information"""
        issues = []
        
        try:
            # This would be expanded to scan multiple file types
            # For now, just handle text-based files
            with open(content_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5KB
                
                # Look for restrictive license indicators
                restrictive_patterns = [
                    "all rights reserved",
                    "proprietary",
                    "confidential",
                    "commercial use prohibited",
                    "gpl license",
                    "copyleft"
                ]
                
                for pattern in restrictive_patterns:
                    if pattern.lower() in content.lower():
                        issues.append(f"Restrictive license text found: {pattern}")
                        
        except Exception:
            # Ignore file reading errors
            pass
        
        return issues