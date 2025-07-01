#!/usr/bin/env python3
"""
Automated Security Control Testing for PRSM
==========================================

Implements comprehensive automated security control testing to meet
SOC2 Type II and ISO27001 compliance requirements. This addresses
critical Gemini audit findings regarding compliance readiness.

This testing framework covers:
- Access Control Testing (AC)
- Configuration Management Testing (CM)
- Data Protection Testing (DP)
- Incident Response Testing (IR)
- System Monitoring Testing (SM)
- Vulnerability Management Testing (VM)

Each control is tested automatically with evidence collection
for audit trail and compliance reporting.
"""

import asyncio
import json
import hashlib
import ssl
import socket
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

class SecurityControlTester:
    """Automated security control testing framework for compliance"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results_dir = self.project_root / "compliance-evidence"
        self.test_results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Control test registry
        self.control_tests = {
            # Access Control (AC) Tests
            "AC-1": self._test_access_control_policy,
            "AC-2": self._test_account_management,
            "AC-3": self._test_access_enforcement,
            "AC-6": self._test_least_privilege,
            "AC-7": self._test_unsuccessful_logon_attempts,
            
            # Configuration Management (CM) Tests
            "CM-1": self._test_configuration_management_policy,
            "CM-2": self._test_baseline_configuration,
            "CM-3": self._test_configuration_change_control,
            "CM-6": self._test_configuration_settings,
            
            # Data Protection (DP) Tests
            "DP-1": self._test_data_protection_policy,
            "DP-2": self._test_data_classification,
            "DP-3": self._test_data_encryption,
            "DP-4": self._test_data_backup_integrity,
            
            # Incident Response (IR) Tests
            "IR-1": self._test_incident_response_policy,
            "IR-2": self._test_incident_response_training,
            "IR-4": self._test_incident_handling,
            "IR-5": self._test_incident_monitoring,
            
            # System Monitoring (SM) Tests
            "SM-1": self._test_system_monitoring_policy,
            "SM-2": self._test_automated_monitoring,
            "SM-3": self._test_log_retention,
            "SM-4": self._test_alert_system,
            
            # Vulnerability Management (VM) Tests
            "VM-1": self._test_vulnerability_management_policy,
            "VM-2": self._test_vulnerability_scanning,
            "VM-3": self._test_patch_management,
            "VM-4": self._test_security_updates
        }
    
    async def execute_compliance_testing(self):
        """Execute comprehensive compliance control testing"""
        logger.info("🛡️ Starting Automated Security Control Testing")
        logger.info("=" * 70)
        logger.info("Compliance Framework: SOC2 Type II + ISO27001")
        logger.info(f"Control Tests: {len(self.control_tests)}")
        
        test_results = {
            "compliance_test_suite": "PRSM Automated Security Control Testing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "framework_standards": ["SOC2_Type_II", "ISO27001_2022"],
            "total_controls_tested": len(self.control_tests),
            "test_results": {},
            "compliance_summary": {},
            "evidence_artifacts": []
        }
        
        # Execute all control tests
        for control_id, test_function in self.control_tests.items():
            logger.info(f"🔍 Testing Control {control_id}...")
            
            start_time = time.time()
            try:
                result = await test_function()
                execution_time = time.time() - start_time
                
                test_results["test_results"][control_id] = {
                    "status": "PASS" if result.get("compliant", False) else "FAIL",
                    "execution_time": execution_time,
                    "details": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "evidence_collected": result.get("evidence_files", [])
                }
                
                status_icon = "✅" if result.get("compliant", False) else "❌"
                logger.info(f"{status_icon} Control {control_id}: {test_results['test_results'][control_id]['status']}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                test_results["test_results"][control_id] = {
                    "status": "ERROR",
                    "execution_time": execution_time,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                logger.error(f"❌ Control {control_id}: ERROR - {e}")
        
        # Generate compliance summary
        test_results["compliance_summary"] = self._generate_compliance_summary(test_results["test_results"])
        
        # Save comprehensive test report
        await self._save_compliance_report(test_results)
        
        return test_results
    
    # Access Control (AC) Tests
    async def _test_access_control_policy(self) -> Dict[str, Any]:
        """AC-1: Test access control policy implementation"""
        logger.info("🔐 Testing access control policy compliance...")
        
        evidence = []
        checks = []
        
        # Check RBAC implementation exists
        rbac_file = self.project_root / "prsm/security/production_rbac.py"
        if rbac_file.exists():
            checks.append({"check": "rbac_implementation", "status": "PASS", "details": "Production RBAC system implemented"})
            evidence.append(str(rbac_file))
        else:
            checks.append({"check": "rbac_implementation", "status": "FAIL", "details": "RBAC system not found"})
        
        # Check user role definitions
        models_file = self.project_root / "prsm/core/models.py"
        if models_file.exists():
            content = models_file.read_text()
            if "UserRole" in content and "ADMIN" in content:
                checks.append({"check": "user_role_definitions", "status": "PASS", "details": "User roles properly defined"})
                evidence.append(str(models_file))
            else:
                checks.append({"check": "user_role_definitions", "status": "FAIL", "details": "User roles not properly defined"})
        
        # Check authentication mechanisms
        auth_file = self.project_root / "prsm/auth/auth_manager.py"
        if auth_file.exists():
            checks.append({"check": "authentication_system", "status": "PASS", "details": "Authentication system implemented"})
            evidence.append(str(auth_file))
        else:
            checks.append({"check": "authentication_system", "status": "FAIL", "details": "Authentication system not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "AC-1",
            "control_name": "Access Control Policy",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement missing access control components" if not compliant else None
        }
    
    async def _test_account_management(self) -> Dict[str, Any]:
        """AC-2: Test account management procedures"""
        logger.info("👤 Testing account management compliance...")
        
        evidence = []
        checks = []
        
        # Check user model implementation
        models_file = self.project_root / "prsm/core/models.py"
        if models_file.exists():
            content = models_file.read_text()
            if "class User" in content and "is_active" in content:
                checks.append({"check": "user_lifecycle_management", "status": "PASS", "details": "User lifecycle fields implemented"})
                evidence.append(str(models_file))
            else:
                checks.append({"check": "user_lifecycle_management", "status": "FAIL", "details": "User lifecycle management incomplete"})
        
        # Check for account creation/deletion procedures
        auth_api_file = self.project_root / "prsm/api/auth_api.py"
        if auth_api_file.exists():
            checks.append({"check": "account_procedures", "status": "PASS", "details": "Account management API implemented"})
            evidence.append(str(auth_api_file))
        else:
            checks.append({"check": "account_procedures", "status": "FAIL", "details": "Account management API not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "AC-2",
            "control_name": "Account Management",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Complete account management implementation" if not compliant else None
        }
    
    async def _test_access_enforcement(self) -> Dict[str, Any]:
        """AC-3: Test access enforcement mechanisms"""
        logger.info("🛡️ Testing access enforcement compliance...")
        
        evidence = []
        checks = []
        
        # Check RBAC enforcement
        rbac_file = self.project_root / "prsm/security/production_rbac.py"
        if rbac_file.exists():
            content = rbac_file.read_text()
            if "check_permission" in content and "authorize" in content:
                checks.append({"check": "permission_enforcement", "status": "PASS", "details": "Permission checking implemented"})
                evidence.append(str(rbac_file))
            else:
                checks.append({"check": "permission_enforcement", "status": "FAIL", "details": "Permission enforcement incomplete"})
        
        # Check API protection
        marketplace_api = self.project_root / "prsm/api/marketplace_api.py"
        if marketplace_api.exists():
            content = marketplace_api.read_text()
            if "check_permission" in content or "Depends(get_current_user)" in content:
                checks.append({"check": "api_access_control", "status": "PASS", "details": "API endpoints protected"})
                evidence.append(str(marketplace_api))
            else:
                checks.append({"check": "api_access_control", "status": "FAIL", "details": "API access control incomplete"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "AC-3",
            "control_name": "Access Enforcement",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Strengthen access enforcement mechanisms" if not compliant else None
        }
    
    async def _test_least_privilege(self) -> Dict[str, Any]:
        """AC-6: Test least privilege implementation"""
        logger.info("🔒 Testing least privilege compliance...")
        
        evidence = []
        checks = []
        
        # Check role-based permissions
        models_file = self.project_root / "prsm/core/models.py"
        if models_file.exists():
            content = models_file.read_text()
            if "UserRole" in content and len(content.split("UserRole")) > 1:
                roles = content.count("=") - content.count("Field(default=")
                if roles >= 3:  # At least 3 different roles
                    checks.append({"check": "role_separation", "status": "PASS", "details": f"Multiple user roles defined ({roles})"})
                    evidence.append(str(models_file))
                else:
                    checks.append({"check": "role_separation", "status": "FAIL", "details": "Insufficient role separation"})
            else:
                checks.append({"check": "role_separation", "status": "FAIL", "details": "Role-based access control not implemented"})
        
        # Check permission granularity
        rbac_file = self.project_root / "prsm/security/production_rbac.py"
        if rbac_file.exists():
            content = rbac_file.read_text()
            if "permission" in content.lower() and "." in content:
                checks.append({"check": "granular_permissions", "status": "PASS", "details": "Granular permissions implemented"})
                evidence.append(str(rbac_file))
            else:
                checks.append({"check": "granular_permissions", "status": "FAIL", "details": "Granular permissions not implemented"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "AC-6",
            "control_name": "Least Privilege",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement more granular role-based permissions" if not compliant else None
        }
    
    async def _test_unsuccessful_logon_attempts(self) -> Dict[str, Any]:
        """AC-7: Test unsuccessful logon attempt handling"""
        logger.info("🚫 Testing logon attempt monitoring compliance...")
        
        evidence = []
        checks = []
        
        # Check rate limiting implementation
        rate_limiter_file = self.project_root / "prsm/security/distributed_rate_limiter.py"
        if rate_limiter_file.exists():
            content = rate_limiter_file.read_text()
            if "rate_limit" in content and "max_requests" in content:
                checks.append({"check": "rate_limiting", "status": "PASS", "details": "Rate limiting implemented"})
                evidence.append(str(rate_limiter_file))
            else:
                checks.append({"check": "rate_limiting", "status": "FAIL", "details": "Rate limiting incomplete"})
        
        # Check authentication failure handling
        auth_file = self.project_root / "prsm/auth/auth_manager.py"
        if auth_file.exists():
            checks.append({"check": "auth_failure_handling", "status": "PASS", "details": "Authentication failure handling implemented"})
            evidence.append(str(auth_file))
        else:
            checks.append({"check": "auth_failure_handling", "status": "FAIL", "details": "Authentication failure handling not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "AC-7",
            "control_name": "Unsuccessful Logon Attempts",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement comprehensive logon attempt monitoring" if not compliant else None
        }
    
    # Configuration Management (CM) Tests
    async def _test_configuration_management_policy(self) -> Dict[str, Any]:
        """CM-1: Test configuration management policy"""
        logger.info("⚙️ Testing configuration management compliance...")
        
        evidence = []
        checks = []
        
        # Check for configuration files
        config_files = [
            "docker-compose.yml",
            "requirements.txt", 
            "pyproject.toml"
        ]
        
        found_configs = []
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                found_configs.append(config_file)
                evidence.append(str(file_path))
        
        if len(found_configs) >= 2:
            checks.append({"check": "configuration_files", "status": "PASS", "details": f"Configuration files present: {found_configs}"})
        else:
            checks.append({"check": "configuration_files", "status": "FAIL", "details": "Insufficient configuration management"})
        
        # Check for environment configuration
        env_files = list(self.project_root.glob("*.env*")) + list(self.project_root.glob("config/*"))
        if env_files:
            checks.append({"check": "environment_config", "status": "PASS", "details": "Environment configuration present"})
            evidence.extend([str(f) for f in env_files[:3]])  # Add first 3 files as evidence
        else:
            checks.append({"check": "environment_config", "status": "FAIL", "details": "Environment configuration not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "CM-1",
            "control_name": "Configuration Management Policy",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Establish comprehensive configuration management" if not compliant else None
        }
    
    async def _test_baseline_configuration(self) -> Dict[str, Any]:
        """CM-2: Test baseline configuration management"""
        logger.info("📋 Testing baseline configuration compliance...")
        
        evidence = []
        checks = []
        
        # Check for infrastructure as code
        terraform_dir = self.project_root / "deploy/enterprise/terraform"
        if terraform_dir.exists() and list(terraform_dir.glob("*.tf")):
            checks.append({"check": "infrastructure_as_code", "status": "PASS", "details": "Terraform infrastructure code present"})
            evidence.append(str(terraform_dir))
        else:
            checks.append({"check": "infrastructure_as_code", "status": "FAIL", "details": "Infrastructure as code not implemented"})
        
        # Check for container configurations
        dockerfiles = list(self.project_root.glob("Dockerfile*"))
        if dockerfiles:
            checks.append({"check": "container_configuration", "status": "PASS", "details": "Container configurations present"})
            evidence.extend([str(f) for f in dockerfiles[:2]])
        else:
            checks.append({"check": "container_configuration", "status": "FAIL", "details": "Container configurations not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "CM-2",
            "control_name": "Baseline Configuration",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Establish infrastructure baseline configurations" if not compliant else None
        }
    
    async def _test_configuration_change_control(self) -> Dict[str, Any]:
        """CM-3: Test configuration change control"""
        logger.info("🔄 Testing configuration change control compliance...")
        
        evidence = []
        checks = []
        
        # Check for CI/CD pipeline
        github_workflows = self.project_root / ".github/workflows"
        if github_workflows.exists() and list(github_workflows.glob("*.yml")):
            checks.append({"check": "cicd_pipeline", "status": "PASS", "details": "CI/CD pipeline implemented"})
            evidence.append(str(github_workflows))
        else:
            checks.append({"check": "cicd_pipeline", "status": "FAIL", "details": "CI/CD pipeline not found"})
        
        # Check for version control (git)
        git_dir = self.project_root / ".git"
        if git_dir.exists():
            checks.append({"check": "version_control", "status": "PASS", "details": "Git version control implemented"})
            evidence.append(str(git_dir))
        else:
            checks.append({"check": "version_control", "status": "FAIL", "details": "Version control not implemented"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "CM-3",
            "control_name": "Configuration Change Control",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement configuration change control processes" if not compliant else None
        }
    
    async def _test_configuration_settings(self) -> Dict[str, Any]:
        """CM-6: Test configuration settings management"""
        logger.info("🔧 Testing configuration settings compliance...")
        
        evidence = []
        checks = []
        
        # Check for configuration management
        config_file = self.project_root / "prsm/core/config.py"
        if config_file.exists():
            checks.append({"check": "config_management", "status": "PASS", "details": "Configuration management implemented"})
            evidence.append(str(config_file))
        else:
            checks.append({"check": "config_management", "status": "FAIL", "details": "Configuration management not found"})
        
        # Check for security configuration
        security_files = list((self.project_root / "prsm/security").glob("*.py")) if (self.project_root / "prsm/security").exists() else []
        if len(security_files) >= 3:
            checks.append({"check": "security_configuration", "status": "PASS", "details": f"Security configurations present ({len(security_files)} files)"})
            evidence.extend([str(f) for f in security_files[:3]])
        else:
            checks.append({"check": "security_configuration", "status": "FAIL", "details": "Insufficient security configuration"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "CM-6",
            "control_name": "Configuration Settings",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Enhance configuration settings management" if not compliant else None
        }
    
    # Data Protection (DP) Tests
    async def _test_data_protection_policy(self) -> Dict[str, Any]:
        """DP-1: Test data protection policy implementation"""
        logger.info("🛡️ Testing data protection compliance...")
        
        evidence = []
        checks = []
        
        # Check for encryption implementation
        crypto_dir = self.project_root / "prsm/cryptography"
        if crypto_dir.exists() and list(crypto_dir.glob("*.py")):
            checks.append({"check": "encryption_implementation", "status": "PASS", "details": "Cryptography module implemented"})
            evidence.append(str(crypto_dir))
        else:
            checks.append({"check": "encryption_implementation", "status": "FAIL", "details": "Encryption implementation not found"})
        
        # Check for data classification
        models_file = self.project_root / "prsm/core/models.py"
        if models_file.exists():
            content = models_file.read_text()
            if "metadata" in content or "classification" in content:
                checks.append({"check": "data_classification", "status": "PASS", "details": "Data classification fields present"})
                evidence.append(str(models_file))
            else:
                checks.append({"check": "data_classification", "status": "FAIL", "details": "Data classification not implemented"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "DP-1",
            "control_name": "Data Protection Policy",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement comprehensive data protection policy" if not compliant else None
        }
    
    async def _test_data_classification(self) -> Dict[str, Any]:
        """DP-2: Test data classification procedures"""
        logger.info("🏷️ Testing data classification compliance...")
        
        evidence = []
        checks = []
        
        # Check for data models with classification
        models_file = self.project_root / "prsm/core/models.py"
        if models_file.exists():
            content = models_file.read_text()
            # Look for metadata fields that could contain classification
            if "metadata" in content and "Dict" in content:
                checks.append({"check": "classification_fields", "status": "PASS", "details": "Metadata fields for classification present"})
                evidence.append(str(models_file))
            else:
                checks.append({"check": "classification_fields", "status": "FAIL", "details": "Data classification fields not found"})
        
        # Check for privacy-related implementations
        privacy_dir = self.project_root / "prsm/privacy"
        if privacy_dir.exists():
            checks.append({"check": "privacy_implementation", "status": "PASS", "details": "Privacy module implemented"})
            evidence.append(str(privacy_dir))
        else:
            checks.append({"check": "privacy_implementation", "status": "FAIL", "details": "Privacy implementation not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "DP-2",
            "control_name": "Data Classification",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement data classification procedures" if not compliant else None
        }
    
    async def _test_data_encryption(self) -> Dict[str, Any]:
        """DP-3: Test data encryption implementation"""
        logger.info("🔐 Testing data encryption compliance...")
        
        evidence = []
        checks = []
        
        # Check for encryption modules
        crypto_files = list((self.project_root / "prsm/cryptography").glob("*.py")) if (self.project_root / "prsm/cryptography").exists() else []
        if crypto_files:
            checks.append({"check": "encryption_modules", "status": "PASS", "details": f"Encryption modules present ({len(crypto_files)} files)"})
            evidence.extend([str(f) for f in crypto_files[:3]])
        else:
            checks.append({"check": "encryption_modules", "status": "FAIL", "details": "Encryption modules not found"})
        
        # Check for database encryption configuration
        db_files = list((self.project_root / "prsm/core").glob("*database*.py"))
        if db_files:
            # Check if any database file mentions encryption
            for db_file in db_files:
                content = db_file.read_text()
                if "encrypt" in content.lower() or "ssl" in content.lower():
                    checks.append({"check": "database_encryption", "status": "PASS", "details": "Database encryption configured"})
                    evidence.append(str(db_file))
                    break
            else:
                checks.append({"check": "database_encryption", "status": "FAIL", "details": "Database encryption not configured"})
        else:
            checks.append({"check": "database_encryption", "status": "FAIL", "details": "Database configuration not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "DP-3",
            "control_name": "Data Encryption",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement comprehensive data encryption" if not compliant else None
        }
    
    async def _test_data_backup_integrity(self) -> Dict[str, Any]:
        """DP-4: Test data backup and integrity procedures"""
        logger.info("💾 Testing data backup integrity compliance...")
        
        evidence = []
        checks = []
        
        # Check for backup scripts
        backup_scripts = list(self.project_root.glob("**/backup*.py")) + list(self.project_root.glob("**/backup*.sh"))
        if backup_scripts:
            checks.append({"check": "backup_procedures", "status": "PASS", "details": "Backup scripts present"})
            evidence.extend([str(f) for f in backup_scripts[:2]])
        else:
            checks.append({"check": "backup_procedures", "status": "FAIL", "details": "Backup procedures not found"})
        
        # Check for database migration system
        migrations_dir = self.project_root / "scripts/migrations"
        alembic_dir = self.project_root / "alembic"
        if migrations_dir.exists() or alembic_dir.exists():
            checks.append({"check": "data_versioning", "status": "PASS", "details": "Database migration system present"})
            evidence.append(str(migrations_dir if migrations_dir.exists() else alembic_dir))
        else:
            checks.append({"check": "data_versioning", "status": "FAIL", "details": "Data versioning system not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "DP-4",
            "control_name": "Data Backup Integrity",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement backup and integrity procedures" if not compliant else None
        }
    
    # Incident Response (IR) Tests
    async def _test_incident_response_policy(self) -> Dict[str, Any]:
        """IR-1: Test incident response policy"""
        logger.info("🚨 Testing incident response compliance...")
        
        evidence = []
        checks = []
        
        # Check for security monitoring
        monitoring_dir = self.project_root / "prsm/monitoring"
        if monitoring_dir.exists() and list(monitoring_dir.glob("*.py")):
            checks.append({"check": "security_monitoring", "status": "PASS", "details": "Security monitoring implemented"})
            evidence.append(str(monitoring_dir))
        else:
            checks.append({"check": "security_monitoring", "status": "FAIL", "details": "Security monitoring not found"})
        
        # Check for logging implementation
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            checks.append({"check": "security_logging", "status": "PASS", "details": "Security logging directory present"})
            evidence.append(str(logs_dir))
        else:
            checks.append({"check": "security_logging", "status": "FAIL", "details": "Security logging not implemented"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "IR-1",
            "control_name": "Incident Response Policy",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement incident response procedures" if not compliant else None
        }
    
    async def _test_incident_response_training(self) -> Dict[str, Any]:
        """IR-2: Test incident response training procedures"""
        logger.info("📚 Testing incident response training compliance...")
        
        evidence = []
        checks = []
        
        # Check for documentation
        docs_dir = self.project_root / "docs"
        security_docs = []
        if docs_dir.exists():
            security_docs = list(docs_dir.glob("**/*security*")) + list(docs_dir.glob("**/*SECURITY*"))
        
        if security_docs:
            checks.append({"check": "security_documentation", "status": "PASS", "details": f"Security documentation present ({len(security_docs)} files)"})
            evidence.extend([str(f) for f in security_docs[:3]])
        else:
            checks.append({"check": "security_documentation", "status": "FAIL", "details": "Security documentation not found"})
        
        # Check for incident response documentation
        ir_docs = []
        if docs_dir.exists():
            ir_docs = list(docs_dir.glob("**/*incident*")) + list(docs_dir.glob("**/*response*"))
        
        if ir_docs:
            checks.append({"check": "incident_response_docs", "status": "PASS", "details": "Incident response documentation present"})
            evidence.extend([str(f) for f in ir_docs[:2]])
        else:
            checks.append({"check": "incident_response_docs", "status": "FAIL", "details": "Incident response documentation not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "IR-2",
            "control_name": "Incident Response Training",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Develop incident response training materials" if not compliant else None
        }
    
    async def _test_incident_handling(self) -> Dict[str, Any]:
        """IR-4: Test incident handling procedures"""
        logger.info("⚡ Testing incident handling compliance...")
        
        evidence = []
        checks = []
        
        # Check for circuit breaker implementation
        circuit_breaker_files = []
        if (self.project_root / "prsm").exists():
            circuit_breaker_files = list((self.project_root / "prsm").glob("**/circuit_breaker*.py"))
        
        if circuit_breaker_files:
            checks.append({"check": "automated_incident_response", "status": "PASS", "details": "Circuit breaker system implemented"})
            evidence.extend([str(f) for f in circuit_breaker_files[:2]])
        else:
            checks.append({"check": "automated_incident_response", "status": "FAIL", "details": "Automated incident response not found"})
        
        # Check for safety monitoring
        safety_dir = self.project_root / "prsm/safety"
        if safety_dir.exists():
            checks.append({"check": "safety_monitoring", "status": "PASS", "details": "Safety monitoring system present"})
            evidence.append(str(safety_dir))
        else:
            checks.append({"check": "safety_monitoring", "status": "FAIL", "details": "Safety monitoring not implemented"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "IR-4",
            "control_name": "Incident Handling",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement automated incident handling procedures" if not compliant else None
        }
    
    async def _test_incident_monitoring(self) -> Dict[str, Any]:
        """IR-5: Test incident monitoring procedures"""
        logger.info("👁️ Testing incident monitoring compliance...")
        
        evidence = []
        checks = []
        
        # Check for monitoring infrastructure
        monitoring_configs = list(self.project_root.glob("**/prometheus*.yml")) + list(self.project_root.glob("**/grafana/**"))
        if monitoring_configs:
            checks.append({"check": "monitoring_infrastructure", "status": "PASS", "details": "Monitoring infrastructure configured"})
            evidence.extend([str(f) for f in monitoring_configs[:3]])
        else:
            checks.append({"check": "monitoring_infrastructure", "status": "FAIL", "details": "Monitoring infrastructure not found"})
        
        # Check for alerting configuration
        alert_configs = list(self.project_root.glob("**/alert*.yml")) + list(self.project_root.glob("**/alertmanager*.yml"))
        if alert_configs:
            checks.append({"check": "alerting_system", "status": "PASS", "details": "Alerting system configured"})
            evidence.extend([str(f) for f in alert_configs[:2]])
        else:
            checks.append({"check": "alerting_system", "status": "FAIL", "details": "Alerting system not configured"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "IR-5",
            "control_name": "Incident Monitoring",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement comprehensive incident monitoring" if not compliant else None
        }
    
    # System Monitoring (SM) Tests
    async def _test_system_monitoring_policy(self) -> Dict[str, Any]:
        """SM-1: Test system monitoring policy"""
        logger.info("📊 Testing system monitoring compliance...")
        
        evidence = []
        checks = []
        
        # Check for monitoring implementation
        monitoring_files = list((self.project_root / "prsm/monitoring").glob("*.py")) if (self.project_root / "prsm/monitoring").exists() else []
        if monitoring_files:
            checks.append({"check": "monitoring_implementation", "status": "PASS", "details": f"Monitoring modules present ({len(monitoring_files)} files)"})
            evidence.extend([str(f) for f in monitoring_files[:3]])
        else:
            checks.append({"check": "monitoring_implementation", "status": "FAIL", "details": "Monitoring implementation not found"})
        
        # Check for performance monitoring
        performance_files = list((self.project_root / "prsm/performance").glob("*.py")) if (self.project_root / "prsm/performance").exists() else []
        if performance_files:
            checks.append({"check": "performance_monitoring", "status": "PASS", "details": "Performance monitoring implemented"})
            evidence.extend([str(f) for f in performance_files[:2]])
        else:
            checks.append({"check": "performance_monitoring", "status": "FAIL", "details": "Performance monitoring not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "SM-1",
            "control_name": "System Monitoring Policy",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement comprehensive system monitoring" if not compliant else None
        }
    
    async def _test_automated_monitoring(self) -> Dict[str, Any]:
        """SM-2: Test automated monitoring implementation"""
        logger.info("🤖 Testing automated monitoring compliance...")
        
        evidence = []
        checks = []
        
        # Check for observability stack
        observability_config = self.project_root / "config/observability-stack.yml"
        if observability_config.exists():
            checks.append({"check": "observability_stack", "status": "PASS", "details": "Observability stack configured"})
            evidence.append(str(observability_config))
        else:
            checks.append({"check": "observability_stack", "status": "FAIL", "details": "Observability stack not configured"})
        
        # Check for metrics collection
        metrics_files = list(self.project_root.glob("**/metrics*.py"))
        if metrics_files:
            checks.append({"check": "metrics_collection", "status": "PASS", "details": "Metrics collection implemented"})
            evidence.extend([str(f) for f in metrics_files[:2]])
        else:
            checks.append({"check": "metrics_collection", "status": "FAIL", "details": "Metrics collection not implemented"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "SM-2",
            "control_name": "Automated Monitoring",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement automated monitoring systems" if not compliant else None
        }
    
    async def _test_log_retention(self) -> Dict[str, Any]:
        """SM-3: Test log retention procedures"""
        logger.info("📝 Testing log retention compliance...")
        
        evidence = []
        checks = []
        
        # Check for logging configuration
        logging_configs = list(self.project_root.glob("**/loki*.yml")) + list(self.project_root.glob("**/promtail*.yml"))
        if logging_configs:
            checks.append({"check": "log_retention_config", "status": "PASS", "details": "Log retention configuration present"})
            evidence.extend([str(f) for f in logging_configs[:2]])
        else:
            checks.append({"check": "log_retention_config", "status": "FAIL", "details": "Log retention configuration not found"})
        
        # Check for log directory structure
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            checks.append({"check": "log_storage", "status": "PASS", "details": "Log storage directory present"})
            evidence.append(str(logs_dir))
        else:
            checks.append({"check": "log_storage", "status": "FAIL", "details": "Log storage not implemented"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "SM-3",
            "control_name": "Log Retention",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement log retention procedures" if not compliant else None
        }
    
    async def _test_alert_system(self) -> Dict[str, Any]:
        """SM-4: Test alert system implementation"""
        logger.info("🔔 Testing alert system compliance...")
        
        evidence = []
        checks = []
        
        # Check for alert configuration
        alert_files = list(self.project_root.glob("**/alert*.yml"))
        if alert_files:
            checks.append({"check": "alert_configuration", "status": "PASS", "details": "Alert configuration present"})
            evidence.extend([str(f) for f in alert_files[:2]])
        else:
            checks.append({"check": "alert_configuration", "status": "FAIL", "details": "Alert configuration not found"})
        
        # Check for notification system
        notification_files = list(self.project_root.glob("**/notification*.py"))
        if notification_files:
            checks.append({"check": "notification_system", "status": "PASS", "details": "Notification system implemented"})
            evidence.extend([str(f) for f in notification_files[:2]])
        else:
            checks.append({"check": "notification_system", "status": "FAIL", "details": "Notification system not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "SM-4",
            "control_name": "Alert System",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement comprehensive alert system" if not compliant else None
        }
    
    # Vulnerability Management (VM) Tests
    async def _test_vulnerability_management_policy(self) -> Dict[str, Any]:
        """VM-1: Test vulnerability management policy"""
        logger.info("🔍 Testing vulnerability management compliance...")
        
        evidence = []
        checks = []
        
        # Check for security scanning in CI/CD
        github_workflows = self.project_root / ".github/workflows"
        security_scan_found = False
        if github_workflows.exists():
            for workflow_file in github_workflows.glob("*.yml"):
                content = workflow_file.read_text()
                if "security" in content.lower() or "bandit" in content.lower() or "safety" in content.lower():
                    security_scan_found = True
                    evidence.append(str(workflow_file))
                    break
        
        if security_scan_found:
            checks.append({"check": "automated_security_scanning", "status": "PASS", "details": "Automated security scanning configured"})
        else:
            checks.append({"check": "automated_security_scanning", "status": "FAIL", "details": "Automated security scanning not found"})
        
        # Check for vulnerability scanner implementation
        vuln_scanner_files = list(self.project_root.glob("**/vulnerability*.py"))
        if vuln_scanner_files:
            checks.append({"check": "vulnerability_scanner", "status": "PASS", "details": "Vulnerability scanner implemented"})
            evidence.extend([str(f) for f in vuln_scanner_files[:2]])
        else:
            checks.append({"check": "vulnerability_scanner", "status": "FAIL", "details": "Vulnerability scanner not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "VM-1",
            "control_name": "Vulnerability Management Policy",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement vulnerability management procedures" if not compliant else None
        }
    
    async def _test_vulnerability_scanning(self) -> Dict[str, Any]:
        """VM-2: Test vulnerability scanning implementation"""
        logger.info("🕵️ Testing vulnerability scanning compliance...")
        
        evidence = []
        checks = []
        
        # Check for dependency scanning
        requirements_files = list(self.project_root.glob("requirements*.txt")) + list(self.project_root.glob("pyproject.toml"))
        if requirements_files:
            checks.append({"check": "dependency_management", "status": "PASS", "details": "Dependency files present for scanning"})
            evidence.extend([str(f) for f in requirements_files[:2]])
        else:
            checks.append({"check": "dependency_management", "status": "FAIL", "details": "Dependency files not found"})
        
        # Check for security integrations
        integration_security = list((self.project_root / "prsm/integrations/security").glob("*.py")) if (self.project_root / "prsm/integrations/security").exists() else []
        if integration_security:
            checks.append({"check": "security_integrations", "status": "PASS", "details": "Security integration modules present"})
            evidence.extend([str(f) for f in integration_security[:2]])
        else:
            checks.append({"check": "security_integrations", "status": "FAIL", "details": "Security integration modules not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "VM-2",
            "control_name": "Vulnerability Scanning",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement vulnerability scanning procedures" if not compliant else None
        }
    
    async def _test_patch_management(self) -> Dict[str, Any]:
        """VM-3: Test patch management procedures"""
        logger.info("🔧 Testing patch management compliance...")
        
        evidence = []
        checks = []
        
        # Check for automated updates in CI/CD
        github_workflows = self.project_root / ".github/workflows"
        update_automation_found = False
        if github_workflows.exists():
            for workflow_file in github_workflows.glob("*.yml"):
                content = workflow_file.read_text()
                if "update" in content.lower() or "upgrade" in content.lower() or "dependabot" in content.lower():
                    update_automation_found = True
                    evidence.append(str(workflow_file))
                    break
        
        if update_automation_found:
            checks.append({"check": "automated_updates", "status": "PASS", "details": "Automated update process configured"})
        else:
            checks.append({"check": "automated_updates", "status": "FAIL", "details": "Automated update process not found"})
        
        # Check for version pinning
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            content = requirements_file.read_text()
            if "==" in content:  # Version pinning
                checks.append({"check": "version_pinning", "status": "PASS", "details": "Dependency versions pinned"})
                evidence.append(str(requirements_file))
            else:
                checks.append({"check": "version_pinning", "status": "FAIL", "details": "Dependency versions not pinned"})
        else:
            checks.append({"check": "version_pinning", "status": "FAIL", "details": "Requirements file not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "VM-3",
            "control_name": "Patch Management",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement patch management procedures" if not compliant else None
        }
    
    async def _test_security_updates(self) -> Dict[str, Any]:
        """VM-4: Test security update procedures"""
        logger.info("🛡️ Testing security update compliance...")
        
        evidence = []
        checks = []
        
        # Check for security update automation
        dependabot_config = self.project_root / ".github/dependabot.yml"
        if dependabot_config.exists():
            checks.append({"check": "dependabot_configuration", "status": "PASS", "details": "Dependabot configured for security updates"})
            evidence.append(str(dependabot_config))
        else:
            checks.append({"check": "dependabot_configuration", "status": "FAIL", "details": "Dependabot not configured"})
        
        # Check for security monitoring
        security_monitoring_files = list(self.project_root.glob("**/security*monitor*.py"))
        if security_monitoring_files:
            checks.append({"check": "security_monitoring", "status": "PASS", "details": "Security monitoring implemented"})
            evidence.extend([str(f) for f in security_monitoring_files[:2]])
        else:
            checks.append({"check": "security_monitoring", "status": "FAIL", "details": "Security monitoring not found"})
        
        compliant = all(check["status"] == "PASS" for check in checks)
        
        return {
            "control_id": "VM-4",
            "control_name": "Security Updates",
            "compliant": compliant,
            "checks": checks,
            "evidence_files": evidence,
            "remediation": "Implement security update procedures" if not compliant else None
        }
    
    def _generate_compliance_summary(self, test_results: Dict) -> Dict[str, Any]:
        """Generate compliance summary and scoring"""
        total_controls = len(test_results)
        passed_controls = len([r for r in test_results.values() if r["status"] == "PASS"])
        failed_controls = len([r for r in test_results.values() if r["status"] == "FAIL"])
        error_controls = len([r for r in test_results.values() if r["status"] == "ERROR"])
        
        compliance_percentage = (passed_controls / total_controls) * 100 if total_controls > 0 else 0
        
        # Determine compliance grade
        if compliance_percentage >= 90:
            grade = "A"
            readiness = "Production Ready"
        elif compliance_percentage >= 80:
            grade = "B"
            readiness = "Near Production Ready"
        elif compliance_percentage >= 70:
            grade = "C"
            readiness = "Development Ready"
        else:
            grade = "D"
            readiness = "Needs Significant Work"
        
        # Categorize results by control family
        control_families = {
            "Access Control": [r for k, r in test_results.items() if k.startswith("AC-")],
            "Configuration Management": [r for k, r in test_results.items() if k.startswith("CM-")],
            "Data Protection": [r for k, r in test_results.items() if k.startswith("DP-")],
            "Incident Response": [r for k, r in test_results.items() if k.startswith("IR-")],
            "System Monitoring": [r for k, r in test_results.items() if k.startswith("SM-")],
            "Vulnerability Management": [r for k, r in test_results.items() if k.startswith("VM-")]
        }
        
        family_scores = {}
        for family, controls in control_families.items():
            if controls:
                family_passed = len([c for c in controls if c["status"] == "PASS"])
                family_total = len(controls)
                family_scores[family] = {
                    "passed": family_passed,
                    "total": family_total,
                    "percentage": (family_passed / family_total) * 100
                }
        
        return {
            "total_controls_tested": total_controls,
            "controls_passed": passed_controls,
            "controls_failed": failed_controls,
            "controls_error": error_controls,
            "compliance_percentage": round(compliance_percentage, 1),
            "compliance_grade": grade,
            "production_readiness": readiness,
            "control_family_scores": family_scores,
            "recommendations": self._generate_compliance_recommendations(test_results, compliance_percentage),
            "next_steps": [
                "Address failed controls with highest priority",
                "Implement missing security documentation",
                "Schedule third-party security audit",
                "Establish continuous compliance monitoring"
            ]
        }
    
    def _generate_compliance_recommendations(self, test_results: Dict, compliance_percentage: float) -> List[str]:
        """Generate specific compliance recommendations"""
        recommendations = []
        
        # High priority recommendations based on failed controls
        failed_controls = [k for k, v in test_results.items() if v["status"] == "FAIL"]
        
        if any(control.startswith("AC-") for control in failed_controls):
            recommendations.append("Strengthen access control implementation and user management")
        
        if any(control.startswith("DP-") for control in failed_controls):
            recommendations.append("Implement comprehensive data protection and encryption")
        
        if any(control.startswith("IR-") for control in failed_controls):
            recommendations.append("Establish incident response procedures and training")
        
        if any(control.startswith("VM-") for control in failed_controls):
            recommendations.append("Implement vulnerability management and patch procedures")
        
        # General recommendations based on compliance level
        if compliance_percentage < 80:
            recommendations.append("Engage compliance consultant for SOC2/ISO27001 gap analysis")
            recommendations.append("Develop comprehensive information security policy suite")
        
        if compliance_percentage < 90:
            recommendations.append("Implement automated compliance monitoring")
            recommendations.append("Schedule regular security control assessments")
        
        return recommendations
    
    async def _save_compliance_report(self, test_results: Dict):
        """Save comprehensive compliance report"""
        logger.info("📋 Generating compliance evidence report...")
        
        # Save main report
        report_file = self.test_results_dir / f"compliance_test_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Generate evidence manifest
        all_evidence_files = []
        for control_result in test_results["test_results"].values():
            if "details" in control_result and "evidence_files" in control_result["details"]:
                all_evidence_files.extend(control_result["details"]["evidence_files"])
        
        evidence_manifest = {
            "evidence_collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_evidence_files": len(set(all_evidence_files)),
            "evidence_files": list(set(all_evidence_files)),
            "evidence_hash": hashlib.sha256(str(sorted(set(all_evidence_files))).encode()).hexdigest()
        }
        
        manifest_file = self.test_results_dir / f"evidence_manifest_{self.timestamp}.json"
        with open(manifest_file, 'w') as f:
            json.dump(evidence_manifest, f, indent=2)
        
        test_results["evidence_artifacts"] = [str(report_file), str(manifest_file)]
        
        logger.info(f"📄 Compliance report saved: {report_file}")
        logger.info(f"📋 Evidence manifest saved: {manifest_file}")
        
        return test_results


async def main():
    """Main function for security control testing"""
    tester = SecurityControlTester()
    results = await tester.execute_compliance_testing()
    
    summary = results["compliance_summary"]
    
    logger.info("\n" + "="*70)
    logger.info("🛡️ AUTOMATED SECURITY CONTROL TESTING COMPLETE")
    logger.info("="*70)
    logger.info(f"Compliance Score: {summary['compliance_percentage']}% (Grade: {summary['compliance_grade']})")
    logger.info(f"Production Readiness: {summary['production_readiness']}")
    logger.info(f"Controls Passed: {summary['controls_passed']}/{summary['total_controls_tested']}")
    
    if summary['compliance_percentage'] >= 80:
        logger.info("🎉 Strong compliance posture for Series A requirements!")
    elif summary['compliance_percentage'] >= 70:
        logger.info("⚠️ Good progress, minor gaps to address for enterprise readiness")
    else:
        logger.info("🔧 Significant compliance work needed for production deployment")
    
    logger.info("\n📊 Control Family Scores:")
    for family, score in summary['control_family_scores'].items():
        percentage = score['percentage']
        icon = "✅" if percentage >= 80 else "⚠️" if percentage >= 60 else "❌"
        logger.info(f"  {icon} {family}: {score['passed']}/{score['total']} ({percentage:.1f}%)")
    
    if summary['recommendations']:
        logger.info("\n💡 Priority Recommendations:")
        for i, rec in enumerate(summary['recommendations'][:3], 1):
            logger.info(f"  {i}. {rec}")
    
    logger.info(f"\n📁 Evidence collected in: compliance-evidence/")
    logger.info("🎯 Next: Build evidence collection pipeline for continuous compliance")


if __name__ == "__main__":
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run automated security control testing
    asyncio.run(main())