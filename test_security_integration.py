#!/usr/bin/env python3
"""
Comprehensive Security Integration Test
Tests all implemented security features in PRSM
"""

import asyncio
import time
import json
from typing import Dict, Any
from pathlib import Path
import structlog

# Configure logging for tests
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class SecurityTestSuite:
    """Comprehensive security test suite for PRSM"""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete security test suite"""
        
        logger.info("🔒 Starting PRSM Security Integration Test Suite")
        
        # Test categories
        test_categories = [
            ("Authentication System", self.test_authentication),
            ("Authorization & RBAC", self.test_authorization),
            ("Rate Limiting", self.test_rate_limiting),
            ("Security Headers", self.test_security_headers),
            ("Input Validation", self.test_input_validation),
            ("JWT Token Security", self.test_jwt_security),
            ("Security Middleware", self.test_middleware),
            ("Audit Logging", self.test_audit_logging),
            ("Threat Detection", self.test_threat_detection),
            ("Vulnerability Scanner", self.test_vulnerability_scanner),
            ("Security Configuration", self.test_security_config)
        ]
        
        # Run tests
        for category_name, test_function in test_categories:
            logger.info(f"Testing {category_name}...")
            
            try:
                result = await test_function()
                self.test_results.append({
                    "category": category_name,
                    "status": "PASSED" if result["passed"] else "FAILED",
                    "details": result
                })
                
                if not result["passed"]:
                    self.failed_tests.append(category_name)
                    
            except Exception as e:
                logger.error(f"Test error in {category_name}", error=str(e))
                self.test_results.append({
                    "category": category_name,
                    "status": "ERROR",
                    "error": str(e)
                })
                self.failed_tests.append(category_name)
        
        # Generate report
        return self.generate_test_report()
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test authentication system components"""
        tests = []
        
        try:
            # Test 1: Auth models exist and are properly defined
            from prsm.auth.models import User, UserRole, Permission, LoginRequest, RegisterRequest
            tests.append({"test": "Import auth models", "status": "PASS"})
            
            # Test 2: JWT handler functionality
            from prsm.auth.jwt_handler import jwt_handler, TokenData
            tests.append({"test": "Import JWT handler", "status": "PASS"})
            
            # Test 3: Auth manager functionality
            from prsm.auth.auth_manager import auth_manager, get_current_user
            tests.append({"test": "Import auth manager", "status": "PASS"})
            
            # Test 4: Password hashing
            test_password = "TestPassword123!"
            hashed = jwt_handler.hash_password(test_password)
            verified = jwt_handler.verify_password(test_password, hashed)
            
            if verified:
                tests.append({"test": "Password hashing/verification", "status": "PASS"})
            else:
                tests.append({"test": "Password hashing/verification", "status": "FAIL", "error": "Verification failed"})
            
            # Test 5: User role permissions
            user = User(
                email="test@prsm.ai",
                username="testuser",
                hashed_password=hashed,
                role=UserRole.ADMIN
            )
            
            has_permission = user.has_permission(Permission.SYSTEM_ADMIN)
            if has_permission:
                tests.append({"test": "Role-based permissions", "status": "PASS"})
            else:
                tests.append({"test": "Role-based permissions", "status": "FAIL", "error": "Admin should have system admin permission"})
            
        except ImportError as e:
            tests.append({"test": "Authentication imports", "status": "FAIL", "error": str(e)})
        except Exception as e:
            tests.append({"test": "Authentication functionality", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_authorization(self) -> Dict[str, Any]:
        """Test authorization and RBAC system"""
        tests = []
        
        try:
            from prsm.auth.models import UserRole, Permission, ROLE_PERMISSIONS
            
            # Test 1: Role permission mappings exist
            if ROLE_PERMISSIONS:
                tests.append({"test": "Role permission mappings", "status": "PASS"})
            else:
                tests.append({"test": "Role permission mappings", "status": "FAIL", "error": "No role permissions defined"})
            
            # Test 2: Admin has all permissions
            admin_permissions = ROLE_PERMISSIONS.get(UserRole.ADMIN, [])
            all_permissions = [p for p in Permission]
            
            if set(admin_permissions) == set(all_permissions):
                tests.append({"test": "Admin has all permissions", "status": "PASS"})
            else:
                missing = set(all_permissions) - set(admin_permissions)
                tests.append({"test": "Admin has all permissions", "status": "FAIL", "error": f"Missing: {missing}"})
            
            # Test 3: Guest has minimal permissions
            guest_permissions = ROLE_PERMISSIONS.get(UserRole.GUEST, [])
            
            if len(guest_permissions) < len(all_permissions) / 2:  # Less than half
                tests.append({"test": "Guest has minimal permissions", "status": "PASS"})
            else:
                tests.append({"test": "Guest has minimal permissions", "status": "FAIL", "error": f"Guest has too many permissions: {len(guest_permissions)}"})
            
            # Test 4: Permission hierarchy makes sense
            user_permissions = ROLE_PERMISSIONS.get(UserRole.USER, [])
            researcher_permissions = ROLE_PERMISSIONS.get(UserRole.RESEARCHER, [])
            
            if len(researcher_permissions) > len(user_permissions):
                tests.append({"test": "Permission hierarchy", "status": "PASS"})
            else:
                tests.append({"test": "Permission hierarchy", "status": "FAIL", "error": "Researcher should have more permissions than user"})
            
        except Exception as e:
            tests.append({"test": "Authorization system", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting system"""
        tests = []
        
        try:
            from prsm.auth.rate_limiter import rate_limiter, RateLimitRule, RateLimitType
            
            # Test 1: Rate limiter import
            tests.append({"test": "Rate limiter import", "status": "PASS"})
            
            # Test 2: Rate limit rules exist
            if rate_limiter.rules:
                tests.append({"test": "Rate limit rules defined", "status": "PASS"})
                
                # Test 3: Different rule types exist
                rule_types = set(rule.limit_type for rule in rate_limiter.rules)
                expected_types = {RateLimitType.GLOBAL, RateLimitType.PER_IP, RateLimitType.PER_USER}
                
                if expected_types.issubset(rule_types):
                    tests.append({"test": "Rule type variety", "status": "PASS"})
                else:
                    missing = expected_types - rule_types
                    tests.append({"test": "Rule type variety", "status": "FAIL", "error": f"Missing rule types: {missing}"})
                
                # Test 4: Auth endpoints have specific limits
                auth_rules = [rule for rule in rate_limiter.rules if "/auth" in rule.endpoint_pattern]
                if auth_rules:
                    tests.append({"test": "Auth endpoint rate limits", "status": "PASS"})
                else:
                    tests.append({"test": "Auth endpoint rate limits", "status": "FAIL", "error": "No specific auth rate limits"})
            else:
                tests.append({"test": "Rate limit rules defined", "status": "FAIL", "error": "No rate limit rules"})
            
        except ImportError as e:
            tests.append({"test": "Rate limiter import", "status": "FAIL", "error": str(e)})
        except Exception as e:
            tests.append({"test": "Rate limiting functionality", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_security_headers(self) -> Dict[str, Any]:
        """Test security headers middleware"""
        tests = []
        
        try:
            from prsm.auth.middleware import SecurityHeadersMiddleware, AuthMiddleware
            
            # Test 1: Middleware import
            tests.append({"test": "Security middleware import", "status": "PASS"})
            
            # Test 2: Security headers defined
            middleware = SecurityHeadersMiddleware(None)
            required_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection",
                "Strict-Transport-Security",
                "Content-Security-Policy"
            ]
            
            missing_headers = []
            for header in required_headers:
                if header not in middleware.security_headers:
                    missing_headers.append(header)
            
            if not missing_headers:
                tests.append({"test": "Required security headers", "status": "PASS"})
            else:
                tests.append({"test": "Required security headers", "status": "FAIL", "error": f"Missing: {missing_headers}"})
            
            # Test 3: Auth middleware has rate limiting
            auth_middleware = AuthMiddleware(None)
            if hasattr(auth_middleware, 'rate_limit_requests'):
                tests.append({"test": "Auth middleware rate limiting", "status": "PASS"})
            else:
                tests.append({"test": "Auth middleware rate limiting", "status": "FAIL", "error": "No rate limiting in auth middleware"})
            
        except ImportError as e:
            tests.append({"test": "Security middleware import", "status": "FAIL", "error": str(e)})
        except Exception as e:
            tests.append({"test": "Security headers functionality", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation using Pydantic models"""
        tests = []
        
        try:
            from pysm.auth.models import LoginRequest, RegisterRequest, PasswordChange
            from pydantic import ValidationError
            
            # Test 1: Valid login request
            try:
                login_req = LoginRequest(username="testuser", password="password123")
                tests.append({"test": "Valid login request", "status": "PASS"})
            except ValidationError:
                tests.append({"test": "Valid login request", "status": "FAIL", "error": "Valid data rejected"})
            
            # Test 2: Invalid login request (missing fields)
            try:
                login_req = LoginRequest(username="testuser")  # Missing password
                tests.append({"test": "Invalid login validation", "status": "FAIL", "error": "Invalid data accepted"})
            except ValidationError:
                tests.append({"test": "Invalid login validation", "status": "PASS"})
            
            # Test 3: Username validation
            try:
                register_req = RegisterRequest(
                    email="test@example.com",
                    username="a",  # Too short
                    password="Password123!",
                    confirm_password="Password123!"
                )
                tests.append({"test": "Username length validation", "status": "FAIL", "error": "Short username accepted"})
            except ValidationError:
                tests.append({"test": "Username length validation", "status": "PASS"})
            
            # Test 4: Email validation
            try:
                register_req = RegisterRequest(
                    email="invalid-email",  # Invalid email
                    username="testuser",
                    password="Password123!",
                    confirm_password="Password123!"
                )
                tests.append({"test": "Email validation", "status": "FAIL", "error": "Invalid email accepted"})
            except ValidationError:
                tests.append({"test": "Email validation", "status": "PASS"})
                
        except ImportError as e:
            tests.append({"test": "Validation models import", "status": "FAIL", "error": str(e)})
        except Exception as e:
            tests.append({"test": "Input validation", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_jwt_security(self) -> Dict[str, Any]:
        """Test JWT token security features"""
        tests = []
        
        try:
            from prsm.auth.jwt_handler import jwt_handler
            
            # Test 1: Token generation
            user_data = {
                "user_id": "12345678-1234-5678-9012-123456789012",
                "username": "testuser",
                "email": "test@example.com",
                "role": "user",
                "permissions": ["model:read"]
            }
            
            token, token_data = await jwt_handler.create_access_token(user_data)
            
            if token and token_data:
                tests.append({"test": "JWT token generation", "status": "PASS"})
            else:
                tests.append({"test": "JWT token generation", "status": "FAIL", "error": "Token creation failed"})
            
            # Test 2: Token verification
            if token:
                verified_data = await jwt_handler.verify_token(token)
                
                if verified_data and verified_data.username == "testuser":
                    tests.append({"test": "JWT token verification", "status": "PASS"})
                else:
                    tests.append({"test": "JWT token verification", "status": "FAIL", "error": "Token verification failed"})
            
            # Test 3: Invalid token rejection
            invalid_token = "invalid.jwt.token"
            verified_invalid = await jwt_handler.verify_token(invalid_token)
            
            if verified_invalid is None:
                tests.append({"test": "Invalid token rejection", "status": "PASS"})
            else:
                tests.append({"test": "Invalid token rejection", "status": "FAIL", "error": "Invalid token accepted"})
            
            # Test 4: Token hash generation
            if token:
                token_hash = jwt_handler.generate_token_hash(token)
                
                if token_hash and len(token_hash) == 64:  # SHA256 hex length
                    tests.append({"test": "Token hash generation", "status": "PASS"})
                else:
                    tests.append({"test": "Token hash generation", "status": "FAIL", "error": "Invalid token hash"})
            
        except Exception as e:
            tests.append({"test": "JWT security", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_middleware(self) -> Dict[str, Any]:
        """Test security middleware integration"""
        tests = []
        
        try:
            # Test 1: Middleware files exist
            middleware_path = Path("prsm/auth/middleware.py")
            if middleware_path.exists():
                tests.append({"test": "Middleware file exists", "status": "PASS"})
            else:
                tests.append({"test": "Middleware file exists", "status": "FAIL", "error": "Middleware file not found"})
            
            # Test 2: Main API integration
            main_api_path = Path("prsm/api/main.py")
            if main_api_path.exists():
                with open(main_api_path, 'r') as f:
                    content = f.read()
                    
                if "AuthMiddleware" in content or "SecurityHeadersMiddleware" in content:
                    tests.append({"test": "Middleware integration", "status": "PASS"})
                else:
                    tests.append({"test": "Middleware integration", "status": "FAIL", "error": "Middleware not integrated in main API"})
            
            # Test 3: Auth router integration
            if main_api_path.exists():
                with open(main_api_path, 'r') as f:
                    content = f.read()
                    
                if "auth_router" in content:
                    tests.append({"test": "Auth router integration", "status": "PASS"})
                else:
                    tests.append({"test": "Auth router integration", "status": "FAIL", "error": "Auth router not integrated"})
            
        except Exception as e:
            tests.append({"test": "Middleware integration", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_audit_logging(self) -> Dict[str, Any]:
        """Test audit logging system"""
        tests = []
        
        try:
            from prsm.integrations.security.audit_logger import audit_logger
            
            # Test 1: Audit logger import
            tests.append({"test": "Audit logger import", "status": "PASS"})
            
            # Test 2: Log auth event method exists
            if hasattr(audit_logger, 'log_auth_event'):
                tests.append({"test": "Auth event logging method", "status": "PASS"})
            else:
                tests.append({"test": "Auth event logging method", "status": "FAIL", "error": "Method not found"})
            
            # Test 3: Log security event method exists  
            if hasattr(audit_logger, 'log_security_event'):
                tests.append({"test": "Security event logging method", "status": "PASS"})
            else:
                tests.append({"test": "Security event logging method", "status": "FAIL", "error": "Method not found"})
            
        except ImportError as e:
            tests.append({"test": "Audit logger import", "status": "FAIL", "error": str(e)})
        except Exception as e:
            tests.append({"test": "Audit logging", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_threat_detection(self) -> Dict[str, Any]:
        """Test threat detection system"""
        tests = []
        
        try:
            from prsm.integrations.security.threat_detector import ThreatDetector
            
            # Test 1: Threat detector import
            tests.append({"test": "Threat detector import", "status": "PASS"})
            
            # Test 2: Threat detection methods exist
            detector = ThreatDetector()
            
            required_methods = ['scan_content', 'analyze_request_patterns', 'detect_sql_injection']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(detector, method):
                    missing_methods.append(method)
            
            if not missing_methods:
                tests.append({"test": "Threat detection methods", "status": "PASS"})
            else:
                tests.append({"test": "Threat detection methods", "status": "FAIL", "error": f"Missing: {missing_methods}"})
            
            # Test 3: SQL injection detection
            if hasattr(detector, 'detect_sql_injection'):
                malicious_input = "'; DROP TABLE users; --"
                result = detector.detect_sql_injection(malicious_input)
                
                if result.is_threat:
                    tests.append({"test": "SQL injection detection", "status": "PASS"})
                else:
                    tests.append({"test": "SQL injection detection", "status": "FAIL", "error": "SQL injection not detected"})
            
        except ImportError as e:
            tests.append({"test": "Threat detector import", "status": "FAIL", "error": str(e)})
        except Exception as e:
            tests.append({"test": "Threat detection", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_vulnerability_scanner(self) -> Dict[str, Any]:
        """Test vulnerability scanner"""
        tests = []
        
        try:
            from prsm.integrations.security.vulnerability_scanner import VulnerabilityScanner
            
            # Test 1: Scanner import
            tests.append({"test": "Vulnerability scanner import", "status": "PASS"})
            
            # Test 2: Scanner methods exist
            scanner = VulnerabilityScanner()
            
            required_methods = ['scan_code', 'check_dependencies', 'scan_for_secrets']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(scanner, method):
                    missing_methods.append(method)
            
            if not missing_methods:
                tests.append({"test": "Scanner methods", "status": "PASS"})
            else:
                tests.append({"test": "Scanner methods", "status": "FAIL", "error": f"Missing: {missing_methods}"})
            
            # Test 3: XSS detection
            if hasattr(scanner, 'scan_code'):
                xss_code = "<script>alert('xss')</script>"
                result = scanner.scan_code(xss_code, "test.html")
                
                # Should detect XSS vulnerability
                has_xss_vuln = any("xss" in vuln.vulnerability_type.lower() for vuln in result.vulnerabilities)
                
                if has_xss_vuln:
                    tests.append({"test": "XSS detection", "status": "PASS"})
                else:
                    tests.append({"test": "XSS detection", "status": "FAIL", "error": "XSS not detected"})
            
        except ImportError as e:
            tests.append({"test": "Vulnerability scanner import", "status": "FAIL", "error": str(e)})
        except Exception as e:
            tests.append({"test": "Vulnerability scanner", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    async def test_security_config(self) -> Dict[str, Any]:
        """Test security configuration"""
        tests = []
        
        try:
            from prsm.core.config import get_settings
            
            settings = get_settings()
            
            # Test 1: Security configuration exists
            if hasattr(settings, 'secret_key'):
                tests.append({"test": "Secret key configured", "status": "PASS"})
            else:
                tests.append({"test": "Secret key configured", "status": "FAIL", "error": "No secret key"})
            
            # Test 2: JWT configuration
            if hasattr(settings, 'jwt_algorithm'):
                tests.append({"test": "JWT configuration", "status": "PASS"})
            else:
                tests.append({"test": "JWT configuration", "status": "FAIL", "error": "No JWT config"})
            
            # Test 3: Database URL validation
            if hasattr(settings, 'validate_database_url'):
                tests.append({"test": "Database URL validation", "status": "PASS"})
            else:
                tests.append({"test": "Database URL validation", "status": "FAIL", "error": "No DB validation"})
            
        except Exception as e:
            tests.append({"test": "Security configuration", "status": "FAIL", "error": str(e)})
        
        passed = all(test["status"] == "PASS" for test in tests)
        
        return {
            "passed": passed,
            "total_tests": len(tests),
            "failed_tests": len([t for t in tests if t["status"] != "PASS"]),
            "details": tests
        }
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results if r["status"] == "ERROR"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall security status
        if success_rate >= 90:
            security_status = "EXCELLENT"
            status_emoji = "🛡️"
        elif success_rate >= 80:
            security_status = "GOOD"
            status_emoji = "✅"
        elif success_rate >= 70:
            security_status = "ACCEPTABLE"
            status_emoji = "⚠️"
        else:
            security_status = "NEEDS_IMPROVEMENT"
            status_emoji = "❌"
        
        report = {
            "overall_status": security_status,
            "status_emoji": status_emoji,
            "success_rate": round(success_rate, 1),
            "summary": {
                "total_categories": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "failed_categories": self.failed_tests
            },
            "test_results": self.test_results,
            "recommendations": self.generate_recommendations(),
            "timestamp": time.time()
        }
        
        return report
    
    def generate_recommendations(self) -> list[str]:
        """Generate security recommendations based on test results"""
        recommendations = []
        
        if "Authentication System" in self.failed_tests:
            recommendations.append("🔐 Implement missing authentication components")
        
        if "Authorization & RBAC" in self.failed_tests:
            recommendations.append("👥 Review and fix role-based access control")
        
        if "Rate Limiting" in self.failed_tests:
            recommendations.append("🚦 Implement comprehensive rate limiting")
        
        if "Security Headers" in self.failed_tests:
            recommendations.append("🛡️ Add missing security headers to responses")
        
        if "Input Validation" in self.failed_tests:
            recommendations.append("✅ Strengthen input validation and sanitization")
        
        if "JWT Token Security" in self.failed_tests:
            recommendations.append("🎫 Fix JWT token generation and validation")
        
        if "Threat Detection" in self.failed_tests:
            recommendations.append("🔍 Enhance threat detection capabilities")
        
        if "Vulnerability Scanner" in self.failed_tests:
            recommendations.append("🔎 Implement vulnerability scanning")
        
        if len(self.failed_tests) == 0:
            recommendations.append("🎉 Excellent security implementation! Continue monitoring and updating.")
        
        return recommendations


async def main():
    """Run security test suite"""
    
    suite = SecurityTestSuite()
    report = await suite.run_all_tests()
    
    # Print report
    print("\n" + "="*60)
    print(f"{report['status_emoji']} PRSM SECURITY TEST REPORT")
    print("="*60)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Success Rate: {report['success_rate']}%")
    print(f"Categories Tested: {report['summary']['total_categories']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Errors: {report['summary']['errors']}")
    
    if report['summary']['failed_categories']:
        print(f"\n❌ Failed Categories:")
        for category in report['summary']['failed_categories']:
            print(f"   - {category}")
    
    print(f"\n📋 Recommendations:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    print("\n" + "="*60)
    
    # Save detailed report
    report_file = f"security_test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_file}")
    
    return report['overall_status'] in ["EXCELLENT", "GOOD"]


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)