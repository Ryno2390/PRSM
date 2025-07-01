#!/usr/bin/env python3
"""
Security Report Analysis for CI/CD Pipeline

Analyzes security scan results and fails the build if critical issues are found.
Addresses Gemini's requirement for automated security validation.
"""

import json
import sys
import os
from pathlib import Path

def check_bandit_report():
    """Check Bandit security scan results for critical issues."""
    report_path = "bandit-report.json"
    if not os.path.exists(report_path):
        print("⚠️ Bandit report not found, skipping...")
        return True
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        high_severity_issues = [
            issue for issue in report.get('results', [])
            if issue.get('issue_severity') == 'HIGH'
        ]
        
        if high_severity_issues:
            print(f"❌ Found {len(high_severity_issues)} HIGH severity security issues:")
            for issue in high_severity_issues[:5]:  # Show first 5
                print(f"  - {issue.get('test_id')}: {issue.get('issue_text')}")
                print(f"    File: {issue.get('filename')}:{issue.get('line_number')}")
            return False
        
        medium_issues = [
            issue for issue in report.get('results', [])
            if issue.get('issue_severity') == 'MEDIUM'
        ]
        
        if len(medium_issues) > 10:
            print(f"⚠️ Found {len(medium_issues)} MEDIUM severity issues (threshold: 10)")
            print("Consider addressing these before production deployment")
        
        print(f"✅ Bandit scan passed: {len(high_severity_issues)} high, {len(medium_issues)} medium issues")
        return True
        
    except Exception as e:
        print(f"❌ Error parsing Bandit report: {e}")
        return False

def check_safety_report():
    """Check Safety dependency vulnerability scan results."""
    report_path = "safety-report.json"
    if not os.path.exists(report_path):
        print("⚠️ Safety report not found, skipping...")
        return True
    
    try:
        with open(report_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print("✅ Safety scan passed: No vulnerabilities found")
            return True
            
        report = json.loads(content)
        vulnerabilities = report.get('vulnerabilities', [])
        
        critical_vulns = [
            vuln for vuln in vulnerabilities
            if vuln.get('vulnerability_id', '').startswith('CVE-') and
            'critical' in vuln.get('advisory', '').lower()
        ]
        
        if critical_vulns:
            print(f"❌ Found {len(critical_vulns)} critical vulnerabilities:")
            for vuln in critical_vulns[:3]:  # Show first 3
                print(f"  - {vuln.get('vulnerability_id')}: {vuln.get('package_name')}")
                print(f"    {vuln.get('advisory', '')[:100]}...")
            return False
        
        if len(vulnerabilities) > 0:
            print(f"⚠️ Found {len(vulnerabilities)} lower-severity vulnerabilities")
            print("Consider updating dependencies")
        
        print(f"✅ Safety scan passed: {len(vulnerabilities)} total vulnerabilities, 0 critical")
        return True
        
    except Exception as e:
        print(f"❌ Error parsing Safety report: {e}")
        return False

def check_semgrep_report():
    """Check Semgrep static analysis results."""
    report_path = "semgrep-report.json"
    if not os.path.exists(report_path):
        print("⚠️ Semgrep report not found, skipping...")
        return True
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        results = report.get('results', [])
        errors = [r for r in results if r.get('severity') == 'ERROR']
        warnings = [r for r in results if r.get('severity') == 'WARNING']
        
        # Block deployment on security-related errors
        security_errors = [
            r for r in errors 
            if any(keyword in r.get('check_id', '').lower() 
                  for keyword in ['security', 'crypto', 'sql-injection', 'xss'])
        ]
        
        if security_errors:
            print(f"❌ Found {len(security_errors)} security-related errors:")
            for error in security_errors[:3]:
                print(f"  - {error.get('check_id')}: {error.get('message')}")
                print(f"    File: {error.get('path')}:{error.get('start', {}).get('line')}")
            return False
        
        if len(errors) > 5:
            print(f"⚠️ Found {len(errors)} total errors (threshold: 5)")
        
        print(f"✅ Semgrep scan passed: {len(security_errors)} security errors, {len(errors)} total errors")
        return True
        
    except Exception as e:
        print(f"❌ Error parsing Semgrep report: {e}")
        return False

def check_ai_auditor_results():
    """Check PRSM AI Auditor validation results."""
    audit_dir = Path("audit_reports")
    if not audit_dir.exists():
        print("⚠️ AI Auditor reports not found, skipping...")
        return True
    
    # Find the most recent audit report
    report_files = list(audit_dir.glob("quick_validation_report_*.json"))
    if not report_files:
        print("⚠️ No AI Auditor validation reports found")
        return True
    
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        status = report.get('audit_status', 'UNKNOWN')
        failed_checks = report.get('summary', {}).get('failed_checks', 0)
        total_checks = report.get('summary', {}).get('total_checks', 0)
        success_rate = report.get('summary', {}).get('success_rate', '0%')
        
        if status == 'FAILED' or failed_checks > 0:
            print(f"❌ AI Auditor validation failed: {failed_checks}/{total_checks} checks failed")
            print(f"Success rate: {success_rate}")
            return False
        
        print(f"✅ AI Auditor validation passed: {success_rate} success rate")
        return True
        
    except Exception as e:
        print(f"❌ Error parsing AI Auditor report: {e}")
        return False

def main():
    """Main security check function."""
    print("🔍 Running Security Report Analysis...")
    print("=" * 50)
    
    checks = [
        ("Bandit Static Security Analysis", check_bandit_report),
        ("Safety Dependency Vulnerability Scan", check_safety_report), 
        ("Semgrep Static Analysis", check_semgrep_report),
        ("PRSM AI Auditor Validation", check_ai_auditor_results)
    ]
    
    all_passed = True
    
    for check_name, check_function in checks:
        print(f"\n🔎 {check_name}:")
        print("-" * 40)
        
        try:
            result = check_function()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL SECURITY CHECKS PASSED - Build can proceed")
        sys.exit(0)
    else:
        print("❌ SECURITY CHECKS FAILED - Blocking deployment")
        print("\nPlease address the security issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()