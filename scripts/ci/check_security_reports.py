#!/usr/bin/env python3
"""
Security Report Analysis for CI/CD Pipeline

Analyzes security scan results and fails the build if critical issues are found.
Addresses Gemini's requirement for automated security validation.
"""

import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check security reports and fail on critical vulnerabilities"
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Fail the build if critical vulnerabilities are found"
    )
    parser.add_argument(
        "--reports-dir",
        default=".",
        help="Directory containing security reports (default: current directory)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    return parser.parse_args()


def get_report_path(filename: str, reports_dir: str) -> str:
    """Get full path to a report file."""
    return os.path.join(reports_dir, filename)


def check_bandit_report(reports_dir: str, verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Check Bandit security scan results for critical issues.
    
    Returns:
        Tuple of (passed, critical_count, total_issues)
    """
    report_path = get_report_path("bandit-report.json", reports_dir)
    if not os.path.exists(report_path):
        print("⚠️ Bandit report not found, skipping...")
        return True, 0, 0
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        results = report.get('results', [])
        
        high_severity_issues = [
            issue for issue in results
            if issue.get('issue_severity') == 'HIGH'
        ]
        
        medium_issues = [
            issue for issue in results
            if issue.get('issue_severity') == 'MEDIUM'
        ]
        
        # Count critical issues (HIGH severity)
        critical_count = len(high_severity_issues)
        total_issues = len(results)
        
        if high_severity_issues:
            print(f"❌ Found {len(high_severity_issues)} HIGH severity security issues:")
            for issue in high_severity_issues[:5]:  # Show first 5
                print(f"  - {issue.get('test_id')}: {issue.get('issue_text')}")
                print(f"    File: {issue.get('filename')}:{issue.get('line_number')}")
                if verbose:
                    print(f"    Details: {issue.get('more_info', 'N/A')}")
            if len(high_severity_issues) > 5:
                print(f"  ... and {len(high_severity_issues) - 5} more")
            return False, critical_count, total_issues
        
        if len(medium_issues) > 10:
            print(f"⚠️ Found {len(medium_issues)} MEDIUM severity issues (threshold: 10)")
            print("Consider addressing these before production deployment")
        
        print(f"✅ Bandit scan passed: {len(high_severity_issues)} high, {len(medium_issues)} medium issues")
        return True, critical_count, total_issues
        
    except Exception as e:
        print(f"❌ Error parsing Bandit report: {e}")
        return False, 0, 0


def check_safety_report(reports_dir: str, verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Check Safety dependency vulnerability scan results.
    
    Returns:
        Tuple of (passed, critical_count, total_vulnerabilities)
    """
    report_path = get_report_path("safety-report.json", reports_dir)
    if not os.path.exists(report_path):
        print("⚠️ Safety report not found, skipping...")
        return True, 0, 0
    
    try:
        with open(report_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print("✅ Safety scan passed: No vulnerabilities found")
            return True, 0, 0
            
        report = json.loads(content)
        
        # Handle different Safety output formats
        if isinstance(report, list):
            vulnerabilities = report
        elif isinstance(report, dict):
            vulnerabilities = report.get('vulnerabilities', [])
            if not vulnerabilities:
                vulnerabilities = report.get('security_issues', [])
        else:
            vulnerabilities = []
        
        # Count critical vulnerabilities
        critical_vulns = []
        for vuln in vulnerabilities:
            if isinstance(vuln, dict):
                severity = vuln.get('severity', '').lower()
                if severity in ['critical', 'high']:
                    critical_vulns.append(vuln)
                elif vuln.get('vulnerability_id', '').startswith('CVE-'):
                    # Check advisory for severity indicators
                    advisory = vuln.get('advisory', '').lower()
                    if 'critical' in advisory or 'high' in advisory:
                        critical_vulns.append(vuln)
        
        critical_count = len(critical_vulns)
        total_vulns = len(vulnerabilities)
        
        if critical_vulns:
            print(f"❌ Found {len(critical_vulns)} critical/high vulnerabilities:")
            for vuln in critical_vulns[:3]:  # Show first 3
                vuln_id = vuln.get('vulnerability_id', vuln.get('id', 'Unknown'))
                package = vuln.get('package_name', vuln.get('package', 'Unknown'))
                print(f"  - {vuln_id}: {package}")
                if verbose:
                    advisory = vuln.get('advisory', 'No advisory available')
                    print(f"    {advisory[:100]}...")
            if len(critical_vulns) > 3:
                print(f"  ... and {len(critical_vulns) - 3} more")
            return False, critical_count, total_vulns
        
        if total_vulns > 0:
            print(f"⚠️ Found {total_vulns} lower-severity vulnerabilities")
            print("Consider updating dependencies")
        
        print(f"✅ Safety scan passed: {total_vulns} total vulnerabilities, 0 critical/high")
        return True, critical_count, total_vulns
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Could not parse Safety report as JSON: {e}")
        return True, 0, 0
    except Exception as e:
        print(f"❌ Error parsing Safety report: {e}")
        return False, 0, 0


def check_semgrep_report(reports_dir: str, verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Check Semgrep static analysis results.
    
    Returns:
        Tuple of (passed, critical_count, total_findings)
    """
    report_path = get_report_path("semgrep-report.json", reports_dir)
    if not os.path.exists(report_path):
        print("⚠️ Semgrep report not found, skipping...")
        return True, 0, 0
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        results = report.get('results', [])
        
        # Categorize findings by severity
        errors = []
        warnings = []
        security_errors = []
        
        for r in results:
            extra = r.get('extra', {})
            severity = extra.get('severity', 'INFO')
            check_id = r.get('check_id', '').lower()
            
            if severity == 'ERROR':
                errors.append(r)
                # Check for security-related errors
                if any(kw in check_id for kw in ['security', 'crypto', 'sql-injection', 'xss', 'injection', 'secrets', 'jwt', 'auth']):
                    security_errors.append(r)
            elif severity == 'WARNING':
                warnings.append(r)
        
        critical_count = len(security_errors)
        total_findings = len(results)
        
        if security_errors:
            print(f"❌ Found {len(security_errors)} security-related errors:")
            for error in security_errors[:3]:
                print(f"  - {error.get('check_id')}: {error.get('extra', {}).get('message', 'No message')}")
                print(f"    File: {error.get('path')}:{error.get('start', {}).get('line', 'unknown')}")
                if verbose:
                    lines = error.get('extra', {}).get('lines', '')
                    if lines:
                        print(f"    Code: {lines[:100]}...")
            if len(security_errors) > 3:
                print(f"  ... and {len(security_errors) - 3} more")
            return False, critical_count, total_findings
        
        if len(errors) > 5:
            print(f"⚠️ Found {len(errors)} total errors (threshold: 5)")
        
        print(f"✅ Semgrep scan passed: {len(security_errors)} security errors, {len(errors)} total errors, {len(warnings)} warnings")
        return True, critical_count, total_findings
        
    except Exception as e:
        print(f"❌ Error parsing Semgrep report: {e}")
        return False, 0, 0


def check_prsm_security_report(reports_dir: str, verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Check PRSM internal security scanner results.
    
    Returns:
        Tuple of (passed, critical_count, total_issues)
    """
    report_path = get_report_path("prsm-security-report.json", reports_dir)
    if not os.path.exists(report_path):
        print("⚠️ PRSM security report not found, skipping...")
        return True, 0, 0
    
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Parse PRSM security report format
        vulnerabilities = report.get('vulnerabilities', [])
        issues = report.get('issues', [])
        secrets = report.get('secrets', [])
        summary = report.get('summary', {})
        
        # Count critical/high severity items
        critical_count = 0
        
        for vuln in vulnerabilities:
            severity = vuln.get('severity', '').lower()
            if severity in ['critical', 'high']:
                critical_count += 1
        
        for issue in issues:
            severity = issue.get('severity', '').lower()
            if severity in ['critical', 'high']:
                critical_count += 1
        
        for secret in secrets:
            severity = secret.get('severity', '').lower()
            if severity in ['critical', 'high']:
                critical_count += 1
        
        total_issues = len(vulnerabilities) + len(issues) + len(secrets)
        risk_level = summary.get('risk_level', 'unknown')
        
        if critical_count > 0:
            print(f"❌ Found {critical_count} critical/high severity issues in PRSM scan:")
            
            for vuln in vulnerabilities[:3]:
                if vuln.get('severity', '').lower() in ['critical', 'high']:
                    print(f"  - {vuln.get('id', 'Unknown')}: {vuln.get('package', 'Unknown')}")
                    print(f"    {vuln.get('description', 'No description')[:100]}...")
            
            for issue in issues[:3]:
                if issue.get('severity', '').lower() in ['critical', 'high']:
                    print(f"  - {issue.get('issue_id', 'Unknown')}: {issue.get('message', 'No message')}")
                    print(f"    File: {issue.get('file_path')}:{issue.get('line_number')}")
            
            for secret in secrets[:3]:
                if secret.get('severity', '').lower() in ['critical', 'high']:
                    print(f"  - Secret leak: {secret.get('secret_type', 'Unknown')}")
                    print(f"    File: {secret.get('file_path')}:{secret.get('line_number')}")
            
            return False, critical_count, total_issues
        
        print(f"✅ PRSM security scan passed: Risk level '{risk_level}', {total_issues} total issues")
        if verbose and total_issues > 0:
            print(f"   Vulnerabilities: {len(vulnerabilities)}")
            print(f"   Code issues: {len(issues)}")
            print(f"   Secret leaks: {len(secrets)}")
        
        return True, critical_count, total_issues
        
    except Exception as e:
        print(f"❌ Error parsing PRSM security report: {e}")
        return False, 0, 0


def check_ai_auditor_results(reports_dir: str, verbose: bool = False) -> Tuple[bool, int, int]:
    """
    Check PRSM AI Auditor validation results.
    
    Returns:
        Tuple of (passed, failed_checks, total_checks)
    """
    audit_dir = Path(reports_dir) / "audit_reports"
    if not audit_dir.exists():
        # Also check in current directory
        audit_dir = Path("audit_reports")
    
    if not audit_dir.exists():
        print("⚠️ AI Auditor reports not found, skipping...")
        return True, 0, 0
    
    # Find the most recent audit report
    report_files = list(audit_dir.glob("quick_validation_report_*.json"))
    if not report_files:
        report_files = list(audit_dir.glob("*.json"))
    
    if not report_files:
        print("⚠️ No AI Auditor validation reports found")
        return True, 0, 0
    
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
            print(f"   Success rate: {success_rate}")
            if verbose:
                for check in report.get('checks', []):
                    if check.get('status') == 'FAILED':
                        print(f"   - {check.get('name')}: {check.get('message', 'No message')}")
            return False, failed_checks, total_checks
        
        print(f"✅ AI Auditor validation passed: {success_rate} success rate")
        return True, failed_checks, total_checks
        
    except Exception as e:
        print(f"❌ Error parsing AI Auditor report: {e}")
        return False, 0, 0


def main():
    """Main security check function."""
    args = parse_args()
    
    print("🔍 Running Security Report Analysis...")
    print("=" * 50)
    print(f"Reports directory: {args.reports_dir}")
    print(f"Fail on critical: {args.fail_on_critical}")
    print("=" * 50)
    
    checks = [
        ("Bandit Static Security Analysis", check_bandit_report),
        ("Safety Dependency Vulnerability Scan", check_safety_report), 
        ("Semgrep Static Analysis", check_semgrep_report),
        ("PRSM Security Scanner", check_prsm_security_report),
        ("PRSM AI Auditor Validation", check_ai_auditor_results)
    ]
    
    all_passed = True
    total_critical = 0
    total_issues = 0
    
    for check_name, check_function in checks:
        print(f"\n🔎 {check_name}:")
        print("-" * 40)
        
        try:
            passed, critical_count, issue_count = check_function(args.reports_dir, args.verbose)
            total_critical += critical_count
            total_issues += issue_count
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    print(f"📊 Security Summary:")
    print(f"   Total critical/high issues: {total_critical}")
    print(f"   Total issues found: {total_issues}")
    print("=" * 50)
    
    if args.fail_on_critical and total_critical > 0:
        print(f"❌ BUILD FAILED: {total_critical} critical/high severity issues found")
        print("\nPlease address the security issues above before proceeding.")
        sys.exit(1)
    elif not all_passed:
        print("❌ SECURITY CHECKS FAILED - Blocking deployment")
        print("\nPlease address the security issues above before proceeding.")
        sys.exit(1)
    else:
        print("✅ ALL SECURITY CHECKS PASSED - Build can proceed")
        sys.exit(0)


if __name__ == "__main__":
    main()