#!/usr/bin/env python3
"""
PRSM Security Audit Script
Automated security scanning for Python dependencies and integration with CI/CD
"""

import json
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SecurityAuditor:
    """
    Automated security auditing for PRSM dependencies
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.timestamp = datetime.now().isoformat()
        
    def run_pip_audit(self, requirements_file: str = "requirements.txt") -> Dict:
        """
        Run pip-audit and return results as JSON
        """
        try:
            result = subprocess.run([
                "pip-audit", 
                "-r", str(self.project_root / requirements_file),
                "--format", "json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # pip-audit returns 0 for no vulnerabilities, non-zero for vulnerabilities found
            if result.stdout:
                audit_data = json.loads(result.stdout)
                if result.returncode == 0:
                    audit_data["status"] = "clean"
                else:
                    audit_data["status"] = "vulnerabilities_found"
                return audit_data
            else:
                return {"error": result.stderr, "status": "error"}
                    
        except subprocess.CalledProcessError as e:
            return {"error": str(e), "status": "error"}
        except FileNotFoundError:
            return {"error": "pip-audit not found. Install with: pipx install pip-audit", "status": "error"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse pip-audit JSON output: {e}", "status": "error"}
    
    def generate_audit_report(self, audit_results: Dict) -> Dict:
        """
        Generate comprehensive audit report
        """
        report = {
            "timestamp": self.timestamp,
            "project": "PRSM",
            "audit_tool": "pip-audit",
            "status": audit_results.get("status", "unknown"),
            "summary": {
                "total_vulnerabilities": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0
            },
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Extract vulnerabilities from dependencies format
        if "dependencies" in audit_results:
            all_vulns = []
            for dep in audit_results["dependencies"]:
                for vuln in dep.get("vulns", []):
                    vuln_with_package = vuln.copy()
                    vuln_with_package["package"] = dep["name"]
                    vuln_with_package["version"] = dep["version"]
                    all_vulns.append(vuln_with_package)
            
            report["summary"]["total_vulnerabilities"] = len(all_vulns)
            
            for vuln in all_vulns:
                severity = self._assess_severity(vuln)
                report["summary"][f"{severity}_count"] += 1
                
                vuln_info = {
                    "package": vuln.get("package", "unknown"),
                    "version": vuln.get("version", "unknown"),
                    "vulnerability_id": vuln.get("id", "unknown"),
                    "severity": severity,
                    "description": vuln.get("description", ""),
                    "fix_versions": vuln.get("fix_versions", []),
                    "aliases": vuln.get("aliases", [])
                }
                report["vulnerabilities"].append(vuln_info)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _assess_severity(self, vulnerability: Dict) -> str:
        """
        Assess vulnerability severity based on available information
        """
        # This is a simplified severity assessment
        # In production, you might integrate with CVSS scores or other severity databases
        
        description = vulnerability.get("description", "").lower()
        vuln_id = vulnerability.get("id", "")
        
        # Look for severity keywords
        if any(keyword in description for keyword in ["critical", "remote code execution", "rce"]):
            return "critical"
        elif any(keyword in description for keyword in ["high", "sql injection", "xss", "csrf"]):
            return "high"
        elif any(keyword in description for keyword in ["denial of service", "dos", "crash"]):
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """
        Generate security recommendations based on audit results
        """
        recommendations = []
        
        total_vulns = report["summary"]["total_vulnerabilities"]
        
        if total_vulns == 0:
            recommendations.append("✅ No known vulnerabilities detected in dependencies")
            recommendations.append("🔄 Continue regular security audits with automated CI/CD integration")
            recommendations.append("📊 Consider implementing dependency scanning in pre-commit hooks")
        else:
            recommendations.append(f"🚨 Found {total_vulns} vulnerabilities that require attention")
            
            if report["summary"]["critical_count"] > 0:
                recommendations.append("🔴 CRITICAL: Address critical vulnerabilities immediately")
                recommendations.append("🚫 Consider blocking deployments until critical issues are resolved")
            
            if report["summary"]["high_count"] > 0:
                recommendations.append("🟠 HIGH: Schedule high-severity vulnerability fixes within 7 days")
            
            # Package-specific recommendations
            for vuln in report["vulnerabilities"]:
                if vuln["fix_versions"]:
                    recommendations.append(
                        f"📦 Update {vuln['package']} from {vuln['version']} to {', '.join(vuln['fix_versions'])}"
                    )
                else:
                    recommendations.append(
                        f"⚠️  No fix available for {vuln['package']} {vuln['version']} "
                        f"(ID: {vuln['vulnerability_id']}). Monitor for updates."
                    )
        
        # General security recommendations
        recommendations.extend([
            "🔒 Implement automated dependency scanning in CI/CD pipeline",
            "📅 Schedule weekly security audits",
            "🔄 Keep dependencies updated regularly",
            "📋 Document security review process for new dependencies"
        ])
        
        return recommendations
    
    def save_report(self, report: Dict, output_file: Optional[str] = None):
        """
        Save audit report to file
        """
        if output_file is None:
            output_file = f"security-audit-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        
        output_path = self.project_root / "security" / "audit-reports" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def print_summary(self, report: Dict):
        """
        Print human-readable audit summary
        """
        print(f"\n🔒 PRSM Security Audit Report")
        print(f"{'='*50}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Status: {report['status'].upper()}")
        print(f"Total Vulnerabilities: {report['summary']['total_vulnerabilities']}")
        
        if report['summary']['total_vulnerabilities'] > 0:
            print(f"\nSeverity Breakdown:")
            print(f"  🔴 Critical: {report['summary']['critical_count']}")
            print(f"  🟠 High: {report['summary']['high_count']}")
            print(f"  🟡 Medium: {report['summary']['medium_count']}")
            print(f"  🔵 Low: {report['summary']['low_count']}")
            
            print(f"\nVulnerabilities Found:")
            for vuln in report['vulnerabilities']:
                severity_emoji = {
                    'critical': '🔴',
                    'high': '🟠', 
                    'medium': '🟡',
                    'low': '🔵'
                }.get(vuln['severity'], '❓')
                
                print(f"  {severity_emoji} {vuln['package']} {vuln['version']} - {vuln['vulnerability_id']}")
                if vuln['fix_versions']:
                    print(f"    ✅ Fix available: {', '.join(vuln['fix_versions'])}")
                else:
                    print(f"    ⚠️  No fix available yet")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'][:10], 1):  # Show top 10
            print(f"  {i}. {rec}")
        
        if len(report['recommendations']) > 10:
            print(f"  ... and {len(report['recommendations']) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="PRSM Security Audit Tool")
    parser.add_argument("--requirements", default="requirements.txt", 
                       help="Requirements file to audit")
    parser.add_argument("--output", help="Output file for detailed report")
    parser.add_argument("--json-only", action="store_true", 
                       help="Output only JSON, no human-readable summary")
    parser.add_argument("--fail-on-vuln", action="store_true",
                       help="Exit with code 1 if vulnerabilities found (for CI/CD)")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    auditor = SecurityAuditor(project_root)
    
    # Run audit
    print("🔍 Running security audit...", file=sys.stderr)
    audit_results = auditor.run_pip_audit(args.requirements)
    
    if audit_results.get("status") == "error":
        print(f"❌ Audit failed: {audit_results.get('error')}", file=sys.stderr)
        sys.exit(1)
    
    # Generate report
    report = auditor.generate_audit_report(audit_results)
    
    # Save report
    if args.output:
        output_path = auditor.save_report(report, args.output)
        print(f"📄 Detailed report saved to: {output_path}", file=sys.stderr)
    
    # Output results
    if args.json_only:
        print(json.dumps(report, indent=2))
    else:
        auditor.print_summary(report)
        
        if args.output:
            print(f"\nDetailed JSON report: {output_path}")
    
    # Exit with error code if vulnerabilities found and fail-on-vuln is set
    if args.fail_on_vuln and report['summary']['total_vulnerabilities'] > 0:
        print(f"\n❌ Exiting with error code due to {report['summary']['total_vulnerabilities']} vulnerabilities", 
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()