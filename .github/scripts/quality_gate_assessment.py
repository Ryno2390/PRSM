#!/usr/bin/env python3
"""
Quality Gate Assessment Script for PRSM CI/CD Pipeline
Evaluates multiple quality metrics to determine deployment readiness.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
try:
    from defusedxml import ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    # Use secure parser settings as fallback
    _original_parse = ET.parse
    def _secure_parse(source):
        parser = ET.XMLParser(resolve_entities=False, forbid_entities=True, forbid_external=True)
        return _original_parse(source, parser=parser)
    ET.parse = _secure_parse

@dataclass
class QualityMetric:
    """Quality metric with scoring and thresholds"""
    name: str
    score: float
    threshold: float
    status: str
    details: Dict[str, Any]

@dataclass
class QualityGateResult:
    """Overall quality gate assessment result"""
    tests: QualityMetric
    validation: QualityMetric
    performance: QualityMetric
    security: QualityMetric
    overall: QualityMetric

class QualityGateAssessor:
    """Assesses quality metrics for deployment decisions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality gate thresholds
        self.thresholds = {
            'test_coverage': 85.0,
            'test_success_rate': 95.0,
            'validation_success_rate': 95.0,
            'performance_regression': 5.0,  # Max % regression allowed
            'security_high_vulns': 0,       # No high severity vulnerabilities
            'security_medium_vulns': 2,     # Max 2 medium severity
            'overall_score': 80.0           # Minimum overall score
        }
    
    def assess_test_quality(self, test_results_dir: Path) -> QualityMetric:
        """Assess test quality from JUnit XML results"""
        try:
            test_files = list(test_results_dir.glob("*-test-results.xml"))
            
            total_tests = 0
            failed_tests = 0
            error_tests = 0
            skipped_tests = 0
            
            for test_file in test_files:
                tree = ET.parse(test_file)
                root = tree.getroot()
                
                # Handle different JUnit XML formats
                if root.tag == 'testsuite':
                    suites = [root]
                else:
                    suites = root.findall('.//testsuite')
                
                for suite in suites:
                    total_tests += int(suite.get('tests', 0))
                    failed_tests += int(suite.get('failures', 0))
                    error_tests += int(suite.get('errors', 0))
                    skipped_tests += int(suite.get('skipped', 0))
            
            if total_tests == 0:
                return QualityMetric(
                    name="tests",
                    score=0.0,
                    threshold=self.thresholds['test_success_rate'],
                    status="FAIL",
                    details={"error": "No test results found"}
                )
            
            success_rate = ((total_tests - failed_tests - error_tests) / total_tests) * 100
            
            # Calculate score based on success rate
            score = min(success_rate, 100.0)
            status = "PASS" if success_rate >= self.thresholds['test_success_rate'] else "FAIL"
            
            return QualityMetric(
                name="tests",
                score=score,
                threshold=self.thresholds['test_success_rate'],
                status=status,
                details={
                    "total_tests": total_tests,
                    "failed_tests": failed_tests,
                    "error_tests": error_tests,
                    "skipped_tests": skipped_tests,
                    "success_rate": success_rate
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess test quality: {e}")
            return QualityMetric(
                name="tests",
                score=0.0,
                threshold=self.thresholds['test_success_rate'],
                status="FAIL",
                details={"error": str(e)}
            )
    
    def assess_validation_quality(self, validation_file: Path) -> QualityMetric:
        """Assess PRSM validation quality"""
        try:
            if not validation_file.exists():
                return QualityMetric(
                    name="validation",
                    score=0.0,
                    threshold=self.thresholds['validation_success_rate'],
                    status="FAIL",
                    details={"error": "Validation results not found"}
                )
            
            with open(validation_file) as f:
                validation_data = json.load(f)
            
            success_rate = validation_data.get('summary', {}).get('success_rate', 0.0)
            
            # Calculate score based on success rate
            score = min(success_rate, 100.0)
            status = "PASS" if success_rate >= self.thresholds['validation_success_rate'] else "FAIL"
            
            return QualityMetric(
                name="validation",
                score=score,
                threshold=self.thresholds['validation_success_rate'],
                status=status,
                details={
                    "success_rate": success_rate,
                    "total_tests": validation_data.get('summary', {}).get('total_tests', 0),
                    "passed_tests": validation_data.get('summary', {}).get('passed_tests', 0),
                    "failed_tests": validation_data.get('summary', {}).get('failed_tests', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess validation quality: {e}")
            return QualityMetric(
                name="validation",
                score=0.0,
                threshold=self.thresholds['validation_success_rate'],
                status="FAIL",
                details={"error": str(e)}
            )
    
    def assess_performance_quality(self, benchmark_file: Path) -> QualityMetric:
        """Assess performance benchmark quality"""
        try:
            if not benchmark_file.exists():
                return QualityMetric(
                    name="performance",
                    score=75.0,  # Default score if no benchmarks
                    threshold=100.0 - self.thresholds['performance_regression'],
                    status="PASS",
                    details={"note": "No performance benchmarks available"}
                )
            
            with open(benchmark_file) as f:
                benchmark_data = json.load(f)
            
            # Check for performance regressions
            overall_score = benchmark_data.get('overall_score', 75.0)
            
            # Assume baseline score of 75.0 for regression calculation
            baseline_score = 75.0
            regression_pct = ((baseline_score - overall_score) / baseline_score) * 100
            
            # Score based on regression level
            if regression_pct <= self.thresholds['performance_regression']:
                score = 100.0 - regression_pct
                status = "PASS"
            else:
                score = 100.0 - regression_pct
                status = "FAIL"
            
            return QualityMetric(
                name="performance",
                score=max(score, 0.0),
                threshold=100.0 - self.thresholds['performance_regression'],
                status=status,
                details={
                    "overall_score": overall_score,
                    "baseline_score": baseline_score,
                    "regression_pct": regression_pct,
                    "benchmarks": benchmark_data.get('benchmarks', {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess performance quality: {e}")
            return QualityMetric(
                name="performance",
                score=50.0,
                threshold=100.0 - self.thresholds['performance_regression'],
                status="FAIL",
                details={"error": str(e)}
            )
    
    def assess_security_quality(self, security_dir: Path) -> QualityMetric:
        """Assess security scan quality"""
        try:
            security_files = list(security_dir.glob("*.json"))
            
            total_high_vulns = 0
            total_medium_vulns = 0
            total_low_vulns = 0
            
            for security_file in security_files:
                with open(security_file) as f:
                    security_data = json.load(f)
                
                # Handle different security report formats
                if 'results' in security_data:  # Bandit format
                    for result in security_data['results']:
                        severity = result.get('issue_severity', 'low').lower()
                        if severity == 'high':
                            total_high_vulns += 1
                        elif severity == 'medium':
                            total_medium_vulns += 1
                        else:
                            total_low_vulns += 1
                
                elif 'vulnerabilities' in security_data:  # Safety format
                    for vuln in security_data['vulnerabilities']:
                        severity = vuln.get('severity', 'low').lower()
                        if severity == 'high':
                            total_high_vulns += 1
                        elif severity == 'medium':
                            total_medium_vulns += 1
                        else:
                            total_low_vulns += 1
            
            # Calculate security score
            score = 100.0
            status = "PASS"
            
            if total_high_vulns > self.thresholds['security_high_vulns']:
                score -= total_high_vulns * 25  # High vulns heavily penalized
                status = "FAIL"
            
            if total_medium_vulns > self.thresholds['security_medium_vulns']:
                score -= (total_medium_vulns - self.thresholds['security_medium_vulns']) * 10
                if total_medium_vulns > 5:  # Too many medium vulns = fail
                    status = "FAIL"
            
            score -= total_low_vulns * 2  # Minor penalty for low vulns
            score = max(score, 0.0)
            
            return QualityMetric(
                name="security",
                score=score,
                threshold=80.0,  # Minimum security score
                status=status,
                details={
                    "high_vulnerabilities": total_high_vulns,
                    "medium_vulnerabilities": total_medium_vulns,
                    "low_vulnerabilities": total_low_vulns,
                    "total_vulnerabilities": total_high_vulns + total_medium_vulns + total_low_vulns
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess security quality: {e}")
            return QualityMetric(
                name="security",
                score=50.0,
                threshold=80.0,
                status="FAIL",
                details={"error": str(e)}
            )
    
    def calculate_overall_score(self, metrics: List[QualityMetric]) -> QualityMetric:
        """Calculate overall quality score"""
        # Weighted scoring
        weights = {
            'tests': 0.3,
            'validation': 0.3,
            'performance': 0.2,
            'security': 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = weights.get(metric.name, 0.0)
            weighted_score += metric.score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0
        
        # Overall status based on individual statuses and overall score
        failed_critical = any(m.status == "FAIL" and m.name in ['tests', 'validation'] for m in metrics)
        overall_status = "FAIL" if (failed_critical or overall_score < self.thresholds['overall_score']) else "PASS"
        
        return QualityMetric(
            name="overall",
            score=overall_score,
            threshold=self.thresholds['overall_score'],
            status=overall_status,
            details={
                "weighted_score": overall_score,
                "weights": weights,
                "critical_failures": failed_critical
            }
        )
    
    def assess_quality_gates(self, 
                           validation_report: Path = None,
                           benchmark_report: Path = None,
                           security_reports: Path = None,
                           test_results: Path = None) -> QualityGateResult:
        """Perform comprehensive quality gate assessment"""
        
        self.logger.info("🎯 Starting quality gate assessment")
        
        # Assess each quality dimension
        tests = self.assess_test_quality(test_results) if test_results else QualityMetric(
            "tests", 75.0, 95.0, "PASS", {"note": "No test results provided"}
        )
        
        validation = self.assess_validation_quality(validation_report) if validation_report else QualityMetric(
            "validation", 75.0, 95.0, "PASS", {"note": "No validation report provided"}
        )
        
        performance = self.assess_performance_quality(benchmark_report) if benchmark_report else QualityMetric(
            "performance", 75.0, 95.0, "PASS", {"note": "No benchmark report provided"}
        )
        
        security = self.assess_security_quality(security_reports) if security_reports else QualityMetric(
            "security", 75.0, 80.0, "PASS", {"note": "No security reports provided"}
        )
        
        # Calculate overall score
        overall = self.calculate_overall_score([tests, validation, performance, security])
        
        result = QualityGateResult(
            tests=tests,
            validation=validation,
            performance=performance,
            security=security,
            overall=overall
        )
        
        self.logger.info(f"✅ Quality gate assessment completed: {overall.status}")
        return result

def main():
    """Main quality gate assessment function"""
    parser = argparse.ArgumentParser(description="PRSM Quality Gate Assessment")
    parser.add_argument("--validation-report", type=Path, help="Path to validation results JSON")
    parser.add_argument("--benchmark-report", type=Path, help="Path to benchmark results JSON")
    parser.add_argument("--security-reports", type=Path, help="Path to security reports directory")
    parser.add_argument("--test-results", type=Path, help="Path to test results directory")
    parser.add_argument("--output", type=Path, default="deployment-decision.json", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    assessor = QualityGateAssessor()
    
    try:
        # Perform assessment
        result = assessor.assess_quality_gates(
            validation_report=args.validation_report,
            benchmark_report=args.benchmark_report,
            security_reports=args.security_reports,
            test_results=args.test_results
        )
        
        # Convert to JSON-serializable format
        output_data = {
            "tests": {
                "score": result.tests.score,
                "threshold": result.tests.threshold,
                "status": result.tests.status,
                "details": result.tests.details
            },
            "validation": {
                "score": result.validation.score,
                "threshold": result.validation.threshold,
                "status": result.validation.status,
                "details": result.validation.details
            },
            "performance": {
                "score": result.performance.score,
                "threshold": result.performance.threshold,
                "status": result.performance.status,
                "details": result.performance.details
            },
            "security": {
                "score": result.security.score,
                "threshold": result.security.threshold,
                "status": result.security.status,
                "details": result.security.details
            },
            "overall": {
                "score": result.overall.score,
                "threshold": result.overall.threshold,
                "status": result.overall.status,
                "details": result.overall.details
            },
            "assessment_summary": {
                "timestamp": "2024-06-21T00:00:00Z",
                "assessor_version": "1.0.0",
                "deployment_approved": result.overall.status == "PASS"
            }
        }
        
        # Write results
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📊 Quality gate assessment completed")
        print(f"📄 Results written to: {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if result.overall.status == "PASS" else 1)
        
    except Exception as e:
        logging.error(f"Quality gate assessment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()