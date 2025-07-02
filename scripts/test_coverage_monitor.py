#!/usr/bin/env python3
"""
Comprehensive Test Coverage Monitor for PRSM
============================================

Monitors and enforces consistent test coverage across all project components:
- Python core modules (pytest + coverage.py)
- JavaScript SDK (Jest)  
- AI Concierge (Jest + TypeScript)
- Smart Contracts (Hardhat + Solidity coverage)

Addresses cold developer audit feedback on inconsistent test coverage measurement.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET

@dataclass
class CoverageResult:
    component: str
    coverage_percentage: float
    lines_covered: int
    lines_total: int
    threshold: float
    status: str  # 'pass', 'fail', 'warning'
    details: Dict[str, any]

class TestCoverageMonitor:
    """Unified test coverage monitoring across all PRSM components"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[CoverageResult] = []
        
        # Coverage thresholds by component type
        self.thresholds = {
            # Critical Security Components: 80%
            'auth': 80.0,
            'security': 80.0,
            'cryptography': 80.0,
            'smart_contracts': 80.0,
            'ai_concierge_llm': 80.0,
            'payment': 80.0,
            
            # Core Framework: 60%
            'api_endpoints': 60.0,
            'agent_framework': 60.0,
            'core_services': 60.0,
            'database': 60.0,
            
            # Supporting Systems: 40%
            'marketplace': 40.0,
            'governance': 40.0,
            'monitoring': 40.0,
            'utilities': 40.0,
            
            # Overall minimum: 50%
            'overall': 50.0
        }
    
    def run_all_coverage_checks(self) -> Dict[str, CoverageResult]:
        """Run coverage checks for all project components"""
        print("🧪 Running Comprehensive Test Coverage Analysis")
        print("=" * 60)
        
        # Run Python coverage
        python_result = self._run_python_coverage()
        if python_result:
            self.results.append(python_result)
        
        # Run JavaScript SDK coverage
        js_sdk_result = self._run_js_sdk_coverage()
        if js_sdk_result:
            self.results.append(js_sdk_result)
        
        # Run AI Concierge coverage
        ai_concierge_result = self._run_ai_concierge_coverage()
        if ai_concierge_result:
            self.results.append(ai_concierge_result)
        
        # Run Smart Contract coverage
        smart_contract_result = self._run_smart_contract_coverage()
        if smart_contract_result:
            self.results.append(smart_contract_result)
        
        # Generate summary report
        self._generate_summary_report()
        
        return {result.component: result for result in self.results}
    
    def _run_python_coverage(self) -> Optional[CoverageResult]:
        """Run Python test coverage using pytest and coverage.py"""
        print("\\n📊 Running Python Test Coverage...")
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, '-m', 'pytest',
                '--cov=prsm',
                '--cov-report=xml:coverage-python.xml',
                '--cov-report=term-missing',
                '--cov-report=html:htmlcov-python',
                'tests/',
                '-v'
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"⚠️  Python tests failed: {result.stderr}")
                return CoverageResult(
                    component="python_core",
                    coverage_percentage=0.0,
                    lines_covered=0,
                    lines_total=0,
                    threshold=self.thresholds['overall'],
                    status="fail",
                    details={"error": result.stderr}
                )
            
            # Parse coverage XML
            coverage_xml = self.project_root / "coverage-python.xml"
            if coverage_xml.exists():
                coverage_data = self._parse_python_coverage_xml(coverage_xml)
                return CoverageResult(
                    component="python_core",
                    coverage_percentage=coverage_data['coverage'],
                    lines_covered=coverage_data['lines_covered'],
                    lines_total=coverage_data['lines_total'],
                    threshold=self.thresholds['overall'],
                    status="pass" if coverage_data['coverage'] >= self.thresholds['overall'] else "fail",
                    details=coverage_data
                )
            
        except subprocess.TimeoutExpired:
            print("⏰ Python test coverage timed out")
        except Exception as e:
            print(f"❌ Python coverage error: {e}")
        
        return None
    
    def _run_js_sdk_coverage(self) -> Optional[CoverageResult]:
        """Run JavaScript SDK test coverage using Jest"""
        print("\\n📊 Running JavaScript SDK Coverage...")
        
        js_sdk_path = self.project_root / "sdks" / "javascript"
        if not js_sdk_path.exists():
            print("⚠️  JavaScript SDK directory not found")
            return None
        
        try:
            # Run Jest with coverage
            result = subprocess.run(
                ["npm", "run", "test:coverage"],
                cwd=js_sdk_path,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )
            
            # Parse Jest coverage output
            coverage_data = self._parse_jest_coverage(js_sdk_path / "coverage")
            
            return CoverageResult(
                component="javascript_sdk",
                coverage_percentage=coverage_data.get('coverage', 0.0),
                lines_covered=coverage_data.get('lines_covered', 0),
                lines_total=coverage_data.get('lines_total', 0),
                threshold=80.0,  # SDK requires 80% coverage
                status="pass" if coverage_data.get('coverage', 0) >= 80.0 else "fail",
                details=coverage_data
            )
            
        except subprocess.TimeoutExpired:
            print("⏰ JavaScript SDK coverage timed out")
        except Exception as e:
            print(f"❌ JavaScript SDK coverage error: {e}")
        
        return None
    
    def _run_ai_concierge_coverage(self) -> Optional[CoverageResult]:
        """Run AI Concierge test coverage using Jest"""
        print("\\n📊 Running AI Concierge Coverage...")
        
        ai_concierge_path = self.project_root / "ai-concierge"
        if not ai_concierge_path.exists():
            print("⚠️  AI Concierge directory not found")
            return None
        
        try:
            # Install dependencies if needed
            if not (ai_concierge_path / "node_modules").exists():
                subprocess.run(["npm", "install"], cwd=ai_concierge_path, check=True)
            
            # Run Jest with coverage
            result = subprocess.run(
                ["npm", "run", "test:ci"],
                cwd=ai_concierge_path,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            # Parse Jest coverage output
            coverage_data = self._parse_jest_coverage(ai_concierge_path / "coverage")
            
            return CoverageResult(
                component="ai_concierge",
                coverage_percentage=coverage_data.get('coverage', 0.0),
                lines_covered=coverage_data.get('lines_covered', 0),
                lines_total=coverage_data.get('lines_total', 0),
                threshold=self.thresholds['ai_concierge_llm'],
                status="pass" if coverage_data.get('coverage', 0) >= self.thresholds['ai_concierge_llm'] else "fail",
                details=coverage_data
            )
            
        except Exception as e:
            print(f"❌ AI Concierge coverage error: {e}")
        
        return None
    
    def _run_smart_contract_coverage(self) -> Optional[CoverageResult]:
        """Run Smart Contract test coverage using Hardhat"""
        print("\\n📊 Running Smart Contract Coverage...")
        
        contracts_path = self.project_root / "contracts"
        if not contracts_path.exists():
            print("⚠️  Contracts directory not found")
            return None
        
        try:
            # Install dependencies if needed
            if not (contracts_path / "node_modules").exists():
                subprocess.run(["npm", "install"], cwd=contracts_path, check=True)
            
            # Run Hardhat coverage
            result = subprocess.run(
                ["npx", "hardhat", "coverage"],
                cwd=contracts_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse Hardhat coverage output
            coverage_data = self._parse_hardhat_coverage(contracts_path)
            
            return CoverageResult(
                component="smart_contracts",
                coverage_percentage=coverage_data.get('coverage', 0.0),
                lines_covered=coverage_data.get('lines_covered', 0),
                lines_total=coverage_data.get('lines_total', 0),
                threshold=self.thresholds['smart_contracts'],
                status="pass" if coverage_data.get('coverage', 0) >= self.thresholds['smart_contracts'] else "fail",
                details=coverage_data
            )
            
        except Exception as e:
            print(f"❌ Smart contract coverage error: {e}")
        
        return None
    
    def _parse_python_coverage_xml(self, xml_path: Path) -> Dict[str, any]:
        """Parse Python coverage XML output"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract overall coverage
            coverage_attr = root.attrib
            lines_covered = int(coverage_attr.get('lines-covered', 0))
            lines_valid = int(coverage_attr.get('lines-valid', 1))
            coverage_percentage = (lines_covered / lines_valid) * 100 if lines_valid > 0 else 0
            
            return {
                'coverage': round(coverage_percentage, 2),
                'lines_covered': lines_covered,
                'lines_total': lines_valid,
                'format': 'python_xml'
            }
        except Exception as e:
            print(f"Error parsing Python coverage XML: {e}")
            return {'coverage': 0.0, 'lines_covered': 0, 'lines_total': 0}
    
    def _parse_jest_coverage(self, coverage_dir: Path) -> Dict[str, any]:
        """Parse Jest coverage JSON output"""
        try:
            coverage_json = coverage_dir / "coverage-final.json"
            if not coverage_json.exists():
                return {'coverage': 0.0, 'lines_covered': 0, 'lines_total': 0}
            
            with open(coverage_json, 'r') as f:
                data = json.load(f)
            
            total_lines = 0
            covered_lines = 0
            
            for file_path, file_data in data.items():
                if 'l' in file_data:  # Line coverage data
                    lines = file_data['l']
                    total_lines += len(lines)
                    covered_lines += sum(1 for count in lines.values() if count > 0)
            
            coverage_percentage = (covered_lines / total_lines) * 100 if total_lines > 0 else 0
            
            return {
                'coverage': round(coverage_percentage, 2),
                'lines_covered': covered_lines,
                'lines_total': total_lines,
                'format': 'jest_json'
            }
        except Exception as e:
            print(f"Error parsing Jest coverage: {e}")
            return {'coverage': 0.0, 'lines_covered': 0, 'lines_total': 0}
    
    def _parse_hardhat_coverage(self, contracts_dir: Path) -> Dict[str, any]:
        """Parse Hardhat coverage output"""
        try:
            # Hardhat generates coverage in coverage.json
            coverage_json = contracts_dir / "coverage" / "coverage.json"
            if not coverage_json.exists():
                return {'coverage': 0.0, 'lines_covered': 0, 'lines_total': 0}
            
            with open(coverage_json, 'r') as f:
                data = json.load(f)
            
            total_lines = 0
            covered_lines = 0
            
            for file_path, file_data in data.items():
                if 'l' in file_data:
                    lines = file_data['l']
                    total_lines += len(lines)
                    covered_lines += sum(1 for count in lines.values() if count > 0)
            
            coverage_percentage = (covered_lines / total_lines) * 100 if total_lines > 0 else 0
            
            return {
                'coverage': round(coverage_percentage, 2),
                'lines_covered': covered_lines,
                'lines_total': total_lines,
                'format': 'hardhat_json'
            }
        except Exception as e:
            print(f"Error parsing Hardhat coverage: {e}")
            return {'coverage': 0.0, 'lines_covered': 0, 'lines_total': 0}
    
    def _generate_summary_report(self):
        """Generate comprehensive coverage summary report"""
        print("\\n" + "=" * 60)
        print("📋 PRSM Test Coverage Summary Report")
        print("=" * 60)
        
        total_coverage = 0
        component_count = 0
        
        for result in self.results:
            status_emoji = "✅" if result.status == "pass" else "❌"
            print(f"{status_emoji} {result.component:20} {result.coverage_percentage:6.1f}% "
                  f"(threshold: {result.threshold:4.1f}%)")
            
            total_coverage += result.coverage_percentage
            component_count += 1
        
        overall_coverage = total_coverage / component_count if component_count > 0 else 0
        overall_status = "✅" if overall_coverage >= self.thresholds['overall'] else "❌"
        
        print("-" * 60)
        print(f"{overall_status} Overall Coverage:    {overall_coverage:6.1f}% "
              f"(threshold: {self.thresholds['overall']:4.1f}%)")
        
        # Generate recommendations
        print("\\n📈 Recommendations:")
        failed_components = [r for r in self.results if r.status == "fail"]
        
        if not failed_components:
            print("✅ All components meet coverage thresholds!")
        else:
            for result in failed_components:
                gap = result.threshold - result.coverage_percentage
                print(f"  • {result.component}: Increase coverage by {gap:.1f}% "
                      f"({int(gap * result.lines_total / 100)} lines)")
        
        # Save detailed report
        self._save_coverage_report()
    
    def _save_coverage_report(self):
        """Save detailed coverage report to JSON"""
        report_data = {
            'timestamp': str(subprocess.check_output(['date'], text=True).strip()),
            'overall_coverage': sum(r.coverage_percentage for r in self.results) / len(self.results) if self.results else 0,
            'components': {
                result.component: {
                    'coverage': result.coverage_percentage,
                    'lines_covered': result.lines_covered,
                    'lines_total': result.lines_total,
                    'threshold': result.threshold,
                    'status': result.status,
                    'details': result.details
                }
                for result in self.results
            }
        }
        
        report_path = self.project_root / "coverage-report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\\n💾 Detailed report saved to: {report_path}")

def main():
    """Main entry point for coverage monitoring"""
    project_root = Path(__file__).parent.parent
    
    if not project_root.exists():
        print("❌ Project root not found")
        sys.exit(1)
    
    monitor = TestCoverageMonitor(project_root)
    results = monitor.run_all_coverage_checks()
    
    # Exit with error code if any critical component fails
    critical_failures = [
        r for r in results.values() 
        if r.status == "fail" and r.component in ["python_core", "ai_concierge", "smart_contracts"]
    ]
    
    if critical_failures:
        print(f"\\n❌ {len(critical_failures)} critical component(s) failed coverage requirements")
        sys.exit(1)
    
    print("\\n✅ Coverage monitoring completed successfully")

if __name__ == "__main__":
    main()