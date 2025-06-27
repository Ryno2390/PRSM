#!/usr/bin/env python3
"""
Comprehensive Link Validation Script for PRSM Repository

This script validates all internal links in markdown files throughout the repository,
with special attention to the main README.md and documentation files that were
reorganized during the AI auditor enhancement project.

Usage:
    python scripts/comprehensive_link_validator.py
    python scripts/comprehensive_link_validator.py --fix-links
    python scripts/comprehensive_link_validator.py --report-only
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from urllib.parse import urlparse, unquote

class LinkValidator:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.broken_links = []
        self.valid_links = []
        self.external_links = []
        self.anchor_links = []
        self.file_moves = {
            # Document the file moves we made during reorganization
            'INVESTMENT_READINESS_REPORT.md': 'docs/business/INVESTMENT_READINESS_REPORT.md',
            'INVESTOR_MATERIALS.md': 'docs/business/INVESTOR_MATERIALS.md',
            'AI_AUDITOR_INDEX.md': 'docs/ai-auditor/AI_AUDITOR_INDEX.md',
            'AI_AUDIT_GUIDE.md': 'docs/ai-auditor/AI_AUDIT_GUIDE.md',
            'TECHNICAL_CLAIMS_VALIDATION.md': 'docs/ai-auditor/TECHNICAL_CLAIMS_VALIDATION.md',
            'ARCHITECTURE_METADATA.json': 'docs/metadata/ARCHITECTURE_METADATA.json',
            'PERFORMANCE_BENCHMARKS.json': 'docs/metadata/PERFORMANCE_BENCHMARKS.json',
            'SECURITY_ATTESTATION.json': 'docs/metadata/SECURITY_ATTESTATION.json'
        }
        
    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files in the repository"""
        md_files = []
        for root, dirs, files in os.walk(self.repo_root):
            # Skip certain directories
            skip_dirs = {'.git', 'node_modules', '.env', '__pycache__', '.pytest_cache'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith('.md'):
                    md_files.append(Path(root) / file)
        
        return sorted(md_files)
    
    def extract_links(self, content: str, source_file: Path) -> List[Dict]:
        """Extract all markdown links from content"""
        links = []
        
        # Pattern for markdown links: [text](url) and [text](url "title")
        link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        
        for match in re.finditer(link_pattern, content):
            link_text = match.group(1)
            link_url = match.group(2).split(' ')[0].strip('"\'')  # Remove titles and quotes
            
            links.append({
                'text': link_text,
                'url': link_url,
                'line': content[:match.start()].count('\n') + 1,
                'source_file': source_file
            })
        
        return links
    
    def categorize_link(self, link_url: str) -> str:
        """Categorize link type"""
        if link_url.startswith(('http://', 'https://')):
            return 'external'
        elif link_url.startswith('#'):
            return 'anchor'
        elif link_url.startswith('mailto:'):
            return 'mailto'
        else:
            return 'internal'
    
    def resolve_internal_link(self, link_url: str, source_file: Path) -> Path:
        """Resolve internal link to absolute path"""
        # Handle anchor links
        if '#' in link_url:
            link_url = link_url.split('#')[0]
            if not link_url:  # Pure anchor link
                return source_file
        
        # Remove URL encoding
        link_url = unquote(link_url)
        
        # Convert relative path to absolute
        if link_url.startswith('/'):
            # Absolute path from repo root
            target_path = self.repo_root / link_url.lstrip('/')
        else:
            # Relative path from source file directory
            target_path = source_file.parent / link_url
        
        # Resolve the path
        try:
            return target_path.resolve()
        except (OSError, ValueError):
            return target_path
    
    def check_file_exists(self, file_path: Path) -> bool:
        """Check if file exists, considering common variations"""
        if file_path.exists():
            return True
        
        # Check if it's a directory link that should point to README.md
        if file_path.is_dir():
            readme_path = file_path / 'README.md'
            if readme_path.exists():
                return True
        
        # Check for case variations
        if file_path.parent.exists():
            actual_files = {f.name.lower(): f for f in file_path.parent.iterdir()}
            if file_path.name.lower() in actual_files:
                return True
        
        return False
    
    def suggest_fix(self, link_url: str, source_file: Path) -> str:
        """Suggest a fix for broken link"""
        # Check if this is a file that was moved during reorganization
        link_filename = Path(link_url).name
        if link_filename in self.file_moves:
            new_path = self.file_moves[link_filename]
            # Calculate relative path from source to new location
            rel_path = os.path.relpath(self.repo_root / new_path, source_file.parent)
            return rel_path
        
        # Look for similar files in the repository
        target_filename = Path(link_url).name
        for md_file in self.find_markdown_files():
            if md_file.name == target_filename:
                rel_path = os.path.relpath(md_file, source_file.parent)
                return rel_path
        
        return None
    
    def validate_links(self, fix_links: bool = False) -> Dict:
        """Validate all links in markdown files"""
        results = {
            'total_files': 0,
            'total_links': 0,
            'broken_links': [],
            'valid_links': [],
            'external_links': [],
            'files_with_issues': [],
            'fixes_applied': []
        }
        
        md_files = self.find_markdown_files()
        results['total_files'] = len(md_files)
        
        print(f"🔍 Scanning {len(md_files)} markdown files for links...")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    original_content = content
                
                links = self.extract_links(content, md_file)
                results['total_links'] += len(links)
                
                file_has_issues = False
                
                for link in links:
                    link_type = self.categorize_link(link['url'])
                    
                    if link_type == 'external' or link_type == 'mailto':
                        results['external_links'].append(link)
                    elif link_type == 'anchor':
                        results['valid_links'].append(link)
                    elif link_type == 'internal':
                        target_path = self.resolve_internal_link(link['url'], md_file)
                        
                        if self.check_file_exists(target_path):
                            results['valid_links'].append(link)
                        else:
                            # Broken link found
                            file_has_issues = True
                            broken_link = {
                                **link,
                                'target_path': str(target_path),
                                'relative_source': str(md_file.relative_to(self.repo_root))
                            }
                            
                            # Try to suggest a fix
                            suggested_fix = self.suggest_fix(link['url'], md_file)
                            if suggested_fix:
                                broken_link['suggested_fix'] = suggested_fix
                                
                                if fix_links:
                                    # Apply the fix
                                    old_link = f"[{link['text']}]({link['url']})"
                                    new_link = f"[{link['text']}]({suggested_fix})"
                                    content = content.replace(old_link, new_link)
                                    results['fixes_applied'].append({
                                        'file': str(md_file.relative_to(self.repo_root)),
                                        'old_url': link['url'],
                                        'new_url': suggested_fix,
                                        'line': link['line']
                                    })
                            
                            results['broken_links'].append(broken_link)
                
                if file_has_issues:
                    results['files_with_issues'].append(str(md_file.relative_to(self.repo_root)))
                
                # Write back fixed content if changes were made
                if fix_links and content != original_content:
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed links in {md_file.relative_to(self.repo_root)}")
                        
            except Exception as e:
                print(f"  ❌ Error processing {md_file}: {e}")
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive report"""
        report = []
        report.append("# PRSM Repository Link Validation Report")
        report.append(f"**Generated**: {os.popen('date').read().strip()}")
        report.append("")
        
        # Summary
        report.append("## 📊 Summary")
        report.append(f"- **Total files scanned**: {results['total_files']}")
        report.append(f"- **Total links found**: {results['total_links']}")
        report.append(f"- **Valid internal links**: {len(results['valid_links'])}")
        report.append(f"- **Broken internal links**: {len(results['broken_links'])}")
        report.append(f"- **External links**: {len(results['external_links'])}")
        report.append(f"- **Files with issues**: {len(results['files_with_issues'])}")
        
        if results['fixes_applied']:
            report.append(f"- **Fixes applied**: {len(results['fixes_applied'])}")
        
        report.append("")
        
        # Validation Score
        total_internal = len(results['valid_links']) + len(results['broken_links'])
        if total_internal > 0:
            score = (len(results['valid_links']) / total_internal) * 100
            report.append(f"## 🎯 Link Validation Score: {score:.1f}%")
            
            if score >= 95:
                report.append("✅ **EXCELLENT** - Repository links are well maintained")
            elif score >= 85:
                report.append("🟡 **GOOD** - Minor link issues to address")
            elif score >= 70:
                report.append("🟠 **NEEDS IMPROVEMENT** - Several broken links found")
            else:
                report.append("🔴 **CRITICAL** - Significant link issues require immediate attention")
        
        report.append("")
        
        # Broken Links Details
        if results['broken_links']:
            report.append("## 🔗 Broken Links")
            for broken in results['broken_links']:
                report.append(f"### {broken['relative_source']}:{broken['line']}")
                report.append(f"- **Link text**: {broken['text']}")
                report.append(f"- **Broken URL**: `{broken['url']}`")
                report.append(f"- **Target path**: `{broken['target_path']}`")
                if 'suggested_fix' in broken:
                    report.append(f"- **Suggested fix**: `{broken['suggested_fix']}`")
                report.append("")
        
        # Files with Issues
        if results['files_with_issues']:
            report.append("## 📄 Files Requiring Attention")
            for file in results['files_with_issues']:
                broken_count = len([b for b in results['broken_links'] if b['relative_source'] == file])
                report.append(f"- **{file}**: {broken_count} broken link(s)")
            report.append("")
        
        # Applied Fixes
        if results['fixes_applied']:
            report.append("## 🔧 Applied Fixes")
            for fix in results['fixes_applied']:
                report.append(f"### {fix['file']}:{fix['line']}")
                report.append(f"- **Old URL**: `{fix['old_url']}`")
                report.append(f"- **New URL**: `{fix['new_url']}`")
                report.append("")
        
        # Recommendations
        report.append("## 💡 Recommendations")
        if results['broken_links']:
            report.append("1. **Fix broken links** using the suggested fixes above")
            report.append("2. **Run validation again** after applying fixes")
            report.append("3. **Add link validation** to CI/CD pipeline")
        else:
            report.append("1. **Excellent link maintenance** - no issues found")
            report.append("2. **Consider adding** automated link validation to CI/CD")
        
        report.append("4. **Review external links periodically** for availability")
        report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Validate links in PRSM repository documentation')
    parser.add_argument('--fix-links', action='store_true', 
                       help='Automatically fix broken links where possible')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report without showing detailed output')
    parser.add_argument('--output', default='docs/LINK_VALIDATION_REPORT.md',
                       help='Output file for the report')
    
    args = parser.parse_args()
    
    # Determine repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    if not (repo_root / '.git').exists():
        print("❌ Error: Not in a git repository")
        sys.exit(1)
    
    validator = LinkValidator(str(repo_root))
    
    print("🔍 PRSM Repository Link Validation")
    print("=" * 40)
    
    # Run validation
    results = validator.validate_links(fix_links=args.fix_links)
    
    # Generate and save report
    report = validator.generate_report(results)
    report_path = repo_root / args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    if not args.report_only:
        print(f"\n📊 Results:")
        print(f"  Total files: {results['total_files']}")
        print(f"  Total links: {results['total_links']}")
        print(f"  Broken links: {len(results['broken_links'])}")
        print(f"  Valid links: {len(results['valid_links'])}")
        print(f"  External links: {len(results['external_links'])}")
        
        if results['fixes_applied']:
            print(f"  Fixes applied: {len(results['fixes_applied'])}")
        
        if results['broken_links']:
            print(f"\n❌ {len(results['broken_links'])} broken links found:")
            for broken in results['broken_links'][:10]:  # Show first 10
                print(f"  - {broken['relative_source']}:{broken['line']} -> {broken['url']}")
            if len(results['broken_links']) > 10:
                print(f"  ... and {len(results['broken_links']) - 10} more")
        else:
            print("\n✅ All internal links are valid!")
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Exit with error code if broken links found (for CI/CD)
    sys.exit(len(results['broken_links']))

if __name__ == '__main__':
    main()