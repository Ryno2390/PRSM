#!/usr/bin/env python3
"""
Create Curated Good First Issues for PRSM

This script creates GitHub issues for the curated good first issues
identified in the CURATED_GOOD_FIRST_ISSUES.md document.

Usage:
    python scripts/create_curated_issues.py --dry-run  # Preview issues
    python scripts/create_curated_issues.py --create   # Actually create issues
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# GitHub API integration (optional - for automated issue creation)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class CuratedIssue:
    """Represents a curated good first issue"""
    
    def __init__(self, title: str, description: str, difficulty: int, 
                 category: str, time_estimate: str, impact: str,
                 skills: List[str], acceptance_criteria: List[str],
                 files_to_modify: List[str], learning_opportunities: List[str]):
        self.title = title
        self.description = description
        self.difficulty = difficulty
        self.category = category
        self.time_estimate = time_estimate
        self.impact = impact
        self.skills = skills
        self.acceptance_criteria = acceptance_criteria
        self.files_to_modify = files_to_modify
        self.learning_opportunities = learning_opportunities
        
        # Generate appropriate labels
        self.labels = self._generate_labels()
    
    def _generate_labels(self) -> List[str]:
        """Generate appropriate GitHub labels for this issue"""
        labels = ['good first issue', 'help wanted']
        
        # Difficulty labels
        if self.difficulty == 1:
            labels.append('beginner-friendly')
        elif self.difficulty == 2:
            labels.append('intermediate')
        elif self.difficulty >= 3:
            labels.append('advanced')
        
        # Category labels
        category_map = {
            'Documentation': 'documentation',
            'Testing': 'testing',
            'SDK Enhancement': 'enhancement',
            'Repository Management': 'infrastructure',
            'Community': 'community',
            'Research/Performance': 'research',
            'Research/Safety': 'research',
            'Infrastructure': 'infrastructure'
        }
        
        if self.category in category_map:
            labels.append(category_map[self.category])
        
        # Component labels based on files
        for file_path in self.files_to_modify:
            if 'prsm/core' in file_path:
                labels.append('component: core')
            elif 'sdks/python' in file_path:
                labels.append('python-sdk')
            elif 'sdks/javascript' in file_path:
                labels.append('javascript-sdk')
            elif 'sdks/go' in file_path:
                labels.append('go-sdk')
            elif 'docs/' in file_path:
                labels.append('documentation')
            elif 'tests/' in file_path:
                labels.append('testing')
        
        # Priority based on impact and difficulty
        if self.impact == '🔥 High' and self.difficulty <= 2:
            labels.append('priority: high')
        elif self.impact == '🔥 High':
            labels.append('priority: medium')
        else:
            labels.append('priority: low')
        
        return list(set(labels))  # Remove duplicates
    
    def to_github_issue(self) -> Dict[str, Any]:
        """Convert to GitHub issue format"""
        
        difficulty_stars = "⭐" * self.difficulty
        skills_str = ", ".join(self.skills)
        files_str = "\n".join(f"- `{file}`" for file in self.files_to_modify)
        criteria_str = "\n".join(f"- [ ] {criterion}" for criterion in self.acceptance_criteria)
        learning_str = "\n".join(f"- {opportunity}" for opportunity in self.learning_opportunities)
        
        body = f"""---
**Type**: {self.category}
**Difficulty**: {difficulty_stars} ({self.difficulty}/4 stars)
**Estimated Time**: {self.time_estimate}
**Skills Needed**: {skills_str}
**Impact**: {self.impact}
**Mentorship Available**: Yes ✅

## 📝 Description

{self.description}

## 🚀 Getting Started

### Prerequisites
- [ ] Read the [Contributing Guide](CONTRIBUTING.md)
- [ ] Set up development environment following [Development Setup](docs/DEVELOPMENT_SETUP.md)
- [ ] Join our [Discord community](https://discord.gg/prsm-ai) for support

### Files to Modify
{files_str}

## ✅ Acceptance Criteria

{criteria_str}

## 📚 Learning Opportunities

{learning_str}

## 🤝 Ready to Start?

**Comment below to claim this issue!** We'll assign it to you and provide guidance.

### New to Open Source?
This issue is perfect for newcomers! Check out:
- [How to contribute to open source](https://opensource.guide/how-to-contribute/)
- [PRSM Contributor Onboarding Guide](docs/CONTRIBUTOR_ONBOARDING.md)
- [Good First Issues Guide](docs/CURATED_GOOD_FIRST_ISSUES.md)

### Need Help?
- 💬 Ask questions in [GitHub Discussions](https://github.com/PRSM-AI/prsm/discussions)
- 🆘 Join our [Discord #help channel](https://discord.gg/prsm-ai)
- 📧 Email: contributors@prsm.ai

---

*This issue was created from our [Curated Good First Issues](docs/CURATED_GOOD_FIRST_ISSUES.md) guide.*
"""
        
        return {
            'title': self.title,
            'body': body,
            'labels': self.labels
        }


# Define the curated issues
CURATED_ISSUES = [
    CuratedIssue(
        title="Set up GitHub Issue Labels System",
        description="""Create a comprehensive GitHub label system to organize issues and improve contributor experience.

This task involves:
1. Researching label best practices for open source projects
2. Creating the label system in GitHub repository settings
3. Documenting the labeling strategy for maintainers
4. Setting up any available label automation

**Label Categories to Create:**
- **Priority**: `priority: critical`, `priority: high`, `priority: medium`, `priority: low`
- **Type**: `type: bug`, `type: feature`, `type: documentation`, `type: enhancement`
- **Component**: `component: core`, `component: sdk`, `component: ui`, `component: auth`
- **Difficulty**: `good first issue`, `help wanted`, `beginner-friendly`, `intermediate`, `advanced`
- **Status**: `needs triage`, `in progress`, `blocked`, `ready for review`

This is essential infrastructure that will help organize the entire project and improve the contributor experience!""",
        difficulty=1,
        category="Repository Management",
        time_estimate="2-3 hours",
        impact="🔥 High",
        skills=["GitHub", "Project Management", "Documentation"],
        acceptance_criteria=[
            "Complete label system is created in GitHub repository settings",
            "Labels follow consistent naming convention and color scheme",
            "Documentation explains when and how to use each label",
            "At least 10 existing issues are properly labeled as examples",
            "Label automation is configured (if available)"
        ],
        files_to_modify=["docs/LABELING_STRATEGY.md"],
        learning_opportunities=[
            "GitHub project management features and best practices",
            "Open source project organization strategies",
            "Community management and contributor experience design",
            "Documentation and process design skills"
        ]
    ),
    
    CuratedIssue(
        title="Complete API Documentation for Core Models Module",
        description="""The `prsm/core/models.py` file contains critical data models for PRSM but lacks comprehensive documentation. This module defines core classes like `UserInput`, `PRSMResponse`, and `ModelMetadata` that are essential for understanding how PRSM works.

**Current State**: The module has ~30 functions and classes with minimal or missing docstrings.

**What Needs Documentation**:
- `UserInput` class and its validation methods
- `PRSMResponse` class and response formatting
- `ModelMetadata` and performance tracking structures
- Error handling patterns and validation logic
- Usage examples for complex model interactions

This is perfect for learning PRSM's architecture while making a high-impact contribution that will help every future developer who works with PRSM!""",
        difficulty=1,
        category="Documentation",
        time_estimate="3-4 hours",
        impact="🔥 High",
        skills=["Python", "Documentation", "Code Reading", "API Design"],
        acceptance_criteria=[
            "All public classes have comprehensive docstrings following Google/NumPy style",
            "All methods include parameter types, return types, and descriptions",
            "Complex classes include usage examples in docstrings",
            "Docstrings explain the purpose and context of each component",
            "Documentation is beginner-friendly and technically accurate"
        ],
        files_to_modify=["prsm/core/models.py"],
        learning_opportunities=[
            "Understanding PRSM's core architecture and data flow",
            "Learning Python docstring conventions and best practices",
            "Developing code reading and comprehension skills",
            "Technical writing and API documentation skills"
        ]
    ),
    
    CuratedIssue(
        title="Add SDK Error Handling Documentation and Examples",
        description="""The Python SDK needs comprehensive error handling examples to help developers build robust applications with PRSM. Currently, while the SDK has good error types, there aren't enough practical examples showing how to handle different scenarios.

**Examples to Create**:
1. **Rate Limiting**: How to handle `PRSMRateLimitError` with exponential backoff
2. **Budget Management**: Graceful handling of `PRSMBudgetExceededError`
3. **Authentication**: Dealing with expired tokens and auth failures
4. **Network Issues**: Timeout handling and connection errors
5. **Validation Errors**: Input validation and error recovery

**Impact**: This will significantly improve the developer experience and help users build production-ready applications with proper error handling.""",
        difficulty=1,
        category="Documentation",
        time_estimate="2-3 hours",
        impact="🔥 High",
        skills=["Python", "Documentation", "Error Handling", "SDK Design"],
        acceptance_criteria=[
            "Create comprehensive `sdks/python/examples/error_handling.py` with all common scenarios",
            "Update Python SDK README with error handling section",
            "Document all SDK exception types with examples",
            "Include production-ready retry patterns and best practices",
            "Add debugging tips and troubleshooting guide"
        ],
        files_to_modify=[
            "sdks/python/examples/error_handling.py",
            "sdks/python/README.md",
            "sdks/python/docs/error_handling.md"
        ],
        learning_opportunities=[
            "Python exception handling patterns and best practices",
            "SDK design principles and user experience considerations",
            "Production application development practices",
            "Technical writing and developer documentation skills"
        ]
    ),
    
    CuratedIssue(
        title="Add Unit Tests for Core Model Classes",
        description="""The `prsm/core/models.py` module is critical to PRSM's functionality but lacks comprehensive unit tests. This creates risk for regressions and makes it harder to safely refactor or extend the codebase.

**Current State**: The module has ~30 classes and functions with minimal test coverage.

**Tests to Create**:
- `UserInput` validation, serialization, and edge cases
- `PRSMResponse` data integrity and format validation  
- Error handling for malformed data
- Performance model metadata accuracy
- Integration between different model classes

**Why This Matters**: Good tests are crucial for maintaining code quality, preventing bugs, and enabling confident refactoring. This is an excellent way to learn the codebase deeply while contributing to its long-term maintainability.""",
        difficulty=2,
        category="Testing",
        time_estimate="4-6 hours",
        impact="🔥 High",
        skills=["Python", "pytest", "Testing", "Code Analysis"],
        acceptance_criteria=[
            "Create `tests/core/test_models.py` with comprehensive test coverage",
            "Test all public methods, properties, and class interactions",
            "Include edge cases, error conditions, and boundary testing",
            "Achieve >90% test coverage for the models module",
            "All tests pass consistently and follow pytest best practices",
            "Include performance tests for critical operations"
        ],
        files_to_modify=["tests/core/test_models.py"],
        learning_opportunities=[
            "Python testing with pytest framework and best practices",
            "Test-driven development methodology and thinking",
            "Code coverage analysis and quality metrics",
            "Deep understanding of PRSM's core data models and validation",
            "Debugging skills and edge case analysis"
        ]
    ),
    
    CuratedIssue(
        title="Create Next.js Chat Interface Example",
        description="""Create a modern, production-ready Next.js example showing how to integrate PRSM in web applications. This will be a flagship example demonstrating best practices for web developers.

**Features to Implement**:
- Server-side API routes with PRSM integration
- Real-time chat interface with streaming responses
- Model selection and parameter controls
- Cost tracking and usage analytics
- Error handling with user-friendly feedback
- Responsive design with modern UI components
- Rate limiting and security considerations

**Impact**: This will be the go-to example for web developers wanting to integrate PRSM, significantly improving adoption in the web development community.""",
        difficulty=2,
        category="SDK Enhancement",
        time_estimate="4-6 hours",
        impact="🔥 High",
        skills=["JavaScript", "React", "Next.js", "API Design", "UI/UX"],
        acceptance_criteria=[
            "Complete Next.js application in `sdks/javascript/examples/nextjs-chat/`",
            "Working chat interface with real-time streaming responses",
            "Professional UI with proper error handling and loading states",
            "Cost tracking dashboard and usage analytics",
            "Comprehensive README with setup and deployment instructions",
            "Docker configuration for easy deployment"
        ],
        files_to_modify=[
            "sdks/javascript/examples/nextjs-chat/",
            "sdks/javascript/examples/nextjs-chat/README.md",
            "sdks/javascript/examples/nextjs-chat/Dockerfile"
        ],
        learning_opportunities=[
            "Next.js API routes and server-side integration patterns",
            "React development with hooks and modern patterns",
            "Real-time web applications with streaming responses",
            "Production web application architecture and deployment",
            "UI/UX design for AI applications"
        ]
    ),
    
    CuratedIssue(
        title="Convert Legacy Test Scripts to Pytest Format",
        description="""Several test files in the PRSM codebase use legacy testing formats and need conversion to proper pytest structure. This will improve test consistency, maintainability, and integration with our CI/CD pipeline.

**Files to Convert**:
- `tests/test_dashboard.py` - Currently uses print statements instead of assertions
- `tests/standalone_pq_test.py` - Needs conversion to pytest structure
- Other legacy test files identified in the codebase

**Why This Matters**: Consistent testing infrastructure is crucial for code quality. Legacy test formats make it harder to run tests, get proper reporting, and integrate with modern tooling.""",
        difficulty=2,
        category="Testing",
        time_estimate="3-5 hours",
        impact="🔥 Medium",
        skills=["Python", "pytest", "Testing", "Code Migration"],
        acceptance_criteria=[
            "All identified legacy tests converted to proper pytest format",
            "Print statements replaced with proper assertions and test reporting",
            "Tests integrated into main test suite and CI pipeline",
            "All converted tests pass consistently",
            "Follow pytest naming conventions and best practices",
            "Update test documentation and runner instructions"
        ],
        files_to_modify=[
            "tests/test_dashboard.py",
            "tests/standalone_pq_test.py",
            "tests/README.md"
        ],
        learning_opportunities=[
            "pytest framework features and conventions",
            "Test migration strategies and best practices",
            "CI/CD integration for automated testing",
            "Code quality improvement and technical debt reduction",
            "Testing infrastructure design and maintenance"
        ]
    )
]


def preview_issues():
    """Preview all issues that would be created"""
    print("🔍 Previewing Curated Good First Issues")
    print("=" * 60)
    
    total_time_min = 0
    total_time_max = 0
    
    for i, issue in enumerate(CURATED_ISSUES, 1):
        print(f"\n{i}. {issue.title}")
        print(f"   Category: {issue.category}")
        print(f"   Difficulty: {'⭐' * issue.difficulty} ({issue.difficulty}/4)")
        print(f"   Time: {issue.time_estimate}")
        print(f"   Impact: {issue.impact}")
        print(f"   Skills: {', '.join(issue.skills)}")
        print(f"   Labels: {', '.join(issue.labels)}")
        
        # Parse time estimate for totals
        if '-' in issue.time_estimate:
            times = issue.time_estimate.split('-')
            try:
                total_time_min += int(times[0])
                total_time_max += int(times[1].split()[0])
            except ValueError:
                pass
    
    print(f"\n📊 Summary:")
    print(f"   Total Issues: {len(CURATED_ISSUES)}")
    print(f"   Estimated Work: {total_time_min}-{total_time_max} hours")
    
    # Category breakdown
    categories = {}
    difficulties = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for issue in CURATED_ISSUES:
        categories[issue.category] = categories.get(issue.category, 0) + 1
        difficulties[issue.difficulty] += 1
    
    print(f"\n📋 By Category:")
    for category, count in categories.items():
        print(f"   {category}: {count}")
    
    print(f"\n⭐ By Difficulty:")
    for difficulty, count in difficulties.items():
        stars = '⭐' * difficulty
        print(f"   {stars} ({difficulty}/4): {count}")


def export_to_json(output_file: str):
    """Export issues to JSON format"""
    issues_data = {
        'total_issues': len(CURATED_ISSUES),
        'issues': [issue.to_github_issue() for issue in CURATED_ISSUES]
    }
    
    with open(output_file, 'w') as f:
        json.dump(issues_data, f, indent=2)
    
    print(f"✅ Issues exported to: {output_file}")


def create_issue_files():
    """Create individual markdown files for each issue"""
    output_dir = Path("scripts/curated_issues")
    output_dir.mkdir(exist_ok=True)
    
    for i, issue in enumerate(CURATED_ISSUES, 1):
        issue_data = issue.to_github_issue()
        
        # Create filename
        safe_title = issue.title.lower().replace(' ', '_').replace('/', '_')
        filename = f"{i:02d}_{safe_title[:50]}.md"
        
        # Write markdown file
        with open(output_dir / filename, 'w') as f:
            f.write(f"# {issue_data['title']}\n\n")
            f.write(f"**Labels**: {', '.join(issue_data['labels'])}\n\n")
            f.write(issue_data['body'])
    
    print(f"✅ Individual issue files created in: {output_dir}")


def create_github_issues(github_token: str, repo: str):
    """Create issues directly in GitHub (requires GitHub token)"""
    if not HAS_REQUESTS:
        print("❌ 'requests' library not available. Install with: pip install requests")
        return
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    base_url = f"https://api.github.com/repos/{repo}/issues"
    
    created_count = 0
    
    for i, issue in enumerate(CURATED_ISSUES, 1):
        issue_data = issue.to_github_issue()
        
        print(f"Creating issue {i}/{len(CURATED_ISSUES)}: {issue_data['title']}")
        
        response = requests.post(base_url, headers=headers, json=issue_data)
        
        if response.status_code == 201:
            created_issue = response.json()
            print(f"✅ Created: {created_issue['html_url']}")
            created_count += 1
        else:
            print(f"❌ Failed to create issue: {response.status_code} - {response.text}")
    
    print(f"\n🎉 Successfully created {created_count}/{len(CURATED_ISSUES)} issues!")


def main():
    parser = argparse.ArgumentParser(description="Create curated good first issues for PRSM")
    parser.add_argument("--dry-run", action="store_true", help="Preview issues without creating them")
    parser.add_argument("--export-json", help="Export issues to JSON file")
    parser.add_argument("--create-files", action="store_true", help="Create individual markdown files")
    parser.add_argument("--create-github", help="Create issues in GitHub (requires token)")
    parser.add_argument("--repo", default="PRSM-AI/prsm", help="GitHub repository (org/repo)")
    
    args = parser.parse_args()
    
    if args.dry_run or len(sys.argv) == 1:
        preview_issues()
    
    if args.export_json:
        export_to_json(args.export_json)
    
    if args.create_files:
        create_issue_files()
    
    if args.create_github:
        github_token = args.create_github
        if github_token == "env":
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                print("❌ GITHUB_TOKEN environment variable not set")
                return
        
        create_github_issues(github_token, args.repo)


if __name__ == "__main__":
    main()