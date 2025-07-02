#!/usr/bin/env python3
"""
PRSM Good First Issues Generator

This script analyzes the PRSM codebase and generates GitHub issues
suitable for new contributors, organized by difficulty and type.
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class GoodFirstIssue:
    """Represents a good first issue for new contributors"""
    title: str
    description: str
    difficulty: int  # 1-4 stars
    estimated_hours: str
    skills_needed: List[str]
    category: str
    component: str
    files_to_modify: List[str]
    acceptance_criteria: List[str]
    learning_opportunities: List[str]
    project_impact: str
    mentorship_available: bool = True
    related_todos: List[str] = None
    

class CodebaseAnalyzer:
    """Analyzes the PRSM codebase to identify good first issue opportunities"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.issues = []
        
    def find_todo_comments(self) -> List[Dict[str, Any]]:
        """Find all TODO comments in the codebase"""
        todos = []
        
        # Search Python files
        for py_file in self.repo_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.pytest_cache', 'venv', 'node_modules']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    if 'TODO' in line or 'FIXME' in line or 'HACK' in line:
                        todos.append({
                            'file': str(py_file.relative_to(self.repo_root)),
                            'line': line_num,
                            'content': line.strip(),
                            'type': 'TODO' if 'TODO' in line else 'FIXME' if 'FIXME' in line else 'HACK'
                        })
            except UnicodeDecodeError:
                continue
                
        return todos
    
    def find_missing_docstrings(self) -> List[Dict[str, Any]]:
        """Find Python modules/functions lacking documentation"""
        missing_docs = []
        
        for py_file in self.repo_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['.git', '__pycache__', 'test', 'venv']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse AST to find functions/classes without docstrings
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            # Check if has docstring
                            has_docstring = (
                                len(node.body) > 0 and
                                isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and
                                isinstance(node.body[0].value.value, str)
                            )
                            
                            if not has_docstring and not node.name.startswith('_'):
                                missing_docs.append({
                                    'file': str(py_file.relative_to(self.repo_root)),
                                    'type': type(node).__name__,
                                    'name': node.name,
                                    'line': node.lineno
                                })
                                
                except SyntaxError:
                    continue
                    
            except UnicodeDecodeError:
                continue
                
        return missing_docs
    
    def find_test_gaps(self) -> List[Dict[str, Any]]:
        """Find modules that lack comprehensive tests"""
        test_gaps = []
        
        # Get all Python modules
        modules = list(self.repo_root.rglob("prsm/**/*.py"))
        
        for module in modules:
            if any(skip in str(module) for skip in ['__pycache__', 'test']):
                continue
                
            # Look for corresponding test file
            relative_path = module.relative_to(self.repo_root)
            
            # Possible test file locations
            possible_tests = [
                self.repo_root / "tests" / f"test_{relative_path.name}",
                self.repo_root / "tests" / relative_path.parent.name / f"test_{relative_path.name}",
                module.parent / f"test_{module.name}",
            ]
            
            has_test = any(test_file.exists() for test_file in possible_tests)
            
            if not has_test and module.stat().st_size > 1000:  # Only for non-trivial files
                test_gaps.append({
                    'file': str(relative_path),
                    'size': module.stat().st_size,
                    'suggested_test_location': f"tests/test_{relative_path.name}"
                })
                
        return test_gaps
    
    def find_empty_modules(self) -> List[Dict[str, Any]]:
        """Find empty or nearly empty Python modules"""
        empty_modules = []
        
        for py_file in self.repo_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['.git', '__pycache__', 'test']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # Check if file is empty or has minimal content
                if len(content) < 50 or content.count('\n') < 3:
                    empty_modules.append({
                        'file': str(py_file.relative_to(self.repo_root)),
                        'size': len(content),
                        'content_preview': content[:100]
                    })
                    
            except UnicodeDecodeError:
                continue
                
        return empty_modules

    def generate_issues(self) -> List[GoodFirstIssue]:
        """Generate good first issues based on codebase analysis"""
        issues = []
        
        # Analyze codebase
        todos = self.find_todo_comments()
        missing_docs = self.find_missing_docstrings()
        test_gaps = self.find_test_gaps()
        empty_modules = self.find_empty_modules()
        
        # Generate documentation issues
        issues.extend(self._create_documentation_issues(missing_docs, empty_modules))
        
        # Generate testing issues
        issues.extend(self._create_testing_issues(test_gaps))
        
        # Generate TODO completion issues
        issues.extend(self._create_todo_issues(todos))
        
        # Generate repository setup issues
        issues.extend(self._create_repo_setup_issues())
        
        # Generate SDK enhancement issues
        issues.extend(self._create_sdk_issues())
        
        return issues
    
    def _create_documentation_issues(self, missing_docs: List[Dict], empty_modules: List[Dict]) -> List[GoodFirstIssue]:
        """Create documentation-related good first issues"""
        issues = []
        
        # Group missing docs by file
        docs_by_file = {}
        for doc in missing_docs:
            file = doc['file']
            if file not in docs_by_file:
                docs_by_file[file] = []
            docs_by_file[file].append(doc)
        
        # Create issues for files with multiple missing docs
        for file, docs in docs_by_file.items():
            if len(docs) >= 3:  # Only files with significant documentation gaps
                component = file.split('/')[1] if '/' in file else 'core'
                
                issues.append(GoodFirstIssue(
                    title=f"Add comprehensive documentation to {file}",
                    description=f"""
The file `{file}` is missing documentation for {len(docs)} functions/classes. This is a great opportunity for new contributors to learn about PRSM's architecture while improving documentation.

**Missing documentation for:**
{chr(10).join(f"- {doc['name']} ({doc['type']}) at line {doc['line']}" for doc in docs[:5])}
{'- ... and more' if len(docs) > 5 else ''}

This task involves:
1. Understanding what each function/class does
2. Writing clear, comprehensive docstrings
3. Adding usage examples where helpful
4. Following Python docstring conventions
                    """.strip(),
                    difficulty=1,
                    estimated_hours="2-4 hours",
                    skills_needed=["Python", "Documentation", "Code Reading"],
                    category="Documentation",
                    component=component,
                    files_to_modify=[file],
                    acceptance_criteria=[
                        "All public functions and classes have docstrings",
                        "Docstrings follow Google/NumPy style conventions",
                        "Complex functions include usage examples",
                        "Parameter and return types are documented",
                        "Documentation is clear and beginner-friendly"
                    ],
                    learning_opportunities=[
                        "Understanding PRSM architecture and components",
                        "Learning Python docstring best practices",
                        "Code reading and comprehension skills",
                        "Technical writing skills"
                    ],
                    project_impact="Improved developer experience and easier onboarding for new contributors",
                    related_todos=[doc['content'] for doc in docs if 'TODO' in doc.get('content', '')]
                ))
        
        # Handle empty modules
        for empty in empty_modules:
            if '__init__.py' in empty['file'] and empty['size'] < 20:
                component = empty['file'].split('/')[1] if '/' in empty['file'] else 'core'
                
                issues.append(GoodFirstIssue(
                    title=f"Add module documentation to {empty['file']}",
                    description=f"""
The module `{empty['file']}` is currently empty or has minimal content. This file should contain:

1. A module-level docstring explaining the purpose of the module
2. Proper imports and exports (if applicable)
3. Module-level constants or configuration (if needed)

**Current content:**
```python
{empty['content_preview']}
```

This is an excellent task for understanding Python module structure and PRSM's architecture.
                    """.strip(),
                    difficulty=1,
                    estimated_hours="1-2 hours",
                    skills_needed=["Python", "Documentation", "Module Design"],
                    category="Documentation",
                    component=component,
                    files_to_modify=[empty['file']],
                    acceptance_criteria=[
                        "Module has a clear docstring explaining its purpose",
                        "Proper imports are defined",
                        "Module follows PRSM architecture patterns",
                        "Code is well-formatted and follows style guide"
                    ],
                    learning_opportunities=[
                        "Python module structure and design",
                        "PRSM codebase organization",
                        "Import/export patterns",
                        "Code organization best practices"
                    ],
                    project_impact="Better code organization and developer understanding"
                ))
        
        return issues
    
    def _create_testing_issues(self, test_gaps: List[Dict]) -> List[GoodFirstIssue]:
        """Create testing-related good first issues"""
        issues = []
        
        # Sort by file size to prioritize important modules
        test_gaps_sorted = sorted(test_gaps, key=lambda x: x['size'], reverse=True)
        
        for gap in test_gaps_sorted[:5]:  # Top 5 most important modules
            component = gap['file'].split('/')[1] if '/' in gap['file'] else 'core'
            
            issues.append(GoodFirstIssue(
                title=f"Add unit tests for {gap['file']}",
                description=f"""
The module `{gap['file']}` currently lacks comprehensive unit tests. Adding tests is crucial for:

- Ensuring code reliability and preventing regressions
- Documenting expected behavior
- Making future refactoring safer
- Improving overall code quality

**Module details:**
- File size: {gap['size']} bytes
- Suggested test location: `{gap['suggested_test_location']}`

**What to test:**
1. All public functions and methods
2. Edge cases and error conditions
3. Integration between components
4. Performance characteristics (if applicable)

This is a great way to learn about the codebase while contributing to its reliability!
                """.strip(),
                difficulty=2,
                estimated_hours="3-6 hours",
                skills_needed=["Python", "pytest", "Testing", "Code Analysis"],
                category="Testing",
                component=component,
                files_to_modify=[gap['suggested_test_location']],
                acceptance_criteria=[
                    "Test file created with comprehensive test cases",
                    "All public functions are tested",
                    "Edge cases and error conditions are covered",
                    "Tests follow pytest conventions",
                    "Test coverage is > 80% for the module",
                    "All tests pass consistently"
                ],
                learning_opportunities=[
                    "Python testing with pytest",
                    "Test-driven development practices",
                    "Code coverage analysis",
                    "Understanding module behavior through testing",
                    "Debugging and error handling"
                ],
                project_impact="Increased code reliability and easier maintenance"
            ))
        
        return issues
    
    def _create_todo_issues(self, todos: List[Dict]) -> List[GoodFirstIssue]:
        """Create issues for completing TODO items"""
        issues = []
        
        # Group TODOs by complexity
        simple_todos = []
        complex_todos = []
        
        for todo in todos:
            content = todo['content'].lower()
            if any(keyword in content for keyword in ['placeholder', 'stub', 'basic', 'simple']):
                simple_todos.append(todo)
            else:
                complex_todos.append(todo)
        
        # Create issues for simple TODOs (group by file)
        todos_by_file = {}
        for todo in simple_todos:
            file = todo['file']
            if file not in todos_by_file:
                todos_by_file[file] = []
            todos_by_file[file].append(todo)
        
        for file, file_todos in todos_by_file.items():
            if len(file_todos) >= 2:  # Only files with multiple TODOs
                component = file.split('/')[1] if '/' in file else 'core'
                
                issues.append(GoodFirstIssue(
                    title=f"Complete TODO implementations in {file}",
                    description=f"""
The file `{file}` contains {len(file_todos)} TODO items that need implementation. These appear to be straightforward implementations based on the TODO comments.

**TODOs to complete:**
{chr(10).join(f"- Line {todo['line']}: {todo['content']}" for todo in file_todos[:3])}
{'- ... and more' if len(file_todos) > 3 else ''}

Most of these TODOs involve:
1. Replacing placeholder implementations
2. Adding basic validation or error handling
3. Implementing simple functionality
4. Adding proper logging or monitoring

This is a great way to contribute meaningful code while learning about PRSM's architecture!
                    """.strip(),
                    difficulty=2,
                    estimated_hours="2-5 hours",
                    skills_needed=["Python", "Code Implementation", "Problem Solving"],
                    category="Implementation",
                    component=component,
                    files_to_modify=[file],
                    acceptance_criteria=[
                        "All identified TODOs are implemented",
                        "Implementations follow existing code patterns",
                        "Code is well-tested and documented",
                        "No breaking changes to existing functionality",
                        "Code follows project style guidelines"
                    ],
                    learning_opportunities=[
                        "PRSM codebase patterns and conventions",
                        "Python implementation techniques",
                        "Error handling and validation",
                        "Code integration and testing"
                    ],
                    project_impact="Reduced technical debt and improved functionality",
                    related_todos=[todo['content'] for todo in file_todos]
                ))
        
        return issues
    
    def _create_repo_setup_issues(self) -> List[GoodFirstIssue]:
        """Create repository setup and infrastructure issues"""
        return [
            GoodFirstIssue(
                title="Set up GitHub issue labels for better project organization",
                description="""
PRSM needs a comprehensive GitHub label system to help organize issues and improve contributor experience. The labels should cover:

**Priority Labels:**
- `priority: critical`, `priority: high`, `priority: medium`, `priority: low`

**Type Labels:**
- `type: bug`, `type: feature`, `type: documentation`, `type: enhancement`

**Component Labels:**
- `component: core`, `component: sdk`, `component: ui`, `component: auth`, `component: marketplace`

**Difficulty Labels:**
- `good first issue`, `help wanted`, `advanced`, `expert`

**Status Labels:**
- `status: needs triage`, `status: in progress`, `status: blocked`, `status: ready for review`

This task involves:
1. Researching label best practices for open source projects
2. Creating the label system in GitHub
3. Documenting the labeling strategy
4. Setting up label automation (if possible)
                """.strip(),
                difficulty=1,
                estimated_hours="2-3 hours",
                skills_needed=["GitHub", "Project Management", "Documentation"],
                category="Repository Management",
                component="infrastructure",
                files_to_modify=["docs/LABELING_STRATEGY.md"],
                acceptance_criteria=[
                    "Complete label system is created in GitHub",
                    "Labels follow consistent naming and color scheme",
                    "Documentation explains when to use each label",
                    "Example issues are properly labeled",
                    "Label automation is set up (if applicable)"
                ],
                learning_opportunities=[
                    "GitHub project management features",
                    "Open source project organization",
                    "Community management best practices",
                    "Documentation and process design"
                ],
                project_impact="Better issue organization and improved contributor experience"
            ),
            
            GoodFirstIssue(
                title="Create GitHub Discussions setup for community engagement",
                description="""
Enable and configure GitHub Discussions to create a space for community questions, ideas, and collaboration beyond just issue tracking.

**Discussion Categories to Create:**
- **💡 Ideas** - Feature ideas and brainstorming
- **❓ Q&A** - Questions about using PRSM
- **📢 Announcements** - Project updates and news  
- **🗣️ General** - General discussions about PRSM
- **🛠️ Development** - Technical discussions for contributors
- **📚 Show and Tell** - Community projects and use cases

**Tasks:**
1. Enable GitHub Discussions on the repository
2. Set up the category structure
3. Create welcome post with guidelines
4. Create initial discussion topics
5. Update README to mention Discussions
6. Set up moderation guidelines

This helps build a stronger community around PRSM!
                """.strip(),
                difficulty=1,
                estimated_hours="1-2 hours",
                skills_needed=["GitHub", "Community Management", "Documentation"],
                category="Community",
                component="infrastructure", 
                files_to_modify=["README.md", "docs/COMMUNITY_GUIDELINES.md"],
                acceptance_criteria=[
                    "GitHub Discussions is enabled with proper categories",
                    "Welcome post explains how to use Discussions",
                    "Initial discussion topics are created",
                    "README mentions Discussions as support channel",
                    "Moderation guidelines are documented"
                ],
                learning_opportunities=[
                    "GitHub Discussions features and management",
                    "Community building strategies",
                    "Online moderation best practices",
                    "Documentation and communication skills"
                ],
                project_impact="Enhanced community engagement and support channel"
            )
        ]
    
    def _create_sdk_issues(self) -> List[GoodFirstIssue]:
        """Create SDK enhancement issues"""
        return [
            GoodFirstIssue(
                title="Add error handling examples to Python SDK documentation",
                description="""
The Python SDK needs comprehensive error handling examples to help developers build robust applications with PRSM.

**Examples to Add:**
1. **Rate Limiting Handling**
   ```python
   try:
       result = await client.infer(prompt="Hello")
   except PRSMRateLimitError as e:
       # Wait and retry logic
       time.sleep(e.retry_after)
       result = await client.infer(prompt="Hello")
   ```

2. **Budget Management**
   ```python
   try:
       result = await client.infer(prompt="Hello")
   except PRSMBudgetExceededError as e:
       print(f"Budget exceeded. Remaining: ${e.remaining_budget}")
       # Handle gracefully
   ```

3. **Authentication Errors**
4. **Network/Timeout Errors**
5. **Validation Errors**

**Files to Update:**
- `sdks/python/README.md`
- `sdks/python/examples/error_handling.py`
- `sdks/python/docs/error_handling.md`

This helps developers build production-ready applications!
                """.strip(),
                difficulty=1,
                estimated_hours="2-3 hours",
                skills_needed=["Python", "Documentation", "Error Handling"],
                category="Documentation",
                component="python-sdk",
                files_to_modify=[
                    "sdks/python/README.md", 
                    "sdks/python/examples/error_handling.py",
                    "sdks/python/docs/error_handling.md"
                ],
                acceptance_criteria=[
                    "Comprehensive error handling examples are added",
                    "All common error scenarios are covered",
                    "Examples show best practices for production apps",
                    "Code examples are tested and work correctly",
                    "Documentation is clear and beginner-friendly"
                ],
                learning_opportunities=[
                    "Python exception handling patterns",
                    "SDK design and documentation",
                    "Production application best practices",
                    "Technical writing skills"
                ],
                project_impact="Better developer experience and more robust applications"
            )
        ]


def export_issues_to_json(issues: List[GoodFirstIssue], output_file: Path):
    """Export issues to JSON format for processing"""
    issues_data = {
        "generated_at": datetime.now().isoformat(),
        "total_issues": len(issues),
        "issues": [asdict(issue) for issue in issues]
    }
    
    with open(output_file, 'w') as f:
        json.dump(issues_data, f, indent=2)


def generate_issue_markdown(issue: GoodFirstIssue) -> str:
    """Generate GitHub issue markdown for a good first issue"""
    
    difficulty_stars = "⭐" * issue.difficulty
    skills_str = ", ".join(issue.skills_needed)
    files_str = ", ".join(issue.files_to_modify)
    criteria_str = "\n".join(f"- [ ] {criterion}" for criterion in issue.acceptance_criteria)
    learning_str = "\n".join(f"- {opportunity}" for opportunity in issue.learning_opportunities)
    
    markdown = f"""---
**Type**: {issue.category}
**Difficulty**: {difficulty_stars} ({issue.difficulty}/4 stars)
**Estimated Time**: {issue.estimated_hours}
**Skills Needed**: {skills_str}
**Mentorship Available**: {'Yes ✅' if issue.mentorship_available else 'No ❌'}

## 📝 Description

{issue.description}

## 🎯 Expected Outcome

{issue.project_impact}

## 🚀 Getting Started

### Prerequisites
- [ ] Read the [Contributing Guide](../../CONTRIBUTING.md)
- [ ] Set up development environment
- [ ] Understand the PRSM architecture basics

### Files to Modify
{files_str}

## ✅ Acceptance Criteria

{criteria_str}

## 📚 Learning Opportunities

{learning_str}

## 🤝 Ready to Start?

Comment below to let us know you're working on this issue! We're here to help guide you through the process.

### New to Open Source?
- [How to contribute to open source](https://opensource.guide/how-to-contribute/)
- [PRSM Contributing Guide](../../CONTRIBUTING.md)
- [Development Setup Guide](../../docs/DEVELOPMENT_SETUP.md)
"""

    if issue.related_todos:
        todos_str = "\n".join(f"- {todo}" for todo in issue.related_todos)
        markdown += f"\n## 🔗 Related TODOs\n\n{todos_str}\n"

    return markdown


def main():
    """Main function to generate good first issues"""
    repo_root = Path(__file__).parent.parent
    
    print("🔍 Analyzing PRSM codebase for good first issue opportunities...")
    
    analyzer = CodebaseAnalyzer(repo_root)
    issues = analyzer.generate_issues()
    
    print(f"✅ Generated {len(issues)} good first issues!")
    
    # Export to JSON
    output_dir = repo_root / "scripts" / "generated_issues"
    output_dir.mkdir(exist_ok=True)
    
    export_issues_to_json(issues, output_dir / "good_first_issues.json")
    
    # Generate individual markdown files
    for i, issue in enumerate(issues):
        issue_file = output_dir / f"issue_{i+1:02d}_{issue.category.lower().replace(' ', '_')}.md"
        
        with open(issue_file, 'w') as f:
            f.write(generate_issue_markdown(issue))
    
    print(f"📁 Issues exported to: {output_dir}")
    print("\n📊 Issue Summary:")
    
    # Summary by category
    categories = {}
    for issue in issues:
        if issue.category not in categories:
            categories[issue.category] = []
        categories[issue.category].append(issue)
    
    for category, cat_issues in categories.items():
        print(f"  {category}: {len(cat_issues)} issues")
        
    # Summary by difficulty
    difficulties = {1: 0, 2: 0, 3: 0, 4: 0}
    for issue in issues:
        difficulties[issue.difficulty] += 1
        
    print("\n📈 Difficulty Distribution:")
    for difficulty, count in difficulties.items():
        stars = "⭐" * difficulty
        print(f"  {stars} ({difficulty}/4): {count} issues")
    
    print(f"\n🎯 Total estimated work: {sum(int(issue.estimated_hours.split('-')[0]) for issue in issues)}-{sum(int(issue.estimated_hours.split('-')[1].split()[0]) for issue in issues)} hours")
    
    print("\n💡 Next steps:")
    print("1. Review generated issues in scripts/generated_issues/")
    print("2. Create GitHub issues using the generated markdown")
    print("3. Apply appropriate labels to each issue")
    print("4. Set up mentorship assignments")


if __name__ == "__main__":
    main()