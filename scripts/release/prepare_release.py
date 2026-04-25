#!/usr/bin/env python3
"""
PRSM Release Preparation Script

Prepares a new release by:
1. Running tests to ensure code quality
2. Bumping version in all relevant files
3. Generating changelog
4. Creating git tag
5. Optionally pushing changes

Usage:
    python prepare_release.py <version> [options]
    
Examples:
    python prepare_release.py 0.2.0
    python prepare_release.py 1.0.0 --skip-tests
    python prepare_release.py 0.1.1 --dry-run
"""

import argparse
import asyncio
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ReleaseType(Enum):
    """Type of release."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"


@dataclass
class ReleaseConfig:
    """Configuration for release preparation."""
    version: str
    dry_run: bool = False
    skip_tests: bool = False
    skip_changelog: bool = False
    push: bool = False
    branch: str = "main"
    
    # Computed values
    current_version: Optional[str] = None
    release_type: Optional[ReleaseType] = None
    tests_passed: bool = False
    changelog_generated: bool = False
    version_bumped: bool = False
    tag_created: bool = False
    errors: list[str] = field(default_factory=list)


class ReleasePreparer:
    """Prepares and validates releases."""
    
    def __init__(self, config: ReleaseConfig):
        self.config = config
        
    def get_current_version(self) -> Optional[str]:
        """Get current version from pyproject.toml."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        if not pyproject_path.exists():
            return None
            
        content = pyproject_path.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            return match.group(1)
        return None
    
    def validate_version(self) -> bool:
        """Validate version format and check it's different from current."""
        # Check format
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$'
        if not re.match(pattern, self.config.version):
            self.config.errors.append(
                f"Invalid version format: {self.config.version}. "
                "Expected: X.Y.Z or X.Y.Z-prerelease"
            )
            return False
        
        # Get current version
        self.config.current_version = self.get_current_version()
        if not self.config.current_version:
            self.config.errors.append("Could not determine current version")
            return False
        
        # Check version is different
        if self.config.version == self.config.current_version:
            self.config.errors.append(
                f"Version {self.config.version} is the same as current version"
            )
            return False
        
        # Determine release type
        current_parts = self.config.current_version.split('-')[0].split('.')
        new_parts = self.config.version.split('-')[0].split('.')
        
        if '-' in self.config.version:
            self.config.release_type = ReleaseType.PRERELEASE
        elif int(new_parts[0]) > int(current_parts[0]):
            self.config.release_type = ReleaseType.MAJOR
        elif int(new_parts[1]) > int(current_parts[1]):
            self.config.release_type = ReleaseType.MINOR
        else:
            self.config.release_type = ReleaseType.PATCH
        
        return True
    
    def check_git_status(self) -> bool:
        """Check git working directory is clean."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                self.config.errors.append(
                    "Git working directory is not clean. "
                    "Please commit or stash changes first."
                )
                return False
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.config.errors.append(f"Failed to check git status: {e}")
            return False
    
    def check_branch(self) -> bool:
        """Check we're on the correct branch."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            
            current_branch = result.stdout.strip()
            if current_branch != self.config.branch:
                self.config.errors.append(
                    f"Not on {self.config.branch} branch. "
                    f"Current branch: {current_branch}"
                )
                return False
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.config.errors.append(f"Failed to check branch: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run test suite."""
        if self.config.skip_tests:
            print("⏭️  Skipping tests (--skip-tests)")
            return True
        
        if self.config.dry_run:
            print("🔍 [DRY RUN] Would run tests")
            return True
        
        print("\n🧪 Running tests...")
        
        try:
            # Run pytest
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=short", "-x"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                print("❌ Tests failed!")
                print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                self.config.errors.append("Tests failed")
                return False
            
            print("✅ All tests passed")
            self.config.tests_passed = True
            return True
            
        except subprocess.TimeoutExpired:
            self.config.errors.append("Tests timed out after 10 minutes")
            return False
        except FileNotFoundError:
            print("⚠️  pytest not found, skipping tests")
            return True
        except Exception as e:
            self.config.errors.append(f"Failed to run tests: {e}")
            return False
    
    def bump_version(self) -> bool:
        """Bump version in all files."""
        if self.config.dry_run:
            print(f"🔍 [DRY RUN] Would bump version to {self.config.version}")
            return True
        
        print(f"\n📌 Bumping version to {self.config.version}...")
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "release" / "bump_version.py"),
                    self.config.version,
                    "--commit"
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            self.config.version_bumped = True
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Version bump failed: {e}")
            print(e.stdout)
            print(e.stderr)
            self.config.errors.append("Version bump failed")
            return False
    
    def generate_changelog(self) -> bool:
        """Generate changelog using git-cliff."""
        if self.config.skip_changelog:
            print("⏭️  Skipping changelog generation (--skip-changelog)")
            return True
        
        if self.config.dry_run:
            print("🔍 [DRY RUN] Would generate changelog")
            return True
        
        print("\n📝 Generating changelog...")
        
        try:
            # Check if git-cliff is installed
            result = subprocess.run(
                ["git-cliff", "--version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("⚠️  git-cliff not installed, skipping changelog")
                return True
            
            # Generate changelog
            result = subprocess.run(
                ["git-cliff", "--tag", f"v{self.config.version}", "-o", "CHANGELOG.md"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            
            print("✅ Changelog generated")
            self.config.changelog_generated = True
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Changelog generation failed: {e}")
            # Don't fail the release if changelog fails
            return True
        except FileNotFoundError:
            print("⚠️  git-cliff not found, skipping changelog")
            return True
    
    def create_tag(self) -> bool:
        """Create git tag for release."""
        if self.config.dry_run:
            print(f"🔍 [DRY RUN] Would create tag v{self.config.version}")
            return True
        
        print(f"\n🏷️  Creating tag v{self.config.version}...")
        
        try:
            tag_name = f"v{self.config.version}"
            result = subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ Created tag: {tag_name}")
            self.config.tag_created = True
            return True
            
        except subprocess.CalledProcessError as e:
            self.config.errors.append(f"Failed to create tag: {e}")
            return False
    
    def push_changes(self) -> bool:
        """Push changes and tag to remote."""
        if not self.config.push:
            print("\n💡 Run with --push to push changes to remote")
            return True
        
        if self.config.dry_run:
            print("🔍 [DRY RUN] Would push changes to remote")
            return True
        
        print("\n📤 Pushing changes to remote...")
        
        try:
            # Push branch
            result = subprocess.run(
                ["git", "push", "origin", self.config.branch],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ Pushed to {self.config.branch}")
            
            # Push tag
            result = subprocess.run(
                ["git", "push", "origin", f"v{self.config.version}"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ Pushed tag v{self.config.version}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.config.errors.append(f"Failed to push changes: {e}")
            return False
    
    def print_summary(self) -> None:
        """Print release summary."""
        print("\n" + "=" * 60)
        print("📋 RELEASE SUMMARY")
        print("=" * 60)
        print(f"  Version:        {self.config.version}")
        print(f"  Previous:       {self.config.current_version}")
        print(f"  Release Type:   {self.config.release_type.value if self.config.release_type else 'unknown'}")
        print(f"  Tests Passed:   {'✅' if self.config.tests_passed else '⏭️ skipped'}")
        print(f"  Version Bumped: {'✅' if self.config.version_bumped else '❌'}")
        print(f"  Changelog:      {'✅' if self.config.changelog_generated else '⏭️ skipped'}")
        print(f"  Tag Created:    {'✅' if self.config.tag_created else '❌'}")
        print("=" * 60)
        
        if self.config.errors:
            print("\n❌ ERRORS:")
            for error in self.config.errors:
                print(f"  - {error}")
        
        if self.config.dry_run:
            print("\n🔍 This was a DRY RUN - no changes were made")
        
        if not self.config.push and not self.config.dry_run:
            print("\n📝 Next steps:")
            print("  1. Review the changes")
            print("  2. Push to remote: git push origin main --tags")
            print("  3. Create GitHub release")
            print("  4. CI/CD will publish to PyPI and Docker Hub")
    
    def prepare(self) -> bool:
        """Run all preparation steps."""
        print("\n🚀 PRSM Release Preparation")
        print("=" * 60)
        print(f"  Target version: {self.config.version}")
        print(f"  Dry run:        {self.config.dry_run}")
        print("=" * 60)
        
        # Validation steps
        print("\n🔍 Validating...")
        
        if not self.validate_version():
            self._print_errors()
            return False
        
        print(f"  ✅ Version format valid")
        print(f"  ✅ Current version: {self.config.current_version}")
        print(f"  ✅ Release type: {self.config.release_type.value}")
        
        if not self.check_git_status():
            self._print_errors()
            return False
        
        print("  ✅ Git working directory clean")
        
        if not self.check_branch():
            self._print_errors()
            return False
        
        print(f"  ✅ On {self.config.branch} branch")
        
        # Preparation steps
        steps = [
            ("Tests", self.run_tests),
            ("Version Bump", self.bump_version),
            ("Changelog", self.generate_changelog),
            ("Tag Creation", self.create_tag),
            ("Push Changes", self.push_changes),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                self._print_errors()
                return False
        
        # Print summary
        self.print_summary()
        
        return len(self.config.errors) == 0
    
    def _print_errors(self) -> None:
        """Print error messages."""
        if self.config.errors:
            print("\n❌ Release preparation failed:")
            for error in self.config.errors:
                print(f"  - {error}")


def suggest_next_version(current: str, release_type: ReleaseType) -> str:
    """Suggest next version based on release type."""
    parts = current.split('-')[0].split('.')
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    
    if release_type == ReleaseType.MAJOR:
        return f"{major + 1}.0.0"
    elif release_type == ReleaseType.MINOR:
        return f"{major}.{minor + 1}.0"
    elif release_type == ReleaseType.PATCH:
        return f"{major}.{minor}.{patch + 1}"
    else:
        return f"{major}.{minor}.{patch + 1}-rc.1"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare a PRSM release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prepare_release.py 0.2.0
    python prepare_release.py 1.0.0 --dry-run
    python prepare_release.py 0.1.1 --skip-tests --push

Release Types:
    - MAJOR: Breaking changes (X.0.0)
    - MINOR: New features (0.X.0)
    - PATCH: Bug fixes (0.0.X)
    - PRERELEASE: Alpha/beta/rc versions (0.0.0-alpha.1)
        """
    )
    parser.add_argument(
        "version",
        help="Version to release (e.g., 0.2.0, 1.0.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-changelog",
        action="store_true",
        help="Skip changelog generation"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push changes and tag to remote"
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to release from (default: main)"
    )
    
    args = parser.parse_args()
    
    config = ReleaseConfig(
        version=args.version,
        dry_run=args.dry_run,
        skip_tests=args.skip_tests,
        skip_changelog=args.skip_changelog,
        push=args.push,
        branch=args.branch,
    )
    
    preparer = ReleasePreparer(config)
    success = preparer.prepare()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
