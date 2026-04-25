#!/usr/bin/env python3
"""
PRSM Version Bumping Script

Bumps version in all relevant files:
- pyproject.toml
- setup.py
- prsm/__init__.py
- package.json (if exists)
- CHANGELOG.md (adds unreleased section)

Usage:
    python bump_version.py <version> [--dry-run]
    
Examples:
    python bump_version.py 0.2.0
    python bump_version.py 1.0.0 --dry-run
    python bump_version.py 0.1.1 --commit
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class VersionBumper:
    """Handles version bumping across multiple files."""
    
    def __init__(self, new_version: str, dry_run: bool = False):
        self.new_version = new_version
        self.dry_run = dry_run
        self.changes: list[tuple[str, str, str]] = []  # (file, old, new)
        
    def validate_version(self) -> bool:
        """Validate version format."""
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$'
        if not re.match(pattern, self.new_version):
            print(f"❌ Invalid version format: {self.new_version}")
            print("   Expected format: X.Y.Z or X.Y.Z-prerelease")
            return False
        return True
    
    def get_current_version(self) -> Optional[str]:
        """Get current version from pyproject.toml."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        if not pyproject_path.exists():
            print("❌ pyproject.toml not found")
            return None
            
        content = pyproject_path.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            return match.group(1)
        return None
    
    def bump_pyproject_toml(self) -> bool:
        """Bump version in pyproject.toml."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        if not pyproject_path.exists():
            print("⚠️  pyproject.toml not found, skipping")
            return True
            
        content = pyproject_path.read_text()
        old_version = self.get_current_version()
        
        if not old_version:
            print("❌ Could not find version in pyproject.toml")
            return False
            
        new_content = re.sub(
            r'^version\s*=\s*"[^"]+"',
            f'version = "{self.new_version}"',
            content,
            flags=re.MULTILINE
        )
        
        if new_content == content:
            print("❌ Failed to update version in pyproject.toml")
            return False
            
        self.changes.append(("pyproject.toml", old_version, self.new_version))
        
        if not self.dry_run:
            pyproject_path.write_text(new_content)
            print(f"✅ Updated pyproject.toml: {old_version} → {self.new_version}")
        else:
            print(f"🔍 [DRY RUN] Would update pyproject.toml: {old_version} → {self.new_version}")
            
        return True
    
    def bump_setup_py(self) -> bool:
        """Bump version in setup.py."""
        setup_path = PROJECT_ROOT / "setup.py"
        if not setup_path.exists():
            print("⚠️  setup.py not found, skipping")
            return True
            
        content = setup_path.read_text()
        old_version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
        
        if not old_version_match:
            print("⚠️  Could not find version in setup.py, skipping")
            return True
            
        old_version = old_version_match.group(1)
        new_content = re.sub(
            r'version\s*=\s*"[^"]+"',
            f'version="{self.new_version}"',
            content
        )
        
        self.changes.append(("setup.py", old_version, self.new_version))
        
        if not self.dry_run:
            setup_path.write_text(new_content)
            print(f"✅ Updated setup.py: {old_version} → {self.new_version}")
        else:
            print(f"🔍 [DRY RUN] Would update setup.py: {old_version} → {self.new_version}")
            
        return True
    
    def bump_init_py(self) -> bool:
        """Bump version in prsm/__init__.py."""
        init_path = PROJECT_ROOT / "prsm" / "__init__.py"
        if not init_path.exists():
            print("⚠️  prsm/__init__.py not found, skipping")
            return True
            
        content = init_path.read_text()
        old_version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        
        if not old_version_match:
            # Add __version__ if not present
            new_content = f'__version__ = "{self.new_version}"\n' + content
            self.changes.append(("prsm/__init__.py", "none", self.new_version))
            
            if not self.dry_run:
                init_path.write_text(new_content)
                print(f"✅ Added __version__ to prsm/__init__.py: {self.new_version}")
            else:
                print(f"🔍 [DRY RUN] Would add __version__ to prsm/__init__.py: {self.new_version}")
            return True
            
        old_version = old_version_match.group(1)
        new_content = re.sub(
            r'__version__\s*=\s*["\'][^"\']+["\']',
            f'__version__ = "{self.new_version}"',
            content
        )
        
        self.changes.append(("prsm/__init__.py", old_version, self.new_version))
        
        if not self.dry_run:
            init_path.write_text(new_content)
            print(f"✅ Updated prsm/__init__.py: {old_version} → {self.new_version}")
        else:
            print(f"🔍 [DRY RUN] Would update prsm/__init__.py: {old_version} → {self.new_version}")
            
        return True
    
    def bump_package_json(self) -> bool:
        """Bump version in package.json (for dashboard/frontend)."""
        package_json_path = PROJECT_ROOT / "package.json"
        if not package_json_path.exists():
            # Check for nested package.json files
            for subdir in ["ai-concierge", "PRSM_ui_mockup"]:
                nested_path = PROJECT_ROOT / subdir / "package.json"
                if nested_path.exists():
                    package_json_path = nested_path
                    break
        
        if not package_json_path.exists() or not package_json_path.is_file():
            print("⚠️  package.json not found, skipping")
            return True
            
        content = package_json_path.read_text()
        old_version_match = re.search(r'"version"\s*:\s*"([^"]+)"', content)
        
        if not old_version_match:
            print("⚠️  Could not find version in package.json, skipping")
            return True
            
        old_version = old_version_match.group(1)
        new_content = re.sub(
            r'"version"\s*:\s*"[^"]+"',
            f'"version": "{self.new_version}"',
            content
        )
        
        self.changes.append((str(package_json_path.relative_to(PROJECT_ROOT)), old_version, self.new_version))
        
        if not self.dry_run:
            package_json_path.write_text(new_content)
            print(f"✅ Updated {package_json_path.relative_to(PROJECT_ROOT)}: {old_version} → {self.new_version}")
        else:
            print(f"🔍 [DRY RUN] Would update {package_json_path.relative_to(PROJECT_ROOT)}: {old_version} → {self.new_version}")
            
        return True
    
    def update_changelog(self) -> bool:
        """Add unreleased section to CHANGELOG.md."""
        changelog_path = PROJECT_ROOT / "CHANGELOG.md"
        
        if not changelog_path.exists():
            print("⚠️  CHANGELOG.md not found, skipping")
            return True
        
        content = changelog_path.read_text()
        
        # Check if version already exists
        if f"## [{self.new_version}]" in content:
            print(f"⚠️  Version {self.new_version} already exists in CHANGELOG.md")
            return True
        
        # Create new section
        today = datetime.now().strftime("%Y-%m-%d")
        new_section = f"""
## [{self.new_version}] - {today}

### Added
- 

### Changed
- 

### Fixed
- 

### Security
- 

"""
        
        # Find insertion point (after header)
        lines = content.split('\n')
        insert_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('# Changelog') or line.startswith('# CHANGELOG'):
                # Skip header and any blank lines
                insert_idx = i + 1
                while insert_idx < len(lines) and lines[insert_idx].strip() == '':
                    insert_idx += 1
                break
        
        # Insert new section
        new_content = '\n'.join(lines[:insert_idx]) + new_section + '\n'.join(lines[insert_idx:])
        
        self.changes.append(("CHANGELOG.md", "added section", self.new_version))
        
        if not self.dry_run:
            changelog_path.write_text(new_content)
            print(f"✅ Added section for {self.new_version} to CHANGELOG.md")
        else:
            print(f"🔍 [DRY RUN] Would add section for {self.new_version} to CHANGELOG.md")
            
        return True
    
    def commit_changes(self) -> bool:
        """Commit version changes to git."""
        if self.dry_run:
            print("🔍 [DRY RUN] Would commit changes")
            return True
            
        try:
            # Stage changed files
            files_to_commit = [change[0] for change in self.changes]
            
            for file_path in files_to_commit:
                subprocess.run(
                    ["git", "add", file_path],
                    cwd=PROJECT_ROOT,
                    check=True,
                    capture_output=True
                )
            
            # Commit
            commit_message = f"chore: bump version to {self.new_version}"
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True
            )
            
            print(f"✅ Committed version bump: {self.new_version}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to commit changes: {e}")
            return False
    
    def create_tag(self) -> bool:
        """Create git tag for new version."""
        if self.dry_run:
            print(f"🔍 [DRY RUN] Would create tag v{self.new_version}")
            return True
            
        try:
            tag_name = f"v{self.new_version}"
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", f"Release {tag_name}"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True
            )
            
            print(f"✅ Created tag: {tag_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create tag: {e}")
            return False
    
    def run(self, commit: bool = False, tag: bool = False) -> bool:
        """Run all version bumping steps."""
        print(f"\n🚀 Bumping version to {self.new_version}")
        print("=" * 50)
        
        # Validate version format
        if not self.validate_version():
            return False
        
        # Get current version
        current = self.get_current_version()
        if current:
            print(f"📌 Current version: {current}")
        print(f"🎯 New version: {self.new_version}")
        print()
        
        # Run all bump operations
        operations = [
            self.bump_pyproject_toml,
            self.bump_setup_py,
            self.bump_init_py,
            self.bump_package_json,
            self.update_changelog,
        ]
        
        for operation in operations:
            if not operation():
                print(f"\n❌ Version bump failed at {operation.__name__}")
                return False
        
        # Commit if requested
        if commit:
            if not self.commit_changes():
                return False
        
        # Create tag if requested
        if tag:
            if not self.create_tag():
                return False
        
        print("\n" + "=" * 50)
        print("✅ Version bump complete!")
        print("\n📝 Files modified:")
        for file_path, old, new in self.changes:
            print(f"   - {file_path}: {old} → {new}")
        
        if not commit:
            print("\n💡 Tip: Run with --commit to commit changes")
        if not tag:
            print("💡 Tip: Run with --tag to create a git tag")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bump version in all PRSM project files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bump_version.py 0.2.0
    python bump_version.py 1.0.0 --dry-run
    python bump_version.py 0.1.1 --commit --tag
        """
    )
    parser.add_argument(
        "version",
        help="New version number (e.g., 0.2.0, 1.0.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit changes to git"
    )
    parser.add_argument(
        "--tag",
        action="store_true",
        help="Create a git tag for the new version"
    )
    
    args = parser.parse_args()
    
    bumper = VersionBumper(args.version, args.dry_run)
    success = bumper.run(commit=args.commit, tag=args.tag)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
