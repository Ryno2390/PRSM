#!/usr/bin/env python3
"""
PRSM Database Migration Helper Script

Provides convenient commands for common database migration operations.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0

def status():
    """Show current migration status"""
    print("📊 Current Migration Status")
    print("=" * 40)
    
    # Show current revision
    if not run_command(["alembic", "current"]):
        return False
    
    print("\n📋 Migration History")
    print("-" * 40)
    return run_command(["alembic", "history", "--verbose"])

def migrate():
    """Apply all pending migrations"""
    print("🚀 Applying Migrations")
    print("=" * 40)
    return run_command(["alembic", "upgrade", "head"])

def rollback(steps=1):
    """Rollback migrations"""
    print(f"⏪ Rolling Back {steps} Migration(s)")
    print("=" * 40)
    return run_command(["alembic", "downgrade", f"-{steps}"])

def create_migration(message, autogenerate=True):
    """Create a new migration"""
    print(f"📝 Creating Migration: {message}")
    print("=" * 40)
    
    cmd = ["alembic", "revision"]
    if autogenerate:
        cmd.append("--autogenerate")
    cmd.extend(["-m", message])
    
    return run_command(cmd)

def check():
    """Check for schema differences"""
    print("🔍 Checking Schema Differences")
    print("=" * 40)
    return run_command(["alembic", "check"])

def reset():
    """Reset database to empty state"""
    print("⚠️  Resetting Database to Empty State")
    print("=" * 40)
    
    response = input("This will remove ALL data. Are you sure? (yes/no): ")
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return True
    
    return run_command(["alembic", "downgrade", "base"])

def main():
    parser = argparse.ArgumentParser(description="PRSM Database Migration Helper")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Show current migration status")
    
    # Migrate command
    subparsers.add_parser("migrate", help="Apply all pending migrations")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("--steps", type=int, default=1, help="Number of migrations to rollback")
    
    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration description")
    create_parser.add_argument("--manual", action="store_true", help="Create manual migration (no autogenerate)")
    
    # Check command
    subparsers.add_parser("check", help="Check for schema differences")
    
    # Reset command
    subparsers.add_parser("reset", help="Reset database to empty state")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    success = True
    
    if args.command == "status":
        success = status()
    elif args.command == "migrate":
        success = migrate()
    elif args.command == "rollback":
        success = rollback(args.steps)
    elif args.command == "create":
        success = create_migration(args.message, not args.manual)
    elif args.command == "check":
        success = check()
    elif args.command == "reset":
        success = reset()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()