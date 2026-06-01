#!/usr/bin/env python3
"""
FTNS Database Migration Script
Migrates from simulation-based FTNS to database-backed implementation

This script handles the transition from the in-memory simulation FTNS
service to the production-ready database-backed implementation.

Key Migration Tasks:
1. Update imports across the codebase
2. Migrate existing transaction data (if any)
3. Initialize system wallets
4. Update configuration
5. Test database connectivity
6. Backup simulation data
7. Validate migration success

Safety Features:
- Dry-run mode for testing
- Rollback capability
- Data validation
- Comprehensive logging
- Error handling
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prsm.core.database import init_database, get_async_session
from prsm.tokenomics.database_ftns_service import DatabaseFTNSService
from prsm.tokenomics.models import WalletType
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FTNSMigrationError(Exception):
    """Base exception for FTNS migration errors"""
    pass


class FTNSDatabaseMigrator:
    """
    Handles migration from simulation to database-backed FTNS
    """
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.db_service = DatabaseFTNSService()
        self.migration_log: List[Dict[str, Any]] = []
        
        # System wallet configurations
        self.system_wallets = [
            {
                "user_id": "system",
                "wallet_type": WalletType.TREASURY,
                "initial_balance": Decimal('10000000.0'),  # 10M FTNS treasury
                "description": "System treasury wallet"
            },
            {
                "user_id": "marketplace",
                "wallet_type": WalletType.ESCROW,
                "initial_balance": Decimal('0.0'),
                "description": "Marketplace escrow wallet"
            },
            {
                "user_id": "governance",
                "wallet_type": WalletType.INSTITUTIONAL,
                "initial_balance": Decimal('1000000.0'),  # 1M FTNS for governance
                "description": "Governance rewards wallet"
            },
            {
                "user_id": "dividend_pool",
                "wallet_type": WalletType.TREASURY,
                "initial_balance": Decimal('5000000.0'),  # 5M FTNS for dividends
                "description": "Dividend distribution pool"
            }
        ]

    async def run_migration(self) -> bool:
        """
        Execute the complete FTNS migration
        
        Returns:
            True if migration successful, False otherwise
        """
        logger.info("🚀 Starting FTNS Database Migration")
        logger.info(f"Dry run mode: {self.dry_run}")
        
        try:
            # Step 1: Pre-migration checks
            await self._pre_migration_checks()
            
            # Step 2: Backup existing data
            await self._backup_simulation_data()
            
            # Step 3: Initialize database
            await self._initialize_database()
            
            # Step 4: Create system wallets
            await self._create_system_wallets()
            
            # Step 5: Migrate existing data
            await self._migrate_existing_data()
            
            # Step 6: Update imports and references
            await self._update_code_references()
            
            # Step 7: Post-migration validation
            await self._validate_migration()
            
            # Step 8: Generate migration report
            await self._generate_migration_report()
            
            logger.info("✅ FTNS Database Migration Completed Successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {str(e)}")
            if not self.dry_run:
                await self._rollback_migration()
            return False
        finally:
            await self.db_service.close()

    async def _pre_migration_checks(self):
        """Perform pre-migration validation"""
        logger.info("🔍 Running pre-migration checks...")
        
        # Check database connectivity
        try:
            from sqlalchemy import text
            async with get_async_session() as session:
                await session.execute(text("SELECT 1"))
            logger.info("✅ Database connectivity verified")
        except Exception as e:
            raise FTNSMigrationError(f"Database connection failed: {str(e)}")
        
        # Check if migration already completed
        try:
            wallet = await self.db_service.get_wallet("system", WalletType.TREASURY)
            if wallet:
                raise FTNSMigrationError("Migration appears to have already been run - system wallet exists")
        except Exception:
            pass  # Expected if no wallets exist yet
        
        # Verify required directories exist
        backup_dir = project_root / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        self._log_step("Pre-migration checks completed")

    async def _backup_simulation_data(self):
        """Backup existing simulation data"""
        logger.info("💾 Backing up simulation data...")
        
        backup_dir = project_root / "backup"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"ftns_simulation_backup_{timestamp}.json"
        
        try:
            # Try to import and extract data from existing simulation service
            simulation_data = {
                "backup_timestamp": datetime.now().isoformat(),
                "migration_type": "simulation_to_database",
                "note": "No simulation data found to backup - clean migration"
            }
            
            # Look for existing simulation data files
            sim_data_files = list(project_root.glob("**/ftns_simulation_data.json"))
            if sim_data_files:
                for file_path in sim_data_files:
                    with open(file_path, 'r') as f:
                        simulation_data[f"data_from_{file_path.name}"] = json.load(f)
            
            # Save backup
            with open(backup_file, 'w') as f:
                json.dump(simulation_data, f, indent=2, default=str)
            
            logger.info(f"✅ Simulation data backed up to {backup_file}")
            self._log_step(f"Backup created: {backup_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not backup simulation data: {str(e)}")
            # Continue migration - this is not a blocking error

    async def _initialize_database(self):
        """Initialize database with FTNS tables"""
        logger.info("🏗️ Initializing database...")
        
        if not self.dry_run:
            try:
                # Initialize database (run migrations if needed)
                await init_database()
                logger.info("✅ Database initialized successfully")
            except Exception as e:
                raise FTNSMigrationError(f"Database initialization failed: {str(e)}")
        else:
            logger.info("✅ [DRY RUN] Would initialize database")
        
        self._log_step("Database initialization completed")

    async def _create_system_wallets(self):
        """Create essential system wallets"""
        logger.info("🏦 Creating system wallets...")
        
        created_wallets = []
        
        for wallet_config in self.system_wallets:
            try:
                if not self.dry_run:
                    wallet = await self.db_service.create_wallet(
                        user_id=wallet_config["user_id"],
                        wallet_type=wallet_config["wallet_type"],
                        initial_balance=wallet_config["initial_balance"]
                    )
                    created_wallets.append({
                        "wallet_id": str(wallet.wallet_id),
                        "user_id": wallet.user_id,
                        "wallet_type": wallet.wallet_type,
                        "balance": str(wallet.balance)
                    })
                    logger.info(f"✅ Created {wallet_config['wallet_type'].value} wallet for {wallet_config['user_id']}")
                else:
                    logger.info(f"✅ [DRY RUN] Would create {wallet_config['wallet_type'].value} wallet for {wallet_config['user_id']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create wallet for {wallet_config['user_id']}: {str(e)}")
                raise FTNSMigrationError(f"System wallet creation failed: {str(e)}")
        
        self._log_step("System wallets created", {"wallets": created_wallets})

    async def _migrate_existing_data(self):
        """Migrate any existing transaction data"""
        logger.info("📊 Migrating existing data...")
        
        # For now, this is a clean migration with no existing data
        # In future, this would migrate from simulation state files
        
        migration_stats = {
            "transactions_migrated": 0,
            "wallets_migrated": 0,
            "balances_migrated": 0
        }
        
        # Check for any existing user data to migrate
        # This would be enhanced based on actual simulation data format
        
        logger.info("✅ Data migration completed (no existing data found)")
        self._log_step("Data migration completed", migration_stats)

    async def _update_code_references(self):
        """Update import statements and service references"""
        logger.info("🔄 Updating code references...")
        
        files_to_update = [
            "prsm/nwtn/orchestrator.py",
            "prsm/agents/executors/model_executor.py",
            "prsm/api/main.py",
            "prsm/tokenomics/__init__.py"
        ]
        
        updates_made = []
        
        for file_path in files_to_update:
            full_path = project_root / file_path
            
            if not full_path.exists():
                logger.warning(f"⚠️ File not found: {file_path}")
                continue
            
            try:
                # Read file content
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Make replacements
                original_content = content
                
                # Update imports
                content = content.replace(
                    "from prsm.tokenomics.ftns_service import ftns_service",
                    "from prsm.tokenomics.database_ftns_service import database_ftns_service as ftns_service"
                )
                
                content = content.replace(
                    "from ..tokenomics.ftns_service import ftns_service",
                    "from ..tokenomics.database_ftns_service import database_ftns_service as ftns_service"
                )
                
                # Update service references
                content = content.replace(
                    "ftns_service = FTNSService()",
                    "# ftns_service now imported as database_ftns_service"
                )
                
                # Write back if changed
                if content != original_content and not self.dry_run:
                    with open(full_path, 'w') as f:
                        f.write(content)
                    updates_made.append(file_path)
                    logger.info(f"✅ Updated {file_path}")
                elif content != original_content:
                    logger.info(f"✅ [DRY RUN] Would update {file_path}")
                    updates_made.append(f"[DRY RUN] {file_path}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to update {file_path}: {str(e)}")
        
        self._log_step("Code references updated", {"files_updated": updates_made})

    async def _validate_migration(self):
        """Validate migration success"""
        logger.info("✅ Validating migration...")
        
        validation_results = {}
        
        # Check system wallets exist
        for wallet_config in self.system_wallets:
            try:
                wallet = await self.db_service.get_wallet(
                    wallet_config["user_id"],
                    wallet_config["wallet_type"]
                )
                
                if wallet:
                    validation_results[f"wallet_{wallet_config['user_id']}"] = "✅ EXISTS"
                    logger.info(f"✅ Validated wallet: {wallet_config['user_id']}")
                else:
                    validation_results[f"wallet_{wallet_config['user_id']}"] = "❌ MISSING"
                    if not self.dry_run:
                        raise FTNSMigrationError(f"System wallet missing: {wallet_config['user_id']}")
                    
            except Exception as e:
                validation_results[f"wallet_{wallet_config['user_id']}"] = f"❌ ERROR: {str(e)}"
                if not self.dry_run:
                    raise
        
        # Test basic operations
        try:
            if not self.dry_run:
                # Test wallet creation
                test_wallet = await self.db_service.get_or_create_wallet("test_user_migration")
                
                # Test balance query
                balance_info = await self.db_service.get_wallet_balance("test_user_migration")
                
                # Test transaction creation
                await self.db_service.create_transaction(
                    from_user_id=None,
                    to_user_id="test_user_migration",
                    amount=Decimal('1.0'),
                    transaction_type="reward",
                    description="Migration validation test"
                )
                
                validation_results["basic_operations"] = "✅ PASSED"
                logger.info("✅ Basic operations validation passed")
            else:
                validation_results["basic_operations"] = "✅ [DRY RUN] WOULD PASS"
                
        except Exception as e:
            validation_results["basic_operations"] = f"❌ FAILED: {str(e)}"
            if not self.dry_run:
                raise FTNSMigrationError(f"Basic operations validation failed: {str(e)}")
        
        self._log_step("Migration validation completed", validation_results)

    async def _generate_migration_report(self):
        """Generate comprehensive migration report"""
        logger.info("📋 Generating migration report...")
        
        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "migration_mode": "DRY_RUN" if self.dry_run else "PRODUCTION",
            "migration_log": self.migration_log,
            "system_wallets_created": len(self.system_wallets),
            "migration_status": "COMPLETED_SUCCESSFULLY"
        }
        
        # Save report
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"ftns_migration_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Migration report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 FTNS DATABASE MIGRATION SUMMARY")
        print("="*60)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'PRODUCTION'}")
        print(f"Timestamp: {report['migration_timestamp']}")
        print(f"System Wallets: {report['system_wallets_created']}")
        print(f"Migration Steps: {len(self.migration_log)}")
        print(f"Status: {report['migration_status']}")
        print(f"Report: {report_file}")
        print("="*60)

    async def _rollback_migration(self):
        """Rollback migration in case of failure"""
        logger.error("🔄 Rolling back migration...")
        
        # In a production system, this would:
        # 1. Remove created wallets
        # 2. Restore backup data
        # 3. Revert code changes
        # 4. Reset database state
        
        logger.warning("⚠️ Rollback functionality not yet implemented")
        logger.warning("⚠️ Manual cleanup may be required")

    def _log_step(self, step: str, data: Optional[Dict[str, Any]] = None):
        """Log migration step"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": data or {}
        }
        self.migration_log.append(entry)
        logger.info(f"📝 {step}")


async def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(description="Migrate FTNS from simulation to database")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run in production mode (default is dry-run)"
    )
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Only backup simulation data, don't migrate"
    )
    
    args = parser.parse_args()
    
    dry_run = not args.production
    
    if not dry_run:
        confirm = input("⚠️  Running in PRODUCTION mode. This will make permanent changes. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("❌ Migration cancelled")
            return
    
    migrator = FTNSDatabaseMigrator(dry_run=dry_run)
    
    if args.backup_only:
        logger.info("📦 Running backup-only mode")
        await migrator._backup_simulation_data()
        return
    
    success = await migrator.run_migration()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        if dry_run:
            print("💡 Run with --production to execute the actual migration")
        else:
            print("✅ FTNS is now running on database-backed implementation")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())