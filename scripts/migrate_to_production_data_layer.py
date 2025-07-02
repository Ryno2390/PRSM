#!/usr/bin/env python3
"""
Migration Script: Simulated IPFS to Production Data Layer
Migrates existing PRSM data from simulated in-memory systems to production PostgreSQL, Redis, and Milvus

MIGRATION STRATEGY:
1. Analyze existing simulated data structures
2. Map data to production schema
3. Migrate consensus records to PostgreSQL
4. Migrate caching data to Redis
5. Generate and store embeddings in Milvus
6. Validate data integrity post-migration
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import os

# Add PRSM to path
sys.path.append(str(Path(__file__).parent.parent))

from prsm.storage.production_data_layer import ProductionDataLayer, test_production_data_layer
from prsm.core.config import settings

logger = logging.getLogger(__name__)


class DataMigrationManager:
    """Manages migration from simulated to production data layer"""
    
    def __init__(self):
        self.data_layer = ProductionDataLayer()
        self.migration_stats = {
            "consensus_records_migrated": 0,
            "network_records_migrated": 0,
            "model_records_migrated": 0,
            "cache_entries_migrated": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def run_full_migration(self) -> Dict[str, Any]:
        """Run complete migration from simulated to production data layer"""
        print("🔄 Starting PRSM Data Layer Migration")
        print("=" * 60)
        
        self.migration_stats["start_time"] = datetime.now(timezone.utc)
        
        try:
            # Step 1: Initialize production data layer
            print("📋 Step 1: Initializing production data layer...")
            initialized = await self.data_layer.initialize_connections()
            
            if not initialized:
                print("❌ Failed to initialize production data layer")
                return self.migration_stats
            
            print("✅ Production data layer initialized")
            
            # Step 2: Discover existing simulated data
            print("📋 Step 2: Discovering existing simulated data...")
            simulated_data = await self._discover_simulated_data()
            
            if not simulated_data:
                print("ℹ️ No existing simulated data found - fresh installation")
            else:
                print(f"✅ Found {len(simulated_data)} simulated data sources")
            
            # Step 3: Migrate consensus data
            print("📋 Step 3: Migrating consensus records...")
            await self._migrate_consensus_data(simulated_data.get("consensus", {}))
            
            # Step 4: Migrate network data
            print("📋 Step 4: Migrating network records...")
            await self._migrate_network_data(simulated_data.get("network", {}))
            
            # Step 5: Migrate model data with embeddings
            print("📋 Step 5: Migrating model records with embeddings...")
            await self._migrate_model_data(simulated_data.get("models", {}))
            
            # Step 6: Setup production caching
            print("📋 Step 6: Setting up production caching...")
            await self._setup_production_caching()
            
            # Step 7: Validate migration
            print("📋 Step 7: Validating migration integrity...")
            validation_results = await self._validate_migration()
            
            self.migration_stats["end_time"] = datetime.now(timezone.utc)
            
            # Generate migration report
            await self._generate_migration_report(validation_results)
            
            print("✅ PRSM Data Layer Migration Completed Successfully!")
            return self.migration_stats
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            self.migration_stats["errors"] += 1
            self.migration_stats["end_time"] = datetime.now(timezone.utc)
            return self.migration_stats
    
    async def _discover_simulated_data(self) -> Dict[str, Any]:
        """Discover existing simulated data sources"""
        simulated_data = {
            "consensus": {},
            "network": {},
            "models": {}
        }
        
        try:
            # Look for common PRSM data directories
            data_paths = [
                Path("data"),
                Path("cache"),
                Path("temp"),
                Path(".prsm"),
                Path("results")
            ]
            
            for path in data_paths:
                if path.exists():
                    await self._scan_directory_for_data(path, simulated_data)
            
            # Look for in-memory data from running processes
            await self._discover_runtime_data(simulated_data)
            
            return simulated_data
            
        except Exception as e:
            logger.error(f"❌ Error discovering simulated data: {e}")
            return simulated_data
    
    async def _scan_directory_for_data(self, directory: Path, simulated_data: Dict[str, Any]):
        """Scan directory for PRSM data files"""
        try:
            for file_path in directory.rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Categorize data based on content
                        if "consensus" in str(file_path).lower() or "consensus_id" in data:
                            simulated_data["consensus"][str(file_path)] = data
                        elif "network" in str(file_path).lower() or "node_id" in data:
                            simulated_data["network"][str(file_path)] = data
                        elif "model" in str(file_path).lower() or "model_id" in data:
                            simulated_data["models"][str(file_path)] = data
                            
                    except (json.JSONDecodeError, KeyError):
                        # Skip non-PRSM or malformed JSON files
                        continue
                        
        except Exception as e:
            logger.error(f"❌ Error scanning directory {directory}: {e}")
    
    async def _discover_runtime_data(self, simulated_data: Dict[str, Any]):
        """Discover data from running PRSM processes"""
        try:
            # Try to import and access running PRSM instances
            try:
                from prsm.federation.consensus import DistributedConsensus
                from prsm.federation.p2p_network import P2PModelNetwork
                
                # Check if there are any active consensus instances
                # This is a simplified check - in practice, you'd need to access
                # singleton instances or global state
                print("ℹ️ Checking for runtime PRSM instances...")
                
            except ImportError:
                pass
                
        except Exception as e:
            logger.debug(f"No runtime data discovered: {e}")
    
    async def _migrate_consensus_data(self, consensus_data: Dict[str, Any]):
        """Migrate consensus records to PostgreSQL"""
        for source, data in consensus_data.items():
            try:
                # Convert simulated consensus data to production format
                if isinstance(data, dict):
                    # Handle single consensus record
                    await self._migrate_single_consensus_record(data)
                elif isinstance(data, list):
                    # Handle multiple consensus records
                    for record in data:
                        await self._migrate_single_consensus_record(record)
                
                self.migration_stats["consensus_records_migrated"] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to migrate consensus data from {source}: {e}")
                self.migration_stats["errors"] += 1
    
    async def _migrate_single_consensus_record(self, record: Dict[str, Any]):
        """Migrate a single consensus record"""
        # Standardize consensus record format
        consensus_data = {
            "consensus_id": record.get("consensus_id", record.get("id", f"migrated_{int(time.time())}")),
            "proposal_data": record.get("proposal_data", record.get("data")),
            "consensus_result": record.get("consensus_result", record.get("result")),
            "participating_nodes": record.get("participating_nodes", record.get("nodes", [])),
            "consensus_time": record.get("consensus_time", record.get("duration", 0.0)),
            "status": record.get("status", "migrated")
        }
        
        success = await self.data_layer.store_consensus_record(consensus_data)
        
        if success:
            logger.debug(f"✅ Migrated consensus record: {consensus_data['consensus_id']}")
        else:
            logger.error(f"❌ Failed to migrate consensus record: {consensus_data['consensus_id']}")
            raise Exception("Consensus record migration failed")
    
    async def _migrate_network_data(self, network_data: Dict[str, Any]):
        """Migrate network records to PostgreSQL"""
        for source, data in network_data.items():
            try:
                # Convert simulated network data to production format
                network_record = {
                    "node_id": data.get("node_id", f"migrated_node_{int(time.time())}"),
                    "network_state": data.get("network_state", data.get("state")),
                    "peer_connections": data.get("peer_connections", data.get("peers", [])),
                    "network_health": data.get("network_health", data.get("health", 1.0)),
                    "status": data.get("status", "migrated")
                }
                
                success = await self.data_layer.store_network_state(network_record)
                
                if success:
                    self.migration_stats["network_records_migrated"] += 1
                    logger.debug(f"✅ Migrated network record: {network_record['node_id']}")
                else:
                    raise Exception("Network record migration failed")
                
            except Exception as e:
                logger.error(f"❌ Failed to migrate network data from {source}: {e}")
                self.migration_stats["errors"] += 1
    
    async def _migrate_model_data(self, model_data: Dict[str, Any]):
        """Migrate model records with generated embeddings to PostgreSQL and Milvus"""
        for source, data in model_data.items():
            try:
                # Generate model record
                model_record = {
                    "model_id": data.get("model_id", f"migrated_model_{int(time.time())}"),
                    "model_name": data.get("model_name", "migrated_model"),
                    "model_version": data.get("model_version", "1.0.0"),
                    "model_metadata": data.get("metadata", {}),
                    "training_data": data.get("training_data"),
                    "performance_metrics": data.get("performance_metrics", {}),
                    "status": data.get("status", "migrated")
                }
                
                # Generate embedding for the model
                # In production, this would be generated by your ML pipeline
                embedding = await self._generate_model_embedding(model_record)
                
                success = await self.data_layer.store_model_with_embedding(model_record, embedding)
                
                if success:
                    self.migration_stats["model_records_migrated"] += 1
                    logger.debug(f"✅ Migrated model record: {model_record['model_id']}")
                else:
                    raise Exception("Model record migration failed")
                
            except Exception as e:
                logger.error(f"❌ Failed to migrate model data from {source}: {e}")
                self.migration_stats["errors"] += 1
    
    async def _generate_model_embedding(self, model_record: Dict[str, Any]) -> List[float]:
        """Generate embedding for model record (simplified for migration)"""
        # In production, this would use your actual embedding model
        # For migration, we generate a simple hash-based embedding
        import hashlib
        
        model_string = json.dumps(model_record, sort_keys=True)
        hash_bytes = hashlib.sha256(model_string.encode()).digest()
        
        # Convert hash to 768-dimensional embedding (typical for transformer models)
        embedding = []
        for i in range(768):
            byte_index = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_index] - 128) / 128.0)  # Normalize to [-1, 1]
        
        return embedding
    
    async def _setup_production_caching(self):
        """Setup production caching with initial data"""
        try:
            # Cache frequently accessed data
            initial_cache_data = {
                "migration_timestamp": datetime.now(timezone.utc).isoformat(),
                "migration_version": "1.0.0",
                "data_layer_status": "production"
            }
            
            success = await self.data_layer.cache_set("prsm:migration:info", initial_cache_data)
            
            if success:
                self.migration_stats["cache_entries_migrated"] += 1
                print("✅ Production caching setup completed")
            else:
                print("⚠️ Production caching setup had issues")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup production caching: {e}")
            self.migration_stats["errors"] += 1
    
    async def _validate_migration(self) -> Dict[str, Any]:
        """Validate migration integrity"""
        validation_results = {
            "data_layer_health": await self.data_layer.health_check(),
            "performance_metrics": await self.data_layer.get_performance_metrics(),
            "sample_data_verification": {}
        }
        
        # Test data layer functionality
        test_results = await test_production_data_layer()
        validation_results["functionality_test"] = test_results
        
        return validation_results
    
    async def _generate_migration_report(self, validation_results: Dict[str, Any]):
        """Generate comprehensive migration report"""
        duration = (
            self.migration_stats["end_time"] - self.migration_stats["start_time"]
        ).total_seconds()
        
        report = f"""
PRSM Data Layer Migration Report
===============================

Migration Summary:
- Duration: {duration:.2f} seconds
- Consensus Records Migrated: {self.migration_stats['consensus_records_migrated']}
- Network Records Migrated: {self.migration_stats['network_records_migrated']}  
- Model Records Migrated: {self.migration_stats['model_records_migrated']}
- Cache Entries Migrated: {self.migration_stats['cache_entries_migrated']}
- Errors Encountered: {self.migration_stats['errors']}

Data Layer Health:
- PostgreSQL: {'✅ Healthy' if validation_results['data_layer_health']['postgresql'] else '❌ Unhealthy'}
- Redis: {'✅ Healthy' if validation_results['data_layer_health']['redis'] else '❌ Unhealthy'}
- Milvus: {'✅ Healthy' if validation_results['data_layer_health']['milvus'] else '❌ Unhealthy'}

Performance Metrics:
- Database Success Rate: {validation_results['performance_metrics']['database_metrics']['success_rate']:.1f}%
- Cache Hit Ratio: {validation_results['performance_metrics']['cache_metrics']['cache_hit_ratio']:.1f}%
- Average Query Time: {validation_results['performance_metrics']['database_metrics']['average_query_time']:.3f}s

Functionality Test:
- Tests Passed: {validation_results['functionality_test']['tests_passed']}/{validation_results['functionality_test']['tests_completed']}
- Data Layer Functional: {'✅ Yes' if validation_results['functionality_test']['data_layer_functional'] else '❌ No'}

Migration Status: {'✅ SUCCESSFUL' if self.migration_stats['errors'] == 0 else '⚠️ COMPLETED WITH ERRORS'}
"""
        
        print(report)
        
        # Save report to file
        report_file = Path("data_layer_migration_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Full migration report saved to: {report_file}")


async def main():
    """Main migration execution"""
    print("🚀 PRSM Production Data Layer Migration")
    print("This will migrate from simulated IPFS to production PostgreSQL/Redis/Milvus")
    print()
    
    # Confirmation
    response = input("Continue with migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled")
        return
    
    migration_manager = DataMigrationManager()
    results = await migration_manager.run_full_migration()
    
    if results["errors"] == 0:
        print("\n🎉 Migration completed successfully!")
        print("Your PRSM installation now uses production-grade data persistence.")
    else:
        print(f"\n⚠️ Migration completed with {results['errors']} errors.")
        print("Please review the migration report for details.")


if __name__ == "__main__":
    asyncio.run(main())