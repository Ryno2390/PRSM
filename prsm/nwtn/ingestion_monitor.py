#!/usr/bin/env python3
"""
Ingestion Progress Monitor
=========================

This tool monitors the progress of the large-scale content ingestion
and provides real-time status updates.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import structlog

logger = structlog.get_logger(__name__)


class IngestionMonitor:
    """Monitor ingestion progress and system status"""
    
    def __init__(self):
        self.storage_path = Path("/Volumes/My Passport/PRSM_Storage")
        self.test_results_path = Path("/tmp/production_components_test_results.json")
        
    async def get_current_status(self) -> Dict[str, Any]:
        """Get current ingestion status"""
        
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "storage_status": await self._get_storage_status(),
            "ingestion_metrics": await self._get_ingestion_metrics(),
            "system_health": await self._get_system_health(),
            "progress_summary": await self._get_progress_summary()
        }
        
        return status
    
    async def _get_storage_status(self) -> Dict[str, Any]:
        """Get storage status"""
        
        try:
            if self.storage_path.exists():
                # Get directory sizes
                content_dir = self.storage_path / "PRSM_Content"
                metadata_dir = self.storage_path / "PRSM_Metadata"
                
                content_size = await self._get_directory_size(content_dir)
                metadata_size = await self._get_directory_size(metadata_dir)
                
                # Get file counts
                content_files = await self._count_files(content_dir)
                metadata_files = await self._count_files(metadata_dir)
                
                # Get disk usage
                import shutil
                disk_usage = shutil.disk_usage(self.storage_path)
                
                return {
                    "external_drive_connected": True,
                    "storage_path": str(self.storage_path),
                    "content_size_mb": content_size / (1024 * 1024),
                    "metadata_size_mb": metadata_size / (1024 * 1024),
                    "total_content_files": content_files,
                    "total_metadata_files": metadata_files,
                    "disk_usage": {
                        "total_gb": disk_usage.total / (1024**3),
                        "used_gb": (disk_usage.total - disk_usage.free) / (1024**3),
                        "free_gb": disk_usage.free / (1024**3),
                        "utilization_percent": ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
                    }
                }
            else:
                return {
                    "external_drive_connected": False,
                    "storage_path": str(self.storage_path),
                    "error": "External drive not accessible"
                }
        except Exception as e:
            return {
                "external_drive_connected": False,
                "error": str(e)
            }
    
    async def _get_ingestion_metrics(self) -> Dict[str, Any]:
        """Get ingestion metrics from database"""
        
        try:
            # Check for storage database
            db_path = self.storage_path / "storage.db"
            if db_path.exists():
                import sqlite3
                
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Get content statistics
                cursor.execute("SELECT COUNT(*) FROM content_storage")
                total_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM content_storage WHERE content_type = 'content'")
                content_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM content_storage WHERE content_type = 'metadata'")
                metadata_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT tier) FROM content_storage")
                tiers_used = cursor.fetchone()[0]
                
                # Get size statistics
                cursor.execute("SELECT SUM(original_size), SUM(stored_size) FROM content_storage")
                size_stats = cursor.fetchone()
                original_size = size_stats[0] or 0
                stored_size = size_stats[1] or 0
                
                # Get recent activity
                cursor.execute("SELECT COUNT(*) FROM content_storage WHERE created_at > datetime('now', '-1 hour')")
                recent_items = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    "total_items_stored": total_items,
                    "content_items": content_items,
                    "metadata_items": metadata_items,
                    "storage_tiers_used": tiers_used,
                    "original_size_mb": original_size / (1024 * 1024),
                    "stored_size_mb": stored_size / (1024 * 1024),
                    "compression_ratio": original_size / stored_size if stored_size > 0 else 0,
                    "items_last_hour": recent_items
                }
            else:
                return {
                    "database_available": False,
                    "message": "Storage database not found"
                }
        except Exception as e:
            return {
                "database_available": False,
                "error": str(e)
            }
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        
        try:
            # Check if test results are available
            if self.test_results_path.exists():
                with open(self.test_results_path, 'r') as f:
                    test_results = json.load(f)
                    
                health_status = {
                    "production_ready": test_results.get("production_ready", False),
                    "components_successful": test_results.get("test_summary", {}).get("successful_components", 0),
                    "total_components": test_results.get("test_summary", {}).get("total_components", 0),
                    "last_test_time": test_results.get("test_summary", {}).get("test_timestamp"),
                    "component_status": test_results.get("component_results", {})
                }
            else:
                health_status = {
                    "production_ready": "unknown",
                    "message": "No test results available"
                }
            
            # Add system resource info
            import psutil
            health_status.update({
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            })
            
            return health_status
            
        except Exception as e:
            return {
                "system_health": "error",
                "error": str(e)
            }
    
    async def _get_progress_summary(self) -> Dict[str, Any]:
        """Get overall progress summary"""
        
        try:
            # Estimate progress based on storage
            storage_status = await self._get_storage_status()
            ingestion_metrics = await self._get_ingestion_metrics()
            
            # Target: 150,000 items (from config)
            target_items = 150000
            current_items = ingestion_metrics.get("content_items", 0)
            
            # Calculate progress
            progress_percent = (current_items / target_items) * 100 if target_items > 0 else 0
            
            # Estimate domains covered (rough estimate based on variety)
            estimated_domains = min(8, max(1, current_items // 5000))  # Roughly 1 domain per 5k items
            
            return {
                "target_items": target_items,
                "current_items": current_items,
                "progress_percent": min(100, progress_percent),
                "estimated_domains_covered": estimated_domains,
                "storage_used_gb": storage_status.get("disk_usage", {}).get("used_gb", 0),
                "ingestion_active": current_items > 0,
                "status": self._get_status_message(progress_percent, current_items)
            }
        except Exception as e:
            return {
                "progress_summary": "error",
                "error": str(e)
            }
    
    def _get_status_message(self, progress_percent: float, current_items: int) -> str:
        """Get status message based on progress"""
        
        if current_items == 0:
            return "🚀 Ingestion starting up..."
        elif progress_percent < 1:
            return "📥 Early ingestion phase - building initial corpus"
        elif progress_percent < 10:
            return "🔄 Active ingestion - collecting diverse content"
        elif progress_percent < 50:
            return "📊 Steady progress - building knowledge base"
        elif progress_percent < 90:
            return "🎯 Advanced stage - approaching target"
        else:
            return "🎉 Nearly complete - finalizing corpus"
    
    async def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory"""
        
        if not path.exists():
            return 0
        
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except (OSError, IOError):
                    pass
        
        return total_size
    
    async def _count_files(self, path: Path) -> int:
        """Count files in directory"""
        
        if not path.exists():
            return 0
        
        return len([f for f in path.rglob("*") if f.is_file()])
    
    async def display_status(self):
        """Display current status in a formatted way"""
        
        status = await self.get_current_status()
        
        print("🌍 PRSM NWTN - INGESTION PROGRESS MONITOR")
        print("=" * 60)
        print(f"📅 Status Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Storage Status
        storage = status["storage_status"]
        print("💾 STORAGE STATUS:")
        if storage.get("external_drive_connected"):
            print(f"   ✅ External Drive: Connected ({storage['storage_path']})")
            print(f"   📁 Content Files: {storage['total_content_files']:,}")
            print(f"   📊 Metadata Files: {storage['total_metadata_files']:,}")
            print(f"   💿 Storage Used: {storage['disk_usage']['used_gb']:.1f} GB / {storage['disk_usage']['total_gb']:.1f} GB")
            print(f"   📈 Disk Utilization: {storage['disk_usage']['utilization_percent']:.1f}%")
        else:
            print(f"   ❌ External Drive: Not connected")
        print()
        
        # Ingestion Metrics
        metrics = status["ingestion_metrics"]
        print("📊 INGESTION METRICS:")
        if metrics.get("database_available", True):
            print(f"   📚 Total Items Stored: {metrics.get('total_items_stored', 0):,}")
            print(f"   📖 Content Items: {metrics.get('content_items', 0):,}")
            print(f"   🏷️  Metadata Items: {metrics.get('metadata_items', 0):,}")
            print(f"   🗜️  Compression Ratio: {metrics.get('compression_ratio', 0):.2f}x")
            print(f"   ⏱️  Items (Last Hour): {metrics.get('items_last_hour', 0):,}")
        else:
            print(f"   ⚠️  Database not available")
        print()
        
        # Progress Summary
        progress = status["progress_summary"]
        print("🎯 PROGRESS SUMMARY:")
        if "error" not in progress:
            print(f"   {progress['status']}")
            print(f"   📈 Progress: {progress['progress_percent']:.1f}% ({progress['current_items']:,} / {progress['target_items']:,})")
            print(f"   🌍 Estimated Domains: {progress['estimated_domains_covered']}")
            print(f"   💾 Storage Used: {progress['storage_used_gb']:.1f} GB")
        else:
            print(f"   ❌ Progress tracking error: {progress['error']}")
        print()
        
        # System Health
        health = status["system_health"]
        print("🏥 SYSTEM HEALTH:")
        print(f"   🔧 Production Ready: {health.get('production_ready', 'unknown')}")
        print(f"   🖥️  CPU Usage: {health.get('cpu_percent', 0):.1f}%")
        print(f"   🧠 Memory Usage: {health.get('memory_percent', 0):.1f}%")
        print(f"   ⚙️  Components: {health.get('components_successful', 0)}/{health.get('total_components', 0)} OK")
        print()
        
        print("=" * 60)
        print("📝 Note: Run this script periodically to monitor ingestion progress")
        print("🔄 The ingestion process runs independently in the background")
        print("=" * 60)
    
    async def continuous_monitor(self, interval: int = 30):
        """Continuously monitor and display status"""
        
        print("🔄 Starting continuous monitoring...")
        print(f"⏰ Update interval: {interval} seconds")
        print("🛑 Press Ctrl+C to stop monitoring")
        print()
        
        try:
            while True:
                await self.display_status()
                print(f"\n⏳ Next update in {interval} seconds...\n")
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            return


async def main():
    """Main function"""
    
    monitor = IngestionMonitor()
    
    # Check if user wants continuous monitoring
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        await monitor.continuous_monitor(interval)
    else:
        await monitor.display_status()


if __name__ == "__main__":
    asyncio.run(main())