#!/usr/bin/env python3
"""
Monitoring Plugin
================

Plugin for enhanced system monitoring and observability.
"""

import logging
from typing import Dict, List, Any, Callable
from ..plugin_manager import Plugin, PluginMetadata
from ..optional_deps import require_optional, has_optional_dependency

logger = logging.getLogger(__name__)


class MonitoringPlugin(Plugin):
    """Plugin for enhanced monitoring capabilities"""
    
    def __init__(self):
        self._psutil = None
        self._metrics_collector = None
        self._initialized = False
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="monitoring",
            version="1.0.0", 
            description="Enhanced system monitoring and metrics collection",
            author="PRSM Core Team",
            dependencies=[],
            optional_dependencies=["psutil", "prometheus_client"],
            entry_points={
                "metrics_collector": "collect_system_metrics",
                "health_checker": "check_system_health"
            }
        )
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the monitoring plugin"""
        try:
            # Try to import optional dependencies
            self._psutil = require_optional("psutil")
            
            if self._psutil:
                logger.info("psutil available - enhanced system monitoring enabled")
            else:
                logger.info("psutil not available - basic monitoring only")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring plugin: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup monitoring resources"""
        self._initialized = False
        return True
    
    def get_capabilities(self) -> List[str]:
        """Get monitoring capabilities"""
        capabilities = ["basic_monitoring", "system_health"]
        
        if self._psutil:
            capabilities.extend([
                "advanced_monitoring",
                "process_monitoring",
                "memory_monitoring",
                "cpu_monitoring",
                "disk_monitoring",
                "network_monitoring"
            ])
        
        return capabilities
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get monitoring hook functions"""
        hooks = {
            "collect_metrics": self.collect_system_metrics,
            "health_check": self.check_system_health,
            "performance_monitor": self.monitor_performance
        }
        
        if self._psutil:
            hooks.update({
                "process_monitor": self.monitor_processes,
                "resource_monitor": self.monitor_resources
            })
        
        return hooks
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {
            "timestamp": self._get_timestamp(),
            "basic": self._collect_basic_metrics()
        }
        
        if self._psutil:
            metrics.update({
                "cpu": self._collect_cpu_metrics(),
                "memory": self._collect_memory_metrics(),
                "disk": self._collect_disk_metrics(),
                "network": self._collect_network_metrics(),
                "processes": self._collect_process_metrics()
            })
        
        return metrics
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            "status": "healthy",
            "timestamp": self._get_timestamp(),
            "checks": {}
        }
        
        # Basic health checks
        health["checks"]["plugin_initialized"] = self._initialized
        
        if self._psutil:
            # Advanced health checks
            try:
                cpu_percent = self._psutil.cpu_percent(interval=1)
                memory = self._psutil.virtual_memory()
                disk = self._psutil.disk_usage('/')
                
                health["checks"]["cpu_usage"] = {
                    "value": cpu_percent,
                    "status": "warning" if cpu_percent > 80 else "healthy",
                    "threshold": 80
                }
                
                health["checks"]["memory_usage"] = {
                    "value": memory.percent,
                    "status": "warning" if memory.percent > 85 else "healthy",
                    "threshold": 85
                }
                
                health["checks"]["disk_usage"] = {
                    "value": disk.percent,
                    "status": "warning" if disk.percent > 90 else "healthy",
                    "threshold": 90
                }
                
                # Overall status
                if any(check.get("status") == "warning" for check in health["checks"].values() if isinstance(check, dict)):
                    health["status"] = "warning"
                    
            except Exception as e:
                health["status"] = "error"
                health["error"] = str(e)
        
        return health
    
    def monitor_performance(self, operation_name: str = "unknown") -> Dict[str, Any]:
        """Monitor performance of an operation"""
        import time
        
        start_time = time.time()
        
        def end_monitoring():
            end_time = time.time()
            duration = end_time - start_time
            
            metrics = {
                "operation": operation_name,
                "duration_seconds": duration,
                "start_time": start_time,
                "end_time": end_time
            }
            
            if self._psutil:
                try:
                    # Get current resource usage
                    process = self._psutil.Process()
                    metrics.update({
                        "cpu_percent": process.cpu_percent(),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "memory_percent": process.memory_percent()
                    })
                except Exception as e:
                    logger.warning(f"Could not collect process metrics: {e}")
            
            return metrics
        
        return {"end_monitoring": end_monitoring}
    
    def monitor_processes(self) -> List[Dict[str, Any]]:
        """Monitor running processes"""
        if not self._psutil:
            return []
        
        processes = []
        try:
            for proc in self._psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.error(f"Error monitoring processes: {e}")
        
        return processes
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor system resources"""
        if not self._psutil:
            return {}
        
        try:
            return {
                "cpu": self._collect_cpu_metrics(),
                "memory": self._collect_memory_metrics(),
                "disk": self._collect_disk_metrics(),
                "network": self._collect_network_metrics()
            }
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return {}
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def _collect_basic_metrics(self) -> Dict[str, Any]:
        """Collect basic system metrics"""
        import time
        import os
        
        return {
            "uptime": time.time(),
            "pid": os.getpid(),
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
        }
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU metrics"""
        if not self._psutil:
            return {}
        
        try:
            return {
                "percent": self._psutil.cpu_percent(interval=0.1),
                "count": self._psutil.cpu_count(),
                "count_logical": self._psutil.cpu_count(logical=True),
                "freq": self._psutil.cpu_freq()._asdict() if self._psutil.cpu_freq() else None
            }
        except Exception as e:
            logger.warning(f"Could not collect CPU metrics: {e}")
            return {}
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory metrics"""
        if not self._psutil:
            return {}
        
        try:
            virtual = self._psutil.virtual_memory()
            swap = self._psutil.swap_memory()
            
            return {
                "virtual": virtual._asdict(),
                "swap": swap._asdict()
            }
        except Exception as e:
            logger.warning(f"Could not collect memory metrics: {e}")
            return {}
    
    def _collect_disk_metrics(self) -> Dict[str, Any]:
        """Collect disk metrics"""
        if not self._psutil:
            return {}
        
        try:
            disk_usage = self._psutil.disk_usage('/')
            disk_io = self._psutil.disk_io_counters()
            
            return {
                "usage": disk_usage._asdict(),
                "io": disk_io._asdict() if disk_io else None
            }
        except Exception as e:
            logger.warning(f"Could not collect disk metrics: {e}")
            return {}
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network metrics"""
        if not self._psutil:
            return {}
        
        try:
            net_io = self._psutil.net_io_counters()
            connections = len(self._psutil.net_connections())
            
            return {
                "io": net_io._asdict() if net_io else None,
                "connections": connections
            }
        except Exception as e:
            logger.warning(f"Could not collect network metrics: {e}")
            return {}