"""
Enhanced Sandbox Manager
=======================

Advanced sandboxing capabilities that extend the base sandbox manager
with additional security features and monitoring.
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

from .sandbox_manager import SandboxManager, SandboxResult
from .audit_logger import audit_logger, SecurityEvent, EventLevel


class EnhancedSandboxResult:
    """Enhanced result from sandbox execution"""
    
    def __init__(self, success: bool, output: str, error_output: str,
                 exit_code: int, execution_time: float, 
                 security_events: List[Dict[str, Any]],
                 resource_usage: Dict[str, Any]):
        self.success = success
        self.output = output
        self.error_output = error_output
        self.exit_code = exit_code
        self.execution_time = execution_time
        self.security_events = security_events
        self.resource_usage = resource_usage
        self.execution_id = str(uuid4())
        self.timestamp = datetime.now(timezone.utc)


class EnhancedSandboxManager:
    """Enhanced sandbox manager with advanced security monitoring"""
    
    def __init__(self, base_sandbox_dir: str = "/tmp/prsm_enhanced_sandbox"):
        """Initialize enhanced sandbox manager"""
        self.base_sandbox_manager = SandboxManager()
        self.base_sandbox_dir = Path(base_sandbox_dir)
        self.base_sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Security monitoring configuration
        self.monitoring_enabled = True
        self.max_execution_time = 300  # 5 minutes
        self.max_memory_mb = 512
        self.max_disk_usage_mb = 100
        
        # Network monitoring
        self.block_network = True
        self.allowed_domains = ["api.github.com", "huggingface.co"]  # Minimal whitelist
        
        print("🛡️ Enhanced Sandbox Manager initialized")
    
    async def execute_with_monitoring(self, content_path: str, 
                                    execution_config: Dict[str, Any],
                                    user_id: str, platform: str) -> EnhancedSandboxResult:
        """
        Execute content in enhanced sandbox with comprehensive monitoring
        
        Args:
            content_path: Path to content to execute
            execution_config: Configuration for execution
            user_id: User requesting execution
            platform: Source platform
            
        Returns:
            EnhancedSandboxResult with detailed execution information
        """
        execution_id = str(uuid4())
        security_events = []
        
        try:
            # Log execution start
            audit_logger.log_event(SecurityEvent(
                event_type="sandbox_execution_start",
                level=EventLevel.INFO,
                user_id=user_id,
                platform=platform,
                description=f"Starting enhanced sandbox execution {execution_id}",
                metadata={
                    "execution_id": execution_id,
                    "content_path": content_path,
                    "config": execution_config
                }
            ))
            
            # Prepare isolated environment
            sandbox_env = await self._prepare_enhanced_environment(execution_id)
            
            # Copy content to sandbox
            sandbox_content_path = await self._copy_content_to_sandbox(
                content_path, sandbox_env["content_dir"]
            )
            
            # Setup monitoring
            monitor_task = None
            if self.monitoring_enabled:
                monitor_task = asyncio.create_task(
                    self._monitor_execution(execution_id, sandbox_env, security_events)
                )
            
            # Execute with timeout
            start_time = time.time()
            
            try:
                # Use base sandbox for actual execution but with our enhanced environment
                base_result = await asyncio.wait_for(
                    self._execute_in_base_sandbox(sandbox_content_path, execution_config),
                    timeout=self.max_execution_time
                )
                
                execution_time = time.time() - start_time
                
                # Stop monitoring
                if monitor_task:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                
                # Collect resource usage
                resource_usage = await self._collect_resource_usage(sandbox_env)
                
                # Create enhanced result
                result = EnhancedSandboxResult(
                    success=base_result.success,
                    output=base_result.output,
                    error_output=base_result.error_output,
                    exit_code=base_result.exit_code,
                    execution_time=execution_time,
                    security_events=security_events,
                    resource_usage=resource_usage
                )
                
                # Log completion
                audit_logger.log_event(SecurityEvent(
                    event_type="sandbox_execution_complete",
                    level=EventLevel.INFO if result.success else EventLevel.WARNING,
                    user_id=user_id,
                    platform=platform,
                    description=f"Enhanced sandbox execution {execution_id} completed",
                    metadata={
                        "execution_id": execution_id,
                        "success": result.success,
                        "execution_time": execution_time,
                        "security_events_count": len(security_events)
                    }
                ))
                
                return result
                
            except asyncio.TimeoutError:
                # Handle timeout
                if monitor_task:
                    monitor_task.cancel()
                
                security_events.append({
                    "type": "execution_timeout",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": f"Execution exceeded {self.max_execution_time} seconds"
                })
                
                return EnhancedSandboxResult(
                    success=False,
                    output="",
                    error_output=f"Execution timed out after {self.max_execution_time} seconds",
                    exit_code=-1,
                    execution_time=self.max_execution_time,
                    security_events=security_events,
                    resource_usage={"timeout": True}
                )
                
        except Exception as e:
            # Handle execution error
            audit_logger.log_event(SecurityEvent(
                event_type="sandbox_execution_error",
                level=EventLevel.ERROR,
                user_id=user_id,
                platform=platform,
                description=f"Enhanced sandbox execution {execution_id} failed: {str(e)}",
                metadata={
                    "execution_id": execution_id,
                    "error": str(e)
                }
            ))
            
            return EnhancedSandboxResult(
                success=False,
                output="",
                error_output=f"Sandbox execution error: {str(e)}",
                exit_code=-2,
                execution_time=0.0,
                security_events=security_events,
                resource_usage={"error": str(e)}
            )
            
        finally:
            # Cleanup sandbox environment
            await self._cleanup_sandbox_environment(execution_id)
    
    async def _prepare_enhanced_environment(self, execution_id: str) -> Dict[str, Any]:
        """Prepare isolated execution environment"""
        sandbox_dir = self.base_sandbox_dir / execution_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        content_dir = sandbox_dir / "content"
        temp_dir = sandbox_dir / "temp"
        logs_dir = sandbox_dir / "logs"
        
        content_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        
        # Setup environment configuration
        env_config = {
            "sandbox_dir": sandbox_dir,
            "content_dir": content_dir,
            "temp_dir": temp_dir,
            "logs_dir": logs_dir,
            "execution_id": execution_id
        }
        
        # Create security configuration file
        security_config = {
            "max_memory_mb": self.max_memory_mb,
            "max_disk_usage_mb": self.max_disk_usage_mb,
            "block_network": self.block_network,
            "allowed_domains": self.allowed_domains,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        with open(sandbox_dir / "security_config.json", 'w') as f:
            json.dump(security_config, f, indent=2)
        
        return env_config
    
    async def _copy_content_to_sandbox(self, source_path: str, 
                                     sandbox_content_dir: Path) -> str:
        """Copy content to sandbox environment"""
        import shutil
        
        if os.path.isfile(source_path):
            # Single file
            filename = os.path.basename(source_path)
            dest_path = sandbox_content_dir / filename
            shutil.copy2(source_path, dest_path)
            return str(dest_path)
            
        elif os.path.isdir(source_path):
            # Directory
            dest_path = sandbox_content_dir / "content"
            shutil.copytree(source_path, dest_path)
            return str(dest_path)
        
        else:
            raise ValueError(f"Invalid source path: {source_path}")
    
    async def _execute_in_base_sandbox(self, content_path: str, 
                                     execution_config: Dict[str, Any]) -> SandboxResult:
        """Execute using base sandbox manager"""
        # Use the base sandbox manager for actual execution
        return await self.base_sandbox_manager.execute_safely(
            content_path, 
            execution_config.get("metadata", {})
        )
    
    async def _monitor_execution(self, execution_id: str, 
                               sandbox_env: Dict[str, Any],
                               security_events: List[Dict[str, Any]]) -> None:
        """Monitor sandbox execution for security events"""
        monitor_interval = 1.0  # Check every second
        
        try:
            while True:
                await asyncio.sleep(monitor_interval)
                
                # Check disk usage
                disk_usage = await self._check_disk_usage(sandbox_env["sandbox_dir"])
                if disk_usage > self.max_disk_usage_mb:
                    security_events.append({
                        "type": "disk_usage_exceeded",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "details": f"Disk usage {disk_usage}MB exceeds limit {self.max_disk_usage_mb}MB"
                    })
                
                # Check for suspicious file creation
                suspicious_files = await self._check_suspicious_files(sandbox_env["sandbox_dir"])
                if suspicious_files:
                    security_events.append({
                        "type": "suspicious_files_created",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "details": f"Suspicious files detected: {suspicious_files}"
                    })
                
        except asyncio.CancelledError:
            # Monitoring was cancelled (normal)
            pass
    
    async def _check_disk_usage(self, sandbox_dir: Path) -> float:
        """Check disk usage of sandbox directory"""
        total_size = 0
        
        try:
            for root, dirs, files in os.walk(sandbox_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        pass
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    async def _check_suspicious_files(self, sandbox_dir: Path) -> List[str]:
        """Check for creation of suspicious files"""
        suspicious_files = []
        suspicious_patterns = [
            ".exe", ".bat", ".cmd", ".scr", ".pif",
            "passwd", "shadow", "hosts", "crontab"
        ]
        
        try:
            for root, dirs, files in os.walk(sandbox_dir):
                for file in files:
                    file_lower = file.lower()
                    for pattern in suspicious_patterns:
                        if pattern in file_lower:
                            suspicious_files.append(file)
                            break
            
        except Exception:
            pass
        
        return suspicious_files
    
    async def _collect_resource_usage(self, sandbox_env: Dict[str, Any]) -> Dict[str, Any]:
        """Collect resource usage statistics"""
        try:
            sandbox_dir = sandbox_env["sandbox_dir"]
            
            # Disk usage
            disk_usage_mb = await self._check_disk_usage(sandbox_dir)
            
            # File count
            file_count = 0
            for root, dirs, files in os.walk(sandbox_dir):
                file_count += len(files)
            
            return {
                "disk_usage_mb": disk_usage_mb,
                "file_count": file_count,
                "sandbox_path": str(sandbox_dir)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _cleanup_sandbox_environment(self, execution_id: str) -> None:
        """Clean up sandbox environment after execution"""
        import shutil
        
        try:
            sandbox_dir = self.base_sandbox_dir / execution_id
            if sandbox_dir.exists():
                shutil.rmtree(sandbox_dir)
                
        except Exception as e:
            print(f"⚠️ Failed to cleanup sandbox {execution_id}: {e}")


# Global enhanced sandbox manager instance
enhanced_sandbox_manager = EnhancedSandboxManager()