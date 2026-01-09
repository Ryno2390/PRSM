"""
Integration Manager
==================

Central coordination system for PRSM's integration layer, managing
all platform connectors and orchestrating import operations while
maintaining safety, security, and economic incentives.

Key Responsibilities:
- Connector lifecycle management and health monitoring
- Import operation coordination and status tracking
- Security and compliance enforcement
- FTNS integration for creator rewards
- Performance optimization and rate limiting
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Type
from uuid import UUID, uuid4

from .base_connector import BaseConnector, ConnectorStatus
from .provenance_engine import ProvenanceEngine
from ..models.integration_models import (
    IntegrationPlatform, ConnectorConfig, IntegrationSource,
    ImportRequest, ImportResult, ConnectorHealth, IntegrationStats,
    IntegrationRecord, ImportStatus
)
from ..security.sandbox_manager import SandboxManager
from prsm.core.config import settings
from prsm.core.models import FTNSTransaction
from ...tokenomics.ftns_service import ftns_service


class IntegrationManager:
    """
    Central coordination system for PRSM's integration layer
    
    Manages all platform connectors, coordinates import operations,
    and ensures compliance with PRSM's safety and economic principles.
    """
    
    def __init__(self):
        """Initialize the integration manager"""
        
        # Connector management
        self.connectors: Dict[IntegrationPlatform, BaseConnector] = {}
        self.connector_configs: Dict[str, ConnectorConfig] = {}  # user_id -> config
        
        # Operation tracking
        self.active_imports: Dict[UUID, ImportRequest] = {}
        self.import_history: List[ImportResult] = []
        self.integration_records: List[IntegrationRecord] = []
        
        # Components
        self.provenance_engine = ProvenanceEngine()
        self.sandbox_manager = SandboxManager()
        
        # Performance and health
        self.stats = IntegrationStats()
        self.last_health_check = None
        
        # Configuration
        self.max_concurrent_imports = int(getattr(settings, "PRSM_MAX_CONCURRENT_IMPORTS", 5))
        self.import_timeout_minutes = int(getattr(settings, "PRSM_IMPORT_TIMEOUT_MINUTES", 30))
        self.health_check_interval = int(getattr(settings, "PRSM_HEALTH_CHECK_INTERVAL", 300))  # 5 minutes
        
        print("🔧 Integration Manager initialized")
    
    # === Connector Management ===
    
    async def register_connector(self, connector_class: Type[BaseConnector], 
                               config: ConnectorConfig) -> bool:
        """
        Register and initialize a platform connector
        
        Args:
            connector_class: Class of the connector to instantiate
            config: Configuration for the connector
            
        Returns:
            True if registration successful
        """
        try:
            platform = config.platform
            user_id = config.user_id
            
            print(f"📝 Registering {platform.value} connector for user {user_id}")
            
            # Create connector instance
            connector = connector_class(config)
            
            # Initialize connector
            init_success = await connector.initialize()
            if not init_success:
                print(f"❌ Failed to initialize {platform.value} connector")
                return False
            
            # Store connector and config
            self.connectors[platform] = connector
            self.connector_configs[f"{user_id}:{platform.value}"] = config
            
            # Update stats
            self.stats.platforms_connected += 1
            if platform not in self.stats.active_connectors:
                self.stats.active_connectors.append(platform)
            
            print(f"✅ Successfully registered {platform.value} connector")
            return True
            
        except Exception as e:
            print(f"❌ Error registering {config.platform.value} connector: {e}")
            return False
    
    async def get_connector(self, platform: IntegrationPlatform) -> Optional[BaseConnector]:
        """
        Get connector for specified platform
        
        Args:
            platform: Platform to get connector for
            
        Returns:
            Connector instance or None if not available
        """
        connector = self.connectors.get(platform)
        
        if connector and connector.is_healthy():
            return connector
        elif connector:
            print(f"⚠️ {platform.value} connector is unhealthy: {connector.status.value}")
            return None
        else:
            print(f"❌ No connector available for {platform.value}")
            return None
    
    async def health_check_all_connectors(self) -> Dict[IntegrationPlatform, ConnectorHealth]:
        """
        Perform health check on all registered connectors
        
        Returns:
            Dictionary mapping platforms to their health status
        """
        print("🔍 Performing health check on all connectors")
        health_results = {}
        
        for platform, connector in self.connectors.items():
            try:
                health = await connector.health_check()
                health_results[platform] = health
                
                if health.status == "unhealthy":
                    print(f"⚠️ {platform.value} connector is unhealthy: {health.issues}")
                
            except Exception as e:
                print(f"❌ Health check failed for {platform.value}: {e}")
                health_results[platform] = ConnectorHealth(
                    platform=platform,
                    status="offline",
                    issues=[f"Health check failed: {str(e)}"]
                )
        
        self.last_health_check = datetime.now(timezone.utc)
        return health_results
    
    # === Import Operations ===
    
    async def submit_import_request(self, request: ImportRequest) -> UUID:
        """
        Submit a content import request
        
        Args:
            request: ImportRequest with content details
            
        Returns:
            Request ID for tracking
        """
        try:
            # Validate request
            if not await self._validate_import_request(request):
                raise ValueError("Import request validation failed")
            
            # Check concurrent import limit
            if len(self.active_imports) >= self.max_concurrent_imports:
                raise ValueError(f"Too many concurrent imports (max: {self.max_concurrent_imports})")
            
            # Get appropriate connector
            connector = await self.get_connector(request.source.platform)
            if not connector:
                raise ValueError(f"No healthy connector available for {request.source.platform.value}")
            
            # Add to active imports
            self.active_imports[request.request_id] = request
            
            print(f"📋 Submitted import request: {request.request_id}")
            print(f"   - Source: {request.source.display_name} ({request.source.platform.value})")
            print(f"   - Type: {request.import_type}")
            print(f"   - User: {request.user_id}")
            
            # Start import operation asynchronously
            asyncio.create_task(self._execute_import(request, connector))
            
            return request.request_id
            
        except Exception as e:
            print(f"❌ Failed to submit import request: {e}")
            raise
    
    async def get_import_status(self, request_id: UUID) -> Optional[ImportStatus]:
        """
        Get status of an import operation
        
        Args:
            request_id: ID of the import request
            
        Returns:
            Current import status or None if not found
        """
        # Check active imports
        if request_id in self.active_imports:
            return self.active_imports[request_id].status
        
        # Check completed imports
        for result in self.import_history:
            if result.request_id == request_id:
                return result.status
        
        return None
    
    async def get_import_result(self, request_id: UUID) -> Optional[ImportResult]:
        """
        Get result of a completed import operation
        
        Args:
            request_id: ID of the import request
            
        Returns:
            ImportResult or None if not found/completed
        """
        for result in self.import_history:
            if result.request_id == request_id:
                return result
        
        return None
    
    async def cancel_import(self, request_id: UUID) -> bool:
        """
        Cancel an active import operation
        
        Args:
            request_id: ID of the import request to cancel
            
        Returns:
            True if cancellation successful
        """
        if request_id in self.active_imports:
            request = self.active_imports[request_id]
            request.status = ImportStatus.CANCELLED
            
            print(f"🚫 Cancelled import request: {request_id}")
            return True
        
        return False
    
    # === Search and Discovery ===
    
    async def search_content(self, query: str, platforms: Optional[List[IntegrationPlatform]] = None,
                           content_type: str = "model", limit: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[IntegrationSource]:
        """
        Search for content across multiple platforms
        
        Args:
            query: Search query string
            platforms: List of platforms to search (all if None)
            content_type: Type of content to search for
            limit: Maximum results per platform
            filters: Search filters
            
        Returns:
            Aggregated list of IntegrationSource results
        """
        if platforms is None:
            platforms = list(self.connectors.keys())
        
        print(f"🔍 Searching for '{query}' across {len(platforms)} platforms")
        
        all_results = []
        search_tasks = []
        
        # Create search tasks for each platform
        for platform in platforms:
            connector = await self.get_connector(platform)
            if connector:
                task = self._search_platform(connector, query, content_type, limit, filters)
                search_tasks.append(task)
        
        # Execute searches concurrently
        if search_tasks:
            platform_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Aggregate results
            for result in platform_results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):
                    print(f"⚠️ Search error: {result}")
        
        print(f"📊 Found {len(all_results)} total results across all platforms")
        return all_results[:limit * 2]  # Return up to 2x limit for better coverage
    
    # === Analytics and Statistics ===
    
    async def get_integration_stats(self) -> IntegrationStats:
        """
        Get comprehensive integration layer statistics
        
        Returns:
            IntegrationStats with current metrics
        """
        # Update calculated fields
        if self.stats.total_imports > 0:
            success_rate = self.stats.successful_imports / self.stats.total_imports
            total_time = sum(result.import_duration for result in self.import_history)
            self.stats.average_import_time = total_time / len(self.import_history) if self.import_history else 0.0
        
        self.stats.platforms_connected = len(self.connectors)
        self.stats.active_connectors = [p for p, c in self.connectors.items() if c.is_healthy()]
        
        return self.stats
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall integration system health
        
        Returns:
            System health summary
        """
        connector_health = await self.health_check_all_connectors()
        
        healthy_connectors = sum(1 for h in connector_health.values() if h.status == "healthy")
        total_connectors = len(connector_health)
        
        health_percentage = (healthy_connectors / max(total_connectors, 1)) * 100
        
        return {
            "overall_status": "healthy" if health_percentage > 80 else "degraded" if health_percentage > 50 else "unhealthy",
            "health_percentage": health_percentage,
            "connectors": {
                "total": total_connectors,
                "healthy": healthy_connectors,
                "degraded": sum(1 for h in connector_health.values() if h.status == "degraded"),
                "unhealthy": sum(1 for h in connector_health.values() if h.status in ["unhealthy", "offline"])
            },
            "imports": {
                "active": len(self.active_imports),
                "total": self.stats.total_imports,
                "success_rate": (self.stats.successful_imports / max(self.stats.total_imports, 1)) * 100
            },
            "last_health_check": self.last_health_check,
            "sandbox_status": await self.sandbox_manager.get_status()
        }
    
    # === Private Helper Methods ===
    
    async def _validate_import_request(self, request: ImportRequest) -> bool:
        """Validate import request before processing"""
        try:
            # Basic validation
            if not request.source.external_id or not request.user_id:
                return False
            
            # Check if user has sufficient FTNS balance for import
            # This would integrate with the actual FTNS service
            return True
            
        except Exception as e:
            print(f"❌ Import request validation failed: {e}")
            return False
    
    async def _execute_import(self, request: ImportRequest, connector: BaseConnector) -> None:
        """Execute import operation with proper tracking and cleanup"""
        start_time = time.time()
        
        try:
            # Phase 1: Execute initial import through connector
            result = await connector.import_content(request)
            
            # Phase 2: Enhanced security scanning if required
            if request.security_scan_required and result.status == ImportStatus.COMPLETED:
                result = await self._perform_security_scanning(request, result)
            
            # Update statistics
            self.stats.total_imports += 1
            if result.status == ImportStatus.COMPLETED:
                self.stats.successful_imports += 1
                
                # Distribute FTNS rewards if applicable
                if request.auto_reward_creator and result.provenance:
                    await self._distribute_creator_rewards(result)
            else:
                self.stats.failed_imports += 1
            
            # Store result
            self.import_history.append(result)
            
            # Create integration record
            record = IntegrationRecord(
                user_id=request.user_id,
                platform=request.source.platform,
                operation_type="import",
                source_reference=request.source.external_id,
                request_id=request.request_id,
                result_id=result.result_id,
                status=result.status.value,
                operation_duration=result.import_duration,
                success=(result.status == ImportStatus.COMPLETED)
            )
            self.integration_records.append(record)
            
            print(f"📊 Import operation completed: {request.request_id} ({result.status.value})")
            
        except Exception as e:
            print(f"❌ Import execution failed for {request.request_id}: {e}")
            
            # Create failed result
            result = ImportResult(
                request_id=request.request_id,
                status=ImportStatus.FAILED,
                error_details={"reason": str(e)},
                import_duration=time.time() - start_time
            )
            self.import_history.append(result)
            self.stats.total_imports += 1
            self.stats.failed_imports += 1
        
        finally:
            # Remove from active imports
            if request.request_id in self.active_imports:
                del self.active_imports[request.request_id]
    
    async def _search_platform(self, connector: BaseConnector, query: str, 
                             content_type: str, limit: int, filters: Optional[Dict[str, Any]]) -> List[IntegrationSource]:
        """Search single platform with error handling"""
        try:
            return await connector.search_content(query, content_type, limit, filters)
        except Exception as e:
            print(f"⚠️ Search failed for {connector.platform.value}: {e}")
            return []
    
    async def _perform_security_scanning(self, request: ImportRequest, result: ImportResult) -> ImportResult:
        """Perform comprehensive security scanning on imported content"""
        try:
            # Import security orchestrator (dynamic import to avoid circular dependencies)
            from ..security.security_orchestrator import security_orchestrator
            
            print(f"🔍 Starting security scan for content: {request.source.external_id}")
            
            # Perform comprehensive security assessment
            security_assessment = await security_orchestrator.comprehensive_security_assessment(
                content_path=result.local_path,
                metadata=result.metadata,
                user_id=request.user_id,
                platform=request.source.platform.value,
                content_id=request.source.external_id,
                enable_sandbox=True
            )
            
            # Update import result with security information
            result.security_scan_results = {
                "assessment_id": security_assessment.assessment_id,
                "security_passed": security_assessment.security_passed,
                "overall_risk_level": security_assessment.overall_risk_level.value,
                "issues": security_assessment.issues,
                "warnings": security_assessment.warnings,
                "recommendations": security_assessment.recommendations,
                "scan_duration": security_assessment.scan_duration,
                "scans_completed": security_assessment.scans_completed,
                "scans_failed": security_assessment.scans_failed
            }
            
            # Block import if security scan failed
            if not security_assessment.security_passed:
                print(f"🚨 Security scan failed - blocking import: {security_assessment.issues}")
                result.status = ImportStatus.SECURITY_BLOCKED
                result.error_details = f"Security validation failed: {'; '.join(security_assessment.issues[:3])}"
                
                # Clean up imported content if security failed
                await self._cleanup_blocked_content(result.local_path)
            else:
                print(f"✅ Security scan passed for content: {request.source.external_id}")
            
            return result
            
        except Exception as e:
            print(f"❌ Security scanning failed: {e}")
            # If security scanning fails, err on the side of caution
            result.status = ImportStatus.SECURITY_ERROR
            result.error_details = f"Security scan error: {str(e)}"
            return result
    
    async def _cleanup_blocked_content(self, content_path: str) -> None:
        """Clean up content that was blocked by security scan"""
        try:
            import os
            import shutil
            
            if os.path.exists(content_path):
                if os.path.isfile(content_path):
                    os.remove(content_path)
                elif os.path.isdir(content_path):
                    shutil.rmtree(content_path)
                print(f"🧹 Cleaned up blocked content: {content_path}")
        except Exception as e:
            print(f"⚠️ Failed to cleanup blocked content: {e}")
    
    async def _distribute_creator_rewards(self, result: ImportResult) -> None:
        """Distribute FTNS rewards to content creators"""
        if not result.provenance or not result.provenance.original_creator:
            return
        
        try:
            # Calculate reward amount based on content type and usage
            reward_amount = 10.0  # Base reward amount (would be configurable)
            
            # Use FTNS service to distribute rewards
            await ftns_service.reward_contribution(
                result.provenance.original_creator,
                "integration_import",
                reward_amount,
                {
                    "content_id": result.provenance.content_id,
                    "platform": result.provenance.platform_source.value,
                    "import_id": str(result.result_id)
                }
            )
            
            result.ftns_rewards_distributed = reward_amount
            self.stats.total_rewards_distributed += reward_amount
            
            print(f"💰 Distributed {reward_amount} FTNS to creator: {result.provenance.original_creator}")
            
        except Exception as e:
            print(f"❌ Failed to distribute creator rewards: {e}")


# === Global Integration Manager Instance ===
integration_manager = IntegrationManager()