"""
Base Connector Abstract Class
=============================

Abstract base class for all platform integration connectors, providing
a consistent interface and common functionality for connecting to external
platforms while maintaining PRSM's safety and economic principles.

Key Features:
- Standardized connector interface for all platforms
- Built-in security and validation hooks
- FTNS integration for creator rewards
- Error handling and rate limiting
- Health monitoring and diagnostics
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID

from ..models.integration_models import (
    IntegrationPlatform, ConnectorConfig, IntegrationSource,
    ImportRequest, ImportResult, SecurityScanResult, ProvenanceMetadata,
    ConnectorHealth, ImportStatus, SecurityRisk
)
from prsm.core.config import settings


class ConnectorStatus(str, Enum):
    """Connector operational status"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"


class BaseConnector(ABC):
    """
    Abstract base class for all platform integration connectors
    
    Provides standardized interface and common functionality for:
    - Authentication and authorization
    - Content discovery and metadata extraction
    - Security validation and compliance checking
    - FTNS provenance tracking and creator rewards
    - Rate limiting and error handling
    """
    
    def __init__(self, config: ConnectorConfig):
        """
        Initialize connector with platform-specific configuration
        
        Args:
            config: ConnectorConfig with authentication and settings
        """
        self.config = config
        self.platform = config.platform
        self.user_id = config.user_id
        
        # Operational state
        self.status = ConnectorStatus.INITIALIZING
        self.last_health_check = None
        self.error_count = 0
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        
        # Configuration from settings
        self.max_retries = int(getattr(settings, "PRSM_CONNECTOR_MAX_RETRIES", 3))
        self.timeout_seconds = int(getattr(settings, "PRSM_CONNECTOR_TIMEOUT", 30))
        self.rate_limit_buffer = float(getattr(settings, "PRSM_RATE_LIMIT_BUFFER", 0.1))
        
        print(f"🔌 Initializing {self.platform.value} connector for user {self.user_id}")
    
    # === Abstract Methods (Platform-Specific Implementation Required) ===
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the external platform
        
        Returns:
            True if authentication successful
        """
        pass
    
    @abstractmethod
    async def search_content(self, query: str, content_type: str = "model", 
                           limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[IntegrationSource]:
        """
        Search for content on the external platform
        
        Args:
            query: Search query string
            content_type: Type of content to search for (model, dataset, repository)
            limit: Maximum number of results to return
            filters: Platform-specific search filters
            
        Returns:
            List of IntegrationSource objects representing found content
        """
        pass
    
    @abstractmethod
    async def get_content_metadata(self, external_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for specific content
        
        Args:
            external_id: Platform-specific content identifier
            
        Returns:
            Dictionary containing detailed content metadata
        """
        pass
    
    @abstractmethod
    async def download_content(self, external_id: str, target_path: str) -> bool:
        """
        Download content from the external platform
        
        Args:
            external_id: Platform-specific content identifier
            target_path: Local path where content should be stored
            
        Returns:
            True if download successful
        """
        pass
    
    @abstractmethod
    async def validate_license(self, external_id: str) -> Dict[str, Any]:
        """
        Validate license compliance for content
        
        Args:
            external_id: Platform-specific content identifier
            
        Returns:
            Dictionary with license information and compliance status
        """
        pass
    
    # === Common Connector Functionality ===
    
    async def initialize(self) -> bool:
        """
        Initialize the connector and verify platform connectivity
        
        Returns:
            True if initialization successful
        """
        try:
            print(f"🔄 Initializing {self.platform.value} connector...")
            
            # Authenticate with platform
            auth_success = await self.authenticate()
            if not auth_success:
                self.status = ConnectorStatus.AUTH_FAILED
                print(f"❌ Authentication failed for {self.platform.value}")
                return False
            
            # Perform initial health check
            health_result = await self.health_check()
            if health_result.status == "healthy":
                self.status = ConnectorStatus.HEALTHY
                print(f"✅ {self.platform.value} connector initialized successfully")
                return True
            else:
                self.status = ConnectorStatus.DEGRADED
                print(f"⚠️ {self.platform.value} connector initialized with issues: {health_result.issues}")
                return True  # Still functional but degraded
                
        except Exception as e:
            self.status = ConnectorStatus.OFFLINE
            self.error_count += 1
            print(f"❌ Failed to initialize {self.platform.value} connector: {e}")
            return False
    
    async def import_content(self, request: ImportRequest) -> ImportResult:
        """
        Import content from external platform into PRSM ecosystem
        
        Args:
            request: ImportRequest with content details and options
            
        Returns:
            ImportResult with status and metadata
        """
        start_time = time.time()
        result = ImportResult(
            request_id=request.request_id,
            status=ImportStatus.PENDING
        )
        
        try:
            print(f"📥 Starting import of {request.source.external_id} from {self.platform.value}")
            
            # Update request status
            request.status = ImportStatus.SCANNING
            
            # Step 1: Get content metadata
            print(f"📋 Fetching metadata for {request.source.external_id}")
            metadata = await self.get_content_metadata(request.source.external_id)
            result.integration_metadata = metadata
            
            # Step 2: Security and license validation
            if request.security_scan_required or request.license_check_required:
                request.status = ImportStatus.SECURITY_CHECK
                scan_result = await self._perform_security_scan(request, metadata)
                result.security_scan = scan_result
                
                if not scan_result.approved_for_import:
                    result.status = ImportStatus.FAILED
                    result.error_details = {
                        "reason": "Security scan failed",
                        "risk_level": scan_result.risk_level,
                        "issues": scan_result.compliance_issues
                    }
                    print(f"❌ Import blocked due to security concerns: {scan_result.risk_level}")
                    return result
            
            # Step 3: Download content
            request.status = ImportStatus.IMPORTING
            print(f"⬇️ Downloading content from {self.platform.value}")
            
            # For now, we'll simulate the download
            # In actual implementation, this would download to a staging area
            download_success = await self._simulate_download(request.source.external_id)
            
            if not download_success:
                result.status = ImportStatus.FAILED
                result.error_details = {"reason": "Download failed"}
                print(f"❌ Failed to download content from {self.platform.value}")
                return result
            
            # Step 4: Create provenance record
            if request.auto_reward_creator:
                provenance = await self._create_provenance_record(request, metadata)
                result.provenance = provenance
            
            # Step 5: Complete import
            result.status = ImportStatus.COMPLETED
            result.imported_content_id = f"{self.platform.value}:{request.source.external_id}"
            result.success_message = f"Successfully imported {request.source.display_name}"
            
            print(f"✅ Successfully imported {request.source.external_id} from {self.platform.value}")
            
        except Exception as e:
            result.status = ImportStatus.FAILED
            result.error_details = {"reason": str(e)}
            self.error_count += 1
            print(f"❌ Import failed for {request.source.external_id}: {e}")
        
        finally:
            result.import_duration = time.time() - start_time
            self.total_requests += 1
            if result.status == ImportStatus.COMPLETED:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            # Update average response time
            total_time = self.average_response_time * (self.total_requests - 1) + result.import_duration
            self.average_response_time = total_time / self.total_requests
        
        return result
    
    async def health_check(self) -> ConnectorHealth:
        """
        Perform health check on the connector
        
        Returns:
            ConnectorHealth with current status and metrics
        """
        start_time = time.time()
        issues = []
        
        try:
            # Check authentication
            auth_valid = await self.authenticate()
            if not auth_valid:
                issues.append("Authentication failed")
            
            # Check rate limits
            if self.rate_limit_remaining is not None and self.rate_limit_remaining < 10:
                issues.append("Rate limit low")
            
            # Check error rate
            error_rate = self.failed_requests / max(self.total_requests, 1)
            if error_rate > 0.1:  # 10% error rate threshold
                issues.append(f"High error rate: {error_rate:.2%}")
            
            # Determine overall status
            if not auth_valid:
                status = "offline"
            elif len(issues) > 2:
                status = "unhealthy"
            elif len(issues) > 0:
                status = "degraded"
            else:
                status = "healthy"
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            health = ConnectorHealth(
                platform=self.platform,
                status=status,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time,
                error_rate=error_rate,
                rate_limit_remaining=self.rate_limit_remaining,
                issues=issues
            )
            
            self.last_health_check = health
            print(f"🔍 Health check for {self.platform.value}: {status}")
            
            return health
            
        except Exception as e:
            print(f"❌ Health check failed for {self.platform.value}: {e}")
            return ConnectorHealth(
                platform=self.platform,
                status="unhealthy",
                issues=[f"Health check failed: {str(e)}"]
            )
    
    # === Private Helper Methods ===
    
    async def _perform_security_scan(self, request: ImportRequest, metadata: Dict[str, Any]) -> SecurityScanResult:
        """Perform security and compliance scanning"""
        print(f"🔒 Performing security scan for {request.source.external_id}")
        
        # This is a simplified implementation
        # In production, this would integrate with actual security scanning tools
        scan_result = SecurityScanResult(
            request_id=request.request_id,
            risk_level=SecurityRisk.LOW,
            approved_for_import=True,
            scan_duration=1.0,
            scanner_version="1.0.0"
        )
        
        # Basic license validation
        if request.license_check_required:
            license_info = await self.validate_license(request.source.external_id)
            scan_result.license_compliance = license_info.get("type", "unknown")
            
            # Check for non-permissive licenses
            if license_info.get("type") in ["proprietary", "copyleft"]:
                scan_result.compliance_issues.append("Non-permissive license detected")
                scan_result.approved_for_import = False
        
        return scan_result
    
    async def _simulate_download(self, external_id: str) -> bool:
        """Simulate content download (placeholder implementation)"""
        # Simulate download delay
        await asyncio.sleep(0.5)
        return True
    
    async def _create_provenance_record(self, request: ImportRequest, metadata: Dict[str, Any]) -> ProvenanceMetadata:
        """Create provenance record for FTNS creator rewards"""
        print(f"📝 Creating provenance record for {request.source.external_id}")
        
        provenance = ProvenanceMetadata(
            content_id=f"{self.platform.value}:{request.source.external_id}",
            original_creator=metadata.get("creator"),
            platform_source=self.platform,
            external_id=request.source.external_id,
            attribution_chain=[{
                "platform": self.platform.value,
                "creator": metadata.get("creator", "unknown"),
                "content_id": request.source.external_id,
                "imported_by": request.user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }],
            license_info=metadata.get("license", {}),
            reward_eligible=True
        )
        
        return provenance
    
    # === Public Status Methods ===
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connector performance metrics"""
        return {
            "platform": self.platform.value,
            "status": self.status.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": self.failed_requests / max(self.total_requests, 1),
            "average_response_time": self.average_response_time,
            "rate_limit_remaining": self.rate_limit_remaining,
            "last_health_check": self.last_health_check.last_check if self.last_health_check else None
        }
    
    def is_healthy(self) -> bool:
        """Check if connector is in healthy state"""
        return self.status in [ConnectorStatus.HEALTHY, ConnectorStatus.DEGRADED]
    
    def __str__(self) -> str:
        return f"{self.platform.value.title()}Connector(status={self.status.value}, user={self.user_id})"
    
    def __repr__(self) -> str:
        return self.__str__()