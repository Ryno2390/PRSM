"""
PRSM API Versioning System
Comprehensive version management and backward compatibility
"""

from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fastapi import Request, Response, HTTPException
from fastapi.routing import APIRoute
import re
import logging
import json

logger = logging.getLogger(__name__)

class APIVersion(Enum):
    """Supported API versions"""
    V1_0 = "1.0"
    V1_1 = "1.1"  # Future version with enhanced features
    V2_0 = "2.0"  # Future major version with breaking changes
    
    @classmethod
    def from_string(cls, version_str: str) -> 'APIVersion':
        """Parse version string to APIVersion enum"""
        # Remove 'v' prefix if present
        clean_version = version_str.lower().replace('v', '')
        
        version_map = {
            '1': cls.V1_0,
            '1.0': cls.V1_0,
            '1.1': cls.V1_1,
            '2': cls.V2_0,
            '2.0': cls.V2_0,
            'latest': cls.V1_0,  # Current latest
            'stable': cls.V1_0,  # Current stable
        }
        
        if clean_version in version_map:
            return version_map[clean_version]
        
        raise ValueError(f"Unsupported API version: {version_str}")
    
    @property
    def major(self) -> int:
        """Get major version number"""
        return int(self.value.split('.')[0])
    
    @property
    def minor(self) -> int:
        """Get minor version number"""
        return int(self.value.split('.')[1])
    
    def is_compatible_with(self, other: 'APIVersion') -> bool:
        """Check if this version is backward compatible with another"""
        return self.major == other.major and self.minor >= other.minor


@dataclass
class VersionInfo:
    """Information about an API version"""
    version: APIVersion
    release_date: datetime
    status: str  # 'active', 'deprecated', 'sunset'
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    migration_guide_url: Optional[str] = None
    changelog: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    
    @property
    def is_deprecated(self) -> bool:
        """Check if version is deprecated"""
        return self.status == 'deprecated' or (
            self.deprecation_date and datetime.now(timezone.utc) >= self.deprecation_date
        )
    
    @property
    def is_sunset(self) -> bool:
        """Check if version is sunset (no longer supported)"""
        return self.status == 'sunset' or (
            self.sunset_date and datetime.now(timezone.utc) >= self.sunset_date
        )


class APIVersionRegistry:
    """Registry for managing API versions and their metadata"""
    
    def __init__(self):
        self.versions: Dict[APIVersion, VersionInfo] = {
            APIVersion.V1_0: VersionInfo(
                version=APIVersion.V1_0,
                release_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
                status='active',
                changelog=[
                    "Initial PRSM API release",
                    "Core authentication system",
                    "FTNS token economy integration",
                    "Marketplace functionality",
                    "Research session management",
                    "Real-time WebSocket support"
                ]
            ),
            APIVersion.V1_1: VersionInfo(
                version=APIVersion.V1_1,
                release_date=datetime(2024, 4, 1, tzinfo=timezone.utc),
                status='active',
                changelog=[
                    "Enhanced AI model orchestration",
                    "Improved performance monitoring",
                    "Extended marketplace categories",
                    "Advanced task hierarchies",
                    "Better error handling"
                ]
            ),
            APIVersion.V2_0: VersionInfo(
                version=APIVersion.V2_0,
                release_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
                status='planned',
                breaking_changes=[
                    "Restructured authentication flow",
                    "New marketplace API endpoints", 
                    "Updated WebSocket message format",
                    "Enhanced FTNS token mechanics"
                ]
            )
        }
        
        self.current_version = APIVersion.V1_0
        self.default_version = APIVersion.V1_0
    
    def get_version_info(self, version: APIVersion) -> VersionInfo:
        """Get information about a specific version"""
        return self.versions.get(version)
    
    def get_supported_versions(self) -> List[APIVersion]:
        """Get list of currently supported versions"""
        return [
            version for version, info in self.versions.items()
            if not info.is_sunset and info.status in ['active', 'deprecated']
        ]
    
    def get_latest_version(self) -> APIVersion:
        """Get the latest stable version"""
        return self.current_version
    
    def is_version_supported(self, version: APIVersion) -> bool:
        """Check if a version is currently supported"""
        info = self.versions.get(version)
        return info is not None and not info.is_sunset


# Global version registry
version_registry = APIVersionRegistry()


class VersionExtractor:
    """Extract API version from various sources"""
    
    @staticmethod
    def from_url_path(path: str) -> Optional[APIVersion]:
        """Extract version from URL path like /api/v1/endpoint"""
        match = re.search(r'/api/v?(\d+(?:\.\d+)?)', path)
        if match:
            try:
                return APIVersion.from_string(match.group(1))
            except ValueError:
                return None
        return None
    
    @staticmethod
    def from_header(request: Request, header_name: str = "API-Version") -> Optional[APIVersion]:
        """Extract version from HTTP header"""
        version_header = request.headers.get(header_name)
        if version_header:
            try:
                return APIVersion.from_string(version_header)
            except ValueError:
                return None
        return None
    
    @staticmethod
    def from_query_param(request: Request, param_name: str = "version") -> Optional[APIVersion]:
        """Extract version from query parameter"""
        version_param = request.query_params.get(param_name)
        if version_param:
            try:
                return APIVersion.from_string(version_param)
            except ValueError:
                return None
        return None
    
    @staticmethod
    def from_accept_header(request: Request) -> Optional[APIVersion]:
        """Extract version from Accept header like application/vnd.prsm.v1+json"""
        accept_header = request.headers.get("Accept", "")
        match = re.search(r'application/vnd\.prsm\.v?(\d+(?:\.\d+)?)', accept_header)
        if match:
            try:
                return APIVersion.from_string(match.group(1))
            except ValueError:
                return None
        return None


class VersionNegotiator:
    """Negotiate the best API version to use for a request"""
    
    def __init__(self, registry: APIVersionRegistry):
        self.registry = registry
    
    def negotiate_version(self, request: Request) -> APIVersion:
        """
        Negotiate the best API version for the request based on:
        1. URL path version (highest priority)
        2. API-Version header
        3. Accept header with vendor-specific media type
        4. Query parameter
        5. Default version (fallback)
        """
        # Try URL path first (most explicit)
        version = VersionExtractor.from_url_path(request.url.path)
        if version and self.registry.is_version_supported(version):
            return version
        
        # Try API-Version header
        version = VersionExtractor.from_header(request)
        if version and self.registry.is_version_supported(version):
            return version
        
        # Try Accept header with vendor media type
        version = VersionExtractor.from_accept_header(request)
        if version and self.registry.is_version_supported(version):
            return version
        
        # Try query parameter
        version = VersionExtractor.from_query_param(request)
        if version and self.registry.is_version_supported(version):
            return version
        
        # Return default version
        return self.registry.default_version


# Global version negotiator
version_negotiator = VersionNegotiator(version_registry)


@dataclass
class RequestContext:
    """Context information about the current request"""
    version: APIVersion
    original_request: Request
    version_source: str  # How the version was determined
    is_deprecated: bool = False
    deprecation_warnings: List[str] = field(default_factory=list)


class VersioningMiddleware:
    """Middleware for handling API versioning"""
    
    def __init__(self, negotiator: VersionNegotiator):
        self.negotiator = negotiator
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request with version information"""
        
        # Negotiate API version
        version = self.negotiator.negotiate_version(request)
        version_info = version_registry.get_version_info(version)
        
        # Check if version is sunset
        if version_info and version_info.is_sunset:
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "API_VERSION_SUNSET",
                    "message": f"API version {version.value} is no longer supported",
                    "sunset_date": version_info.sunset_date.isoformat() if version_info.sunset_date else None,
                    "migration_guide": version_info.migration_guide_url,
                    "supported_versions": [v.value for v in version_registry.get_supported_versions()]
                }
            )
        
        # Create request context
        context = RequestContext(
            version=version,
            original_request=request,
            version_source=self._determine_version_source(request, version),
            is_deprecated=version_info.is_deprecated if version_info else False
        )
        
        # Store context in request state
        request.state.api_version = version
        request.state.version_context = context
        
        # Process request
        response = await call_next(request)
        
        # Add version headers to response
        self._add_version_headers(response, context, version_info)
        
        return response
    
    def _determine_version_source(self, request: Request, version: APIVersion) -> str:
        """Determine how the version was specified"""
        if VersionExtractor.from_url_path(request.url.path) == version:
            return "url_path"
        elif VersionExtractor.from_header(request) == version:
            return "header"
        elif VersionExtractor.from_accept_header(request) == version:
            return "accept_header"
        elif VersionExtractor.from_query_param(request) == version:
            return "query_param"
        else:
            return "default"
    
    def _add_version_headers(self, response: Response, context: RequestContext, version_info: Optional[VersionInfo]):
        """Add version-related headers to response"""
        response.headers["API-Version"] = context.version.value
        response.headers["API-Version-Source"] = context.version_source
        response.headers["API-Supported-Versions"] = ",".join([v.value for v in version_registry.get_supported_versions()])
        
        if version_info:
            if version_info.is_deprecated:
                response.headers["API-Deprecation-Warning"] = "true"
                response.headers["API-Deprecation-Date"] = version_info.deprecation_date.isoformat() if version_info.deprecation_date else ""
                if version_info.sunset_date:
                    response.headers["API-Sunset-Date"] = version_info.sunset_date.isoformat()
                if version_info.migration_guide_url:
                    response.headers["API-Migration-Guide"] = version_info.migration_guide_url


class BackwardCompatibilityTransformer:
    """Transform requests and responses for backward compatibility"""
    
    def __init__(self):
        self.transformers: Dict[APIVersion, Dict[str, Callable]] = {
            # Example transformers for different versions
            APIVersion.V1_0: {
                "request": self._transform_v1_0_request,
                "response": self._transform_v1_0_response
            }
        }
    
    def transform_request(self, request_data: Dict[str, Any], from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Transform request data between API versions"""
        if from_version == to_version:
            return request_data
        
        # Apply transformations step by step
        transformed_data = request_data.copy()
        
        # For now, implement basic transformation logic
        # In a real system, this would have comprehensive transformation rules
        if from_version == APIVersion.V1_0 and to_version.major > 1:
            transformed_data = self._migrate_v1_to_v2_request(transformed_data)
        
        return transformed_data
    
    def transform_response(self, response_data: Dict[str, Any], from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Transform response data between API versions"""
        if from_version == to_version:
            return response_data
        
        # Apply reverse transformations
        transformed_data = response_data.copy()
        
        if from_version.major > 1 and to_version == APIVersion.V1_0:
            transformed_data = self._migrate_v2_to_v1_response(transformed_data)
        
        return transformed_data
    
    def _transform_v1_0_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v1.0 request format"""
        # Example: Handle legacy field names
        if 'user_email' in data:
            data['email'] = data.pop('user_email')
        return data
    
    def _transform_v1_0_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform response to v1.0 format"""
        # Example: Add legacy fields for backward compatibility
        if 'email' in data and 'user_email' not in data:
            data['user_email'] = data['email']
        return data
    
    def _migrate_v1_to_v2_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 request format to v2"""
        # Example migration logic
        migrated = data.copy()
        
        # Handle authentication format changes
        if 'auth_token' in migrated:
            migrated['authorization'] = {
                'type': 'bearer',
                'token': migrated.pop('auth_token')
            }
        
        return migrated
    
    def _migrate_v2_to_v1_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v2 response format to v1"""
        # Example reverse migration
        migrated = data.copy()
        
        # Handle response format changes
        if 'authorization' in migrated:
            auth = migrated.pop('authorization')
            if isinstance(auth, dict) and auth.get('type') == 'bearer':
                migrated['auth_token'] = auth.get('token')
        
        return migrated


# Global compatibility transformer
compatibility_transformer = BackwardCompatibilityTransformer()


def get_request_version(request: Request) -> APIVersion:
    """Get the API version for the current request"""
    return getattr(request.state, 'api_version', version_registry.default_version)


def get_version_context(request: Request) -> Optional[RequestContext]:
    """Get the version context for the current request"""
    return getattr(request.state, 'version_context', None)


def deprecated_endpoint(version: APIVersion, migration_guide: str = None):
    """Decorator to mark an endpoint as deprecated"""
    def decorator(func):
        func._deprecated_version = version
        func._migration_guide = migration_guide
        return func
    return decorator


def version_added(version: APIVersion):
    """Decorator to mark when an endpoint was added"""
    def decorator(func):
        func._version_added = version
        return func
    return decorator


def version_changed(version: APIVersion, changes: List[str]):
    """Decorator to document changes in a specific version"""
    def decorator(func):
        func._version_changes = {version: changes}
        return func
    return decorator


class VersionedRoute(APIRoute):
    """Custom route class that handles versioning"""
    
    def __init__(self, *args, **kwargs):
        self.min_version = kwargs.pop('min_version', None)
        self.max_version = kwargs.pop('max_version', None)
        self.deprecated_version = kwargs.pop('deprecated_version', None)
        super().__init__(*args, **kwargs)
    
    def matches(self, scope: Dict[str, Any]) -> tuple:
        """Check if route matches and version is supported"""
        match, child_scope = super().matches(scope)
        
        if not match:
            return match, child_scope
        
        # Extract version from request (simplified for this example)
        # In a real implementation, this would use the full version negotiation
        request_version = version_registry.default_version
        
        # Check version compatibility
        if self.min_version and request_version.value < self.min_version.value:
            return False, {}
        
        if self.max_version and request_version.value > self.max_version.value:
            return False, {}
        
        return match, child_scope


# Utility functions for common version operations
def requires_version(min_version: APIVersion, max_version: APIVersion = None):
    """Decorator to enforce version requirements on endpoints"""
    def decorator(func):
        def wrapper(request: Request, *args, **kwargs):
            current_version = get_request_version(request)
            
            if current_version.value < min_version.value:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "API_VERSION_TOO_OLD",
                        "message": f"This endpoint requires API version {min_version.value} or newer",
                        "current_version": current_version.value,
                        "required_version": min_version.value
                    }
                )
            
            if max_version and current_version.value > max_version.value:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "API_VERSION_TOO_NEW",
                        "message": f"This endpoint does not support API version {current_version.value}",
                        "current_version": current_version.value,
                        "max_supported_version": max_version.value
                    }
                )
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator


def version_specific_response(versions: Dict[APIVersion, Callable]):
    """Decorator to provide version-specific responses"""
    def decorator(func):
        def wrapper(request: Request, *args, **kwargs):
            current_version = get_request_version(request)
            
            if current_version in versions:
                return versions[current_version](request, *args, **kwargs)
            
            # Find the closest compatible version
            compatible_versions = [v for v in versions.keys() if v.is_compatible_with(current_version)]
            if compatible_versions:
                best_version = max(compatible_versions, key=lambda v: (v.major, v.minor))
                return versions[best_version](request, *args, **kwargs)
            
            # Fallback to original function
            return func(request, *args, **kwargs)
        return wrapper
    return decorator