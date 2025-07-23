"""
PRSM API Backward Compatibility Layer
Handles transformations and migrations between API versions
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from fastapi import Request, Response, HTTPException
from pydantic import BaseModel, Field
import json
import logging

from .versioning import APIVersion, get_request_version, version_registry

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Mapping configuration for field transformations"""
    old_field: str
    new_field: str
    transformer: Optional[Callable[[Any], Any]] = None
    reverse_transformer: Optional[Callable[[Any], Any]] = None
    required: bool = True
    deprecated_in: Optional[APIVersion] = None
    removed_in: Optional[APIVersion] = None


@dataclass
class SchemaTransformation:
    """Schema transformation rules between versions"""
    from_version: APIVersion
    to_version: APIVersion
    field_mappings: List[FieldMapping]
    custom_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    validation_rules: List[Callable[[Dict[str, Any]], bool]] = None


class CompatibilityEngine:
    """Core engine for handling backward compatibility transformations"""
    
    def __init__(self):
        self.transformations: Dict[tuple, SchemaTransformation] = {}
        self._initialize_default_transformations()
    
    def _initialize_default_transformations(self):
        """Initialize default transformation rules"""
        
        # Authentication endpoint transformations
        self.register_transformation(SchemaTransformation(
            from_version=APIVersion.V1_0,
            to_version=APIVersion.V1_1,
            field_mappings=[
                FieldMapping(
                    old_field="user_email",
                    new_field="email",
                    deprecated_in=APIVersion.V1_1
                ),
                FieldMapping(
                    old_field="user_password",
                    new_field="password",
                    deprecated_in=APIVersion.V1_1
                )
            ]
        ))
        
        # Marketplace resource transformations
        self.register_transformation(SchemaTransformation(
            from_version=APIVersion.V1_0,
            to_version=APIVersion.V1_1,
            field_mappings=[
                FieldMapping(
                    old_field="resource_price",
                    new_field="price",
                    deprecated_in=APIVersion.V1_1
                ),
                FieldMapping(
                    old_field="seller_info",
                    new_field="seller_name",
                    transformer=lambda x: x.get("name") if isinstance(x, dict) else str(x),
                    reverse_transformer=lambda x: {"name": x, "id": None},
                    deprecated_in=APIVersion.V1_1
                )
            ]
        ))
        
        # FTNS token transformations
        self.register_transformation(SchemaTransformation(
            from_version=APIVersion.V1_0,
            to_version=APIVersion.V1_1,
            field_mappings=[
                FieldMapping(
                    old_field="token_balance",
                    new_field="available_balance",
                    deprecated_in=APIVersion.V1_1
                ),
                FieldMapping(
                    old_field="locked_tokens",
                    new_field="locked_balance",
                    deprecated_in=APIVersion.V1_1
                )
            ]
        ))
        
        # Session management transformations
        self.register_transformation(SchemaTransformation(
            from_version=APIVersion.V1_0,
            to_version=APIVersion.V1_1,
            field_mappings=[
                FieldMapping(
                    old_field="session_budget",
                    new_field="ftns_budget",
                    deprecated_in=APIVersion.V1_1
                ),
                FieldMapping(
                    old_field="session_spent",
                    new_field="ftns_spent",
                    deprecated_in=APIVersion.V1_1
                )
            ]
        ))
    
    def register_transformation(self, transformation: SchemaTransformation):
        """Register a new transformation rule"""
        key = (transformation.from_version, transformation.to_version)
        self.transformations[key] = transformation
        logger.info(f"Registered transformation from {transformation.from_version.value} to {transformation.to_version.value}")
    
    def transform_request(self, data: Dict[str, Any], from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Transform request data from one version to another"""
        if from_version == to_version:
            return data
        
        # Find transformation path
        transformation = self._find_transformation_path(from_version, to_version)
        if not transformation:
            logger.warning(f"No transformation path found from {from_version.value} to {to_version.value}")
            return data
        
        return self._apply_transformation(data, transformation, direction="forward")
    
    def transform_response(self, data: Dict[str, Any], from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Transform response data from one version to another"""
        if from_version == to_version:
            return data
        
        # Find reverse transformation path
        transformation = self._find_transformation_path(to_version, from_version)
        if not transformation:
            logger.warning(f"No reverse transformation path found from {from_version.value} to {to_version.value}")
            return data
        
        return self._apply_transformation(data, transformation, direction="reverse")
    
    def _find_transformation_path(self, from_version: APIVersion, to_version: APIVersion) -> Optional[SchemaTransformation]:
        """Find transformation path between versions"""
        # Direct transformation
        direct_key = (from_version, to_version)
        if direct_key in self.transformations:
            return self.transformations[direct_key]
        
        # For now, only support direct transformations
        # In a more complex system, this would implement path finding through intermediate versions
        return None
    
    def _apply_transformation(self, data: Dict[str, Any], transformation: SchemaTransformation, direction: str = "forward") -> Dict[str, Any]:
        """Apply transformation rules to data"""
        if not isinstance(data, dict):
            return data
        
        transformed_data = data.copy()
        
        # Apply custom transformer first if available
        if transformation.custom_transformer and direction == "forward":
            transformed_data = transformation.custom_transformer(transformed_data)
        
        # Apply field mappings
        for mapping in transformation.field_mappings:
            if direction == "forward":
                self._apply_forward_mapping(transformed_data, mapping)
            else:
                self._apply_reverse_mapping(transformed_data, mapping)
        
        # Apply validation rules
        if transformation.validation_rules:
            for rule in transformation.validation_rules:
                if not rule(transformed_data):
                    logger.warning(f"Validation rule failed during transformation")
        
        return transformed_data
    
    def _apply_forward_mapping(self, data: Dict[str, Any], mapping: FieldMapping):
        """Apply forward field mapping (old -> new)"""
        if mapping.old_field in data:
            value = data.pop(mapping.old_field)
            
            # Apply transformer if available
            if mapping.transformer:
                try:
                    value = mapping.transformer(value)
                except Exception as e:
                    logger.error(f"Error applying transformer for {mapping.old_field}: {e}")
                    return
            
            data[mapping.new_field] = value
    
    def _apply_reverse_mapping(self, data: Dict[str, Any], mapping: FieldMapping):
        """Apply reverse field mapping (new -> old)"""
        if mapping.new_field in data:
            value = data.get(mapping.new_field)
            
            # Apply reverse transformer if available
            if mapping.reverse_transformer:
                try:
                    value = mapping.reverse_transformer(value)
                except Exception as e:
                    logger.error(f"Error applying reverse transformer for {mapping.new_field}: {e}")
                    return
            
            # Add old field while keeping new field for compatibility
            data[mapping.old_field] = value


class DeprecationManager:
    """Manages API deprecation warnings and lifecycle"""
    
    def __init__(self):
        self.deprecation_warnings: Dict[str, Dict[str, Any]] = {}
    
    def add_deprecation_warning(self, endpoint: str, version: APIVersion, message: str, 
                              sunset_date: Optional[datetime] = None, 
                              migration_guide: Optional[str] = None):
        """Add a deprecation warning for an endpoint"""
        self.deprecation_warnings[endpoint] = {
            "deprecated_in": version,
            "message": message,
            "sunset_date": sunset_date,
            "migration_guide": migration_guide,
            "first_warned": datetime.utcnow()
        }
    
    def get_deprecation_warning(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Get deprecation warning for an endpoint"""
        return self.deprecation_warnings.get(endpoint)
    
    def should_warn(self, endpoint: str, request_version: APIVersion) -> bool:
        """Check if deprecation warning should be shown"""
        warning = self.get_deprecation_warning(endpoint)
        if not warning:
            return False
        
        deprecated_version = warning["deprecated_in"]
        return request_version.value >= deprecated_version.value


class MigrationGuide:
    """Generates migration guides between API versions"""
    
    def __init__(self, compatibility_engine: CompatibilityEngine):
        self.engine = compatibility_engine
        self.guides: Dict[tuple, Dict[str, Any]] = {}
    
    def generate_migration_guide(self, from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Generate comprehensive migration guide"""
        guide_key = (from_version, to_version)
        
        if guide_key in self.guides:
            return self.guides[guide_key]
        
        # Find transformation
        transformation = self.engine._find_transformation_path(from_version, to_version)
        
        guide = {
            "from_version": from_version.value,
            "to_version": to_version.value,
            "migration_type": self._determine_migration_type(from_version, to_version),
            "breaking_changes": self._identify_breaking_changes(transformation),
            "field_changes": self._document_field_changes(transformation),
            "examples": self._generate_examples(transformation),
            "timeline": self._get_migration_timeline(from_version, to_version)
        }
        
        self.guides[guide_key] = guide
        return guide
    
    def _determine_migration_type(self, from_version: APIVersion, to_version: APIVersion) -> str:
        """Determine the type of migration required"""
        if from_version.major != to_version.major:
            return "major"  # Breaking changes expected
        elif from_version.minor != to_version.minor:
            return "minor"  # Backward compatible
        else:
            return "patch"  # Bug fixes only
    
    def _identify_breaking_changes(self, transformation: Optional[SchemaTransformation]) -> List[str]:
        """Identify breaking changes in the transformation"""
        if not transformation:
            return []
        
        breaking_changes = []
        for mapping in transformation.field_mappings:
            if mapping.removed_in:
                breaking_changes.append(f"Field '{mapping.old_field}' has been removed")
            elif mapping.deprecated_in:
                breaking_changes.append(f"Field '{mapping.old_field}' is deprecated, use '{mapping.new_field}' instead")
        
        return breaking_changes
    
    def _document_field_changes(self, transformation: Optional[SchemaTransformation]) -> List[Dict[str, Any]]:
        """Document field-level changes"""
        if not transformation:
            return []
        
        changes = []
        for mapping in transformation.field_mappings:
            change = {
                "old_field": mapping.old_field,
                "new_field": mapping.new_field,
                "required": mapping.required,
                "has_transformer": mapping.transformer is not None
            }
            
            if mapping.deprecated_in:
                change["deprecated_in"] = mapping.deprecated_in.value
            if mapping.removed_in:
                change["removed_in"] = mapping.removed_in.value
            
            changes.append(change)
        
        return changes
    
    def _generate_examples(self, transformation: Optional[SchemaTransformation]) -> Dict[str, Any]:
        """Generate before/after examples"""
        if not transformation:
            return {}
        
        # Generate example request/response transformations
        example_request_old = {
            "user_email": "researcher@university.edu",
            "user_password": "password123",
            "resource_price": 50.0,
            "token_balance": 1000.0
        }
        
        example_request_new = self.engine.transform_request(
            example_request_old, 
            transformation.from_version, 
            transformation.to_version
        )
        
        return {
            "request_transformation": {
                "before": example_request_old,
                "after": example_request_new
            }
        }
    
    def _get_migration_timeline(self, from_version: APIVersion, to_version: APIVersion) -> Dict[str, Any]:
        """Get timeline information for migration"""
        from_info = version_registry.get_version_info(from_version)
        to_info = version_registry.get_version_info(to_version)
        
        timeline = {}
        
        if from_info:
            timeline["from_release_date"] = from_info.release_date.isoformat()
            if from_info.deprecation_date:
                timeline["deprecation_date"] = from_info.deprecation_date.isoformat()
            if from_info.sunset_date:
                timeline["sunset_date"] = from_info.sunset_date.isoformat()
        
        if to_info:
            timeline["to_release_date"] = to_info.release_date.isoformat()
        
        return timeline


class CompatibilityMiddleware:
    """Middleware to handle backward compatibility transformations"""
    
    def __init__(self, compatibility_engine: CompatibilityEngine):
        self.engine = compatibility_engine
        self.deprecation_manager = DeprecationManager()
        self.migration_guide = MigrationGuide(compatibility_engine)
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request with compatibility transformations"""
        
        request_version = get_request_version(request)
        current_version = version_registry.get_latest_version()
        
        # Transform request if needed
        if hasattr(request, 'json') and request_version != current_version:
            try:
                # Get request body
                body = await request.body()
                if body:
                    request_data = json.loads(body)
                    
                    # Transform to current version
                    transformed_data = self.engine.transform_request(
                        request_data, request_version, current_version
                    )
                    
                    # Update request body (this is a simplified approach)
                    # In practice, you'd need to handle this more carefully
                    request._body = json.dumps(transformed_data).encode()
                    
            except Exception as e:
                logger.error(f"Error transforming request: {e}")
        
        # Process request
        response = await call_next(request)
        
        # Transform response if needed
        if request_version != current_version:
            try:
                # This is a simplified approach - in practice you'd intercept the response body
                response_content = response.body
                if response_content:
                    response_data = json.loads(response_content)
                    
                    # Transform back to requested version
                    transformed_response = self.engine.transform_response(
                        response_data, current_version, request_version
                    )
                    
                    # Update response
                    response._content = json.dumps(transformed_response).encode()
                    
            except Exception as e:
                logger.error(f"Error transforming response: {e}")
        
        # Add deprecation warnings
        self._add_deprecation_headers(request, response, request_version)
        
        return response
    
    def _add_deprecation_headers(self, request: Request, response: Response, request_version: APIVersion):
        """Add deprecation headers to response"""
        endpoint = request.url.path
        
        if self.deprecation_manager.should_warn(endpoint, request_version):
            warning = self.deprecation_manager.get_deprecation_warning(endpoint)
            response.headers["Deprecation"] = "true"
            response.headers["Deprecation-Message"] = warning["message"]
            
            if warning.get("sunset_date"):
                response.headers["Sunset"] = warning["sunset_date"].isoformat()
            
            if warning.get("migration_guide"):
                response.headers["Migration-Guide"] = warning["migration_guide"]


# Global instances
compatibility_engine = CompatibilityEngine()
compatibility_middleware = CompatibilityMiddleware(compatibility_engine)
deprecation_manager = DeprecationManager()
migration_guide_generator = MigrationGuide(compatibility_engine)


# Utility functions for endpoint developers
def add_backward_compatibility(old_field: str, new_field: str, version: APIVersion, 
                             transformer: Optional[Callable] = None):
    """Decorator to add backward compatibility for field changes"""
    def decorator(func):
        # Store compatibility information for documentation
        if not hasattr(func, '_compatibility_mappings'):
            func._compatibility_mappings = []
        
        func._compatibility_mappings.append({
            'old_field': old_field,
            'new_field': new_field,
            'version': version,
            'transformer': transformer
        })
        
        return func
    return decorator


def mark_deprecated(version: APIVersion, message: str, sunset_date: Optional[datetime] = None):
    """Decorator to mark an endpoint as deprecated"""
    def decorator(func):
        func._deprecated_in = version
        func._deprecation_message = message
        func._sunset_date = sunset_date
        
        # Register with deprecation manager
        endpoint_name = getattr(func, '__name__', 'unknown')
        deprecation_manager.add_deprecation_warning(
            endpoint_name, version, message, sunset_date
        )
        
        return func
    return decorator


def version_compatibility_check(request: Request, min_version: APIVersion, max_version: Optional[APIVersion] = None):
    """Check if request version is compatible with endpoint requirements"""
    request_version = get_request_version(request)
    
    if request_version.value < min_version.value:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "API_VERSION_TOO_OLD",
                "message": f"This endpoint requires API version {min_version.value} or newer",
                "current_version": request_version.value,
                "required_version": min_version.value,
                "migration_guide": f"/docs/migration/{request_version.value}-to-{min_version.value}"
            }
        )
    
    if max_version and request_version.value > max_version.value:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "API_VERSION_TOO_NEW",
                "message": f"This endpoint does not support API version {request_version.value}",
                "current_version": request_version.value,
                "max_supported_version": max_version.value
            }
        )