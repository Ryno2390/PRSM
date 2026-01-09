"""
PRSM API Version-Specific OpenAPI Schemas
Generates different OpenAPI specifications for different API versions
"""

from typing import Dict, Any, List, Optional, Set
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import copy
import json

from .versioning import APIVersion, version_registry, VersionInfo
from .openapi_config import API_TAGS_METADATA, SECURITY_SCHEMES, COMMON_RESPONSES


class VersionedOpenAPIGenerator:
    """Generates OpenAPI specifications for different API versions"""
    
    def __init__(self):
        self.version_schemas: Dict[APIVersion, Dict[str, Any]] = {}
        self.base_schema_cache: Optional[Dict[str, Any]] = None
    
    def generate_versioned_schema(self, app: FastAPI, version: APIVersion) -> Dict[str, Any]:
        """Generate OpenAPI schema for a specific API version"""
        
        if version in self.version_schemas:
            return self.version_schemas[version]
        
        # Get base schema
        base_schema = self._get_base_schema(app)
        
        # Create version-specific schema
        versioned_schema = self._create_version_specific_schema(base_schema, version)
        
        # Cache the schema
        self.version_schemas[version] = versioned_schema
        
        return versioned_schema
    
    def _get_base_schema(self, app: FastAPI) -> Dict[str, Any]:
        """Get the base OpenAPI schema"""
        if self.base_schema_cache is None:
            self.base_schema_cache = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
                tags=API_TAGS_METADATA
            )
            
            # Add security schemes and common responses
            if "components" not in self.base_schema_cache:
                self.base_schema_cache["components"] = {}
            
            self.base_schema_cache["components"]["securitySchemes"] = SECURITY_SCHEMES
            
            if "responses" not in self.base_schema_cache["components"]:
                self.base_schema_cache["components"]["responses"] = {}
            self.base_schema_cache["components"]["responses"].update(COMMON_RESPONSES)
        
        return copy.deepcopy(self.base_schema_cache)
    
    def _create_version_specific_schema(self, base_schema: Dict[str, Any], version: APIVersion) -> Dict[str, Any]:
        """Create a version-specific OpenAPI schema"""
        
        # Update info section
        version_info = version_registry.get_version_info(version)
        self._update_info_section(base_schema, version, version_info)
        
        # Filter paths based on version
        self._filter_paths_by_version(base_schema, version)
        
        # Update schemas based on version
        self._update_schemas_for_version(base_schema, version)
        
        # Add version-specific examples
        self._add_version_specific_examples(base_schema, version)
        
        # Add deprecation information
        self._add_deprecation_info(base_schema, version, version_info)
        
        return base_schema
    
    def _update_info_section(self, schema: Dict[str, Any], version: APIVersion, version_info: Optional[VersionInfo]):
        """Update the info section for the specific version"""
        
        schema["info"]["version"] = version.value
        schema["info"]["title"] = f"PRSM API v{version.value}"
        
        # Add version-specific description
        version_description = self._get_version_description(version, version_info)
        schema["info"]["description"] = f"{schema['info']['description']}\n\n{version_description}"
        
        # Update servers to include version in path
        if "servers" in schema:
            for server in schema["servers"]:
                # Add version-specific server paths
                original_url = server["url"]
                server["url"] = f"{original_url}/v{version.major}"
        
        # Add version-specific external docs
        schema["externalDocs"] = {
            "description": f"PRSM API v{version.value} Documentation",
            "url": f"https://docs.prsm.org/api/v{version.value}"
        }
    
    def _get_version_description(self, version: APIVersion, version_info: Optional[VersionInfo]) -> str:
        """Get version-specific description"""
        
        description_parts = [f"## API Version {version.value}"]
        
        if version_info:
            if version_info.release_date:
                description_parts.append(f"**Released:** {version_info.release_date.strftime('%B %d, %Y')}")
            
            description_parts.append(f"**Status:** {version_info.status.title()}")
            
            if version_info.is_deprecated:
                description_parts.append("⚠️  **This version is deprecated**")
                if version_info.deprecation_date:
                    description_parts.append(f"**Deprecated:** {version_info.deprecation_date.strftime('%B %d, %Y')}")
                if version_info.sunset_date:
                    description_parts.append(f"**Sunset Date:** {version_info.sunset_date.strftime('%B %d, %Y')}")
                if version_info.migration_guide_url:
                    description_parts.append(f"**Migration Guide:** {version_info.migration_guide_url}")
            
            if version_info.changelog:
                description_parts.append("### What's New")
                for change in version_info.changelog:
                    description_parts.append(f"- {change}")
            
            if version_info.breaking_changes:
                description_parts.append("### Breaking Changes")
                for change in version_info.breaking_changes:
                    description_parts.append(f"- ⚠️  {change}")
        
        return "\n".join(description_parts)
    
    def _filter_paths_by_version(self, schema: Dict[str, Any], version: APIVersion):
        """Filter API paths based on version availability"""
        
        if "paths" not in schema:
            return
        
        paths_to_remove = []
        
        for path, path_item in schema["paths"].items():
            # Check if path should be available in this version
            if not self._is_path_available_in_version(path, path_item, version):
                paths_to_remove.append(path)
            else:
                # Filter operations within the path
                self._filter_operations_by_version(path_item, version)
        
        # Remove unavailable paths
        for path in paths_to_remove:
            del schema["paths"][path]
    
    def _is_path_available_in_version(self, path: str, path_item: Dict[str, Any], version: APIVersion) -> bool:
        """Check if a path should be available in a specific version"""
        
        # Check if any operation in the path is available in this version
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch", "head", "options"]:
                if self._is_operation_available_in_version(operation, version):
                    return True
        
        return False
    
    def _is_operation_available_in_version(self, operation: Dict[str, Any], version: APIVersion) -> bool:
        """Check if an operation should be available in a specific version"""
        
        # Check for version-specific tags
        operation_tags = operation.get("tags", [])
        
        # Check for version annotations (if we had them)
        # For now, assume all operations are available unless specifically marked
        
        # Future: Check operation metadata for version availability
        min_version = operation.get("x-min-version")
        max_version = operation.get("x-max-version")
        
        if min_version and version.value < min_version:
            return False
        
        if max_version and version.value > max_version:
            return False
        
        return True
    
    def _filter_operations_by_version(self, path_item: Dict[str, Any], version: APIVersion):
        """Filter operations within a path based on version"""
        
        operations_to_remove = []
        
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch", "head", "options"]:
                if not self._is_operation_available_in_version(operation, version):
                    operations_to_remove.append(method)
                else:
                    # Update operation for this version
                    self._update_operation_for_version(operation, version)
        
        # Remove unavailable operations
        for method in operations_to_remove:
            del path_item[method]
    
    def _update_operation_for_version(self, operation: Dict[str, Any], version: APIVersion):
        """Update an operation for a specific version"""
        
        # Add version-specific examples
        self._add_operation_examples_for_version(operation, version)
        
        # Update parameter schemas if needed
        if "parameters" in operation:
            for param in operation["parameters"]:
                self._update_parameter_for_version(param, version)
        
        # Update request body schemas if needed
        if "requestBody" in operation:
            self._update_request_body_for_version(operation["requestBody"], version)
        
        # Update response schemas if needed
        if "responses" in operation:
            for response_code, response in operation["responses"].items():
                self._update_response_for_version(response, version)
    
    def _update_schemas_for_version(self, schema: Dict[str, Any], version: APIVersion):
        """Update component schemas for a specific version"""
        
        if "components" not in schema or "schemas" not in schema["components"]:
            return
        
        schemas = schema["components"]["schemas"]
        
        # Update schemas based on version-specific field mappings
        if version == APIVersion.V1_0:
            self._add_v1_0_schema_compatibility(schemas)
        elif version == APIVersion.V1_1:
            self._add_v1_1_schema_enhancements(schemas)
    
    def _add_v1_0_schema_compatibility(self, schemas: Dict[str, Any]):
        """Add v1.0 compatibility to schemas"""
        
        # Add legacy field names for backward compatibility
        if "LoginRequest" in schemas:
            login_schema = schemas["LoginRequest"]
            if "properties" in login_schema:
                # Add deprecated field names
                props = login_schema["properties"]
                if "email" in props:
                    props["user_email"] = {
                        **props["email"],
                        "deprecated": True,
                        "description": "Deprecated: Use 'email' instead"
                    }
                if "password" in props:
                    props["user_password"] = {
                        **props["password"],
                        "deprecated": True,
                        "description": "Deprecated: Use 'password' instead"
                    }
        
        # Similar updates for other schemas
        if "FTNSBalance" in schemas:
            balance_schema = schemas["FTNSBalance"]
            if "properties" in balance_schema:
                props = balance_schema["properties"]
                if "available_balance" in props:
                    props["token_balance"] = {
                        **props["available_balance"],
                        "deprecated": True,
                        "description": "Deprecated: Use 'available_balance' instead"
                    }
                if "locked_balance" in props:
                    props["locked_tokens"] = {
                        **props["locked_balance"],
                        "deprecated": True,
                        "description": "Deprecated: Use 'locked_balance' instead"
                    }
    
    def _add_v1_1_schema_enhancements(self, schemas: Dict[str, Any]):
        """Add v1.1 enhancements to schemas"""
        
        # Add new fields and improved validation for v1.1
        if "MarketplaceResource" in schemas:
            resource_schema = schemas["MarketplaceResource"]
            if "properties" in resource_schema:
                props = resource_schema["properties"]
                
                # Add enhanced metadata fields
                props["metadata"] = {
                    "type": "object",
                    "description": "Enhanced resource metadata (v1.1+)",
                    "properties": {
                        "license": {"type": "string"},
                        "version": {"type": "string"},
                        "size_mb": {"type": "number"},
                        "framework": {"type": "string"}
                    }
                }
    
    def _add_version_specific_examples(self, schema: Dict[str, Any], version: APIVersion):
        """Add version-specific examples to the schema"""
        
        # Add version-specific code examples
        schema["x-code-samples"] = self._get_version_code_samples(version)
        
        # Add version-specific response examples
        if "paths" in schema:
            for path, path_item in schema["paths"].items():
                for method, operation in path_item.items():
                    if method in ["get", "post", "put", "delete", "patch"]:
                        self._add_operation_examples_for_version(operation, version)
    
    def _get_version_code_samples(self, version: APIVersion) -> List[Dict[str, Any]]:
        """Get code samples for a specific version"""
        
        samples = []
        
        # Python example
        samples.append({
            "lang": "python",
            "label": f"Python SDK v{version.value}",
            "source": f'''from prsm_sdk import PRSMClient

# Initialize client for v{version.value}
client = PRSMClient(
    api_version="{version.value}",
    base_url="https://api.prsm.org"
)

# Authenticate
await client.auth.login("user@example.com", "password")

# Search marketplace
resources = await client.marketplace.search(
    query="machine learning",
    resource_type="ai_model"
)'''
        })
        
        # JavaScript example
        samples.append({
            "lang": "javascript",
            "label": f"JavaScript SDK v{version.value}",
            "source": f'''import {{ PRSMClient }} from '@prsm/js-sdk';

// Initialize client for v{version.value}
const client = new PRSMClient({{
  apiVersion: '{version.value}',
  baseURL: 'https://api.prsm.org'
}});

// Authenticate
await client.auth.login('user@example.com', 'password');

// Search marketplace
const resources = await client.marketplace.search({{
  query: 'machine learning',
  resourceType: 'ai_model'
}});'''
        })
        
        # cURL example
        samples.append({
            "lang": "shell",
            "label": f"cURL v{version.value}",
            "source": f'''# Login (v{version.value})
curl -X POST "https://api.prsm.org/api/v{version.major}/auth/login" \\
  -H "Content-Type: application/json" \\
  -H "API-Version: {version.value}" \\
  -d '{{"email": "user@example.com", "password": "password"}}'

# Search marketplace
curl -X GET "https://api.prsm.org/api/v{version.major}/marketplace/resources?query=machine%20learning" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "API-Version: {version.value}"'''
        })
        
        return samples
    
    def _add_operation_examples_for_version(self, operation: Dict[str, Any], version: APIVersion):
        """Add version-specific examples to an operation"""
        
        # Add request examples based on version
        if "requestBody" in operation:
            self._add_request_examples_for_version(operation["requestBody"], version)
        
        # Add response examples based on version
        if "responses" in operation:
            for response in operation["responses"].values():
                self._add_response_examples_for_version(response, version)
    
    def _add_request_examples_for_version(self, request_body: Dict[str, Any], version: APIVersion):
        """Add version-specific request examples"""
        
        if "content" not in request_body:
            return
        
        for media_type, content in request_body["content"].items():
            if "examples" not in content:
                content["examples"] = {}
            
            # Add version-specific examples
            if version == APIVersion.V1_0:
                content["examples"][f"v{version.value}_example"] = {
                    "summary": f"v{version.value} format",
                    "description": f"Request format for API version {version.value}",
                    "value": self._get_v1_0_request_example()
                }
    
    def _add_response_examples_for_version(self, response: Dict[str, Any], version: APIVersion):
        """Add version-specific response examples"""
        
        if "content" not in response:
            return
        
        for media_type, content in response["content"].items():
            if "examples" not in content:
                content["examples"] = {}
            
            # Add version-specific examples
            content["examples"][f"v{version.value}_example"] = {
                "summary": f"v{version.value} format",
                "description": f"Response format for API version {version.value}",
                "value": self._get_version_response_example(version)
            }
    
    def _get_v1_0_request_example(self) -> Dict[str, Any]:
        """Get v1.0 specific request example"""
        return {
            "user_email": "researcher@university.edu",
            "user_password": "password123"
        }
    
    def _get_version_response_example(self, version: APIVersion) -> Dict[str, Any]:
        """Get version-specific response example"""
        if version == APIVersion.V1_0:
            return {
                "success": True,
                "message": "Operation successful",
                "timestamp": "2024-01-15T10:00:00Z",
                "token_balance": 1000.0,  # v1.0 field name
                "locked_tokens": 100.0   # v1.0 field name
            }
        else:
            return {
                "success": True,
                "message": "Operation successful",
                "timestamp": "2024-01-15T10:00:00Z",
                "available_balance": 1000.0,  # v1.1+ field name
                "locked_balance": 100.0       # v1.1+ field name
            }
    
    def _add_deprecation_info(self, schema: Dict[str, Any], version: APIVersion, version_info: Optional[VersionInfo]):
        """Add deprecation information to the schema"""
        
        if not version_info or not version_info.is_deprecated:
            return
        
        # Add deprecation notice to description
        deprecation_notice = "\n\n---\n⚠️  **DEPRECATION NOTICE**\n\n"
        deprecation_notice += f"This API version ({version.value}) is deprecated"
        
        if version_info.deprecation_date:
            deprecation_notice += f" as of {version_info.deprecation_date.strftime('%B %d, %Y')}"
        
        if version_info.sunset_date:
            deprecation_notice += f" and will be discontinued on {version_info.sunset_date.strftime('%B %d, %Y')}"
        
        deprecation_notice += ".\n\n"
        
        if version_info.migration_guide_url:
            deprecation_notice += f"Please refer to the [migration guide]({version_info.migration_guide_url}) for upgrade instructions.\n\n"
        
        latest_version = version_registry.get_latest_version()
        deprecation_notice += f"We recommend upgrading to API version {latest_version.value}."
        
        schema["info"]["description"] += deprecation_notice
        
        # Add deprecation headers to all responses
        if "paths" in schema:
            for path_item in schema["paths"].values():
                for operation in path_item.values():
                    if isinstance(operation, dict) and "responses" in operation:
                        for response in operation["responses"].values():
                            if isinstance(response, dict):
                                if "headers" not in response:
                                    response["headers"] = {}
                                
                                response["headers"]["Deprecation"] = {
                                    "description": "Indicates that the API version is deprecated",
                                    "schema": {"type": "string", "example": "true"}
                                }
                                
                                if version_info.sunset_date:
                                    response["headers"]["Sunset"] = {
                                        "description": "Date when this API version will be discontinued",
                                        "schema": {"type": "string", "format": "date-time"}
                                    }
    
    def _update_parameter_for_version(self, param: Dict[str, Any], version: APIVersion):
        """Update parameter definition for specific version"""
        pass  # Implementation would depend on specific parameter changes
    
    def _update_request_body_for_version(self, request_body: Dict[str, Any], version: APIVersion):
        """Update request body definition for specific version"""
        pass  # Implementation would depend on specific schema changes
    
    def _update_response_for_version(self, response: Dict[str, Any], version: APIVersion):
        """Update response definition for specific version"""
        pass  # Implementation would depend on specific response changes


# Global versioned OpenAPI generator
versioned_openapi_generator = VersionedOpenAPIGenerator()


def get_versioned_openapi_schema(app: FastAPI, version: APIVersion) -> Dict[str, Any]:
    """Get OpenAPI schema for a specific API version"""
    return versioned_openapi_generator.generate_versioned_schema(app, version)


def create_version_specific_docs_endpoints(app: FastAPI):
    """Create version-specific documentation endpoints"""
    
    @app.get("/openapi/v{version}.json", include_in_schema=False)
    async def get_version_openapi(version: str):
        """Get OpenAPI schema for specific version"""
        try:
            api_version = APIVersion.from_string(version)
            schema = get_versioned_openapi_schema(app, api_version)
            return schema
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"API version {version} not found"
            )
    
    @app.get("/docs/v{version}", include_in_schema=False)
    async def get_version_docs(version: str):
        """Get Swagger UI for specific version"""
        try:
            api_version = APIVersion.from_string(version)
            
            # Return HTML page with version-specific Swagger UI
            from fastapi.responses import HTMLResponse
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>PRSM API v{version} Documentation</title>
                <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui.css" />
                <style>
                    .swagger-ui .topbar {{ display: none; }}
                    .version-banner {{
                        background: #1f4788;
                        color: white;
                        padding: 10px;
                        text-align: center;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="version-banner">
                    PRSM API Documentation - Version {version}
                </div>
                <div id="swagger-ui"></div>
                <script src="https://unpkg.com/swagger-ui-dist@3.52.5/swagger-ui-bundle.js"></script>
                <script>
                    SwaggerUIBundle({{
                        url: '/openapi/v{version}.json',
                        dom_id: '#swagger-ui',
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIBundle.presets.standalone
                        ]
                    }});
                </script>
            </body>
            </html>
            """
            
            return HTMLResponse(content=html_content)
            
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"API version {version} documentation not found"
            )