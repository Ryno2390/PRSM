# PRSM API Versioning & Migration Guide

## Table of Contents
- [Overview](#overview)
- [Version Strategy](#version-strategy)
- [Specifying API Versions](#specifying-api-versions)
- [Backward Compatibility](#backward-compatibility)
- [Migration Process](#migration-process)
- [Deprecation Policy](#deprecation-policy)
- [Version-Specific Documentation](#version-specific-documentation)
- [Best Practices](#best-practices)

## Overview

The PRSM API uses semantic versioning to manage changes and ensure backward compatibility. Our versioning system provides:

- **Predictable Evolution**: Clear versioning strategy with well-defined compatibility rules
- **Smooth Migrations**: Automated compatibility layer for seamless transitions
- **Comprehensive Documentation**: Version-specific documentation and migration guides
- **Deprecation Management**: Structured deprecation process with advance notice

## Version Strategy

### Version Format
PRSM API versions follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes that require code updates
- **MINOR**: New features that maintain backward compatibility
- **PATCH**: Bug fixes and improvements (not exposed in API versioning)

### Current Versions

| Version | Status | Release Date | End of Life | Description |
|---------|--------|--------------|-------------|-------------|
| **v1.0** | ✅ Active | Jan 15, 2024 | TBD | Initial stable release |
| **v1.1** | ✅ Active | Apr 1, 2024 | TBD | Enhanced features and improvements |
| **v2.0** | 🚧 Planned | Jul 1, 2024 | - | Major architectural improvements |

### Compatibility Promise

- **Backward Compatibility**: Minor versions maintain full backward compatibility
- **Migration Support**: Automatic transformation between compatible versions
- **Deprecation Notice**: 6-month minimum notice before breaking changes
- **Sunset Timeline**: 12-month support after deprecation announcement

## Specifying API Versions

### 1. URL Path (Recommended)
```bash
curl https://api.prsm.org/api/v1/marketplace/resources
curl https://api.prsm.org/api/v1.1/marketplace/resources
curl https://api.prsm.org/api/v2/marketplace/resources
```

### 2. API-Version Header
```bash
curl -H "API-Version: 1.1" https://api.prsm.org/api/marketplace/resources
```

### 3. Accept Header (Content Negotiation)
```bash
curl -H "Accept: application/vnd.prsm.v1.1+json" https://api.prsm.org/api/marketplace/resources
```

### 4. Query Parameter
```bash
curl https://api.prsm.org/api/marketplace/resources?version=1.1
```

**Priority Order**: URL Path > API-Version Header > Accept Header > Query Parameter > Default

### SDK Examples

#### Python SDK
```python
from prsm_sdk import PRSMClient

# Specify version during initialization
client = PRSMClient(
    api_version="1.1",
    base_url="https://api.prsm.org"
)

# Or set version per request
response = await client.marketplace.search(
    query="machine learning",
    api_version="1.1"
)
```

#### JavaScript SDK
```javascript
import { PRSMClient } from '@prsm/js-sdk';

// Global version setting
const client = new PRSMClient({
  apiVersion: '1.1',
  baseURL: 'https://api.prsm.org'
});

// Per-request version
const resources = await client.marketplace.search({
  query: 'machine learning'
}, { apiVersion: '1.1' });
```

## Backward Compatibility

### Automatic Transformation

The PRSM API automatically transforms requests and responses between compatible versions:

```python
# Your v1.0 request (old field names)
{
    "user_email": "researcher@university.edu",
    "user_password": "password123",
    "token_balance": 1000.0
}

# Automatically transformed to v1.1 format internally
{
    "email": "researcher@university.edu",
    "password": "password123",
    "available_balance": 1000.0
}
```

### Response Headers

Version information is included in all API responses:

```http
HTTP/1.1 200 OK
API-Version: 1.1
API-Version-Source: header
API-Supported-Versions: 1.0,1.1,2.0
API-Deprecation-Warning: false
Content-Type: application/json
```

### Deprecation Headers

When using deprecated versions:

```http
HTTP/1.1 200 OK
API-Version: 1.0
API-Deprecation-Warning: true
API-Deprecation-Date: 2024-07-01T00:00:00Z
API-Sunset-Date: 2025-01-01T00:00:00Z
API-Migration-Guide: https://docs.prsm.org/migration/v1.0-to-v1.1
```

## Migration Process

### 1. Check Current Version Usage

```bash
# Get your current API usage
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.prsm.org/api/migration/paths?from_version=1.0
```

### 2. Review Migration Checklist

```bash
# Get detailed migration checklist
curl https://api.prsm.org/api/migration/checklist?from_version=1.0&to_version=1.1
```

### 3. Identify Breaking Changes

```bash
# Get breaking changes summary
curl https://api.prsm.org/api/migration/breaking-changes?from_version=1.0&to_version=1.1
```

### 4. Access Migration Guide

Visit the interactive migration guide:
```
https://api.prsm.org/migration/guide/1.0/to/1.1
```

### Example Migration: v1.0 to v1.1

#### Step 1: Update Authentication Fields
```python
# Before (v1.0)
login_data = {
    "user_email": "researcher@university.edu",
    "user_password": "password123"
}

# After (v1.1)  
login_data = {
    "email": "researcher@university.edu",
    "password": "password123"
}
```

#### Step 2: Update Balance Field Names
```python
# Before (v1.0)
balance = response_data["token_balance"]
locked = response_data["locked_tokens"]

# After (v1.1)
balance = response_data["available_balance"] 
locked = response_data["locked_balance"]
```

#### Step 3: Handle Enhanced Error Responses
```python
# v1.1 includes more detailed error information
try:
    response = await client.marketplace.purchase(resource_id)
except PRSMAPIError as e:
    if e.error_code == "INSUFFICIENT_BALANCE":
        required = e.details["required_amount"]
        available = e.details["available_balance"]
        print(f"Need {required} FTNS, but only have {available}")
```

#### Step 4: Update SDK Version
```bash
# Python
pip install --upgrade prsm-sdk>=1.1.0

# JavaScript
npm install @prsm/js-sdk@^1.1.0
```

### Testing Your Migration

1. **Test in Staging**: Always test migrations in a staging environment first
2. **Gradual Rollout**: Use feature flags to gradually migrate traffic
3. **Monitor Metrics**: Watch error rates and response times during migration
4. **Rollback Plan**: Keep v1.0 compatibility available for quick rollback

## Deprecation Policy

### Deprecation Timeline

1. **Announcement**: New version released, previous version marked as deprecated
2. **Warning Period**: 6 months minimum of deprecation warnings
3. **Sunset Notice**: 3 months advance notice before discontinuation
4. **End of Life**: Version no longer supported

### Deprecation Process

#### Phase 1: Deprecation Announcement (Month 0)
- Version marked as deprecated in documentation
- Deprecation headers added to responses
- Migration guides published

#### Phase 2: Active Deprecation (Months 1-6)
- Regular deprecation warnings in responses
- Email notifications to active users
- Enhanced migration support

#### Phase 3: Sunset Warning (Months 6-9)
- Increased warning frequency
- Direct outreach to high-usage clients
- Final migration assistance

#### Phase 4: End of Life (Month 9+)
- Version no longer accepts requests
- HTTP 410 Gone responses
- Automatic redirect to latest documentation

### Staying Informed

1. **API Headers**: Monitor deprecation headers in responses
2. **Email Notifications**: Subscribe to API announcements
3. **Changelog**: Review release notes regularly
4. **Developer Dashboard**: Check version usage analytics

## Version-Specific Documentation

### Interactive Documentation

Each version has dedicated interactive documentation:

- **v1.0**: https://api.prsm.org/docs/v1.0
- **v1.1**: https://api.prsm.org/docs/v1.1  
- **v2.0**: https://api.prsm.org/docs/v2.0

### OpenAPI Specifications

Download version-specific OpenAPI specs:

- **v1.0**: https://api.prsm.org/openapi/v1.0.json
- **v1.1**: https://api.prsm.org/openapi/v1.1.json
- **v2.0**: https://api.prsm.org/openapi/v2.0.json

### Migration Guides

Interactive migration guides with checklists:

- **v1.0 → v1.1**: https://api.prsm.org/migration/guide/1.0/to/1.1
- **v1.1 → v2.0**: https://api.prsm.org/migration/guide/1.1/to/2.0

## Best Practices

### For API Consumers

#### 1. Explicit Version Specification
```python
# ✅ Good: Always specify version explicitly
client = PRSMClient(api_version="1.1")

# ❌ Avoid: Relying on default version
client = PRSMClient()  # Uses whatever default is set
```

#### 2. Version Headers Monitoring
```python
import requests

response = requests.get(
    "https://api.prsm.org/api/v1/marketplace/resources",
    headers={"Authorization": "Bearer TOKEN"}
)

# Check for deprecation warnings
if response.headers.get("API-Deprecation-Warning") == "true":
    migration_guide = response.headers.get("API-Migration-Guide")
    print(f"API version deprecated. Migration guide: {migration_guide}")
```

#### 3. Error Handling for Version Issues
```python
try:
    response = await client.api_call()
except PRSMAPIError as e:
    if e.error_code == "API_VERSION_TOO_OLD":
        print(f"Upgrade required. Current: {e.current_version}")
        print(f"Required: {e.required_version}")
    elif e.error_code == "API_VERSION_SUNSET":
        print(f"Version no longer supported: {e.sunset_date}")
```

#### 4. Configuration Management
```python
# Use environment variables for version management
import os

API_VERSION = os.getenv("PRSM_API_VERSION", "1.1")
client = PRSMClient(api_version=API_VERSION)
```

### For Integration Planning

#### 1. Version Lifecycle Tracking
```python
# Monitor version lifecycle in your deployment pipeline
def check_api_version_status():
    response = requests.get("https://api.prsm.org/api/versions")
    versions = response.json()["supported_versions"]
    
    current_version = os.getenv("PRSM_API_VERSION")
    version_info = next(v for v in versions if v["version"] == current_version)
    
    if version_info["status"] == "deprecated":
        days_until_sunset = calculate_days_until(version_info["sunset_date"])
        if days_until_sunset < 30:
            alert_ops_team(f"API version sunset in {days_until_sunset} days")
```

#### 2. Automated Migration Testing
```python
# Test your code against multiple API versions
API_VERSIONS_TO_TEST = ["1.0", "1.1", "2.0"]

@pytest.mark.parametrize("api_version", API_VERSIONS_TO_TEST)
def test_marketplace_search(api_version):
    client = PRSMClient(api_version=api_version)
    response = client.marketplace.search(query="test")
    assert response.success
```

#### 3. Gradual Migration Strategy
```python
# Feature flag for gradual API version migration
def get_api_version_for_user(user_id):
    # Start with 10% of users on new version
    if feature_flag("api_v1_1_migration", user_id, rollout_percentage=10):
        return "1.1"
    return "1.0"
```

### Common Pitfalls to Avoid

1. **Hard-coding Version Numbers**: Use configuration for version management
2. **Ignoring Deprecation Warnings**: Monitor headers and notifications
3. **Skipping Version Testing**: Test against multiple versions during development
4. **Last-minute Migrations**: Start migration planning early in deprecation cycle
5. **Assuming Compatibility**: Always review breaking changes documentation

---

## Support and Resources

- **📚 Version Documentation**: [https://docs.prsm.org/versioning](https://docs.prsm.org/versioning)
- **🔄 Migration Tools**: [https://tools.prsm.org/migration](https://tools.prsm.org/migration)
- **📧 Version Announcements**: [api-announcements@prsm.org](mailto:api-announcements@prsm.org)
- **💬 Developer Community**: [https://community.prsm.org/api-versioning](https://community.prsm.org/api-versioning)
- **🆘 Migration Support**: [migration-support@prsm.org](mailto:migration-support@prsm.org)

For enterprise customers with custom SLAs, contact [enterprise@prsm.org](mailto:enterprise@prsm.org) for dedicated migration assistance.