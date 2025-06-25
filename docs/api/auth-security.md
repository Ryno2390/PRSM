# Authentication & Security API

Handle access control, security policies, and authentication mechanisms within the PRSM ecosystem.

## 🎯 Overview

The Authentication & Security API provides comprehensive security management including user authentication, API key management, access control, security policies, and threat detection.

## 📋 Base URL

```
https://api.prsm.ai/v1/auth
```

## 🔐 Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.prsm.ai/v1/auth
```

## 🚀 Quick Start

### Generate API Key

```python
import prsm

# Using master credentials to create API key
client = prsm.Client(
    email="user@company.com",
    password="secure_password"
)

# Generate new API key
api_key = client.auth.create_api_key(
    name="Production API Key",
    scopes=["inference", "data:read", "monitoring:read"],
    expires_at="2024-12-31T23:59:59Z"
)

print(f"API Key: {api_key.key}")
```

## 📊 Endpoints

### POST /auth/login
Authenticate user with email/password.

**Request Body:**
```json
{
  "email": "user@company.com",
  "password": "secure_password",
  "mfa_token": "123456"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "rt_abc123def456...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user_id": "user_123",
  "permissions": ["inference", "data:read", "monitoring:read"],
  "session_id": "session_xyz789"
}
```

### POST /auth/refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "rt_abc123def456..."
}
```

### POST /auth/logout
Logout and invalidate tokens.

**Request Body:**
```json
{
  "session_id": "session_xyz789",
  "invalidate_all_sessions": false
}
```

### GET /auth/me
Get current user information and permissions.

**Response:**
```json
{
  "user_id": "user_123",
  "email": "user@company.com",
  "name": "John Doe",
  "organization": "Acme Corp",
  "role": "admin",
  "permissions": [
    "inference:create",
    "data:read",
    "data:write",
    "monitoring:read",
    "users:manage"
  ],
  "subscription": {
    "plan": "enterprise",
    "status": "active",
    "expires_at": "2024-12-31T23:59:59Z"
  },
  "last_login": "2024-01-15T10:30:00Z",
  "mfa_enabled": true
}
```

## 🔑 API Key Management

### POST /auth/api-keys
Create new API key.

**Request Body:**
```json
{
  "name": "Production API Key",
  "description": "API key for production environment",
  "scopes": ["inference", "data:read", "monitoring:read"],
  "expires_at": "2024-12-31T23:59:59Z",
  "rate_limit": {
    "requests_per_minute": 1000,
    "requests_per_day": 100000
  },
  "ip_restrictions": ["192.168.1.0/24", "10.0.0.0/8"],
  "environment": "production"
}
```

**Response:**
```json
{
  "key_id": "key_abc123",
  "api_key": "sk-proj-abc123def456ghi789...",
  "name": "Production API Key",
  "scopes": ["inference", "data:read", "monitoring:read"],
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-12-31T23:59:59Z",
  "last_used": null,
  "usage_count": 0
}
```

### GET /auth/api-keys
List all API keys for current user.

**Response:**
```json
{
  "api_keys": [
    {
      "key_id": "key_abc123",
      "name": "Production API Key",
      "scopes": ["inference", "data:read"],
      "created_at": "2024-01-15T10:30:00Z",
      "expires_at": "2024-12-31T23:59:59Z",
      "last_used": "2024-01-15T14:22:00Z",
      "usage_count": 2847,
      "status": "active"
    }
  ],
  "total_keys": 5,
  "active_keys": 4
}
```

### DELETE /auth/api-keys/{key_id}
Revoke API key.

### PUT /auth/api-keys/{key_id}
Update API key settings.

## 👥 User Management

### POST /auth/users
Create new user (admin only).

**Request Body:**
```json
{
  "email": "newuser@company.com",
  "name": "Jane Smith",
  "role": "developer",
  "permissions": ["inference", "data:read"],
  "send_invitation": true,
  "temporary_password": "temp_password_123"
}
```

### GET /auth/users
List users in organization.

### PUT /auth/users/{user_id}/permissions
Update user permissions.

**Request Body:**
```json
{
  "permissions": [
    "inference:create",
    "data:read",
    "data:write",
    "monitoring:read"
  ],
  "role": "senior_developer"
}
```

### POST /auth/users/{user_id}/disable
Disable user account.

## 🔐 Multi-Factor Authentication

### POST /auth/mfa/setup
Setup MFA for current user.

**Request Body:**
```json
{
  "method": "totp",
  "device_name": "iPhone 12"
}
```

**Response:**
```json
{
  "secret_key": "JBSWY3DPEHPK3PXP",
  "qr_code_url": "data:image/png;base64,iVBORw0KGgoAAAANSUh...",
  "backup_codes": [
    "12345678",
    "87654321",
    "13579246"
  ]
}
```

### POST /auth/mfa/verify
Verify MFA token.

**Request Body:**
```json
{
  "token": "123456",
  "remember_device": true
}
```

### GET /auth/mfa/methods
List available MFA methods.

**Response:**
```json
{
  "methods": [
    {
      "type": "totp",
      "name": "Authenticator App",
      "enabled": true,
      "device_name": "iPhone 12"
    },
    {
      "type": "sms",
      "name": "SMS",
      "enabled": false,
      "phone_number": "+1***-***-1234"
    }
  ]
}
```

## 🛡️ Access Control

### Role-Based Access Control (RBAC)

```python
# Define custom roles
role = client.auth.create_role(
    name="ml_engineer",
    description="Machine Learning Engineer",
    permissions=[
        "inference:create",
        "inference:read",
        "models:read",
        "models:deploy",
        "data:read",
        "monitoring:read"
    ]
)

# Assign role to user
client.auth.assign_role(
    user_id="user_123",
    role="ml_engineer"
)
```

### Permission Management

```python
# Check user permissions
permissions = client.auth.check_permissions(
    user_id="user_123",
    resource="dataset_456",
    action="read"
)

# Grant specific permissions
client.auth.grant_permission(
    user_id="user_123",
    resource="dataset_456",
    permissions=["read", "write"]
)
```

### Resource-Level Access

```python
# Set resource-specific access
client.auth.set_resource_access(
    resource_type="dataset",
    resource_id="ds_abc123",
    access_rules=[
        {
            "principal": "user_123",
            "permissions": ["read", "write"]
        },
        {
            "principal": "team:data_science",
            "permissions": ["read"]
        }
    ]
)
```

## 🔒 Security Policies

### Password Policies

```python
# Set organization password policy
password_policy = client.auth.set_password_policy(
    min_length=12,
    require_uppercase=True,
    require_lowercase=True,
    require_numbers=True,
    require_symbols=True,
    prevent_reuse=5,
    max_age_days=90,
    require_mfa=True
)
```

### Session Management

```python
# Configure session policies
session_policy = client.auth.set_session_policy(
    max_session_duration=28800,  # 8 hours
    idle_timeout=3600,  # 1 hour
    require_mfa_for_sensitive=True,
    concurrent_sessions_limit=3,
    force_logout_on_ip_change=True
)
```

### API Security Policies

```python
# Set API security policies
api_policy = client.auth.set_api_policy(
    rate_limiting={
        "default": {"requests_per_minute": 100},
        "premium": {"requests_per_minute": 1000}
    },
    ip_whitelisting_required=False,
    require_https=True,
    api_key_rotation_days=90
)
```

## 🚨 Security Monitoring

### Threat Detection

```python
# Configure threat detection
threat_detection = client.auth.configure_threat_detection(
    enable_anomaly_detection=True,
    suspicious_patterns=[
        "multiple_failed_logins",
        "unusual_api_usage",
        "geographic_anomaly",
        "privilege_escalation_attempt"
    ],
    alert_thresholds={
        "failed_login_attempts": 5,
        "api_rate_anomaly": 3.0,
        "geographic_distance_km": 1000
    }
)
```

### Security Events

```python
# Get security events
security_events = client.auth.get_security_events(
    timeframe="24h",
    event_types=["login", "failed_login", "api_key_created", "permission_changed"],
    severity=["medium", "high", "critical"]
)

# Report security incident
incident = client.auth.report_incident(
    type="suspicious_activity",
    description="Multiple failed login attempts from unusual location",
    severity="high",
    affected_users=["user_123"],
    evidence={
        "ip_address": "192.168.1.100",
        "user_agent": "curl/7.68.0",
        "attempts": 10
    }
)
```

### Audit Logging

```python
# Get comprehensive audit logs
audit_logs = client.auth.get_audit_logs(
    timeframe="7d",
    actions=["login", "api_key_usage", "permission_change", "data_access"],
    users=["user_123", "user_456"],
    include_ip_addresses=True
)

# Export audit logs for compliance
audit_export = client.auth.export_audit_logs(
    timeframe="90d",
    format="csv",
    include_all_metadata=True,
    encryption_required=True
)
```

## 🔐 Encryption and Data Protection

### Data Encryption

```python
# Configure encryption settings
encryption_config = client.auth.configure_encryption(
    data_at_rest={
        "algorithm": "AES-256-GCM",
        "key_rotation_days": 90,
        "customer_managed_keys": True
    },
    data_in_transit={
        "min_tls_version": "1.3",
        "cipher_suites": ["TLS_AES_256_GCM_SHA384"],
        "certificate_pinning": True
    }
)
```

### Key Management

```python
# Manage encryption keys
key = client.auth.create_encryption_key(
    purpose="data_encryption",
    algorithm="AES-256",
    auto_rotation=True,
    rotation_period_days=90
)

# Rotate encryption keys
rotation = client.auth.rotate_keys(
    key_ids=["key_123", "key_456"],
    force_immediate=False
)
```

## 🌐 OAuth and External Authentication

### OAuth Configuration

```python
# Configure OAuth providers
oauth_config = client.auth.configure_oauth(
    providers=[
        {
            "name": "google",
            "client_id": "your-google-client-id",
            "client_secret": "your-google-client-secret",
            "scopes": ["email", "profile"],
            "auto_create_users": True
        },
        {
            "name": "microsoft",
            "client_id": "your-microsoft-client-id",
            "client_secret": "your-microsoft-client-secret",
            "tenant_id": "your-tenant-id"
        }
    ]
)
```

### SAML Integration

```python
# Configure SAML SSO
saml_config = client.auth.configure_saml(
    entity_id="https://your-company.com",
    sso_url="https://your-idp.com/sso",
    certificate="-----BEGIN CERTIFICATE-----...",
    attribute_mapping={
        "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
        "name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
        "role": "http://schemas.microsoft.com/ws/2008/06/identity/claims/role"
    }
)
```

## 🔍 Compliance and Governance

### Compliance Controls

```python
# Configure compliance controls
compliance = client.auth.configure_compliance(
    standards=["SOC2", "GDPR", "HIPAA"],
    controls={
        "data_retention_days": 2555,  # 7 years
        "audit_log_retention_days": 2555,
        "automatic_user_deprovisioning": True,
        "regular_access_reviews": True,
        "encryption_at_rest_required": True
    }
)
```

### Privacy Controls

```python
# Implement privacy controls
privacy_controls = client.auth.configure_privacy(
    data_minimization=True,
    purpose_limitation=True,
    retention_policies={
        "user_data": "7_years",
        "logs": "1_year",
        "analytics": "2_years"
    },
    consent_management=True,
    right_to_deletion=True
)
```

## 📊 Security Reporting

### Security Dashboard

```python
# Get security dashboard data
security_dashboard = client.auth.security_dashboard(
    timeframe="30d",
    include_trends=True
)

print(f"Total logins: {security_dashboard.total_logins}")
print(f"Failed login rate: {security_dashboard.failed_login_rate}%")
print(f"Active sessions: {security_dashboard.active_sessions}")
print(f"Security incidents: {security_dashboard.security_incidents}")
```

### Compliance Reports

```python
# Generate compliance reports
compliance_report = client.auth.generate_compliance_report(
    standard="SOC2",
    timeframe="quarter",
    include_evidence=True,
    controls=[
        "access_control",
        "authentication",
        "encryption",
        "audit_logging"
    ]
)
```

## 🧪 Security Testing

### Penetration Testing

```python
# Schedule security assessments
security_assessment = client.auth.schedule_security_assessment(
    assessment_type="penetration_test",
    scope=["api_endpoints", "authentication", "authorization"],
    frequency="quarterly",
    external_provider="security_firm_xyz"
)
```

### Vulnerability Scanning

```python
# Configure vulnerability scanning
vuln_scanning = client.auth.configure_vulnerability_scanning(
    scan_frequency="weekly",
    scan_scope=["api", "infrastructure", "dependencies"],
    auto_remediation=True,
    severity_threshold="medium"
)
```

## 📞 Support

- **Security Issues**: security@prsm.ai
- **Access Problems**: access-support@prsm.ai
- **Compliance**: compliance@prsm.ai
- **Emergency**: security-emergency@prsm.ai