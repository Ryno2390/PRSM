# Security Hardening: Request Size Limits & Input Sanitization

## Overview

PRSM implements comprehensive security hardening through request size limits, input sanitization, and protection against common web application attacks. This system prevents DoS attacks, injection vulnerabilities, and malicious input while maintaining application functionality.

## Security Features Implemented

### 🛡️ Request Size Limits & Rate Limiting

**Purpose**: Prevent DoS attacks and resource exhaustion
**Implementation**: Multi-layer protection with configurable limits

#### HTTP Request Protection
- **Default body size limit**: 16MB for most endpoints
- **Endpoint-specific limits**: Customized per API endpoint
- **Rate limiting**: IP-based protection with different tiers
- **Request timeout**: Protection against slow attacks

#### WebSocket Protection
- **Message size limits**: 256KB per WebSocket message
- **Rate limiting**: 60 messages per minute per connection
- **Connection timeouts**: 5-minute idle connection limit
- **Memory monitoring**: Tracks total bytes transferred

### 🧹 Input Sanitization & Validation

**Purpose**: Prevent XSS, SQL injection, and other injection attacks
**Implementation**: Multi-stage sanitization with allowlist approach

#### HTML Sanitization
- **Allowlist approach**: Only permitted HTML tags allowed
- **Attribute filtering**: Strict control over HTML attributes
- **Script prevention**: Automatic removal of script tags and events
- **Link validation**: URL scheme validation and filtering

#### SQL Injection Prevention
- **Pattern detection**: Regex-based detection of SQL injection patterns
- **Input validation**: Parameterized query enforcement
- **Error handling**: Secure error responses without information leakage

#### JSON Structure Protection
- **Depth limiting**: Prevention of deeply nested JSON attacks
- **Key count limits**: Protection against object key flooding
- **String length limits**: Prevention of memory exhaustion
- **Type validation**: Strict type checking for JSON values

## Configuration

### Request Limits Configuration

```python
# Default configuration in RequestLimitsConfig
default_max_body_size: 16MB
endpoint_body_limits: {
    '/api/v1/sessions': 1MB,
    '/api/v1/nwtn/execute': 512KB,
    '/api/v1/credentials/register': 64KB,
    '/api/v1/conversations/*/messages': 256KB,
    '/api/v1/web3/transfer': 32KB,
    '/api/v1/upload': 100MB,
    '/api/v1/ipfs/upload': 50MB
}

websocket_max_message_size: 256KB
rate_limit_requests_per_minute: 100
rate_limit_expensive_requests_per_minute: 10
```

### Sanitization Configuration

```python
# Default configuration in SanitizationConfig
allowed_html_tags: {'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'blockquote', 'h1-h6', 'code', 'pre'}
max_json_depth: 10
max_json_keys: 1000
max_string_length: 100KB
allowed_url_schemes: {'http', 'https', 'mailto'}
```

## API Endpoints

### Security Status Monitoring

#### Get Security Status
```http
GET /api/v1/security/status
Authorization: Bearer <jwt-token>
```

**Response**:
```json
{
    "request_limits": {
        "enabled": true,
        "default_max_body_size": 16777216,
        "rate_limit_requests_per_minute": 100,
        "expensive_endpoints_count": 5
    },
    "websocket_limits": {
        "active_connections": 12,
        "max_message_size": 262144,
        "max_messages_per_minute": 60
    },
    "input_sanitization": {
        "enabled": true,
        "max_json_depth": 10,
        "allowed_html_tags_count": 12,
        "sql_injection_patterns_count": 10
    },
    "system_status": {
        "security_middleware_active": true,
        "user_permissions": "admin"
    }
}
```

#### Test Input Sanitization (Admin Only)
```http
POST /api/v1/security/test/sanitization
Authorization: Bearer <jwt-token>
Content-Type: application/json

{
    "test_string": "<script>alert('xss')</script>Hello <strong>World</strong>",
    "allow_html": true,
    "test_json": {
        "user_input": "'; DROP TABLE users; --",
        "nested": {"very": {"deep": "object"}}
    }
}
```

## Implementation Details

### Middleware Architecture

#### Request Limits Middleware
```python
from prsm.security import RequestLimitsMiddleware, RequestLimitsConfig

# Applied to FastAPI app
app.add_middleware(RequestLimitsMiddleware, config=request_limits_config)
```

**Features**:
- Request body size validation
- Rate limiting per IP address
- Timeout enforcement
- Attack pattern detection
- Comprehensive logging

#### Input Sanitization Integration
```python
from prsm.security import sanitize_string, sanitize_json

# Automatic sanitization in Pydantic models
class SecureUserInput(SecureBaseModel):
    query: str = Field(..., max_length=10000)
    
    @validator('query')
    def validate_query(cls, v):
        # Automatic sanitization occurs in SecureBaseModel
        return v.strip()
```

### WebSocket Security

#### Message Validation
```python
from prsm.security import validate_websocket_message

# In WebSocket endpoints
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    while True:
        data = await websocket.receive_text()
        
        # Validate message size and rate limits
        await validate_websocket_message(websocket, data, user_id)
        
        message = json.loads(data)
        await handle_message(message)
```

#### Connection Management
```python
from prsm.security import cleanup_websocket_connection

# Cleanup on disconnect
finally:
    await cleanup_websocket_connection(websocket)
```

## Secure Models

### Enhanced Pydantic Models

PRSM provides secure Pydantic models with built-in sanitization:

```python
from prsm.security import SecureUserInput, SecureWebSocketMessage

# Automatically sanitized user input
user_input = SecureUserInput(
    user_id="user123",
    query="<script>alert('xss')</script>Hello world",
    preferences={"theme": "dark"}
)
# query is automatically sanitized to: "&lt;script&gt;alert('xss')&lt;/script&gt;Hello world"

# Secure WebSocket messages
ws_message = SecureWebSocketMessage(
    type="send_message",
    content="User message content",
    conversation_id="conv123"
)
```

### Available Secure Models

- **SecureUserInput**: User queries and preferences
- **SecureCredentialData**: API credentials and tokens
- **SecureWebSocketMessage**: WebSocket message content
- **SecureTransferRequest**: Cryptocurrency transfers
- **SecureConversationMessage**: Chat messages
- **SecureFileUpload**: File upload validation

## Security Patterns

### String Sanitization

```python
from prsm.security import sanitize_string

# Basic sanitization (HTML escaped)
clean_text = await sanitize_string(
    "User <script>alert('xss')</script> input",
    allow_html=False,
    max_length=1000,
    field_name="user_comment"
)

# HTML sanitization (allowlist approach)
clean_html = await sanitize_string(
    "User <strong>bold</strong> <script>alert('xss')</script> content",
    allow_html=True,
    max_length=5000,
    field_name="user_post"
)
# Result: "User <strong>bold</strong>  content"
```

### JSON Sanitization

```python
from prsm.security import sanitize_json

# Sanitize complex JSON structures
user_data = {
    "name": "<script>alert('xss')</script>John",
    "profile": {
        "bio": "'; DROP TABLE users; --",
        "settings": {"theme": "dark"}
    }
}

clean_data = await sanitize_json(
    user_data,
    max_depth=5,
    field_name="user_profile"
)
```

### URL Validation

```python
from prsm.security import validate_url

# Validate and sanitize URLs
safe_url = await validate_url(
    "javascript:alert('xss')",
    field_name="profile_link"
)
# Raises HTTPException for dangerous URLs
```

## Attack Prevention

### Cross-Site Scripting (XSS)
- **HTML escaping**: All user content HTML-escaped by default
- **Allowlist sanitization**: Permitted HTML tags with strict attribute filtering
- **Content Security Policy**: Headers prevent script execution
- **Output encoding**: Proper encoding for different contexts

### SQL Injection
- **Pattern detection**: Regex-based detection of SQL injection attempts
- **Parameterized queries**: Enforcement through ORM usage
- **Input validation**: Strict validation of all database inputs
- **Error handling**: No database error information exposed

### Denial of Service (DoS)
- **Request size limits**: Body size limits prevent memory exhaustion
- **Rate limiting**: IP-based rate limiting prevents request flooding
- **Connection limits**: WebSocket connection limits per user
- **Timeout protection**: Request timeouts prevent slow attacks

### Path Traversal
- **Path validation**: Automatic detection of directory traversal attempts
- **File name sanitization**: Strict file name validation for uploads
- **Absolute path detection**: Warning and logging for absolute paths

### JSON Attacks
- **Depth limiting**: Prevention of deeply nested JSON structures
- **Key count limits**: Protection against object key flooding
- **Size limits**: String and structure size limitations
- **Type validation**: Strict type checking for JSON values

## Monitoring & Alerting

### Security Event Logging

All security events are logged with comprehensive details:

```json
{
    "event_type": "input_sanitization_sql_injection_attempt",
    "user_id": "user123",
    "details": {
        "field": "user_query",
        "pattern": "(?i)(union\\s+select)",
        "client_ip": "192.168.1.100",
        "user_agent": "Mozilla/5.0..."
    },
    "security_level": "warning",
    "timestamp": "2025-06-10T20:00:00Z"
}
```

### Security Metrics

Monitor security system effectiveness:

```python
# Get security statistics
security_status = await get_security_status()

# Key metrics to monitor:
# - Request blocking rate
# - Sanitization trigger rate  
# - Rate limit violations
# - WebSocket connection patterns
# - Attack pattern detection
```

## Best Practices

### For Developers

1. **Always use secure models**: Use `SecureBaseModel` derivatives for all user input
2. **Validate at boundaries**: Sanitize input at application boundaries
3. **Principle of least privilege**: Only allow necessary HTML tags/attributes
4. **Monitor security logs**: Regularly review security event logs
5. **Test security measures**: Use admin sanitization testing endpoints

### For System Administrators

1. **Configure appropriate limits**: Adjust size limits based on application needs
2. **Monitor attack patterns**: Watch for repeated attack attempts
3. **Update security rules**: Regularly update sanitization patterns
4. **Capacity planning**: Monitor resource usage under limits
5. **Incident response**: Have procedures for security event escalation

### For DevOps Teams

1. **Environment-specific limits**: Different limits for dev/staging/production
2. **Load testing**: Test security limits under load
3. **Monitoring integration**: Integrate security logs with monitoring systems
4. **Automated alerting**: Set up alerts for security threshold violations
5. **Regular updates**: Keep security dependencies updated

## Troubleshooting

### Common Issues

#### Request Size Exceeded
```
HTTP 413 Request Entity Too Large
```
**Solutions**:
- Check request body size against endpoint limits
- Verify content-length header accuracy
- Consider compressing large payloads
- Request limit increase if legitimate

#### Rate Limit Exceeded
```
HTTP 429 Too Many Requests
```
**Solutions**:
- Implement client-side rate limiting
- Add retry logic with backoff
- Check for expensive endpoint usage
- Verify legitimate traffic patterns

#### Input Sanitization Rejection
```
HTTP 400 Invalid input detected
```
**Solutions**:
- Review input for suspicious patterns
- Check SQL injection pattern matches
- Validate input encoding
- Use admin testing endpoint for debugging

#### WebSocket Validation Failure
```
WebSocket closed: 1008 Message validation failed
```
**Solutions**:
- Check message size limits
- Verify message rate limits
- Review connection timeout settings
- Validate message JSON structure

### Debug Steps

1. **Check security status endpoint**: Get current system status
2. **Review security logs**: Look for relevant security events
3. **Test with admin endpoints**: Use sanitization testing endpoints
4. **Validate configuration**: Check limits configuration
5. **Monitor metrics**: Track security system performance

## Configuration Examples

### Development Environment
```python
# Relaxed limits for development
RequestLimitsConfig(
    default_max_body_size=32 * 1024 * 1024,  # 32MB
    rate_limit_requests_per_minute=1000,
    websocket_max_message_size=1024 * 1024   # 1MB
)
```

### Production Environment
```python
# Strict limits for production
RequestLimitsConfig(
    default_max_body_size=8 * 1024 * 1024,   # 8MB
    rate_limit_requests_per_minute=60,
    websocket_max_message_size=128 * 1024,   # 128KB
    expensive_endpoints={'all_ai_endpoints'}
)
```

### High-Security Environment
```python
# Maximum security configuration
RequestLimitsConfig(
    default_max_body_size=1 * 1024 * 1024,   # 1MB
    rate_limit_requests_per_minute=30,
    websocket_max_message_size=64 * 1024,    # 64KB
    request_timeout_seconds=30
)

SanitizationConfig(
    max_json_depth=5,
    max_json_keys=100,
    allowed_html_tags={'p', 'br', 'strong'}  # Minimal HTML
)
```

This comprehensive security hardening system provides enterprise-grade protection against common web application attacks while maintaining usability and performance for legitimate users.