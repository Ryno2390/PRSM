# PRSM API Comprehensive Documentation

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Core Concepts](#core-concepts)
- [API Endpoints](#api-endpoints)
- [Real-time Features](#real-time-features)
- [SDKs and Examples](#sdks-and-examples)
- [Rate Limits and Quotas](#rate-limits-and-quotas)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Overview

The PRSM (Protocol for Recursive Scientific Modeling) API provides comprehensive access to a decentralized AI framework designed for scientific discovery and collaboration. This API enables researchers, developers, and organizations to:

- 🧠 **Orchestrate AI Models**: Coordinate multiple AI systems through the NWTN (Neural Work Token Network)
- 💰 **Trade Resources**: Buy and sell AI models, datasets, and tools using FTNS tokens
- 🔬 **Manage Research**: Track research sessions with real-time collaboration
- 🏪 **Access Marketplace**: Discover and acquire resources from a global marketplace
- 🔐 **Ensure Security**: Benefit from enterprise-grade security and compliance

### Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| **AI Orchestration** | Multi-agent reasoning and task decomposition | ✅ Active |
| **Token Economy** | FTNS cryptocurrency integration | ✅ Active |
| **Marketplace** | Universal resource trading platform | ✅ Active |
| **Real-time Updates** | WebSocket-based live communication | ✅ Active |
| **P2P Network** | Distributed computing capabilities | 🚧 Beta |
| **Governance** | Community-driven decision making | ✅ Active |

## Authentication

### Overview
PRSM API uses JWT (JSON Web Token) based authentication with support for:
- **Access Tokens**: Short-lived tokens for API requests (1 hour default)
- **Refresh Tokens**: Long-lived tokens for obtaining new access tokens (30 days)
- **API Keys**: Service-to-service authentication
- **Role-based Access**: Granular permissions based on user roles

### Getting Started

#### 1. Register an Account
```bash
curl -X POST "https://api.prsm.org/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "researcher@university.edu",
    "password": "secure_password_123",
    "full_name": "Dr. Jane Smith",
    "organization": "University of Science",
    "role": "researcher"
  }'
```

#### 2. Login and Get Tokens
```bash
curl -X POST "https://api.prsm.org/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "researcher@university.edu",
    "password": "secure_password_123"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "email": "researcher@university.edu",
    "role": "researcher",
    "ftns_balance": 1000.0
  }
}
```

#### 3. Use Access Token
Include the access token in the Authorization header for all API requests:
```bash
curl -X GET "https://api.prsm.org/api/v1/marketplace/resources" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### User Roles and Permissions

| Role | Permissions | FTNS Grant | Description |
|------|-------------|------------|-------------|
| **Guest** | Read-only access | 0 | Limited marketplace browsing |
| **Researcher** | Full research features | 1,000 | Academic and research use |
| **Developer** | API access, tool creation | 2,000 | Software development |
| **Enterprise** | Team management | 10,000 | Commercial applications |
| **Admin** | System administration | Unlimited | Platform management |

## Core Concepts

### FTNS Token Economy

**FTNS (Functional Token Network System)** is the native cryptocurrency that powers the PRSM ecosystem:

- **Earning FTNS**: Contribute models, datasets, or compute resources
- **Spending FTNS**: Purchase resources, API calls, or premium features
- **Staking FTNS**: Participate in governance and earn rewards
- **Real Transactions**: Full blockchain integration with smart contracts

#### Token Usage Examples

| Action | Cost (FTNS) | Description |
|--------|-------------|-------------|
| AI Model Query | 0.1 - 10 | Based on model complexity |
| Dataset Download | 5 - 500 | Based on size and quality |
| Compute Hour | 10 - 100 | Based on resource requirements |
| Storage (GB/month) | 1 - 5 | Distributed IPFS storage |

### Resource Types

The PRSM marketplace supports 9 types of resources:

1. **🤖 AI Models**: Pre-trained models, fine-tuned models, custom architectures
2. **📊 Datasets**: Training data, benchmarks, specialized collections
3. **🔧 Tools**: Analysis tools, preprocessing scripts, utilities
4. **⚡ Compute Time**: GPU/CPU hours, specialized hardware access
5. **💾 Storage**: IPFS storage, database access, data hosting
6. **🔌 API Access**: Third-party APIs, specialized services
7. **📄 Research Papers**: Academic papers, tutorials, documentation
8. **📋 Templates**: Project templates, workflow patterns
9. **🔌 Plugins**: Extensions, integrations, custom components

### Session Management

Research sessions provide structured collaboration:

- **Session Lifecycle**: Create → Configure → Execute → Review → Archive
- **Real-time Collaboration**: Multiple researchers can work simultaneously
- **Version Control**: Track changes and manage contributions
- **Budget Management**: Allocate and track FTNS spending
- **Progress Monitoring**: Real-time updates on task completion

## API Endpoints

### Authentication Endpoints

#### POST `/api/v1/auth/register`
Register a new user account.

**Request Body:**
```json
{
  "email": "string",
  "password": "string",
  "full_name": "string",
  "organization": "string (optional)",
  "role": "researcher|developer|enterprise"
}
```

#### POST `/api/v1/auth/login`
Authenticate and receive access tokens.

#### POST `/api/v1/auth/refresh`
Refresh access token using refresh token.

#### POST `/api/v1/auth/logout`
Invalidate current session and tokens.

### Marketplace Endpoints

#### GET `/api/v1/marketplace/resources`
Search and browse marketplace resources.

**Query Parameters:**
- `query`: Search query string
- `resource_type`: Filter by resource type
- `max_price`: Maximum price in FTNS
- `min_rating`: Minimum rating (0-5)
- `tags`: Comma-separated tags
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20)

**Example Request:**
```bash
curl -X GET "https://api.prsm.org/api/v1/marketplace/resources?query=machine%20learning&resource_type=ai_model&max_price=200&min_rating=4.0" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "items": [
    {
      "id": "res_123456789",
      "title": "Advanced Computer Vision Model",
      "description": "State-of-the-art CNN for image classification",
      "resource_type": "ai_model",
      "price": 120.0,
      "seller_name": "Vision Research Lab",
      "rating": 4.8,
      "reviews_count": 15,
      "tags": ["computer-vision", "cnn", "pytorch"],
      "created_at": "2024-01-10T08:00:00Z"
    }
  ],
  "total": 45,
  "page": 1,
  "per_page": 20,
  "has_next": true,
  "has_prev": false
}
```

#### POST `/api/v1/marketplace/resources/{resource_id}/purchase`
Purchase a marketplace resource.

#### GET `/api/v1/marketplace/resources/{resource_id}`
Get detailed information about a specific resource.

### Session Management Endpoints

#### POST `/api/v1/sessions`
Create a new research session.

**Request Body:**
```json
{
  "title": "Climate Change Analysis",
  "description": "ML analysis of climate data",
  "collaborators": ["user_id_1", "user_id_2"],
  "ftns_budget": 500.0,
  "tags": ["climate", "machine-learning"]
}
```

#### GET `/api/v1/sessions`
List user's research sessions.

#### GET `/api/v1/sessions/{session_id}`
Get detailed session information.

#### PUT `/api/v1/sessions/{session_id}`
Update session configuration.

### Task Management Endpoints

#### POST `/api/v1/tasks`
Create a new task within a session.

#### GET `/api/v1/tasks`
List tasks with filtering options.

#### GET `/api/v1/tasks/{task_id}`
Get detailed task information including subtasks.

### FTNS Token Endpoints

#### GET `/api/v1/users/{user_id}/balance`
Get user's FTNS token balance.

**Response:**
```json
{
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "available_balance": 1250.75,
  "locked_balance": 250.0,
  "total_balance": 1500.75,
  "last_updated": "2024-01-15T10:00:00Z"
}
```

#### GET `/api/v1/transactions`
Get transaction history.

#### POST `/api/v1/transactions/transfer`
Transfer FTNS tokens between users.

### Health and Monitoring

#### GET `/api/v1/health`
Comprehensive system health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "components": {
    "database": {"status": "healthy", "response_time_ms": 15.2},
    "redis": {"status": "healthy", "response_time_ms": 8.1},
    "ipfs": {"status": "healthy", "response_time_ms": 45.6},
    "vector_db": {"status": "healthy", "response_time_ms": 23.4}
  },
  "response_time_ms": 245.5
}
```

## Real-time Features

### WebSocket Connections

PRSM provides real-time communication through WebSocket connections:

#### General Updates: `/ws/{user_id}`
Connect to receive real-time notifications and updates.

```javascript
const ws = new WebSocket('wss://api.prsm.org/ws/user_123?token=your_jwt_token');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```

**Message Types:**
- `session_update`: Research session status changes
- `task_completion`: Task completion notifications
- `ftns_transaction`: Token transaction notifications
- `marketplace_activity`: New listings or purchases

#### Conversation Streaming: `/ws/conversation/{user_id}/{conversation_id}`
Real-time AI conversation streaming.

```python
import asyncio
import websockets
import json

async def stream_conversation():
    uri = "wss://api.prsm.org/ws/conversation/user_123/conv_456?token=jwt_token"
    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "type": "user_message",
            "content": "Analyze this dataset for climate patterns"
        }))
        
        # Receive streaming response
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "ai_response_chunk":
                print(data["content"], end="", flush=True)
```

### Real-time Events

| Event Type | Description | Example Payload |
|------------|-------------|-----------------|
| `session_started` | New session created | `{"session_id": "sess_123", "title": "New Research"}` |
| `task_completed` | Task finished | `{"task_id": "task_456", "status": "completed", "cost": 25.0}` |
| `balance_updated` | FTNS balance changed | `{"new_balance": 1275.0, "change": 25.0}` |
| `message_received` | New collaboration message | `{"from": "user_789", "content": "Results look good!"}` |

## SDKs and Examples

### Python SDK

Install the official Python SDK:
```bash
pip install prsm-sdk
```

**Basic Usage:**
```python
from prsm_sdk import PRSMClient

# Initialize client
client = PRSMClient(
    api_key="your_api_key",
    base_url="https://api.prsm.org"
)

# Authenticate
client.login("researcher@university.edu", "password")

# Search marketplace
resources = client.marketplace.search(
    query="machine learning",
    resource_type="ai_model",
    max_price=200.0
)

# Create research session
session = client.sessions.create(
    title="Climate Analysis",
    description="ML analysis of climate data",
    ftns_budget=500.0
)

# Execute AI query
response = client.ai.query(
    prompt="Analyze this climate dataset",
    model="gpt-4-scientific",
    session_id=session.id
)
```

### JavaScript SDK

Install via npm:
```bash
npm install @prsm/js-sdk
```

**Basic Usage:**
```javascript
import { PRSMClient } from '@prsm/js-sdk';

const client = new PRSMClient({
  apiKey: 'your_api_key',
  baseURL: 'https://api.prsm.org'
});

// Authenticate
await client.auth.login('researcher@university.edu', 'password');

// Search marketplace
const resources = await client.marketplace.search({
  query: 'machine learning',
  resourceType: 'ai_model',
  maxPrice: 200.0
});

// Real-time updates
client.websocket.connect('user_123');
client.websocket.on('session_update', (data) => {
  console.log('Session updated:', data);
});
```

### cURL Examples

**Purchase a Resource:**
```bash
curl -X POST "https://api.prsm.org/api/v1/marketplace/resources/res_123/purchase" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"payment_method": "ftns_balance"}'
```

**Create a Task:**
```bash
curl -X POST "https://api.prsm.org/api/v1/tasks" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Data Preprocessing",
    "description": "Clean and prepare dataset for analysis",
    "session_id": "session_123",
    "estimated_cost": 25.0,
    "priority": 3
  }'
```

## Rate Limits and Quotas

### Rate Limiting

API requests are limited based on your subscription tier:

| Tier | Requests/Hour | Concurrent Connections | Burst Limit |
|------|---------------|----------------------|-------------|
| **Free** | 100 | 2 | 10/minute |
| **Pro** | 1,000 | 5 | 50/minute |
| **Enterprise** | 10,000 | 20 | 200/minute |
| **Custom** | Negotiable | Negotiable | Negotiable |

### Rate Limit Headers

All API responses include rate limit information:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1642345678
X-RateLimit-Type: user
```

### Quota Management

Resource usage is tracked and limited:

- **FTNS Token Spending**: Daily/monthly limits based on tier
- **Storage Usage**: IPFS storage allocation
- **Compute Time**: Maximum hours per month
- **API Calls**: Request count across all endpoints

## Error Handling

### Standard Error Format

All API errors follow a consistent format:

```json
{
  "success": false,
  "message": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "details": {
    "field": "Additional context",
    "suggestion": "How to fix the error"
  },
  "timestamp": "2024-01-15T10:00:00Z",
  "request_id": "req_123456789"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | `INVALID_INPUT` | Request parameters are invalid | Check request format and required fields |
| 401 | `UNAUTHORIZED` | Authentication required | Provide valid access token |
| 403 | `FORBIDDEN` | Insufficient permissions | Check user role and permissions |
| 404 | `NOT_FOUND` | Resource does not exist | Verify resource ID or path |
| 429 | `RATE_LIMITED` | Too many requests | Wait before making more requests |
| 500 | `INTERNAL_ERROR` | Server error | Retry request or contact support |

### Error Handling Best Practices

1. **Always Check Status Codes**: Handle different HTTP status codes appropriately
2. **Use Error Codes**: Machine-readable error codes for programmatic handling
3. **Implement Retry Logic**: Exponential backoff for temporary failures
4. **Log Request IDs**: Include request IDs in support tickets

**Example Error Handling:**
```python
import requests
import time

def make_api_request(url, headers, data=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                continue
            elif response.status_code >= 500:  # Server error
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            # Handle other errors
            error_data = response.json()
            raise APIError(error_data['error_code'], error_data['message'])
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    
    raise APIError('MAX_RETRIES_EXCEEDED', 'Request failed after maximum retries')
```

## Best Practices

### Authentication
- **Secure Token Storage**: Store tokens securely (not in client-side code)
- **Token Refresh**: Implement automatic token refresh logic
- **Environment Variables**: Use environment variables for API keys

### Performance
- **Connection Pooling**: Reuse HTTP connections for better performance
- **Pagination**: Use pagination for large result sets
- **Caching**: Cache responses when appropriate (respect cache headers)
- **Compression**: Enable gzip compression for large responses

### Security
- **HTTPS Only**: Always use HTTPS in production
- **Input Validation**: Validate all inputs before sending to API
- **Rate Limiting**: Implement client-side rate limiting
- **Audit Logging**: Log API interactions for security monitoring

### Reliability
- **Error Handling**: Implement comprehensive error handling
- **Timeouts**: Set appropriate request timeouts
- **Circuit Breakers**: Implement circuit breakers for external dependencies
- **Health Checks**: Monitor API health and availability

### Example Production Implementation

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

class PRSMAPIClient:
    def __init__(self, base_url, api_key, timeout=30):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'PRSM-Python-Client/1.0.0'
        })
        
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def make_request(self, method, endpoint, **kwargs):
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
            
            # Log request details
            self.logger.info(f"{method} {url} - {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                self.logger.error(f"API Error: {error_data}")
                raise APIError(response.status_code, error_data)
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
```

---

## Support and Resources

- **📚 Full Documentation**: [https://docs.prsm.org](https://docs.prsm.org)
- **🐛 Issue Tracking**: [https://github.com/prsm-org/prsm/issues](https://github.com/prsm-org/prsm/issues)
- **💬 Community Forum**: [https://community.prsm.org](https://community.prsm.org)
- **📧 API Support**: [api-support@prsm.org](mailto:api-support@prsm.org)
- **🔒 Security Issues**: [security@prsm.org](mailto:security@prsm.org)

For enterprise support and custom implementations, contact [enterprise@prsm.org](mailto:enterprise@prsm.org).