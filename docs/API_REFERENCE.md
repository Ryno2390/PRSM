# PRSM API Reference

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core API Endpoints](#core-api-endpoints)
4. [Health & Monitoring](#health--monitoring)
5. [User Management](#user-management)
6. [Model & Training](#model--training)
7. [Marketplace](#marketplace)
8. [Governance](#governance)
9. [Web3 & FTNS](#web3--ftns)
10. [Security](#security)
11. [WebSocket API](#websocket-api)
12. [Error Handling](#error-handling)
13. [Rate Limiting](#rate-limiting)
14. [SDK Examples](#sdk-examples)

## Overview

The PRSM API provides a comprehensive RESTful interface for interacting with the Protocol for Recursive Scientific Modeling. All endpoints follow REST conventions and return JSON responses.

### Base URLs

- **Production**: `https://api.prsm.org`
- **Staging**: `https://staging-api.prsm.org`
- **Development**: `http://localhost:8000`

### API Versioning

The API uses URL-based versioning:
- Current version: `v1`
- Full endpoint format: `{base_url}/api/v1/{endpoint}`

### Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **File Upload**: `multipart/form-data`

## Authentication

### JWT Bearer Authentication

Most endpoints require JWT bearer token authentication:

```http
Authorization: Bearer <jwt_token>
```

### Obtaining a Token

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user123",
    "username": "your_username",
    "email": "user@example.com",
    "role": "researcher"
  }
}
```

#### Register
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "new_user",
  "email": "user@example.com",
  "password": "secure_password",
  "role": "researcher"
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Authorization: Bearer <jwt_token>
```

### API Key Authentication

For programmatic access, use API keys:

```http
X-API-Key: your_api_key
```

## Core API Endpoints

### NWTN (Neural Web of Thought Networks)

#### Submit Research Query
```http
POST /api/v1/nwtn/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "Analyze the impact of climate change on marine ecosystems",
  "domain": "environmental_science",
  "methodology": "comprehensive_analysis",
  "max_iterations": 5,
  "include_citations": true
}
```

**Response**:
```json
{
  "session_id": "sess_abc123",
  "status": "processing",
  "query": "Analyze the impact of climate change on marine ecosystems",
  "estimated_completion": "2025-06-11T14:30:00Z",
  "cost_estimate": {
    "ftns_tokens": 150,
    "usd_equivalent": 0.75
  }
}
```

#### Get Session Status
```http
GET /api/v1/nwtn/sessions/{session_id}
Authorization: Bearer <token>
```

**Response**:
```json
{
  "session_id": "sess_abc123",
  "status": "completed",
  "progress": 100,
  "results": {
    "summary": "Climate change significantly impacts marine ecosystems through...",
    "key_findings": [
      "Ocean acidification affects coral reef biodiversity",
      "Rising temperatures alter fish migration patterns"
    ],
    "citations": [
      {
        "title": "Marine Ecosystem Response to Climate Change",
        "authors": ["Smith, J.", "Doe, A."],
        "journal": "Nature Climate Change",
        "year": 2024
      }
    ],
    "confidence_score": 0.92
  },
  "cost_actual": {
    "ftns_tokens": 145,
    "usd_equivalent": 0.73
  }
}
```

#### List User Sessions
```http
GET /api/v1/nwtn/sessions?limit=20&offset=0&status=completed
Authorization: Bearer <token>
```

### Model Management

#### List Available Models
```http
GET /api/v1/models?category=language&provider=openai&limit=50
```

**Response**:
```json
{
  "models": [
    {
      "id": "model_123",
      "name": "GPT-4 Turbo",
      "provider": "openai",
      "category": "language",
      "description": "Advanced language model for complex reasoning",
      "pricing": {
        "input_tokens": 0.01,
        "output_tokens": 0.03,
        "currency": "USD"
      },
      "capabilities": ["text_generation", "analysis", "reasoning"],
      "context_length": 128000,
      "available": true
    }
  ],
  "total": 45,
  "page": 1,
  "pages": 3
}
```

#### Get Model Details
```http
GET /api/v1/models/{model_id}
```

#### Train Custom Model
```http
POST /api/v1/models/train
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Custom Research Model",
  "base_model": "llama2_7b",
  "training_data": "ipfs://QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "training_parameters": {
    "epochs": 3,
    "learning_rate": 0.0001,
    "batch_size": 16
  },
  "domain": "biomedical_research"
}
```

## Health & Monitoring

### System Health
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "environment": "production",
  "timestamp": "2025-06-11T12:00:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12.5,
      "last_check": "2025-06-11T12:00:00Z"
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 2.1,
      "last_check": "2025-06-11T12:00:00Z"
    },
    "ipfs": {
      "status": "healthy",
      "response_time_ms": 45.3,
      "last_check": "2025-06-11T12:00:00Z"
    }
  }
}
```

### Liveness & Readiness Probes
```http
GET /health/liveness    # Kubernetes liveness probe
GET /health/readiness   # Kubernetes readiness probe
```

### Detailed Status (Authenticated)
```http
GET /health/detailed
Authorization: Bearer <token>
```

### Metrics (Prometheus)
```http
GET /health/metrics
```

## User Management

### Get Current User Profile
```http
GET /api/v1/users/me
Authorization: Bearer <token>
```

**Response**:
```json
{
  "id": "user_123",
  "username": "researcher1",
  "email": "researcher@university.edu",
  "role": "researcher",
  "organization": "University Research Lab",
  "created_at": "2025-01-15T10:30:00Z",
  "last_login": "2025-06-11T08:15:00Z",
  "ftns_balance": 1250.50,
  "research_credits": 500,
  "permissions": [
    "submit_queries",
    "train_models",
    "access_marketplace"
  ]
}
```

### Update User Profile
```http
PUT /api/v1/users/me
Authorization: Bearer <token>
Content-Type: application/json

{
  "organization": "Updated Research Institution",
  "bio": "Climate science researcher focused on marine ecosystems",
  "research_interests": ["climate_change", "marine_biology", "data_analysis"]
}
```

### Change Password
```http
POST /api/v1/users/me/password
Authorization: Bearer <token>
Content-Type: application/json

{
  "current_password": "old_password",
  "new_password": "new_secure_password"
}
```

## Marketplace

### Browse Models
```http
GET /api/v1/marketplace/models?category=language&featured=true&limit=20
```

**Response**:
```json
{
  "models": [
    {
      "id": "marketplace_model_123",
      "name": "Advanced Research Assistant",
      "description": "Specialized model for scientific literature analysis",
      "creator": {
        "username": "ai_researcher",
        "verified": true
      },
      "category": "language",
      "tags": ["research", "analysis", "scientific"],
      "pricing": {
        "ftns_per_request": 5,
        "bulk_discount": 0.1
      },
      "stats": {
        "downloads": 1547,
        "rating": 4.8,
        "reviews": 89
      },
      "featured": true,
      "created_at": "2025-05-15T14:20:00Z"
    }
  ],
  "total": 156,
  "filters": {
    "categories": ["language", "vision", "audio", "scientific"],
    "providers": ["community", "verified", "official"]
  }
}
```

### Rent Model
```http
POST /api/v1/marketplace/models/{model_id}/rent
Authorization: Bearer <token>
Content-Type: application/json

{
  "duration_hours": 24,
  "max_requests": 1000
}
```

### Submit Model to Marketplace
```http
POST /api/v1/marketplace/models
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Custom Scientific Model",
  "description": "Specialized for climate data analysis",
  "category": "scientific",
  "model_file": "ipfs://QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "pricing": {
    "ftns_per_request": 10,
    "revenue_share": 0.7
  },
  "tags": ["climate", "data_analysis", "python"]
}
```

## Governance

### List Active Proposals
```http
GET /api/v1/governance/proposals?status=active&limit=10
```

**Response**:
```json
{
  "proposals": [
    {
      "id": "prop_123",
      "title": "Increase Research Grant Pool",
      "description": "Proposal to allocate additional FTNS tokens for research grants",
      "proposer": "governance_council",
      "status": "active",
      "voting_ends": "2025-06-18T23:59:59Z",
      "votes": {
        "yes": 1250000,
        "no": 340000,
        "abstain": 89000
      },
      "quorum_required": 1000000,
      "threshold": 0.6,
      "category": "funding"
    }
  ]
}
```

### Submit Proposal
```http
POST /api/v1/governance/proposals
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "New Research Domain Integration",
  "description": "Proposal to add quantum computing as a supported research domain",
  "category": "technical",
  "implementation_plan": "Detailed plan for quantum computing integration...",
  "budget_required": 50000
}
```

### Vote on Proposal
```http
POST /api/v1/governance/proposals/{proposal_id}/vote
Authorization: Bearer <token>
Content-Type: application/json

{
  "vote": "yes",
  "voting_power": 1000,
  "comment": "This proposal aligns with our research goals"
}
```

## Web3 & FTNS

### Get FTNS Balance
```http
GET /api/v1/web3/balance
Authorization: Bearer <token>
```

**Response**:
```json
{
  "balance": {
    "ftns_tokens": 1250.50,
    "staked_tokens": 500.00,
    "available_tokens": 750.50
  },
  "wallet_address": "0x742d35cc6bf4532c95a0e96a7bdc86c0b3e11888",
  "network": "polygon",
  "last_updated": "2025-06-11T12:00:00Z"
}
```

### Transfer FTNS Tokens
```http
POST /api/v1/web3/transfer
Authorization: Bearer <token>
Content-Type: application/json

{
  "to_address": "0x456def789ghi012jkl345mno678pqr901stu234vw",
  "amount": 100.0,
  "note": "Research collaboration payment"
}
```

### Purchase FTNS with Fiat
```http
POST /api/v1/payments/purchase
Authorization: Bearer <token>
Content-Type: application/json

{
  "amount_usd": 50.00,
  "payment_method": "stripe",
  "payment_token": "tok_1234567890abcdef"
}
```

### Transaction History
```http
GET /api/v1/web3/transactions?limit=50&type=all
Authorization: Bearer <token>
```

## Security

### Security Status
```http
GET /api/v1/security/status
Authorization: Bearer <token>
```

**Response**:
```json
{
  "security_level": "normal",
  "threats_detected": 0,
  "last_scan": "2025-06-11T11:45:00Z",
  "account_status": "verified",
  "recent_activity": {
    "login_attempts": 1,
    "successful_logins": 1,
    "api_calls": 156,
    "suspicious_activity": false
  }
}
```

### Enable Two-Factor Authentication
```http
POST /api/v1/security/2fa/enable
Authorization: Bearer <token>
```

### Generate API Key
```http
POST /api/v1/security/api-keys
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Research Project API Key",
  "permissions": ["read", "submit_queries"],
  "expires_in_days": 90
}
```

## WebSocket API

### Connection
```javascript
wss://api.prsm.org/ws?token=<jwt_token>
```

### Message Format
```json
{
  "type": "message_type",
  "payload": {
    "key": "value"
  },
  "timestamp": "2025-06-11T12:00:00Z"
}
```

### Subscribe to Session Updates
```json
{
  "type": "subscribe",
  "payload": {
    "channel": "session_updates",
    "session_id": "sess_abc123"
  }
}
```

### Real-time Progress Updates
```json
{
  "type": "session_progress",
  "payload": {
    "session_id": "sess_abc123",
    "progress": 45,
    "current_step": "model_reasoning",
    "estimated_completion": "2025-06-11T14:30:00Z"
  }
}
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "query",
      "issue": "Query text is required"
    },
    "request_id": "req_abc123xyz"
  }
}
```

### Common Error Codes

- `AUTHENTICATION_FAILED` - Invalid credentials
- `INSUFFICIENT_BALANCE` - Not enough FTNS tokens
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `VALIDATION_ERROR` - Invalid input data
- `RESOURCE_NOT_FOUND` - Requested resource doesn't exist
- `PERMISSION_DENIED` - Insufficient permissions
- `SERVICE_UNAVAILABLE` - External service temporarily unavailable

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1623456789
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/v1/nwtn/query` | 100 requests | 1 hour |
| `/api/v1/models/train` | 5 requests | 1 hour |
| `/api/v1/auth/login` | 5 attempts | 15 minutes |
| `/api/v1/marketplace/*` | 1000 requests | 1 hour |
| General API | 10000 requests | 1 hour |

### Rate Limit Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "retry_after": 3600
  }
}
```

## SDK Examples

### Python SDK
```python
from prsm_sdk import PRSMClient

# Initialize client
client = PRSMClient(
    api_key="your_api_key",
    base_url="https://api.prsm.org"
)

# Submit research query
response = client.nwtn.submit_query(
    query="Analyze protein folding mechanisms",
    domain="biochemistry",
    max_iterations=3
)

print(f"Session ID: {response.session_id}")

# Monitor progress
for update in client.nwtn.watch_session(response.session_id):
    print(f"Progress: {update.progress}%")
    if update.status == "completed":
        print(f"Results: {update.results}")
        break
```

### JavaScript SDK
```javascript
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.prsm.org'
});

// Submit research query
const response = await client.nwtn.submitQuery({
  query: 'Analyze protein folding mechanisms',
  domain: 'biochemistry',
  maxIterations: 3
});

console.log(`Session ID: ${response.sessionId}`);

// WebSocket for real-time updates
const ws = client.createWebSocket();
ws.subscribe('session_updates', response.sessionId, (update) => {
  console.log(`Progress: ${update.progress}%`);
  if (update.status === 'completed') {
    console.log('Results:', update.results);
  }
});
```

### cURL Examples

#### Submit Query
```bash
curl -X POST "https://api.prsm.org/api/v1/nwtn/query" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze climate change impacts",
    "domain": "environmental_science",
    "max_iterations": 5
  }'
```

#### Check Session Status
```bash
curl -X GET "https://api.prsm.org/api/v1/nwtn/sessions/sess_abc123" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### Get FTNS Balance
```bash
curl -X GET "https://api.prsm.org/api/v1/web3/balance" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Document Information

**Version**: 1.0  
**Last Updated**: June 11, 2025  
**API Version**: v1  
**Contact**: api-support@prsm.org  

**Related Documentation**:
- [WebSocket API Documentation](WEBSOCKET_API.md)
- [Authentication Guide](API_KEY_MANAGEMENT.md)
- [SDK Documentation](https://docs.prsm.org/sdks)
- [Postman Collection](https://docs.prsm.org/postman)

---

*For support and questions, please contact our API support team at api-support@prsm.org or visit our [Developer Portal](https://developers.prsm.org).*