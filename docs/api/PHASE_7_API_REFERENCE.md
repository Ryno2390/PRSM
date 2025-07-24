# Phase 7 API Reference

## Overview

This document provides comprehensive API reference for all Phase 7 enterprise architecture components, including endpoints, request/response formats, authentication requirements, and usage examples.

## Authentication

All Phase 7 APIs use JWT-based authentication with the following header:

```
Authorization: Bearer <jwt_token>
```

## Base URLs

- **Production**: `https://api.prsm.ai/v1`
- **Staging**: `https://staging.api.prsm.ai/v1`
- **Development**: `http://localhost:8000/v1`

## API Components

### 1. Global Infrastructure API

Base path: `/infrastructure`

#### Get System Health

```http
GET /infrastructure/health
```

**Response:**
```json
{
  "status": "healthy",
  "regions": [
    {
      "name": "us-west-1",
      "status": "healthy",
      "load": 0.45,
      "latency": 12.3,
      "availability": 99.9
    }
  ],
  "global_metrics": {
    "total_nodes": 50,
    "active_connections": 1247,
    "average_response_time": 156.2
  }
}
```

#### Add Region

```http
POST /infrastructure/regions
```

**Request Body:**
```json
{
  "name": "eu-west-1",
  "capacity": 1000,
  "availability_zone": "eu-west-1a",
  "configuration": {
    "auto_scaling": true,
    "min_nodes": 2,
    "max_nodes": 20
  }
}
```

**Response:**
```json
{
  "region_id": "region_12345",
  "status": "initializing",
  "estimated_ready_time": "2025-07-23T16:30:00Z"
}
```

#### Get Optimal Region

```http
GET /infrastructure/optimal-region?workload_type=reasoning&user_location=us-west
```

**Response:**
```json
{
  "recommended_region": "us-west-1",
  "estimated_latency": 23.5,
  "load_factor": 0.67,
  "reason": "Optimal for reasoning workloads in your location"
}
```

### 2. Analytics Dashboard API

Base path: `/analytics`

#### Create Dashboard

```http
POST /analytics/dashboards
```

**Request Body:**
```json
{
  "name": "Executive Dashboard",
  "type": "executive",
  "widgets": [
    {
      "type": "metric_card",
      "title": "Active Users",
      "data_source": "user_metrics",
      "refresh_interval": 300
    },
    {
      "type": "line_chart",
      "title": "Query Volume",
      "data_source": "query_metrics",
      "time_range": "24h"
    }
  ],
  "layout": {
    "columns": 3,
    "responsive": true
  }
}
```

**Response:**
```json
{
  "dashboard_id": "dash_67890",
  "status": "created",
  "url": "https://dashboard.prsm.ai/d/dash_67890",
  "embed_url": "https://dashboard.prsm.ai/embed/dash_67890"
}
```

#### Update Dashboard Data

```http
PUT /analytics/dashboards/{dashboard_id}/data
```

**Request Body:**
```json
{
  "data": {
    "user_metrics": {
      "active_users": 1247,
      "new_users_today": 89,
      "retention_rate": 0.85
    },
    "query_metrics": {
      "total_queries": 15634,
      "successful_queries": 15012,
      "average_response_time": 18.4
    }
  },
  "timestamp": "2025-07-23T16:00:00Z"
}
```

**Response:**
```json
{
  "status": "updated",
  "widgets_updated": 8,
  "cache_invalidated": true
}
```

#### Get Dashboard Metrics

```http
GET /analytics/dashboards/{dashboard_id}/metrics?time_range=24h&granularity=1h
```

**Response:**
```json
{
  "dashboard_id": "dash_67890",
  "time_range": "24h",
  "metrics": [
    {
      "timestamp": "2025-07-23T15:00:00Z",
      "active_users": 1156,
      "query_volume": 234,
      "success_rate": 0.96
    }
  ],
  "aggregated": {
    "total_queries": 15634,
    "average_success_rate": 0.96,
    "peak_concurrent_users": 1289
  }
}
```

### 3. Enterprise Integration API

Base path: `/integrations`

#### Create Integration

```http
POST /integrations
```

**Request Body:**
```json
{
  "name": "Salesforce CRM Integration",
  "type": "salesforce",
  "configuration": {
    "instance_url": "https://company.salesforce.com",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "username": "integration@company.com"
  },
  "sync_settings": {
    "auto_sync": true,
    "sync_interval": 3600,
    "sync_objects": ["Account", "Contact", "Opportunity"]
  }
}
```

**Response:**
```json
{
  "integration_id": "int_54321",
  "status": "connecting",
  "connection_test": "passed",
  "next_sync": "2025-07-23T17:00:00Z"
}
```

#### Sync Integration

```http
POST /integrations/{integration_id}/sync
```

**Request Body:**
```json
{
  "sync_type": "incremental",
  "objects": ["Account", "Contact"],
  "filters": {
    "modified_since": "2025-07-23T00:00:00Z"
  }
}
```

**Response:**
```json
{
  "sync_id": "sync_98765",
  "status": "running",
  "estimated_duration": 300,
  "progress": {
    "total_records": 1500,
    "processed_records": 0,
    "errors": 0
  }
}
```

#### Get Integration Status

```http
GET /integrations/{integration_id}/status
```

**Response:**
```json
{
  "integration_id": "int_54321",
  "name": "Salesforce CRM Integration",
  "status": "active",
  "health": "healthy",
  "last_sync": "2025-07-23T16:00:00Z",
  "next_sync": "2025-07-23T17:00:00Z",
  "statistics": {
    "total_records_synced": 15000,
    "sync_success_rate": 0.99,
    "average_sync_duration": 245
  }
}
```

### 4. AI Orchestration API

Base path: `/orchestration`

#### Register AI Model

```http
POST /orchestration/models
```

**Request Body:**
```json
{
  "name": "Advanced Reasoning Model",
  "type": "reasoning",
  "provider": "openai",
  "model_id": "gpt-4",
  "capabilities": ["reasoning", "analysis", "synthesis"],
  "configuration": {
    "temperature": 0.7,
    "max_tokens": 4000,
    "timeout": 30
  },
  "cost_per_token": 0.00003
}
```

**Response:**
```json
{
  "model_id": "model_11111",
  "status": "registered",
  "health_check": "passed",
  "estimated_availability": "immediate"
}
```

#### Execute Task

```http
POST /orchestration/tasks
```

**Request Body:**
```json
{
  "type": "reasoning",
  "content": "Analyze the implications of quantum computing on current cryptographic systems",
  "priority": "high",
  "requirements": {
    "reasoning_depth": "deep",
    "max_cost_ftns": 1000,
    "preferred_models": ["model_11111"],
    "timeout": 60
  },
  "context": {
    "user_id": "user_123",
    "session_id": "session_456"
  }
}
```

**Response:**
```json
{
  "task_id": "task_22222",
  "status": "queued",
  "estimated_completion": "2025-07-23T16:02:00Z",
  "assigned_models": ["model_11111"],
  "estimated_cost_ftns": 535
}
```

#### Get Task Status

```http
GET /orchestration/tasks/{task_id}
```

**Response:**
```json
{
  "task_id": "task_22222",
  "status": "completed",
  "result": {
    "content": "Quantum computing poses significant challenges...",
    "confidence_score": 0.95,
    "reasoning_steps": [
      "Analyzed current cryptographic vulnerabilities",
      "Evaluated quantum algorithm capabilities",
      "Assessed timeline for practical quantum computers"
    ]
  },
  "metrics": {
    "processing_time": 18.4,
    "cost_ftns": 535,
    "models_used": ["model_11111"],
    "tokens_consumed": 3847
  }
}
```

### 5. Marketplace API

Base path: `/marketplace`

#### Register Integration

```http
POST /marketplace/integrations
```

**Request Body:**
```json
{
  "name": "Advanced Analytics Plugin",
  "type": "plugin",
  "version": "1.2.0",
  "description": "Comprehensive analytics suite for enterprise users",
  "developer_id": "dev_analytics_team",
  "capabilities": {
    "real_time_analytics": true,
    "custom_dashboards": true,
    "data_export": true,
    "automated_reporting": true
  },
  "pricing": {
    "model": "subscription",
    "base_price": 29.99,
    "billing_period": "monthly",
    "free_trial_days": 14
  },
  "technical_details": {
    "entry_point": "analytics_plugin:AnalyticsPlugin",
    "dependencies": ["pandas>=1.5.0", "plotly>=5.0.0"],
    "permissions": ["read_data", "write_reports"],
    "security_level": "sandbox"
  }
}
```

**Response:**
```json
{
  "integration_id": "int_market_33333",
  "status": "pending_review",
  "security_scan_id": "scan_44444",
  "estimated_review_time": "2-3 business days",
  "developer_dashboard_url": "https://marketplace.prsm.ai/developer/int_market_33333"
}
```

#### Search Integrations

```http
GET /marketplace/integrations/search?q=analytics&category=business&sort=popularity&limit=10
```

**Response:**
```json
{
  "query": "analytics",
  "total_results": 47,
  "results": [
    {
      "integration_id": "int_market_33333",
      "name": "Advanced Analytics Plugin",
      "developer": "Analytics Team",
      "rating": 4.8,
      "reviews_count": 156,
      "price": "$29.99/month",
      "description": "Comprehensive analytics suite...",
      "featured": true,
      "tags": ["analytics", "dashboards", "reporting"]
    }
  ],
  "facets": {
    "categories": {
      "business": 23,
      "development": 15,
      "security": 9
    },
    "pricing": {
      "free": 12,
      "freemium": 18,
      "paid": 17
    }
  }
}
```

#### Submit Review

```http
POST /marketplace/integrations/{integration_id}/reviews
```

**Request Body:**
```json
{
  "rating": 5,
  "title": "Excellent plugin for enterprise analytics",
  "content": "This plugin has significantly improved our data analysis capabilities. The dashboards are intuitive and the automated reporting saves us hours each week.",
  "pros": ["Easy to use", "Great visualizations", "Excellent support"],
  "cons": ["Could use more export formats"],
  "recommended": true
}
```

**Response:**
```json
{
  "review_id": "review_55555",
  "status": "published",
  "sentiment_score": 0.89,
  "helpfulness_score": 0.92,
  "moderation_status": "approved"
}
```

### 6. Unified Pipeline API

Base path: `/pipeline`

#### Process Query

```http
POST /pipeline/process
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "query": "What are the latest developments in quantum computing for cryptography?",
  "verbosity_level": "detailed",
  "enable_deep_reasoning": true,
  "context": {
    "session_id": "session_789",
    "previous_queries": ["quantum computing basics"],
    "user_expertise": "intermediate"
  },
  "preferences": {
    "max_cost_ftns": 1000,
    "max_processing_time": 60,
    "include_citations": true,
    "format": "markdown"
  }
}
```

**Response:**
```json
{
  "query_id": "query_66666",
  "status": "completed",
  "result": {
    "natural_language_response": "Recent developments in quantum computing for cryptography include...",
    "confidence_score": 0.95,
    "content_grounding_score": 0.87,
    "citations": [
      {
        "title": "Quantum-Resistant Cryptographic Algorithms",
        "authors": ["Smith, J.", "Doe, A."],
        "journal": "Nature Quantum Information",
        "year": 2025,
        "doi": "10.1038/s41534-025-00123-4"
      }
    ]
  },
  "processing_details": {
    "stages_completed": 7,
    "deep_reasoning_applied": true,
    "reasoning_engines_used": [
      "deductive", "inductive", "causal", "probabilistic"
    ],
    "content_sources": 23,
    "marketplace_assets_used": 2
  },
  "metrics": {
    "processing_time": 18.4,
    "cost_ftns": 535,
    "tokens_generated": 2847,
    "papers_analyzed": 23
  }
}
```

#### Get Pipeline Status

```http
GET /pipeline/queries/{query_id}
```

**Response:**
```json
{
  "query_id": "query_66666",
  "status": "processing",
  "current_stage": "deep_reasoning",
  "progress": 0.71,
  "estimated_completion": "2025-07-23T16:02:30Z",
  "stages": [
    {
      "name": "query_analysis",
      "status": "completed",
      "duration": 2.1
    },
    {
      "name": "content_search",
      "status": "completed",
      "duration": 5.3
    },
    {
      "name": "deep_reasoning",
      "status": "in_progress",
      "progress": 0.45
    }
  ]
}
```

## Error Handling

All APIs use standardized error responses:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request body is missing required field 'name'",
    "details": {
      "field": "name",
      "expected_type": "string",
      "received": null
    },
    "request_id": "req_12345",
    "timestamp": "2025-07-23T16:00:00Z"
  }
}
```

### Common Error Codes

- `AUTHENTICATION_REQUIRED`: Missing or invalid authentication token
- `AUTHORIZATION_DENIED`: Insufficient permissions for requested action
- `INVALID_REQUEST`: Malformed request body or parameters
- `RESOURCE_NOT_FOUND`: Requested resource does not exist
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded
- `INTERNAL_ERROR`: Internal server error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## Rate Limiting

API endpoints are rate-limited based on subscription tier:

| Tier | Requests/Minute | Burst Limit |
|------|----------------|-------------|
| Free | 60 | 100 |
| Professional | 300 | 500 |
| Enterprise | 1000 | 2000 |
| Custom | Negotiated | Negotiated |

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 299
X-RateLimit-Reset: 1642777200
```

## Webhooks

Phase 7 APIs support webhook notifications for key events:

### Webhook Configuration

```http
POST /webhooks/subscriptions
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhooks/prsm",
  "events": [
    "pipeline.query.completed",
    "integration.sync.completed",
    "marketplace.review.submitted"
  ],
  "secret": "your_webhook_secret",
  "active": true
}
```

### Webhook Payload Example

```json
{
  "event": "pipeline.query.completed",
  "timestamp": "2025-07-23T16:00:00Z",
  "data": {
    "query_id": "query_66666",
    "user_id": "user_123",
    "status": "completed",
    "processing_time": 18.4,
    "cost_ftns": 535
  },
  "signature": "sha256=abcd1234..."
}
```

## SDK Libraries

Official SDK libraries are available for:

- **Python**: `pip install prsm-sdk`
- **JavaScript/Node.js**: `npm install @prsm/sdk`
- **Java**: Maven/Gradle dependency
- **Go**: Go module
- **Rust**: Cargo crate

### Python SDK Example

```python
from prsm_sdk import PRSMClient

# Initialize client
client = PRSMClient(api_key="your_api_key")

# Process query through unified pipeline
result = await client.pipeline.process_query(
    query="Analyze quantum computing implications",
    verbosity_level="detailed",
    enable_deep_reasoning=True
)

print(f"Response: {result.natural_language_response}")
print(f"Confidence: {result.confidence_score}")
print(f"Cost: {result.cost_ftns} FTNS")
```

This comprehensive API reference provides developers with all the information needed to integrate with PRSM's Phase 7 enterprise architecture components.