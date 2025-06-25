# REST API Integration Guide

Complete guide for integrating PRSM into applications using REST API endpoints.

## Overview

PRSM provides a comprehensive REST API that allows any application to integrate AI capabilities regardless of programming language or framework. This guide covers everything from basic queries to advanced features like streaming and session management.

## Prerequisites

- PRSM server running (`prsm serve`)
- API keys configured
- Basic understanding of HTTP requests

## API Endpoints Overview

| Endpoint | Method | Purpose | Auth Required |
|----------|---------|---------|--------------|
| `/health` | GET | Check server status | No |
| `/api/v1/query` | POST | Submit AI queries | Yes |
| `/api/v1/stream` | POST | Stream AI responses | Yes |
| `/api/v1/sessions` | POST | Create sessions | Yes |
| `/api/v1/budget` | GET | Check FTNS balance | Yes |
| `/api/v1/models` | GET | List available models | Yes |

## Quick Start

### 1. Health Check

First, verify PRSM is running:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "components": {
    "nwtn": "operational",
    "ftns": "operational",
    "database": "connected"
  }
}
```

### 2. Basic Query

Submit a simple AI query:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "user_id": "user123",
    "context_allocation": 50
  }'
```

Response:
```json
{
  "query_id": "q_abc123",
  "final_answer": "Quantum computing is a revolutionary technology...",
  "ftns_charged": 45.2,
  "processing_time": 2.8,
  "quality_score": 92,
  "reasoning_trace": [
    "Analyzed query complexity",
    "Selected appropriate explanation level",
    "Generated clear analogies"
  ]
}
```

## Authentication

PRSM supports multiple authentication methods:

### API Key Authentication (Recommended)

```bash
# Header-based
curl -H "Authorization: Bearer your-api-key" ...

# Query parameter (less secure)
curl "http://localhost:8000/api/v1/query?api_key=your-api-key" ...
```

### JWT Token Authentication

```bash
# Get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_user", "password": "your_pass"}'

# Use token
curl -H "Authorization: Bearer jwt-token-here" ...
```

## Core API Usage

### Simple Query Request

```http
POST /api/v1/query
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "prompt": "Your question or instruction here",
  "user_id": "unique_user_identifier",
  "context_allocation": 50,
  "model_preference": "gpt-4",
  "quality_threshold": 85,
  "timeout": 30
}
```

**Parameters:**
- `prompt` (required): The AI query or instruction
- `user_id` (required): Unique identifier for the user
- `context_allocation` (optional): FTNS tokens to allocate (default: 50)
- `model_preference` (optional): Preferred AI model
- `quality_threshold` (optional): Minimum quality score (0-100)
- `timeout` (optional): Request timeout in seconds

### Response Format

```json
{
  "query_id": "unique_query_identifier",
  "final_answer": "AI response content",
  "ftns_charged": 45.2,
  "processing_time": 2.8,
  "quality_score": 92,
  "model_used": "gpt-4",
  "reasoning_trace": ["step1", "step2"],
  "metadata": {
    "tokens_used": 1250,
    "model_calls": 2,
    "cache_hit": false
  }
}
```

### Streaming Responses

For real-time AI responses:

```bash
curl -X POST http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -H "Accept: text/event-stream" \
  -d '{
    "prompt": "Write a story about AI and humans",
    "user_id": "user123"
  }'
```

Stream format (Server-Sent Events):
```
data: {"type": "start", "query_id": "q_123"}

data: {"type": "content", "content": "Once upon a time..."}

data: {"type": "content", "content": " there was an AI..."}

data: {"type": "end", "final_answer": "complete story", "ftns_charged": 30.5}
```

## Session Management

### Create Session

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "user_id": "user123",
    "initial_context": {
      "project": "customer_service",
      "language": "en"
    },
    "session_timeout": 3600
  }'
```

Response:
```json
{
  "session_id": "sess_abc123",
  "user_id": "user123",
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2024-01-15T11:30:00Z",
  "context": {
    "project": "customer_service",
    "language": "en"
  }
}
```

### Use Session in Queries

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "prompt": "What was our previous conversation about?",
    "user_id": "user123",
    "session_id": "sess_abc123"
  }'
```

## Language-Specific Examples

### Python with requests

```python
import requests
import json

class PRSMClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def query(self, prompt, user_id, **kwargs):
        data = {
            "prompt": prompt,
            "user_id": user_id,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/query",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def stream_query(self, prompt, user_id, **kwargs):
        data = {
            "prompt": prompt,
            "user_id": user_id,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/stream",
            headers={**self.headers, "Accept": "text/event-stream"},
            json=data,
            stream=True
        )
        
        for line in response.iter_lines():
            if line.startswith(b'data: '):
                yield json.loads(line[6:])

# Usage
client = PRSMClient("http://localhost:8000", "your-api-key")

# Simple query
result = client.query("What is machine learning?", "user123")
print(result["final_answer"])

# Streaming query
for chunk in client.stream_query("Tell me a story", "user123"):
    if chunk["type"] == "content":
        print(chunk["content"], end="")
```

### JavaScript/Node.js with fetch

```javascript
class PRSMClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }

    async query(prompt, userId, options = {}) {
        const response = await fetch(`${this.baseUrl}/api/v1/query`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                prompt,
                user_id: userId,
                ...options
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async *streamQuery(prompt, userId, options = {}) {
        const response = await fetch(`${this.baseUrl}/api/v1/stream`, {
            method: 'POST',
            headers: {
                ...this.headers,
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify({
                prompt,
                user_id: userId,
                ...options
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    yield JSON.parse(line.substring(6));
                }
            }
        }
    }
}

// Usage
const client = new PRSMClient('http://localhost:8000', 'your-api-key');

// Simple query
client.query('What is blockchain?', 'user123')
    .then(result => console.log(result.final_answer));

// Streaming query
(async () => {
    for await (const chunk of client.streamQuery('Explain AI', 'user123')) {
        if (chunk.type === 'content') {
            process.stdout.write(chunk.content);
        }
    }
})();
```

### Go with net/http

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type PRSMClient struct {
    BaseURL string
    APIKey  string
    Client  *http.Client
}

type QueryRequest struct {
    Prompt            string `json:"prompt"`
    UserID           string `json:"user_id"`
    ContextAllocation int   `json:"context_allocation,omitempty"`
}

type QueryResponse struct {
    QueryID       string   `json:"query_id"`
    FinalAnswer   string   `json:"final_answer"`
    FTNSCharged   float64  `json:"ftns_charged"`
    ProcessingTime float64 `json:"processing_time"`
    QualityScore  int      `json:"quality_score"`
}

func NewPRSMClient(baseURL, apiKey string) *PRSMClient {
    return &PRSMClient{
        BaseURL: baseURL,
        APIKey:  apiKey,
        Client:  &http.Client{},
    }
}

func (c *PRSMClient) Query(prompt, userID string) (*QueryResponse, error) {
    reqBody := QueryRequest{
        Prompt: prompt,
        UserID: userID,
        ContextAllocation: 50,
    }

    jsonBody, _ := json.Marshal(reqBody)
    
    req, _ := http.NewRequest("POST", c.BaseURL+"/api/v1/query", 
        bytes.NewBuffer(jsonBody))
    req.Header.Set("Authorization", "Bearer "+c.APIKey)
    req.Header.Set("Content-Type", "application/json")

    resp, err := c.Client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result QueryResponse
    err = json.NewDecoder(resp.Body).Decode(&result)
    return &result, err
}

// Usage
func main() {
    client := NewPRSMClient("http://localhost:8000", "your-api-key")
    
    result, err := client.Query("What is Go programming?", "user123")
    if err != nil {
        panic(err)
    }
    
    fmt.Println(result.FinalAnswer)
}
```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Response |
|------|---------|----------|
| 200 | Success | Query processed successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Invalid or missing API key |
| 403 | Forbidden | Insufficient permissions |
| 429 | Rate Limited | Too many requests |
| 500 | Server Error | Internal server error |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "Missing required parameter: user_id",
    "details": {
      "parameter": "user_id",
      "expected": "string",
      "provided": "null"
    }
  },
  "request_id": "req_abc123"
}
```

### Robust Error Handling Example

```python
import requests
import time
from typing import Optional, Dict, Any

class PRSMAPIError(Exception):
    def __init__(self, status_code: int, error_data: Dict[str, Any]):
        self.status_code = status_code
        self.error_data = error_data
        super().__init__(f"PRSM API Error {status_code}: {error_data}")

class PRSMClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def query_with_retry(self, prompt: str, user_id: str, 
                        max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """Query with automatic retry and backoff"""
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/query",
                    json={
                        "prompt": prompt,
                        "user_id": user_id,
                        **kwargs
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                else:
                    # Other HTTP errors
                    error_data = response.json() if response.content else {}
                    raise PRSMAPIError(response.status_code, error_data)
                    
            except requests.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"Request failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise
        
        raise Exception("Max retries exceeded")

# Usage with error handling
client = PRSMClient("http://localhost:8000", "your-api-key")

try:
    result = client.query_with_retry("Explain AI", "user123")
    print(result["final_answer"])
except PRSMAPIError as e:
    print(f"API Error: {e.error_data}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedPRSMClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Headers
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PRSM-Client/1.0"
        })
```

### Caching Strategy

```python
import time
from typing import Dict, Any, Tuple

class CachedPRSMClient:
    def __init__(self, base_url: str, api_key: str, cache_ttl: int = 300):
        self.client = PRSMClient(base_url, api_key)
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = cache_ttl
    
    def _cache_key(self, prompt: str, user_id: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        return f"{user_id}:{hash(prompt)}:{hash(str(sorted(kwargs.items())))}"
    
    def query(self, prompt: str, user_id: str, **kwargs) -> Dict[str, Any]:
        cache_key = self._cache_key(prompt, user_id, **kwargs)
        now = time.time()
        
        # Check cache
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if now - timestamp < self.cache_ttl:
                return result
        
        # Query and cache
        result = self.client.query(prompt, user_id, **kwargs)
        self.cache[cache_key] = (result, now)
        
        # Clean expired entries
        self._cleanup_cache(now)
        
        return result
    
    def _cleanup_cache(self, now: float):
        """Remove expired cache entries"""
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
```

## Production Deployment

### Environment Configuration

```bash
# Environment variables
export PRSM_API_URL="https://api.prsm.yourcompany.com"
export PRSM_API_KEY="your-production-api-key"
export PRSM_TIMEOUT=30
export PRSM_MAX_RETRIES=3
export PRSM_CACHE_TTL=300
```

### Health Monitoring

```python
import logging
from datetime import datetime, timedelta

class HealthMonitor:
    def __init__(self, client: PRSMClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.last_health_check = None
        self.health_status = True
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            response = requests.get(f"{self.client.base_url}/health", timeout=5)
            self.health_status = response.status_code == 200
            self.last_health_check = datetime.now()
            
            if not self.health_status:
                self.logger.error(f"Health check failed: {response.status_code}")
            
            return self.health_status
        except Exception as e:
            self.health_status = False
            self.logger.error(f"Health check exception: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        if self.last_health_check is None:
            return False
        
        # Consider stale if no check in last 5 minutes
        if datetime.now() - self.last_health_check > timedelta(minutes=5):
            return False
        
        return self.health_status
```

## Security Best Practices

### API Key Management

```python
import os
from typing import Optional

class SecurePRSMClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # Load from environment if not provided
        self.base_url = base_url or os.getenv('PRSM_API_URL')
        self.api_key = api_key or os.getenv('PRSM_API_KEY')
        
        if not self.base_url or not self.api_key:
            raise ValueError("PRSM_API_URL and PRSM_API_KEY must be set")
        
        # Validate API key format
        if not self.api_key.startswith(('pk_', 'sk_')):
            raise ValueError("Invalid API key format")
```

### Input Validation

```python
import re
from typing import Any, Dict

class ValidatedPRSMClient:
    MAX_PROMPT_LENGTH = 10000
    ALLOWED_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    def validate_input(self, prompt: str, user_id: str, **kwargs) -> None:
        """Validate input parameters"""
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt too long: {len(prompt)} > {self.MAX_PROMPT_LENGTH}")
        
        if not self.ALLOWED_USER_ID_PATTERN.match(user_id):
            raise ValueError("Invalid user_id format")
        
        # Validate numeric parameters
        if 'context_allocation' in kwargs:
            allocation = kwargs['context_allocation']
            if not isinstance(allocation, int) or allocation < 1 or allocation > 1000:
                raise ValueError("context_allocation must be between 1 and 1000")
    
    def query(self, prompt: str, user_id: str, **kwargs) -> Dict[str, Any]:
        self.validate_input(prompt, user_id, **kwargs)
        return super().query(prompt, user_id, **kwargs)
```

## Troubleshooting

### Common Issues

**Connection Refused**
```bash
# Check if PRSM server is running
curl http://localhost:8000/health

# Check logs
docker logs prsm-api
```

**Authentication Errors**
```bash
# Verify API key
echo $PRSM_API_KEY

# Test with curl
curl -H "Authorization: Bearer $PRSM_API_KEY" \
     http://localhost:8000/api/v1/models
```

**Rate Limiting**
```python
# Implement exponential backoff
def exponential_backoff(attempt: int) -> float:
    return min(60, (2 ** attempt) + random.uniform(0, 1))
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add request logging
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1
```

---

**Next Steps:**
- [Python Application Integration](./python-app-integration.md)
- [Framework Integration Guides](../framework-integration/)
- [Production Deployment Guide](../platform-integration/)