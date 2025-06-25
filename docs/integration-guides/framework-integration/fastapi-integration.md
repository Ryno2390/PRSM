# FastAPI Integration Guide

Complete guide for integrating PRSM with FastAPI applications to create AI-powered APIs.

## Overview

FastAPI is a modern, fast web framework for building APIs with Python. This guide shows how to integrate PRSM into FastAPI applications to add AI capabilities with minimal setup.

## Benefits of FastAPI + PRSM

- **Async Support**: Native async/await support for optimal performance
- **Type Safety**: Pydantic models for request/response validation
- **Auto Documentation**: OpenAPI/Swagger docs generation
- **High Performance**: One of the fastest Python web frameworks
- **Easy Integration**: Simple PRSM client integration patterns

## Installation

```bash
# Install FastAPI and PRSM SDK
pip install fastapi uvicorn prsm-sdk

# Optional: Authentication and monitoring
pip install python-jose[cryptography] prometheus-client
```

## Quick Start

### Basic AI-Powered API

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from prsm_sdk import PRSMClient, PRSMError
import asyncio
from typing import Optional

app = FastAPI(title="AI-Powered API", version="1.0.0")

# Global PRSM client
prsm_client = None

@app.on_event("startup")
async def startup_event():
    global prsm_client
    prsm_client = PRSMClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )
    print("🚀 PRSM client initialized")

@app.on_event("shutdown")
async def shutdown_event():
    if prsm_client:
        await prsm_client.close()
    print("🛑 PRSM client closed")

# Pydantic models
class QueryRequest(BaseModel):
    prompt: str
    user_id: str
    context_allocation: Optional[int] = 50
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    cost: float
    processing_time: float
    quality_score: int

@app.post("/ai/query", response_model=QueryResponse)
async def ai_query(request: QueryRequest):
    """AI-powered query endpoint"""
    try:
        response = await prsm_client.query(
            prompt=request.prompt,
            user_id=request.user_id,
            context_allocation=request.context_allocation,
            session_id=request.session_id
        )
        
        return QueryResponse(
            query_id=response.query_id,
            answer=response.final_answer,
            cost=response.ftns_charged,
            processing_time=response.processing_time,
            quality_score=response.quality_score
        )
    
    except PRSMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        prsm_healthy = await prsm_client.ping()
        return {
            "status": "healthy",
            "prsm_connected": prsm_healthy
        }
    except:
        return {
            "status": "degraded",
            "prsm_connected": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Advanced Integration Patterns

### Dependency Injection for PRSM Client

```python
from fastapi import Depends
from contextlib import asynccontextmanager

class PRSMService:
    def __init__(self):
        self.client: Optional[PRSMClient] = None
    
    async def initialize(self):
        self.client = PRSMClient(
            base_url="http://localhost:8000",
            api_key="your-api-key"
        )
    
    async def close(self):
        if self.client:
            await self.client.close()
    
    async def query(self, prompt: str, user_id: str, **kwargs):
        if not self.client:
            raise HTTPException(status_code=503, detail="PRSM service not available")
        return await self.client.query(prompt, user_id=user_id, **kwargs)

# Global service instance
prsm_service = PRSMService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await prsm_service.initialize()
    yield
    # Shutdown
    await prsm_service.close()

app = FastAPI(lifespan=lifespan)

# Dependency injection
async def get_prsm_service() -> PRSMService:
    return prsm_service

@app.post("/ai/query")
async def ai_query(
    request: QueryRequest,
    prsm: PRSMService = Depends(get_prsm_service)
):
    response = await prsm.query(
        request.prompt,
        request.user_id,
        context_allocation=request.context_allocation
    )
    return QueryResponse(...)
```

### Streaming Responses

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/ai/stream")
async def ai_stream(request: QueryRequest):
    """Streaming AI response endpoint"""
    async def generate():
        try:
            async for chunk in prsm_client.stream(
                request.prompt, 
                user_id=request.user_id
            ):
                yield f"data: {json.dumps(chunk.dict())}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
```

### Authentication and Authorization

```python
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserInDB(BaseModel):
    username: str
    hashed_password: str
    prsm_quota: int = 1000
    security_level: str = "standard"

# Mock user database
fake_users_db = {
    "testuser": UserInDB(
        username="testuser",
        hashed_password=pwd_context.hash("testpass"),
        prsm_quota=1000
    )
}

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/auth/login")
async def login(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not pwd_context.verify(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = jwt.encode(
        {"sub": username, "exp": datetime.utcnow() + access_token_expires},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/ai/query")
async def ai_query(
    request: QueryRequest,
    current_user: str = Depends(verify_token)
):
    # Check user quota
    user = fake_users_db.get(current_user)
    if user and user.prsm_quota <= 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="PRSM quota exceeded"
        )
    
    # Process query with user context
    response = await prsm_client.query(
        prompt=request.prompt,
        user_id=current_user,  # Use authenticated user ID
        context_allocation=request.context_allocation
    )
    
    # Update user quota
    if user:
        user.prsm_quota -= int(response.ftns_charged)
    
    return QueryResponse(...)
```

### Background Tasks and Job Queues

```python
from fastapi import BackgroundTasks
import asyncio
from uuid import uuid4

# Job storage (use Redis in production)
job_storage = {}

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

async def process_ai_job(job_id: str, prompt: str, user_id: str):
    """Background task for processing AI queries"""
    job_storage[job_id]["status"] = "processing"
    
    try:
        response = await prsm_client.query(prompt, user_id=user_id)
        job_storage[job_id].update({
            "status": "completed",
            "result": {
                "answer": response.final_answer,
                "cost": response.ftns_charged,
                "processing_time": response.processing_time
            },
            "completed_at": datetime.utcnow()
        })
    except Exception as e:
        job_storage[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow()
        })

@app.post("/ai/jobs")
async def create_ai_job(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Create an AI processing job"""
    job_id = str(uuid4())
    
    # Create job entry
    job_storage[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow()
    }
    
    # Schedule background task
    background_tasks.add_task(
        process_ai_job,
        job_id,
        request.prompt,
        request.user_id
    )
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/ai/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_storage[job_id]
```

### Caching with Redis

```python
import redis.asyncio as redis
import json
import hashlib
from typing import Optional

class CacheService:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = 300  # 5 minutes
    
    def _cache_key(self, prompt: str, user_id: str, **kwargs) -> str:
        key_data = f"{prompt}:{user_id}:{json.dumps(sorted(kwargs.items()))}"
        return f"prsm:query:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def get_cached_response(self, prompt: str, user_id: str, **kwargs) -> Optional[dict]:
        key = self._cache_key(prompt, user_id, **kwargs)
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    async def cache_response(self, prompt: str, user_id: str, response: dict, **kwargs):
        key = self._cache_key(prompt, user_id, **kwargs)
        await self.redis.setex(key, self.ttl, json.dumps(response))

# Initialize cache service
cache_service = CacheService()

@app.post("/ai/query")
async def ai_query_with_cache(request: QueryRequest):
    """AI query with caching"""
    # Check cache first
    cached_response = await cache_service.get_cached_response(
        request.prompt,
        request.user_id,
        context_allocation=request.context_allocation
    )
    
    if cached_response:
        cached_response["from_cache"] = True
        return cached_response
    
    # Process query
    response = await prsm_client.query(
        prompt=request.prompt,
        user_id=request.user_id,
        context_allocation=request.context_allocation
    )
    
    # Cache the response
    response_data = {
        "query_id": response.query_id,
        "answer": response.final_answer,
        "cost": response.ftns_charged,
        "processing_time": response.processing_time,
        "quality_score": response.quality_score,
        "from_cache": False
    }
    
    await cache_service.cache_response(
        request.prompt,
        request.user_id,
        response_data,
        context_allocation=request.context_allocation
    )
    
    return response_data
```

## Monitoring and Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
query_counter = Counter('prsm_queries_total', 'Total PRSM queries', ['user_id', 'status'])
query_duration = Histogram('prsm_query_duration_seconds', 'Query duration')
ftns_usage = Histogram('prsm_ftns_usage', 'FTNS tokens used per query')
active_queries = Gauge('prsm_active_queries', 'Currently active queries')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    if request.url.path.startswith("/ai/"):
        active_queries.inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record success metrics
            processing_time = time.time() - start_time
            query_duration.observe(processing_time)
            
            return response
        except Exception as e:
            # Record error metrics
            raise
        finally:
            active_queries.dec()
    else:
        return await call_next(request)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("prsm_api")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_query(self, user_id: str, prompt: str, response: dict, processing_time: float):
        log_data = {
            "event": "ai_query",
            "user_id": user_id,
            "prompt_length": len(prompt),
            "ftns_charged": response.get("cost", 0),
            "processing_time": processing_time,
            "quality_score": response.get("quality_score", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))

logger = StructuredLogger()

@app.post("/ai/query")
async def ai_query_with_logging(request: QueryRequest):
    start_time = time.time()
    
    try:
        response = await prsm_client.query(
            prompt=request.prompt,
            user_id=request.user_id,
            context_allocation=request.context_allocation
        )
        
        processing_time = time.time() - start_time
        
        response_data = QueryResponse(
            query_id=response.query_id,
            answer=response.final_answer,
            cost=response.ftns_charged,
            processing_time=response.processing_time,
            quality_score=response.quality_score
        )
        
        # Log successful query
        logger.log_query(
            request.user_id,
            request.prompt,
            response_data.dict(),
            processing_time
        )
        
        return response_data
        
    except Exception as e:
        logger.logger.error(f"Query failed for user {request.user_id}: {str(e)}")
        raise
```

## Testing

### Unit Tests with pytest

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

@pytest.fixture
def client():
    from main import app
    return TestClient(app)

@pytest.fixture
def mock_prsm_client():
    with patch('main.prsm_client') as mock:
        mock.query = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        yield mock

def test_health_check(client, mock_prsm_client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ai_query(client, mock_prsm_client):
    # Mock PRSM response
    mock_response = type('MockResponse', (), {
        'query_id': 'test_123',
        'final_answer': 'Test response',
        'ftns_charged': 25.0,
        'processing_time': 1.5,
        'quality_score': 90
    })()
    mock_prsm_client.query.return_value = mock_response
    
    response = client.post("/ai/query", json={
        "prompt": "Test prompt",
        "user_id": "test_user"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test response"
    assert data["cost"] == 25.0

def test_ai_query_error(client, mock_prsm_client):
    from prsm_sdk import PRSMError
    mock_prsm_client.query.side_effect = PRSMError("Test error")
    
    response = client.post("/ai/query", json={
        "prompt": "Test prompt",
        "user_id": "test_user"
    })
    
    assert response.status_code == 400
    assert "Test error" in response.json()["detail"]
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_integration_with_real_prsm():
    """Integration test with real PRSM instance"""
    from prsm_sdk import PRSMClient
    
    client = PRSMClient("http://localhost:8000", "test-api-key")
    
    try:
        response = await client.query("What is 2+2?", user_id="test_user")
        assert response.final_answer is not None
        assert "4" in response.final_answer
    finally:
        await client.close()
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PRSM_API_URL=http://prsm:8000
      - PRSM_API_KEY=${PRSM_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - prsm
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  prsm:
    image: prsm:latest
    ports:
      - "8000:8000"
```

### Production Configuration

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # PRSM Configuration
    prsm_api_url: str = "http://localhost:8000"
    prsm_api_key: str
    prsm_timeout: int = 30
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 300
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Monitoring
    log_level: str = "INFO"
    metrics_enabled: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

**Next Steps:**
- [Django Integration](./django-integration.md)
- [Flask Integration](./flask-integration.md)
- [React Integration](./react-integration.md)
- [Production Deployment](../platform-integration/)
