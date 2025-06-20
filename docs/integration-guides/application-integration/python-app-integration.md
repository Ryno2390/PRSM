# Python Application Integration Guide

Complete guide for integrating PRSM into Python applications using the native SDK.

## Overview

The PRSM Python SDK provides the most comprehensive integration experience with full async support, automatic error handling, and rich features for Python applications. This guide covers everything from basic usage to advanced patterns.

## Installation

### Standard Installation

```bash
# Install PRSM Python SDK
pip install prsm-sdk

# Or install with all optional dependencies
pip install prsm-sdk[all]
```

### Development Installation

```bash
# For development or bleeding edge features
pip install git+https://github.com/prsm-org/prsm.git#subdirectory=sdks/python

# Or install in editable mode
git clone https://github.com/prsm-org/prsm.git
cd prsm/sdks/python
pip install -e ".[dev]"
```

### Verify Installation

```python
import prsm_sdk
print(f"PRSM SDK version: {prsm_sdk.__version__}")

# Test connection
from prsm_sdk import PRSMClient
client = PRSMClient("http://localhost:8000")
print(f"SDK initialized: {client}")
```

## Quick Start

### Basic Usage

```python
import asyncio
from prsm_sdk import PRSMClient

async def main():
    # Initialize client
    client = PRSMClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )
    
    # Simple query
    response = await client.query(
        prompt="Explain machine learning in simple terms",
        user_id="user123"
    )
    
    print(f"Response: {response.final_answer}")
    print(f"Cost: {response.ftns_charged} FTNS")
    print(f"Time: {response.processing_time:.2f}s")
    
    await client.close()

# Run the example
asyncio.run(main())
```

### Synchronous Wrapper

For non-async applications:

```python
from prsm_sdk import PRSMClient
import asyncio

class SyncPRSMClient:
    def __init__(self, base_url: str, api_key: str):
        self._client = PRSMClient(base_url, api_key)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
    
    def query(self, prompt: str, user_id: str, **kwargs):
        return self._loop.run_until_complete(
            self._client.query(prompt, user_id, **kwargs)
        )
    
    def close(self):
        self._loop.run_until_complete(self._client.close())
        self._loop.close()

# Usage
client = SyncPRSMClient("http://localhost:8000", "your-api-key")
result = client.query("What is AI?", "user123")
print(result.final_answer)
client.close()
```

## Web Framework Integration

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from prsm_sdk import PRSMClient, PRSMError
import asyncio
from typing import Optional

app = FastAPI(title="AI-Powered API")

# Global PRSM client
prsm_client = None

@app.on_event("startup")
async def startup_event():
    global prsm_client
    prsm_client = PRSMClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"
    )

@app.on_event("shutdown")
async def shutdown_event():
    if prsm_client:
        await prsm_client.close()

class QueryRequest(BaseModel):
    prompt: str
    user_id: str
    context_allocation: Optional[int] = 50
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
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
            answer=response.final_answer,
            cost=response.ftns_charged,
            processing_time=response.processing_time,
            quality_score=response.quality_score
        )
    
    except PRSMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/ai/stream/{query}")
async def ai_stream(query: str, user_id: str):
    """Streaming AI response"""
    async def generate():
        try:
            async for chunk in prsm_client.stream(query, user_id=user_id):
                yield f"data: {chunk.json()}\\n\\n"
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\\n\\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test PRSM connection
        health = await prsm_client.ping()
        return {"status": "healthy", "prsm_connected": health}
    except:
        return {"status": "degraded", "prsm_connected": False}
```

### Django Integration

```python
# settings.py
PRSM_CONFIG = {
    'BASE_URL': 'http://localhost:8000',
    'API_KEY': 'your-api-key',
    'DEFAULT_TIMEOUT': 30,
    'MAX_RETRIES': 3
}

# apps.py
from django.apps import AppConfig
from prsm_sdk import PRSMClient
import asyncio

class AIConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai'
    
    def ready(self):
        # Initialize PRSM client
        from django.conf import settings
        self.prsm_client = PRSMClient(
            base_url=settings.PRSM_CONFIG['BASE_URL'],
            api_key=settings.PRSM_CONFIG['API_KEY']
        )

# views.py
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.apps import apps
import json
import asyncio

@csrf_exempt
@require_http_methods(["POST"])
def ai_query_view(request):
    """Django view for AI queries"""
    try:
        data = json.loads(request.body)
        prompt = data.get('prompt')
        user_id = data.get('user_id')
        
        if not prompt or not user_id:
            return JsonResponse({'error': 'Missing prompt or user_id'}, status=400)
        
        # Get PRSM client from app config
        ai_app = apps.get_app_config('ai')
        client = ai_app.prsm_client
        
        # Run async query in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                client.query(prompt, user_id=user_id)
            )
            
            return JsonResponse({
                'answer': response.final_answer,
                'cost': response.ftns_charged,
                'processing_time': response.processing_time
            })
        finally:
            loop.close()
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# For async Django (3.1+)
from django.http import JsonResponse
from asgiref.sync import async_to_sync
import asyncio

async def async_ai_query_view(request):
    """Async Django view for AI queries"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            prompt = data.get('prompt')
            user_id = data.get('user_id')
            
            ai_app = apps.get_app_config('ai')
            response = await ai_app.prsm_client.query(prompt, user_id=user_id)
            
            return JsonResponse({
                'answer': response.final_answer,
                'cost': response.ftns_charged
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
```

### Flask Integration

```python
from flask import Flask, request, jsonify, Response
from prsm_sdk import PRSMClient, PRSMError
import asyncio
import json
from functools import wraps

app = Flask(__name__)

# Initialize PRSM client
prsm_client = PRSMClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

def async_route(f):
    """Decorator to run async functions in Flask"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

@app.route('/ai/query', methods=['POST'])
@async_route
async def ai_query():
    """AI query endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing prompt or user_id'}), 400
        
        response = await prsm_client.query(
            prompt=data['prompt'],
            user_id=data['user_id'],
            context_allocation=data.get('context_allocation', 50)
        )
        
        return jsonify({
            'answer': response.final_answer,
            'cost': response.ftns_charged,
            'processing_time': response.processing_time,
            'quality_score': response.quality_score
        })
    
    except PRSMError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/ai/stream', methods=['POST'])
@async_route
async def ai_stream():
    """Streaming AI response"""
    data = request.get_json()
    
    async def generate():
        try:
            async for chunk in prsm_client.stream(
                data['prompt'], 
                user_id=data['user_id']
            ):
                yield f"data: {json.dumps(chunk.dict())}\\n\\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\\n\\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health')
@async_route
async def health():
    """Health check endpoint"""
    try:
        prsm_healthy = await prsm_client.ping()
        return jsonify({
            'status': 'healthy',
            'prsm_connected': prsm_healthy
        })
    except:
        return jsonify({
            'status': 'degraded',
            'prsm_connected': False
        }), 503

if __name__ == '__main__':
    app.run(debug=True)
```

## Advanced Integration Patterns

### Background Tasks with Celery

```python
# celery_app.py
from celery import Celery
from prsm_sdk import PRSMClient
import asyncio

# Configure Celery
celery_app = Celery('prsm_tasks')
celery_app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0'
)

# Initialize PRSM client
prsm_client = PRSMClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

@celery_app.task(bind=True, max_retries=3)
def process_ai_query(self, prompt: str, user_id: str, **kwargs):
    """Background AI processing task"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                prsm_client.query(prompt, user_id=user_id, **kwargs)
            )
            
            return {
                'success': True,
                'answer': response.final_answer,
                'cost': response.ftns_charged,
                'processing_time': response.processing_time
            }
        finally:
            loop.close()
            
    except Exception as e:
        # Retry on failure
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60, exc=e)
        else:
            return {
                'success': False,
                'error': str(e)
            }

# Usage in your application
from celery.result import AsyncResult

def submit_ai_task(prompt: str, user_id: str):
    """Submit AI processing task"""
    task = process_ai_query.delay(prompt, user_id)
    return task.id

def get_ai_result(task_id: str):
    """Get AI processing result"""
    result = AsyncResult(task_id, app=celery_app)
    
    if result.ready():
        return {
            'status': 'completed',
            'result': result.result
        }
    else:
        return {
            'status': 'pending',
            'progress': result.info if result.info else {}
        }
```

### Database Integration with SQLAlchemy

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from prsm_sdk import PRSMClient
import asyncio

Base = declarative_base()

class AIQuery(Base):
    __tablename__ = 'ai_queries'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    ftns_cost = Column(Float)
    processing_time = Column(Float)
    quality_score = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20), default='pending')  # pending, completed, failed
    error_message = Column(Text)

class AIQueryService:
    def __init__(self, database_url: str, prsm_client: PRSMClient):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.prsm_client = prsm_client
    
    async def process_query(self, prompt: str, user_id: str, **kwargs):
        """Process AI query and store in database"""
        session = self.Session()
        
        # Create query record
        query_record = AIQuery(
            user_id=user_id,
            prompt=prompt,
            status='pending'
        )
        session.add(query_record)
        session.commit()
        
        try:
            # Process with PRSM
            response = await self.prsm_client.query(
                prompt, user_id=user_id, **kwargs
            )
            
            # Update record with results
            query_record.response = response.final_answer
            query_record.ftns_cost = response.ftns_charged
            query_record.processing_time = response.processing_time
            query_record.quality_score = response.quality_score
            query_record.status = 'completed'
            query_record.completed_at = datetime.utcnow()
            
        except Exception as e:
            # Update record with error
            query_record.status = 'failed'
            query_record.error_message = str(e)
            query_record.completed_at = datetime.utcnow()
        
        finally:
            session.commit()
            session.close()
        
        return query_record
    
    def get_user_queries(self, user_id: str, limit: int = 10):
        """Get recent queries for a user"""
        session = self.Session()
        queries = session.query(AIQuery).filter(
            AIQuery.user_id == user_id
        ).order_by(AIQuery.created_at.desc()).limit(limit).all()
        session.close()
        return queries
    
    def get_query_stats(self):
        """Get query statistics"""
        session = self.Session()
        
        stats = {
            'total_queries': session.query(AIQuery).count(),
            'completed_queries': session.query(AIQuery).filter(
                AIQuery.status == 'completed'
            ).count(),
            'failed_queries': session.query(AIQuery).filter(
                AIQuery.status == 'failed'
            ).count(),
            'average_cost': session.query(AIQuery).filter(
                AIQuery.ftns_cost.isnot(None)
            ).with_entities(func.avg(AIQuery.ftns_cost)).scalar() or 0,
            'average_processing_time': session.query(AIQuery).filter(
                AIQuery.processing_time.isnot(None)
            ).with_entities(func.avg(AIQuery.processing_time)).scalar() or 0
        }
        
        session.close()
        return stats

# Usage
async def main():
    prsm_client = PRSMClient("http://localhost:8000", "your-api-key")
    ai_service = AIQueryService("sqlite:///ai_queries.db", prsm_client)
    
    # Process a query
    result = await ai_service.process_query(
        "Explain quantum computing",
        user_id="user123"
    )
    
    print(f"Query {result.id}: {result.status}")
    
    # Get user history
    history = ai_service.get_user_queries("user123")
    for query in history:
        print(f"{query.created_at}: {query.prompt[:50]}...")
    
    # Get statistics
    stats = ai_service.get_query_stats()
    print(f"Statistics: {stats}")

asyncio.run(main())
```

### Caching with Redis

```python
import redis
import json
import hashlib
from typing import Optional, Any
from datetime import datetime, timedelta
from prsm_sdk import PRSMClient

class CachedPRSMClient:
    def __init__(self, prsm_client: PRSMClient, redis_url: str = "redis://localhost:6379"):
        self.client = prsm_client
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 300  # 5 minutes
    
    def _cache_key(self, prompt: str, user_id: str, **kwargs) -> str:
        """Generate cache key"""
        key_data = f"{prompt}:{user_id}:{json.dumps(sorted(kwargs.items()))}"
        return f"prsm:query:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def query(self, prompt: str, user_id: str, use_cache: bool = True, 
                   cache_ttl: Optional[int] = None, **kwargs):
        """Query with caching support"""
        cache_key = self._cache_key(prompt, user_id, **kwargs)
        
        # Try cache first
        if use_cache:
            cached = self.redis.get(cache_key)
            if cached:
                cached_data = json.loads(cached)
                cached_data['from_cache'] = True
                return type('CachedResponse', (), cached_data)()
        
        # Query PRSM
        response = await self.client.query(prompt, user_id=user_id, **kwargs)
        
        # Cache the response
        if use_cache:
            cache_data = {
                'final_answer': response.final_answer,
                'ftns_charged': response.ftns_charged,
                'processing_time': response.processing_time,
                'quality_score': response.quality_score,
                'cached_at': datetime.utcnow().isoformat(),
                'from_cache': False
            }
            
            ttl = cache_ttl or self.default_ttl
            self.redis.setex(cache_key, ttl, json.dumps(cache_data))
        
        return response
    
    def invalidate_cache(self, pattern: str = "prsm:query:*"):
        """Invalidate cache entries"""
        keys = self.redis.keys(pattern)
        if keys:
            return self.redis.delete(*keys)
        return 0
    
    def cache_stats(self) -> dict:
        """Get cache statistics"""
        all_keys = self.redis.keys("prsm:query:*")
        
        stats = {
            'total_entries': len(all_keys),
            'memory_usage': sum(self.redis.memory_usage(key) for key in all_keys),
            'hit_rate': 0  # Would need to track hits/misses
        }
        
        return stats

# Usage
async def main():
    prsm_client = PRSMClient("http://localhost:8000", "your-api-key")
    cached_client = CachedPRSMClient(prsm_client)
    
    # First query (will hit PRSM and cache)
    response1 = await cached_client.query("What is AI?", "user123")
    print(f"First query: {response1.processing_time:.2f}s")
    
    # Second query (will hit cache)
    response2 = await cached_client.query("What is AI?", "user123")
    print(f"Cached query: from_cache={getattr(response2, 'from_cache', False)}")

asyncio.run(main())
```

## Error Handling and Resilience

### Circuit Breaker Pattern

```python
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class ResilientPRSMClient:
    def __init__(self, base_url: str, api_key: str):
        self.client = PRSMClient(base_url, api_key)
        self.circuit_breaker = CircuitBreaker()
    
    async def query(self, prompt: str, user_id: str, **kwargs):
        """Query with circuit breaker protection"""
        return await self.circuit_breaker.call(
            self.client.query, prompt, user_id=user_id, **kwargs
        )

# Usage
async def main():
    resilient_client = ResilientPRSMClient("http://localhost:8000", "your-api-key")
    
    try:
        response = await resilient_client.query("Test query", "user123")
        print(f"Success: {response.final_answer}")
    except Exception as e:
        print(f"Circuit breaker prevented call: {e}")

asyncio.run(main())
```

### Retry with Exponential Backoff

```python
import asyncio
import random
from typing import Callable, Any
from prsm_sdk import PRSMClient, PRSMError

class RetryConfig:
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

async def retry_with_backoff(func: Callable, retry_config: RetryConfig, *args, **kwargs) -> Any:
    """Retry function with exponential backoff"""
    last_exception = None
    
    for attempt in range(retry_config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt == retry_config.max_attempts - 1:
                break
            
            # Calculate delay with jitter
            delay = min(
                retry_config.base_delay * (retry_config.exponential_base ** attempt),
                retry_config.max_delay
            )
            jitter = random.uniform(0.1, 0.2) * delay
            total_delay = delay + jitter
            
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.2f}s...")
            await asyncio.sleep(total_delay)
    
    raise last_exception

class ReliablePRSMClient:
    def __init__(self, base_url: str, api_key: str, retry_config: Optional[RetryConfig] = None):
        self.client = PRSMClient(base_url, api_key)
        self.retry_config = retry_config or RetryConfig()
    
    async def query(self, prompt: str, user_id: str, **kwargs):
        """Query with automatic retry"""
        return await retry_with_backoff(
            self.client.query,
            self.retry_config,
            prompt,
            user_id=user_id,
            **kwargs
        )

# Usage
async def main():
    reliable_client = ReliablePRSMClient(
        "http://localhost:8000", 
        "your-api-key",
        RetryConfig(max_attempts=5, base_delay=2.0)
    )
    
    try:
        response = await reliable_client.query("Test query", "user123")
        print(f"Success after retries: {response.final_answer}")
    except Exception as e:
        print(f"Failed after all retries: {e}")

asyncio.run(main())
```

## Testing

### Unit Testing

```python
import unittest
from unittest.mock import AsyncMock, patch
import asyncio
from prsm_sdk import PRSMClient, PRSMResponse

class TestPRSMIntegration(unittest.TestCase):
    def setUp(self):
        self.client = PRSMClient("http://localhost:8000", "test-api-key")
    
    @patch('prsm_sdk.PRSMClient.query')
    async def test_successful_query(self, mock_query):
        """Test successful query processing"""
        # Mock response
        mock_response = PRSMResponse(
            final_answer="Mocked AI response",
            ftns_charged=25.0,
            processing_time=1.5,
            quality_score=90
        )
        mock_query.return_value = mock_response
        
        # Test query
        result = await self.client.query("Test prompt", "user123")
        
        # Assertions
        self.assertEqual(result.final_answer, "Mocked AI response")
        self.assertEqual(result.ftns_charged, 25.0)
        mock_query.assert_called_once_with("Test prompt", "user123")
    
    @patch('prsm_sdk.PRSMClient.query')
    async def test_query_error_handling(self, mock_query):
        """Test error handling"""
        from prsm_sdk import PRSMError
        mock_query.side_effect = PRSMError("API Error")
        
        with self.assertRaises(PRSMError):
            await self.client.query("Test prompt", "user123")
    
    def test_sync_wrapper(self):
        """Test synchronous wrapper"""
        async def mock_query(prompt, user_id):
            return PRSMResponse(
                final_answer="Sync test response",
                ftns_charged=10.0,
                processing_time=1.0,
                quality_score=85
            )
        
        with patch.object(self.client, 'query', new=mock_query):
            # Use asyncio.run for testing
            result = asyncio.run(self.client.query("Test", "user123"))
            self.assertEqual(result.final_answer, "Sync test response")

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
import pytest
import asyncio
from prsm_sdk import PRSMClient, PRSMError
import os

@pytest.fixture
async def prsm_client():
    """Create PRSM client for testing"""
    client = PRSMClient(
        base_url=os.getenv("PRSM_TEST_URL", "http://localhost:8000"),
        api_key=os.getenv("PRSM_TEST_API_KEY", "test-key")
    )
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_health_check(prsm_client):
    """Test PRSM health check"""
    health = await prsm_client.ping()
    assert health is True

@pytest.mark.asyncio
async def test_simple_query(prsm_client):
    """Test simple query processing"""
    response = await prsm_client.query(
        "What is 2+2?",
        user_id="test_user"
    )
    
    assert response.final_answer is not None
    assert "4" in response.final_answer
    assert response.ftns_charged > 0
    assert response.processing_time > 0

@pytest.mark.asyncio
async def test_streaming_query(prsm_client):
    """Test streaming response"""
    chunks = []
    async for chunk in prsm_client.stream("Count to 5", user_id="test_user"):
        chunks.append(chunk)
        if chunk.is_final:
            break
    
    assert len(chunks) > 1
    assert chunks[-1].is_final

@pytest.mark.asyncio
async def test_error_handling(prsm_client):
    """Test error scenarios"""
    with pytest.raises(PRSMError):
        await prsm_client.query("", user_id="")  # Invalid input

@pytest.mark.asyncio
async def test_session_management(prsm_client):
    """Test session functionality"""
    # Create session
    session = await prsm_client.create_session(
        user_id="test_user",
        initial_context={"test": "context"}
    )
    
    assert session.session_id is not None
    
    # Use session in query
    response = await prsm_client.query(
        "Remember our context",
        user_id="test_user",
        session_id=session.session_id
    )
    
    assert response.final_answer is not None

# Run with: pytest test_integration.py -v
```

## Monitoring and Observability

### Logging Configuration

```python
import logging
from prsm_sdk import PRSMClient

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LoggingPRSMClient:
    def __init__(self, base_url: str, api_key: str):
        self.client = PRSMClient(base_url, api_key)
        self.logger = logging.getLogger(__name__)
    
    async def query(self, prompt: str, user_id: str, **kwargs):
        """Query with comprehensive logging"""
        self.logger.info(
            "Starting PRSM query",
            extra={
                'user_id': user_id,
                'prompt_length': len(prompt),
                'context_allocation': kwargs.get('context_allocation', 50)
            }
        )
        
        try:
            response = await self.client.query(prompt, user_id=user_id, **kwargs)
            
            self.logger.info(
                "PRSM query completed successfully",
                extra={
                    'user_id': user_id,
                    'ftns_charged': response.ftns_charged,
                    'processing_time': response.processing_time,
                    'quality_score': response.quality_score
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "PRSM query failed",
                extra={
                    'user_id': user_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            raise
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
query_counter = Counter('prsm_queries_total', 'Total PRSM queries', ['user_id', 'status'])
query_duration = Histogram('prsm_query_duration_seconds', 'Query duration')
ftns_usage = Histogram('prsm_ftns_usage', 'FTNS tokens used per query')
active_queries = Gauge('prsm_active_queries', 'Currently active queries')

class MetricsPRSMClient:
    def __init__(self, base_url: str, api_key: str):
        self.client = PRSMClient(base_url, api_key)
    
    async def query(self, prompt: str, user_id: str, **kwargs):
        """Query with metrics collection"""
        active_queries.inc()
        start_time = time.time()
        
        try:
            response = await self.client.query(prompt, user_id=user_id, **kwargs)
            
            # Record success metrics
            query_counter.labels(user_id=user_id, status='success').inc()
            query_duration.observe(response.processing_time)
            ftns_usage.observe(response.ftns_charged)
            
            return response
            
        except Exception as e:
            query_counter.labels(user_id=user_id, status='error').inc()
            raise
        finally:
            active_queries.dec()

# Start Prometheus metrics server
start_http_server(8001)
```

## Deployment Best Practices

### Configuration Management

```python
from pydantic import BaseSettings
from typing import Optional

class PRSMSettings(BaseSettings):
    """PRSM configuration with environment variables"""
    prsm_base_url: str = "http://localhost:8000"
    prsm_api_key: str
    prsm_timeout: int = 30
    prsm_max_retries: int = 3
    prsm_cache_ttl: int = 300
    prsm_rate_limit: int = 100
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    
    # Database configuration
    database_url: str = "sqlite:///prsm_app.db"
    
    class Config:
        env_file = ".env"

settings = PRSMSettings()

# Usage in application
prsm_client = PRSMClient(
    base_url=settings.prsm_base_url,
    api_key=settings.prsm_api_key,
    timeout=settings.prsm_timeout
)
```

### Docker Integration

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
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8000"
    environment:
      - PRSM_BASE_URL=http://prsm-server:8000
      - PRSM_API_KEY=${PRSM_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - prsm-server
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  prsm-server:
    image: prsm:latest
    ports:
      - "8000:8000"
    environment:
      - PRSM_ENV=production
```

---

**Next Steps:**
- [Framework Integration Guides](../framework-integration/)
- [Database Integration](../database-integration/)
- [Production Deployment](../platform-integration/)