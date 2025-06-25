# API Fundamentals

Learn to use PRSM's REST API and Python SDK effectively for integration and development.

## ⏱️ Time: 15 minutes

## 🎯 Learning Goals

- Master PRSM REST API endpoints
- Use Python SDK for programmatic access
- Handle authentication and rate limiting
- Implement error handling and retries

## 🌐 PRSM API Overview

PRSM provides two primary interfaces:

1. **REST API**: HTTP endpoints for any programming language
2. **Python SDK**: Native Python integration with async support

## 🔌 REST API Basics

### Starting the API Server

```bash
# Start PRSM API server
prsm serve --host 0.0.0.0 --port 8000

# Or with auto-reload for development
prsm serve --reload
```

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/health` | GET | System health check |
| `/api/v1/query` | POST | Submit AI queries |
| `/api/v1/models` | GET | List available models |
| `/api/v1/sessions` | POST | Create user sessions |
| `/api/v1/budget` | GET | Check FTNS balance |

### API Examples

#### 1. Health Check

```bash
# Check if PRSM is running
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
    "database": "connected",
    "redis": "connected"
  }
}
```

#### 2. Submit Query via REST

```bash
# Submit a query using curl
curl -X POST http://localhost:8000/api/v1/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "api_test_user",
    "prompt": "Explain quantum computing in simple terms",
    "context_allocation": 30,
    "model_preference": "gpt-4"
  }'
```

Response:
```json
{
  "query_id": "q_12345",
  "final_answer": "Quantum computing uses quantum mechanics...",
  "ftns_charged": 28.5,
  "processing_time": 3.2,
  "quality_score": 95,
  "reasoning_trace": [
    "Analyzed query complexity",
    "Selected appropriate explanation level",
    "Generated clear analogies"
  ]
}
```

#### 3. Python REST Client

```python
#!/usr/bin/env python3
\"\"\"
PRSM REST API Client Example
Using requests library to interact with PRSM API
\"\"\"

import requests
import json
import time

class PRSMRestClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        \"\"\"Check if PRSM API is healthy\"\"\"
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def submit_query(self, user_id, prompt, context_allocation=50, **kwargs):
        \"\"\"Submit a query to PRSM\"\"\"
        payload = {
            "user_id": user_id,
            "prompt": prompt,
            "context_allocation": context_allocation,
            **kwargs
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/query",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_models(self):
        \"\"\"List available AI models\"\"\"
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

# Example usage
def main():
    print("🌐 PRSM REST API Demo")
    print("-" * 30)
    
    client = PRSMRestClient()
    
    # Health check
    print("🏥 Checking API health...")
    health = client.health_check()
    if "error" in health:
        print(f"❌ API unavailable: {health['error']}")
        return
    print(f"✅ API Status: {health['status']}")
    
    # List models
    print("\\n🤖 Available models...")
    models = client.get_models()
    if "models" in models:
        for model in models["models"]:
            print(f"   • {model['name']} - {model['description']}")
    
    # Submit query
    print("\\n📝 Submitting query...")
    result = client.submit_query(
        user_id="rest_demo_user",
        prompt="What are the benefits of renewable energy?",
        context_allocation=40
    )
    
    if "error" in result:
        print(f"❌ Query failed: {result['error']}")
        return
    
    print("✅ Query successful!")
    print(f"💰 FTNS Used: {result['ftns_charged']}")
    print(f"⏱️  Time: {result['processing_time']:.2f}s")
    print(f"🤖 Response: {result['final_answer'][:100]}...")

if __name__ == "__main__":
    main()
```

## 🐍 Python SDK

The Python SDK provides a more integrated experience with async support and automatic error handling.

### Installation and Import

```python
# SDK is included with PRSM installation
from prsm_sdk import PRSMClient
from prsm_sdk.models import QueryRequest, PRSMSession
```

### Basic SDK Usage

```python
#!/usr/bin/env python3
\"\"\"
PRSM Python SDK Examples
Comprehensive guide to using the PRSM SDK
\"\"\"

import asyncio
from prsm_sdk import PRSMClient
from prsm_sdk.models import QueryRequest
from prsm_sdk.exceptions import PRSMError, InsufficientFTNSError

async def sdk_examples():
    print("🐍 PRSM Python SDK Demo")
    print("-" * 30)
    
    # Initialize client
    client = PRSMClient(
        base_url="http://localhost:8000",
        timeout=30
    )
    
    try:
        # Example 1: Simple Query
        print("\\n📝 Example 1: Simple Query")
        response = await client.query(
            prompt="What is machine learning?",
            user_id="sdk_demo_user",
            context_allocation=25
        )
        print(f"✅ Response: {response.final_answer[:100]}...")
        print(f"💰 Cost: {response.ftns_charged} FTNS")
        
        # Example 2: Streaming Response
        print("\\n🌊 Example 2: Streaming Response")
        print("Streaming response chunks:")
        async for chunk in client.stream(
            prompt="Tell me a story about AI and humans working together",
            user_id="sdk_demo_user"
        ):
            print(f"   📦 Chunk: {chunk.content[:50]}...")
            if chunk.is_final:
                print(f"   ✅ Final cost: {chunk.ftns_charged} FTNS")
                break
        
        # Example 3: Cost Estimation
        print("\\n💰 Example 3: Cost Estimation")
        estimated_cost = await client.estimate_cost(
            prompt="Write a detailed research paper on climate change impacts",
            model="gpt-4"
        )
        print(f"📊 Estimated cost: {estimated_cost} FTNS")
        
        # Example 4: Batch Processing
        print("\\n📦 Example 4: Batch Processing")
        queries = [
            "What is photosynthesis?",
            "How do solar panels work?", 
            "Explain the water cycle"
        ]
        
        results = await client.batch_query(
            prompts=queries,
            user_id="sdk_demo_user",
            context_allocation=20
        )
        
        for i, result in enumerate(results):
            print(f"   Query {i+1}: {result.ftns_charged} FTNS")
        
    except InsufficientFTNSError as e:
        print(f"❌ Insufficient FTNS tokens: {e}")
    except PRSMError as e:
        print(f"❌ PRSM Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(sdk_examples())
```

### Advanced SDK Features

```python
#!/usr/bin/env python3
\"\"\"
Advanced PRSM SDK Features
Rate limiting, retries, custom configurations
\"\"\"

import asyncio
from prsm_sdk import PRSMClient
from prsm_sdk.config import ClientConfig

async def advanced_sdk_features():
    print("🚀 Advanced PRSM SDK Features")
    print("-" * 35)
    
    # Custom configuration
    config = ClientConfig(
        retry_attempts=3,
        retry_delay=1.0,
        rate_limit_per_minute=30,
        default_timeout=45
    )
    
    client = PRSMClient(config=config)
    
    # Example 1: Custom Model Selection
    print("\\n🎯 Example 1: Custom Model Selection")
    response = await client.query(
        prompt="Solve this calculus problem: ∫x²dx",
        model_preference="math_specialist",
        fallback_models=["gpt-4", "claude-3"],
        user_id="advanced_user"
    )
    print(f"✅ Model used: {response.model_used}")
    print(f"📝 Solution: {response.final_answer}")
    
    # Example 2: Context-Aware Conversation
    print("\\n💬 Example 2: Context-Aware Conversation")
    
    # Start a session
    session = await client.create_session(
        user_id="conversation_user",
        initial_context="You are helping a student learn physics"
    )
    
    # Multiple related queries
    questions = [
        "What is Newton's first law?",
        "Can you give me an example?",
        "How does this relate to momentum?"
    ]
    
    for question in questions:
        response = await client.query(
            prompt=question,
            session_id=session.session_id,
            context_allocation=30
        )
        print(f"Q: {question}")
        print(f"A: {response.final_answer[:80]}...")
        print()
    
    # Example 3: Quality and Performance Monitoring
    print("\\n📊 Example 3: Quality Monitoring")
    
    # Query with quality tracking
    response = await client.query(
        prompt="Explain blockchain technology for beginners",
        quality_threshold=85,  # Minimum quality score
        user_id="quality_user"
    )
    
    print(f"📈 Quality Score: {response.quality_score}/100")
    print(f"⏱️  Processing Time: {response.processing_time:.2f}s")
    print(f"🎯 Met Quality Threshold: {response.quality_score >= 85}")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(advanced_sdk_features())
```

## 🔐 Authentication & Security

PRSM uses API keys and user sessions for security:

```python
# Authentication example
from prsm_sdk import PRSMClient

# Using API key authentication
client = PRSMClient(
    api_key="your-prsm-api-key",
    base_url="https://api.prsm.org"
)

# Session-based authentication
session = await client.authenticate(
    username="your_username",
    password="your_password"
)
```

## 🚨 Error Handling Best Practices

```python
import asyncio
from prsm_sdk import PRSMClient
from prsm_sdk.exceptions import (
    PRSMError, 
    InsufficientFTNSError,
    ModelUnavailableError,
    RateLimitError
)

async def robust_query_handler():
    client = PRSMClient()
    
    try:
        response = await client.query(
            prompt="Your query here",
            user_id="your_user"
        )
        return response
        
    except InsufficientFTNSError:
        print("❌ Not enough FTNS tokens. Please add more to your account.")
        
    except ModelUnavailableError as e:
        print(f"❌ Requested model unavailable: {e.model}")
        print("🔄 Trying with fallback model...")
        # Retry with different model
        response = await client.query(
            prompt="Your query here",
            model_preference="gpt-3.5-turbo",
            user_id="your_user"
        )
        return response
        
    except RateLimitError as e:
        print(f"⏳ Rate limited. Retry after {e.retry_after} seconds")
        await asyncio.sleep(e.retry_after)
        # Retry the request
        
    except PRSMError as e:
        print(f"❌ PRSM API Error: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return None
```

## 🎯 Key Takeaways

1. **REST API**: Universal access from any programming language
2. **Python SDK**: Native async integration with rich features
3. **Error Handling**: Robust error handling prevents failures
4. **Authentication**: Secure API access with keys and sessions
5. **Monitoring**: Track costs, quality, and performance

## 🚀 Practice Exercises

1. **Build a CLI Tool**: Create a command-line interface using the REST API
2. **Chat Bot**: Use the SDK to build a conversational AI assistant
3. **Batch Processor**: Process multiple queries efficiently
4. **Monitor Dashboard**: Track API usage and costs

## 📚 Reference Links

- [Complete API Documentation](../../API_REFERENCE.md)
- [SDK Source Code](../../../sdks/python/)
- [Error Handling Guide](../../TROUBLESHOOTING_GUIDE.md)

---

**API Mastery Achieved!** 🌐

**Next Tutorial** → [Configuration Deep Dive](./configuration.md)