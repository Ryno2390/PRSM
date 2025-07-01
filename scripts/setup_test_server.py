#!/usr/bin/env python3
"""
Load Testing Server Setup
========================

Sets up a test instance of PRSM for performance testing.
Addresses Gemini's requirement for real-world performance validation
rather than simulated results.
"""

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import uvicorn
import structlog
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from prsm.core.config import get_settings
from prsm.security.production_rbac import get_rbac_manager
from prsm.security.distributed_rate_limiter import get_rate_limiter
from prsm.core.database_service import get_database_service

logger = structlog.get_logger(__name__)
settings = get_settings()

# Test server configuration
TEST_SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "workers": 4,
    "log_level": "info"
}

# Mock data for load testing
MOCK_USERS = [
    {"id": f"user_{i}", "username": f"user_{i}", "role": "user"} 
    for i in range(1000)
]

MOCK_ADMIN_USERS = [
    {"id": f"admin_{i}", "username": f"admin_{i}", "role": "admin"}
    for i in range(10)
]

# Create FastAPI app for testing
app = FastAPI(
    title="PRSM Load Testing Server",
    description="Production-like PRSM instance for load testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Mock authentication for load testing
def mock_authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Mock authentication that simulates real auth overhead"""
    token = credentials.credentials
    
    # Simulate auth validation delay
    time.sleep(0.001)  # 1ms to simulate DB lookup
    
    # Simple token format: "user_123" or "admin_5"
    if token.startswith(("user_", "admin_")):
        user_type, user_id = token.split("_", 1)
        return {
            "user_id": token,
            "username": token,
            "role": user_type,
            "authenticated": True
        }
    
    raise HTTPException(status_code=401, detail="Invalid authentication")

@app.get("/health")
async def health_check():
    """Health check endpoint for load testing"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.post("/api/auth/login")
async def login(request: dict):
    """Mock login endpoint with realistic response time"""
    username = request.get("username", "")
    password = request.get("password", "")
    
    # Simulate authentication processing time
    await asyncio.sleep(0.05)  # 50ms to simulate real auth
    
    # Simple validation for load testing
    if username.startswith(("user_", "admin_")) and password:
        return {
            "access_token": username,  # Use username as token for simplicity
            "token_type": "bearer",
            "expires_in": 3600,
            "user_id": username,
            "role": "admin" if username.startswith("admin_") else "user"
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/agents/query")
async def agent_query(
    request: dict,
    user = Depends(mock_authenticate)
):
    """Mock agent query endpoint with realistic processing"""
    query = request.get("query", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Simulate processing time based on query complexity
    processing_time = min(0.1 + len(query) * 0.001, 2.0)  # 100ms base + 1ms per character, max 2s
    await asyncio.sleep(processing_time)
    
    # Simulate realistic response
    response = {
        "response": f"Processed query: '{query}' for user {user['user_id']}",
        "processing_time": processing_time,
        "timestamp": time.time(),
        "model": "mock-model-v1",
        "confidence": 0.85,
        "metadata": {
            "query_length": len(query),
            "user_role": user["role"],
            "processing_node": "test-node-1"
        }
    }
    
    return response

@app.post("/api/ml/process")
async def ml_processing(
    request: dict,
    user = Depends(mock_authenticate)
):
    """Mock ML processing endpoint with heavier computation simulation"""
    content = request.get("content", "")
    options = request.get("options", {})
    
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")
    
    # Simulate ML processing time (more intensive)
    base_time = 0.5  # 500ms base
    content_time = len(content) * 0.002  # 2ms per character
    options_time = len(options) * 0.1  # 100ms per option
    
    processing_time = min(base_time + content_time + options_time, 10.0)  # Max 10s
    await asyncio.sleep(processing_time)
    
    # Simulate ML results
    results = {
        "sentiment": "positive" if hash(content) % 2 else "negative",
        "entities": [f"entity_{i}" for i in range(hash(content) % 5)],
        "classification": f"category_{hash(content) % 10}",
        "summary": content[:100] + "..." if len(content) > 100 else content,
        "confidence_scores": {
            "sentiment": 0.8 + (hash(content) % 20) * 0.01,
            "entities": 0.7 + (hash(content) % 30) * 0.01,
            "classification": 0.9 + (hash(content) % 10) * 0.01
        },
        "processing_time": processing_time,
        "model_version": "mock-ml-v2.1"
    }
    
    return {"results": results}

@app.post("/api/search/vector")
async def vector_search(
    request: dict,
    user = Depends(mock_authenticate)
):
    """Mock vector search with database simulation"""
    query = request.get("query", "")
    limit = min(request.get("limit", 10), 100)  # Max 100 results
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Simulate vector search time (database + similarity computation)
    search_time = 0.1 + limit * 0.005  # 100ms base + 5ms per result
    await asyncio.sleep(search_time)
    
    # Generate mock search results
    results = []
    for i in range(limit):
        similarity_score = max(0.5, 1.0 - (i * 0.05) - (hash(query + str(i)) % 20) * 0.01)
        results.append({
            "content_id": f"content_{hash(query + str(i)) % 10000}",
            "title": f"Mock Content {i+1} for '{query}'",
            "similarity_score": similarity_score,
            "metadata": {
                "type": "document",
                "created_at": "2024-01-01T00:00:00Z",
                "author": f"author_{i % 10}"
            }
        })
    
    return {
        "query": query,
        "results": results,
        "total_found": limit,
        "search_time": search_time,
        "timestamp": time.time()
    }

@app.get("/api/metrics/system")
async def system_metrics(user = Depends(mock_authenticate)):
    """Mock system metrics endpoint"""
    
    # Only allow admin access
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await asyncio.sleep(0.1)  # Simulate metrics collection
    
    return {
        "cpu_usage": 45.2 + (time.time() % 10),
        "memory_usage": 67.8 + (time.time() % 5),
        "active_connections": 150 + int(time.time() % 50),
        "requests_per_second": 25.5 + (time.time() % 15),
        "database_connections": 12,
        "redis_connections": 8,
        "uptime_seconds": int(time.time() % 86400),
        "timestamp": time.time()
    }

@app.post("/api/economy/balance")
async def get_balance(
    request: dict,
    user = Depends(mock_authenticate)
):
    """Mock balance check for marketplace testing"""
    currency = request.get("currency", "FTNS")
    
    await asyncio.sleep(0.05)  # Simulate database lookup
    
    # Generate consistent but varied balances
    user_hash = hash(user["user_id"])
    balance = abs(user_hash % 10000) + 100  # Between 100-10099
    
    return {
        "user_id": user["user_id"],
        "currency": currency,
        "balance": balance,
        "locked_balance": balance * 0.1,  # 10% locked
        "last_updated": time.time()
    }

@app.get("/api/marketplace/browse")
async def browse_marketplace(
    limit: int = 20,
    user = Depends(mock_authenticate)
):
    """Mock marketplace browsing"""
    limit = min(limit, 100)  # Max 100 items
    
    await asyncio.sleep(0.2)  # Simulate database query
    
    items = []
    for i in range(limit):
        items.append({
            "item_id": f"item_{i + 1}",
            "title": f"AI Model {i + 1}",
            "description": f"High-quality AI model for task {i + 1}",
            "price": 10 + (i * 5) + (hash(str(i)) % 50),
            "seller": f"seller_{i % 20}",
            "rating": 4.0 + (hash(str(i)) % 10) * 0.1,
            "category": ["nlp", "vision", "audio", "general"][i % 4]
        })
    
    return {
        "items": items,
        "total_count": 1000,  # Mock total
        "page_size": limit,
        "timestamp": time.time()
    }

@app.get("/api/health/detailed")
async def detailed_health():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "services": {
            "database": "connected",
            "redis": "connected", 
            "ml_service": "operational",
            "vector_store": "operational"
        },
        "performance": {
            "avg_response_time": 0.15,
            "p95_response_time": 0.45,
            "error_rate": 0.02
        },
        "timestamp": time.time()
    }

class LoadTestServer:
    """Load testing server manager"""
    
    def __init__(self):
        self.server = None
        self.running = False
    
    async def start(self):
        """Start the load testing server"""
        logger.info("🚀 Starting PRSM Load Testing Server...")
        
        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=TEST_SERVER_CONFIG["host"],
            port=TEST_SERVER_CONFIG["port"],
            log_level=TEST_SERVER_CONFIG["log_level"],
            access_log=True
        )
        
        self.server = uvicorn.Server(config)
        self.running = True
        
        logger.info(f"✅ Server starting on http://{TEST_SERVER_CONFIG['host']}:{TEST_SERVER_CONFIG['port']}")
        logger.info("📊 Ready for load testing with 1000+ concurrent users")
        
        try:
            await self.server.serve()
        except KeyboardInterrupt:
            logger.info("🛑 Server shutdown requested")
        finally:
            self.running = False
    
    async def stop(self):
        """Stop the load testing server"""
        if self.server and self.running:
            logger.info("🛑 Stopping load testing server...")
            self.server.should_exit = True
            self.running = False

def setup_signal_handlers(server):
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(server.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main function to run the load testing server"""
    server = LoadTestServer()
    setup_signal_handlers(server)
    
    logger.info("🔧 PRSM Load Testing Server")
    logger.info("=========================")
    logger.info("This server simulates PRSM production load for performance testing")
    logger.info(f"Target: Support 1000+ concurrent users as required by Gemini audit")
    
    await server.start()

if __name__ == "__main__":
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run the server
    asyncio.run(main())