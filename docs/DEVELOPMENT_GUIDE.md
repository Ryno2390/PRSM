# PRSM Development Guide

## 🎯 Comprehensive Development Setup

This guide provides step-by-step instructions for setting up a complete PRSM development environment with practical examples, common troubleshooting scenarios, and advanced configuration options.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Development Workflow](#development-workflow)
4. [Testing & Debugging](#testing--debugging)
5. [Common Development Tasks](#common-development-tasks)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Configuration](#advanced-configuration)

## 🖥️ System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space (SSD recommended)
- **OS**: macOS 10.15+, Ubuntu 20.04+, Windows 10+
- **Python**: 3.9+ (3.11+ recommended)

### Recommended Development Setup
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB for full AI model testing
- **Storage**: 100GB+ SSD for datasets and models
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for local model training)

## 🛠️ Environment Setup

### Option 1: Quick Setup (Recommended for Beginners)

```bash
# 1. Clone the repository
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM

# 2. Run the automated setup script
chmod +x scripts/setup_dev_environment.sh
./scripts/setup_dev_environment.sh

# 3. Activate the environment
source prsm-dev/bin/activate  # Linux/macOS
# or
prsm-dev\Scripts\activate     # Windows

# 4. Verify installation
python -c "import prsm; print('✅ PRSM installation successful!')"
```

### Option 2: Manual Setup (Advanced Users)

#### Step 1: Python Environment
```bash
# Create isolated environment
python3.11 -m venv prsm-dev
source prsm-dev/bin/activate

# Upgrade pip and essential tools
pip install --upgrade pip setuptools wheel
```

#### Step 2: Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# Optional: AI/ML dependencies for local model training
pip install -r requirements-ml.txt
```

#### Step 3: Database Setup
```bash
# Initialize PostgreSQL (production-like setup)
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
brew install postgresql                              # macOS

# Create development database
createdb prsm_dev

# Run migrations
alembic upgrade head
```

#### Step 4: Environment Configuration
```bash
# Copy example configuration
cp config/development.env.example .env

# Edit configuration
nano .env
```

Example `.env` configuration:
```bash
# Database
DATABASE_URL="postgresql://user:pass@localhost/prsm_dev"
REDIS_URL="redis://localhost:6379/0"

# API Keys (optional for basic development)
OPENAI_API_KEY="your_openai_key"
ANTHROPIC_API_KEY="your_anthropic_key"

# Development settings
PRSM_ENV="development"
DEBUG_LEVEL="INFO"
ENABLE_PERFORMANCE_MONITORING="true"

# Security (development only)
JWT_SECRET_KEY="dev_secret_key_change_in_production"
ENCRYPTION_KEY="dev_encryption_key_32_chars_long"
```

### Option 3: Docker Development Environment

```bash
# Quick start with Docker Compose
docker-compose -f docker/dev-compose.yml up -d

# Access development container
docker exec -it prsm_dev_container bash

# Or use VS Code Dev Containers
code --install-extension ms-vscode-remote.remote-containers
# Open in container: Ctrl+Shift+P > "Remote-Containers: Reopen in Container"
```

## 🔄 Development Workflow

### Daily Development Cycle

```bash
# 1. Start development services
make dev-services

# 2. Run in development mode with hot reload
python -m uvicorn prsm.api.main:app --reload --host 0.0.0.0 --port 8000

# 3. In another terminal, start the background workers
celery -A prsm.workers worker --loglevel=info

# 4. Monitor logs in real-time
tail -f logs/prsm-dev.log
```

### Testing Your Changes

```bash
# Quick smoke test
python -m pytest tests/test_smoke.py -v

# Test specific component
python -m pytest tests/test_agents/ -v

# Run with coverage report
python -m pytest --cov=prsm --cov-report=html tests/

# Performance benchmarks
python scripts/run_benchmarks.py
```

### Code Quality Checks

```bash
# Format code
black prsm/ tests/
isort prsm/ tests/

# Lint code
flake8 prsm/ tests/
mypy prsm/

# Security scan
bandit -r prsm/

# All quality checks at once
make quality-check
```

## 🧪 Testing & Debugging

### Running Specific Test Suites

```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# End-to-end tests
python -m pytest tests/e2e/ -v

# AI model tests (requires API keys)
python -m pytest tests/ai_models/ -v --api-keys

# Performance tests
python -m pytest tests/performance/ -v --benchmark-only
```

### Interactive Testing Examples

```python
# test_interactive.py - Run this for hands-on testing
import asyncio
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import PRSMSession

async def interactive_test():
    """Interactive test session for development"""
    orchestrator = NWTNOrchestrator()
    
    # Test 1: Basic query processing
    print("🧪 Test 1: Basic Query Processing")
    session = PRSMSession(user_id="dev_test")
    response = await orchestrator.process_query(
        "Explain quantum computing in simple terms",
        session
    )
    print(f"✅ Response length: {len(response.content)} characters")
    print(f"✅ Context used: {response.context_used} FTNS")
    
    # Test 2: Multi-agent coordination
    print("\n🧪 Test 2: Multi-Agent Coordination")
    complex_query = "Design a sustainable energy system for a city of 1 million people"
    response = await orchestrator.process_query(complex_query, session)
    print(f"✅ Reasoning steps: {len(response.reasoning_trace)}")
    
    # Test 3: Error handling
    print("\n🧪 Test 3: Error Handling")
    try:
        await orchestrator.process_query("", session)  # Empty query
    except ValueError as e:
        print(f"✅ Proper error handling: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_test())
```

### Debugging Tools and Techniques

#### 1. Structured Logging
```python
import structlog

logger = structlog.get_logger(__name__)

# Add debug logging to your code
logger.debug("Processing query", 
            query_length=len(query),
            session_id=session.session_id,
            user_id=session.user_id)
```

#### 2. Performance Profiling
```bash
# Profile your code
python -m cProfile -o profile.stats your_script.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

#### 3. Memory Usage Monitoring
```python
# memory_monitor.py
import tracemalloc
import psutil
import os

def start_memory_monitoring():
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def check_memory_usage(label=""):
    process = psutil.Process(os.getpid())
    current, peak = tracemalloc.get_traced_memory()
    print(f"{label} - Current: {current / 1024 / 1024:.2f} MB, "
          f"Peak: {peak / 1024 / 1024:.2f} MB, "
          f"Process: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## 🔧 Common Development Tasks

### Adding a New API Endpoint

```python
# 1. Define the endpoint in the appropriate router file
# prsm/api/my_new_api.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/my-feature", tags=["My Feature"])

class MyRequest(BaseModel):
    param1: str
    param2: int = 10

class MyResponse(BaseModel):
    result: str
    processing_time: float

@router.post("/process", response_model=MyResponse)
async def process_my_feature(
    request: MyRequest,
    current_user: str = Depends(get_current_user)
):
    """Process my feature with proper error handling and logging."""
    start_time = time.time()
    
    try:
        logger.info("Processing feature request",
                   user_id=current_user,
                   param1=request.param1,
                   param2=request.param2)
        
        # Your processing logic here
        result = f"Processed {request.param1} with value {request.param2}"
        
        processing_time = time.time() - start_time
        
        logger.info("Feature processing completed",
                   user_id=current_user,
                   processing_time=processing_time)
        
        return MyResponse(
            result=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error("Feature processing failed",
                    user_id=current_user,
                    error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# 2. Register the router in main.py
# app.include_router(my_new_api.router)

# 3. Add tests
# tests/test_my_new_api.py
import pytest
from fastapi.testclient import TestClient

def test_my_new_endpoint():
    response = client.post("/api/v1/my-feature/process", json={
        "param1": "test",
        "param2": 20
    })
    assert response.status_code == 200
    assert "Processed test" in response.json()["result"]
```

### Adding a New Agent Component

```python
# 1. Create the agent class
# prsm/agents/my_new_agent.py

from typing import Dict, Any, List
import structlog
from .base import BaseAgent

logger = structlog.get_logger(__name__)

class MyNewAgent(BaseAgent):
    """Agent for handling specific domain tasks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.agent_type = "my_new_agent"
        self.capabilities = ["capability1", "capability2"]
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task specific to this agent's domain."""
        logger.info("Processing task", agent_type=self.agent_type, task_id=task.get("id"))
        
        try:
            # Your processing logic
            result = await self._specific_processing(task)
            
            logger.info("Task completed successfully", 
                       agent_type=self.agent_type,
                       task_id=task.get("id"))
            
            return {
                "status": "completed",
                "result": result,
                "agent_type": self.agent_type
            }
            
        except Exception as e:
            logger.error("Task processing failed",
                        agent_type=self.agent_type,
                        error=str(e))
            raise
    
    async def _specific_processing(self, task: Dict[str, Any]) -> Any:
        """Implement your specific processing logic here."""
        # Placeholder for your implementation
        return f"Processed task: {task.get('description', 'No description')}"

# 2. Register the agent
# prsm/agents/__init__.py
from .my_new_agent import MyNewAgent

AVAILABLE_AGENTS = {
    "my_new_agent": MyNewAgent,
    # ... other agents
}

# 3. Add configuration
# config/agents.yaml
my_new_agent:
  enabled: true
  max_concurrent_tasks: 5
  timeout_seconds: 300
  capabilities:
    - capability1
    - capability2

# 4. Add tests
# tests/test_my_new_agent.py
import pytest
from prsm.agents.my_new_agent import MyNewAgent

@pytest.mark.asyncio
async def test_my_new_agent_processing():
    agent = MyNewAgent()
    task = {"id": "test_task", "description": "Test task"}
    
    result = await agent.process_task(task)
    
    assert result["status"] == "completed"
    assert "Processed task" in result["result"]
```

### Database Migrations

```bash
# 1. Generate a new migration
alembic revision --autogenerate -m "Add new feature table"

# 2. Review the generated migration file
# alembic/versions/xxxx_add_new_feature_table.py

# 3. Apply the migration
alembic upgrade head

# 4. Rollback if needed
alembic downgrade -1

# 5. Check migration status
alembic current
alembic history --verbose
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'prsm'
# Solution: Ensure PYTHONPATH is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### Issue 2: Database Connection Issues
```bash
# Error: FATAL: database "prsm_dev" does not exist
# Solution: Create the database
createdb prsm_dev

# Or reset the database completely
dropdb prsm_dev && createdb prsm_dev
alembic upgrade head
```

#### Issue 3: Port Already in Use
```bash
# Error: [Errno 48] Address already in use
# Solution: Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
python -m uvicorn prsm.api.main:app --port 8001
```

#### Issue 4: Memory Issues with Large Models
```python
# Add memory monitoring to your code
import gc
import torch

def clear_memory():
    """Clear memory between model operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use smaller batch sizes
config = {
    "batch_size": 1,  # Reduce from default
    "max_length": 512,  # Reduce context length
}
```

#### Issue 5: API Rate Limiting
```python
# Implement backoff for external APIs
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
            return None
        return wrapper
    return decorator
```

### Performance Debugging

#### Slow Query Identification
```python
# Add to your database queries
import time
import structlog

logger = structlog.get_logger(__name__)

async def logged_query(query, params=None):
    start_time = time.time()
    try:
        result = await database.fetch_all(query, params)
        duration = time.time() - start_time
        
        if duration > 1.0:  # Log slow queries
            logger.warning("Slow query detected",
                         query=query[:100],
                         duration=duration,
                         row_count=len(result))
        
        return result
    except Exception as e:
        logger.error("Query failed", query=query[:100], error=str(e))
        raise
```

#### Memory Leak Detection
```python
# memory_leak_detector.py
import psutil
import time
import threading

class MemoryLeakDetector:
    def __init__(self, threshold_mb=100):
        self.threshold_mb = threshold_mb
        self.initial_memory = None
        self.monitoring = False
    
    def start_monitoring(self):
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        while self.monitoring:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            increase = current_memory - self.initial_memory
            
            if increase > self.threshold_mb:
                logger.warning("Potential memory leak detected",
                             initial_mb=self.initial_memory,
                             current_mb=current_memory,
                             increase_mb=increase)
            
            time.sleep(30)  # Check every 30 seconds

# Usage
detector = MemoryLeakDetector()
detector.start_monitoring()
```

## ⚡ Performance Optimization

### Database Optimization

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "poolclass": QueuePool
}

# Optimize queries with indexes
# migrations/add_performance_indexes.py
def upgrade():
    op.create_index('idx_sessions_user_id', 'sessions', ['user_id'])
    op.create_index('idx_tasks_status_created', 'tasks', ['status', 'created_at'])
    op.create_index('idx_models_active_type', 'models', ['is_active', 'model_type'])
```

### Caching Strategy

```python
# Redis caching
import redis
import json
import pickle
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiry_seconds=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, expiry_seconds, pickle.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expiry_seconds=600)
async def expensive_computation(param1, param2):
    # Your expensive computation here
    await asyncio.sleep(2)  # Simulate expensive operation
    return f"Result for {param1}, {param2}"
```

### Async Optimization

```python
# Batch processing
async def process_items_batch(items, batch_size=10):
    """Process items in batches to avoid overwhelming the system."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            process_single_item(item) for item in batch
        ])
        results.extend(batch_results)
        
        # Add small delay between batches
        await asyncio.sleep(0.1)
    
    return results

# Connection limiting
semaphore = asyncio.Semaphore(5)  # Limit concurrent connections

async def limited_api_call(data):
    async with semaphore:
        return await make_api_call(data)
```

## 🔧 Advanced Configuration

### Environment-Specific Settings

```python
# config/settings.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Database
    database_url: str
    database_pool_size: int = 10
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Security
    jwt_secret_key: str
    jwt_expiry_hours: int = 24
    
    # Performance
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load settings
settings = Settings()
```

### Custom Development Scripts

```python
# scripts/dev_utils.py
import asyncio
import click
from prsm.core.database import init_database
from prsm.core.models import PRSMSession

@click.group()
def cli():
    """PRSM Development Utilities"""
    pass

@cli.command()
@click.option('--count', default=10, help='Number of test sessions to create')
async def create_test_data(count):
    """Create test data for development."""
    await init_database()
    
    for i in range(count):
        session = PRSMSession(
            user_id=f"test_user_{i}",
            session_id=f"test_session_{i}",
            status="active"
        )
        # Save to database
        
    click.echo(f"Created {count} test sessions")

@cli.command()
async def reset_dev_database():
    """Reset development database."""
    click.confirm("This will delete all data. Continue?", abort=True)
    
    # Reset database logic here
    click.echo("Database reset complete")

if __name__ == '__main__':
    cli()
```

### VS Code Configuration

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./prsm-dev/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "-v"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}

// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "PRSM API Server",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "prsm.api.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000"
            ],
            "env": {
                "PRSM_ENV": "development"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

## 📈 Monitoring and Metrics

### Development Metrics Dashboard

```python
# Create a simple metrics endpoint for development
from fastapi import APIRouter
import psutil
import time

dev_router = APIRouter(prefix="/dev", tags=["Development"])

@dev_router.get("/metrics")
async def get_dev_metrics():
    """Get development metrics."""
    process = psutil.Process()
    
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "process": {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files())
        },
        "timestamp": time.time()
    }
```

This comprehensive development guide provides practical examples and real-world scenarios that developers will encounter when working with PRSM. It builds on the existing documentation while adding substantial value through detailed examples, troubleshooting scenarios, and advanced configuration options.