#!/usr/bin/env python3
"""
PRSM Python SDK - FastAPI Integration Example

This example demonstrates how to integrate PRSM SDK with FastAPI for production
web applications, including proper error handling, authentication, and monitoring.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from prsm_sdk import PRSMClient, PRSMError, BudgetExceededError, RateLimitError


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global PRSM client instance
prsm_client: Optional[PRSMClient] = None


# Request/Response Models
class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="The query prompt")
    model: Optional[str] = Field(None, description="Specific model to use")
    max_tokens: Optional[int] = Field(500, ge=1, le=4000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: Optional[bool] = Field(False, description="Enable streaming response")


class QueryResponse(BaseModel):
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    cost: float = Field(..., description="Request cost in dollars")
    request_id: str = Field(..., description="Unique request identifier")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    prsm_connected: bool = Field(..., description="PRSM API connection status")
    version: str = Field(..., description="Service version")


class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_cost: Optional[float] = Field(None, gt=0, description="Maximum cost constraint")
    min_quality: Optional[float] = Field(None, ge=0, le=1, description="Minimum quality threshold")
    max_latency: Optional[float] = Field(None, gt=0, description="Maximum latency in seconds")


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    global prsm_client
    
    # Startup
    logger.info("Starting PRSM FastAPI integration service")
    
    api_key = os.getenv("PRSM_API_KEY")
    if not api_key:
        logger.error("PRSM_API_KEY environment variable not set")
        raise RuntimeError("PRSM_API_KEY is required")
    
    try:
        prsm_client = PRSMClient(
            api_key=api_key,
            base_url=os.getenv("PRSM_BASE_URL"),
            timeout=60.0,
            max_retries=3
        )
        
        # Test connection
        await prsm_client.health_check()
        logger.info("Connected to PRSM API successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize PRSM client: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down PRSM FastAPI integration service")
    if prsm_client:
        await prsm_client.close()


# Create FastAPI app
app = FastAPI(
    title="PRSM AI Service",
    description="Production-ready AI service powered by PRSM SDK",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get PRSM client
async def get_prsm_client() -> PRSMClient:
    """Get the initialized PRSM client"""
    if not prsm_client:
        raise HTTPException(status_code=503, detail="PRSM client not initialized")
    return prsm_client


# Error handlers
async def handle_prsm_error(error: PRSMError) -> HTTPException:
    """Convert PRSM errors to HTTP exceptions"""
    if isinstance(error, BudgetExceededError):
        return HTTPException(
            status_code=402,
            detail={
                "error": "Budget exceeded",
                "message": error.message,
                "remaining_budget": error.remaining_budget
            }
        )
    elif isinstance(error, RateLimitError):
        return HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": error.message,
                "retry_after": error.retry_after_seconds
            }
        )
    else:
        return HTTPException(
            status_code=500,
            detail={
                "error": "PRSM API error",
                "message": error.message
            }
        )


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check(client: PRSMClient = Depends(get_prsm_client)):
    """Health check endpoint"""
    try:
        await client.health_check()
        prsm_connected = True
    except Exception:
        prsm_connected = False
    
    return HealthResponse(
        status="healthy" if prsm_connected else "degraded",
        prsm_connected=prsm_connected,
        version="1.0.0"
    )


@app.post("/query", response_model=QueryResponse)
async def query_model(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    client: PRSMClient = Depends(get_prsm_client)
):
    """Query an AI model with the provided prompt"""
    try:
        # Log the request for monitoring
        request_id = http_request.headers.get("X-Request-ID", "unknown")
        logger.info(f"Processing query request {request_id}", extra={
            "request_id": request_id,
            "model": request.model,
            "prompt_length": len(request.prompt),
            "stream": request.stream
        })
        
        # Execute the query
        result = await client.models.infer(
            model=request.model or "gpt-4",
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Log usage for analytics (background task)
        background_tasks.add_task(
            log_usage,
            request_id=request_id,
            model=result.model,
            tokens_used=result.usage.total_tokens,
            cost=result.cost
        )
        
        return QueryResponse(
            content=result.content,
            model=result.model,
            usage=result.usage.__dict__,
            cost=result.cost,
            request_id=request_id
        )
        
    except PRSMError as e:
        logger.error(f"PRSM error: {e.message}", extra={"request_id": request_id})
        raise await handle_prsm_error(e)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query/stream")
async def stream_query(
    request: QueryRequest,
    http_request: Request,
    client: PRSMClient = Depends(get_prsm_client)
):
    """Stream AI model responses for real-time applications"""
    if not request.stream:
        request.stream = True
    
    request_id = http_request.headers.get("X-Request-ID", "unknown")
    
    async def generate_stream():
        try:
            logger.info(f"Starting stream for request {request_id}")
            
            async for chunk in client.models.stream(
                model=request.model or "gpt-4",
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                # Format as Server-Sent Events
                yield f"data: {chunk.content}\n\n"
                
            yield "data: [DONE]\n\n"
            
        except PRSMError as e:
            logger.error(f"Stream error: {e.message}", extra={"request_id": request_id})
            yield f"data: {{\"error\": \"{e.message}\"}}\n\n"
        except Exception as e:
            logger.error(f"Unexpected stream error: {e}", extra={"request_id": request_id})
            yield f"data: {{\"error\": \"Internal server error\"}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/optimize")
async def optimize_request(
    request: OptimizeRequest,
    http_request: Request,
    client: PRSMClient = Depends(get_prsm_client)
):
    """Optimize model selection based on cost and quality constraints"""
    try:
        request_id = http_request.headers.get("X-Request-ID", "unknown")
        
        constraints = {}
        if request.max_cost:
            constraints["max_cost"] = request.max_cost
        if request.min_quality:
            constraints["min_quality"] = request.min_quality
        if request.max_latency:
            constraints["max_latency"] = request.max_latency
        
        optimization = await client.cost_optimization.optimize_request(
            prompt=request.prompt,
            constraints=constraints
        )
        
        logger.info(f"Optimization completed for request {request_id}", extra={
            "selected_model": optimization.selected_model,
            "estimated_cost": optimization.estimated_cost,
            "cost_savings": optimization.cost_savings
        })
        
        return {
            "selected_model": optimization.selected_model,
            "estimated_cost": optimization.estimated_cost,
            "quality_score": optimization.quality_score,
            "cost_savings": optimization.cost_savings,
            "reasoning": optimization.reasoning,
            "request_id": request_id
        }
        
    except PRSMError as e:
        logger.error(f"Optimization error: {e.message}")
        raise await handle_prsm_error(e)
    except Exception as e:
        logger.error(f"Unexpected optimization error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def list_available_models(client: PRSMClient = Depends(get_prsm_client)):
    """List all available models in the marketplace"""
    try:
        models = await client.marketplace.list_models()
        
        return {
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "description": model.description,
                    "provider": model.provider,
                    "cost_per_1k_tokens": model.pricing.cost_per_1k_tokens,
                    "capabilities": model.capabilities,
                    "performance_score": model.performance_score
                }
                for model in models
            ],
            "total_count": len(models)
        }
        
    except PRSMError as e:
        logger.error(f"Error listing models: {e.message}")
        raise await handle_prsm_error(e)


@app.get("/budget")
async def get_budget_status(client: PRSMClient = Depends(get_prsm_client)):
    """Get current budget status and usage statistics"""
    try:
        budget = await client.cost_optimization.get_budget()
        
        return {
            "total_budget": budget.total_budget,
            "spent": budget.spent,
            "remaining": budget.remaining,
            "utilization_percentage": (budget.spent / budget.total_budget) * 100,
            "projected_exhaustion": budget.projected_exhaustion,
            "daily_average": budget.daily_average if hasattr(budget, 'daily_average') else None
        }
        
    except PRSMError as e:
        logger.error(f"Error getting budget: {e.message}")
        raise await handle_prsm_error(e)


# Background task functions
async def log_usage(request_id: str, model: str, tokens_used: int, cost: float):
    """Log usage metrics for analytics (background task)"""
    logger.info("Usage logged", extra={
        "request_id": request_id,
        "model": model,
        "tokens_used": tokens_used,
        "cost": cost,
        "timestamp": asyncio.get_event_loop().time()
    })
    
    # In production, you might send this to a metrics service like:
    # - Prometheus
    # - DataDog
    # - CloudWatch
    # - Custom analytics database


# Production deployment helpers
def create_production_app():
    """Create app configured for production deployment"""
    # Configure for production
    app.debug = False
    
    # Add production middleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost").split(",")
    )
    
    return app


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "fastapi_integration:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )