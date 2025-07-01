"""
API Response Optimizer for PRSM
===============================

Optimizes API response times through compression, caching,
connection pooling, and efficient request handling.
"""

import gzip
import json
from typing import Any, Dict
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
import time
import structlog

logger = structlog.get_logger(__name__)

class APIOptimizer:
    """API performance optimization utilities"""
    
    @staticmethod
    def optimize_fastapi_app(app: FastAPI) -> FastAPI:
        """Apply performance optimizations to FastAPI app"""
        
        # Add compression middleware
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Add CORS with optimized settings
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            max_age=86400,  # Cache preflight requests for 24 hours
        )
        
        # Add performance monitoring middleware
        @app.middleware("http")
        async def performance_monitoring(request: Request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log slow requests
            if process_time > 1.0:
                logger.warning(
                    f"Slow request detected",
                    path=request.url.path,
                    method=request.method,
                    process_time=process_time
                )
            
            return response
        
        # Add response optimization middleware
        @app.middleware("http")
        async def response_optimization(request: Request, call_next):
            response = await call_next(request)
            
            # Add performance headers
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-cache"
            
            return response
        
        logger.info("✅ FastAPI app optimized for performance")
        return app
    
    @staticmethod
    def create_optimized_response(data: Any, status_code: int = 200) -> Response:
        """Create optimized JSON response"""
        json_data = json.dumps(data, separators=(',', ':'), default=str)
        
        # Compress large responses
        if len(json_data) > 1024:
            compressed = gzip.compress(json_data.encode('utf-8'))
            return Response(
                content=compressed,
                status_code=status_code,
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(compressed))
                }
            )
        
        return Response(
            content=json_data,
            status_code=status_code,
            headers={"Content-Type": "application/json"}
        )
    
    @staticmethod
    async def batch_process_requests(requests: list, batch_size: int = 10) -> list:
        """Process requests in optimized batches"""
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        
        return results

# Response optimization utilities
def optimize_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize JSON response size"""
    # Remove null values
    return {k: v for k, v in data.items() if v is not None}

def add_cache_headers(response: Response, max_age: int = 300):
    """Add optimized cache headers"""
    response.headers["Cache-Control"] = f"public, max-age={max_age}"
    response.headers["ETag"] = f'"{hash(str(response.body))}"'
    return response

# Connection optimization
class ConnectionOptimizer:
    """Optimize external connections"""
    
    @staticmethod
    def get_optimized_client_settings() -> Dict[str, Any]:
        """Get optimized HTTP client settings"""
        return {
            "timeout": 10.0,
            "limits": {
                "max_keepalive_connections": 20,
                "max_connections": 100,
                "keepalive_expiry": 30.0
            },
            "headers": {
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=30, max=100"
            }
        }