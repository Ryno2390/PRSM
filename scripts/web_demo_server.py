#!/usr/bin/env python3
"""
PRSM Web Demo Server

FastAPI backend that integrates our PostgreSQL + pgvector implementation 
with the existing UI/UX mockup for investor demonstrations.

Features:
- REST API endpoints for PRSM query processing
- WebSocket support for real-time communication
- Live metrics and performance dashboards
- FTNS token economics tracking
- Static file serving for the existing UI

Run with: uvicorn web_demo_server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from prsm.vector_store.base import VectorStoreConfig, VectorStoreType, ContentType, SearchFilters
    from prsm.vector_store.implementations.pgvector_store import create_development_pgvector_store
    from integration_demo_pgvector import RealEmbeddingService, FTNSTokenService, PRSMProductionDemo
except ImportError as e:
    logger.error(f"Failed to import PRSM components: {e}")
    logger.error("Make sure you're running from the PRSM root directory")
    raise


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    user_id: str = "web_user"
    show_reasoning: bool = True
    max_results: int = 5

class QueryResponse(BaseModel):
    success: bool
    response: str
    processing_time: float
    query_cost: float
    results_found: int
    intent_category: str
    complexity_score: float
    embedding_provider: str
    creators_compensated: int
    error: Optional[str] = None

class SystemStatus(BaseModel):
    database_status: Dict[str, Any]
    embedding_service: Dict[str, Any]
    ftns_economics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    system_health: Dict[str, str]

class UserBalance(BaseModel):
    user_id: str
    balance: float
    recent_transactions: List[Dict[str, Any]]


# Global application state
class AppState:
    def __init__(self):
        self.demo: Optional[PRSMProductionDemo] = None
        self.connected_clients: List[WebSocket] = []
        self.is_initialized = False
        self.initialization_error: Optional[str] = None

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize PRSM demo on startup"""
    logger.info("🚀 Initializing PRSM Web Demo Server...")
    
    try:
        # Initialize the production demo
        app_state.demo = PRSMProductionDemo(embedding_provider="auto")
        await app_state.demo.initialize()
        app_state.is_initialized = True
        logger.info("✅ PRSM demo initialized successfully")
        
        # Broadcast initialization success to any connected clients
        await broadcast_message({
            "type": "system_status",
            "status": "ready",
            "message": "PRSM system initialized and ready for queries"
        })
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize PRSM demo: {e}")
        app_state.initialization_error = str(e)
        
        # Broadcast error to clients
        await broadcast_message({
            "type": "system_error", 
            "error": str(e),
            "message": "Failed to initialize PRSM system"
        })
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down PRSM Web Demo Server...")
    if app_state.demo and app_state.demo.vector_store:
        await app_state.demo.vector_store.disconnect()


# Create FastAPI app with lifespan management
app = FastAPI(
    title="PRSM Web Demo API",
    description="REST API for PRSM investor demonstrations with PostgreSQL + pgvector",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients"""
    if not app_state.connected_clients:
        return
    
    disconnected_clients = []
    for client in app_state.connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send message to client: {e}")
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        app_state.connected_clients.remove(client)


def check_initialization():
    """Check if the demo is properly initialized"""
    if not app_state.is_initialized:
        if app_state.initialization_error:
            raise HTTPException(
                status_code=503, 
                detail=f"PRSM system not initialized: {app_state.initialization_error}"
            )
        else:
            raise HTTPException(
                status_code=503, 
                detail="PRSM system is still initializing, please wait"
            )


# API Routes

@app.get("/")
async def serve_ui():
    """Serve the main UI from the existing mockup"""
    return FileResponse("/Users/ryneschultz/Documents/GitHub/PRSM/PRSM_ui_mockup/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if app_state.is_initialized and app_state.demo:
        try:
            # Test database connection
            health = await app_state.demo.vector_store.health_check()
            return {
                "status": "healthy",
                "database": health,
                "embedding_service": app_state.demo.embedding_service.get_usage_stats(),
                "initialized": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": app_state.is_initialized
            }
    else:
        return {
            "status": "initializing" if not app_state.initialization_error else "error",
            "error": app_state.initialization_error,
            "initialized": False
        }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a PRSM query with full pipeline"""
    check_initialization()
    
    logger.info(f"Processing query from {request.user_id}: {request.query[:50]}...")
    
    try:
        # Process query through PRSM pipeline
        result = await app_state.demo.process_query(
            request.user_id, 
            request.query, 
            show_reasoning=request.show_reasoning
        )
        
        if result.get("success"):
            response = QueryResponse(
                success=True,
                response=result["response"],
                processing_time=result["processing_time"],
                query_cost=result["query_cost"],
                results_found=result["results_found"],
                intent_category=result["intent_category"],
                complexity_score=result["complexity_score"],
                embedding_provider=result["embedding_provider"],
                creators_compensated=result.get("creators_compensated", 0)
            )
            
            # Broadcast query completion to WebSocket clients
            background_tasks.add_task(broadcast_message, {
                "type": "query_completed",
                "user_id": request.user_id,
                "query": request.query[:100] + "..." if len(request.query) > 100 else request.query,
                "results_found": result["results_found"],
                "processing_time": result["processing_time"],
                "query_cost": result["query_cost"]
            })
            
            return response
        else:
            error_msg = result.get("error", "Unknown error occurred")
            background_tasks.add_task(broadcast_message, {
                "type": "query_error",
                "user_id": request.user_id,
                "error": error_msg
            })
            
            return QueryResponse(
                success=False,
                response="",
                processing_time=result.get("processing_time", 0),
                query_cost=0,
                results_found=0,
                intent_category="error",
                complexity_score=0,
                embedding_provider="unknown",
                creators_compensated=0,
                error=error_msg
            )
            
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        error_msg = f"Internal server error: {str(e)}"
        
        return QueryResponse(
            success=False,
            response="",
            processing_time=0,
            query_cost=0,
            results_found=0,
            intent_category="error",
            complexity_score=0,
            embedding_provider="unknown",
            creators_compensated=0,
            error=error_msg
        )

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status for dashboard"""
    check_initialization()
    
    try:
        # Get database status
        db_health = await app_state.demo.vector_store.health_check()
        db_stats = await app_state.demo.vector_store.get_collection_stats()
        
        # Get performance metrics
        if hasattr(app_state.demo.vector_store, 'performance_metrics'):
            db_performance = app_state.demo.vector_store.performance_metrics
        else:
            db_performance = {"note": "Performance metrics not available"}
        
        # Get embedding service stats
        embedding_stats = app_state.demo.embedding_service.get_usage_stats()
        
        # Get FTNS economics
        economics = app_state.demo.ftns_service.get_economics_summary()
        
        # Get demo performance
        demo_performance = app_state.demo.demo_stats
        
        return SystemStatus(
            database_status={
                "health": db_health,
                "stats": db_stats,
                "performance": db_performance
            },
            embedding_service=embedding_stats,
            ftns_economics=economics,
            performance_metrics=demo_performance,
            system_health={
                "database": "operational",
                "embedding_service": "operational",
                "ftns_economy": "active",
                "overall": "ready"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system status: {str(e)}")

@app.get("/api/user/{user_id}/balance", response_model=UserBalance)
async def get_user_balance(user_id: str):
    """Get user FTNS balance and recent transactions"""
    check_initialization()
    
    try:
        balance = await app_state.demo.ftns_service.get_user_balance(user_id)
        
        # Get recent transactions for this user
        all_transactions = app_state.demo.ftns_service.transactions
        user_transactions = [
            t for t in all_transactions[-10:]  # Last 10 transactions
            if t.get("user_id") == user_id
        ]
        
        # Convert datetime objects to strings for JSON serialization
        for transaction in user_transactions:
            if "timestamp" in transaction:
                transaction["timestamp"] = transaction["timestamp"].isoformat()
        
        return UserBalance(
            user_id=user_id,
            balance=balance,
            recent_transactions=user_transactions
        )
        
    except Exception as e:
        logger.error(f"Failed to get user balance for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user balance: {str(e)}")

@app.get("/api/metrics/live")
async def get_live_metrics():
    """Get live metrics for real-time dashboard updates"""
    check_initialization()
    
    try:
        # Get current system metrics
        economics = app_state.demo.ftns_service.get_economics_summary()
        embedding_stats = app_state.demo.embedding_service.get_usage_stats()
        demo_stats = app_state.demo.demo_stats
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "queries_processed": demo_stats.get("queries_processed", 0),
            "success_rate": (
                demo_stats.get("successful_queries", 0) / 
                max(1, demo_stats.get("queries_processed", 1)) * 100
            ),
            "average_response_time": demo_stats.get("average_response_time", 0),
            "total_ftns_volume": economics.get("query_volume", 0),
            "total_creators_compensated": len(economics.get("creator_earnings", {})),
            "embedding_api_calls": embedding_stats.get("api_calls", 0),
            "embedding_provider": embedding_stats.get("provider", "unknown"),
            "database_operations": demo_stats.get("database_operations", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get live metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve live metrics: {str(e)}")


# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    app_state.connected_clients.append(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {len(app_state.connected_clients)}")
    
    try:
        # Send initial system status
        if app_state.is_initialized:
            await websocket.send_text(json.dumps({
                "type": "system_status",
                "status": "ready",
                "message": "Connected to PRSM system"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "system_status",
                "status": "initializing",
                "message": "PRSM system is initializing..."
            }))
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "request_status":
                if app_state.is_initialized:
                    live_metrics = await get_live_metrics()
                    await websocket.send_text(json.dumps({
                        "type": "live_metrics",
                        "data": live_metrics
                    }))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in app_state.connected_clients:
            app_state.connected_clients.remove(websocket)
        logger.info(f"WebSocket client removed. Total clients: {len(app_state.connected_clients)}")


# Mount static files for the UI
app.mount("/static", StaticFiles(directory="/Users/ryneschultz/Documents/GitHub/PRSM/PRSM_ui_mockup"), name="static")


# Additional utility endpoints
@app.get("/api/demo/scenarios")
async def get_demo_scenarios():
    """Get predefined demo scenarios for investor presentations"""
    return {
        "scenarios": [
            {
                "id": "academic_research",
                "title": "Academic Research Scenario",
                "description": "Demonstrate PRSM for academic research and paper discovery",
                "sample_queries": [
                    "How can AI governance be democratized through blockchain technology?",
                    "What are the legal requirements for content provenance in AI training?",
                    "Show me research on climate change datasets for machine learning"
                ],
                "user_persona": "researcher_alice",
                "expected_results": "3-5 high-quality research papers with creator compensation"
            },
            {
                "id": "enterprise_knowledge",
                "title": "Enterprise Knowledge Management",
                "description": "Showcase PRSM for enterprise knowledge discovery and compliance",
                "sample_queries": [
                    "What are best practices for vector database implementation?",
                    "How do token economics incentivize quality contributions?",
                    "Find technical documentation for similarity search algorithms"
                ],
                "user_persona": "enterprise_org",
                "expected_results": "Technical content with provenance tracking and licensing"
            },
            {
                "id": "investor_technical",
                "title": "Technical Deep Dive",
                "description": "Technical demonstration for investor due diligence",
                "sample_queries": [
                    "What are the technical advantages of PRSM's vector database?",
                    "How does PRSM ensure legal compliance in AI training?",
                    "Demonstrate real-time performance monitoring capabilities"
                ],
                "user_persona": "demo_investor",
                "expected_results": "Performance metrics, scalability indicators, compliance features"
            }
        ]
    }

@app.post("/api/demo/run_scenario/{scenario_id}")
async def run_demo_scenario(scenario_id: str, background_tasks: BackgroundTasks):
    """Run a predefined demo scenario"""
    check_initialization()
    
    scenarios_response = await get_demo_scenarios()
    scenarios = {s["id"]: s for s in scenarios_response["scenarios"]}
    
    if scenario_id not in scenarios:
        raise HTTPException(status_code=404, detail="Demo scenario not found")
    
    scenario = scenarios[scenario_id]
    results = []
    
    try:
        for query in scenario["sample_queries"]:
            result = await app_state.demo.process_query(
                scenario["user_persona"], 
                query, 
                show_reasoning=False
            )
            results.append({
                "query": query,
                "success": result.get("success", False),
                "results_found": result.get("results_found", 0),
                "processing_time": result.get("processing_time", 0),
                "query_cost": result.get("query_cost", 0)
            })
        
        # Broadcast scenario completion
        background_tasks.add_task(broadcast_message, {
            "type": "scenario_completed",
            "scenario_id": scenario_id,
            "title": scenario["title"],
            "queries_completed": len(results),
            "total_results": sum(r["results_found"] for r in results)
        })
        
        return {
            "scenario": scenario,
            "results": results,
            "summary": {
                "queries_completed": len(results),
                "total_results_found": sum(r["results_found"] for r in results),
                "total_processing_time": sum(r["processing_time"] for r in results),
                "total_cost": sum(r["query_cost"] for r in results)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to run demo scenario {scenario_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run demo scenario: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting PRSM Web Demo Server")
    print("📍 Navigate to: http://localhost:8000")
    print("🔧 API Documentation: http://localhost:8000/docs")
    print("📊 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "web_demo_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )