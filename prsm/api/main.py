"""
PRSM FastAPI Application
Main API entry point for PRSM
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List, Set
from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import asyncio

from prsm.core.config import get_settings
from prsm.core.models import UserInput, PRSMResponse
from prsm.core.database import (
    init_database, close_database, db_manager,
    SessionQueries, FTNSQueries, ModelQueries
)
from prsm.core.redis_client import (
    init_redis, close_redis, redis_manager,
    get_session_cache, get_model_cache, get_task_queue, get_pubsub
)
from prsm.core.vector_db import (
    init_vector_databases, close_vector_databases, get_vector_db_manager,
    embedding_generator
)
from prsm.core.ipfs_client import (
    init_ipfs, close_ipfs, get_ipfs_client, prsm_ipfs
)
from prsm.api.teams_api import router as teams_router
from prsm.api.auth_api import router as auth_router
from prsm.api.credential_api import router as credential_router
from prsm.api.security_status_api import router as security_router
from prsm.api.security_logging_api import router as security_logging_router
from prsm.api.payment_api import router as payment_router
from prsm.api.cryptography_api import router as crypto_router
# MARKETPLACE API: Fully consolidated universal endpoints (real_marketplace_api.py)
from prsm.api.real_marketplace_api import router as marketplace_router
from prsm.api.governance_api import router as governance_router
from prsm.api.mainnet_deployment_api import router as mainnet_router
from prsm.api.health_api import router as health_router
from prsm.api.budget_api import router as budget_router
from prsm.web3.frontend_integration import router as web3_router
from prsm.chronos.api import router as chronos_router
from prsm.api.ui_api import router as ui_router
from prsm.api.ipfs_api import router as ipfs_router
from prsm.api.session_api import router as session_router
from prsm.api.task_api import router as task_router
from prsm.api.recommendation_api import router as recommendation_router
from prsm.api.reputation_api import router as reputation_router
from prsm.api.distillation_api import router as distillation_router
from prsm.api.monitoring_api import router as monitoring_router
from prsm.api.compliance_api import router as compliance_router
from prsm.auth.auth_manager import auth_manager
from prsm.auth import get_current_user
from prsm.auth.middleware import AuthMiddleware, SecurityHeadersMiddleware
from prsm.security.middleware import (
    get_security_middleware_stack, configure_cors,
    SecurityHeadersMiddleware as EnhancedSecurityHeaders,
    RateLimitingMiddleware, RequestValidationMiddleware
)


# Configure structured logging
logger = structlog.get_logger(__name__)
settings = get_settings()


# === WebSocket Connection Manager ===

class WebSocketManager:
    """
    Manages WebSocket connections for real-time communication
    
    🔌 REAL-TIME FEATURES:
    - Connection lifecycle management
    - Message broadcasting to specific users or all clients
    - Conversation-specific subscriptions
    - Automatic cleanup and reconnection handling
    """
    
    def __init__(self):
        # Active connections: user_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Conversation subscriptions: conversation_id -> Set[WebSocket]
        self.conversation_subscriptions: Dict[str, Set[WebSocket]] = {}
        # Connection metadata: WebSocket -> Dict[str, Any]
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str, connection_type: str = "general"):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        # Add to user connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        
        # Store connection metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connection_type": connection_type,
            "connected_at": asyncio.get_event_loop().time(),
            "last_activity": asyncio.get_event_loop().time()
        }
        
        logger.info("WebSocket connected",
                   user_id=user_id,
                   connection_type=connection_type,
                   total_connections=len(self.connection_metadata))
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "user_id": user_id,
            "timestamp": asyncio.get_event_loop().time(),
            "message": "Real-time connection established"
        }, websocket)
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket not in self.connection_metadata:
            return
            
        metadata = self.connection_metadata[websocket]
        user_id = metadata["user_id"]
        
        # Remove from user connections
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove from conversation subscriptions
        for conversation_id, subscribers in self.conversation_subscriptions.items():
            subscribers.discard(websocket)
        
        # Clean up empty conversation subscriptions
        self.conversation_subscriptions = {
            conv_id: subs for conv_id, subs in self.conversation_subscriptions.items()
            if subs
        }
        
        # Remove metadata
        del self.connection_metadata[websocket]
        
        logger.info("WebSocket disconnected",
                   user_id=user_id,
                   total_connections=len(self.connection_metadata))
    
    async def subscribe_to_conversation(self, websocket: WebSocket, conversation_id: str):
        """Subscribe WebSocket to conversation updates"""
        if conversation_id not in self.conversation_subscriptions:
            self.conversation_subscriptions[conversation_id] = set()
        
        self.conversation_subscriptions[conversation_id].add(websocket)
        
        # Update metadata
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            if "subscriptions" not in metadata:
                metadata["subscriptions"] = set()
            metadata["subscriptions"].add(conversation_id)
        
        logger.debug("WebSocket subscribed to conversation",
                    conversation_id=conversation_id,
                    subscribers=len(self.conversation_subscriptions[conversation_id]))
    
    async def unsubscribe_from_conversation(self, websocket: WebSocket, conversation_id: str):
        """Unsubscribe WebSocket from conversation updates"""
        if conversation_id in self.conversation_subscriptions:
            self.conversation_subscriptions[conversation_id].discard(websocket)
            
            if not self.conversation_subscriptions[conversation_id]:
                del self.conversation_subscriptions[conversation_id]
        
        # Update metadata
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            if "subscriptions" in metadata:
                metadata["subscriptions"].discard(conversation_id)
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
            
            # Update last activity
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["last_activity"] = asyncio.get_event_loop().time()
                
        except Exception as e:
            logger.error("Failed to send personal message", error=str(e))
            await self.disconnect(websocket)
    
    async def send_to_user(self, message: Dict[str, Any], user_id: str):
        """Send message to all connections for a specific user"""
        if user_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error("Failed to send message to user", user_id=user_id, error=str(e))
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                await self.disconnect(ws)
    
    async def broadcast_to_conversation(self, message: Dict[str, Any], conversation_id: str):
        """Send message to all subscribers of a conversation"""
        if conversation_id in self.conversation_subscriptions:
            disconnected = []
            for websocket in self.conversation_subscriptions[conversation_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error("Failed to broadcast to conversation",
                               conversation_id=conversation_id,
                               error=str(e))
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                await self.disconnect(ws)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Send message to all connected clients"""
        disconnected = []
        for websocket in self.connection_metadata.keys():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to broadcast to all", error=str(e))
                disconnected.append(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            await self.disconnect(ws)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        return {
            "total_connections": len(self.connection_metadata),
            "unique_users": len(self.active_connections),
            "conversation_subscriptions": len(self.conversation_subscriptions),
            "connections_by_user": {
                user_id: len(connections) 
                for user_id, connections in self.active_connections.items()
            }
        }


# Create global WebSocket manager
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager
    
    🚀 STARTUP SEQUENCE:
    1. Initialize PostgreSQL database connections with pooling
    2. Run database health checks and create tables if needed
    3. Initialize IPFS client for distributed storage
    4. Set up vector database connections for model embeddings
    5. Start background monitoring and maintenance tasks
    """
    # Startup
    logger.info("🚀 Starting PRSM API server", environment=settings.environment)
    
    # Validate required configuration
    missing_config = settings.validate_required_config()
    if missing_config:
        logger.warning("⚠️ Missing configuration detected", missing_items=missing_config)
        for item in missing_config:
            logger.warning(f"   • {item}")
        
        if settings.is_production and missing_config:
            logger.error("❌ Cannot start in production with missing critical configuration")
            raise RuntimeError(f"Missing required production configuration: {', '.join(missing_config)}")
        else:
            logger.info("🔧 Continuing startup - some features may be limited")
    else:
        logger.info("✅ Configuration validation passed")
    
    try:
        # 🗄️ Initialize PostgreSQL database connections
        await init_database()
        logger.info("✅ Database connections established")
        
        # 🔴 Initialize Redis caching and session management
        await init_redis()
        logger.info("✅ Redis caching and session management initialized")
        
        # 🔍 Initialize vector databases for semantic search
        await init_vector_databases()
        logger.info("✅ Vector databases initialized for semantic search")
        
        # 🌐 Initialize IPFS distributed storage
        await init_ipfs()
        logger.info("✅ IPFS distributed storage initialized")
        
        # 🔐 Initialize secure credential management
        from prsm.integrations.security.secure_config_manager import initialize_secure_configuration
        secure_config_success = await initialize_secure_configuration()
        if secure_config_success:
            logger.info("✅ Secure credential management initialized")
        else:
            logger.warning("⚠️ Secure credential management initialization incomplete")
        
        # 📊 Start background monitoring tasks
        # await start_background_tasks()
        
        logger.info("🎉 PRSM API server startup completed successfully")
        
    except Exception as e:
        logger.error("❌ Failed to start PRSM API server", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PRSM API server")
    
    try:
        # 🗄️ Close database connections
        await close_database()
        logger.info("✅ Database connections closed")
        
        # 🔴 Close Redis connections
        await close_redis()
        logger.info("✅ Redis connections closed")
        
        # 🔍 Close vector database connections
        await close_vector_databases()
        logger.info("✅ Vector database connections closed")
        
        # 🌐 Close IPFS connections
        await close_ipfs()
        logger.info("✅ IPFS connections closed")
        
        # 📊 Stop background tasks
        # await stop_background_tasks()
        
        logger.info("✅ PRSM API server shutdown completed")
        
    except Exception as e:
        logger.error("❌ Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="PRSM API",
    description="Protocol for Recursive Scientific Modeling - API for decentralized AI collaboration",
    version="0.1.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan
)
# Add enhanced security middleware stack (order matters - most specific first)
from prsm.security import RequestLimitsMiddleware, request_limits_config

# Original middleware for compatibility
app.add_middleware(RequestLimitsMiddleware, config=request_limits_config)
app.add_middleware(AuthMiddleware, rate_limit_requests=100, rate_limit_window=60)

# Enhanced security middleware stack
app.add_middleware(RequestValidationMiddleware)
app.add_middleware(RateLimitingMiddleware) 
app.add_middleware(EnhancedSecurityHeaders)

logger.info("Enhanced security middleware stack initialized",
           middleware_count=3,
           features=["request_validation", "rate_limiting", "security_headers"])

# Initialize auth manager
@app.on_event("startup")
async def initialize_auth():
    """Initialize authentication system"""
    try:
        await auth_manager.initialize()
        logger.info("Authentication system initialized")
    except Exception as e:
        logger.error("Failed to initialize auth system", error=str(e))

# Include authentication router
app.include_router(auth_router)


# Add secure CORS middleware
cors_middleware = configure_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_middleware.allow_origins if not settings.is_development else ["*"],
    allow_credentials=cors_middleware.allow_credentials,
    allow_methods=cors_middleware.allow_methods,
    allow_headers=cors_middleware.allow_headers,
    expose_headers=cors_middleware.expose_headers,
    max_age=cors_middleware.max_age
)

logger.info("Secure CORS configuration applied",
           development_mode=settings.is_development,
           origins_count=len(cors_middleware.allow_origins) if not settings.is_development else "all")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("Unhandled exception", exc_info=exc, path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with system information"""
    return {
        "name": "PRSM API",
        "version": "0.1.0",
        "description": "Protocol for Recursive Scientific Modeling",
        "environment": settings.environment.value,
        "status": "operational",
        "features": {
            "nwtn_enabled": settings.nwtn_enabled,
            "ftns_enabled": settings.ftns_enabled,
            "p2p_enabled": settings.p2p_enabled,
            "governance_enabled": settings.governance_enabled,
            "rsi_enabled": settings.rsi_enabled,
        }
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint
    
    🏥 HEALTH MONITORING:
    Tests all critical PRSM subsystems to ensure operational readiness:
    - PostgreSQL database connectivity and performance
    - IPFS distributed storage availability  
    - Vector database embedding services
    - P2P network peer connectivity
    - Circuit breaker safety system status
    """
    from datetime import datetime
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # 🗄️ Database Health Check
    try:
        db_healthy = await db_manager.health_check()
        health_status["components"]["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "last_check": db_manager.last_health_check.isoformat() if db_manager.last_health_check else None,
            "connection_pool": "active" if db_healthy else "failed"
        }
        
        if not db_healthy:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # 🔴 Redis Health Check
    try:
        redis_healthy = await redis_manager.health_check()
        health_status["components"]["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "connected": redis_manager.client.connected,
            "last_check": redis_manager.client.last_health_check.isoformat() if redis_manager.client.last_health_check else None
        }
        
        if not redis_healthy:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # 🌐 IPFS Health Check
    try:
        ipfs_client = get_ipfs_client()
        ipfs_healthy_nodes = await ipfs_client.health_check()
        
        overall_ipfs_health = ipfs_healthy_nodes > 0
        
        health_status["components"]["ipfs"] = {
            "status": "healthy" if overall_ipfs_health else "unhealthy",
            "connected": ipfs_client.connected,
            "healthy_nodes": f"{ipfs_healthy_nodes}/{len(ipfs_client.nodes)}",
            "primary_node": ipfs_client.primary_node.url if ipfs_client.primary_node else None,
            "last_check": ipfs_client.last_health_check.isoformat() if ipfs_client.last_health_check else None
        }
        
        if not overall_ipfs_health:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["components"]["ipfs"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # 🔍 Vector Database Health Check
    try:
        vector_db_manager = get_vector_db_manager()
        vector_health = await vector_db_manager.health_check()
        
        healthy_providers = sum(1 for status in vector_health.values() if status)
        total_providers = len(vector_health)
        
        overall_vector_health = healthy_providers > 0
        
        health_status["components"]["vector_db"] = {
            "status": "healthy" if overall_vector_health else "unhealthy",
            "providers": vector_health,
            "healthy_providers": f"{healthy_providers}/{total_providers}",
            "primary_provider": vector_db_manager.primary_provider.value if vector_db_manager.primary_provider else None
        }
        
        if not overall_vector_health:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["components"]["vector_db"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # 🌐 P2P Network Health Check (placeholder)
    health_status["components"]["p2p_network"] = {
        "status": "not_implemented",
        "message": "P2P network integration pending"
    }
    
    # 🛡️ Safety System Health Check (placeholder)
    health_status["components"]["safety_system"] = {
        "status": "not_implemented",
        "message": "Safety monitoring integration pending"
    }
    
    return health_status


@app.post("/query", response_model=PRSMResponse)
async def process_query(user_input: UserInput, current_user: str = Depends(get_current_user)) -> PRSMResponse:
    """
    Process a user query through the NWTN system
    
    This is the main entry point for PRSM queries that will:
    1. Validate FTNS context allocation
    2. Clarify user intent via NWTN
    3. Decompose query via Architect AIs
    4. Route tasks to appropriate agents
    5. Execute tasks and compile results
    6. Return synthesized response
    """
    logger.info("Processing user query", user_id=user_input.user_id, 
                prompt_length=len(user_input.prompt))
    
    # TODO: Implement full NWTN orchestration pipeline
    # This endpoint is planned for v0.2.0 release
    
    raise HTTPException(
        status_code=501,
        detail={
            "message": "NWTN orchestration coming in v0.2.0",
            "current_status": "development",
            "available_endpoints": [
                "/health - System health check",
                "/models - List available models", 
                "/teachers/* - Teacher model operations",
                "/ipfs/* - Distributed storage",
                "/vectors/* - Semantic search"
            ]
        }
    )


@app.get("/models")
async def list_models(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """
    List available models in the PRSM network
    
    🤖 MODEL DISCOVERY:
    Queries the real PostgreSQL model registry to return currently
    available models across all categories in the PRSM ecosystem
    """
    try:
        # 🧠 Query teacher models from database
        teacher_models = await ModelQueries.get_models_by_type("teacher")
        
        # 🎯 Query specialist models
        specialist_models = await ModelQueries.get_models_by_type("specialist") 
        
        # 🌐 Query general models
        general_models = await ModelQueries.get_models_by_type("general")
        
        total_count = len(teacher_models) + len(specialist_models) + len(general_models)
        
        logger.info("Model registry queried", 
                   teacher_count=len(teacher_models),
                   specialist_count=len(specialist_models), 
                   general_count=len(general_models),
                   total_count=total_count)
        
        return {
            "teacher_models": teacher_models,
            "specialist_models": specialist_models,
            "general_models": general_models,
            "total_count": total_count,
            "registry_status": "active"
        }
        
    except Exception as e:
        logger.error("Failed to query model registry", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve models from registry"
        )


@app.get("/users/{user_id}/balance")
async def get_user_balance(user_id: str, current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get user's FTNS token balance
    
    🪙 FTNS BALANCE QUERY:
    Retrieves real FTNS token balance from PostgreSQL database
    including locked amounts for governance voting and active transactions
    """
    try:
        # 💰 Query user balance from database
        balance_data = await FTNSQueries.get_user_balance(user_id)
        
        logger.debug("User balance queried",
                    user_id=user_id,
                    balance=balance_data["balance"],
                    locked_balance=balance_data["locked_balance"])
        
        return {
            "user_id": user_id,
            "balance": balance_data["balance"],
            "locked_balance": balance_data["locked_balance"], 
            "available_balance": balance_data["balance"] - balance_data["locked_balance"],
            "currency": "FTNS",
            "last_transaction": None  # Coming in v0.2.0 with transaction history
        }
        
    except Exception as e:
        logger.error("Failed to query user balance", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve user balance"
        )


@app.get("/network/status")
async def network_status() -> Dict[str, Any]:
    """Get P2P network status"""
    # P2P networking planned for v0.3.0
    return {
        "connected_peers": 0,
        "total_models": 0,
        "network_health": "single_node_mode",
        "status": "P2P networking coming in v0.3.0"
    }


@app.get("/governance/proposals")
async def list_proposals() -> Dict[str, Any]:
    """List active governance proposals"""
    # Governance system planned for v0.4.0
    return {
        "active_proposals": [],
        "total_count": 0,
        "status": "Governance system coming in v0.4.0"
    }


# === New Database-Backed Endpoints ===


@app.post("/ftns/transactions")
async def create_ftns_transaction(
    transaction_data: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Create a new FTNS transaction
    
    🪙 TRANSACTION PROCESSING:
    Records FTNS token transactions in PostgreSQL for complete
    audit trail and economic transparency
    """
    try:
        # 💳 Validate required fields
        required_fields = ["to_user", "amount", "transaction_type", "description"]
        for field in required_fields:
            if field not in transaction_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # 🔄 Create transaction record
        transaction_data["transaction_id"] = uuid4()
        transaction_id = await FTNSQueries.create_transaction(transaction_data)
        
        # 💰 Update user balance if not a charge
        if transaction_data["transaction_type"] != "charge":
            await FTNSQueries.update_balance(
                transaction_data["to_user"], 
                transaction_data["amount"]
            )
        
        logger.info("FTNS transaction created",
                   transaction_id=transaction_id,
                   transaction_type=transaction_data["transaction_type"],
                   amount=transaction_data["amount"])
        
        return {
            "transaction_id": transaction_id,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create FTNS transaction", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process transaction"
        )


@app.post("/cache/model_output")
async def cache_model_output(
    cache_request: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Cache model output for performance optimization
    
    🚀 MODEL CACHING:
    Stores model outputs in Redis to reduce API costs and improve
    response times for similar queries across the PRSM network
    """
    try:
        # Validate required fields
        required_fields = ["cache_key", "output_data"]
        for field in required_fields:
            if field not in cache_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        model_cache = get_model_cache()
        if not model_cache:
            raise HTTPException(
                status_code=503,
                detail="Model cache not available"
            )
        
        # Cache the model output
        success = await model_cache.store_model_output(
            cache_key=cache_request["cache_key"],
            output_data=cache_request["output_data"],
            ttl=cache_request.get("ttl", 1800)  # 30 minutes default
        )
        
        if success:
            logger.info("Model output cached",
                       cache_key=cache_request["cache_key"])
            
            return {
                "status": "cached",
                "cache_key": cache_request["cache_key"]
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to cache model output"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error caching model output", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/cache/model_output/{cache_key}")
async def get_cached_model_output(
    cache_key: str,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get cached model output
    
    ⚡ CACHE RETRIEVAL:
    Fast retrieval of previously cached model outputs to avoid
    redundant API calls and improve system responsiveness
    """
    try:
        model_cache = get_model_cache()
        if not model_cache:
            raise HTTPException(
                status_code=503,
                detail="Model cache not available"
            )
        
        cached_output = await model_cache.get_model_output(cache_key)
        
        if cached_output:
            logger.debug("Cache hit for model output", cache_key=cache_key)
            return {
                "status": "hit",
                "cache_key": cache_key,
                "output_data": cached_output
            }
        else:
            logger.debug("Cache miss for model output", cache_key=cache_key)
            return {
                "status": "miss",
                "cache_key": cache_key,
                "message": "No cached output found"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving cached model output",
                    cache_key=cache_key,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )




@app.post("/models/register")
async def register_model_with_embedding(
    model_data: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Register a new model with semantic embedding
    
    🤖 MODEL REGISTRATION:
    Registers model in both PostgreSQL database and vector database
    for semantic discovery and similarity matching
    """
    try:
        # Validate required fields
        required_fields = ["model_id", "name", "description", "model_type", "owner_id"]
        for field in required_fields:
            if field not in model_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # 🗄️ Register in PostgreSQL database
        await ModelQueries.register_model(model_data)
        
        # 🧠 Generate embedding for semantic search
        model_embedding = await embedding_generator.generate_model_embedding(model_data)
        
        if model_embedding:
            # 🔍 Store in vector database
            vector_db_manager = get_vector_db_manager()
            
            embedding_metadata = {
                "content": f"{model_data['name']} - {model_data['description']}",
                "model_id": model_data["model_id"],
                "model_type": model_data["model_type"],
                "specialization": model_data.get("specialization", ""),
                "owner_id": model_data["owner_id"],
                "performance_score": model_data.get("performance_score", 0.0)
            }
            
            success = await vector_db_manager.upsert_embedding(
                index_name="models",
                vector_id=model_data["model_id"],
                embedding=model_embedding,
                metadata=embedding_metadata
            )
            
            if success:
                logger.info("Model registered with embedding",
                           model_id=model_data["model_id"],
                           model_type=model_data["model_type"])
                
                return {
                    "model_id": model_data["model_id"],
                    "status": "registered",
                    "semantic_search_enabled": True
                }
            else:
                logger.warning("Model registered but embedding failed",
                              model_id=model_data["model_id"])
                
                return {
                    "model_id": model_data["model_id"],
                    "status": "registered",
                    "semantic_search_enabled": False,
                    "warning": "Vector database indexing failed"
                }
        else:
            logger.warning("Model registered but embedding generation failed",
                          model_id=model_data["model_id"])
            
            return {
                "model_id": model_data["model_id"],
                "status": "registered", 
                "semantic_search_enabled": False,
                "warning": "Embedding generation failed"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register model", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to register model"
        )


@app.post("/models/search/semantic")
async def search_models_semantic(
    search_request: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Semantic search for models using natural language queries
    
    🔍 SEMANTIC DISCOVERY:
    Uses vector similarity to find models that match the intent
    and requirements expressed in natural language queries
    """
    try:
        # Validate required fields
        if "query" not in search_request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: query"
            )
        
        query = search_request["query"]
        top_k = search_request.get("top_k", 10)
        model_type = search_request.get("model_type")
        specialization = search_request.get("specialization")
        
        # 🧠 Generate embedding for the search query
        query_embedding = await embedding_generator.generate_embedding(query)
        
        if not query_embedding:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate query embedding"
            )
        
        # 🔍 Search similar models using vector similarity
        vector_db_manager = get_vector_db_manager()
        
        similar_models = await vector_db_manager.search_similar_models(
            query_embedding=query_embedding,
            top_k=top_k,
            model_type=model_type,
            specialization=specialization
        )
        
        # Format results with similarity scores
        results = []
        for result in similar_models:
            model_info = {
                "model_id": result.metadata.get("model_id"),
                "name": result.metadata.get("content", "").split(" - ")[0] if " - " in result.metadata.get("content", "") else "Unknown",
                "model_type": result.metadata.get("model_type"),
                "specialization": result.metadata.get("specialization"),
                "similarity_score": result.score,
                "performance_score": result.metadata.get("performance_score", 0.0)
            }
            results.append(model_info)
        
        logger.info("Semantic model search completed",
                   query_length=len(query),
                   results_count=len(results),
                   top_k=top_k)
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_type": "semantic",
            "model_filters": {
                "model_type": model_type,
                "specialization": specialization
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Semantic model search failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Semantic search failed"
        )


@app.get("/models/{model_id}/similar")
async def find_similar_models(
    model_id: str,
    top_k: int = 5,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Find models similar to a specific model
    
    🔗 MODEL SIMILARITY:
    Uses vector similarity to discover models with similar
    capabilities, specializations, or performance characteristics
    """
    try:
        # 🗄️ Get model metadata from database
        models = await ModelQueries.get_models_by_type("teacher")  # Get all models for now
        models.extend(await ModelQueries.get_models_by_type("specialist"))
        models.extend(await ModelQueries.get_models_by_type("general"))
        
        target_model = None
        for model in models:
            if model["model_id"] == model_id:
                target_model = model
                break
        
        if not target_model:
            raise HTTPException(
                status_code=404,
                detail="Model not found"
            )
        
        # 🧠 Generate embedding for target model
        target_embedding = await embedding_generator.generate_model_embedding(target_model)
        
        if not target_embedding:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate model embedding"
            )
        
        # 🔍 Search for similar models
        vector_db_manager = get_vector_db_manager()
        
        similar_models = await vector_db_manager.search_similar_models(
            query_embedding=target_embedding,
            top_k=top_k + 1,  # +1 to account for the model itself
            model_type=target_model.get("model_type"),
            specialization=target_model.get("specialization")
        )
        
        # Filter out the target model itself
        filtered_results = []
        for result in similar_models:
            if result.metadata.get("model_id") != model_id:
                model_info = {
                    "model_id": result.metadata.get("model_id"),
                    "name": result.metadata.get("content", "").split(" - ")[0] if " - " in result.metadata.get("content", "") else "Unknown",
                    "model_type": result.metadata.get("model_type"),
                    "specialization": result.metadata.get("specialization"),
                    "similarity_score": result.score,
                    "performance_score": result.metadata.get("performance_score", 0.0)
                }
                filtered_results.append(model_info)
                
                if len(filtered_results) >= top_k:
                    break
        
        logger.info("Similar models found",
                   target_model_id=model_id,
                   similar_models_count=len(filtered_results))
        
        return {
            "target_model_id": model_id,
            "target_model_name": target_model["name"],
            "similar_models": filtered_results,
            "total_found": len(filtered_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to find similar models",
                    model_id=model_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to find similar models"
        )


@app.get("/vectors/stats")
async def get_vector_database_stats(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive vector database statistics
    
    📊 VECTOR ANALYTICS:
    Provides insights into vector database usage, performance,
    and indexing status across all providers
    """
    try:
        vector_db_manager = get_vector_db_manager()
        
        # Get health status
        health_status = await vector_db_manager.health_check()
        
        # Get provider statistics
        provider_stats = await vector_db_manager.get_provider_stats()
        
        # Calculate summary statistics
        total_vectors = 0
        healthy_providers = 0
        
        for provider, status in health_status.items():
            if status:
                healthy_providers += 1
                
                # Sum vectors across indexes for this provider
                provider_vectors = 0
                if provider in provider_stats:
                    for index_name, index_stats in provider_stats[provider].items():
                        provider_vectors += index_stats.get("total_vector_count", 0)
                
                total_vectors = max(total_vectors, provider_vectors)  # Use max across providers
        
        return {
            "summary": {
                "total_providers": len(health_status),
                "healthy_providers": healthy_providers,
                "primary_provider": vector_db_manager.primary_provider.value if vector_db_manager.primary_provider else None,
                "total_vectors_indexed": total_vectors,
                "indexes_configured": list(vector_db_manager.indexes.keys())
            },
            "provider_health": health_status,
            "provider_statistics": provider_stats,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions
        }
        
    except Exception as e:
        logger.error("Failed to get vector database stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve vector database statistics"
        )


@app.post("/embeddings/generate")
async def generate_embedding_endpoint(
    embedding_request: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate embedding for arbitrary text
    
    🧠 EMBEDDING SERVICE:
    Provides embedding generation as a service with caching
    for integration with external systems and development
    """
    try:
        if "text" not in embedding_request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: text"
            )
        
        text = embedding_request["text"]
        
        if len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        # Generate embedding
        embedding = await embedding_generator.generate_embedding(text)
        
        if embedding is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embedding"
            )
        
        logger.debug("Embedding generated via API",
                    text_length=len(text),
                    embedding_dimension=len(embedding))
        
        return {
            "text": text,
            "embedding": embedding,
            "model": settings.embedding_model,
            "dimensions": len(embedding),
            "text_length": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate embedding", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to generate embedding"
        )












# === Teacher Model Endpoints ===

@app.post("/teachers/create")
async def create_teacher_endpoint(
    teacher_request: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new teacher model with real ML implementation
    
    🎓 TEACHER CREATION:
    Creates production-ready teacher models with actual ML training
    capabilities for knowledge distillation and curriculum generation
    """
    try:
        from prsm.teachers.teacher_model import create_teacher_with_specialization
        
        # Validate required fields
        required_fields = ["specialization", "domain"]
        for field in required_fields:
            if field not in teacher_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        specialization = teacher_request["specialization"]
        domain = teacher_request["domain"]
        use_real_implementation = teacher_request.get("use_real_implementation", True)
        
        # Create teacher
        teacher = await create_teacher_with_specialization(
            specialization=specialization,
            domain=domain,
            use_real_implementation=use_real_implementation
        )
        
        # Get teacher info
        teacher_id = str(teacher.teacher_model.teacher_id)
        implementation_type = "real" if hasattr(teacher, 'capabilities_assessor') else "simulated"
        
        logger.info("Teacher model created via API",
                   teacher_id=teacher_id,
                   specialization=specialization,
                   domain=domain,
                   implementation=implementation_type)
        
        return {
            "success": True,
            "teacher_id": teacher_id,
            "specialization": specialization,
            "domain": domain,
            "implementation": implementation_type,
            "name": teacher.teacher_model.name,
            "capabilities": "real_ml_training" if implementation_type == "real" else "simulated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create teacher model", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create teacher model"
        )


@app.post("/teachers/{teacher_id}/teach")
async def conduct_teaching_session(
    teacher_id: str,
    teaching_request: Dict[str, Any],
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Conduct a teaching session with a specific teacher
    
    🎯 TEACHING SESSION:
    Runs actual teaching sessions with real capability assessment,
    adaptive curriculum generation, and ML-based training
    """
    try:
        # This would typically retrieve the teacher from a registry
        # For now, we'll demonstrate with a new teacher instance
        from prsm.teachers.teacher_model import create_teacher_with_specialization
        
        # Validate required fields
        required_fields = ["student_model_id", "domain", "learning_objectives"]
        for field in required_fields:
            if field not in teaching_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        student_model_id = teaching_request["student_model_id"]
        domain = teaching_request["domain"]
        learning_objectives = teaching_request["learning_objectives"]
        
        # Create teacher for this session (in production, would retrieve from registry)
        teacher = await create_teacher_with_specialization(
            specialization="adaptive_learning",
            domain=domain,
            use_real_implementation=True
        )
        
        # Conduct teaching session
        if hasattr(teacher, 'teach_student'):
            # Real implementation
            learning_session = await teacher.teach_student(
                student_model_id=student_model_id,
                domain=domain,
                learning_objectives=learning_objectives
            )
            
            session_type = "real_ml_training"
            session_details = {
                "session_id": str(learning_session.session_id),
                "student_id": str(learning_session.student_id),
                "teacher_id": str(learning_session.teacher_id),
                "learning_gain": learning_session.learning_gain,
                "completed": learning_session.completed,
                "pre_assessment": learning_session.performance_before,
                "post_assessment": learning_session.performance_after
            }
        else:
            # Simulated implementation fallback
            from prsm.core.models import LearningSession
            from uuid import uuid4
            
            learning_session = LearningSession(
                session_id=uuid4(),
                teacher_id=teacher.teacher_model.teacher_id,
                student_id=uuid4(),
                curriculum_id=uuid4(),
                performance_before={"accuracy": 0.6},
                performance_after={"accuracy": 0.75},
                learning_gain=0.15,
                completed=True
            )
            
            session_type = "simulated"
            session_details = {
                "session_id": str(learning_session.session_id),
                "student_id": str(learning_session.student_id),
                "teacher_id": str(learning_session.teacher_id),
                "learning_gain": learning_session.learning_gain,
                "completed": learning_session.completed
            }
        
        logger.info("Teaching session completed",
                   teacher_id=teacher_id,
                   student_model=student_model_id,
                   learning_gain=learning_session.learning_gain,
                   session_type=session_type)
        
        return {
            "success": True,
            "session_type": session_type,
            "domain": domain,
            "learning_objectives": learning_objectives,
            **session_details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Teaching session failed",
                    teacher_id=teacher_id,
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Teaching session failed"
        )


@app.get("/teachers/available_backends")
async def get_available_teacher_backends(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get information about available teacher model backends
    
    📊 BACKEND STATUS:
    Returns information about ML frameworks and capabilities
    available for real teacher model implementations
    """
    try:
        from prsm.teachers.real_teacher_implementation import get_available_training_backends
        
        backends = get_available_training_backends()
        
        backend_info = {
            "available_backends": backends,
            "total_backends": len(backends),
            "real_implementation_available": len(backends) > 0,
            "capabilities": {
                "knowledge_distillation": len(backends) > 0,
                "real_model_training": len(backends) > 0,
                "adaptive_curriculum": True,  # Always available
                "performance_assessment": len(backends) > 0
            }
        }
        
        return backend_info
        
    except Exception as e:
        logger.error("Failed to get teacher backend information", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve backend information"
        )



# === WebSocket Endpoints ===

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    Main WebSocket endpoint for real-time communication
    
    🔌 REAL-TIME CONNECTION:
    Establishes persistent connection for live updates including:
    - Conversation message streaming
    - Real-time notifications  
    - Live data updates (tokenomics, tasks, files)
    - System status updates
    
    🔐 SECURITY:
    Requires JWT authentication via query parameter, Authorization header, or cookie.
    Connection authenticated before acceptance to prevent unauthorized access.
    """
    from .websocket_auth import authenticate_websocket_connection, cleanup_websocket_connection, WebSocketAuthError
    
    try:
        # 🛡️ AUTHENTICATE CONNECTION BEFORE ACCEPTING
        connection = await authenticate_websocket_connection(websocket, user_id, "general")
        await websocket.accept()
        
        # Connect to websocket manager after authentication
        await websocket_manager.connect(websocket, user_id, "general")
        
        logger.info("Secure WebSocket connection established",
                   user_id=user_id,
                   username=connection.username,
                   role=connection.role.value,
                   ip_address=connection.ip_address)
        
        while True:
            # Listen for messages from client
            data = await websocket.receive_text()
            
            # 🛡️ VALIDATE MESSAGE SIZE AND RATE LIMITS
            from prsm.security import validate_websocket_message
            try:
                await validate_websocket_message(websocket, data, user_id)
            except Exception as e:
                logger.warning("WebSocket message validation failed",
                             user_id=user_id,
                             error=str(e))
                await websocket.close(code=1008, reason="Message validation failed")
                return
            
            message = json.loads(data)
            
            # Handle different message types with authentication context
            await handle_websocket_message(websocket, user_id, message, connection)
            
    except WebSocketAuthError as e:
        # Authentication failed - close with appropriate code
        logger.warning("WebSocket authentication failed",
                      user_id=user_id,
                      error=e.message,
                      code=e.code)
        await websocket.close(code=e.code, reason=e.message)
        return
        
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
        await cleanup_websocket_connection(websocket)
        
    except Exception as e:
        logger.error("WebSocket error", user_id=user_id, error=str(e))
        await websocket_manager.disconnect(websocket)
        await cleanup_websocket_connection(websocket)
        
        # 🛡️ CLEANUP SECURITY TRACKING
        from prsm.security import cleanup_websocket_connection as cleanup_security
        await cleanup_security(websocket)


@app.websocket("/ws/conversation/{user_id}/{conversation_id}")
async def conversation_websocket(websocket: WebSocket, user_id: str, conversation_id: str):
    """
    Conversation-specific WebSocket for streaming AI responses
    
    💬 CONVERSATION STREAMING:
    Provides real-time streaming for conversation messages with:
    - Token-by-token AI response streaming
    - Live typing indicators
    - Message status updates
    - Context usage tracking
    
    🔐 SECURITY:
    Requires JWT authentication and verifies user has access to the specific conversation.
    Prevents unauthorized access to private conversation streams.
    """
    from .websocket_auth import authenticate_websocket_connection, cleanup_websocket_connection, WebSocketAuthError
    
    try:
        # 🛡️ AUTHENTICATE CONNECTION AND VERIFY CONVERSATION ACCESS
        connection = await authenticate_websocket_connection(
            websocket, user_id, "conversation", conversation_id
        )
        await websocket.accept()
        
        # Connect to websocket manager after authentication
        await websocket_manager.connect(websocket, user_id, "conversation")
        await websocket_manager.subscribe_to_conversation(websocket, conversation_id)
        
        logger.info("Secure conversation WebSocket established",
                   user_id=user_id,
                   conversation_id=conversation_id,
                   username=connection.username,
                   role=connection.role.value,
                   ip_address=connection.ip_address)
        
        while True:
            # Listen for conversation messages
            data = await websocket.receive_text()
            
            # 🛡️ VALIDATE MESSAGE SIZE AND RATE LIMITS
            from prsm.security import validate_websocket_message
            try:
                await validate_websocket_message(websocket, data, user_id)
            except Exception as e:
                logger.warning("Conversation WebSocket message validation failed",
                             user_id=user_id,
                             conversation_id=conversation_id,
                             error=str(e))
                await websocket.close(code=1008, reason="Message validation failed")
                return
            
            message = json.loads(data)
            
            # Handle conversation-specific messages with authentication context
            await handle_conversation_message(websocket, user_id, conversation_id, message, connection)
            
    except WebSocketAuthError as e:
        # Authentication failed - close with appropriate code
        logger.warning("Conversation WebSocket authentication failed",
                      user_id=user_id,
                      conversation_id=conversation_id,
                      error=e.message,
                      code=e.code)
        await websocket.close(code=e.code, reason=e.message)
        return
        
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
        await cleanup_websocket_connection(websocket)
        
        # 🛡️ CLEANUP SECURITY TRACKING
        from prsm.security import cleanup_websocket_connection as cleanup_security
        await cleanup_security(websocket)
        
    except Exception as e:
        logger.error("Conversation WebSocket error",
                    user_id=user_id,
                    conversation_id=conversation_id,
                    error=str(e))
        await websocket_manager.disconnect(websocket)
        await cleanup_websocket_connection(websocket)
        
        # 🛡️ CLEANUP SECURITY TRACKING
        from prsm.security import cleanup_websocket_connection as cleanup_security
        await cleanup_security(websocket)


async def handle_websocket_message(websocket: WebSocket, user_id: str, message: Dict[str, Any], connection=None):
    """Handle incoming WebSocket messages with authentication context"""
    from .websocket_auth import require_websocket_permission, WebSocketAuthError
    
    message_type = message.get("type")
    
    if message_type == "ping":
        # Respond to ping with pong
        await websocket_manager.send_personal_message({
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)
        
    elif message_type == "subscribe_conversation":
        # Subscribe to conversation updates (requires conversation permission)
        try:
            await require_websocket_permission(websocket, "conversation.read")
        except WebSocketAuthError:
            await websocket_manager.send_personal_message({
                "type": "error",
                "message": "Permission denied: conversation.read required",
                "timestamp": asyncio.get_event_loop().time()
            }, websocket)
            return
            
        conversation_id = message.get("conversation_id")
        if conversation_id:
            await websocket_manager.subscribe_to_conversation(websocket, conversation_id)
            await websocket_manager.send_personal_message({
                "type": "subscribed",
                "conversation_id": conversation_id,
                "message": f"Subscribed to conversation {conversation_id}"
            }, websocket)
    
    elif message_type == "unsubscribe_conversation":
        # Unsubscribe from conversation updates
        conversation_id = message.get("conversation_id")
        if conversation_id:
            await websocket_manager.unsubscribe_from_conversation(websocket, conversation_id)
            await websocket_manager.send_personal_message({
                "type": "unsubscribed",
                "conversation_id": conversation_id,
                "message": f"Unsubscribed from conversation {conversation_id}"
            }, websocket)
    
    elif message_type == "request_status":
        # Send current status information
        stats = websocket_manager.get_connection_stats()
        await websocket_manager.send_personal_message({
            "type": "status_update",
            "user_id": user_id,
            "connection_stats": stats,
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)
    
    else:
        # Unknown message type
        await websocket_manager.send_personal_message({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)


async def handle_conversation_message(websocket: WebSocket, user_id: str, conversation_id: str, message: Dict[str, Any], connection=None):
    """Handle conversation-specific WebSocket messages with authentication context"""
    from .websocket_auth import require_websocket_permission, WebSocketAuthError
    
    message_type = message.get("type")
    
    if message_type == "send_message":
        # Handle real-time message sending with streaming response
        content = message.get("content")
        if not content:
            await websocket_manager.send_personal_message({
                "type": "error",
                "message": "Message content is required"
            }, websocket)
            return
        
        # Broadcast user message to conversation subscribers
        await websocket_manager.broadcast_to_conversation({
            "type": "user_message",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "content": content,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)
        
        # Start streaming AI response
        await stream_ai_response(conversation_id, content, user_id)
        
    elif message_type == "typing_start":
        # Broadcast typing indicator
        await websocket_manager.broadcast_to_conversation({
            "type": "typing_indicator",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "typing": True,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)
        
    elif message_type == "typing_stop":
        # Broadcast typing stop
        await websocket_manager.broadcast_to_conversation({
            "type": "typing_indicator",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "typing": False,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)


async def stream_ai_response(conversation_id: str, user_message: str, user_id: str):
    """
    Stream AI response token by token for real-time experience
    
    🤖 AI RESPONSE STREAMING:
    Simulates streaming AI responses for demonstration.
    In production, this would integrate with actual NWTN orchestrator.
    """
    # Simulate AI response generation
    ai_response = f"Thank you for your message: '{user_message}'. This is a streaming response from NWTN. In production, this would be connected to the actual NWTN orchestrator for real AI processing."
    
    # Send typing indicator
    await websocket_manager.broadcast_to_conversation({
        "type": "ai_typing",
        "conversation_id": conversation_id,
        "typing": True,
        "timestamp": asyncio.get_event_loop().time()
    }, conversation_id)
    
    # Stream response token by token
    words = ai_response.split()
    streamed_content = ""
    
    for i, word in enumerate(words):
        # Add word to streamed content
        if i > 0:
            streamed_content += " "
        streamed_content += word
        
        # Send partial response
        await websocket_manager.broadcast_to_conversation({
            "type": "ai_response_chunk",
            "conversation_id": conversation_id,
            "partial_content": streamed_content,
            "is_complete": False,
            "timestamp": asyncio.get_event_loop().time()
        }, conversation_id)
        
        # Simulate typing delay
        await asyncio.sleep(0.1)
    
    # Send completion message
    await websocket_manager.broadcast_to_conversation({
        "type": "ai_response_complete",
        "conversation_id": conversation_id,
        "final_content": streamed_content,
        "user_id": user_id,
        "model_used": "nwtn-v1",
        "context_tokens": len(user_message.split()) * 1.3 + len(words),
        "timestamp": asyncio.get_event_loop().time()
    }, conversation_id)
    
    # Stop typing indicator
    await websocket_manager.broadcast_to_conversation({
        "type": "ai_typing",
        "conversation_id": conversation_id,
        "typing": False,
        "timestamp": asyncio.get_event_loop().time()
    }, conversation_id)


@app.get("/ws/stats")
async def get_websocket_stats(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get WebSocket connection statistics
    
    📊 CONNECTION MONITORING:
    Returns real-time statistics about WebSocket connections
    for monitoring and debugging purposes
    
    🔐 SECURITY:
    Requires authentication and admin permissions to view system statistics
    """
    from .websocket_auth import websocket_auth
    
    try:
        # Check admin permissions
        user = await auth_manager.get_user_by_id(user_id)
        if not user or user.role not in ["admin", "moderator"]:
            raise HTTPException(
                status_code=403,
                detail="Admin permissions required to view WebSocket statistics"
            )
        
        # Get combined stats from both managers
        websocket_stats = websocket_manager.get_connection_stats()
        auth_stats = await websocket_auth.get_connection_stats()
        
        return {
            "success": True,
            "stats": {
                "websocket_manager": websocket_stats,
                "authentication": auth_stats,
                "security": {
                    "authenticated_connections": auth_stats["active_connections"],
                    "unique_authenticated_users": auth_stats["unique_users"],
                    "max_connections_per_user": auth_stats["max_connections_per_user"]
                }
            },
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get WebSocket stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve WebSocket statistics"
        )


# Enhanced UI endpoints with WebSocket integration

@app.post("/ui/conversations/{conversation_id}/messages/streaming")
async def send_streaming_message(conversation_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send message with WebSocket streaming response
    
    💬 STREAMING INTEGRATION:
    Sends message via REST API and triggers WebSocket streaming
    for real-time AI response delivery
    """
    try:
        from uuid import uuid4
        from datetime import datetime
        
        # Validate message
        if "content" not in message_data:
            raise HTTPException(
                status_code=400,
                detail="Missing message content"
            )
        
        user_id = message_data.get("user_id", "anonymous")
        content = message_data["content"]
        
        # Create user message
        message_id = str(uuid4())
        user_message = {
            "message_id": message_id,
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "context_tokens": len(content.split()) * 1.3
        }
        
        # Start streaming response via WebSocket
        asyncio.create_task(stream_ai_response(conversation_id, content, user_id))
        
        logger.info("Streaming message sent",
                   conversation_id=conversation_id,
                   user_id=user_id,
                   message_length=len(content))
        
        return {
            "success": True,
            "message_id": message_id,
            "conversation_id": conversation_id,
            "streaming": True,
            "message": "Response will be streamed via WebSocket"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to send streaming message", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to send streaming message"
        )


# Notification system endpoints

@app.post("/notifications/send")
async def send_notification(notification_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send real-time notification to users
    
    🔔 NOTIFICATION SYSTEM:
    Sends notifications via WebSocket to specific users or all users
    for system updates, task notifications, etc.
    """
    try:
        notification_type = notification_data.get("type", "general")
        target = notification_data.get("target", "all")  # "all", "user", or user_id
        message = notification_data.get("message", "")
        
        notification = {
            "type": "notification",
            "notification_type": notification_type,
            "message": message,
            "timestamp": asyncio.get_event_loop().time(),
            "data": notification_data.get("data", {})
        }
        
        if target == "all":
            # Broadcast to all connected users
            await websocket_manager.broadcast_to_all(notification)
            logger.info("Notification broadcast to all users", type=notification_type)
            
        elif target == "user" and "user_id" in notification_data:
            # Send to specific user
            user_id = notification_data["user_id"]
            await websocket_manager.send_to_user(notification, user_id)
            logger.info("Notification sent to user", user_id=user_id, type=notification_type)
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid notification target"
            )
        
        return {
            "success": True,
            "notification_type": notification_type,
            "target": target,
            "message": "Notification sent successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to send notification", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to send notification"
        )


# === Integration Layer Routers ===
try:
    from prsm.integrations.api.integration_api import integration_router
    from prsm.integrations.api.config_api import config_router
    from prsm.integrations.api.security_api import security_router
    
    app.include_router(integration_router, prefix="/integrations", tags=["Integrations"])
    app.include_router(config_router, prefix="/integrations/config", tags=["Configuration"])
    app.include_router(security_router, prefix="/integrations/security", tags=["Security"])
    logger.info("✅ Integration layer API endpoints enabled (including enhanced security)")
except ImportError as e:
    logger.warning(f"⚠️ Integration layer not available: {e}")

# Include teams API router
app.include_router(teams_router, prefix="/api/v1", tags=["Teams"])
app.include_router(credential_router, tags=["Credentials"])  # Already has /api/v1/credentials prefix
app.include_router(security_router, tags=["Security"])  # Already has /api/v1/security prefix
app.include_router(security_logging_router, tags=["Security Logging"])  # Already has /api/v1/security/logging prefix
app.include_router(payment_router, tags=["Payments"])  # Already has /api/v1/payments prefix
app.include_router(crypto_router, tags=["Cryptography"])  # Already has /api/v1/crypto prefix
# MARKETPLACE API: Universal endpoints with complete consolidation
app.include_router(marketplace_router, prefix="/api/v1/marketplace", tags=["Marketplace"])  # Universal /resources endpoints
app.include_router(recommendation_router, prefix="/api/v1/marketplace", tags=["Recommendations"])  # AI-powered recommendations
app.include_router(reputation_router, prefix="/api/v1", tags=["Reputation"])  # User reputation and trust system
app.include_router(distillation_router, prefix="/api/v1", tags=["Distillation"])  # Automated knowledge distillation
app.include_router(monitoring_router, prefix="/api/v1", tags=["Monitoring"])  # Enterprise monitoring and observability
app.include_router(compliance_router, prefix="/api/v1", tags=["Compliance"])  # SOC2/ISO27001 compliance framework
app.include_router(governance_router, tags=["Governance"])  # Already has /api/v1/governance prefix
app.include_router(mainnet_router, prefix="/api/v1", tags=["Mainnet Deployment"])
app.include_router(health_router, prefix="/api/v1", tags=["Health"])  # Has /health prefix, needs /api/v1
app.include_router(budget_router, tags=["Budget Management"])  # Already has /api/v1/budget prefix
app.include_router(web3_router, prefix="/api/v1", tags=["Web3"])
app.include_router(chronos_router, prefix="/api/v1", tags=["CHRONOS"])
app.include_router(ui_router, prefix="/ui", tags=["UI"])
app.include_router(ipfs_router, prefix="/ipfs", tags=["IPFS"])
app.include_router(session_router, prefix="/sessions", tags=["Sessions"])
app.include_router(task_router, prefix="/tasks", tags=["Tasks"])
logger.info("✅ IPFS API endpoints enabled")
logger.info("✅ Session API endpoints enabled")
logger.info("✅ Task API endpoints enabled")
logger.info("✅ Teams API endpoints enabled")
logger.info("✅ Security Logging API endpoints enabled")
logger.info("✅ Payment Processing API endpoints enabled")
logger.info("✅ CHRONOS Clearing Protocol API endpoints enabled")
logger.info("✅ Cryptography API endpoints enabled")
logger.info("✅ Governance API endpoints enabled")
logger.info("✅ Mainnet Deployment API endpoints enabled")
logger.info("✅ UI API endpoints enabled")

# Include additional routers when implemented
# app.include_router(nwtn_router, prefix="/nwtn", tags=["NWTN"])
# app.include_router(agents_router, prefix="/agents", tags=["Agents"])
# app.include_router(tokenomics_router, prefix="/ftns", tags=["FTNS"])


if __name__ == "__main__":
    import uvicorn
    # Use environment variable for host, default to localhost for security
    import os
    host = os.getenv("PRSM_API_HOST", "127.0.0.1")  # Default to localhost
    uvicorn.run(app, host=host, port=8000, reload=True)
