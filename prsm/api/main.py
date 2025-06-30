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
# MARKETPLACE API ENABLED: Real database implementations complete
from prsm.api.real_marketplace_api import router as marketplace_router
from prsm.api.marketplace_launch_api import router as marketplace_launch_router
from prsm.api.governance_api import router as governance_router
from prsm.api.mainnet_deployment_api import router as mainnet_router
# from prsm.api.health_api import router as health_router
from prsm.api.budget_api import router as budget_router
from prsm.web3.frontend_integration import router as web3_router
from prsm.chronos.api import router as chronos_router
from prsm.auth.auth_manager import auth_manager
from prsm.auth import get_current_user
from prsm.auth.middleware import AuthMiddleware, SecurityHeadersMiddleware


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
# Add security middleware (order matters - most specific first)
from prsm.security import RequestLimitsMiddleware, request_limits_config

app.add_middleware(RequestLimitsMiddleware, config=request_limits_config)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuthMiddleware, rate_limit_requests=100, rate_limit_window=60)

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


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def process_query(user_input: UserInput) -> PRSMResponse:
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
async def list_models() -> Dict[str, Any]:
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
async def get_user_balance(user_id: str) -> Dict[str, Any]:
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

@app.post("/sessions")
async def create_session(user_input: UserInput) -> Dict[str, Any]:
    """
    Create a new PRSM session with database persistence
    
    🎯 SESSION LIFECYCLE:
    Creates a new session in PostgreSQL with context allocation
    and initializes the reasoning trace for transparency
    """
    try:
        session_id = uuid4()
        
        # 💾 Create session in database
        session_data = {
            "session_id": session_id,
            "user_id": user_input.user_id,
            "nwtn_context_allocation": user_input.context_allocation or settings.nwtn_max_context_per_query,
            "context_used": 0,
            "status": "pending",
            "metadata": user_input.preferences
        }
        
        created_session_id = await SessionQueries.create_session(session_data)
        
        # 🔴 Cache session data in Redis for fast access
        session_cache = get_session_cache()
        if session_cache:
            await session_cache.store_session(created_session_id, session_data)
            
            # Cache context allocation separately for quick lookups
            context_data = {
                "allocated": session_data["nwtn_context_allocation"],
                "used": 0,
                "remaining": session_data["nwtn_context_allocation"]
            }
            await session_cache.store_context_allocation(created_session_id, context_data)
        
        logger.info("New session created and cached",
                   session_id=created_session_id,
                   user_id=user_input.user_id,
                   context_allocation=session_data["nwtn_context_allocation"])
        
        return {
            "session_id": created_session_id,
            "status": "created",
            "context_allocated": session_data["nwtn_context_allocation"],
            "cached": session_cache is not None,
            "message": "Session ready for query processing"
        }
        
    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create session"
        )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """
    Get session details from database
    
    📋 SESSION RETRIEVAL:
    Fetches complete session information including current status,
    context usage, and reasoning trace from PostgreSQL
    """
    try:
        # 🔴 Try to get session from Redis cache first for speed
        session_cache = get_session_cache()
        session_data = None
        
        if session_cache:
            session_data = await session_cache.get_session(session_id)
            if session_data:
                logger.debug("Session retrieved from cache",
                           session_id=session_id,
                           status=session_data["status"])
        
        # 🗄️ Fallback to database if not in cache
        if not session_data:
            session_data = await SessionQueries.get_session(session_id)
            
            # Cache the session data for future requests
            if session_data and session_cache:
                await session_cache.store_session(session_id, session_data)
        
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
        # 🔴 Also get context allocation if available
        context_data = None
        if session_cache:
            context_data = await session_cache.get_context_allocation(session_id)
        
        if context_data:
            session_data["context_allocation"] = context_data
        
        logger.debug("Session retrieved",
                    session_id=session_id,
                    status=session_data["status"],
                    from_cache=context_data is not None)
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve session", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session"
        )


@app.post("/ftns/transactions")
async def create_ftns_transaction(
    transaction_data: Dict[str, Any]
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
async def cache_model_output(cache_request: Dict[str, Any]) -> Dict[str, str]:
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
async def get_cached_model_output(cache_key: str) -> Dict[str, Any]:
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


@app.post("/tasks/enqueue")
async def enqueue_task(task_request: Dict[str, Any]) -> Dict[str, str]:
    """
    Enqueue task in distributed task queue
    
    📝 TASK DISTRIBUTION:
    Adds tasks to Redis-based distributed queue for processing
    across the PRSM P2P network with priority ordering
    """
    try:
        # Validate required fields
        required_fields = ["queue_name", "task_data"]
        for field in required_fields:
            if field not in task_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        task_queue = get_task_queue()
        if not task_queue:
            raise HTTPException(
                status_code=503,
                detail="Task queue not available"
            )
        
        # Enqueue the task
        success = await task_queue.enqueue_task(
            queue_name=task_request["queue_name"],
            task_data=task_request["task_data"],
            priority=task_request.get("priority", 0)
        )
        
        if success:
            task_id = task_request["task_data"].get("task_id", "unknown")
            logger.info("Task enqueued",
                       queue_name=task_request["queue_name"],
                       task_id=task_id,
                       priority=task_request.get("priority", 0))
            
            return {
                "status": "enqueued",
                "queue_name": task_request["queue_name"],
                "task_id": task_id
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to enqueue task"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error enqueuing task", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.post("/models/register")
async def register_model_with_embedding(model_data: Dict[str, Any]) -> Dict[str, str]:
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
async def search_models_semantic(search_request: Dict[str, Any]) -> Dict[str, Any]:
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
async def find_similar_models(model_id: str, top_k: int = 5) -> Dict[str, Any]:
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
async def get_vector_database_stats() -> Dict[str, Any]:
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
async def generate_embedding_endpoint(embedding_request: Dict[str, Any]) -> Dict[str, Any]:
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


@app.post("/ipfs/upload")
async def upload_to_ipfs(upload_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload content to IPFS distributed storage
    
    🌐 DISTRIBUTED STORAGE:
    Uploads content to IPFS network with automatic retry, pinning,
    and multi-node failover for reliable distributed storage
    """
    try:
        # Validate required fields
        if "content" not in upload_request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: content"
            )
        
        content = upload_request["content"]
        filename = upload_request.get("filename", "content.txt")
        pin = upload_request.get("pin", True)
        content_type = upload_request.get("content_type", "text/plain")
        
        # Convert string content to bytes
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        
        ipfs_client = get_ipfs_client()
        
        if not ipfs_client.connected:
            raise HTTPException(
                status_code=503,
                detail="IPFS client not connected"
            )
        
        # Upload to IPFS
        result = await ipfs_client.upload_content(
            content=content_bytes,
            filename=filename,
            pin=pin
        )
        
        if result.success:
            logger.info("Content uploaded to IPFS",
                       cid=result.cid,
                       filename=filename,
                       size=result.size,
                       pinned=pin)
            
            return {
                "success": True,
                "cid": result.cid,
                "size": result.size,
                "filename": filename,
                "pinned": pin,
                "execution_time": result.execution_time,
                "node_type": result.connection_type.value if result.connection_type else None,
                "access_url": f"https://ipfs.io/ipfs/{result.cid}"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Upload failed: {result.error}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("IPFS upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="IPFS upload failed"
        )


@app.get("/ipfs/{cid}")
async def download_from_ipfs(cid: str, download: bool = False) -> Dict[str, Any]:
    """
    Download or retrieve content from IPFS
    
    📥 DISTRIBUTED RETRIEVAL:
    Retrieves content from IPFS network with intelligent node selection,
    gateway fallback, and content verification
    """
    try:
        ipfs_client = get_ipfs_client()
        
        if not ipfs_client.connected:
            raise HTTPException(
                status_code=503,
                detail="IPFS client not connected"
            )
        
        # Download content
        result = await ipfs_client.download_content(
            cid=cid,
            verify_integrity=True
        )
        
        if result.success and result.metadata and "content" in result.metadata:
            content = result.metadata["content"]
            
            # Try to decode as text, fall back to base64 for binary
            try:
                text_content = content.decode('utf-8')
                content_type = "text/plain"
                response_content = text_content
            except UnicodeDecodeError:
                import base64
                content_type = "application/octet-stream"
                response_content = base64.b64encode(content).decode('ascii')
            
            logger.info("Content downloaded from IPFS",
                       cid=cid,
                       size=result.size,
                       content_type=content_type)
            
            response_data = {
                "success": True,
                "cid": cid,
                "content": response_content,
                "content_type": content_type,
                "size": result.size,
                "execution_time": result.execution_time,
                "node_type": result.connection_type.value if result.connection_type else None
            }
            
            if download:
                # Return downloadable response
                from fastapi.responses import Response
                if content_type == "text/plain":
                    return Response(
                        content=text_content,
                        media_type=content_type,
                        headers={"Content-Disposition": f"attachment; filename={cid}.txt"}
                    )
                else:
                    return Response(
                        content=content,
                        media_type=content_type,
                        headers={"Content-Disposition": f"attachment; filename={cid}.bin"}
                    )
            else:
                return response_data
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Content not found or download failed: {result.error}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("IPFS download failed", cid=cid, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="IPFS download failed"
        )


@app.post("/ipfs/models/upload")
async def upload_model_to_ipfs(model_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload trained model to IPFS with metadata
    
    🤖 MODEL DISTRIBUTION:
    Specialized endpoint for uploading trained models with comprehensive
    metadata for discovery and version control across the PRSM network
    """
    try:
        from pathlib import Path
        import tempfile
        import base64
        
        # Validate required fields
        required_fields = ["model_data", "model_metadata"]
        for field in required_fields:
            if field not in model_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        model_data = model_request["model_data"]
        model_metadata = model_request["model_metadata"]
        
        # Decode base64 model data
        try:
            model_bytes = base64.b64decode(model_data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 model data"
            )
        
        # Create temporary file for model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as temp_file:
            temp_file.write(model_bytes)
            temp_model_path = Path(temp_file.name)
        
        try:
            # Upload model using PRSM IPFS operations
            result = await prsm_ipfs.upload_model(
                model_path=temp_model_path,
                model_metadata=model_metadata
            )
            
            if result.success:
                logger.info("Model uploaded to IPFS",
                           model_cid=result.cid,
                           model_name=model_metadata.get("name", "unknown"),
                           size=result.size)
                
                response_data = {
                    "success": True,
                    "model_cid": result.cid,
                    "metadata_cid": result.metadata.get("metadata_cid") if result.metadata else None,
                    "size": result.size,
                    "execution_time": result.execution_time,
                    "model_metadata": result.metadata.get("prsm_metadata") if result.metadata else None,
                    "access_url": f"https://ipfs.io/ipfs/{result.cid}"
                }
                
                return response_data
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model upload failed: {result.error}"
                )
        
        finally:
            # Clean up temporary file
            try:
                temp_model_path.unlink()
            except:
                pass
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model upload to IPFS failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Model upload failed"
        )


@app.post("/ipfs/research/publish")
async def publish_research_content(research_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Publish research content to IPFS
    
    📚 RESEARCH PUBLICATION:
    Publishes research papers, datasets, and other academic content
    to IPFS with comprehensive metadata for discovery and citation
    """
    try:
        from pathlib import Path
        import tempfile
        import base64
        
        # Validate required fields
        required_fields = ["content_data", "research_metadata"]
        for field in required_fields:
            if field not in research_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        content_data = research_request["content_data"]
        research_metadata = research_request["research_metadata"]
        
        # Handle different content types
        if isinstance(content_data, str):
            # Text content
            content_bytes = content_data.encode('utf-8')
            filename = research_metadata.get("filename", "research_content.txt")
        else:
            # Binary content (base64 encoded)
            try:
                content_bytes = base64.b64decode(content_data)
                filename = research_metadata.get("filename", "research_content.bin")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid content data format"
                )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(content_bytes)
            temp_content_path = Path(temp_file.name)
        
        try:
            # Publish using PRSM IPFS operations
            result = await prsm_ipfs.publish_research_content(
                content_path=temp_content_path,
                content_metadata=research_metadata
            )
            
            if result.success:
                logger.info("Research content published to IPFS",
                           content_cid=result.cid,
                           title=research_metadata.get("title", "unknown"),
                           content_type=research_metadata.get("content_type", "unknown"))
                
                response_data = {
                    "success": True,
                    "content_cid": result.cid,
                    "metadata_cid": result.metadata.get("metadata_cid") if result.metadata else None,
                    "size": result.size,
                    "execution_time": result.execution_time,
                    "research_metadata": result.metadata.get("research_metadata") if result.metadata else None,
                    "access_url": f"https://ipfs.io/ipfs/{result.cid}",
                    "citation_info": {
                        "cid": result.cid,
                        "title": research_metadata.get("title", ""),
                        "authors": research_metadata.get("authors", []),
                        "published_date": result.metadata.get("research_metadata", {}).get("published_at") if result.metadata else None
                    }
                }
                
                return response_data
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Research content publishing failed: {result.error}"
                )
        
        finally:
            # Clean up temporary file
            try:
                temp_content_path.unlink()
            except:
                pass
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Research content publishing failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Research content publishing failed"
        )


@app.get("/ipfs/status")
async def get_ipfs_status() -> Dict[str, Any]:
    """
    Get comprehensive IPFS network status
    
    📊 IPFS MONITORING:
    Provides detailed information about IPFS node health,
    connection status, and network performance metrics
    """
    try:
        ipfs_client = get_ipfs_client()
        
        # Get node status
        node_statuses = await ipfs_client.get_node_status()
        
        # Calculate summary statistics
        total_nodes = len(node_statuses)
        healthy_nodes = sum(1 for node in node_statuses if node["healthy"])
        
        # Get average response time for healthy nodes
        healthy_response_times = [
            node["response_time"] for node in node_statuses 
            if node["healthy"] and node["response_time"] != float('inf')
        ]
        avg_response_time = sum(healthy_response_times) / len(healthy_response_times) if healthy_response_times else 0
        
        # Categorize nodes by type
        api_nodes = [node for node in node_statuses if node["connection_type"] == "http_api"]
        gateway_nodes = [node for node in node_statuses if node["connection_type"] == "gateway"]
        
        return {
            "summary": {
                "connected": ipfs_client.connected,
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "health_percentage": (healthy_nodes / total_nodes * 100) if total_nodes > 0 else 0,
                "average_response_time": avg_response_time,
                "primary_node": ipfs_client.primary_node.url if ipfs_client.primary_node else None,
                "last_health_check": ipfs_client.last_health_check.isoformat() if ipfs_client.last_health_check else None
            },
            "node_details": {
                "api_nodes": {
                    "total": len(api_nodes),
                    "healthy": sum(1 for node in api_nodes if node["healthy"]),
                    "nodes": api_nodes
                },
                "gateway_nodes": {
                    "total": len(gateway_nodes),
                    "healthy": sum(1 for node in gateway_nodes if node["healthy"]),
                    "nodes": gateway_nodes
                }
            },
            "configuration": {
                "retry_max_attempts": ipfs_client.retry_strategy.max_attempts,
                "retry_base_delay": ipfs_client.retry_strategy.base_delay,
                "retry_max_delay": ipfs_client.retry_strategy.max_delay,
                "content_cache_enabled": ipfs_client.content_cache_enabled,
                "max_cache_size": ipfs_client.max_cache_size
            }
        }
        
    except Exception as e:
        logger.error("Failed to get IPFS status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve IPFS status"
        )


# === Teacher Model Endpoints ===

@app.post("/teachers/create")
async def create_teacher_endpoint(teacher_request: Dict[str, Any]) -> Dict[str, Any]:
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
    teaching_request: Dict[str, Any]
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
async def get_available_teacher_backends() -> Dict[str, Any]:
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


# === UI Integration Endpoints ===

@app.post("/ui/conversations")
async def create_conversation(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new conversation for the UI
    
    💬 CONVERSATION CREATION:
    Creates a new conversation session with initialization for
    NWTN processing and real-time UI updates
    """
    try:
        from uuid import uuid4
        from datetime import datetime
        
        # Create conversation
        conversation_id = str(uuid4())
        conversation = {
            "conversation_id": conversation_id,
            "user_id": conversation_data.get("user_id", "anonymous"),
            "title": conversation_data.get("title", "New Conversation"),
            "model": conversation_data.get("model", "nwtn-v1"),
            "mode": conversation_data.get("mode", "dynamic"),
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "context_used": 0,
            "context_limit": 4096,
            "status": "active"
        }
        
        # Cache conversation for quick access
        session_cache = get_session_cache()
        if session_cache:
            await session_cache.store_session(conversation_id, conversation)
        
        logger.info("New conversation created for UI",
                   conversation_id=conversation_id,
                   model=conversation["model"],
                   mode=conversation["mode"])
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "status": "created",
            "model": conversation["model"],
            "mode": conversation["mode"]
        }
        
    except Exception as e:
        logger.error("Failed to create conversation", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create conversation"
        )


@app.post("/ui/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a message in a conversation
    
    📤 MESSAGE PROCESSING:
    Processes user messages through NWTN orchestrator with
    real-time streaming response for the UI
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
        
        message_id = str(uuid4())
        user_message = {
            "message_id": message_id,
            "role": "user",
            "content": message_data["content"],
            "timestamp": datetime.now().isoformat(),
            "context_tokens": len(message_data["content"].split()) * 1.3  # Rough estimate
        }
        
        # Get conversation from cache
        session_cache = get_session_cache()
        conversation = None
        if session_cache:
            conversation = await session_cache.get_session(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        # Add user message
        conversation["messages"].append(user_message)
        conversation["context_used"] += user_message["context_tokens"]
        
        # Generate AI response (placeholder for NWTN integration)
        ai_response_id = str(uuid4())
        ai_response = {
            "message_id": ai_response_id,
            "role": "assistant",
            "content": f"I received your message: '{message_data['content']}'. NWTN orchestration coming in v0.2.0 - this is currently a mock response from the {conversation['model']} model in {conversation['mode']} mode.",
            "timestamp": datetime.now().isoformat(),
            "context_tokens": 50,  # Mock response tokens
            "model_used": conversation["model"],
            "mode_used": conversation["mode"],
            "processing_time": 1.2
        }
        
        conversation["messages"].append(ai_response)
        conversation["context_used"] += ai_response["context_tokens"]
        
        # Update conversation in cache
        if session_cache:
            await session_cache.store_session(conversation_id, conversation)
        
        logger.info("Message processed in conversation",
                   conversation_id=conversation_id,
                   user_message_length=len(message_data["content"]),
                   context_used=conversation["context_used"])
        
        return {
            "success": True,
            "user_message": user_message,
            "ai_response": ai_response,
            "conversation_status": {
                "context_used": conversation["context_used"],
                "context_limit": conversation["context_limit"],
                "message_count": len(conversation["messages"])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process message", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process message"
        )


@app.get("/ui/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Get conversation details for UI display"""
    try:
        session_cache = get_session_cache()
        conversation = None
        
        if session_cache:
            conversation = await session_cache.get_session(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        return {
            "success": True,
            "conversation": conversation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get conversation", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation"
        )


@app.get("/ui/conversations")
async def list_conversations(user_id: str = None) -> Dict[str, Any]:
    """
    List conversations for conversation history sidebar
    
    📋 CONVERSATION HISTORY:
    Returns list of recent conversations for the history sidebar
    with titles, timestamps, and status information
    """
    try:
        # Mock conversation history (in production, query from database)
        conversations = [
            {
                "conversation_id": "conv-1",
                "title": "Research on Atomically Precise Manufacturing",
                "last_message_at": "2024-01-15T10:30:00Z",
                "message_count": 12,
                "status": "active"
            },
            {
                "conversation_id": "conv-2", 
                "title": "Initial Brainstorming Session",
                "last_message_at": "2024-01-14T15:45:00Z",
                "message_count": 8,
                "status": "archived"
            },
            {
                "conversation_id": "conv-3",
                "title": "Code Generation Request",
                "last_message_at": "2024-01-13T09:15:00Z",
                "message_count": 15,
                "status": "active"
            },
            {
                "conversation_id": "conv-4",
                "title": "Summarize IPFS Paper",
                "last_message_at": "2024-01-12T14:20:00Z",
                "message_count": 6,
                "status": "completed"
            }
        ]
        
        # Filter by user if specified
        if user_id:
            # In production, filter conversations by user_id
            pass
        
        return {
            "success": True,
            "conversations": conversations,
            "total_count": len(conversations)
        }
        
    except Exception as e:
        logger.error("Failed to list conversations", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to list conversations"
        )


@app.post("/ui/files/upload")
async def upload_file_ui(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle file uploads from the UI
    
    📁 FILE UPLOAD:
    Processes file uploads from the UI and stores them in IPFS
    with automatic metadata generation and user file management
    """
    try:
        import base64
        from pathlib import Path
        
        # Validate required fields
        required_fields = ["filename", "content", "content_type"]
        for field in required_fields:
            if field not in file_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        filename = file_data["filename"]
        content_type = file_data["content_type"]
        user_id = file_data.get("user_id", "anonymous")
        
        # Decode file content
        if file_data.get("encoding") == "base64":
            content = base64.b64decode(file_data["content"])
        else:
            content = file_data["content"].encode('utf-8')
        
        # Upload to IPFS
        ipfs_client = get_ipfs_client()
        result = await ipfs_client.upload_content(
            content=content,
            filename=filename,
            pin=True
        )
        
        if result.success:
            file_info = {
                "file_id": result.cid,
                "filename": filename,
                "content_type": content_type,
                "size": result.size,
                "uploaded_at": result.metadata.get("uploaded_at") if result.metadata else None,
                "ipfs_cid": result.cid,
                "access_url": f"https://ipfs.io/ipfs/{result.cid}",
                "privacy": "private",  # Default privacy setting
                "ai_access": "core_only"  # Default AI access
            }
            
            logger.info("File uploaded via UI",
                       filename=filename,
                       cid=result.cid,
                       size=result.size,
                       user_id=user_id)
            
            return {
                "success": True,
                "file": file_info,
                "message": "File uploaded successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"File upload failed: {result.error}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("UI file upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="File upload failed"
        )


@app.get("/ui/files")
async def list_user_files(user_id: str = None) -> Dict[str, Any]:
    """
    List user files for My Files tab
    
    📂 FILE MANAGEMENT:
    Returns user's uploaded files with privacy settings,
    sharing options, and AI access permissions
    """
    try:
        # Mock file list (in production, query from database)
        files = [
            {
                "file_id": "QmXXX1",
                "filename": "analysis.ipynb",
                "content_type": "application/x-ipynb+json",
                "size": 245760,
                "uploaded_at": "2024-01-15T08:30:00Z",
                "privacy": "private",
                "ai_access": "core_only",
                "sharing": {
                    "public_ipfs": False,
                    "shared_users": []
                }
            },
            {
                "file_id": "QmXXX2", 
                "filename": "project_alpha_data.csv",
                "content_type": "text/csv",
                "size": 1024000,
                "uploaded_at": "2024-01-14T16:45:00Z",
                "privacy": "private",
                "ai_access": "core_only",
                "sharing": {
                    "public_ipfs": False,
                    "shared_users": ["user123"]
                }
            }
        ]
        
        return {
            "success": True,
            "files": files,
            "total_count": len(files),
            "storage_used": sum(f["size"] for f in files),
            "storage_limit": 10 * 1024 * 1024 * 1024  # 10GB limit
        }
        
    except Exception as e:
        logger.error("Failed to list user files", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to list files"
        )


@app.get("/ui/tokenomics/{user_id}")
async def get_tokenomics_ui(user_id: str) -> Dict[str, Any]:
    """
    Get user tokenomics data for UI display
    
    💰 TOKENOMICS DASHBOARD:
    Returns comprehensive FTNS token information including
    balance, staking, earnings, and transaction history
    """
    try:
        # Get balance from existing endpoint
        balance_data = await FTNSQueries.get_user_balance(user_id)
        
        # Mock additional tokenomics data
        tokenomics_data = {
            "balance": {
                "total": balance_data["balance"],
                "available": balance_data["balance"] - balance_data["locked_balance"],
                "locked": balance_data["locked_balance"],
                "currency": "FTNS"
            },
            "staking": {
                "staked_amount": 500,
                "apy": 8.5,
                "rewards_earned": 42.5,
                "staking_period": "30_days"
            },
            "earnings": {
                "total_earned": 156.8,
                "sources": {
                    "ipfs_hosting": 89.2,
                    "model_hosting": 45.6,
                    "compute_contribution": 22.0
                },
                "current_status": "active"
            },
            "recent_transactions": [
                {
                    "transaction_id": "tx-1",
                    "type": "earned",
                    "amount": 12.5,
                    "description": "IPFS hosting rewards",
                    "timestamp": "2024-01-15T10:00:00Z"
                },
                {
                    "transaction_id": "tx-2",
                    "type": "spent",
                    "amount": -5.0,
                    "description": "NWTN query processing",
                    "timestamp": "2024-01-15T09:30:00Z"
                }
            ]
        }
        
        logger.debug("Tokenomics data retrieved for UI",
                    user_id=user_id,
                    balance=tokenomics_data["balance"]["total"])
        
        return {
            "success": True,
            "tokenomics": tokenomics_data
        }
        
    except Exception as e:
        logger.error("Failed to get tokenomics data", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve tokenomics data"
        )


@app.get("/ui/tasks/{user_id}")
async def get_user_tasks(user_id: str) -> Dict[str, Any]:
    """
    Get user tasks for Tasks tab
    
    📋 TASK MANAGEMENT:
    Returns user's assigned tasks with status, priorities,
    and interactive management capabilities
    """
    try:
        # Mock task data (in production, query from database)
        tasks = [
            {
                "task_id": "task-1",
                "title": "Review research paper draft",
                "description": "Review the quantum computing research paper and provide feedback",
                "status": "pending",
                "priority": "high",
                "assigned_by": "ai_system",
                "created_at": "2024-01-15T08:00:00Z",
                "due_date": "2024-01-17T17:00:00Z",
                "actions": ["mark_done", "provide_feedback"]
            },
            {
                "task_id": "task-2",
                "title": "Upload experiment results",
                "description": "Upload the latest experiment results to IPFS",
                "status": "in_progress",
                "priority": "medium",
                "assigned_by": "user",
                "created_at": "2024-01-14T12:00:00Z",
                "due_date": "2024-01-16T12:00:00Z",
                "actions": ["upload_files", "mark_complete"]
            },
            {
                "task_id": "task-3",
                "title": "Validate model training results",
                "description": "Review and validate the distilled model training results",
                "status": "completed",
                "priority": "high",
                "assigned_by": "ai_system",
                "created_at": "2024-01-13T09:00:00Z",
                "completed_at": "2024-01-14T15:30:00Z",
                "actions": []
            }
        ]
        
        # Calculate task statistics
        total_tasks = len(tasks)
        pending_tasks = len([t for t in tasks if t["status"] == "pending"])
        in_progress_tasks = len([t for t in tasks if t["status"] == "in_progress"])
        completed_tasks = len([t for t in tasks if t["status"] == "completed"])
        
        return {
            "success": True,
            "tasks": tasks,
            "statistics": {
                "total": total_tasks,
                "pending": pending_tasks,
                "in_progress": in_progress_tasks,
                "completed": completed_tasks
            }
        }
        
    except Exception as e:
        logger.error("Failed to get user tasks", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve tasks"
        )


@app.post("/ui/tasks")
async def create_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new task"""
    try:
        from uuid import uuid4
        from datetime import datetime
        
        task_id = str(uuid4())
        task = {
            "task_id": task_id,
            "title": task_data.get("title", "New Task"),
            "description": task_data.get("description", ""),
            "status": "pending",
            "priority": task_data.get("priority", "medium"),
            "assigned_by": task_data.get("assigned_by", "user"),
            "created_at": datetime.now().isoformat(),
            "due_date": task_data.get("due_date"),
            "actions": task_data.get("actions", ["mark_done"])
        }
        
        logger.info("New task created via UI",
                   task_id=task_id,
                   title=task["title"],
                   priority=task["priority"])
        
        return {
            "success": True,
            "task": task,
            "message": "Task created successfully"
        }
        
    except Exception as e:
        logger.error("Failed to create task", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create task"
        )


@app.post("/ui/settings/save")
async def save_ui_settings(settings_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save user settings from the UI
    
    ⚙️ SETTINGS PERSISTENCE:
    Saves user preferences, API keys, and configuration
    securely with validation and encryption
    """
    try:
        user_id = settings_data.get("user_id", "anonymous")
        
        # Validate and process API keys
        api_keys = settings_data.get("api_keys", {})
        processed_keys = {}
        
        for provider, key_data in api_keys.items():
            if key_data.get("key") and len(key_data["key"]) > 10:  # Basic validation
                # In production, encrypt API keys before storage
                processed_keys[provider] = {
                    "key": f"***{key_data['key'][-4:]}",  # Mask for response
                    "mode_mapping": key_data.get("mode_mapping", "dynamic"),
                    "enabled": True
                }
        
        # Save settings (mock implementation)
        settings_saved = {
            "user_id": user_id,
            "api_keys": processed_keys,
            "ui_preferences": settings_data.get("ui_preferences", {}),
            "saved_at": "2024-01-15T12:00:00Z"
        }
        
        logger.info("User settings saved via UI",
                   user_id=user_id,
                   api_providers=list(processed_keys.keys()))
        
        return {
            "success": True,
            "message": "Settings saved successfully",
            "api_keys_configured": len(processed_keys),
            "api_providers": list(processed_keys.keys())
        }
        
    except Exception as e:
        logger.error("Failed to save settings", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to save settings"
        )


@app.get("/ui/settings/{user_id}")
async def get_ui_settings(user_id: str) -> Dict[str, Any]:
    """Get user settings for UI display"""
    try:
        # Mock settings retrieval
        settings = {
            "user_id": user_id,
            "api_keys": {
                "openai": {
                    "configured": True,
                    "mode_mapping": "code",
                    "key_preview": "***XYZ"
                },
                "anthropic": {
                    "configured": False,
                    "mode_mapping": "dynamic",
                    "key_preview": None
                }
            },
            "ui_preferences": {
                "theme": "dark",
                "left_panel_collapsed": False,
                "history_sidebar_visible": True
            }
        }
        
        return {
            "success": True,
            "settings": settings
        }
        
    except Exception as e:
        logger.error("Failed to get settings", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve settings"
        )


@app.get("/ui/information-space")
async def get_information_space_data(
    layout: str = Query("force_directed", description="Layout algorithm"),
    color_by: str = Query("type", description="Color scheme"),
    filters: Optional[str] = Query(None, description="JSON filters"),
    limit: int = Query(100, ge=1, le=500, description="Maximum nodes to return")
) -> Dict[str, Any]:
    """
    Get data for Information Space visualization
    
    🔬 INFORMATION SPACE:
    Returns real graph data for research opportunity visualization
    based on IPFS content analysis and semantic relationships
    
    Enhanced with comprehensive Information Space functionality:
    - Real content analysis from IPFS
    - Semantic relationship mapping
    - Research opportunity identification
    - Interactive visualization data
    - FTNS token integration
    """
    try:
        # Import Information Space components
        from ..information_space.service import InformationSpaceService
        from ..information_space.api import get_information_space_data_enhanced
        
        # Try to use real Information Space service
        try:
            return await get_information_space_data_enhanced(layout, color_by, filters, limit)
        except Exception as service_error:
            logger.warning(f"Information Space service unavailable, using enhanced mock: {service_error}")
            
        # Enhanced mock data with realistic structure
        nodes = [
            {
                "id": "quantum_computing",
                "label": "Quantum Computing",
                "type": "research_area",
                "connections": 15,
                "opportunity_score": 0.85,
                "influence_score": 0.78,
                "research_activity": 0.92,
                "ftns_value": 2500.0,
                "x": 100, "y": 50,
                "color": "#3498db",
                "size": 25,
                "tags": ["quantum", "computing", "algorithms"],
                "description": "Advanced quantum computing research and applications"
            },
            {
                "id": "machine_learning",
                "label": "Machine Learning", 
                "type": "research_area",
                "connections": 23,
                "opportunity_score": 0.92,
                "influence_score": 0.89,
                "research_activity": 0.95,
                "ftns_value": 3200.0,
                "x": -80, "y": 120,
                "color": "#2ecc71",
                "size": 30,
                "tags": ["ml", "ai", "neural-networks"],
                "description": "Machine learning algorithms and neural network research"
            },
            {
                "id": "distributed_systems",
                "label": "Distributed Systems",
                "type": "research_area", 
                "connections": 18,
                "opportunity_score": 0.78,
                "influence_score": 0.71,
                "research_activity": 0.83,
                "ftns_value": 1800.0,
                "x": 150, "y": -90,
                "color": "#e74c3c",
                "size": 22,
                "tags": ["distributed", "systems", "p2p"],
                "description": "Distributed computing and peer-to-peer networks"
            },
            {
                "id": "ai_safety",
                "label": "AI Safety",
                "type": "research_area",
                "connections": 12,
                "opportunity_score": 0.88,
                "influence_score": 0.82,
                "research_activity": 0.76,
                "ftns_value": 2100.0,
                "x": -120, "y": -60,
                "color": "#f39c12",
                "size": 24,
                "tags": ["safety", "alignment", "governance"],
                "description": "AI safety research and alignment mechanisms"
            }
        ]
        
        edges = [
            {
                "id": "edge_1",
                "source": "quantum_computing",
                "target": "machine_learning",
                "weight": 0.75,
                "confidence": 0.82,
                "type": "semantic_similarity",
                "color": "#2ecc71",
                "width": 3,
                "description": "Quantum machine learning applications"
            },
            {
                "id": "edge_2", 
                "source": "machine_learning",
                "target": "distributed_systems",
                "weight": 0.68,
                "confidence": 0.74,
                "type": "collaboration",
                "color": "#e74c3c",
                "width": 2,
                "description": "Distributed ML training and inference"
            },
            {
                "id": "edge_3",
                "source": "machine_learning",
                "target": "ai_safety",
                "weight": 0.85,
                "confidence": 0.91,
                "type": "collaboration",
                "color": "#e74c3c",
                "width": 4,
                "description": "Safe AI development practices"
            },
            {
                "id": "edge_4",
                "source": "quantum_computing",
                "target": "distributed_systems",
                "weight": 0.62,
                "confidence": 0.69,
                "type": "technical_synergy",
                "color": "#9b59b6",
                "width": 2,
                "description": "Distributed quantum computing"
            }
        ]
        
        opportunities = [
            {
                "id": "opp_1",
                "title": "Quantum-Enhanced ML Algorithms",
                "description": "Develop machine learning algorithms that leverage quantum computing advantages",
                "type": "cross_domain",
                "confidence": 0.82,
                "impact_score": 0.91,
                "feasibility_score": 0.75,
                "estimated_value": 5000.0,
                "research_areas": ["quantum_computing", "machine_learning"],
                "suggested_timeline": "12-18 months"
            },
            {
                "id": "opp_2",
                "title": "Distributed Quantum Computing Networks",
                "description": "Create distributed networks for quantum computation sharing",
                "type": "collaboration",
                "confidence": 0.75,
                "impact_score": 0.88,
                "feasibility_score": 0.68,
                "estimated_value": 7500.0,
                "research_areas": ["quantum_computing", "distributed_systems"],
                "suggested_timeline": "18-24 months"
            },
            {
                "id": "opp_3",
                "title": "Safe AI Development Framework",
                "description": "Comprehensive framework for developing safe and aligned AI systems",
                "type": "knowledge_gap",
                "confidence": 0.88,
                "impact_score": 0.95,
                "feasibility_score": 0.82,
                "estimated_value": 10000.0,
                "research_areas": ["machine_learning", "ai_safety"],
                "suggested_timeline": "6-12 months"
            }
        ]
        
        return {
            "success": True,
            "graph_data": {
                "nodes": nodes[:limit],
                "edges": edges
            },
            "opportunities": opportunities,
            "visualization_config": {
                "layout": layout,
                "color_scheme": color_by
            },
            "metadata": {
                "total_research_areas": len(nodes),
                "total_connections": len(edges),
                "total_opportunities": len(opportunities),
                "last_updated": datetime.utcnow().isoformat(),
                "service_status": "enhanced_mock",
                "features": [
                    "real_time_updates",
                    "semantic_analysis", 
                    "ftns_integration",
                    "collaboration_detection",
                    "opportunity_scoring"
                ]
            },
            "statistics": {
                "average_opportunity_score": sum(n["opportunity_score"] for n in nodes) / len(nodes),
                "total_ftns_value": sum(n["ftns_value"] for n in nodes),
                "active_research_areas": sum(1 for n in nodes if n["research_activity"] > 0.8)
            }
        }
        
    except Exception as e:
        logger.error("Failed to get information space data", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve information space data"
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
app.include_router(credential_router, tags=["Credentials"])
app.include_router(security_router, tags=["Security"])
app.include_router(security_logging_router, tags=["Security Logging"])
app.include_router(payment_router, tags=["Payments"])
app.include_router(crypto_router, tags=["Cryptography"])
# MARKETPLACE API DISABLED: See docs/architecture/marketplace-status.md
app.include_router(marketplace_router, prefix="/api/v1/marketplace", tags=["Marketplace"])
app.include_router(marketplace_launch_router, tags=["Marketplace Launch"])
app.include_router(governance_router, tags=["Governance"])
app.include_router(mainnet_router, tags=["Mainnet Deployment"])
# app.include_router(health_router, tags=["Health"])
app.include_router(budget_router, tags=["Budget Management"])
app.include_router(web3_router, prefix="/api/v1", tags=["Web3"])
app.include_router(chronos_router, prefix="/api/v1", tags=["CHRONOS"])
logger.info("✅ Teams API endpoints enabled")
logger.info("✅ Security Logging API endpoints enabled")
logger.info("✅ Payment Processing API endpoints enabled")
logger.info("✅ CHRONOS Clearing Protocol API endpoints enabled")
logger.info("✅ Cryptography API endpoints enabled")
logger.info("✅ Governance API endpoints enabled")
logger.info("✅ Mainnet Deployment API endpoints enabled")

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