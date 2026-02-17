"""
Core API Endpoints
==================

Root, health, and foundational PRSM API endpoints.
These are included directly in the main application, not as a router.
"""

import asyncio
import json
import structlog
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from prsm.core.config import get_settings
from prsm.core.models import UserInput, PRSMResponse
from prsm.core.auth import get_current_user

logger = structlog.get_logger(__name__)
settings = get_settings()


def register_core_endpoints(app: FastAPI) -> None:
    """
    Register core endpoints directly on the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    _register_root_endpoint(app)
    _register_health_endpoint(app)
    _register_query_endpoint(app)
    _register_model_endpoints(app)
    _register_user_endpoints(app)
    _register_network_endpoints(app)
    _register_websocket_endpoints(app)
    _register_notification_endpoints(app)

    logger.info("Core endpoints registered")


def _register_root_endpoint(app: FastAPI) -> None:
    """Register root endpoint."""

    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint with system information"""
        _settings = settings or get_settings()
        return {
            "name": "PRSM API",
            "version": "0.1.0",
            "description": "Protocol for Recursive Scientific Modeling",
            "environment": _settings.environment.value if _settings else "unknown",
            "status": "operational",
            "features": {
                "nwtn_enabled": getattr(_settings, "nwtn_enabled", True),
                "ftns_enabled": getattr(_settings, "ftns_enabled", True),
                "p2p_enabled": getattr(_settings, "p2p_enabled", False),
                "governance_enabled": getattr(_settings, "governance_enabled", True),
                "rsi_enabled": getattr(_settings, "rsi_enabled", False),
            } if _settings else {}
        }


def _register_health_endpoint(app: FastAPI) -> None:
    """Register comprehensive health check endpoint."""

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """
        Comprehensive health check endpoint

        Tests all critical PRSM subsystems to ensure operational readiness:
        - PostgreSQL database connectivity and performance
        - Redis caching availability
        - IPFS distributed storage
        - Vector database embedding services
        """
        from prsm.core.database import db_manager
        from prsm.core.redis_client import redis_manager
        from prsm.core.ipfs_client import get_ipfs_client
        from prsm.core.vector_db import get_vector_db_manager

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }

        # Database Health Check
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
            health_status["components"]["database"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "unhealthy"

        # Redis Health Check
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
            health_status["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"

        # IPFS Health Check
        try:
            ipfs_client = get_ipfs_client()
            ipfs_healthy_nodes = await ipfs_client.health_check()
            overall_ipfs_health = ipfs_healthy_nodes > 0
            health_status["components"]["ipfs"] = {
                "status": "healthy" if overall_ipfs_health else "unhealthy",
                "connected": ipfs_client.connected,
                "healthy_nodes": f"{ipfs_healthy_nodes}/{len(ipfs_client.nodes)}",
                "primary_node": ipfs_client.primary_node.url if ipfs_client.primary_node else None
            }
            if not overall_ipfs_health:
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["ipfs"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "unhealthy"

        # Vector Database Health Check
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
            health_status["components"]["vector_db"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "unhealthy"

        # Placeholder for future components
        health_status["components"]["p2p_network"] = {
            "status": "not_implemented",
            "message": "P2P network integration pending"
        }
        health_status["components"]["safety_system"] = {
            "status": "not_implemented",
            "message": "Safety monitoring integration pending"
        }

        return health_status


def _register_query_endpoint(app: FastAPI) -> None:
    """Register main query processing endpoint."""

    @app.post("/query", response_model=PRSMResponse)
    async def process_query(
        user_input: UserInput,
        current_user: str = Depends(get_current_user)
    ) -> PRSMResponse:
        """
        Process a user query through the NWTN system

        This is the main entry point for PRSM queries.
        """
        from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
        from prsm.core.models import AgentType

        logger.info("Processing user query",
                   user_id=user_input.user_id,
                   prompt_length=len(user_input.prompt))

        orchestrator = NeuroSymbolicOrchestrator(node_id="api_node_primary")
        result = await orchestrator.solve_task(
            user_input.prompt,
            user_input.context_allocation or ""
        )

        # Map the neuro-symbolic trace to the API response model
        mapped_trace = []
        for step in result.get("trace", []):
            agent_type = AgentType.CANDIDATE_GENERATOR
            if "VERIFICATION" in step.get("a", "") or "CHECK" in step.get("a", ""):
                agent_type = AgentType.CANDIDATE_EVALUATOR
            elif "STRATEGY" in step.get("a", ""):
                agent_type = AgentType.ARCHITECT

            mapped_trace.append({
                "step_id": uuid4(),
                "agent_type": agent_type,
                "agent_id": result.get("node_id", "system"),
                "input_data": {"action": step.get("a")},
                "output_data": {"content": step.get("c"), "metadata": step.get("m")},
                "execution_time": 0.1,
                "confidence_score": step.get("s", 1.0),
                "timestamp": datetime.now()
            })

        return PRSMResponse(
            session_id=user_input.session_id or uuid4(),
            user_id=user_input.user_id,
            final_answer=result["output"],
            reasoning_trace=mapped_trace,
            confidence_score=result.get("reward", 1.0),
            context_used=len(result["output"].split()),
            ftns_charged=0.0,
            safety_validated=True,
            metadata={
                "verification_hash": result.get("verification_hash"),
                "input_hash": result.get("input_hash"),
                "pq_signature": result.get("pq_signature"),
                "raw_trace": result.get("trace"),
                "mode": result.get("mode")
            }
        )


def _register_model_endpoints(app: FastAPI) -> None:
    """Register model-related endpoints."""

    @app.get("/models")
    async def list_models(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
        """List available models in the PRSM network"""
        from prsm.core.database import ModelQueries

        try:
            teacher_models = await ModelQueries.get_models_by_type("teacher")
            specialist_models = await ModelQueries.get_models_by_type("specialist")
            general_models = await ModelQueries.get_models_by_type("general")

            total_count = len(teacher_models) + len(specialist_models) + len(general_models)

            return {
                "teacher_models": teacher_models,
                "specialist_models": specialist_models,
                "general_models": general_models,
                "total_count": total_count,
                "registry_status": "active"
            }
        except Exception as e:
            logger.error("Failed to query model registry", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to retrieve models from registry")


def _register_user_endpoints(app: FastAPI) -> None:
    """Register user-related endpoints."""

    @app.get("/users/{user_id}/balance")
    async def get_user_balance(
        user_id: str,
        current_user: str = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Get user's FTNS token balance"""
        from prsm.core.database import FTNSQueries

        try:
            balance_data = await FTNSQueries.get_user_balance(user_id)
            return {
                "user_id": user_id,
                "balance": balance_data["balance"],
                "locked_balance": balance_data["locked_balance"],
                "available_balance": balance_data["balance"] - balance_data["locked_balance"],
                "currency": "FTNS",
                "last_transaction": None
            }
        except Exception as e:
            logger.error("Failed to query user balance", user_id=user_id, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to retrieve user balance")


def _register_network_endpoints(app: FastAPI) -> None:
    """Register network status endpoints."""

    @app.get("/network/status")
    async def network_status() -> Dict[str, Any]:
        """Get P2P network status"""
        return {
            "connected_peers": 0,
            "total_models": 0,
            "network_health": "single_node_mode",
            "status": "P2P networking coming in v0.3.0"
        }

    @app.get("/governance/proposals")
    async def list_proposals() -> Dict[str, Any]:
        """List active governance proposals"""
        return {
            "active_proposals": [],
            "total_count": 0,
            "status": "Governance system coming in v0.4.0"
        }


def _register_websocket_endpoints(app: FastAPI) -> None:
    """Register WebSocket endpoints."""
    from prsm.interface.api.websocket import (
        websocket_manager,
        handle_websocket_message,
        handle_conversation_message,
        stream_ai_response
    )

    @app.websocket("/ws/{user_id}")
    async def websocket_endpoint(websocket: WebSocket, user_id: str):
        """Main WebSocket endpoint for real-time communication"""
        from prsm.interface.api.websocket_auth import (
            authenticate_websocket_connection,
            cleanup_websocket_connection,
            WebSocketAuthError
        )
        from prsm.core.security import validate_websocket_message

        try:
            connection = await authenticate_websocket_connection(websocket, user_id, "general")
            await websocket.accept()
            await websocket_manager.connect(websocket, user_id, "general")

            logger.info("Secure WebSocket connection established",
                       user_id=user_id,
                       username=connection.username,
                       role=connection.role.value,
                       ip_address=connection.ip_address)

            while True:
                data = await websocket.receive_text()

                try:
                    await validate_websocket_message(websocket, data, user_id)
                except Exception as e:
                    logger.warning("WebSocket message validation failed",
                                 user_id=user_id, error=str(e))
                    await websocket.close(code=1008, reason="Message validation failed")
                    return

                message = json.loads(data)
                await handle_websocket_message(websocket, user_id, message, connection)

        except WebSocketAuthError as e:
            logger.warning("WebSocket authentication failed",
                          user_id=user_id, error=e.message, code=e.code)
            await websocket.close(code=e.code, reason=e.message)
            return

        except WebSocketDisconnect:
            await websocket_manager.disconnect(websocket)
            await cleanup_websocket_connection(websocket)

        except Exception as e:
            logger.error("WebSocket error", user_id=user_id, error=str(e))
            await websocket_manager.disconnect(websocket)
            await cleanup_websocket_connection(websocket)

    @app.websocket("/ws/conversation/{user_id}/{conversation_id}")
    async def conversation_websocket(
        websocket: WebSocket,
        user_id: str,
        conversation_id: str
    ):
        """Conversation-specific WebSocket for streaming AI responses"""
        from prsm.interface.api.websocket_auth import (
            authenticate_websocket_connection,
            cleanup_websocket_connection,
            WebSocketAuthError
        )
        from prsm.core.security import validate_websocket_message

        try:
            connection = await authenticate_websocket_connection(
                websocket, user_id, "conversation", conversation_id
            )
            await websocket.accept()
            await websocket_manager.connect(websocket, user_id, "conversation")
            await websocket_manager.subscribe_to_conversation(websocket, conversation_id)

            logger.info("Secure conversation WebSocket established",
                       user_id=user_id,
                       conversation_id=conversation_id,
                       username=connection.username)

            while True:
                data = await websocket.receive_text()

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
                await handle_conversation_message(
                    websocket, user_id, conversation_id, message, connection
                )

        except WebSocketAuthError as e:
            logger.warning("Conversation WebSocket authentication failed",
                          user_id=user_id,
                          conversation_id=conversation_id,
                          error=e.message)
            await websocket.close(code=e.code, reason=e.message)
            return

        except WebSocketDisconnect:
            await websocket_manager.disconnect(websocket)
            await cleanup_websocket_connection(websocket)

        except Exception as e:
            logger.error("Conversation WebSocket error",
                        user_id=user_id,
                        conversation_id=conversation_id,
                        error=str(e))
            await websocket_manager.disconnect(websocket)
            await cleanup_websocket_connection(websocket)

    @app.get("/ws/stats")
    async def get_websocket_stats(
        user_id: str = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        from prsm.interface.api.websocket_auth import websocket_auth
        from prsm.core.auth.auth_manager import auth_manager

        try:
            user = await auth_manager.get_user_by_id(user_id)
            if not user or user.role not in ["admin", "moderator"]:
                raise HTTPException(
                    status_code=403,
                    detail="Admin permissions required to view WebSocket statistics"
                )

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
            raise HTTPException(status_code=500, detail="Failed to retrieve WebSocket statistics")


def _register_notification_endpoints(app: FastAPI) -> None:
    """Register notification system endpoints."""
    from prsm.interface.api.websocket import websocket_manager, stream_ai_response

    @app.post("/ui/conversations/{conversation_id}/messages/streaming")
    async def send_streaming_message(
        conversation_id: str,
        message_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send message with WebSocket streaming response"""
        try:
            if "content" not in message_data:
                raise HTTPException(status_code=400, detail="Missing message content")

            user_id = message_data.get("user_id", "anonymous")
            content = message_data["content"]
            message_id = str(uuid4())

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
            raise HTTPException(status_code=500, detail="Failed to send streaming message")

    @app.post("/notifications/send")
    async def send_notification(notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send real-time notification to users"""
        try:
            notification_type = notification_data.get("type", "general")
            target = notification_data.get("target", "all")
            message = notification_data.get("message", "")

            notification = {
                "type": "notification",
                "notification_type": notification_type,
                "message": message,
                "timestamp": asyncio.get_event_loop().time(),
                "data": notification_data.get("data", {})
            }

            if target == "all":
                await websocket_manager.broadcast_to_all(notification)
                logger.info("Notification broadcast to all users", type=notification_type)
            elif target == "user" and "user_id" in notification_data:
                user_id = notification_data["user_id"]
                await websocket_manager.send_to_user(notification, user_id)
                logger.info("Notification sent to user", user_id=user_id, type=notification_type)
            else:
                raise HTTPException(status_code=400, detail="Invalid notification target")

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
            raise HTTPException(status_code=500, detail="Failed to send notification")
