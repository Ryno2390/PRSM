"""
Dashboard Server Application
============================

FastAPI-based web dashboard for PRSM researchers.
Provides real-time monitoring, job submission, and FTNS management.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import structlog
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    Request,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prsm.core.auth.jwt_handler import jwt_handler
from prsm.core.auth.auth_manager import AuthManager, AuthenticationError, security

logger = structlog.get_logger(__name__)


# ── Request/Response Models ─────────────────────────────────────────────────────

class JobSubmitRequest(BaseModel):
    """Request body for submitting a compute job."""
    job_type: str = Field(..., description="Type of job: inference, embedding, benchmark")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Job-specific payload")
    ftns_budget: float = Field(default=1.0, ge=0.01, description="FTNS budget for job")
    model_id: Optional[str] = Field(default=None, description="Model ID for inference jobs")
    input_data: Optional[str] = Field(default=None, description="Input data for the job")


class StakeRequest(BaseModel):
    """Request body for staking FTNS."""
    amount: float = Field(..., gt=0, description="Amount to stake")
    duration_epochs: int = Field(default=10, ge=1, description="Staking duration in epochs")


class TransferRequest(BaseModel):
    """Request body for transferring FTNS."""
    to_wallet: str = Field(..., description="Destination wallet ID")
    amount: float = Field(..., gt=0, description="Amount to transfer")
    description: Optional[str] = Field(default=None, description="Transfer description")


class LoginRequest(BaseModel):
    """Request body for dashboard login."""
    username: str
    password: str


class StatusResponse(BaseModel):
    """System status response."""
    status: str
    node_id: str
    uptime_seconds: float
    connected_peers: int
    ftns_balance: float
    active_jobs: int
    timestamp: str


# ── WebSocket Connection Manager ────────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket):
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("WebSocket connected", total_connections=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket."""
        self.active_connections.discard(websocket)
        logger.info("WebSocket disconnected", total_connections=len(self.active_connections))
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning("Failed to send personal message", error=str(e))
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)


# ── Dashboard Server ──────────────────────────────────────────────────────────────

class DashboardServer:
    """
    Web dashboard server for PRSM.
    
    Provides:
    - Real-time node status via WebSocket
    - Job submission interface
    - Transaction history
    - FTNS balance and staking views
    - Authentication integration
    """
    
    def __init__(self, node=None, host: str = "0.0.0.0", port: int = 8080):
        """
        Initialize the dashboard server.
        
        Args:
            node: PRSM node instance to monitor/control
            host: Host address to bind to
            port: Port to listen on
        """
        self.node = node
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="PRSM Dashboard",
            description="Web dashboard for PRSM researchers",
            version="1.0.0",
        )
        self.manager = ConnectionManager()
        self.auth_manager: Optional[AuthManager] = None
        self.start_time: Optional[datetime] = None
        self._server = None
        
        # Setup routes and middleware
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup all API routes for the dashboard."""
        
        # ── Authentication Endpoints ─────────────────────────────────────────────
        
        @self.app.post("/api/auth/login")
        async def login(request: LoginRequest):
            """Authenticate user and return JWT token."""
            if not self.auth_manager:
                # Fallback to node identity if auth manager not available
                if self.node and self.node.identity:
                    return {
                        "access_token": "demo-token",
                        "token_type": "bearer",
                        "user_id": self.node.identity.node_id,
                        "username": "researcher",
                    }
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Authentication not available"
                )
            
            try:
                # Use auth manager for authentication
                token_data = await self.auth_manager.authenticate_user(
                    username=request.username,
                    password=request.password,
                    client_info={"source": "dashboard"}
                )
                return {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "token_type": "bearer",
                    "user_id": token_data["user_id"],
                    "username": request.username,
                }
            except AuthenticationError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(e)
                )
        
        @self.app.post("/api/auth/logout")
        async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Logout user and invalidate token."""
            if self.auth_manager and credentials:
                await self.auth_manager.logout(credentials.credentials)
            return {"status": "logged_out"}
        
        @self.app.get("/api/auth/me")
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Get current authenticated user info."""
            if not self.auth_manager:
                # Fallback for demo mode
                if self.node and self.node.identity:
                    return {
                        "user_id": self.node.identity.node_id,
                        "username": "researcher",
                        "role": "user",
                    }
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            try:
                user = await self.auth_manager.get_current_user(credentials.credentials)
                return user
            except AuthenticationError:
                raise HTTPException(status_code=401, detail="Invalid token")
        
        # ── Status Endpoints ─────────────────────────────────────────────────────
        
        @self.app.get("/api/status")
        async def get_status():
            """Get overall system status."""
            if not self.node:
                return StatusResponse(
                    status="demo",
                    node_id="demo-node",
                    uptime_seconds=0,
                    connected_peers=0,
                    ftns_balance=1000.0,
                    active_jobs=0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ).dict()
            
            # Get real status from node
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
            connected_peers = len(self.node.transport.peers) if self.node.transport else 0
            balance = await self._get_balance()
            active_jobs = self._get_active_job_count()
            
            return {
                "status": "online",
                "node_id": self.node.identity.node_id if self.node.identity else "unknown",
                "uptime_seconds": uptime,
                "connected_peers": connected_peers,
                "ftns_balance": balance,
                "active_jobs": active_jobs,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        @self.app.get("/api/node")
        async def get_node_info():
            """Get detailed node information."""
            if not self.node:
                return {
                    "node_id": "demo-node",
                    "status": "demo",
                    "version": "1.0.0",
                    "roles": ["researcher"],
                    "address": "0.0.0.0:8080",
                }
            
            roles = []
            if self.node.compute_provider:
                roles.append("compute")
            if self.node.storage_provider:
                roles.append("storage")
            if self.node.agent_registry:
                roles.append("agent")
            
            return {
                "node_id": self.node.identity.node_id if self.node.identity else "unknown",
                "status": "online",
                "version": "1.0.0",
                "roles": roles or ["researcher"],
                "address": str(self.node.transport.address) if self.node.transport else "unknown",
                "peers_count": len(self.node.transport.peers) if self.node.transport else 0,
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0,
            }
        
        # ── Job Endpoints ─────────────────────────────────────────────────────────
        
        @self.app.get("/api/jobs")
        async def get_jobs(status_filter: Optional[str] = None, limit: int = 50):
            """Get compute jobs with optional status filter."""
            if not self.node or not self.node.compute_requester:
                return {"jobs": [], "count": 0}
            
            jobs = []
            for job_id, job in list(self.node.compute_requester.submitted_jobs.items())[:limit]:
                if status_filter and job.status.value != status_filter:
                    continue
                jobs.append({
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "job_type": job.job_type.value,
                    "ftns_budget": job.ftns_budget,
                    "provider_id": job.provider_id,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error": job.error,
                })
            
            return {"jobs": jobs, "count": len(jobs)}
        
        @self.app.post("/api/jobs/submit")
        async def submit_job(job: JobSubmitRequest):
            """Submit a compute job to the network."""
            if not self.node or not self.node.compute_requester:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Compute requester not initialized"
                )
            
            from prsm.node.compute_provider import JobType
            try:
                job_type = JobType(job.job_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid job type: {job.job_type}. Valid types: inference, embedding, benchmark"
                )
            
            try:
                submitted = await self.node.compute_requester.submit_job(
                    job_type=job_type,
                    payload=job.payload,
                    ftns_budget=job.ftns_budget,
                )
                
                # Broadcast job submission to WebSocket clients
                await self.manager.broadcast({
                    "type": "job_submitted",
                    "data": {
                        "job_id": submitted.job_id,
                        "status": submitted.status.value,
                        "job_type": submitted.job_type.value,
                    }
                })
                
                return {
                    "job_id": submitted.job_id,
                    "status": submitted.status.value,
                    "job_type": submitted.job_type.value,
                    "ftns_budget": submitted.ftns_budget,
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/jobs/{job_id}")
        async def get_job_status(job_id: str):
            """Get status of a specific job."""
            if not self.node or not self.node.compute_requester:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Compute requester not initialized"
                )
            
            job = self.node.compute_requester.submitted_jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return {
                "job_id": job.job_id,
                "status": job.status.value,
                "job_type": job.job_type.value,
                "ftns_budget": job.ftns_budget,
                "provider_id": job.provider_id,
                "result": job.result,
                "result_verified": job.result_verified,
                "error": job.error,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
        
        # ── FTNS Balance Endpoints ───────────────────────────────────────────────
        
        @self.app.get("/api/ftns/balance")
        async def get_ftns_balance():
            """Get FTNS balance and staking info."""
            balance = await self._get_balance()
            staked = await self._get_staked_balance()
            
            return {
                "available_balance": balance,
                "staked_balance": staked,
                "total_balance": balance + staked,
                "currency": "FTNS",
            }
        
        @self.app.get("/api/ftns/history")
        async def get_transaction_history(limit: int = 50):
            """Get transaction history."""
            if not self.node or not self.node.ledger or not self.node.identity:
                return {"transactions": [], "count": 0}
            
            history = await self.node.ledger.get_transaction_history(
                self.node.identity.node_id, limit=min(limit, 200)
            )
            
            transactions = [
                {
                    "tx_id": tx.tx_id,
                    "type": tx.tx_type.value,
                    "from": tx.from_wallet,
                    "to": tx.to_wallet,
                    "amount": tx.amount,
                    "description": tx.description,
                    "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
                }
                for tx in history
            ]
            
            return {"transactions": transactions, "count": len(transactions)}
        
        @self.app.post("/api/ftns/transfer")
        async def transfer_ftns(request: TransferRequest):
            """Transfer FTNS to another wallet."""
            if not self.node or not self.node.ledger_sync:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Ledger not initialized"
                )
            
            if request.amount <= 0:
                raise HTTPException(status_code=400, detail="Amount must be positive")
            
            tx = await self.node.ledger_sync.signed_transfer(
                to_wallet=request.to_wallet,
                amount=request.amount,
                description=request.description or f"Transfer to {request.to_wallet[:12]}...",
            )
            
            if not tx:
                raise HTTPException(status_code=400, detail="Insufficient balance")
            
            # Broadcast transfer to WebSocket clients
            await self.manager.broadcast({
                "type": "transfer",
                "data": {
                    "tx_id": tx.tx_id,
                    "amount": tx.amount,
                    "to": tx.to_wallet,
                }
            })
            
            return {
                "tx_id": tx.tx_id,
                "from": tx.from_wallet,
                "to": tx.to_wallet,
                "amount": tx.amount,
                "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
            }
        
        @self.app.post("/api/ftns/stake")
        async def stake_ftns(request: StakeRequest):
            """Stake FTNS tokens."""
            if not self.node or not self.node.ledger:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Ledger not initialized"
                )
            
            # Check if staking is implemented
            if hasattr(self.node.ledger, 'stake_tokens'):
                result = await self.node.ledger.stake_tokens(
                    wallet_id=self.node.identity.node_id,
                    amount=request.amount,
                    duration_epochs=request.duration_epochs,
                )
                return result
            else:
                # Demo response if staking not implemented
                return {
                    "status": "staked",
                    "amount": request.amount,
                    "duration_epochs": request.duration_epochs,
                    "message": "Staking simulated (not yet implemented on ledger)",
                }
        
        # ── Peer Network Endpoints ───────────────────────────────────────────────
        
        @self.app.get("/api/peers")
        async def get_peers():
            """Get connected and known peers."""
            if not self.node:
                return {"connected": [], "known": [], "connected_count": 0, "known_count": 0}
            
            connected = []
            if self.node.transport:
                for pid, peer in self.node.transport.peers.items():
                    connected.append({
                        "peer_id": pid,
                        "address": str(peer.address),
                        "display_name": peer.display_name,
                        "connected_at": peer.connected_at.isoformat() if peer.connected_at else None,
                        "last_seen": peer.last_seen.isoformat() if peer.last_seen else None,
                        "outbound": peer.outbound,
                    })
            
            known = []
            if self.node.discovery:
                for info in self.node.discovery.get_known_peers():
                    known.append({
                        "node_id": info.node_id,
                        "address": str(info.address),
                        "display_name": info.display_name,
                        "last_seen": info.last_seen.isoformat() if info.last_seen else None,
                    })
            
            return {
                "connected": connected,
                "known": known,
                "connected_count": len(connected),
                "known_count": len(known),
            }
        
        # ── Agent Endpoints ───────────────────────────────────────────────────────
        
        @self.app.get("/api/agents")
        async def list_agents(local_only: bool = False):
            """List known agents."""
            if not self.node or not self.node.agent_registry:
                return {"agents": [], "count": 0}
            
            agents = self.node.agent_registry.get_local_agents() if local_only else self.node.agent_registry.get_all_agents()
            
            return {
                "agents": [a.to_dict() for a in agents],
                "count": len(agents),
            }
        
        @self.app.get("/api/agents/{agent_id}")
        async def get_agent(agent_id: str):
            """Get agent details."""
            if not self.node or not self.node.agent_registry:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Agent registry not initialized"
                )
            
            record = self.node.agent_registry.lookup(agent_id)
            if not record:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            result = record.to_dict()
            if self.node.ledger:
                result["allowance"] = await self.node.ledger.get_agent_allowance(agent_id)
            return result
        
        # ── Content Endpoints ─────────────────────────────────────────────────────
        
        @self.app.get("/api/content/search")
        async def search_content(q: str = "", limit: int = 20):
            """Search the network content index."""
            if not self.node or not self.node.content_index:
                return {"query": q, "results": [], "count": 0}
            
            results = self.node.content_index.search(q, limit=min(limit, 100))
            
            return {
                "query": q,
                "results": [
                    {
                        "cid": r.cid,
                        "filename": r.filename,
                        "size_bytes": r.size_bytes,
                        "creator_id": r.creator_id,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in results
                ],
                "count": len(results),
            }
        
        # ── Teacher Endpoints ─────────────────────────────────────────────────────
        
        @self.app.get("/api/teacher/list")
        async def list_teachers():
            """List available teacher models."""
            if not self.node:
                return {"teachers": [], "count": 0}
            
            # Check if node has teacher registry
            teachers = []
            if hasattr(self.node, 'teacher_registry') and self.node.teacher_registry:
                for teacher_id, teacher in self.node.teacher_registry.items():
                    teachers.append({
                        "teacher_id": teacher_id,
                        "specialization": getattr(teacher, 'specialization', 'general'),
                        "domain": getattr(teacher, 'domain', 'unknown'),
                        "status": getattr(teacher, 'status', 'active'),
                        "created_at": getattr(teacher, 'created_at', None),
                    })
            
            return {"teachers": teachers, "count": len(teachers)}
        
        @self.app.post("/api/teacher/create")
        async def create_teacher(request: dict):
            """Create a new teacher model."""
            if not self.node:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Node not connected"
                )
            
            specialization = request.get("specialization", "general")
            domain = request.get("domain", "general")
            
            # Check if node supports teacher creation
            if hasattr(self.node, 'create_teacher'):
                teacher = await self.node.create_teacher(
                    specialization=specialization,
                    domain=domain,
                )
                return {
                    "teacher_id": teacher.teacher_id,
                    "specialization": specialization,
                    "domain": domain,
                    "status": "active",
                }
            else:
                # Demo response
                from uuid import uuid4
                return {
                    "teacher_id": f"teacher_{uuid4().hex[:12]}",
                    "specialization": specialization,
                    "domain": domain,
                    "status": "active",
                    "message": "Teacher created (demo mode)",
                }
        
        @self.app.get("/api/teacher/{teacher_id}")
        async def get_teacher(teacher_id: str):
            """Get teacher model details."""
            if not self.node:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Node not connected"
                )
            
            if hasattr(self.node, 'teacher_registry') and self.node.teacher_registry:
                teacher = self.node.teacher_registry.get(teacher_id)
                if teacher:
                    return {
                        "teacher_id": teacher_id,
                        "specialization": getattr(teacher, 'specialization', 'general'),
                        "domain": getattr(teacher, 'domain', 'unknown'),
                        "status": getattr(teacher, 'status', 'active'),
                    }
            
            raise HTTPException(status_code=404, detail="Teacher not found")
        
        # ── Distillation Endpoints ────────────────────────────────────────────────
        
        @self.app.get("/api/distillation")
        async def list_distillation_jobs():
            """List distillation jobs."""
            if not self.node:
                return {"jobs": [], "count": 0}
            
            jobs = []
            # Check if node has distillation tracking
            if hasattr(self.node, 'distillation_jobs'):
                for job_id, job in self.node.distillation_jobs.items():
                    jobs.append({
                        "job_id": job_id,
                        "teacher_id": getattr(job, 'teacher_id', None),
                        "status": getattr(job, 'status', 'unknown'),
                        "progress": getattr(job, 'progress', 0),
                        "created_at": getattr(job, 'created_at', None),
                    })
            
            return {"jobs": jobs, "count": len(jobs)}
        
        @self.app.post("/api/distillation/submit")
        async def submit_distillation(request: dict):
            """Submit a distillation job."""
            if not self.node:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Node not connected"
                )
            
            teacher_id = request.get("teacher_id")
            ftns_budget = request.get("ftns_budget", 1.0)
            dataset_cid = request.get("dataset_cid")
            
            if not teacher_id:
                raise HTTPException(status_code=400, detail="teacher_id is required")
            
            # Check if node supports distillation
            if hasattr(self.node, 'submit_distillation'):
                job = await self.node.submit_distillation(
                    teacher_id=teacher_id,
                    ftns_budget=ftns_budget,
                    dataset_cid=dataset_cid,
                )
                return {
                    "job_id": job.job_id,
                    "teacher_id": teacher_id,
                    "status": "pending",
                }
            else:
                # Demo response
                from uuid import uuid4
                return {
                    "job_id": f"distill_{uuid4().hex[:12]}",
                    "teacher_id": teacher_id,
                    "status": "pending",
                    "message": "Distillation job submitted (demo mode)",
                }
        
        @self.app.get("/api/distillation/{job_id}")
        async def get_distillation_job(job_id: str):
            """Get distillation job status."""
            if not self.node:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Node not connected"
                )
            
            if hasattr(self.node, 'distillation_jobs'):
                job = self.node.distillation_jobs.get(job_id)
                if job:
                    return {
                        "job_id": job_id,
                        "teacher_id": getattr(job, 'teacher_id', None),
                        "status": getattr(job, 'status', 'unknown'),
                        "progress": getattr(job, 'progress', 0),
                        "created_at": getattr(job, 'created_at', None),
                    }
            
            raise HTTPException(status_code=404, detail="Distillation job not found")
        
        # ── Health Endpoint ───────────────────────────────────────────────────────
        
        @self.app.get("/api/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "node_id": self.node.identity.node_id if self.node and self.node.identity else "demo",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        # ── WebSocket Endpoint ────────────────────────────────────────────────────
        
        @self.app.websocket("/ws/status")
        async def websocket_status(websocket: WebSocket):
            """WebSocket endpoint for real-time status updates."""
            await self.manager.connect(websocket)
            try:
                # Send initial status
                status = await get_status()
                await self.manager.send_personal_message(
                    {"type": "status_update", "data": status},
                    websocket
                )
                
                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                        
                        # Handle ping/pong
                        if data.get("type") == "ping":
                            await self.manager.send_personal_message(
                                {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()},
                                websocket
                            )
                        # Handle status request
                        elif data.get("type") == "get_status":
                            status = await get_status()
                            await self.manager.send_personal_message(
                                {"type": "status_update", "data": status},
                                websocket
                            )
                    except asyncio.TimeoutError:
                        # Send periodic status update
                        status = await get_status()
                        await self.manager.send_personal_message(
                            {"type": "status_update", "data": status},
                            websocket
                        )
                        
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception as e:
                logger.error("WebSocket error", error=str(e))
                self.manager.disconnect(websocket)
        
        # ── Dashboard HTML Route ──────────────────────────────────────────────────
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_root(request: Request):
            """Serve the main dashboard page."""
            # Get the templates directory
            templates_dir = Path(__file__).parent / "templates"
            if templates_dir.exists():
                templates = Jinja2Templates(directory=str(templates_dir))
                return templates.TemplateResponse("dashboard.html", {"request": request})
            else:
                # Fallback to static HTML
                static_dir = Path(__file__).parent / "static"
                index_path = static_dir / "index.html"
                if index_path.exists():
                    return HTMLResponse(content=index_path.read_text())
                else:
                    return HTMLResponse(content=self._get_default_html())
    
    async def _get_balance(self) -> float:
        """Get current FTNS balance."""
        if not self.node or not self.node.ledger or not self.node.identity:
            return 1000.0  # Demo balance
        return await self.node.ledger.get_balance(self.node.identity.node_id)
    
    async def _get_staked_balance(self) -> float:
        """Get staked FTNS balance."""
        if not self.node or not self.node.ledger or not self.node.identity:
            return 0.0
        # Check if staking is implemented
        if hasattr(self.node.ledger, 'get_staked_balance'):
            return await self.node.ledger.get_staked_balance(self.node.identity.node_id)
        return 0.0
    
    def _get_active_job_count(self) -> int:
        """Get count of active jobs."""
        if not self.node or not self.node.compute_requester:
            return 0
        count = 0
        for job in self.node.compute_requester.submitted_jobs.values():
            if job.status.value in ["pending", "running"]:
                count += 1
        return count
    
    def _get_default_html(self) -> str:
        """Get default HTML for dashboard when template not found."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>PRSM Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>PRSM Dashboard</h1>
    <p>Dashboard is running. Please ensure static files are properly configured.</p>
    <p>API endpoints available at /api/*</p>
</body>
</html>
        """
    
    async def start(self):
        """Start the dashboard server."""
        import uvicorn
        
        self.start_time = datetime.now(timezone.utc)
        
        # Initialize auth manager if available
        try:
            self.auth_manager = AuthManager()
            await self.auth_manager.initialize()
        except Exception as e:
            logger.warning("Auth manager initialization failed, using demo mode", error=str(e))
            self.auth_manager = None
        
        # Mount static files
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        
        logger.info("Starting dashboard server", host=self.host, port=self.port)
        await self._server.serve()
    
    async def stop(self):
        """Stop the dashboard server."""
        if self._server:
            self._server.should_exit = True
            logger.info("Dashboard server stopped")


def create_dashboard_app(node=None) -> FastAPI:
    """
    Factory function to create a dashboard FastAPI app.
    
    Args:
        node: PRSM node instance to monitor/control
        
    Returns:
        Configured FastAPI application
    """
    server = DashboardServer(node=node)
    return server.app