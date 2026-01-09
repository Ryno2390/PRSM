"""
CDN API Endpoints
================

REST API endpoints for PRSM CDN node registration, content serving,
and performance monitoring. Enables external nodes to participate in
the decentralized content delivery network.

Key Features:
- Node registration and capability discovery
- Content serving with automatic routing
- Performance metrics and monitoring
- FTNS reward distribution
- Real-time status updates
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import aiohttp

from ..infrastructure.cdn_layer import (
    prsm_cdn, NodeType, ContentPriority, BandwidthMetrics, 
    GeographicLocation, CDNNode, RequestRoute
)
from ..infrastructure.ipfs_cdn_bridge import ipfs_cdn_bridge
from ..infrastructure.sybil_resistance import sybil_resistance, ChallengeType, ValidationStatus


router = APIRouter(prefix="/cdn", tags=["CDN"])


# Request/Response Models
class NodeRegistrationRequest(BaseModel):
    """Request to register a new CDN node"""
    node_type: NodeType
    operator_id: UUID
    storage_capacity_gb: float = Field(gt=0, description="Available storage in GB")
    
    # Geographic location
    continent: str
    country: str
    region: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    
    # Bandwidth capabilities
    download_mbps: float = Field(gt=0, description="Download bandwidth in Mbps")
    upload_mbps: float = Field(gt=0, description="Upload bandwidth in Mbps")
    latency_ms: float = Field(gt=0, description="Average latency in ms")
    packet_loss_rate: float = Field(ge=0, le=1, description="Packet loss rate (0-1)")
    uptime_percentage: float = Field(ge=0, le=1, description="Historical uptime (0-1)")
    
    # Optional institutional voucher
    institutional_voucher: Optional[str] = None


class NodeRegistrationResponse(BaseModel):
    """Response for successful node registration"""
    node_id: UUID
    validation_status: ValidationStatus
    challenges_issued: List[UUID]
    estimated_earnings_ftns_per_day: Decimal
    next_challenge_deadline: datetime


class ContentRequest(BaseModel):
    """Request to serve content through CDN"""
    content_hash: str
    client_location: GeographicLocation
    performance_requirements: Optional[Dict[str, Any]] = None


class ContentResponse(BaseModel):
    """Response with content routing information"""
    content_hash: str
    primary_node: UUID
    fallback_nodes: List[UUID]
    estimated_latency_ms: float
    estimated_bandwidth_mbps: float
    ftns_cost: Decimal
    cache_probability: float


class PerformanceReport(BaseModel):
    """Performance report from serving content"""
    request_id: UUID
    node_id: UUID
    content_hash: str
    bytes_served: int
    actual_latency_ms: float
    success: bool
    error_message: Optional[str] = None


class NodeStatus(BaseModel):
    """Current status of a CDN node"""
    node_id: UUID
    is_active: bool
    total_requests_served: int
    total_bytes_served: int
    average_response_time_ms: float
    cache_hit_rate: float
    ftns_earned_total: Decimal
    ftns_earned_last_24h: Decimal
    last_challenge_date: Optional[datetime]
    validation_status: ValidationStatus
    reputation_score: float


class ChallengeResponse(BaseModel):
    """Response to a validation challenge"""
    challenge_id: UUID
    response_data: Dict[str, Any]


# API Endpoints

@router.post("/nodes/register", response_model=NodeRegistrationResponse)
async def register_cdn_node(request: NodeRegistrationRequest, background_tasks: BackgroundTasks):
    """
    Register a new node in the PRSM CDN network.
    """
    try:
        # Check if this is an institutional participant
        institutional_integration = None
        from ..institutional.gateway import institutional_gateway
        
        # Look for existing institutional participant
        for participant in institutional_gateway.participants.values():
            if participant.participant_id == request.operator_id:
                institutional_integration = await institutional_gateway.integrate_with_cdn_infrastructure(
                    participant_id=request.operator_id
                )
                break
        
        # If not institutional, proceed with standard registration
        if not institutional_integration:
            # Create geographic location
            location = GeographicLocation(
                continent=request.continent,
                country=request.country,
                region=request.region,
                latitude=request.latitude,
                longitude=request.longitude
            )
            
            # Create bandwidth metrics
            bandwidth_metrics = BandwidthMetrics(
                download_mbps=request.download_mbps,
                upload_mbps=request.upload_mbps,
                latency_ms=request.latency_ms,
                packet_loss_rate=request.packet_loss_rate,
                uptime_percentage=request.uptime_percentage,
                geographic_location=location
            )
            
            # Register with CDN layer
            node = await prsm_cdn.register_cdn_node(
                node_type=request.node_type,
                operator_id=request.operator_id,
                storage_capacity_gb=request.storage_capacity_gb,
                bandwidth_metrics=bandwidth_metrics
            )
            
            # Validate with sybil resistance
            claimed_capabilities = {
                "bandwidth_mbps": request.download_mbps,
                "storage_gb": request.storage_capacity_gb,
                "geographic_location": f"{request.continent},{request.country},{request.region}"
            }
            
            validation_status = await sybil_resistance.validate_new_node(
                node_id=node.node_id,
                claimed_capabilities=claimed_capabilities,
                institutional_voucher=request.institutional_voucher
            )
            
            node_id = node.node_id
            estimated_earnings = Decimal('0.5')  # Base estimate per day
            if request.node_type == NodeType.RESEARCH_INSTITUTION:
                estimated_earnings *= Decimal('3')
            elif request.node_type == NodeType.ENTERPRISE_GATEWAY:
                estimated_earnings *= Decimal('2')
        else:
            # Use institutional integration results
            node_id = institutional_integration["cdn_node_id"]
            validation_status = institutional_integration["validation_status"]
            estimated_earnings = Decimal(str(institutional_integration["estimated_monthly_ftns_earnings"] / 30))  # Convert monthly to daily
        
        # Get issued challenges
        challenges_issued = [
            challenge.challenge_id 
            for challenge in sybil_resistance.active_challenges.values()
            if challenge.node_id == node_id
        ]
        
        # Find next challenge deadline
        next_deadline = None
        if challenges_issued:
            active_challenges = [
                sybil_resistance.active_challenges[cid] 
                for cid in challenges_issued
                if cid in sybil_resistance.active_challenges
            ]
            if active_challenges:
                next_deadline = min(c.response_deadline for c in active_challenges)
        
        # Schedule periodic validation
        background_tasks.add_task(_schedule_periodic_validation, node_id)
        
        response = NodeRegistrationResponse(
            node_id=node_id,
            validation_status=validation_status,
            challenges_issued=challenges_issued,
            estimated_earnings_ftns_per_day=estimated_earnings,
            next_challenge_deadline=next_deadline or datetime.now(timezone.utc) + timedelta(hours=24)
        )
        
        # Add institutional integration info if applicable
        if institutional_integration:
            print(f"🏛️ Institutional CDN integration completed")
            print(f"   - Tier-based earnings multiplier applied")
            print(f"   - Enterprise SLA guarantees active")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Node registration failed: {str(e)}")


@router.get("/nodes/{node_id}/status", response_model=NodeStatus)
async def get_node_status(node_id: UUID = Path(..., description="Node ID")):
    """
    Get current status and performance metrics for a CDN node.
    """
    if node_id not in prsm_cdn.cdn_nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = prsm_cdn.cdn_nodes[node_id]
    
    # Get reputation score
    reputation_score = await sybil_resistance.get_node_trust_score(node_id)
    
    # Get validation status
    node_reputation = sybil_resistance.node_reputations.get(node_id)
    validation_status = node_reputation.validation_status if node_reputation else ValidationStatus.UNVERIFIED
    last_challenge_date = node_reputation.last_challenge_date if node_reputation else None
    
    # Calculate 24h earnings
    ftns_24h = node.ftns_earned_last_period  # Reset daily in production
    
    return NodeStatus(
        node_id=node_id,
        is_active=node.is_active,
        total_requests_served=node.total_requests_served,
        total_bytes_served=node.total_bytes_served,
        average_response_time_ms=node.average_response_time_ms,
        cache_hit_rate=node.cache_hit_rate,
        ftns_earned_total=node.ftns_earned_total,
        ftns_earned_last_24h=ftns_24h,
        last_challenge_date=last_challenge_date,
        validation_status=validation_status,
        reputation_score=reputation_score
    )


@router.post("/content/request", response_model=ContentResponse)
async def request_content(request: ContentRequest):
    """
    Request optimal routing for content delivery.
    """
    try:
        route = await prsm_cdn.optimize_content_routing(
            content_hash=request.content_hash,
            client_location=request.client_location,
            performance_requirements=request.performance_requirements
        )
        
        return ContentResponse(
            content_hash=request.content_hash,
            primary_node=route.primary_node,
            fallback_nodes=route.fallback_nodes,
            estimated_latency_ms=route.estimated_latency_ms,
            estimated_bandwidth_mbps=route.estimated_bandwidth_mbps,
            ftns_cost=route.ftns_cost,
            cache_probability=route.cache_probability
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content request failed: {str(e)}")


@router.get("/content/{content_hash}")
async def serve_content(content_hash: str, node_id: UUID = Query(..., description="Serving node ID")):
    """
    Serve content directly through the CDN.
    """
    try:
        # Verify node exists and is active
        if node_id not in prsm_cdn.cdn_nodes or not prsm_cdn.cdn_nodes[node_id].is_active:
            raise HTTPException(status_code=404, detail="Node not found or inactive")
        
        # Retrieve content from IPFS
        content = await ipfs_cdn_bridge.retrieve_content(content_hash, node_id)
        
        if content is None:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Stream content response
        def content_stream():
            chunk_size = 8192
            for i in range(0, len(content), chunk_size):
                yield content[i:i + chunk_size]
        
        return StreamingResponse(
            content_stream(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(len(content)),
                "Content-Hash": content_hash,
                "Served-By": str(node_id)
            }
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Content serving failed: {str(e)}")


@router.post("/performance/report")
async def report_performance(report: PerformanceReport, background_tasks: BackgroundTasks):
    """
    Report performance metrics for content serving and distribute rewards.
    """
    try:
        # Verify the request and node
        if report.node_id not in prsm_cdn.cdn_nodes:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Create a mock route for reward calculation
        route = RequestRoute(
            content_hash=report.content_hash,
            client_location=GeographicLocation(
                continent="unknown", country="unknown", region="unknown",
                latitude=0.0, longitude=0.0
            ),
            primary_node=report.node_id,
            fallback_nodes=[],
            estimated_latency_ms=report.actual_latency_ms * 1.1,  # Assume slightly higher estimate
            estimated_bandwidth_mbps=100.0,  # Default estimate
            cache_probability=0.8,
            ftns_cost=Decimal('0.01'),
            node_rewards={report.node_id: Decimal('0.01')}
        )
        
        if report.success:
            # Record successful serving and distribute rewards
            result = await prsm_cdn.serve_content_request(
                route=route,
                actual_bytes_served=report.bytes_served,
                actual_latency_ms=report.actual_latency_ms
            )
            
            # Schedule optimization tasks
            background_tasks.add_task(_optimize_caching)
            
            return {"status": "success", "ftns_awarded": result["ftns_reward"]}
        else:
            # Handle failed request
            return {"status": "failure", "message": report.error_message}
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Performance report failed: {str(e)}")


@router.get("/challenges/{node_id}")
async def get_pending_challenges(node_id: UUID = Path(..., description="Node ID")):
    """
    Get pending validation challenges for a node.
    """
    pending_challenges = []
    
    for challenge in sybil_resistance.active_challenges.values():
        if challenge.node_id == node_id:
            pending_challenges.append({
                "challenge_id": challenge.challenge_id,
                "challenge_type": challenge.challenge_type,
                "challenge_data": challenge.challenge_data,
                "response_deadline": challenge.response_deadline,
                "issued_at": challenge.issued_at
            })
    
    return {"node_id": node_id, "pending_challenges": pending_challenges}


@router.post("/challenges/respond")
async def respond_to_challenge(response: ChallengeResponse):
    """
    Submit response to a validation challenge.
    """
    try:
        success = await sybil_resistance.submit_challenge_response(
            challenge_id=response.challenge_id,
            response_data=response.response_data
        )
        
        return {
            "challenge_id": response.challenge_id,
            "validation_successful": success,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Challenge response failed: {str(e)}")


@router.get("/network/health")
async def get_network_health():
    """
    Get comprehensive CDN network health metrics.
    """
    try:
        cdn_health = await prsm_cdn.get_network_health()
        ipfs_health = await ipfs_cdn_bridge.get_ipfs_health_metrics()
        sybil_validation = await sybil_resistance.periodic_validation_sweep()
        
        return {
            "timestamp": datetime.now(timezone.utc),
            "cdn_metrics": cdn_health,
            "ipfs_metrics": ipfs_health,
            "validation_metrics": sybil_validation,
            "overall_status": "healthy"  # Could be computed based on metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/content/pin")
async def pin_content(
    content_hash: str = Query(..., description="IPFS content hash"),
    priority: ContentPriority = Query(ContentPriority.NORMAL, description="Content priority"),
    metadata: Dict[str, Any] = None
):
    """
    Pin content to IPFS nodes and register with CDN.
    """
    try:
        success = await ipfs_cdn_bridge.pin_content(
            content_hash=content_hash,
            priority=priority,
            metadata=metadata or {}
        )
        
        if success:
            return {
                "content_hash": content_hash,
                "priority": priority,
                "status": "pinned",
                "timestamp": datetime.now(timezone.utc)
            }
        else:
            raise HTTPException(status_code=500, detail="Content pinning failed")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pin content failed: {str(e)}")


@router.get("/optimize/caching")
async def optimize_caching(background_tasks: BackgroundTasks):
    """
    Trigger caching optimization across the network.
    """
    background_tasks.add_task(_optimize_caching)
    return {
        "status": "optimization_scheduled",
        "timestamp": datetime.now(timezone.utc)
    }


# Background task functions
async def _schedule_periodic_validation(node_id: UUID):
    """Schedule periodic validation for a node"""
    await asyncio.sleep(3600)  # Wait 1 hour
    try:
        await sybil_resistance.issue_challenge(
            node_id=node_id,
            challenge_type=ChallengeType.UPTIME_CHECK
        )
    except Exception as e:
        print(f"❌ Periodic validation failed for {node_id}: {e}")


async def _optimize_caching():
    """Optimize content caching across the network"""
    try:
        cdn_optimization = await prsm_cdn.optimize_content_caching()
        ipfs_optimization = await ipfs_cdn_bridge.optimize_pinning_strategy()
        print(f"🔧 Caching optimization completed")
        print(f"   - CDN optimizations: {len(cdn_optimization.get('new_replications', []))}")
        print(f"   - IPFS optimizations: {len(ipfs_optimization.get('pins_added', []))}")
    except Exception as e:
        print(f"❌ Caching optimization failed: {e}")