"""
PRSM IPFS API Router
Handles all IPFS (Distributed Storage) related endpoints
"""

from typing import Dict, Any

import structlog
from fastapi import APIRouter, HTTPException

from prsm.core.ipfs_client import get_ipfs_client

# Initialize router
router = APIRouter()
logger = structlog.get_logger(__name__)

@router.post("/upload")
async def upload_to_ipfs(upload_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload content to IPFS distributed storage
    
    🌐 DISTRIBUTED STORAGE:
    Stores content on IPFS network with automatic pinning
    and comprehensive metadata generation
    """
    try:
        ipfs_client = get_ipfs_client()
        
        if not ipfs_client:
            raise HTTPException(
                status_code=503,
                detail="IPFS service not available"
            )
        
        # Validate required fields
        content = upload_request.get("content")
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Content is required"
            )
        
        filename = upload_request.get("filename", "unnamed_file")
        content_type = upload_request.get("content_type", "application/octet-stream")
        pin = upload_request.get("pin", True)
        
        # Upload to IPFS
        result = await ipfs_client.upload_content(
            content=content,
            filename=filename,
            content_type=content_type,
            pin=pin
        )
        
        if result.success:
            logger.info("Content uploaded to IPFS",
                       cid=result.cid,
                       filename=filename,
                       size=result.size)
            
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
            
    except Exception as e:
        logger.error("IPFS upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="IPFS upload failed"
        )


@router.get("/{cid}")
async def download_from_ipfs(cid: str, download: bool = False) -> Dict[str, Any]:
    """
    Download or retrieve content from IPFS
    
    📥 DISTRIBUTED RETRIEVAL:
    Retrieves content from IPFS network with automatic
    caching and metadata extraction
    """
    try:
        ipfs_client = get_ipfs_client()
        
        if not ipfs_client:
            raise HTTPException(
                status_code=503,
                detail="IPFS service not available"
            )
        
        # Retrieve from IPFS
        result = await ipfs_client.retrieve_content(cid)
        
        if result.success:
            logger.info("Content retrieved from IPFS",
                       cid=cid,
                       size=result.size if hasattr(result, 'size') else None)
            
            return {
                "success": True,
                "cid": cid,
                "content": result.content if not download else None,
                "metadata": result.metadata if hasattr(result, 'metadata') else None,
                "size": result.size if hasattr(result, 'size') else None,
                "content_type": result.content_type if hasattr(result, 'content_type') else None,
                "download_url": f"https://ipfs.io/ipfs/{cid}" if download else None
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Content not found: {result.error if hasattr(result, 'error') else 'Unknown error'}"
            )
            
    except Exception as e:
        logger.error("IPFS download failed", error=str(e), cid=cid)
        raise HTTPException(
            status_code=500,
            detail="IPFS download failed"
        )


@router.post("/models/upload")
async def upload_model_to_ipfs(model_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload trained model to IPFS with metadata
    
    🤖 MODEL DISTRIBUTION:
    Stores ML models on IPFS with comprehensive metadata
    including training parameters, performance metrics, and versioning
    """
    try:
        ipfs_client = get_ipfs_client()
        
        if not ipfs_client:
            raise HTTPException(
                status_code=503,
                detail="IPFS service not available"
            )
        
        # Validate model data
        model_data = model_request.get("model_data")
        if not model_data:
            raise HTTPException(
                status_code=400,
                detail="Model data is required"
            )
        
        # Prepare model metadata
        model_metadata = {
            "model_name": model_request.get("name", "unnamed_model"),
            "model_type": model_request.get("type", "unknown"),
            "framework": model_request.get("framework", "unknown"),
            "version": model_request.get("version", "1.0.0"),
            "description": model_request.get("description", ""),
            "training_data": model_request.get("training_data", {}),
            "performance_metrics": model_request.get("metrics", {}),
            "parameters": model_request.get("parameters", {}),
            "uploaded_at": model_request.get("uploaded_at"),
            "creator": model_request.get("creator", "anonymous")
        }
        
        # Upload model to IPFS
        result = await ipfs_client.upload_model(
            model_data=model_data,
            metadata=model_metadata
        )
        
        if result.success:
            logger.info("Model uploaded to IPFS",
                       cid=result.cid,
                       model_name=model_metadata["model_name"])
            
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
            
    except Exception as e:
        logger.error("Model upload failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Model upload failed"
        )


@router.post("/research/publish")
async def publish_research_content(research_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Publish research content to IPFS
    
    📚 RESEARCH PUBLICATION:
    Publishes research papers, datasets, and academic content
    to IPFS with proper attribution and metadata
    """
    try:
        ipfs_client = get_ipfs_client()
        
        if not ipfs_client:
            raise HTTPException(
                status_code=503,
                detail="IPFS service not available"
            )
        
        # Validate research content
        content = research_request.get("content")
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Research content is required"
            )
        
        # Prepare research metadata
        research_metadata = {
            "title": research_request.get("title", "Untitled Research"),
            "authors": research_request.get("authors", []),
            "abstract": research_request.get("abstract", ""),
            "keywords": research_request.get("keywords", []),
            "publication_type": research_request.get("type", "paper"),
            "institution": research_request.get("institution", ""),
            "published_at": research_request.get("published_at"),
            "license": research_request.get("license", "CC-BY-4.0"),
            "doi": research_request.get("doi", ""),
            "research_area": research_request.get("research_area", ""),
            "funding": research_request.get("funding", [])
        }
        
        # Upload research content to IPFS
        result = await ipfs_client.upload_research(
            content=content,
            metadata=research_metadata
        )
        
        if result.success:
            logger.info("Research content published to IPFS",
                       cid=result.cid,
                       title=research_metadata["title"])
            
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
                detail=f"Research publication failed: {result.error}"
            )
            
    except Exception as e:
        logger.error("Research content publishing failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Research content publishing failed"
        )


@router.get("/status")
async def get_ipfs_status() -> Dict[str, Any]:
    """
    Get comprehensive IPFS network status
    
    📊 IPFS MONITORING:
    Returns detailed status of IPFS network connectivity,
    node information, and performance metrics
    """
    try:
        ipfs_client = get_ipfs_client()
        
        if not ipfs_client:
            return {
                "success": False,
                "status": "disconnected",
                "message": "IPFS service not available"
            }
        
        # Get IPFS node status
        status = await ipfs_client.get_status()
        
        logger.info("IPFS status retrieved",
                   node_id=status.get("node_id", "unknown"))
        
        return {
            "success": True,
            "status": "connected",
            "node_info": {
                "node_id": status.get("node_id", "unknown"),
                "version": status.get("version", "unknown"),
                "network": status.get("network", "unknown"),
                "connection_type": status.get("connection_type", "unknown")
            },
            "performance": {
                "peers_connected": status.get("peers", 0),
                "storage_used": status.get("storage_used", "unknown"),
                "bandwidth_in": status.get("bandwidth_in", "unknown"),
                "bandwidth_out": status.get("bandwidth_out", "unknown")
            },
            "capabilities": {
                "gateway_available": status.get("gateway", False),
                "pinning_service": status.get("pinning", False),
                "pubsub_enabled": status.get("pubsub", False),
                "experimental_features": status.get("experimental", [])
            }
        }
        
    except Exception as e:
        logger.error("Failed to get IPFS status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve IPFS status"
        )