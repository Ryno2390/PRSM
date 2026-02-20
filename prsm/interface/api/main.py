"""
PRSM FastAPI Application
========================

Main API entry point for PRSM.

This module provides backward compatibility while using the modular
app_factory pattern internally. The application is created using
create_app() which handles all initialization.

Architecture:
- app_factory.py: Application factory and configuration
- lifecycle/: Startup and shutdown sequences
- router_registry.py: Centralized router management
- middleware.py: Middleware stack configuration
- core_endpoints.py: Root, health, and core API endpoints
- websocket/: WebSocket connection management

For new deployments, you can use the factory directly:
    from prsm.interface.api.app_factory import create_app
    app = create_app()
"""

import os
import structlog
from typing import Dict, Any

from prsm.interface.api.app_factory import create_app
from prsm.core.config import get_settings

# Configure structured logging
logger = structlog.get_logger(__name__)
settings = get_settings()

# Create the FastAPI application using the factory pattern
# This handles all initialization:
# - Middleware configuration
# - Router registration
# - Core endpoint registration
# - Exception handlers
# - Lifespan management (startup/shutdown)
app = create_app()

# Re-export WebSocket manager for backward compatibility
from prsm.interface.api.websocket import websocket_manager

# Log application creation
logger.info(
    "PRSM API application initialized",
    environment=settings.environment.value if settings else "unknown",
    version="0.1.0"
)


# === Legacy Endpoints ===
# The following endpoints are kept here for backward compatibility
# and will be migrated to separate routers in future versions.

@app.post("/ftns/transactions")
async def create_ftns_transaction(
    transaction_data: Dict[str, Any],
    current_user: str = None
) -> Dict[str, Any]:
    """
    Create a new FTNS transaction with atomic guarantees
    
    Security features:
    - Idempotency key prevents duplicate transactions
    - Atomic balance updates prevent race conditions
    - Proper error handling for insufficient balance
    
    Required fields:
    - to_user: Recipient user ID
    - amount: Amount to transfer (positive number)
    - transaction_type: 'transfer', 'reward', 'charge', etc.
    - description: Transaction description
    
    Optional fields:
    - idempotency_key: Unique key for duplicate prevention (auto-generated if not provided)
    - from_user: Sender user ID (defaults to current_user or 'system' for rewards)
    """
    from uuid import uuid4
    from fastapi import HTTPException
    from prsm.core.database import FTNSQueries
    import hashlib
    
    try:
        required_fields = ["to_user", "amount", "transaction_type", "description"]
        for field in required_fields:
            if field not in transaction_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        amount = float(transaction_data["amount"])
        if amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="Amount must be positive"
            )
        
        to_user = transaction_data["to_user"]
        transaction_type = transaction_data["transaction_type"]
        description = transaction_data["description"]
        
        from_user = transaction_data.get("from_user") or current_user or "system"
        
        idempotency_key = transaction_data.get("idempotency_key")
        if not idempotency_key:
            key_components = f"{from_user}:{to_user}:{amount}:{transaction_type}:{uuid4().hex}"
            idempotency_key = f"ftns:{hashlib.sha256(key_components.encode()).hexdigest()[:32]}"
        
        if transaction_type == "transfer":
            result = await FTNSQueries.execute_atomic_transfer(
                from_user_id=from_user,
                to_user_id=to_user,
                amount=amount,
                idempotency_key=idempotency_key,
                description=description
            )
        elif transaction_type == "reward":
            result = await FTNSQueries.execute_atomic_deduct(
                user_id=from_user if from_user != "system" else "system_mint",
                amount=-amount,
                idempotency_key=idempotency_key,
                description=description,
                transaction_type="reward"
            )
            if result["success"]:
                deduct_result = await FTNSQueries.execute_atomic_deduct(
                    user_id=to_user,
                    amount=-amount,
                    idempotency_key=f"{idempotency_key}:credit",
                    description=description,
                    transaction_type="reward_credit"
                )
                result = deduct_result
        else:
            result = await FTNSQueries.execute_atomic_deduct(
                user_id=from_user,
                amount=amount,
                idempotency_key=idempotency_key,
                description=description,
                transaction_type=transaction_type
            )
        
        if not result["success"]:
            error_msg = result.get("error_message", "Transaction failed")
            if "Insufficient balance" in error_msg:
                raise HTTPException(status_code=402, detail=error_msg)
            elif "Duplicate request" in error_msg or "idempotency" in error_msg.lower():
                return {
                    "transaction_id": result.get("transaction_id"),
                    "status": "completed",
                    "idempotent_replay": True
                }
            else:
                raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info("FTNS transaction created",
                   transaction_id=result["transaction_id"],
                   transaction_type=transaction_type,
                   amount=amount,
                   idempotency_key=idempotency_key)
        
        return {
            "transaction_id": str(result["transaction_id"]),
            "status": "completed",
            "new_balance": result.get("new_balance"),
            "idempotency_key": idempotency_key
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
    current_user: str = None
) -> Dict[str, str]:
    """Cache model output for performance optimization"""
    from fastapi import HTTPException
    from prsm.core.redis_client import get_model_cache

    try:
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

        success = await model_cache.store_model_output(
            cache_key=cache_request["cache_key"],
            output_data=cache_request["output_data"],
            ttl=cache_request.get("ttl", 1800)
        )

        if success:
            logger.info("Model output cached", cache_key=cache_request["cache_key"])
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
    current_user: str = None
) -> Dict[str, Any]:
    """Get cached model output"""
    from fastapi import HTTPException
    from prsm.core.redis_client import get_model_cache

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
    current_user: str = None
) -> Dict[str, str]:
    """Register a new model with semantic embedding"""
    from fastapi import HTTPException
    from prsm.core.database import ModelQueries
    from prsm.core.vector_db import embedding_generator, get_vector_db_manager

    try:
        required_fields = ["model_id", "name", "description", "model_type", "owner_id"]
        for field in required_fields:
            if field not in model_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )

        await ModelQueries.register_model(model_data)
        model_embedding = await embedding_generator.generate_model_embedding(model_data)

        if model_embedding:
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
                return {
                    "model_id": model_data["model_id"],
                    "status": "registered",
                    "semantic_search_enabled": False,
                    "warning": "Vector database indexing failed"
                }
        else:
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
    current_user: str = None
) -> Dict[str, Any]:
    """Semantic search for models using natural language queries"""
    from fastapi import HTTPException
    from prsm.core.vector_db import embedding_generator, get_vector_db_manager

    try:
        if "query" not in search_request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: query"
            )

        query = search_request["query"]
        top_k = search_request.get("top_k", 10)
        model_type = search_request.get("model_type")
        specialization = search_request.get("specialization")

        query_embedding = await embedding_generator.generate_embedding(query)

        if not query_embedding:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate query embedding"
            )

        vector_db_manager = get_vector_db_manager()
        similar_models = await vector_db_manager.search_similar_models(
            query_embedding=query_embedding,
            top_k=top_k,
            model_type=model_type,
            specialization=specialization
        )

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


@app.get("/vectors/stats")
async def get_vector_database_stats(
    current_user: str = None
) -> Dict[str, Any]:
    """Get comprehensive vector database statistics"""
    from fastapi import HTTPException
    from prsm.core.vector_db import get_vector_db_manager

    try:
        vector_db_manager = get_vector_db_manager()
        health_status = await vector_db_manager.health_check()
        provider_stats = await vector_db_manager.get_provider_stats()

        total_vectors = 0
        healthy_providers = 0

        for provider, status in health_status.items():
            if status:
                healthy_providers += 1
                provider_vectors = 0
                if provider in provider_stats:
                    for index_name, index_stats in provider_stats[provider].items():
                        provider_vectors += index_stats.get("total_vector_count", 0)
                total_vectors = max(total_vectors, provider_vectors)

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
    current_user: str = None
) -> Dict[str, Any]:
    """Generate embedding for arbitrary text"""
    from fastapi import HTTPException
    from prsm.core.vector_db import embedding_generator

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
# These endpoints will be migrated to a separate router in future versions

@app.post("/teachers/create")
async def create_teacher_endpoint(
    teacher_request: Dict[str, Any],
    current_user: str = None
) -> Dict[str, Any]:
    """Create a new teacher model with real ML implementation"""
    from fastapi import HTTPException

    try:
        from prsm.compute.teachers.teacher_model import create_teacher_with_specialization

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

        teacher = await create_teacher_with_specialization(
            specialization=specialization,
            domain=domain,
            use_real_implementation=use_real_implementation
        )

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


@app.get("/teachers/available_backends")
async def get_available_teacher_backends(
    current_user: str = None
) -> Dict[str, Any]:
    """Get information about available teacher model backends"""
    from fastapi import HTTPException

    try:
        from prsm.compute.teachers.real_teacher_implementation import get_available_training_backends

        backends = get_available_training_backends()

        return {
            "available_backends": backends,
            "total_backends": len(backends),
            "real_implementation_available": len(backends) > 0,
            "capabilities": {
                "knowledge_distillation": len(backends) > 0,
                "real_model_training": len(backends) > 0,
                "adaptive_curriculum": True,
                "performance_assessment": len(backends) > 0
            }
        }

    except Exception as e:
        logger.error("Failed to get teacher backend information", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve backend information"
        )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("PRSM_API_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=8000, reload=True)
