"""NWTN Model Service — register, deploy, and serve fine-tuned models."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from prsm.compute.nwtn.training.models import ModelCard, DeploymentStatus

logger = logging.getLogger(__name__)


class NWTNModelService:
    """Manages NWTN model lifecycle: register, shard, deploy, serve, retire."""

    def __init__(self, tensor_executor=None):
        self._tensor_executor = tensor_executor
        self._models: Dict[str, ModelCard] = {}
        self._deployments: Dict[str, Dict[str, Any]] = {}

    def register_model(self, model_card: ModelCard) -> str:
        """Register a model for deployment."""
        self._models[model_card.model_id] = model_card
        model_card.status = DeploymentStatus.REGISTERED
        logger.info(f"Model registered: {model_card.model_id} ({model_card.model_name})")
        return model_card.model_id

    def deploy_model(
        self,
        model_id: str,
        weight_tensors: Optional[Dict[str, np.ndarray]] = None,
        n_shards: int = 4,
    ) -> Dict[str, Any]:
        """Shard and deploy a model via Ring 8 infrastructure."""
        card = self._models.get(model_id)
        if card is None:
            raise ValueError(f"Model {model_id} not registered")

        card.status = DeploymentStatus.SHARDING

        deployment = {
            "model_id": model_id,
            "n_shards": n_shards,
            "deployed_at": time.time(),
            "shard_ids": [],
        }

        if weight_tensors:
            from prsm.compute.model_sharding.sharder import ModelSharder
            sharder = ModelSharder()
            sharded = sharder.shard_model(
                model_id=model_id,
                model_name=card.model_name,
                weight_tensors=weight_tensors,
                n_shards=n_shards,
            )
            deployment["shard_ids"] = [s.shard_id for s in sharded.shards]
            deployment["total_size_bytes"] = sharded.total_size_bytes

        card.status = DeploymentStatus.DEPLOYED
        self._deployments[model_id] = deployment

        logger.info(f"Model deployed: {model_id} ({n_shards} shards)")
        return deployment

    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        card = self._models.get(model_id)
        if card is None:
            return None
        deployment = self._deployments.get(model_id, {})
        return {
            **card.to_dict(),
            "deployment": deployment,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [card.to_dict() for card in self._models.values()]

    def retire_model(self, model_id: str) -> bool:
        card = self._models.get(model_id)
        if card is None:
            return False
        card.status = DeploymentStatus.RETIRED
        self._deployments.pop(model_id, None)
        logger.info(f"Model retired: {model_id}")
        return True
