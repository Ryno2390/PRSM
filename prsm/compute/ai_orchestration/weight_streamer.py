"""
PRSM Weights-as-a-Stream (Dynamic Sharding)
===========================================

Implements demand-paging for neural weights. Allows edge nodes to stream 
specific model layers from IPFS/P2P on-demand, enabling large-model 
inference on memory-constrained devices.
"""

import asyncio
import logging
import torch
from typing import Dict, List, Optional, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

class WeightStreamer:
    """
    Manages the 'Working Set' of model weights in RAM.
    Acts as a bridge between the P2P storage and the Model Instance.
    Supports both Layer-level and Expert-level (MoE) streaming.
    """
    def __init__(self, ipfs_client, max_layers_in_ram: int = 2, max_experts_in_ram: int = 8):
        self.ipfs_client = ipfs_client
        self.max_layers_in_ram = max_layers_in_ram
        self.max_experts_in_ram = max_experts_in_ram
        
        # Mapping of index -> StateDict
        self.layer_cache: Dict[int, Any] = OrderedDict()
        self.expert_cache: Dict[str, Any] = OrderedDict() # Key: "layer_idx:expert_idx"
        
        self.layer_to_cid: Dict[int, str] = {}
        self.expert_to_cid: Dict[str, str] = {} # Key: "layer_idx:expert_idx"
        
        # Synchronization
        self._lock = asyncio.Lock()

    def register_shards(self, shard_map: Dict[int, str]):
        """Register which IPFS CIDs correspond to which model layers"""
        self.layer_to_cid.update(shard_map)
        logger.info(f"Registered {len(shard_map)} streamable layer shards")

    def register_expert_shards(self, layer_idx: int, expert_map: Dict[int, str]):
        """Register CIDs for specific experts within a layer (MoE)"""
        for expert_idx, cid in expert_map.items():
            key = f"{layer_idx}:{expert_idx}"
            self.expert_to_cid[key] = cid
        logger.info(f"Registered {len(expert_map)} expert shards for layer {layer_idx}")

    async def get_layer(self, layer_index: int) -> Any:
        """
        Retrieves a full layer from RAM or network.
        """
        async with self._lock:
            if layer_index in self.layer_cache:
                self.layer_cache.move_to_end(layer_index)
                return self.layer_cache[layer_index]

            logger.info(f"📡 Streaming layer {layer_index} from network...")
            cid = self.layer_to_cid.get(layer_index)
            if not cid:
                raise ValueError(f"No CID registered for layer {layer_index}")

            layer_data, _ = await self.ipfs_client.retrieve_with_provenance(cid)
            layer_weights = self._reconstruct_weights(layer_data)
            
            if len(self.layer_cache) >= self.max_layers_in_ram:
                self.layer_cache.popitem(last=False)

            self.layer_cache[layer_index] = layer_weights
            return layer_weights

    async def get_expert(self, layer_index: int, expert_index: int) -> Any:
        """
        Retrieves a specific expert from RAM or network (MoE support).
        This allows small nodes to only load the few experts they are compute-eligible for.
        """
        key = f"{layer_index}:{expert_index}"
        async with self._lock:
            if key in self.expert_cache:
                self.expert_cache.move_to_end(key)
                return self.expert_cache[key]

            logger.info(f"🧩 Streaming expert {key} from network...")
            cid = self.expert_to_cid.get(key)
            if not cid:
                # If we don't have a granular expert shard, we might need to load the whole layer
                # But for MoE goals, we assume expert-sharding is enabled.
                raise ValueError(f"No CID registered for expert {key}")

            expert_data, _ = await self.ipfs_client.retrieve_with_provenance(cid)
            expert_weights = self._reconstruct_weights(expert_data)
            
            if len(self.expert_cache) >= self.max_experts_in_ram:
                evicted_key, _ = self.expert_cache.popitem(last=False)
                logger.debug(f"🧹 Evicted expert {evicted_key} from RAM")

            self.expert_cache[key] = expert_weights
            return expert_weights

    def _reconstruct_weights(self, raw_data: bytes) -> Any:
        """Reconstructs torch parameters from raw bytes"""
        # In production: return torch.load(io.BytesIO(raw_data))
        return f"Weight_Shard_{len(raw_data)}"

    def cleanup_caching(self):
        """Force clear RAM cache"""
        self.layer_cache.clear()
        self.expert_cache.clear()
        logger.info("RAM cache cleared")
