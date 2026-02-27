#!/usr/bin/env python3
"""
NWTN Mock Services - Test Fixtures
==================================

This module contains mock service classes for testing the NWTN orchestrator.
These mocks are intentionally kept separate from production code to prevent
accidental use in production environments.

IMPORTANT:
- These classes should ONLY be used in test contexts
- Production code should use real service implementations
- Use pytest fixtures or dependency injection to inject these mocks

Usage:
    from tests.fixtures.nwtn_mocks import (
        MockContextManager,
        MockFTNSService,
        MockIPFSClient,
        MockModelRegistry
    )

    # In tests, inject mocks via constructor
    orchestrator = NWTNOrchestrator(
        context_manager=MockContextManager(),
        ftns_service=MockFTNSService(),
        ipfs_client=MockIPFSClient(),
        model_registry=MockModelRegistry()
    )
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from uuid import UUID

from prsm.core.models import TeacherModel


class MockContextManager:
    """
    Mock context manager for testing NWTN orchestrator.
    
    Simulates context allocation and usage tracking without
    requiring a real context management infrastructure.
    
    WARNING: This class is for testing only. Do not use in production.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.usage_history: List[Dict[str, Any]] = []
    
    async def get_session_usage(self, session_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """Get usage data for a session."""
        return self.sessions.get(str(session_id))
    
    async def optimize_context_allocation(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate optimization metrics from historical usage data."""
        if not historical_data:
            return {"avg_efficiency": 0.7, "over_allocation_rate": 0.1, "under_allocation_rate": 0.1}
        
        efficiencies = [d.get("used", 0) / max(d.get("allocated", 1), 1) for d in historical_data]
        avg_eff = sum(efficiencies) / len(efficiencies)
        
        return {
            "avg_efficiency": avg_eff,
            "over_allocation_rate": sum(1 for e in efficiencies if e < 0.7) / len(efficiencies),
            "under_allocation_rate": sum(1 for e in efficiencies if e > 0.9) / len(efficiencies),
            "optimization_potential": (1 - avg_eff) * 100
        }
    
    def record_usage(self, session_id: str, context_used: int, allocated: int):
        """Record context usage for a session."""
        self.usage_history.append({
            "session_id": session_id,
            "used": context_used,
            "allocated": allocated,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


class MockFTNSService:
    """
    Mock FTNS (Fungible Token for Network Services) for testing.
    
    Simulates token balance management and transactions without
    requiring a real blockchain or token infrastructure.
    
    WARNING: This class is for testing only. Do not use in production.
    """
    
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self.transactions: List[Dict[str, Any]] = []
    
    async def get_user_balance(self, user_id: str):
        """Get balance for a user, returning a Balance dataclass."""
        @dataclass
        class Balance:
            balance: float
            user_id: str
        return Balance(balance=self.balances.get(user_id, 0.0), user_id=user_id)
    
    def get_user_balance_sync(self, user_id: str) -> float:
        """Synchronously get user balance."""
        return self.balances.get(user_id, 0.0)
    
    async def reward_contribution(self, user_id: str, contribution_type: str, amount: float) -> bool:
        """Reward a user for their contribution."""
        self.balances[user_id] = self.balances.get(user_id, 0.0) + amount
        self.transactions.append({
            "user_id": user_id,
            "type": "reward",
            "amount": amount,
            "contribution_type": contribution_type
        })
        return True
    
    def award_tokens(self, user_id: str, amount: float, description: str = "") -> bool:
        """Award tokens to a user."""
        self.balances[user_id] = self.balances.get(user_id, 0.0) + amount
        self.transactions.append({
            "user_id": user_id,
            "type": "award",
            "amount": amount,
            "description": description
        })
        return True
    
    async def charge_user(self, user_id: str, amount: float, description: str = "") -> bool:
        """Charge a user's balance (async version with validation)."""
        if self.balances.get(user_id, 0.0) < amount:
            raise ValueError(f"Insufficient balance for user {user_id}")
        self.balances[user_id] -= amount
        self.transactions.append({
            "user_id": user_id,
            "type": "charge",
            "amount": amount,
            "description": description
        })
        return True
    
    def deduct_tokens(self, user_id: str, amount: float, description: str = "") -> bool:
        """Deduct tokens from a user's balance (sync version)."""
        if self.balances.get(user_id, 0.0) < amount:
            return False
        self.balances[user_id] -= amount
        self.transactions.append({
            "user_id": user_id,
            "type": "deduct",
            "amount": amount,
            "description": description
        })
        return True


class MockIPFSClient:
    """
    Mock IPFS (InterPlanetary File System) client for testing.
    
    Simulates distributed storage operations without requiring
    a real IPFS node or network connection.
    
    WARNING: This class is for testing only. Do not use in production.
    """
    
    def __init__(self):
        self.storage: Dict[str, bytes] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
    
    async def store_model(self, model_data: bytes, metadata: Dict[str, Any]) -> str:
        """Store model data and return a content identifier (CID)."""
        import hashlib
        cid = f"Qm{hashlib.sha256(model_data).hexdigest()[:44]}"
        self.storage[cid] = model_data
        self.models[cid] = metadata
        return cid
    
    async def retrieve_model(self, cid: str) -> Optional[bytes]:
        """Retrieve model data by content identifier."""
        return self.storage.get(cid)


class MockModelRegistry:
    """
    Mock model registry for testing NWTN orchestrator.
    
    Simulates model registration and discovery without requiring
    a real model registry infrastructure.
    
    WARNING: This class is for testing only. Do not use in production.
    """
    
    def __init__(self):
        self.registered_models: Dict[str, TeacherModel] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default test models."""
        default_models = [
            TeacherModel(name="Research Assistant", specialization="research", performance_score=0.92),
            TeacherModel(name="Data Analyzer", specialization="data_analysis", performance_score=0.88),
            TeacherModel(name="General Helper", specialization="general", performance_score=0.85),
            TeacherModel(name="Physics Expert", specialization="theoretical_physics", performance_score=0.90),
            TeacherModel(name="ML Specialist", specialization="machine_learning", performance_score=0.91),
        ]
        for model in default_models:
            self.registered_models[model.name] = model
    
    async def register_teacher_model(self, model: TeacherModel, cid: str) -> bool:
        """Register a teacher model with its IPFS content identifier."""
        model.ipfs_cid = cid
        self.registered_models[model.name] = model
        return True
    
    async def discover_specialists(self, domain: str) -> List[TeacherModel]:
        """Discover specialist models for a given domain."""
        domain_lower = domain.lower()
        return [
            m for m in self.registered_models.values()
            if domain_lower in m.specialization.lower() or m.specialization.lower() == "general"
        ]


# Pytest fixtures for convenience
# These can be used directly in tests with pytest
# Import pytest conditionally to avoid errors when importing mocks outside pytest context

try:
    import pytest
    
    @pytest.fixture
    def mock_context_manager():
        """Pytest fixture providing a MockContextManager instance."""
        return MockContextManager()


    @pytest.fixture
    def mock_ftns_service():
        """Pytest fixture providing a MockFTNSService instance."""
        return MockFTNSService()


    @pytest.fixture
    def mock_ipfs_client():
        """Pytest fixture providing a MockIPFSClient instance."""
        return MockIPFSClient()


    @pytest.fixture
    def mock_model_registry():
        """Pytest fixture providing a MockModelRegistry instance."""
        return MockModelRegistry()


    @pytest.fixture
    def nwtn_mock_services(mock_context_manager, mock_ftns_service, mock_ipfs_client, mock_model_registry):
        """
        Pytest fixture providing all NWTN mock services as a dictionary.
        
        Usage:
            def test_something(nwtn_mock_services):
                orchestrator = NWTNOrchestrator(**nwtn_mock_services)
        """
        return {
            "context_manager": mock_context_manager,
            "ftns_service": mock_ftns_service,
            "ipfs_client": mock_ipfs_client,
            "model_registry": mock_model_registry
        }

except ImportError:
    # pytest not available - fixtures will not be defined
    # but the mock classes can still be imported and used directly
    pass
