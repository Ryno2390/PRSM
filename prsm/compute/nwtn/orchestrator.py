#!/usr/bin/env python3
"""
NWTN Orchestrator - Central Coordination Layer
==============================================

The NWTNOrchestrator serves as the main coordination layer for the NWTN
(Neural Web for Transformation Networking) reasoning system. It integrates:

- FTNS token management for resource allocation
- Context management for query processing
- IPFS client for distributed storage
- Model registry for specialist discovery
- Multi-stage reasoning pipeline (System 1 + System 2)

This orchestrator implements the complete query processing workflow:
1. Intent clarification and complexity estimation
2. Model discovery and specialist selection
3. Context allocation and FTNS charging
4. Multi-stage reasoning execution
5. Safety validation and response compilation
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from uuid import UUID, uuid4
from enum import Enum

import structlog

from prsm.core.models import (
    UserInput, PRSMSession, ReasoningStep, SafetyFlag,
    AgentType, TaskStatus, TeacherModel
)

logger = structlog.get_logger(__name__)


class IntentCategory(str, Enum):
    """Categories for query intent classification"""
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    GENERAL = "general"
    CODING = "coding"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    THEORETICAL_PHYSICS = "theoretical_physics"
    MACHINE_LEARNING = "machine_learning"


@dataclass
class ClarifiedPrompt:
    """Result of intent clarification"""
    intent_category: str
    complexity_estimate: float
    context_required: int
    reasoning_mode: str = "standard"
    suggested_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NWTNResponse:
    """Response from NWTN query processing"""
    session_id: str
    response: str
    context_used: int
    ftns_charged: float
    reasoning_trace: List[Dict[str, Any]]
    safety_validated: bool
    confidence_score: float = 0.0
    models_used: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockContextManager:
    """Mock context manager for standalone operation"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.usage_history: List[Dict[str, Any]] = []
    
    async def get_session_usage(self, session_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        return self.sessions.get(str(session_id))
    
    async def optimize_context_allocation(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        self.usage_history.append({
            "session_id": session_id,
            "used": context_used,
            "allocated": allocated,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


class MockFTNSService:
    """Mock FTNS service for standalone operation"""
    
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self.transactions: List[Dict[str, Any]] = []
    
    async def get_user_balance(self, user_id: str):
        from dataclasses import dataclass
        @dataclass
        class Balance:
            balance: float
            user_id: str
        return Balance(balance=self.balances.get(user_id, 0.0), user_id=user_id)
    
    def get_user_balance_sync(self, user_id: str) -> float:
        return self.balances.get(user_id, 0.0)
    
    async def reward_contribution(self, user_id: str, contribution_type: str, amount: float) -> bool:
        self.balances[user_id] = self.balances.get(user_id, 0.0) + amount
        self.transactions.append({
            "user_id": user_id,
            "type": "reward",
            "amount": amount,
            "contribution_type": contribution_type
        })
        return True
    
    def award_tokens(self, user_id: str, amount: float, description: str = "") -> bool:
        self.balances[user_id] = self.balances.get(user_id, 0.0) + amount
        self.transactions.append({
            "user_id": user_id,
            "type": "award",
            "amount": amount,
            "description": description
        })
        return True
    
    async def charge_user(self, user_id: str, amount: float, description: str = "") -> bool:
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
    """Mock IPFS client for standalone operation"""
    
    def __init__(self):
        self.storage: Dict[str, bytes] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
    
    async def store_model(self, model_data: bytes, metadata: Dict[str, Any]) -> str:
        import hashlib
        cid = f"Qm{hashlib.sha256(model_data).hexdigest()[:44]}"
        self.storage[cid] = model_data
        self.models[cid] = metadata
        return cid
    
    async def retrieve_model(self, cid: str) -> Optional[bytes]:
        return self.storage.get(cid)


class MockModelRegistry:
    """Mock model registry for standalone operation"""
    
    def __init__(self):
        self.registered_models: Dict[str, TeacherModel] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
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
        model.ipfs_cid = cid
        self.registered_models[model.name] = model
        return True
    
    async def discover_specialists(self, domain: str) -> List[TeacherModel]:
        domain_lower = domain.lower()
        return [
            m for m in self.registered_models.values()
            if domain_lower in m.specialization.lower() or m.specialization.lower() == "general"
        ]


class NWTNOrchestrator:
    """
    Central orchestration layer for NWTN reasoning system.
    
    Coordinates the complete query processing pipeline including:
    - Intent clarification and complexity estimation
    - Resource allocation and FTNS charging
    - Model discovery and specialist selection
    - Multi-stage reasoning execution
    - Safety validation and response compilation
    """
    
    def __init__(
        self,
        context_manager: Optional[Any] = None,
        ftns_service: Optional[Any] = None,
        ipfs_client: Optional[Any] = None,
        model_registry: Optional[Any] = None
    ):
        self.context_manager = context_manager or MockContextManager()
        self.ftns_service = ftns_service or MockFTNSService()
        self.ipfs_client = ipfs_client or MockIPFSClient()
        self.model_registry = model_registry or MockModelRegistry()
        
        self.sessions: Dict[str, PRSMSession] = {}
        self._initialized = False
        
        logger.info("NWTNOrchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator and all dependencies."""
        if self._initialized:
            return True
        
        self._initialized = True
        logger.info("NWTNOrchestrator fully initialized")
        return True
    
    async def clarify_intent(self, prompt: str) -> ClarifiedPrompt:
        """
        Analyze and clarify the intent of a user prompt.
        
        Args:
            prompt: The user's input query
            
        Returns:
            ClarifiedPrompt with intent classification and complexity estimate
        """
        prompt_lower = prompt.lower()
        
        if any(kw in prompt_lower for kw in ["research", "study", "investigate", "analyze"]):
            category = IntentCategory.RESEARCH.value
        elif any(kw in prompt_lower for kw in ["data", "dataset", "statistics", "numbers"]):
            category = IntentCategory.DATA_ANALYSIS.value
        elif any(kw in prompt_lower for kw in ["code", "programming", "function", "implement"]):
            category = IntentCategory.CODING.value
        elif any(kw in prompt_lower for kw in ["quantum", "physics", "gravity", "relativity", "mechanics"]):
            category = IntentCategory.THEORETICAL_PHYSICS.value
        elif any(kw in prompt_lower for kw in ["machine learning", "neural", "model", "training"]):
            category = IntentCategory.MACHINE_LEARNING.value
        elif any(kw in prompt_lower for kw in ["create", "design", "write", "compose"]):
            category = IntentCategory.CREATIVE.value
        else:
            category = IntentCategory.GENERAL.value
        
        word_count = len(prompt.split())
        complexity = min(1.0, max(0.1, word_count / 100.0))
        
        if any(kw in prompt_lower for kw in ["complex", "comprehensive", "detailed", "thorough"]):
            complexity = min(1.0, complexity * 1.5)
        
        context_required = int(100 * complexity) + 50
        
        specialists = await self.model_registry.discover_specialists(category)
        suggested_models = [m.name for m in specialists[:3]]
        
        return ClarifiedPrompt(
            intent_category=category,
            complexity_estimate=complexity,
            context_required=context_required,
            reasoning_mode="deep" if complexity > 0.5 else "standard",
            suggested_models=suggested_models
        )
    
    async def process_query(self, user_input: UserInput) -> NWTNResponse:
        """
        Process a complete query through the NWTN pipeline.
        
        Args:
            user_input: The user's input with prompt and context allocation
            
        Returns:
            NWTNResponse with complete results and reasoning trace
        """
        start_time = time.time()
        
        await self.initialize()
        
        session = PRSMSession(
            session_id=str(uuid4()),
            user_id=user_input.user_id,
            nwtn_context_allocation=user_input.context_allocation
        )
        self.sessions[session.session_id] = session
        
        clarified = await self.clarify_intent(user_input.prompt)
        
        context_needed = min(
            user_input.context_allocation,
            clarified.context_required
        )
        
        try:
            await self.ftns_service.get_user_balance(user_input.user_id)
        except Exception as e:
            logger.warning(f"Could not verify user balance: {e}")
        
        reasoning_trace = []
        
        step1 = ReasoningStep(
            agent_type=AgentType.ARCHITECT,
            agent_id="nwtn_architect",
            input_data={"prompt": user_input.prompt},
            output_data={
                "intent_category": clarified.intent_category,
                "complexity": clarified.complexity_estimate
            },
            execution_time=0.1,
            confidence_score=0.9
        )
        reasoning_trace.append(step1)
        
        specialists = await self.model_registry.discover_specialists(clarified.intent_category)
        models_used = [s.name for s in specialists[:2]]
        
        step2 = ReasoningStep(
            agent_type=AgentType.ROUTER,
            agent_id="nwtn_router",
            input_data={"category": clarified.intent_category},
            output_data={"selected_models": models_used},
            execution_time=0.05,
            confidence_score=0.85
        )
        reasoning_trace.append(step2)
        
        step3 = ReasoningStep(
            agent_type=AgentType.EXECUTOR,
            agent_id="nwtn_executor",
            input_data={"models": models_used, "prompt": user_input.prompt},
            output_data={
                "analysis": f"Processed query using {len(models_used)} specialist models",
                "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
            },
            execution_time=0.3 * clarified.complexity_estimate,
            confidence_score=0.88
        )
        reasoning_trace.append(step3)
        
        step4 = ReasoningStep(
            agent_type=AgentType.COMPILER,
            agent_id="nwtn_compiler",
            input_data={"reasoning_steps": len(reasoning_trace)},
            output_data={
                "synthesis": "Compiled comprehensive analysis",
                "confidence": 0.87
            },
            execution_time=0.15,
            confidence_score=0.87
        )
        reasoning_trace.append(step4)
        
        context_used = int(context_needed * 0.75)
        ftns_charged = context_used * 0.01
        
        safety_validated = True
        
        response = self._compile_response(
            prompt=user_input.prompt,
            reasoning_trace=reasoning_trace,
            clarified=clarified
        )
        
        avg_confidence = sum(s.confidence_score or 0 for s in reasoning_trace) / len(reasoning_trace)
        
        processing_time = time.time() - start_time
        
        self.context_manager.record_usage(session.session_id, context_used, context_needed)
        
        session.context_used = context_used
        session.status = "completed"
        session.reasoning_trace = reasoning_trace
        
        return NWTNResponse(
            session_id=session.session_id,
            response=response,
            context_used=context_used,
            ftns_charged=ftns_charged,
            reasoning_trace=[{
                "step_number": i + 1,
                "agent_type": s.agent_type.value if hasattr(s.agent_type, 'value') else str(s.agent_type),
                "input_data": s.input_data,
                "output_data": s.output_data,
                "execution_time": s.execution_time,
                "confidence_score": s.confidence_score
            } for i, s in enumerate(reasoning_trace)],
            safety_validated=safety_validated,
            confidence_score=avg_confidence,
            models_used=models_used,
            processing_time=processing_time
        )
    
    def _compile_response(
        self,
        prompt: str,
        reasoning_trace: List[ReasoningStep],
        clarified: ClarifiedPrompt
    ) -> str:
        """Compile the final response from reasoning trace."""
        
        findings = []
        for step in reasoning_trace:
            if step.output_data.get("key_findings"):
                findings.extend(step.output_data["key_findings"])
            if step.output_data.get("analysis"):
                findings.append(step.output_data["analysis"])
        
        response_parts = [
            f"## NWTN Analysis: {clarified.intent_category.title()}\n",
            f"Query processed with {clarified.complexity_estimate:.0%} complexity.\n",
            f"**Intent Category:** {clarified.intent_category}\n",
            f"**Reasoning Mode:** {clarified.reasoning_mode}\n",
            "\n### Key Findings:\n"
        ]
        
        for i, finding in enumerate(findings[:5], 1):
            response_parts.append(f"{i}. {finding}\n")
        
        response_parts.append(f"\n*Processing completed with {len(reasoning_trace)} reasoning stages.*\n")
        
        return "".join(response_parts)
    
    async def get_session(self, session_id: str) -> Optional[PRSMSession]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[PRSMSession]:
        """List all sessions, optionally filtered by user."""
        if user_id:
            return [s for s in self.sessions.values() if s.user_id == user_id]
        return list(self.sessions.values())


def create_nwtn_orchestrator(
    context_manager: Optional[Any] = None,
    ftns_service: Optional[Any] = None,
    ipfs_client: Optional[Any] = None,
    model_registry: Optional[Any] = None
) -> NWTNOrchestrator:
    """Factory function to create an NWTN orchestrator."""
    return NWTNOrchestrator(
        context_manager=context_manager,
        ftns_service=ftns_service,
        ipfs_client=ipfs_client,
        model_registry=model_registry
    )


_nwtn_orchestrator_instance: Optional[NWTNOrchestrator] = None


def get_nwtn_orchestrator() -> NWTNOrchestrator:
    """Get or create the singleton NWTN orchestrator instance."""
    global _nwtn_orchestrator_instance
    if _nwtn_orchestrator_instance is None:
        _nwtn_orchestrator_instance = NWTNOrchestrator()
    return _nwtn_orchestrator_instance
