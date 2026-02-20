#!/usr/bin/env python3
"""
Model Registry for PRSM Federation

Provides model discovery, registration, and metadata management for the
distributed model federation system.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
import json

logger = structlog.get_logger(__name__)


class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"


class ModelProvider(Enum):
    """Model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    MARKETPLACE = "marketplace"
    PRSM = "prsm"


@dataclass
class ModelDetails:
    """Detailed model information"""
    model_id: str
    name: str
    provider: ModelProvider
    capabilities: List[ModelCapability]
    context_length: int = 4096
    max_tokens: int = 2048
    cost_per_token: float = 0.0
    availability: float = 1.0  # 0.0 to 1.0
    performance_score: float = 0.8  # 0.0 to 1.0
    specialization_domains: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_available(self) -> bool:
        """Check if model is currently available"""
        return self.availability > 0.5
    
    @property
    def is_high_performance(self) -> bool:
        """Check if model has high performance score"""
        return self.performance_score > 0.8


class ModelRegistry:
    """
    Central registry for model discovery and management
    
    Manages information about available models across different providers,
    their capabilities, performance metrics, and availability status.
    """
    
    def __init__(self):
        """Initialize the model registry"""
        self.models: Dict[str, ModelDetails] = {}
        self.provider_models: Dict[ModelProvider, Set[str]] = {}
        self.capability_models: Dict[ModelCapability, Set[str]] = {}
        self.domain_specialists: Dict[str, Set[str]] = {}
        
        # Initialize with default models
        self._initialize_default_models()
        
        logger.info("ModelRegistry initialized", 
                   total_models=len(self.models),
                   providers=len(self.provider_models))
    
    def _initialize_default_models(self):
        """Initialize registry with default available models"""
        
        # OpenAI Models
        openai_models = [
            ModelDetails(
                model_id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider=ModelProvider.OPENAI,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.ANALYSIS, ModelCapability.REASONING],
                context_length=16385,
                max_tokens=4096,
                cost_per_token=0.002,
                performance_score=0.85,
                specialization_domains=["general", "coding", "analysis"]
            ),
            ModelDetails(
                model_id="gpt-4",
                name="GPT-4",
                provider=ModelProvider.OPENAI,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.CODE_GENERATION],
                context_length=128000,
                max_tokens=4096,
                cost_per_token=0.03,
                performance_score=0.95,
                specialization_domains=["reasoning", "complex_analysis", "coding"]
            )
        ]
        
        # Anthropic Models
        anthropic_models = [
            ModelDetails(
                model_id="claude-3-sonnet-20240229",
                name="Claude 3 Sonnet",
                provider=ModelProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.ANALYSIS, ModelCapability.REASONING],
                context_length=200000,
                max_tokens=4096,
                cost_per_token=0.015,
                performance_score=0.92,
                specialization_domains=["analysis", "reasoning", "creative_writing"]
            ),
            ModelDetails(
                model_id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider=ModelProvider.ANTHROPIC,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.CODE_GENERATION],
                context_length=200000,
                max_tokens=4096,
                cost_per_token=0.075,
                performance_score=0.98,
                specialization_domains=["complex_reasoning", "creative_tasks", "analysis"]
            )
        ]
        
        # HuggingFace Models
        hf_models = [
            ModelDetails(
                model_id="microsoft/DialoGPT-medium",
                name="DialoGPT Medium",
                provider=ModelProvider.HUGGINGFACE,
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=1024,
                max_tokens=512,
                cost_per_token=0.0,  # Free
                performance_score=0.7,
                specialization_domains=["conversation", "dialogue"]
            )
        ]
        
        # Register all models
        for model_list in [openai_models, anthropic_models, hf_models]:
            for model in model_list:
                self.register_model(model)
    
    def register_model(self, model: ModelDetails):
        """Register a new model in the registry"""
        self.models[model.model_id] = model
        
        # Update provider index
        if model.provider not in self.provider_models:
            self.provider_models[model.provider] = set()
        self.provider_models[model.provider].add(model.model_id)
        
        # Update capability index
        for capability in model.capabilities:
            if capability not in self.capability_models:
                self.capability_models[capability] = set()
            self.capability_models[capability].add(model.model_id)
        
        # Update domain specialist index
        for domain in model.specialization_domains:
            if domain not in self.domain_specialists:
                self.domain_specialists[domain] = set()
            self.domain_specialists[domain].add(model.model_id)
        
        logger.debug("Model registered", model_id=model.model_id, provider=model.provider.value)
    
    async def discover_specialists(self, task_category: str) -> List[str]:
        """Discover specialist models for a given task category"""
        specialists = []
        
        # Direct domain match
        if task_category in self.domain_specialists:
            specialists.extend(list(self.domain_specialists[task_category]))
        
        # Fuzzy matching for related domains
        related_domains = self._find_related_domains(task_category)
        for domain in related_domains:
            if domain in self.domain_specialists:
                specialists.extend(list(self.domain_specialists[domain]))
        
        # Remove duplicates and filter by availability
        unique_specialists = []
        for model_id in set(specialists):
            if model_id in self.models and self.models[model_id].is_available:
                unique_specialists.append(model_id)
        
        # Sort by performance score
        unique_specialists.sort(
            key=lambda mid: self.models[mid].performance_score,
            reverse=True
        )
        
        logger.debug("Discovered specialists", 
                    task_category=task_category,
                    specialists_count=len(unique_specialists))
        
        return unique_specialists[:10]  # Return top 10
    
    async def get_model_details(self, model_id: str) -> Optional[ModelDetails]:
        """Get detailed information about a specific model"""
        model = self.models.get(model_id)
        if model:
            logger.debug("Model details retrieved", model_id=model_id)
            return model
        else:
            logger.warning("Model not found", model_id=model_id)
            return None
    
    def get_models_by_capability(self, capability: ModelCapability) -> List[str]:
        """Get all models that have a specific capability"""
        if capability in self.capability_models:
            return [mid for mid in self.capability_models[capability] 
                   if mid in self.models and self.models[mid].is_available]
        return []
    
    def get_models_by_provider(self, provider: ModelProvider) -> List[str]:
        """Get all models from a specific provider"""
        if provider in self.provider_models:
            return [mid for mid in self.provider_models[provider]
                   if mid in self.models and self.models[mid].is_available]
        return []
    
    def get_high_performance_models(self) -> List[str]:
        """Get all high-performance models"""
        return [mid for mid, model in self.models.items() 
               if model.is_high_performance and model.is_available]
    
    def get_available_models(self) -> List[str]:
        """Get all currently available models"""
        return [mid for mid, model in self.models.items() if model.is_available]
    
    def update_model_performance(self, model_id: str, performance_score: float):
        """Update performance score for a model"""
        if model_id in self.models:
            self.models[model_id].performance_score = max(0.0, min(1.0, performance_score))
            self.models[model_id].last_updated = datetime.now(timezone.utc)
            logger.debug("Model performance updated", 
                        model_id=model_id, 
                        performance_score=performance_score)
    
    def update_model_availability(self, model_id: str, availability: float):
        """Update availability score for a model"""
        if model_id in self.models:
            self.models[model_id].availability = max(0.0, min(1.0, availability))
            self.models[model_id].last_updated = datetime.now(timezone.utc)
            logger.debug("Model availability updated",
                        model_id=model_id,
                        availability=availability)
    
    def _find_related_domains(self, task_category: str) -> List[str]:
        """Find domains related to the given task category"""
        # Simple fuzzy matching - in production this could be more sophisticated
        related = []
        category_lower = task_category.lower()
        
        domain_mappings = {
            'code': ['coding', 'programming', 'development'],
            'analysis': ['reasoning', 'complex_analysis', 'analytical'],
            'text': ['general', 'writing', 'creative_writing'],
            'reasoning': ['analysis', 'complex_reasoning', 'logical'],
            'creative': ['creative_writing', 'creative_tasks', 'general'],
            'conversation': ['dialogue', 'general']
        }
        
        for key, domains in domain_mappings.items():
            if key in category_lower:
                related.extend(domains)
        
        return list(set(related))
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry"""
        return {
            'total_models': len(self.models),
            'available_models': len(self.get_available_models()),
            'high_performance_models': len(self.get_high_performance_models()),
            'providers': len(self.provider_models),
            'capabilities': len(self.capability_models),
            'specialization_domains': len(self.domain_specialists),
            'provider_distribution': {
                provider.value: len(models) 
                for provider, models in self.provider_models.items()
            }
        }
    
    async def register_teacher_model(self, teacher_model: Any, cid: str) -> bool:
        """Register a teacher model with IPFS CID (for NWTN compatibility)"""
        try:
            model_id = getattr(teacher_model, 'teacher_id', None) or getattr(teacher_model, 'name', str(uuid4()))
            name = getattr(teacher_model, 'name', 'Unknown')
            specialization = getattr(teacher_model, 'specialization', 'general')
            performance_score = getattr(teacher_model, 'performance_score', 0.8)
            
            model_details = ModelDetails(
                model_id=str(model_id),
                name=name,
                provider=ModelProvider.LOCAL,
                capabilities=[ModelCapability.REASONING, ModelCapability.ANALYSIS],
                performance_score=performance_score,
                specialization_domains=[specialization, "general"]
            )
            model_details.metadata['ipfs_cid'] = cid
            
            self.register_model(model_details)
            return True
        except Exception as e:
            logger.error(f"Failed to register teacher model: {e}")
            return False
    
    @property
    def registered_models(self) -> Dict[str, Any]:
        """Property for backward compatibility with tests"""
        return self.models


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create the global model registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry