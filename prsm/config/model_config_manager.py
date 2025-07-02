"""
Model Configuration Manager for PRSM

Centralized management of model configurations, pricing, capabilities,
and provider mappings. Replaces hardcoded configurations throughout
the codebase with flexible, externalized configuration files.

🎯 Features:
- Centralized model catalog management
- Dynamic pricing configuration
- Capability-based routing configuration  
- Provider mapping management
- Environment-specific overrides
- Runtime configuration updates
- Comprehensive validation
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import structlog

logger = structlog.get_logger(__name__)


class ModelTier(Enum):
    """Model tier classification"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ModelCapability(Enum):
    """Model capability types"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    MULTIMODAL = "multimodal"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    STREAMING = "streaming"


@dataclass
class ModelPricing:
    """Model pricing configuration"""
    input_cost_per_1k: Decimal
    output_cost_per_1k: Decimal
    currency: str = "USD"
    effective_date: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ModelConfiguration:
    """Complete model configuration"""
    id: str
    name: str
    provider: str
    tier: ModelTier
    pricing: ModelPricing
    context_length: int
    max_tokens: int
    capabilities: List[ModelCapability]
    quality_score: float
    latency_score: float
    cost_score: float
    supports_streaming: bool = False
    supports_tools: bool = False
    supports_vision: bool = False
    model_family: Optional[str] = None
    model_version: Optional[str] = None
    deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfiguration:
    """Provider-specific configuration"""
    name: str
    base_url: Optional[str] = None
    auth_type: str = "api_key"
    rate_limits: Dict[str, int] = field(default_factory=dict)
    default_models: List[str] = field(default_factory=list)
    supported_capabilities: List[ModelCapability] = field(default_factory=list)
    health_check_endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityMapping:
    """Task capability to provider mapping"""
    task_type: str
    preferred_providers: List[str]
    fallback_providers: List[str]
    model_requirements: Dict[str, Any] = field(default_factory=dict)
    quality_threshold: float = 0.7
    cost_threshold: Optional[float] = None
    latency_threshold: Optional[float] = None


class ModelConfigManager:
    """
    Centralized model configuration manager
    
    Manages all model-related configurations including:
    - Model catalogs and specifications
    - Pricing and cost information
    - Provider mappings and capabilities
    - Routing and selection strategies
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to config directory in the same location as this file
            config_dir = Path(__file__).parent / "models"
        
        self.config_dir = Path(config_dir)
        self._model_catalog: Dict[str, ModelConfiguration] = {}
        self._provider_configs: Dict[str, ProviderConfiguration] = {}
        self._capability_mappings: Dict[str, CapabilityMapping] = {}
        self._pricing_config: Dict[str, Dict[str, ModelPricing]] = {}
        self._loaded = False
    
    def load_all_configurations(self, force_reload: bool = False) -> None:
        """Load all configuration files"""
        if self._loaded and not force_reload:
            return
        
        try:
            self._load_model_catalog()
            self._load_provider_configs()
            self._load_capability_mappings()
            self._load_pricing_configs()
            self._loaded = True
            
            logger.info("Model configurations loaded successfully", 
                       models_count=len(self._model_catalog),
                       providers_count=len(self._provider_configs),
                       capabilities_count=len(self._capability_mappings))
            
        except Exception as e:
            logger.error("Failed to load model configurations", error=str(e))
            # Load fallback configurations
            self._load_fallback_configs()
    
    def _load_model_catalog(self) -> None:
        """Load model catalog from configuration files"""
        catalog_file = self.config_dir / "model_catalog.yaml"
        
        if not catalog_file.exists():
            logger.warning("Model catalog file not found, using fallback", 
                          expected_path=str(catalog_file))
            self._load_fallback_model_catalog()
            return
        
        try:
            with open(catalog_file, 'r') as f:
                catalog_data = yaml.safe_load(f)
            
            for model_id, model_data in catalog_data.get("models", {}).items():
                pricing_data = model_data.get("pricing", {})
                pricing = ModelPricing(
                    input_cost_per_1k=Decimal(str(pricing_data.get("input", 0))),
                    output_cost_per_1k=Decimal(str(pricing_data.get("output", 0))),
                    currency=pricing_data.get("currency", "USD")
                )
                
                capabilities = [
                    ModelCapability(cap) for cap in model_data.get("capabilities", [])
                    if cap in [c.value for c in ModelCapability]
                ]
                
                model_config = ModelConfiguration(
                    id=model_id,
                    name=model_data.get("name", model_id),
                    provider=model_data.get("provider", "unknown"),
                    tier=ModelTier(model_data.get("tier", "basic")),
                    pricing=pricing,
                    context_length=model_data.get("context_length", 4096),
                    max_tokens=model_data.get("max_tokens", 1000),
                    capabilities=capabilities,
                    quality_score=model_data.get("quality_score", 0.7),
                    latency_score=model_data.get("latency_score", 0.7),
                    cost_score=model_data.get("cost_score", 0.7),
                    supports_streaming=model_data.get("supports_streaming", False),
                    supports_tools=model_data.get("supports_tools", False),
                    supports_vision=model_data.get("supports_vision", False),
                    model_family=model_data.get("model_family"),
                    model_version=model_data.get("model_version"),
                    deprecated=model_data.get("deprecated", False),
                    metadata=model_data.get("metadata", {})
                )
                
                self._model_catalog[model_id] = model_config
                
        except Exception as e:
            logger.error("Failed to load model catalog", error=str(e), file=str(catalog_file))
            self._load_fallback_model_catalog()
    
    def _load_provider_configs(self) -> None:
        """Load provider configurations"""
        providers_file = self.config_dir / "providers.yaml"
        
        if not providers_file.exists():
            logger.warning("Provider config file not found, using fallback",
                          expected_path=str(providers_file))
            self._load_fallback_provider_configs()
            return
        
        try:
            with open(providers_file, 'r') as f:
                providers_data = yaml.safe_load(f)
            
            for provider_name, provider_data in providers_data.get("providers", {}).items():
                capabilities = [
                    ModelCapability(cap) for cap in provider_data.get("supported_capabilities", [])
                    if cap in [c.value for c in ModelCapability]
                ]
                
                provider_config = ProviderConfiguration(
                    name=provider_name,
                    base_url=provider_data.get("base_url"),
                    auth_type=provider_data.get("auth_type", "api_key"),
                    rate_limits=provider_data.get("rate_limits", {}),
                    default_models=provider_data.get("default_models", []),
                    supported_capabilities=capabilities,
                    health_check_endpoint=provider_data.get("health_check_endpoint"),
                    metadata=provider_data.get("metadata", {})
                )
                
                self._provider_configs[provider_name] = provider_config
                
        except Exception as e:
            logger.error("Failed to load provider configs", error=str(e))
            self._load_fallback_provider_configs()
    
    def _load_capability_mappings(self) -> None:
        """Load capability mappings"""
        mappings_file = self.config_dir / "capability_mappings.yaml"
        
        if not mappings_file.exists():
            logger.warning("Capability mappings file not found, using fallback",
                          expected_path=str(mappings_file))
            self._load_fallback_capability_mappings()
            return
        
        try:
            with open(mappings_file, 'r') as f:
                mappings_data = yaml.safe_load(f)
            
            for task_type, mapping_data in mappings_data.get("task_mappings", {}).items():
                mapping = CapabilityMapping(
                    task_type=task_type,
                    preferred_providers=mapping_data.get("preferred_providers", []),
                    fallback_providers=mapping_data.get("fallback_providers", []),
                    model_requirements=mapping_data.get("model_requirements", {}),
                    quality_threshold=mapping_data.get("quality_threshold", 0.7),
                    cost_threshold=mapping_data.get("cost_threshold"),
                    latency_threshold=mapping_data.get("latency_threshold")
                )
                
                self._capability_mappings[task_type] = mapping
                
        except Exception as e:
            logger.error("Failed to load capability mappings", error=str(e))
            self._load_fallback_capability_mappings()
    
    def _load_pricing_configs(self) -> None:
        """Load pricing configurations"""
        pricing_file = self.config_dir / "pricing.yaml"
        
        if not pricing_file.exists():
            logger.warning("Pricing config file not found, using model catalog pricing")
            return
        
        try:
            with open(pricing_file, 'r') as f:
                pricing_data = yaml.safe_load(f)
            
            for provider, models in pricing_data.get("pricing", {}).items():
                self._pricing_config[provider] = {}
                for model_id, pricing_info in models.items():
                    pricing = ModelPricing(
                        input_cost_per_1k=Decimal(str(pricing_info.get("input", 0))),
                        output_cost_per_1k=Decimal(str(pricing_info.get("output", 0))),
                        currency=pricing_info.get("currency", "USD"),
                        effective_date=pricing_info.get("effective_date"),
                        notes=pricing_info.get("notes")
                    )
                    self._pricing_config[provider][model_id] = pricing
                    
        except Exception as e:
            logger.error("Failed to load pricing configs", error=str(e))
    
    # Fallback configurations for when config files are missing
    def _load_fallback_configs(self) -> None:
        """Load minimal fallback configurations"""
        self._load_fallback_model_catalog()
        self._load_fallback_provider_configs()
        self._load_fallback_capability_mappings()
        logger.warning("Using fallback configurations - please create proper config files")
    
    def _load_fallback_model_catalog(self) -> None:
        """Load fallback model catalog with essential models"""
        fallback_models = {
            "gpt-3.5-turbo": ModelConfiguration(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider="openai",
                tier=ModelTier.BASIC,
                pricing=ModelPricing(Decimal("0.0005"), Decimal("0.0015")),
                context_length=4096,
                max_tokens=1000,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
                quality_score=0.85,
                latency_score=0.9,
                cost_score=0.9,
                supports_streaming=True
            ),
            "claude-3-sonnet": ModelConfiguration(
                id="claude-3-sonnet",
                name="Claude 3 Sonnet",
                provider="anthropic",
                tier=ModelTier.PREMIUM,
                pricing=ModelPricing(Decimal("0.003"), Decimal("0.015")),
                context_length=200000,
                max_tokens=1000,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                quality_score=0.90,
                latency_score=0.8,
                cost_score=0.7,
                supports_streaming=True
            ),
            "llama2-7b-chat": ModelConfiguration(
                id="llama2-7b-chat",
                name="Llama 2 7B Chat",
                provider="ollama",
                tier=ModelTier.FREE,
                pricing=ModelPricing(Decimal("0"), Decimal("0")),
                context_length=4096,
                max_tokens=1000,
                capabilities=[ModelCapability.TEXT_GENERATION],
                quality_score=0.75,
                latency_score=0.6,
                cost_score=1.0
            )
        }
        
        self._model_catalog.update(fallback_models)
    
    def _load_fallback_provider_configs(self) -> None:
        """Load fallback provider configurations"""
        fallback_providers = {
            "openai": ProviderConfiguration(
                name="openai",
                auth_type="api_key",
                default_models=["gpt-3.5-turbo"],
                supported_capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION]
            ),
            "anthropic": ProviderConfiguration(
                name="anthropic",
                auth_type="api_key",
                default_models=["claude-3-sonnet"],
                supported_capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING]
            ),
            "ollama": ProviderConfiguration(
                name="ollama",
                auth_type="none",
                default_models=["llama2-7b-chat"],
                supported_capabilities=[ModelCapability.TEXT_GENERATION]
            )
        }
        
        self._provider_configs.update(fallback_providers)
    
    def _load_fallback_capability_mappings(self) -> None:
        """Load fallback capability mappings"""
        fallback_mappings = {
            "code_generation": CapabilityMapping(
                task_type="code_generation",
                preferred_providers=["openai", "anthropic"],
                fallback_providers=["ollama"]
            ),
            "reasoning": CapabilityMapping(
                task_type="reasoning",
                preferred_providers=["anthropic", "openai"],
                fallback_providers=["ollama"]
            ),
            "general_chat": CapabilityMapping(
                task_type="general_chat",
                preferred_providers=["ollama", "openai", "anthropic"],
                fallback_providers=[]
            )
        }
        
        self._capability_mappings.update(fallback_mappings)
    
    # Public API methods
    def get_model_config(self, model_id: str) -> Optional[ModelConfiguration]:
        """Get configuration for a specific model"""
        self.load_all_configurations()
        return self._model_catalog.get(model_id)
    
    def get_all_models(self) -> Dict[str, ModelConfiguration]:
        """Get all model configurations"""
        self.load_all_configurations()
        return self._model_catalog.copy()
    
    def get_models_by_provider(self, provider: str) -> Dict[str, ModelConfiguration]:
        """Get all models for a specific provider"""
        self.load_all_configurations()
        return {
            model_id: config for model_id, config in self._model_catalog.items()
            if config.provider == provider
        }
    
    def get_models_by_capability(self, capability: ModelCapability) -> Dict[str, ModelConfiguration]:
        """Get all models that support a specific capability"""
        self.load_all_configurations()
        return {
            model_id: config for model_id, config in self._model_catalog.items()
            if capability in config.capabilities
        }
    
    def get_provider_config(self, provider: str) -> Optional[ProviderConfiguration]:
        """Get configuration for a specific provider"""
        self.load_all_configurations()
        return self._provider_configs.get(provider)
    
    def get_all_providers(self) -> Dict[str, ProviderConfiguration]:
        """Get all provider configurations"""
        self.load_all_configurations()
        return self._provider_configs.copy()
    
    def get_capability_mapping(self, task_type: str) -> Optional[CapabilityMapping]:
        """Get capability mapping for a task type"""
        self.load_all_configurations()
        return self._capability_mappings.get(task_type)
    
    def get_all_capability_mappings(self) -> Dict[str, CapabilityMapping]:
        """Get all capability mappings"""
        self.load_all_configurations()
        return self._capability_mappings.copy()
    
    def get_model_pricing(self, model_id: str, provider: str = None) -> Optional[ModelPricing]:
        """Get pricing for a specific model"""
        self.load_all_configurations()
        
        # First check dedicated pricing config
        if provider and provider in self._pricing_config:
            if model_id in self._pricing_config[provider]:
                return self._pricing_config[provider][model_id]
        
        # Fallback to model catalog pricing
        model_config = self.get_model_config(model_id)
        if model_config:
            return model_config.pricing
        
        return None
    
    def get_recommended_models(self, 
                             task_type: str, 
                             max_cost: Optional[float] = None,
                             min_quality: Optional[float] = None,
                             provider_preference: Optional[List[str]] = None) -> List[str]:
        """Get recommended models for a task type with optional constraints"""
        self.load_all_configurations()
        
        # Get capability mapping
        mapping = self.get_capability_mapping(task_type)
        if not mapping:
            logger.warning("No capability mapping found for task type", task_type=task_type)
            return []
        
        # Get candidate providers
        candidate_providers = mapping.preferred_providers.copy()
        if provider_preference:
            # Reorder based on preferences
            preferred = [p for p in provider_preference if p in candidate_providers]
            others = [p for p in candidate_providers if p not in provider_preference]
            candidate_providers = preferred + others
        
        # Find suitable models
        recommended = []
        for provider in candidate_providers:
            models = self.get_models_by_provider(provider)
            for model_id, model_config in models.items():
                # Check quality constraint
                if min_quality and model_config.quality_score < min_quality:
                    continue
                
                # Check cost constraint
                if max_cost:
                    pricing = self.get_model_pricing(model_id, provider)
                    if pricing and float(pricing.input_cost_per_1k) > max_cost:
                        continue
                
                recommended.append(model_id)
        
        return recommended
    
    def reload_configurations(self) -> None:
        """Reload all configurations from files"""
        self._model_catalog.clear()
        self._provider_configs.clear()
        self._capability_mappings.clear()
        self._pricing_config.clear()
        self._loaded = False
        self.load_all_configurations(force_reload=True)
        logger.info("Model configurations reloaded")


# Global instance for easy access
_global_config_manager: Optional[ModelConfigManager] = None


def get_model_config_manager(config_dir: Optional[Union[str, Path]] = None) -> ModelConfigManager:
    """Get the global model configuration manager instance"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ModelConfigManager(config_dir)
    
    return _global_config_manager


def reload_model_configurations() -> None:
    """Reload model configurations from files"""
    global _global_config_manager
    
    if _global_config_manager:
        _global_config_manager.reload_configurations()
    else:
        _global_config_manager = ModelConfigManager()


# Convenience functions
def get_model_pricing(model_id: str, provider: str = None) -> Optional[ModelPricing]:
    """Get pricing for a model"""
    manager = get_model_config_manager()
    return manager.get_model_pricing(model_id, provider)


def get_recommended_models_for_task(task_type: str, **constraints) -> List[str]:
    """Get recommended models for a task type"""
    manager = get_model_config_manager()
    return manager.get_recommended_models(task_type, **constraints)