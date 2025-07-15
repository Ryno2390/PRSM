#!/usr/bin/env python3
"""
Ollama Model Discovery and Integration
=====================================

This module discovers and integrates Ollama models from your external drive
('/Volumes/My Passport') for NWTN optimization, providing a practical approach
to leveraging your existing model collection.

Key Features:
- Automatically discovers Ollama models on external drive
- Analyzes model capabilities and NWTN suitability
- Provides optimization recommendations
- Integrates with NWTN optimization pipeline
- Supports direct model loading and fine-tuning

Ollama Model Support:
- Llama 2/3 variants (7B, 13B, 70B)
- Mistral models (7B, 8x7B Mixtral)
- Code Llama models
- Phi models
- Gemma models
- Custom fine-tuned models

Discovery Process:
1. Scan external drive for Ollama model files
2. Parse model metadata and capabilities
3. Assess NWTN optimization potential
4. Generate optimization recommendations
5. Provide integration pathway

Usage:
    from prsm.nwtn.ollama_model_discovery import OllamaModelDiscovery
    
    discovery = OllamaModelDiscovery()
    await discovery.initialize()
    
    # Discover models on external drive
    models = await discovery.discover_models("/Volumes/My Passport")
    
    # Get NWTN optimization recommendations
    recommendations = await discovery.get_nwtn_recommendations(models)
    
    # Optimize selected model
    optimized_model = await discovery.optimize_model_for_nwtn(
        model_path=models[0].path,
        output_path="models/nwtn_optimized/"
    )
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
from datetime import datetime, timezone
import structlog

from prsm.nwtn.open_source_optimization import NWTNOpenSourceOptimizer, BaseModelType, OptimizationStrategy
from prsm.nwtn.nwtn_optimized_voicebox import NWTNReasoningMode, ScientificDomain
from prsm.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class OllamaModelFamily(str, Enum):
    """Ollama model families"""
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    CODE_LLAMA = "codellama"
    PHI = "phi"
    GEMMA = "gemma"
    VICUNA = "vicuna"
    ORCA = "orca"
    NEURAL_CHAT = "neural-chat"
    STARLING = "starling"
    OPENCHAT = "openchat"
    CUSTOM = "custom"


@dataclass
class OllamaModelInfo:
    """Information about discovered Ollama model"""
    name: str
    family: OllamaModelFamily
    path: str
    size_gb: float
    parameters: str
    quantization: str
    capabilities: Dict[str, float]
    nwtn_suitability: float
    optimization_potential: float
    last_modified: datetime
    file_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NWTNOptimizationRecommendation:
    """NWTN optimization recommendation for Ollama model"""
    model_info: OllamaModelInfo
    recommended_strategy: OptimizationStrategy
    expected_performance: Dict[str, float]
    resource_requirements: Dict[str, Any]
    optimization_time_estimate: float
    priority_score: float
    reasoning: str
    specific_improvements: List[str]
    target_domains: List[ScientificDomain]
    reasoning_modes: List[NWTNReasoningMode]


class OllamaModelDiscovery:
    """
    Ollama Model Discovery and Integration
    
    Discovers and analyzes Ollama models from your external drive,
    providing recommendations for NWTN optimization and integration.
    """
    
    def __init__(self):
        self.external_drive_path = "/Volumes/My Passport"
        self.ollama_models_path = None
        self.nwtn_optimizer = None
        
        # Model discovery state
        self.discovered_models: List[OllamaModelInfo] = []
        self.model_capabilities_db = {}
        self.optimization_recommendations: List[NWTNOptimizationRecommendation] = []
        
        # Model family characteristics
        self.family_characteristics = self._initialize_family_characteristics()
        
        # Discovery patterns
        self.model_file_patterns = [
            "*.bin",
            "*.safetensors", 
            "*.gguf",
            "*.ggml",
            "model.json",
            "config.json"
        ]
        
        logger.info("Ollama Model Discovery initialized")
    
    async def initialize(self):
        """Initialize the discovery system"""
        try:
            logger.info("🔍 Initializing Ollama Model Discovery...")
            
            # Check if external drive is connected
            if not os.path.exists(self.external_drive_path):
                logger.warning(f"⚠️ External drive not found at {self.external_drive_path}")
                return False
            
            # Find Ollama models directory
            self.ollama_models_path = await self._find_ollama_models_directory()
            
            if not self.ollama_models_path:
                logger.warning("⚠️ Ollama models directory not found on external drive")
                return False
            
            # Initialize NWTN optimizer
            self.nwtn_optimizer = NWTNOpenSourceOptimizer()
            await self.nwtn_optimizer.initialize()
            
            logger.info("✅ Ollama Model Discovery initialized successfully")
            logger.info(f"📁 Ollama models path: {self.ollama_models_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama discovery: {e}")
            return False
    
    async def discover_models(self, drive_path: Optional[str] = None) -> List[OllamaModelInfo]:
        """
        Discover all Ollama models on the external drive
        
        Returns list of discovered models with their capabilities and
        NWTN optimization potential.
        """
        try:
            search_path = drive_path or self.external_drive_path
            logger.info(f"🔍 Discovering Ollama models in {search_path}...")
            
            # Find all potential model directories
            model_directories = await self._find_model_directories(search_path)
            
            discovered_models = []
            
            for model_dir in model_directories:
                logger.info(f"📁 Analyzing model directory: {model_dir}")
                
                # Extract model information
                model_info = await self._analyze_model_directory(model_dir)
                
                if model_info:
                    discovered_models.append(model_info)
                    logger.info(f"✅ Discovered model: {model_info.name} ({model_info.family.value})")
            
            # Sort by NWTN suitability
            discovered_models.sort(key=lambda x: x.nwtn_suitability, reverse=True)
            
            # Store discovered models
            self.discovered_models = discovered_models
            
            logger.info(f"🎯 Discovery complete: {len(discovered_models)} models found")
            
            # Log top models
            for i, model in enumerate(discovered_models[:5]):
                logger.info(f"  {i+1}. {model.name}: {model.nwtn_suitability:.3f} suitability")
            
            return discovered_models
            
        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            return []
    
    async def get_nwtn_recommendations(
        self,
        models: Optional[List[OllamaModelInfo]] = None,
        target_domains: Optional[List[ScientificDomain]] = None,
        resource_constraints: Optional[Dict[str, Any]] = None
    ) -> List[NWTNOptimizationRecommendation]:
        """
        Get NWTN optimization recommendations for discovered models
        
        Analyzes each model and provides specific recommendations for
        NWTN optimization including strategy, timeline, and expected results.
        """
        try:
            models_to_analyze = models or self.discovered_models
            
            if not models_to_analyze:
                logger.warning("No models to analyze for NWTN recommendations")
                return []
            
            logger.info(f"🎯 Generating NWTN recommendations for {len(models_to_analyze)} models...")
            
            recommendations = []
            
            for model in models_to_analyze:
                logger.info(f"📊 Analyzing {model.name} for NWTN optimization...")
                
                # Generate recommendation
                recommendation = await self._generate_nwtn_recommendation(
                    model, target_domains, resource_constraints
                )
                
                recommendations.append(recommendation)
                
                logger.info(f"✅ Recommendation generated for {model.name}")
                logger.info(f"   Strategy: {recommendation.recommended_strategy.value}")
                logger.info(f"   Priority: {recommendation.priority_score:.3f}")
                logger.info(f"   Time estimate: {recommendation.optimization_time_estimate:.1f} hours")
            
            # Sort by priority score
            recommendations.sort(key=lambda x: x.priority_score, reverse=True)
            
            # Store recommendations
            self.optimization_recommendations = recommendations
            
            # Log top recommendations
            logger.info(f"🏆 Top NWTN optimization recommendations:")
            for i, rec in enumerate(recommendations[:3]):
                logger.info(f"  {i+1}. {rec.model_info.name}: {rec.priority_score:.3f} priority")
                logger.info(f"     Strategy: {rec.recommended_strategy.value}")
                logger.info(f"     Reasoning: {rec.reasoning[:100]}...")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate NWTN recommendations: {e}")
            return []
    
    async def optimize_model_for_nwtn(
        self,
        model_path: str,
        output_path: str,
        optimization_strategy: Optional[OptimizationStrategy] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize selected Ollama model for NWTN integration
        
        Takes an Ollama model and optimizes it for NWTN's multi-modal
        reasoning capabilities through fine-tuning and specialization.
        """
        try:
            logger.info(f"🔧 Starting NWTN optimization for model: {model_path}")
            
            # Find model info
            model_info = await self._find_model_info_by_path(model_path)
            if not model_info:
                raise ValueError(f"Model not found in discovery cache: {model_path}")
            
            # Get optimization recommendation if strategy not specified
            if not optimization_strategy:
                recommendations = await self.get_nwtn_recommendations([model_info])
                if recommendations:
                    optimization_strategy = recommendations[0].recommended_strategy
                else:
                    optimization_strategy = OptimizationStrategy.LORA
            
            # Prepare optimization configuration
            optimization_config = await self._prepare_optimization_config(
                model_info, optimization_strategy, config_overrides
            )
            
            # Convert Ollama model to standard format
            converted_model_path = await self._convert_ollama_model(model_path, optimization_config)
            
            # Map to base model type for optimizer
            base_model_type = await self._map_to_base_model_type(model_info)
            
            # Create NWTN training dataset
            training_data_path = await self._create_nwtn_training_data(model_info)
            
            # Optimize using NWTN optimizer
            optimization_result = await self.nwtn_optimizer.optimize_for_nwtn(
                base_model=base_model_type,
                nwtn_data_path=training_data_path,
                output_path=output_path,
                optimization_strategy=optimization_strategy,
                config_overrides=config_overrides
            )
            
            # Add Ollama-specific metadata
            optimization_result.optimization_metadata.update({
                "original_ollama_model": model_info.name,
                "original_model_path": model_path,
                "model_family": model_info.family.value,
                "quantization": model_info.quantization,
                "original_size_gb": model_info.size_gb
            })
            
            logger.info(f"✅ NWTN optimization completed successfully!")
            logger.info(f"📊 Integration score: {optimization_result.nwtn_integration_score:.3f}")
            logger.info(f"💾 Optimized model saved to: {optimization_result.optimized_model_path}")
            
            return {
                "optimization_result": optimization_result,
                "original_model_info": model_info,
                "optimization_config": optimization_config
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize model for NWTN: {e}")
            raise
    
    async def get_model_comparison(self, models: Optional[List[OllamaModelInfo]] = None) -> Dict[str, Any]:
        """
        Compare discovered models for NWTN suitability
        
        Provides comprehensive comparison of models across various
        dimensions relevant to NWTN optimization.
        """
        try:
            models_to_compare = models or self.discovered_models
            
            if not models_to_compare:
                logger.warning("No models to compare")
                return {}
            
            logger.info(f"📊 Comparing {len(models_to_compare)} models for NWTN suitability...")
            
            comparison = {
                "model_count": len(models_to_compare),
                "comparison_matrix": {},
                "rankings": {},
                "recommendations": {},
                "summary": {}
            }
            
            # Create comparison matrix
            for model in models_to_compare:
                comparison["comparison_matrix"][model.name] = {
                    "family": model.family.value,
                    "size_gb": model.size_gb,
                    "parameters": model.parameters,
                    "quantization": model.quantization,
                    "nwtn_suitability": model.nwtn_suitability,
                    "optimization_potential": model.optimization_potential,
                    "capabilities": model.capabilities
                }
            
            # Generate rankings
            comparison["rankings"] = {
                "by_nwtn_suitability": sorted(
                    [(m.name, m.nwtn_suitability) for m in models_to_compare],
                    key=lambda x: x[1], reverse=True
                ),
                "by_optimization_potential": sorted(
                    [(m.name, m.optimization_potential) for m in models_to_compare],
                    key=lambda x: x[1], reverse=True
                ),
                "by_size": sorted(
                    [(m.name, m.size_gb) for m in models_to_compare],
                    key=lambda x: x[1]
                )
            }
            
            # Generate recommendations
            best_overall = max(models_to_compare, key=lambda x: x.nwtn_suitability)
            best_potential = max(models_to_compare, key=lambda x: x.optimization_potential)
            smallest_viable = min(
                [m for m in models_to_compare if m.nwtn_suitability >= 0.7],
                key=lambda x: x.size_gb,
                default=None
            )
            
            comparison["recommendations"] = {
                "best_overall": {
                    "model": best_overall.name,
                    "reason": f"Highest NWTN suitability ({best_overall.nwtn_suitability:.3f})"
                },
                "best_potential": {
                    "model": best_potential.name,
                    "reason": f"Highest optimization potential ({best_potential.optimization_potential:.3f})"
                },
                "most_efficient": {
                    "model": smallest_viable.name if smallest_viable else "None",
                    "reason": f"Smallest viable model ({smallest_viable.size_gb:.1f}GB)" if smallest_viable else "No viable models under size threshold"
                }
            }
            
            # Generate summary
            comparison["summary"] = {
                "total_models": len(models_to_compare),
                "families_represented": len(set(m.family for m in models_to_compare)),
                "average_nwtn_suitability": sum(m.nwtn_suitability for m in models_to_compare) / len(models_to_compare),
                "total_storage_gb": sum(m.size_gb for m in models_to_compare),
                "viable_models": len([m for m in models_to_compare if m.nwtn_suitability >= 0.7])
            }
            
            logger.info(f"✅ Model comparison completed")
            logger.info(f"📈 Best overall: {comparison['recommendations']['best_overall']['model']}")
            logger.info(f"🎯 Best potential: {comparison['recommendations']['best_potential']['model']}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}
    
    async def export_model_inventory(self, output_path: str) -> str:
        """Export complete model inventory to JSON file"""
        try:
            inventory = {
                "discovery_timestamp": datetime.now(timezone.utc),
                "external_drive_path": self.external_drive_path,
                "ollama_models_path": self.ollama_models_path,
                "discovered_models": [
                    {
                        "name": model.name,
                        "family": model.family.value,
                        "path": model.path,
                        "size_gb": model.size_gb,
                        "parameters": model.parameters,
                        "quantization": model.quantization,
                        "nwtn_suitability": model.nwtn_suitability,
                        "optimization_potential": model.optimization_potential,
                        "capabilities": model.capabilities,
                        "metadata": model.metadata
                    }
                    for model in self.discovered_models
                ],
                "optimization_recommendations": [
                    {
                        "model_name": rec.model_info.name,
                        "recommended_strategy": rec.recommended_strategy.value,
                        "priority_score": rec.priority_score,
                        "optimization_time_estimate": rec.optimization_time_estimate,
                        "reasoning": rec.reasoning,
                        "expected_performance": rec.expected_performance,
                        "resource_requirements": rec.resource_requirements
                    }
                    for rec in self.optimization_recommendations
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(inventory, f, indent=2, default=str)
            
            logger.info(f"📤 Model inventory exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export inventory: {e}")
            raise
    
    # === Private Methods ===
    
    def _initialize_family_characteristics(self) -> Dict[OllamaModelFamily, Dict[str, Any]]:
        """Initialize characteristics of different model families"""
        return {
            OllamaModelFamily.LLAMA2: {
                "reasoning_strength": 0.85,
                "scientific_knowledge": 0.78,
                "instruction_following": 0.92,
                "nwtn_base_suitability": 0.85,
                "optimization_potential": 0.88
            },
            OllamaModelFamily.LLAMA3: {
                "reasoning_strength": 0.90,
                "scientific_knowledge": 0.82,
                "instruction_following": 0.94,
                "nwtn_base_suitability": 0.90,
                "optimization_potential": 0.92
            },
            OllamaModelFamily.MISTRAL: {
                "reasoning_strength": 0.87,
                "scientific_knowledge": 0.80,
                "instruction_following": 0.89,
                "nwtn_base_suitability": 0.87,
                "optimization_potential": 0.90
            },
            OllamaModelFamily.MIXTRAL: {
                "reasoning_strength": 0.92,
                "scientific_knowledge": 0.85,
                "instruction_following": 0.91,
                "nwtn_base_suitability": 0.90,
                "optimization_potential": 0.93
            },
            OllamaModelFamily.CODE_LLAMA: {
                "reasoning_strength": 0.93,
                "scientific_knowledge": 0.75,
                "instruction_following": 0.88,
                "nwtn_base_suitability": 0.83,
                "optimization_potential": 0.87
            },
            OllamaModelFamily.PHI: {
                "reasoning_strength": 0.82,
                "scientific_knowledge": 0.77,
                "instruction_following": 0.85,
                "nwtn_base_suitability": 0.80,
                "optimization_potential": 0.83
            },
            OllamaModelFamily.GEMMA: {
                "reasoning_strength": 0.84,
                "scientific_knowledge": 0.79,
                "instruction_following": 0.87,
                "nwtn_base_suitability": 0.82,
                "optimization_potential": 0.85
            }
        }
    
    async def _find_ollama_models_directory(self) -> Optional[str]:
        """Find Ollama models directory on external drive"""
        try:
            # Common Ollama model locations
            possible_paths = [
                os.path.join(self.external_drive_path, ".ollama", "models"),
                os.path.join(self.external_drive_path, "ollama", "models"),
                os.path.join(self.external_drive_path, "models"),
                os.path.join(self.external_drive_path, "AI_Models", "ollama"),
                os.path.join(self.external_drive_path, "LLM", "ollama"),
                os.path.join(self.external_drive_path, "ML", "ollama")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"📁 Found Ollama models directory: {path}")
                    return path
            
            # Search for directories containing model files
            for root, dirs, files in os.walk(self.external_drive_path):
                if any(f.endswith(('.bin', '.gguf', '.ggml', '.safetensors')) for f in files):
                    if 'ollama' in root.lower() or 'model' in root.lower():
                        logger.info(f"📁 Found potential Ollama models directory: {root}")
                        return root
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find Ollama models directory: {e}")
            return None
    
    async def _find_model_directories(self, search_path: str) -> List[str]:
        """Find all potential model directories"""
        try:
            model_directories = []
            
            # Walk through directory structure
            for root, dirs, files in os.walk(search_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                # Check if directory contains model files
                has_model_files = any(
                    f.endswith(('.bin', '.gguf', '.ggml', '.safetensors')) or
                    f in ['config.json', 'model.json', 'tokenizer.json']
                    for f in files
                )
                
                if has_model_files:
                    model_directories.append(root)
            
            logger.info(f"📁 Found {len(model_directories)} potential model directories")
            return model_directories
            
        except Exception as e:
            logger.error(f"Failed to find model directories: {e}")
            return []
    
    async def _analyze_model_directory(self, model_dir: str) -> Optional[OllamaModelInfo]:
        """Analyze individual model directory"""
        try:
            # Extract model name from directory structure
            model_name = await self._extract_model_name(model_dir)
            
            # Determine model family
            model_family = await self._determine_model_family(model_name, model_dir)
            
            # Calculate directory size
            size_gb = await self._calculate_directory_size(model_dir)
            
            # Extract model parameters
            parameters = await self._extract_model_parameters(model_name, model_dir)
            
            # Determine quantization
            quantization = await self._determine_quantization(model_dir)
            
            # Calculate capabilities
            capabilities = await self._calculate_model_capabilities(model_family, parameters)
            
            # Calculate NWTN suitability
            nwtn_suitability = await self._calculate_nwtn_suitability(
                model_family, parameters, quantization, size_gb
            )
            
            # Calculate optimization potential
            optimization_potential = await self._calculate_optimization_potential(
                model_family, nwtn_suitability, size_gb
            )
            
            # Get file modification time
            last_modified = datetime.fromtimestamp(
                os.path.getmtime(model_dir), tz=timezone.utc
            )
            
            # Calculate file hash
            file_hash = await self._calculate_directory_hash(model_dir)
            
            # Create model info
            model_info = OllamaModelInfo(
                name=model_name,
                family=model_family,
                path=model_dir,
                size_gb=size_gb,
                parameters=parameters,
                quantization=quantization,
                capabilities=capabilities,
                nwtn_suitability=nwtn_suitability,
                optimization_potential=optimization_potential,
                last_modified=last_modified,
                file_hash=file_hash,
                metadata=await self._extract_model_metadata(model_dir)
            )
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to analyze model directory {model_dir}: {e}")
            return None
    
    async def _extract_model_name(self, model_dir: str) -> str:
        """Extract model name from directory path"""
        # Extract from path components
        path_parts = model_dir.split(os.sep)
        
        # Look for model name in path
        for part in reversed(path_parts):
            if any(family.value in part.lower() for family in OllamaModelFamily):
                return part
        
        # Fallback to directory name
        return os.path.basename(model_dir)
    
    async def _determine_model_family(self, model_name: str, model_dir: str) -> OllamaModelFamily:
        """Determine model family from name and directory"""
        name_lower = model_name.lower()
        
        # Check name against known families
        if "llama3" in name_lower or "llama-3" in name_lower:
            return OllamaModelFamily.LLAMA3
        elif "llama2" in name_lower or "llama-2" in name_lower:
            return OllamaModelFamily.LLAMA2
        elif "mixtral" in name_lower:
            return OllamaModelFamily.MIXTRAL
        elif "mistral" in name_lower:
            return OllamaModelFamily.MISTRAL
        elif "codellama" in name_lower or "code-llama" in name_lower:
            return OllamaModelFamily.CODE_LLAMA
        elif "phi" in name_lower:
            return OllamaModelFamily.PHI
        elif "gemma" in name_lower:
            return OllamaModelFamily.GEMMA
        elif "vicuna" in name_lower:
            return OllamaModelFamily.VICUNA
        elif "orca" in name_lower:
            return OllamaModelFamily.ORCA
        else:
            return OllamaModelFamily.CUSTOM
    
    async def _calculate_directory_size(self, directory: str) -> float:
        """Calculate directory size in GB"""
        try:
            total_size = 0
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            
            return total_size / (1024 ** 3)  # Convert to GB
            
        except Exception as e:
            logger.error(f"Failed to calculate directory size: {e}")
            return 0.0
    
    async def _extract_model_parameters(self, model_name: str, model_dir: str) -> str:
        """Extract model parameters from name or config"""
        name_lower = model_name.lower()
        
        # Look for parameter indicators in name
        if "70b" in name_lower:
            return "70B"
        elif "13b" in name_lower:
            return "13B"
        elif "7b" in name_lower:
            return "7B"
        elif "3b" in name_lower:
            return "3B"
        elif "1b" in name_lower:
            return "1B"
        
        # Try to read from config file
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'num_parameters' in config:
                        return f"{config['num_parameters'] / 1e9:.1f}B"
            except:
                pass
        
        return "Unknown"
    
    async def _determine_quantization(self, model_dir: str) -> str:
        """Determine quantization type from files"""
        # Check for specific file types
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.q4_0.bin') or 'q4_0' in file:
                    return "Q4_0"
                elif file.endswith('.q4_1.bin') or 'q4_1' in file:
                    return "Q4_1"
                elif file.endswith('.q5_0.bin') or 'q5_0' in file:
                    return "Q5_0"
                elif file.endswith('.q5_1.bin') or 'q5_1' in file:
                    return "Q5_1"
                elif file.endswith('.q8_0.bin') or 'q8_0' in file:
                    return "Q8_0"
                elif file.endswith('.f16.bin') or 'f16' in file:
                    return "F16"
                elif file.endswith('.f32.bin') or 'f32' in file:
                    return "F32"
                elif file.endswith('.gguf'):
                    return "GGUF"
                elif file.endswith('.ggml'):
                    return "GGML"
                elif file.endswith('.safetensors'):
                    return "SafeTensors"
        
        return "Unknown"
    
    async def _calculate_model_capabilities(self, family: OllamaModelFamily, parameters: str) -> Dict[str, float]:
        """Calculate model capabilities based on family and parameters"""
        base_capabilities = self.family_characteristics.get(family, {
            "reasoning_strength": 0.7,
            "scientific_knowledge": 0.7,
            "instruction_following": 0.7
        })
        
        # Parameter size adjustments
        parameter_multiplier = 1.0
        if "70B" in parameters:
            parameter_multiplier = 1.3
        elif "13B" in parameters:
            parameter_multiplier = 1.1
        elif "7B" in parameters:
            parameter_multiplier = 1.0
        elif "3B" in parameters:
            parameter_multiplier = 0.9
        elif "1B" in parameters:
            parameter_multiplier = 0.8
        
        return {
            "reasoning_strength": min(base_capabilities.get("reasoning_strength", 0.7) * parameter_multiplier, 1.0),
            "scientific_knowledge": min(base_capabilities.get("scientific_knowledge", 0.7) * parameter_multiplier, 1.0),
            "instruction_following": min(base_capabilities.get("instruction_following", 0.7) * parameter_multiplier, 1.0)
        }
    
    async def _calculate_nwtn_suitability(self, family: OllamaModelFamily, parameters: str, quantization: str, size_gb: float) -> float:
        """Calculate NWTN suitability score"""
        base_suitability = self.family_characteristics.get(family, {}).get("nwtn_base_suitability", 0.7)
        
        # Parameter size bonus
        parameter_bonus = 0.0
        if "70B" in parameters:
            parameter_bonus = 0.15
        elif "13B" in parameters:
            parameter_bonus = 0.10
        elif "7B" in parameters:
            parameter_bonus = 0.05
        
        # Quantization penalty
        quantization_penalty = 0.0
        if quantization in ["Q4_0", "Q4_1"]:
            quantization_penalty = 0.05
        elif quantization in ["Q5_0", "Q5_1"]:
            quantization_penalty = 0.02
        
        # Size efficiency bonus (more efficient models get slight bonus)
        size_efficiency = 1.0
        if size_gb < 5.0:
            size_efficiency = 1.05
        elif size_gb > 30.0:
            size_efficiency = 0.95
        
        suitability = (base_suitability + parameter_bonus - quantization_penalty) * size_efficiency
        return min(max(suitability, 0.0), 1.0)
    
    async def _calculate_optimization_potential(self, family: OllamaModelFamily, nwtn_suitability: float, size_gb: float) -> float:
        """Calculate optimization potential"""
        base_potential = self.family_characteristics.get(family, {}).get("optimization_potential", 0.8)
        
        # Higher suitability = higher potential
        suitability_bonus = nwtn_suitability * 0.2
        
        # Size considerations (medium sized models often have best potential)
        size_factor = 1.0
        if 5.0 <= size_gb <= 25.0:  # Sweet spot
            size_factor = 1.1
        elif size_gb > 50.0:  # Very large models harder to optimize
            size_factor = 0.9
        
        potential = (base_potential + suitability_bonus) * size_factor
        return min(max(potential, 0.0), 1.0)
    
    async def _calculate_directory_hash(self, directory: str) -> str:
        """Calculate hash of directory contents"""
        try:
            hasher = hashlib.sha256()
            
            # Hash directory structure and key files
            for root, dirs, files in os.walk(directory):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    hasher.update(file_path.encode())
                    
                    # Hash file size and modification time
                    if os.path.exists(file_path):
                        stat = os.stat(file_path)
                        hasher.update(str(stat.st_size).encode())
                        hasher.update(str(stat.st_mtime).encode())
            
            return hasher.hexdigest()[:16]  # Short hash
            
        except Exception as e:
            logger.error(f"Failed to calculate directory hash: {e}")
            return "unknown"
    
    async def _extract_model_metadata(self, model_dir: str) -> Dict[str, Any]:
        """Extract additional model metadata"""
        metadata = {}
        
        # Look for config files
        config_files = ['config.json', 'model.json', 'tokenizer.json']
        for config_file in config_files:
            config_path = os.path.join(model_dir, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                        metadata[config_file] = config_data
                except:
                    pass
        
        # Directory structure
        metadata['directory_structure'] = []
        for root, dirs, files in os.walk(model_dir):
            rel_root = os.path.relpath(root, model_dir)
            metadata['directory_structure'].append({
                'path': rel_root,
                'files': files,
                'subdirs': dirs
            })
        
        return metadata
    
    async def _generate_nwtn_recommendation(
        self,
        model_info: OllamaModelInfo,
        target_domains: Optional[List[ScientificDomain]],
        resource_constraints: Optional[Dict[str, Any]]
    ) -> NWTNOptimizationRecommendation:
        """Generate NWTN optimization recommendation for model"""
        
        # Determine optimal strategy based on model characteristics
        if model_info.size_gb > 30.0:
            recommended_strategy = OptimizationStrategy.QLORA
        elif model_info.size_gb > 15.0:
            recommended_strategy = OptimizationStrategy.LORA
        else:
            recommended_strategy = OptimizationStrategy.FULL_FINE_TUNING
        
        # Estimate performance improvements
        expected_performance = {
            "reasoning_improvement": 0.15 * model_info.optimization_potential,
            "scientific_accuracy": 0.18 * model_info.optimization_potential,
            "instruction_following": 0.12 * model_info.optimization_potential,
            "nwtn_integration": 0.20 * model_info.optimization_potential
        }
        
        # Calculate resource requirements
        resource_requirements = {
            "gpu_memory_gb": model_info.size_gb * 1.5,
            "training_time_hours": 2.0 + (model_info.size_gb * 0.1),
            "storage_gb": model_info.size_gb * 2.0,
            "compute_requirements": "Medium" if model_info.size_gb < 20 else "High"
        }
        
        # Calculate priority score
        priority_score = (
            model_info.nwtn_suitability * 0.4 +
            model_info.optimization_potential * 0.3 +
            (1.0 - model_info.size_gb / 100.0) * 0.2 +  # Size efficiency
            model_info.capabilities.get("reasoning_strength", 0.7) * 0.1
        )
        
        # Generate reasoning
        reasoning = f"Model {model_info.name} ({model_info.family.value}) shows {model_info.nwtn_suitability:.1%} NWTN suitability with {model_info.optimization_potential:.1%} optimization potential. "
        reasoning += f"Recommended {recommended_strategy.value} strategy based on {model_info.size_gb:.1f}GB size. "
        reasoning += f"Expected {expected_performance['reasoning_improvement']:.1%} reasoning improvement."
        
        # Specific improvements
        specific_improvements = [
            f"Enhanced {mode.value} reasoning" for mode in NWTNReasoningMode
        ][:3]  # Top 3
        
        # Target domains (default to top domains if not specified)
        target_domains = target_domains or [
            ScientificDomain.PHYSICS,
            ScientificDomain.CHEMISTRY,
            ScientificDomain.BIOLOGY,
            ScientificDomain.ENGINEERING
        ]
        
        # Reasoning modes (all modes)
        reasoning_modes = [mode for mode in NWTNReasoningMode]
        
        return NWTNOptimizationRecommendation(
            model_info=model_info,
            recommended_strategy=recommended_strategy,
            expected_performance=expected_performance,
            resource_requirements=resource_requirements,
            optimization_time_estimate=resource_requirements["training_time_hours"],
            priority_score=priority_score,
            reasoning=reasoning,
            specific_improvements=specific_improvements,
            target_domains=target_domains,
            reasoning_modes=reasoning_modes
        )
    
    async def _find_model_info_by_path(self, model_path: str) -> Optional[OllamaModelInfo]:
        """Find model info by path"""
        for model in self.discovered_models:
            if model.path == model_path:
                return model
        return None
    
    async def _prepare_optimization_config(
        self,
        model_info: OllamaModelInfo,
        optimization_strategy: OptimizationStrategy,
        config_overrides: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare optimization configuration"""
        config = {
            "model_family": model_info.family.value,
            "model_size": model_info.parameters,
            "quantization": model_info.quantization,
            "optimization_strategy": optimization_strategy.value,
            "target_domains": ["physics", "chemistry", "biology", "engineering"],
            "reasoning_modes": ["analogical", "causal", "deductive", "inductive"]
        }
        
        if config_overrides:
            config.update(config_overrides)
        
        return config
    
    async def _convert_ollama_model(self, model_path: str, optimization_config: Dict[str, Any]) -> str:
        """Convert Ollama model to standard format"""
        # Placeholder - would implement actual conversion
        logger.info(f"🔄 Converting Ollama model: {model_path}")
        return model_path
    
    async def _map_to_base_model_type(self, model_info: OllamaModelInfo) -> BaseModelType:
        """Map Ollama model to base model type"""
        if model_info.family == OllamaModelFamily.LLAMA2:
            if "13B" in model_info.parameters:
                return BaseModelType.LLAMA2_13B
            else:
                return BaseModelType.LLAMA2_7B
        elif model_info.family == OllamaModelFamily.LLAMA3:
            if "70B" in model_info.parameters:
                return BaseModelType.LLAMA3_70B
            else:
                return BaseModelType.LLAMA3_8B
        elif model_info.family == OllamaModelFamily.MISTRAL:
            return BaseModelType.MISTRAL_7B
        elif model_info.family == OllamaModelFamily.MIXTRAL:
            return BaseModelType.MISTRAL_8X7B
        elif model_info.family == OllamaModelFamily.CODE_LLAMA:
            if "13B" in model_info.parameters:
                return BaseModelType.CODE_LLAMA_13B
            else:
                return BaseModelType.CODE_LLAMA_7B
        else:
            return BaseModelType.LLAMA2_7B  # Default fallback
    
    async def _create_nwtn_training_data(self, model_info: OllamaModelInfo) -> str:
        """Create NWTN training data specific to model"""
        # Placeholder - would create actual training data
        training_data_path = f"training_data/{model_info.name.replace('/', '_')}"
        logger.info(f"📚 Creating NWTN training data: {training_data_path}")
        return training_data_path


# Global discovery instance
_discovery = None

async def get_discovery() -> OllamaModelDiscovery:
    """Get the global discovery instance"""
    global _discovery
    if _discovery is None:
        _discovery = OllamaModelDiscovery()
        await _discovery.initialize()
    return _discovery