"""
PRSM Architecture Generator
Automated generation of optimal student model architectures

The Architecture Generator creates optimal neural network architectures
for distilled models based on teacher analysis, requirements, and
performance targets.
"""

import asyncio
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import structlog

from .models import (
    StudentArchitecture, DistillationRequest, TeacherAnalysis,
    ModelSize, OptimizationTarget, TrainingStrategy
)

from prsm.compute.nwtn.architectures.ssm_core import SSMConfig, get_ssm_reasoner
from prsm.compute.nwtn.architectures.liquid_core import get_liquid_reasoner

logger = structlog.get_logger(__name__)


class ArchitectureGenerator:
    """
    Automated neural architecture generator for distilled models
    
    Generates optimal architectures through:
    
    Architecture Search:
    - Automated neural architecture search (NAS)
    - Constraint-based architecture optimization
    - Multi-objective optimization (accuracy vs efficiency)
    - Hardware-aware architecture design
    
    Optimization Strategies:
    - Layer pruning and compression
    - Attention head reduction
    - Knowledge distillation-aware design
    - Progressive architecture refinement
    
    Performance Prediction:
    - Parameter count estimation
    - Inference speed prediction
    - Memory usage calculation
    - Energy efficiency modeling
    
    Customization Features:
    - Domain-specific architectural patterns
    - Hardware target optimization
    - Quality-efficiency trade-off balancing
    - Scalability and modularity design
    """
    
    def __init__(self):
        # Architecture templates by size
        self.size_templates = {
            ModelSize.TINY: {
                "layer_range": (6, 12),
                "hidden_range": (256, 512),
                "attention_heads_range": (4, 8),
                "ssm_state_range": (8, 16),
                "ssm_conv_range": (2, 4),
                "liquid_hidden_range": (64, 128),
                "intermediate_factor": 2.0,
                "max_parameters": 100_000_000  # 100M
            },
            ModelSize.SMALL: {
                "layer_range": (12, 18),
                "hidden_range": (512, 768),
                "attention_heads_range": (8, 12),
                "ssm_state_range": (16, 32),
                "ssm_conv_range": (4, 4),
                "liquid_hidden_range": (128, 256),
                "intermediate_factor": 3.0,
                "max_parameters": 1_000_000_000  # 1B
            },
            ModelSize.MEDIUM: {
                "layer_range": (18, 24),
                "hidden_range": (768, 1024),
                "attention_heads_range": (12, 16),
                "ssm_state_range": (32, 64),
                "ssm_conv_range": (4, 8),
                "liquid_hidden_range": (256, 512),
                "intermediate_factor": 4.0,
                "max_parameters": 10_000_000_000  # 10B
            },
            ModelSize.LARGE: {
                "layer_range": (24, 32),
                "hidden_range": (1024, 2048),
                "attention_heads_range": (16, 32),
                "ssm_state_range": (64, 128),
                "ssm_conv_range": (8, 8),
                "liquid_hidden_range": (512, 1024),
                "intermediate_factor": 4.0,
                "max_parameters": 100_000_000_000  # 100B+
            }
        }
        
        # Domain-specific architectural patterns
        self.domain_patterns = {
            "medical_research": {
                "attention_pattern": "local_global",
                "specialized_layers": ["medical_terminology", "clinical_reasoning"],
                "knowledge_injection": "domain_adapter",
                "safety_requirements": "high"
            },
            "legal_analysis": {
                "attention_pattern": "hierarchical",
                "specialized_layers": ["legal_reasoning", "precedent_analysis"],
                "knowledge_injection": "citation_aware",
                "safety_requirements": "very_high"
            },
            "scientific_reasoning": {
                "attention_pattern": "multi_hop",
                "specialized_layers": ["hypothesis_generation", "evidence_evaluation"],
                "knowledge_injection": "causal_reasoning",
                "safety_requirements": "medium"
            },
            "code_generation": {
                "attention_pattern": "structural",
                "specialized_layers": ["syntax_awareness", "semantic_analysis"],
                "knowledge_injection": "code_structure",
                "safety_requirements": "medium"
            },
            "creative_writing": {
                "attention_pattern": "creative",
                "specialized_layers": ["style_adaptation", "narrative_flow"],
                "knowledge_injection": "style_transfer",
                "safety_requirements": "medium"
            }
        }
        
        # Optimization target configurations
        self.optimization_configs = {
            OptimizationTarget.SPEED: {
                "layer_penalty": 0.8,
                "attention_penalty": 0.7,
                "parameter_penalty": 0.9,
                "efficiency_weight": 0.6
            },
            OptimizationTarget.ACCURACY: {
                "layer_penalty": 1.2,
                "attention_penalty": 1.1,
                "parameter_penalty": 1.1,
                "efficiency_weight": 0.2
            },
            OptimizationTarget.EFFICIENCY: {
                "layer_penalty": 0.6,
                "attention_penalty": 0.5,
                "parameter_penalty": 0.7,
                "efficiency_weight": 0.8
            },
            OptimizationTarget.SIZE: {
                "layer_penalty": 0.5,
                "attention_penalty": 0.4,
                "parameter_penalty": 0.3,
                "efficiency_weight": 0.9
            },
            OptimizationTarget.BALANCED: {
                "layer_penalty": 1.0,
                "attention_penalty": 1.0,
                "parameter_penalty": 1.0,
                "efficiency_weight": 0.5
            }
        }
    
    async def generate_architecture(
        self, 
        request: DistillationRequest, 
        teacher_analysis: TeacherAnalysis
    ) -> StudentArchitecture:
        """
        Generate optimal student architecture
        
        Args:
            request: Distillation request with requirements
            teacher_analysis: Analysis of teacher model
            
        Returns:
            StudentArchitecture: Optimized architecture specification
        """
        # Handle Pydantic use_enum_values=True conversion
        target_size = ModelSize(request.target_size) if isinstance(request.target_size, str) else request.target_size
        optimization_target = OptimizationTarget(request.optimization_target) if isinstance(request.optimization_target, str) else request.optimization_target

        logger.info("Generating student architecture",
                   target_size=target_size.value,
                   optimization_target=optimization_target.value,
                   domain=request.domain)
        
        try:
            # Initialize architecture
            architecture = StudentArchitecture(
                distillation_request_id=request.request_id,
                model_type=request.target_architecture or "transformer"
            )
            
            # Get base template and constraints
            template = self.size_templates[target_size]
            domain_pattern = self.domain_patterns.get(request.domain, {})
            optimization_config = self.optimization_configs[optimization_target]
            
            # Generate core architecture parameters
            await self._generate_core_parameters(architecture, request, template, optimization_config)
            
            # Add domain-specific components
            await self._add_domain_specific_components(architecture, request, domain_pattern)
            
            # Optimize architecture based on teacher analysis
            await self._optimize_for_teacher(architecture, teacher_analysis, request)
            
            # Add compression and optimization techniques
            await self._add_compression_techniques(architecture, request, optimization_config)
            
            # Calculate performance predictions
            await self._predict_performance(architecture, request, teacher_analysis)
            
            # Generate design rationale
            await self._generate_design_rationale(architecture, request, teacher_analysis, optimization_target)
            
            logger.info("Architecture generation completed",
                       estimated_parameters=architecture.estimated_parameters,
                       predicted_accuracy=architecture.predicted_accuracy,
                       compression_ratio=architecture.compression_ratio)
            
            return architecture
            
        except Exception as e:
            logger.error("Architecture generation failed", error=str(e))
            raise
    
    async def _generate_core_parameters(
        self, 
        architecture: StudentArchitecture, 
        request: DistillationRequest,
        template: Dict[str, Any],
        optimization_config: Dict[str, float]
    ):
        """Generate core architecture parameters"""
        try:
            model_type = architecture.model_type.lower()
            
            # Layer count optimization (common to most)
            if request.layer_count:
                layer_count = request.layer_count
            else:
                layer_count = await self._optimize_layer_count(template, optimization_config)
            architecture.layer_count = layer_count
            
            # Vocabulary size
            vocabulary_size = request.vocabulary_size or 32000
            architecture.vocabulary_size = vocabulary_size

            if model_type == "transformer":
                if request.hidden_size:
                    hidden_size = request.hidden_size
                else:
                    hidden_size = await self._optimize_hidden_size(template, optimization_config)
                
                if request.attention_heads:
                    attention_heads = request.attention_heads
                else:
                    attention_heads = await self._optimize_attention_heads(template, optimization_config, hidden_size)
                
                architecture.hidden_size = hidden_size
                architecture.attention_heads = attention_heads
                architecture.intermediate_size = int(hidden_size * template["intermediate_factor"])
                
            elif model_type == "ssm":
                await self._generate_ssm_parameters(architecture, request, template, optimization_config)
                
            elif model_type == "liquid":
                await self._generate_liquid_parameters(architecture, request, template, optimization_config)
            
            logger.info("Core parameters generated",
                       model_type=model_type,
                       layers=layer_count,
                       hidden_size=architecture.hidden_size)
            
        except Exception as e:
            logger.error("Core parameter generation failed", error=str(e))

    async def _generate_ssm_parameters(
        self,
        architecture: StudentArchitecture,
        request: DistillationRequest,
        template: Dict[str, Any],
        optimization_config: Dict[str, float]
    ):
        """Generate SSM-specific parameters"""
        hidden_size = request.hidden_size or await self._optimize_hidden_size(template, optimization_config)
        d_state = await self._optimize_ssm_state(template, optimization_config)
        d_conv = await self._optimize_ssm_conv(template, optimization_config)
        
        architecture.hidden_size = hidden_size
        architecture.config_overrides.update({
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": 2
        })

    async def _generate_liquid_parameters(
        self,
        architecture: StudentArchitecture,
        request: DistillationRequest,
        template: Dict[str, Any],
        optimization_config: Dict[str, float]
    ):
        """Generate Liquid-specific parameters"""
        # For liquid, hidden_size refers to the ODE state size
        liquid_hidden = await self._optimize_liquid_hidden(template, optimization_config)
        architecture.hidden_size = request.hidden_size or liquid_hidden
        architecture.config_overrides.update({
            "dt": 0.1,
            "solver": "euler"
        })

    async def _add_domain_specific_components(
        self, 
        architecture: StudentArchitecture, 
        request: DistillationRequest,
        domain_pattern: Dict[str, Any]
    ):
        """Add domain-specific architectural components"""
        try:
            specialized_layers = []
            model_type = architecture.model_type.lower()
            
            # Add domain-specific layers
            for layer_type in domain_pattern.get("specialized_layers", []):
                layer_config = {
                    "type": layer_type,
                    "position": "post_block",
                    "size": architecture.hidden_size // 2,
                    "activation": "gelu"
                }
                specialized_layers.append(layer_config)
            
            # Add pattern modifications based on architecture
            if model_type == "transformer":
                attention_pattern = domain_pattern.get("attention_pattern", "standard")
                if attention_pattern != "standard":
                    attention_config = {
                        "type": "attention_modification",
                        "pattern": attention_pattern,
                        "heads_affected": "all",
                        "modification_strength": 0.3
                    }
                    specialized_layers.append(attention_config)
            elif model_type == "ssm":
                # SSM specific scaling enhancements
                specialized_layers.append({
                    "type": "selective_scan_optimization",
                    "strength": 1.2
                })
            
            # Add knowledge injection mechanism
            knowledge_injection = domain_pattern.get("knowledge_injection")
            if knowledge_injection:
                injection_config = {
                    "type": "knowledge_injection",
                    "method": knowledge_injection,
                    "layers": list(range(architecture.layer_count // 2, architecture.layer_count)),
                    "injection_size": architecture.hidden_size // 4
                }
                specialized_layers.append(injection_config)
            
            architecture.specialized_layers = specialized_layers
            
            logger.info("Domain-specific components added",
                       domain=request.domain,
                       specialized_layers=len(specialized_layers))
            
        except Exception as e:
            logger.error("Domain-specific component addition failed", error=str(e))
    
    async def _optimize_for_teacher(
        self, 
        architecture: StudentArchitecture, 
        teacher_analysis: TeacherAnalysis,
        request: DistillationRequest
    ):
        """Optimize architecture based on teacher analysis"""
        try:
            # Adjust based on teacher complexity
            if teacher_analysis.distillation_difficulty > 0.8:
                # High difficulty - need more capacity
                architecture.layer_count = min(
                    architecture.layer_count + 2,
                    self.size_templates[request.target_size]["layer_range"][1]
                )
                architecture.hidden_size = min(
                    int(architecture.hidden_size * 1.2),
                    self.size_templates[request.target_size]["hidden_range"][1]
                )
            
            # Adjust based on critical knowledge areas
            critical_areas = len(teacher_analysis.critical_knowledge_areas)
            if critical_areas > 5:
                # Many critical areas - add knowledge preservation layers
                knowledge_preservation = {
                    "type": "knowledge_preservation",
                    "areas": teacher_analysis.critical_knowledge_areas,
                    "preservation_strength": 0.8,
                    "layer_positions": [architecture.layer_count // 3, 2 * architecture.layer_count // 3]
                }
                architecture.specialized_layers.append(knowledge_preservation)
            
            # Adjust based on reasoning patterns
            complex_reasoning = ["metacognitive_reasoning", "creative_problem_solving"]
            if any(pattern in teacher_analysis.reasoning_patterns for pattern in complex_reasoning):
                # Add reasoning enhancement layer
                reasoning_enhancement = {
                    "type": "reasoning_enhancement",
                    "patterns": [p for p in teacher_analysis.reasoning_patterns if p in complex_reasoning],
                    "enhancement_factor": 1.5,
                    "position": "pre_output"
                }
                architecture.specialized_layers.append(reasoning_enhancement)
            
            logger.info("Teacher-based optimization completed",
                       difficulty=teacher_analysis.distillation_difficulty,
                       critical_areas=critical_areas,
                       reasoning_patterns=len(teacher_analysis.reasoning_patterns))
            
        except Exception as e:
            logger.error("Teacher-based optimization failed", error=str(e))
    
    async def _add_compression_techniques(
        self, 
        architecture: StudentArchitecture, 
        request: DistillationRequest,
        optimization_config: Dict[str, float]
    ):
        """Add compression and optimization techniques"""
        try:
            compression_techniques = []
            optimization_strategies = []
            
            # Add compression based on optimization target
            if request.optimization_target in [OptimizationTarget.SPEED, OptimizationTarget.SIZE, OptimizationTarget.EFFICIENCY]:
                compression_techniques.extend([
                    "weight_quantization",
                    "activation_quantization",
                    "knowledge_distillation"
                ])
                
                if request.optimization_target == OptimizationTarget.SIZE:
                    compression_techniques.extend([
                        "layer_pruning",
                        "attention_head_pruning",
                        "weight_sharing"
                    ])
            
            # Add optimization strategies
            optimization_strategies = [
                "gradient_checkpointing",
                "mixed_precision_training",
                "dynamic_batching"
            ]
            
            if request.optimization_target == OptimizationTarget.SPEED:
                optimization_strategies.extend([
                    "layer_fusion",
                    "operator_optimization",
                    "memory_layout_optimization"
                ])
            
            # Add custom compression techniques from request
            if request.compression_techniques:
                compression_techniques.extend(request.compression_techniques)
            
            architecture.compression_techniques = list(set(compression_techniques))
            architecture.optimization_strategies = list(set(optimization_strategies))
            
            logger.info("Compression techniques added",
                       compression_count=len(compression_techniques),
                       optimization_count=len(optimization_strategies))
            
        except Exception as e:
            logger.error("Compression technique addition failed", error=str(e))
    
    async def _predict_performance(
        self, 
        architecture: StudentArchitecture, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis
    ):
        """Predict performance characteristics of the architecture"""
        try:
            model_type = architecture.model_type.lower()
            h = architecture.hidden_size
            l = architecture.layer_count
            v = architecture.vocabulary_size
            
            # Estimate parameters
            embedding_params = v * h
            
            if model_type == "transformer":
                i = architecture.intermediate_size or (h * 4)
                # Attention (4 * h^2) + FFN (2 * h * i) per layer
                block_params = l * (4 * h * h + 2 * h * i + 2 * h)
            elif model_type == "ssm":
                d_state = architecture.config_overrides.get("d_state", 16)
                # S6 parameters are significantly leaner than attention
                # in_proj (2*h*2*h) + x_proj (h*dt_rank+2*n) + dt_proj (dt_rank*h) + out_proj (h*h)
                block_params = l * (5 * h * h + h * d_state)
            elif model_type == "liquid":
                # Continuous neurons ODE parameters
                block_params = l * (3 * h * h + 3 * h)
            else:
                block_params = l * (h * h * 4) # Default fallback
                
            output_params = v * h
            
            # Add specialized layer parameters
            specialized_params = sum(
                layer.get("size", h // 2) * h
                for layer in architecture.specialized_layers
            )
            
            total_params = embedding_params + block_params + output_params + specialized_params
            architecture.estimated_parameters = int(total_params)
            
            # Estimate model size (2 bytes per parameter for float16)
            size_mb = (total_params * 2) / (1024 * 1024)
            architecture.estimated_size_mb = size_mb
            
            # Estimate inference speed (tokens/second)
            # SSMs and Liquids scale linearly with sequence length O(N)
            # Transformers scale quadratically O(N^2)
            base_speed = 1000
            if model_type in ["ssm", "liquid"]:
                # 2x - 3x faster than Transformers for edge nodes
                arch_multiplier = 2.5
            else:
                arch_multiplier = 1.0
                
            param_factor = max(0.1, 1.0 - (total_params / 1_000_000_000) * 0.8)
            speed = base_speed * param_factor * arch_multiplier
            
            # Apply optimization adjustments
            if "weight_quantization" in architecture.compression_techniques:
                speed *= 1.5
                
            architecture.estimated_inference_speed = speed
            
            # Estimate memory usage
            # SSMs have constant memory footprint for KV-cache
            if model_type in ["ssm", "liquid"]:
                base_memory = size_mb * 1.1 
            else:
                base_memory = size_mb * 1.5
                
            architecture.estimated_memory_usage = int(base_memory)
            
            # Predict accuracy retention
            teacher_params = teacher_analysis.estimated_parameters or 100_000_000_000
            compression_ratio = teacher_params / total_params
            
            # Base accuracy retention
            if compression_ratio <= 10:
                base_accuracy = 0.95
            elif compression_ratio <= 50:
                base_accuracy = 0.88
            else:
                base_accuracy = 0.80
            
            # SSMs are surprisingly good at long-range reasoning
            if model_type == "ssm" and request.domain == "scientific_reasoning":
                base_accuracy *= 1.02

            predicted_accuracy = min(0.98, base_accuracy)
            architecture.predicted_accuracy = predicted_accuracy
            
            # Calculate compression ratio and efficiency
            architecture.compression_ratio = compression_ratio
            architecture.efficiency_score = min(1.0, (speed / 1000) * (1000 / base_memory) * predicted_accuracy)
            
            logger.info("Performance prediction completed",
                       parameters=total_params,
                       speed=speed,
                       accuracy=predicted_accuracy)
            
        except Exception as e:
            logger.error("Performance prediction failed", error=str(e))
    
    async def _generate_design_rationale(
        self, 
        architecture: StudentArchitecture, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        optimization_target: OptimizationTarget
    ):
        """Generate design rationale and alternative architectures"""
        try:
            # Design decisions rationale
            design_decisions = {
                "architecture_choice": f"Selected {architecture.model_type} for optimal {optimization_target.value} performance",
                "layer_count": f"Selected {architecture.layer_count} layers to balance capacity and efficiency",
                "hidden_size": f"Hidden size of {architecture.hidden_size} provides good expressiveness",
                "compression_ratio": f"Target compression ratio of {architecture.compression_ratio:.1f}x achieves size reduction"
            }
            
            if architecture.model_type == "ssm":
                design_decisions["efficiency"] = "SSM architecture provides linear scaling for long-context scientific reasoning"
            
            # Trade-off analysis
            trade_off_analysis = {
                "accuracy_vs_speed": {
                    "chosen_balance": optimization_target.value,
                    "speed_gain": f"{architecture.estimated_inference_speed / 100:.1f}x faster than baseline",
                    "accuracy_retention": f"{architecture.predicted_accuracy:.1%} of teacher performance"
                }
            }
            
            # Generate alternative architectures
            alternatives = await self._generate_alternative_architectures(request, teacher_analysis, optimization_target)
            
            # Update architecture
            architecture.design_decisions = design_decisions
            architecture.trade_off_analysis = trade_off_analysis
            architecture.alternative_architectures = alternatives
            
        except Exception as e:
            logger.error("Design rationale generation failed", error=str(e))
    
    async def _generate_alternative_architectures(
        self, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        optimization_target: OptimizationTarget
    ) -> List[Dict[str, Any]]:
        """Generate alternative architecture options"""
        try:
            alternatives = []
            
            # Architectural alternatives
            if (request.target_architecture or "transformer") == "transformer":
                alternatives.append({
                    "name": "SSM Efficient",
                    "model_type": "ssm",
                    "trade_offs": "3x faster inference, constant memory, experimental reasoning",
                    "use_case": "Extreme edge nodes with FTNS budget constraints"
                })
            
            # Speed-optimized alternative
            if optimization_target != OptimizationTarget.SPEED:
                speed_alt = {
                    "name": "Speed Optimized",
                    "layer_count": max(6, request.layer_count // 2 if request.layer_count else 8),
                    "hidden_size": max(256, request.hidden_size // 2 if request.hidden_size else 384),
                    "trade_offs": "2x faster inference, ~10% accuracy reduction",
                    "use_case": "Real-time applications requiring fast response"
                }
                alternatives.append(speed_alt)
            
            return alternatives
            
        except Exception as e:
            logger.error("Alternative architecture generation failed", error=str(e))
            return []
    
    # === Helper Methods ===
    
    async def _optimize_layer_count(self, template: Dict[str, Any], optimization_config: Dict[str, float]) -> int:
        """Optimize number of layers"""
        layer_range = template["layer_range"]
        penalty = optimization_config["layer_penalty"]
        
        # Bias towards fewer layers for speed/efficiency optimization
        if penalty < 1.0:
            return layer_range[0] + int((layer_range[1] - layer_range[0]) * 0.3)
        else:
            return layer_range[0] + int((layer_range[1] - layer_range[0]) * 0.7)
    
    async def _optimize_hidden_size(self, template: Dict[str, Any], optimization_config: Dict[str, float]) -> int:
        """Optimize hidden size"""
        hidden_range = template["hidden_range"]
        penalty = optimization_config["parameter_penalty"]
        
        # Choose hidden size based on optimization target
        if penalty < 1.0:
            size = hidden_range[0] + int((hidden_range[1] - hidden_range[0]) * 0.3)
        else:
            size = hidden_range[0] + int((hidden_range[1] - hidden_range[0]) * 0.7)
        
        return (size // 64) * 64 
    
    async def _optimize_attention_heads(self, template: Dict[str, Any], optimization_config: Dict[str, float], hidden_size: int) -> int:
        """Optimize number of attention heads"""
        heads_range = template["attention_heads_range"]
        penalty = optimization_config["attention_penalty"]
        
        max_heads = min(heads_range[1], hidden_size // 64)
        min_heads = max(heads_range[0], 4)
        
        if penalty < 1.0:
            heads = min_heads + int((max_heads - min_heads) * 0.3)
        else:
            heads = min_heads + int((max_heads - min_heads) * 0.7)
        
        return heads

    async def _optimize_ssm_state(self, template: Dict[str, Any], optimization_config: Dict[str, float]) -> int:
        """Optimize SSM state dimension"""
        state_range = template.get("ssm_state_range", (16, 32))
        penalty = optimization_config["parameter_penalty"]
        if penalty < 1.0:
            return state_range[0]
        return state_range[1]

    async def _optimize_ssm_conv(self, template: Dict[str, Any], optimization_config: Dict[str, float]) -> int:
        """Optimize SSM convolution kernel size"""
        conv_range = template.get("ssm_conv_range", (4, 4))
        return conv_range[0]

    async def _optimize_liquid_hidden(self, template: Dict[str, Any], optimization_config: Dict[str, float]) -> int:
        """Optimize Liquid hidden state size"""
        hidden_range = template.get("liquid_hidden_range", (128, 256))
        penalty = optimization_config["parameter_penalty"]
        if penalty < 1.0:
            return hidden_range[0]
        return hidden_range[1]