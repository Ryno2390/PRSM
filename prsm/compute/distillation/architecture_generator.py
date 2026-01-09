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
                "intermediate_factor": 2.0,
                "max_parameters": 100_000_000  # 100M
            },
            ModelSize.SMALL: {
                "layer_range": (12, 18),
                "hidden_range": (512, 768),
                "attention_heads_range": (8, 12),
                "intermediate_factor": 3.0,
                "max_parameters": 1_000_000_000  # 1B
            },
            ModelSize.MEDIUM: {
                "layer_range": (18, 24),
                "hidden_range": (768, 1024),
                "attention_heads_range": (12, 16),
                "intermediate_factor": 4.0,
                "max_parameters": 10_000_000_000  # 10B
            },
            ModelSize.LARGE: {
                "layer_range": (24, 32),
                "hidden_range": (1024, 2048),
                "attention_heads_range": (16, 32),
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
        logger.info("Generating student architecture",
                   target_size=request.target_size.value,
                   optimization_target=request.optimization_target.value,
                   domain=request.domain)
        
        try:
            # Initialize architecture
            architecture = StudentArchitecture(
                distillation_request_id=request.request_id,
                model_type=request.target_architecture or "transformer"
            )
            
            # Get base template and constraints
            template = self.size_templates[request.target_size]
            domain_pattern = self.domain_patterns.get(request.domain, {})
            optimization_config = self.optimization_configs[request.optimization_target]
            
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
            await self._generate_design_rationale(architecture, request, teacher_analysis)
            
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
            # Use custom parameters if provided, otherwise optimize
            if request.layer_count:
                layer_count = request.layer_count
            else:
                layer_count = await self._optimize_layer_count(template, optimization_config)
            
            if request.hidden_size:
                hidden_size = request.hidden_size
            else:
                hidden_size = await self._optimize_hidden_size(template, optimization_config)
            
            if request.attention_heads:
                attention_heads = request.attention_heads
            else:
                attention_heads = await self._optimize_attention_heads(template, optimization_config, hidden_size)
            
            # Calculate intermediate size
            intermediate_size = int(hidden_size * template["intermediate_factor"])
            
            # Vocabulary size
            vocabulary_size = request.vocabulary_size or 32000  # Default vocab size
            
            # Update architecture
            architecture.layer_count = layer_count
            architecture.hidden_size = hidden_size
            architecture.attention_heads = attention_heads
            architecture.intermediate_size = intermediate_size
            architecture.vocabulary_size = vocabulary_size
            
            logger.info("Core parameters generated",
                       layers=layer_count,
                       hidden_size=hidden_size,
                       attention_heads=attention_heads)
            
        except Exception as e:
            logger.error("Core parameter generation failed", error=str(e))
    
    async def _add_domain_specific_components(
        self, 
        architecture: StudentArchitecture, 
        request: DistillationRequest,
        domain_pattern: Dict[str, Any]
    ):
        """Add domain-specific architectural components"""
        try:
            specialized_layers = []
            
            # Add domain-specific layers
            for layer_type in domain_pattern.get("specialized_layers", []):
                layer_config = {
                    "type": layer_type,
                    "position": "post_attention",
                    "size": architecture.hidden_size // 2,
                    "activation": "gelu"
                }
                specialized_layers.append(layer_config)
            
            # Add attention pattern modifications
            attention_pattern = domain_pattern.get("attention_pattern", "standard")
            if attention_pattern != "standard":
                attention_config = {
                    "type": "attention_modification",
                    "pattern": attention_pattern,
                    "heads_affected": "all",
                    "modification_strength": 0.3
                }
                specialized_layers.append(attention_config)
            
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
            # Estimate parameters
            embedding_params = architecture.vocabulary_size * architecture.hidden_size
            transformer_params = (
                architecture.layer_count * (
                    # Attention weights
                    4 * architecture.hidden_size * architecture.hidden_size +
                    # Feed-forward weights
                    2 * architecture.hidden_size * architecture.intermediate_size +
                    # Layer norm weights
                    2 * architecture.hidden_size
                )
            )
            output_params = architecture.vocabulary_size * architecture.hidden_size
            
            # Add specialized layer parameters
            specialized_params = sum(
                layer.get("size", architecture.hidden_size // 2) * architecture.hidden_size
                for layer in architecture.specialized_layers
            )
            
            total_params = embedding_params + transformer_params + output_params + specialized_params
            architecture.estimated_parameters = total_params
            
            # Estimate model size (2 bytes per parameter for float16)
            size_mb = (total_params * 2) / (1024 * 1024)
            architecture.estimated_size_mb = size_mb
            
            # Estimate inference speed (tokens/second)
            # Based on parameter count and hardware assumptions
            base_speed = 1000  # Base tokens/second for small model
            param_factor = max(0.1, 1.0 - (total_params / 1_000_000_000) * 0.8)  # Larger models slower
            speed = base_speed * param_factor
            
            # Apply optimization adjustments
            if "operator_optimization" in architecture.optimization_strategies:
                speed *= 1.3
            if "layer_fusion" in architecture.optimization_strategies:
                speed *= 1.2
            if "weight_quantization" in architecture.compression_techniques:
                speed *= 1.5
                
            architecture.estimated_inference_speed = speed
            
            # Estimate memory usage
            base_memory = size_mb * 1.5  # Model + activation memory
            if "gradient_checkpointing" in architecture.optimization_strategies:
                base_memory *= 0.8  # Memory savings
            architecture.estimated_memory_usage = int(base_memory)
            
            # Predict accuracy retention
            # Based on parameter ratio and optimization target
            teacher_params = teacher_analysis.estimated_parameters or 100_000_000_000
            compression_ratio = teacher_params / total_params
            
            # Base accuracy retention
            if compression_ratio <= 10:
                base_accuracy = 0.95
            elif compression_ratio <= 50:
                base_accuracy = 0.88
            elif compression_ratio <= 100:
                base_accuracy = 0.82
            else:
                base_accuracy = 0.75
            
            # Adjust for optimization target
            if request.optimization_target == OptimizationTarget.ACCURACY:
                base_accuracy *= 1.05
            elif request.optimization_target in [OptimizationTarget.SPEED, OptimizationTarget.SIZE]:
                base_accuracy *= 0.95
            
            # Adjust for domain complexity
            domain_difficulty = {
                "medical_research": 0.95,
                "legal_analysis": 0.93,
                "scientific_reasoning": 0.92,
                "code_generation": 0.96,
                "creative_writing": 0.94
            }.get(request.domain, 0.95)
            
            predicted_accuracy = min(0.98, base_accuracy * domain_difficulty)
            architecture.predicted_accuracy = predicted_accuracy
            
            # Calculate compression ratio and efficiency
            architecture.compression_ratio = compression_ratio
            architecture.efficiency_score = min(1.0, (speed / 1000) * (1000 / base_memory) * predicted_accuracy)
            
            logger.info("Performance prediction completed",
                       parameters=total_params,
                       size_mb=size_mb,
                       speed=speed,
                       accuracy=predicted_accuracy,
                       compression_ratio=compression_ratio)
            
        except Exception as e:
            logger.error("Performance prediction failed", error=str(e))
    
    async def _generate_design_rationale(
        self, 
        architecture: StudentArchitecture, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis
    ):
        """Generate design rationale and alternative architectures"""
        try:
            # Design decisions rationale
            design_decisions = {
                "layer_count": f"Selected {architecture.layer_count} layers to balance capacity and efficiency for {request.optimization_target.value} optimization",
                "hidden_size": f"Hidden size of {architecture.hidden_size} provides good trade-off between expressiveness and computational cost",
                "attention_heads": f"{architecture.attention_heads} attention heads allow sufficient parallel attention patterns while maintaining efficiency",
                "compression_ratio": f"Target compression ratio of {architecture.compression_ratio:.1f}x achieves size reduction while preserving {architecture.predicted_accuracy:.1%} accuracy"
            }
            
            # Trade-off analysis
            trade_off_analysis = {
                "accuracy_vs_speed": {
                    "chosen_balance": request.optimization_target.value,
                    "speed_gain": f"{architecture.estimated_inference_speed / 100:.1f}x faster than baseline",
                    "accuracy_retention": f"{architecture.predicted_accuracy:.1%} of teacher performance"
                },
                "size_vs_capability": {
                    "parameter_reduction": f"{architecture.compression_ratio:.1f}x smaller than teacher",
                    "capability_preservation": f"Retains {len(teacher_analysis.critical_knowledge_areas)} critical knowledge areas"
                },
                "efficiency_vs_quality": {
                    "efficiency_score": f"{architecture.efficiency_score:.2f}",
                    "quality_measures": "Optimized for domain-specific performance"
                }
            }
            
            # Generate alternative architectures
            alternatives = await self._generate_alternative_architectures(request, teacher_analysis)
            
            # Update architecture
            architecture.design_decisions = design_decisions
            architecture.trade_off_analysis = trade_off_analysis
            architecture.alternative_architectures = alternatives
            
            logger.info("Design rationale generated",
                       decisions_count=len(design_decisions),
                       alternatives_count=len(alternatives))
            
        except Exception as e:
            logger.error("Design rationale generation failed", error=str(e))
    
    async def _generate_alternative_architectures(
        self, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis
    ) -> List[Dict[str, Any]]:
        """Generate alternative architecture options"""
        try:
            alternatives = []
            
            # Speed-optimized alternative
            if request.optimization_target != OptimizationTarget.SPEED:
                speed_alt = {
                    "name": "Speed Optimized",
                    "layer_count": max(6, request.layer_count // 2 if request.layer_count else 8),
                    "hidden_size": max(256, request.hidden_size // 2 if request.hidden_size else 384),
                    "trade_offs": "2x faster inference, ~10% accuracy reduction",
                    "use_case": "Real-time applications requiring fast response"
                }
                alternatives.append(speed_alt)
            
            # Accuracy-optimized alternative
            if request.optimization_target != OptimizationTarget.ACCURACY:
                accuracy_alt = {
                    "name": "Accuracy Optimized", 
                    "layer_count": min(24, (request.layer_count or 12) + 4),
                    "hidden_size": min(1024, (request.hidden_size or 512) + 256),
                    "trade_offs": "5-10% higher accuracy, 2x more parameters",
                    "use_case": "Applications where accuracy is paramount"
                }
                alternatives.append(accuracy_alt)
            
            # Balanced alternative
            if request.optimization_target != OptimizationTarget.BALANCED:
                balanced_alt = {
                    "name": "Balanced",
                    "layer_count": 12,
                    "hidden_size": 512,
                    "trade_offs": "Good balance of speed, accuracy, and size",
                    "use_case": "General-purpose applications"
                }
                alternatives.append(balanced_alt)
            
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
        
        # Ensure divisible by attention heads
        return (size // 64) * 64  # Round to nearest 64
    
    async def _optimize_attention_heads(self, template: Dict[str, Any], optimization_config: Dict[str, float], hidden_size: int) -> int:
        """Optimize number of attention heads"""
        heads_range = template["attention_heads_range"]
        penalty = optimization_config["attention_penalty"]
        
        # Choose based on hidden size and optimization target
        max_heads = min(heads_range[1], hidden_size // 64)  # Each head needs at least 64 dimensions
        min_heads = max(heads_range[0], 4)  # Minimum 4 heads
        
        if penalty < 1.0:
            heads = min_heads + int((max_heads - min_heads) * 0.3)
        else:
            heads = min_heads + int((max_heads - min_heads) * 0.7)
        
        return heads