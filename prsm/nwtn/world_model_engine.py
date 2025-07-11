"""
World Model Engine for NWTN Hybrid Architecture
First-principles reasoning and causal validation system

This module implements the System 2 component of the hybrid architecture,
providing structured reasoning about causal relationships, physical principles,
and logical consistency. It serves as the "slow, deliberative" reasoning system
that validates and refines the rapid pattern recognition from System 1.

Key Features:
1. Domain-specific world models (physics, chemistry, biology, etc.)
2. Causal relationship tracking and validation
3. First-principles reasoning from fundamental laws
4. Logical consistency checking
5. Cross-domain knowledge transfer
6. Hierarchical model organization (core -> domain -> specialized)

Integration Points:
- PRSM Marketplace: Share validated world models
- IPFS: Distributed storage of world model graphs
- Federation: Consensus on core principles across nodes
- Tokenomics: FTNS rewards for model improvements
"""

import asyncio
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from prsm.core.models import PRSMBaseModel, TimestampMixin
from prsm.core.config import get_settings
from prsm.nwtn.hybrid_architecture import SOC, SOCType, ConfidenceLevel

logger = structlog.get_logger(__name__)
settings = get_settings()


class DomainType(str, Enum):
    """Types of knowledge domains"""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATHEMATICS = "mathematics"
    LOGIC = "logic"
    COMPUTER_SCIENCE = "computer_science"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    GENERAL = "general"


class CausalRelationType(str, Enum):
    """Types of causal relationships"""
    CAUSES = "causes"                    # A causes B
    ENABLES = "enables"                  # A enables B
    PREVENTS = "prevents"                # A prevents B
    CORRELATES = "correlates"            # A correlates with B
    IMPLIES = "implies"                  # A implies B
    REQUIRES = "requires"                # A requires B
    INSTANTIATES = "instantiates"        # A is an instance of B
    COMPOSED_OF = "composed_of"          # A is composed of B


class ValidationResult(PRSMBaseModel):
    """Result of world model validation"""
    
    is_valid: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_principles: List[str] = Field(default_factory=list)
    conflicting_principles: List[str] = Field(default_factory=list)
    confidence_adjustment: float = Field(default=0.0)
    explanation: str = ""
    
    
class CausalRelation(PRSMBaseModel):
    """Represents a causal relationship between SOCs"""
    
    id: UUID = Field(default_factory=uuid4)
    source_soc: str = Field(..., description="Source SOC name")
    target_soc: str = Field(..., description="Target SOC name")
    relation_type: CausalRelationType
    strength: float = Field(ge=0.0, le=1.0, description="Strength of causal relation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in relation")
    
    # Evidence and validation
    evidence_count: int = Field(default=0)
    supporting_experiments: List[str] = Field(default_factory=list)
    
    # Context
    domain: str = Field(default="general")
    conditions: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_validated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DomainModel(PRSMBaseModel):
    """Domain-specific world model"""
    
    domain: DomainType
    core_principles: Dict[str, SOC] = Field(default_factory=dict)
    specialized_socs: Dict[str, SOC] = Field(default_factory=dict)
    causal_relations: Dict[str, CausalRelation] = Field(default_factory=dict)
    
    # Hierarchy
    parent_domains: List[str] = Field(default_factory=list)
    child_domains: List[str] = Field(default_factory=list)
    
    # Validation rules
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    consistency_checks: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(default="1.0")


class WorldModelEngine:
    """
    Core world model engine implementing System 2 reasoning
    
    Manages hierarchical world models from core principles to specialized domains,
    provides causal reasoning, and validates SOCs against first principles.
    """
    
    def __init__(self):
        self.core_model = DomainModel(domain=DomainType.GENERAL)
        self.domain_models: Dict[str, DomainModel] = {}
        self.global_relations: Dict[str, CausalRelation] = {}
        
        # Initialize with fundamental models
        self._initialize_core_world_model()
        self._initialize_domain_models()
        
        logger.info("World Model Engine initialized")
        
    def _initialize_core_world_model(self):
        """Initialize core world model with fundamental principles"""
        
        # Physics principles
        physics_principles = {
            "conservation_of_energy": {
                "description": "Energy cannot be created or destroyed, only transformed",
                "confidence": 0.98,
                "domain": "physics",
                "type": "conservation_law"
            },
            "conservation_of_momentum": {
                "description": "Total momentum of isolated system remains constant",
                "confidence": 0.98,
                "domain": "physics",
                "type": "conservation_law"
            },
            "causality": {
                "description": "Every effect has a preceding cause",
                "confidence": 0.95,
                "domain": "physics",
                "type": "fundamental_principle"
            },
            "thermodynamics_second_law": {
                "description": "Entropy of isolated system always increases",
                "confidence": 0.97,
                "domain": "physics",
                "type": "thermodynamic_law"
            }
        }
        
        # Mathematical principles
        mathematical_principles = {
            "mathematical_consistency": {
                "description": "Mathematical statements are either true or false",
                "confidence": 0.99,
                "domain": "mathematics",
                "type": "logical_principle"
            },
            "transitivity": {
                "description": "If A relates to B and B relates to C, then A relates to C",
                "confidence": 0.95,
                "domain": "mathematics",
                "type": "relational_property"
            }
        }
        
        # Logical principles
        logical_principles = {
            "non_contradiction": {
                "description": "A statement cannot be both true and false simultaneously",
                "confidence": 0.99,
                "domain": "logic",
                "type": "logical_axiom"
            },
            "excluded_middle": {
                "description": "A statement is either true or false, no middle ground",
                "confidence": 0.95,
                "domain": "logic",
                "type": "logical_axiom"
            },
            "identity": {
                "description": "Every entity is identical to itself",
                "confidence": 0.99,
                "domain": "logic",
                "type": "logical_axiom"
            }
        }
        
        # Create core SOCs
        all_principles = {**physics_principles, **mathematical_principles, **logical_principles}
        
        for name, props in all_principles.items():
            soc = SOC(
                name=name,
                soc_type=SOCType.PRINCIPLE,
                confidence=props["confidence"],
                confidence_level=ConfidenceLevel.CORE,
                domain=props["domain"],
                properties={
                    "description": props["description"],
                    "type": props["type"],
                    "fundamental": True
                }
            )
            self.core_model.core_principles[name] = soc
            
        # Create fundamental causal relations
        self._create_fundamental_causal_relations()
        
        logger.info(
            "Core world model initialized",
            principles_count=len(self.core_model.core_principles),
            relations_count=len(self.core_model.causal_relations)
        )
        
    def _create_fundamental_causal_relations(self):
        """Create fundamental causal relationships"""
        
        # Physics causality
        physics_relations = [
            ("force", "acceleration", CausalRelationType.CAUSES),
            ("mass", "gravitational_attraction", CausalRelationType.CAUSES),
            ("energy_input", "temperature_increase", CausalRelationType.CAUSES),
            ("friction", "energy_dissipation", CausalRelationType.CAUSES),
        ]
        
        # Logic causality
        logic_relations = [
            ("premises", "conclusion", CausalRelationType.IMPLIES),
            ("contradiction", "inconsistency", CausalRelationType.CAUSES),
            ("logical_axiom", "logical_derivation", CausalRelationType.ENABLES),
        ]
        
        # Create relations
        for source, target, relation_type in physics_relations + logic_relations:
            relation = CausalRelation(
                source_soc=source,
                target_soc=target,
                relation_type=relation_type,
                strength=0.9,
                confidence=0.9,
                domain="physics" if (source, target, relation_type) in physics_relations else "logic"
            )
            self.core_model.causal_relations[str(relation.id)] = relation
            
    def _initialize_domain_models(self):
        """Initialize domain-specific world models"""
        
        # Physics domain
        physics_model = DomainModel(
            domain=DomainType.PHYSICS,
            parent_domains=["general"],
            validation_rules={
                "energy_conservation": "Total energy must be conserved",
                "momentum_conservation": "Total momentum must be conserved",
                "causality_requirement": "Effects must have preceding causes"
            }
        )
        
        # Chemistry domain
        chemistry_model = DomainModel(
            domain=DomainType.CHEMISTRY,
            parent_domains=["physics"],
            validation_rules={
                "mass_conservation": "Mass must be conserved in reactions",
                "charge_conservation": "Charge must be conserved in reactions",
                "thermodynamic_feasibility": "Reactions must be thermodynamically feasible"
            }
        )
        
        # Biology domain
        biology_model = DomainModel(
            domain=DomainType.BIOLOGY,
            parent_domains=["chemistry", "physics"],
            validation_rules={
                "evolutionary_consistency": "Biological features must have evolutionary basis",
                "energy_efficiency": "Biological processes must be energy efficient",
                "information_flow": "Biological information must flow in consistent patterns"
            }
        )
        
        # Store domain models
        self.domain_models["physics"] = physics_model
        self.domain_models["chemistry"] = chemistry_model
        self.domain_models["biology"] = biology_model
        
        logger.info(
            "Domain models initialized",
            domains=list(self.domain_models.keys())
        )
        
    async def validate_soc_against_world_model(
        self, 
        soc: SOC, 
        domain: str = None
    ) -> ValidationResult:
        """
        Validate SOC against world model principles
        
        This is the core System 2 reasoning function that checks SOCs
        against first principles and causal relationships.
        """
        
        target_domain = domain or soc.domain
        
        # Get relevant domain model
        domain_model = self.domain_models.get(target_domain, self.core_model)
        
        # Validate against core principles
        core_validation = await self._validate_against_core_principles(soc, domain_model)
        
        # Validate against causal relations
        causal_validation = await self._validate_causal_consistency(soc, domain_model)
        
        # Validate against domain-specific rules
        domain_validation = await self._validate_domain_rules(soc, domain_model)
        
        # Combine validation results
        overall_confidence = (
            core_validation.confidence_score * 0.4 +
            causal_validation.confidence_score * 0.4 +
            domain_validation.confidence_score * 0.2
        )
        
        is_valid = overall_confidence > 0.6
        
        confidence_adjustment = 0.0
        if overall_confidence > 0.8:
            confidence_adjustment = 0.1
        elif overall_confidence < 0.4:
            confidence_adjustment = -0.1
            
        # Compile supporting and conflicting principles
        supporting_principles = (
            core_validation.supporting_principles +
            causal_validation.supporting_principles +
            domain_validation.supporting_principles
        )
        
        conflicting_principles = (
            core_validation.conflicting_principles +
            causal_validation.conflicting_principles +
            domain_validation.conflicting_principles
        )
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            supporting_principles=supporting_principles,
            conflicting_principles=conflicting_principles,
            confidence_adjustment=confidence_adjustment,
            explanation=f"SOC '{soc.name}' validated against {target_domain} world model"
        )
        
        logger.info(
            "SOC validation completed",
            soc_name=soc.name,
            domain=target_domain,
            is_valid=is_valid,
            confidence_score=overall_confidence
        )
        
        return result
        
    async def _validate_against_core_principles(
        self, 
        soc: SOC, 
        domain_model: DomainModel
    ) -> ValidationResult:
        """Validate SOC against core principles"""
        
        supporting_principles = []
        conflicting_principles = []
        confidence_score = 0.5  # Neutral starting point
        
        # Check against each core principle
        for principle_name, principle_soc in domain_model.core_principles.items():
            consistency_score = await self._check_soc_consistency(soc, principle_soc)
            
            if consistency_score > 0.7:
                supporting_principles.append(principle_name)
                confidence_score += 0.1
            elif consistency_score < 0.3:
                conflicting_principles.append(principle_name)
                confidence_score -= 0.1
                
        # Normalize confidence score
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return ValidationResult(
            is_valid=len(conflicting_principles) == 0,
            confidence_score=confidence_score,
            supporting_principles=supporting_principles,
            conflicting_principles=conflicting_principles
        )
        
    async def _validate_causal_consistency(
        self, 
        soc: SOC, 
        domain_model: DomainModel
    ) -> ValidationResult:
        """Validate SOC against causal relations"""
        
        supporting_principles = []
        conflicting_principles = []
        confidence_score = 0.5
        
        # Check if SOC participates in any causal relations
        for relation_id, relation in domain_model.causal_relations.items():
            if soc.name in [relation.source_soc, relation.target_soc]:
                # SOC is part of established causal relation
                if relation.confidence > 0.7:
                    supporting_principles.append(f"causal_relation_{relation_id}")
                    confidence_score += 0.05
                    
        # Check for causal contradictions
        # (Simplified - full implementation would do sophisticated causal reasoning)
        
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return ValidationResult(
            is_valid=len(conflicting_principles) == 0,
            confidence_score=confidence_score,
            supporting_principles=supporting_principles,
            conflicting_principles=conflicting_principles
        )
        
    async def _validate_domain_rules(
        self, 
        soc: SOC, 
        domain_model: DomainModel
    ) -> ValidationResult:
        """Validate SOC against domain-specific rules"""
        
        supporting_principles = []
        conflicting_principles = []
        confidence_score = 0.5
        
        # Apply domain-specific validation rules
        for rule_name, rule_description in domain_model.validation_rules.items():
            rule_satisfaction = await self._check_domain_rule(soc, rule_name, rule_description)
            
            if rule_satisfaction > 0.6:
                supporting_principles.append(rule_name)
                confidence_score += 0.05
            elif rule_satisfaction < 0.4:
                conflicting_principles.append(rule_name)
                confidence_score -= 0.05
                
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return ValidationResult(
            is_valid=len(conflicting_principles) == 0,
            confidence_score=confidence_score,
            supporting_principles=supporting_principles,
            conflicting_principles=conflicting_principles
        )
        
    async def _check_soc_consistency(self, soc1: SOC, soc2: SOC) -> float:
        """Check consistency between two SOCs"""
        
        # Simplified consistency check
        # Full implementation would use sophisticated NLP and domain knowledge
        
        # Check for direct contradictions in descriptions
        desc1 = soc1.properties.get("description", "").lower()
        desc2 = soc2.properties.get("description", "").lower()
        
        # Basic keyword analysis
        contradiction_pairs = [
            ("create", "destroy"),
            ("increase", "decrease"),
            ("cause", "prevent"),
            ("enable", "disable"),
            ("conserve", "violate")
        ]
        
        for word1, word2 in contradiction_pairs:
            if word1 in desc1 and word2 in desc2:
                return 0.2  # Strong contradiction
            if word2 in desc1 and word1 in desc2:
                return 0.2  # Strong contradiction
                
        # Check for supporting keywords
        support_groups = [
            ["energy", "force", "momentum", "conservation"],
            ["cause", "effect", "relationship", "dependency"],
            ["logic", "reasoning", "consistency", "validation"]
        ]
        
        for group in support_groups:
            if any(word in desc1 for word in group) and any(word in desc2 for word in group):
                return 0.8  # Strong support
                
        return 0.5  # Neutral
        
    async def _check_domain_rule(self, soc: SOC, rule_name: str, rule_description: str) -> float:
        """Check if SOC satisfies domain-specific rule"""
        
        # Simplified rule checking
        # Full implementation would use domain-specific validators
        
        soc_desc = soc.properties.get("description", "").lower()
        rule_desc = rule_description.lower()
        
        # Check for rule-related keywords in SOC
        rule_keywords = rule_desc.split()
        matches = sum(1 for word in rule_keywords if word in soc_desc)
        
        if matches > len(rule_keywords) * 0.3:
            return 0.7  # Good rule satisfaction
        else:
            return 0.5  # Neutral
            
    async def create_causal_relation(
        self,
        source_soc: str,
        target_soc: str,
        relation_type: CausalRelationType,
        strength: float,
        domain: str = "general",
        evidence: List[str] = None
    ) -> CausalRelation:
        """Create new causal relation between SOCs"""
        
        relation = CausalRelation(
            source_soc=source_soc,
            target_soc=target_soc,
            relation_type=relation_type,
            strength=strength,
            confidence=0.5,  # Start with neutral confidence
            domain=domain,
            supporting_experiments=evidence or []
        )
        
        # Add to appropriate domain model
        if domain in self.domain_models:
            self.domain_models[domain].causal_relations[str(relation.id)] = relation
        else:
            self.global_relations[str(relation.id)] = relation
            
        logger.info(
            "Created causal relation",
            source=source_soc,
            target=target_soc,
            relation_type=relation_type.value,
            strength=strength,
            domain=domain
        )
        
        return relation
        
    def update_soc_in_world_model(self, soc: SOC, validation_result: ValidationResult):
        """Update SOC in world model based on validation results"""
        
        # Update SOC confidence based on validation
        if validation_result.confidence_adjustment != 0:
            soc.update_confidence(soc.confidence + validation_result.confidence_adjustment)
            
        # If SOC reaches core confidence, add to appropriate domain model
        if soc.confidence_level == ConfidenceLevel.CORE:
            domain_model = self.domain_models.get(soc.domain, self.core_model)
            domain_model.core_principles[soc.name] = soc
            
            logger.info(
                "SOC promoted to core principles",
                soc_name=soc.name,
                domain=soc.domain,
                confidence=soc.confidence
            )
            
    def get_domain_model(self, domain: str) -> Optional[DomainModel]:
        """Get domain model by name"""
        return self.domain_models.get(domain)
        
    def get_related_socs(self, soc_name: str, relation_type: CausalRelationType = None) -> List[Tuple[str, CausalRelation]]:
        """Get SOCs related to given SOC"""
        
        related = []
        
        # Search all domain models
        for domain_model in self.domain_models.values():
            for relation in domain_model.causal_relations.values():
                if soc_name in [relation.source_soc, relation.target_soc]:
                    if relation_type is None or relation.relation_type == relation_type:
                        other_soc = relation.target_soc if relation.source_soc == soc_name else relation.source_soc
                        related.append((other_soc, relation))
                        
        return related
        
    def get_world_model_stats(self) -> Dict[str, Any]:
        """Get world model statistics"""
        
        total_principles = len(self.core_model.core_principles)
        total_relations = len(self.core_model.causal_relations)
        
        domain_stats = {}
        for domain, model in self.domain_models.items():
            domain_stats[domain] = {
                "principles": len(model.core_principles),
                "specialized_socs": len(model.specialized_socs),
                "causal_relations": len(model.causal_relations)
            }
            total_principles += len(model.core_principles)
            total_relations += len(model.causal_relations)
            
        return {
            "total_principles": total_principles,
            "total_relations": total_relations,
            "core_model_principles": len(self.core_model.core_principles),
            "domain_models": len(self.domain_models),
            "domain_stats": domain_stats
        }
        
    async def export_world_model(self, format: str = "json") -> Dict[str, Any]:
        """Export world model for sharing or storage"""
        
        if format == "json":
            return {
                "core_model": self.core_model.dict(),
                "domain_models": {k: v.dict() for k, v in self.domain_models.items()},
                "global_relations": {k: v.dict() for k, v in self.global_relations.items()},
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"
            }
        else:
            raise ValueError(f"Export format '{format}' not supported")
            
    async def import_world_model(self, model_data: Dict[str, Any]):
        """Import world model from external source"""
        
        try:
            # Import core model
            if "core_model" in model_data:
                self.core_model = DomainModel(**model_data["core_model"])
                
            # Import domain models
            if "domain_models" in model_data:
                for domain, model_dict in model_data["domain_models"].items():
                    self.domain_models[domain] = DomainModel(**model_dict)
                    
            # Import global relations
            if "global_relations" in model_data:
                for relation_id, relation_dict in model_data["global_relations"].items():
                    self.global_relations[relation_id] = CausalRelation(**relation_dict)
                    
            logger.info(
                "World model imported successfully",
                domains=len(self.domain_models),
                principles=len(self.core_model.core_principles)
            )
            
        except Exception as e:
            logger.error("Failed to import world model", error=str(e))
            raise


# Factory functions for integration

def create_world_model_engine() -> WorldModelEngine:
    """Create world model engine instance"""
    return WorldModelEngine()


def create_domain_specialized_engine(domain: DomainType) -> WorldModelEngine:
    """Create world model engine specialized for specific domain"""
    
    engine = WorldModelEngine()
    
    # Add domain-specific initialization
    if domain == DomainType.PHYSICS:
        # Add more physics-specific principles
        pass
    elif domain == DomainType.CHEMISTRY:
        # Add more chemistry-specific principles
        pass
    elif domain == DomainType.BIOLOGY:
        # Add more biology-specific principles
        pass
        
    return engine


def create_base_world_model() -> Dict[str, Any]:
    """
    Create base world model that can be copied for new agents
    
    This implements the "base instincts" concept from the brainstorming document
    """
    
    engine = WorldModelEngine()
    
    # Export core model as base
    base_model = {
        "core_principles": {k: v.dict() for k, v in engine.core_model.core_principles.items()},
        "fundamental_relations": {k: v.dict() for k, v in engine.core_model.causal_relations.items()},
        "validation_rules": engine.core_model.validation_rules,
        "version": "1.0",
        "type": "base_world_model"
    }
    
    return base_model