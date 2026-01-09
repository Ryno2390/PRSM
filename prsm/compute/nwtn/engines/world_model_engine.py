#!/usr/bin/env python3
"""
World Model Engine for NWTN

Maintains and updates a comprehensive world model that integrates knowledge
from multiple domains, tracks causal relationships, and enables advanced reasoning.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from uuid import uuid4

logger = structlog.get_logger(__name__)


class DomainType(Enum):
    """Domain types for world model organization"""
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    SOCIAL = "social"
    ECONOMIC = "economic"
    POLITICAL = "political"
    CULTURAL = "cultural"
    PHILOSOPHICAL = "philosophical"
    MATHEMATICAL = "mathematical"
    GENERAL = "general"


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSATION = "direct_causation"
    INDIRECT_CAUSATION = "indirect_causation"
    CORRELATION = "correlation"
    CONTRADICTION = "contradiction"
    IMPLICATION = "implication"
    ANALOGY = "analogy"
    DEPENDENCY = "dependency"


@dataclass
class CausalRelation:
    """Represents a causal relationship between concepts"""
    relation_id: str = field(default_factory=lambda: str(uuid4()))
    source_concept: str = ""
    target_concept: str = ""
    relation_type: CausalRelationType = CausalRelationType.CORRELATION
    strength: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    domain: DomainType = DomainType.GENERAL
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainModel:
    """Model for a specific domain"""
    domain: DomainType
    concepts: Set[str] = field(default_factory=set)
    relations: List[CausalRelation] = field(default_factory=list)
    core_principles: List[str] = field(default_factory=list)
    knowledge_graph: Dict[str, List[str]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_score: float = 0.7


@dataclass
class ValidationResult:
    """Result of world model validation"""
    is_valid: bool = True
    confidence: float = 0.8
    inconsistencies: List[str] = field(default_factory=list)
    missing_relations: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)


class WorldModelEngine:
    """
    Comprehensive world model engine that maintains and updates knowledge
    across multiple domains with causal relationships and validation.
    """
    
    def __init__(self):
        """Initialize the world model engine"""
        self.domain_models: Dict[DomainType, DomainModel] = {}
        self.cross_domain_relations: List[CausalRelation] = []
        self.global_concepts: Set[str] = set()
        self.validation_history: List[ValidationResult] = []
        
        # Initialize default domain models
        self._initialize_domain_models()
        
        logger.info("WorldModelEngine initialized", 
                   domains=len(self.domain_models),
                   concepts=len(self.global_concepts))
    
    def _initialize_domain_models(self):
        """Initialize basic domain models with core concepts"""
        domain_concepts = {
            DomainType.SCIENTIFIC: [
                "hypothesis", "experiment", "theory", "evidence", "causation",
                "correlation", "methodology", "peer_review", "reproducibility"
            ],
            DomainType.TECHNICAL: [
                "algorithm", "system", "architecture", "optimization", "scalability",
                "reliability", "performance", "integration", "innovation"
            ],
            DomainType.MATHEMATICAL: [
                "proof", "theorem", "axiom", "logic", "function", "relationship",
                "pattern", "structure", "abstraction"
            ],
            DomainType.PHILOSOPHICAL: [
                "reasoning", "argument", "premise", "conclusion", "validity",
                "truth", "knowledge", "belief", "ethics"
            ]
        }
        
        for domain, concepts in domain_concepts.items():
            domain_model = DomainModel(
                domain=domain,
                concepts=set(concepts),
                core_principles=[f"Core principle for {domain.value}"],
                confidence_score=0.8
            )
            self.domain_models[domain] = domain_model
            self.global_concepts.update(concepts)
    
    async def add_causal_relation(self, relation: CausalRelation) -> bool:
        """Add a causal relation to the world model"""
        try:
            # Validate the relation
            if not await self._validate_relation(relation):
                logger.warning("Invalid causal relation rejected",
                             source=relation.source_concept,
                             target=relation.target_concept)
                return False
            
            # Add to appropriate domain model
            if relation.domain in self.domain_models:
                self.domain_models[relation.domain].relations.append(relation)
                self.domain_models[relation.domain].last_updated = datetime.now(timezone.utc)
            else:
                # Add as cross-domain relation
                self.cross_domain_relations.append(relation)
            
            # Update global concepts
            self.global_concepts.add(relation.source_concept)
            self.global_concepts.add(relation.target_concept)
            
            logger.debug("Causal relation added",
                        relation_id=relation.relation_id,
                        domain=relation.domain.value)
            return True
            
        except Exception as e:
            logger.error("Failed to add causal relation", error=str(e))
            return False
    
    async def _validate_relation(self, relation: CausalRelation) -> bool:
        """Validate a causal relation before adding it"""
        # Basic validation checks
        if not relation.source_concept or not relation.target_concept:
            return False
        
        if relation.strength < 0.0 or relation.strength > 1.0:
            return False
        
        if relation.confidence < 0.0 or relation.confidence > 1.0:
            return False
        
        # Check for contradictions with existing relations
        existing_relations = self.get_relations_for_concept(relation.source_concept)
        for existing in existing_relations:
            if (existing.target_concept == relation.target_concept and
                existing.relation_type == CausalRelationType.CONTRADICTION and
                relation.relation_type != CausalRelationType.CONTRADICTION):
                logger.warning("Potential contradiction detected",
                             concept=relation.source_concept,
                             target=relation.target_concept)
                return relation.confidence > 0.8  # Only accept if very confident
        
        return True
    
    def get_relations_for_concept(self, concept: str) -> List[CausalRelation]:
        """Get all relations involving a specific concept"""
        relations = []
        
        # Check domain models
        for domain_model in self.domain_models.values():
            for relation in domain_model.relations:
                if relation.source_concept == concept or relation.target_concept == concept:
                    relations.append(relation)
        
        # Check cross-domain relations
        for relation in self.cross_domain_relations:
            if relation.source_concept == concept or relation.target_concept == concept:
                relations.append(relation)
        
        return relations
    
    def get_domain_model(self, domain: DomainType) -> Optional[DomainModel]:
        """Get domain model for a specific domain"""
        return self.domain_models.get(domain)
    
    async def update_concept_knowledge(self, concept: str, knowledge: Dict[str, Any]) -> bool:
        """Update knowledge about a specific concept"""
        try:
            # Add concept to global set
            self.global_concepts.add(concept)
            
            # Extract domain from knowledge or infer
            domain = knowledge.get('domain', DomainType.GENERAL)
            if isinstance(domain, str):
                domain = DomainType(domain)
            
            # Ensure domain model exists
            if domain not in self.domain_models:
                self.domain_models[domain] = DomainModel(domain=domain)
            
            # Add concept to domain model
            domain_model = self.domain_models[domain]
            domain_model.concepts.add(concept)
            
            # Update knowledge graph
            related_concepts = knowledge.get('related_concepts', [])
            domain_model.knowledge_graph[concept] = related_concepts
            
            # Add any embedded relations
            if 'relations' in knowledge:
                for rel_data in knowledge['relations']:
                    relation = CausalRelation(
                        source_concept=concept,
                        target_concept=rel_data.get('target', ''),
                        relation_type=CausalRelationType(rel_data.get('type', 'correlation')),
                        strength=rel_data.get('strength', 0.5),
                        confidence=rel_data.get('confidence', 0.5),
                        domain=domain
                    )
                    await self.add_causal_relation(relation)
            
            domain_model.last_updated = datetime.now(timezone.utc)
            
            logger.debug("Concept knowledge updated",
                        concept=concept,
                        domain=domain.value)
            return True
            
        except Exception as e:
            logger.error("Failed to update concept knowledge",
                        concept=concept,
                        error=str(e))
            return False
    
    async def validate_world_model(self) -> ValidationResult:
        """Validate the consistency and completeness of the world model"""
        try:
            inconsistencies = []
            missing_relations = []
            
            # Check for contradictory relations
            for domain_model in self.domain_models.values():
                for i, rel1 in enumerate(domain_model.relations):
                    for rel2 in domain_model.relations[i+1:]:
                        if (rel1.source_concept == rel2.source_concept and
                            rel1.target_concept == rel2.target_concept and
                            rel1.relation_type == CausalRelationType.CONTRADICTION and
                            rel2.relation_type != CausalRelationType.CONTRADICTION):
                            inconsistencies.append(
                                f"Contradiction: {rel1.source_concept} -> {rel1.target_concept}"
                            )
            
            # Check for missing expected relations
            for concept in list(self.global_concepts)[:10]:  # Sample check
                relations = self.get_relations_for_concept(concept)
                if len(relations) == 0:
                    missing_relations.append(f"No relations found for concept: {concept}")
            
            # Calculate overall confidence
            total_confidence = 0.0
            total_relations = 0
            
            for domain_model in self.domain_models.values():
                for relation in domain_model.relations:
                    total_confidence += relation.confidence
                    total_relations += 1
            
            for relation in self.cross_domain_relations:
                total_confidence += relation.confidence
                total_relations += 1
            
            overall_confidence = total_confidence / max(total_relations, 1)
            
            # Determine validity
            is_valid = len(inconsistencies) < 3 and overall_confidence > 0.6
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=overall_confidence,
                inconsistencies=inconsistencies,
                missing_relations=missing_relations,
                details={
                    'total_concepts': len(self.global_concepts),
                    'total_relations': total_relations,
                    'domains_count': len(self.domain_models),
                    'cross_domain_relations': len(self.cross_domain_relations)
                }
            )
            
            self.validation_history.append(result)
            
            logger.info("World model validation complete",
                       is_valid=is_valid,
                       confidence=overall_confidence,
                       inconsistencies_count=len(inconsistencies))
            
            return result
            
        except Exception as e:
            logger.error("World model validation failed", error=str(e))
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                inconsistencies=[f"Validation error: {str(e)}"],
                missing_relations=[],
                details={'error': str(e)}
            )
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[str]:
        """Get concepts related to the given concept up to max_depth"""
        related = set()
        to_process = [(concept, 0)]
        processed = set()
        
        while to_process:
            current_concept, depth = to_process.pop(0)
            
            if current_concept in processed or depth >= max_depth:
                continue
            
            processed.add(current_concept)
            
            # Get direct relations
            relations = self.get_relations_for_concept(current_concept)
            for relation in relations:
                if relation.source_concept == current_concept:
                    target = relation.target_concept
                    if target not in processed:
                        related.add(target)
                        to_process.append((target, depth + 1))
                elif relation.target_concept == current_concept:
                    source = relation.source_concept  
                    if source not in processed:
                        related.add(source)
                        to_process.append((source, depth + 1))
        
        return list(related)
    
    def get_world_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the world model"""
        total_relations = sum(len(dm.relations) for dm in self.domain_models.values())
        total_relations += len(self.cross_domain_relations)
        
        return {
            'total_concepts': len(self.global_concepts),
            'total_relations': total_relations,
            'domains': len(self.domain_models),
            'cross_domain_relations': len(self.cross_domain_relations),
            'validation_history_count': len(self.validation_history),
            'last_validation': self.validation_history[-1].validation_timestamp.isoformat() if self.validation_history else None,
            'domain_breakdown': {
                domain.value: {
                    'concepts': len(model.concepts),
                    'relations': len(model.relations),
                    'confidence': model.confidence_score
                }
                for domain, model in self.domain_models.items()
            }
        }


# Factory functions
def create_world_model_engine() -> WorldModelEngine:
    """Create a new world model engine instance"""
    return WorldModelEngine()


def create_domain_specialized_engine(domain: DomainType) -> WorldModelEngine:
    """Create a world model engine specialized for a specific domain"""
    engine = WorldModelEngine()
    
    # Initialize with expanded concepts for the target domain
    if domain == DomainType.SCIENTIFIC:
        additional_concepts = [
            "paradigm", "falsifiability", "replication", "statistical_significance",
            "bias", "control_group", "variable", "observation", "measurement"
        ]
    elif domain == DomainType.TECHNICAL:
        additional_concepts = [
            "modularity", "abstraction", "encapsulation", "coupling", "cohesion",
            "pattern", "refactoring", "testing", "debugging", "deployment"
        ]
    else:
        additional_concepts = ["specialized_concept_1", "specialized_concept_2"]
    
    # Add specialized concepts
    for concept in additional_concepts:
        engine.global_concepts.add(concept)
        if domain in engine.domain_models:
            engine.domain_models[domain].concepts.add(concept)
    
    logger.info("Domain specialized engine created",
               domain=domain.value,
               specialized_concepts=len(additional_concepts))
    
    return engine


def create_base_world_model() -> WorldModelEngine:
    """Create a basic world model engine with minimal initialization"""
    engine = WorldModelEngine()
    logger.info("Base world model engine created")
    return engine


# Global instance management
_global_world_model: Optional[WorldModelEngine] = None


def get_world_model_engine() -> WorldModelEngine:
    """Get or create the global world model engine instance"""
    global _global_world_model
    if _global_world_model is None:
        _global_world_model = create_world_model_engine()
    return _global_world_model