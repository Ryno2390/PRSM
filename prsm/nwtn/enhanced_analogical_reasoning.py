#!/usr/bin/env python3
"""
Enhanced Analogical Reasoning Engine for NWTN
=============================================

This module implements a comprehensive analogical reasoning system based on
cognitive science research (Gentner, Holyoak, Hofstadter) with multiple
modalities and elemental components.

Key Features:
- Structure-Mapping Theory implementation
- Multi-modal analogical reasoning
- Topographical pattern mapping
- Developmental analogies for breakthrough discovery
- Quality assessment and refinement
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

import structlog

logger = structlog.get_logger(__name__)


class AnalogicalReasoningType(Enum):
    """Types of analogical reasoning modalities"""
    DEVELOPMENTAL = "developmental"        # Mature → less developed domains
    EXPLANATORY = "explanatory"          # Complex → simple for understanding
    PROBLEM_SOLVING = "problem_solving"  # Solution transfer
    CREATIVE = "creative"                # Novel idea generation
    STRUCTURAL = "structural"            # Formal system modeling
    NORMATIVE = "normative"              # Decision/ethics guidance


class RelationType(Enum):
    """Types of relations that can be mapped"""
    CAUSAL = "causal"                    # A causes B
    TEMPORAL = "temporal"                # A happens before B
    SPATIAL = "spatial"                  # A is connected to B
    FUNCTIONAL = "functional"            # A serves purpose B
    HIERARCHICAL = "hierarchical"        # A contains B
    SIMILARITY = "similarity"            # A is similar to B
    OPPOSITION = "opposition"            # A opposes B
    TRANSFORMATION = "transformation"    # A becomes B


@dataclass
class ConceptualObject:
    """Represents an object in a conceptual domain"""
    id: str
    name: str
    properties: Dict[str, Any]
    relations: Dict[str, List[str]]  # relation_type -> [connected_object_ids]
    embedding: Optional[np.ndarray] = None
    domain: Optional[str] = None
    
    def __post_init__(self):
        if self.embedding is None:
            # Generate embedding from properties and relations
            self.embedding = self._generate_embedding()
    
    def _generate_embedding(self) -> np.ndarray:
        """Generate embedding from object properties and relations"""
        # Simplified embedding generation
        # In practice, this would use semantic embeddings
        feature_vector = []
        
        # Property features
        for prop, value in self.properties.items():
            if isinstance(value, (int, float)):
                feature_vector.append(value)
            else:
                feature_vector.append(hash(str(value)) % 1000 / 1000.0)
        
        # Relation features
        for rel_type, connections in self.relations.items():
            feature_vector.append(len(connections))
        
        # Pad or truncate to fixed size
        while len(feature_vector) < 64:
            feature_vector.append(0.0)
        
        return np.array(feature_vector[:64])


@dataclass
class StructuralRelation:
    """Represents a structural relation between objects"""
    relation_type: RelationType
    source_object: str
    target_object: str
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.relation_type, self.source_object, self.target_object))


@dataclass
class ConceptualDomain:
    """Represents a conceptual domain with objects and relations"""
    name: str
    objects: Dict[str, ConceptualObject]
    relations: Set[StructuralRelation]
    maturity_level: float = 0.0  # 0.0 = nascent, 1.0 = highly developed
    domain_embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.domain_embedding is None:
            self.domain_embedding = self._generate_domain_embedding()
    
    def _generate_domain_embedding(self) -> np.ndarray:
        """Generate domain-level embedding from objects and relations"""
        if not self.objects:
            return np.zeros(256)
        
        # Aggregate object embeddings
        object_embeddings = [obj.embedding for obj in self.objects.values()]
        domain_embedding = np.mean(object_embeddings, axis=0)
        
        # Add relational structure features
        relation_features = np.zeros(64)
        for relation in self.relations:
            rel_idx = list(RelationType).index(relation.relation_type)
            relation_features[rel_idx % 64] += relation.strength
        
        # Combine object and relational features
        combined = np.concatenate([domain_embedding, relation_features])
        
        # Pad to 256 dimensions
        while len(combined) < 256:
            combined = np.append(combined, 0.0)
        
        return combined[:256]
    
    def get_topographical_complexity(self) -> float:
        """Calculate topographical complexity of the domain"""
        if not self.objects or not self.relations:
            return 0.0
        
        # Factors contributing to complexity:
        # 1. Number of objects and relations
        # 2. Diversity of relation types
        # 3. Interconnectedness
        # 4. Hierarchical depth
        
        object_count = len(self.objects)
        relation_count = len(self.relations)
        relation_types = len(set(r.relation_type for r in self.relations))
        
        # Calculate interconnectedness
        total_connections = sum(len(obj.relations) for obj in self.objects.values())
        avg_connections = total_connections / object_count if object_count > 0 else 0
        
        # Normalize and combine factors
        complexity = (
            (object_count / 100.0) * 0.3 +
            (relation_count / 500.0) * 0.3 +
            (relation_types / len(RelationType)) * 0.2 +
            (avg_connections / 10.0) * 0.2
        )
        
        return min(complexity, 1.0)


@dataclass
class AnalogicalMapping:
    """Represents a mapping between source and target domains"""
    source_domain: ConceptualDomain
    target_domain: ConceptualDomain
    object_mappings: Dict[str, str]  # source_obj_id -> target_obj_id
    relation_mappings: Dict[StructuralRelation, StructuralRelation]
    mapping_quality: float = 0.0
    reasoning_type: AnalogicalReasoningType = AnalogicalReasoningType.DEVELOPMENTAL
    
    def __post_init__(self):
        if self.mapping_quality == 0.0:
            self.mapping_quality = self._calculate_mapping_quality()
    
    def _calculate_mapping_quality(self) -> float:
        """Calculate quality of the analogical mapping"""
        if not self.object_mappings or not self.relation_mappings:
            return 0.0
        
        # Factors for mapping quality:
        # 1. Structural consistency
        # 2. Semantic similarity
        # 3. Relational alignment
        # 4. Coverage (how much of each domain is mapped)
        
        # Structural consistency
        consistent_relations = 0
        total_relations = len(self.relation_mappings)
        
        for source_rel, target_rel in self.relation_mappings.items():
            if source_rel.relation_type == target_rel.relation_type:
                consistent_relations += 1
        
        structural_score = consistent_relations / total_relations if total_relations > 0 else 0
        
        # Semantic similarity (simplified)
        semantic_scores = []
        for source_id, target_id in self.object_mappings.items():
            source_obj = self.source_domain.objects.get(source_id)
            target_obj = self.target_domain.objects.get(target_id)
            
            if source_obj and target_obj:
                similarity = np.dot(source_obj.embedding, target_obj.embedding)
                semantic_scores.append(similarity)
        
        semantic_score = np.mean(semantic_scores) if semantic_scores else 0
        
        # Coverage
        source_coverage = len(self.object_mappings) / len(self.source_domain.objects)
        target_coverage = len(set(self.object_mappings.values())) / len(self.target_domain.objects)
        coverage_score = (source_coverage + target_coverage) / 2
        
        # Combine scores
        quality = (structural_score * 0.4 + semantic_score * 0.3 + coverage_score * 0.3)
        return min(quality, 1.0)


@dataclass
class AnalogicalInference:
    """Represents an inference generated from analogical reasoning"""
    inference_type: str
    content: str
    confidence: float
    supporting_mappings: List[AnalogicalMapping]
    predicted_outcomes: List[str]
    verification_steps: List[str]
    
    def __str__(self):
        return f"Inference[{self.inference_type}]: {self.content} (confidence: {self.confidence:.2f})"


class AnalogicalReasoningEngine:
    """Main engine for analogical reasoning"""
    
    def __init__(self):
        self.domains: Dict[str, ConceptualDomain] = {}
        self.mappings: List[AnalogicalMapping] = []
        self.inference_history: List[AnalogicalInference] = []
        
        # Configuration
        self.min_mapping_quality = 0.3
        self.max_mappings_per_query = 10
        self.similarity_threshold = 0.5
    
    def add_domain(self, domain: ConceptualDomain):
        """Add a conceptual domain to the engine"""
        self.domains[domain.name] = domain
        logger.info(f"Added domain: {domain.name}", 
                   objects=len(domain.objects), 
                   relations=len(domain.relations),
                   maturity=domain.maturity_level)
    
    def find_analogical_mappings(
        self, 
        source_domain: str, 
        target_domain: str,
        reasoning_type: AnalogicalReasoningType = AnalogicalReasoningType.DEVELOPMENTAL
    ) -> List[AnalogicalMapping]:
        """Find analogical mappings between two domains"""
        
        if source_domain not in self.domains or target_domain not in self.domains:
            return []
        
        source = self.domains[source_domain]
        target = self.domains[target_domain]
        
        # For developmental reasoning, ensure source is more mature
        if reasoning_type == AnalogicalReasoningType.DEVELOPMENTAL:
            if source.maturity_level < target.maturity_level:
                source, target = target, source
        
        # Generate mappings using structure-mapping theory
        mappings = self._generate_structure_mappings(source, target, reasoning_type)
        
        # Filter by quality
        quality_mappings = [m for m in mappings if m.mapping_quality >= self.min_mapping_quality]
        
        # Sort by quality and return top mappings
        quality_mappings.sort(key=lambda x: x.mapping_quality, reverse=True)
        
        return quality_mappings[:self.max_mappings_per_query]
    
    def _generate_structure_mappings(
        self, 
        source: ConceptualDomain, 
        target: ConceptualDomain,
        reasoning_type: AnalogicalReasoningType
    ) -> List[AnalogicalMapping]:
        """Generate structure mappings using cognitive science principles"""
        
        mappings = []
        
        # Find object correspondences based on structural similarity
        object_mappings = self._find_object_correspondences(source, target)
        
        # Find relation correspondences
        relation_mappings = self._find_relation_correspondences(
            source, target, object_mappings
        )
        
        # Create mapping
        if object_mappings and relation_mappings:
            mapping = AnalogicalMapping(
                source_domain=source,
                target_domain=target,
                object_mappings=object_mappings,
                relation_mappings=relation_mappings,
                reasoning_type=reasoning_type
            )
            mappings.append(mapping)
        
        return mappings
    
    def _find_object_correspondences(
        self, 
        source: ConceptualDomain, 
        target: ConceptualDomain
    ) -> Dict[str, str]:
        """Find object correspondences based on structural and semantic similarity"""
        
        correspondences = {}
        
        # Calculate similarity matrix
        for source_id, source_obj in source.objects.items():
            best_match = None
            best_score = 0.0
            
            for target_id, target_obj in target.objects.items():
                # Semantic similarity
                semantic_sim = np.dot(source_obj.embedding, target_obj.embedding)
                
                # Structural similarity (relation patterns)
                structural_sim = self._calculate_structural_similarity(source_obj, target_obj)
                
                # Combined similarity
                combined_sim = semantic_sim * 0.6 + structural_sim * 0.4
                
                if combined_sim > best_score and combined_sim > self.similarity_threshold:
                    best_score = combined_sim
                    best_match = target_id
            
            if best_match:
                correspondences[source_id] = best_match
        
        return correspondences
    
    def _calculate_structural_similarity(
        self, 
        obj1: ConceptualObject, 
        obj2: ConceptualObject
    ) -> float:
        """Calculate structural similarity between two objects"""
        
        # Compare relation patterns
        obj1_relations = set(obj1.relations.keys())
        obj2_relations = set(obj2.relations.keys())
        
        if not obj1_relations and not obj2_relations:
            return 1.0
        
        intersection = obj1_relations.intersection(obj2_relations)
        union = obj1_relations.union(obj2_relations)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_relation_correspondences(
        self, 
        source: ConceptualDomain, 
        target: ConceptualDomain,
        object_mappings: Dict[str, str]
    ) -> Dict[StructuralRelation, StructuralRelation]:
        """Find relation correspondences based on object mappings"""
        
        relation_mappings = {}
        
        # Map relations based on object correspondences
        for source_rel in source.relations:
            # Check if both objects in the relation are mapped
            if (source_rel.source_object in object_mappings and 
                source_rel.target_object in object_mappings):
                
                target_source = object_mappings[source_rel.source_object]
                target_target = object_mappings[source_rel.target_object]
                
                # Find corresponding relation in target domain
                for target_rel in target.relations:
                    if (target_rel.source_object == target_source and
                        target_rel.target_object == target_target and
                        target_rel.relation_type == source_rel.relation_type):
                        
                        relation_mappings[source_rel] = target_rel
                        break
        
        return relation_mappings
    
    def generate_inferences(
        self, 
        mapping: AnalogicalMapping,
        max_inferences: int = 5
    ) -> List[AnalogicalInference]:
        """Generate inferences based on analogical mapping"""
        
        inferences = []
        
        # Different inference types based on reasoning type
        if mapping.reasoning_type == AnalogicalReasoningType.DEVELOPMENTAL:
            inferences.extend(self._generate_developmental_inferences(mapping))
        elif mapping.reasoning_type == AnalogicalReasoningType.PROBLEM_SOLVING:
            inferences.extend(self._generate_problem_solving_inferences(mapping))
        elif mapping.reasoning_type == AnalogicalReasoningType.CREATIVE:
            inferences.extend(self._generate_creative_inferences(mapping))
        
        # Sort by confidence and return top inferences
        inferences.sort(key=lambda x: x.confidence, reverse=True)
        return inferences[:max_inferences]
    
    def _generate_developmental_inferences(
        self, 
        mapping: AnalogicalMapping
    ) -> List[AnalogicalInference]:
        """Generate developmental inferences for domain advancement"""
        
        inferences = []
        
        # Look for gaps in target domain that exist in source domain
        source_objects = set(mapping.source_domain.objects.keys())
        mapped_source_objects = set(mapping.object_mappings.keys())
        unmapped_source_objects = source_objects - mapped_source_objects
        
        for source_obj_id in unmapped_source_objects:
            source_obj = mapping.source_domain.objects[source_obj_id]
            
            # Generate inference for missing object/concept
            inference = AnalogicalInference(
                inference_type="developmental_gap",
                content=f"Target domain may benefit from developing concept similar to '{source_obj.name}' from source domain",
                confidence=0.7,
                supporting_mappings=[mapping],
                predicted_outcomes=[f"Enhanced {mapping.target_domain.name} framework"],
                verification_steps=[f"Investigate if {source_obj.name} analogue exists in {mapping.target_domain.name}"]
            )
            inferences.append(inference)
        
        return inferences
    
    def _generate_problem_solving_inferences(
        self, 
        mapping: AnalogicalMapping
    ) -> List[AnalogicalInference]:
        """Generate problem-solving inferences"""
        
        inferences = []
        
        # Look for successful patterns in source that could solve target problems
        for source_rel in mapping.source_domain.relations:
            if source_rel.relation_type == RelationType.CAUSAL:
                # Look for causal patterns that could be transferred
                inference = AnalogicalInference(
                    inference_type="solution_transfer",
                    content=f"Causal pattern from {mapping.source_domain.name} may solve problems in {mapping.target_domain.name}",
                    confidence=0.6,
                    supporting_mappings=[mapping],
                    predicted_outcomes=["Problem resolution", "Improved efficiency"],
                    verification_steps=["Test causal relationship in target domain"]
                )
                inferences.append(inference)
        
        return inferences
    
    def _generate_creative_inferences(
        self, 
        mapping: AnalogicalMapping
    ) -> List[AnalogicalInference]:
        """Generate creative inferences for innovation"""
        
        inferences = []
        
        # Look for novel combinations or unexpected connections
        for source_obj_id, target_obj_id in mapping.object_mappings.items():
            source_obj = mapping.source_domain.objects[source_obj_id]
            target_obj = mapping.target_domain.objects[target_obj_id]
            
            # Generate creative combination
            inference = AnalogicalInference(
                inference_type="creative_combination",
                content=f"Novel approach: combine {source_obj.name} properties with {target_obj.name} context",
                confidence=0.5,
                supporting_mappings=[mapping],
                predicted_outcomes=["Innovative solution", "Novel methodology"],
                verification_steps=["Prototype combination", "Test feasibility"]
            )
            inferences.append(inference)
        
        return inferences
    
    async def process_analogical_query(
        self, 
        query: str, 
        source_domain: Optional[str] = None,
        reasoning_type: AnalogicalReasoningType = AnalogicalReasoningType.DEVELOPMENTAL
    ) -> List[AnalogicalInference]:
        """Process a natural language query for analogical reasoning"""
        
        # This would integrate with NLP to understand the query
        # For now, simplified implementation
        
        all_inferences = []
        
        if source_domain:
            # Find analogies from specified source domain
            for target_name in self.domains:
                if target_name != source_domain:
                    mappings = self.find_analogical_mappings(
                        source_domain, target_name, reasoning_type
                    )
                    for mapping in mappings:
                        inferences = self.generate_inferences(mapping)
                        all_inferences.extend(inferences)
        else:
            # Find analogies across all domain pairs
            domain_names = list(self.domains.keys())
            for i, source_name in enumerate(domain_names):
                for target_name in domain_names[i+1:]:
                    mappings = self.find_analogical_mappings(
                        source_name, target_name, reasoning_type
                    )
                    for mapping in mappings:
                        inferences = self.generate_inferences(mapping)
                        all_inferences.extend(inferences)
        
        # Sort by confidence and return top results
        all_inferences.sort(key=lambda x: x.confidence, reverse=True)
        return all_inferences[:20]


# Example usage and testing
async def demo_analogical_reasoning():
    """Demonstrate the analogical reasoning engine"""
    
    engine = AnalogicalReasoningEngine()
    
    # Create example domains
    
    # Biology domain (mature)
    biology_objects = {
        "heart": ConceptualObject("heart", "Heart", 
                                 {"function": "pump", "material": "muscle"}, 
                                 {"causal": ["circulation"]}, domain="biology"),
        "circulation": ConceptualObject("circulation", "Blood Circulation", 
                                       {"type": "fluid_flow", "direction": "cyclic"}, 
                                       {"functional": ["oxygen_transport"]}, domain="biology"),
        "oxygen_transport": ConceptualObject("oxygen_transport", "Oxygen Transport", 
                                            {"mechanism": "hemoglobin", "efficiency": 0.95}, 
                                            {}, domain="biology")
    }
    
    biology_relations = {
        StructuralRelation(RelationType.CAUSAL, "heart", "circulation", 0.9),
        StructuralRelation(RelationType.FUNCTIONAL, "circulation", "oxygen_transport", 0.8)
    }
    
    biology_domain = ConceptualDomain("biology", biology_objects, biology_relations, 0.8)
    
    # Engineering domain (less mature in bio-inspired design)
    engineering_objects = {
        "pump": ConceptualObject("pump", "Mechanical Pump", 
                               {"function": "fluid_movement", "material": "metal"}, 
                               {"causal": ["fluid_flow"]}, domain="engineering"),
        "fluid_flow": ConceptualObject("fluid_flow", "Fluid Flow", 
                                     {"type": "liquid_movement", "direction": "directional"}, 
                                     {"functional": ["transport"]}, domain="engineering"),
        "transport": ConceptualObject("transport", "Material Transport", 
                                    {"mechanism": "pressure", "efficiency": 0.7}, 
                                    {}, domain="engineering")
    }
    
    engineering_relations = {
        StructuralRelation(RelationType.CAUSAL, "pump", "fluid_flow", 0.9),
        StructuralRelation(RelationType.FUNCTIONAL, "fluid_flow", "transport", 0.7)
    }
    
    engineering_domain = ConceptualDomain("engineering", engineering_objects, engineering_relations, 0.6)
    
    # Add domains to engine
    engine.add_domain(biology_domain)
    engine.add_domain(engineering_domain)
    
    # Find analogical mappings
    mappings = engine.find_analogical_mappings("biology", "engineering", AnalogicalReasoningType.DEVELOPMENTAL)
    
    print("🧠 Analogical Mappings Found:")
    for mapping in mappings:
        print(f"  Quality: {mapping.mapping_quality:.2f}")
        print(f"  Object mappings: {mapping.object_mappings}")
        print(f"  Relation mappings: {len(mapping.relation_mappings)}")
        print()
    
    # Generate inferences
    if mappings:
        inferences = engine.generate_inferences(mappings[0])
        print("💡 Generated Inferences:")
        for inference in inferences:
            print(f"  {inference}")
            print(f"    Predicted outcomes: {inference.predicted_outcomes}")
            print()


if __name__ == "__main__":
    asyncio.run(demo_analogical_reasoning())