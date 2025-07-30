#!/usr/bin/env python3
"""
Analogical Chain Reasoning System for NWTN
==========================================

This module implements Phase 4.3: Analogical Chain Reasoning from the NWTN roadmap.
It enables multi-hop analogical reasoning through A→B→C→D chains with semantic consistency
validation, extending the breakthrough capabilities of the existing analogical reasoning system.

Key Innovations:
1. **Multi-Hop Analogical Chains**: Complex A→B→C→D analogical reasoning paths
2. **Semantic Consistency Validation**: Advanced coherence checking across reasoning chains
3. **Path Discovery Algorithms**: Intelligent discovery of non-obvious analogical connections
4. **Chain Quality Scoring**: Assessment of analogical chain strength and breakthrough potential
5. **Creative Solution Generation**: Multi-hop chains for discovering innovative solutions

Architecture Components:
- AnalogicalChainEngine: Core engine for multi-hop analogical reasoning
- ChainPathFinder: Discovery of analogical paths across multiple domains
- SemanticConsistencyValidator: Validation of chain coherence and logical consistency
- ChainQualityAssessor: Assessment of chain strength, novelty, and breakthrough potential
- BreakthroughChainIdentifier: Identification of chains with highest innovation potential

Based on NWTN Roadmap Phase 4.3 - Analogical Chain Reasoning
Expected Impact: Enhanced cross-domain innovation through sophisticated analogical path discovery
"""

import asyncio
import time
import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from uuid import uuid4
from collections import defaultdict, deque
import networkx as nx
import random
import statistics
import structlog

logger = structlog.get_logger(__name__)

class ChainType(Enum):
    """Types of analogical chains"""
    LINEAR = "linear"                   # A→B→C→D straight path
    BRANCHING = "branching"             # A→B→{C1,C2}→D multiple paths
    CONVERGENT = "convergent"           # {A1,A2}→B→C→D paths converge
    CYCLICAL = "cyclical"               # A→B→C→A with return
    HIERARCHICAL = "hierarchical"       # Multi-level abstraction chains
    TEMPORAL = "temporal"               # Time-based analogical progression

class AnalogicalRelationType(Enum):
    """Types of analogical relationships between chain elements"""
    STRUCTURAL = "structural"           # Similar structure or organization
    FUNCTIONAL = "functional"           # Similar function or purpose
    CAUSAL = "causal"                  # Similar cause-effect relationships
    BEHAVIORAL = "behavioral"           # Similar behaviors or patterns
    COMPOSITIONAL = "compositional"     # Similar composition or parts
    PROCEDURAL = "procedural"          # Similar processes or methods
    CONTEXTUAL = "contextual"          # Similar contexts or environments
    EMERGENT = "emergent"              # Similar emergent properties

class ConsistencyLevel(Enum):
    """Levels of semantic consistency in analogical chains"""
    STRONG = "strong"                   # High consistency (>0.8)
    MODERATE = "moderate"               # Moderate consistency (0.6-0.8)
    WEAK = "weak"                      # Weak consistency (0.4-0.6)
    INCONSISTENT = "inconsistent"       # Poor consistency (<0.4)

@dataclass
class AnalogicalElement:
    """Represents an element in an analogical chain"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    domain: str = ""
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    structural_features: List[str] = field(default_factory=list)
    functional_features: List[str] = field(default_factory=list)
    behavioral_features: List[str] = field(default_factory=list)
    abstraction_level: int = 0          # 0=concrete, higher=more abstract
    domain_expertise_required: float = 0.5  # 0=common knowledge, 1=expert knowledge
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AnalogicalLink:
    """Represents a link between two elements in an analogical chain"""
    source_element: AnalogicalElement
    target_element: AnalogicalElement
    relation_type: AnalogicalRelationType
    id: str = field(default_factory=lambda: str(uuid4()))
    similarity_score: float = 0.0
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    transformation_description: str = ""
    breakthrough_potential: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AnalogicalChain:
    """Represents a complete analogical reasoning chain"""
    chain_type: ChainType
    id: str = field(default_factory=lambda: str(uuid4()))
    elements: List[AnalogicalElement] = field(default_factory=list)
    links: List[AnalogicalLink] = field(default_factory=list)
    source_query: str = ""
    target_domain: str = ""
    
    # Quality metrics
    overall_consistency: float = 0.0
    semantic_coherence: float = 0.0
    novelty_score: float = 0.0
    breakthrough_potential: float = 0.0
    practical_applicability: float = 0.0
    
    # Chain characteristics
    chain_length: int = 0
    domains_traversed: List[str] = field(default_factory=list)
    abstraction_levels: List[int] = field(default_factory=list)
    complexity_score: float = 0.0
    
    # Validation results
    consistency_level: ConsistencyLevel = ConsistencyLevel.MODERATE
    validation_evidence: List[str] = field(default_factory=list)
    potential_weaknesses: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ChainDiscoveryResult:
    """Result of analogical chain discovery process"""
    query: str = ""
    discovered_chains: List[AnalogicalChain] = field(default_factory=list)
    best_chains: List[AnalogicalChain] = field(default_factory=list)
    breakthrough_chains: List[AnalogicalChain] = field(default_factory=list)
    total_paths_explored: int = 0
    discovery_time: float = 0.0
    chain_diversity: float = 0.0
    average_chain_quality: float = 0.0
    recommended_applications: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ChainPathFinder:
    """Discovery of analogical paths across multiple domains"""
    
    def __init__(self, max_chain_length: int = 6):
        self.max_chain_length = max_chain_length
        self.knowledge_graph = nx.DiGraph()
        self.domain_elements = defaultdict(list)  # domain -> [elements]
        self.element_connections = defaultdict(list)  # element_id -> [connected_elements]
        
    async def initialize_knowledge_base(self, context: Dict[str, Any]):
        """Initialize knowledge base with domain elements and connections"""
        
        # Create sample knowledge base for analogical reasoning
        await self._create_sample_elements()
        await self._establish_connections()
        
        logger.info("Knowledge base initialized", 
                   elements=len(self.domain_elements),
                   connections=len(self.knowledge_graph.edges))
        
    async def _create_sample_elements(self):
        """Create sample elements across different domains"""
        
        # Biological domain elements
        bio_elements = [
            AnalogicalElement(
                name="protein_folding",
                domain="biology",
                description="Process of protein structure formation",
                structural_features=["3D_structure", "folding_patterns", "active_sites"],
                functional_features=["catalysis", "binding", "regulation"],
                behavioral_features=["conformational_change", "stability", "interaction"]
            ),
            AnalogicalElement(
                name="neural_networks",
                domain="biology",
                description="Interconnected network of neurons",
                structural_features=["nodes", "connections", "hierarchies"],
                functional_features=["signal_processing", "learning", "memory"],
                behavioral_features=["adaptation", "pattern_recognition", "emergence"]
            ),
            AnalogicalElement(
                name="ecosystem_dynamics",
                domain="biology",
                description="Interactions within ecological systems",
                structural_features=["food_webs", "hierarchies", "cycles"],
                functional_features=["energy_flow", "resource_allocation", "balance"],
                behavioral_features=["adaptation", "competition", "cooperation"]
            )
        ]
        
        # Engineering domain elements
        eng_elements = [
            AnalogicalElement(
                name="origami",
                domain="engineering",
                description="Art and science of paper folding",
                structural_features=["fold_patterns", "geometric_constraints", "3D_forms"],
                functional_features=["structural_optimization", "deployment", "compactness"],
                behavioral_features=["transformation", "constraint_satisfaction", "efficiency"]
            ),
            AnalogicalElement(
                name="algorithm_optimization",
                domain="engineering",
                description="Process of improving algorithm efficiency",
                structural_features=["data_structures", "control_flow", "modularity"],
                functional_features=["performance", "resource_usage", "scalability"],
                behavioral_features=["convergence", "adaptation", "trade_offs"]
            ),
            AnalogicalElement(
                name="network_topology",
                domain="engineering",
                description="Structure of network connections",
                structural_features=["nodes", "edges", "paths"],
                functional_features=["connectivity", "routing", "load_balancing"],
                behavioral_features=["resilience", "efficiency", "emergence"]
            )
        ]
        
        # Urban planning elements
        urban_elements = [
            AnalogicalElement(
                name="urban_planning",
                domain="urban_design",
                description="Design and organization of urban spaces",
                structural_features=["zones", "infrastructure", "connectivity"],
                functional_features=["efficiency", "livability", "sustainability"],
                behavioral_features=["growth", "adaptation", "optimization"]
            ),
            AnalogicalElement(
                name="traffic_flow",
                domain="urban_design",
                description="Movement patterns in transportation systems",
                structural_features=["networks", "bottlenecks", "routing"],
                functional_features=["throughput", "efficiency", "load_balancing"],
                behavioral_features=["congestion", "optimization", "adaptation"]
            )
        ]
        
        # Business domain elements
        business_elements = [
            AnalogicalElement(
                name="organizational_structure",
                domain="business",
                description="Hierarchical arrangement of business entities",
                structural_features=["hierarchies", "departments", "reporting_lines"],
                functional_features=["coordination", "decision_making", "resource_allocation"],
                behavioral_features=["efficiency", "adaptation", "communication"]
            ),
            AnalogicalElement(
                name="supply_chain",
                domain="business",
                description="Network of suppliers and distributors",
                structural_features=["networks", "nodes", "flows"],
                functional_features=["logistics", "inventory_management", "optimization"],
                behavioral_features=["resilience", "efficiency", "adaptation"]
            )
        ]
        
        # Store elements in domain mapping
        all_elements = bio_elements + eng_elements + urban_elements + business_elements
        
        for element in all_elements:
            self.domain_elements[element.domain].append(element)
            # Add to knowledge graph
            self.knowledge_graph.add_node(element.id, element=element)
    
    async def _establish_connections(self):
        """Establish connections between elements based on similarity"""
        
        all_elements = []
        for domain_list in self.domain_elements.values():
            all_elements.extend(domain_list)
            
        # Create connections based on feature similarity
        for i, elem1 in enumerate(all_elements):
            for elem2 in all_elements[i+1:]:
                similarity = self._calculate_element_similarity(elem1, elem2)
                
                if similarity > 0.3:  # Threshold for meaningful connection
                    # Determine relationship type based on strongest feature overlap
                    relation_type = self._determine_relation_type(elem1, elem2)
                    
                    # Create bidirectional connection
                    self.knowledge_graph.add_edge(elem1.id, elem2.id, 
                                                weight=similarity, 
                                                relation_type=relation_type)
                    self.knowledge_graph.add_edge(elem2.id, elem1.id, 
                                                weight=similarity, 
                                                relation_type=relation_type)
                    
                    self.element_connections[elem1.id].append(elem2.id)
                    self.element_connections[elem2.id].append(elem1.id)
    
    def _calculate_element_similarity(self, elem1: AnalogicalElement, elem2: AnalogicalElement) -> float:
        """Calculate similarity between two analogical elements"""
        
        if elem1.domain == elem2.domain:
            return 0.1  # Low similarity within same domain
        
        # Calculate feature overlaps
        structural_overlap = len(set(elem1.structural_features) & set(elem2.structural_features))
        functional_overlap = len(set(elem1.functional_features) & set(elem2.functional_features))
        behavioral_overlap = len(set(elem1.behavioral_features) & set(elem2.behavioral_features))
        
        # Normalize by total unique features
        total_features = len(set(elem1.structural_features + elem1.functional_features + elem1.behavioral_features +
                                elem2.structural_features + elem2.functional_features + elem2.behavioral_features))
        
        if total_features == 0:
            return 0.0
            
        similarity = (structural_overlap * 0.4 + functional_overlap * 0.4 + behavioral_overlap * 0.2) / (total_features * 0.3)
        
        return min(1.0, similarity)
    
    def _determine_relation_type(self, elem1: AnalogicalElement, elem2: AnalogicalElement) -> AnalogicalRelationType:
        """Determine the type of analogical relationship between elements"""
        
        structural_overlap = len(set(elem1.structural_features) & set(elem2.structural_features))
        functional_overlap = len(set(elem1.functional_features) & set(elem2.functional_features))
        behavioral_overlap = len(set(elem1.behavioral_features) & set(elem2.behavioral_features))
        
        if structural_overlap >= functional_overlap and structural_overlap >= behavioral_overlap:
            return AnalogicalRelationType.STRUCTURAL
        elif functional_overlap >= behavioral_overlap:
            return AnalogicalRelationType.FUNCTIONAL
        else:
            return AnalogicalRelationType.BEHAVIORAL
    
    async def discover_analogical_paths(self, 
                                      source_domain: str, 
                                      target_domain: str,
                                      query_context: str,
                                      max_paths: int = 10) -> List[List[str]]:
        """Discover analogical paths from source to target domain"""
        
        if source_domain not in self.domain_elements or target_domain not in self.domain_elements:
            return []
        
        source_elements = [elem.id for elem in self.domain_elements[source_domain]]
        target_elements = [elem.id for elem in self.domain_elements[target_domain]]
        
        all_paths = []
        
        # Find paths from each source element to each target element
        for source_id in source_elements:
            for target_id in target_elements:
                try:
                    # Use networkx to find all simple paths
                    paths = list(nx.all_simple_paths(
                        self.knowledge_graph, 
                        source_id, 
                        target_id, 
                        cutoff=self.max_chain_length
                    ))
                    
                    # Limit paths per pair
                    all_paths.extend(paths[:3])
                    
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.warning("Path finding failed", 
                                 source=source_id, target=target_id, error=str(e))
                    continue
        
        # Score and rank paths
        scored_paths = []
        for path in all_paths:
            score = await self._score_analogical_path(path, query_context)
            scored_paths.append((path, score))
        
        # Sort by score and return top paths
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return [path for path, score in scored_paths[:max_paths]]
    
    async def _score_analogical_path(self, path: List[str], query_context: str) -> float:
        """Score the quality of an analogical path"""
        
        if len(path) < 2:
            return 0.0
        
        # Get elements for the path
        elements = []
        for element_id in path:
            if element_id in self.knowledge_graph:
                elements.append(self.knowledge_graph.nodes[element_id]['element'])
        
        if not elements:
            return 0.0
        
        # Calculate path score based on multiple factors
        scores = []
        
        # 1. Average link strength
        link_strengths = []
        for i in range(len(path) - 1):
            if self.knowledge_graph.has_edge(path[i], path[i+1]):
                weight = self.knowledge_graph[path[i]][path[i+1]]['weight']
                link_strengths.append(weight)
        
        if link_strengths:
            scores.append(statistics.mean(link_strengths))
        
        # 2. Domain diversity (more diverse = higher score)
        domains = set(elem.domain for elem in elements)
        diversity_score = min(1.0, len(domains) / 4.0)  # Normalize by max expected domains
        scores.append(diversity_score)
        
        # 3. Length penalty (shorter paths preferred, but not too short)
        length_score = max(0.1, 1.0 - (len(path) - 2) * 0.15)  # Penalty for length > 2
        scores.append(length_score)
        
        # 4. Query relevance (simple keyword matching)
        relevance_score = 0.0
        query_words = set(query_context.lower().split())
        for element in elements:
            element_words = set((element.name + " " + element.description).lower().split())
            overlap = len(query_words & element_words)
            if overlap > 0:
                relevance_score += overlap / len(query_words)
        
        relevance_score = min(1.0, relevance_score / len(elements))
        scores.append(relevance_score)
        
        # Overall score
        return statistics.mean(scores) if scores else 0.0

class SemanticConsistencyValidator:
    """Validation of chain coherence and logical consistency"""
    
    def __init__(self):
        self.consistency_thresholds = {
            ConsistencyLevel.STRONG: 0.8,
            ConsistencyLevel.MODERATE: 0.6,
            ConsistencyLevel.WEAK: 0.4
        }
    
    async def validate_chain_consistency(self, chain: AnalogicalChain) -> Dict[str, Any]:
        """Validate the semantic consistency of an analogical chain"""
        
        if not chain.links:
            return {
                'consistency_score': 0.0,
                'consistency_level': ConsistencyLevel.INCONSISTENT,
                'validation_details': ['No links to validate'],
                'improvement_suggestions': ['Add analogical links to the chain']
            }
        
        # Validate different aspects of consistency
        link_consistency = await self._validate_link_consistency(chain.links)
        semantic_flow = await self._validate_semantic_flow(chain.elements, chain.links)
        domain_transitions = await self._validate_domain_transitions(chain.elements)
        abstraction_coherence = await self._validate_abstraction_coherence(chain.elements)
        
        # Calculate overall consistency
        consistency_scores = [
            link_consistency['score'],
            semantic_flow['score'],
            domain_transitions['score'],
            abstraction_coherence['score']
        ]
        
        overall_score = statistics.mean(consistency_scores)
        consistency_level = self._determine_consistency_level(overall_score)
        
        # Compile validation details
        validation_details = []
        validation_details.extend(link_consistency['details'])
        validation_details.extend(semantic_flow['details'])
        validation_details.extend(domain_transitions['details'])
        validation_details.extend(abstraction_coherence['details'])
        
        # Compile improvement suggestions
        suggestions = []
        suggestions.extend(link_consistency['suggestions'])
        suggestions.extend(semantic_flow['suggestions'])
        suggestions.extend(domain_transitions['suggestions'])
        suggestions.extend(abstraction_coherence['suggestions'])
        
        # Update chain with validation results
        chain.overall_consistency = overall_score
        chain.consistency_level = consistency_level
        chain.validation_evidence = validation_details[:10]  # Top 10
        chain.potential_weaknesses = suggestions[:5]  # Top 5
        
        return {
            'consistency_score': overall_score,
            'consistency_level': consistency_level,
            'validation_details': validation_details,
            'improvement_suggestions': suggestions,
            'component_scores': {
                'link_consistency': link_consistency['score'],
                'semantic_flow': semantic_flow['score'],
                'domain_transitions': domain_transitions['score'],
                'abstraction_coherence': abstraction_coherence['score']
            }
        }
    
    async def _validate_link_consistency(self, links: List[AnalogicalLink]) -> Dict[str, Any]:
        """Validate consistency of individual analogical links"""
        
        if not links:
            return {'score': 0.0, 'details': [], 'suggestions': []}
        
        details = []
        suggestions = []
        scores = []
        
        for link in links:
            # Check similarity score threshold
            if link.similarity_score >= 0.7:
                details.append(f"Strong link: {link.source_element.name} → {link.target_element.name}")
                scores.append(link.similarity_score)
            elif link.similarity_score >= 0.5:
                details.append(f"Moderate link: {link.source_element.name} → {link.target_element.name}")
                scores.append(link.similarity_score)
            else:
                suggestions.append(f"Weak link needs strengthening: {link.source_element.name} → {link.target_element.name}")
                scores.append(link.similarity_score * 0.5)  # Penalty for weak links
            
            # Check confidence levels
            if link.confidence < 0.5:
                suggestions.append(f"Low confidence link: {link.source_element.name} → {link.target_element.name}")
        
        avg_score = statistics.mean(scores) if scores else 0.0
        
        return {
            'score': avg_score,
            'details': details,
            'suggestions': suggestions
        }
    
    async def _validate_semantic_flow(self, elements: List[AnalogicalElement], links: List[AnalogicalLink]) -> Dict[str, Any]:
        """Validate semantic flow through the chain"""
        
        if len(elements) < 2:
            return {'score': 0.0, 'details': [], 'suggestions': []}
        
        details = []
        suggestions = []
        flow_scores = []
        
        # Check feature consistency through the chain
        for i in range(len(elements) - 1):
            elem1 = elements[i]
            elem2 = elements[i + 1]
            
            # Calculate feature preservation
            structural_preservation = self._calculate_feature_preservation(
                elem1.structural_features, elem2.structural_features
            )
            functional_preservation = self._calculate_feature_preservation(
                elem1.functional_features, elem2.functional_features
            )
            
            preservation_score = (structural_preservation + functional_preservation) / 2
            flow_scores.append(preservation_score)
            
            if preservation_score >= 0.6:
                details.append(f"Good semantic flow: {elem1.name} → {elem2.name}")
            else:
                suggestions.append(f"Improve semantic connection: {elem1.name} → {elem2.name}")
        
        avg_flow_score = statistics.mean(flow_scores) if flow_scores else 0.0
        
        return {
            'score': avg_flow_score,
            'details': details,
            'suggestions': suggestions
        }
    
    def _calculate_feature_preservation(self, features1: List[str], features2: List[str]) -> float:
        """Calculate how well features are preserved between elements"""
        
        if not features1 and not features2:
            return 1.0
        
        if not features1 or not features2:
            return 0.0
        
        # Calculate Jaccard similarity
        set1 = set(features1)
        set2 = set(features2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _validate_domain_transitions(self, elements: List[AnalogicalElement]) -> Dict[str, Any]:
        """Validate transitions between domains"""
        
        if len(elements) < 2:
            return {'score': 0.0, 'details': [], 'suggestions': []}
        
        details = []
        suggestions = []
        transition_scores = []
        
        domains = [elem.domain for elem in elements]
        
        # Check for meaningful domain diversity
        unique_domains = len(set(domains))
        if unique_domains >= 3:
            details.append(f"Good domain diversity: {unique_domains} domains traversed")
            transition_scores.append(0.8)
        elif unique_domains == 2:
            details.append(f"Moderate domain diversity: {unique_domains} domains")
            transition_scores.append(0.6)
        else:
            suggestions.append("Add more domain diversity to strengthen analogical reasoning")
            transition_scores.append(0.3)
        
        # Check for smooth transitions (not too many domain jumps)
        domain_changes = sum(1 for i in range(len(domains) - 1) if domains[i] != domains[i+1])
        if domain_changes <= len(domains) // 2:  # Reasonable number of transitions
            details.append("Smooth domain transitions")
            transition_scores.append(0.7)
        else:
            suggestions.append("Reduce excessive domain jumping for better coherence")
            transition_scores.append(0.4)
        
        avg_score = statistics.mean(transition_scores) if transition_scores else 0.0
        
        return {
            'score': avg_score,
            'details': details,
            'suggestions': suggestions
        }
    
    async def _validate_abstraction_coherence(self, elements: List[AnalogicalElement]) -> Dict[str, Any]:
        """Validate coherence of abstraction levels through chain"""
        
        if len(elements) < 2:
            return {'score': 0.0, 'details': [], 'suggestions': []}
        
        details = []
        suggestions = []
        
        abstraction_levels = [elem.abstraction_level for elem in elements]
        
        # Check for reasonable abstraction level progression
        max_jump = max(abs(abstraction_levels[i+1] - abstraction_levels[i]) 
                      for i in range(len(abstraction_levels) - 1))
        
        if max_jump <= 1:
            details.append("Consistent abstraction levels")
            coherence_score = 0.8
        elif max_jump <= 2:
            details.append("Moderate abstraction level changes")
            coherence_score = 0.6
        else:
            suggestions.append("Large abstraction jumps may weaken analogical coherence")
            coherence_score = 0.4
        
        return {
            'score': coherence_score,
            'details': details,
            'suggestions': suggestions
        }
    
    def _determine_consistency_level(self, score: float) -> ConsistencyLevel:
        """Determine consistency level based on score"""
        
        if score >= self.consistency_thresholds[ConsistencyLevel.STRONG]:
            return ConsistencyLevel.STRONG
        elif score >= self.consistency_thresholds[ConsistencyLevel.MODERATE]:
            return ConsistencyLevel.MODERATE
        elif score >= self.consistency_thresholds[ConsistencyLevel.WEAK]:
            return ConsistencyLevel.WEAK
        else:
            return ConsistencyLevel.INCONSISTENT

class ChainQualityAssessor:
    """Assessment of chain strength, novelty, and breakthrough potential"""
    
    def __init__(self):
        self.quality_weights = {
            'consistency': 0.25,
            'novelty': 0.25,
            'practicality': 0.20,
            'breakthrough_potential': 0.30
        }
    
    async def assess_chain_quality(self, chain: AnalogicalChain) -> Dict[str, Any]:
        """Comprehensive assessment of analogical chain quality"""
        
        # Assess different quality dimensions
        consistency_assessment = await self._assess_consistency(chain)
        novelty_assessment = await self._assess_novelty(chain)
        practicality_assessment = await self._assess_practicality(chain)
        breakthrough_assessment = await self._assess_breakthrough_potential(chain)
        
        # Calculate weighted overall quality
        overall_quality = (
            consistency_assessment * self.quality_weights['consistency'] +
            novelty_assessment * self.quality_weights['novelty'] +
            practicality_assessment * self.quality_weights['practicality'] +
            breakthrough_assessment * self.quality_weights['breakthrough_potential']
        )
        
        # Update chain with quality metrics
        chain.semantic_coherence = consistency_assessment
        chain.novelty_score = novelty_assessment
        chain.practical_applicability = practicality_assessment
        chain.breakthrough_potential = breakthrough_assessment
        chain.complexity_score = await self._calculate_complexity_score(chain)
        
        return {
            'overall_quality': overall_quality,
            'consistency_score': consistency_assessment,
            'novelty_score': novelty_assessment,
            'practicality_score': practicality_assessment,
            'breakthrough_potential': breakthrough_assessment,
            'complexity_score': chain.complexity_score,
            'quality_grade': self._assign_quality_grade(overall_quality),
            'strengths': await self._identify_strengths(chain),
            'improvement_areas': await self._identify_improvement_areas(chain)
        }
    
    async def _assess_consistency(self, chain: AnalogicalChain) -> float:
        """Assess consistency quality of the chain"""
        return chain.overall_consistency if chain.overall_consistency > 0 else 0.5
    
    async def _assess_novelty(self, chain: AnalogicalChain) -> float:
        """Assess novelty of the analogical chain"""
        
        novelty_factors = []
        
        # Domain diversity contributes to novelty
        unique_domains = len(set(elem.domain for elem in chain.elements))
        domain_novelty = min(1.0, unique_domains / 4.0)  # Normalize by expected max
        novelty_factors.append(domain_novelty)
        
        # Chain length contributes to novelty (longer chains are more novel but harder)
        length_novelty = min(1.0, (chain.chain_length - 2) / 4.0)  # 2-6 length range
        novelty_factors.append(length_novelty)
        
        # Uncommon connections contribute to novelty
        connection_novelty = 0.0
        for link in chain.links:
            if link.similarity_score < 0.6:  # Uncommon connections
                connection_novelty += 0.2
        connection_novelty = min(1.0, connection_novelty)
        novelty_factors.append(connection_novelty)
        
        return statistics.mean(novelty_factors) if novelty_factors else 0.5
    
    async def _assess_practicality(self, chain: AnalogicalChain) -> float:
        """Assess practical applicability of the chain"""
        
        practicality_factors = []
        
        # Shorter chains are generally more practical
        if chain.chain_length <= 4:
            practicality_factors.append(0.8)
        elif chain.chain_length <= 6:
            practicality_factors.append(0.6)
        else:
            practicality_factors.append(0.4)
        
        # Strong links contribute to practicality
        if chain.links:
            strong_links = sum(1 for link in chain.links if link.similarity_score >= 0.7)
            link_strength_ratio = strong_links / len(chain.links)
            practicality_factors.append(link_strength_ratio)
        
        # Lower abstraction levels are more practical
        if chain.elements:
            avg_abstraction = statistics.mean(elem.abstraction_level for elem in chain.elements)
            abstraction_practicality = max(0.2, 1.0 - avg_abstraction / 3.0)  # Normalize by max abstraction
            practicality_factors.append(abstraction_practicality)
        
        return statistics.mean(practicality_factors) if practicality_factors else 0.5
    
    async def _assess_breakthrough_potential(self, chain: AnalogicalChain) -> float:
        """Assess breakthrough potential of the chain"""
        
        breakthrough_factors = []
        
        # Novel domain combinations increase breakthrough potential
        if chain.elements:
            domains = [elem.domain for elem in chain.elements]
            unique_domains = len(set(domains))
            if unique_domains >= 4:
                breakthrough_factors.append(0.9)
            elif unique_domains >= 3:
                breakthrough_factors.append(0.7)
            else:
                breakthrough_factors.append(0.5)
        
        # Complex but coherent chains have higher breakthrough potential
        complexity_coherence_balance = 0.0
        if chain.complexity_score > 0 and chain.overall_consistency > 0:
            # Sweet spot: moderate complexity with high consistency
            if 0.5 <= chain.complexity_score <= 0.8 and chain.overall_consistency >= 0.7:
                complexity_coherence_balance = 0.8
            elif chain.complexity_score > 0.3 and chain.overall_consistency > 0.5:
                complexity_coherence_balance = 0.6
            else:
                complexity_coherence_balance = 0.4
        
        breakthrough_factors.append(complexity_coherence_balance)
        
        # Uncommon but strong connections suggest breakthrough potential
        if chain.links:
            novel_strong_links = sum(1 for link in chain.links 
                                   if 0.4 <= link.similarity_score < 0.7)  # Novel but meaningful
            if novel_strong_links > 0:
                breakthrough_factors.append(0.7)
            else:
                breakthrough_factors.append(0.4)
        
        return statistics.mean(breakthrough_factors) if breakthrough_factors else 0.5
    
    async def _calculate_complexity_score(self, chain: AnalogicalChain) -> float:
        """Calculate complexity score of the chain"""
        
        complexity_factors = []
        
        # Chain length contributes to complexity
        length_complexity = min(1.0, chain.chain_length / 6.0)
        complexity_factors.append(length_complexity)
        
        # Domain diversity contributes to complexity
        if chain.elements:
            unique_domains = len(set(elem.domain for elem in chain.elements))
            domain_complexity = min(1.0, unique_domains / 4.0)
            complexity_factors.append(domain_complexity)
        
        # Abstraction level variation contributes to complexity
        if chain.elements:
            abstractions = [elem.abstraction_level for elem in chain.elements]
            if len(abstractions) > 1:
                abstraction_variance = np.var(abstractions)
                abstraction_complexity = min(1.0, abstraction_variance / 2.0)
                complexity_factors.append(abstraction_complexity)
        
        return statistics.mean(complexity_factors) if complexity_factors else 0.5
    
    def _assign_quality_grade(self, quality_score: float) -> str:
        """Assign quality grade based on overall score"""
        
        if quality_score >= 0.9:
            return "A+ (Exceptional)"
        elif quality_score >= 0.8:
            return "A (Excellent)"
        elif quality_score >= 0.7:
            return "B+ (Very Good)"
        elif quality_score >= 0.6:
            return "B (Good)"
        elif quality_score >= 0.5:
            return "C+ (Fair)"
        elif quality_score >= 0.4:
            return "C (Needs Improvement)"
        else:
            return "D (Poor)"
    
    async def _identify_strengths(self, chain: AnalogicalChain) -> List[str]:
        """Identify strengths of the analogical chain"""
        
        strengths = []
        
        if chain.overall_consistency >= 0.7:
            strengths.append("High semantic consistency")
        
        if chain.novelty_score >= 0.7:
            strengths.append("High novelty and creativity")
        
        if chain.practical_applicability >= 0.7:
            strengths.append("Strong practical applicability")
        
        if chain.breakthrough_potential >= 0.7:
            strengths.append("High breakthrough potential")
        
        if len(set(elem.domain for elem in chain.elements)) >= 3:
            strengths.append("Excellent domain diversity")
        
        if chain.consistency_level in [ConsistencyLevel.STRONG, ConsistencyLevel.MODERATE]:
            strengths.append("Well-validated semantic coherence")
        
        return strengths
    
    async def _identify_improvement_areas(self, chain: AnalogicalChain) -> List[str]:
        """Identify areas for improvement in the analogical chain"""
        
        improvements = []
        
        if chain.overall_consistency < 0.6:
            improvements.append("Strengthen semantic consistency between elements")
        
        if chain.novelty_score < 0.5:
            improvements.append("Increase novelty through more diverse domain connections")
        
        if chain.practical_applicability < 0.5:
            improvements.append("Improve practical applicability by strengthening key links")
        
        if chain.breakthrough_potential < 0.6:
            improvements.append("Enhance breakthrough potential through more creative connections")
        
        if len(chain.domains_traversed) < 3:
            improvements.append("Add more domain diversity for richer analogical reasoning")
        
        return improvements

class BreakthroughChainIdentifier:
    """Identification of chains with highest innovation potential"""
    
    def __init__(self):
        self.breakthrough_thresholds = {
            'minimum_quality': 0.6,
            'minimum_novelty': 0.7,
            'minimum_breakthrough_potential': 0.7,
            'minimum_consistency': 0.5
        }
    
    async def identify_breakthrough_chains(self, chains: List[AnalogicalChain]) -> List[AnalogicalChain]:
        """Identify chains with highest breakthrough potential"""
        
        if not chains:
            return []
        
        breakthrough_chains = []
        
        for chain in chains:
            if await self._is_breakthrough_chain(chain):
                breakthrough_chains.append(chain)
        
        # Sort by breakthrough potential
        breakthrough_chains.sort(key=lambda x: x.breakthrough_potential, reverse=True)
        
        logger.info("Breakthrough chains identified",
                   total_chains=len(chains),
                   breakthrough_chains=len(breakthrough_chains))
        
        return breakthrough_chains
    
    async def _is_breakthrough_chain(self, chain: AnalogicalChain) -> bool:
        """Determine if a chain qualifies as breakthrough"""
        
        # Check all breakthrough criteria
        criteria_met = [
            chain.novelty_score >= self.breakthrough_thresholds['minimum_novelty'],
            chain.breakthrough_potential >= self.breakthrough_thresholds['minimum_breakthrough_potential'],
            chain.overall_consistency >= self.breakthrough_thresholds['minimum_consistency']
        ]
        
        # Additional breakthrough indicators
        breakthrough_indicators = []
        
        # High domain diversity
        if len(set(elem.domain for elem in chain.elements)) >= 4:
            breakthrough_indicators.append(True)
        
        # Novel connections (moderate similarity scores)
        novel_connections = sum(1 for link in chain.links 
                              if 0.4 <= link.similarity_score < 0.7)
        if novel_connections >= len(chain.links) * 0.5:
            breakthrough_indicators.append(True)
        
        # Complex but coherent structure
        if (0.5 <= chain.complexity_score <= 0.8 and 
            chain.overall_consistency >= 0.6):
            breakthrough_indicators.append(True)
        
        # Must meet basic criteria AND have breakthrough indicators
        return (all(criteria_met) and 
                len(breakthrough_indicators) >= 2)
    
    async def analyze_breakthrough_patterns(self, breakthrough_chains: List[AnalogicalChain]) -> Dict[str, Any]:
        """Analyze patterns in breakthrough chains"""
        
        if not breakthrough_chains:
            return {'error': 'No breakthrough chains to analyze'}
        
        # Analyze common patterns
        domain_combinations = defaultdict(int)
        chain_lengths = []
        breakthrough_scores = []
        
        for chain in breakthrough_chains:
            # Track domain combinations
            domains = tuple(sorted(set(elem.domain for elem in chain.elements)))
            domain_combinations[domains] += 1
            
            chain_lengths.append(chain.chain_length)
            breakthrough_scores.append(chain.breakthrough_potential)
        
        # Find most successful patterns
        top_domain_combinations = dict(sorted(domain_combinations.items(), 
                                            key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'total_breakthrough_chains': len(breakthrough_chains),
            'average_breakthrough_potential': statistics.mean(breakthrough_scores),
            'optimal_chain_length': statistics.mode(chain_lengths) if chain_lengths else 0,
            'successful_domain_combinations': top_domain_combinations,
            'breakthrough_rate': len(breakthrough_chains) / len(breakthrough_chains),  # Always 1.0 for filtered list
            'top_performing_chain': max(breakthrough_chains, 
                                      key=lambda x: x.breakthrough_potential) if breakthrough_chains else None
        }

class AnalogicalChainEngine:
    """Core engine for multi-hop analogical reasoning"""
    
    def __init__(self):
        self.path_finder = ChainPathFinder()
        self.consistency_validator = SemanticConsistencyValidator()
        self.quality_assessor = ChainQualityAssessor()
        self.breakthrough_identifier = BreakthroughChainIdentifier()
        
        self.chain_cache = {}  # query -> ChainDiscoveryResult
        
    async def discover_analogical_chains(self, 
                                       query: str,
                                       source_domain: str,
                                       target_domain: str,
                                       context: Dict[str, Any]) -> ChainDiscoveryResult:
        """Discover and analyze analogical chains for breakthrough reasoning"""
        
        start_time = time.time()
        
        try:
            # Initialize knowledge base
            await self.path_finder.initialize_knowledge_base(context)
            
            # Discover analogical paths
            paths = await self.path_finder.discover_analogical_paths(
                source_domain, target_domain, query, max_paths=20
            )
            
            logger.info("Analogical paths discovered", 
                       query=query[:50],
                       source_domain=source_domain,
                       target_domain=target_domain,
                       paths_found=len(paths))
            
            # Convert paths to analogical chains
            chains = []
            for path in paths:
                chain = await self._create_analogical_chain(path, query, target_domain)
                if chain:
                    chains.append(chain)
            
            # Validate and assess chain quality
            for chain in chains:
                # Validate consistency
                await self.consistency_validator.validate_chain_consistency(chain)
                
                # Assess quality
                await self.quality_assessor.assess_chain_quality(chain)
            
            # Identify breakthrough chains
            breakthrough_chains = await self.breakthrough_identifier.identify_breakthrough_chains(chains)
            
            # Select best chains
            best_chains = sorted(chains, 
                               key=lambda x: (x.breakthrough_potential + x.overall_consistency + x.novelty_score) / 3,
                               reverse=True)[:10]
            
            # Create discovery result
            result = ChainDiscoveryResult(
                query=query,
                discovered_chains=chains,
                best_chains=best_chains,
                breakthrough_chains=breakthrough_chains,
                total_paths_explored=len(paths),
                discovery_time=time.time() - start_time,
                chain_diversity=await self._calculate_chain_diversity(chains),
                average_chain_quality=await self._calculate_average_quality(chains),
                recommended_applications=await self._generate_application_recommendations(best_chains)
            )
            
            # Cache result
            self.chain_cache[query] = result
            
            logger.info("Analogical chain discovery completed",
                       query=query[:50],
                       total_chains=len(chains),
                       breakthrough_chains=len(breakthrough_chains),
                       discovery_time=result.discovery_time)
            
            return result
            
        except Exception as e:
            logger.error("Analogical chain discovery failed", error=str(e))
            return ChainDiscoveryResult(
                query=query,
                discovery_time=time.time() - start_time
            )
    
    async def _create_analogical_chain(self, 
                                     path: List[str], 
                                     query: str, 
                                     target_domain: str) -> Optional[AnalogicalChain]:
        """Create analogical chain from discovered path"""
        
        try:
            # Get elements from path
            elements = []
            for element_id in path:
                if element_id in self.path_finder.knowledge_graph:
                    element = self.path_finder.knowledge_graph.nodes[element_id]['element']
                    elements.append(element)
            
            if len(elements) < 2:
                return None
            
            # Create links between consecutive elements
            links = []
            for i in range(len(elements) - 1):
                source_elem = elements[i]
                target_elem = elements[i + 1]
                
                # Get relationship information from graph
                relation_type = AnalogicalRelationType.STRUCTURAL  # Default
                similarity_score = 0.5  # Default
                
                if self.path_finder.knowledge_graph.has_edge(source_elem.id, target_elem.id):
                    edge_data = self.path_finder.knowledge_graph[source_elem.id][target_elem.id]
                    similarity_score = edge_data.get('weight', 0.5)
                    relation_type = edge_data.get('relation_type', AnalogicalRelationType.STRUCTURAL)
                
                link = AnalogicalLink(
                    source_element=source_elem,
                    target_element=target_elem,
                    relation_type=relation_type,
                    similarity_score=similarity_score,
                    confidence=similarity_score * 0.8,  # Slightly lower than similarity
                    evidence=[f"Connection found between {source_elem.name} and {target_elem.name}"],
                    transformation_description=f"Analogical transformation from {source_elem.domain} to {target_elem.domain}",
                    breakthrough_potential=similarity_score if 0.4 <= similarity_score < 0.7 else similarity_score * 0.8
                )
                links.append(link)
            
            # Determine chain type
            domains = [elem.domain for elem in elements]
            unique_domains = len(set(domains))
            
            if unique_domains == len(domains):
                chain_type = ChainType.LINEAR
            elif unique_domains < len(domains) // 2:
                chain_type = ChainType.CYCLICAL
            else:
                chain_type = ChainType.BRANCHING
            
            # Create chain
            chain = AnalogicalChain(
                chain_type=chain_type,
                elements=elements,
                links=links,
                source_query=query,
                target_domain=target_domain,
                chain_length=len(elements),
                domains_traversed=list(set(domains)),
                abstraction_levels=[elem.abstraction_level for elem in elements]
            )
            
            return chain
            
        except Exception as e:
            logger.error("Failed to create analogical chain", error=str(e))
            return None
    
    async def _calculate_chain_diversity(self, chains: List[AnalogicalChain]) -> float:
        """Calculate diversity among discovered chains"""
        
        if not chains:
            return 0.0
        
        # Calculate diversity based on different factors
        diversity_factors = []
        
        # Domain combination diversity
        domain_combinations = set()
        for chain in chains:
            domains = tuple(sorted(chain.domains_traversed))
            domain_combinations.add(domains)
        
        domain_diversity = len(domain_combinations) / len(chains)
        diversity_factors.append(domain_diversity)
        
        # Chain length diversity
        lengths = [chain.chain_length for chain in chains]
        if lengths:
            length_variance = np.var(lengths)
            length_diversity = min(1.0, length_variance / 4.0)  # Normalize
            diversity_factors.append(length_diversity)
        
        # Chain type diversity
        chain_types = set(chain.chain_type for chain in chains)
        type_diversity = len(chain_types) / len(ChainType)
        diversity_factors.append(type_diversity)
        
        return statistics.mean(diversity_factors) if diversity_factors else 0.0
    
    async def _calculate_average_quality(self, chains: List[AnalogicalChain]) -> float:
        """Calculate average quality of discovered chains"""
        
        if not chains:
            return 0.0
        
        quality_scores = []
        for chain in chains:
            # Calculate composite quality score
            composite_quality = (
                chain.overall_consistency * 0.25 +
                chain.novelty_score * 0.25 +
                chain.practical_applicability * 0.20 +
                chain.breakthrough_potential * 0.30
            )
            quality_scores.append(composite_quality)
        
        return statistics.mean(quality_scores)
    
    async def _generate_application_recommendations(self, chains: List[AnalogicalChain]) -> List[str]:
        """Generate application recommendations based on best chains"""
        
        if not chains:
            return []
        
        recommendations = []
        
        # Analyze chain patterns for recommendations
        for chain in chains[:5]:  # Top 5 chains
            if chain.breakthrough_potential >= 0.7:
                recommendations.append(f"Explore breakthrough application: {chain.elements[0].name} → {chain.elements[-1].name}")
            
            if chain.practical_applicability >= 0.7:
                recommendations.append(f"Practical implementation: Apply {chain.elements[0].domain} insights to {chain.target_domain}")
            
            if len(chain.domains_traversed) >= 4:
                recommendations.append(f"Cross-domain innovation opportunity: {' → '.join(chain.domains_traversed[:3])}...")
        
        # Remove duplicates and limit
        return list(set(recommendations))[:10]
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        
        return {
            'total_queries_processed': len(self.chain_cache),
            'knowledge_base_elements': len([elem for domain_list in self.path_finder.domain_elements.values() 
                                          for elem in domain_list]),
            'knowledge_base_connections': len(self.path_finder.knowledge_graph.edges),
            'domains_available': len(self.path_finder.domain_elements),
            'cache_size': len(self.chain_cache)
        }

# Main interface function for integration with meta-reasoning engine
async def analogical_chain_reasoning_integration(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Analogical chain reasoning integration for multi-hop breakthrough reasoning"""
    
    # Extract domain information from context
    source_domain = context.get('source_domain', 'biology')
    target_domain = context.get('target_domain', 'engineering')
    
    # Create analogical chain engine
    chain_engine = AnalogicalChainEngine()
    
    # Discover analogical chains
    result = await chain_engine.discover_analogical_chains(query, source_domain, target_domain, context)
    
    # Convert to meta-reasoning format
    return {
        'conclusion': f"Analogical chain reasoning discovered {len(result.best_chains)} high-quality chains with {len(result.breakthrough_chains)} breakthrough opportunities",
        'confidence': min(1.0, result.average_chain_quality),
        'evidence': [f"Chain {i+1}: {' → '.join([elem.name for elem in chain.elements])}" 
                    for i, chain in enumerate(result.best_chains[:3])],
        'reasoning_chain': [
            f"Discovered {result.total_paths_explored} analogical paths",
            f"Generated {len(result.discovered_chains)} validated chains",
            f"Identified {len(result.breakthrough_chains)} breakthrough chains",
            f"Average chain quality: {result.average_chain_quality:.2f}"
        ],
        'processing_time': result.discovery_time,
        'quality_score': result.average_chain_quality,
        
        # Analogical chain specific results
        'analogical_chains': [
            {
                'chain_id': chain.id,
                'elements': [elem.name for elem in chain.elements],
                'domains': chain.domains_traversed,
                'chain_type': chain.chain_type.value,
                'consistency_score': chain.overall_consistency,
                'novelty_score': chain.novelty_score,
                'breakthrough_potential': chain.breakthrough_potential,
                'quality_grade': await chain_engine.quality_assessor._assign_quality_grade(
                    (chain.overall_consistency + chain.novelty_score + chain.breakthrough_potential) / 3
                )
            }
            for chain in result.best_chains[:5]
        ],
        'breakthrough_chains': len(result.breakthrough_chains),
        'chain_diversity': result.chain_diversity,
        'recommended_applications': result.recommended_applications,
        'analogical_chain_reasoning_enabled': True
    }

if __name__ == "__main__":
    # Test the analogical chain reasoning system
    async def test_analogical_chain_reasoning():
        test_query = "biomimetic approaches to solving urban traffic flow optimization"
        test_context = {
            "source_domain": "biology",
            "target_domain": "urban_design",
            "breakthrough_mode": "creative"
        }
        
        result = await analogical_chain_reasoning_integration(test_query, test_context)
        
        print("Analogical Chain Reasoning Test Results:")
        print("=" * 60)
        print(f"Query: {test_query}")
        print(f"Conclusion: {result['conclusion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Quality Score: {result['quality_score']:.2f}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        
        print(f"\nAnalogical Chains Discovered:")
        for i, chain in enumerate(result.get('analogical_chains', [])[:3], 1):
            print(f"{i}. {' → '.join(chain['elements'])}")
            print(f"   Domains: {', '.join(chain['domains'])}")
            print(f"   Quality: {chain['quality_grade']}")
            print(f"   Breakthrough Potential: {chain['breakthrough_potential']:.2f}")
        
        print(f"\nBreakthrough Chains: {result.get('breakthrough_chains', 0)}")
        print(f"Chain Diversity: {result.get('chain_diversity', 0):.2f}")
        
        if result.get('recommended_applications'):
            print(f"\nRecommended Applications:")
            for app in result['recommended_applications'][:3]:
                print(f"• {app}")
    
    asyncio.run(test_analogical_chain_reasoning())