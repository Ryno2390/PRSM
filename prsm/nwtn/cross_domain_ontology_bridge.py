#!/usr/bin/env python3
"""
Cross-Domain Ontology Bridge System for NWTN
===========================================

This module implements the Cross-Domain Ontology Bridge from the NWTN Novel Idea Generation Roadmap Phase 6.
It builds domain ontology graphs mapping concepts across disciplines and creates "conceptual bridges" 
between seemingly unrelated fields for breakthrough cross-domain insights.

Key Innovations:
1. **Domain Ontology Graph**: Maps concepts, relationships, and hierarchies across multiple research domains
2. **Conceptual Bridge Discovery**: Identifies semantic connections between seemingly unrelated fields
3. **Cross-Domain Concept Mapping**: Enables transfer of concepts, patterns, and solutions across domains
4. **Semantic Similarity Bridge**: Advanced algorithms for identifying conceptual relationships across domains
5. **Multi-Hop Bridge Traversal**: Enables complex cross-domain reasoning through connected concept chains

Architecture:
- DomainOntologyBuilder: Constructs domain-specific ontology graphs from research corpora
- ConceptualBridgeDetector: Identifies semantic bridges between different domains
- CrossDomainConceptMapper: Maps concepts and relationships across domain boundaries
- BridgeTraversalEngine: Enables multi-hop reasoning across conceptual bridges
- SemanticSimilarityCalculator: Computes semantic similarity scores for bridge quality assessment

Based on NWTN Roadmap Phase 6 - Cross-Domain Ontology Bridge (P3 Priority, High Effort)
Expected Impact: Revolutionary cross-domain insight discovery through systematic concept bridging
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
from collections import defaultdict, Counter, deque
import networkx as nx
import json
import re
import statistics
from itertools import combinations, product
import structlog
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os
import pickle

logger = structlog.get_logger(__name__)

class DomainType(Enum):
    """Types of knowledge domains"""
    SCIENTIFIC = "scientific"           # Natural sciences, mathematics
    TECHNOLOGICAL = "technological"     # Engineering, computer science
    MEDICAL = "medical"                 # Healthcare, biology, medicine
    SOCIAL = "social"                   # Psychology, sociology, anthropology
    ECONOMIC = "economic"               # Economics, finance, business
    PHILOSOPHICAL = "philosophical"     # Philosophy, ethics, logic
    ARTISTIC = "artistic"               # Arts, design, creativity
    HISTORICAL = "historical"           # History, archaeology, cultural studies
    LINGUISTIC = "linguistic"           # Language, communication, semiotics
    ENVIRONMENTAL = "environmental"     # Ecology, sustainability, climate

class BridgeType(Enum):
    """Types of conceptual bridges between domains"""
    ANALOGICAL = "analogical"           # Structural or functional similarity
    CAUSAL = "causal"                   # Cause-effect relationships
    TEMPORAL = "temporal"               # Time-based patterns or sequences
    SPATIAL = "spatial"                 # Spatial or geometric relationships
    HIERARCHICAL = "hierarchical"       # Nested or compositional structures
    FUNCTIONAL = "functional"           # Similar functions or purposes
    SEMANTIC = "semantic"               # Meaning or conceptual similarity
    MATHEMATICAL = "mathematical"       # Mathematical models or patterns
    SYSTEMIC = "systemic"               # System-level behaviors or properties
    EVOLUTIONARY = "evolutionary"       # Development or adaptation patterns

class ConceptType(Enum):
    """Types of concepts in the ontology"""
    ENTITY = "entity"                   # Things, objects, phenomena
    PROCESS = "process"                 # Actions, procedures, changes
    PROPERTY = "property"               # Attributes, characteristics, qualities
    RELATIONSHIP = "relationship"       # Connections, associations, dependencies
    PRINCIPLE = "principle"             # Laws, rules, fundamental truths
    PATTERN = "pattern"                 # Recurring structures or behaviors
    METHOD = "method"                   # Techniques, approaches, procedures
    GOAL = "goal"                       # Objectives, purposes, intentions
    CONSTRAINT = "constraint"           # Limitations, boundaries, restrictions
    EMERGENT = "emergent"               # Properties arising from complexity

@dataclass
class DomainConcept:
    """Represents a concept within a specific domain"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    domain: DomainType = DomainType.SCIENTIFIC
    concept_type: ConceptType = ConceptType.ENTITY
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # relation_type -> [concept_ids]
    examples: List[str] = field(default_factory=list)
    frequency: float = 0.0              # Frequency in domain corpus
    importance: float = 0.0             # Importance within domain
    abstraction_level: int = 0          # 0=concrete, higher=more abstract
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ConceptualBridge:
    """Represents a bridge between concepts across domains"""
    source_concept: DomainConcept
    target_concept: DomainConcept
    bridge_type: BridgeType
    id: str = field(default_factory=lambda: str(uuid4()))
    similarity_score: float = 0.0
    bridge_strength: float = 0.0        # Overall strength of the bridge
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    bidirectional: bool = True
    confidence: float = 0.0
    supporting_papers: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    breakthrough_potential: float = 0.0
    validation_status: str = "pending"  # pending, validated, rejected
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class BridgeTraversalPath:
    """Represents a multi-hop path across conceptual bridges"""
    start_concept: DomainConcept
    end_concept: DomainConcept
    id: str = field(default_factory=lambda: str(uuid4()))
    bridge_path: List[ConceptualBridge] = field(default_factory=list)
    path_length: int = 0
    total_similarity: float = 0.0
    average_bridge_strength: float = 0.0
    domains_traversed: List[DomainType] = field(default_factory=list)
    bridge_types_used: List[BridgeType] = field(default_factory=list)
    path_coherence: float = 0.0         # How well the path holds together
    novelty_score: float = 0.0          # How novel this path is
    breakthrough_potential: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class CrossDomainInsight:
    """Represents an insight discovered through cross-domain bridging"""
    source_domain: DomainType
    target_domain: DomainType
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    conceptual_bridges: List[ConceptualBridge] = field(default_factory=list)
    traversal_paths: List[BridgeTraversalPath] = field(default_factory=list)
    insight_type: str = ""              # discovery, solution, analogy, pattern
    novelty_score: float = 0.0
    breakthrough_potential: float = 0.0
    validation_confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    practical_applications: List[str] = field(default_factory=list)
    research_implications: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class DomainOntologyBuilder:
    """Builds domain-specific ontology graphs from research corpora"""
    
    def __init__(self):
        self.domain_graphs = {}  # domain -> NetworkX graph
        self.concept_extractors = {}
        self.relationship_detectors = {}
    
    async def build_domain_ontology(self, 
                                   domain: DomainType, 
                                   papers: List[Dict[str, Any]], 
                                   context: Dict[str, Any]) -> nx.DiGraph:
        """Build ontology graph for a specific domain"""
        try:
            # Create domain graph
            graph = nx.DiGraph()
            domain_concepts = {}
            
            # Extract concepts from papers
            concepts = await self._extract_domain_concepts(domain, papers, context)
            
            # Add concepts to graph
            for concept in concepts:
                graph.add_node(concept.id, 
                             concept=concept,
                             name=concept.name,
                             domain=concept.domain.value,
                             type=concept.concept_type.value,
                             importance=concept.importance)
                domain_concepts[concept.id] = concept
            
            # Detect relationships between concepts
            relationships = await self._detect_concept_relationships(concepts, papers, context)
            
            # Add relationships to graph
            for source_id, target_id, relation_type, strength in relationships:
                if source_id in domain_concepts and target_id in domain_concepts:
                    graph.add_edge(source_id, target_id,
                                 relation_type=relation_type,
                                 strength=strength)
            
            # Calculate centrality measures
            await self._calculate_concept_centralities(graph, domain_concepts)
            
            # Store domain graph
            self.domain_graphs[domain] = graph
            
            logger.info("Domain ontology built successfully",
                       domain=domain.value,
                       concepts=len(concepts),
                       relationships=len(relationships))
            
            return graph
            
        except Exception as e:
            logger.error("Failed to build domain ontology", 
                        domain=domain.value, error=str(e))
            return nx.DiGraph()
    
    async def _extract_domain_concepts(self, 
                                     domain: DomainType, 
                                     papers: List[Dict[str, Any]], 
                                     context: Dict[str, Any]) -> List[DomainConcept]:
        """Extract concepts from domain papers"""
        concepts = []
        
        # Domain-specific concept patterns
        concept_patterns = {
            DomainType.SCIENTIFIC: {
                ConceptType.PRINCIPLE: ["law", "theorem", "principle", "rule", "equation"],
                ConceptType.PHENOMENON: ["effect", "phenomenon", "behavior", "pattern"],
                ConceptType.METHOD: ["method", "technique", "approach", "algorithm"],
                ConceptType.ENTITY: ["particle", "system", "structure", "component"]
            },
            DomainType.TECHNOLOGICAL: {
                ConceptType.SYSTEM: ["system", "architecture", "platform", "framework"],
                ConceptType.PROCESS: ["algorithm", "protocol", "procedure", "workflow"],
                ConceptType.PROPERTY: ["performance", "efficiency", "scalability", "reliability"],
                ConceptType.METHOD: ["technique", "methodology", "approach", "strategy"]
            },
            DomainType.MEDICAL: {
                ConceptType.ENTITY: ["disease", "symptom", "organ", "cell", "gene"],
                ConceptType.PROCESS: ["diagnosis", "treatment", "therapy", "prevention"],
                ConceptType.PROPERTY: ["pathology", "physiology", "metabolism", "immunity"],
                ConceptType.METHOD: ["surgery", "medication", "intervention", "screening"]
            }
        }
        
        domain_patterns = concept_patterns.get(domain, {})
        
        # Extract concepts from papers
        for paper in papers[:50]:  # Limit for performance
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            text = f"{title} {abstract}".lower()
            
            # Extract concepts using patterns
            for concept_type, patterns in domain_patterns.items():
                for pattern in patterns:
                    if pattern in text:
                        concept = DomainConcept(
                            name=pattern,
                            domain=domain,
                            concept_type=concept_type,
                            description=f"{concept_type.value} concept from {domain.value} domain",
                            keywords=[pattern],
                            frequency=text.count(pattern),
                            importance=min(1.0, text.count(pattern) / 10.0)
                        )
                        concepts.append(concept)
        
        # Remove duplicates and rank by importance
        unique_concepts = {}
        for concept in concepts:
            if concept.name not in unique_concepts:
                unique_concepts[concept.name] = concept
            else:
                # Merge concepts with same name
                existing = unique_concepts[concept.name]
                existing.frequency += concept.frequency
                existing.importance = max(existing.importance, concept.importance)
        
        # Return top concepts
        sorted_concepts = sorted(unique_concepts.values(), 
                               key=lambda x: x.importance, reverse=True)
        return sorted_concepts[:100]  # Top 100 concepts per domain
    
    async def _detect_concept_relationships(self, 
                                          concepts: List[DomainConcept], 
                                          papers: List[Dict[str, Any]], 
                                          context: Dict[str, Any]) -> List[Tuple[str, str, str, float]]:
        """Detect relationships between concepts"""
        relationships = []
        
        # Common relationship patterns
        relation_patterns = {
            "causes": ["causes", "leads to", "results in", "produces"],
            "enables": ["enables", "allows", "facilitates", "supports"],
            "requires": ["requires", "needs", "depends on", "relies on"],
            "similar_to": ["similar to", "like", "analogous to", "comparable to"],
            "part_of": ["part of", "component of", "element of", "subset of"],
            "opposite_to": ["opposite to", "contrary to", "unlike", "different from"]
        }
        
        # Detect relationships in paper text
        for paper in papers[:30]:  # Limit for performance
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            for concept1, concept2 in combinations(concepts, 2):
                for relation_type, patterns in relation_patterns.items():
                    for pattern in patterns:
                        # Look for "concept1 pattern concept2"
                        if (concept1.name in text and concept2.name in text and 
                            pattern in text):
                            strength = 0.5  # Base strength
                            if f"{concept1.name} {pattern} {concept2.name}" in text:
                                strength = 1.0  # Direct relationship
                            elif (text.find(concept1.name) < text.find(pattern) < text.find(concept2.name)):
                                strength = 0.8  # Ordered relationship
                            
                            relationships.append((concept1.id, concept2.id, relation_type, strength))
        
        return relationships
    
    async def _calculate_concept_centralities(self, 
                                            graph: nx.DiGraph, 
                                            concepts: Dict[str, DomainConcept]):
        """Calculate centrality measures for concepts"""
        if len(graph.nodes) == 0:
            return
        
        try:
            # Calculate centrality measures
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            eigenvector = nx.eigenvector_centrality(graph, max_iter=100)
            pagerank = nx.pagerank(graph)
            
            # Update concept importance based on centrality
            for concept_id, concept in concepts.items():
                if concept_id in graph.nodes:
                    centrality_score = (
                        betweenness.get(concept_id, 0) * 0.3 +
                        closeness.get(concept_id, 0) * 0.3 +
                        eigenvector.get(concept_id, 0) * 0.2 +
                        pagerank.get(concept_id, 0) * 0.2
                    )
                    concept.importance = max(concept.importance, centrality_score)
        
        except Exception as e:
            logger.warning("Failed to calculate centralities", error=str(e))

class ConceptualBridgeDetector:
    """Detects semantic bridges between concepts across domains using embedding-based analysis"""
    
    def __init__(self, embeddings_path: str = "/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/embeddings"):
        self.similarity_threshold = 0.3
        self.bridge_validators = {}
        self.embeddings_path = embeddings_path
        self._embedding_cache = {}
        self._domain_embeddings = {}
        self._load_embeddings()
    
    async def detect_conceptual_bridges(self, 
                                      source_domain_graph: nx.DiGraph, 
                                      target_domain_graph: nx.DiGraph,
                                      source_domain: DomainType,
                                      target_domain: DomainType,
                                      context: Dict[str, Any]) -> List[ConceptualBridge]:
        """Detect conceptual bridges between two domain graphs using embedding-based analysis"""
        bridges = []
        
        try:
            # Get concepts from both domains
            source_concepts = [graph.nodes[node]['concept'] for node in source_domain_graph.nodes()]
            target_concepts = [graph.nodes[node]['concept'] for node in target_domain_graph.nodes()]
            
            # Detect bridges using multiple strategies including embeddings
            embedding_bridges = await self._detect_embedding_bridges(
                source_concepts, target_concepts, source_domain, target_domain
            )
            bridges.extend(embedding_bridges)
            
            semantic_bridges = await self._detect_semantic_bridges(
                source_concepts, target_concepts, source_domain, target_domain
            )
            bridges.extend(semantic_bridges)
            
            functional_bridges = await self._detect_functional_bridges(
                source_concepts, target_concepts, source_domain, target_domain
            )
            bridges.extend(functional_bridges)
            
            structural_bridges = await self._detect_structural_bridges(
                source_concepts, target_concepts, source_domain, target_domain
            )
            bridges.extend(structural_bridges)
            
            # Automated domain clustering bridges
            cluster_bridges = await self._detect_cluster_bridges(
                source_concepts, target_concepts, source_domain, target_domain
            )
            bridges.extend(cluster_bridges)
            
            # Calculate bridge quality scores
            for bridge in bridges:
                await self._calculate_bridge_quality(bridge, context)
            
            # Filter and rank bridges with embedding bonus
            quality_bridges = [b for b in bridges if b.similarity_score >= self.similarity_threshold]
            quality_bridges.sort(key=self._bridge_ranking_score, reverse=True)
            
            logger.info("Conceptual bridges detected with embedding analysis",
                       source_domain=source_domain.value,
                       target_domain=target_domain.value,
                       total_bridges=len(bridges),
                       quality_bridges=len(quality_bridges),
                       embedding_bridges=len(embedding_bridges))
            
            return quality_bridges[:50]  # Top 50 bridges
            
        except Exception as e:
            logger.error("Failed to detect conceptual bridges", error=str(e))
            return []
    
    async def _detect_semantic_bridges(self, 
                                     source_concepts: List[DomainConcept],
                                     target_concepts: List[DomainConcept],
                                     source_domain: DomainType,
                                     target_domain: DomainType) -> List[ConceptualBridge]:
        """Detect bridges based on semantic similarity"""
        bridges = []
        
        # Semantic similarity patterns
        similarity_indicators = [
            ("wave", "oscillation", "vibration", "frequency", "amplitude"),
            ("network", "graph", "connection", "node", "edge"),
            ("flow", "current", "stream", "transfer", "movement"),
            ("growth", "development", "evolution", "progression", "expansion"),
            ("structure", "architecture", "organization", "hierarchy", "framework"),
            ("pattern", "sequence", "order", "arrangement", "configuration"),
            ("system", "mechanism", "process", "function", "operation"),
            ("energy", "power", "force", "potential", "kinetic"),
            ("information", "data", "signal", "message", "communication"),
            ("feedback", "response", "adaptation", "control", "regulation")
        ]
        
        for source_concept in source_concepts[:20]:  # Limit for performance
            for target_concept in target_concepts[:20]:
                # Check semantic similarity
                similarity = self._calculate_semantic_similarity(
                    source_concept, target_concept, similarity_indicators
                )
                
                if similarity >= self.similarity_threshold:
                    bridge = ConceptualBridge(
                        source_concept=source_concept,
                        target_concept=target_concept,
                        bridge_type=BridgeType.SEMANTIC,
                        similarity_score=similarity,
                        explanation=f"Semantic similarity between {source_concept.name} and {target_concept.name}",
                        evidence=[f"Shared semantic patterns in {source_domain.value} and {target_domain.value}"]
                    )
                    bridges.append(bridge)
        
        return bridges
    
    async def _detect_functional_bridges(self, 
                                       source_concepts: List[DomainConcept],
                                       target_concepts: List[DomainConcept],
                                       source_domain: DomainType,
                                       target_domain: DomainType) -> List[ConceptualBridge]:
        """Detect bridges based on functional similarity"""
        bridges = []
        
        # Functional similarity patterns
        functional_patterns = {
            "optimization": ["minimize", "maximize", "optimize", "improve", "enhance"],
            "transformation": ["convert", "transform", "change", "modify", "adapt"],
            "coordination": ["coordinate", "synchronize", "align", "organize", "manage"],
            "protection": ["protect", "defend", "shield", "preserve", "maintain"],
            "transmission": ["transmit", "send", "carry", "convey", "propagate"],
            "detection": ["detect", "sense", "identify", "recognize", "discover"],
            "storage": ["store", "save", "retain", "preserve", "cache"],
            "filtering": ["filter", "select", "separate", "purify", "refine"],
            "amplification": ["amplify", "strengthen", "boost", "enhance", "increase"],
            "regulation": ["regulate", "control", "manage", "govern", "moderate"]
        }
        
        for source_concept in source_concepts[:15]:
            for target_concept in target_concepts[:15]:
                # Check functional similarity
                functional_similarity = self._calculate_functional_similarity(
                    source_concept, target_concept, functional_patterns
                )
                
                if functional_similarity >= self.similarity_threshold:
                    bridge = ConceptualBridge(
                        source_concept=source_concept,
                        target_concept=target_concept,
                        bridge_type=BridgeType.FUNCTIONAL,
                        similarity_score=functional_similarity,
                        explanation=f"Functional similarity between {source_concept.name} and {target_concept.name}",
                        evidence=[f"Shared functional patterns across {source_domain.value} and {target_domain.value}"]
                    )
                    bridges.append(bridge)
        
        return bridges
    
    async def _detect_structural_bridges(self, 
                                       source_concepts: List[DomainConcept],
                                       target_concepts: List[DomainConcept],
                                       source_domain: DomainType,
                                       target_domain: DomainType) -> List[ConceptualBridge]:
        """Detect bridges based on structural similarity"""
        bridges = []
        
        # Structural similarity patterns
        structural_patterns = {
            "hierarchical": ["hierarchy", "levels", "layers", "tree", "branches"],
            "network": ["network", "connections", "links", "nodes", "graph"],
            "cyclical": ["cycle", "loop", "circular", "periodic", "recurring"],
            "linear": ["linear", "sequence", "chain", "series", "progression"],
            "parallel": ["parallel", "concurrent", "simultaneous", "multiple"],
            "recursive": ["recursive", "self-similar", "fractal", "nested", "repeated"],
            "modular": ["modular", "components", "parts", "modules", "blocks"],
            "distributed": ["distributed", "spread", "scattered", "decentralized"],
            "centralized": ["centralized", "focused", "concentrated", "unified"],
            "emergent": ["emergent", "arising", "collective", "system-level"]
        }
        
        for source_concept in source_concepts[:15]:
            for target_concept in target_concepts[:15]:
                # Check structural similarity
                structural_similarity = self._calculate_structural_similarity(
                    source_concept, target_concept, structural_patterns
                )
                
                if structural_similarity >= self.similarity_threshold:
                    bridge = ConceptualBridge(
                        source_concept=source_concept,
                        target_concept=target_concept,
                        bridge_type=BridgeType.STRUCTURAL,
                        similarity_score=structural_similarity,
                        explanation=f"Structural similarity between {source_concept.name} and {target_concept.name}",
                        evidence=[f"Shared structural patterns across {source_domain.value} and {target_domain.value}"]
                    )
                    bridges.append(bridge)
        
        return bridges
    
    def _calculate_semantic_similarity(self, 
                                     concept1: DomainConcept,
                                     concept2: DomainConcept,
                                     similarity_indicators: List[Tuple]) -> float:
        """Calculate semantic similarity between concepts"""
        
        # Keywords overlap
        keywords1 = set([concept1.name] + concept1.keywords + concept1.synonyms)
        keywords2 = set([concept2.name] + concept2.keywords + concept2.synonyms)
        keyword_overlap = len(keywords1.intersection(keywords2)) / max(len(keywords1.union(keywords2)), 1)
        
        # Semantic pattern matching
        pattern_score = 0.0
        for pattern in similarity_indicators:
            pattern_set = set(pattern)
            score1 = len(keywords1.intersection(pattern_set)) / len(pattern_set)
            score2 = len(keywords2.intersection(pattern_set)) / len(pattern_set)
            if score1 > 0 and score2 > 0:
                pattern_score = max(pattern_score, min(score1, score2))
        
        # Description similarity (simple word overlap)
        desc1_words = set(concept1.description.lower().split())
        desc2_words = set(concept2.description.lower().split())
        desc_overlap = len(desc1_words.intersection(desc2_words)) / max(len(desc1_words.union(desc2_words)), 1)
        
        return (keyword_overlap * 0.4 + pattern_score * 0.4 + desc_overlap * 0.2)
    
    def _calculate_functional_similarity(self, 
                                       concept1: DomainConcept,
                                       concept2: DomainConcept,
                                       functional_patterns: Dict[str, List[str]]) -> float:
        """Calculate functional similarity between concepts"""
        
        text1 = f"{concept1.name} {concept1.description} {' '.join(concept1.keywords)}".lower()
        text2 = f"{concept2.name} {concept2.description} {' '.join(concept2.keywords)}".lower()
        
        functional_score = 0.0
        for function_type, patterns in functional_patterns.items():
            score1 = sum(1 for pattern in patterns if pattern in text1) / len(patterns)
            score2 = sum(1 for pattern in patterns if pattern in text2) / len(patterns)
            if score1 > 0 and score2 > 0:
                functional_score = max(functional_score, min(score1, score2))
        
        return functional_score
    
    def _calculate_structural_similarity(self, 
                                       concept1: DomainConcept,
                                       concept2: DomainConcept,
                                       structural_patterns: Dict[str, List[str]]) -> float:
        """Calculate structural similarity between concepts"""
        
        text1 = f"{concept1.name} {concept1.description} {' '.join(concept1.keywords)}".lower()
        text2 = f"{concept2.name} {concept2.description} {' '.join(concept2.keywords)}".lower()
        
        structural_score = 0.0
        for structure_type, patterns in structural_patterns.items():
            score1 = sum(1 for pattern in patterns if pattern in text1) / len(patterns)
            score2 = sum(1 for pattern in patterns if pattern in text2) / len(patterns)
            if score1 > 0 and score2 > 0:
                structural_score = max(structural_score, min(score1, score2))
        
        return structural_score
    
    def _load_embeddings(self):
        """Load 100K embeddings for cross-domain analysis"""
        try:
            # Load embeddings if they exist
            if os.path.exists(self.embeddings_path):
                embedding_files = [f for f in os.listdir(self.embeddings_path) if f.endswith('.pkl')]
                logger.info(f"Loading embeddings from {len(embedding_files)} files", 
                           embeddings_path=self.embeddings_path)
                
                # Load a sample of embeddings for performance
                for file in embedding_files[:10]:  # Limit to first 10 files
                    try:
                        file_path = os.path.join(self.embeddings_path, file)
                        with open(file_path, 'rb') as f:
                            embeddings_data = pickle.load(f)
                            if isinstance(embeddings_data, dict):
                                self._embedding_cache.update(embeddings_data)
                    except Exception as e:
                        logger.warning(f"Failed to load embedding file {file}", error=str(e))
                        
                logger.info(f"Loaded {len(self._embedding_cache)} embeddings for cross-domain analysis")
            else:
                logger.warning("Embeddings path not found, using pattern-based analysis only")
                
        except Exception as e:
            logger.error("Failed to load embeddings", error=str(e))
    
    async def _detect_embedding_bridges(self, 
                                      source_concepts: List[DomainConcept],
                                      target_concepts: List[DomainConcept],
                                      source_domain: DomainType,
                                      target_domain: DomainType) -> List[ConceptualBridge]:
        """Detect bridges using embedding-based cosine similarity"""
        bridges = []
        
        if not self._embedding_cache:
            return bridges
        
        try:
            # Get embeddings for concepts
            source_embeddings = []
            source_concept_map = []
            
            for concept in source_concepts[:20]:  # Limit for performance
                embedding = self._get_concept_embedding(concept)
                if embedding is not None:
                    source_embeddings.append(embedding)
                    source_concept_map.append(concept)
            
            target_embeddings = []
            target_concept_map = []
            
            for concept in target_concepts[:20]:  # Limit for performance
                embedding = self._get_concept_embedding(concept)
                if embedding is not None:
                    target_embeddings.append(embedding)
                    target_concept_map.append(concept)
            
            if not source_embeddings or not target_embeddings:
                return bridges
            
            # Calculate cosine similarities
            source_matrix = np.array(source_embeddings)
            target_matrix = np.array(target_embeddings)
            
            similarity_matrix = cosine_similarity(source_matrix, target_matrix)
            
            # Create bridges for high similarity pairs
            for i, source_concept in enumerate(source_concept_map):
                for j, target_concept in enumerate(target_concept_map):
                    similarity = similarity_matrix[i, j]
                    
                    if similarity >= self.similarity_threshold:
                        bridge = ConceptualBridge(
                            source_concept=source_concept,
                            target_concept=target_concept,
                            bridge_type=BridgeType.SEMANTIC,
                            similarity_score=float(similarity),
                            explanation=f"Embedding-based similarity: {similarity:.3f} between {source_concept.name} and {target_concept.name}",
                            evidence=[f"Cosine similarity of embeddings: {similarity:.3f}"],
                            confidence=min(1.0, similarity * 1.2)
                        )
                        # Add embedding similarity as additional attribute
                        bridge.embedding_similarity = similarity
                        bridges.append(bridge)
            
            logger.info(f"Detected {len(bridges)} embedding-based bridges",
                       source_domain=source_domain.value,
                       target_domain=target_domain.value)
            
        except Exception as e:
            logger.error("Failed to detect embedding bridges", error=str(e))
        
        return bridges
    
    async def _detect_cluster_bridges(self, 
                                    source_concepts: List[DomainConcept],
                                    target_concepts: List[DomainConcept],
                                    source_domain: DomainType,
                                    target_domain: DomainType) -> List[ConceptualBridge]:
        """Detect bridges using automated domain clustering"""
        bridges = []
        
        if not self._embedding_cache:
            return bridges
        
        try:
            # Combine all concepts and their embeddings
            all_concepts = source_concepts[:15] + target_concepts[:15]
            all_embeddings = []
            concept_map = []
            
            for concept in all_concepts:
                embedding = self._get_concept_embedding(concept)
                if embedding is not None:
                    all_embeddings.append(embedding)
                    concept_map.append(concept)
            
            if len(all_embeddings) < 4:  # Need minimum concepts for clustering
                return bridges
            
            # Perform DBSCAN clustering
            embeddings_matrix = np.array(all_embeddings)
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(embeddings_matrix)
            
            # Group concepts by clusters
            clusters = defaultdict(list)
            for i, label in enumerate(clustering.labels_):
                if label != -1:  # Ignore noise points
                    clusters[label].append(concept_map[i])
            
            # Create bridges within clusters that span domains
            for cluster_id, cluster_concepts in clusters.items():
                if len(cluster_concepts) >= 2:
                    # Check if cluster spans multiple domains
                    domains_in_cluster = set(c.domain for c in cluster_concepts)
                    if len(domains_in_cluster) >= 2:
                        # Create bridges between concepts from different domains
                        for concept1, concept2 in combinations(cluster_concepts, 2):
                            if concept1.domain != concept2.domain:
                                # Calculate cluster coherence
                                coherence = self._calculate_cluster_coherence(cluster_concepts)
                                
                                bridge = ConceptualBridge(
                                    source_concept=concept1,
                                    target_concept=concept2,
                                    bridge_type=BridgeType.SEMANTIC,
                                    similarity_score=coherence,
                                    explanation=f"Automated clustering bridge in cluster {cluster_id} (coherence: {coherence:.3f})",
                                    evidence=[f"DBSCAN clustering grouped concepts with coherence {coherence:.3f}"],
                                    confidence=coherence
                                )
                                bridges.append(bridge)
            
            logger.info(f"Detected {len(bridges)} cluster-based bridges from {len(clusters)} clusters",
                       source_domain=source_domain.value,
                       target_domain=target_domain.value)
            
        except Exception as e:
            logger.error("Failed to detect cluster bridges", error=str(e))
        
        return bridges
    
    def _get_concept_embedding(self, concept: DomainConcept) -> Optional[np.ndarray]:
        """Get embedding for a concept by matching name/keywords"""
        # Try exact name match first
        if concept.name in self._embedding_cache:
            return self._embedding_cache[concept.name]
        
        # Try keyword matches
        for keyword in concept.keywords:
            if keyword in self._embedding_cache:
                return self._embedding_cache[keyword]
        
        # Try partial matches
        concept_text = concept.name.lower()
        for key, embedding in list(self._embedding_cache.items())[:100]:  # Limit search
            if concept_text in key.lower() or key.lower() in concept_text:
                return embedding
        
        return None
    
    def _calculate_cluster_coherence(self, cluster_concepts: List[DomainConcept]) -> float:
        """Calculate coherence score for a cluster of concepts"""
        if len(cluster_concepts) < 2:
            return 0.0
        
        # Get embeddings for cluster concepts
        embeddings = []
        for concept in cluster_concepts:
            embedding = self._get_concept_embedding(concept)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) < 2:
            return 0.5  # Default coherence
        
        # Calculate average pairwise similarity
        embeddings_matrix = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        return float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.5
    
    def _bridge_ranking_score(self, bridge: ConceptualBridge) -> float:
        """Enhanced bridge ranking that includes embedding similarity"""
        base_score = bridge.bridge_strength
        
        # Add embedding similarity bonus if available
        embedding_bonus = 0.0
        if hasattr(bridge, 'embedding_similarity'):
            embedding_bonus = bridge.embedding_similarity * 0.2
        
        return base_score + embedding_bonus
    
    async def _calculate_bridge_quality(self, bridge: ConceptualBridge, context: Dict[str, Any]):
        """Calculate overall bridge quality and strength with embedding enhancement"""
        
        # Bridge strength factors (adjusted for embedding analysis)
        similarity_weight = 0.25
        importance_weight = 0.2
        novelty_weight = 0.2
        coherence_weight = 0.15
        embedding_weight = 0.2  # New embedding factor
        
        # Importance factor (average of concept importances)
        importance_factor = (bridge.source_concept.importance + bridge.target_concept.importance) / 2
        
        # Novelty factor (cross-domain bridges are inherently novel)
        novelty_factor = 0.8 if bridge.source_concept.domain != bridge.target_concept.domain else 0.3
        
        # Coherence factor (based on bridge type and concept types)
        coherence_factor = 0.7  # Base coherence
        if bridge.source_concept.concept_type == bridge.target_concept.concept_type:
            coherence_factor += 0.2
        
        # Embedding factor (bonus for embedding-based bridges)
        embedding_factor = 0.5  # Default
        if hasattr(bridge, 'embedding_similarity'):
            embedding_factor = bridge.embedding_similarity
        
        # Calculate overall bridge strength
        bridge.bridge_strength = (
            bridge.similarity_score * similarity_weight +
            importance_factor * importance_weight +
            novelty_factor * novelty_weight +
            coherence_factor * coherence_weight +
            embedding_factor * embedding_weight
        )
        
        # Calculate confidence and breakthrough potential with embedding bonus
        embedding_bonus = 0.1 if hasattr(bridge, 'embedding_similarity') else 0.0
        bridge.confidence = min(1.0, (bridge.bridge_strength + embedding_bonus) * 1.2)
        bridge.breakthrough_potential = bridge.bridge_strength * novelty_factor * (1.0 + embedding_bonus)

class CrossDomainConceptMapper:
    """Maps concepts and relationships across domain boundaries using 100K embeddings"""
    
    def __init__(self, embeddings_path: str = "/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/embeddings"):
        self.mapping_strategies = {}
        self.validation_rules = {}
        self.embeddings_path = embeddings_path
        self._embedding_cache = {}
        self._load_embeddings()
    
    async def map_cross_domain_concepts(self, 
                                      bridges: List[ConceptualBridge],
                                      source_query: str,
                                      context: Dict[str, Any]) -> List[CrossDomainInsight]:
        """Map concepts across domains to generate insights using embedding-enhanced analysis"""
        insights = []
        
        try:
            # Group bridges by domain pairs
            domain_pairs = defaultdict(list)
            for bridge in bridges:
                pair = (bridge.source_concept.domain, bridge.target_concept.domain)
                domain_pairs[pair].append(bridge)
            
            # Generate embedding-enhanced insights for each domain pair
            for (source_domain, target_domain), pair_bridges in domain_pairs.items():
                if len(pair_bridges) < 2:  # Need multiple bridges for insights
                    continue
                
                pair_insights = await self._generate_domain_pair_insights(
                    source_domain, target_domain, pair_bridges, source_query, context
                )
                insights.extend(pair_insights)
            
            # Generate multi-domain insights with embedding clustering
            multi_domain_insights = await self._generate_multi_domain_insights(
                bridges, source_query, context
            )
            insights.extend(multi_domain_insights)
            
            # Generate embedding-based pattern insights
            embedding_insights = await self._generate_embedding_pattern_insights(
                bridges, source_query, context
            )
            insights.extend(embedding_insights)
            
            # Enhanced ranking with embedding similarity
            for insight in insights:
                await self._enhance_insight_with_embeddings(insight)
            
            insights.sort(key=lambda x: x.breakthrough_potential, reverse=True)
            
            logger.info("Enhanced cross-domain concept mapping completed",
                       total_insights=len(insights),
                       domain_pairs=len(domain_pairs),
                       embedding_enhanced=len([i for i in insights if hasattr(i, 'embedding_coherence')]))
            
            return insights[:20]  # Top 20 insights
            
        except Exception as e:
            logger.error("Failed to map cross-domain concepts", error=str(e))
            return []
    
    async def _generate_domain_pair_insights(self, 
                                           source_domain: DomainType,
                                           target_domain: DomainType,
                                           bridges: List[ConceptualBridge],
                                           query: str,
                                           context: Dict[str, Any]) -> List[CrossDomainInsight]:
        """Generate insights for a specific domain pair"""
        insights = []
        
        # Insight generation patterns
        insight_patterns = {
            "analogy_transfer": "Concepts from {source} domain can be understood through {target} analogies",
            "solution_transfer": "Solutions from {target} domain may apply to {source} problems", 
            "pattern_recognition": "Similar patterns in {source} and {target} suggest underlying principles",
            "method_adaptation": "Methods from {target} could be adapted for {source} applications",
            "principle_unification": "Common principles underlying both {source} and {target} domains"
        }
        
        # Select strongest bridges for insight generation
        strong_bridges = sorted(bridges, key=lambda x: x.bridge_strength, reverse=True)[:10]
        
        for pattern_name, pattern_template in insight_patterns.items():
            if len(strong_bridges) >= 2:
                insight = CrossDomainInsight(
                    title=f"{pattern_name.replace('_', ' ').title()}: {source_domain.value} ↔ {target_domain.value}",
                    description=pattern_template.format(source=source_domain.value, target=target_domain.value),
                    source_domain=source_domain,
                    target_domain=target_domain,
                    conceptual_bridges=strong_bridges[:3],  # Top 3 bridges
                    insight_type=pattern_name
                )
                
                # Calculate insight metrics
                await self._calculate_insight_metrics(insight, query, context)
                insights.append(insight)
        
        return insights
    
    async def _generate_multi_domain_insights(self, 
                                            bridges: List[ConceptualBridge],
                                            query: str,
                                            context: Dict[str, Any]) -> List[CrossDomainInsight]:
        """Generate insights involving multiple domains"""
        insights = []
        
        # Group bridges by concepts for multi-hop insights
        concept_bridges = defaultdict(list)
        for bridge in bridges:
            concept_bridges[bridge.source_concept.id].append(bridge)
            concept_bridges[bridge.target_concept.id].append(bridge)
        
        # Find concepts that bridge multiple domains
        multi_domain_concepts = {
            concept_id: bridges_list for concept_id, bridges_list in concept_bridges.items()
            if len(set(b.source_concept.domain for b in bridges_list) | 
                   set(b.target_concept.domain for b in bridges_list)) >= 3
        }
        
        # Generate multi-domain insights
        for concept_id, concept_bridges in list(multi_domain_concepts.items())[:5]:
            domains_involved = set(b.source_concept.domain for b in concept_bridges) | \
                             set(b.target_concept.domain for b in concept_bridges)
            
            insight = CrossDomainInsight(
                title=f"Multi-Domain Hub: {len(domains_involved)} domains connected",
                description=f"Central concept connecting {', '.join(d.value for d in domains_involved)} domains",
                source_domain=list(domains_involved)[0],
                target_domain=list(domains_involved)[1],
                conceptual_bridges=concept_bridges[:5],
                insight_type="multi_domain_hub"
            )
            
            await self._calculate_insight_metrics(insight, query, context)
            insights.append(insight)
        
        return insights
    
    async def _calculate_insight_metrics(self, 
                                       insight: CrossDomainInsight, 
                                       query: str, 
                                       context: Dict[str, Any]):
        """Calculate metrics for cross-domain insight quality with embedding enhancement"""
        
        if not insight.conceptual_bridges:
            return
        
        # Novelty score based on domain diversity
        domains_involved = set([insight.source_domain, insight.target_domain])
        domains_involved.update([b.source_concept.domain for b in insight.conceptual_bridges])
        domains_involved.update([b.target_concept.domain for b in insight.conceptual_bridges])
        insight.novelty_score = min(1.0, len(domains_involved) / 5.0)
        
        # Breakthrough potential based on bridge strengths, novelty, and embedding similarity
        avg_bridge_strength = statistics.mean([b.bridge_strength for b in insight.conceptual_bridges])
        
        # Embedding enhancement bonus
        embedding_bonus = 0.0
        embedding_bridges = [b for b in insight.conceptual_bridges if hasattr(b, 'embedding_similarity')]
        if embedding_bridges:
            avg_embedding_similarity = statistics.mean([b.embedding_similarity for b in embedding_bridges])
            embedding_bonus = avg_embedding_similarity * 0.2
        
        insight.breakthrough_potential = (avg_bridge_strength + insight.novelty_score + embedding_bonus) / 2
        
        # Validation confidence based on bridge confidences
        avg_confidence = statistics.mean([b.confidence for b in insight.conceptual_bridges])
        insight.validation_confidence = avg_confidence + embedding_bonus * 0.1
        
        # Generate enhanced practical applications with embedding insights
        base_applications = [
            f"Apply {insight.target_domain.value} methods to {insight.source_domain.value} problems",
            f"Transfer insights from {insight.source_domain.value} to {insight.target_domain.value}",
            f"Develop hybrid approaches combining {insight.source_domain.value} and {insight.target_domain.value}"
        ]
        
        if embedding_bridges:
            base_applications.append(
                f"Leverage embedding-based similarities for automated cross-domain discovery"
            )
        
        insight.practical_applications = base_applications
        
        # Generate enhanced research implications
        base_implications = [
            f"Investigate common principles underlying {insight.source_domain.value} and {insight.target_domain.value}",
            f"Develop formal models bridging {insight.source_domain.value} and {insight.target_domain.value}",
            f"Explore novel applications at the intersection of these domains"
        ]
        
        if embedding_bridges:
            base_implications.append(
                f"Use 100K embedding space to discover latent cross-domain patterns"
            )
        
        insight.research_implications = base_implications
    
    def _load_embeddings(self):
        """Load 100K embeddings for concept mapping"""
        try:
            if os.path.exists(self.embeddings_path):
                embedding_files = [f for f in os.listdir(self.embeddings_path) if f.endswith('.pkl')]
                logger.info(f"Loading embeddings for concept mapping from {len(embedding_files)} files")
                
                # Load a sample of embeddings
                for file in embedding_files[:10]:  # Limit to first 10 files
                    try:
                        file_path = os.path.join(self.embeddings_path, file)
                        with open(file_path, 'rb') as f:
                            embeddings_data = pickle.load(f)
                            if isinstance(embeddings_data, dict):
                                self._embedding_cache.update(embeddings_data)
                    except Exception as e:
                        logger.warning(f"Failed to load embedding file {file}", error=str(e))
                        
                logger.info(f"Loaded {len(self._embedding_cache)} embeddings for concept mapping")
            else:
                logger.warning("Embeddings path not found for concept mapping")
                
        except Exception as e:
            logger.error("Failed to load embeddings for concept mapping", error=str(e))
    
    async def _generate_embedding_pattern_insights(self, 
                                                 bridges: List[ConceptualBridge],
                                                 query: str,
                                                 context: Dict[str, Any]) -> List[CrossDomainInsight]:
        """Generate insights based on embedding space patterns"""
        insights = []
        
        if not self._embedding_cache:
            return insights
        
        try:
            # Get embedding bridges
            embedding_bridges = [b for b in bridges if hasattr(b, 'embedding_similarity')]
            
            if len(embedding_bridges) < 3:
                return insights
            
            # Group by similarity ranges
            high_similarity_bridges = [b for b in embedding_bridges if b.embedding_similarity > 0.7]
            medium_similarity_bridges = [b for b in embedding_bridges if 0.4 <= b.embedding_similarity <= 0.7]
            
            # Generate high-similarity insight
            if high_similarity_bridges:
                domains_involved = set()
                for bridge in high_similarity_bridges:
                    domains_involved.add(bridge.source_concept.domain)
                    domains_involved.add(bridge.target_concept.domain)
                
                insight = CrossDomainInsight(
                    title=f"High-Similarity Embedding Patterns: {len(domains_involved)} domains",
                    description=f"Strong embedding similarities suggest deep conceptual connections across {', '.join(d.value for d in domains_involved)}",
                    source_domain=list(domains_involved)[0],
                    target_domain=list(domains_involved)[1] if len(domains_involved) > 1 else list(domains_involved)[0],
                    conceptual_bridges=high_similarity_bridges[:5],
                    insight_type="embedding_high_similarity"
                )
                
                await self._calculate_insight_metrics(insight, query, context)
                insights.append(insight)
            
            # Generate medium-similarity insight for broader patterns
            if medium_similarity_bridges:
                domains_involved = set()
                for bridge in medium_similarity_bridges:
                    domains_involved.add(bridge.source_concept.domain)
                    domains_involved.add(bridge.target_concept.domain)
                
                insight = CrossDomainInsight(
                    title=f"Moderate Embedding Patterns: Broader cross-domain connections",
                    description=f"Moderate embedding similarities reveal potential conceptual bridges across {len(domains_involved)} domains",
                    source_domain=list(domains_involved)[0],
                    target_domain=list(domains_involved)[1] if len(domains_involved) > 1 else list(domains_involved)[0],
                    conceptual_bridges=medium_similarity_bridges[:5],
                    insight_type="embedding_moderate_similarity"
                )
                
                await self._calculate_insight_metrics(insight, query, context)
                insights.append(insight)
            
        except Exception as e:
            logger.error("Failed to generate embedding pattern insights", error=str(e))
        
        return insights
    
    async def _enhance_insight_with_embeddings(self, insight: CrossDomainInsight):
        """Enhance insight with embedding-based analysis"""
        if not self._embedding_cache or not insight.conceptual_bridges:
            return
        
        try:
            # Calculate embedding coherence for the insight
            embedding_bridges = [b for b in insight.conceptual_bridges if hasattr(b, 'embedding_similarity')]
            
            if embedding_bridges:
                # Calculate average embedding similarity
                avg_embedding_similarity = statistics.mean([b.embedding_similarity for b in embedding_bridges])
                insight.embedding_coherence = avg_embedding_similarity
                
                # Boost breakthrough potential based on embedding coherence
                embedding_boost = avg_embedding_similarity * 0.15
                insight.breakthrough_potential = min(1.0, insight.breakthrough_potential + embedding_boost)
                
                # Add embedding-based evidence
                insight.supporting_evidence.append(
                    f"Embedding analysis shows {avg_embedding_similarity:.3f} average similarity across {len(embedding_bridges)} concept pairs"
                )
        
        except Exception as e:
            logger.error("Failed to enhance insight with embeddings", error=str(e))

class BridgeTraversalEngine:
    """Enables multi-hop reasoning across conceptual bridges"""
    
    def __init__(self, max_path_length: int = 5):
        self.max_path_length = max_path_length
        self.traversal_strategies = {}
    
    async def find_traversal_paths(self, 
                                 start_concept: DomainConcept,
                                 end_concept: DomainConcept,
                                 bridges: List[ConceptualBridge],
                                 context: Dict[str, Any]) -> List[BridgeTraversalPath]:
        """Find paths from start to end concept through bridges"""
        try:
            # Build bridge graph
            bridge_graph = self._build_bridge_graph(bridges)
            
            # Find all paths using different strategies
            paths = []
            
            # Breadth-first search for shortest paths
            bfs_paths = await self._find_bfs_paths(start_concept, end_concept, bridge_graph)
            paths.extend(bfs_paths)
            
            # Quality-first search for highest quality paths
            quality_paths = await self._find_quality_paths(start_concept, end_concept, bridge_graph)
            paths.extend(quality_paths)
            
            # Diversity-first search for most diverse paths
            diverse_paths = await self._find_diverse_paths(start_concept, end_concept, bridge_graph)
            paths.extend(diverse_paths)
            
            # Remove duplicates and calculate path metrics
            unique_paths = self._remove_duplicate_paths(paths)
            for path in unique_paths:
                await self._calculate_path_metrics(path)
            
            # Rank paths by breakthrough potential
            unique_paths.sort(key=lambda x: x.breakthrough_potential, reverse=True)
            
            logger.info("Bridge traversal completed",
                       start_concept=start_concept.name,
                       end_concept=end_concept.name,
                       paths_found=len(unique_paths))
            
            return unique_paths[:10]  # Top 10 paths
            
        except Exception as e:
            logger.error("Failed to find traversal paths", error=str(e))
            return []
    
    def _build_bridge_graph(self, bridges: List[ConceptualBridge]) -> nx.DiGraph:
        """Build graph representation of conceptual bridges"""
        graph = nx.DiGraph()
        
        for bridge in bridges:
            # Add nodes
            graph.add_node(bridge.source_concept.id, concept=bridge.source_concept)
            graph.add_node(bridge.target_concept.id, concept=bridge.target_concept)
            
            # Add edges (bidirectional if specified)
            graph.add_edge(bridge.source_concept.id, bridge.target_concept.id, 
                          bridge=bridge, weight=bridge.bridge_strength)
            
            if bridge.bidirectional:
                graph.add_edge(bridge.target_concept.id, bridge.source_concept.id,
                              bridge=bridge, weight=bridge.bridge_strength)
        
        return graph
    
    async def _find_bfs_paths(self, 
                            start_concept: DomainConcept,
                            end_concept: DomainConcept,
                            bridge_graph: nx.DiGraph) -> List[BridgeTraversalPath]:
        """Find shortest paths using breadth-first search"""
        paths = []
        
        try:
            # Use NetworkX to find all simple paths
            all_paths = list(nx.all_simple_paths(
                bridge_graph, start_concept.id, end_concept.id, 
                cutoff=self.max_path_length
            ))
            
            # Convert to BridgeTraversalPath objects
            for node_path in all_paths[:5]:  # Limit to 5 shortest paths
                bridge_path = []
                for i in range(len(node_path) - 1):
                    edge_data = bridge_graph[node_path[i]][node_path[i+1]]
                    bridge_path.append(edge_data['bridge'])
                
                path = BridgeTraversalPath(
                    start_concept=start_concept,
                    end_concept=end_concept,
                    bridge_path=bridge_path,
                    path_length=len(bridge_path)
                )
                paths.append(path)
                
        except nx.NetworkXNoPath:
            pass  # No path found
        except Exception as e:
            logger.warning("BFS path finding failed", error=str(e))
        
        return paths
    
    async def _find_quality_paths(self, 
                                start_concept: DomainConcept,
                                end_concept: DomainConcept,
                                bridge_graph: nx.DiGraph) -> List[BridgeTraversalPath]:
        """Find highest quality paths"""
        paths = []
        
        try:
            # Use Dijkstra's algorithm with inverted weights (higher weight = better)
            inverted_graph = bridge_graph.copy()
            for u, v, data in inverted_graph.edges(data=True):
                data['weight'] = 1.0 - data['weight']  # Invert weights
            
            try:
                shortest_path = nx.shortest_path(
                    inverted_graph, start_concept.id, end_concept.id, weight='weight'
                )
                
                # Convert to BridgeTraversalPath
                bridge_path = []
                for i in range(len(shortest_path) - 1):
                    edge_data = bridge_graph[shortest_path[i]][shortest_path[i+1]]
                    bridge_path.append(edge_data['bridge'])
                
                path = BridgeTraversalPath(
                    start_concept=start_concept,
                    end_concept=end_concept,
                    bridge_path=bridge_path,
                    path_length=len(bridge_path)
                )
                paths.append(path)
                
            except nx.NetworkXNoPath:
                pass
                
        except Exception as e:
            logger.warning("Quality path finding failed", error=str(e))
        
        return paths
    
    async def _find_diverse_paths(self, 
                                start_concept: DomainConcept,
                                end_concept: DomainConcept,
                                bridge_graph: nx.DiGraph) -> List[BridgeTraversalPath]:
        """Find most diverse paths (different domains/bridge types)"""
        paths = []
        
        try:
            # Find all simple paths and rank by diversity
            all_paths = list(nx.all_simple_paths(
                bridge_graph, start_concept.id, end_concept.id, 
                cutoff=self.max_path_length
            ))
            
            path_diversity_scores = []
            for node_path in all_paths:
                # Calculate diversity score
                domains_in_path = set()
                bridge_types_in_path = set()
                
                for i in range(len(node_path) - 1):
                    edge_data = bridge_graph[node_path[i]][node_path[i+1]]
                    bridge = edge_data['bridge']
                    domains_in_path.add(bridge.source_concept.domain)
                    domains_in_path.add(bridge.target_concept.domain)
                    bridge_types_in_path.add(bridge.bridge_type)
                
                diversity_score = len(domains_in_path) + len(bridge_types_in_path)
                path_diversity_scores.append((node_path, diversity_score))
            
            # Sort by diversity and take top paths
            path_diversity_scores.sort(key=lambda x: x[1], reverse=True)
            
            for node_path, _ in path_diversity_scores[:3]:  # Top 3 diverse paths
                bridge_path = []
                for i in range(len(node_path) - 1):
                    edge_data = bridge_graph[node_path[i]][node_path[i+1]]
                    bridge_path.append(edge_data['bridge'])
                
                path = BridgeTraversalPath(
                    start_concept=start_concept,
                    end_concept=end_concept,
                    bridge_path=bridge_path,
                    path_length=len(bridge_path)
                )
                paths.append(path)
                
        except Exception as e:
            logger.warning("Diverse path finding failed", error=str(e))
        
        return paths
    
    def _remove_duplicate_paths(self, paths: List[BridgeTraversalPath]) -> List[BridgeTraversalPath]:
        """Remove duplicate paths based on bridge sequences"""
        unique_paths = []
        seen_signatures = set()
        
        for path in paths:
            # Create signature based on bridge IDs
            signature = tuple(bridge.id for bridge in path.bridge_path)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_paths.append(path)
        
        return unique_paths
    
    async def _calculate_path_metrics(self, path: BridgeTraversalPath):
        """Calculate metrics for bridge traversal path"""
        if not path.bridge_path:
            return
        
        # Calculate total similarity and average bridge strength
        path.total_similarity = sum(bridge.similarity_score for bridge in path.bridge_path)
        path.average_bridge_strength = statistics.mean([bridge.bridge_strength for bridge in path.bridge_path])
        
        # Collect domains and bridge types traversed
        path.domains_traversed = []
        path.bridge_types_used = []
        for bridge in path.bridge_path:
            if bridge.source_concept.domain not in path.domains_traversed:
                path.domains_traversed.append(bridge.source_concept.domain)
            if bridge.target_concept.domain not in path.domains_traversed:
                path.domains_traversed.append(bridge.target_concept.domain)
            if bridge.bridge_type not in path.bridge_types_used:
                path.bridge_types_used.append(bridge.bridge_type)
        
        # Calculate path coherence (how well bridges connect)
        coherence_scores = []
        for i in range(len(path.bridge_path) - 1):
            current_bridge = path.bridge_path[i]
            next_bridge = path.bridge_path[i + 1]
            
            # Check if bridges share a concept
            if (current_bridge.target_concept.id == next_bridge.source_concept.id or
                current_bridge.source_concept.id == next_bridge.target_concept.id):
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(0.5)  # Loose connection
        
        path.path_coherence = statistics.mean(coherence_scores) if coherence_scores else 1.0
        
        # Calculate novelty score based on domain diversity
        path.novelty_score = min(1.0, len(path.domains_traversed) / 4.0)
        
        # Calculate breakthrough potential
        path.breakthrough_potential = (
            path.average_bridge_strength * 0.4 +
            path.path_coherence * 0.3 +
            path.novelty_score * 0.3
        )
        
        # Generate reasoning chain
        path.reasoning_chain = []
        for i, bridge in enumerate(path.bridge_path):
            reasoning_step = f"Step {i+1}: {bridge.source_concept.name} → {bridge.target_concept.name} " \
                           f"via {bridge.bridge_type.value} bridge " \
                           f"(strength: {bridge.bridge_strength:.2f})"
            path.reasoning_chain.append(reasoning_step)

class CrossDomainOntologyBridge:
    """Main orchestrator for cross-domain ontology bridging using 100K embeddings"""
    
    def __init__(self, embeddings_path: str = "/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/embeddings"):
        self.ontology_builder = DomainOntologyBuilder()
        self.bridge_detector = ConceptualBridgeDetector(embeddings_path)
        self.concept_mapper = CrossDomainConceptMapper(embeddings_path)
        self.traversal_engine = BridgeTraversalEngine()
        
        # Domain ontology cache
        self.domain_ontologies = {}
        self.cross_domain_bridges = {}
        self.embeddings_path = embeddings_path
    
    async def perform_cross_domain_analysis(self,
                                          query: str,
                                          context: Dict[str, Any],
                                          papers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive cross-domain ontology analysis"""
        start_time = time.time()
        
        try:
            # Identify relevant domains from query
            relevant_domains = await self._identify_relevant_domains(query, context, papers)
            
            # Build or retrieve domain ontologies
            domain_graphs = {}
            for domain in relevant_domains:
                if domain in self.domain_ontologies:
                    domain_graphs[domain] = self.domain_ontologies[domain]
                else:
                    domain_papers = self._filter_papers_by_domain(papers or [], domain)
                    graph = await self.ontology_builder.build_domain_ontology(domain, domain_papers, context)
                    self.domain_ontologies[domain] = graph
                    domain_graphs[domain] = graph
            
            # Detect conceptual bridges between domains
            all_bridges = []
            for domain1, domain2 in combinations(relevant_domains, 2):
                pair_key = (domain1, domain2)
                if pair_key in self.cross_domain_bridges:
                    bridges = self.cross_domain_bridges[pair_key]
                else:
                    bridges = await self.bridge_detector.detect_conceptual_bridges(
                        domain_graphs[domain1], domain_graphs[domain2], 
                        domain1, domain2, context
                    )
                    self.cross_domain_bridges[pair_key] = bridges
                all_bridges.extend(bridges)
            
            # Map cross-domain concepts and generate insights
            cross_domain_insights = await self.concept_mapper.map_cross_domain_concepts(
                all_bridges, query, context
            )
            
            # Find traversal paths for multi-hop reasoning
            traversal_paths = []
            if len(all_bridges) >= 3:
                # Find paths between different domain concepts
                domain_concepts = {}
                for domain, graph in domain_graphs.items():
                    if graph.nodes():
                        # Get most important concept from each domain
                        concepts = [graph.nodes[node]['concept'] for node in graph.nodes()]
                        most_important = max(concepts, key=lambda c: c.importance)
                        domain_concepts[domain] = most_important
                
                # Find paths between domain concepts
                for domain1, domain2 in combinations(domain_concepts.keys(), 2):
                    paths = await self.traversal_engine.find_traversal_paths(
                        domain_concepts[domain1], domain_concepts[domain2], all_bridges, context
                    )
                    traversal_paths.extend(paths)
            
            # Calculate overall analysis metrics
            processing_time = time.time() - start_time
            
            result = {
                "conclusion": f"Cross-domain analysis identified {len(all_bridges)} conceptual bridges across {len(relevant_domains)} domains",
                "confidence": self._calculate_overall_confidence(all_bridges, cross_domain_insights),
                "evidence": [insight.description for insight in cross_domain_insights[:5]],
                "reasoning_chain": [
                    f"Analyzed {len(relevant_domains)} relevant domains: {', '.join(d.value for d in relevant_domains)}",
                    f"Built ontology graphs with {sum(len(g.nodes()) for g in domain_graphs.values())} total concepts",
                    f"Detected {len(all_bridges)} conceptual bridges across domains",
                    f"Generated {len(cross_domain_insights)} cross-domain insights",
                    f"Found {len(traversal_paths)} multi-hop traversal paths"
                ],
                "processing_time": processing_time,
                "quality_score": self._calculate_quality_score(all_bridges, cross_domain_insights),
                
                # Detailed results
                "relevant_domains": [d.value for d in relevant_domains],
                "domain_ontologies": {d.value: {"concepts": len(g.nodes()), "relationships": len(g.edges())} 
                                    for d, g in domain_graphs.items()},
                "conceptual_bridges": [self._bridge_to_dict(bridge) for bridge in all_bridges[:20]],
                "cross_domain_insights": [self._insight_to_dict(insight) for insight in cross_domain_insights],
                "traversal_paths": [self._path_to_dict(path) for path in traversal_paths[:10]],
                "bridge_statistics": self._calculate_bridge_statistics(all_bridges),
                "domain_connectivity": self._calculate_domain_connectivity(all_bridges, relevant_domains)
            }
            
            logger.info("Cross-domain ontology analysis completed",
                       query=query,
                       domains=len(relevant_domains),
                       bridges=len(all_bridges),
                       insights=len(cross_domain_insights),
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Cross-domain ontology analysis failed", error=str(e))
            return {
                "conclusion": "Cross-domain ontology analysis encountered errors",
                "confidence": 0.0,
                "evidence": [],
                "reasoning_chain": [f"Analysis failed: {str(e)}"],
                "processing_time": time.time() - start_time,
                "quality_score": 0.0,
                "error": str(e)
            }
    
    async def _identify_relevant_domains(self, 
                                       query: str, 
                                       context: Dict[str, Any], 
                                       papers: List[Dict[str, Any]]) -> List[DomainType]:
        """Identify domains relevant to the query"""
        
        query_lower = query.lower()
        relevant_domains = []
        
        # Domain keywords mapping
        domain_keywords = {
            DomainType.SCIENTIFIC: ["science", "physics", "chemistry", "biology", "mathematics", "research", "theory", "experiment"],
            DomainType.TECHNOLOGICAL: ["technology", "engineering", "software", "algorithm", "system", "computer", "digital", "automation"],
            DomainType.MEDICAL: ["medical", "health", "disease", "treatment", "patient", "clinical", "therapeutic", "diagnosis"],
            DomainType.SOCIAL: ["social", "psychology", "behavior", "society", "human", "culture", "community", "interaction"],
            DomainType.ECONOMIC: ["economic", "business", "finance", "market", "cost", "profit", "trade", "investment"],
            DomainType.PHILOSOPHICAL: ["philosophy", "ethics", "logic", "moral", "reasoning", "knowledge", "truth", "meaning"],
            DomainType.ARTISTIC: ["art", "design", "creative", "aesthetic", "music", "visual", "artistic", "beauty"],
            DomainType.HISTORICAL: ["history", "historical", "past", "evolution", "development", "timeline", "ancient", "traditional"],
            DomainType.LINGUISTIC: ["language", "communication", "linguistic", "speech", "text", "semantic", "syntax", "meaning"],
            DomainType.ENVIRONMENTAL: ["environment", "ecology", "climate", "sustainability", "green", "natural", "conservation", "ecosystem"]
        }
        
        # Check query against domain keywords
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_domains.append(domain)
        
        # If no domains found, add default scientific domains
        if not relevant_domains:
            relevant_domains = [DomainType.SCIENTIFIC, DomainType.TECHNOLOGICAL]
        
        # Ensure we have at least 2 domains for cross-domain analysis
        if len(relevant_domains) == 1:
            # Add a complementary domain
            if relevant_domains[0] == DomainType.SCIENTIFIC:
                relevant_domains.append(DomainType.TECHNOLOGICAL)
            else:
                relevant_domains.append(DomainType.SCIENTIFIC)
        
        return relevant_domains[:4]  # Maximum 4 domains for performance
    
    def _filter_papers_by_domain(self, papers: List[Dict[str, Any]], domain: DomainType) -> List[Dict[str, Any]]:
        """Filter papers relevant to a specific domain"""
        domain_keywords = {
            DomainType.SCIENTIFIC: ["physics", "chemistry", "biology", "mathematics", "science"],
            DomainType.TECHNOLOGICAL: ["engineering", "computer", "software", "technology", "algorithm"],
            DomainType.MEDICAL: ["medical", "health", "clinical", "patient", "treatment"],
            # Add more as needed
        }
        
        keywords = domain_keywords.get(domain, [])
        if not keywords:
            return papers[:20]  # Return first 20 if no specific keywords
        
        filtered = []
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            text = f"{title} {abstract}"
            
            if any(keyword in text for keyword in keywords):
                filtered.append(paper)
        
        return filtered[:20] if filtered else papers[:20]
    
    def _calculate_overall_confidence(self, 
                                    bridges: List[ConceptualBridge], 
                                    insights: List[CrossDomainInsight]) -> float:
        """Calculate overall confidence in cross-domain analysis"""
        if not bridges:
            return 0.0
        
        bridge_confidence = statistics.mean([bridge.confidence for bridge in bridges])
        insight_confidence = statistics.mean([insight.validation_confidence for insight in insights]) if insights else 0.5
        
        return (bridge_confidence + insight_confidence) / 2
    
    def _calculate_quality_score(self, 
                               bridges: List[ConceptualBridge], 
                               insights: List[CrossDomainInsight]) -> float:
        """Calculate overall quality score"""
        if not bridges:
            return 0.0
        
        bridge_quality = statistics.mean([bridge.bridge_strength for bridge in bridges])
        insight_quality = statistics.mean([insight.breakthrough_potential for insight in insights]) if insights else 0.5
        
        return (bridge_quality + insight_quality) / 2
    
    def _bridge_to_dict(self, bridge: ConceptualBridge) -> Dict[str, Any]:
        """Convert ConceptualBridge to dictionary"""
        return {
            "id": bridge.id,
            "source_concept": bridge.source_concept.name,
            "source_domain": bridge.source_concept.domain.value,
            "target_concept": bridge.target_concept.name,
            "target_domain": bridge.target_concept.domain.value,
            "bridge_type": bridge.bridge_type.value,
            "similarity_score": bridge.similarity_score,
            "bridge_strength": bridge.bridge_strength,
            "confidence": bridge.confidence,
            "explanation": bridge.explanation,
            "breakthrough_potential": bridge.breakthrough_potential
        }
    
    def _insight_to_dict(self, insight: CrossDomainInsight) -> Dict[str, Any]:
        """Convert CrossDomainInsight to dictionary"""
        return {
            "id": insight.id,
            "title": insight.title,
            "description": insight.description,
            "source_domain": insight.source_domain.value,
            "target_domain": insight.target_domain.value,
            "insight_type": insight.insight_type,
            "novelty_score": insight.novelty_score,
            "breakthrough_potential": insight.breakthrough_potential,
            "validation_confidence": insight.validation_confidence,
            "practical_applications": insight.practical_applications[:3],
            "research_implications": insight.research_implications[:3]
        }
    
    def _path_to_dict(self, path: BridgeTraversalPath) -> Dict[str, Any]:
        """Convert BridgeTraversalPath to dictionary"""
        return {
            "id": path.id,
            "start_concept": path.start_concept.name,
            "end_concept": path.end_concept.name,
            "path_length": path.path_length,
            "average_bridge_strength": path.average_bridge_strength,
            "domains_traversed": [d.value for d in path.domains_traversed],
            "bridge_types_used": [bt.value for bt in path.bridge_types_used],
            "path_coherence": path.path_coherence,
            "novelty_score": path.novelty_score,
            "breakthrough_potential": path.breakthrough_potential,
            "reasoning_chain": path.reasoning_chain
        }
    
    def _calculate_bridge_statistics(self, bridges: List[ConceptualBridge]) -> Dict[str, Any]:
        """Calculate statistics about conceptual bridges"""
        if not bridges:
            return {}
        
        bridge_types = [bridge.bridge_type for bridge in bridges]
        type_counts = Counter(bridge_types)
        
        return {
            "total_bridges": len(bridges),
            "average_similarity": statistics.mean([b.similarity_score for b in bridges]),
            "average_strength": statistics.mean([b.bridge_strength for b in bridges]),
            "bridge_type_distribution": {bt.value: count for bt, count in type_counts.items()},
            "high_quality_bridges": len([b for b in bridges if b.bridge_strength > 0.7]),
            "breakthrough_potential_bridges": len([b for b in bridges if b.breakthrough_potential > 0.6])
        }
    
    def _calculate_domain_connectivity(self, 
                                     bridges: List[ConceptualBridge], 
                                     domains: List[DomainType]) -> Dict[str, Any]:
        """Calculate connectivity metrics between domains"""
        if not bridges:
            return {}
        
        # Build connectivity matrix
        domain_connections = defaultdict(int)
        for bridge in bridges:
            pair = tuple(sorted([bridge.source_concept.domain.value, bridge.target_concept.domain.value]))
            domain_connections[pair] += 1
        
        # Calculate average connectivity
        total_possible_pairs = len(domains) * (len(domains) - 1) // 2
        connected_pairs = len(domain_connections)
        connectivity_ratio = connected_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
        
        return {
            "domain_pairs_connected": connected_pairs,
            "total_possible_pairs": total_possible_pairs,
            "connectivity_ratio": connectivity_ratio,
            "strongest_connections": dict(sorted(domain_connections.items(), 
                                               key=lambda x: x[1], reverse=True)[:5])
        }

# Main interface function for integration with meta-reasoning engine
async def cross_domain_ontology_bridge_integration(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced cross-domain ontology bridge integration using 100K embeddings for breakthrough insights"""
    
    # Get papers from context if available
    papers = context.get('external_papers', [])
    
    # Get embeddings path from context or use default
    embeddings_path = context.get('embeddings_path', "/Users/ryneschultz/Documents/GitHub/PRSM_Storage_Local/03_NWTN_READY/embeddings")
    
    # Create and use enhanced cross-domain ontology bridge with 100K embeddings
    ontology_bridge = CrossDomainOntologyBridge(embeddings_path)
    result = await ontology_bridge.perform_cross_domain_analysis(query, context, papers)
    
    # Add embedding analysis metadata
    result['embedding_analysis'] = {
        'embeddings_loaded': len(ontology_bridge.bridge_detector._embedding_cache) > 0,
        'embedding_count': len(ontology_bridge.bridge_detector._embedding_cache),
        'embedding_enhanced_bridges': len([b for b in result.get('conceptual_bridges', []) 
                                         if 'embedding_similarity' in b]),
        'phase_completion': 'Phase 2.3: Cross-Domain Ontology Bridge Enhanced with 100K Embeddings'
    }
    
    return result

if __name__ == "__main__":
    # Test the cross-domain ontology bridge system
    async def test_cross_domain_ontology_bridge():
        test_query = "machine learning optimization techniques for drug discovery"
        test_context = {
            "domain": "cross_domain_research",
            "breakthrough_mode": "creative"
        }
        
        result = await cross_domain_ontology_bridge_integration(test_query, test_context)
        
        print("Cross-Domain Ontology Bridge Test Results:")
        print("=" * 60)
        print(f"Query: {test_query}")
        print(f"Conclusion: {result['conclusion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Quality Score: {result['quality_score']:.2f}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"\nRelevant Domains: {', '.join(result['relevant_domains'])}")
        print(f"\nCross-Domain Insights:")
        for i, insight in enumerate(result.get('cross_domain_insights', [])[:3], 1):
            print(f"{i}. {insight['title']}")
            print(f"   {insight['description']}")
            print(f"   Breakthrough Potential: {insight['breakthrough_potential']:.2f}")
        print(f"\nBridge Statistics:")
        stats = result.get('bridge_statistics', {})
        print(f"• Total Bridges: {stats.get('total_bridges', 0)}")
        print(f"• Average Strength: {stats.get('average_strength', 0):.2f}")
        print(f"• High Quality Bridges: {stats.get('high_quality_bridges', 0)}")
    
    asyncio.run(test_cross_domain_ontology_bridge())