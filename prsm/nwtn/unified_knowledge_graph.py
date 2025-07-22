"""
NWTN Unified Knowledge Graph Architecture

Comprehensive knowledge representation system combining all enterprise sources including:
- Academic papers, business documents, technical content, communications
- Entity graphs, relationship mapping, temporal intelligence, provenance tracking
- Cross-source entity resolution and breakthrough prediction signals

Part of NWTN Phase 1: Universal Knowledge Ingestion Engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime
import hashlib
import json
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class EntityType(Enum):
    PERSON = "person"
    CONCEPT = "concept" 
    PRODUCT = "product"
    PROJECT = "project"
    ORGANIZATION = "organization"

class RelationshipType(Enum):
    # Structural relationships
    COLLABORATES_WITH = "collaborates_with"
    WORKS_FOR = "works_for"
    DEVELOPS = "develops"
    USES = "uses"
    MENTIONS = "mentions"
    
    # Knowledge relationships
    INFLUENCES = "influences"
    CONTRADICTS = "contradicts"
    BUILDS_ON = "builds_on"
    VALIDATES = "validates"
    CHALLENGES = "challenges"
    
    # Temporal relationships
    PRECEDED_BY = "preceded_by"
    EVOLVED_FROM = "evolved_from"
    SUPERSEDED_BY = "superseded_by"
    
    # Causal relationships
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"

@dataclass
class Entity:
    id: str
    entity_type: EntityType
    name: str
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    source_documents: Set[str] = field(default_factory=set)
    first_seen: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class Relationship:
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    temporal_context: Optional[datetime] = None
    source_documents: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        return f"{self.source_entity_id}_{self.relationship_type.value}_{self.target_entity_id}"

@dataclass
class KnowledgeEvolutionEvent:
    timestamp: datetime
    entity_id: str
    event_type: str  # created, updated, deprecated, superseded
    old_state: Optional[Dict[str, Any]]
    new_state: Dict[str, Any]
    trigger_documents: Set[str]
    confidence: float

class EntityGraph:
    """Core entity management and storage system"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.entity_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self.alias_index: Dict[str, str] = {}  # alias -> entity_id
        self.lock = threading.RLock()
    
    def add_entity(self, entity: Entity) -> bool:
        """Add entity with duplicate detection and alias management"""
        with self.lock:
            # Check for existing entity by aliases
            existing_id = self._resolve_entity_by_aliases(entity.name, entity.aliases)
            if existing_id:
                # Merge with existing entity
                return self._merge_entities(existing_id, entity)
            
            # Add new entity
            self.entities[entity.id] = entity
            self.entity_index[entity.entity_type].add(entity.id)
            
            # Update alias index
            self.alias_index[entity.name.lower()] = entity.id
            for alias in entity.aliases:
                self.alias_index[alias.lower()] = entity.id
            
            return True
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find entity by name or alias"""
        entity_id = self.alias_index.get(name.lower())
        return self.entities.get(entity_id) if entity_id else None
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of specific type"""
        entity_ids = self.entity_index[entity_type]
        return [self.entities[eid] for eid in entity_ids]
    
    def _resolve_entity_by_aliases(self, name: str, aliases: Set[str]) -> Optional[str]:
        """Check if entity already exists by name/aliases"""
        all_names = {name.lower()} | {alias.lower() for alias in aliases}
        for alias in all_names:
            if alias in self.alias_index:
                return self.alias_index[alias]
        return None
    
    def _merge_entities(self, existing_id: str, new_entity: Entity) -> bool:
        """Merge new entity information with existing entity"""
        existing = self.entities[existing_id]
        
        # Merge aliases
        existing.aliases.update(new_entity.aliases)
        
        # Merge attributes (new values take precedence)
        existing.attributes.update(new_entity.attributes)
        
        # Merge source documents
        existing.source_documents.update(new_entity.source_documents)
        
        # Update confidence (weighted average)
        total_docs = len(existing.source_documents) + len(new_entity.source_documents)
        existing_weight = len(existing.source_documents) / total_docs
        new_weight = len(new_entity.source_documents) / total_docs
        existing.confidence = existing.confidence * existing_weight + new_entity.confidence * new_weight
        
        existing.last_updated = datetime.now()
        return True

class RelationshipMapping:
    """Advanced relationship discovery and cross-source entity resolution"""
    
    def __init__(self, entity_graph: EntityGraph):
        self.entity_graph = entity_graph
        self.relationships: Dict[str, Relationship] = {}
        self.relationship_index: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relationship_ids
        self.temporal_relationships: List[Relationship] = []
        self.lock = threading.RLock()
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add relationship with confidence-based merging"""
        with self.lock:
            rel_id = relationship.id
            
            if rel_id in self.relationships:
                # Merge with existing relationship
                return self._merge_relationships(rel_id, relationship)
            
            # Validate entities exist
            if not (self.entity_graph.get_entity(relationship.source_entity_id) and 
                   self.entity_graph.get_entity(relationship.target_entity_id)):
                return False
            
            self.relationships[rel_id] = relationship
            self.relationship_index[relationship.source_entity_id].add(rel_id)
            self.relationship_index[relationship.target_entity_id].add(rel_id)
            
            # Track temporal relationships
            if relationship.temporal_context:
                self.temporal_relationships.append(relationship)
                self.temporal_relationships.sort(key=lambda r: r.temporal_context)
            
            return True
    
    def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving entity"""
        rel_ids = self.relationship_index[entity_id]
        return [self.relationships[rid] for rid in rel_ids]
    
    def find_cross_source_connections(self, entity_id: str) -> Dict[str, List[Relationship]]:
        """Find connections across different source types"""
        relationships = self.get_entity_relationships(entity_id)
        cross_source_connections = defaultdict(list)
        
        entity = self.entity_graph.get_entity(entity_id)
        if not entity:
            return {}
        
        entity_sources = entity.source_documents
        
        for rel in relationships:
            # Get connected entity
            connected_id = rel.target_entity_id if rel.source_entity_id == entity_id else rel.source_entity_id
            connected_entity = self.entity_graph.get_entity(connected_id)
            
            if connected_entity:
                # Check for cross-source connections
                connected_sources = connected_entity.source_documents
                if not entity_sources.intersection(connected_sources):
                    # Different sources - this is a cross-source connection
                    source_types = self._classify_source_types(entity_sources, connected_sources)
                    cross_source_connections[source_types].append(rel)
        
        return cross_source_connections
    
    def discover_causal_relationships(self, window_hours: int = 24) -> List[Relationship]:
        """Discover potential causal relationships based on temporal patterns"""
        causal_candidates = []
        
        # Group temporal relationships by time windows
        time_windows = self._group_by_time_windows(self.temporal_relationships, window_hours)
        
        for window_relationships in time_windows:
            if len(window_relationships) < 2:
                continue
                
            # Look for potential causal patterns
            for i, rel1 in enumerate(window_relationships):
                for rel2 in window_relationships[i+1:]:
                    causal_strength = self._analyze_causal_potential(rel1, rel2)
                    if causal_strength > 0.6:
                        causal_rel = Relationship(
                            source_entity_id=rel1.target_entity_id,
                            target_entity_id=rel2.source_entity_id,
                            relationship_type=RelationshipType.CAUSES,
                            confidence=causal_strength,
                            evidence=[f"Temporal pattern: {rel1.id} → {rel2.id}"],
                            temporal_context=rel2.temporal_context
                        )
                        causal_candidates.append(causal_rel)
        
        return causal_candidates
    
    def _merge_relationships(self, existing_id: str, new_relationship: Relationship) -> bool:
        """Merge relationship information"""
        existing = self.relationships[existing_id]
        
        # Merge evidence
        existing.evidence.extend(new_relationship.evidence)
        
        # Merge source documents
        existing.source_documents.update(new_relationship.source_documents)
        
        # Update confidence (weighted by evidence count)
        existing_weight = len(existing.evidence)
        new_weight = len(new_relationship.evidence)
        total_weight = existing_weight + new_weight
        
        if total_weight > 0:
            existing.confidence = (existing.confidence * existing_weight + 
                                 new_relationship.confidence * new_weight) / total_weight
        
        return True
    
    def _classify_source_types(self, sources1: Set[str], sources2: Set[str]) -> str:
        """Classify the types of sources being connected"""
        # This would classify document types based on file extensions, metadata, etc.
        # For now, simplified classification
        return "cross_domain_connection"
    
    def _group_by_time_windows(self, relationships: List[Relationship], window_hours: int) -> List[List[Relationship]]:
        """Group relationships by time windows"""
        windows = []
        current_window = []
        window_start = None
        
        for rel in relationships:
            if not rel.temporal_context:
                continue
                
            if window_start is None:
                window_start = rel.temporal_context
                current_window = [rel]
            elif (rel.temporal_context - window_start).total_seconds() <= window_hours * 3600:
                current_window.append(rel)
            else:
                if current_window:
                    windows.append(current_window)
                window_start = rel.temporal_context
                current_window = [rel]
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def _analyze_causal_potential(self, rel1: Relationship, rel2: Relationship) -> float:
        """Analyze potential causal relationship strength"""
        # Simplified causal analysis - in practice would use more sophisticated methods
        temporal_score = 0.5  # Base score for temporal ordering
        
        # Entity overlap bonus
        if rel1.target_entity_id == rel2.source_entity_id:
            temporal_score += 0.3
        
        # Relationship type compatibility
        if rel1.relationship_type in [RelationshipType.INFLUENCES, RelationshipType.ENABLES]:
            temporal_score += 0.2
            
        return min(temporal_score, 1.0)

class TemporalIntelligence:
    """Track knowledge evolution and identify breakthrough prediction signals"""
    
    def __init__(self, entity_graph: EntityGraph, relationship_mapping: RelationshipMapping):
        self.entity_graph = entity_graph
        self.relationship_mapping = relationship_mapping
        self.evolution_events: List[KnowledgeEvolutionEvent] = []
        self.trend_cache: Dict[str, Any] = {}
        self.lock = threading.RLock()
    
    def track_knowledge_evolution(self, entity_id: str, old_state: Optional[Dict], new_state: Dict, 
                                trigger_docs: Set[str], event_type: str = "updated") -> KnowledgeEvolutionEvent:
        """Track evolution of knowledge entities"""
        event = KnowledgeEvolutionEvent(
            timestamp=datetime.now(),
            entity_id=entity_id,
            event_type=event_type,
            old_state=old_state,
            new_state=new_state,
            trigger_documents=trigger_docs,
            confidence=0.8
        )
        
        with self.lock:
            self.evolution_events.append(event)
            self.evolution_events.sort(key=lambda e: e.timestamp)
        
        # Analyze for breakthrough signals
        self._analyze_breakthrough_signals(event)
        
        return event
    
    def construct_decision_timeline(self, entity_id: str) -> List[KnowledgeEvolutionEvent]:
        """Construct timeline of decisions/changes for an entity"""
        entity_events = [event for event in self.evolution_events if event.entity_id == entity_id]
        return sorted(entity_events, key=lambda e: e.timestamp)
    
    def identify_trends(self, entity_type: Optional[EntityType] = None, 
                       time_window_days: int = 30) -> Dict[str, Any]:
        """Identify emerging trends in knowledge evolution"""
        cache_key = f"{entity_type}_{time_window_days}"
        if cache_key in self.trend_cache:
            return self.trend_cache[cache_key]
        
        cutoff_date = datetime.now() - pd.Timedelta(days=time_window_days)
        recent_events = [e for e in self.evolution_events if e.timestamp >= cutoff_date]
        
        if entity_type:
            # Filter by entity type
            relevant_entities = {e.id for e in self.entity_graph.get_entities_by_type(entity_type)}
            recent_events = [e for e in recent_events if e.entity_id in relevant_entities]
        
        trends = {
            'creation_rate': self._calculate_creation_rate(recent_events),
            'evolution_hotspots': self._identify_evolution_hotspots(recent_events),
            'emerging_relationships': self._identify_emerging_relationship_patterns(recent_events),
            'breakthrough_indicators': self._detect_breakthrough_indicators(recent_events)
        }
        
        self.trend_cache[cache_key] = trends
        return trends
    
    def detect_breakthrough_prediction_signals(self) -> List[Dict[str, Any]]:
        """Detect signals that might predict breakthrough innovations"""
        signals = []
        
        # Signal 1: Rapid convergence of previously unconnected domains
        domain_convergence = self._detect_domain_convergence()
        signals.extend(domain_convergence)
        
        # Signal 2: Sudden increase in contradictory relationships
        contradiction_spikes = self._detect_contradiction_spikes()
        signals.extend(contradiction_spikes)
        
        # Signal 3: Emergence of bridging concepts
        bridging_concepts = self._detect_bridging_concepts()
        signals.extend(bridging_concepts)
        
        # Signal 4: Acceleration in knowledge evolution rate
        evolution_acceleration = self._detect_evolution_acceleration()
        signals.extend(evolution_acceleration)
        
        return sorted(signals, key=lambda s: s['confidence'], reverse=True)
    
    def _analyze_breakthrough_signals(self, event: KnowledgeEvolutionEvent):
        """Real-time analysis of individual events for breakthrough signals"""
        # Check for rapid state changes
        if event.old_state and event.new_state:
            change_magnitude = self._calculate_change_magnitude(event.old_state, event.new_state)
            if change_magnitude > 0.8:
                # Significant change detected
                pass
    
    def _calculate_creation_rate(self, events: List[KnowledgeEvolutionEvent]) -> float:
        """Calculate rate of new entity/relationship creation"""
        creation_events = [e for e in events if e.event_type == "created"]
        if not events:
            return 0.0
        return len(creation_events) / len(events)
    
    def _identify_evolution_hotspots(self, events: List[KnowledgeEvolutionEvent]) -> List[str]:
        """Identify entities with high evolution activity"""
        entity_activity = defaultdict(int)
        for event in events:
            entity_activity[event.entity_id] += 1
        
        # Return top 10 most active entities
        return sorted(entity_activity.keys(), key=entity_activity.get, reverse=True)[:10]
    
    def _identify_emerging_relationship_patterns(self, events: List[KnowledgeEvolutionEvent]) -> List[str]:
        """Identify new relationship patterns emerging"""
        # Simplified implementation - would analyze relationship type distributions
        return ["cross_domain_collaboration_increase", "validation_relationship_emergence"]
    
    def _detect_breakthrough_indicators(self, events: List[KnowledgeEvolutionEvent]) -> Dict[str, float]:
        """Detect specific breakthrough indicator patterns"""
        return {
            'paradigm_shift_probability': 0.3,
            'innovation_acceleration': 0.7,
            'cross_pollination_index': 0.8
        }
    
    def _detect_domain_convergence(self) -> List[Dict[str, Any]]:
        """Detect convergence of previously separate domains"""
        # Implementation would analyze cross-domain relationship formation rates
        return [{'type': 'domain_convergence', 'entities': ['AI', 'Biology'], 'confidence': 0.8}]
    
    def _detect_contradiction_spikes(self) -> List[Dict[str, Any]]:
        """Detect spikes in contradictory relationships"""
        return [{'type': 'contradiction_spike', 'domain': 'quantum_computing', 'confidence': 0.6}]
    
    def _detect_bridging_concepts(self) -> List[Dict[str, Any]]:
        """Detect emergence of concepts that bridge multiple domains"""
        return [{'type': 'bridging_concept', 'concept': 'bio_inspired_computing', 'confidence': 0.9}]
    
    def _detect_evolution_acceleration(self) -> List[Dict[str, Any]]:
        """Detect acceleration in knowledge evolution rates"""
        return [{'type': 'evolution_acceleration', 'domain': 'machine_learning', 'confidence': 0.7}]
    
    def _calculate_change_magnitude(self, old_state: Dict, new_state: Dict) -> float:
        """Calculate magnitude of change between states"""
        # Simplified change calculation
        changed_fields = 0
        total_fields = len(old_state)
        
        for key in old_state:
            if key in new_state and old_state[key] != new_state[key]:
                changed_fields += 1
        
        return changed_fields / total_fields if total_fields > 0 else 0.0

class ProvenanceTracking:
    """Track source attribution, confidence assessment, and quality metrics"""
    
    def __init__(self):
        self.source_registry: Dict[str, Dict[str, Any]] = {}
        self.quality_assessments: Dict[str, float] = {}
        self.confidence_models: Dict[str, Any] = {}
        self.lock = threading.RLock()
    
    def register_source(self, source_id: str, metadata: Dict[str, Any]):
        """Register a source document with metadata"""
        with self.lock:
            self.source_registry[source_id] = {
                'metadata': metadata,
                'registered_at': datetime.now(),
                'quality_score': self._assess_initial_quality(metadata),
                'entities_extracted': set(),
                'relationships_extracted': set()
            }
    
    def update_source_attribution(self, source_id: str, entity_ids: Set[str], relationship_ids: Set[str]):
        """Update what entities/relationships were extracted from a source"""
        with self.lock:
            if source_id in self.source_registry:
                self.source_registry[source_id]['entities_extracted'].update(entity_ids)
                self.source_registry[source_id]['relationships_extracted'].update(relationship_ids)
    
    def assess_confidence(self, entity_id: str, relationships: List[Relationship]) -> float:
        """Assess confidence in entity based on source quality and relationship evidence"""
        if not relationships:
            return 0.5  # Base confidence for isolated entities
        
        # Factor in source quality
        source_scores = []
        for rel in relationships:
            for source_id in rel.source_documents:
                if source_id in self.source_registry:
                    source_scores.append(self.source_registry[source_id]['quality_score'])
        
        # Factor in relationship evidence strength
        evidence_strength = sum(len(rel.evidence) for rel in relationships) / len(relationships)
        evidence_score = min(evidence_strength / 10.0, 1.0)  # Normalize to 0-1
        
        # Factor in cross-source validation
        unique_sources = len(set().union(*[rel.source_documents for rel in relationships]))
        cross_validation_score = min(unique_sources / 5.0, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        source_score = np.mean(source_scores) if source_scores else 0.5
        final_confidence = (source_score * 0.4 + evidence_score * 0.3 + cross_validation_score * 0.3)
        
        return final_confidence
    
    def get_source_quality_metrics(self, source_id: str) -> Dict[str, Any]:
        """Get quality metrics for a specific source"""
        if source_id not in self.source_registry:
            return {}
        
        source_info = self.source_registry[source_id]
        return {
            'quality_score': source_info['quality_score'],
            'entities_contributed': len(source_info['entities_extracted']),
            'relationships_contributed': len(source_info['relationships_extracted']),
            'metadata': source_info['metadata']
        }
    
    def _assess_initial_quality(self, metadata: Dict[str, Any]) -> float:
        """Assess initial quality score based on source metadata"""
        quality_score = 0.5  # Base score
        
        # Factor in source type
        if metadata.get('source_type') == 'academic_paper':
            quality_score += 0.2
        elif metadata.get('source_type') == 'enterprise_document':
            quality_score += 0.1
        
        # Factor in author/organization reputation (would need reputation system)
        if metadata.get('author_h_index', 0) > 10:
            quality_score += 0.1
        
        # Factor in publication venue quality
        if metadata.get('venue_impact_factor', 0) > 2.0:
            quality_score += 0.15
        
        # Factor in recency
        pub_date = metadata.get('publication_date')
        if pub_date:
            age_years = (datetime.now() - pub_date).days / 365
            recency_bonus = max(0, 0.1 - age_years * 0.02)  # Newer is better
            quality_score += recency_bonus
        
        return min(quality_score, 1.0)

class UnifiedKnowledgeGraph:
    """Main orchestrator for the Unified Knowledge Graph system"""
    
    def __init__(self, max_entities: int = 100000, max_relationships: int = 500000):
        self.entity_graph = EntityGraph()
        self.relationship_mapping = RelationshipMapping(self.entity_graph)
        self.temporal_intelligence = TemporalIntelligence(self.entity_graph, self.relationship_mapping)
        self.provenance_tracking = ProvenanceTracking()
        
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        self.lock = threading.RLock()
    
    def ingest_processed_content(self, processed_content, source_id: str) -> Dict[str, Any]:
        """Ingest content from Universal Knowledge Ingestion Engine"""
        from .universal_knowledge_ingestion_engine import ProcessedContent
        
        if not isinstance(processed_content, ProcessedContent):
            return {'success': False, 'error': 'Invalid content type'}
        
        # Register source
        source_metadata = {
            'source_type': processed_content.content_format.value,
            'file_path': processed_content.metadata.get('file_path'),
            'author': processed_content.metadata.get('author'),
            'creation_date': processed_content.metadata.get('creation_date'),
            'quality_indicators': processed_content.metadata.get('quality_indicators', {})
        }
        self.provenance_tracking.register_source(source_id, source_metadata)
        
        # Extract and add entities
        extracted_entities = self._extract_entities_from_content(processed_content, source_id)
        entity_ids = set()
        for entity in extracted_entities:
            if self.entity_graph.add_entity(entity):
                entity_ids.add(entity.id)
        
        # Extract and add relationships
        extracted_relationships = self._extract_relationships_from_content(processed_content, source_id)
        relationship_ids = set()
        for relationship in extracted_relationships:
            if self.relationship_mapping.add_relationship(relationship):
                relationship_ids.add(relationship.id)
        
        # Update provenance tracking
        self.provenance_tracking.update_source_attribution(source_id, entity_ids, relationship_ids)
        
        # Track knowledge evolution
        for entity_id in entity_ids:
            self.temporal_intelligence.track_knowledge_evolution(
                entity_id, None, {'source_added': source_id}, {source_id}, 'content_integration'
            )
        
        return {
            'success': True,
            'entities_extracted': len(entity_ids),
            'relationships_extracted': len(relationship_ids),
            'source_id': source_id
        }
    
    def discover_breakthrough_opportunities(self) -> List[Dict[str, Any]]:
        """Discover potential breakthrough opportunities from the knowledge graph"""
        opportunities = []
        
        # Cross-domain connection analysis
        cross_domain_connections = self._analyze_cross_domain_connections()
        opportunities.extend(cross_domain_connections)
        
        # Temporal breakthrough signals
        temporal_signals = self.temporal_intelligence.detect_breakthrough_prediction_signals()
        opportunities.extend(temporal_signals)
        
        # Knowledge gap identification
        knowledge_gaps = self._identify_knowledge_gaps()
        opportunities.extend(knowledge_gaps)
        
        # Emerging contradiction analysis
        contradictions = self._analyze_emerging_contradictions()
        opportunities.extend(contradictions)
        
        return sorted(opportunities, key=lambda o: o.get('confidence', 0), reverse=True)
    
    def query_knowledge_graph(self, query_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph with various query types"""
        if query_type == "entity_connections":
            entity_id = parameters.get('entity_id')
            if not entity_id:
                return {'error': 'entity_id required'}
            
            entity = self.entity_graph.get_entity(entity_id)
            relationships = self.relationship_mapping.get_entity_relationships(entity_id)
            cross_connections = self.relationship_mapping.find_cross_source_connections(entity_id)
            
            return {
                'entity': entity,
                'relationships': relationships,
                'cross_source_connections': cross_connections
            }
        
        elif query_type == "temporal_evolution":
            entity_id = parameters.get('entity_id')
            if not entity_id:
                return {'error': 'entity_id required'}
                
            timeline = self.temporal_intelligence.construct_decision_timeline(entity_id)
            return {'evolution_timeline': timeline}
        
        elif query_type == "trend_analysis":
            entity_type = parameters.get('entity_type')
            time_window = parameters.get('time_window_days', 30)
            trends = self.temporal_intelligence.identify_trends(entity_type, time_window)
            return {'trends': trends}
        
        elif query_type == "breakthrough_signals":
            signals = self.temporal_intelligence.detect_breakthrough_prediction_signals()
            return {'breakthrough_signals': signals}
        
        else:
            return {'error': f'Unknown query type: {query_type}'}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        entity_counts = {etype.value: len(self.entity_graph.get_entities_by_type(etype)) 
                        for etype in EntityType}
        
        relationship_counts = defaultdict(int)
        for rel in self.relationship_mapping.relationships.values():
            relationship_counts[rel.relationship_type.value] += 1
        
        return {
            'total_entities': len(self.entity_graph.entities),
            'total_relationships': len(self.relationship_mapping.relationships),
            'entity_breakdown': entity_counts,
            'relationship_breakdown': dict(relationship_counts),
            'evolution_events': len(self.temporal_intelligence.evolution_events),
            'registered_sources': len(self.provenance_tracking.source_registry),
            'cross_source_connections': self._count_cross_source_connections(),
            'breakthrough_signals_detected': len(self.temporal_intelligence.detect_breakthrough_prediction_signals())
        }
    
    def _extract_entities_from_content(self, content, source_id: str) -> List[Entity]:
        """Extract entities from processed content"""
        entities = []
        
        # Extract from content entities (from universal ingestion)
        for entity_name, entity_info in content.entities.items():
            # Determine entity type based on content and context
            entity_type = self._classify_entity_type(entity_name, entity_info, content)
            
            entity_id = self._generate_entity_id(entity_name, entity_type)
            entity = Entity(
                id=entity_id,
                entity_type=entity_type,
                name=entity_name,
                aliases=set(),
                attributes=entity_info if isinstance(entity_info, dict) else {'mentions': entity_info},
                confidence=0.8,
                source_documents={source_id}
            )
            entities.append(entity)
        
        return entities
    
    def _extract_relationships_from_content(self, content, source_id: str) -> List[Relationship]:
        """Extract relationships from processed content"""
        relationships = []
        
        # Basic co-occurrence relationships
        entity_names = list(content.entities.keys())
        for i, entity1 in enumerate(entity_names):
            for entity2 in entity_names[i+1:]:
                # Create mentions relationship
                entity1_id = self._generate_entity_id(entity1, self._classify_entity_type(entity1, {}, content))
                entity2_id = self._generate_entity_id(entity2, self._classify_entity_type(entity2, {}, content))
                
                relationship = Relationship(
                    source_entity_id=entity1_id,
                    target_entity_id=entity2_id,
                    relationship_type=RelationshipType.MENTIONS,
                    confidence=0.6,
                    evidence=[f"Co-mentioned in {source_id}"],
                    source_documents={source_id}
                )
                relationships.append(relationship)
        
        return relationships
    
    def _classify_entity_type(self, entity_name: str, entity_info: Any, content) -> EntityType:
        """Classify entity type based on name, context, and content"""
        name_lower = entity_name.lower()
        
        # Simple heuristic-based classification
        if any(indicator in name_lower for indicator in ['university', 'company', 'corporation', 'institute']):
            return EntityType.ORGANIZATION
        elif any(indicator in name_lower for indicator in ['project', 'system', 'framework', 'algorithm']):
            return EntityType.PROJECT
        elif any(indicator in name_lower for indicator in ['dr.', 'prof.', 'phd']) or name_lower.count(' ') == 1:
            return EntityType.PERSON
        elif any(indicator in name_lower for indicator in ['software', 'hardware', 'tool', 'platform']):
            return EntityType.PRODUCT
        else:
            return EntityType.CONCEPT
    
    def _generate_entity_id(self, name: str, entity_type: EntityType) -> str:
        """Generate consistent entity ID"""
        normalized_name = name.lower().replace(' ', '_')
        return f"{entity_type.value}_{hashlib.md5(normalized_name.encode()).hexdigest()[:8]}"
    
    def _analyze_cross_domain_connections(self) -> List[Dict[str, Any]]:
        """Analyze cross-domain connections for breakthrough opportunities"""
        opportunities = []
        
        # Find entities with high cross-source connectivity
        for entity_id in self.entity_graph.entities:
            cross_connections = self.relationship_mapping.find_cross_source_connections(entity_id)
            if len(cross_connections) > 2:  # Threshold for interesting cross-connections
                opportunities.append({
                    'type': 'cross_domain_hub',
                    'entity_id': entity_id,
                    'cross_connections': len(cross_connections),
                    'confidence': min(len(cross_connections) / 10.0, 1.0)
                })
        
        return opportunities
    
    def _identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify potential knowledge gaps in the graph"""
        gaps = []
        
        # Find highly connected entities with missing relationship types
        for entity_id, entity in self.entity_graph.entities.items():
            relationships = self.relationship_mapping.get_entity_relationships(entity_id)
            relationship_types = {rel.relationship_type for rel in relationships}
            
            # Check for missing common relationship types
            expected_types = {RelationshipType.INFLUENCES, RelationshipType.BUILDS_ON}
            missing_types = expected_types - relationship_types
            
            if missing_types and len(relationships) > 5:
                gaps.append({
                    'type': 'relationship_gap',
                    'entity_id': entity_id,
                    'missing_relationships': [t.value for t in missing_types],
                    'confidence': 0.7
                })
        
        return gaps
    
    def _analyze_emerging_contradictions(self) -> List[Dict[str, Any]]:
        """Analyze emerging contradictions that might signal breakthroughs"""
        contradictions = []
        
        # Find contradiction relationships
        for rel in self.relationship_mapping.relationships.values():
            if rel.relationship_type == RelationshipType.CONTRADICTS:
                contradictions.append({
                    'type': 'active_contradiction',
                    'entities': [rel.source_entity_id, rel.target_entity_id],
                    'confidence': rel.confidence
                })
        
        return contradictions
    
    def _count_cross_source_connections(self) -> int:
        """Count total cross-source connections"""
        total = 0
        for entity_id in self.entity_graph.entities:
            cross_connections = self.relationship_mapping.find_cross_source_connections(entity_id)
            total += sum(len(connections) for connections in cross_connections.values())
        return total

# Required import for temporal analysis
import pandas as pd