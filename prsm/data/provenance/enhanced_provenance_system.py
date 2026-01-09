#!/usr/bin/env python3
"""
Enhanced Provenance System

Provides comprehensive tracking and verification of data lineage,
reasoning chains, and computational provenance throughout the PRSM ecosystem.

Core Functions:
- Reasoning chain provenance tracking
- Data source verification and lineage
- Computational reproducibility assurance
- Trust and credibility scoring
"""

import structlog
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4
import hashlib
import json

logger = structlog.get_logger(__name__)


class ProvenanceType(Enum):
    """Types of provenance records"""
    DATA_SOURCE = "data_source"
    REASONING_STEP = "reasoning_step"
    MODEL_PREDICTION = "model_prediction"
    COMPUTATION = "computation"
    TRANSFORMATION = "transformation"
    CITATION = "citation"
    VALIDATION = "validation"


class TrustLevel(Enum):
    """Trust levels for provenance records"""
    VERIFIED = "verified"          # Fully verified and trusted
    TRUSTED = "trusted"            # High confidence, peer reviewed
    CREDIBLE = "credible"          # Good confidence, some verification
    PROVISIONAL = "provisional"    # Limited verification
    UNCERTAIN = "uncertain"        # Low confidence, needs verification


@dataclass
class ProvenanceRecord:
    """Individual provenance record"""
    record_id: str
    provenance_type: ProvenanceType
    timestamp: datetime
    source_entity: str  # What created this record
    target_entity: str  # What this record describes
    operation: str      # What operation was performed
    inputs: List[str]   # Input record IDs
    outputs: List[str]  # Output record IDs
    metadata: Dict[str, Any]
    trust_level: TrustLevel = TrustLevel.CREDIBLE
    verification_status: bool = False
    content_hash: str = ""


@dataclass
class ReasoningProvenance:
    """Provenance chain for reasoning operations"""
    reasoning_id: str
    query: str
    reasoning_steps: List[ProvenanceRecord]
    data_sources: List[ProvenanceRecord]
    model_calls: List[ProvenanceRecord]
    final_conclusion: str
    confidence_score: float
    total_steps: int
    verification_chain: List[str] = field(default_factory=list)


class EnhancedProvenanceSystem:
    """
    Enhanced Provenance System for PRSM
    
    Tracks and verifies the complete lineage of reasoning, data,
    and computational operations within the PRSM ecosystem.
    """
    
    def __init__(self):
        """Initialize enhanced provenance system"""
        self.provenance_records: Dict[str, ProvenanceRecord] = {}
        self.reasoning_chains: Dict[str, ReasoningProvenance] = {}
        self.trust_scores: Dict[str, float] = {}
        self.verification_cache: Dict[str, bool] = {}
        
        logger.info("EnhancedProvenanceSystem initialized")
    
    def create_provenance_record(self,
                               provenance_type: ProvenanceType,
                               source_entity: str,
                               target_entity: str,
                               operation: str,
                               inputs: Optional[List[str]] = None,
                               outputs: Optional[List[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None,
                               trust_level: TrustLevel = TrustLevel.CREDIBLE) -> ProvenanceRecord:
        """Create new provenance record"""
        try:
            record_id = f"prov_{uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            # Create content hash for integrity
            content_data = {
                'type': provenance_type.value,
                'source': source_entity,
                'target': target_entity,
                'operation': operation,
                'inputs': inputs or [],
                'outputs': outputs or [],
                'metadata': metadata or {}
            }
            content_hash = hashlib.sha256(json.dumps(content_data, sort_keys=True).encode()).hexdigest()[:16]
            
            record = ProvenanceRecord(
                record_id=record_id,
                provenance_type=provenance_type,
                timestamp=datetime.now(timezone.utc),
                source_entity=source_entity,
                target_entity=target_entity,
                operation=operation,
                inputs=inputs or [],
                outputs=outputs or [],
                metadata=metadata or {},
                trust_level=trust_level,
                content_hash=content_hash
            )
            
            self.provenance_records[record_id] = record
            
            logger.debug("Provenance record created",
                        record_id=record_id,
                        type=provenance_type.value,
                        operation=operation)
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to create provenance record: {e}")
            raise
    
    def start_reasoning_chain(self, query: str) -> str:
        """Start tracking a reasoning chain"""
        reasoning_id = f"reasoning_{uuid4().hex[:8]}"
        
        # Create initial data source record
        initial_record = self.create_provenance_record(
            provenance_type=ProvenanceType.DATA_SOURCE,
            source_entity="user_query",
            target_entity=reasoning_id,
            operation="query_initiation",
            metadata={
                'query_text': query,
                'query_length': len(query),
                'reasoning_initiated': True
            }
        )
        
        reasoning_chain = ReasoningProvenance(
            reasoning_id=reasoning_id,
            query=query,
            reasoning_steps=[initial_record],
            data_sources=[initial_record],
            model_calls=[],
            final_conclusion="",
            confidence_score=0.0,
            total_steps=1
        )
        
        self.reasoning_chains[reasoning_id] = reasoning_chain
        
        logger.info("Reasoning chain started", reasoning_id=reasoning_id, query=query[:100])
        
        return reasoning_id
    
    def add_reasoning_step(self,
                          reasoning_id: str,
                          step_type: str,
                          operation: str,
                          inputs: List[str],
                          outputs: List[str],
                          metadata: Optional[Dict[str, Any]] = None) -> ProvenanceRecord:
        """Add step to reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Create reasoning step record
        step_record = self.create_provenance_record(
            provenance_type=ProvenanceType.REASONING_STEP,
            source_entity=step_type,
            target_entity=reasoning_id,
            operation=operation,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata or {}
        )
        
        reasoning_chain.reasoning_steps.append(step_record)
        reasoning_chain.total_steps = len(reasoning_chain.reasoning_steps)
        
        logger.debug("Reasoning step added",
                    reasoning_id=reasoning_id,
                    step_type=step_type,
                    operation=operation)
        
        return step_record
    
    def add_data_source(self,
                       reasoning_id: str,
                       source_type: str,
                       source_id: str,
                       metadata: Optional[Dict[str, Any]] = None) -> ProvenanceRecord:
        """Add data source to reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Create data source record
        source_record = self.create_provenance_record(
            provenance_type=ProvenanceType.DATA_SOURCE,
            source_entity=source_type,
            target_entity=reasoning_id,
            operation="data_retrieval",
            outputs=[source_id],
            metadata=metadata or {}
        )
        
        reasoning_chain.data_sources.append(source_record)
        
        logger.debug("Data source added",
                    reasoning_id=reasoning_id,
                    source_type=source_type,
                    source_id=source_id)
        
        return source_record
    
    def add_model_call(self,
                      reasoning_id: str,
                      model_id: str,
                      input_text: str,
                      output_text: str,
                      metadata: Optional[Dict[str, Any]] = None) -> ProvenanceRecord:
        """Add model call to reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Create model prediction record
        model_record = self.create_provenance_record(
            provenance_type=ProvenanceType.MODEL_PREDICTION,
            source_entity=model_id,
            target_entity=reasoning_id,
            operation="model_inference",
            metadata={
                'input_text': input_text[:500],  # Truncate for storage
                'output_text': output_text[:500],
                'input_length': len(input_text),
                'output_length': len(output_text),
                **(metadata or {})
            }
        )
        
        reasoning_chain.model_calls.append(model_record)
        
        logger.debug("Model call added",
                    reasoning_id=reasoning_id,
                    model_id=model_id,
                    input_length=len(input_text))
        
        return model_record
    
    def finalize_reasoning_chain(self,
                               reasoning_id: str,
                               final_conclusion: str,
                               confidence_score: float) -> ReasoningProvenance:
        """Finalize reasoning chain with conclusion"""
        if reasoning_id not in self.reasoning_chains:
            raise ValueError(f"Reasoning chain {reasoning_id} not found")
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        reasoning_chain.final_conclusion = final_conclusion
        reasoning_chain.confidence_score = confidence_score
        
        # Create final conclusion record
        conclusion_record = self.create_provenance_record(
            provenance_type=ProvenanceType.COMPUTATION,
            source_entity="reasoning_system",
            target_entity=reasoning_id,
            operation="reasoning_conclusion",
            inputs=[r.record_id for r in reasoning_chain.reasoning_steps],
            metadata={
                'conclusion': final_conclusion[:500],
                'confidence_score': confidence_score,
                'total_reasoning_steps': reasoning_chain.total_steps,
                'data_sources_count': len(reasoning_chain.data_sources),
                'model_calls_count': len(reasoning_chain.model_calls)
            },
            trust_level=TrustLevel.TRUSTED if confidence_score > 0.8 else TrustLevel.CREDIBLE
        )
        
        reasoning_chain.reasoning_steps.append(conclusion_record)
        
        logger.info("Reasoning chain finalized",
                   reasoning_id=reasoning_id,
                   confidence_score=confidence_score,
                   total_steps=reasoning_chain.total_steps)
        
        return reasoning_chain
    
    def verify_provenance_chain(self, record_id: str) -> bool:
        """Verify integrity of provenance chain"""
        try:
            if record_id in self.verification_cache:
                return self.verification_cache[record_id]
            
            record = self.provenance_records.get(record_id)
            if not record:
                return False
            
            # Check content hash integrity
            content_data = {
                'type': record.provenance_type.value,
                'source': record.source_entity,
                'target': record.target_entity,
                'operation': record.operation,
                'inputs': record.inputs,
                'outputs': record.outputs,
                'metadata': record.metadata
            }
            expected_hash = hashlib.sha256(json.dumps(content_data, sort_keys=True).encode()).hexdigest()[:16]
            
            if record.content_hash != expected_hash:
                logger.warning("Provenance record hash mismatch", record_id=record_id)
                self.verification_cache[record_id] = False
                return False
            
            # Verify input dependencies
            for input_id in record.inputs:
                if input_id in self.provenance_records:
                    if not self.verify_provenance_chain(input_id):
                        self.verification_cache[record_id] = False
                        return False
            
            record.verification_status = True
            self.verification_cache[record_id] = True
            
            logger.debug("Provenance record verified", record_id=record_id)
            return True
            
        except Exception as e:
            logger.error(f"Provenance verification failed for {record_id}: {e}")
            self.verification_cache[record_id] = False
            return False
    
    def calculate_trust_score(self, reasoning_id: str) -> float:
        """Calculate trust score for reasoning chain"""
        if reasoning_id not in self.reasoning_chains:
            return 0.0
        
        if reasoning_id in self.trust_scores:
            return self.trust_scores[reasoning_id]
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        
        # Trust factors
        verification_score = 0.0
        trust_level_score = 0.0
        completeness_score = 0.0
        
        total_records = len(reasoning_chain.reasoning_steps)
        if total_records == 0:
            return 0.0
        
        # Calculate verification score
        verified_records = sum(1 for step in reasoning_chain.reasoning_steps 
                             if self.verify_provenance_chain(step.record_id))
        verification_score = verified_records / total_records
        
        # Calculate trust level score
        trust_values = {
            TrustLevel.VERIFIED: 1.0,
            TrustLevel.TRUSTED: 0.9,
            TrustLevel.CREDIBLE: 0.7,
            TrustLevel.PROVISIONAL: 0.5,
            TrustLevel.UNCERTAIN: 0.3
        }
        
        avg_trust = sum(trust_values.get(step.trust_level, 0.5) 
                       for step in reasoning_chain.reasoning_steps) / total_records
        trust_level_score = avg_trust
        
        # Calculate completeness score
        has_data_sources = len(reasoning_chain.data_sources) > 0
        has_model_calls = len(reasoning_chain.model_calls) > 0
        has_conclusion = bool(reasoning_chain.final_conclusion)
        
        completeness_score = sum([has_data_sources, has_model_calls, has_conclusion]) / 3.0
        
        # Combine scores
        final_trust_score = (
            verification_score * 0.4 +
            trust_level_score * 0.4 +
            completeness_score * 0.2
        )
        
        self.trust_scores[reasoning_id] = final_trust_score
        
        logger.debug("Trust score calculated",
                    reasoning_id=reasoning_id,
                    trust_score=final_trust_score,
                    verification=verification_score,
                    trust_level=trust_level_score,
                    completeness=completeness_score)
        
        return final_trust_score
    
    def get_reasoning_provenance(self, reasoning_id: str) -> Optional[ReasoningProvenance]:
        """Get complete provenance for reasoning chain"""
        return self.reasoning_chains.get(reasoning_id)
    
    def get_provenance_summary(self, reasoning_id: str) -> Dict[str, Any]:
        """Get summary of provenance information"""
        if reasoning_id not in self.reasoning_chains:
            return {}
        
        reasoning_chain = self.reasoning_chains[reasoning_id]
        trust_score = self.calculate_trust_score(reasoning_id)
        
        return {
            'reasoning_id': reasoning_id,
            'query': reasoning_chain.query,
            'total_steps': reasoning_chain.total_steps,
            'data_sources_count': len(reasoning_chain.data_sources),
            'model_calls_count': len(reasoning_chain.model_calls),
            'confidence_score': reasoning_chain.confidence_score,
            'trust_score': trust_score,
            'verification_status': trust_score > 0.7,
            'final_conclusion_preview': reasoning_chain.final_conclusion[:200] + "..." if reasoning_chain.final_conclusion else "",
            'created_at': reasoning_chain.reasoning_steps[0].timestamp.isoformat() if reasoning_chain.reasoning_steps else None
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get provenance system statistics"""
        total_records = len(self.provenance_records)
        total_chains = len(self.reasoning_chains)
        verified_records = sum(1 for record_id in self.provenance_records 
                             if self.verify_provenance_chain(record_id))
        
        # Type distribution
        type_counts = {}
        for record in self.provenance_records.values():
            ptype = record.provenance_type.value
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        # Trust level distribution
        trust_counts = {}
        for record in self.provenance_records.values():
            trust = record.trust_level.value
            trust_counts[trust] = trust_counts.get(trust, 0) + 1
        
        return {
            'total_provenance_records': total_records,
            'total_reasoning_chains': total_chains,
            'verified_records': verified_records,
            'verification_rate': (verified_records / max(total_records, 1)) * 100,
            'record_types': type_counts,
            'trust_levels': trust_counts,
            'average_chain_length': sum(len(chain.reasoning_steps) for chain in self.reasoning_chains.values()) / max(total_chains, 1)
        }


# Global provenance system instance
_global_provenance_system: Optional[EnhancedProvenanceSystem] = None


def get_enhanced_provenance_system() -> EnhancedProvenanceSystem:
    """Get global enhanced provenance system instance"""
    global _global_provenance_system
    if _global_provenance_system is None:
        _global_provenance_system = EnhancedProvenanceSystem()
    return _global_provenance_system


# Convenience functions
def track_reasoning_step(reasoning_id: str, step_type: str, operation: str, 
                        inputs: List[str] = None, outputs: List[str] = None,
                        metadata: Dict[str, Any] = None) -> ProvenanceRecord:
    """Convenience function to track reasoning step"""
    system = get_enhanced_provenance_system()
    return system.add_reasoning_step(
        reasoning_id=reasoning_id,
        step_type=step_type,
        operation=operation,
        inputs=inputs or [],
        outputs=outputs or [],
        metadata=metadata
    )


def start_reasoning_tracking(query: str) -> str:
    """Start tracking a new reasoning chain"""
    system = get_enhanced_provenance_system()
    return system.start_reasoning_chain(query)
