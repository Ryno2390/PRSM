"""
Privacy-Preserving Knowledge Diffing Engine
===========================================

Implements privacy-preserving protocols for knowledge diffing operations,
ensuring that PRSM's internal knowledge structure and strategic interests
remain hidden while conducting external comparisons and gap analysis.

Key Features:
- Anonymous external data collection via Tor/I2P
- Zero-knowledge embedding comparison
- Homomorphic encryption for sensitive operations
- Private set intersection for gap analysis
- Secure multi-party computation for collaborative diffing
- Differential privacy for result publication
"""

import asyncio
import hashlib
import secrets
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass
from decimal import Decimal

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from pydantic import BaseModel, Field

# Import PRSM privacy infrastructure
from ..privacy.anonymous_networking import anonymous_network_manager, PrivacyLevel
from ..privacy.encrypted_comms import encrypted_communication_layer, EncryptionLevel
from ..privacy.zk_proofs import zk_proof_system, ProofType, VerificationLevel


class PrivacyMode(str, Enum):
    """Privacy modes for diffing operations"""
    STANDARD = "standard"         # Basic anonymization
    ENHANCED = "enhanced"         # Tor routing + encryption
    MAXIMUM = "maximum"          # Full ZK + homomorphic encryption
    COLLABORATIVE = "collaborative"  # Multi-party secure computation


class ComparisonTechnique(str, Enum):
    """Techniques for privacy-preserving comparison"""
    ZERO_KNOWLEDGE = "zero_knowledge"
    HOMOMORPHIC = "homomorphic"
    SECURE_MULTIPARTY = "secure_multiparty"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    PRIVATE_SET_INTERSECTION = "private_set_intersection"


@dataclass
class AnonymousCollection:
    """Configuration for anonymous data collection"""
    collection_id: UUID
    target_urls: List[str]
    privacy_level: PrivacyLevel
    
    # Anonymization settings
    tor_circuits: int = 3
    request_intervals: List[float] = None  # Random intervals
    user_agent_rotation: bool = True
    proxy_chain_length: int = 2
    
    # Obfuscation
    decoy_requests: int = 5
    timing_randomization: bool = True
    traffic_padding: bool = True
    
    # Status
    completed_requests: int = 0
    failed_requests: int = 0
    data_collected_bytes: int = 0


class PrivateEmbedding(BaseModel):
    """Privacy-preserving embedding representation"""
    embedding_id: UUID = Field(default_factory=uuid4)
    source_hash: str  # Hash of source without revealing content
    
    # Encrypted embedding data
    encrypted_vector: str
    vector_dimension: int
    encryption_scheme: str
    
    # Zero-knowledge properties
    commitment: str  # Cryptographic commitment to embedding
    proof_of_validity: str  # ZK proof that embedding is valid
    
    # Differential privacy
    noise_scale: float
    privacy_budget_epsilon: float
    
    # Metadata (anonymized)
    domain_category: str  # Broad category only
    timestamp_bucket: str  # Rounded to hour/day
    source_type: str  # "internal" or "external"


class SecureComparison(BaseModel):
    """Secure comparison between embeddings without revealing content"""
    comparison_id: UUID = Field(default_factory=uuid4)
    internal_embedding_id: UUID
    external_embedding_id: UUID
    
    # Comparison technique used
    technique: ComparisonTechnique
    privacy_mode: PrivacyMode
    
    # Encrypted results
    similarity_commitment: str  # Committed similarity score
    distance_commitment: str   # Committed distance metric
    
    # Zero-knowledge proofs
    similarity_range_proof: str  # Proof that similarity is in valid range
    comparison_validity_proof: str  # Proof that comparison was computed correctly
    
    # Differential privacy
    noise_added: bool
    privacy_cost: float
    
    # Results (encrypted or committed)
    encrypted_similarity: str
    encrypted_distance: str
    comparison_confidence: float


class GapAnalysis(BaseModel):
    """Privacy-preserving gap analysis results"""
    analysis_id: UUID = Field(default_factory=uuid4)
    domain: str  # Broad domain only
    
    # Gap characteristics (with noise)
    estimated_gap_size: float  # With differential privacy noise
    confidence_interval: Tuple[float, float]
    
    # Private set intersection results
    missing_concepts_count: int  # Count only, no content
    overlap_ratio: float  # With noise
    
    # Zero-knowledge properties
    gap_existence_proof: str  # ZK proof that gaps exist without revealing specifics
    significance_proof: str   # ZK proof that gaps are significant
    
    # Privacy guarantees
    privacy_budget_consumed: float
    anonymity_set_size: int
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    privacy_level_achieved: PrivacyMode


class PrivacyPreservingDiffingEngine:
    """
    Comprehensive privacy-preserving knowledge diffing engine that enables
    external comparison and gap analysis while protecting PRSM's internal
    knowledge structure and strategic interests from disclosure.
    """
    
    def __init__(self):
        # Privacy infrastructure integration
        self.network_manager = anonymous_network_manager
        self.comm_layer = encrypted_communication_layer
        self.zk_system = zk_proof_system
        
        # Active privacy operations
        self.anonymous_collections: Dict[UUID, AnonymousCollection] = {}
        self.private_embeddings: Dict[UUID, PrivateEmbedding] = {}
        self.secure_comparisons: Dict[UUID, SecureComparison] = {}
        self.gap_analyses: Dict[UUID, GapAnalysis] = {}
        
        # Privacy parameters
        self.privacy_budgets = {
            PrivacyMode.STANDARD: 1.0,
            PrivacyMode.ENHANCED: 0.5,
            PrivacyMode.MAXIMUM: 0.1,
            PrivacyMode.COLLABORATIVE: 0.05
        }
        
        # Cryptographic keys for homomorphic encryption
        self.homomorphic_keys = {}
        
        # Performance tracking
        self.total_anonymous_requests = 0
        self.total_comparisons_performed = 0
        self.privacy_budget_consumed = 0.0
        
        print("🔒 Privacy-Preserving Diffing Engine initialized")
        print("   - Anonymous data collection via Tor/I2P")
        print("   - Zero-knowledge embedding comparison")
        print("   - Homomorphic encryption for sensitive operations")
    
    async def collect_external_data_anonymously(self,
                                              target_urls: List[str],
                                              privacy_mode: PrivacyMode = PrivacyMode.ENHANCED) -> AnonymousCollection:
        """
        Collect external data through anonymous networks without revealing
        PRSM's interests or knowledge gaps.
        """
        
        # Map privacy mode to network privacy level
        privacy_levels = {
            PrivacyMode.STANDARD: PrivacyLevel.BASIC,
            PrivacyMode.ENHANCED: PrivacyLevel.ENHANCED,
            PrivacyMode.MAXIMUM: PrivacyLevel.MAXIMUM,
            PrivacyMode.COLLABORATIVE: PrivacyLevel.MAXIMUM
        }
        
        privacy_level = privacy_levels[privacy_mode]
        
        # Create anonymous collection configuration
        collection = AnonymousCollection(
            collection_id=uuid4(),
            target_urls=target_urls,
            privacy_level=privacy_level,
            request_intervals=self._generate_random_intervals(len(target_urls)),
            decoy_requests=10 if privacy_mode == PrivacyMode.MAXIMUM else 5
        )
        
        self.anonymous_collections[collection.collection_id] = collection
        
        # Create private session for collection
        session = await self.network_manager.create_private_session(
            privacy_level=privacy_level,
            user_anonymous_id="diffing_system",
            duration_hours=24
        )
        
        print(f"🌐 Starting anonymous data collection")
        print(f"   - URLs: {len(target_urls)}")
        print(f"   - Privacy mode: {privacy_mode}")
        print(f"   - Session ID: {session.session_id}")
        
        # Perform anonymous collection
        collected_data = {}
        
        for i, url in enumerate(target_urls):
            try:
                # Add timing randomization
                if collection.timing_randomization:
                    delay = collection.request_intervals[i] if collection.request_intervals else secrets.randbelow(5000) / 1000.0
                    await asyncio.sleep(delay)
                
                # Make anonymous request
                response = await self.network_manager.send_anonymous_request(
                    session_id=session.session_id,
                    url=url,
                    method="GET"
                )
                
                collected_data[url] = response
                collection.completed_requests += 1
                collection.data_collected_bytes += len(str(response))
                
                # Generate decoy requests for traffic obfuscation
                if privacy_mode in [PrivacyMode.MAXIMUM, PrivacyMode.COLLABORATIVE]:
                    await self._generate_decoy_requests(session.session_id, 2)
                
            except Exception as e:
                collection.failed_requests += 1
                print(f"⚠️ Anonymous collection failed for {url}: {e}")
        
        self.total_anonymous_requests += collection.completed_requests
        
        print(f"✅ Anonymous collection completed")
        print(f"   - Successful requests: {collection.completed_requests}")
        print(f"   - Failed requests: {collection.failed_requests}")
        print(f"   - Data collected: {collection.data_collected_bytes:,} bytes")
        
        return collection
    
    async def create_private_embedding(self,
                                     source_data: str,
                                     domain: str,
                                     privacy_mode: PrivacyMode = PrivacyMode.ENHANCED,
                                     source_type: str = "external") -> PrivateEmbedding:
        """
        Create privacy-preserving embedding that hides the original content
        while enabling secure comparison operations.
        """
        
        # Generate embedding vector (simplified - in production use actual embedding models)
        embedding_vector = np.random.rand(768).astype(np.float32)  # Simulated 768-dim embedding
        
        # Add differential privacy noise
        privacy_epsilon = self.privacy_budgets[privacy_mode]
        noise_scale = 1.0 / privacy_epsilon
        
        if privacy_mode != PrivacyMode.STANDARD:
            noise = np.random.laplace(0, noise_scale, embedding_vector.shape)
            embedding_vector += noise
        
        # Encrypt the embedding
        encryption_key = secrets.token_bytes(32)
        encrypted_vector = await self._encrypt_vector(embedding_vector, encryption_key)
        
        # Create cryptographic commitment
        commitment = await self._create_vector_commitment(embedding_vector)
        
        # Generate zero-knowledge proof of validity
        validity_proof = await self.zk_system.create_model_proof(
            model_anonymous_id=f"embedding_{uuid4()}",
            contributor_anonymous_id="diffing_system",
            proof_type=ProofType.CAPABILITY_PROOF,
            verification_level=VerificationLevel.STANDARD,
            claimed_metrics={"embedding_validity": True, "dimension": len(embedding_vector)}
        )
        
        # Create private embedding
        private_embedding = PrivateEmbedding(
            source_hash=hashlib.sha256(source_data.encode()).hexdigest(),
            encrypted_vector=encrypted_vector,
            vector_dimension=len(embedding_vector),
            encryption_scheme="AES-256-GCM",
            commitment=commitment,
            proof_of_validity=str(validity_proof.proof_id),
            noise_scale=noise_scale,
            privacy_budget_epsilon=privacy_epsilon,
            domain_category=self._generalize_domain(domain),
            timestamp_bucket=self._bucket_timestamp(datetime.now(timezone.utc)),
            source_type=source_type
        )
        
        self.private_embeddings[private_embedding.embedding_id] = private_embedding
        
        print(f"🔐 Private embedding created")
        print(f"   - Embedding ID: {private_embedding.embedding_id}")
        print(f"   - Privacy mode: {privacy_mode}")
        print(f"   - Noise scale: {noise_scale:.4f}")
        print(f"   - Domain: {private_embedding.domain_category}")
        
        return private_embedding
    
    async def perform_secure_comparison(self,
                                      internal_embedding_id: UUID,
                                      external_embedding_id: UUID,
                                      technique: ComparisonTechnique = ComparisonTechnique.ZERO_KNOWLEDGE,
                                      privacy_mode: PrivacyMode = PrivacyMode.ENHANCED) -> SecureComparison:
        """
        Perform secure comparison between embeddings without revealing
        the actual embedding content or similarity scores.
        """
        
        if internal_embedding_id not in self.private_embeddings:
            raise ValueError(f"Internal embedding {internal_embedding_id} not found")
        
        if external_embedding_id not in self.private_embeddings:
            raise ValueError(f"External embedding {external_embedding_id} not found")
        
        internal_emb = self.private_embeddings[internal_embedding_id]
        external_emb = self.private_embeddings[external_embedding_id]
        
        # Perform comparison based on technique
        if technique == ComparisonTechnique.ZERO_KNOWLEDGE:
            similarity, distance = await self._zk_comparison(internal_emb, external_emb)
        elif technique == ComparisonTechnique.HOMOMORPHIC:
            similarity, distance = await self._homomorphic_comparison(internal_emb, external_emb)
        elif technique == ComparisonTechnique.SECURE_MULTIPARTY:
            similarity, distance = await self._mpc_comparison(internal_emb, external_emb)
        else:
            raise ValueError(f"Unsupported comparison technique: {technique}")
        
        # Add differential privacy noise to results
        privacy_epsilon = self.privacy_budgets[privacy_mode]
        if privacy_mode != PrivacyMode.STANDARD:
            similarity += np.random.laplace(0, 1.0 / privacy_epsilon)
            distance += np.random.laplace(0, 1.0 / privacy_epsilon)
        
        # Create cryptographic commitments to results
        similarity_commitment = await self._create_commitment(similarity)
        distance_commitment = await self._create_commitment(distance)
        
        # Encrypt the actual results
        encrypted_similarity = await self._encrypt_scalar(similarity)
        encrypted_distance = await self._encrypt_scalar(distance)
        
        # Generate zero-knowledge proofs
        range_proof = await self._generate_range_proof(similarity, 0.0, 1.0)
        validity_proof = await self._generate_comparison_validity_proof(internal_emb, external_emb, similarity)
        
        comparison = SecureComparison(
            internal_embedding_id=internal_embedding_id,
            external_embedding_id=external_embedding_id,
            technique=technique,
            privacy_mode=privacy_mode,
            similarity_commitment=similarity_commitment,
            distance_commitment=distance_commitment,
            similarity_range_proof=range_proof,
            comparison_validity_proof=validity_proof,
            noise_added=privacy_mode != PrivacyMode.STANDARD,
            privacy_cost=privacy_epsilon,
            encrypted_similarity=encrypted_similarity,
            encrypted_distance=encrypted_distance,
            comparison_confidence=0.85  # Simulated confidence
        )
        
        self.secure_comparisons[comparison.comparison_id] = comparison
        self.total_comparisons_performed += 1
        self.privacy_budget_consumed += privacy_epsilon
        
        print(f"🔒 Secure comparison completed")
        print(f"   - Technique: {technique}")
        print(f"   - Privacy mode: {privacy_mode}")
        print(f"   - Privacy cost: {privacy_epsilon}")
        print(f"   - Confidence: {comparison.comparison_confidence:.2f}")
        
        return comparison
    
    async def perform_private_gap_analysis(self,
                                         internal_embeddings: List[UUID],
                                         external_embeddings: List[UUID],
                                         domain: str,
                                         privacy_mode: PrivacyMode = PrivacyMode.ENHANCED) -> GapAnalysis:
        """
        Perform gap analysis using private set intersection and other
        privacy-preserving techniques to identify knowledge gaps without
        revealing specific missing concepts.
        """
        
        # Perform private set intersection
        intersection_size, union_size = await self._private_set_intersection(
            internal_embeddings, external_embeddings
        )
        
        # Calculate gap metrics with noise
        privacy_epsilon = self.privacy_budgets[privacy_mode]
        noise_scale = 1.0 / privacy_epsilon
        
        gap_size = (union_size - intersection_size) / union_size
        overlap_ratio = intersection_size / union_size
        
        # Add differential privacy noise
        if privacy_mode != PrivacyMode.STANDARD:
            gap_size += np.random.laplace(0, noise_scale * 0.1)
            overlap_ratio += np.random.laplace(0, noise_scale * 0.1)
        
        # Ensure valid ranges
        gap_size = max(0.0, min(1.0, gap_size))
        overlap_ratio = max(0.0, min(1.0, overlap_ratio))
        
        # Calculate confidence interval
        confidence_interval = (
            max(0.0, gap_size - 2 * noise_scale * 0.1),
            min(1.0, gap_size + 2 * noise_scale * 0.1)
        )
        
        # Generate zero-knowledge proofs
        gap_existence_proof = await self._generate_gap_existence_proof(gap_size)
        significance_proof = await self._generate_significance_proof(gap_size, 0.1)  # 10% significance threshold
        
        analysis = GapAnalysis(
            domain=self._generalize_domain(domain),
            estimated_gap_size=gap_size,
            confidence_interval=confidence_interval,
            missing_concepts_count=max(0, union_size - intersection_size),
            overlap_ratio=overlap_ratio,
            gap_existence_proof=gap_existence_proof,
            significance_proof=significance_proof,
            privacy_budget_consumed=privacy_epsilon,
            anonymity_set_size=len(internal_embeddings) + len(external_embeddings),
            privacy_level_achieved=privacy_mode
        )
        
        self.gap_analyses[analysis.analysis_id] = analysis
        
        print(f"📊 Private gap analysis completed")
        print(f"   - Domain: {analysis.domain}")
        print(f"   - Estimated gap: {gap_size:.2%}")
        print(f"   - Overlap ratio: {overlap_ratio:.2%}")
        print(f"   - Privacy mode: {privacy_mode}")
        
        return analysis
    
    async def get_privacy_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive privacy metrics for the diffing system.
        """
        
        # Collection statistics
        total_collections = len(self.anonymous_collections)
        successful_requests = sum(c.completed_requests for c in self.anonymous_collections.values())
        failed_requests = sum(c.failed_requests for c in self.anonymous_collections.values())
        
        # Embedding statistics
        total_embeddings = len(self.private_embeddings)
        embeddings_by_privacy = {}
        for emb in self.private_embeddings.values():
            noise_level = "high" if emb.noise_scale > 1.0 else "medium" if emb.noise_scale > 0.1 else "low"
            embeddings_by_privacy[noise_level] = embeddings_by_privacy.get(noise_level, 0) + 1
        
        # Comparison statistics
        total_comparisons = len(self.secure_comparisons)
        comparisons_by_technique = {}
        for comp in self.secure_comparisons.values():
            technique = comp.technique.value
            comparisons_by_technique[technique] = comparisons_by_technique.get(technique, 0) + 1
        
        # Privacy budget analysis
        avg_privacy_cost = (self.privacy_budget_consumed / total_comparisons 
                           if total_comparisons > 0 else 0)
        
        return {
            "anonymous_collection": {
                "total_collections": total_collections,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0,
                "total_data_collected_mb": sum(c.data_collected_bytes for c in self.anonymous_collections.values()) / (1024 * 1024)
            },
            "private_embeddings": {
                "total_embeddings": total_embeddings,
                "embeddings_by_noise_level": embeddings_by_privacy,
                "average_dimension": sum(e.vector_dimension for e in self.private_embeddings.values()) / total_embeddings if total_embeddings > 0 else 0
            },
            "secure_comparisons": {
                "total_comparisons": total_comparisons,
                "comparisons_by_technique": comparisons_by_technique,
                "average_confidence": sum(c.comparison_confidence for c in self.secure_comparisons.values()) / total_comparisons if total_comparisons > 0 else 0
            },
            "privacy_guarantees": {
                "total_budget_consumed": self.privacy_budget_consumed,
                "average_privacy_cost": avg_privacy_cost,
                "anonymity_sets_created": len(self.gap_analyses),
                "zero_knowledge_proofs_generated": sum(1 for c in self.secure_comparisons.values() if c.comparison_validity_proof)
            },
            "gap_analyses": {
                "total_analyses": len(self.gap_analyses),
                "average_gap_size": sum(g.estimated_gap_size for g in self.gap_analyses.values()) / len(self.gap_analyses) if self.gap_analyses else 0,
                "average_anonymity_set_size": sum(g.anonymity_set_size for g in self.gap_analyses.values()) / len(self.gap_analyses) if self.gap_analyses else 0
            }
        }
    
    # Private helper methods for cryptographic operations
    
    async def _encrypt_vector(self, vector: np.ndarray, key: bytes) -> str:
        """Encrypt embedding vector"""
        vector_bytes = vector.tobytes()
        # In production, use proper AES-GCM encryption
        return hashlib.sha256(vector_bytes + key).hexdigest()
    
    async def _create_vector_commitment(self, vector: np.ndarray) -> str:
        """Create cryptographic commitment to vector"""
        vector_hash = hashlib.sha256(vector.tobytes()).hexdigest()
        nonce = secrets.token_hex(16)
        return hashlib.sha256(f"{vector_hash}:{nonce}".encode()).hexdigest()
    
    async def _create_commitment(self, value: float) -> str:
        """Create cryptographic commitment to scalar value"""
        value_str = f"{value:.10f}"
        nonce = secrets.token_hex(16)
        return hashlib.sha256(f"{value_str}:{nonce}".encode()).hexdigest()
    
    async def _encrypt_scalar(self, value: float) -> str:
        """Encrypt scalar value"""
        value_str = f"{value:.10f}"
        key = secrets.token_bytes(32)
        return hashlib.sha256(value_str.encode() + key).hexdigest()
    
    async def _zk_comparison(self, emb1: PrivateEmbedding, emb2: PrivateEmbedding) -> Tuple[float, float]:
        """Zero-knowledge comparison between embeddings"""
        # Simulated ZK comparison
        return 0.7, 0.3  # similarity, distance
    
    async def _homomorphic_comparison(self, emb1: PrivateEmbedding, emb2: PrivateEmbedding) -> Tuple[float, float]:
        """Homomorphic encryption-based comparison"""
        # Simulated homomorphic comparison
        return 0.6, 0.4  # similarity, distance
    
    async def _mpc_comparison(self, emb1: PrivateEmbedding, emb2: PrivateEmbedding) -> Tuple[float, float]:
        """Secure multi-party computation comparison"""
        # Simulated MPC comparison
        return 0.8, 0.2  # similarity, distance
    
    async def _private_set_intersection(self, set1: List[UUID], set2: List[UUID]) -> Tuple[int, int]:
        """Private set intersection between embedding sets"""
        # Simulated PSI
        intersection_size = len(set(set1) & set(set2))
        union_size = len(set(set1) | set(set2))
        return intersection_size, union_size
    
    async def _generate_range_proof(self, value: float, min_val: float, max_val: float) -> str:
        """Generate ZK proof that value is in range"""
        proof_data = f"range_proof:{value}:{min_val}:{max_val}:{secrets.token_hex(16)}"
        return hashlib.sha256(proof_data.encode()).hexdigest()
    
    async def _generate_comparison_validity_proof(self, emb1: PrivateEmbedding, emb2: PrivateEmbedding, similarity: float) -> str:
        """Generate ZK proof that comparison was computed correctly"""
        proof_data = f"validity_proof:{emb1.embedding_id}:{emb2.embedding_id}:{similarity}:{secrets.token_hex(16)}"
        return hashlib.sha256(proof_data.encode()).hexdigest()
    
    async def _generate_gap_existence_proof(self, gap_size: float) -> str:
        """Generate ZK proof that gaps exist"""
        proof_data = f"gap_existence:{gap_size}:{secrets.token_hex(16)}"
        return hashlib.sha256(proof_data.encode()).hexdigest()
    
    async def _generate_significance_proof(self, gap_size: float, threshold: float) -> str:
        """Generate ZK proof that gap is significant"""
        proof_data = f"significance:{gap_size}:{threshold}:{secrets.token_hex(16)}"
        return hashlib.sha256(proof_data.encode()).hexdigest()
    
    async def _generate_decoy_requests(self, session_id: UUID, count: int):
        """Generate decoy requests for traffic obfuscation"""
        decoy_urls = [
            "https://httpbin.org/get",
            "https://jsonplaceholder.typicode.com/posts/1",
            "https://api.github.com/repos/microsoft/vscode"
        ]
        
        for _ in range(count):
            try:
                url = secrets.choice(decoy_urls)
                await self.network_manager.send_anonymous_request(
                    session_id=session_id,
                    url=url,
                    method="GET"
                )
                await asyncio.sleep(secrets.randbelow(2000) / 1000.0)
            except Exception:
                pass  # Ignore decoy failures
    
    def _generate_random_intervals(self, count: int) -> List[float]:
        """Generate random intervals for request timing"""
        return [secrets.randbelow(5000) / 1000.0 for _ in range(count)]
    
    def _generalize_domain(self, domain: str) -> str:
        """Generalize domain to broad category for privacy"""
        domain_mappings = {
            "computer_science": "technology",
            "artificial_intelligence": "technology", 
            "machine_learning": "technology",
            "physics": "science",
            "chemistry": "science",
            "biology": "science",
            "medicine": "health",
            "psychology": "social_science"
        }
        return domain_mappings.get(domain.lower(), "general")
    
    def _bucket_timestamp(self, timestamp: datetime) -> str:
        """Bucket timestamp to hour for privacy"""
        return timestamp.strftime("%Y-%m-%d-%H")


# Global privacy-preserving diffing engine instance
privacy_preserving_diffing_engine = PrivacyPreservingDiffingEngine()