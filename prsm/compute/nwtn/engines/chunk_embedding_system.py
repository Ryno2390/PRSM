#!/usr/bin/env python3
"""
Chunk Embedding and Ranking System for Meta-Paper Generation
============================================================

This module implements multi-dimensional embedding strategies for semantic chunks,
enabling sophisticated ranking and selection for meta-paper assembly. The system
supports query-specific relevance scoring, reasoning-engine-specific weighting,
and intelligent chunk assembly for optimal research contexts.

Key Features:
- Multiple embedding types (content, concept, mathematical, methodological)
- Query-specific similarity scoring with reasoning engine adaptation
- Multi-objective chunk selection with diversity constraints
- Evidence strength and novelty integration
- Semantic overlap detection for deduplication

Integration: Works with chunk_classification_system.py for meta-paper generation
"""

import re
import os
import time
import asyncio
import logging
import hashlib
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# Import from our chunk classification system
from .chunk_classification_system import ChunkType, SemanticChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkEmbeddings:
    """Container for multiple embedding representations of a chunk"""
    chunk_id: str
    content_embedding: Optional[List[float]] = None      # General content similarity
    concept_embedding: Optional[List[float]] = None      # Concept-based similarity  
    math_embedding: Optional[List[float]] = None         # Mathematical content similarity
    method_embedding: Optional[List[float]] = None       # Methodological similarity
    results_embedding: Optional[List[float]] = None      # Results-focused similarity
    embedding_dim: int = 384  # Default embedding dimension
    
    def get_embedding_for_type(self, chunk_type: ChunkType) -> Optional[List[float]]:
        """Get most appropriate embedding for chunk type"""
        type_mapping = {
            ChunkType.MATHEMATICAL_FORMULATION: self.math_embedding,
            ChunkType.ALGORITHM_DESCRIPTION: self.method_embedding,
            ChunkType.METHODOLOGY: self.method_embedding,
            ChunkType.EXPERIMENTAL_SETUP: self.method_embedding,
            ChunkType.RESULTS_QUANTITATIVE: self.results_embedding,
            ChunkType.RESULTS_QUALITATIVE: self.results_embedding,
            ChunkType.PERFORMANCE_METRICS: self.results_embedding
        }
        
        # Return specialized embedding if available, otherwise content embedding
        specialized = type_mapping.get(chunk_type)
        return specialized if specialized is not None else self.content_embedding


@dataclass
class QueryRelevanceScore:
    """Relevance score with breakdown"""
    total_score: float
    content_similarity: float
    concept_similarity: float
    type_weight: float
    evidence_bonus: float
    novelty_bonus: float
    recency_bonus: float
    reasoning_engine_bonus: float


@dataclass
class ChunkSelectionResult:
    """Result of chunk selection process"""
    selected_chunks: List[SemanticChunk]
    selection_time: float
    total_chunks_considered: int
    diversity_score: float
    coverage_score: float
    average_relevance: float
    token_usage: int


class SimpleEmbeddingGenerator:
    """Simple embedding generation without external dependencies"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.concept_weights = self._initialize_concept_weights()
        
    def _initialize_concept_weights(self) -> Dict[str, float]:
        """Initialize concept importance weights"""
        return {
            # Core ML/AI concepts
            'neural': 0.9, 'network': 0.9, 'learning': 0.9, 'model': 0.8,
            'algorithm': 0.8, 'training': 0.8, 'accuracy': 0.9, 'performance': 0.8,
            
            # Research methodology
            'experiment': 0.7, 'evaluation': 0.7, 'method': 0.8, 'approach': 0.7,
            'framework': 0.7, 'analysis': 0.6, 'results': 0.8, 'findings': 0.8,
            
            # Mathematical/statistical
            'equation': 0.8, 'formula': 0.8, 'optimization': 0.7, 'probability': 0.7,
            'statistical': 0.7, 'significant': 0.8, 'hypothesis': 0.6,
            
            # Problem-solving
            'problem': 0.7, 'solution': 0.8, 'challenge': 0.6, 'improvement': 0.7,
            'novel': 0.8, 'innovative': 0.8, 'breakthrough': 0.9
        }
    
    async def generate_content_embedding(self, text: str) -> List[float]:
        """Generate content-based embedding using TF-IDF-like approach"""
        # Tokenize and clean text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Create term frequency map
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1
        
        # Generate embedding vector
        embedding = [0.0] * self.embedding_dim
        
        # Map words to embedding dimensions using hash
        for word, freq in term_freq.items():
            # Use concept weight if available
            weight = self.concept_weights.get(word, 0.1)
            
            # Map word to multiple dimensions using hash
            for i in range(3):  # Use 3 dimensions per word for better distribution
                hash_val = hash(f"{word}_{i}") % self.embedding_dim
                embedding[hash_val] += freq * weight
        
        # Normalize embedding
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def generate_concept_embedding(self, concepts: List[str]) -> List[float]:
        """Generate concept-based embedding"""
        embedding = [0.0] * self.embedding_dim
        
        if not concepts:
            return embedding
        
        for concept in concepts:
            concept_words = concept.lower().split()
            for word in concept_words:
                weight = self.concept_weights.get(word, 0.5)
                
                # Map concept words to embedding dimensions
                hash_val = hash(word) % self.embedding_dim
                embedding[hash_val] += weight
        
        # Normalize
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def generate_math_embedding(self, text: str) -> List[float]:
        """Generate mathematics-focused embedding"""
        # Look for mathematical patterns
        math_patterns = {
            'equation': r'[a-zA-Z]\s*=\s*[^.!?]*[.!?]',
            'formula': r'\$[^$]+\$|\\[a-zA-Z]+\{[^}]*\}',
            'variables': r'\b[a-zA-Z]\s*[=<>]\s*\w+',
            'greek': r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega)',
            'operators': r'[+\-*/^]|\\(sum|prod|int|frac|sqrt)',
            'numbers': r'\b\d+\.?\d*\b'
        }
        
        math_features = {}
        for pattern_name, pattern in math_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            math_features[pattern_name] = len(matches)
        
        # Create math-specific embedding
        embedding = [0.0] * self.embedding_dim
        
        for i, (feature, count) in enumerate(math_features.items()):
            # Distribute mathematical features across embedding dimensions
            start_dim = (i * self.embedding_dim // len(math_features))
            end_dim = ((i + 1) * self.embedding_dim // len(math_features))
            
            for dim in range(start_dim, end_dim):
                embedding[dim] = count * 0.1  # Scale appropriately
        
        # Normalize
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def generate_method_embedding(self, text: str) -> List[float]:
        """Generate methodology-focused embedding"""
        method_keywords = [
            'method', 'approach', 'technique', 'procedure', 'algorithm',
            'framework', 'system', 'model', 'implementation', 'design',
            'architecture', 'structure', 'process', 'workflow', 'pipeline',
            'evaluation', 'assessment', 'measurement', 'analysis', 'experiment'
        ]
        
        text_lower = text.lower()
        method_scores = {}
        
        for keyword in method_keywords:
            count = text_lower.count(keyword)
            if count > 0:
                method_scores[keyword] = count
        
        # Create methodology embedding
        embedding = [0.0] * self.embedding_dim
        
        for i, (keyword, count) in enumerate(method_scores.items()):
            # Map methodology terms to embedding space
            hash_val = hash(keyword) % self.embedding_dim
            embedding[hash_val] = count * 0.2
            
            # Also distribute to nearby dimensions for smoother representation
            for offset in [-1, 1]:
                neighbor_dim = (hash_val + offset) % self.embedding_dim
                embedding[neighbor_dim] += count * 0.1
        
        # Normalize
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def generate_results_embedding(self, text: str, quantitative_data: List[Dict]) -> List[float]:
        """Generate results-focused embedding"""
        results_keywords = [
            'results', 'findings', 'outcomes', 'performance', 'accuracy',
            'precision', 'recall', 'improvement', 'comparison', 'evaluation',
            'significant', 'better', 'superior', 'outperforms', 'achieves'
        ]
        
        embedding = [0.0] * self.embedding_dim
        text_lower = text.lower()
        
        # Score based on results keywords
        for keyword in results_keywords:
            count = text_lower.count(keyword)
            if count > 0:
                hash_val = hash(keyword) % self.embedding_dim
                embedding[hash_val] += count * 0.2
        
        # Boost based on quantitative data presence
        quant_boost = min(1.0, len(quantitative_data) * 0.1)
        for i in range(self.embedding_dim):
            embedding[i] += quant_boost
        
        # Special boost for performance metrics
        for data_item in quantitative_data:
            if data_item.get('type') == 'performance_metric':
                # Boost dimensions related to performance
                for i in range(0, self.embedding_dim, 10):  # Every 10th dimension
                    embedding[i] += 0.3
        
        # Normalize
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding


class ChunkEmbeddingSystem:
    """Main system for generating and managing chunk embeddings"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.generator = SimpleEmbeddingGenerator(embedding_dim)
        self.embedding_cache = {}  # Cache embeddings to avoid recomputation
        
    async def generate_chunk_embeddings(self, chunk: SemanticChunk) -> ChunkEmbeddings:
        """Generate comprehensive embeddings for a semantic chunk"""
        
        # Check cache first
        cache_key = f"{chunk.chunk_hash}_{chunk.chunk_type.value}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        start_time = time.time()
        
        # Generate different types of embeddings based on chunk content
        embeddings = ChunkEmbeddings(
            chunk_id=chunk.chunk_hash,
            embedding_dim=self.embedding_dim
        )
        
        # Always generate content embedding
        embeddings.content_embedding = await self.generator.generate_content_embedding(chunk.content)
        
        # Generate concept embedding if concepts are available
        if chunk.key_concepts:
            embeddings.concept_embedding = await self.generator.generate_concept_embedding(chunk.key_concepts)
        
        # Generate specialized embeddings based on chunk type
        if chunk.chunk_type == ChunkType.MATHEMATICAL_FORMULATION:
            embeddings.math_embedding = await self.generator.generate_math_embedding(chunk.content)
        
        elif chunk.chunk_type in [ChunkType.METHODOLOGY, ChunkType.ALGORITHM_DESCRIPTION, ChunkType.EXPERIMENTAL_SETUP]:
            embeddings.method_embedding = await self.generator.generate_method_embedding(chunk.content)
        
        elif chunk.chunk_type in [ChunkType.RESULTS_QUANTITATIVE, ChunkType.RESULTS_QUALITATIVE, ChunkType.PERFORMANCE_METRICS]:
            embeddings.results_embedding = await self.generator.generate_results_embedding(chunk.content, chunk.quantitative_data)
        
        # Cache the embeddings
        self.embedding_cache[cache_key] = embeddings
        
        generation_time = time.time() - start_time
        logger.debug(f"Generated embeddings for chunk {chunk.chunk_hash} in {generation_time:.3f}s")
        
        return embeddings
    
    async def calculate_query_relevance(self, 
                                      chunk: SemanticChunk,
                                      chunk_embeddings: ChunkEmbeddings,
                                      query: str,
                                      reasoning_engine: str) -> QueryRelevanceScore:
        """Calculate comprehensive query relevance score"""
        
        # Generate query embedding
        query_embedding = await self.generator.generate_content_embedding(query)
        
        # Calculate base content similarity
        content_similarity = self._cosine_similarity(
            chunk_embeddings.content_embedding, 
            query_embedding
        ) if chunk_embeddings.content_embedding else 0.0
        
        # Calculate concept similarity if available
        concept_similarity = 0.0
        if chunk_embeddings.concept_embedding:
            query_concepts = await self.generator.generate_concept_embedding([query])
            concept_similarity = self._cosine_similarity(
                chunk_embeddings.concept_embedding,
                query_concepts
            )
        
        # Apply reasoning-engine-specific weights
        type_weight = self._get_reasoning_engine_weight(chunk.chunk_type, reasoning_engine)
        
        # Calculate bonuses
        evidence_bonus = chunk.evidence_strength * 0.1
        novelty_bonus = chunk.novelty_score * 0.05
        recency_bonus = chunk.recency_score * 0.05
        
        # Special reasoning engine bonus
        reasoning_bonus = self._calculate_reasoning_engine_bonus(chunk, reasoning_engine)
        
        # Combine all factors
        total_score = (
            content_similarity * 0.4 +
            concept_similarity * 0.3 +
            type_weight * 0.2 +
            evidence_bonus + novelty_bonus + recency_bonus + reasoning_bonus
        )
        
        return QueryRelevanceScore(
            total_score=min(1.0, total_score),
            content_similarity=content_similarity,
            concept_similarity=concept_similarity,
            type_weight=type_weight,
            evidence_bonus=evidence_bonus,
            novelty_bonus=novelty_bonus,
            recency_bonus=recency_bonus,
            reasoning_engine_bonus=reasoning_bonus
        )
    
    def _cosine_similarity(self, vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Return cosine similarity
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _get_reasoning_engine_weight(self, chunk_type: ChunkType, reasoning_engine: str) -> float:
        """Get type weight based on reasoning engine"""
        engine_preferences = {
            "causal": {
                ChunkType.METHODOLOGY: 0.9,
                ChunkType.THEORETICAL_FRAMEWORK: 0.8,
                ChunkType.RESULTS_QUANTITATIVE: 0.7,
                ChunkType.EXPERIMENTAL_SETUP: 0.8
            },
            "probabilistic": {
                ChunkType.RESULTS_QUANTITATIVE: 0.9,
                ChunkType.PERFORMANCE_METRICS: 0.8,
                ChunkType.MATHEMATICAL_FORMULATION: 0.7
            },
            "abductive": {
                ChunkType.KEY_FINDINGS: 0.9,
                ChunkType.IMPLICATIONS: 0.8,
                ChunkType.THEORETICAL_FRAMEWORK: 0.7
            },
            "deductive": {
                ChunkType.MATHEMATICAL_FORMULATION: 0.9,
                ChunkType.THEORETICAL_FRAMEWORK: 0.8,
                ChunkType.ALGORITHM_DESCRIPTION: 0.7
            },
            "inductive": {
                ChunkType.RESULTS_QUANTITATIVE: 0.8,
                ChunkType.KEY_FINDINGS: 0.8,
                ChunkType.EXPERIMENTAL_SETUP: 0.7
            },
            "analogical": {
                ChunkType.RELATED_WORK: 0.8,
                ChunkType.IMPLICATIONS: 0.7,
                ChunkType.THEORETICAL_FRAMEWORK: 0.7
            }
        }
        
        engine_prefs = engine_preferences.get(reasoning_engine, {})
        return engine_prefs.get(chunk_type, 0.5)  # Default weight
    
    def _calculate_reasoning_engine_bonus(self, chunk: SemanticChunk, reasoning_engine: str) -> float:
        """Calculate bonus based on reasoning engine alignment"""
        content_lower = chunk.content.lower()
        
        engine_keywords = {
            "causal": ['cause', 'effect', 'because', 'leads to', 'results in', 'due to'],
            "probabilistic": ['probability', 'likely', 'chance', 'random', 'distribution', 'statistical'],
            "abductive": ['explains', 'suggests', 'indicates', 'implies', 'hypothesis', 'theory'],
            "deductive": ['therefore', 'thus', 'consequently', 'follows', 'proof', 'conclude'],
            "inductive": ['pattern', 'trend', 'generally', 'typically', 'observed', 'evidence'],
            "analogical": ['similar', 'like', 'analogous', 'comparable', 'parallel', 'resembles']
        }
        
        keywords = engine_keywords.get(reasoning_engine, [])
        keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
        
        return min(0.15, keyword_count * 0.03)  # Max 0.15 bonus


class ChunkRankingSystem:
    """System for ranking and selecting chunks for meta-paper assembly"""
    
    def __init__(self):
        self.embedding_system = ChunkEmbeddingSystem()
    
    async def rank_chunks_for_query(self,
                                  chunks: List[SemanticChunk],
                                  query: str,
                                  reasoning_engine: str,
                                  top_k: int = 50) -> List[Tuple[SemanticChunk, QueryRelevanceScore]]:
        """Rank chunks by relevance to query and reasoning engine"""
        
        ranked_chunks = []
        
        # Generate embeddings and calculate relevance for each chunk
        for chunk in chunks:
            embeddings = await self.embedding_system.generate_chunk_embeddings(chunk)
            relevance = await self.embedding_system.calculate_query_relevance(
                chunk, embeddings, query, reasoning_engine
            )
            ranked_chunks.append((chunk, relevance))
        
        # Sort by total relevance score
        ranked_chunks.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return ranked_chunks[:top_k]
    
    async def select_optimal_chunks(self,
                                  ranked_chunks: List[Tuple[SemanticChunk, QueryRelevanceScore]],
                                  token_budget: int = 1500,
                                  diversity_threshold: float = 0.8) -> ChunkSelectionResult:
        """Select optimal chunks with diversity and token budget constraints"""
        
        start_time = time.time()
        selected_chunks = []
        used_tokens = 0
        covered_concepts = set()
        
        for chunk, relevance_score in ranked_chunks:
            # Check token budget (estimate 1.3 tokens per word)
            estimated_tokens = chunk.word_count * 1.3
            if used_tokens + estimated_tokens > token_budget:
                continue
            
            # Check conceptual diversity
            chunk_concepts = set(chunk.key_concepts)
            if covered_concepts:
                concept_overlap = len(chunk_concepts & covered_concepts) / max(len(chunk_concepts), 1)
                if concept_overlap > 0.7:  # Skip if >70% concept overlap
                    continue
            
            # Check content diversity (simplified without embeddings)
            content_similarity = self._calculate_content_similarity(chunk, selected_chunks)
            if content_similarity > diversity_threshold:
                continue
            
            # Add chunk to selection
            selected_chunks.append(chunk)
            used_tokens += estimated_tokens
            covered_concepts.update(chunk_concepts)
            
            # Stop if we have enough high-quality chunks
            if len(selected_chunks) >= 20 or used_tokens > token_budget * 0.9:
                break
        
        selection_time = time.time() - start_time
        
        # Calculate result metrics
        diversity_score = self._calculate_diversity_score(selected_chunks)
        coverage_score = len(covered_concepts) / max(len(set().union(*(c.key_concepts for c in selected_chunks))), 1)
        avg_relevance = sum(r.total_score for _, r in ranked_chunks[:len(selected_chunks)]) / max(len(selected_chunks), 1)
        
        return ChunkSelectionResult(
            selected_chunks=selected_chunks,
            selection_time=selection_time,
            total_chunks_considered=len(ranked_chunks),
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            average_relevance=avg_relevance,
            token_usage=int(used_tokens)
        )
    
    def _calculate_content_similarity(self, chunk: SemanticChunk, selected_chunks: List[SemanticChunk]) -> float:
        """Calculate content similarity using simple text overlap"""
        if not selected_chunks:
            return 0.0
        
        chunk_words = set(chunk.content.lower().split())
        max_similarity = 0.0
        
        for selected in selected_chunks:
            selected_words = set(selected.content.lower().split())
            if chunk_words and selected_words:
                intersection = len(chunk_words & selected_words)
                union = len(chunk_words | selected_words)
                similarity = intersection / union if union > 0 else 0.0
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_diversity_score(self, chunks: List[SemanticChunk]) -> float:
        """Calculate diversity score of selected chunks"""
        if len(chunks) <= 1:
            return 1.0
        
        # Calculate average pairwise diversity
        total_diversity = 0.0
        comparisons = 0
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                # Type diversity
                type_diversity = 1.0 if chunk1.chunk_type != chunk2.chunk_type else 0.5
                
                # Concept diversity
                concepts1 = set(chunk1.key_concepts)
                concepts2 = set(chunk2.key_concepts)
                if concepts1 or concepts2:
                    concept_overlap = len(concepts1 & concepts2) / max(len(concepts1 | concepts2), 1)
                    concept_diversity = 1.0 - concept_overlap
                else:
                    concept_diversity = 0.5
                
                # Content diversity (simple word overlap)
                words1 = set(chunk1.content.lower().split()[:50])  # First 50 words
                words2 = set(chunk2.content.lower().split()[:50])
                word_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                content_diversity = 1.0 - word_overlap
                
                # Combined diversity
                chunk_diversity = (type_diversity + concept_diversity + content_diversity) / 3
                total_diversity += chunk_diversity
                comparisons += 1
        
        return total_diversity / max(comparisons, 1)


# Main interface functions
async def rank_and_select_chunks(chunks: List[SemanticChunk],
                                query: str,
                                reasoning_engine: str,
                                token_budget: int = 1500,
                                top_k: int = 50) -> ChunkSelectionResult:
    """
    Main interface for ranking and selecting chunks for meta-paper generation
    
    Args:
        chunks: List of classified semantic chunks
        query: User query for relevance ranking
        reasoning_engine: Target reasoning engine for type weighting
        token_budget: Maximum tokens for selected chunks
        top_k: Number of top chunks to consider
        
    Returns:
        ChunkSelectionResult with optimally selected chunks
    """
    ranking_system = ChunkRankingSystem()
    
    # Rank chunks by query relevance
    ranked_chunks = await ranking_system.rank_chunks_for_query(
        chunks, query, reasoning_engine, top_k
    )
    
    # Select optimal subset with diversity constraints
    selection_result = await ranking_system.select_optimal_chunks(
        ranked_chunks, token_budget
    )
    
    return selection_result


def get_reasoning_engine_descriptions() -> Dict[str, str]:
    """Get descriptions of reasoning engine preferences"""
    return {
        "causal": "Focuses on cause-and-effect relationships, methodological explanations",
        "probabilistic": "Emphasizes quantitative results, statistical significance, metrics",
        "abductive": "Values key findings, implications, theoretical explanations",
        "deductive": "Prioritizes mathematical formulations, logical proofs, algorithms", 
        "inductive": "Seeks empirical results, experimental evidence, observed patterns",
        "analogical": "Looks for related work, comparable approaches, structural similarities"
    }


# Export main classes and functions
__all__ = [
    'ChunkEmbeddings', 'QueryRelevanceScore', 'ChunkSelectionResult',
    'ChunkEmbeddingSystem', 'ChunkRankingSystem',
    'rank_and_select_chunks', 'get_reasoning_engine_descriptions'
]