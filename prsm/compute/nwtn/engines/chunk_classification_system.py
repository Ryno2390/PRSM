#!/usr/bin/env python3
"""
Semantic Chunk Classification System for Meta-Paper Generation
=============================================================

This module implements intelligent chunk type classification for research papers,
enabling the meta-paper generation approach where chunks from multiple papers
are ranked by query relevance and assembled into coherent research contexts.

Key Features:
- 15 semantic chunk types for comprehensive paper analysis
- Pattern-based classification with confidence scoring  
- Section-aware chunking with natural boundary detection
- Query-specific relevance scoring for meta-paper assembly
- Evidence strength and novelty assessment

Integrates with: complete_nwtn_pipeline_v4.py for enhanced candidate generation
"""

import re
import os
import time
import asyncio
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Semantic chunk types for research paper content"""
    # Core content types
    ABSTRACT = "abstract"                    # Paper summary/overview
    PROBLEM_STATEMENT = "problem_statement" # What problem is being solved
    METHODOLOGY = "methodology"             # How the research was conducted
    EXPERIMENTAL_SETUP = "experimental"     # Specific experimental conditions
    RESULTS_QUANTITATIVE = "results_quant"  # Numbers, statistics, measurements
    RESULTS_QUALITATIVE = "results_qual"    # Observations, patterns, insights
    KEY_FINDINGS = "key_findings"           # Main discoveries/conclusions
    THEORETICAL_FRAMEWORK = "theory"        # Conceptual models, hypotheses
    RELATED_WORK = "related_work"          # Background, prior research
    LIMITATIONS = "limitations"             # Study constraints, caveats
    FUTURE_WORK = "future_work"            # Recommendations, next steps
    IMPLICATIONS = "implications"           # Practical applications
    MATHEMATICAL_FORMULATION = "math"       # Equations, proofs, formulas
    ALGORITHM_DESCRIPTION = "algorithm"     # Pseudocode, procedures
    PERFORMANCE_METRICS = "metrics"         # Evaluation criteria, benchmarks


@dataclass
class SemanticChunk:
    """Individual semantic chunk with classification and metadata"""
    content: str
    chunk_type: ChunkType
    paper_id: str
    paper_title: str
    section_name: str
    confidence_score: float        # How confident are we in the chunk type?
    evidence_strength: float       # How well-supported are the claims?
    novelty_score: float          # How unique/novel is this information?
    citation_count: int           # Number of times this finding is cited
    recency_score: float          # How recent is this research?
    word_count: int
    start_position: int           # Character position in original document
    end_position: int
    key_concepts: List[str] = field(default_factory=list)
    quantitative_data: List[Dict] = field(default_factory=list)  # Extracted numbers
    author_credibility: float = 0.5  # Based on author h-index, institution, etc.
    chunk_hash: str = ""           # For deduplication
    
    def __post_init__(self):
        """Generate hash for deduplication"""
        content_for_hash = f"{self.content}:{self.chunk_type.value}"
        self.chunk_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:12]
    
    def is_valid(self) -> bool:
        """Check if chunk meets basic validity requirements"""
        return (
            len(self.content.strip()) >= 50 and  # Minimum meaningful content length
            0.0 <= self.confidence_score <= 1.0 and
            0.0 <= self.evidence_strength <= 1.0 and
            0.0 <= self.novelty_score <= 1.0 and
            self.word_count > 5
        )


@dataclass  
class ChunkClassificationResult:
    """Result of chunk classification process"""
    chunks: List[SemanticChunk]
    classification_time: float
    total_chunks_created: int
    valid_chunks: int
    paper_metadata: Dict[str, Any]
    classification_confidence: float  # Overall confidence in classification
    

class ChunkTypeClassifier:
    """Intelligent classification of text chunks into semantic types"""
    
    def __init__(self):
        self.section_patterns = self._initialize_section_patterns()
        self.content_patterns = self._initialize_content_patterns()
        self.quantitative_patterns = self._initialize_quantitative_patterns()
        self.concept_extractors = self._initialize_concept_extractors()
        
    def _initialize_section_patterns(self) -> Dict[ChunkType, List[str]]:
        """Initialize regex patterns for identifying chunk types by section headers"""
        return {
            ChunkType.ABSTRACT: [
                r'^\s*abstract\s*$',
                r'^\s*summary\s*$',
                r'^\s*overview\s*$'
            ],
            ChunkType.PROBLEM_STATEMENT: [
                r'^\s*(problem\s+statement|research\s+problem|motivation)\s*$',
                r'^\s*(introduction|background)\s*$',
                r'^\s*1\.?\s*introduction\s*$'
            ],
            ChunkType.METHODOLOGY: [
                r'^\s*(methodology|methods|approach|experimental\s+design)\s*$',
                r'^\s*\d+\.?\s*(methodology|methods|approach)\s*$',
                r'^\s*(materials\s+and\s+methods|experimental\s+procedure)\s*$'
            ],
            ChunkType.EXPERIMENTAL_SETUP: [
                r'^\s*(experimental\s+setup|setup|configuration|implementation)\s*$',
                r'^\s*(data\s+collection|data\s+gathering|experiment\s+design)\s*$',
                r'^\s*(evaluation\s+setup|experimental\s+conditions)\s*$'
            ],
            ChunkType.RESULTS_QUANTITATIVE: [
                r'^\s*(results|findings|outcomes)\s*$',
                r'^\s*\d+\.?\s*results\s*$',
                r'^\s*(experimental\s+results|evaluation\s+results)\s*$',
                r'^\s*(performance|benchmarks|metrics)\s*$'
            ],
            ChunkType.KEY_FINDINGS: [
                r'^\s*(key\s+findings|main\s+results|conclusions)\s*$',
                r'^\s*(discussion|analysis)\s*$',
                r'^\s*\d+\.?\s*(discussion|conclusions)\s*$'
            ],
            ChunkType.THEORETICAL_FRAMEWORK: [
                r'^\s*(theoretical\s+framework|theory|model)\s*$',
                r'^\s*(conceptual\s+model|framework)\s*$',
                r'^\s*(related\s+theory|background\s+theory)\s*$'
            ],
            ChunkType.RELATED_WORK: [
                r'^\s*(related\s+work|literature\s+review|prior\s+work)\s*$',
                r'^\s*\d+\.?\s*(related\s+work|background)\s*$',
                r'^\s*(previous\s+research|state\s+of\s+the\s+art)\s*$'
            ],
            ChunkType.LIMITATIONS: [
                r'^\s*(limitations|constraints|assumptions)\s*$',
                r'^\s*(threats\s+to\s+validity|study\s+limitations)\s*$'
            ],
            ChunkType.FUTURE_WORK: [
                r'^\s*(future\s+work|future\s+research|next\s+steps)\s*$',
                r'^\s*(recommendations|future\s+directions)\s*$'
            ],
            ChunkType.IMPLICATIONS: [
                r'^\s*(implications|applications|practical\s+significance)\s*$',
                r'^\s*(practical\s+applications|real.world\s+impact)\s*$'
            ]
        }
    
    def _initialize_content_patterns(self) -> Dict[ChunkType, List[str]]:
        """Initialize patterns for identifying chunk types by content characteristics"""
        return {
            ChunkType.ABSTRACT: [
                r'this\s+(paper|study|research|work)\s+(presents|proposes|introduces|investigates)',
                r'we\s+(present|propose|introduce|investigate|demonstrate|show)',
                r'(in\s+this\s+(paper|study|work))',
            ],
            ChunkType.PROBLEM_STATEMENT: [
                r'(the\s+problem|challenge|issue)\s+(is|lies|stems\s+from)',
                r'(existing|current)\s+(approaches|methods|solutions)\s+(suffer\s+from|have\s+limitations)',
                r'(however|unfortunately|despite)',
                r'(there\s+is\s+a\s+need|it\s+is\s+necessary|we\s+need)'
            ],
            ChunkType.METHODOLOGY: [
                r'(we\s+(use|employ|apply|adopt|utilize))',
                r'(the\s+(method|approach|technique|algorithm)\s+(is|involves))',
                r'(our\s+(approach|method|strategy))',
                r'(step\s+\d+|first|second|third|finally|next)'
            ],
            ChunkType.EXPERIMENTAL_SETUP: [
                r'(dataset|corpus|benchmark|evaluation\s+set)',
                r'(we\s+(evaluate|test|experiment|validate))',
                r'(experimental\s+(setup|configuration|conditions))',
                r'(baseline|comparison|control\s+group)'
            ],
            ChunkType.RESULTS_QUANTITATIVE: [
                r'(\d+\.?\d*%|\d+\.?\d*\s+(accuracy|precision|recall|f1))',
                r'(table\s+\d+|figure\s+\d+|\d+\.?\d*\s*(improvement|increase|decrease))',
                r'(statistically\s+significant|p\s*[<>=]\s*\d)',
                r'(outperform|better\s+than|superior\s+to)'
            ],
            ChunkType.KEY_FINDINGS: [
                r'(we\s+(find|found|observe|discovered|show|demonstrate))',
                r'(the\s+results\s+(show|indicate|suggest|reveal))',
                r'(our\s+(findings|results|experiments)\s+(show|indicate|suggest))',
                r'(this\s+(suggests|indicates|implies|demonstrates))'
            ],
            ChunkType.THEORETICAL_FRAMEWORK: [
                r'(theorem|lemma|proposition|hypothesis|conjecture)',
                r'(let\s+\w+\s+be|assume|suppose|given)',
                r'(theoretical|conceptual|formal)\s+(model|framework)',
                r'(definition|axiom|postulate)'
            ],
            ChunkType.LIMITATIONS: [
                r'(limitation|constraint|shortcoming|weakness)',
                r'(however|unfortunately|despite|nevertheless)',
                r'(cannot|unable\s+to|fails\s+to|does\s+not)',
                r'(future\s+work\s+could|improvement|enhancement)'
            ],
            ChunkType.MATHEMATICAL_FORMULATION: [
                r'(equation|formula|\$.*\$|\\begin\{|\\end\{)',
                r'(let\s+\w+\s*=|given\s+that|where\s+\w+\s+is)',
                r'(\\\w+|\^\{\w+\}|\\_\{\w+\})',  # LaTeX patterns
                r'(minimize|maximize|subject\s+to|optimization)'
            ],
            ChunkType.ALGORITHM_DESCRIPTION: [
                r'(algorithm\s+\d+|procedure|pseudocode)',
                r'(input:|output:|return|while|for\s+each|if\s+then)',
                r'(step\s+\d+:|begin|end|repeat|until)',
                r'(time\s+complexity|space\s+complexity|o\(\w+\))'
            ],
            ChunkType.PERFORMANCE_METRICS: [
                r'(accuracy|precision|recall|f1.score|auc|mae|rmse)',
                r'(benchmark|evaluation\s+metric|performance\s+measure)',
                r'(baseline|state.of.the.art|sota)',
                r'(runtime|execution\s+time|memory\s+usage)'
            ]
        }
    
    def _initialize_quantitative_patterns(self) -> List[str]:
        """Initialize patterns for extracting quantitative data"""
        return [
            r'(\d+\.?\d*)\s*%',  # Percentages
            r'(\d+\.?\d*)\s*(accuracy|precision|recall|f1)',  # Performance metrics
            r'p\s*[<>=]\s*(\d+\.?\d*)',  # P-values
            r'(\d+\.?\d*)\s*(seconds|minutes|hours|ms|μs)',  # Time measurements
            r'(\d+\.?\d*)\s*(mb|gb|kb|bytes)',  # Memory/size measurements
            r'(\d+\.?\d*)\s*(fold|times)\s+(faster|slower|better|worse)',  # Comparisons
            r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)',  # Statistical measurements
            r'n\s*=\s*(\d+)',  # Sample sizes
        ]
    
    def _simple_sentence_tokenize(self, text: str) -> List[str]:
        """Simple sentence tokenization without NLTK dependency"""
        # Split on common sentence endings, handling common abbreviations
        sentence_endings = r'[.!?]+\s+'
        
        # Split text by sentence endings
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only keep meaningful sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _initialize_concept_extractors(self) -> Dict[str, List[str]]:
        """Initialize patterns for extracting key concepts"""
        return {
            'machine_learning': [
                'neural network', 'deep learning', 'machine learning', 'artificial intelligence',
                'supervised learning', 'unsupervised learning', 'reinforcement learning',
                'convolutional', 'recurrent', 'transformer', 'attention', 'lstm', 'gru'
            ],
            'nlp': [
                'natural language processing', 'text mining', 'sentiment analysis',
                'named entity recognition', 'language model', 'tokenization',
                'parsing', 'semantic analysis', 'syntactic analysis'
            ],
            'computer_vision': [
                'computer vision', 'image processing', 'object detection',
                'image classification', 'segmentation', 'feature extraction',
                'optical character recognition', 'face recognition'
            ],
            'statistics': [
                'hypothesis test', 'confidence interval', 'regression',
                'correlation', 'anova', 'chi-square', 'bayesian',
                'frequentist', 'distribution', 'probability'
            ],
            'algorithms': [
                'algorithm', 'data structure', 'complexity', 'optimization',
                'search', 'sort', 'graph', 'tree', 'dynamic programming',
                'greedy', 'divide and conquer'
            ]
        }
    
    async def classify_chunks(self, 
                            text: str, 
                            paper_metadata: Dict[str, Any],
                            chunk_size: int = 400,
                            overlap: int = 50) -> ChunkClassificationResult:
        """
        Classify text content into semantic chunks
        
        Args:
            text: Full paper content
            paper_metadata: Paper metadata (title, authors, etc.)
            chunk_size: Target chunk size in words
            overlap: Overlap between chunks in words
            
        Returns:
            ChunkClassificationResult with classified chunks
        """
        start_time = time.time()
        
        # Step 1: Detect document structure and sections
        sections = await self._detect_document_sections(text)
        
        # Step 2: Create chunks respecting section boundaries
        raw_chunks = await self._create_semantic_chunks(text, sections, chunk_size, overlap)
        
        # Step 3: Classify each chunk
        classified_chunks = []
        for chunk_data in raw_chunks:
            chunk = await self._classify_single_chunk(chunk_data, paper_metadata)
            if chunk and chunk.is_valid():
                classified_chunks.append(chunk)
        
        # Step 4: Post-process and enhance chunks
        enhanced_chunks = await self._enhance_chunks(classified_chunks)
        
        classification_time = time.time() - start_time
        
        # Calculate overall classification confidence
        avg_confidence = sum(chunk.confidence_score for chunk in enhanced_chunks) / max(len(enhanced_chunks), 1)
        
        return ChunkClassificationResult(
            chunks=enhanced_chunks,
            classification_time=classification_time,
            total_chunks_created=len(raw_chunks),
            valid_chunks=len(enhanced_chunks),
            paper_metadata=paper_metadata,
            classification_confidence=avg_confidence
        )
    
    async def _detect_document_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect document sections and their boundaries"""
        sections = []
        lines = text.split('\n')
        
        current_section = None
        section_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            section_type = self._identify_section_type(line)
            
            if section_type:
                # Close previous section
                if current_section:
                    sections.append({
                        'type': current_section,
                        'title': current_section.value.replace('_', ' ').title(),
                        'start_line': section_start,
                        'end_line': i - 1,
                        'content': '\n'.join(lines[section_start:i])
                    })
                
                # Start new section
                current_section = section_type
                section_start = i
        
        # Close final section
        if current_section:
            sections.append({
                'type': current_section,
                'title': current_section.value.replace('_', ' ').title(), 
                'start_line': section_start,
                'end_line': len(lines) - 1,
                'content': '\n'.join(lines[section_start:])
            })
        
        # If no sections detected, treat entire text as general content
        if not sections:
            sections.append({
                'type': ChunkType.RELATED_WORK,  # Default type
                'title': 'General Content',
                'start_line': 0,
                'end_line': len(lines) - 1,
                'content': text
            })
        
        return sections
    
    def _identify_section_type(self, line: str) -> Optional[ChunkType]:
        """Identify section type from header line"""
        line_lower = line.lower().strip()
        
        # Remove common section numbering patterns
        clean_line = re.sub(r'^\d+\.?\d*\s*', '', line_lower)
        clean_line = re.sub(r'^[ivx]+\.?\s*', '', clean_line)  # Roman numerals
        
        # Check against section patterns
        for chunk_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, clean_line, re.IGNORECASE):
                    return chunk_type
        
        return None
    
    async def _create_semantic_chunks(self, 
                                    text: str, 
                                    sections: List[Dict[str, Any]], 
                                    chunk_size: int, 
                                    overlap: int) -> List[Dict[str, Any]]:
        """Create chunks respecting semantic boundaries"""
        chunks = []
        
        for section in sections:
            section_content = section['content']
            section_type = section['type']
            
            # Split section into sentences for better boundaries
            sentences = self._simple_sentence_tokenize(section_content)
            
            # Group sentences into chunks of target size
            current_chunk = []
            current_word_count = 0
            chunk_start_pos = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # If adding this sentence would exceed chunk size, finalize current chunk
                if current_word_count > 0 and current_word_count + sentence_words > chunk_size:
                    chunk_content = ' '.join(current_chunk)
                    chunks.append({
                        'content': chunk_content,
                        'section_type': section_type,
                        'section_name': section['title'],
                        'word_count': current_word_count,
                        'start_position': chunk_start_pos,
                        'end_position': chunk_start_pos + len(chunk_content)
                    })
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > 1:
                        overlap_sentences = current_chunk[-1:]  # Keep last sentence for overlap
                        current_chunk = overlap_sentences
                        current_word_count = sum(len(s.split()) for s in overlap_sentences)
                        chunk_start_pos += len(' '.join(current_chunk[:-1])) + 1
                    else:
                        current_chunk = []
                        current_word_count = 0
                        chunk_start_pos += len(chunk_content) + 1
                
                current_chunk.append(sentence)
                current_word_count += sentence_words
            
            # Finalize last chunk in section
            if current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'section_type': section_type,
                    'section_name': section['title'],
                    'word_count': current_word_count,
                    'start_position': chunk_start_pos,
                    'end_position': chunk_start_pos + len(chunk_content)
                })
        
        return chunks
    
    async def _classify_single_chunk(self, 
                                   chunk_data: Dict[str, Any], 
                                   paper_metadata: Dict[str, Any]) -> Optional[SemanticChunk]:
        """Classify a single chunk based on content and context"""
        content = chunk_data['content'].strip()
        if len(content) < 50:  # Skip very short chunks
            return None
        
        # Start with section-based classification
        base_type = chunk_data['section_type']
        confidence = 0.6  # Base confidence from section classification
        
        # Refine classification based on content patterns
        content_scores = {}
        for chunk_type, patterns in self.content_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches * 0.1  # Each pattern match adds to score
            content_scores[chunk_type] = score
        
        # Find best content-based classification
        if content_scores:
            best_content_type = max(content_scores, key=content_scores.get)
            best_content_score = content_scores[best_content_type]
            
            # Combine section and content classification
            if best_content_score > 0.3 and best_content_type != base_type:
                # Content classification overrides section if confidence is high
                base_type = best_content_type
                confidence = min(0.9, 0.6 + best_content_score)
        
        # Extract quantitative data
        quantitative_data = self._extract_quantitative_data(content)
        
        # Adjust classification for quantitative chunks
        if quantitative_data and base_type not in [ChunkType.RESULTS_QUANTITATIVE, ChunkType.PERFORMANCE_METRICS]:
            if any('accuracy' in str(item) or 'precision' in str(item) for item in quantitative_data):
                base_type = ChunkType.PERFORMANCE_METRICS
                confidence = min(0.95, confidence + 0.2)
            elif len(quantitative_data) > 2:
                base_type = ChunkType.RESULTS_QUANTITATIVE
                confidence = min(0.9, confidence + 0.15)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(content)
        
        # Calculate evidence strength based on quantitative data and citations
        evidence_strength = self._calculate_evidence_strength(content, quantitative_data)
        
        # Calculate novelty score based on language patterns
        novelty_score = self._calculate_novelty_score(content)
        
        # Estimate recency and credibility (would be enhanced with real metadata)
        recency_score = 0.7  # Default - would use publication date
        author_credibility = 0.6  # Default - would use author metrics
        
        return SemanticChunk(
            content=content,
            chunk_type=base_type,
            paper_id=paper_metadata.get('paper_id', 'unknown'),
            paper_title=paper_metadata.get('title', 'Unknown Paper'),
            section_name=chunk_data['section_name'],
            confidence_score=confidence,
            evidence_strength=evidence_strength,
            novelty_score=novelty_score,
            citation_count=0,  # Would be populated from citation data
            recency_score=recency_score,
            word_count=chunk_data['word_count'],
            start_position=chunk_data['start_position'],
            end_position=chunk_data['end_position'],
            key_concepts=key_concepts,
            quantitative_data=quantitative_data,
            author_credibility=author_credibility
        )
    
    def _extract_quantitative_data(self, content: str) -> List[Dict[str, Any]]:
        """Extract quantitative data from chunk content"""
        quantitative_data = []
        
        for pattern in self.quantitative_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(quantitative_data) >= 10:  # Limit per chunk
                    break
                    
                data_item = {
                    'value': match.group(1) if match.groups() else match.group(0),
                    'context': match.group(0),
                    'type': self._classify_quantitative_type(match.group(0)),
                    'position': match.start()
                }
                quantitative_data.append(data_item)
        
        return quantitative_data
    
    def _classify_quantitative_type(self, match_text: str) -> str:
        """Classify type of quantitative data"""
        match_lower = match_text.lower()
        
        if any(metric in match_lower for metric in ['accuracy', 'precision', 'recall', 'f1']):
            return 'performance_metric'
        elif '%' in match_text:
            return 'percentage'
        elif any(time_unit in match_lower for time_unit in ['second', 'minute', 'hour', 'ms']):
            return 'time_measurement'
        elif any(size_unit in match_lower for size_unit in ['mb', 'gb', 'kb', 'byte']):
            return 'size_measurement'
        elif 'p' in match_lower and any(op in match_lower for op in ['<', '>', '=']):
            return 'statistical_significance'
        else:
            return 'general_numeric'
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from chunk content"""
        concepts = []
        content_lower = content.lower()
        
        # Extract concepts based on predefined categories
        for category, concept_list in self.concept_extractors.items():
            for concept in concept_list:
                if concept in content_lower:
                    concepts.append(concept)
        
        # Extract capitalized terms (likely proper nouns/technical terms)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        for term in capitalized_terms[:5]:  # Limit to 5 per chunk
            if len(term) > 3 and term not in concepts:
                concepts.append(term)
        
        # Extract technical acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', content)
        for acronym in acronyms[:3]:  # Limit to 3 per chunk
            if acronym not in concepts:
                concepts.append(acronym)
        
        return concepts[:15]  # Limit total concepts per chunk
    
    def _calculate_evidence_strength(self, content: str, quantitative_data: List[Dict]) -> float:
        """Calculate evidence strength based on content characteristics"""
        strength = 0.5  # Base strength
        
        # Boost for quantitative data
        strength += min(0.3, len(quantitative_data) * 0.1)
        
        # Boost for statistical significance indicators
        if any('significant' in item.get('context', '').lower() for item in quantitative_data):
            strength += 0.2
        
        # Boost for comparative language
        comparative_terms = ['better than', 'outperforms', 'superior to', 'improvement', 'significant']
        for term in comparative_terms:
            if term in content.lower():
                strength += 0.05
        
        # Boost for experimental validation
        validation_terms = ['experiment', 'validation', 'evaluation', 'benchmark', 'test']
        for term in validation_terms:
            if term in content.lower():
                strength += 0.03
        
        return min(1.0, strength)
    
    def _calculate_novelty_score(self, content: str) -> float:
        """Calculate novelty score based on language patterns"""
        novelty = 0.5  # Base novelty
        
        # Boost for novel contribution language
        novel_terms = ['novel', 'new', 'first', 'innovative', 'breakthrough', 'pioneer']
        for term in novel_terms:
            if term in content.lower():
                novelty += 0.08
        
        # Boost for problem-solving language
        solution_terms = ['solve', 'address', 'overcome', 'tackle', 'solution']
        for term in solution_terms:
            if term in content.lower():
                novelty += 0.05
        
        # Boost for limitation-addressing language  
        limitation_terms = ['unlike previous', 'in contrast', 'however', 'overcome']
        for term in limitation_terms:
            if term in content.lower():
                novelty += 0.04
        
        return min(1.0, novelty)
    
    async def _enhance_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Post-process chunks for quality and consistency"""
        enhanced_chunks = []
        
        # Remove near-duplicates based on content similarity
        unique_chunks = self._remove_duplicate_chunks(chunks)
        
        # Enhance chunk relationships and context
        for chunk in unique_chunks:
            # Enhance mathematical chunks
            if chunk.chunk_type == ChunkType.MATHEMATICAL_FORMULATION:
                chunk = self._enhance_mathematical_chunk(chunk)
            
            # Enhance algorithm chunks
            elif chunk.chunk_type == ChunkType.ALGORITHM_DESCRIPTION:
                chunk = self._enhance_algorithm_chunk(chunk)
            
            # Enhance results chunks
            elif chunk.chunk_type in [ChunkType.RESULTS_QUANTITATIVE, ChunkType.PERFORMANCE_METRICS]:
                chunk = self._enhance_results_chunk(chunk)
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _remove_duplicate_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Remove duplicate chunks based on content similarity"""
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            if chunk.chunk_hash not in seen_hashes:
                unique_chunks.append(chunk)
                seen_hashes.add(chunk.chunk_hash)
        
        return unique_chunks
    
    def _enhance_mathematical_chunk(self, chunk: SemanticChunk) -> SemanticChunk:
        """Enhance mathematical formulation chunks"""
        # Extract equations and formulas
        equation_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\\\[[^\]]+\\\]',  # LaTeX display math
            r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}',  # LaTeX environments
            r'[a-zA-Z]\s*=\s*[^.!?]*[.!?]'  # Simple equations
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.findall(pattern, chunk.content, re.DOTALL)
            equations.extend(matches)
        
        # Add equations to quantitative data
        for eq in equations[:5]:  # Limit to 5 equations
            chunk.quantitative_data.append({
                'type': 'equation',
                'value': eq,
                'context': 'mathematical_formulation'
            })
        
        # Boost confidence if mathematical content is found
        if equations:
            chunk.confidence_score = min(0.95, chunk.confidence_score + 0.1)
        
        return chunk
    
    def _enhance_algorithm_chunk(self, chunk: SemanticChunk) -> SemanticChunk:
        """Enhance algorithm description chunks"""
        # Look for algorithm keywords
        algorithm_keywords = [
            'input:', 'output:', 'return', 'while', 'for', 'if', 'then', 'else',
            'begin', 'end', 'repeat', 'until', 'step', 'procedure', 'algorithm'
        ]
        
        keyword_count = sum(1 for keyword in algorithm_keywords if keyword in chunk.content.lower())
        
        # Boost confidence based on algorithm structure
        if keyword_count >= 3:
            chunk.confidence_score = min(0.95, chunk.confidence_score + 0.15)
        elif keyword_count >= 1:
            chunk.confidence_score = min(0.85, chunk.confidence_score + 0.1)
        
        return chunk
    
    def _enhance_results_chunk(self, chunk: SemanticChunk) -> SemanticChunk:
        """Enhance results and performance chunks"""
        # Boost evidence strength if many quantitative measures
        if len(chunk.quantitative_data) >= 3:
            chunk.evidence_strength = min(1.0, chunk.evidence_strength + 0.2)
        
        # Look for significance indicators
        significance_indicators = ['significant', 'p <', 'p>', 'confidence', 'statistically']
        significance_count = sum(1 for indicator in significance_indicators if indicator in chunk.content.lower())
        
        if significance_count > 0:
            chunk.evidence_strength = min(1.0, chunk.evidence_strength + 0.1)
            
        return chunk


# Integration helper functions
async def classify_paper_chunks(paper_content: str, 
                              paper_metadata: Dict[str, Any],
                              chunk_size: int = 400,
                              overlap: int = 50) -> ChunkClassificationResult:
    """
    Main interface for classifying paper content into semantic chunks
    
    Args:
        paper_content: Full text content of the paper
        paper_metadata: Metadata about the paper (title, authors, etc.)
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words
        
    Returns:
        ChunkClassificationResult with classified chunks
    """
    classifier = ChunkTypeClassifier()
    return await classifier.classify_chunks(paper_content, paper_metadata, chunk_size, overlap)


def get_chunk_type_descriptions() -> Dict[ChunkType, str]:
    """Get human-readable descriptions of chunk types"""
    return {
        ChunkType.ABSTRACT: "Paper summary and overview",
        ChunkType.PROBLEM_STATEMENT: "Problem definition and motivation",
        ChunkType.METHODOLOGY: "Research methods and approaches",
        ChunkType.EXPERIMENTAL_SETUP: "Experimental design and conditions",
        ChunkType.RESULTS_QUANTITATIVE: "Numerical results and measurements",
        ChunkType.RESULTS_QUALITATIVE: "Observational results and patterns",
        ChunkType.KEY_FINDINGS: "Main discoveries and conclusions",
        ChunkType.THEORETICAL_FRAMEWORK: "Theoretical models and concepts",
        ChunkType.RELATED_WORK: "Background and prior research",
        ChunkType.LIMITATIONS: "Study constraints and limitations",
        ChunkType.FUTURE_WORK: "Future research directions",
        ChunkType.IMPLICATIONS: "Practical applications and significance",
        ChunkType.MATHEMATICAL_FORMULATION: "Mathematical equations and proofs",
        ChunkType.ALGORITHM_DESCRIPTION: "Algorithms and procedures",
        ChunkType.PERFORMANCE_METRICS: "Evaluation metrics and benchmarks"
    }


# Export main classes and functions
__all__ = [
    'ChunkType', 'SemanticChunk', 'ChunkClassificationResult', 'ChunkTypeClassifier',
    'classify_paper_chunks', 'get_chunk_type_descriptions'
]