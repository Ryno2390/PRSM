"""
Information Space Content Analyzer

Analyzes IPFS content to extract semantic relationships and build the Information Space graph.
Integrates with existing PRSM systems for comprehensive knowledge mapping.
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from decimal import Decimal
import hashlib

# NLP and ML imports
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("NLP libraries not available. Content analysis will be limited.")

from .models import (
    InfoNode, InfoEdge, ResearchOpportunity, ContentAnalysis,
    NodeType, EdgeType, OpportunityType, InformationGraph
)

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzes IPFS content to extract metadata and relationships."""
    
    def __init__(self, ipfs_client=None):
        self.ipfs_client = ipfs_client
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Analysis cache
        self.analysis_cache: Dict[str, ContentAnalysis] = {}
        
        # Content type processors
        self.processors = {
            'text/plain': self._process_text,
            'application/pdf': self._process_pdf,
            'application/json': self._process_json,
            'text/markdown': self._process_markdown,
            'text/html': self._process_html
        }
        
    async def analyze_content(self, ipfs_hash: str, content_type: str = None) -> Optional[ContentAnalysis]:
        """Analyze content from IPFS and extract Information Space data."""
        
        # Check cache first
        if ipfs_hash in self.analysis_cache:
            return self.analysis_cache[ipfs_hash]
            
        if not self.ipfs_client:
            logger.error("IPFS client not available for content analysis")
            return None
            
        try:
            start_time = datetime.utcnow()
            
            # Fetch content from IPFS
            content = await self.ipfs_client.get_content(ipfs_hash)
            if not content:
                logger.warning(f"Could not fetch content from IPFS: {ipfs_hash}")
                return None
                
            # Detect content type if not provided
            if not content_type:
                content_type = self._detect_content_type(content)
                
            # Process content based on type
            processor = self.processors.get(content_type, self._process_text)
            analysis = await processor(content, ipfs_hash)
            
            # Add processing metadata
            analysis.processing_time = (datetime.utcnow() - start_time).total_seconds()
            analysis.analyzed_at = datetime.utcnow()
            
            # Enhance with semantic analysis
            if NLP_AVAILABLE:
                analysis = await self.semantic_analyzer.enhance_analysis(analysis, content)
                
            # Cache the analysis
            self.analysis_cache[ipfs_hash] = analysis
            
            logger.info(f"Successfully analyzed content: {ipfs_hash}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content {ipfs_hash}: {e}")
            return None
            
    async def _process_text(self, content: str, ipfs_hash: str) -> ContentAnalysis:
        """Process plain text content."""
        
        analysis = ContentAnalysis(
            content_id=self._generate_content_id(content),
            ipfs_hash=ipfs_hash
        )
        
        # Extract basic metadata
        lines = content.split('\n')
        analysis.title = lines[0].strip() if lines else "Untitled"
        
        # Extract keywords using simple frequency analysis
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Top keywords
        analysis.keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        analysis.keywords = [word for word, _ in analysis.keywords]
        
        # Extract potential citations (simple heuristic)
        citation_pattern = r'\b(?:et al\.|doi:|arxiv:|http[s]?://)'
        analysis.cited_works = re.findall(citation_pattern, content, re.IGNORECASE)
        
        return analysis
        
    async def _process_pdf(self, content: bytes, ipfs_hash: str) -> ContentAnalysis:
        """Process PDF content."""
        
        # For now, treat as text (would need PDF parsing library)
        try:
            text_content = content.decode('utf-8', errors='ignore')
            return await self._process_text(text_content, ipfs_hash)
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ContentAnalysis(
                content_id=self._generate_content_id(str(content)),
                ipfs_hash=ipfs_hash,
                title="PDF Document"
            )
            
    async def _process_json(self, content: str, ipfs_hash: str) -> ContentAnalysis:
        """Process JSON content."""
        
        try:
            data = json.loads(content)
            
            analysis = ContentAnalysis(
                content_id=self._generate_content_id(content),
                ipfs_hash=ipfs_hash
            )
            
            # Extract metadata from JSON structure
            if isinstance(data, dict):
                analysis.title = data.get('title', data.get('name', 'JSON Document'))
                analysis.abstract = data.get('abstract', data.get('description', ''))
                analysis.authors = data.get('authors', data.get('contributors', []))
                analysis.keywords = data.get('keywords', data.get('tags', []))
                
            return analysis
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON content: {ipfs_hash}")
            return ContentAnalysis(
                content_id=self._generate_content_id(content),
                ipfs_hash=ipfs_hash,
                title="Invalid JSON"
            )
            
    async def _process_markdown(self, content: str, ipfs_hash: str) -> ContentAnalysis:
        """Process Markdown content."""
        
        analysis = ContentAnalysis(
            content_id=self._generate_content_id(content),
            ipfs_hash=ipfs_hash
        )
        
        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        analysis.title = title_match.group(1) if title_match else "Markdown Document"
        
        # Extract links as potential citations
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)
        analysis.cited_works = [url for _, url in links if url.startswith('http')]
        
        # Extract keywords from content
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        analysis.keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        analysis.keywords = [word for word, _ in analysis.keywords]
        
        return analysis
        
    async def _process_html(self, content: str, ipfs_hash: str) -> ContentAnalysis:
        """Process HTML content."""
        
        # Simple HTML processing (would benefit from BeautifulSoup)
        analysis = ContentAnalysis(
            content_id=self._generate_content_id(content),
            ipfs_hash=ipfs_hash
        )
        
        # Extract title from <title> tag
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        analysis.title = title_match.group(1) if title_match else "HTML Document"
        
        # Extract links
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
        links = re.findall(link_pattern, content, re.IGNORECASE)
        analysis.cited_works = [url for url in links if url.startswith('http')]
        
        # Remove HTML tags for text analysis
        text_content = re.sub(r'<[^>]+>', ' ', content)
        text_analysis = await self._process_text(text_content, ipfs_hash)
        analysis.keywords = text_analysis.keywords
        
        return analysis
        
    def _detect_content_type(self, content: Any) -> str:
        """Detect content type from content."""
        
        if isinstance(content, bytes):
            # Check for PDF magic bytes
            if content.startswith(b'%PDF'):
                return 'application/pdf'
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                return 'application/octet-stream'
                
        if isinstance(content, str):
            content_lower = content.lower().strip()
            
            # Check for JSON
            if content_lower.startswith('{') or content_lower.startswith('['):
                try:
                    json.loads(content)
                    return 'application/json'
                except json.JSONDecodeError:
                    pass
                    
            # Check for HTML
            if '<html' in content_lower or '<!doctype html' in content_lower:
                return 'text/html'
                
            # Check for Markdown
            if re.search(r'^#+\s', content, re.MULTILINE) or '```' in content:
                return 'text/markdown'
                
        return 'text/plain'
        
    def _generate_content_id(self, content: str) -> str:
        """Generate unique content ID."""
        return hashlib.sha256(str(content).encode()).hexdigest()[:16]


class SemanticAnalyzer:
    """Advanced semantic analysis using NLP models."""
    
    def __init__(self):
        self.nlp = None
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        
        if NLP_AVAILABLE:
            self._initialize_models()
            
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            
        try:
            # Load sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    async def enhance_analysis(self, analysis: ContentAnalysis, content: str) -> ContentAnalysis:
        """Enhance analysis with semantic information."""
        
        if not NLP_AVAILABLE or not self.nlp:
            return analysis
            
        try:
            # Process with spaCy
            doc = self.nlp(content[:1000000])  # Limit to 1M chars
            
            # Extract named entities
            entities = []
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART']:
                    entities.append(ent.text)
                    
            # Update analysis with entities
            if entities:
                analysis.concepts.extend(entities[:20])
                
            # Extract key phrases
            key_phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:
                    key_phrases.append(chunk.text)
                    
            analysis.keywords.extend(key_phrases[:10])
            
            # Generate embeddings
            if self.sentence_transformer:
                embeddings = self.sentence_transformer.encode([content[:5000]])
                analysis.embeddings = embeddings[0].tolist()
                
            # Calculate novelty score (simple heuristic)
            unique_concepts = len(set(analysis.concepts))
            total_concepts = len(analysis.concepts)
            analysis.novelty_score = unique_concepts / max(total_concepts, 1)
            
            # Calculate quality score based on various factors
            quality_factors = [
                len(analysis.title) > 0,
                len(analysis.abstract) > 0,
                len(analysis.keywords) > 5,
                len(analysis.concepts) > 3,
                analysis.citation_count > 0
            ]
            analysis.quality_score = sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            
        return analysis
        
    def calculate_semantic_similarity(self, analysis1: ContentAnalysis, analysis2: ContentAnalysis) -> float:
        """Calculate semantic similarity between two content analyses."""
        
        if not self.sentence_transformer or not analysis1.embeddings or not analysis2.embeddings:
            return 0.0
            
        try:
            emb1 = np.array(analysis1.embeddings).reshape(1, -1)
            emb2 = np.array(analysis2.embeddings).reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
            
    def extract_topics(self, analyses: List[ContentAnalysis], n_topics: int = 10) -> Dict[str, List[str]]:
        """Extract topics from multiple content analyses."""
        
        if not analyses:
            return {}
            
        try:
            # Combine all text content
            documents = []
            for analysis in analyses:
                text = f"{analysis.title} {analysis.abstract} {' '.join(analysis.keywords)}"
                documents.append(text)
                
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Cluster documents
            kmeans = KMeans(n_clusters=min(n_topics, len(documents)), random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Extract topics
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            topics = {}
            
            for i in range(min(n_topics, len(set(clusters)))):
                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                topics[f"topic_{i}"] = top_terms
                
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return {}


class GraphBuilder:
    """Builds Information Space graph from content analyses."""
    
    def __init__(self, content_analyzer: ContentAnalyzer):
        self.content_analyzer = content_analyzer
        self.similarity_threshold = 0.7
        
    async def build_graph_from_analyses(self, analyses: List[ContentAnalysis]) -> InformationGraph:
        """Build an Information Space graph from content analyses."""
        
        graph = InformationGraph()
        
        # Create nodes from analyses
        for analysis in analyses:
            node = self._create_node_from_analysis(analysis)
            graph.add_node(node)
            
        # Create edges based on similarities and relationships
        await self._create_edges(graph, analyses)
        
        # Generate research opportunities
        opportunities = await self._generate_opportunities(graph, analyses)
        for opportunity in opportunities:
            graph.add_opportunity(opportunity)
            
        # Update graph metrics
        graph.update_node_metrics()
        
        return graph
        
    def _create_node_from_analysis(self, analysis: ContentAnalysis) -> InfoNode:
        """Create an InfoNode from ContentAnalysis."""
        
        # Determine node type based on content
        node_type = NodeType.DOCUMENT
        if any(keyword in analysis.title.lower() for keyword in ['dataset', 'data']):
            node_type = NodeType.DATASET
        elif any(keyword in analysis.title.lower() for keyword in ['model', 'algorithm']):
            node_type = NodeType.MODEL
        elif analysis.authors:
            node_type = NodeType.RESEARCH_AREA
            
        # Calculate opportunity score
        opportunity_score = (
            analysis.novelty_score * 0.3 +
            analysis.quality_score * 0.3 +
            min(analysis.citation_count / 100, 1.0) * 0.2 +
            min(len(analysis.concepts) / 20, 1.0) * 0.2
        )
        
        return InfoNode(
            id=analysis.content_id,
            label=analysis.title,
            node_type=node_type,
            description=analysis.abstract,
            tags=set(analysis.keywords),
            ipfs_hash=analysis.ipfs_hash,
            opportunity_score=opportunity_score,
            research_activity=min(analysis.citation_count / 50, 1.0),
            metadata={
                'authors': analysis.authors,
                'concepts': analysis.concepts,
                'topics': analysis.topics,
                'citation_count': analysis.citation_count,
                'novelty_score': analysis.novelty_score,
                'quality_score': analysis.quality_score
            }
        )
        
    async def _create_edges(self, graph: InformationGraph, analyses: List[ContentAnalysis]) -> None:
        """Create edges between nodes based on content similarity and relationships."""
        
        # Create similarity edges
        for i, analysis1 in enumerate(analyses):
            for j, analysis2 in enumerate(analyses[i+1:], i+1):
                similarity = self.content_analyzer.semantic_analyzer.calculate_semantic_similarity(
                    analysis1, analysis2
                )
                
                if similarity > self.similarity_threshold:
                    edge = InfoEdge(
                        source=analysis1.content_id,
                        target=analysis2.content_id,
                        edge_type=EdgeType.SEMANTIC_SIMILARITY,
                        weight=similarity,
                        confidence=similarity,
                        description=f"Semantic similarity: {similarity:.2f}"
                    )
                    graph.add_edge(edge)
                    
        # Create citation edges
        for analysis in analyses:
            for cited_work in analysis.cited_works:
                # Find matching analyses (simple heuristic)
                for other_analysis in analyses:
                    if cited_work in other_analysis.title or cited_work in other_analysis.abstract:
                        edge = InfoEdge(
                            source=analysis.content_id,
                            target=other_analysis.content_id,
                            edge_type=EdgeType.CITATION,
                            weight=1.0,
                            confidence=0.8,
                            description="Citation relationship"
                        )
                        graph.add_edge(edge)
                        
    async def _generate_opportunities(self, graph: InformationGraph, analyses: List[ContentAnalysis]) -> List[ResearchOpportunity]:
        """Generate research opportunities from graph analysis."""
        
        opportunities = []
        
        # Find high-opportunity nodes
        high_opportunity_nodes = [
            node for node in graph.nodes.values()
            if node.opportunity_score > 0.7
        ]
        
        # Generate collaboration opportunities
        for node in high_opportunity_nodes:
            neighbors = graph.get_neighbors(node.id)
            if len(neighbors) > 2:
                opportunity = ResearchOpportunity(
                    title=f"Collaboration Opportunity: {node.label}",
                    description=f"High-potential collaboration around {node.label} with {len(neighbors)} connected research areas",
                    opportunity_type=OpportunityType.COLLABORATION,
                    confidence=node.opportunity_score,
                    impact_score=node.influence_score,
                    feasibility_score=0.8,
                    research_areas=[node.id] + neighbors[:3],
                    estimated_value=Decimal('1000') * Decimal(str(node.opportunity_score)),
                    suggested_timeline="6-12 months"
                )
                opportunities.append(opportunity)
                
        # Find knowledge gaps (isolated high-quality nodes)
        for node in graph.nodes.values():
            if node.metadata.get('quality_score', 0) > 0.8 and len(graph.get_neighbors(node.id)) < 2:
                opportunity = ResearchOpportunity(
                    title=f"Knowledge Gap: {node.label}",
                    description=f"High-quality research area with limited connections - opportunity for expansion",
                    opportunity_type=OpportunityType.KNOWLEDGE_GAP,
                    confidence=0.7,
                    impact_score=node.metadata.get('quality_score', 0.5),
                    feasibility_score=0.6,
                    research_areas=[node.id],
                    estimated_value=Decimal('500'),
                    suggested_timeline="3-6 months"
                )
                opportunities.append(opportunity)
                
        return opportunities[:10]  # Limit to top 10 opportunities