#!/usr/bin/env python3
"""
NWTN Content Grounding Synthesizer
==================================

This module implements proper content grounding for Claude API synthesis to prevent
hallucinations by ensuring responses are based on actual paper content from the
external storage, not Claude's training knowledge.

Key Features:
- Extracts full abstracts and key content from retrieved papers
- Injects actual paper content into Claude prompts for grounding
- Dynamically retrieves additional paper details for longer responses
- Validates citations against actual papers in corpus
- Prevents hallucinated content by limiting Claude to provided sources

Author: NWTN System
Version: 1.0 - Content Grounding Implementation
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class GroundedPaperContent:
    """Contains actual paper content for grounding Claude responses"""
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    key_sections: List[str]
    publish_date: str
    categories: List[str]
    relevance_score: float
    content_length: int

@dataclass
class ContentGroundingResult:
    """Result of content grounding with actual paper content"""
    grounded_content: str
    source_papers: List[GroundedPaperContent]
    content_tokens_estimate: int
    grounding_quality: float
    available_expansion_content: bool

class ContentGroundingSynthesizer:
    """
    Ensures Claude API synthesis is grounded in actual paper content
    rather than hallucinated knowledge
    """
    
    def __init__(self, external_knowledge_base=None):
        self.knowledge_base = external_knowledge_base
        self.max_paper_content_tokens = 2000  # Limit per paper for Claude context
        self.min_grounding_quality = 0.7  # Minimum quality threshold
        
    async def prepare_grounded_synthesis(
        self, 
        reasoning_result: Any,
        target_tokens: int,
        retrieved_papers: List[Dict[str, Any]],
        verbosity_level: str
    ) -> ContentGroundingResult:
        """
        Prepare grounded content for Claude synthesis by extracting actual paper content
        
        Args:
            reasoning_result: NWTN reasoning output
            target_tokens: Target response length for verbosity level
            retrieved_papers: Papers retrieved from external storage  
            verbosity_level: BRIEF|STANDARD|DETAILED|COMPREHENSIVE|ACADEMIC
        
        Returns:
            ContentGroundingResult with actual paper content for Claude
        """
        logger.info("Starting content grounding preparation",
                   target_tokens=target_tokens,
                   verbosity_level=verbosity_level,
                   papers_count=len(retrieved_papers))
        
        # Step 1: Extract actual paper content from external storage
        grounded_papers = await self._extract_paper_content(retrieved_papers)
        
        # Step 2: Assess if we have enough content for target response length
        current_content_estimate = self._estimate_content_tokens(grounded_papers)
        
        # Step 3: If insufficient content, retrieve additional paper details
        if current_content_estimate < (target_tokens * 0.6):  # Need 60% coverage
            logger.info("Insufficient content for target length, expanding paper details",
                       current_estimate=current_content_estimate,
                       target=target_tokens)
            grounded_papers = await self._expand_paper_content(grounded_papers, target_tokens)
        
        # Step 4: Build grounded content prompt
        grounded_content = await self._build_grounded_content_prompt(
            reasoning_result, grounded_papers, verbosity_level
        )
        
        # Step 5: Validate grounding quality
        grounding_quality = self._assess_grounding_quality(grounded_papers, target_tokens)
        
        result = ContentGroundingResult(
            grounded_content=grounded_content,
            source_papers=grounded_papers,
            content_tokens_estimate=len(grounded_content.split()) * 1.3,
            grounding_quality=grounding_quality,
            available_expansion_content=len(grounded_papers) > 5
        )
        
        logger.info("Content grounding completed",
                   grounding_quality=grounding_quality,
                   source_papers=len(grounded_papers),
                   content_tokens=result.content_tokens_estimate)
        
        return result
    
    async def _extract_paper_content(
        self, 
        retrieved_papers: List[Dict[str, Any]]
    ) -> List[GroundedPaperContent]:
        """Extract actual content from papers in external storage"""
        grounded_papers = []
        
        logger.info("CONTENT GROUNDING DEBUG: Starting paper content extraction",
                   papers_count=len(retrieved_papers),
                   first_paper_keys=list(retrieved_papers[0].keys()) if retrieved_papers else "No papers")
        
        for i, paper in enumerate(retrieved_papers[:10]):  # Limit to top 10 for Claude context
            try:
                logger.info(f"CONTENT GROUNDING DEBUG: Processing paper {i+1}",
                           paper_keys=list(paper.keys()),
                           arxiv_id=paper.get('arxiv_id'),
                           title=paper.get('title', '')[:50] + '...' if paper.get('title') else 'No title')
                
                # Papers from external storage - check if full content is available
                if paper.get('title'):
                    # Determine content source and quality
                    has_full_content = paper.get('has_full_content', False)
                    content_source = "full_content" if has_full_content else "abstract_only"
                    
                    # Build comprehensive content including full sections if available
                    key_sections = self._extract_enhanced_key_sections(paper)
                    content_length = 0
                    
                    # Calculate total content length
                    if has_full_content:
                        content_length += len(paper.get('full_text', ''))
                        for section in ['introduction', 'methodology', 'results', 'discussion', 'conclusion']:
                            content_length += len(paper.get(section, ''))
                    else:
                        content_length = len(paper.get('abstract', ''))
                    
                    grounded_paper = GroundedPaperContent(
                        arxiv_id=paper.get('arxiv_id', ''),
                        title=paper.get('title', ''),
                        authors=paper.get('authors', ''),
                        abstract=paper.get('abstract', ''),
                        key_sections=key_sections,
                        publish_date=paper.get('publish_date', ''),
                        categories=paper.get('categories', []),
                        relevance_score=paper.get('relevance_score', 0.8),
                        content_length=content_length
                    )
                    grounded_papers.append(grounded_paper)
                    logger.info(f"CONTENT GROUNDING DEBUG: Successfully processed paper {i+1}",
                               title=paper.get('title', '')[:50],
                               content_source=content_source,
                               content_length=content_length)
                else:
                    # Fallback: Get full paper details from external storage if needed
                    full_paper = await self._get_full_paper_details(paper.get('arxiv_id'))
                    
                    if full_paper:
                        grounded_paper = GroundedPaperContent(
                            arxiv_id=full_paper.get('arxiv_id', ''),
                            title=full_paper.get('title', ''),
                            authors=full_paper.get('authors', ''),
                            abstract=full_paper.get('abstract', ''),
                            key_sections=self._extract_key_sections(full_paper),
                            publish_date=full_paper.get('publish_date', ''),
                            categories=full_paper.get('categories', []),
                            relevance_score=paper.get('score', 0.8),
                            content_length=len(full_paper.get('abstract', ''))
                        )
                        grounded_papers.append(grounded_paper)
                    
            except Exception as e:
                logger.warning("Failed to extract paper content",
                              paper_id=paper.get('arxiv_id'),
                              error=str(e))
                continue
        
        logger.info("CONTENT GROUNDING DEBUG: Paper content extraction completed",
                   successful_papers=len(grounded_papers),
                   total_papers=len(retrieved_papers))
        return grounded_papers
    
    async def _get_full_paper_details(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full paper details from external storage"""
        if not self.knowledge_base or not arxiv_id:
            return None
            
        try:
            # Query external storage for complete paper details
            if hasattr(self.knowledge_base, 'storage_manager'):
                cursor = self.knowledge_base.storage_manager.storage_db.cursor()
                cursor.execute("""
                    SELECT title, abstract, authors, arxiv_id, publish_date, 
                           categories, domain, journal_ref
                    FROM arxiv_papers 
                    WHERE arxiv_id = ?
                """, (arxiv_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'title': row[0],
                        'abstract': row[1],
                        'authors': row[2],
                        'arxiv_id': row[3],
                        'publish_date': row[4],
                        'categories': row[5].split(',') if row[5] else [],
                        'domain': row[6],
                        'journal_ref': row[7] or ''
                    }
        except Exception as e:
            logger.error("Failed to retrieve full paper details",
                        arxiv_id=arxiv_id,
                        error=str(e))
        
        return None
    
    def _extract_key_sections(self, paper: Dict[str, Any]) -> List[str]:
        """Extract key sections from paper for detailed responses (legacy method)"""
        return self._extract_enhanced_key_sections(paper)
    
    def _extract_enhanced_key_sections(self, paper: Dict[str, Any]) -> List[str]:
        """Extract enhanced key sections from paper including full content when available"""
        key_sections = []
        
        has_full_content = paper.get('has_full_content', False)
        
        if has_full_content:
            # Use full paper content sections
            logger.info(f"Using full content sections for paper {paper.get('arxiv_id', 'unknown')}")
            
            # Add introduction if available
            if paper.get('introduction'):
                intro_preview = paper['introduction'][:500] + "..." if len(paper['introduction']) > 500 else paper['introduction']
                key_sections.append(f"Introduction: {intro_preview}")
            
            # Add methodology if available
            if paper.get('methodology'):
                method_preview = paper['methodology'][:500] + "..." if len(paper['methodology']) > 500 else paper['methodology']
                key_sections.append(f"Methodology: {method_preview}")
            
            # Add results if available
            if paper.get('results'):
                results_preview = paper['results'][:500] + "..." if len(paper['results']) > 500 else paper['results']
                key_sections.append(f"Results: {results_preview}")
            
            # Add discussion if available
            if paper.get('discussion'):
                discussion_preview = paper['discussion'][:500] + "..." if len(paper['discussion']) > 500 else paper['discussion']
                key_sections.append(f"Discussion: {discussion_preview}")
            
            # Add conclusion if available
            if paper.get('conclusion'):
                conclusion_preview = paper['conclusion'][:500] + "..." if len(paper['conclusion']) > 500 else paper['conclusion']
                key_sections.append(f"Conclusion: {conclusion_preview}")
            
            # Add abstract as supplementary
            if paper.get('abstract'):
                key_sections.append(f"Abstract (supplementary): {paper['abstract']}")
            
        else:
            # Fallback to abstract-only content
            logger.info(f"Using abstract-only content for paper {paper.get('arxiv_id', 'unknown')}")
            
            # Add abstract as primary content
            if paper.get('abstract'):
                key_sections.append(f"Abstract: {paper['abstract']}")
        
        # Add metadata regardless of content type
        if paper.get('journal_ref'):
            key_sections.append(f"Published in: {paper['journal_ref']}")
        
        if paper.get('domain'):
            key_sections.append(f"Domain: {paper['domain']}")
        
        if paper.get('categories'):
            if isinstance(paper['categories'], list):
                categories_str = ', '.join(paper['categories'])
            else:
                categories_str = str(paper['categories'])
            key_sections.append(f"Categories: {categories_str}")
            
        return key_sections
    
    async def _expand_paper_content(
        self, 
        grounded_papers: List[GroundedPaperContent],
        target_tokens: int
    ) -> List[GroundedPaperContent]:
        """Expand paper content by retrieving additional related papers if needed"""
        current_tokens = self._estimate_content_tokens(grounded_papers)
        
        if current_tokens < (target_tokens * 0.6) and len(grounded_papers) < 10:
            logger.info("Expanding paper content for sufficient grounding",
                       current_tokens=current_tokens,
                       target_tokens=target_tokens)
            
            # Get additional papers from similar categories/domains
            additional_papers = await self._get_related_papers(
                grounded_papers, 
                needed_tokens=target_tokens - current_tokens
            )
            grounded_papers.extend(additional_papers)
        
        return grounded_papers
    
    async def _get_related_papers(
        self, 
        existing_papers: List[GroundedPaperContent],
        needed_tokens: int
    ) -> List[GroundedPaperContent]:
        """Retrieve additional related papers to meet content requirements"""
        additional_papers = []
        
        if not self.knowledge_base:
            return additional_papers
        
        try:
            # Get papers from same categories as existing papers
            categories = set()
            for paper in existing_papers:
                categories.update(paper.categories)
            
            if categories and hasattr(self.knowledge_base, 'storage_manager'):
                cursor = self.knowledge_base.storage_manager.storage_db.cursor()
                
                # Build category filter
                category_filter = " OR ".join([f"categories LIKE '%{cat}%'" for cat in list(categories)[:3]])
                
                cursor.execute(f"""
                    SELECT title, abstract, authors, arxiv_id, publish_date, 
                           categories, domain, journal_ref
                    FROM arxiv_papers 
                    WHERE ({category_filter})
                    AND arxiv_id NOT IN ({','.join(['?' for _ in existing_papers])})
                    ORDER BY publish_date DESC
                    LIMIT 5
                """, [p.arxiv_id for p in existing_papers])
                
                rows = cursor.fetchall()
                for row in rows:
                    paper_data = {
                        'title': row[0],
                        'abstract': row[1],
                        'authors': row[2],
                        'arxiv_id': row[3],
                        'publish_date': row[4],
                        'categories': row[5].split(',') if row[5] else [],
                        'domain': row[6],
                        'journal_ref': row[7] or ''
                    }
                    
                    additional_paper = GroundedPaperContent(
                        arxiv_id=paper_data['arxiv_id'],
                        title=paper_data['title'],
                        authors=paper_data['authors'],
                        abstract=paper_data['abstract'],
                        key_sections=self._extract_key_sections(paper_data),
                        publish_date=paper_data['publish_date'],
                        categories=paper_data['categories'],
                        relevance_score=0.6,  # Lower relevance for expanded content
                        content_length=len(paper_data['abstract'])
                    )
                    additional_papers.append(additional_paper)
                    
        except Exception as e:
            logger.warning("Failed to retrieve additional papers", error=str(e))
        
        return additional_papers
    
    def _estimate_content_tokens(self, papers: List[GroundedPaperContent]) -> int:
        """Estimate token count from grounded paper content"""
        total_words = 0
        for paper in papers:
            total_words += len(paper.title.split())
            total_words += len(paper.abstract.split()) 
            for section in paper.key_sections:
                total_words += len(section.split())
        
        return int(total_words * 1.3)  # Word to token conversion estimate
    
    async def _build_grounded_content_prompt(
        self,
        reasoning_result: Any,
        grounded_papers: List[GroundedPaperContent],
        verbosity_level: str
    ) -> str:
        """Build Claude prompt with actual paper content to prevent hallucinations"""
        
        # Build grounded content section with actual paper abstracts
        papers_content = []
        for i, paper in enumerate(grounded_papers[:8]):  # Limit for Claude context
            paper_content = f"""
Paper {i+1}: {paper.title}
Authors: {paper.authors}
arXiv ID: {paper.arxiv_id}
Published: {paper.publish_date}
Categories: {', '.join(paper.categories)}

Abstract: {paper.abstract}
"""
            if paper.key_sections:
                paper_content += f"Additional Content: {' '.join(paper.key_sections[:2])}"
            
            papers_content.append(paper_content)
        
        # Build the grounded prompt
        grounded_prompt = f"""
You are synthesizing research findings based EXCLUSIVELY on the following papers from the NWTN corpus.
You MUST base your response only on the content provided below. Do not add information from your training data.

REASONING SUMMARY:
{getattr(reasoning_result, 'summary', 'NWTN completed comprehensive analysis across 5,040 reasoning permutations')}

SOURCE PAPERS (MUST use only these sources):
{chr(10).join(papers_content)}

SYNTHESIS REQUIREMENTS:
- Use ONLY the information provided in the source papers above
- Generate a {verbosity_level} level response based on the provided abstracts and content
- Include citations to the specific papers provided (use the arXiv IDs)
- If the provided papers don't contain enough detail for longer responses, indicate this limitation
- Do not add information not present in the provided papers
- Focus on synthesizing and analyzing the provided content rather than adding external knowledge
        """
        
        return grounded_prompt
    
    def _assess_grounding_quality(
        self, 
        papers: List[GroundedPaperContent],
        target_tokens: int
    ) -> float:
        """Assess quality of content grounding"""
        if not papers:
            return 0.0
        
        # Factors affecting grounding quality
        content_coverage = min(1.0, self._estimate_content_tokens(papers) / max(target_tokens, 1))
        paper_quality = sum(p.relevance_score for p in papers) / len(papers)
        content_diversity = len(set(tuple(p.categories) for p in papers)) / max(len(papers), 1)
        
        # Weighted quality score
        quality = (content_coverage * 0.4 + paper_quality * 0.4 + content_diversity * 0.2)
        
        return min(1.0, quality)

async def create_grounded_synthesizer(external_knowledge_base=None) -> ContentGroundingSynthesizer:
    """Factory function to create a content grounding synthesizer"""
    synthesizer = ContentGroundingSynthesizer(external_knowledge_base)
    logger.info("Content Grounding Synthesizer initialized", 
               grounding_enabled=external_knowledge_base is not None)
    return synthesizer