#!/usr/bin/env python3
"""
PRSM Python SDK - Research Paper Analysis Example

This example demonstrates how to use PRSM for scientific research tasks,
including paper analysis, citation extraction, and research insights generation.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

from prsm_sdk import PRSMClient, PRSMError


@dataclass
class Citation:
    """Represents a research paper citation"""
    title: str
    authors: List[str]
    year: int
    venue: str
    doi: Optional[str] = None
    relevance_score: Optional[float] = None


@dataclass
class ResearchInsight:
    """Represents insights extracted from research papers"""
    key_findings: List[str]
    methodologies: List[str]
    limitations: List[str]
    future_work: List[str]
    confidence_score: float


@dataclass
class PaperAnalysis:
    """Complete analysis of a research paper"""
    title: str
    abstract_summary: str
    key_contributions: List[str]
    methodology: str
    results_summary: str
    citations: List[Citation]
    insights: ResearchInsight
    research_gaps: List[str]
    impact_assessment: str


class ResearchPaperAnalyzer:
    """PRSM-powered research paper analysis system"""
    
    def __init__(self, api_key: str):
        self.client = PRSMClient(api_key=api_key)
        self.analysis_model = "gpt-4"  # Use most capable model for research
        self.extraction_model = "claude-3"  # Alternative for variety
    
    async def analyze_paper_from_text(self, paper_text: str, title: str = None) -> PaperAnalysis:
        """Analyze a research paper from its full text"""
        
        print(f"🔬 Analyzing research paper: {title or 'Untitled'}")
        
        # Step 1: Extract basic information
        basic_info = await self._extract_basic_information(paper_text, title)
        
        # Step 2: Identify key contributions and methodology
        contributions = await self._extract_contributions(paper_text)
        methodology = await self._extract_methodology(paper_text)
        
        # Step 3: Summarize results and findings
        results = await self._summarize_results(paper_text)
        
        # Step 4: Extract and analyze citations
        citations = await self._extract_citations(paper_text)
        
        # Step 5: Generate research insights
        insights = await self._generate_insights(paper_text, contributions, methodology)
        
        # Step 6: Identify research gaps and future work
        gaps = await self._identify_research_gaps(paper_text, insights)
        
        # Step 7: Assess potential impact
        impact = await self._assess_impact(basic_info, contributions, insights)
        
        return PaperAnalysis(
            title=basic_info.get("title", title or "Unknown"),
            abstract_summary=basic_info.get("abstract_summary", ""),
            key_contributions=contributions,
            methodology=methodology,
            results_summary=results,
            citations=citations,
            insights=insights,
            research_gaps=gaps,
            impact_assessment=impact
        )
    
    async def _extract_basic_information(self, paper_text: str, title: str) -> Dict[str, Any]:
        """Extract title, abstract, and basic paper information"""
        
        prompt = f"""
        Analyze this research paper and extract the following information:
        
        1. Title (if not provided: {title})
        2. Abstract summary (2-3 sentences)
        3. Research domain/field
        4. Publication type (conference, journal, preprint, etc.)
        
        Paper text (first 2000 characters):
        {paper_text[:2000]}
        
        Provide the response in JSON format:
        {{
            "title": "Paper title",
            "abstract_summary": "Brief summary of the abstract",
            "research_domain": "Field of research",
            "publication_type": "Type of publication"
        }}
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            return json.loads(result.content)
            
        except (PRSMError, json.JSONDecodeError) as e:
            print(f"⚠️ Error extracting basic info: {e}")
            return {"title": title or "Unknown", "abstract_summary": ""}
    
    async def _extract_contributions(self, paper_text: str) -> List[str]:
        """Extract key contributions from the paper"""
        
        prompt = f"""
        Identify the key contributions of this research paper. Focus on:
        1. Novel methodologies or techniques
        2. Theoretical contributions
        3. Empirical findings
        4. Practical applications
        5. Improvements over existing work
        
        Paper text excerpt:
        {paper_text[:3000]}
        
        List the top 3-5 key contributions, each in 1-2 sentences.
        Return as a JSON array of strings.
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=600,
                temperature=0.2
            )
            
            contributions = json.loads(result.content)
            return contributions if isinstance(contributions, list) else []
            
        except (PRSMError, json.JSONDecodeError) as e:
            print(f"⚠️ Error extracting contributions: {e}")
            return []
    
    async def _extract_methodology(self, paper_text: str) -> str:
        """Extract and summarize the research methodology"""
        
        prompt = f"""
        Summarize the research methodology used in this paper. Include:
        1. Research approach (experimental, theoretical, survey, etc.)
        2. Data collection methods
        3. Analysis techniques
        4. Tools and technologies used
        5. Evaluation metrics
        
        Paper text excerpt:
        {paper_text[1000:4000]}  # Focus on methods section
        
        Provide a concise 2-3 paragraph summary of the methodology.
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=400,
                temperature=0.1
            )
            
            return result.content.strip()
            
        except PRSMError as e:
            print(f"⚠️ Error extracting methodology: {e}")
            return "Methodology extraction failed"
    
    async def _summarize_results(self, paper_text: str) -> str:
        """Summarize the key results and findings"""
        
        prompt = f"""
        Summarize the key results and findings from this research paper:
        1. Main experimental/analytical results
        2. Performance metrics and comparisons
        3. Statistical significance
        4. Validation outcomes
        
        Focus on the results and discussion sections.
        
        Paper text excerpt:
        {paper_text[-3000:]}  # Focus on results/conclusion sections
        
        Provide a clear 2-3 paragraph summary of the results.
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=400,
                temperature=0.1
            )
            
            return result.content.strip()
            
        except PRSMError as e:
            print(f"⚠️ Error summarizing results: {e}")
            return "Results summary failed"
    
    async def _extract_citations(self, paper_text: str) -> List[Citation]:
        """Extract and analyze citations from the paper"""
        
        prompt = f"""
        Extract the most important citations from this research paper's references.
        Focus on:
        1. Foundational works
        2. Recent related work
        3. Methodological references
        4. Comparison baselines
        
        For each citation, extract:
        - Title
        - Authors (first 3 if more)
        - Publication year
        - Venue (journal/conference)
        
        Paper text (references section):
        {paper_text[-2000:]}
        
        Return as JSON array with max 10 most relevant citations:
        [
            {{
                "title": "Paper title",
                "authors": ["Author 1", "Author 2"],
                "year": 2023,
                "venue": "Conference/Journal name"
            }}
        ]
        """
        
        try:
            result = await self.client.models.infer(
                model=self.extraction_model,
                prompt=prompt,
                max_tokens=800,
                temperature=0.1
            )
            
            citations_data = json.loads(result.content)
            
            citations = []
            for citation_data in citations_data:
                citation = Citation(
                    title=citation_data.get("title", ""),
                    authors=citation_data.get("authors", []),
                    year=citation_data.get("year", 0),
                    venue=citation_data.get("venue", "")
                )
                citations.append(citation)
            
            return citations
            
        except (PRSMError, json.JSONDecodeError) as e:
            print(f"⚠️ Error extracting citations: {e}")
            return []
    
    async def _generate_insights(self, paper_text: str, contributions: List[str], methodology: str) -> ResearchInsight:
        """Generate deep insights about the research"""
        
        prompt = f"""
        Based on this research paper, generate deep insights about:
        
        Key Contributions:
        {json.dumps(contributions, indent=2)}
        
        Methodology:
        {methodology}
        
        Analyze the paper and provide insights on:
        1. Key findings and their significance
        2. Methodological strengths and innovations
        3. Limitations and potential weaknesses
        4. Suggested future research directions
        5. Overall confidence in the work (0.0-1.0)
        
        Paper excerpt:
        {paper_text[500:2500]}
        
        Return as JSON:
        {{
            "key_findings": ["Finding 1", "Finding 2", ...],
            "methodologies": ["Method strength 1", ...],
            "limitations": ["Limitation 1", ...],
            "future_work": ["Future direction 1", ...],
            "confidence_score": 0.85
        }}
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=700,
                temperature=0.3
            )
            
            insights_data = json.loads(result.content)
            
            return ResearchInsight(
                key_findings=insights_data.get("key_findings", []),
                methodologies=insights_data.get("methodologies", []),
                limitations=insights_data.get("limitations", []),
                future_work=insights_data.get("future_work", []),
                confidence_score=insights_data.get("confidence_score", 0.5)
            )
            
        except (PRSMError, json.JSONDecodeError) as e:
            print(f"⚠️ Error generating insights: {e}")
            return ResearchInsight([], [], [], [], 0.0)
    
    async def _identify_research_gaps(self, paper_text: str, insights: ResearchInsight) -> List[str]:
        """Identify research gaps and opportunities"""
        
        prompt = f"""
        Based on this research paper and the insights generated, identify research gaps and opportunities:
        
        Current Limitations:
        {json.dumps(insights.limitations, indent=2)}
        
        Future Work Suggestions:
        {json.dumps(insights.future_work, indent=2)}
        
        Paper context:
        {paper_text[2000:3500]}
        
        Identify 3-5 specific research gaps or opportunities that could lead to future work.
        Each gap should be actionable and specific.
        
        Return as JSON array of strings.
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=400,
                temperature=0.4
            )
            
            gaps = json.loads(result.content)
            return gaps if isinstance(gaps, list) else []
            
        except (PRSMError, json.JSONDecodeError) as e:
            print(f"⚠️ Error identifying research gaps: {e}")
            return []
    
    async def _assess_impact(self, basic_info: Dict, contributions: List[str], insights: ResearchInsight) -> str:
        """Assess the potential impact of the research"""
        
        prompt = f"""
        Assess the potential impact of this research work:
        
        Research Domain: {basic_info.get('research_domain', 'Unknown')}
        
        Key Contributions:
        {json.dumps(contributions, indent=2)}
        
        Key Findings:
        {json.dumps(insights.key_findings, indent=2)}
        
        Confidence Score: {insights.confidence_score}
        
        Evaluate:
        1. Scientific/theoretical impact
        2. Practical applications
        3. Industry relevance
        4. Long-term significance
        5. Reproducibility and scalability
        
        Provide a comprehensive impact assessment in 2-3 paragraphs.
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            return result.content.strip()
            
        except PRSMError as e:
            print(f"⚠️ Error assessing impact: {e}")
            return "Impact assessment failed"
    
    async def compare_papers(self, analyses: List[PaperAnalysis]) -> Dict[str, Any]:
        """Compare multiple paper analyses"""
        
        if len(analyses) < 2:
            return {"error": "At least 2 papers required for comparison"}
        
        comparison_data = {
            "papers": [
                {
                    "title": analysis.title,
                    "contributions": analysis.key_contributions,
                    "methodology": analysis.methodology,
                    "confidence": analysis.insights.confidence_score
                }
                for analysis in analyses
            ]
        }
        
        prompt = f"""
        Compare these research papers across multiple dimensions:
        
        {json.dumps(comparison_data, indent=2)}
        
        Provide a comparison focusing on:
        1. Methodological approaches and their strengths
        2. Contribution significance and novelty
        3. Research quality and rigor
        4. Potential impact and applications
        5. Recommendations for which work to build upon
        
        Return a structured comparison analysis.
        """
        
        try:
            result = await self.client.models.infer(
                model=self.analysis_model,
                prompt=prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            return {
                "comparison_analysis": result.content,
                "paper_count": len(analyses),
                "average_confidence": sum(a.insights.confidence_score for a in analyses) / len(analyses)
            }
            
        except PRSMError as e:
            print(f"⚠️ Error comparing papers: {e}")
            return {"error": str(e)}


async def analyze_paper_from_file(file_path: str, analyzer: ResearchPaperAnalyzer) -> PaperAnalysis:
    """Analyze a research paper from a text file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            paper_text = f.read()
        
        title = Path(file_path).stem.replace('_', ' ').title()
        
        analysis = await analyzer.analyze_paper_from_text(paper_text, title)
        return analysis
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        raise
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        raise


def print_analysis_report(analysis: PaperAnalysis):
    """Print a formatted analysis report"""
    
    print("\n" + "=" * 80)
    print(f"📄 RESEARCH PAPER ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\n📝 Title: {analysis.title}")
    print(f"\n📋 Abstract Summary:")
    print(f"   {analysis.abstract_summary}")
    
    print(f"\n🔍 Key Contributions:")
    for i, contribution in enumerate(analysis.key_contributions, 1):
        print(f"   {i}. {contribution}")
    
    print(f"\n🔬 Methodology:")
    print(f"   {analysis.methodology}")
    
    print(f"\n📊 Results Summary:")
    print(f"   {analysis.results_summary}")
    
    print(f"\n💡 Key Insights:")
    insights = analysis.insights
    print(f"   Confidence Score: {insights.confidence_score:.2f}")
    
    if insights.key_findings:
        print(f"   Key Findings:")
        for finding in insights.key_findings:
            print(f"     • {finding}")
    
    if insights.limitations:
        print(f"   Limitations:")
        for limitation in insights.limitations:
            print(f"     • {limitation}")
    
    if analysis.research_gaps:
        print(f"\n🔍 Research Gaps:")
        for gap in analysis.research_gaps:
            print(f"   • {gap}")
    
    print(f"\n🎯 Impact Assessment:")
    print(f"   {analysis.impact_assessment}")
    
    if analysis.citations:
        print(f"\n📚 Key Citations ({len(analysis.citations)}):")
        for citation in analysis.citations[:5]:  # Show top 5
            authors_str = ", ".join(citation.authors[:2])
            if len(citation.authors) > 2:
                authors_str += " et al."
            print(f"   • {citation.title} ({authors_str}, {citation.year})")


async def demo_research_analysis():
    """Demonstrate research paper analysis capabilities"""
    
    # Check for API key
    api_key = os.getenv("PRSM_API_KEY")
    if not api_key:
        print("❌ Please set PRSM_API_KEY environment variable")
        return
    
    print("🔬 PRSM Research Paper Analysis Demo")
    print("=" * 50)
    
    analyzer = ResearchPaperAnalyzer(api_key)
    
    # Example: Analyze a sample research abstract/excerpt
    sample_paper = """
    Title: Attention Is All You Need
    
    Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show that these models are superior in quality while being more parallelizable and requiring significantly less time to train.
    
    Introduction: Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.
    
    The Transformer Model: The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. In the following sections, we describe the Transformer in detail.
    
    Experiments: We evaluate our model on English-to-German and English-to-French translation tasks using the WMT 2014 datasets. For both tasks we used byte-pair encoding which has a shared source-target vocabulary of about 37000 tokens for English-German and 32000 for English-French.
    
    Results: On the WMT 2014 English-to-German translation task, the big Transformer model (Transformer (big)) outperforms the best previously reported models by more than 2.0 BLEU, establishing a new single-model state-of-the-art BLEU score of 28.4. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight P100 GPUs.
    
    Conclusion: In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.
    """
    
    try:
        # Analyze the sample paper
        print("🔍 Analyzing 'Attention Is All You Need' paper...")
        analysis = await analyzer.analyze_paper_from_text(
            sample_paper, 
            "Attention Is All You Need"
        )
        
        # Print the analysis report
        print_analysis_report(analysis)
        
        # Demonstrate cost tracking
        budget = await analyzer.client.cost_optimization.get_budget()
        print(f"\n💰 Analysis Cost: ~${budget.spent:.4f}")
        
    except PRSMError as e:
        print(f"❌ PRSM Error: {e.message}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        await analyzer.client.close()


async def demo_batch_analysis():
    """Demonstrate batch analysis of multiple papers"""
    
    api_key = os.getenv("PRSM_API_KEY")
    if not api_key:
        print("❌ Please set PRSM_API_KEY environment variable")
        return
    
    print("\n📚 Batch Research Paper Analysis Demo")
    print("=" * 50)
    
    analyzer = ResearchPaperAnalyzer(api_key)
    
    # Sample papers for comparison
    papers = {
        "BERT": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers...",
        "GPT": "Improving Language Understanding by Generative Pre-Training. Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering, semantic similarity assessment, and document classification...",
        "T5": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task..."
    }
    
    try:
        analyses = []
        
        # Analyze each paper
        for title, text in papers.items():
            print(f"🔍 Analyzing {title}...")
            analysis = await analyzer.analyze_paper_from_text(text, title)
            analyses.append(analysis)
        
        # Compare the papers
        print("\n🔍 Comparing papers...")
        comparison = await analyzer.compare_papers(analyses)
        
        print("\n" + "=" * 80)
        print("📊 PAPER COMPARISON REPORT")
        print("=" * 80)
        print(comparison["comparison_analysis"])
        print(f"\nAverage Confidence Score: {comparison['average_confidence']:.2f}")
        
    except PRSMError as e:
        print(f"❌ PRSM Error: {e.message}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        await analyzer.client.close()


async def main():
    """Run the research paper analysis examples"""
    
    await demo_research_analysis()
    await demo_batch_analysis()
    
    print("\n" + "=" * 80)
    print("✅ Research paper analysis examples completed!")
    print("\n💡 Use cases:")
    print("• Literature review automation")
    print("• Research gap identification")
    print("• Citation analysis and recommendations")
    print("• Impact assessment for funding decisions")
    print("• Peer review assistance")
    print("• Research trend analysis")


if __name__ == "__main__":
    asyncio.run(main())