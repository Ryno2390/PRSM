#!/usr/bin/env python3
"""
Paper Selector for 100-Paper Test
Selects 100 high-value papers from arXiv focusing on biomimetics and materials science
"""

import arxiv
import requests
import time
from typing import List, Dict
import json

class PaperSelector:
    """Selects high-value papers for analogical reasoning testing"""
    
    def __init__(self):
        self.target_count = 100
        self.selected_papers = []
        
        # Focus on high-value research areas
        self.search_queries = [
            "biomimetics AND materials",
            "bio-inspired materials",
            "nature-inspired design",
            "biomimetic surface",
            "gecko adhesion",
            "shark skin drag",
            "butterfly wing",
            "spider silk properties",
            "plant surface",
            "bio-inspired robotics",
            "smart materials",
            "adaptive materials",
            "self-healing materials",
            "responsive materials",
            "functional surfaces"
        ]
        
        # Quality filters
        self.min_year = 2020  # Recent papers only
        self.preferred_categories = [
            "cond-mat.mtrl-sci",  # Materials science
            "physics.bio-ph",     # Biological physics
            "q-bio.TO",          # Tissue and organs
            "cs.RO",             # Robotics
            "physics.class-ph"   # Classical physics
        ]
    
    def search_papers_by_query(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search for papers using a specific query"""
        
        print(f"Searching for papers: '{query}'")
        
        try:
            # Create search client
            client = arxiv.Client()
            
            # Build search with filters
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in client.results(search):
                # Filter by year
                if result.published.year < self.min_year:
                    continue
                
                # Create paper info
                paper_info = {
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'categories': result.categories,
                    'pdf_url': result.pdf_url,
                    'query': query,
                    'relevance_score': self._calculate_relevance_score(result, query)
                }
                
                papers.append(paper_info)
                
                # Rate limiting
                time.sleep(0.1)
            
            return papers
            
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            return []
    
    def _calculate_relevance_score(self, result, query: str) -> float:
        """Calculate relevance score for paper selection"""
        
        score = 0.0
        title_lower = result.title.lower()
        abstract_lower = result.summary.lower()
        
        # Keyword matching
        biomimetic_keywords = [
            'biomimetic', 'bio-inspired', 'nature-inspired', 'bioinspired',
            'gecko', 'shark', 'butterfly', 'spider', 'plant', 'leaf',
            'lotus', 'velcro', 'burdock', 'adhesion', 'drag reduction'
        ]
        
        materials_keywords = [
            'material', 'surface', 'coating', 'film', 'composite',
            'nanostructure', 'microstructure', 'texture', 'roughness'
        ]
        
        discovery_keywords = [
            'novel', 'new', 'innovative', 'breakthrough', 'discovery',
            'mechanism', 'property', 'performance', 'enhancement'
        ]
        
        # Score based on keyword presence
        for keyword in biomimetic_keywords:
            if keyword in title_lower:
                score += 3.0
            if keyword in abstract_lower:
                score += 1.0
        
        for keyword in materials_keywords:
            if keyword in title_lower:
                score += 2.0
            if keyword in abstract_lower:
                score += 0.5
        
        for keyword in discovery_keywords:
            if keyword in title_lower:
                score += 1.5
            if keyword in abstract_lower:
                score += 0.3
        
        # Bonus for preferred categories
        for category in result.categories:
            if category in self.preferred_categories:
                score += 2.0
        
        # Recent publication bonus
        year_bonus = (result.published.year - 2020) * 0.5
        score += year_bonus
        
        return score
    
    def select_diverse_papers(self) -> List[Dict]:
        """Select diverse set of 100 papers across different queries"""
        
        all_papers = []
        
        # Search with each query
        for query in self.search_queries:
            papers = self.search_papers_by_query(query, max_results=15)
            all_papers.extend(papers)
            
            print(f"Found {len(papers)} papers for '{query}'")
            time.sleep(1)  # Rate limiting between queries
        
        print(f"\nTotal papers found: {len(all_papers)}")
        
        # Remove duplicates (same arxiv_id)
        seen_ids = set()
        unique_papers = []
        
        for paper in all_papers:
            if paper['arxiv_id'] not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper['arxiv_id'])
        
        print(f"Unique papers: {len(unique_papers)}")
        
        # Sort by relevance score
        unique_papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Select top 100 with diversity across queries
        selected = []
        query_counts = {}
        
        for paper in unique_papers:
            if len(selected) >= self.target_count:
                break
            
            query = paper['query']
            if query_counts.get(query, 0) < 10:  # Max 10 per query for diversity
                selected.append(paper)
                query_counts[query] = query_counts.get(query, 0) + 1
        
        # Fill remaining slots if needed
        for paper in unique_papers:
            if len(selected) >= self.target_count:
                break
            if paper not in selected:
                selected.append(paper)
        
        self.selected_papers = selected[:self.target_count]
        return self.selected_papers
    
    def save_paper_list(self, filename: str = "selected_papers.json"):
        """Save selected papers to JSON file"""
        
        with open(filename, 'w') as f:
            json.dump(self.selected_papers, f, indent=2)
        
        print(f"Saved {len(self.selected_papers)} papers to {filename}")
    
    def analyze_selection(self):
        """Analyze the quality and diversity of selected papers"""
        
        if not self.selected_papers:
            print("No papers selected yet!")
            return
        
        print(f"\n📊 PAPER SELECTION ANALYSIS")
        print(f"=" * 50)
        print(f"Total Papers Selected: {len(self.selected_papers)}")
        
        # Year distribution
        year_counts = {}
        for paper in self.selected_papers:
            year = paper['published'][:4]
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print(f"\n📅 Year Distribution:")
        for year in sorted(year_counts.keys()):
            print(f"   {year}: {year_counts[year]} papers")
        
        # Query distribution
        query_counts = {}
        for paper in self.selected_papers:
            query = paper['query']
            query_counts[query] = query_counts.get(query, 0) + 1
        
        print(f"\n🔍 Query Distribution:")
        for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   '{query}': {count} papers")
        
        # Relevance scores
        scores = [paper['relevance_score'] for paper in self.selected_papers]
        print(f"\n⭐ Relevance Scores:")
        print(f"   Average: {sum(scores)/len(scores):.2f}")
        print(f"   Range: {min(scores):.2f} - {max(scores):.2f}")
        
        # Top 5 papers
        print(f"\n🏆 Top 5 Papers by Relevance:")
        for i, paper in enumerate(self.selected_papers[:5], 1):
            print(f"   {i}. {paper['title'][:60]}... (score: {paper['relevance_score']:.2f})")
        
        print(f"\n✅ Paper selection complete and ready for processing!")

def main():
    selector = PaperSelector()
    
    print("🔍 Starting paper selection for 100-paper test...")
    print("This will search arXiv for high-value biomimetics and materials papers")
    
    # Select papers
    papers = selector.select_diverse_papers()
    
    if len(papers) >= 100:
        # Save to file
        selector.save_paper_list()
        
        # Analyze selection
        selector.analyze_selection()
        
        print(f"\n🎯 SUCCESS: Selected {len(papers)} high-quality papers")
        print("Ready for SOC extraction and pattern analysis!")
    else:
        print(f"⚠️  Warning: Only found {len(papers)} papers (target: 100)")
        print("Consider adjusting search criteria or proceeding with available papers")

if __name__ == "__main__":
    main()