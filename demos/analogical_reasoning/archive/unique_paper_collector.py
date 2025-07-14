#!/usr/bin/env python3
"""
Unique Paper Collector for NWTN Testing
Collects diverse, unique scientific papers from arXiv for breakthrough testing
"""

import json
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Set
from datetime import datetime, timedelta
import random

class UniquePaperCollector:
    """Collects unique scientific papers from arXiv across diverse domains"""
    
    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.collected_papers = []
        self.used_arxiv_ids = set()
        
        # Diverse scientific domains for breakthrough discovery
        self.search_queries = [
            # Biomimetics and Bio-inspired Engineering
            "cat:physics.bio-ph AND (biomimetic OR bio-inspired OR biological)",
            "cat:cond-mat.soft AND (adhesion OR gecko OR spider)",
            "cat:physics.bio-ph AND (attachment OR surface OR interface)",
            
            # Materials Science and Engineering
            "cat:cond-mat.mtrl-sci AND (surface AND adhesion)",
            "cat:cond-mat.mtrl-sci AND (mechanical AND properties)",
            "cat:cond-mat.soft AND (polymer OR elastomer)",
            
            # Nanotechnology and Surfaces
            "cat:cond-mat.mes-hall AND (nano AND surface)",
            "cat:physics.app-ph AND (nanoscale AND adhesion)",
            "cat:cond-mat.mtrl-sci AND (nanostructure OR nanomaterial)",
            
            # Robotics and Engineering Applications
            "cat:cs.RO AND (bio-inspired OR biomimetic)",
            "cat:cs.RO AND (adhesion OR climbing OR attachment)",
            "cat:physics.app-ph AND (robotic AND surface)",
            
            # Mechanical Engineering and Design
            "cat:physics.app-ph AND (mechanical AND design)",
            "cat:cond-mat.soft AND (mechanical AND optimization)",
            "cat:physics.app-ph AND (engineering AND bio)",
            
            # Advanced Materials and Manufacturing
            "cat:cond-mat.mtrl-sci AND (manufacturing OR fabrication)",
            "cat:physics.app-ph AND (functional AND material)",
            "cat:cond-mat.mtrl-sci AND (smart AND material)",
            
            # Interdisciplinary Applications
            "cat:physics.bio-ph AND (engineering OR application)",
            "cat:cond-mat.soft AND (biophysics OR biomechanics)",
            "cat:physics.app-ph AND (interdisciplinary OR multidisciplinary)"
        ]
    
    def collect_unique_papers(self, target_count: int = 500) -> List[Dict]:
        """Collect unique papers across diverse scientific domains"""
        
        print(f"🔍 COLLECTING {target_count} UNIQUE SCIENTIFIC PAPERS")
        print(f"=" * 60)
        print(f"📚 Searching across {len(self.search_queries)} diverse scientific domains")
        print(f"🎯 Target: {target_count} unique papers for breakthrough discovery testing")
        
        papers_per_query = max(25, target_count // len(self.search_queries))
        
        for i, query in enumerate(self.search_queries, 1):
            print(f"\n🔬 Domain {i}/{len(self.search_queries)}: {self._get_domain_name(query)}")
            print(f"   Query: {query[:60]}...")
            
            try:
                domain_papers = self._search_arxiv(query, papers_per_query)
                new_papers = self._filter_unique_papers(domain_papers)
                
                self.collected_papers.extend(new_papers)
                print(f"   ✅ Added {len(new_papers)} unique papers (total: {len(self.collected_papers)})")
                
                # Rate limiting
                time.sleep(2)
                
                if len(self.collected_papers) >= target_count:
                    print(f"\n🎯 Target reached: {len(self.collected_papers)} papers collected")
                    break
                    
            except Exception as e:
                print(f"   ❌ Error searching domain: {e}")
                continue
        
        # Shuffle for diversity and trim to target
        random.shuffle(self.collected_papers)
        final_papers = self.collected_papers[:target_count]
        
        print(f"\n✅ COLLECTION COMPLETE:")
        print(f"   📊 Total unique papers: {len(final_papers)}")
        print(f"   🔬 Domains covered: {len(set(p.get('domain', 'unknown') for p in final_papers))}")
        print(f"   📅 Date range: {self._get_date_range(final_papers)}")
        
        return final_papers
    
    def _search_arxiv(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search arXiv for papers matching query"""
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        response = requests.get(self.arxiv_base_url, params=params)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        entries = root.findall('atom:entry', namespaces)
        
        for entry in entries:
            try:
                # Extract paper information
                title = entry.find('atom:title', namespaces)
                title_text = title.text.strip().replace('\n', ' ') if title is not None else "No title"
                
                summary = entry.find('atom:summary', namespaces)
                abstract_text = summary.text.strip().replace('\n', ' ') if summary is not None else "No abstract"
                
                # Get arXiv ID
                id_elem = entry.find('atom:id', namespaces)
                arxiv_url = id_elem.text if id_elem is not None else ""
                arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else f"unknown_{len(papers)}"
                
                # Get authors
                authors = []
                author_elements = entry.findall('atom:author', namespaces)
                for author_elem in author_elements:
                    name_elem = author_elem.find('atom:name', namespaces)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                
                # Get categories
                categories = []
                category_elements = entry.findall('atom:category', namespaces)
                for cat_elem in category_elements:
                    term = cat_elem.get('term')
                    if term:
                        categories.append(term)
                
                # Get publication date
                published = entry.find('atom:published', namespaces)
                pub_date = published.text if published is not None else datetime.now().isoformat()
                
                paper = {
                    'arxiv_id': arxiv_id,
                    'title': title_text,
                    'abstract': abstract_text,
                    'authors': authors,
                    'categories': categories,
                    'published_date': pub_date,
                    'query': query,
                    'domain': self._get_domain_name(query),
                    'collection_timestamp': datetime.now().isoformat()
                }
                
                papers.append(paper)
                
            except Exception as e:
                print(f"      ⚠️ Error parsing entry: {e}")
                continue
        
        return papers
    
    def _filter_unique_papers(self, papers: List[Dict]) -> List[Dict]:
        """Filter out papers we've already collected"""
        
        unique_papers = []
        
        for paper in papers:
            arxiv_id = paper['arxiv_id']
            
            if arxiv_id not in self.used_arxiv_ids:
                self.used_arxiv_ids.add(arxiv_id)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _get_domain_name(self, query: str) -> str:
        """Get readable domain name from query"""
        
        if "bio-ph" in query and "biomimetic" in query:
            return "Biomimetics"
        elif "mtrl-sci" in query and "adhesion" in query:
            return "Materials_Adhesion"
        elif "soft" in query and "gecko" in query:
            return "Bio_Inspired_Surfaces"
        elif "mes-hall" in query and "nano" in query:
            return "Nanotechnology"
        elif "cs.RO" in query:
            return "Robotics"
        elif "app-ph" in query and "mechanical" in query:
            return "Mechanical_Engineering"
        elif "mtrl-sci" in query and "manufacturing" in query:
            return "Advanced_Manufacturing"
        elif "bio-ph" in query and "engineering" in query:
            return "Bioengineering"
        else:
            return "Interdisciplinary"
    
    def _get_date_range(self, papers: List[Dict]) -> str:
        """Get date range of collected papers"""
        
        if not papers:
            return "No papers"
        
        dates = [p.get('published_date', '') for p in papers if p.get('published_date')]
        
        if not dates:
            return "Unknown date range"
        
        try:
            # Parse dates and find range
            parsed_dates = []
            for date_str in dates:
                try:
                    # Handle different date formats from arXiv
                    if 'T' in date_str:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')
                    parsed_dates.append(date_obj)
                except:
                    continue
            
            if parsed_dates:
                min_date = min(parsed_dates)
                max_date = max(parsed_dates)
                return f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}"
            else:
                return "Date parsing failed"
                
        except Exception as e:
            return f"Date range error: {e}"
    
    def save_unique_papers(self, papers: List[Dict], filename: str = "unique_papers_collection.json"):
        """Save collected unique papers to file"""
        
        # Add collection metadata
        collection_data = {
            'collection_metadata': {
                'total_papers': len(papers),
                'collection_date': datetime.now().isoformat(),
                'domains_covered': len(set(p.get('domain', 'unknown') for p in papers)),
                'unique_arxiv_ids': len(set(p['arxiv_id'] for p in papers)),
                'query_count': len(self.search_queries),
                'date_range': self._get_date_range(papers)
            },
            'papers': papers
        }
        
        with open(filename, 'w') as f:
            json.dump(collection_data, f, indent=2)
        
        print(f"\n💾 SAVED UNIQUE PAPER COLLECTION:")
        print(f"   File: {filename}")
        print(f"   Papers: {len(papers)}")
        print(f"   Domains: {collection_data['collection_metadata']['domains_covered']}")
        print(f"   Date range: {collection_data['collection_metadata']['date_range']}")
        
        return filename
    
    def validate_uniqueness(self, papers: List[Dict]) -> Dict:
        """Validate that all papers are truly unique"""
        
        arxiv_ids = [p['arxiv_id'] for p in papers]
        titles = [p['title'] for p in papers]
        
        unique_ids = set(arxiv_ids)
        unique_titles = set(titles)
        
        validation_results = {
            'total_papers': len(papers),
            'unique_arxiv_ids': len(unique_ids),
            'unique_titles': len(unique_titles),
            'arxiv_id_duplicates': len(arxiv_ids) - len(unique_ids),
            'title_duplicates': len(titles) - len(unique_titles),
            'is_fully_unique': len(unique_ids) == len(papers) and len(unique_titles) == len(papers)
        }
        
        print(f"\n🔍 UNIQUENESS VALIDATION:")
        print(f"   Total papers: {validation_results['total_papers']}")
        print(f"   Unique arXiv IDs: {validation_results['unique_arxiv_ids']}")
        print(f"   Unique titles: {validation_results['unique_titles']}")
        print(f"   ID duplicates: {validation_results['arxiv_id_duplicates']}")
        print(f"   Title duplicates: {validation_results['title_duplicates']}")
        
        if validation_results['is_fully_unique']:
            print(f"   ✅ All papers are unique!")
        else:
            print(f"   ⚠️ Some duplicates found")
        
        return validation_results

def main():
    """Collect unique papers for testing"""
    
    collector = UniquePaperCollector()
    
    # Collect 500 unique papers for comprehensive testing
    papers = collector.collect_unique_papers(target_count=500)
    
    # Validate uniqueness
    validation = collector.validate_uniqueness(papers)
    
    # Save collection
    filename = collector.save_unique_papers(papers)
    
    print(f"\n🎉 UNIQUE PAPER COLLECTION COMPLETE!")
    print(f"Ready for breakthrough discovery testing with {len(papers)} unique papers")
    
    return papers, filename

if __name__ == "__main__":
    main()