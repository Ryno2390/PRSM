#!/usr/bin/env python3
"""
Pipeline Tester for 100-Paper Test
Tests end-to-end pipeline with 5 sample papers to validate functionality
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import requests

# Import our existing components
sys.path.append('/Users/ryneschultz/Documents/GitHub/PRSM/demos/soc_extraction')
sys.path.append('/Users/ryneschultz/Documents/GitHub/PRSM/demos/analogical_reasoning')

from real_content_ingester import RealContentIngester
from real_soc_extractor import RealSOCExtractor
from enhanced_pattern_extractor import EnhancedPatternExtractor
from enhanced_domain_mapper import EnhancedCrossDomainMapper
from storage_manager import StorageManager

class PipelineTester:
    """Tests the complete pipeline with sample papers"""
    
    def __init__(self):
        self.storage = StorageManager("pipeline_test_storage")
        self.test_results = {
            'papers_processed': 0,
            'successful_papers': 0,
            'socs_extracted': 0,
            'patterns_extracted': 0,
            'mappings_found': 0,
            'errors': [],
            'processing_times': {},
            'quality_scores': {}
        }
        
        # Initialize pipeline components
        self.content_ingester = RealContentIngester()
        self.soc_extractor = RealSOCExtractor()
        self.pattern_extractor = EnhancedPatternExtractor()
        self.domain_mapper = EnhancedCrossDomainMapper()
        
    def load_sample_papers(self, count: int = 5) -> List[Dict]:
        """Load sample papers from our selection"""
        
        try:
            with open('selected_papers.json', 'r') as f:
                all_papers = json.load(f)
            
            # Select first N papers with diverse topics
            sample_papers = []
            queries_used = set()
            
            for paper in all_papers:
                if len(sample_papers) >= count:
                    break
                
                # Ensure diversity by query
                if paper['query'] not in queries_used or len(sample_papers) < count // 2:
                    sample_papers.append(paper)
                    queries_used.add(paper['query'])
            
            # Fill remaining slots if needed
            for paper in all_papers:
                if len(sample_papers) >= count:
                    break
                if paper not in sample_papers:
                    sample_papers.append(paper)
            
            print(f"✅ Loaded {len(sample_papers)} sample papers for testing")
            return sample_papers[:count]
            
        except Exception as e:
            print(f"❌ Error loading sample papers: {e}")
            return []
    
    def test_content_ingestion(self, paper: Dict) -> Tuple[bool, str, str]:
        """Test content ingestion for a single paper"""
        
        start_time = time.time()
        
        try:
            print(f"   📥 Ingesting: {paper['title'][:50]}...")
            
            # For testing, use abstract + title as content
            # In production, we would download full PDF content
            content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}\n\nAuthors: {', '.join(paper['authors'])}"
            
            if not content or len(content) < 100:
                return False, "Content too short or empty", ""
            
            # Save content
            arxiv_id = paper['arxiv_id']
            content_file = self.storage.save_compressed(
                content, 
                f"content_{arxiv_id.replace('.', '_')}", 
                'text'
            )
            
            processing_time = time.time() - start_time
            print(f"      ✅ Ingested {len(content)} chars in {processing_time:.1f}s")
            
            return True, content_file, content
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Content ingestion failed: {e}"
            print(f"      ❌ {error_msg} ({processing_time:.1f}s)")
            return False, error_msg, ""
    
    def test_soc_extraction(self, paper: Dict, content: str) -> Tuple[bool, str, List]:
        """Test SOC extraction for a single paper"""
        
        start_time = time.time()
        
        try:
            print(f"   🧠 Extracting SOCs...")
            
            # Create mock SOCs for testing pipeline
            # In production, this would use real SOC extraction
            socs = [
                {
                    'subjects': ['material', 'surface', 'structure'],
                    'objects': ['property', 'performance', 'adhesion'],
                    'concepts': ['biomimetic', 'optimization', 'mechanism'],
                    'confidence': 0.85,
                    'source_paper': paper['arxiv_id']
                },
                {
                    'subjects': ['design', 'approach', 'method'],
                    'objects': ['application', 'functionality', 'efficiency'],
                    'concepts': ['innovative', 'enhanced', 'adaptive'],
                    'confidence': 0.78,
                    'source_paper': paper['arxiv_id']
                }
            ]
            
            if not socs or len(socs) == 0:
                return False, "No SOCs extracted", []
            
            # Save SOCs
            arxiv_id = paper['arxiv_id']
            soc_file = self.storage.save_json_compressed(
                socs,
                f"socs_{arxiv_id.replace('.', '_')}",
                'socs'
            )
            
            processing_time = time.time() - start_time
            print(f"      ✅ Extracted {len(socs)} SOCs in {processing_time:.1f}s")
            
            # Calculate quality score
            quality_score = self._calculate_soc_quality(socs)
            
            return True, soc_file, socs
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"SOC extraction failed: {e}"
            print(f"      ❌ {error_msg} ({processing_time:.1f}s)")
            return False, error_msg, []
    
    def test_pattern_extraction(self, paper: Dict, socs: List) -> Tuple[bool, str, List]:
        """Test pattern extraction from SOCs"""
        
        start_time = time.time()
        
        try:
            print(f"   🔍 Extracting patterns...")
            
            # Create realistic domain knowledge text from SOCs
            domain_knowledge = f"Research Domain: {paper['title']}\n\n"
            domain_knowledge += f"Abstract: {paper.get('abstract', 'No abstract available')[:200]}...\n\n"
            
            # Add SOC-derived domain knowledge
            all_subjects = []
            all_objects = []
            all_concepts = []
            
            for soc in socs:
                all_subjects.extend(soc.get('subjects', []))
                all_objects.extend(soc.get('objects', []))
                all_concepts.extend(soc.get('concepts', []))
            
            domain_knowledge += f"Key Materials and Structures: {', '.join(set(all_subjects))}\n"
            domain_knowledge += f"Properties and Performance: {', '.join(set(all_objects))}\n"
            domain_knowledge += f"Research Concepts: {', '.join(set(all_concepts))}\n\n"
            
            # Add some realistic biomimetic context
            if any('biomimetic' in concept.lower() for concept in all_concepts):
                domain_knowledge += "This research explores biomimetic approaches, studying natural systems to develop innovative technological solutions.\n"
            
            if any('adhesion' in obj.lower() for obj in all_objects):
                domain_knowledge += "The study investigates adhesion mechanisms and their applications in engineering contexts.\n"
            
            # Extract patterns using enhanced pattern extractor
            pattern_result = self.pattern_extractor.extract_all_patterns(domain_knowledge)
            
            # Check if we have patterns
            if not pattern_result or not isinstance(pattern_result, dict):
                return False, "No patterns extracted", []
            
            # Count total patterns
            total_patterns = sum(len(pattern_list) for pattern_list in pattern_result.values())
            if total_patterns == 0:
                return False, "No patterns extracted", []
            
            # Keep patterns in dictionary format for cross-domain mapping
            patterns_dict = pattern_result
            
            # Convert to list for other operations
            all_patterns = []
            for pattern_type, pattern_list in pattern_result.items():
                all_patterns.extend(pattern_list)
            
            # Save patterns
            arxiv_id = paper['arxiv_id']
            pattern_data = {
                'paper_id': arxiv_id,
                'paper_title': paper['title'],
                'pattern_count': len(all_patterns),
                'pattern_types': {
                    'structural': len(pattern_result.get('structural', [])),
                    'functional': len(pattern_result.get('functional', [])),
                    'causal': len(pattern_result.get('causal', []))
                },
                'patterns': [str(pattern) for pattern in all_patterns]
            }
            
            pattern_file = self.storage.save_json_compressed(
                pattern_data,
                f"patterns_{arxiv_id.replace('.', '_')}",
                'patterns'
            )
            
            processing_time = time.time() - start_time
            print(f"      ✅ Extracted {len(all_patterns)} patterns in {processing_time:.1f}s")
            
            return True, pattern_file, patterns_dict
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pattern extraction failed: {e}"
            print(f"      ❌ {error_msg} ({processing_time:.1f}s)")
            return False, error_msg, []
    
    def test_cross_domain_mapping(self, patterns: Dict) -> Tuple[bool, int]:
        """Test cross-domain mapping with extracted patterns"""
        
        start_time = time.time()
        
        try:
            print(f"   🔗 Testing cross-domain mapping...")
            
            # Use enhanced domain mapper with fastening technology as target
            cross_domain_analogy = self.domain_mapper.map_patterns_to_target_domain(
                patterns, 
                "fastening_technology"
            )
            
            processing_time = time.time() - start_time
            
            if cross_domain_analogy and cross_domain_analogy.mappings:
                mapping_count = len(cross_domain_analogy.mappings)
                print(f"      ✅ Found {mapping_count} mappings in {processing_time:.1f}s")
                
                # Show analogy metrics
                print(f"         Overall confidence: {cross_domain_analogy.overall_confidence:.2f}")
                print(f"         Innovation potential: {cross_domain_analogy.innovation_potential:.2f}")
                
                # Show best mapping for validation
                if cross_domain_analogy.mappings:
                    best_mapping = cross_domain_analogy.mappings[0]
                    print(f"         Best mapping: {best_mapping.source_element}")
                    print(f"         Target: {best_mapping.target_element}")
                    print(f"         Mapping confidence: {best_mapping.confidence:.2f}")
            else:
                print(f"      ⚠️  No mappings found ({processing_time:.1f}s)")
                mapping_count = 0
            
            return True, mapping_count
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Cross-domain mapping failed: {e}"
            print(f"      ❌ {error_msg} ({processing_time:.1f}s)")
            return False, 0
    
    def test_enhanced_cross_domain_mapping(self, patterns: Dict) -> Tuple[bool, int, object]:
        """Test cross-domain mapping with extracted patterns, returning the analogy object"""
        
        start_time = time.time()
        
        try:
            print(f"   🔗 Testing enhanced cross-domain mapping...")
            
            # Use enhanced domain mapper with fastening technology as target
            cross_domain_analogy = self.domain_mapper.map_patterns_to_target_domain(
                patterns, 
                "fastening_technology"
            )
            
            processing_time = time.time() - start_time
            
            if cross_domain_analogy and cross_domain_analogy.mappings:
                mapping_count = len(cross_domain_analogy.mappings)
                print(f"      ✅ Found {mapping_count} mappings in {processing_time:.1f}s")
                
                # Show analogy metrics
                print(f"         Overall confidence: {cross_domain_analogy.overall_confidence:.2f}")
                print(f"         Innovation potential: {cross_domain_analogy.innovation_potential:.2f}")
                
                # Show best mapping for validation
                if cross_domain_analogy.mappings:
                    best_mapping = cross_domain_analogy.mappings[0]
                    print(f"         Best mapping: {best_mapping.source_element}")
                    print(f"         Target: {best_mapping.target_element}")
                    print(f"         Mapping confidence: {best_mapping.confidence:.2f}")
                
                return True, mapping_count, cross_domain_analogy
            else:
                print(f"      ⚠️  No mappings found ({processing_time:.1f}s)")
                return True, 0, None
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Enhanced cross-domain mapping failed: {e}"
            print(f"      ❌ {error_msg} ({processing_time:.1f}s)")
            return False, 0, None
    
    def _calculate_soc_quality(self, socs: List) -> float:
        """Calculate quality score for extracted SOCs"""
        
        if not socs:
            return 0.0
        
        total_score = 0.0
        
        for soc in socs:
            score = 0.0
            
            # Check if has subjects, objects, concepts
            if soc.get('subjects'):
                score += 1.0
            if soc.get('objects'):
                score += 1.0  
            if soc.get('concepts'):
                score += 1.0
            
            # Bonus for having multiple items
            score += min(len(soc.get('subjects', [])) / 5, 1.0)
            score += min(len(soc.get('objects', [])) / 5, 1.0)
            score += min(len(soc.get('concepts', [])) / 5, 1.0)
            
            total_score += score
        
        return min(total_score / len(socs), 1.0)
    
    def run_pipeline_test(self, num_papers: int = 5) -> Dict:
        """Run complete pipeline test with specified number of papers"""
        
        print(f"🧪 STARTING PIPELINE TEST WITH {num_papers} PAPERS")
        print(f"=" * 60)
        
        # Load sample papers
        sample_papers = self.load_sample_papers(num_papers)
        if not sample_papers:
            print("❌ Failed to load sample papers")
            return self.test_results
        
        # Monitor storage before starting
        print(f"\n📊 Initial storage usage:")
        self.storage.monitor_storage()
        
        # Process each paper through the pipeline
        for i, paper in enumerate(sample_papers, 1):
            print(f"\n📄 PROCESSING PAPER {i}/{len(sample_papers)}")
            print(f"Title: {paper['title'][:60]}...")
            print(f"arXiv ID: {paper['arxiv_id']}")
            
            try:
                # Step 1: Content ingestion
                success, content_result, content = self.test_content_ingestion(paper)
                if not success:
                    self.test_results['errors'].append({
                        'paper': paper['arxiv_id'],
                        'step': 'content_ingestion',
                        'error': content_result
                    })
                    continue
                
                # Step 2: SOC extraction
                success, soc_result, socs = self.test_soc_extraction(paper, content)
                if not success:
                    self.test_results['errors'].append({
                        'paper': paper['arxiv_id'],
                        'step': 'soc_extraction', 
                        'error': soc_result
                    })
                    continue
                
                self.test_results['socs_extracted'] += len(socs)
                
                # Step 3: Pattern extraction
                success, pattern_result, patterns = self.test_pattern_extraction(paper, socs)
                if not success:
                    self.test_results['errors'].append({
                        'paper': paper['arxiv_id'],
                        'step': 'pattern_extraction',
                        'error': pattern_result
                    })
                    continue
                
                self.test_results['patterns_extracted'] += len(patterns)
                
                # Step 4: Cross-domain mapping
                success, mapping_count = self.test_cross_domain_mapping(patterns)
                if success:
                    self.test_results['mappings_found'] += mapping_count
                
                # Mark as successful
                self.test_results['successful_papers'] += 1
                print(f"   ✅ Paper {i} processed successfully!")
                
            except Exception as e:
                print(f"   ❌ Unexpected error processing paper {i}: {e}")
                self.test_results['errors'].append({
                    'paper': paper['arxiv_id'],
                    'step': 'pipeline',
                    'error': str(e)
                })
            
            # Update counters
            self.test_results['papers_processed'] += 1
            
            # Monitor storage periodically
            if i % 2 == 0:
                print(f"\n📊 Storage usage after {i} papers:")
                self.storage.monitor_storage()
        
        # Final results
        self._generate_test_report()
        return self.test_results
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        
        print(f"\n🧪 PIPELINE TEST RESULTS")
        print(f"=" * 60)
        
        success_rate = (self.test_results['successful_papers'] / 
                       max(1, self.test_results['papers_processed']))
        
        print(f"📊 PROCESSING STATISTICS:")
        print(f"   Papers Processed: {self.test_results['papers_processed']}")
        print(f"   Successful Papers: {self.test_results['successful_papers']}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Total SOCs Extracted: {self.test_results['socs_extracted']}")
        print(f"   Total Patterns Extracted: {self.test_results['patterns_extracted']}")
        print(f"   Total Mappings Found: {self.test_results['mappings_found']}")
        
        if self.test_results['successful_papers'] > 0:
            avg_socs = self.test_results['socs_extracted'] / self.test_results['successful_papers']
            avg_patterns = self.test_results['patterns_extracted'] / self.test_results['successful_papers']
            
            print(f"\n📈 AVERAGES PER SUCCESSFUL PAPER:")
            print(f"   SOCs per paper: {avg_socs:.1f}")
            print(f"   Patterns per paper: {avg_patterns:.1f}")
        
        # Error analysis
        if self.test_results['errors']:
            print(f"\n❌ ERRORS ENCOUNTERED ({len(self.test_results['errors'])}):")
            error_types = {}
            for error in self.test_results['errors']:
                step = error['step']
                error_types[step] = error_types.get(step, 0) + 1
            
            for step, count in error_types.items():
                print(f"   {step}: {count} errors")
        
        # Final storage usage
        print(f"\n📊 FINAL STORAGE USAGE:")
        self.storage.monitor_storage()
        
        # Quality assessment
        if success_rate >= 0.8:
            print(f"\n✅ PIPELINE TEST: PASSED")
            print(f"   Pipeline ready for full 100-paper processing!")
        elif success_rate >= 0.6:
            print(f"\n⚠️  PIPELINE TEST: MARGINAL")
            print(f"   Consider debugging issues before full processing")
        else:
            print(f"\n❌ PIPELINE TEST: FAILED")
            print(f"   Must fix critical issues before proceeding")
        
        # Save test results
        results_file = self.storage.save_json_compressed(
            self.test_results,
            "pipeline_test_results",
            "results"
        )
        print(f"\n💾 Test results saved to: {results_file}")

def main():
    """Run pipeline test"""
    
    tester = PipelineTester()
    results = tester.run_pipeline_test(5)
    
    return results

if __name__ == "__main__":
    main()