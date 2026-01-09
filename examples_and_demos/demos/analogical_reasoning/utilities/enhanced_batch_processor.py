#!/usr/bin/env python3
"""
Enhanced Batch Processor with Integrated Quality Ranking
Processes papers through complete NWTN pipeline WITH breakthrough quality assessment
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import traceback

# Import our validated pipeline components
from pipeline_tester import PipelineTester
from storage_manager import StorageManager
from breakthrough_ranker import BreakthroughRanker, RankedBreakthrough
from calibrated_breakthrough_ranker import CalibratedBreakthroughRanker
from enhanced_breakthrough_assessment import EnhancedBreakthroughAssessor, BreakthroughProfile
from multi_dimensional_ranking import BreakthroughRanker as MultiDimensionalRanker

class EnhancedBatchProcessor:
    """
    Unified enhanced batch processor with:
    - Integrated breakthrough quality ranking
    - Calibrated quality scoring (Phase A)
    - Multi-tier relevance analysis (Phase B)
    - Configurable test modes
    """
    
    def __init__(self, test_mode: str = "standard", use_calibrated_scoring: bool = False, 
                 use_multi_dimensional: bool = True, organization_type: str = "industry"):
        self.test_mode = test_mode  # "standard", "phase_a", "phase_b", "baseline"
        self.use_calibrated_scoring = use_calibrated_scoring
        self.use_multi_dimensional = use_multi_dimensional
        self.organization_type = organization_type
        
        self.storage = StorageManager(f"enhanced_{test_mode}_processing")
        self.pipeline_tester = PipelineTester()
        
        # Choose assessment system based on configuration
        if use_multi_dimensional:
            self.breakthrough_assessor = EnhancedBreakthroughAssessor()
            self.breakthrough_ranker = MultiDimensionalRanker(organization_type)
            print(f"🧠 Using multi-dimensional breakthrough assessment ({organization_type})")
        elif use_calibrated_scoring:
            self.breakthrough_ranker = CalibratedBreakthroughRanker()
            print(f"🔧 Using calibrated breakthrough ranking")
        else:
            self.breakthrough_ranker = BreakthroughRanker()
            print(f"📊 Using standard breakthrough ranking")
        
        # Enhanced batch processing metrics
        self.batch_results = {
            'start_time': None,
            'end_time': None,
            'papers_processed': 0,
            'successful_papers': 0,
            'total_socs': 0,
            'total_patterns': 0,
            'total_mappings': 0,
            'total_discoveries': 0,
            'processing_errors': [],
            'performance_metrics': {},
            
            # Multi-dimensional quality metrics
            'quality_metrics': {
                # Legacy threshold metrics (for compatibility)
                'breakthrough_tier': 0,
                'high_potential_tier': 0,
                'promising_tier': 0,
                'moderate_tier': 0,
                'low_priority_tier': 0,
                'minimal_tier': 0,
                'average_quality_score': 0.0,
                'top_quality_score': 0.0,
                
                # New multi-dimensional metrics
                'revolutionary_discoveries': 0,
                'high_impact_discoveries': 0,
                'incremental_discoveries': 0,
                'speculative_discoveries': 0,
                'niche_discoveries': 0,
                
                'immediate_opportunities': 0,   # 0-2 years
                'near_term_opportunities': 0,   # 2-5 years
                'long_term_opportunities': 0,   # 5-10 years
                'blue_sky_opportunities': 0,    # 10+ years
                
                'low_risk_count': 0,
                'moderate_risk_count': 0,
                'high_risk_count': 0,
                'extreme_risk_count': 0,
                
                'avg_success_probability': 0.0,
                'portfolio_balance_score': 0.0
            },
            
            # Enhanced breakthrough analysis
            'breakthrough_profiles': [],        # Full multi-dimensional profiles
            'ranked_breakthroughs': [],        # Legacy format for compatibility
            'multi_dimensional_rankings': {    # Rankings by different strategies
                'quality_weighted': [],
                'risk_adjusted': [],
                'commercial_focused': [],
                'innovation_focused': [],
                'portfolio_optimized': []
            },
            'top_breakthroughs': [],           # Top 10 by primary strategy
            'commercial_opportunities': [],    # High commercial potential
            'innovation_opportunities': [],    # High innovation potential
            'immediate_opportunities': [],     # Ready for implementation
            
            # NEW: Multi-tier relevance analysis (Phase B)
            'relevance_analysis': {
                'tier_1_direct': {'papers': [], 'breakthroughs': [], 'avg_quality': 0.0},
                'tier_2_adjacent': {'papers': [], 'breakthroughs': [], 'avg_quality': 0.0},
                'tier_3_distant': {'papers': [], 'breakthroughs': [], 'avg_quality': 0.0},
                'tier_comparison': {}
            },
            
            'pattern_catalog': {},
            'discovery_confidence_scores': []
        }
        
        # Domain relevance keywords for Phase B
        self.relevance_tiers = {
            'tier_1_direct': {
                'keywords': ['adhesion', 'fastening', 'attachment', 'bonding', 'stick', 'grip', 'fasten', 'bond', 'adhere'],
                'description': 'Direct Relevance - Fastening/Adhesion papers'
            },
            'tier_2_adjacent': {
                'keywords': ['biomimetic', 'bio-inspired', 'surface', 'interface', 'contact', 'gecko', 'spider', 'material'],
                'description': 'Adjacent Domains - Biomimetics/Materials papers'
            },
            'tier_3_distant': {
                'keywords': [],  # Everything else
                'description': 'Distant Domains - All other scientific papers'
            }
        }
        
        # Papers will be loaded based on test mode
        self.papers = []
        
    def load_papers_for_test(self, count: int = 100, source: str = "unique") -> List[Dict]:
        """Load papers based on test mode and source"""
        
        if source == "unique":
            return self._load_unique_papers(count)
        elif source == "original":
            return self._load_original_papers(count)
        else:
            raise ValueError(f"Unknown paper source: {source}")
    
    def _load_unique_papers(self, count: int) -> List[Dict]:
        """Load unique papers from collection"""
        
        try:
            with open('unique_papers_collection.json', 'r') as f:
                data = json.load(f)
            
            papers = data['papers'][:count]
            print(f"✅ Loaded {len(papers)} unique papers")
            return papers
            
        except Exception as e:
            print(f"❌ Error loading unique papers: {e}")
            return []
    
    def _load_original_papers(self, count: int) -> List[Dict]:
        """Load original papers from selection"""
        
        try:
            with open('selected_papers.json', 'r') as f:
                papers = json.load(f)
            
            papers = papers[:count]
            print(f"✅ Loaded {len(papers)} original papers")
            return papers
            
        except Exception as e:
            print(f"❌ Error loading original papers: {e}")
            return []
    
    def categorize_paper_by_relevance(self, paper: Dict) -> str:
        """Categorize paper by relevance tier for Phase B analysis"""
        
        # Combine title and abstract for analysis
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Check Tier 1: Direct relevance
        tier_1_matches = sum(1 for keyword in self.relevance_tiers['tier_1_direct']['keywords'] 
                           if keyword in text)
        if tier_1_matches >= 2:  # Need multiple matches for direct relevance
            return 'tier_1_direct'
        
        # Check Tier 2: Adjacent domains
        tier_2_matches = sum(1 for keyword in self.relevance_tiers['tier_2_adjacent']['keywords'] 
                           if keyword in text)
        if tier_2_matches >= 1:  # Single match sufficient for adjacent
            return 'tier_2_adjacent'
        
        # Default: Tier 3: Distant domains
        return 'tier_3_distant'
    
    def run_unified_test(self, test_mode: str, paper_count: int = 100, paper_source: str = "unique") -> Dict:
        """
        Unified test runner for all test modes:
        - phase_a: Calibrated quality scoring test
        - phase_b: Multi-tier relevance analysis 
        - baseline: Standard scoring baseline
        - comparison: Phase A vs Phase B comparison
        """
        
        print(f"\n🚀 STARTING UNIFIED TEST: {test_mode.upper()}")
        print(f"=" * 80)
        
        # Configure test mode
        self.test_mode = test_mode
        
        if test_mode == "phase_a":
            print(f"🔧 PHASE A: Calibrated Quality Scoring Test")
            print(f"📊 Purpose: Establish realistic quality baseline with conservative scoring")
            if not self.use_multi_dimensional:
                self.use_calibrated_scoring = True
                self.breakthrough_ranker = CalibratedBreakthroughRanker()
            
        elif test_mode == "phase_b":
            print(f"🎯 PHASE B: Multi-Tier Relevance Analysis")
            print(f"📊 Purpose: Compare discovery rates across domain relevance tiers")
            if not self.use_multi_dimensional:
                self.use_calibrated_scoring = True
                self.breakthrough_ranker = CalibratedBreakthroughRanker()
            
        elif test_mode == "baseline":
            print(f"📈 BASELINE: Standard Scoring Test")
            print(f"📊 Purpose: Original scoring for comparison")
            if not self.use_multi_dimensional:
                self.use_calibrated_scoring = False
                self.breakthrough_ranker = BreakthroughRanker()
            
        elif test_mode == "comparison":
            print(f"⚖️ COMPARISON: Phase A vs Phase B Analysis") 
            print(f"📊 Purpose: Direct comparison of calibrated vs tiered approaches")
            if not self.use_multi_dimensional:
                self.use_calibrated_scoring = True
                self.breakthrough_ranker = CalibratedBreakthroughRanker()
        
        # Load papers
        self.papers = self.load_papers_for_test(paper_count, paper_source)
        if not self.papers:
            print("❌ No papers loaded. Cannot proceed.")
            return {}
        
        # Phase B: Categorize papers by relevance tier
        if test_mode in ["phase_b", "comparison"]:
            self._categorize_papers_by_tier()
            
            # Process each tier separately for Phase B
            if test_mode == "phase_b":
                self._process_phase_b_by_tiers()
                return self.batch_results
        
        # Execute standard processing for non-Phase B tests
        results = self.run_enhanced_batch_processing()
        
        # Mode-specific analysis
        if test_mode == "phase_a":
            self._analyze_phase_a_results()
        elif test_mode == "comparison":
            self._analyze_comparison_results()
        
        return results
    
    def _categorize_papers_by_tier(self):
        """Categorize all papers by relevance tier for Phase B analysis"""
        
        print(f"\n🎯 CATEGORIZING PAPERS BY RELEVANCE TIER:")
        
        for paper in self.papers:
            tier = self.categorize_paper_by_relevance(paper)
            self.batch_results['relevance_analysis'][tier]['papers'].append(paper)
        
        # Show distribution
        for tier_name, tier_data in self.batch_results['relevance_analysis'].items():
            if tier_name != 'tier_comparison':
                count = len(tier_data['papers'])
                description = self.relevance_tiers[tier_name]['description']
                print(f"   {tier_name}: {count} papers ({description})")
    
    def run_enhanced_batch_processing(self) -> Dict:
        """Execute enhanced 100-paper batch processing with quality ranking"""
        
        print(f"\n🚀 STARTING ENHANCED 100-PAPER BATCH PROCESSING")
        print(f"==" * 35)
        print(f"Processing {len(self.papers)} papers through complete NWTN pipeline")
        print(f"Expected patterns: ~{len(self.papers) * 2.6:.0f}")
        print(f"Expected cost: ${len(self.papers) * 0.327:.2f}")
        print(f"✨ NEW: Integrated breakthrough quality ranking and assessment")
        
        self.batch_results['start_time'] = time.time()
        
        # Initial storage check
        print(f"\n📊 Initial storage status:")
        self.storage.monitor_storage()
        
        # Process papers in batches for memory management
        batch_size = 20  # Process 20 papers at a time
        total_batches = (len(self.papers) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(self.papers))
            batch_papers = self.papers[start_idx:end_idx]
            
            print(f"\n📦 PROCESSING BATCH {batch_num + 1}/{total_batches}")
            print(f"Papers {start_idx + 1}-{end_idx} ({len(batch_papers)} papers)")
            
            self._process_enhanced_paper_batch(batch_papers, batch_num + 1)
            
            # Storage monitoring every 2 batches
            if (batch_num + 1) % 2 == 0:
                print(f"\n📊 Storage status after batch {batch_num + 1}:")
                storage_ok = self.storage.monitor_storage()
                
                if not storage_ok:
                    print("⚠️ Storage getting low, running cleanup...")
                    self.storage.cleanup_storage(target_free_gb=1.0)
        
        # Final processing and analysis with quality ranking
        self._finalize_enhanced_batch_processing()
        
        return self.batch_results
    
    def _process_enhanced_paper_batch(self, papers: List[Dict], batch_num: int):
        """Process a batch of papers with quality assessment"""
        
        batch_start_time = time.time()
        
        for i, paper in enumerate(papers):
            paper_num = (batch_num - 1) * 20 + i + 1
            
            print(f"\n📄 PAPER {paper_num}/100")
            print(f"Title: {paper['title'][:50]}...")
            print(f"arXiv ID: {paper['arxiv_id']}")
            
            try:
                # Process through complete pipeline with enhanced assessment
                success, quality_score, assessment_result = self._process_enhanced_single_paper(paper, paper_num)
                
                if success:
                    self.batch_results['successful_papers'] += 1
                    print(f"   ✅ Paper {paper_num} completed successfully")
                    
                    # Handle both multi-dimensional and legacy assessment results
                    if assessment_result:
                        if self.use_multi_dimensional and isinstance(assessment_result, BreakthroughProfile):
                            # Multi-dimensional assessment
                            self.batch_results['breakthrough_profiles'].append(assessment_result)
                            print(f"   🧠 Success Probability: {quality_score:.1%} ({assessment_result.category.value}, {assessment_result.risk_level.value} risk)")
                            
                            # Update multi-dimensional metrics
                            self._update_multi_dimensional_metrics(assessment_result)
                            
                            # Create legacy compatibility entry
                            legacy_breakthrough = self._create_legacy_breakthrough(assessment_result)
                            self.batch_results['ranked_breakthroughs'].append(legacy_breakthrough)
                            
                        else:
                            # Legacy assessment
                            self.batch_results['ranked_breakthroughs'].append(assessment_result)
                            print(f"   🏆 Quality Score: {quality_score:.3f} ({assessment_result.breakthrough_score.quality_tier})")
                            
                            # Update legacy quality tier counts
                            tier = assessment_result.breakthrough_score.quality_tier
                            tier_key = tier.lower() + '_tier'
                            if tier_key in self.batch_results['quality_metrics']:
                                self.batch_results['quality_metrics'][tier_key] += 1
                else:
                    print(f"   ❌ Paper {paper_num} failed processing")
                
                self.batch_results['papers_processed'] += 1
                
                # Progress update every 10 papers
                if paper_num % 10 == 0:
                    self._print_enhanced_progress_update(paper_num)
                
            except Exception as e:
                print(f"   💥 Unexpected error processing paper {paper_num}: {e}")
                self.batch_results['processing_errors'].append({
                    'paper_id': paper['arxiv_id'],
                    'paper_num': paper_num,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        batch_time = time.time() - batch_start_time
        print(f"\n📊 Batch {batch_num} completed in {batch_time:.1f}s")
    
    def _process_enhanced_single_paper(self, paper: Dict, paper_num: int) -> Tuple[bool, float, RankedBreakthrough]:
        """Process a single paper with integrated quality assessment"""
        
        try:
            # Step 1: Content ingestion
            success, content_result, content = self.pipeline_tester.test_content_ingestion(paper)
            if not success:
                return False, 0.0, None
            
            # Step 2: SOC extraction
            success, soc_result, socs = self.pipeline_tester.test_soc_extraction(paper, content)
            if not success:
                return False, 0.0, None
            
            self.batch_results['total_socs'] += len(socs)
            
            # Step 3: Pattern extraction
            success, pattern_result, patterns = self.pipeline_tester.test_pattern_extraction(paper, socs)
            if not success:
                return False, 0.0, None
            
            # Count patterns
            if isinstance(patterns, dict):
                pattern_count = sum(len(pattern_list) for pattern_list in patterns.values())
                self.batch_results['total_patterns'] += pattern_count
                
                # Store patterns in catalog
                self.batch_results['pattern_catalog'][paper['arxiv_id']] = {
                    'paper_title': paper['title'],
                    'pattern_count': pattern_count,
                    'pattern_types': {k: len(v) for k, v in patterns.items()}
                }
            
            # Step 4: Cross-domain mapping with quality assessment
            success, mapping_count, cross_domain_analogy = self.pipeline_tester.test_enhanced_cross_domain_mapping(patterns)
            if success:
                self.batch_results['total_mappings'] += mapping_count
                
                # Count as discovery if mappings found
                if mapping_count > 0:
                    self.batch_results['total_discoveries'] += 1
                
                # NEW: Step 5: Enhanced breakthrough assessment
                if cross_domain_analogy and mapping_count > 0:
                    if self.use_multi_dimensional:
                        breakthrough_profile = self._assess_breakthrough_multi_dimensional(paper, patterns, cross_domain_analogy)
                        quality_score = breakthrough_profile.success_probability  # Use success probability as primary score
                        return True, quality_score, breakthrough_profile
                    else:
                        ranked_breakthrough = self._assess_breakthrough_quality(paper, patterns, cross_domain_analogy)
                        quality_score = ranked_breakthrough.breakthrough_score.overall_score
                        return True, quality_score, ranked_breakthrough
            
            return True, 0.0, None
            
        except Exception as e:
            print(f"      ❌ Enhanced pipeline error: {e}")
            return False, 0.0, None
    
    def _assess_breakthrough_quality(self, paper: Dict, patterns: Dict, cross_domain_analogy) -> RankedBreakthrough:
        """Assess breakthrough quality using integrated ranking system"""
        
        # Create mapping data for quality assessment
        mapping_data = {
            'source_papers': [paper['arxiv_id']],
            'target_domain': cross_domain_analogy.target_domain,
            'confidence': cross_domain_analogy.overall_confidence,
            'innovation_potential': cross_domain_analogy.innovation_potential,
            'description': paper['title'],
            'source_patterns': [str(p) for pattern_list in patterns.values() for p in pattern_list],
            'target_applications': [m.target_element for m in cross_domain_analogy.mappings[:3]],
            'key_innovations': self._extract_innovations_from_paper(paper),
            'testable_predictions': self._generate_predictions_from_mappings(cross_domain_analogy.mappings),
            'constraints': [c for m in cross_domain_analogy.mappings for c in m.constraints],
            'reasoning': f"Cross-domain analogical mapping from: {paper['title']}"
        }
        
        # Use breakthrough ranker to assess quality
        ranked_breakthrough = self.breakthrough_ranker.rank_breakthrough(mapping_data)
        
        return ranked_breakthrough
    
    def _assess_breakthrough_multi_dimensional(self, paper: Dict, patterns: Dict, cross_domain_analogy) -> BreakthroughProfile:
        """Multi-dimensional breakthrough assessment using enhanced system"""
        
        # Create mapping data for multi-dimensional assessment
        mapping_data = {
            'discovery_id': f"discovery_{paper['arxiv_id']}",
            'source_papers': [paper['arxiv_id']],
            'target_domain': cross_domain_analogy.target_domain,
            'confidence': cross_domain_analogy.overall_confidence,
            'innovation_potential': cross_domain_analogy.innovation_potential,
            'description': paper['title'],
            'source_patterns': [str(p) for pattern_list in patterns.values() for p in pattern_list],
            'target_applications': [m.target_element for m in cross_domain_analogy.mappings[:3]],
            'key_innovations': self._extract_innovations_from_paper(paper),
            'testable_predictions': self._generate_predictions_from_mappings(cross_domain_analogy.mappings),
            'constraints': [c for m in cross_domain_analogy.mappings for c in m.constraints],
            'reasoning': f"Cross-domain analogical mapping from: {paper['title']}"
        }
        
        # Use enhanced breakthrough assessor
        context = {'organization_type': self.organization_type}
        breakthrough_profile = self.breakthrough_assessor.assess_breakthrough(mapping_data, context)
        
        return breakthrough_profile
    
    def _extract_innovations_from_paper(self, paper: Dict) -> List[str]:
        """Extract key innovations from paper content"""
        title = paper['title'].lower()
        abstract = paper.get('abstract', '').lower()
        all_text = f"{title} {abstract}"
        
        innovations = []
        
        if any(word in all_text for word in ['biomimetic', 'bio-inspired', 'biological']):
            innovations.append("biomimetic_design_approach")
        if any(word in all_text for word in ['adhesion', 'adhesive', 'attachment']):
            innovations.append("advanced_adhesion_mechanism")
        if any(word in all_text for word in ['surface', 'interface', 'contact']):
            innovations.append("engineered_surface_properties")
        if any(word in all_text for word in ['optimization', 'enhanced', 'improved']):
            innovations.append("performance_optimization")
        if any(word in all_text for word in ['gecko', 'spider', 'insect']):
            innovations.append("biologically_inspired_mechanics")
        if any(word in all_text for word in ['nano', 'micro', 'scale']):
            innovations.append("multi_scale_design_principles")
        
        return innovations if innovations else ["cross_domain_pattern_transfer"]
    
    def _generate_predictions_from_mappings(self, mappings: List) -> List[str]:
        """Generate testable predictions from analogical mappings"""
        predictions = []
        
        if not mappings:
            return ["system_performance_validation_required"]
        
        for mapping in mappings[:3]:  # Top 3 mappings
            if hasattr(mapping, 'mapping_type'):
                if mapping.mapping_type == 'structural':
                    predictions.append("Structural features will replicate source mechanism")
                elif mapping.mapping_type == 'functional':
                    predictions.append("Functional performance will transfer to target domain")
                elif mapping.mapping_type == 'causal':
                    predictions.append("Causal relationships will be preserved in implementation")
            
            if hasattr(mapping, 'confidence') and mapping.confidence > 0.7:
                predictions.append("High-confidence mapping enables reliable prediction")
        
        # Add general predictions
        predictions.extend([
            "Cross-domain analogical transfer will be experimentally validatable",
            "Implementation will demonstrate measurable performance improvement"
        ])
        
        return list(set(predictions))  # Remove duplicates
    
    def _update_multi_dimensional_metrics(self, profile: BreakthroughProfile):
        """Update multi-dimensional quality metrics"""
        
        metrics = self.batch_results['quality_metrics']
        
        # Update category counts
        category_map = {
            'REVOLUTIONARY': 'revolutionary_discoveries',
            'HIGH_IMPACT': 'high_impact_discoveries',
            'INCREMENTAL': 'incremental_discoveries',
            'SPECULATIVE': 'speculative_discoveries',
            'NICHE': 'niche_discoveries'
        }
        
        category_key = category_map.get(profile.category.value.upper())
        if category_key:
            metrics[category_key] += 1
        
        # Update time horizon counts
        time_map = {
            '0-2_years': 'immediate_opportunities',
            '2-5_years': 'near_term_opportunities',
            '5-10_years': 'long_term_opportunities',
            '10+_years': 'blue_sky_opportunities'
        }
        
        time_key = time_map.get(profile.time_horizon.value)
        if time_key:
            metrics[time_key] += 1
        
        # Update risk counts
        risk_map = {
            'low': 'low_risk_count',
            'moderate': 'moderate_risk_count',
            'high': 'high_risk_count',
            'extreme': 'extreme_risk_count'
        }
        
        risk_key = risk_map.get(profile.risk_level.value)
        if risk_key:
            metrics[risk_key] += 1
    
    def _create_legacy_breakthrough(self, profile: BreakthroughProfile):
        """Create legacy RankedBreakthrough for compatibility"""
        
        # Create a simplified legacy object for compatibility
        class LegacyBreakthrough:
            def __init__(self, profile):
                self.discovery_id = profile.discovery_id
                self.discovery_description = profile.description
                self.source_papers = profile.source_papers
                
                # Create simplified breakthrough score
                class SimpleScore:
                    def __init__(self, profile):
                        self.overall_score = profile.success_probability
                        self.scientific_novelty = profile.scientific_novelty
                        self.commercial_potential = profile.commercial_potential
                        self.technical_feasibility = profile.technical_feasibility
                        self.evidence_quality = profile.evidence_strength
                        
                        # Map to legacy quality tiers based on success probability
                        if profile.success_probability >= 0.8:
                            self.quality_tier = "BREAKTHROUGH"
                        elif profile.success_probability >= 0.6:
                            self.quality_tier = "HIGH_POTENTIAL"
                        elif profile.success_probability >= 0.4:
                            self.quality_tier = "PROMISING"
                        elif profile.success_probability >= 0.2:
                            self.quality_tier = "MODERATE"
                        else:
                            self.quality_tier = "MINIMAL"
                        
                        self.recommendation = f"{profile.category.value} breakthrough, {profile.time_horizon.value} timeline"
                
                self.breakthrough_score = SimpleScore(profile)
        
        return LegacyBreakthrough(profile)
    
    def _finalize_multi_dimensional_analysis(self) -> Tuple[float, float]:
        """Finalize multi-dimensional breakthrough analysis"""
        
        profiles = self.batch_results['breakthrough_profiles']
        
        if not profiles:
            return 0.0, 0.0
        
        # Calculate aggregate metrics
        success_probabilities = [p.success_probability for p in profiles]
        avg_success_probability = sum(success_probabilities) / len(success_probabilities)
        top_success_probability = max(success_probabilities)
        
        # Update metrics
        self.batch_results['quality_metrics']['avg_success_probability'] = avg_success_probability
        self.batch_results['quality_metrics']['top_quality_score'] = top_success_probability
        self.batch_results['quality_metrics']['average_quality_score'] = avg_success_probability
        
        # Generate multi-dimensional rankings
        ranking_strategies = ['quality_weighted', 'risk_adjusted', 'commercial_focused', 'innovation_focused', 'portfolio_optimized']
        
        for strategy in ranking_strategies:
            ranked = self.breakthrough_ranker.rank_breakthroughs(profiles, strategy=strategy)
            # Store top 10 for each strategy
            self.batch_results['multi_dimensional_rankings'][strategy] = ranked[:10]
        
        # Set primary rankings (using quality_weighted as default)
        primary_ranked = self.batch_results['multi_dimensional_rankings']['quality_weighted']
        self.batch_results['top_breakthroughs'] = [item[0] for item in primary_ranked]
        
        # Identify specialized opportunities
        self.batch_results['commercial_opportunities'] = [
            p for p in profiles if p.commercial_potential >= 0.7
        ]
        
        self.batch_results['innovation_opportunities'] = [
            p for p in profiles if p.scientific_novelty >= 0.7
        ]
        
        self.batch_results['immediate_opportunities'] = [
            p for p in profiles 
            if p.time_horizon.value == '0-2_years' and p.risk_level.value in ['low', 'moderate']
        ]
        
        return avg_success_probability, top_success_probability
    
    def _finalize_legacy_analysis(self) -> Tuple[float, float]:
        """Finalize legacy breakthrough analysis"""
        
        breakthroughs = self.batch_results['ranked_breakthroughs']
        
        if not breakthroughs:
            return 0.0, 0.0
        
        quality_scores = [b.breakthrough_score.overall_score for b in breakthroughs]
        avg_quality_score = sum(quality_scores) / len(quality_scores)
        top_quality_score = max(quality_scores)
        
        # Update quality metrics
        self.batch_results['quality_metrics']['average_quality_score'] = avg_quality_score
        self.batch_results['quality_metrics']['top_quality_score'] = top_quality_score
        
        # Sort breakthroughs by quality
        self.batch_results['top_breakthroughs'] = sorted(
            breakthroughs, 
            key=lambda x: x.breakthrough_score.overall_score, 
            reverse=True
        )[:10]
        
        # Identify commercial opportunities (score ≥ 0.6)
        self.batch_results['commercial_opportunities'] = [
            b for b in breakthroughs 
            if b.breakthrough_score.overall_score >= 0.6
        ]
        
        return avg_quality_score, top_quality_score
    
    def _print_multi_dimensional_results(self):
        """Print multi-dimensional breakthrough analysis results"""
        
        metrics = self.batch_results['quality_metrics']
        profiles = self.batch_results['breakthrough_profiles']
        
        print(f"\n🧠 MULTI-DIMENSIONAL BREAKTHROUGH ANALYSIS:")
        print(f"   Total breakthrough profiles: {len(profiles)}")
        print(f"   Average success probability: {metrics['avg_success_probability']:.1%}")
        print(f"   Top success probability: {metrics['top_quality_score']:.1%}")
        
        # Category distribution
        print(f"\n📊 BREAKTHROUGH CATEGORIES:")
        categories = ['revolutionary_discoveries', 'high_impact_discoveries', 'incremental_discoveries', 
                     'speculative_discoveries', 'niche_discoveries']
        for category in categories:
            count = metrics.get(category, 0)
            if count > 0:
                category_name = category.replace('_discoveries', '').replace('_', ' ').title()
                print(f"   {category_name}: {count}")
        
        # Time horizon distribution
        print(f"\n⏰ TIME HORIZON DISTRIBUTION:")
        horizons = ['immediate_opportunities', 'near_term_opportunities', 'long_term_opportunities', 'blue_sky_opportunities']
        for horizon in horizons:
            count = metrics.get(horizon, 0)
            if count > 0:
                horizon_name = horizon.replace('_opportunities', '').replace('_', ' ').title()
                print(f"   {horizon_name}: {count}")
        
        # Risk distribution
        print(f"\n⚠️ RISK DISTRIBUTION:")
        risks = ['low_risk_count', 'moderate_risk_count', 'high_risk_count', 'extreme_risk_count']
        for risk in risks:
            count = metrics.get(risk, 0)
            if count > 0:
                risk_name = risk.replace('_risk_count', '').replace('_', ' ').title() + " Risk"
                print(f"   {risk_name}: {count}")
        
        # Strategic opportunities
        commercial_count = len(self.batch_results['commercial_opportunities'])
        innovation_count = len(self.batch_results['innovation_opportunities'])
        immediate_count = len(self.batch_results['immediate_opportunities'])
        
        print(f"\n🎯 STRATEGIC OPPORTUNITIES:")
        print(f"   High commercial potential (≥0.7): {commercial_count}")
        print(f"   High innovation potential (≥0.7): {innovation_count}")
        print(f"   Immediate implementation ready: {immediate_count}")
        
        # Top rankings by different strategies
        print(f"\n🏆 TOP BREAKTHROUGHS BY STRATEGY:")
        
        rankings = self.batch_results['multi_dimensional_rankings']
        for strategy, ranked_list in rankings.items():
            if ranked_list:
                top_breakthrough = ranked_list[0]
                profile = top_breakthrough[0]
                score = top_breakthrough[1]
                strategy_name = strategy.replace('_', ' ').title()
                print(f"   {strategy_name}: {profile.discovery_id} (Score: {score:.3f})")
                print(f"      Category: {profile.category.value}, Risk: {profile.risk_level.value}")
    
    def _print_legacy_quality_results(self):
        """Print legacy quality analysis results"""
        
        metrics = self.batch_results['performance_metrics']
        quality_metrics = self.batch_results['quality_metrics']
        
        print(f"\n🏆 QUALITY ANALYSIS:")
        print(f"   Quality breakthroughs analyzed: {metrics.get('quality_breakthroughs_count', 0)}")
        print(f"   Average quality score: {metrics.get('avg_quality_score', 0):.3f}")
        print(f"   Top quality score: {metrics.get('top_quality_score', 0):.3f}")
        print(f"   Commercial opportunities (≥0.6): {metrics.get('commercial_opportunities_count', 0)}")
        
        # Quality tier distribution
        print(f"\n📊 QUALITY TIER DISTRIBUTION:")
        tiers = ['breakthrough', 'high_potential', 'promising', 'moderate', 'low_priority', 'minimal']
        for tier in tiers:
            tier_key = tier + '_tier'
            count = quality_metrics.get(tier_key, 0)
            if count > 0:
                print(f"   {tier.upper()}: {count} discoveries")
        
        # Top breakthroughs
        top_breakthroughs = self.batch_results['top_breakthroughs']
        if top_breakthroughs:
            print(f"\n🚀 TOP 3 QUALITY BREAKTHROUGHS:")
            for i, breakthrough in enumerate(top_breakthroughs[:3], 1):
                print(f"   #{i}: Score {breakthrough.breakthrough_score.overall_score:.3f} "
                      f"({breakthrough.breakthrough_score.quality_tier})")
                print(f"       {breakthrough.discovery_description[:60]}...")
    
    def _print_enhanced_progress_update(self, paper_num: int):
        """Print enhanced progress update with quality metrics"""
        
        success_rate = (self.batch_results['successful_papers'] / paper_num) * 100
        avg_patterns = self.batch_results['total_patterns'] / max(1, self.batch_results['successful_papers'])
        avg_mappings = self.batch_results['total_mappings'] / max(1, self.batch_results['successful_papers'])
        
        # Calculate quality metrics
        breakthroughs = self.batch_results['ranked_breakthroughs']
        if breakthroughs:
            avg_quality = sum(b.breakthrough_score.overall_score for b in breakthroughs) / len(breakthroughs)
            top_quality = max(b.breakthrough_score.overall_score for b in breakthroughs)
            
            # Count quality tiers
            tier_counts = {}
            for breakthrough in breakthroughs:
                tier = breakthrough.breakthrough_score.quality_tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        else:
            avg_quality = 0.0
            top_quality = 0.0
            tier_counts = {}
        
        print(f"\n📈 ENHANCED PROGRESS UPDATE - {paper_num}/100 papers")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Total SOCs: {self.batch_results['total_socs']}")
        print(f"   Total patterns: {self.batch_results['total_patterns']}")
        print(f"   Total mappings: {self.batch_results['total_mappings']}")
        print(f"   Papers with discoveries: {self.batch_results['total_discoveries']}")
        
        # NEW: Quality metrics
        print(f"\n🏆 QUALITY METRICS:")
        print(f"   Average quality score: {avg_quality:.3f}")
        print(f"   Top quality score: {top_quality:.3f}")
        print(f"   Quality breakthroughs: {len(breakthroughs)}")
        
        if tier_counts:
            print(f"   Quality distribution:")
            for tier, count in sorted(tier_counts.items()):
                print(f"      {tier}: {count}")
    
    def _finalize_enhanced_batch_processing(self):
        """Finalize enhanced batch processing with comprehensive quality analysis"""
        
        self.batch_results['end_time'] = time.time()
        processing_time = self.batch_results['end_time'] - self.batch_results['start_time']
        
        print(f"\n🎯 ENHANCED BATCH PROCESSING COMPLETE")
        print(f"==" * 35)
        
        # Calculate final metrics
        success_rate = (self.batch_results['successful_papers'] / 
                       self.batch_results['papers_processed']) * 100
        
        discovery_rate = (self.batch_results['total_discoveries'] / 
                         max(1, self.batch_results['successful_papers'])) * 100
        
        avg_patterns_per_paper = (self.batch_results['total_patterns'] / 
                                 max(1, self.batch_results['successful_papers']))
        
        avg_mappings_per_paper = (self.batch_results['total_mappings'] / 
                                 max(1, self.batch_results['successful_papers']))
        
        # Enhanced quality analysis based on assessment type
        if self.use_multi_dimensional:
            avg_quality_score, top_quality_score = self._finalize_multi_dimensional_analysis()
        else:
            avg_quality_score, top_quality_score = self._finalize_legacy_analysis()
        
        # Store performance metrics
        self.batch_results['performance_metrics'] = {
            'total_processing_time_seconds': processing_time,
            'processing_time_hours': processing_time / 3600,
            'papers_per_hour': self.batch_results['papers_processed'] / (processing_time / 3600),
            'success_rate_percent': success_rate,
            'discovery_rate_percent': discovery_rate,
            'avg_patterns_per_paper': avg_patterns_per_paper,
            'avg_mappings_per_paper': avg_mappings_per_paper,
            'patterns_per_hour': self.batch_results['total_patterns'] / (processing_time / 3600),
            'mappings_per_hour': self.batch_results['total_mappings'] / (processing_time / 3600),
            
            # NEW: Quality performance metrics
            'avg_quality_score': avg_quality_score,
            'top_quality_score': top_quality_score,
            'quality_breakthroughs_count': len(self.batch_results.get('breakthrough_profiles', [])) if self.use_multi_dimensional else len(self.batch_results.get('ranked_breakthroughs', [])),
            'commercial_opportunities_count': len(self.batch_results['commercial_opportunities'])
        }
        
        # Print comprehensive results
        self._print_enhanced_final_results()
        
        # Save enhanced results
        self._save_enhanced_batch_results()
    
    def _print_enhanced_final_results(self):
        """Print comprehensive final results with quality analysis"""
        
        metrics = self.batch_results['performance_metrics']
        
        print(f"📊 PROCESSING STATISTICS:")
        print(f"   Papers processed: {self.batch_results['papers_processed']}")
        print(f"   Successful papers: {self.batch_results['successful_papers']}")
        print(f"   Success rate: {metrics['success_rate_percent']:.1f}%")
        print(f"   Processing time: {metrics['processing_time_hours']:.2f} hours")
        print(f"   Papers per hour: {metrics['papers_per_hour']:.1f}")
        
        print(f"\n🔬 DISCOVERY STATISTICS:")
        print(f"   Total SOCs extracted: {self.batch_results['total_socs']}")
        print(f"   Total patterns generated: {self.batch_results['total_patterns']}")
        print(f"   Total cross-domain mappings: {self.batch_results['total_mappings']}")
        print(f"   Papers with discoveries: {self.batch_results['total_discoveries']}")
        print(f"   Discovery rate: {metrics['discovery_rate_percent']:.1f}%")
        
        # Enhanced Quality Analysis
        if self.use_multi_dimensional:
            self._print_multi_dimensional_results()
        else:
            self._print_legacy_quality_results()
        
        print(f"\n📈 EFFICIENCY METRICS:")
        print(f"   Average patterns per paper: {metrics['avg_patterns_per_paper']:.1f}")
        print(f"   Average mappings per paper: {metrics['avg_mappings_per_paper']:.1f}")
        print(f"   Patterns generated per hour: {metrics['patterns_per_hour']:.0f}")
        print(f"   Mappings generated per hour: {metrics['mappings_per_hour']:.0f}")
        
        if self.batch_results['processing_errors']:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            print(f"   Total errors: {len(self.batch_results['processing_errors'])}")
            print(f"   Error rate: {(len(self.batch_results['processing_errors']) / self.batch_results['papers_processed']) * 100:.1f}%")
        
        # Final storage status
        print(f"\n📊 FINAL STORAGE STATUS:")
        self.storage.monitor_storage()
        
        # Enhanced success assessment
        if metrics['success_rate_percent'] >= 80 and metrics['avg_quality_score'] >= 0.4:
            print(f"\n✅ ENHANCED BATCH PROCESSING: HIGHLY SUCCESSFUL")
            print(f"   Ready for quality-filtered 1K paper scaling!")
        elif metrics['success_rate_percent'] >= 60 and metrics['avg_quality_score'] >= 0.3:
            print(f"\n⚠️ ENHANCED BATCH PROCESSING: SUCCESSFUL WITH QUALITY IMPROVEMENTS NEEDED")
            print(f"   Optimize quality filtering before scaling")
        else:
            print(f"\n❌ ENHANCED BATCH PROCESSING: NEEDS QUALITY ENHANCEMENT")
            print(f"   Address quality ranking issues before scaling")
    
    def _save_enhanced_batch_results(self):
        """Save comprehensive enhanced batch results"""
        
        # Convert RankedBreakthrough objects to serializable format
        serializable_results = dict(self.batch_results)
        
        # Convert breakthrough_profiles to dictionaries (multi-dimensional)
        if 'breakthrough_profiles' in serializable_results:
            serializable_results['breakthrough_profiles'] = [
                self._convert_breakthrough_to_dict(b)
                for b in self.batch_results['breakthrough_profiles']
            ]
        
        # Convert multi_dimensional_rankings to dictionaries
        if 'multi_dimensional_rankings' in serializable_results:
            rankings_dict = {}
            for strategy, ranked_list in self.batch_results['multi_dimensional_rankings'].items():
                rankings_dict[strategy] = [
                    {
                        **self._convert_breakthrough_to_dict(breakthrough),
                        'score': score,
                        'reasoning': reasoning
                    }
                    for breakthrough, score, reasoning in ranked_list
                ]
            serializable_results['multi_dimensional_rankings'] = rankings_dict
        
        # Convert ranked_breakthroughs to dictionaries (legacy)
        if 'ranked_breakthroughs' in serializable_results:
            serializable_results['ranked_breakthroughs'] = [
                self._convert_breakthrough_to_dict(b)
                for b in self.batch_results['ranked_breakthroughs']
            ]
        
        # Convert top_breakthroughs to dictionaries
        if 'top_breakthroughs' in serializable_results:
            serializable_results['top_breakthroughs'] = [
                self._convert_breakthrough_to_dict(b, simple=True)
                for b in self.batch_results['top_breakthroughs']
            ]
        
        # Convert commercial_opportunities to dictionaries
        if 'commercial_opportunities' in serializable_results:
            serializable_results['commercial_opportunities'] = [
                self._convert_breakthrough_to_dict(b, simple=True)
                for b in self.batch_results['commercial_opportunities']
            ]
        
        # Save detailed results with quality analysis
        results_file = self.storage.save_json_compressed(
            serializable_results,
            "enhanced_batch_processing_results_100_papers",
            "results"
        )
        
        # Save top breakthroughs separately for easy access
        top_breakthroughs_data = [
            self._convert_breakthrough_to_dict(b)
            for b in self.batch_results['top_breakthroughs']
        ]
        
        top_breakthroughs_file = self.storage.save_json_compressed(
            top_breakthroughs_data,
            "top_quality_breakthroughs_100_papers", 
            "results"
        )
        
        print(f"\n💾 ENHANCED RESULTS SAVED:")
        print(f"   Enhanced batch results: {results_file}")
        print(f"   Top quality breakthroughs: {top_breakthroughs_file}")
        
        # Create enhanced summary report
        summary = {
            'test_name': '100-Paper Enhanced NWTN Batch Processing Test',
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'papers_processed': self.batch_results['papers_processed'],
            'success_rate': self.batch_results['performance_metrics']['success_rate_percent'],
            'total_patterns': self.batch_results['total_patterns'],
            'total_mappings': self.batch_results['total_mappings'],
            'discovery_rate': self.batch_results['performance_metrics']['discovery_rate_percent'],
            'processing_time_hours': self.batch_results['performance_metrics']['processing_time_hours'],
            
            # NEW: Quality metrics in summary
            'avg_quality_score': self.batch_results['performance_metrics']['avg_quality_score'],
            'top_quality_score': self.batch_results['performance_metrics']['top_quality_score'],
            'commercial_opportunities': self.batch_results['performance_metrics']['commercial_opportunities_count'],
            'quality_recommendation': self._generate_quality_recommendation()
        }
        
        summary_file = self.storage.save_json_compressed(
            summary,
            "100_paper_enhanced_test_summary",
            "results"
        )
        print(f"   Enhanced test summary: {summary_file}")
    
    def _convert_breakthrough_to_dict(self, breakthrough, simple: bool = False) -> Dict:
        """Convert breakthrough object to dictionary, handling both legacy and multi-dimensional formats"""
        
        if self.use_multi_dimensional and hasattr(breakthrough, 'category'):
            # Multi-dimensional BreakthroughProfile
            base_dict = {
                'discovery_id': breakthrough.discovery_id,
                'quality_score': breakthrough.success_probability,
                'quality_tier': breakthrough.category.value,
                'description': breakthrough.description
            }
            
            if not simple:
                base_dict.update({
                    'source_papers': breakthrough.source_papers,
                    'commercial_potential': breakthrough.commercial_potential,
                    'technical_feasibility': breakthrough.technical_feasibility,
                    'scientific_novelty': breakthrough.scientific_novelty,
                    'recommendation': f"{breakthrough.category.value} breakthrough, {breakthrough.risk_level.value} risk, {breakthrough.time_horizon.value} timeline"
                })
        else:
            # Legacy RankedBreakthrough with breakthrough_score
            base_dict = {
                'discovery_id': breakthrough.discovery_id,
                'quality_score': breakthrough.breakthrough_score.overall_score,
                'quality_tier': breakthrough.breakthrough_score.quality_tier,
                'description': breakthrough.discovery_description
            }
            
            if not simple:
                base_dict.update({
                    'source_papers': breakthrough.source_papers,
                    'commercial_potential': breakthrough.breakthrough_score.commercial_potential,
                    'technical_feasibility': breakthrough.breakthrough_score.technical_feasibility,
                    'scientific_novelty': breakthrough.breakthrough_score.scientific_novelty,
                    'recommendation': breakthrough.breakthrough_score.recommendation
                })
                
        return base_dict

    def _generate_quality_recommendation(self) -> str:
        """Generate recommendation based on quality analysis"""
        
        avg_quality = self.batch_results['performance_metrics']['avg_quality_score']
        commercial_ops = self.batch_results['performance_metrics']['commercial_opportunities_count']
        success_rate = self.batch_results['performance_metrics']['success_rate_percent']
        
        if success_rate >= 80 and avg_quality >= 0.5 and commercial_ops >= 5:
            return "FULL GO: High-quality breakthroughs identified - proceed to 1K papers immediately"
        elif success_rate >= 70 and avg_quality >= 0.4 and commercial_ops >= 2:
            return "CONDITIONAL GO: Good quality baseline - optimize filtering then scale to 1K papers"
        elif success_rate >= 60 and avg_quality >= 0.3:
            return "ENHANCE QUALITY: Improve ranking algorithms before scaling"
        else:
            return "REASSESS: Address fundamental quality issues before proceeding"

    def _analyze_phase_a_results(self):
        """Phase A specific analysis with calibrated scoring"""
        
        print(f"\n🔧 PHASE A CALIBRATED ANALYSIS")
        print(f"=" * 70)
        
        breakthroughs = self.batch_results['ranked_breakthroughs']
        papers_processed = self.batch_results['papers_processed']
        
        # Count by calibrated thresholds
        hq_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.75)
        commercial_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.65)
        promising_count = sum(1 for b in breakthroughs if b.breakthrough_score.overall_score >= 0.55)
        
        print(f"📊 CALIBRATED BREAKTHROUGH METRICS:")
        print(f"   Papers processed: {papers_processed} unique papers")
        print(f"   High-quality breakthroughs (≥0.75): {hq_count}")
        print(f"   Commercial opportunities (≥0.65): {commercial_count}")
        print(f"   Promising discoveries (≥0.55): {promising_count}")
        
        # Calculate realistic rates
        hq_rate = (hq_count / papers_processed) * 100
        commercial_rate = (commercial_count / papers_processed) * 100
        promising_rate = (promising_count / papers_processed) * 100
        
        print(f"\n📈 CALIBRATED DISCOVERY RATES:")
        print(f"   High-quality rate: {hq_rate:.1f}% (target: 0.5-2%)")
        print(f"   Commercial rate: {commercial_rate:.1f}% (target: 5-15%)")
        print(f"   Promising rate: {promising_rate:.1f}% (target: 10-25%)")
        
        # Success assessment for Phase A
        if 0.5 <= hq_rate <= 3.0 and 3.0 <= commercial_rate <= 20.0:
            print(f"\n✅ PHASE A SUCCESS: Calibrated scoring working correctly")
            print(f"   🚀 Ready for Phase B domain relevance filtering")
        elif hq_rate == 0 and commercial_rate < 3:
            print(f"\n⚠️ PHASE A: Over-calibrated (too conservative)")
            print(f"   🔧 Slightly relax scoring thresholds for Phase B")
        else:
            print(f"\n❌ PHASE A: Under-calibrated (still too generous)")
            print(f"   🔧 Further tighten scoring before Phase B")

    def _analyze_phase_b_results(self):
        """Phase B specific analysis with tier-based results"""
        
        print(f"\n🎯 PHASE B TIER-BASED ANALYSIS")
        print(f"=" * 70)
        
        # Analyze breakthroughs by tier
        for tier_name, tier_data in self.batch_results['relevance_analysis'].items():
            if tier_name == 'tier_comparison':
                continue
                
            tier_papers = len(tier_data['papers'])
            tier_breakthroughs = len(tier_data['breakthroughs'])
            
            if tier_papers > 0:
                tier_rate = (tier_breakthroughs / tier_papers) * 100
                tier_data['breakthrough_rate'] = tier_rate
                
                description = self.relevance_tiers[tier_name]['description']
                print(f"\n📊 {tier_name.upper()} RESULTS:")
                print(f"   {description}")
                print(f"   Papers: {tier_papers}")
                print(f"   Breakthroughs: {tier_breakthroughs}")
                print(f"   Discovery rate: {tier_rate:.1f}%")
                
                if tier_data['breakthroughs']:
                    scores = [b.breakthrough_score.overall_score for b in tier_data['breakthroughs']]
                    avg_score = sum(scores) / len(scores)
                    tier_data['avg_quality'] = avg_score
                    print(f"   Average quality: {avg_score:.3f}")

    def _analyze_comparison_results(self):
        """Compare Phase A vs Phase B results"""
        
        print(f"\n⚖️ PHASE A vs PHASE B COMPARISON")
        print(f"=" * 70)
        
        # This would compare cached Phase A results with current Phase B results
        print(f"   Phase A: Calibrated scoring across all papers")
        print(f"   Phase B: Tier-based relevance filtering + calibrated scoring")
        print(f"   Comparison: Phase B should show higher discovery rates in relevant tiers")

    def _process_phase_b_by_tiers(self):
        """Process Phase B by analyzing each tier separately"""
        
        print(f"\n🎯 PHASE B: PROCESSING PAPERS BY RELEVANCE TIER")
        print(f"=" * 70)
        
        self.batch_results['start_time'] = time.time()
        
        # Process each tier separately
        for tier_name, tier_data in self.batch_results['relevance_analysis'].items():
            if tier_name == 'tier_comparison' or not tier_data['papers']:
                continue
                
            tier_papers = tier_data['papers']
            tier_description = self.relevance_tiers[tier_name]['description']
            
            print(f"\n📦 PROCESSING {tier_name.upper()}")
            print(f"   {tier_description}")
            print(f"   Papers to process: {len(tier_papers)}")
            
            # Process papers in this tier
            for i, paper in enumerate(tier_papers):
                paper_num = i + 1
                print(f"\n📄 TIER PAPER {paper_num}/{len(tier_papers)}")
                print(f"Title: {paper['title'][:50]}...")
                
                try:
                    success, quality_score, ranked_breakthrough = self._process_enhanced_single_paper(paper, paper_num)
                    
                    if success and ranked_breakthrough:
                        tier_data['breakthroughs'].append(ranked_breakthrough)
                        self.batch_results['ranked_breakthroughs'].append(ranked_breakthrough)
                        print(f"   ✅ Quality Score: {quality_score:.3f} ({ranked_breakthrough.breakthrough_score.quality_tier})")
                    
                    self.batch_results['papers_processed'] += 1
                    if success:
                        self.batch_results['successful_papers'] += 1
                        
                except Exception as e:
                    print(f"   ❌ Error processing tier paper: {e}")
            
            # Tier summary
            tier_breakthroughs = len(tier_data['breakthroughs'])
            tier_rate = (tier_breakthroughs / len(tier_papers)) * 100 if tier_papers else 0
            
            print(f"\n📊 {tier_name.upper()} SUMMARY:")
            print(f"   Papers processed: {len(tier_papers)}")
            print(f"   Breakthroughs found: {tier_breakthroughs}")
            print(f"   Discovery rate: {tier_rate:.1f}%")
            
            if tier_data['breakthroughs']:
                scores = [b.breakthrough_score.overall_score for b in tier_data['breakthroughs']]
                avg_score = sum(scores) / len(scores)
                tier_data['avg_quality'] = avg_score
                print(f"   Average quality: {avg_score:.3f}")
        
        self.batch_results['end_time'] = time.time()
        
        # Final Phase B analysis
        self._finalize_phase_b_processing()

    def _finalize_phase_b_processing(self):
        """Finalize Phase B processing with tier-based analysis"""
        
        processing_time = self.batch_results['end_time'] - self.batch_results['start_time']
        
        print(f"\n🎯 PHASE B PROCESSING COMPLETE")
        print(f"=" * 70)
        
        # Calculate tier-based metrics
        tier_1_papers = len(self.batch_results['relevance_analysis']['tier_1_direct']['papers'])
        tier_1_breakthroughs = len(self.batch_results['relevance_analysis']['tier_1_direct']['breakthroughs'])
        tier_1_rate = (tier_1_breakthroughs / tier_1_papers) * 100 if tier_1_papers > 0 else 0
        
        tier_2_papers = len(self.batch_results['relevance_analysis']['tier_2_adjacent']['papers'])
        tier_2_breakthroughs = len(self.batch_results['relevance_analysis']['tier_2_adjacent']['breakthroughs'])
        tier_2_rate = (tier_2_breakthroughs / tier_2_papers) * 100 if tier_2_papers > 0 else 0
        
        tier_3_papers = len(self.batch_results['relevance_analysis']['tier_3_distant']['papers'])
        tier_3_breakthroughs = len(self.batch_results['relevance_analysis']['tier_3_distant']['breakthroughs'])
        tier_3_rate = (tier_3_breakthroughs / tier_3_papers) * 100 if tier_3_papers > 0 else 0
        
        total_breakthroughs = tier_1_breakthroughs + tier_2_breakthroughs + tier_3_breakthroughs
        overall_rate = (total_breakthroughs / self.batch_results['papers_processed']) * 100
        
        print(f"📊 PHASE B TIER RESULTS:")
        print(f"   Tier 1 (Direct): {tier_1_breakthroughs}/{tier_1_papers} papers ({tier_1_rate:.1f}% rate)")
        print(f"   Tier 2 (Adjacent): {tier_2_breakthroughs}/{tier_2_papers} papers ({tier_2_rate:.1f}% rate)")
        print(f"   Tier 3 (Distant): {tier_3_breakthroughs}/{tier_3_papers} papers ({tier_3_rate:.1f}% rate)")
        print(f"   Overall: {total_breakthroughs}/{self.batch_results['papers_processed']} papers ({overall_rate:.1f}% rate)")
        
        # Store performance metrics
        self.batch_results['performance_metrics'] = {
            'total_processing_time_seconds': processing_time,
            'processing_time_hours': processing_time / 3600,
            'papers_per_hour': self.batch_results['papers_processed'] / (processing_time / 3600),
            'success_rate_percent': (self.batch_results['successful_papers'] / self.batch_results['papers_processed']) * 100,
            'overall_discovery_rate': overall_rate,
            'tier_1_discovery_rate': tier_1_rate,
            'tier_2_discovery_rate': tier_2_rate,
            'tier_3_discovery_rate': tier_3_rate,
            'total_breakthroughs': total_breakthroughs
        }
        
        # Phase B specific analysis
        self._analyze_phase_b_results()
        
        # Success assessment
        if overall_rate >= 5 and tier_1_rate > tier_2_rate:
            print(f"\n✅ PHASE B SUCCESS: Tier-based approach validates relevance filtering")
            print(f"   🚀 Ready for 1K paper scaling with tier-optimized strategy")
        elif overall_rate >= 2:
            print(f"\n⚠️ PHASE B PARTIAL SUCCESS: Some tier differentiation observed")
            print(f"   🔧 Refine tier criteria before scaling")
        else:
            print(f"\n❌ PHASE B NEEDS IMPROVEMENT: Limited tier effectiveness")
            print(f"   🛠️ Reassess relevance categorization approach")

def main():
    """Execute enhanced 100-paper batch processing test"""
    
    processor = EnhancedBatchProcessor()
    
    if not processor.papers:
        print("❌ No papers loaded. Cannot proceed with enhanced batch processing.")
        return
    
    if len(processor.papers) < 100:
        print(f"⚠️ Only {len(processor.papers)} papers available (need 100)")
        response = input("Proceed anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Execute enhanced batch processing
    results = processor.run_enhanced_batch_processing()
    
    print(f"\n🎉 ENHANCED 100-PAPER BATCH PROCESSING COMPLETE!")
    print(f"Check results in enhanced_batch_storage/results/")
    print(f"🏆 Quality analysis integrated throughout pipeline")
    
    return results

if __name__ == "__main__":
    main()