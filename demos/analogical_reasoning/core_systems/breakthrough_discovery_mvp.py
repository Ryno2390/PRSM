#!/usr/bin/env python3
"""
PRSM Breakthrough Discovery MVP
Complete reasoning system for scientific breakthrough discovery

This MVP integrates:
- SOC extraction from research papers
- Pattern recognition and analogical reasoning
- Multi-dimensional breakthrough assessment
- Strategic portfolio analysis
- Economic projections and ROI analysis
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import our validated components
from enhanced_batch_processor import EnhancedBatchProcessor
from enhanced_breakthrough_assessment import EnhancedBreakthroughAssessor, BreakthroughProfile
from multi_dimensional_ranking import BreakthroughRanker
from storage_manager import StorageManager

class MVPMode(Enum):
    QUICK_DISCOVERY = "quick"      # 10-50 papers, fast insights
    STANDARD_ANALYSIS = "standard" # 100-200 papers, comprehensive
    DEEP_PORTFOLIO = "deep"        # 500+ papers, full portfolio analysis
    CUSTOM = "custom"              # User-defined parameters

@dataclass
class MVPConfiguration:
    """Configuration for MVP breakthrough discovery session"""
    mode: MVPMode
    paper_count: int
    organization_type: str  # 'academic', 'industry', 'startup', 'vc_fund'
    target_domains: List[str]
    assessment_focus: str   # 'quality_weighted', 'commercial_focused', 'innovation_focused'
    budget_limit: Optional[float] = None
    time_limit_hours: Optional[float] = None

class BreakthroughDiscoveryMVP:
    """
    MVP of the complete PRSM breakthrough discovery system
    
    Provides a unified interface for:
    1. Scientific paper analysis and SOC extraction
    2. Analogical reasoning and cross-domain mapping
    3. Multi-dimensional breakthrough assessment
    4. Strategic ranking and portfolio optimization
    5. Economic analysis and ROI projections
    """
    
    def __init__(self, config: MVPConfiguration):
        self.config = config
        self.session_id = f"mvp_session_{int(time.time())}"
        
        # Initialize storage for this MVP session
        self.storage = StorageManager(f"mvp_{config.mode.value}_{self.session_id}")
        
        # Initialize the enhanced batch processor with multi-dimensional assessment
        self.processor = EnhancedBatchProcessor(
            test_mode=f"mvp_{config.mode.value}",
            use_multi_dimensional=True,
            organization_type=config.organization_type
        )
        
        # Initialize breakthrough assessor and ranker
        self.assessor = EnhancedBreakthroughAssessor()
        self.ranker = BreakthroughRanker(config.organization_type)
        
        # Session results
        self.session_results = {}
        self.breakthrough_portfolio = []
        self.economic_analysis = {}
        
        print(f"🚀 PRSM Breakthrough Discovery MVP Initialized")
        print(f"   Mode: {config.mode.value.upper()}")
        print(f"   Organization: {config.organization_type}")
        print(f"   Target Papers: {config.paper_count}")
        print(f"   Assessment Focus: {config.assessment_focus}")
    
    def run_discovery_session(self) -> Dict:
        """
        Execute complete breakthrough discovery session
        Returns comprehensive results and analysis
        """
        print(f"\n🧠 STARTING BREAKTHROUGH DISCOVERY SESSION")
        print("=" * 80)
        
        session_start = time.time()
        
        try:
            # Phase 1: Paper Processing and Pattern Discovery
            print(f"📊 PHASE 1: Scientific Paper Analysis")
            paper_results = self._process_papers()
            
            # Phase 2: Multi-Dimensional Assessment
            print(f"\n🔍 PHASE 2: Multi-Dimensional Breakthrough Assessment")
            assessment_results = self._assess_breakthroughs(paper_results)
            
            # Phase 3: Strategic Ranking and Portfolio Analysis
            print(f"\n🎯 PHASE 3: Strategic Portfolio Analysis")
            portfolio_results = self._analyze_portfolio(assessment_results)
            
            # Phase 4: Economic Analysis and ROI Projections
            print(f"\n💰 PHASE 4: Economic Analysis and ROI Projections")
            economic_results = self._generate_economic_analysis(portfolio_results)
            
            # Phase 5: Generate Comprehensive Report
            print(f"\n📋 PHASE 5: Generating Comprehensive Report")
            final_report = self._generate_final_report(
                paper_results, assessment_results, portfolio_results, economic_results
            )
            
            session_duration = time.time() - session_start
            
            print(f"\n✅ BREAKTHROUGH DISCOVERY SESSION COMPLETE")
            print(f"   Duration: {session_duration:.2f} seconds")
            print(f"   Papers analyzed: {paper_results.get('papers_processed', 0)}")
            print(f"   Breakthroughs discovered: {len(assessment_results.get('breakthrough_profiles', []))}")
            print(f"   Investment opportunities: {len(portfolio_results.get('immediate_action', []))}")
            
            return final_report
            
        except Exception as e:
            print(f"\n❌ ERROR in breakthrough discovery session: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "session_id": self.session_id}
    
    def _process_papers(self) -> Dict:
        """Phase 1: Process papers through complete NWTN pipeline"""
        
        # Determine test mode based on MVP mode
        test_mode_mapping = {
            MVPMode.QUICK_DISCOVERY: "phase_a",
            MVPMode.STANDARD_ANALYSIS: "phase_b", 
            MVPMode.DEEP_PORTFOLIO: "scaling_test",
            MVPMode.CUSTOM: "phase_b"
        }
        
        test_mode = test_mode_mapping[self.config.mode]
        
        print(f"   📥 Processing {self.config.paper_count} papers...")
        print(f"   🔬 Pipeline mode: {test_mode}")
        
        # Execute the processing
        results = self.processor.run_unified_test(
            test_mode=test_mode,
            paper_count=self.config.paper_count,
            paper_source="unique"
        )
        
        if results:
            print(f"   ✅ Paper processing complete")
            print(f"   📊 Success rate: {results['performance_metrics']['success_rate_percent']:.1f}%")
            print(f"   🔍 Patterns extracted: {results['total_patterns']}")
            print(f"   🔗 Mappings generated: {results['total_mappings']}")
        
        return results
    
    def _assess_breakthroughs(self, paper_results: Dict) -> Dict:
        """Phase 2: Multi-dimensional breakthrough assessment"""
        
        breakthrough_profiles = paper_results.get('breakthrough_profiles', [])
        
        print(f"   🧠 Assessing {len(breakthrough_profiles)} breakthrough profiles...")
        print(f"   📋 Using {self.config.organization_type} assessment framework")
        
        # Enhance profiles with additional analysis
        enhanced_profiles = []
        for profile in breakthrough_profiles:
            # Add market readiness assessment
            profile['market_readiness'] = self._assess_market_readiness(profile)
            
            # Add competitive landscape analysis
            profile['competitive_landscape'] = self._analyze_competitive_landscape(profile)
            
            enhanced_profiles.append(profile)
        
        assessment_results = {
            'breakthrough_profiles': enhanced_profiles,
            'assessment_summary': self._generate_assessment_summary(enhanced_profiles),
            'multi_dimensional_rankings': paper_results.get('multi_dimensional_rankings', {})
        }
        
        print(f"   ✅ Multi-dimensional assessment complete")
        print(f"   🎯 Average success probability: {assessment_results['assessment_summary']['avg_success_probability']:.1%}")
        
        return assessment_results
    
    def _analyze_portfolio(self, assessment_results: Dict) -> Dict:
        """Phase 3: Strategic portfolio analysis and optimization"""
        
        profiles = assessment_results['breakthrough_profiles']
        
        print(f"   📊 Analyzing breakthrough portfolio...")
        print(f"   🎯 Focus: {self.config.assessment_focus}")
        
        # Generate portfolio recommendations
        portfolio_recommendations = self._generate_portfolio_recommendations(profiles)
        
        # Optimize portfolio based on organization type and assessment focus
        optimized_portfolio = self._optimize_portfolio(profiles, portfolio_recommendations)
        
        # Generate strategic insights
        strategic_insights = self._generate_strategic_insights(profiles, optimized_portfolio)
        
        portfolio_results = {
            'portfolio_recommendations': portfolio_recommendations,
            'optimized_portfolio': optimized_portfolio,
            'strategic_insights': strategic_insights,
            'risk_analysis': self._analyze_portfolio_risk(profiles)
        }
        
        print(f"   ✅ Portfolio analysis complete")
        print(f"   🚀 Immediate opportunities: {len(portfolio_recommendations.get('immediate_action', []))}")
        print(f"   📈 Strategic bets: {len(portfolio_recommendations.get('strategic_bets', []))}")
        
        return portfolio_results
    
    def _generate_economic_analysis(self, portfolio_results: Dict) -> Dict:
        """Phase 4: Economic analysis and ROI projections"""
        
        print(f"   💰 Generating economic projections...")
        
        # Calculate processing costs
        processing_cost = self.config.paper_count * 0.327  # Cost per paper
        
        # Calculate expected returns
        profiles = portfolio_results['portfolio_recommendations']
        immediate_opportunities = profiles.get('immediate_action', [])
        strategic_bets = profiles.get('strategic_bets', [])
        
        # ROI calculations
        expected_successes = len(immediate_opportunities) * 0.6 + len(strategic_bets) * 0.3
        cost_per_success = processing_cost / expected_successes if expected_successes > 0 else float('inf')
        
        # Market value projections
        market_value_projection = self._project_market_value(immediate_opportunities, strategic_bets)
        
        economic_analysis = {
            'processing_cost': processing_cost,
            'expected_successes': expected_successes,
            'cost_per_success': cost_per_success,
            'market_value_projection': market_value_projection,
            'roi_analysis': self._calculate_roi_scenarios(processing_cost, market_value_projection),
            'investment_recommendations': self._generate_investment_recommendations(portfolio_results)
        }
        
        print(f"   ✅ Economic analysis complete")
        print(f"   💵 Processing cost: ${processing_cost:.2f}")
        print(f"   📈 Expected successes: {expected_successes:.1f}")
        print(f"   🎯 Cost per success: ${cost_per_success:.2f}")
        
        return economic_analysis
    
    def _generate_final_report(self, paper_results: Dict, assessment_results: Dict, 
                              portfolio_results: Dict, economic_results: Dict) -> Dict:
        """Phase 5: Generate comprehensive final report"""
        
        print(f"   📋 Compiling comprehensive report...")
        
        # Executive summary
        executive_summary = self._create_executive_summary(
            paper_results, assessment_results, portfolio_results, economic_results
        )
        
        # Detailed findings
        detailed_findings = {
            'paper_analysis': self._summarize_paper_analysis(paper_results),
            'breakthrough_assessment': self._summarize_breakthrough_assessment(assessment_results),
            'portfolio_analysis': self._summarize_portfolio_analysis(portfolio_results),
            'economic_projections': self._summarize_economic_analysis(economic_results)
        }
        
        # Action items and recommendations
        action_items = self._generate_action_items(portfolio_results, economic_results)
        
        # Save comprehensive results
        final_report = {
            'session_info': {
                'session_id': self.session_id,
                'configuration': {
                    'mode': self.config.mode.value,
                    'paper_count': self.config.paper_count,
                    'organization_type': self.config.organization_type,
                    'assessment_focus': self.config.assessment_focus
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'executive_summary': executive_summary,
            'detailed_findings': detailed_findings,
            'action_items': action_items,
            'raw_data': {
                'paper_results': paper_results,
                'assessment_results': assessment_results,
                'portfolio_results': portfolio_results,
                'economic_results': economic_results
            }
        }
        
        # Save to storage
        report_file = self.storage.save_json_compressed(
            final_report,
            f"mvp_breakthrough_discovery_report_{self.session_id}",
            "results"
        )
        
        print(f"   ✅ Comprehensive report generated")
        print(f"   💾 Saved to: {report_file}")
        
        return final_report
    
    # Helper methods for detailed analysis
    def _assess_market_readiness(self, profile: Dict) -> str:
        """Assess market readiness of a breakthrough"""
        success_prob = profile.get('quality_score', 0)
        category = profile.get('quality_tier', 'unknown')
        
        if success_prob > 0.3 and category in ['high_impact', 'revolutionary']:
            return "Market ready"
        elif success_prob > 0.2:
            return "Near market ready"
        elif success_prob > 0.1:
            return "Development required"
        else:
            return "Research stage"
    
    def _analyze_competitive_landscape(self, profile: Dict) -> str:
        """Analyze competitive landscape for breakthrough"""
        # Simplified competitive analysis based on profile characteristics
        commercial_potential = profile.get('commercial_potential', 0)
        technical_feasibility = profile.get('technical_feasibility', 0)
        
        if commercial_potential > 0.7 and technical_feasibility > 0.6:
            return "Competitive advantage opportunity"
        elif commercial_potential > 0.5:
            return "Moderate competition expected"
        else:
            return "Niche market, limited competition"
    
    def _generate_assessment_summary(self, profiles: List[Dict]) -> Dict:
        """Generate summary of breakthrough assessment"""
        if not profiles:
            return {'avg_success_probability': 0, 'total_count': 0}
        
        success_probs = [p.get('quality_score', 0) for p in profiles]
        
        return {
            'total_count': len(profiles),
            'avg_success_probability': sum(success_probs) / len(success_probs),
            'max_success_probability': max(success_probs),
            'categories': self._count_categories(profiles)
        }
    
    def _count_categories(self, profiles: List[Dict]) -> Dict:
        """Count breakthrough categories"""
        categories = {}
        for profile in profiles:
            cat = profile.get('quality_tier', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
    def _generate_portfolio_recommendations(self, profiles: List[Dict]) -> Dict:
        """Generate portfolio-level recommendations"""
        immediate_action = []
        strategic_bets = []
        research_pipeline = []
        monitoring_list = []
        
        for profile in profiles:
            success_prob = profile.get('quality_score', 0)
            market_readiness = profile.get('market_readiness', 'unknown')
            
            if market_readiness == "Market ready" and success_prob > 0.2:
                immediate_action.append(profile)
            elif success_prob > 0.15:
                strategic_bets.append(profile)
            elif success_prob > 0.1:
                research_pipeline.append(profile)
            else:
                monitoring_list.append(profile)
        
        return {
            'immediate_action': immediate_action,
            'strategic_bets': strategic_bets,
            'research_pipeline': research_pipeline,
            'monitoring_list': monitoring_list
        }
    
    def _optimize_portfolio(self, profiles: List[Dict], recommendations: Dict) -> Dict:
        """Optimize portfolio based on organization type and focus"""
        # Simplified portfolio optimization
        optimized = {
            'high_priority': recommendations['immediate_action'][:5],
            'medium_priority': recommendations['strategic_bets'][:10],
            'low_priority': recommendations['research_pipeline'][:15]
        }
        
        return optimized
    
    def _generate_strategic_insights(self, profiles: List[Dict], optimized_portfolio: Dict) -> List[str]:
        """Generate strategic insights from portfolio analysis"""
        insights = []
        
        total_profiles = len(profiles)
        high_priority_count = len(optimized_portfolio['high_priority'])
        
        if high_priority_count > 0:
            insights.append(f"Found {high_priority_count} high-priority breakthrough opportunities ready for immediate action")
        
        if total_profiles > 50:
            insights.append("Large portfolio provides good risk diversification opportunities")
        
        avg_success = sum(p.get('quality_score', 0) for p in profiles) / len(profiles) if profiles else 0
        if avg_success > 0.15:
            insights.append("Above-average success probabilities indicate strong portfolio quality")
        else:
            insights.append("Conservative success probabilities suggest focus on portfolio approach")
        
        return insights
    
    def _analyze_portfolio_risk(self, profiles: List[Dict]) -> Dict:
        """Analyze portfolio risk distribution"""
        if not profiles:
            return {'risk_distribution': {}, 'overall_risk': 'unknown'}
        
        # Simplified risk analysis
        high_risk = sum(1 for p in profiles if p.get('quality_score', 0) < 0.1)
        medium_risk = sum(1 for p in profiles if 0.1 <= p.get('quality_score', 0) < 0.2)
        low_risk = sum(1 for p in profiles if p.get('quality_score', 0) >= 0.2)
        
        total = len(profiles)
        
        return {
            'risk_distribution': {
                'high_risk': high_risk / total,
                'medium_risk': medium_risk / total,
                'low_risk': low_risk / total
            },
            'overall_risk': 'high' if high_risk / total > 0.7 else 'medium' if medium_risk / total > 0.5 else 'low'
        }
    
    def _project_market_value(self, immediate_opportunities: List[Dict], strategic_bets: List[Dict]) -> Dict:
        """Project market value of breakthrough portfolio"""
        # Simplified market value projection
        immediate_value = len(immediate_opportunities) * 5000000  # $5M average per immediate opportunity
        strategic_value = len(strategic_bets) * 20000000  # $20M average per strategic bet
        
        return {
            'immediate_market_value': immediate_value,
            'strategic_market_value': strategic_value,
            'total_projected_value': immediate_value + strategic_value
        }
    
    def _calculate_roi_scenarios(self, processing_cost: float, market_value: Dict) -> Dict:
        """Calculate ROI scenarios"""
        total_value = market_value['total_projected_value']
        
        conservative_roi = (total_value * 0.1) / processing_cost if processing_cost > 0 else 0
        moderate_roi = (total_value * 0.05) / processing_cost if processing_cost > 0 else 0
        optimistic_roi = (total_value * 0.02) / processing_cost if processing_cost > 0 else 0
        
        return {
            'conservative_scenario': {'success_rate': 0.1, 'roi': conservative_roi},
            'moderate_scenario': {'success_rate': 0.05, 'roi': moderate_roi},
            'optimistic_scenario': {'success_rate': 0.02, 'roi': optimistic_roi}
        }
    
    def _generate_investment_recommendations(self, portfolio_results: Dict) -> List[str]:
        """Generate investment recommendations"""
        recommendations = []
        
        immediate_count = len(portfolio_results['portfolio_recommendations'].get('immediate_action', []))
        strategic_count = len(portfolio_results['portfolio_recommendations'].get('strategic_bets', []))
        
        if immediate_count > 0:
            recommendations.append(f"Prioritize {immediate_count} immediate opportunities for quick wins")
        
        if strategic_count > 5:
            recommendations.append(f"Consider portfolio approach for {strategic_count} strategic bets")
        
        risk_analysis = portfolio_results.get('risk_analysis', {})
        if risk_analysis.get('overall_risk') == 'high':
            recommendations.append("High-risk portfolio requires larger investment fund and longer timeline")
        
        return recommendations
    
    def _create_executive_summary(self, paper_results: Dict, assessment_results: Dict, 
                                 portfolio_results: Dict, economic_results: Dict) -> Dict:
        """Create executive summary"""
        
        papers_processed = paper_results.get('papers_processed', 0)
        breakthroughs_found = len(assessment_results.get('breakthrough_profiles', []))
        immediate_opportunities = len(portfolio_results['portfolio_recommendations'].get('immediate_action', []))
        processing_cost = economic_results.get('processing_cost', 0)
        expected_roi = economic_results['roi_analysis']['moderate_scenario']['roi']
        
        return {
            'key_metrics': {
                'papers_analyzed': papers_processed,
                'breakthroughs_discovered': breakthroughs_found,
                'immediate_opportunities': immediate_opportunities,
                'processing_cost': processing_cost,
                'expected_roi': expected_roi
            },
            'key_findings': [
                f"Analyzed {papers_processed} scientific papers and identified {breakthroughs_found} breakthrough opportunities",
                f"Found {immediate_opportunities} opportunities ready for immediate action",
                f"Processing cost of ${processing_cost:.2f} with projected ROI of {expected_roi:.1f}x",
                "Multi-dimensional assessment provides actionable insights beyond traditional threshold analysis"
            ],
            'strategic_recommendation': self._get_strategic_recommendation(economic_results, portfolio_results)
        }
    
    def _get_strategic_recommendation(self, economic_results: Dict, portfolio_results: Dict) -> str:
        """Get overall strategic recommendation"""
        roi = economic_results['roi_analysis']['moderate_scenario']['roi']
        immediate_count = len(portfolio_results['portfolio_recommendations'].get('immediate_action', []))
        
        if roi > 10 and immediate_count > 3:
            return "Strong investment opportunity - recommend immediate deployment"
        elif roi > 5:
            return "Viable investment opportunity - recommend pilot program"
        elif immediate_count > 0:
            return "Moderate opportunity - focus on immediate wins"
        else:
            return "Research-stage portfolio - recommend longer-term approach"
    
    def _summarize_paper_analysis(self, results: Dict) -> Dict:
        """Summarize paper analysis results"""
        return {
            'processing_summary': {
                'papers_processed': results.get('papers_processed', 0),
                'success_rate': results['performance_metrics']['success_rate_percent'],
                'patterns_extracted': results.get('total_patterns', 0),
                'mappings_generated': results.get('total_mappings', 0)
            },
            'efficiency_metrics': {
                'processing_time': results['performance_metrics']['processing_time_hours'],
                'papers_per_hour': results['performance_metrics'].get('papers_per_hour', 0)
            }
        }
    
    def _summarize_breakthrough_assessment(self, results: Dict) -> Dict:
        """Summarize breakthrough assessment results"""
        return {
            'assessment_overview': results['assessment_summary'],
            'multi_dimensional_rankings': {
                strategy: len(rankings) for strategy, rankings in results['multi_dimensional_rankings'].items()
            }
        }
    
    def _summarize_portfolio_analysis(self, results: Dict) -> Dict:
        """Summarize portfolio analysis results"""
        return {
            'portfolio_breakdown': {
                category: len(items) for category, items in results['portfolio_recommendations'].items()
            },
            'risk_analysis': results['risk_analysis'],
            'strategic_insights': results['strategic_insights']
        }
    
    def _summarize_economic_analysis(self, results: Dict) -> Dict:
        """Summarize economic analysis results"""
        return {
            'cost_analysis': {
                'processing_cost': results['processing_cost'],
                'cost_per_success': results['cost_per_success']
            },
            'roi_projections': results['roi_analysis'],
            'market_value_projection': results['market_value_projection']
        }
    
    def _generate_action_items(self, portfolio_results: Dict, economic_results: Dict) -> List[Dict]:
        """Generate specific action items"""
        action_items = []
        
        immediate_opportunities = portfolio_results['portfolio_recommendations'].get('immediate_action', [])
        if immediate_opportunities:
            action_items.append({
                'priority': 'high',
                'action': f'Evaluate {len(immediate_opportunities)} immediate breakthrough opportunities',
                'timeline': '1-2 weeks',
                'resources_required': 'Technical evaluation team'
            })
        
        strategic_bets = portfolio_results['portfolio_recommendations'].get('strategic_bets', [])
        if strategic_bets:
            action_items.append({
                'priority': 'medium', 
                'action': f'Develop investment strategy for {len(strategic_bets)} strategic opportunities',
                'timeline': '1-2 months',
                'resources_required': 'Investment committee review'
            })
        
        roi = economic_results['roi_analysis']['moderate_scenario']['roi']
        if roi > 5:
            action_items.append({
                'priority': 'high',
                'action': 'Secure funding for breakthrough development based on strong ROI projections',
                'timeline': '2-4 weeks',
                'resources_required': 'Financial planning and leadership approval'
            })
        
        return action_items

# Convenience functions for easy MVP usage
def create_quick_discovery_session(organization_type: str = "industry") -> BreakthroughDiscoveryMVP:
    """Create a quick discovery session (10-50 papers)"""
    config = MVPConfiguration(
        mode=MVPMode.QUICK_DISCOVERY,
        paper_count=25,
        organization_type=organization_type,
        target_domains=["materials", "engineering"],
        assessment_focus="quality_weighted"
    )
    return BreakthroughDiscoveryMVP(config)

def create_standard_analysis_session(organization_type: str = "industry") -> BreakthroughDiscoveryMVP:
    """Create a standard analysis session (100-200 papers)"""
    config = MVPConfiguration(
        mode=MVPMode.STANDARD_ANALYSIS,
        paper_count=150,
        organization_type=organization_type,
        target_domains=["materials", "engineering", "biotech"],
        assessment_focus="commercial_focused"
    )
    return BreakthroughDiscoveryMVP(config)

def create_deep_portfolio_session(organization_type: str = "vc_fund") -> BreakthroughDiscoveryMVP:
    """Create a deep portfolio analysis session (500+ papers)"""
    config = MVPConfiguration(
        mode=MVPMode.DEEP_PORTFOLIO,
        paper_count=500,
        organization_type=organization_type,
        target_domains=["ai", "biotech", "materials", "energy"],
        assessment_focus="portfolio_optimized"
    )
    return BreakthroughDiscoveryMVP(config)

# Main execution for testing
if __name__ == "__main__":
    print("🚀 PRSM Breakthrough Discovery MVP")
    print("=" * 50)
    
    # Demo: Quick discovery session
    print("\n📊 Demo: Quick Discovery Session")
    mvp = create_quick_discovery_session("industry")
    results = mvp.run_discovery_session()
    
    if "error" not in results:
        print(f"\n✅ MVP Demo Complete!")
        print(f"Session ID: {results['session_info']['session_id']}")
        print(f"Breakthroughs found: {results['executive_summary']['key_metrics']['breakthroughs_discovered']}")
        print(f"Immediate opportunities: {results['executive_summary']['key_metrics']['immediate_opportunities']}")
        print(f"Expected ROI: {results['executive_summary']['key_metrics']['expected_roi']:.1f}x")
    else:
        print(f"❌ MVP Demo failed: {results['error']}")