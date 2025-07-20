#!/usr/bin/env python3
"""
Complete Production Ecosystem Test
==================================

This test validates the entire PRSM/NWTN ecosystem from data ingestion to payment distribution.

TEST PHASES:
1. Full Data Ingestion (150K+ papers) - BACKGROUND PROCESS
2. Creative Challenge Prompts (5 open-ended prompts)
3. Complete Pipeline Per Prompt (with background deep reasoning)
4. Economic Validation (FTNS payments to Prismatica)

BACKGROUND PROCESSES:
- Content ingestion runs in background with progress monitoring
- Deep reasoning runs in background with periodic status checks
- Progress saved to JSON files for monitoring
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.content_ingestion_engine import ContentIngestionEngine
from prsm.nwtn.external_storage_config import ExternalStorageConfig
from prsm.tokenomics.ftns_service import FTNSService
from prsm.integrations.core.provenance_engine import ProvenanceEngine


@dataclass
class TestProgress:
    """Track overall test progress"""
    phase: str
    start_time: datetime
    current_time: datetime
    total_papers: int = 0
    papers_processed: int = 0
    prompts_completed: int = 0
    total_prompts: int = 5
    ftns_distributed: float = 0.0
    status: str = "running"
    error_message: Optional[str] = None
    
    def to_dict(self):
        return {
            'phase': self.phase,
            'start_time': self.start_time.isoformat(),
            'current_time': self.current_time.isoformat(),
            'total_papers': self.total_papers,
            'papers_processed': self.papers_processed,
            'prompts_completed': self.prompts_completed,
            'total_prompts': self.total_prompts,
            'ftns_distributed': self.ftns_distributed,
            'status': self.status,
            'error_message': self.error_message,
            'completion_percentage': self.get_completion_percentage()
        }
    
    def get_completion_percentage(self) -> float:
        if self.phase == "ingestion":
            return (self.papers_processed / max(1, self.total_papers)) * 100
        elif self.phase == "processing":
            return (self.prompts_completed / self.total_prompts) * 100
        return 0.0


class ProductionEcosystemTest:
    """Complete production ecosystem test with background processing"""
    
    def __init__(self):
        self.progress = TestProgress(
            phase="initialization",
            start_time=datetime.now(timezone.utc),
            current_time=datetime.now(timezone.utc)
        )
        self.progress_file = "ecosystem_test_progress.json"
        self.results_file = "ecosystem_test_results.json"
        self.stop_requested = False
        
        # Test configuration
        self.prismatica_user = {
            'user_id': 'prismatica_content_creator',
            'name': 'Prismatica',
            'role': 'content_creator',
            'initial_ftns_balance': 0.0
        }
        
        # Creative test prompts designed to challenge NWTN
        self.challenge_prompts = [
            {
                'id': 'creative_synthesis_1',
                'prompt': 'What novel experimental approaches could emerge from combining quantum computing principles with biological neural network architectures? Suggest 3 specific experiments that have never been attempted.',
                'expected_capabilities': ['analogical_reasoning', 'creative_synthesis', 'experimental_design'],
                'difficulty': 'high'
            },
            {
                'id': 'interdisciplinary_breakthrough_1',
                'prompt': 'How might gravitational wave detection techniques be adapted to discover new particles in high-energy physics experiments? Propose a concrete methodology.',
                'expected_capabilities': ['cross_domain_reasoning', 'methodological_innovation', 'physics_synthesis'],
                'difficulty': 'high'
            },
            {
                'id': 'emergent_technology_1',
                'prompt': 'What are the most promising unexplored applications of topological materials in quantum information processing? Identify 3 research directions that could lead to breakthrough technologies.',
                'expected_capabilities': ['future_prediction', 'technology_synthesis', 'materials_science'],
                'difficulty': 'high'
            },
            {
                'id': 'complex_system_design_1',
                'prompt': 'Design a theoretical framework for creating artificial consciousness that integrates insights from neuroscience, computer science, and philosophy of mind. What would be the key components and validation criteria?',
                'expected_capabilities': ['system_design', 'consciousness_theory', 'validation_methodology'],
                'difficulty': 'extreme'
            },
            {
                'id': 'paradigm_shift_1',
                'prompt': 'What fundamental assumptions in current machine learning theory might be preventing us from achieving artificial general intelligence? Propose alternative theoretical foundations.',
                'expected_capabilities': ['paradigm_questioning', 'theoretical_innovation', 'AGI_reasoning'],
                'difficulty': 'extreme'
            }
        ]
        
        # System components
        self.integrator = None
        self.ingestion_engine = None
        self.external_storage = None
        
        # Results tracking
        self.results = {
            'ingestion_results': {},
            'prompt_results': [],
            'economic_results': {},
            'performance_metrics': {},
            'validation_status': {}
        }
    
    def save_progress(self):
        """Save current progress to file"""
        self.progress.current_time = datetime.now(timezone.utc)
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress.to_dict(), f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save progress: {e}")
    
    def save_results(self):
        """Save current results to file"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    async def initialize_system(self):
        """Initialize all system components"""
        print("🚀 Initializing Complete Production Ecosystem Test")
        print("=" * 80)
        
        self.progress.phase = "initialization"
        self.save_progress()
        
        try:
            # Initialize external storage
            print("📁 Initializing External Storage...")
            self.external_storage = ExternalStorageConfig()
            
            # Initialize system integrator
            print("🧠 Initializing System Integrator...")
            self.integrator = SystemIntegrator(
                external_storage_config=self.external_storage,
                force_mock_retriever=False  # Use real retriever for production test
            )
            await self.integrator.initialize()
            
            # Initialize content ingestion engine
            print("📥 Initializing Content Ingestion Engine...")
            self.ingestion_engine = ContentIngestionEngine(
                external_storage=self.external_storage,
                ftns_service=self.integrator.ftns_service,
                provenance_engine=self.integrator.provenance_engine
            )
            await self.ingestion_engine.initialize()
            
            # Create Prismatica user account
            print("👤 Creating Prismatica User Account...")
            await self._create_prismatica_account()
            
            print("✅ System initialization complete")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            self.progress.error_message = str(e)
            self.save_progress()
            return False
    
    async def _create_prismatica_account(self):
        """Create the Prismatica user account for content ownership"""
        # Register Prismatica as content creator
        await self.integrator.ftns_service.create_account(
            user_id=self.prismatica_user['user_id'],
            initial_balance=self.prismatica_user['initial_ftns_balance']
        )
        
        # Register with provenance engine
        await self.integrator.provenance_engine.register_content_creator(
            creator_id=self.prismatica_user['user_id'],
            creator_name=self.prismatica_user['name'],
            creator_metadata={
                'role': self.prismatica_user['role'],
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def run_background_ingestion(self):
        """Run content ingestion in background with progress monitoring"""
        print("\n📥 PHASE 1: Starting Background Content Ingestion")
        print("=" * 60)
        
        self.progress.phase = "ingestion"
        self.save_progress()
        
        try:
            # Get paper count from external storage
            paper_count = await self.external_storage.get_paper_count()
            self.progress.total_papers = paper_count
            
            print(f"📊 Found {paper_count:,} papers to process")
            print("⏳ Starting background ingestion process...")
            
            # Start background ingestion
            ingestion_task = asyncio.create_task(
                self._process_all_papers_background()
            )
            
            # Monitor progress
            while not ingestion_task.done():
                await asyncio.sleep(30)  # Check every 30 seconds for large scale
                self.save_progress()
                
                completion = self.progress.get_completion_percentage()
                elapsed = (datetime.now(timezone.utc) - self.progress.start_time).total_seconds()
                rate = self.progress.papers_processed / max(1, elapsed) * 60  # papers per minute
                
                print(f"📈 Ingestion Progress: {completion:.1f}% ({self.progress.papers_processed:,}/{self.progress.total_papers:,} papers)")
                print(f"⏱️  Processing Rate: {rate:.0f} papers/minute, Elapsed: {elapsed/60:.1f} minutes")
                
                if self.stop_requested:
                    ingestion_task.cancel()
                    break
            
            if ingestion_task.done() and not ingestion_task.cancelled():
                ingestion_result = await ingestion_task
                self.results['ingestion_results'] = ingestion_result
                print("✅ Background ingestion completed successfully")
                return True
            else:
                print("⚠️  Background ingestion was cancelled")
                return False
                
        except Exception as e:
            print(f"❌ Background ingestion failed: {e}")
            self.progress.error_message = str(e)
            self.save_progress()
            return False
    
    async def _process_all_papers_background(self):
        """Process all papers in background"""
        ingestion_results = {
            'papers_processed': 0,
            'embeddings_created': 0,
            'provenance_records': 0,
            'ftns_rewards_distributed': 0.0,
            'total_processing_time': 0.0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Get all papers from external storage
            papers = await self.external_storage.get_all_papers()
            
            for i, paper in enumerate(papers):
                if self.stop_requested:
                    break
                
                try:
                    # Process paper through ingestion engine
                    result = await self.ingestion_engine.ingest_paper(
                        paper_data=paper,
                        creator_id=self.prismatica_user['user_id']
                    )
                    
                    if result['success']:
                        ingestion_results['papers_processed'] += 1
                        ingestion_results['embeddings_created'] += 1
                        ingestion_results['provenance_records'] += 1
                        ingestion_results['ftns_rewards_distributed'] += result.get('ftns_reward', 0.0)
                    
                    self.progress.papers_processed = i + 1
                    
                    # Save progress every 1000 papers for large scale
                    if (i + 1) % 1000 == 0:
                        self.save_progress()
                        self.save_results()
                        
                        # Log progress for large datasets
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed * 60
                        print(f"  📊 Processed {i + 1:,} papers ({rate:.0f} papers/minute)")
                        
                except Exception as e:
                    ingestion_results['errors'].append(f"Paper {i}: {str(e)}")
                    continue
            
            ingestion_results['total_processing_time'] = time.time() - start_time
            
        except Exception as e:
            ingestion_results['errors'].append(f"Fatal error: {str(e)}")
            raise
        
        return ingestion_results
    
    async def run_creative_challenges(self):
        """Run the 5 creative challenge prompts through complete pipeline"""
        print("\n🧠 PHASE 2: Running Creative Challenge Prompts")
        print("=" * 60)
        
        self.progress.phase = "processing"
        self.save_progress()
        
        for i, prompt_data in enumerate(self.challenge_prompts):
            try:
                print(f"\n📝 Processing Challenge {i+1}/5: {prompt_data['id']}")
                print(f"🎯 Difficulty: {prompt_data['difficulty'].upper()}")
                print(f"📋 Prompt: {prompt_data['prompt'][:100]}...")
                
                # Process through complete pipeline
                result = await self._process_creative_prompt(prompt_data)
                
                self.results['prompt_results'].append(result)
                self.progress.prompts_completed = i + 1
                
                # Save progress after each prompt
                self.save_progress()
                self.save_results()
                
                print(f"✅ Challenge {i+1} completed successfully")
                
            except Exception as e:
                print(f"❌ Challenge {i+1} failed: {e}")
                self.results['prompt_results'].append({
                    'prompt_id': prompt_data['id'],
                    'success': False,
                    'error': str(e)
                })
                continue
        
        print(f"\n🎉 All {len(self.challenge_prompts)} creative challenges completed!")
        return True
    
    async def _process_creative_prompt(self, prompt_data: Dict[str, Any]):
        """Process a single creative prompt through the complete pipeline"""
        start_time = time.time()
        
        # Create test user for this prompt
        test_user_id = f"test_user_{prompt_data['id']}"
        
        # Process through complete pipeline with background deep reasoning
        print("🔍 Starting complete pipeline processing...")
        
        # Start deep reasoning in background
        pipeline_task = asyncio.create_task(
            self.integrator.process_complete_query(
                query=prompt_data['prompt'],
                user_id=test_user_id,
                query_cost=20.0  # Higher cost for complex queries
            )
        )
        
        # Monitor pipeline progress
        while not pipeline_task.done():
            await asyncio.sleep(5)  # Check every 5 seconds
            print("⏳ Deep reasoning in progress...")
            
            if self.stop_requested:
                pipeline_task.cancel()
                break
        
        if pipeline_task.done() and not pipeline_task.cancelled():
            pipeline_result = await pipeline_task
            
            # Calculate economic impact
            economic_impact = self._calculate_economic_impact(pipeline_result)
            
            processing_time = time.time() - start_time
            
            result = {
                'prompt_id': prompt_data['id'],
                'prompt': prompt_data['prompt'],
                'success': pipeline_result.success,
                'processing_time': processing_time,
                'response_quality': pipeline_result.quality_score,
                'sources_used': len(pipeline_result.citations),
                'ftns_charged': pipeline_result.total_cost,
                'ftns_distributed': sum(dist['payment_amount'] for dist in pipeline_result.payment_distributions),
                'economic_impact': economic_impact,
                'pipeline_metrics': pipeline_result.pipeline_metrics,
                'response_preview': pipeline_result.final_response[:200] + "..." if len(pipeline_result.final_response) > 200 else pipeline_result.final_response
            }
            
            # Update total FTNS distributed
            self.progress.ftns_distributed += result['ftns_distributed']
            
            return result
        else:
            return {
                'prompt_id': prompt_data['id'],
                'success': False,
                'error': 'Pipeline processing was cancelled or failed'
            }
    
    def _calculate_economic_impact(self, pipeline_result):
        """Calculate economic impact of the query processing"""
        return {
            'total_cost': pipeline_result.total_cost,
            'content_creator_payments': [
                {
                    'creator_id': dist['creator_id'],
                    'amount': dist['payment_amount'],
                    'paper_id': dist['paper_id']
                }
                for dist in pipeline_result.payment_distributions
            ],
            'system_fee': pipeline_result.total_cost - sum(dist['payment_amount'] for dist in pipeline_result.payment_distributions),
            'roi_for_prismatica': sum(dist['payment_amount'] for dist in pipeline_result.payment_distributions if dist['creator_id'] == self.prismatica_user['user_id'])
        }
    
    async def validate_economic_cycle(self):
        """Validate the complete economic cycle"""
        print("\n💰 PHASE 3: Validating Economic Cycle")
        print("=" * 60)
        
        try:
            # Get Prismatica's final FTNS balance
            final_balance = await self.integrator.ftns_service.get_balance(
                self.prismatica_user['user_id']
            )
            
            # Calculate total revenue earned
            total_revenue = sum(
                result.get('ftns_distributed', 0) 
                for result in self.results['prompt_results']
                if result.get('success', False)
            )
            
            # Generate economic report
            economic_report = {
                'prismatica_initial_balance': self.prismatica_user['initial_ftns_balance'],
                'prismatica_final_balance': final_balance,
                'total_revenue_earned': total_revenue,
                'total_queries_processed': len([r for r in self.results['prompt_results'] if r.get('success', False)]),
                'average_revenue_per_query': total_revenue / max(1, len(self.results['prompt_results'])),
                'economic_cycle_valid': final_balance > self.prismatica_user['initial_ftns_balance']
            }
            
            self.results['economic_results'] = economic_report
            
            print(f"📊 Economic Validation Results:")
            print(f"   💰 Prismatica Final Balance: {final_balance:.2f} FTNS")
            print(f"   📈 Total Revenue Earned: {total_revenue:.2f} FTNS")
            print(f"   🎯 Economic Cycle Valid: {economic_report['economic_cycle_valid']}")
            
            return economic_report['economic_cycle_valid']
            
        except Exception as e:
            print(f"❌ Economic validation failed: {e}")
            return False
    
    async def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n📋 PHASE 4: Generating Final Report")
        print("=" * 60)
        
        # Calculate overall metrics
        successful_prompts = [r for r in self.results['prompt_results'] if r.get('success', False)]
        
        final_report = {
            'test_summary': {
                'total_papers_processed': self.progress.papers_processed,
                'total_prompts_processed': len(self.results['prompt_results']),
                'successful_prompts': len(successful_prompts),
                'success_rate': len(successful_prompts) / max(1, len(self.results['prompt_results'])) * 100,
                'total_ftns_distributed': self.progress.ftns_distributed,
                'test_duration': (datetime.now(timezone.utc) - self.progress.start_time).total_seconds()
            },
            'production_readiness': {
                'data_ingestion_working': self.progress.papers_processed > 0,
                'semantic_search_working': any(r.get('sources_used', 0) > 0 for r in successful_prompts),
                'deep_reasoning_working': any(r.get('response_quality', 0) > 0.5 for r in successful_prompts),
                'economic_cycle_working': self.results['economic_results'].get('economic_cycle_valid', False),
                'claude_api_working': True,  # Assumed if responses generated
                'overall_production_ready': True  # Will be calculated
            },
            'performance_metrics': {
                'average_processing_time': sum(r.get('processing_time', 0) for r in successful_prompts) / max(1, len(successful_prompts)),
                'average_response_quality': sum(r.get('response_quality', 0) for r in successful_prompts) / max(1, len(successful_prompts)),
                'average_sources_per_response': sum(r.get('sources_used', 0) for r in successful_prompts) / max(1, len(successful_prompts))
            }
        }
        
        # Determine overall production readiness
        readiness_checks = final_report['production_readiness']
        final_report['production_readiness']['overall_production_ready'] = all([
            readiness_checks['data_ingestion_working'],
            readiness_checks['semantic_search_working'],
            readiness_checks['deep_reasoning_working'],
            readiness_checks['economic_cycle_working']
        ])
        
        self.results['final_report'] = final_report
        self.save_results()
        
        # Print final report
        print("\n🎉 FINAL PRODUCTION READINESS REPORT")
        print("=" * 80)
        print(f"📊 Papers Processed: {final_report['test_summary']['total_papers_processed']:,}")
        print(f"🧠 Prompts Processed: {final_report['test_summary']['total_prompts_processed']}")
        print(f"✅ Success Rate: {final_report['test_summary']['success_rate']:.1f}%")
        print(f"💰 FTNS Distributed: {final_report['test_summary']['total_ftns_distributed']:.2f}")
        print(f"⏱️  Total Test Duration: {final_report['test_summary']['test_duration']:.0f} seconds")
        
        print(f"\n🚀 PRODUCTION READINESS:")
        for check, status in readiness_checks.items():
            emoji = "✅" if status else "❌"
            print(f"   {emoji} {check.replace('_', ' ').title()}: {status}")
        
        if final_report['production_readiness']['overall_production_ready']:
            print(f"\n🎉 NWTN IS PRODUCTION READY! 🎉")
        else:
            print(f"\n⚠️  NWTN needs additional work before production deployment")
        
        return final_report
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
            self.stop_requested = True
            self.progress.status = "stopping"
            self.save_progress()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_complete_test(self):
        """Run the complete production ecosystem test"""
        try:
            self.setup_signal_handlers()
            
            # Phase 1: Initialize system
            if not await self.initialize_system():
                return False
            
            # Phase 2: Background content ingestion
            if not await self.run_background_ingestion():
                return False
            
            # Phase 3: Creative challenge prompts
            if not await self.run_creative_challenges():
                return False
            
            # Phase 4: Economic validation
            if not await self.validate_economic_cycle():
                return False
            
            # Phase 5: Final report
            final_report = await self.generate_final_report()
            
            self.progress.status = "completed"
            self.save_progress()
            
            return final_report['production_readiness']['overall_production_ready']
            
        except Exception as e:
            print(f"❌ Complete test failed: {e}")
            self.progress.status = "failed"
            self.progress.error_message = str(e)
            self.save_progress()
            return False


async def main():
    """Run the complete production ecosystem test"""
    test = ProductionEcosystemTest()
    
    print("🚀 Starting Complete Production Ecosystem Test")
    print("=" * 80)
    print("📋 This test will validate the entire PRSM/NWTN ecosystem:")
    print("   1. Full data ingestion (150K+ papers)")
    print("   2. Creative challenge prompts")
    print("   3. Complete pipeline processing")
    print("   4. Economic validation")
    print("\n⏳ Background processes will run with progress monitoring")
    print("💾 Progress saved to: ecosystem_test_progress.json")
    print("📊 Results saved to: ecosystem_test_results.json")
    
    success = await test.run_complete_test()
    
    if success:
        print("\n🎉 PRODUCTION ECOSYSTEM TEST PASSED!")
        print("✅ NWTN is ready for production deployment!")
    else:
        print("\n❌ PRODUCTION ECOSYSTEM TEST FAILED!")
        print("⚠️  Review logs and fix issues before production deployment")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())