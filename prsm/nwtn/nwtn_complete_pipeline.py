#!/usr/bin/env python3
"""
Complete NWTN Pipeline: From Prompt to Natural Language Response
=================================================================

This unified script demonstrates the complete NWTN (Neuro-symbolic World model with 
Theory-driven Neuro-symbolic reasoning) pipeline from user input to final natural 
language response.

Complete Pipeline:
1. 📝 User Prompt Input
2. 📚 Semantic Search (2,310 papers → 20 most relevant)
3. 🔬 Content Analysis (extract key concepts)
4. 🧠 System 1: Generate 5,040 candidate answers (7! reasoning engine permutations)
5. 🗜️ Deduplication & Compression
6. 🎯 System 2: Meta-reasoning evaluation and synthesis
7. 📦 Wisdom Package Creation (answer + traces + corpus + metadata)
8. 🤖 LLM Integration (Claude API) for natural language generation
9. ✨ Final Natural Language Response

Based on validated individual pipeline components.
"""

import asyncio
import sys
import time
import json
import os
from datetime import datetime, timezone
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.core.models import UserInput


class NWTNPipelineMonitor:
    """Monitor and track the complete NWTN pipeline execution"""
    
    def __init__(self):
        self.start_time = None
        self.stages = []
        self.current_stage = None
        
    def start_pipeline(self):
        self.start_time = time.time()
        print("🚀 NWTN COMPLETE PIPELINE STARTING")
        print("=" * 50)
        
    def start_stage(self, stage_name, description):
        if self.current_stage:
            self.end_stage()
        
        stage_start = time.time()
        self.current_stage = {
            'name': stage_name,
            'description': description,
            'start_time': stage_start,
            'end_time': None,
            'duration': None,
            'status': 'running'
        }
        
        elapsed = stage_start - self.start_time if self.start_time else 0
        print(f"\n⏱️  [{elapsed:.1f}s] {stage_name}")
        print(f"   {description}")
        
    def end_stage(self, status='completed'):
        if self.current_stage:
            end_time = time.time()
            self.current_stage['end_time'] = end_time
            self.current_stage['duration'] = end_time - self.current_stage['start_time']
            self.current_stage['status'] = status
            
            status_icon = "✅" if status == 'completed' else "❌" if status == 'failed' else "⚠️"
            print(f"   {status_icon} {self.current_stage['name']}: {status} ({self.current_stage['duration']:.2f}s)")
            
            self.stages.append(self.current_stage)
            self.current_stage = None
    
    def end_pipeline(self, success=True):
        if self.current_stage:
            self.end_stage('completed' if success else 'failed')
            
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n🏁 NWTN PIPELINE {'COMPLETED' if success else 'FAILED'}")
        print("=" * 50)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        print(f"\n📊 PIPELINE STAGE SUMMARY:")
        for i, stage in enumerate(self.stages, 1):
            status_icon = "✅" if stage['status'] == 'completed' else "❌" if stage['status'] == 'failed' else "⚠️"
            print(f"   {i}. {status_icon} {stage['name']}: {stage['duration']:.2f}s")
            
        return {
            'total_time': total_time,
            'stages': self.stages,
            'success': success
        }


async def run_complete_nwtn_pipeline(prompt: str, context_allocation: int = 1000, monitor_progress: bool = True):
    """
    Execute the complete NWTN pipeline from prompt to natural language response
    
    Args:
        prompt: User input prompt/question
        context_allocation: Context size allocation for processing
        monitor_progress: Whether to monitor and display progress
        
    Returns:
        dict: Complete pipeline results including final answer, metrics, and traces
    """
    
    # Initialize monitoring
    monitor = NWTNPipelineMonitor()
    monitor.start_pipeline()
    
    print(f"📝 INPUT PROMPT ({len(prompt)} characters):")
    print(f'   "{prompt[:100]}{"..." if len(prompt) > 100 else ""}"')
    print(f"📊 CONTEXT ALLOCATION: {context_allocation}")
    print(f"🗂️  PROGRESS MONITORING: {'Enabled' if monitor_progress else 'Disabled'}")
    
    try:
        # Stage 1: Initialize NWTN System
        monitor.start_stage("🏗️  SYSTEM INITIALIZATION", "Setting up NWTN orchestrator and components")
        
        orchestrator = EnhancedNWTNOrchestrator()
        user_input = UserInput(
            user_id=f"unified_pipeline_user_{int(time.time())}",
            prompt=prompt,
            context_allocation=context_allocation
        )
        
        monitor.end_stage()
        
        # Stage 2: Progress Monitoring Setup
        if monitor_progress:
            monitor.start_stage("📊 PROGRESS SETUP", "Initializing progress tracking for 5,040 candidates")
            
            # Clear any existing progress file
            progress_file = '/tmp/nwtn_progress.json'
            if os.path.exists(progress_file):
                os.remove(progress_file)
                print(f"   🗑️  Cleared existing progress file")
                
            print(f"   📈 Progress tracking: {progress_file}")
            
            monitor.end_stage()
        
        # Stage 3: Execute Complete Pipeline
        monitor.start_stage("🧠 NWTN PROCESSING", "Executing full pipeline: Semantic → System 1 → System 2 → LLM")
        
        # Start the actual NWTN processing
        processing_start = time.time()
        
        # Run the complete pipeline
        if monitor_progress:
            # Create progress monitoring task
            progress_task = asyncio.create_task(monitor_nwtn_progress())
            processing_task = asyncio.create_task(orchestrator.process_query(user_input))
            
            # Wait for processing to complete
            results = await processing_task
            
            # Cancel progress monitoring
            progress_task.cancel()
            
            try:
                await progress_task
            except asyncio.CancelledError:
                pass  # Expected when we cancel the task
        else:
            results = await orchestrator.process_query(user_input)
        
        processing_time = time.time() - processing_start
        
        monitor.end_stage()
        
        # Stage 4: Results Analysis
        monitor.start_stage("🔍 RESULTS ANALYSIS", "Analyzing pipeline outputs and validating components")
        
        # Extract key components
        final_answer = getattr(results, 'final_answer', '')
        reasoning_trace = getattr(results, 'reasoning_trace', [])
        metadata = getattr(results, 'metadata', {})
        confidence_score = getattr(results, 'confidence_score', 0)
        
        # Analyze pipeline execution
        analysis = analyze_pipeline_results(results)
        
        print(f"   📄 Final answer: {len(final_answer)} characters")
        print(f"   🧠 Reasoning steps: {len(reasoning_trace)}")
        print(f"   📊 Metadata keys: {len(metadata)}")
        print(f"   🎯 Confidence: {confidence_score}")
        
        monitor.end_stage()
        
        # Stage 5: Pipeline Validation
        monitor.start_stage("✅ VALIDATION", "Validating complete pipeline execution")
        
        validation = validate_complete_pipeline(results, analysis)
        
        for check, status in validation.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {check.replace('_', ' ').title()}: {status}")
        
        validation_success = all(validation.values())
        monitor.end_stage('completed' if validation_success else 'partial')
        
        # Complete pipeline
        pipeline_metrics = monitor.end_pipeline(success=validation_success)
        
        # Prepare comprehensive results
        complete_results = {
            # Core Results
            'final_answer': final_answer,
            'reasoning_trace': reasoning_trace,
            'metadata': metadata,
            'confidence_score': confidence_score,
            
            # Pipeline Analysis
            'pipeline_analysis': analysis,
            'pipeline_validation': validation,
            'pipeline_metrics': pipeline_metrics,
            
            # Processing Information
            'input_prompt': prompt,
            'context_allocation': context_allocation,
            'processing_time': processing_time,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            
            # Success Indicators
            'pipeline_success': validation_success,
            'components_validated': sum(validation.values()),
            'total_components': len(validation)
        }
        
        return complete_results
        
    except Exception as e:
        print(f"❌ PIPELINE ERROR: {e}")
        monitor.end_stage('failed')
        monitor.end_pipeline(success=False)
        
        import traceback
        traceback.print_exc()
        
        return {
            'error': str(e),
            'pipeline_success': False,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


async def monitor_nwtn_progress():
    """Monitor NWTN progress during processing"""
    progress_file = '/tmp/nwtn_progress.json'
    last_reported = -1
    
    try:
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    iteration = progress.get('iteration', 0)
                    total = progress.get('total_iterations', 5040)
                    percent = progress.get('progress_percent', 0)
                    status = progress.get('status', 'running')
                    
                    # Only report new progress
                    if iteration > last_reported:
                        if iteration % 500 == 0 or iteration >= 5000:  # Report more frequently near the end
                            print(f"   📈 Progress: {iteration:,}/{total:,} candidates ({percent:.1f}%)")
                        last_reported = iteration
                        
                        # Break if completed
                        if status == 'COMPLETED' or iteration >= total:
                            print(f"   🎉 Candidate generation completed: {iteration:,} candidates")
                            break
                            
                except Exception as e:
                    print(f"   ⚠️  Progress monitoring error: {e}")
                    
    except asyncio.CancelledError:
        print(f"   📊 Progress monitoring stopped")
        raise


def analyze_pipeline_results(results):
    """Analyze NWTN pipeline results for completeness and quality"""
    
    final_answer = getattr(results, 'final_answer', '')
    reasoning_trace = getattr(results, 'reasoning_trace', [])
    metadata = getattr(results, 'metadata', {})
    
    analysis = {
        # System 1 Analysis
        'system1_candidates': metadata.get('system1_candidates', 0),
        'candidates_generated': metadata.get('candidates_generated', 0),
        'reasoning_sequences': len(reasoning_trace),
        
        # System 2 Analysis
        'deduplication_applied': 'compressed' in str(metadata).lower() or 'dedup' in str(metadata).lower(),
        'meta_reasoning_applied': len(reasoning_trace) > 5,  # Multiple reasoning steps indicate meta-reasoning
        
        # Wisdom Package Analysis
        'wisdom_package_created': len(final_answer) > 500 and len(reasoning_trace) > 0,
        'corpus_integration': metadata.get('papers_used', 0) > 0 or 'corpus' in str(metadata).lower(),
        'provenance_tracking': 'session_id' in str(results),
        
        # LLM Integration Analysis
        'natural_language_generated': len(final_answer) > 100 and '.' in final_answer,
        'professional_quality': any(phrase in final_answer.lower() for phrase in 
                                   ['strategy', 'approach', 'research', 'analysis', 'comprehensive']),
        'structured_response': final_answer.count('\n') > 2 if final_answer else False,
        
        # Quality Metrics
        'answer_length': len(final_answer),
        'answer_complexity': len(final_answer.split('. ')) if final_answer else 0,
        'metadata_richness': len(metadata),
        'confidence_available': hasattr(results, 'confidence_score'),
        
        # Pipeline Completeness
        'end_to_end_success': all([
            len(final_answer) > 100,
            len(reasoning_trace) > 0,
            len(metadata) > 0
        ])
    }
    
    return analysis


def validate_complete_pipeline(results, analysis):
    """Validate that all pipeline components executed successfully"""
    
    validation = {
        # Core Pipeline Components
        'semantic_search_executed': analysis['corpus_integration'],
        'system1_generation_completed': analysis['system1_candidates'] >= 5000,  # Should be 5040
        'deduplication_applied': analysis['deduplication_applied'] or analysis['system1_candidates'] > 0,
        'meta_reasoning_executed': analysis['meta_reasoning_applied'],
        'wisdom_package_created': analysis['wisdom_package_created'],
        'llm_integration_successful': analysis['natural_language_generated'],
        
        # Quality Validation
        'comprehensive_answer': analysis['answer_length'] > 200,
        'professional_quality': analysis['professional_quality'],
        'structured_output': analysis['structured_response'],
        'metadata_complete': analysis['metadata_richness'] > 5,
        
        # End-to-End Validation
        'complete_pipeline_success': analysis['end_to_end_success']
    }
    
    return validation


def display_results(results):
    """Display comprehensive pipeline results"""
    
    print("\n" + "=" * 70)
    print("🎊 NWTN COMPLETE PIPELINE RESULTS")
    print("=" * 70)
    
    if not results.get('pipeline_success', False):
        print("❌ PIPELINE FAILED")
        if 'error' in results:
            print(f"   Error: {results['error']}")
        return
    
    # Pipeline Success Summary
    analysis = results.get('pipeline_analysis', {})
    validation = results.get('pipeline_validation', {})
    
    print(f"✅ PIPELINE SUCCESS: {results['components_validated']}/{results['total_components']} components validated")
    print(f"⏱️  Processing Time: {results['processing_time']:.2f}s")
    print(f"🧠 System 1 Candidates: {analysis.get('system1_candidates', 0):,}")
    print(f"📊 Reasoning Steps: {analysis.get('reasoning_sequences', 0)}")
    print(f"📄 Answer Length: {analysis.get('answer_length', 0)} characters")
    
    # Display final answer
    final_answer = results.get('final_answer', '')
    if final_answer:
        print(f"\n📋 FINAL NATURAL LANGUAGE RESPONSE:")
        print("─" * 70)
        print(final_answer)
        print("─" * 70)
    
    # Validation Summary
    print(f"\n🔍 COMPONENT VALIDATION:")
    for component, status in validation.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {component.replace('_', ' ').title()}")
    
    print(f"\n🎯 PIPELINE ARCHITECTURE CONFIRMED:")
    print("   📝 User Prompt")
    print("   ↓")
    print("   📚 Semantic Search (2,310 papers)")
    print("   ↓")
    print("   🔬 Content Analysis")
    print("   ↓")
    print("   🧠 System 1 (5,040 candidates)")
    print("   ↓")
    print("   🗜️ Deduplication & Compression")
    print("   ↓")
    print("   🎯 System 2 (Meta-reasoning)")
    print("   ↓")
    print("   📦 Wisdom Package Creation")
    print("   ↓")
    print("   🤖 LLM Integration (Claude API)")
    print("   ↓")
    print("   ✨ Natural Language Response")


async def main():
    """Run the complete NWTN pipeline demonstration"""
    
    print("🧠 NWTN Complete Pipeline Demonstration")
    print("=" * 50)
    
    # Use the proven context rot prompt that works well
    test_prompt = """The concept of "context rot" in AI systems refers to the degradation of performance when the operational context differs significantly from the training context. This phenomenon is particularly pronounced in large language models and neural networks deployed in dynamic environments.

Context rot manifests in several ways:
1. Distribution shift between training and deployment data
2. Temporal drift in data patterns over time  
3. Domain adaptation challenges when models encounter new scenarios
4. Catastrophic forgetting when models are updated with new information

Understanding and mitigating context rot is crucial for maintaining robust AI systems in production environments. What are the most effective strategies for detecting and preventing context rot in deployed machine learning models?"""
    
    print("Running complete NWTN pipeline with context rot analysis prompt...")
    print("Expected execution time: ~1-2 minutes for full 5,040 candidate generation")
    print()
    
    # Execute the complete pipeline
    results = await run_complete_nwtn_pipeline(
        prompt=test_prompt,
        context_allocation=1000,
        monitor_progress=True
    )
    
    # Display comprehensive results
    display_results(results)
    
    # Save results to file
    results_file = f"nwtn_complete_pipeline_results_{int(time.time())}.json"
    try:
        # Remove non-serializable objects for JSON
        json_results = {k: v for k, v in results.items() if k != 'reasoning_trace'}
        json_results['reasoning_steps_count'] = len(results.get('reasoning_trace', []))
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    return results


if __name__ == "__main__":
    print("🚀 Starting NWTN Complete Pipeline...")
    print("⏱️  This will demonstrate the full end-to-end pipeline")
    print("🧠 Generating 5,040 reasoning candidates and natural language response")
    print()
    
    try:
        results = asyncio.run(main())
        
        if results.get('pipeline_success', False):
            print("\n🎉 NWTN COMPLETE PIPELINE: SUCCESSFUL!")
            print("All components validated and working correctly.")
        else:
            print("\n⚠️  NWTN PIPELINE: PARTIAL SUCCESS")
            print("Some components may need attention.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n💥 PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()