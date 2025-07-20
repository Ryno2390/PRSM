#!/usr/bin/env python3
"""
Extract and synthesize results from the completed NWTN deep reasoning system
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

async def extract_reasoning_results():
    """Extract the final reasoning results from the deep reasoning log"""
    
    log_file = Path('/tmp/deep_reasoning_fixed.log')
    
    if not log_file.exists():
        print("❌ Deep reasoning log file not found!")
        return None
    
    print("🔍 Extracting reasoning results from completed deep reasoning system...")
    
    # Read the entire log file
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract key information
    results = {
        'query': 'What are the most promising approaches for improving transformer attention mechanisms to handle very long sequences efficiently?',
        'total_sequences': 5040,
        'completed_sequences': 0,
        'runtime_minutes': 0,
        'reasoning_approaches': [],
        'world_model_validations': [],
        'performance_metrics': {},
        'synthesis_results': [],
        'final_recommendations': []
    }
    
    # Extract progress checkpoints
    progress_matches = re.findall(
        r'DEEP REASONING PROGRESS.*?completed=(\d+).*?elapsed_minutes=([\d.]+)',
        log_content
    )
    
    if progress_matches:
        final_progress = progress_matches[-1]
        results['completed_sequences'] = int(final_progress[0])
        results['runtime_minutes'] = float(final_progress[1])
    
    # Extract world model validation information
    validation_matches = re.findall(
        r'Enhanced reasoning with world model.*?adjusted_confidence=([\d.]+).*?conflicts=(\d+).*?supporting_knowledge=(\d+)',
        log_content
    )
    
    if validation_matches:
        # Get statistics from all validations
        confidences = [float(m[0]) for m in validation_matches]
        conflicts = [int(m[1]) for m in validation_matches]
        supporting_knowledge = [int(m[2]) for m in validation_matches]
        
        results['world_model_validations'] = {
            'total_validations': len(validation_matches),
            'average_confidence': sum(confidences) / len(confidences),
            'average_conflicts': sum(conflicts) / len(conflicts),
            'average_supporting_knowledge': sum(supporting_knowledge) / len(supporting_knowledge),
            'confidence_range': [min(confidences), max(confidences)],
            'conflict_range': [min(conflicts), max(conflicts)]
        }
    
    # Extract reasoning engine sequences
    sequence_matches = re.findall(
        r'current_sequence=\[(.*?)\]',
        log_content
    )
    
    if sequence_matches:
        # Parse the sequences to understand reasoning approaches
        sequences = []
        for match in sequence_matches[-20:]:  # Get last 20 sequences
            # Clean up the sequence string
            sequence_str = match.replace("'", "").replace(" ", "")
            engines = sequence_str.split(',')
            sequences.append(engines)
        
        results['reasoning_approaches'] = sequences
    
    # Extract synthesis results if available
    synthesis_matches = re.findall(
        r'Synthesis complete.*?quality=([\d.]+).*?confidence=([\d.]+)',
        log_content
    )
    
    if synthesis_matches:
        results['synthesis_results'] = [
            {
                'quality': float(m[0]),
                'confidence': float(m[1])
            }
            for m in synthesis_matches
        ]
    
    # Performance metrics
    results['performance_metrics'] = {
        'sequences_per_minute': results['completed_sequences'] / results['runtime_minutes'] if results['runtime_minutes'] > 0 else 0,
        'total_runtime_hours': results['runtime_minutes'] / 60,
        'completion_percentage': (results['completed_sequences'] / results['total_sequences']) * 100
    }
    
    return results

async def main():
    print("🚀 NWTN Deep Reasoning Results Extraction")
    print("=" * 50)
    
    # Extract results
    results = await extract_reasoning_results()
    
    if not results:
        print("❌ Failed to extract results")
        return
    
    # Save results to file
    output_file = Path('/tmp/nwtn_deep_reasoning_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results extracted and saved to {output_file}")
    print()
    print("📊 SUMMARY:")
    print(f"Query: {results['query']}")
    print(f"Completed Sequences: {results['completed_sequences']:,} / {results['total_sequences']:,}")
    print(f"Runtime: {results['runtime_minutes']:.1f} minutes ({results['performance_metrics']['total_runtime_hours']:.2f} hours)")
    print(f"Completion: {results['performance_metrics']['completion_percentage']:.1f}%")
    print(f"Processing Rate: {results['performance_metrics']['sequences_per_minute']:.1f} sequences/minute")
    
    if results['world_model_validations']:
        wm = results['world_model_validations']
        print()
        print("🌍 WORLD MODEL VALIDATION:")
        print(f"Total Validations: {wm['total_validations']:,}")
        print(f"Average Confidence: {wm['average_confidence']:.3f}")
        print(f"Average Conflicts: {wm['average_conflicts']:.1f}")
        print(f"Average Supporting Knowledge: {wm['average_supporting_knowledge']:.1f}")
    
    print()
    print("✅ Ready for Claude API synthesis!")

if __name__ == "__main__":
    asyncio.run(main())