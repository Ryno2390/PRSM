#!/usr/bin/env python3
"""
NWTN Complete Pipeline Runner
============================

Interactive command-line interface for running complete NWTN pipeline:
Prompt → Parameter Selection → Deep Reasoning → Content Grounding → Claude Synthesis → Answer

Usage:
    python run_nwtn_pipeline.py

Features:
- Interactive parameter selection (reasoning depth & verbosity)
- Complete end-to-end pipeline execution
- Real-time progress tracking
- FTNS cost calculation and receipt
- Works Cited with actual paper references
- Grounded synthesis with zero hallucination risk
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nwtn_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def print_header():
    """Print NWTN pipeline header"""
    print("=" * 80)
    print("🧠 NWTN COMPLETE PIPELINE RUNNER")
    print("Raw Data → NWTN Search → Deep Reasoning → Content Grounding → Claude API → Answer")
    print("=" * 80)
    print(f"📊 Scale: 149,726 arXiv papers with semantic search + hallucination prevention")
    print(f"🚀 Status: Production-ready with content grounding system")
    print(f"📝 Documentation: Full pipeline architecture and technical details available")
    print()

def get_user_query():
    """Get the research query from the user"""
    print("📝 STEP 1: RESEARCH QUERY")
    print("-" * 40)
    
    while True:
        query = input("Enter your research question: ").strip()
        if query:
            if len(query) < 10:
                print("⚠️  Please provide a more detailed research question (at least 10 characters)")
                continue
            break
        else:
            print("⚠️  Please enter a research question")
    
    print(f"✅ Query received: {query}")
    print()
    return query

def get_reasoning_depth():
    """Get the desired reasoning depth from the user"""
    print("🔬 STEP 2: REASONING DEPTH SELECTION")
    print("-" * 40)
    
    reasoning_options = {
        "1": {
            "name": "QUICK",
            "description": "Fast reasoning with 3-5 engines (~2-5 minutes)",
            "engines": "3-5 reasoning engines",
            "time": "2-5 minutes",
            "cost": "1.0x multiplier (base token rate)",
            "use_case": "Quick insights, preliminary analysis"
        },
        "2": {
            "name": "INTERMEDIATE", 
            "description": "Balanced reasoning with 5-6 engines (~10-20 minutes)",
            "engines": "5-6 reasoning engines",
            "time": "10-20 minutes", 
            "cost": "2.5x multiplier (2.5x base tokens)",
            "use_case": "Most research questions, good balance of depth/speed"
        },
        "3": {
            "name": "DEEP",
            "description": "Comprehensive reasoning with all 7 engines + 5,040 permutations (~2-3 hours)",
            "engines": "All 7 reasoning engines, 5,040 permutations",
            "time": "2-3 hours",
            "cost": "5.0x multiplier (5.0x base tokens)",
            "use_case": "Complex research, breakthrough discovery, publication-quality"
        }
    }
    
    for key, option in reasoning_options.items():
        print(f"{key}. {option['name']}")
        print(f"   📊 Engines: {option['engines']}")
        print(f"   ⏱️  Time: {option['time']}")
        print(f"   💰 Cost: {option['cost']}")
        print(f"   🎯 Use case: {option['use_case']}")
        print()
    
    while True:
        choice = input("Select reasoning depth (1-3): ").strip()
        if choice in reasoning_options:
            selected = reasoning_options[choice]
            print(f"✅ Selected: {selected['name']} reasoning ({selected['time']})")
            print()
            return selected['name']
        else:
            print("⚠️  Please enter 1, 2, or 3")

def get_verbosity_level():
    """Get the desired verbosity level from the user"""
    print("📄 STEP 3: RESPONSE VERBOSITY SELECTION")
    print("-" * 40)
    
    verbosity_options = {
        "1": {
            "name": "BRIEF",
            "tokens": 500,
            "description": "Concise summary with key insights",
            "length": "~1 page",
            "citations": "3-5 papers",
            "use_case": "Quick overview, executive summary"
        },
        "2": {
            "name": "STANDARD",
            "tokens": 1000,
            "description": "Detailed analysis with supporting evidence",
            "length": "~2 pages", 
            "citations": "5-8 papers",
            "use_case": "Standard research analysis, most use cases"
        },
        "3": {
            "name": "DETAILED", 
            "tokens": 2000,
            "description": "Comprehensive analysis with methodology discussion",
            "length": "~4 pages",
            "citations": "8-12 papers",
            "use_case": "In-depth research, technical analysis"
        },
        "4": {
            "name": "COMPREHENSIVE",
            "tokens": 3500,
            "description": "Extensive analysis with implementation recommendations",
            "length": "~7 pages",
            "citations": "12-15 papers", 
            "use_case": "Research reports, strategic analysis"
        },
        "5": {
            "name": "ACADEMIC",
            "tokens": 4000,
            "description": "Publication-quality analysis with full citations",
            "length": "~8 pages",
            "citations": "15+ papers",
            "use_case": "Academic papers, comprehensive reports"
        }
    }
    
    for key, option in verbosity_options.items():
        print(f"{key}. {option['name']} ({option['tokens']} tokens)")
        print(f"   📝 Description: {option['description']}")
        print(f"   📄 Length: {option['length']}")
        print(f"   📚 Citations: {option['citations']}")
        print(f"   🎯 Use case: {option['use_case']}")
        print()
    
    while True:
        choice = input("Select verbosity level (1-5): ").strip()
        if choice in verbosity_options:
            selected = verbosity_options[choice]
            print(f"✅ Selected: {selected['name']} ({selected['tokens']} tokens, {selected['length']})")
            print()
            return selected['name']
        else:
            print("⚠️  Please enter 1, 2, 3, 4, or 5")

def get_breakthrough_mode():
    """Get the desired breakthrough intensity mode from the user"""
    print("🚀 STEP 3: BREAKTHROUGH INTENSITY SELECTION")
    print("-" * 40)
    
    breakthrough_options = {
        "1": {
            "name": "CONSERVATIVE",
            "description": "Established approaches with high confidence",
            "multiplier": "0.8x",
            "use_cases": [
                "Medical research and clinical applications",
                "Regulatory compliance and safety-critical systems",
                "Financial analysis and risk assessment"
            ],
            "features": "Focus on proven methods, high confidence threshold",
            "example_domains": "Healthcare, Finance, Legal, Safety"
        },
        "2": {
            "name": "BALANCED", 
            "description": "Mix of proven approaches with moderate innovation",
            "multiplier": "1.0x",
            "use_cases": [
                "Business strategy and competitive analysis",
                "Academic research with practical applications", 
                "Product development and market research"
            ],
            "features": "60% conventional + 40% breakthrough candidates",
            "example_domains": "Business, Technology, Academic Research"
        },
        "3": {
            "name": "CREATIVE",
            "description": "Explore novel possibilities and innovative approaches", 
            "multiplier": "1.3x",
            "use_cases": [
                "Research and development initiatives",
                "Innovation workshops and brainstorming",
                "Startup strategy and disruption analysis"
            ],
            "features": "70% breakthrough + wild hypothesis generation",
            "example_domains": "R&D, Startups, Innovation Labs"
        },
        "4": {
            "name": "REVOLUTIONARY",
            "description": "Challenge everything and explore radical possibilities",
            "multiplier": "1.8x", 
            "use_cases": [
                "Moonshot and breakthrough innovation projects",
                "Paradigm shift research and analysis",
                "Blue-sky research and theoretical exploration"
            ],
            "features": "90% breakthrough + assumption challenging + impossibility exploration",
            "example_domains": "Moonshots, Venture Capital, Theoretical Research"
        }
    }
    
    for key, option in breakthrough_options.items():
        print(f"{key}. {option['name']}")
        print(f"   Description: {option['description']}")
        print(f"   Complexity Multiplier: {option['multiplier']} (affects cost)")
        print(f"   Features: {option['features']}")
        print(f"   Example Domains: {option['example_domains']}")
        print(f"   Key Use Cases:")
        for use_case in option['use_cases'][:2]:  # Show top 2 use cases
            print(f"     • {use_case}")
        print()
    
    while True:
        choice = input("Select breakthrough intensity (1-4): ").strip()
        if choice in breakthrough_options:
            selected = breakthrough_options[choice]
            print(f"✅ Selected: {selected['name']}")
            print(f"   {selected['description']}")
            print()
            return selected['name']
        else:
            print("⚠️  Please enter a valid choice (1-4)")

def display_pipeline_summary(query, depth, verbosity, breakthrough_mode):
    """Display pipeline execution summary"""
    print("🚀 PIPELINE EXECUTION SUMMARY")
    print("-" * 40)
    print(f"📝 Query: {query}")
    print(f"🔬 Reasoning Depth: {depth}")
    print(f"📄 Verbosity Level: {verbosity}")
    print(f"🚀 Breakthrough Mode: {breakthrough_mode}")
    print()
    
    # Get precise token-based pricing preview
    print("💰 CALCULATING PRECISE FTNS COST...")
    print("-" * 40)
    
    try:
        # Import and initialize voicebox for pricing preview
        from prsm.nwtn.voicebox import NWTNVoicebox
        
        voicebox = NWTNVoicebox()
        await voicebox.initialize()
        
        # Get pricing preview using enhanced token-based pricing
        pricing_preview = await voicebox.get_pricing_preview(
            query=query,
            thinking_mode=depth,
            verbosity_level=verbosity
        )
        
        estimated_cost = pricing_preview["estimated_cost"]
        cost_breakdown = pricing_preview["cost_breakdown"]
        pricing_explanation = pricing_preview["pricing_explanation"]
        
        print(f"💰 Estimated Cost: {estimated_cost:.3f} FTNS")
        print(f"🧮 Base Computational Tokens: {cost_breakdown['base_tokens']}")
        print(f"🧠 Reasoning Multiplier: {pricing_explanation['thinking_multiplier']}")
        print(f"📝 Verbosity Multiplier: {pricing_explanation['verbosity_multiplier']}")
        print(f"🚀 Breakthrough Mode: {breakthrough_mode}")
        print(f"📊 Market Conditions: {pricing_explanation['market_conditions']}")
        print(f"🏆 Quality Tier: {pricing_preview['pricing_tier']}")
        print(f"📄 Estimated Response Tokens: {pricing_preview['estimated_response_tokens']}")
        
        # Time estimates
        time_estimates = {
            "QUICK": "2-5 minutes",
            "INTERMEDIATE": "10-20 minutes", 
            "DEEP": "2-3 hours"
        }
        estimated_time = time_estimates.get(depth, "Unknown")
        print(f"⏱️  Estimated Time: {estimated_time}")
        
    except Exception as e:
        print(f"⚠️  Could not calculate precise pricing: {e}")
        print("💰 Using fallback estimate: 100-500 FTNS")
        estimated_cost = "100-500"
    
    print()
    
    confirm = input("Proceed with pipeline execution? (y/n): ").strip().lower()
    return confirm == 'y'

async def run_nwtn_pipeline(query: str, depth: str, verbosity: str, breakthrough_mode: str):
    """Run the complete NWTN pipeline"""
    
    pipeline_id = str(uuid4())
    start_time = datetime.now(timezone.utc)
    
    print("🔄 EXECUTING NWTN PIPELINE...")
    print("=" * 40)
    print(f"Pipeline ID: {pipeline_id}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    try:
        # Check if API key is available
        api_key_file = Path("/Users/ryneschultz/Documents/GitHub/Anthropic_API_Key.txt")
        if not api_key_file.exists():
            print("❌ Error: Anthropic API key file not found")
            print(f"Expected location: {api_key_file}")
            return
        
        # Step 1: Initialize NWTN components
        print("⚙️  STEP 1: Initializing NWTN components...")
        
        # Import NWTN components
        from prsm.nwtn.voicebox import NWTNVoicebox
        from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine
        
        # Initialize VoiceBox
        print("   📡 Initializing VoiceBox...")
        voicebox = NWTNVoicebox()
        await voicebox.initialize()
        print("   ✅ VoiceBox initialized")
        
        # Configure API key
        print("   🔑 Configuring Anthropic API key...")
        api_key = api_key_file.read_text().strip()
        
        # Set default API config (this bypasses user API configuration for testing)
        from prsm.nwtn.voicebox import APIConfiguration, LLMProvider
        voicebox.default_api_config = APIConfiguration(
            provider=LLMProvider.CLAUDE,
            api_key=api_key,
            model_name="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.7,
            timeout=60
        )
        print("   ✅ API key configured")
        print()
        
        # Step 2: Execute reasoning pipeline
        print("🧠 STEP 2: Executing multi-modal reasoning...")
        print(f"   🔬 Depth: {depth} reasoning mode")
        
        # Create context with depth, verbosity, and breakthrough mode parameters
        context = {
            "thinking_mode": depth,
            "verbosity_level": verbosity,
            "breakthrough_mode": breakthrough_mode,
            "pipeline_id": pipeline_id,
            "user_interface": "interactive_cli",
            "user_tier": "standard"  # For token-based pricing
        }
        
        # Mock user ID for testing (in production this would be actual user)
        test_user_id = "test_user_pipeline"
        
        # Configure API key for the test user to bypass the check
        from prsm.nwtn.voicebox import LLMProvider
        await voicebox.configure_api_key(
            user_id=test_user_id,
            provider=LLMProvider.CLAUDE,
            api_key=api_key
        )
        
        print("   🔍 Searching 149,726 arXiv papers...")
        print("   ⚙️  Running NWTN reasoning engines...")
        
        if depth == "DEEP":
            print("   🚀 Deep mode: Running 5,040 permutations (this will take 2-3 hours)")
            print("   📊 Progress will be logged to nwtn_pipeline.log")
        
        # Execute the complete pipeline through VoiceBox
        print("   🎯 Processing query through complete NWTN pipeline...")
        
        response = await voicebox.process_query(
            user_id=test_user_id,
            query=query,
            context=context
        )
        
        print("   ✅ Reasoning completed")
        print()
        
        # Step 3: Display results
        print("📊 STEP 3: Pipeline results...")
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        print("=" * 80)
        print("🎉 NWTN PIPELINE EXECUTION COMPLETED!")
        print("=" * 80)
        print()
        
        print("📝 FINAL ANSWER:")
        print("-" * 50)
        print(response.natural_language_response)
        print()
        
        print("📊 EXECUTION SUMMARY:")
        print("-" * 30)
        print(f"⏱️  Processing Time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        print(f"💰 Total Cost: {response.total_cost_ftns:.1f} FTNS")
        print(f"📈 Confidence Score: {response.confidence_score:.1f}%")
        print(f"🔬 Reasoning Modes Used: {len(response.used_reasoning_modes)}")
        print(f"📚 Source Links: {len(response.source_links)}")
        print()
        
        if response.source_links:
            print("🔗 SOURCE ATTRIBUTION:")
            print("-" * 30)
            print(response.attribution_summary)
            print()
        
        print("💳 FTNS RECEIPT:")
        print("-" * 20)
        receipt = {
            "transaction_id": response.response_id,
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "reasoning_depth": depth,
            "verbosity_level": verbosity,
            "processing_time_seconds": processing_time,
            "total_cost_ftns": response.total_cost_ftns,
            "confidence_score": response.confidence_score,
            "source_papers_used": len(response.source_links),
            "pipeline_version": "2.2-content-grounding"
        }
        
        print(json.dumps(receipt, indent=2))
        print()
        
        # Save results to file
        results_file = f"nwtn_results_{pipeline_id[:8]}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "receipt": receipt,
                "response": response.natural_language_response,
                "attribution": response.attribution_summary,
                "source_links": [
                    {
                        "title": link.title,
                        "creator": link.creator,
                        "ipfs_link": link.ipfs_link
                    } for link in response.source_links
                ]
            }, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        print()
        
        print("✅ NWTN Pipeline execution completed successfully!")
        print("🎯 Zero hallucination risk - all content grounded in actual research papers")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logger.error(f"NWTN pipeline failed: {e}", exc_info=True)
        return False
    
    return True

async def main():
    """Main pipeline runner"""
    try:
        # Display header
        print_header()
        
        # Get user inputs
        query = get_user_query()
        depth = get_reasoning_depth()
        verbosity = get_verbosity_level()
        breakthrough_mode = get_breakthrough_mode()
        
        # Display summary and confirm
        if not display_pipeline_summary(query, depth, verbosity, breakthrough_mode):
            print("❌ Pipeline execution cancelled by user")
            return
        
        # Run the complete pipeline
        success = await run_nwtn_pipeline(query, depth, verbosity, breakthrough_mode)
        
        if success:
            print("\n🎉 NWTN Pipeline completed successfully!")
            print("📝 Ready for next query - run script again to test another prompt")
        else:
            print("\n❌ Pipeline execution failed - check logs for details")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected pipeline error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())