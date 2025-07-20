#!/usr/bin/env python3
"""
Quick verification test for the royalty calculation fix
"""
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.nwtn.content_royalty_engine import ContentRoyaltyEngine, QueryComplexity
from prsm.provenance.enhanced_provenance_system import EnhancedProvenanceSystem
from prsm.core.database_service import get_database_service
from prsm.core.content_types import ContentType


async def test_royalty_calculation():
    """Test that royalty calculations now work properly"""
    
    print("🔧 Testing royalty calculation fix...")
    
    # Initialize systems
    royalty_engine = ContentRoyaltyEngine()
    provenance_system = EnhancedProvenanceSystem()
    database_service = get_database_service()
    
    await royalty_engine.initialize()
    
    # Create a test content with provenance
    test_content = b"This is a test research paper about machine learning."
    test_metadata = {
        "title": "Test Paper",
        "authors": ["Test Author"],
        "domain": "computer_science"
    }
    
    creator_info = {
        'name': 'test_creator_001',
        'platform': 'PRSM_TEST',
        'contributors': [],
        'parent_content': None
    }
    
    license_info = {
        'type': 'open_source',
        'terms': {'attribution_required': True}
    }
    
    print("📝 Creating test content with provenance...")
    
    # Register content with provenance
    content_id, fingerprint, attribution_chain = await provenance_system.register_content_with_provenance(
        content_data=test_content,
        content_type=ContentType.RESEARCH_PAPER,
        creator_info=creator_info,
        license_info=license_info,
        metadata=test_metadata
    )
    
    print(f"✅ Content created with ID: {content_id}")
    print(f"✅ Attribution chain created for: {attribution_chain.original_creator}")
    
    # Test royalty calculation
    print("💰 Testing royalty calculation...")
    
    content_sources = [content_id]
    reasoning_context = {
        'reasoning_path': ['deductive', 'inductive'],
        'overall_confidence': 0.8,
        'content_weights': {str(content_id): 1.0}
    }
    
    royalty_calculations = await royalty_engine.calculate_usage_royalty(
        content_sources=content_sources,
        query_complexity=QueryComplexity.COMPLEX,
        user_tier="premium",
        reasoning_context=reasoning_context
    )
    
    print(f"📊 Royalty calculations: {len(royalty_calculations)} items")
    
    if royalty_calculations:
        for calc in royalty_calculations:
            print(f"  💰 Content: {calc.content_id}")
            print(f"  👤 Creator: {calc.creator_id}")
            print(f"  💵 Final Royalty: {calc.final_royalty}")
            print(f"  📈 Importance: {calc.importance_multiplier}")
            print(f"  ⭐ Quality: {calc.quality_multiplier}")
        
        total_royalty = sum(calc.final_royalty for calc in royalty_calculations)
        print(f"✅ Total royalty: {total_royalty} FTNS")
        
        if total_royalty > 0:
            print("🎉 SUCCESS: Royalty calculation is working!")
            return True
        else:
            print("❌ FAILURE: Royalty calculation returned 0.0")
            return False
    else:
        print("❌ FAILURE: No royalty calculations returned")
        return False


async def main():
    """Run the verification test"""
    print("🚀 Royalty Calculation Fix Verification")
    print("=" * 60)
    
    try:
        success = await test_royalty_calculation()
        if success:
            print("\n✅ All tests passed! The royalty calculation fix is working.")
            return 0
        else:
            print("\n❌ Test failed. The royalty calculation issue persists.")
            return 1
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))