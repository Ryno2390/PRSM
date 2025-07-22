#!/usr/bin/env python3
"""
Test PDF Processing for NWTN Pipeline
=====================================

Quick test to verify PDF download and processing works with real papers.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.nwtn.external_storage_config import ExternalKnowledgeBase

async def test_pdf_processing():
    """Test downloading and processing a real arXiv PDF"""
    
    print("🧪 Testing PDF Processing")
    print("=" * 50)
    
    # Initialize knowledge base
    kb = ExternalKnowledgeBase()
    await kb.initialize()
    
    # Test with a recent superconductor paper
    test_arxiv_id = "0908.1126"
    print(f"📄 Testing with arXiv ID: {test_arxiv_id}")
    
    try:
        # Test PDF download
        print("📥 Downloading PDF...")
        pdf_content = await kb._download_arxiv_pdf(test_arxiv_id)
        
        if pdf_content:
            print(f"✅ PDF downloaded successfully! Size: {len(pdf_content)/1024:.1f} KB")
            
            # Test PDF text extraction
            print("📝 Extracting text from PDF...")
            structured_content = kb._extract_text_from_pdf(pdf_content)
            
            if structured_content:
                print("✅ Text extraction successful!")
                print(f"📊 Full text length: {len(structured_content.get('full_text', ''))}")
                print(f"📊 Introduction: {len(structured_content.get('introduction', ''))}")
                print(f"📊 Methodology: {len(structured_content.get('methodology', ''))}")
                print(f"📊 Results: {len(structured_content.get('results', ''))}")
                print(f"📊 Discussion: {len(structured_content.get('discussion', ''))}")
                print(f"📊 Conclusion: {len(structured_content.get('conclusion', ''))}")
                
                # Show a preview of extracted content
                if structured_content.get('introduction'):
                    print("\n📄 Introduction Preview:")
                    print("-" * 30)
                    print(structured_content['introduction'][:500] + "...")
                    
            else:
                print("❌ Text extraction failed")
                
        else:
            print("❌ PDF download failed")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pdf_processing())