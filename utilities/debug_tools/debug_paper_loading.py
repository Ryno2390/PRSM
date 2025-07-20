#!/usr/bin/env python3
"""
Debug script to check paper loading from external drive
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.test_150k_papers_provenance import Large150KPaperProvenanceTest

def main():
    # Create test instance
    test = Large150KPaperProvenanceTest()
    
    # Check if external drive is accessible
    papers_directory = Path("/Volumes/My Passport/PRSM_Storage/PRSM_Content/hot")
    print(f"📁 Checking papers directory: {papers_directory}")
    print(f"📁 Directory exists: {papers_directory.exists()}")
    
    if papers_directory.exists():
        # Check what files are in the directory
        print(f"📁 Directory contents:")
        dat_files = list(papers_directory.glob("**/*.dat"))
        print(f"📊 Found {len(dat_files)} .dat files")
        
        if len(dat_files) > 0:
            print("📝 First 5 .dat files:")
            for i, file in enumerate(dat_files[:5]):
                print(f"  {i+1}. {file.name}")
            
            # Test loading a single paper
            print("\n🔍 Testing metadata extraction on first paper:")
            try:
                metadata = test._extract_arxiv_metadata(dat_files[0])
                print(f"✅ Metadata extracted: {metadata}")
            except Exception as e:
                print(f"❌ Metadata extraction failed: {e}")
                
            # Test loading papers from directory
            print("\n🔄 Testing paper loading from directory:")
            try:
                papers = test._load_papers_from_directory(papers_directory)
                print(f"✅ Loaded {len(papers)} papers")
                
                if len(papers) > 0:
                    print(f"📄 First paper: {papers[0]['title']}")
                    
            except Exception as e:
                print(f"❌ Paper loading failed: {e}")
        
        else:
            print("❌ No .dat files found in directory")
    else:
        print("❌ Papers directory not found")
        
        # Check if external drive is mounted
        volumes = Path("/Volumes")
        if volumes.exists():
            print("📁 Available volumes:")
            for vol in volumes.iterdir():
                if vol.is_dir():
                    print(f"  - {vol.name}")
        else:
            print("❌ /Volumes directory not found")

if __name__ == "__main__":
    main()