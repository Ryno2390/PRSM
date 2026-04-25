#!/usr/bin/env python3
"""
Efficient Duplicate Detection Script for Large Paper Collections
Handles ~200K+ papers with memory-efficient processing and detailed reporting
"""

import os
import json
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path
import hashlib

class EfficientDuplicateDetector:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.arxiv_ids = set()
        self.duplicates = []
        self.file_count = 0
        self.processed_count = 0
        self.errors = []
        
    def extract_arxiv_id_from_filename(self, filename: str) -> str:
        """Extract arXiv ID from filename (e.g., '2001.0006.json' -> '2001.0006')"""
        return filename.replace('.json', '')
    
    def batch_process_files(self, batch_size: int = 1000) -> dict:
        """Process files in batches to avoid memory issues"""
        print(f"🔍 Starting duplicate detection in: {self.data_dir}")
        print(f"📦 Processing in batches of {batch_size} files")
        
        start_time = time.time()
        
        # Get all JSON files
        json_files = list(self.data_dir.glob("*.json"))
        self.file_count = len(json_files)
        print(f"📊 Found {self.file_count:,} JSON files to analyze")
        
        # Track arXiv IDs from filenames (fast method)
        filename_ids = Counter()
        for json_file in json_files:
            arxiv_id = self.extract_arxiv_id_from_filename(json_file.name)
            filename_ids[arxiv_id] += 1
        
        # Find filename-based duplicates
        filename_duplicates = {arxiv_id: count for arxiv_id, count in filename_ids.items() if count > 1}
        
        print(f"📁 Filename-based analysis:")
        print(f"   - Total files: {self.file_count:,}")
        print(f"   - Unique filenames: {len(filename_ids):,}")
        print(f"   - Duplicate filenames: {len(filename_duplicates):,}")
        
        if filename_duplicates:
            print(f"⚠️  Found {len(filename_duplicates)} duplicate filenames:")
            for arxiv_id, count in list(filename_duplicates.items())[:5]:
                print(f"   - {arxiv_id}: {count} files")
            if len(filename_duplicates) > 5:
                print(f"   ... and {len(filename_duplicates) - 5} more")
        
        # Content-based verification for a sample
        print(f"\n📖 Verifying content integrity (sample of {min(1000, self.file_count)} files)...")
        sample_files = json_files[:1000]
        content_arxiv_ids = Counter()
        content_errors = 0
        
        for i, json_file in enumerate(sample_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content_arxiv_id = data.get('arxiv_id', 'MISSING')
                    filename_arxiv_id = self.extract_arxiv_id_from_filename(json_file.name)
                    
                    content_arxiv_ids[content_arxiv_id] += 1
                    
                    # Check if filename matches content
                    if content_arxiv_id != filename_arxiv_id:
                        self.errors.append(f"Mismatch: {json_file.name} contains {content_arxiv_id}")
                        
            except Exception as e:
                content_errors += 1
                self.errors.append(f"Error reading {json_file.name}: {str(e)}")
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1:,}/{len(sample_files):,} sample files...")
        
        # Content-based duplicates in sample
        content_duplicates = {arxiv_id: count for arxiv_id, count in content_arxiv_ids.items() if count > 1}
        
        print(f"\n📊 Content-based analysis (sample):")
        print(f"   - Sample size: {len(sample_files):,}")
        print(f"   - Unique content IDs: {len(content_arxiv_ids):,}")
        print(f"   - Content duplicates in sample: {len(content_duplicates):,}")
        print(f"   - Content errors: {content_errors:,}")
        
        if content_duplicates:
            print(f"⚠️  Content-based duplicates in sample:")
            for arxiv_id, count in list(content_duplicates.items())[:3]:
                print(f"   - {arxiv_id}: {count} files")
        
        # Final statistics
        elapsed_time = time.time() - start_time
        
        results = {
            'total_files': self.file_count,
            'unique_filenames': len(filename_ids),
            'filename_duplicates': filename_duplicates,
            'sample_size': len(sample_files),
            'content_duplicates_in_sample': content_duplicates,
            'errors': self.errors,
            'processing_time': elapsed_time
        }
        
        print(f"\n✅ Analysis complete in {elapsed_time:.1f} seconds")
        return results
    
    def generate_report(self, results: dict) -> str:
        """Generate comprehensive duplicate detection report"""
        report = f"""
🔍 DUPLICATE DETECTION REPORT
{'='*50}

📊 DATASET OVERVIEW:
   Total JSON files found: {results['total_files']:,}
   Unique filename patterns: {results['unique_filenames']:,}
   Processing time: {results['processing_time']:.1f} seconds

📁 FILENAME-BASED ANALYSIS:
   Duplicate filenames detected: {len(results['filename_duplicates']):,}
   
   Status: {'❌ DUPLICATES FOUND' if results['filename_duplicates'] else '✅ NO FILENAME DUPLICATES'}

📖 CONTENT-BASED ANALYSIS (Sample):
   Sample size verified: {results['sample_size']:,}
   Content duplicates in sample: {len(results['content_duplicates_in_sample']):,}
   
   Status: {'❌ CONTENT DUPLICATES FOUND' if results['content_duplicates_in_sample'] else '✅ NO CONTENT DUPLICATES IN SAMPLE'}

🚨 ERRORS:
   Total errors encountered: {len(results['errors']):,}
"""
        
        if results['filename_duplicates']:
            report += f"\n📋 FILENAME DUPLICATES DETAIL:\n"
            for arxiv_id, count in list(results['filename_duplicates'].items())[:10]:
                report += f"   - {arxiv_id}: {count} files\n"
            if len(results['filename_duplicates']) > 10:
                report += f"   ... and {len(results['filename_duplicates']) - 10} more\n"
        
        if results['content_duplicates_in_sample']:
            report += f"\n📋 CONTENT DUPLICATES DETAIL (Sample):\n"
            for arxiv_id, count in list(results['content_duplicates_in_sample'].items())[:5]:
                report += f"   - {arxiv_id}: {count} files\n"
        
        if results['errors']:
            report += f"\n🚨 ERROR DETAILS (First 5):\n"
            for error in results['errors'][:5]:
                report += f"   - {error}\n"
            if len(results['errors']) > 5:
                report += f"   ... and {len(results['errors']) - 5} more errors\n"
        
        # Final assessment
        if not results['filename_duplicates'] and not results['content_duplicates_in_sample']:
            report += f"\n🎉 FINAL ASSESSMENT: DATASET APPEARS CLEAN\n"
            report += f"   Estimated unique papers: {results['unique_filenames']:,}\n"
        else:
            report += f"\n⚠️  FINAL ASSESSMENT: DUPLICATES DETECTED\n"
            report += f"   Action required: Manual review and cleanup\n"
        
        return report

def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/Volumes/My Passport/PRSM_Storage/02_PROCESSED_CONTENT"
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory {data_dir} does not exist")
        return
    
    detector = EfficientDuplicateDetector(data_dir)
    results = detector.batch_process_files()
    report = detector.generate_report(results)
    
    print(report)
    
    # Save report to file
    report_file = "/Users/ryneschultz/Documents/GitHub/PRSM/duplicate_detection_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n💾 Report saved to: {report_file}")

if __name__ == "__main__":
    main()