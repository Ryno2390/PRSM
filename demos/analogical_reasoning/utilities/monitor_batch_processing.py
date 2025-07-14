#!/usr/bin/env python3
"""
Batch Processing Monitor
Simple script to check the status of the 100-paper batch processing test
"""

import os
import json
import time
import subprocess
import glob
from pathlib import Path
from datetime import datetime

def check_process_running():
    """Check if batch_processor.py is still running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'batch_processor.py'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def get_storage_usage():
    """Get current storage usage"""
    storage_path = Path("batch_processing_storage")
    if not storage_path.exists():
        return "Storage directory not found"
    
    total_size = 0
    file_counts = {}
    
    for subfolder in ['pdfs', 'extracted_text', 'socs', 'patterns', 'results', 'temp']:
        folder_path = storage_path / subfolder
        if folder_path.exists():
            folder_size = 0
            file_count = 0
            for file_path in folder_path.rglob('*'):
                if file_path.is_file():
                    folder_size += file_path.stat().st_size
                    file_count += 1
            file_counts[subfolder] = file_count
            total_size += folder_size
    
    return {
        'total_mb': total_size / (1024 * 1024),
        'file_counts': file_counts
    }

def check_results_files():
    """Check for completion results files"""
    results_path = Path("batch_processing_storage/results")
    if not results_path.exists():
        return {}
    
    results_files = {}
    
    # Check for key completion files
    completion_files = [
        "batch_processing_results_100_papers.json.gz",
        "pattern_catalog_100_papers.json.gz", 
        "100_paper_test_summary.json.gz"
    ]
    
    for filename in completion_files:
        file_path = results_path / filename
        if file_path.exists():
            results_files[filename] = {
                'exists': True,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            results_files[filename] = {'exists': False}
    
    return results_files

def estimate_progress():
    """Estimate processing progress based on stored files"""
    storage_usage = get_storage_usage()
    file_counts = storage_usage.get('file_counts', {})
    
    # Estimate based on pattern files (each paper should generate 1 pattern file)
    patterns_processed = file_counts.get('patterns', 0)
    
    return {
        'estimated_papers_processed': patterns_processed,
        'progress_percent': min(100, (patterns_processed / 100) * 100),
        'estimated_remaining': max(0, 100 - patterns_processed)
    }

def get_system_load():
    """Get basic system load info"""
    try:
        # Get CPU usage
        result = subprocess.run(['top', '-l', '1', '-n', '0'], 
                              capture_output=True, text=True)
        cpu_line = [line for line in result.stdout.split('\n') if 'CPU usage' in line]
        cpu_info = cpu_line[0] if cpu_line else "CPU info unavailable"
        
        # Get memory usage
        result = subprocess.run(['vm_stat'], capture_output=True, text=True)
        memory_lines = result.stdout.split('\n')[:5]
        
        return {
            'cpu': cpu_info,
            'memory_sample': memory_lines[1] if len(memory_lines) > 1 else "Memory info unavailable"
        }
    except:
        return {'cpu': 'System info unavailable', 'memory_sample': ''}

def print_status():
    """Print comprehensive status report"""
    print("=" * 70)
    print("🔄 100-PAPER BATCH PROCESSING MONITOR")
    print("=" * 70)
    print(f"📅 Status Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if process is running
    is_running = check_process_running()
    print(f"\n🖥️  PROCESS STATUS:")
    if is_running:
        print("   ✅ batch_processor.py is RUNNING")
    else:
        print("   ❌ batch_processor.py is NOT RUNNING")
    
    # Get storage usage
    storage = get_storage_usage()
    if isinstance(storage, dict):
        print(f"\n📊 STORAGE USAGE:")
        print(f"   Total storage: {storage['total_mb']:.1f} MB")
        print(f"   File counts by type:")
        for folder, count in storage['file_counts'].items():
            print(f"      {folder}: {count} files")
    
    # Estimate progress
    progress = estimate_progress()
    print(f"\n📈 ESTIMATED PROGRESS:")
    print(f"   Papers processed: ~{progress['estimated_papers_processed']}/100")
    print(f"   Progress: {progress['progress_percent']:.1f}%")
    if progress['estimated_remaining'] > 0:
        print(f"   Estimated remaining: {progress['estimated_remaining']} papers")
    
    # Check for completion files
    results = check_results_files()
    print(f"\n🎯 COMPLETION STATUS:")
    completion_count = sum(1 for file_info in results.values() if file_info.get('exists', False))
    total_files = len(results)
    
    if completion_count == total_files:
        print("   🎉 BATCH PROCESSING COMPLETE!")
        print("   All result files have been generated.")
    elif completion_count > 0:
        print(f"   ⚠️  Partial completion: {completion_count}/{total_files} result files found")
    else:
        print("   ⏳ Processing still in progress...")
    
    # Show result files status
    for filename, info in results.items():
        if info['exists']:
            print(f"      ✅ {filename} ({info['size_mb']:.2f} MB, {info['modified']})")
        else:
            print(f"      ⏳ {filename} (not yet created)")
    
    # System load
    system = get_system_load()
    print(f"\n💻 SYSTEM STATUS:")
    print(f"   {system['cpu']}")
    print(f"   {system['memory_sample']}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    if completion_count == total_files:
        print("   🚀 Processing complete! You can now:")
        print("      - Review results with: python analyze_batch_results.py")
        print("      - Make go/no-go decision for 1K paper scaling")
        print("      - Begin next phase of exponential scaling")
    elif is_running:
        print("   ⏳ Processing is active. Check back in 30-60 minutes.")
        print("   💡 You can run this monitor script periodically:")
        print("      python monitor_batch_processing.py")
    else:
        print("   ❌ Process appears stopped. You may need to:")
        print("      - Check for errors in the terminal")
        print("      - Restart with: python batch_processor.py")
        print("      - Review logs for troubleshooting")
    
    print("=" * 70)

def main():
    """Main monitoring function"""
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print_status()

if __name__ == "__main__":
    main()