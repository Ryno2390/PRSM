#!/usr/bin/env python3
"""
Storage Manager for 100-Paper Test
Manages local storage with automatic cleanup to stay under 3GB total footprint
"""

import os
import shutil
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

class StorageManager:
    """Manages storage for 100-paper test with automatic cleanup"""
    
    def __init__(self, base_dir: str = "paper_test_storage"):
        self.base_dir = Path(base_dir)
        self.max_storage_gb = 3.0  # 3GB limit
        self.max_storage_bytes = int(self.max_storage_gb * 1024 * 1024 * 1024)
        
        # Storage directories
        self.dirs = {
            'pdfs': self.base_dir / 'pdfs',
            'text': self.base_dir / 'extracted_text', 
            'socs': self.base_dir / 'socs',
            'patterns': self.base_dir / 'patterns',
            'temp': self.base_dir / 'temp',
            'results': self.base_dir / 'results'
        }
        
        # Initialize storage
        self._setup_directories()
        
        # Track file sizes and cleanup priorities
        self.file_registry = {}
        self.cleanup_priority = [
            'pdfs',      # Delete first - largest files
            'text',      # Delete after SOC extraction
            'temp',      # Always clean temp files
        ]
        
    def _setup_directories(self):
        """Create storage directory structure"""
        
        # Create base directory
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        print(f"✅ Storage directories created in: {self.base_dir}")
        print(f"📊 Storage limit: {self.max_storage_gb:.1f} GB")
    
    def get_storage_usage(self) -> Dict[str, float]:
        """Get current storage usage by directory (in MB)"""
        
        usage = {}
        total_bytes = 0
        
        for name, dir_path in self.dirs.items():
            if dir_path.exists():
                size_bytes = sum(
                    f.stat().st_size 
                    for f in dir_path.rglob('*') 
                    if f.is_file()
                )
                usage[name] = size_bytes / (1024 * 1024)  # Convert to MB
                total_bytes += size_bytes
            else:
                usage[name] = 0.0
        
        usage['total_mb'] = total_bytes / (1024 * 1024)
        usage['total_gb'] = total_bytes / (1024 * 1024 * 1024)
        usage['remaining_gb'] = self.max_storage_gb - usage['total_gb']
        
        return usage
    
    def check_storage_space(self, required_mb: float = 0) -> bool:
        """Check if we have enough storage space"""
        
        usage = self.get_storage_usage()
        available_mb = usage['remaining_gb'] * 1024
        
        if required_mb > available_mb:
            print(f"⚠️  Storage warning: Need {required_mb:.1f}MB, only {available_mb:.1f}MB available")
            return False
        
        return True
    
    def cleanup_storage(self, target_free_gb: float = 1.0) -> bool:
        """Clean up storage to free target amount of space"""
        
        print(f"🧹 Starting storage cleanup to free {target_free_gb:.1f}GB...")
        
        initial_usage = self.get_storage_usage()
        bytes_to_free = target_free_gb * 1024 * 1024 * 1024
        bytes_freed = 0
        
        # Clean up in priority order
        for category in self.cleanup_priority:
            if bytes_freed >= bytes_to_free:
                break
                
            dir_path = self.dirs[category]
            if not dir_path.exists():
                continue
            
            print(f"   Cleaning {category} directory...")
            
            # Get all files sorted by size (largest first)
            files = []
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    files.append((file_path, file_path.stat().st_size))
            
            files.sort(key=lambda x: x[1], reverse=True)
            
            # Delete files until we have enough space
            for file_path, file_size in files:
                if bytes_freed >= bytes_to_free:
                    break
                
                try:
                    file_path.unlink()
                    bytes_freed += file_size
                    print(f"      Deleted: {file_path.name} ({file_size / (1024*1024):.1f}MB)")
                except Exception as e:
                    print(f"      Error deleting {file_path}: {e}")
        
        final_usage = self.get_storage_usage()
        freed_gb = (initial_usage['total_gb'] - final_usage['total_gb'])
        
        print(f"✅ Cleanup complete: Freed {freed_gb:.2f}GB")
        print(f"📊 Storage now: {final_usage['total_gb']:.2f}GB / {self.max_storage_gb:.1f}GB")
        
        return freed_gb >= target_free_gb * 0.8  # Allow 20% tolerance
    
    def save_compressed(self, data: str, filename: str, directory: str) -> str:
        """Save data with compression to save space"""
        
        dir_path = self.dirs[directory]
        file_path = dir_path / f"{filename}.gz"
        
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            f.write(data)
        
        # Track file size
        file_size = file_path.stat().st_size
        self.file_registry[str(file_path)] = file_size
        
        return str(file_path)
    
    def load_compressed(self, file_path: str) -> str:
        """Load compressed data"""
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return f.read()
    
    def save_json_compressed(self, data: Dict, filename: str, directory: str) -> str:
        """Save JSON data with compression"""
        
        json_str = json.dumps(data, indent=None, separators=(',', ':'))
        return self.save_compressed(json_str, filename, directory)
    
    def load_json_compressed(self, file_path: str) -> Dict:
        """Load compressed JSON data"""
        
        json_str = self.load_compressed(file_path)
        return json.loads(json_str)
    
    def get_temp_file(self, suffix: str = '') -> str:
        """Get temporary file path that will be auto-cleaned"""
        
        temp_file = tempfile.NamedTemporaryFile(
            dir=self.dirs['temp'],
            suffix=suffix,
            delete=False
        )
        temp_file.close()
        
        return temp_file.name
    
    def monitor_storage(self):
        """Monitor and report current storage usage"""
        
        usage = self.get_storage_usage()
        
        print(f"\n📊 STORAGE USAGE REPORT")
        print(f"=" * 40)
        
        for category, size_mb in usage.items():
            if category.endswith('_mb') or category.endswith('_gb'):
                continue
            
            print(f"{category:12}: {size_mb:6.1f} MB")
        
        print(f"{'='*40}")
        print(f"{'Total':12}: {usage['total_mb']:6.1f} MB ({usage['total_gb']:.2f} GB)")
        print(f"{'Remaining':12}: {usage['remaining_gb']*1024:6.1f} MB ({usage['remaining_gb']:.2f} GB)")
        print(f"{'Usage':12}: {(usage['total_gb']/self.max_storage_gb)*100:5.1f}%")
        
        # Warn if getting close to limit
        if usage['remaining_gb'] < 0.5:
            print(f"⚠️  WARNING: Less than 500MB remaining!")
            return False
        elif usage['remaining_gb'] < 1.0:
            print(f"⚠️  Low space: Less than 1GB remaining")
            
        return True
    
    def estimate_batch_requirements(self, num_papers: int) -> Dict[str, float]:
        """Estimate storage requirements for a batch of papers"""
        
        # Conservative estimates based on paper characteristics
        avg_pdf_mb = 2.5      # Average scientific PDF
        avg_text_kb = 50      # Extracted text
        avg_soc_kb = 5        # SOC JSON data (compressed)
        avg_patterns_kb = 2   # Pattern data (compressed)
        
        estimates = {
            'pdfs_mb': num_papers * avg_pdf_mb,
            'text_mb': num_papers * avg_text_kb / 1024,
            'socs_mb': num_papers * avg_soc_kb / 1024,
            'patterns_mb': num_papers * avg_patterns_kb / 1024,
            'temp_mb': num_papers * 0.5,  # Temporary processing files
        }
        
        estimates['total_mb'] = sum(estimates.values())
        estimates['total_gb'] = estimates['total_mb'] / 1024
        
        return estimates
    
    def plan_batch_processing(self, num_papers: int) -> Dict:
        """Plan batch processing to stay within storage limits"""
        
        estimates = self.estimate_batch_requirements(num_papers)
        current_usage = self.get_storage_usage()
        
        print(f"\n📋 BATCH PROCESSING PLAN")
        print(f"=" * 40)
        print(f"Papers to process: {num_papers}")
        print(f"Estimated space needed: {estimates['total_gb']:.2f} GB")
        print(f"Current usage: {current_usage['total_gb']:.2f} GB")
        print(f"Available space: {current_usage['remaining_gb']:.2f} GB")
        
        plan = {
            'can_process_all': estimates['total_gb'] <= current_usage['remaining_gb'],
            'recommended_batch_size': num_papers,
            'cleanup_needed': False,
            'processing_strategy': 'single_batch'
        }
        
        if not plan['can_process_all']:
            # Calculate optimal batch size
            available_gb = current_usage['remaining_gb'] - 0.5  # Keep 0.5GB buffer
            papers_per_gb = num_papers / estimates['total_gb']
            max_batch_size = int(available_gb * papers_per_gb)
            
            plan['recommended_batch_size'] = max(max_batch_size, 10)  # Minimum 10 papers
            plan['cleanup_needed'] = True
            plan['processing_strategy'] = 'multiple_batches'
            
            print(f"\n⚠️  Not enough space for all papers!")
            print(f"Recommended batch size: {plan['recommended_batch_size']}")
            print(f"Processing strategy: {plan['processing_strategy']}")
        else:
            print(f"\n✅ Sufficient space available!")
            print(f"Can process all {num_papers} papers in single batch")
        
        return plan
    
    def cleanup_temp_files(self):
        """Clean up all temporary files"""
        
        temp_dir = self.dirs['temp']
        if temp_dir.exists():
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Error deleting temp file {file_path}: {e}")
        
        print("🧹 Temporary files cleaned up")

def main():
    """Test storage manager functionality"""
    
    storage = StorageManager()
    
    # Monitor initial storage
    storage.monitor_storage()
    
    # Plan for 100-paper processing
    plan = storage.plan_batch_processing(100)
    
    # Test storage estimation
    estimates = storage.estimate_batch_requirements(100)
    print(f"\n💾 STORAGE ESTIMATES FOR 100 PAPERS:")
    for category, size_mb in estimates.items():
        if category.endswith('_mb'):
            print(f"   {category}: {size_mb:.1f} MB")
        elif category.endswith('_gb'):
            print(f"   {category}: {size_mb:.2f} GB")
    
    print(f"\n✅ Storage manager ready for 100-paper test!")
    return storage

if __name__ == "__main__":
    main()