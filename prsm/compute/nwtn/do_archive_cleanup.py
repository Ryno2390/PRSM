#!/usr/bin/env python3
"""
Auto-Archive Unused NWTN Files
==============================

Automatically moves completely unused files to archive/unused_files/ 
to clean up the NWTN directory for external audit readiness.
"""

import os
import shutil
from pathlib import Path

# Files that ARE used by the complete NWTN pipeline (from import tracing)
USED_FILES = {
    'breakthrough_reasoning_coordinator.py',
    'complete_nwtn_pipeline_v4.py', 
    'engines/universal_knowledge_ingestion_engine.py',
    'enhanced_semantic_retriever.py',
    'multi_layer_validation.py',
    'pipeline_health_monitor.py',
    'pipeline_reliability_fixes.py',
    'test_complete_nwtn_v4.py',
    'archive_unused_files.py',  # Keep our archiving script
    'do_archive_cleanup.py',    # Keep this script
    'streamlined_nwtn_analysis.py',  # Keep recent analysis
    '__init__.py',
}

def should_keep_file(file_path: str) -> bool:
    """Determine if a file should be kept (not archived)"""
    
    # Convert to relative path from nwtn directory
    rel_path = file_path.replace('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn/', '')
    
    # Always keep files already in archive
    if rel_path.startswith('archive/'):
        return True
    
    # Always keep corpus and processed_corpus directories
    if rel_path.startswith('corpus/') or rel_path.startswith('processed_corpus/'):
        return True
    
    # Keep files directly used by pipeline
    if rel_path in USED_FILES:
        return True
    
    # Keep important reference files
    if any(pattern in rel_path for pattern in [
        'README.md', 'context_rot_prompt.md', 'nwtn_complete_pipeline.py'
    ]):
        return True
    
    # Keep JSON results files
    if rel_path.endswith('.json'):
        return True
    
    # Keep directory structure files
    if rel_path.endswith('__init__.py'):
        return True
    
    return False

def main():
    """Archive unused files automatically"""
    
    print("🧹 NWTN Directory Cleanup - Archiving Unused Files")
    print("=" * 55)
    
    nwtn_dir = Path('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn')
    archive_dir = nwtn_dir / 'archive' / 'unused_files'
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files to archive
    unused_files = []
    kept_files = []
    
    for py_file in nwtn_dir.rglob('*.py'):
        if should_keep_file(str(py_file)):
            kept_files.append(py_file)
        else:
            unused_files.append(py_file)
    
    print(f"📊 Analysis:")
    print(f"   Total Python files: {len(unused_files) + len(kept_files)}")
    print(f"   Files to keep active: {len(kept_files)}")
    print(f"   Files to archive: {len(unused_files)}")
    print()
    
    # Archive unused files
    moved_count = 0
    for source_path in unused_files:
        # Skip if already in archive
        if 'archive' in str(source_path):
            continue
            
        # Create relative path from nwtn directory
        rel_path = source_path.relative_to(nwtn_dir)
        dest_path = archive_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(source_path), str(dest_path))
            print(f"📦 Archived: {rel_path}")
            moved_count += 1
        except Exception as e:
            print(f"❌ Error archiving {rel_path}: {e}")
    
    print()
    print(f"✅ Cleanup Complete!")
    print(f"   {moved_count} files archived to archive/unused_files/")
    print(f"   {len(kept_files)} active files remain")
    print(f"   Repository is now audit-ready ✨")
    
    # Show remaining active files (excluding corpus/archive)
    print(f"\n📁 Active NWTN files (excluding corpus data):")
    for file_path in sorted(kept_files):
        rel_path = file_path.relative_to(nwtn_dir)
        if not str(rel_path).startswith(('corpus/', 'processed_corpus/', 'archive/')):
            print(f"   {rel_path}")

if __name__ == "__main__":
    main()