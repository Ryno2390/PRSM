#!/usr/bin/env python3
"""
Archive Unused NWTN Files
========================

Identifies files that are COMPLETELY unused by the working NWTN pipeline
and moves them to the archive folder to clean up the repository.

Only moves files that are genuinely unused by any component of the pipeline.
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
    '__init__.py',  # Always keep
}

# Additional files that should be kept (important for context/reference)
KEEP_FILES = {
    'README.md',
    'context_rot_prompt.md',
    'nwtn_complete_pipeline.py',  # Original reference implementation
    'streamlined_nwtn_analysis.py',  # Recent analysis
    # Keep any JSON result files
    '*.json',
    # Keep the main working directory structure
    'corpus/',
    'processed_corpus/',
}

def should_keep_file(file_path: str) -> bool:
    """Determine if a file should be kept (not archived)"""
    
    # Convert to relative path from nwtn directory
    rel_path = file_path.replace('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn/', '')
    
    # Always keep files in archive (avoid recursion)
    if rel_path.startswith('archive/'):
        return True
    
    # Always keep corpus and processed_corpus directories
    if rel_path.startswith('corpus/') or rel_path.startswith('processed_corpus/'):
        return True
    
    # Keep files directly used by pipeline
    if rel_path in USED_FILES:
        return True
    
    # Keep important reference files
    if any(keep_pattern in rel_path for keep_pattern in KEEP_FILES):
        return True
    
    # Keep JSON results files
    if rel_path.endswith('.json'):
        return True
    
    # Keep directory structure files
    if rel_path.endswith('__init__.py'):
        return True
    
    return False

def find_unused_files():
    """Find all Python files that are completely unused"""
    
    nwtn_dir = Path('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn')
    unused_files = []
    kept_files = []
    
    # Walk through all Python files
    for py_file in nwtn_dir.rglob('*.py'):
        if should_keep_file(str(py_file)):
            kept_files.append(str(py_file))
        else:
            unused_files.append(str(py_file))
    
    return unused_files, kept_files

def archive_unused_files(unused_files, dry_run=True):
    """Move unused files to the archive directory"""
    
    archive_dir = Path('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn/archive')
    archive_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    
    print(f"📂 Archive directory: {archive_dir}")
    print(f"🔄 Dry run: {'Yes' if dry_run else 'No'}")
    print()
    
    for file_path in unused_files:
        source_path = Path(file_path)
        
        # Skip if already in archive
        if 'archive' in str(source_path):
            continue
            
        # Create relative path from nwtn directory
        nwtn_base = Path('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn')
        rel_path = source_path.relative_to(nwtn_base)
        
        # Create destination path in archive
        dest_path = archive_dir / 'unused_files' / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"{'[DRY RUN] ' if dry_run else ''}Moving: {rel_path} -> archive/unused_files/{rel_path}")
        
        if not dry_run:
            try:
                shutil.move(str(source_path), str(dest_path))
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moving {rel_path}: {e}")
        else:
            moved_count += 1
    
    return moved_count

def analyze_unused_files():
    """Analyze which files are unused and provide summary"""
    
    print("🔍 NWTN UNUSED FILE ANALYSIS")
    print("=" * 50)
    
    unused_files, kept_files = find_unused_files()
    
    print(f"📊 ANALYSIS SUMMARY:")
    print(f"   Total Python files found: {len(unused_files) + len(kept_files)}")
    print(f"   Files used by pipeline: {len(kept_files)}")
    print(f"   Unused files identified: {len(unused_files)}")
    print(f"   Archive ratio: {len(unused_files) / (len(unused_files) + len(kept_files)) * 100:.1f}%")
    print()
    
    # Show files that ARE being used
    print(f"✅ USED FILES ({len(kept_files)} files):")
    for file_path in sorted(kept_files):
        rel_path = file_path.replace('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn/', '')
        if not rel_path.startswith('corpus/') and not rel_path.startswith('processed_corpus/'):
            print(f"   {rel_path}")
    print()
    
    # Group unused files by category
    categories = {}
    for file_path in unused_files:
        rel_path = file_path.replace('/Users/ryneschultz/Documents/GitHub/PRSM/prsm/nwtn/', '')
        
        # Skip files already in archive
        if rel_path.startswith('archive/'):
            continue
            
        # Categorize by directory
        category = rel_path.split('/')[0] if '/' in rel_path else 'root'
        if category not in categories:
            categories[category] = []
        categories[category].append(rel_path)
    
    print(f"❌ UNUSED FILES BY CATEGORY:")
    for category, files in sorted(categories.items()):
        print(f"\n   {category.upper()} ({len(files)} files):")
        for file_path in sorted(files)[:5]:  # Show first 5 files per category
            print(f"     • {file_path}")
        if len(files) > 5:
            print(f"     ... and {len(files) - 5} more files")
    
    return unused_files, kept_files

def main():
    """Main function to analyze and optionally archive unused files"""
    
    # First, analyze what would be archived
    unused_files, kept_files = analyze_unused_files()
    
    # Filter out files already in archive
    unused_files = [f for f in unused_files if 'archive/' not in f]
    
    print(f"\n📋 ARCHIVAL PLAN:")
    print(f"   Files to move to archive: {len(unused_files)}")
    print(f"   Files to keep active: {len(kept_files)}")
    print()
    
    # Show what would be moved (dry run)
    print("🔄 DRY RUN - Files that would be archived:")
    moved_count = archive_unused_files(unused_files, dry_run=True)
    print(f"\nDry run complete: {moved_count} files would be moved to archive/unused_files/")
    
    # Ask for confirmation for actual move
    print(f"\n❓ CONFIRMATION:")
    print("Do you want to actually move these unused files to the archive?")
    print("This will clean up the NWTN directory but preserve all files in archive/unused_files/")
    print("(Enter 'yes' to proceed, anything else to cancel)")
    
    response = input("> ").strip().lower()
    
    if response == 'yes':
        print("\n🚀 EXECUTING ARCHIVAL:")
        actual_moved = archive_unused_files(unused_files, dry_run=False)
        print(f"\n✅ Archival complete: {actual_moved} files moved to archive/unused_files/")
        print("The NWTN directory is now cleaned up while preserving all files.")
    else:
        print("\n❌ Archival cancelled. No files were moved.")
    
    print(f"\n📁 FINAL STATUS:")
    print(f"   Active NWTN files: {len(kept_files)}")
    print(f"   Archived files: {len(unused_files) if response == 'yes' else 0}")
    print(f"   Repository is ready for external audit ✨")

if __name__ == "__main__":
    main()