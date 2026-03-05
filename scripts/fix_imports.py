#!/usr/bin/env python3
"""
Bulk fix import paths in test files to match actual nested package structure.
"""
import os
import re
from pathlib import Path

# Mapping of old imports to new imports (verified against actual structure)
IMPORT_MAPPINGS = {
    'from prsm.nwtn': 'from prsm.compute.nwtn',
    'from prsm.agents': 'from prsm.compute.agents',
    'from prsm.tokenomics': 'from prsm.economy.tokenomics',
    'from prsm.federation': 'from prsm.compute.federation',
    'from prsm.marketplace': 'from prsm.economy.marketplace',
    'from prsm.governance': 'from prsm.economy.governance',
    'from prsm.api': 'from prsm.interface.api',
    'from prsm.safety': 'from prsm.core.safety',
    'from prsm.collaboration': 'from prsm.compute.collaboration',
    'from prsm.teachers': 'from prsm.compute.teachers',
    'from prsm.integrations': 'from prsm.core.integrations',
    'from prsm.auth': 'from prsm.core.auth',
    'from prsm.improvement': 'from prsm.compute.improvement',
    'from prsm.data_layer': 'from prsm.data.data_layer',
    'from prsm.ipfs': 'from prsm.data.ipfs',
    'from prsm.distillation': 'from prsm.compute.distillation',
    'from prsm.web': 'from prsm.interface.web',
    'from prsm.provenance': 'from prsm.data.provenance',
    'from prsm.content_processing': 'from prsm.data.content_processing',
    'from prsm.analytics': 'from prsm.data.analytics',
    'from prsm.plugins': 'from prsm.compute.plugins',
    'from prsm.monitoring': 'from prsm.core.monitoring',
    'from prsm.enterprise': 'from prsm.core.enterprise',
    'from prsm.embeddings': 'from prsm.data.embeddings',
    'from prsm.chronos': 'from prsm.compute.chronos',
    'from prsm.quality': 'from prsm.compute.quality',
    'from prsm.evaluation': 'from prsm.compute.evaluation',
    'from prsm.cryptography': 'from prsm.core.cryptography',
    'from prsm.benchmarking': 'from prsm.compute.benchmarking',
    'from prsm.ai_orchestration': 'from prsm.compute.ai_orchestration',
    
    # Also fix import statements
    'import prsm.nwtn': 'import prsm.compute.nwtn',
    'import prsm.agents': 'import prsm.compute.agents',
    'import prsm.tokenomics': 'import prsm.economy.tokenomics',
    'import prsm.federation': 'import prsm.compute.federation',
    'import prsm.marketplace': 'import prsm.economy.marketplace',
    'import prsm.governance': 'import prsm.economy.governance',
    'import prsm.api': 'import prsm.interface.api',
    'import prsm.safety': 'import prsm.core.safety',
    'import prsm.collaboration': 'import prsm.compute.collaboration',
}

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        replacements_made = []
        
        # Apply each mapping
        for old_import, new_import in IMPORT_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                replacements_made.append((old_import, new_import))
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(replacements_made)
        
        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    """Process all test files."""
    tests_dir = Path('tests')
    
    if not tests_dir.exists():
        print(f"Error: {tests_dir} directory not found")
        return
    
    total_files = 0
    modified_files = 0
    total_replacements = 0
    
    # Walk through all Python files in tests/
    for py_file in tests_dir.rglob('*.py'):
        total_files += 1
        replacements = fix_imports_in_file(py_file)
        if replacements > 0:
            modified_files += 1
            total_replacements += replacements
            print(f"✓ {py_file.relative_to(tests_dir)}: {replacements} replacements")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Files modified: {modified_files}")
    print(f"  Total replacements: {total_replacements}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
