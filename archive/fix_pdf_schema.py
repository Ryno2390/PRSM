#!/usr/bin/env python3
"""
Fix PDF Processing Schema
========================

Adds missing database columns to fix the running PDF processing.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.nwtn.external_storage_config import ExternalKnowledgeBase

async def fix_schema():
    """Fix database schema for PDF processing"""
    
    print("🔧 FIXING PDF PROCESSING SCHEMA")
    print("=" * 50)
    print("Adding missing database columns for content_length and processed_date")
    print()
    
    try:
        # Initialize knowledge base
        kb = ExternalKnowledgeBase()
        await kb.initialize()
        
        # The schema upgrade should happen automatically during initialization
        print("✅ Schema upgrade completed!")
        print("The PDF download process should now store content successfully.")
        
        # Check that the columns exist
        cursor = kb.storage_manager.storage_db.cursor()
        cursor.execute("PRAGMA table_info(arxiv_papers)")
        columns = [row[1] for row in cursor.fetchall()]
        
        missing_columns = []
        required_columns = ['content_length', 'processed_date']
        
        for col in required_columns:
            if col in columns:
                print(f"✅ Column '{col}' exists")
            else:
                missing_columns.append(col)
                print(f"❌ Column '{col}' missing")
        
        if missing_columns:
            print(f"⚠️  Still missing columns: {missing_columns}")
            print("The schema upgrade may not have completed properly.")
        else:
            print("🎉 All required columns are present!")
            print("The PDF processing should now work correctly.")
            
    except Exception as e:
        print(f"❌ Error fixing schema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(fix_schema())