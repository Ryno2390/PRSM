"""
PRSM Legacy Integration API
===========================

The "PRSM-Sync" plugin for traditional scientific software.
Automates data flow from Excel, MATLAB, and ELNs into the Knowledge Graph.
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# v1.6.0 scope alignment: prsm.knowledge_system deleted in PR 3
try:
    from prsm.knowledge_system import UnifiedKnowledgeSystem
except ImportError:
    UnifiedKnowledgeSystem = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

class LegacySyncPlugin:
    """
    Auto-Save for Science.
    Hooks into legacy software to stream findings into PRSM.
    """
    def __init__(self, user_id: str, ks: UnifiedKnowledgeSystem):
        self.user_id = user_id
        self.ks = ks
        self.sync_history: List[str] = []

    async def sync_from_excel(self, file_path: str, data: List[Dict[str, Any]], domain: str):
        """
        Simulates a background sync from an Excel spreadsheet.
        """
        payload = json.dumps(data)
        logger.info(f"📊 Legacy-Sync: Syncing {len(data)} rows from {file_path}")
        
        # Automatically ingest into the knowledge system
        cid = await self.ks.ingest_content(
            content=payload,
            title=f"Legacy Sync from {file_path}",
            domain=domain
        )
        
        self.sync_history.append(cid)
        return cid

    async def sync_from_matlab(self, variable_name: str, matrix_data: Any, domain: str):
        """
        Simulates a sync from a MATLAB workspace.
        """
        logger.info(f"🧬 Legacy-Sync: Syncing MATLAB variable {variable_name}")
        
        cid = await self.ks.ingest_content(
            content=str(matrix_data),
            title=f"MATLAB Sync: {variable_name}",
            domain=domain
        )
        
        self.sync_history.append(cid)
        return cid
