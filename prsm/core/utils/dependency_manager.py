"""
PRSM Lazy Dependency Manager (Hydration Engine)
==============================================

Ensures that heavy dependencies (torch, transformers, etc.) are 
only installed when the node actually needs to perform a heavy task.
"""

import sys
import subprocess
import logging
import importlib.util
from typing import List

logger = logging.getLogger(__name__)

class DependencyManager:
    """
    Manages the 'Hydration' of the local environment.
    """
    HEAVY_DEPS = [
        "torch", "transformers", "numpy", "sentence-transformers",
        "web3", "eth-account", "fastapi", "uvicorn", "bleach"
    ]

    @staticmethod
    def is_installed(package_name: str) -> bool:
        """Checks if a package is available in the current environment"""
        # Mapping common import names to package names if they differ
        import_name = package_name.replace("-", "_")
        return importlib.util.find_spec(import_name) is not None

    @classmethod
    def hydrate_environment(cls):
        """
        Installs all heavy dependencies in the background.
        Used when an agent transitions from 'Wizard' to 'Active Research'.
        """
        missing = [p for p in cls.HEAVY_DEPS if not cls.is_installed(p)]
        
        if not missing:
            logger.info("✅ Environment is already fully hydrated.")
            return True

        logger.info(f"🌊 Hydrating environment... Installing missing: {missing}")
        
        try:
            # We use the same python executable that is currently running
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            logger.info("🎉 Hydration Complete! All scientific cores are now active.")
            return True
        except Exception as e:
            logger.error(f"❌ Hydration Failed: {e}")
            return False

    @classmethod
    def ensure_scientific_stack(cls):
        """Blocking check to ensure heavy deps are ready before a task starts"""
        if not all(cls.is_installed(p) for p in cls.HEAVY_DEPS):
            return cls.hydrate_environment()
        return True
