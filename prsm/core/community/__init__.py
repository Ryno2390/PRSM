"""
PRSM Community Management Package
=================================

Community onboarding, early adopter programs, and user engagement systems
for PRSM production launch and ecosystem growth.
"""

from .onboarding import CommunityOnboardingService
from .early_adopters import EarlyAdopterProgram
from .models import *

# Note: the `engagement` submodule (CommunityEngagement) was removed; its
# import was left behind and broke `import prsm.core.community` with a
# ModuleNotFoundError. Nothing imports CommunityEngagement from this
# package, so the stale import + __all__ entry are dropped (the DB model
# CommunityEngagementRecord still lives in .models and is unaffected).
__all__ = [
    "CommunityOnboardingService",
    "EarlyAdopterProgram",
]