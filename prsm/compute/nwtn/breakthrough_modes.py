#!/usr/bin/env python3
"""
NWTN Breakthrough Modes
=======================

Breakthrough mode definitions for NWTN reasoning system.

Modes:
- CONSERVATIVE: Established consensus, proven approaches, high confidence threshold
- BALANCED: Balanced approach between conservative and revolutionary
- REVOLUTIONARY: Novel connections, speculative breakthroughs, lower confidence threshold
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class BreakthroughMode(str, Enum):
    """Modes for breakthrough reasoning"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    CREATIVE = "creative"
    REVOLUTIONARY = "revolutionary"
    CUSTOM = "custom"


@dataclass
class CandidateDistribution:
    """Distribution of candidate types for a breakthrough mode"""
    # Conventional candidate types
    synthesis: float = 0.0
    methodological: float = 0.0
    empirical: float = 0.0
    applied: float = 0.0
    theoretical: float = 0.0
    # Breakthrough candidate types
    contrarian: float = 0.0
    cross_domain_transplant: float = 0.0
    assumption_flip: float = 0.0
    speculative_moonshot: float = 0.0
    historical_analogy: float = 0.0


@dataclass
class BreakthroughModeConfig:
    """Configuration for a breakthrough mode"""
    mode: BreakthroughMode
    confidence_threshold: float
    exploration_depth: int
    novelty_weight: float
    risk_tolerance: float
    max_speculative_branches: int
    require_consensus: bool
    description: str = ""
    name: str = ""
    complexity_multiplier: float = 1.0
    quality_tier: str = "standard"
    assumption_challenging_enabled: bool = False
    wild_hypothesis_enabled: bool = False
    impossibility_exploration_enabled: bool = False
    use_cases: List[str] = field(default_factory=list)
    candidate_distribution: CandidateDistribution = field(default_factory=CandidateDistribution)

    def __post_init__(self):
        if not self.name:
            self.name = self.mode.value.replace("_", " ").title()


# Default configurations for each mode
DEFAULT_CONFIGS: Dict[BreakthroughMode, BreakthroughModeConfig] = {
    BreakthroughMode.CONSERVATIVE: BreakthroughModeConfig(
        mode=BreakthroughMode.CONSERVATIVE,
        confidence_threshold=0.85,
        exploration_depth=3,
        novelty_weight=0.2,
        risk_tolerance=0.1,
        max_speculative_branches=1,
        require_consensus=True,
        description="Established consensus, proven approaches, high confidence threshold",
        name="Conservative",
        complexity_multiplier=0.8,
        quality_tier="high_confidence",
        assumption_challenging_enabled=False,
        wild_hypothesis_enabled=False,
        impossibility_exploration_enabled=False,
        use_cases=["Medical/clinical decisions", "Safety-critical analysis",
                   "Regulatory compliance", "Established science questions"],
        candidate_distribution=CandidateDistribution(
            synthesis=0.40, methodological=0.25, empirical=0.20, applied=0.10, theoretical=0.05,
            contrarian=0.0, cross_domain_transplant=0.0, assumption_flip=0.0,
            speculative_moonshot=0.0, historical_analogy=0.0
        ),
    ),
    BreakthroughMode.BALANCED: BreakthroughModeConfig(
        mode=BreakthroughMode.BALANCED,
        confidence_threshold=0.65,
        exploration_depth=5,
        novelty_weight=0.5,
        risk_tolerance=0.4,
        max_speculative_branches=3,
        require_consensus=False,
        description="Balanced approach between conservative and revolutionary",
        name="Balanced",
        complexity_multiplier=1.0,
        quality_tier="standard",
        assumption_challenging_enabled=False,
        wild_hypothesis_enabled=False,
        impossibility_exploration_enabled=False,
        use_cases=["General research questions", "Business strategy",
                   "Technology evaluation", "Market analysis"],
        candidate_distribution=CandidateDistribution(
            synthesis=0.25, methodological=0.20, empirical=0.15, applied=0.10, theoretical=0.10,
            contrarian=0.05, cross_domain_transplant=0.05, assumption_flip=0.05,
            speculative_moonshot=0.03, historical_analogy=0.02
        ),
    ),
    BreakthroughMode.CREATIVE: BreakthroughModeConfig(
        mode=BreakthroughMode.CREATIVE,
        confidence_threshold=0.55,
        exploration_depth=6,
        novelty_weight=0.65,
        risk_tolerance=0.55,
        max_speculative_branches=4,
        require_consensus=False,
        description="Creative exploration with novel connections, moderate breakthrough focus",
        name="Creative",
        complexity_multiplier=1.3,
        quality_tier="exploratory",
        assumption_challenging_enabled=True,
        wild_hypothesis_enabled=False,
        impossibility_exploration_enabled=False,
        use_cases=["Innovation projects", "Product development",
                   "Research ideation", "Cross-domain exploration"],
        candidate_distribution=CandidateDistribution(
            synthesis=0.15, methodological=0.10, empirical=0.10, applied=0.10, theoretical=0.05,
            contrarian=0.15, cross_domain_transplant=0.15, assumption_flip=0.10,
            speculative_moonshot=0.05, historical_analogy=0.05
        ),
    ),
    BreakthroughMode.REVOLUTIONARY: BreakthroughModeConfig(
        mode=BreakthroughMode.REVOLUTIONARY,
        confidence_threshold=0.45,
        exploration_depth=7,
        novelty_weight=0.8,
        risk_tolerance=0.7,
        max_speculative_branches=5,
        require_consensus=False,
        description="Novel connections, speculative breakthroughs, lower confidence threshold",
        name="Revolutionary",
        complexity_multiplier=1.8,
        quality_tier="breakthrough",
        assumption_challenging_enabled=True,
        wild_hypothesis_enabled=True,
        impossibility_exploration_enabled=True,
        use_cases=["Moonshot projects", "Fundamental research",
                   "Paradigm-challenging questions", "Disruptive innovation"],
        candidate_distribution=CandidateDistribution(
            synthesis=0.05, methodological=0.05, empirical=0.05, applied=0.05, theoretical=0.05,
            contrarian=0.20, cross_domain_transplant=0.20, assumption_flip=0.15,
            speculative_moonshot=0.10, historical_analogy=0.10
        ),
    ),
}


class BreakthroughModeManager:
    """Manager for breakthrough mode configurations"""

    def __init__(self):
        self._configs = DEFAULT_CONFIGS.copy()
        self._custom_configs: Dict[str, BreakthroughModeConfig] = {}

    def get_config(self, mode: BreakthroughMode) -> BreakthroughModeConfig:
        """Get configuration for a mode"""
        if mode == BreakthroughMode.CUSTOM:
            raise ValueError("Custom mode requires a specific config name")
        return self._configs.get(mode, self._configs[BreakthroughMode.BALANCED])

    def get_mode_config(self, mode: BreakthroughMode) -> BreakthroughModeConfig:
        """Alias for get_config - get configuration for a mode"""
        return self.get_config(mode)

    def set_config(self, mode: BreakthroughMode, config: BreakthroughModeConfig) -> None:
        """Set configuration for a mode"""
        self._configs[mode] = config

    def register_custom_config(self, name: str, config: BreakthroughModeConfig) -> None:
        """Register a custom configuration"""
        self._custom_configs[name] = config

    def list_modes(self) -> list:
        """List all available modes"""
        return list(BreakthroughMode)

    def _calculate_breakthrough_intensity(self, config: BreakthroughModeConfig) -> float:
        """Calculate a scalar breakthrough intensity score for a mode configuration.

        Returns a value between 0.0 (fully conservative) and 1.0 (fully revolutionary).
        """
        dist = config.candidate_distribution
        breakthrough_total = (
            dist.contrarian + dist.cross_domain_transplant + dist.assumption_flip
            + dist.speculative_moonshot + dist.historical_analogy
        )
        # Combine distribution with other indicators
        risk_component = config.risk_tolerance * 0.4
        novelty_component = config.novelty_weight * 0.4
        distribution_component = breakthrough_total * 0.2
        return round(min(1.0, risk_component + novelty_component + distribution_component), 3)

    def create_reasoning_context(self, mode: BreakthroughMode, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create an enhanced reasoning context dict for a given breakthrough mode."""
        config = self.get_config(mode)
        enhanced = dict(base_context)
        enhanced.update({
            "breakthrough_mode": mode.value,
            "complexity_multiplier": config.complexity_multiplier,
            "quality_tier": config.quality_tier,
            "confidence_threshold": config.confidence_threshold,
            "exploration_depth": config.exploration_depth,
            "assumption_challenging_enabled": config.assumption_challenging_enabled,
            "wild_hypothesis_enabled": config.wild_hypothesis_enabled,
            "impossibility_exploration_enabled": config.impossibility_exploration_enabled,
            "breakthrough_intensity": self._calculate_breakthrough_intensity(config),
        })
        return enhanced

    def get_mode_pricing_info(self, mode: BreakthroughMode) -> Dict[str, Any]:
        """Get pricing-relevant information for a breakthrough mode."""
        config = self.get_config(mode)
        intensity = self._calculate_breakthrough_intensity(config)
        # Estimate processing time based on complexity
        base_time_seconds = 30
        estimated_time = base_time_seconds * config.complexity_multiplier * (1 + config.exploration_depth * 0.1)
        return {
            "mode": mode.value,
            "complexity_multiplier": config.complexity_multiplier,
            "quality_tier": config.quality_tier,
            "estimated_processing_time": f"{estimated_time:.0f}s",
            "breakthrough_intensity": intensity,
            "confidence_threshold": config.confidence_threshold,
        }


# Global manager instance
breakthrough_mode_manager = BreakthroughModeManager()


def get_breakthrough_mode_config(mode: BreakthroughMode) -> BreakthroughModeConfig:
    """Get configuration for a breakthrough mode"""
    return breakthrough_mode_manager.get_config(mode)


def suggest_breakthrough_mode(query: str) -> BreakthroughMode:
    """
    Suggest a breakthrough mode based on query content.

    Analyzes the query for keywords and patterns that indicate
    the appropriate reasoning mode.
    """
    query_lower = query.lower()

    # Keywords suggesting conservative mode
    conservative_keywords = [
        'safety', 'risk', 'medical', 'clinical', 'approved', 'standard',
        'proven', 'established', 'consensus', 'validated', 'regulatory'
    ]

    # Keywords suggesting revolutionary mode
    revolutionary_keywords = [
        'impossible', 'breakthrough', 'revolutionary', 'moonshot', 'novel',
        'innovate', 'disrupt', 'transform', 'paradigm', 'unprecedented'
    ]

    # Count keyword matches
    conservative_score = sum(1 for kw in conservative_keywords if kw in query_lower)
    revolutionary_score = sum(1 for kw in revolutionary_keywords if kw in query_lower)

    # Determine mode based on scores
    if revolutionary_score > conservative_score + 1:
        return BreakthroughMode.REVOLUTIONARY
    elif conservative_score > revolutionary_score + 1:
        return BreakthroughMode.CONSERVATIVE
    else:
        return BreakthroughMode.BALANCED


__all__ = [
    "BreakthroughMode",
    "BreakthroughModeConfig",
    "CandidateDistribution",
    "BreakthroughModeManager",
    "breakthrough_mode_manager",
    "get_breakthrough_mode_config",
    "suggest_breakthrough_mode",
]
