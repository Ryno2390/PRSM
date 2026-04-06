"""NWTN Training Pipeline — collect, validate, export AgentTrace corpus."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from prsm.compute.nwtn.training.models import (
    TrainingConfig,
    TrainingCorpus,
    ModelCard,
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Manages training data collection and export for NWTN fine-tuning."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self._corpus = TrainingCorpus()

    @property
    def corpus(self) -> TrainingCorpus:
        return self._corpus

    def ingest_traces(self, traces: List) -> int:
        """Ingest AgentTrace objects or dicts into the corpus."""
        count = 0
        for trace in traces:
            if hasattr(trace, "to_dict"):
                self._corpus.add_trace(trace.to_dict())
            elif isinstance(trace, dict):
                self._corpus.add_trace(trace)
            count += 1
        return count

    def validate_corpus(self) -> Tuple[bool, List[str]]:
        """Validate the corpus meets minimum requirements."""
        is_valid, errors = self._corpus.validate()
        if self._corpus.total_traces < self.config.min_corpus_size:
            errors.append(
                f"Corpus size {self._corpus.total_traces} below minimum {self.config.min_corpus_size}"
            )
            is_valid = False
        return is_valid, errors

    def export_dataset(self, output_path: Optional[str] = None) -> str:
        """Export corpus as JSONL for fine-tuning."""
        jsonl = self._corpus.export_jsonl()
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(jsonl)
            logger.info(f"Exported {self._corpus.total_traces} traces to {output_path}")
        return jsonl

    def get_corpus_stats(self) -> Dict[str, Any]:
        return self._corpus.stats()

    def create_model_card(
        self,
        model_id: str,
        model_name: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelCard:
        """Create a model card after training."""
        return ModelCard(
            model_id=model_id,
            model_name=model_name,
            base_model=self.config.base_model,
            training_config=self.config,
            corpus_stats=self._corpus.stats(),
            metrics=metrics or {},
        )
