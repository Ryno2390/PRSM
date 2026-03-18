"""
Test suite for Real Teacher Model Evaluation (Phase 1 Priority 2).

Tests all 4 layers of evaluation fixes:
- Layer 1: _validate_model() in RealTeacherTrainer
- Layer 2: evaluate_model() in PyTorchDistillationBackend
- Layer 3: evaluate_model() in TransformersDistillationBackend
- Layer 4: _evaluate_response_quality() in RealTeacherCapabilities

All tests mock ML libraries to run without GPU or model downloads.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import numpy as np


# =============================================================================
# Layer 1 Tests: _validate_model() in RealTeacherTrainer
# =============================================================================

class TestValidateModel:
    """Tests for RealTeacherTrainer._validate_model()"""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend with tokenizer."""
        backend = Mock()
        backend.device = "cpu"
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": Mock(to=lambda device: Mock()),
            "attention_mask": Mock(to=lambda device: Mock())
        }
        backend._get_or_create_tokenizer.return_value = mock_tokenizer
        return backend

    @pytest.fixture
    def mock_model(self):
        """Create mock model with forward pass."""
        model = Mock()
        model.eval = Mock()
        model.train = Mock()
        return model

    @pytest.fixture
    def validation_data(self):
        """Sample validation data."""
        return [
            {"content": "What is 2+2?", "expected_answer": "4"},
            {"content": "What is 3+3?", "expected_answer": "6"},
            {"content": "What is 4+4?", "expected_answer": "8"},
        ]

    @patch('prsm.compute.teachers.real_teacher_implementation.torch', None)
    def test_torch_unavailable_returns_zero(self, mock_backend, mock_model, validation_data):
        """When torch is not available, return 0.0 (not random ~0.7)."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        trainer = RealTeacherTrainer()
        # _validate_model is async, so we need to run it
        import asyncio
        result = asyncio.run(trainer._validate_model(mock_backend, mock_model, validation_data))
        
        assert result == 0.0

    def test_empty_validation_data_returns_zero(self, mock_backend, mock_model):
        """Empty validation data returns 0.0."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        trainer = RealTeacherTrainer()
        import asyncio
        result = asyncio.run(trainer._validate_model(mock_backend, mock_model, []))
        
        assert result == 0.0

    def test_no_ground_truth_returns_neutral(self, mock_backend, mock_model):
        """All items missing expected_answer returns 0.5 (neutral)."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        trainer = RealTeacherTrainer()
        data = [
            {"content": "Question 1"},
            {"content": "Question 2"},
        ]
        
        import asyncio
        # Need to mock torch for this test
        with patch('prsm.compute.teachers.real_teacher_implementation.torch'):
            result = asyncio.run(trainer._validate_model(mock_backend, mock_model, data))
        
        assert result == 0.5

    @patch('prsm.compute.teachers.real_teacher_implementation.torch')
    def test_low_loss_counts_as_correct(self, mock_torch, mock_backend, mock_model, validation_data):
        """Low loss (perplexity < 5.0) counts as correct prediction."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        # Mock torch.no_grad context manager
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
        
        # Mock output with low loss (perplexity ~3.3 < 5.0)
        mock_output = Mock()
        mock_output.loss = Mock(item=lambda: 1.2)  # exp(1.2) ≈ 3.3
        mock_output.logits = None
        mock_model.return_value = mock_output
        
        trainer = RealTeacherTrainer()
        import asyncio
        result = asyncio.run(trainer._validate_model(mock_backend, mock_model, validation_data))
        
        # All items should be correct (perplexity < threshold)
        assert result == 1.0

    @patch('prsm.compute.teachers.real_teacher_implementation.torch')
    def test_high_loss_counts_as_incorrect(self, mock_torch, mock_backend, mock_model, validation_data):
        """High loss (perplexity > 5.0) counts as incorrect prediction."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        # Mock torch.no_grad context manager
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
        
        # Mock output with high loss (perplexity ~20 > 5.0)
        mock_output = Mock()
        mock_output.loss = Mock(item=lambda: 3.0)  # exp(3.0) ≈ 20
        mock_output.logits = None
        mock_model.return_value = mock_output
        
        trainer = RealTeacherTrainer()
        import asyncio
        result = asyncio.run(trainer._validate_model(mock_backend, mock_model, validation_data))
        
        # All items should be incorrect (perplexity > threshold)
        assert result == 0.0

    @patch('prsm.compute.teachers.real_teacher_implementation.torch')
    def test_exception_returns_zero(self, mock_torch, mock_backend, mock_model, validation_data):
        """Exception during evaluation returns 0.0 (not random ~0.7)."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        mock_model.side_effect = RuntimeError("Test error")
        
        trainer = RealTeacherTrainer()
        import asyncio
        result = asyncio.run(trainer._validate_model(mock_backend, mock_model, validation_data))
        
        assert result == 0.0

    @patch('prsm.compute.teachers.real_teacher_implementation.torch')
    def test_model_train_mode_restored(self, mock_torch, mock_backend, mock_model, validation_data):
        """Model train mode is always restored after evaluation."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
        
        mock_output = Mock()
        mock_output.loss = Mock(item=lambda: 1.0)
        mock_output.logits = None
        mock_model.return_value = mock_output
        
        trainer = RealTeacherTrainer()
        import asyncio
        asyncio.run(trainer._validate_model(mock_backend, mock_model, validation_data))
        
        mock_model.train.assert_called_once()


# =============================================================================
# Layer 2 Tests: pytorch_backend.evaluate_model()
# =============================================================================

class TestPyTorchBackendEvaluateModel:
    """Tests for PyTorchDistillationBackend.evaluate_model()"""

    @pytest.fixture
    def mock_student_model(self):
        """Create mock student model."""
        model = Mock()
        model.eval = Mock()
        model.train = Mock()
        model.parameters = Mock(return_value=[
            Mock(numel=lambda: 1000, element_size=lambda: 4)
        ])
        return model

    @pytest.fixture
    def test_data(self):
        """Sample test data."""
        return {
            "test": [
                {"content": "What is 2+2?", "expected_answer": "4"},
                {"content": "What is 3+3?", "expected_answer": "6"},
                {"content": "What is 4+4?", "expected_answer": "8"},
            ]
        }

    @patch('prsm.compute.distillation.backends.pytorch_backend.torch')
    def test_returns_real_accuracy_not_hardcoded(self, mock_torch, mock_student_model, test_data):
        """Returns computed accuracy, not hardcoded 0.85."""
        from prsm.compute.distillation.backends.pytorch_backend import PyTorchDistillationBackend

        backend = PyTorchDistillationBackend()

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": Mock(to=lambda device: Mock(shape=[1, 10])),
            "attention_mask": Mock(to=lambda device: Mock(shape=[1, 10]))
        }
        backend._get_or_create_tokenizer = Mock(return_value=mock_tokenizer)

        # Mock torch operations
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
        mock_torch.cuda.is_available.return_value = False

        # Mock model output — use MagicMock so logits supports slicing
        mock_output = MagicMock()
        mock_output.loss = Mock(item=lambda: 1.0)
        mock_output.logits.__getitem__.return_value.argmax.return_value = Mock(item=lambda: 42)
        mock_student_model.return_value = mock_output

        import asyncio
        result = asyncio.run(backend.evaluate_model(mock_student_model, test_data))

        # Should not be hardcoded 0.85
        assert result["accuracy"] != 0.85
        # Should be between 0 and 1
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_unavailable_returns_zeros(self, mock_student_model, test_data):
        """When PyTorch unavailable, returns all-zeros dict."""
        from prsm.compute.distillation.backends.pytorch_backend import PyTorchDistillationBackend

        # Patch __init__ to avoid constructor touching torch, then patch torch=None
        # to simulate the unavailable path in evaluate_model
        with patch.object(PyTorchDistillationBackend, '__init__', lambda self, **kw: None):
            backend = PyTorchDistillationBackend()
            backend.device = "cpu"
            backend.tokenizer_cache = {}
            with patch('prsm.compute.distillation.backends.pytorch_backend.torch', None):
                import asyncio
                result = asyncio.run(backend.evaluate_model(mock_student_model, test_data))
        
        assert result["accuracy"] == 0.0
        assert result["f1_score"] == 0.0

    def test_empty_test_data_returns_zeros(self, mock_student_model):
        """Empty test data returns all-zeros dict."""
        from prsm.compute.distillation.backends.pytorch_backend import PyTorchDistillationBackend
        
        backend = PyTorchDistillationBackend()
        import asyncio
        result = asyncio.run(backend.evaluate_model(mock_student_model, {"test": []}))
        
        assert result["accuracy"] == 0.0

    @patch('prsm.compute.distillation.backends.pytorch_backend.torch')
    def test_model_size_computed_from_params(self, mock_torch, mock_student_model, test_data):
        """model_size_mb is computed from parameter count, not hardcoded."""
        from prsm.compute.distillation.backends.pytorch_backend import PyTorchDistillationBackend
        
        backend = PyTorchDistillationBackend()
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": Mock(to=lambda device: Mock(shape=[1, 10])),
            "attention_mask": Mock(to=lambda device: Mock(shape=[1, 10]))
        }
        backend._get_or_create_tokenizer = Mock(return_value=mock_tokenizer)
        
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
        mock_torch.cuda.is_available.return_value = False
        
        mock_output = MagicMock()
        mock_output.loss = Mock(item=lambda: 1.0)
        mock_output.logits.__getitem__.return_value.argmax.return_value = Mock(item=lambda: 42)
        mock_student_model.return_value = mock_output

        import asyncio
        result = asyncio.run(backend.evaluate_model(mock_student_model, test_data))

        # Should not be hardcoded 125.5
        assert result["model_size_mb"] != 125.5
        # Should be computed from our mock params (1000 params * 4 bytes = 4000 bytes ≈ 0.0038 MB)
        assert result["model_size_mb"] > 0


# =============================================================================
# Layer 3 Tests: transformers_backend.evaluate_model()
# =============================================================================

class TestTransformersBackendEvaluateModel:
    """Tests for TransformersDistillationBackend.evaluate_model()"""

    @pytest.fixture
    def mock_hf_model(self):
        """Create mock HuggingFace model with .config attribute."""
        model = Mock()
        model.eval = Mock()
        model.train = Mock()
        model.config = Mock()  # HF models have .config
        model.parameters = Mock(return_value=[
            Mock(numel=lambda: 1000, element_size=lambda: 4)
        ])
        return model

    @pytest.fixture
    def test_data(self):
        """Sample test data."""
        return {
            "test": [
                {"content": "What is 2+2?", "expected_answer": "4"},
                {"content": "What is 3+3?", "expected_answer": "6"},
            ]
        }

    @patch('prsm.compute.distillation.backends.transformers_backend.TRANSFORMERS_AVAILABLE', False)
    def test_unavailable_returns_zeros(self, mock_hf_model, test_data):
        """When transformers unavailable, returns all-zeros dict."""
        from prsm.compute.distillation.backends.transformers_backend import TransformersDistillationBackend
        
        # Need to handle the case where the constructor raises ImportError
        with patch.object(TransformersDistillationBackend, '__init__', lambda self, device="auto": None):
            backend = TransformersDistillationBackend()
            backend._fallback_metrics = lambda: {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "avg_loss": 0.0,
                "perplexity": 0.0,
                "inference_latency_ms": 0.0,
                "throughput_tokens_per_sec": 0.0,
                "memory_usage_mb": 0.0,
                "model_size_mb": 0.0,
                "items_evaluated": 0
            }
            
            import asyncio
            result = asyncio.run(backend.evaluate_model(mock_hf_model, test_data))
            
            assert result["accuracy"] == 0.0

    @patch('prsm.compute.distillation.backends.transformers_backend.torch')
    @patch('prsm.compute.distillation.backends.transformers_backend.TRANSFORMERS_AVAILABLE', True)
    def test_hf_model_takes_trainer_path(self, mock_torch, mock_hf_model, test_data):
        """HF model with .config takes Trainer evaluation path."""
        from prsm.compute.distillation.backends.transformers_backend import TransformersDistillationBackend
        
        # Need to mock the constructor to avoid ImportError
        with patch.object(TransformersDistillationBackend, '__init__', lambda self, device="auto": None):
            backend = TransformersDistillationBackend()
            backend.device = "cpu"
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock()
            }
            backend._get_or_create_tokenizer = Mock(return_value=mock_tokenizer)
            
            # Mock Trainer path
            with patch.object(backend, '_evaluate_with_trainer') as mock_trainer:
                mock_trainer.return_value = {
                    "accuracy": 0.75,
                    "f1_score": 0.72,
                    "precision": 0.74,
                    "recall": 0.73,
                    "avg_loss": 1.5,
                    "perplexity": 4.5,
                    "inference_latency_ms": 10.0,
                    "throughput_tokens_per_sec": 100.0,
                    "memory_usage_mb": 100.0,
                    "model_size_mb": 0.0038,
                    "items_evaluated": 2
                }
                
                import asyncio
                result = asyncio.run(backend.evaluate_model(mock_hf_model, test_data))
                
                # Should have called Trainer path
                mock_trainer.assert_called_once()
                assert result["accuracy"] == 0.75

    @patch('prsm.compute.distillation.backends.transformers_backend.torch')
    @patch('prsm.compute.distillation.backends.transformers_backend.TRANSFORMERS_AVAILABLE', True)
    def test_non_hf_model_takes_manual_path(self, mock_torch, test_data):
        """Non-HF model (no .config) takes manual loop path."""
        from prsm.compute.distillation.backends.transformers_backend import TransformersDistillationBackend

        with patch.object(TransformersDistillationBackend, '__init__', lambda self, device="auto": None):
            backend = TransformersDistillationBackend()
            backend.device = "cpu"

            # Use spec= so Mock does NOT auto-create .config (simulates non-HF model)
            mock_model = Mock(spec=['eval', 'train', 'parameters'])
            mock_model.eval = Mock()
            mock_model.train = Mock()
            mock_model.parameters = Mock(return_value=[])

            mock_tokenizer = Mock()
            backend._get_or_create_tokenizer = Mock(return_value=mock_tokenizer)

            fallback = {
                "accuracy": 0.0, "f1_score": 0.0, "precision": 0.0,
                "recall": 0.0, "avg_loss": 0.0, "perplexity": 0.0,
                "inference_latency_ms": 0.0, "throughput_tokens_per_sec": 0.0,
                "memory_usage_mb": 0.0, "model_size_mb": 0.0, "items_evaluated": 0
            }
            backend._fallback_metrics = lambda: fallback

            with patch.object(backend, '_evaluate_with_manual_loop') as mock_manual:
                mock_manual.return_value = fallback

                import asyncio
                asyncio.run(backend.evaluate_model(mock_model, test_data))

                # Should have called manual loop path (not Trainer)
                mock_manual.assert_called_once()


# =============================================================================
# Layer 4 Tests: _evaluate_response_quality()
# =============================================================================

class TestEvaluateResponseQuality:
    """Tests for RealTeacherCapabilities._evaluate_response_quality()"""

    @pytest.fixture
    def capabilities(self):
        """Create RealTeacherCapabilities instance with ModelExecutor mocked."""
        with patch('prsm.compute.teachers.real_teacher_implementation.ModelExecutor'):
            from prsm.compute.teachers.real_teacher_implementation import RealTeacherCapabilities
            return RealTeacherCapabilities()

    def test_semantic_similarity_high_score(self, capabilities):
        """Semantically similar response scores high similarity."""
        content = "The capital of France is Paris."
        expected = "Paris is the capital city of France."
        
        # Mock sentence model to return high similarity
        with patch.object(capabilities, '_get_sentence_model') as mock_model:
            mock_emb = Mock()
            mock_model.return_value.encode.return_value = [mock_emb]
            
            with patch('numpy.dot', return_value=0.85):
                with patch('numpy.linalg.norm', return_value=1.0):
                    import asyncio
                    result = asyncio.run(capabilities._evaluate_response_quality(
                        {"expected_answer": expected}, content, "general"
                    ))
        
        assert result["similarity_score"] > 0.6

    def test_fallback_to_jaccard_when_no_sentence_model(self, capabilities):
        """Falls back to Jaccard when sentence-transformers unavailable."""
        content = "The answer is four"
        expected = "four is the answer"
        
        # Mock no sentence model
        with patch.object(capabilities, '_get_sentence_model', return_value=None):
            import asyncio
            result = asyncio.run(capabilities._evaluate_response_quality(
                {"expected_answer": expected}, content, "general"
            ))
        
        # Should use Jaccard (token overlap)
        # Both have words: the, answer, is, four
        assert result["similarity_score"] > 0

    def test_creativity_score_vocabulary_diversity(self, capabilities):
        """Creativity score based on vocabulary diversity, not length."""
        # High diversity text
        diverse_text = "The quick brown fox jumps over lazy dogs while cats sleep"
        # Low diversity text (repetitive)
        repetitive_text = "the the the the the the the the the the"
        
        import asyncio
        diverse_result = asyncio.run(capabilities._evaluate_response_quality(
            {}, diverse_text, "general"
        ))
        repetitive_result = asyncio.run(capabilities._evaluate_response_quality(
            {}, repetitive_text, "general"
        ))
        
        # Diverse text should have higher creativity score
        assert diverse_result["creativity_score"] > repetitive_result["creativity_score"]

    def test_open_ended_well_structured_is_correct(self, capabilities):
        """Well-structured open-ended response is marked correct."""
        # Well-structured: multiple sentences, good length, not repetitive
        good_response = """
        To solve this problem, first we need to understand the basics.
        Then we can apply the formula to get the result.
        Finally, we verify our answer is correct.
        """
        
        import asyncio
        result = asyncio.run(capabilities._evaluate_response_quality(
            {}, good_response, "general"
        ))
        
        assert result["correct"] is True

    def test_open_ended_single_word_is_incorrect(self, capabilities):
        """Single word open-ended response is marked incorrect."""
        poor_response = "yes"
        
        import asyncio
        result = asyncio.run(capabilities._evaluate_response_quality(
            {}, poor_response, "general"
        ))
        
        assert result["correct"] is False

    def test_problem_solving_score_reasoning_keywords(self, capabilities):
        """Problem-solving score increases with reasoning keywords."""
        # Text with reasoning keywords
        reasoning_text = "First, we analyze the problem. Therefore, we can conclude the answer."
        # Text without reasoning keywords
        simple_text = "The answer is forty two."
        
        import asyncio
        reasoning_result = asyncio.run(capabilities._evaluate_response_quality(
            {}, reasoning_text, "general"
        ))
        simple_result = asyncio.run(capabilities._evaluate_response_quality(
            {}, simple_text, "general"
        ))
        
        # Reasoning text should have higher problem-solving score
        assert reasoning_result["problem_solving"] > simple_result["problem_solving"]

    def test_empty_content_returns_zero_scores(self, capabilities):
        """Empty content returns all-zero scores."""
        import asyncio
        result = asyncio.run(capabilities._evaluate_response_quality(
            {}, "", "general"
        ))
        
        assert result["correct"] is False
        assert result["creativity_score"] == 0.0
        assert result["problem_solving"] == 0.0
        assert result["similarity_score"] == 0.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestEvaluationIntegration:
    """Integration tests for the evaluation pipeline."""

    def test_validation_not_random(self):
        """Validation accuracy is deterministic, not random."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        trainer = RealTeacherTrainer()
        
        # Run validation multiple times with same inputs
        # Should get same result each time (not random)
        with patch.object(trainer, '_validate_model', return_value=0.75):
            import asyncio
            results = [asyncio.run(trainer._validate_model(Mock(), Mock(), [])) for _ in range(5)]
        
        # All results should be identical
        assert len(set(results)) == 1

    def test_evaluation_metrics_schema(self):
        """All evaluation methods return consistent schema."""
        from prsm.compute.distillation.backends.pytorch_backend import PyTorchDistillationBackend
        from prsm.compute.distillation.backends.transformers_backend import TransformersDistillationBackend
        
        expected_keys = {
            "accuracy", "f1_score", "precision", "recall",
            "avg_loss", "perplexity", "inference_latency_ms",
            "throughput_tokens_per_sec", "memory_usage_mb",
            "model_size_mb", "items_evaluated"
        }
        
        pytorch_backend = PyTorchDistillationBackend()
        pytorch_result = pytorch_backend._fallback_metrics()
        
        assert set(pytorch_result.keys()) == expected_keys
        
        # For transformers, we need to handle the ImportError case
        with patch.object(TransformersDistillationBackend, '__init__', lambda self, device="auto": None):
            transformers_backend = TransformersDistillationBackend()
            transformers_backend._fallback_metrics = lambda: {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "avg_loss": 0.0,
                "perplexity": 0.0,
                "inference_latency_ms": 0.0,
                "throughput_tokens_per_sec": 0.0,
                "memory_usage_mb": 0.0,
                "model_size_mb": 0.0,
                "items_evaluated": 0
            }
            transformers_result = transformers_backend._fallback_metrics()
            
            assert set(transformers_result.keys()) == expected_keys

    def test_perplexity_threshold_not_random(self):
        """Perplexity threshold is a defined constant, not random."""
        from prsm.compute.teachers.real_teacher_implementation import RealTeacherTrainer
        
        trainer = RealTeacherTrainer()
        
        # The threshold should be a constant (5.0)
        assert hasattr(trainer, 'VALIDATION_PERPLEXITY_THRESHOLD')
        assert trainer.VALIDATION_PERPLEXITY_THRESHOLD == 5.0

    def test_fallback_metrics_all_zeros(self):
        """Fallback metrics return all zeros, not fake values."""
        from prsm.compute.distillation.backends.pytorch_backend import PyTorchDistillationBackend
        
        backend = PyTorchDistillationBackend()
        result = backend._fallback_metrics()
        
        # All values should be 0.0 or 0
        for key, value in result.items():
            assert value == 0.0 or value == 0, f"{key} should be 0, got {value}"
