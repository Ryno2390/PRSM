"""
Comprehensive Test Suite for teacher_loader.py

Phase 1 Priority 3: Tests for the teacher model loading infrastructure.

This test suite covers:
- classify_teacher_source() for all 6 source types
- TeacherModelWrapper class functionality
- load_teacher_model() for HF Hub, local, and API paths
- Graceful degradation behavior
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import os

# Import the module under test
from prsm.compute.distillation.backends.teacher_loader import (
    TeacherSource,
    TeacherModelWrapper,
    classify_teacher_source,
    load_teacher_model,
    _resolve_device,
    _load_local_checkpoint,
    _load_hf_hub,
    _create_api_wrapper,
)


# =============================================================================
# Test Class: TestClassifyTeacherSource
# =============================================================================

class TestClassifyTeacherSource:
    """Tests for classify_teacher_source() function."""
    
    # -------------------------------------------------------------------------
    # LOCAL_CHECKPOINT tests
    # -------------------------------------------------------------------------
    
    @patch('os.path.exists')
    def test_classify_local_checkpoint_existing_file(self, mock_exists):
        """Test LOCAL_CHECKPOINT detection with existing file path."""
        mock_exists.return_value = True
        result = classify_teacher_source("/path/to/model.pt")
        assert result == TeacherSource.LOCAL_CHECKPOINT
        mock_exists.assert_called_once_with("/path/to/model.pt")
    
    @patch('os.path.exists')
    def test_classify_local_checkpoint_directory(self, mock_exists):
        """Test LOCAL_CHECKPOINT detection with existing directory."""
        mock_exists.return_value = True
        result = classify_teacher_source("/models/llama-7b")
        assert result == TeacherSource.LOCAL_CHECKPOINT
    
    @patch('os.path.exists')
    def test_classify_local_checkpoint_relative_path(self, mock_exists):
        """Test LOCAL_CHECKPOINT detection with relative path."""
        mock_exists.return_value = True
        result = classify_teacher_source("./local_models/model.pt")
        assert result == TeacherSource.LOCAL_CHECKPOINT
    
    @patch('os.path.exists')
    def test_classify_local_checkpoint_pth_extension(self, mock_exists):
        """Test LOCAL_CHECKPOINT detection with .pth extension."""
        mock_exists.return_value = True
        result = classify_teacher_source("/checkpoints/model.pth")
        assert result == TeacherSource.LOCAL_CHECKPOINT
    
    # -------------------------------------------------------------------------
    # ANTHROPIC_API tests
    # -------------------------------------------------------------------------
    
    @patch('os.path.exists')
    def test_classify_anthropic_claude_3_opus(self, mock_exists):
        """Test ANTHROPIC_API detection for claude-3-opus."""
        mock_exists.return_value = False
        result = classify_teacher_source("claude-3-opus")
        assert result == TeacherSource.ANTHROPIC_API
    
    @patch('os.path.exists')
    def test_classify_anthropic_claude_2(self, mock_exists):
        """Test ANTHROPIC_API detection for claude-2."""
        mock_exists.return_value = False
        result = classify_teacher_source("claude-2")
        assert result == TeacherSource.ANTHROPIC_API
    
    @patch('os.path.exists')
    def test_classify_anthropic_claude_instant(self, mock_exists):
        """Test ANTHROPIC_API detection for claude-instant."""
        mock_exists.return_value = False
        result = classify_teacher_source("claude-instant-1")
        assert result == TeacherSource.ANTHROPIC_API
    
    @patch('os.path.exists')
    def test_classify_anthropic_explicit_prefix(self, mock_exists):
        """Test ANTHROPIC_API detection with anthropic/ prefix."""
        mock_exists.return_value = False
        result = classify_teacher_source("anthropic/claude-2")
        assert result == TeacherSource.ANTHROPIC_API
    
    @patch('os.path.exists')
    def test_classify_anthropic_case_insensitive(self, mock_exists):
        """Test ANTHROPIC_API detection is case insensitive."""
        mock_exists.return_value = False
        result = classify_teacher_source("CLAUDE-3-OPUS")
        assert result == TeacherSource.ANTHROPIC_API
    
    # -------------------------------------------------------------------------
    # OPENAI_API tests
    # -------------------------------------------------------------------------
    
    @patch('os.path.exists')
    def test_classify_openai_gpt4(self, mock_exists):
        """Test OPENAI_API detection for gpt-4."""
        mock_exists.return_value = False
        result = classify_teacher_source("gpt-4")
        assert result == TeacherSource.OPENAI_API
    
    @patch('os.path.exists')
    def test_classify_openai_gpt35_turbo(self, mock_exists):
        """Test OPENAI_API detection for gpt-3.5-turbo."""
        mock_exists.return_value = False
        result = classify_teacher_source("gpt-3.5-turbo")
        assert result == TeacherSource.OPENAI_API
    
    @patch('os.path.exists')
    def test_classify_openai_o1_preview(self, mock_exists):
        """Test OPENAI_API detection for o1-preview."""
        mock_exists.return_value = False
        result = classify_teacher_source("o1-preview")
        assert result == TeacherSource.OPENAI_API
    
    @patch('os.path.exists')
    def test_classify_openai_o1_mini(self, mock_exists):
        """Test OPENAI_API detection for o1-mini."""
        mock_exists.return_value = False
        result = classify_teacher_source("o1-mini")
        assert result == TeacherSource.OPENAI_API
    
    @patch('os.path.exists')
    def test_classify_openai_text_davinci(self, mock_exists):
        """Test OPENAI_API detection for text-davinci-003."""
        mock_exists.return_value = False
        result = classify_teacher_source("text-davinci-003")
        assert result == TeacherSource.OPENAI_API
    
    @patch('os.path.exists')
    def test_classify_openai_davinci(self, mock_exists):
        """Test OPENAI_API detection for davinci."""
        mock_exists.return_value = False
        result = classify_teacher_source("davinci-002")
        assert result == TeacherSource.OPENAI_API
    
    @patch('os.path.exists')
    def test_classify_openai_case_insensitive(self, mock_exists):
        """Test OPENAI_API detection is case insensitive."""
        mock_exists.return_value = False
        result = classify_teacher_source("GPT-4-TURBO")
        assert result == TeacherSource.OPENAI_API
    
    # -------------------------------------------------------------------------
    # OLLAMA tests
    # -------------------------------------------------------------------------
    
    @patch('os.path.exists')
    def test_classify_ollama_colon_prefix(self, mock_exists):
        """Test OLLAMA detection with ollama: prefix."""
        mock_exists.return_value = False
        result = classify_teacher_source("ollama:llama2")
        assert result == TeacherSource.OLLAMA
    
    @patch('os.path.exists')
    def test_classify_ollama_slash_prefix(self, mock_exists):
        """Test OLLAMA detection with ollama/ prefix."""
        mock_exists.return_value = False
        result = classify_teacher_source("ollama/mistral")
        assert result == TeacherSource.OLLAMA
    
    @patch('os.path.exists')
    def test_classify_ollama_llama2(self, mock_exists):
        """Test OLLAMA detection for llama2 model."""
        mock_exists.return_value = False
        result = classify_teacher_source("ollama:llama2:13b")
        assert result == TeacherSource.OLLAMA
    
    @patch('os.path.exists')
    def test_classify_ollama_mistral(self, mock_exists):
        """Test OLLAMA detection for mistral model."""
        mock_exists.return_value = False
        result = classify_teacher_source("ollama/mistral:7b")
        assert result == TeacherSource.OLLAMA
    
    @patch('os.path.exists')
    def test_classify_ollama_case_insensitive(self, mock_exists):
        """Test OLLAMA detection is case insensitive."""
        mock_exists.return_value = False
        result = classify_teacher_source("OLLAMA:LLAMA2")
        assert result == TeacherSource.OLLAMA
    
    # -------------------------------------------------------------------------
    # HF_HUB tests
    # -------------------------------------------------------------------------
    
    @patch('os.path.exists')
    def test_classify_hf_hub_llama2(self, mock_exists):
        """Test HF_HUB detection for meta-llama/Llama-2-7b-hf."""
        mock_exists.return_value = False
        result = classify_teacher_source("meta-llama/Llama-2-7b-hf")
        assert result == TeacherSource.HF_HUB
    
    @patch('os.path.exists')
    def test_classify_hf_hub_mistral(self, mock_exists):
        """Test HF_HUB detection for mistralai/Mistral-7B-v0.1."""
        mock_exists.return_value = False
        result = classify_teacher_source("mistralai/Mistral-7B-v0.1")
        assert result == TeacherSource.HF_HUB
    
    @patch('os.path.exists')
    def test_classify_hf_hub_gpt2(self, mock_exists):
        """Test HF_HUB detection for gpt2."""
        mock_exists.return_value = False
        result = classify_teacher_source("openai-community/gpt2")
        assert result == TeacherSource.HF_HUB
    
    @patch('os.path.exists')
    def test_classify_hf_hub_qwen(self, mock_exists):
        """Test HF_HUB detection for Qwen models."""
        mock_exists.return_value = False
        result = classify_teacher_source("Qwen/Qwen2-7B")
        assert result == TeacherSource.HF_HUB
    
    # -------------------------------------------------------------------------
    # UNKNOWN tests
    # -------------------------------------------------------------------------
    
    @patch('os.path.exists')
    def test_classify_unknown_format(self, mock_exists):
        """Test UNKNOWN detection for unrecognized format."""
        mock_exists.return_value = False
        result = classify_teacher_source("unknown-model-format")
        assert result == TeacherSource.UNKNOWN
    
    @patch('os.path.exists')
    def test_classify_unknown_single_word(self, mock_exists):
        """Test UNKNOWN detection for single word model name."""
        mock_exists.return_value = False
        result = classify_teacher_source("randommodel")
        assert result == TeacherSource.UNKNOWN
    
    @patch('os.path.exists')
    def test_classify_unknown_empty_string(self, mock_exists):
        """Test UNKNOWN detection for empty string."""
        mock_exists.return_value = False
        result = classify_teacher_source("")
        assert result == TeacherSource.UNKNOWN
    
    @patch('os.path.exists')
    def test_classify_unknown_none_handling(self, mock_exists):
        """Test UNKNOWN detection handles None-like input."""
        mock_exists.return_value = False
        # Note: None would cause TypeError, but empty string is handled
        result = classify_teacher_source("")
        assert result == TeacherSource.UNKNOWN
    
    # -------------------------------------------------------------------------
    # Priority order tests
    # -------------------------------------------------------------------------
    
    @patch('os.path.exists')
    def test_priority_local_over_hf_format(self, mock_exists):
        """Test that existing local path takes priority over HF format detection."""
        # A path like "meta-llama/Llama-2-7b-hf" could be HF format OR local
        # If it exists locally, it should be LOCAL_CHECKPOINT
        mock_exists.return_value = True
        result = classify_teacher_source("meta-llama/Llama-2-7b-hf")
        assert result == TeacherSource.LOCAL_CHECKPOINT
    
    @patch('os.path.exists')
    def test_priority_anthropic_over_hf_format(self, mock_exists):
        """Test Anthropic detection takes priority over HF format for claude models."""
        mock_exists.return_value = False
        # "anthropic/claude-2" has "/" but should be ANTHROPIC_API
        result = classify_teacher_source("anthropic/claude-2")
        assert result == TeacherSource.ANTHROPIC_API
    
    @patch('os.path.exists')
    def test_priority_ollama_over_hf_format(self, mock_exists):
        """Test Ollama detection takes priority over HF format."""
        mock_exists.return_value = False
        # "ollama/mistral" has "/" but should be OLLAMA
        result = classify_teacher_source("ollama/mistral")
        assert result == TeacherSource.OLLAMA


# =============================================================================
# Test Class: TestTeacherModelWrapper
# =============================================================================

class TestTeacherModelWrapper:
    """Tests for TeacherModelWrapper class."""
    
    # -------------------------------------------------------------------------
    # supports_soft_labels property tests
    # -------------------------------------------------------------------------
    
    def test_supports_soft_labels_true_with_local_model_hf_hub(self):
        """Test supports_soft_labels is True with local model and HF_HUB source."""
        mock_model = MagicMock()
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=mock_model
        )
        assert wrapper.supports_soft_labels is True
    
    def test_supports_soft_labels_true_with_local_model_local_checkpoint(self):
        """Test supports_soft_labels is True with local model and LOCAL_CHECKPOINT source."""
        mock_model = MagicMock()
        wrapper = TeacherModelWrapper(
            model_id="/path/to/model.pt",
            source=TeacherSource.LOCAL_CHECKPOINT,
            local_model=mock_model
        )
        assert wrapper.supports_soft_labels is True
    
    def test_supports_soft_labels_true_with_local_model_unknown(self):
        """Test supports_soft_labels is True with local model and UNKNOWN source."""
        mock_model = MagicMock()
        wrapper = TeacherModelWrapper(
            model_id="unknown-model",
            source=TeacherSource.UNKNOWN,
            local_model=mock_model
        )
        assert wrapper.supports_soft_labels is True
    
    def test_supports_soft_labels_false_with_none_model(self):
        """Test supports_soft_labels is False when model is None."""
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=None
        )
        assert wrapper.supports_soft_labels is False
    
    def test_supports_soft_labels_false_anthropic_api(self):
        """Test supports_soft_labels is False for ANTHROPIC_API source."""
        mock_executor = MagicMock()
        wrapper = TeacherModelWrapper(
            model_id="claude-3-opus",
            source=TeacherSource.ANTHROPIC_API,
            executor=mock_executor
        )
        assert wrapper.supports_soft_labels is False
    
    def test_supports_soft_labels_false_openai_api(self):
        """Test supports_soft_labels is False for OPENAI_API source."""
        mock_executor = MagicMock()
        wrapper = TeacherModelWrapper(
            model_id="gpt-4",
            source=TeacherSource.OPENAI_API,
            executor=mock_executor
        )
        assert wrapper.supports_soft_labels is False
    
    def test_supports_soft_labels_false_ollama(self):
        """Test supports_soft_labels is False for OLLAMA source."""
        mock_executor = MagicMock()
        wrapper = TeacherModelWrapper(
            model_id="ollama:llama2",
            source=TeacherSource.OLLAMA,
            executor=mock_executor
        )
        assert wrapper.supports_soft_labels is False
    
    def test_supports_soft_labels_false_load_error(self):
        """Test supports_soft_labels is False when load_error is set."""
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            load_error="Failed to load model"
        )
        assert wrapper.supports_soft_labels is False
    
    # -------------------------------------------------------------------------
    # get_soft_labels() tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_get_soft_labels_returns_logits(self):
        """Test get_soft_labels returns logits from mock local model."""
        mock_model = MagicMock()
        mock_logits = MagicMock()
        mock_logits.to.return_value = mock_logits
        
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        
        mock_model.return_value = mock_outputs
        mock_model.eval = MagicMock()
        
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=mock_model
        )
        
        input_ids = MagicMock()
        input_ids.to.return_value = input_ids
        attention_mask = MagicMock()
        attention_mask.to.return_value = attention_mask
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.no_grad = MagicMock()
                mock_torch.no_grad.return_value.__enter__ = MagicMock()
                mock_torch.no_grad.return_value.__exit__ = MagicMock()
                
                result = await wrapper.get_soft_labels(input_ids, attention_mask)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_soft_labels_returns_none_for_api_wrapper(self):
        """Test get_soft_labels returns None for API wrapper (no local model)."""
        mock_executor = MagicMock()
        wrapper = TeacherModelWrapper(
            model_id="gpt-4",
            source=TeacherSource.OPENAI_API,
            executor=mock_executor
        )
        
        result = await wrapper.get_soft_labels(MagicMock(), MagicMock())
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_soft_labels_returns_none_no_model(self):
        """Test get_soft_labels returns None when model is None."""
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=None
        )
        
        result = await wrapper.get_soft_labels(MagicMock(), MagicMock())
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_soft_labels_handles_exception(self):
        """Test get_soft_labels handles exceptions gracefully (returns None)."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.side_effect = RuntimeError("Model error")
        
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=mock_model
        )
        
        input_ids = MagicMock()
        input_ids.to.return_value = input_ids
        attention_mask = MagicMock()
        attention_mask.to.return_value = attention_mask
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.no_grad = MagicMock()
                mock_torch.no_grad.return_value.__enter__ = MagicMock()
                mock_torch.no_grad.return_value.__exit__ = MagicMock()
                
                result = await wrapper.get_soft_labels(input_ids, attention_mask)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_soft_labels_dict_output(self):
        """Test get_soft_labels handles dict output with 'logits' key."""
        mock_model = MagicMock()
        mock_logits = MagicMock()
        
        mock_model.return_value = {"logits": mock_logits}
        mock_model.eval = MagicMock()
        
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=mock_model
        )
        
        input_ids = MagicMock()
        input_ids.to.return_value = input_ids
        attention_mask = MagicMock()
        attention_mask.to.return_value = attention_mask
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.no_grad = MagicMock()
                mock_torch.no_grad.return_value.__enter__ = MagicMock()
                mock_torch.no_grad.return_value.__exit__ = MagicMock()
                
                result = await wrapper.get_soft_labels(input_ids, attention_mask)
        
        assert result == mock_logits
    
    # -------------------------------------------------------------------------
    # get_response() tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_get_response_api_path(self):
        """Test get_response with API path and mocked executor."""
        mock_executor = MagicMock()
        mock_executor.generate = AsyncMock(return_value="Test response")
        
        wrapper = TeacherModelWrapper(
            model_id="gpt-4",
            source=TeacherSource.OPENAI_API,
            executor=mock_executor
        )
        
        result = await wrapper.get_response("What is AI?")
        assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_get_response_api_complete_method(self):
        """Test get_response with executor.complete method."""
        # Create a mock with only the 'complete' method (no 'generate')
        mock_executor = MagicMock(spec=['complete'])
        mock_executor.complete = AsyncMock(return_value="Complete response")
        
        wrapper = TeacherModelWrapper(
            model_id="claude-3-opus",
            source=TeacherSource.ANTHROPIC_API,
            executor=mock_executor
        )
        
        result = await wrapper.get_response("Hello")
        assert result == "Complete response"
    
    @pytest.mark.asyncio
    async def test_get_response_api_chat_method(self):
        """Test get_response with executor.chat method."""
        # Create a mock with only the 'chat' method (no 'generate' or 'complete')
        mock_executor = MagicMock(spec=['chat'])
        mock_executor.chat = AsyncMock(return_value="Chat response")
        
        wrapper = TeacherModelWrapper(
            model_id="ollama:llama2",
            source=TeacherSource.OLLAMA,
            executor=mock_executor
        )
        
        result = await wrapper.get_response("Hi there")
        assert result == "Chat response"
    
    @pytest.mark.asyncio
    async def test_get_response_local_path(self):
        """Test get_response with local model path and mocked generate()."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer
        mock_tokenizer.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock()
        }
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Generated response"
        
        # Setup model generate
        mock_output_ids = MagicMock()
        mock_output_ids.__getitem__ = lambda self, idx: MagicMock(shape=[1, 10])
        mock_model.generate = MagicMock(return_value=mock_output_ids)
        mock_model.eval = MagicMock()
        
        wrapper = TeacherModelWrapper(
            model_id="/path/to/model",
            source=TeacherSource.LOCAL_CHECKPOINT,
            local_model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.no_grad = MagicMock()
                mock_torch.no_grad.return_value.__enter__ = MagicMock()
                mock_torch.no_grad.return_value.__exit__ = MagicMock()
                
                result = await wrapper.get_response("Test prompt")
        
        # Result should be from local generation
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_get_response_handles_exception(self):
        """Test get_response handles exceptions gracefully (returns empty string)."""
        mock_executor = MagicMock()
        mock_executor.generate = AsyncMock(side_effect=RuntimeError("API error"))
        
        wrapper = TeacherModelWrapper(
            model_id="gpt-4",
            source=TeacherSource.OPENAI_API,
            executor=mock_executor
        )
        
        result = await wrapper.get_response("Test prompt")
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_get_response_empty_prompt(self):
        """Test get_response returns empty string for empty prompt."""
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB
        )
        
        result = await wrapper.get_response("")
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_get_response_no_model_no_executor(self):
        """Test get_response returns empty string when no model and no executor."""
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=None,
            executor=None
        )
        
        result = await wrapper.get_response("Test prompt")
        assert result == ""
    
    # -------------------------------------------------------------------------
    # __repr__ tests
    # -------------------------------------------------------------------------
    
    def test_repr(self):
        """Test __repr__ returns expected string format."""
        wrapper = TeacherModelWrapper(
            model_id="test/model",
            source=TeacherSource.HF_HUB,
            local_model=MagicMock(),
            device="cuda"
        )
        
        repr_str = repr(wrapper)
        assert "test/model" in repr_str
        assert "hf_hub" in repr_str
        assert "cuda" in repr_str


# =============================================================================
# Test Class: TestLoadTeacherModel
# =============================================================================

class TestLoadTeacherModel:
    """Tests for load_teacher_model() function."""
    
    # -------------------------------------------------------------------------
    # HF Hub path tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_load_hf_hub_success(self):
        """Test load_teacher_model with HF Hub success."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    with patch('prsm.compute.distillation.backends.teacher_loader.AutoTokenizer') as mock_auto_tokenizer:
                        mock_auto_model.from_pretrained.return_value = mock_model
                        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                        
                        wrapper = await load_teacher_model("meta-llama/Llama-2-7b-hf")
        
        assert wrapper.source == TeacherSource.HF_HUB
        assert wrapper.model_id == "meta-llama/Llama-2-7b-hf"
    
    @pytest.mark.asyncio
    async def test_load_hf_hub_oserror_failure(self):
        """Test load_teacher_model with HF Hub OSError failure."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    mock_auto_model.from_pretrained.side_effect = OSError("Model not found")
                    
                    wrapper = await load_teacher_model("nonexistent/model")
        
        assert wrapper.load_error is not None
        assert "Failed to load from HF Hub" in wrapper.load_error
    
    @pytest.mark.asyncio
    async def test_load_hf_hub_sets_pad_token(self):
        """Test that tokenizer pad_token is set when missing."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    with patch('prsm.compute.distillation.backends.teacher_loader.AutoTokenizer') as mock_auto_tokenizer:
                        mock_auto_model.from_pretrained.return_value = mock_model
                        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                        
                        wrapper = await load_teacher_model("test/model")
        
        # Verify pad_token was set
        assert mock_tokenizer.pad_token == "<eos>"
    
    @pytest.mark.asyncio
    async def test_load_hf_hub_model_eval_mode(self):
        """Test that model is put in eval mode."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    with patch('prsm.compute.distillation.backends.teacher_loader.AutoTokenizer') as mock_auto_tokenizer:
                        mock_auto_model.from_pretrained.return_value = mock_model
                        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                        
                        await load_teacher_model("test/model")
        
        mock_model.eval.assert_called_once()
    
    # -------------------------------------------------------------------------
    # Local path tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_load_local_pt_file(self):
        """Test load_teacher_model with local .pt file."""
        mock_model = MagicMock()
        
        with patch('os.path.exists', return_value=True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                    mock_torch.load.return_value = mock_model
                    
                    wrapper = await load_teacher_model("/path/to/model.pt")
        
        assert wrapper.source == TeacherSource.LOCAL_CHECKPOINT
        assert wrapper.model_id == "/path/to/model.pt"
    
    @pytest.mark.asyncio
    async def test_load_local_pth_file(self):
        """Test load_teacher_model with local .pth file."""
        mock_model = MagicMock()
        
        with patch('os.path.exists', return_value=True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                    mock_torch.load.return_value = mock_model
                    
                    wrapper = await load_teacher_model("/path/to/model.pth")
        
        assert wrapper.source == TeacherSource.LOCAL_CHECKPOINT
    
    @pytest.mark.asyncio
    async def test_load_local_hf_directory(self):
        """Test load_teacher_model with HF format directory."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
                    with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                        with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                            with patch('prsm.compute.distillation.backends.teacher_loader.AutoTokenizer') as mock_auto_tokenizer:
                                mock_auto_model.from_pretrained.return_value = mock_model
                                mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                                
                                wrapper = await load_teacher_model("/models/llama-7b")
        
        assert wrapper.source == TeacherSource.LOCAL_CHECKPOINT
    
    # -------------------------------------------------------------------------
    # API path tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_load_api_anthropic(self):
        """Test load_teacher_model with Anthropic API path."""
        mock_executor = MagicMock()
        
        with patch('os.path.exists', return_value=False):
            wrapper = await load_teacher_model("claude-3-opus", executor=mock_executor)
        
        assert wrapper.source == TeacherSource.ANTHROPIC_API
        assert wrapper.model is None
        assert wrapper.executor == mock_executor
    
    @pytest.mark.asyncio
    async def test_load_api_openai(self):
        """Test load_teacher_model with OpenAI API path."""
        mock_executor = MagicMock()
        
        with patch('os.path.exists', return_value=False):
            wrapper = await load_teacher_model("gpt-4", executor=mock_executor)
        
        assert wrapper.source == TeacherSource.OPENAI_API
        assert wrapper.model is None
        assert wrapper.executor == mock_executor
    
    @pytest.mark.asyncio
    async def test_load_api_ollama(self):
        """Test load_teacher_model with Ollama path."""
        mock_executor = MagicMock()
        
        with patch('os.path.exists', return_value=False):
            wrapper = await load_teacher_model("ollama:llama2", executor=mock_executor)
        
        assert wrapper.source == TeacherSource.OLLAMA
        assert wrapper.model is None
        assert wrapper.executor == mock_executor
    
    @pytest.mark.asyncio
    async def test_load_api_no_executor(self):
        """Test load_teacher_model with API path but no executor."""
        with patch('os.path.exists', return_value=False):
            wrapper = await load_teacher_model("gpt-4", executor=None)
        
        assert wrapper.source == TeacherSource.OPENAI_API
        assert wrapper.executor is None
        # Should still return a valid wrapper
    
    # -------------------------------------------------------------------------
    # Graceful degradation tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.asyncio
    async def test_hf_load_failure_returns_wrapper(self):
        """Test HF load failure returns wrapper with model=None."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    mock_auto_model.from_pretrained.side_effect = RuntimeError("Load failed")
                    
                    wrapper = await load_teacher_model("test/model")
        
        assert wrapper is not None
        assert wrapper.model is None
        assert wrapper.load_error is not None
    
    @pytest.mark.asyncio
    async def test_supports_soft_labels_false_on_failure(self):
        """Test wrapper.supports_soft_labels is False on failure."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    mock_auto_model.from_pretrained.side_effect = RuntimeError("Load failed")
                    
                    wrapper = await load_teacher_model("test/model")
        
        assert wrapper.supports_soft_labels is False
    
    @pytest.mark.asyncio
    async def test_training_continues_no_crash(self):
        """Test that training can continue (no crash) after load failure."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    mock_auto_model.from_pretrained.side_effect = OSError("Network error")
                    
                    wrapper = await load_teacher_model("test/model")
        
        # Training should be able to continue
        assert wrapper is not None
        assert wrapper.model is None
        # Student-only training would proceed
    
    @pytest.mark.asyncio
    async def test_never_raises_empty_model_id(self):
        """Test load_teacher_model never raises even with empty model_id."""
        wrapper = await load_teacher_model("")
        
        assert wrapper is not None
        assert wrapper.load_error is not None
    
    @pytest.mark.asyncio
    async def test_never_raises_unexpected_exception(self):
        """Test load_teacher_model catches unexpected exceptions during load."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    mock_auto_model.from_pretrained.side_effect = RuntimeError("Unexpected error")
                    
                    wrapper = await load_teacher_model("test/model")
        
        assert wrapper is not None
        assert wrapper.load_error is not None
    
    @pytest.mark.asyncio
    async def test_unknown_source_fallback(self):
        """Test UNKNOWN source attempts HF Hub fallback."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        
        with patch('os.path.exists', return_value=False):
            with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                    with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                        mock_auto_model.from_pretrained.return_value = mock_model
                        
                        wrapper = await load_teacher_model("unknown-model-format")
        
        # Should attempt HF Hub lookup
        assert wrapper is not None


# =============================================================================
# Test Class: TestResolveDevice
# =============================================================================

class TestResolveDevice:
    """Tests for _resolve_device() function."""
    
    def test_resolve_device_cpu(self):
        """Test _resolve_device returns 'cpu' when specified."""
        result = _resolve_device("cpu")
        assert result == "cpu"
    
    def test_resolve_device_cuda(self):
        """Test _resolve_device returns 'cuda' when specified."""
        result = _resolve_device("cuda")
        assert result == "cuda"
    
    def test_resolve_device_auto_with_cuda(self):
        """Test _resolve_device returns 'cuda' when auto and CUDA available."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                
                result = _resolve_device("auto")
        
        assert result == "cuda"
    
    def test_resolve_device_auto_without_cuda(self):
        """Test _resolve_device returns 'cpu' when auto and CUDA not available."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                
                result = _resolve_device("auto")
        
        assert result == "cpu"
    
    def test_resolve_device_auto_no_torch(self):
        """Test _resolve_device returns 'cpu' when auto and PyTorch not available."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', False):
            result = _resolve_device("auto")
        
        assert result == "cpu"


# =============================================================================
# Test Class: TestCreateApiWrapper
# =============================================================================

class TestCreateApiWrapper:
    """Tests for _create_api_wrapper() function."""
    
    def test_create_api_wrapper_anthropic(self):
        """Test _create_api_wrapper for Anthropic source."""
        mock_executor = MagicMock()
        
        wrapper = _create_api_wrapper(
            model_id="claude-3-opus",
            source=TeacherSource.ANTHROPIC_API,
            executor=mock_executor,
            device="cpu"
        )
        
        assert wrapper.model_id == "claude-3-opus"
        assert wrapper.source == TeacherSource.ANTHROPIC_API
        assert wrapper.model is None
        assert wrapper.executor == mock_executor
    
    def test_create_api_wrapper_openai(self):
        """Test _create_api_wrapper for OpenAI source."""
        mock_executor = MagicMock()
        
        wrapper = _create_api_wrapper(
            model_id="gpt-4",
            source=TeacherSource.OPENAI_API,
            executor=mock_executor,
            device="cpu"
        )
        
        assert wrapper.model_id == "gpt-4"
        assert wrapper.source == TeacherSource.OPENAI_API
    
    def test_create_api_wrapper_ollama(self):
        """Test _create_api_wrapper for Ollama source."""
        mock_executor = MagicMock()
        
        wrapper = _create_api_wrapper(
            model_id="ollama:llama2",
            source=TeacherSource.OLLAMA,
            executor=mock_executor,
            device="cpu"
        )
        
        assert wrapper.model_id == "ollama:llama2"
        assert wrapper.source == TeacherSource.OLLAMA
    
    def test_create_api_wrapper_no_executor(self):
        """Test _create_api_wrapper with no executor."""
        wrapper = _create_api_wrapper(
            model_id="gpt-4",
            source=TeacherSource.OPENAI_API,
            executor=None,
            device="cpu"
        )
        
        assert wrapper.executor is None
        assert wrapper.model is None


# =============================================================================
# Test Class: TestLoadLocalCheckpoint
# =============================================================================

class TestLoadLocalCheckpoint:
    """Tests for _load_local_checkpoint() function."""
    
    @pytest.mark.asyncio
    async def test_load_pt_file(self):
        """Test loading .pt file."""
        mock_model = MagicMock()
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.load.return_value = mock_model
                
                wrapper = await _load_local_checkpoint("/path/to/model.pt", "cpu")
        
        assert wrapper.model == mock_model
        assert wrapper.source == TeacherSource.LOCAL_CHECKPOINT
    
    @pytest.mark.asyncio
    async def test_load_pth_file(self):
        """Test loading .pth file."""
        mock_model = MagicMock()
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.load.return_value = mock_model
                
                wrapper = await _load_local_checkpoint("/path/to/model.pth", "cpu")
        
        assert wrapper.model == mock_model
    
    @pytest.mark.asyncio
    async def test_load_no_torch(self):
        """Test loading without PyTorch available."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', False):
            wrapper = await _load_local_checkpoint("/path/to/model.pt", "cpu")
        
        assert wrapper.model is None
        assert wrapper.load_error is not None
    
    @pytest.mark.asyncio
    async def test_load_exception(self):
        """Test loading with exception."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.torch') as mock_torch:
                mock_torch.load.side_effect = RuntimeError("Load error")
                
                wrapper = await _load_local_checkpoint("/path/to/model.pt", "cpu")
        
        assert wrapper.model is None
        assert wrapper.load_error is not None


# =============================================================================
# Test Class: TestLoadHfHub
# =============================================================================

class TestLoadHfHub:
    """Tests for _load_hf_hub() function."""
    
    @pytest.mark.asyncio
    async def test_load_success(self):
        """Test successful HF Hub load."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    with patch('prsm.compute.distillation.backends.teacher_loader.AutoTokenizer') as mock_auto_tokenizer:
                        mock_auto_model.from_pretrained.return_value = mock_model
                        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
                        
                        wrapper = await _load_hf_hub("test/model", "cpu")
        
        assert wrapper.model == mock_model
        assert wrapper.source == TeacherSource.HF_HUB
    
    @pytest.mark.asyncio
    async def test_load_no_transformers(self):
        """Test loading without Transformers available."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', False):
            wrapper = await _load_hf_hub("test/model", "cpu")
        
        assert wrapper.model is None
        assert wrapper.load_error is not None
    
    @pytest.mark.asyncio
    async def test_load_no_torch(self):
        """Test loading without PyTorch available."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', False):
                wrapper = await _load_hf_hub("test/model", "cpu")
        
        assert wrapper.model is None
        assert wrapper.load_error is not None
    
    @pytest.mark.asyncio
    async def test_load_exception(self):
        """Test loading with exception."""
        with patch('prsm.compute.distillation.backends.teacher_loader.TRANSFORMERS_AVAILABLE', True):
            with patch('prsm.compute.distillation.backends.teacher_loader.TORCH_AVAILABLE', True):
                with patch('prsm.compute.distillation.backends.teacher_loader.AutoModelForCausalLM') as mock_auto_model:
                    mock_auto_model.from_pretrained.side_effect = OSError("Network error")
                    
                    wrapper = await _load_hf_hub("test/model", "cpu")
        
        assert wrapper.model is None
        assert wrapper.load_error is not None


# =============================================================================
# Test Class: TestTeacherSourceEnum
# =============================================================================

class TestTeacherSourceEnum:
    """Tests for TeacherSource enum."""
    
    def test_enum_values(self):
        """Test TeacherSource enum has expected values."""
        assert TeacherSource.HF_HUB.value == "hf_hub"
        assert TeacherSource.LOCAL_CHECKPOINT.value == "local"
        assert TeacherSource.ANTHROPIC_API.value == "anthropic_api"
        assert TeacherSource.OPENAI_API.value == "openai_api"
        assert TeacherSource.OLLAMA.value == "ollama"
        assert TeacherSource.UNKNOWN.value == "unknown"
    
    def test_enum_string_comparison(self):
        """Test TeacherSource enum can be compared to strings."""
        assert TeacherSource.HF_HUB == TeacherSource.HF_HUB
        assert TeacherSource.HF_HUB != TeacherSource.OPENAI_API
    
    def test_enum_str_representation(self):
        """Test TeacherSource enum string representation."""
        assert str(TeacherSource.HF_HUB) == "TeacherSource.HF_HUB"
        assert TeacherSource.HF_HUB.value == "hf_hub"
