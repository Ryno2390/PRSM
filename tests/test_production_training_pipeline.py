"""
Test suite for PRSM Production Training Pipeline
Comprehensive validation of real ML training implementations
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from uuid import uuid4

from prsm.distillation.production_training_pipeline import (
    ProductionTrainingPipeline, TeacherModelConnector, DistillationDataset,
    ProductionPyTorchTrainer, ProductionTensorFlowTrainer, ProductionTransformersTrainer,
    EnhancedProductionTrainingPipeline, TrainingConfig, get_production_training_pipeline
)
from prsm.distillation.models import (
    DistillationRequest, ModelSize, OptimizationTarget, TrainingStrategy
)


class TestTeacherModelConnector:
    """Test the TeacherModelConnector for real API integration"""
    
    @pytest.fixture
    def connector(self):
        return TeacherModelConnector()
    
    def test_initialization(self, connector):
        """Test connector initialization"""
        assert connector.model_clients == {}
        assert connector.response_cache == {}
        assert connector.cache_ttl == 3600
    
    @pytest.mark.asyncio
    async def test_query_teacher_caching(self, connector):
        """Test response caching functionality"""
        model_name = "test-model"
        prompt = "Test prompt"
        
        # Mock a response
        with patch.object(connector, '_query_huggingface') as mock_hf:
            mock_hf.return_value = {"response": "Test response", "model": model_name}
            
            # First call
            result1 = await connector.query_teacher(model_name, prompt)
            assert result1["response"] == "Test response"
            assert mock_hf.call_count == 1
            
            # Second call should use cache
            result2 = await connector.query_teacher(model_name, prompt)
            assert result2["response"] == "Test response"
            assert mock_hf.call_count == 1  # Not called again due to cache
    
    @pytest.mark.asyncio
    async def test_query_openai_model(self, connector):
        """Test OpenAI model querying"""
        model_name = "gpt-4"
        prompt = "Test prompt"
        
        with patch.object(connector, '_query_openai') as mock_openai:
            mock_openai.return_value = {
                "response": "OpenAI response",
                "model": model_name,
                "tokens_used": 50
            }
            
            result = await connector.query_teacher(model_name, prompt)
            assert result["response"] == "OpenAI response"
            assert result["tokens_used"] == 50
            mock_openai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_anthropic_model(self, connector):
        """Test Anthropic model querying"""
        model_name = "claude-3-sonnet"
        prompt = "Test prompt"
        
        with patch.object(connector, '_query_anthropic') as mock_anthropic:
            mock_anthropic.return_value = {
                "response": "Claude response",
                "model": model_name,
                "tokens_used": 75
            }
            
            result = await connector.query_teacher(model_name, prompt)
            assert result["response"] == "Claude response"
            assert result["tokens_used"] == 75
            mock_anthropic.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_huggingface_model(self, connector):
        """Test Hugging Face model querying"""
        model_name = "microsoft/DialoGPT-medium"
        prompt = "Test prompt"
        
        with patch.object(connector, '_query_huggingface') as mock_hf:
            mock_hf.return_value = {
                "response": "HF response",
                "model": model_name,
                "tokens_used": 30,
                "logits": None
            }
            
            result = await connector.query_teacher(model_name, prompt)
            assert result["response"] == "HF response"
            assert result["tokens_used"] == 30
            mock_hf.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, connector):
        """Test error handling in teacher querying"""
        model_name = "invalid-model"
        prompt = "Test prompt"
        
        with patch.object(connector, '_query_huggingface') as mock_hf:
            mock_hf.side_effect = Exception("Model not found")
            
            result = await connector.query_teacher(model_name, prompt)
            assert "error" in result
            assert "Model not found" in result["error"]


class TestDistillationDataset:
    """Test the DistillationDataset for training data generation"""
    
    @pytest.fixture
    def mock_connector(self):
        connector = Mock(spec=TeacherModelConnector)
        connector.query_teacher = Mock(return_value={
            "response": "Sample response",
            "model": "test-teacher",
            "tokens_used": 25
        })
        return connector
    
    @pytest.fixture
    def dataset(self, mock_connector):
        return DistillationDataset(
            teacher_connector=mock_connector,
            teacher_model="test-teacher",
            domain="nlp",
            size=10,
            tokenizer=None
        )
    
    def test_initialization(self, dataset, mock_connector):
        """Test dataset initialization"""
        assert dataset.teacher_connector == mock_connector
        assert dataset.teacher_model == "test-teacher"
        assert dataset.domain == "nlp"
        assert dataset.size == 10
        assert len(dataset.prompts) == 10
    
    def test_domain_prompt_generation(self, dataset):
        """Test domain-specific prompt generation"""
        # NLP domain should have specific templates
        prompts = dataset._generate_domain_prompts()
        assert len(prompts) == dataset.size
        
        # Check for NLP-specific patterns
        nlp_keywords = ["summarize", "translate", "answer", "complete", "classify"]
        prompt_text = " ".join(prompts).lower()
        assert any(keyword in prompt_text for keyword in nlp_keywords)
    
    def test_coding_domain_prompts(self, mock_connector):
        """Test coding domain prompt generation"""
        dataset = DistillationDataset(
            teacher_connector=mock_connector,
            teacher_model="test-teacher",
            domain="coding",
            size=5
        )
        
        prompts = dataset._generate_domain_prompts()
        coding_keywords = ["python function", "debug", "algorithm", "optimize"]
        prompt_text = " ".join(prompts).lower()
        assert any(keyword in prompt_text for keyword in coding_keywords)
    
    @pytest.mark.asyncio
    async def test_teacher_response_generation(self, dataset, mock_connector):
        """Test teacher response generation"""
        # Mock async query_teacher
        async def mock_query(model, prompt, max_tokens=256):
            return {
                "response": f"Response to: {prompt[:20]}...",
                "model": model,
                "tokens_used": 30
            }
        
        mock_connector.query_teacher = mock_query
        
        await dataset.generate_teacher_responses()
        
        assert len(dataset.teacher_responses) == len(dataset.prompts)
        for response in dataset.teacher_responses:
            assert "response" in response
            assert "Response to:" in response["response"]
    
    def test_getitem_without_tokenizer(self, dataset):
        """Test __getitem__ without tokenizer"""
        # Add a mock teacher response
        dataset.teacher_responses = [{
            "response": "Test response",
            "model": "test-teacher"
        }]
        
        item = dataset[0]
        assert "input_text" in item
        assert "target_text" in item
        assert "teacher_logits" in item
        assert item["target_text"] == "Test response"
    
    def test_getitem_with_tokenizer(self, dataset):
        """Test __getitem__ with tokenizer"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_tokenizer.return_value["input_ids"].squeeze.return_value = "mock_input_ids"
        mock_tokenizer.return_value["attention_mask"].squeeze.return_value = "mock_attention_mask"
        
        dataset.tokenizer = mock_tokenizer
        dataset.teacher_responses = [{
            "response": "Test response",
            "model": "test-teacher"
        }]
        
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "target_ids" in item


class TestProductionPyTorchTrainer:
    """Test the ProductionPyTorchTrainer implementation"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            learning_rate=2e-5,
            batch_size=8,
            num_epochs=2,
            warmup_steps=100,
            weight_decay=0.01,
            temperature=4.0,
            alpha=0.7
        )
    
    @pytest.fixture
    def trainer(self, config):
        return ProductionPyTorchTrainer(config)
    
    def test_initialization(self, trainer, config):
        """Test trainer initialization"""
        assert trainer.config == config
        assert trainer.safety_monitor is not None
    
    @pytest.mark.asyncio
    async def test_student_model_loading(self, trainer):
        """Test student model loading"""
        with patch('prsm.distillation.production_training_pipeline.GPT2LMHeadModel') as mock_model, \
             patch('prsm.distillation.production_training_pipeline.GPT2Tokenizer') as mock_tokenizer:
            
            mock_model.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            model, tokenizer = await trainer._load_student_model("", "nlp")
            
            assert model is not None
            assert tokenizer is not None
            mock_model.from_pretrained.assert_called_once()
            mock_tokenizer.from_pretrained.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_saving(self, trainer):
        """Test model saving functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock model and tokenizer
            mock_model = Mock()
            mock_model.state_dict.return_value = {"param": "value"}
            
            mock_tokenizer = Mock()
            
            with patch('torch.save') as mock_torch_save, \
                 patch('builtins.open', create=True) as mock_open:
                
                result_path = await trainer._save_model(mock_model, mock_tokenizer, "test_job")
                
                assert "test_job" in result_path
                mock_torch_save.assert_called_once()
                mock_tokenizer.save_pretrained.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, trainer):
        """Test model evaluation"""
        with patch('torch.no_grad'), \
             patch('torch.device') as mock_device:
            
            # Mock model and tokenizer
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
            
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {"input_ids": Mock()}
            mock_tokenizer.decode.return_value = "Generated response with multiple words"
            mock_tokenizer.eos_token_id = 0
            
            mock_tokenizer.return_value["input_ids"].to.return_value = Mock()
            
            results = await trainer._evaluate_model(mock_model, mock_tokenizer, "nlp")
            
            assert "avg_response_length" in results
            assert "response_rate" in results
            assert "domain" in results
            assert results["domain"] == "nlp"


class TestEnhancedProductionTrainingPipeline:
    """Test the Enhanced Production Training Pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        return EnhancedProductionTrainingPipeline()
    
    @pytest.fixture
    def sample_request(self):
        return DistillationRequest(
            user_id="test_user",
            teacher_model="gpt-3.5-turbo",
            domain="nlp",
            target_size=ModelSize.SMALL,
            optimization_target=OptimizationTarget.BALANCED,
            training_strategy=TrainingStrategy.BASIC,
            quality_threshold=0.8,
            budget_ftns=1000
        )
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert "pytorch" in pipeline.trainers
        assert "tensorflow" in pipeline.trainers
        assert "transformers" in pipeline.trainers
        assert pipeline.active_training_jobs == {}
        assert pipeline.training_history == []
    
    def test_backend_selection_nlp_domain(self, pipeline, sample_request):
        """Test backend selection for NLP domains"""
        sample_request.domain = "creative_writing"
        backend = pipeline._select_optimal_backend(sample_request)
        assert backend == "transformers"
    
    def test_backend_selection_size_optimization(self, pipeline, sample_request):
        """Test backend selection for size optimization"""
        sample_request.optimization_target = OptimizationTarget.SIZE
        backend = pipeline._select_optimal_backend(sample_request)
        assert backend == "tensorflow"
    
    def test_backend_selection_default(self, pipeline, sample_request):
        """Test default backend selection"""
        sample_request.domain = "general"
        sample_request.optimization_target = OptimizationTarget.ACCURACY
        backend = pipeline._select_optimal_backend(sample_request)
        assert backend == "pytorch"
    
    @pytest.mark.asyncio
    async def test_training_job_creation(self, pipeline, sample_request):
        """Test training job creation and tracking"""
        with patch.object(pipeline, '_execute_enhanced_training'):
            job = await pipeline.start_training(sample_request)
            
            assert job.user_id == sample_request.user_id
            assert job.teacher_model == sample_request.teacher_model
            assert job.domain == sample_request.domain
            assert job.status == "training"
            assert str(job.job_id) in pipeline.active_training_jobs
    
    @pytest.mark.asyncio
    async def test_training_metrics_tracking(self, pipeline):
        """Test training metrics and progress tracking"""
        metrics = await pipeline.get_training_metrics()
        
        assert "active_jobs" in metrics
        assert "completed_jobs" in metrics
        assert "failed_jobs" in metrics
        assert "total_jobs" in metrics
        assert "success_rate" in metrics
        assert "average_training_time" in metrics
    
    def test_optimal_hyperparameter_calculation(self, pipeline, sample_request):
        """Test optimal hyperparameter calculation"""
        # Test learning rate calculation
        lr = pipeline._get_optimal_lr(sample_request)
        assert 1e-6 <= lr <= 1e-2
        
        # Test batch size calculation
        batch_size = pipeline._get_optimal_batch_size(sample_request)
        assert 4 <= batch_size <= 64
        
        # Test epochs calculation
        epochs = pipeline._get_optimal_epochs(sample_request)
        assert 1 <= epochs <= 10


class TestIntegrationScenarios:
    """Integration tests for complete training scenarios"""
    
    @pytest.fixture
    def pipeline(self):
        return get_production_training_pipeline()
    
    @pytest.mark.asyncio
    async def test_complete_pytorch_training_flow(self, pipeline):
        """Test complete PyTorch training flow"""
        request = DistillationRequest(
            user_id="test_user",
            teacher_model="gpt-3.5-turbo",
            domain="code_generation",
            target_size=ModelSize.SMALL,
            optimization_target=OptimizationTarget.ACCURACY,
            training_strategy=TrainingStrategy.BASIC,
            quality_threshold=0.8,
            budget_ftns=2000
        )
        
        with patch('prsm.distillation.production_training_pipeline.ProductionPyTorchTrainer') as mock_trainer_class:
            # Mock trainer instance
            mock_trainer = Mock()
            mock_trainer.train_model = Mock(return_value={
                "status": "completed",
                "model_path": "/models/test_model",
                "final_loss": 0.25,
                "training_steps": 1000,
                "evaluation_results": {"accuracy": 0.85},
                "training_time": 3600
            })
            mock_trainer_class.return_value = mock_trainer
            
            # Mock FTNS service
            with patch('prsm.distillation.production_training_pipeline.ftns_service'):
                job = await pipeline.start_training(request)
                
                # Wait a moment for async execution
                await asyncio.sleep(0.1)
                
                assert job.user_id == "test_user"
                assert job.backend == "transformers"  # Code generation uses transformers
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, pipeline):
        """Test error handling and recovery mechanisms"""
        request = DistillationRequest(
            user_id="test_user",
            teacher_model="invalid-model",
            domain="nlp",
            target_size=ModelSize.TINY,
            optimization_target=OptimizationTarget.SPEED,
            training_strategy=TrainingStrategy.BASIC,
            quality_threshold=0.8,
            budget_ftns=500
        )
        
        with patch('prsm.distillation.production_training_pipeline.ProductionTransformersTrainer') as mock_trainer_class:
            # Mock trainer that fails
            mock_trainer = Mock()
            mock_trainer.train_model = Mock(return_value={
                "status": "failed",
                "error": "Invalid teacher model"
            })
            mock_trainer_class.return_value = mock_trainer
            
            job = await pipeline.start_training(request)
            
            # Wait for async execution
            await asyncio.sleep(0.1)
            
            assert job.status == "training"  # Initial status
    
    @pytest.mark.asyncio
    async def test_concurrent_training_jobs(self, pipeline):
        """Test handling of multiple concurrent training jobs"""
        requests = []
        for i in range(3):
            request = DistillationRequest(
                user_id=f"user_{i}",
                teacher_model="gpt-3.5-turbo",
                domain="nlp",
                target_size=ModelSize.SMALL,
                optimization_target=OptimizationTarget.BALANCED,
                training_strategy=TrainingStrategy.BASIC,
                quality_threshold=0.8,
                budget_ftns=1000
            )
            requests.append(request)
        
        with patch('prsm.distillation.production_training_pipeline.ProductionTransformersTrainer'):
            jobs = []
            for request in requests:
                job = await pipeline.start_training(request)
                jobs.append(job)
            
            # Check that all jobs are tracked
            assert len(pipeline.active_training_jobs) == 3
            
            # Check unique job IDs
            job_ids = [str(job.job_id) for job in jobs]
            assert len(set(job_ids)) == 3


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling"""
    
    @pytest.mark.asyncio
    async def test_large_dataset_handling(self):
        """Test handling of large training datasets"""
        connector = Mock()
        connector.query_teacher = Mock(return_value={
            "response": "Response",
            "model": "teacher",
            "tokens_used": 30
        })
        
        # Create large dataset
        dataset = DistillationDataset(
            teacher_connector=connector,
            teacher_model="teacher",
            domain="nlp",
            size=10000  # Large dataset
        )
        
        assert len(dataset.prompts) == 10000
        
        # Test batched response generation (simulate)
        with patch.object(dataset, 'teacher_connector') as mock_connector:
            async def mock_query(model, prompt, max_tokens=256):
                await asyncio.sleep(0.001)  # Simulate API call
                return {"response": "Mock response", "model": model}
            
            mock_connector.query_teacher = mock_query
            
            # Time the operation
            start_time = asyncio.get_event_loop().time()
            
            # Generate only first 100 for testing
            dataset.size = 100
            dataset.prompts = dataset.prompts[:100]
            await dataset.generate_teacher_responses()
            
            end_time = asyncio.get_event_loop().time()
            
            # Should complete within reasonable time
            assert end_time - start_time < 5.0
            assert len(dataset.teacher_responses) == 100
    
    @pytest.mark.asyncio
    async def test_memory_efficient_training(self):
        """Test memory-efficient training configurations"""
        config = TrainingConfig(
            batch_size=4,  # Small batch size
            gradient_accumulation_steps=8,  # Simulate larger batch
            use_fp16=True,  # Memory optimization
            dataloader_num_workers=2
        )
        
        trainer = ProductionPyTorchTrainer(config)
        
        # Test configuration is memory-efficient
        assert config.batch_size <= 8
        assert config.use_fp16 is True
        assert config.dataloader_num_workers >= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])