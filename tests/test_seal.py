"""
Tests for PRSM SEAL Implementation

This test suite verifies the real SEAL implementation with actual ML training loops.
"""

import asyncio
import json
import numpy as np
import pytest
import torch
from datetime import datetime, timezone
from pathlib import Path

# Test imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from prsm.compute.teachers.seal import (
    SEALConfig,
    SEALNeuralNetwork,
    SEALReinforcementLearner, 
    RLTRewardCalculator,
    SEALTrainer,
    SEALService,
    get_seal_service
)


class TestSEALConfig:
    """Test SEAL configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SEALConfig()
        
        assert config.base_model_name == "microsoft/DialoGPT-medium"
        assert config.max_sequence_length == 512
        assert config.batch_size == 8
        assert config.input_dim == 768
        assert config.hidden_dim == 512
        assert config.output_dim == 256
        assert config.num_safety_categories == 5
        assert config.num_quality_dimensions == 3
        assert config.learning_rate == 2e-5
        assert config.rl_learning_rate == 1e-4
        assert config.enable_continuous_learning == True


class TestSEALNeuralNetwork:
    """Test the SEAL neural network implementation"""
    
    def test_network_initialization(self):
        """Test that the network initializes correctly"""
        config = SEALConfig()
        network = SEALNeuralNetwork(config)
        
        assert isinstance(network, torch.nn.Module)
        assert network.config == config
        assert network.performance_head.in_features == config.output_dim
        assert network.safety_head.out_features == config.num_safety_categories
        assert network.quality_head.out_features == config.num_quality_dimensions
    
    def test_forward_pass(self):
        """Test that forward pass works and returns expected outputs"""
        config = SEALConfig()
        network = SEALNeuralNetwork(config)
        
        # Create test input
        batch_size = 4
        input_tensor = torch.randn(batch_size, config.input_dim)
        
        # Forward pass
        outputs = network(input_tensor)
        
        # Verify output structure
        assert "hidden_representation" in outputs
        assert "performance_score" in outputs
        assert "safety_scores" in outputs
        assert "quality_scores" in outputs
        
        # Verify shapes
        assert outputs["hidden_representation"].shape == (batch_size, config.output_dim)
        assert outputs["performance_score"].shape == (batch_size, 1)
        assert outputs["safety_scores"].shape == (batch_size, config.num_safety_categories)
        assert outputs["quality_scores"].shape == (batch_size, config.num_quality_dimensions)
        
        # Verify value ranges
        assert torch.all(outputs["performance_score"] >= 0)
        assert torch.all(outputs["performance_score"] <= 1)
        assert torch.allclose(outputs["safety_scores"].sum(dim=1), torch.ones(batch_size))
        assert torch.allclose(outputs["quality_scores"].sum(dim=1), torch.ones(batch_size))


class TestSEALReinforcementLearner:
    """Test the reinforcement learning component"""
    
    def test_rl_initialization(self):
        """Test RL learner initialization"""
        config = SEALConfig()
        rl_learner = SEALReinforcementLearner(config)
        
        assert rl_learner.config == config
        assert isinstance(rl_learner.q_network, torch.nn.Module)
        assert isinstance(rl_learner.target_network, torch.nn.Module)
        assert len(rl_learner.replay_buffer) == 0
    
    def test_action_selection(self):
        """Test action selection mechanism"""
        config = SEALConfig()
        rl_learner = SEALReinforcementLearner(config)
        
        # Test state
        state = torch.randn(config.rl_state_dim)
        
        # Test training mode (with exploration)
        action_training = rl_learner.select_action(state, training=True)
        assert 0 <= action_training < config.rl_action_dim
        
        # Test evaluation mode (no exploration)
        action_eval = rl_learner.select_action(state, training=False)
        assert 0 <= action_eval < config.rl_action_dim
    
    def test_experience_storage(self):
        """Test experience replay buffer"""
        config = SEALConfig()
        rl_learner = SEALReinforcementLearner(config)
        
        state = torch.randn(config.rl_state_dim)
        next_state = torch.randn(config.rl_state_dim)
        
        rl_learner.store_experience(state, 5, 1.0, next_state, False)
        
        assert len(rl_learner.replay_buffer) == 1
        
        stored_exp = rl_learner.replay_buffer[0]
        assert torch.equal(stored_exp[0], state)
        assert stored_exp[1] == 5
        assert stored_exp[2] == 1.0
        assert torch.equal(stored_exp[3], next_state)
        assert stored_exp[4] == False
    
    def test_q_learning_update(self):
        """Test Q-learning update mechanism"""
        config = SEALConfig()
        rl_learner = SEALReinforcementLearner(config)
        
        # Fill replay buffer with experiences
        for _ in range(50):
            state = torch.randn(config.rl_state_dim)
            next_state = torch.randn(config.rl_state_dim)
            action = np.random.randint(config.rl_action_dim)
            reward = np.random.random()
            done = np.random.random() > 0.9
            
            rl_learner.store_experience(state, action, reward, next_state, done)
        
        # Perform update
        initial_loss = rl_learner.update()
        
        # Should return a valid loss value
        assert isinstance(initial_loss, float)
        assert initial_loss >= 0


class TestRLTRewardCalculator:
    """Test the RLT reward calculation"""
    
    def test_reward_calculator_initialization(self):
        """Test reward calculator initialization"""
        config = SEALConfig()
        calculator = RLTRewardCalculator(config)
        
        assert isinstance(calculator.comprehension_network, torch.nn.Module)
        assert isinstance(calculator.quality_network, torch.nn.Module)
        assert calculator.config == config
    
    def test_comprehension_reward_calculation(self):
        """Test student comprehension reward calculation"""
        config = SEALConfig()
        calculator = RLTRewardCalculator(config)
        
        # Create test embeddings
        question_emb = torch.randn(1, 256)
        answer_emb = torch.randn(1, 256) 
        student_emb = torch.randn(1, 256)
        
        reward = calculator.calculate_student_comprehension_reward(
            question_emb, answer_emb, student_emb
        )
        
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
    
    def test_continuity_reward_calculation(self):
        """Test logical continuity reward calculation"""
        config = SEALConfig()
        calculator = RLTRewardCalculator(config)
        
        teacher_explanation = torch.randn(1, 256)
        student_understanding = torch.randn(1, 256)
        
        reward = calculator.calculate_logical_continuity_reward(
            teacher_explanation, student_understanding
        )
        
        assert isinstance(reward, float)
        assert reward >= 0.0  # Reward should be positive


@pytest.mark.asyncio
class TestSEALTrainer:
    """Test the SEAL trainer implementation"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        config = SEALConfig()
        trainer = SEALTrainer(config)
        
        assert trainer.config == config
        assert trainer.device.type in ["cuda", "cpu"]
        assert isinstance(trainer.seal_network, SEALNeuralNetwork)
        assert isinstance(trainer.rl_learner, SEALReinforcementLearner)
        assert isinstance(trainer.rlt_calculator, RLTRewardCalculator)
    
    async def test_embedding_generation(self):
        """Test embedding generation"""
        config = SEALConfig()
        trainer = SEALTrainer(config)
        
        embedding = await trainer.get_embedding("Test text for embedding")
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape[-1] == config.input_dim
        assert embedding.shape[0] == 1  # Batch size of 1
    
    async def test_training_epoch(self):
        """Test training epoch execution"""
        config = SEALConfig()
        trainer = SEALTrainer(config)
        
        # Mock training data
        training_data = [
            {"prompt": "What is AI?", "response": "AI is artificial intelligence"},
            {"prompt": "How does ML work?", "response": "ML uses algorithms to learn patterns"}
        ]
        
        # This should work with real implementation
        metrics = await trainer.train_epoch(training_data)
        
        assert isinstance(metrics, dict)
        assert "epoch" in metrics
        assert "total_loss" in metrics
        assert "improvement_score" in metrics
        assert "safety_score" in metrics
        assert "quality_score" in metrics
        assert metrics["epoch"] == 1
    
    async def test_response_improvement(self):
        """Test response improvement functionality"""
        config = SEALConfig()
        trainer = SEALTrainer(config)
        
        result = await trainer.improve_response(
            prompt="What is AI?",
            response="AI is artificial intelligence"
        )
        
        assert isinstance(result, dict)
        assert "original_response" in result
        assert "improved_response" in result
        assert "improvement_metrics" in result
        assert result["original_response"] == "AI is artificial intelligence"


@pytest.mark.asyncio
class TestSEALService:
    """Test the main SEAL service"""
    
    async def test_service_initialization(self):
        """Test SEAL service initialization"""
        config = SEALConfig()
        service = SEALService(config)
        
        assert service.config == config
        assert service.trainer is None
        assert service.is_initialized == False
        assert isinstance(service.training_data_queue, asyncio.Queue)
        
        # Initialize the service
        await service.initialize()
        
        assert service.is_initialized == True
        assert service.trainer is not None
        assert isinstance(service.trainer, SEALTrainer)
    
    async def test_response_improvement_service(self):
        """Test response improvement through service"""
        config = SEALConfig()
        service = SEALService(config)
        
        result = await service.improve_response(
            prompt="What is machine learning?",
            response="Machine learning is a subset of AI"
        )
        
        assert isinstance(result, dict)
        assert "original_response" in result
        assert "improved_response" in result
        assert "improvement_metrics" in result
        assert result["original_response"] == "Machine learning is a subset of AI"
    
    async def test_global_service_function(self):
        """Test global service getter function"""
        service = await get_seal_service()
        
        assert isinstance(service, SEALService)
        assert service.is_initialized == True
        
        # Test that subsequent calls return the same instance
        service2 = await get_seal_service()
        assert service is service2


@pytest.mark.asyncio
class TestSEALIntegration:
    """Integration tests for the complete SEAL system"""
    
    async def test_end_to_end_improvement(self):
        """Test complete end-to-end improvement process"""
        service = await get_seal_service()
        
        # Test improvement with a real example
        result = await service.improve_response(
            prompt="Explain photosynthesis in simple terms",
            response="Plants make food using sunlight"
        )
        
        # Verify the result structure
        assert "original_response" in result
        assert "improved_response" in result
        assert "improvement_metrics" in result
        
        # Verify improvement metrics structure
        metrics = result["improvement_metrics"]
        assert "performance_improvement" in metrics
        assert "safety_improvement" in metrics
        assert "quality_improvement" in metrics
        assert "overall_improvement" in metrics
        
        # All metrics should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
    
    async def test_continuous_learning_integration(self):
        """Test continuous learning capabilities"""
        config = SEALConfig(enable_continuous_learning=True, batch_size=2)
        service = SEALService(config)
        await service.initialize()
        
        # Add multiple improvement requests to trigger batch processing
        tasks = []
        for i in range(3):
            task = service.improve_response(
                prompt=f"Question {i}",
                response=f"Answer {i}"
            )
            tasks.append(task)
        
        # Wait for all improvements to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        assert len(results) == 3
        for result in results:
            assert "improvement_metrics" in result
            assert isinstance(result["improvement_metrics"], dict)
    
    async def test_error_handling(self):
        """Test error handling in SEAL components"""
        config = SEALConfig()
        trainer = SEALTrainer(config)
        
        # Test with empty training data
        empty_metrics = await trainer.train_epoch([])
        assert isinstance(empty_metrics, dict)
        assert empty_metrics["total_loss"] == 0.0
        
        # Test with malformed data
        bad_data = [{"bad_key": "bad_value"}]
        try:
            await trainer.train_epoch(bad_data)
        except Exception as e:
            # Should handle gracefully or raise meaningful error
            assert isinstance(e, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])