"""
Test suite for Model Conversion Utilities
Tests PyTorch ↔ ONNX ↔ TorchScript conversions
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
import asyncio
import os

# Import the conversion utilities
from prsm.distillation.model_conversion_utilities import (
    ModelConversionEngine,
    ConversionConfig,
    PyTorchToONNXConverter,
    PyTorchToTorchScriptConverter,
    ONNXToPyTorchConverter
)

class SimpleTestModel(nn.Module):
    """Simple model for testing conversions"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

class ConvTestModel(nn.Module):
    """Convolutional model for testing image-based conversions"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def simple_pytorch_model(temp_dir):
    """Create and save a simple PyTorch model"""
    model = SimpleTestModel()
    model_path = Path(temp_dir) / "simple_model.pth"
    torch.save(model, model_path)
    return str(model_path), model

@pytest.fixture
def conv_pytorch_model(temp_dir):
    """Create and save a convolutional PyTorch model"""
    model = ConvTestModel()
    model_path = Path(temp_dir) / "conv_model.pth"
    torch.save(model, model_path)
    return str(model_path), model

@pytest.fixture
def conversion_engine():
    """Create model conversion engine"""
    return ModelConversionEngine()

class TestModelConversionEngine:
    """Test the main conversion engine"""
    
    def test_engine_initialization(self, conversion_engine):
        """Test engine initializes correctly"""
        assert isinstance(conversion_engine, ModelConversionEngine)
        assert conversion_engine.conversion_history == []
    
    def test_supported_conversions(self, conversion_engine):
        """Test that supported conversions are returned"""
        conversions = conversion_engine.get_supported_conversions()
        assert isinstance(conversions, list)
        assert len(conversions) > 0
        
        # Check format
        for conversion in conversions:
            assert "from" in conversion
            assert "to" in conversion
            assert "description" in conversion
    
    def test_conversion_summary_empty(self, conversion_engine):
        """Test summary with no conversions"""
        summary = conversion_engine.get_conversion_summary()
        assert "message" in summary
        assert summary["message"] == "No conversions performed yet"
    
    def test_clear_history(self, conversion_engine):
        """Test clearing conversion history"""
        # Add dummy history entry
        conversion_engine.conversion_history.append("dummy")
        assert len(conversion_engine.conversion_history) == 1
        
        conversion_engine.clear_history()
        assert len(conversion_engine.conversion_history) == 0

class TestPyTorchToONNXConversion:
    """Test PyTorch to ONNX conversion"""
    
    @pytest.mark.asyncio
    async def test_simple_pytorch_to_onnx(self, simple_pytorch_model, temp_dir, conversion_engine):
        """Test converting simple PyTorch model to ONNX"""
        model_path, model = simple_pytorch_model
        output_path = Path(temp_dir) / "simple_model.onnx"
        
        config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path=model_path,
            output_path=str(output_path),
            input_shapes={"input": [1, 10]},
            optimize=True,
            simplify=False  # Skip simplification to avoid optional dependencies
        )
        
        result = await conversion_engine.convert_model(config)
        
        assert result.success
        assert result.input_format == "pytorch"
        assert result.output_format == "onnx"
        assert result.validation_passed
        assert output_path.exists()
        assert result.model_size_mb > 0
        assert result.conversion_time > 0
    
    @pytest.mark.asyncio
    async def test_conv_pytorch_to_onnx(self, conv_pytorch_model, temp_dir, conversion_engine):
        """Test converting convolutional PyTorch model to ONNX"""
        model_path, model = conv_pytorch_model
        output_path = Path(temp_dir) / "conv_model.onnx"
        
        config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path=model_path,
            output_path=str(output_path),
            input_shapes={"input": [1, 3, 32, 32]},
            optimize=False,  # Skip optimization for faster testing
            simplify=False
        )
        
        result = await conversion_engine.convert_model(config)
        
        assert result.success
        assert output_path.exists()
        assert result.validation_passed
    
    @pytest.mark.asyncio
    async def test_pytorch_to_onnx_missing_file(self, temp_dir, conversion_engine):
        """Test conversion with missing input file"""
        config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path="nonexistent_model.pth",
            output_path=str(Path(temp_dir) / "output.onnx"),
            input_shapes={"input": [1, 10]}
        )
        
        result = await conversion_engine.convert_model(config)
        
        assert not result.success
        assert "not found" in result.error_message

class TestPyTorchToTorchScriptConversion:
    """Test PyTorch to TorchScript conversion"""
    
    @pytest.mark.asyncio
    async def test_simple_pytorch_to_torchscript(self, simple_pytorch_model, temp_dir, conversion_engine):
        """Test converting simple PyTorch model to TorchScript"""
        model_path, model = simple_pytorch_model
        output_path = Path(temp_dir) / "simple_model.pt"
        
        config = ConversionConfig(
            input_format="pytorch",
            output_format="torchscript",
            input_path=model_path,
            output_path=str(output_path),
            input_shapes={"input": [1, 10]},
            optimize=True
        )
        
        result = await conversion_engine.convert_model(config)
        
        assert result.success
        assert result.input_format == "pytorch"
        assert result.output_format == "torchscript"
        assert result.validation_passed
        assert output_path.exists()
        assert result.model_size_mb > 0
        assert "conversion_method" in result.metadata
    
    @pytest.mark.asyncio
    async def test_conv_pytorch_to_torchscript(self, conv_pytorch_model, temp_dir, conversion_engine):
        """Test converting convolutional PyTorch model to TorchScript"""
        model_path, model = conv_pytorch_model
        output_path = Path(temp_dir) / "conv_model.pt"
        
        config = ConversionConfig(
            input_format="pytorch",
            output_format="torchscript",
            input_path=model_path,
            output_path=str(output_path),
            input_shapes={"input": [1, 3, 32, 32]},
            optimize=False
        )
        
        result = await conversion_engine.convert_model(config)
        
        assert result.success
        assert output_path.exists()
        assert result.validation_passed

class TestONNXToPyTorchConversion:
    """Test ONNX to PyTorch conversion"""
    
    @pytest.mark.asyncio
    async def test_onnx_to_pytorch_wrapper(self, simple_pytorch_model, temp_dir, conversion_engine):
        """Test creating PyTorch wrapper for ONNX model"""
        # First convert PyTorch to ONNX
        model_path, model = simple_pytorch_model
        onnx_path = Path(temp_dir) / "simple_model.onnx"
        
        onnx_config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path=model_path,
            output_path=str(onnx_path),
            input_shapes={"input": [1, 10]},
            optimize=False,
            simplify=False
        )
        
        onnx_result = await conversion_engine.convert_model(onnx_config)
        assert onnx_result.success
        
        # Then convert ONNX to PyTorch wrapper
        wrapper_path = Path(temp_dir) / "onnx_wrapper.py"
        
        wrapper_config = ConversionConfig(
            input_format="onnx",
            output_format="pytorch",
            input_path=str(onnx_path),
            output_path=str(wrapper_path)
        )
        
        wrapper_result = await conversion_engine.convert_model(wrapper_config)
        
        assert wrapper_result.success
        assert wrapper_path.exists()
        assert "wrapper_approach" in wrapper_result.metadata
        
        # Verify wrapper code contains expected components
        wrapper_code = wrapper_path.read_text()
        assert "ONNXModelWrapper" in wrapper_code
        assert "onnxruntime" in wrapper_code
        assert "load_model" in wrapper_code

class TestBatchConversion:
    """Test batch conversion functionality"""
    
    @pytest.mark.asyncio
    async def test_batch_conversion(self, simple_pytorch_model, temp_dir, conversion_engine):
        """Test converting multiple models in batch"""
        model_path, model = simple_pytorch_model
        
        # Create multiple conversion configs
        configs = []
        for i in range(3):
            configs.append(ConversionConfig(
                input_format="pytorch",
                output_format="onnx",
                input_path=model_path,
                output_path=str(Path(temp_dir) / f"batch_model_{i}.onnx"),
                input_shapes={"input": [1, 10]},
                optimize=False,
                simplify=False
            ))
        
        results = await conversion_engine.batch_convert(configs)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Check all files were created
        for i in range(3):
            assert (Path(temp_dir) / f"batch_model_{i}.onnx").exists()

class TestConversionValidation:
    """Test conversion validation functionality"""
    
    def test_pytorch_model_output_consistency(self, simple_pytorch_model):
        """Test that PyTorch model produces consistent outputs"""
        model_path, model = simple_pytorch_model
        model.eval()
        
        # Create test input
        test_input = torch.randn(1, 10)
        
        # Run multiple times
        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)
    
    def test_conversion_config_validation(self):
        """Test conversion config validation"""
        # Test valid config
        config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path="input.pth",
            output_path="output.onnx"
        )
        
        assert config.input_format == "pytorch"
        assert config.output_format == "onnx"
        assert config.opset_version == 11  # Default value
        assert config.optimize is True  # Default value

class TestPerformanceMetrics:
    """Test performance tracking in conversions"""
    
    @pytest.mark.asyncio
    async def test_conversion_metrics(self, simple_pytorch_model, temp_dir, conversion_engine):
        """Test that conversion metrics are properly tracked"""
        model_path, model = simple_pytorch_model
        output_path = Path(temp_dir) / "metrics_test.onnx"
        
        config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path=model_path,
            output_path=str(output_path),
            input_shapes={"input": [1, 10]},
            optimize=False,
            simplify=False
        )
        
        result = await conversion_engine.convert_model(config)
        
        # Check that metrics are tracked
        assert result.conversion_time > 0
        assert result.model_size_mb > 0
        
        # Check summary includes this conversion
        summary = conversion_engine.get_conversion_summary()
        assert summary["total_conversions"] == 1
        assert summary["successful_conversions"] == 1
        assert summary["success_rate"] == 1.0
        assert len(summary["recent_conversions"]) == 1

class TestErrorHandling:
    """Test error handling in conversions"""
    
    @pytest.mark.asyncio
    async def test_unsupported_conversion(self, temp_dir, conversion_engine):
        """Test handling of unsupported conversion types"""
        config = ConversionConfig(
            input_format="unsupported",
            output_format="unknown",
            input_path="dummy.model",
            output_path=str(Path(temp_dir) / "output.model")
        )
        
        result = await conversion_engine.convert_model(config)
        
        assert not result.success
        assert "Unsupported conversion" in result.error_message
    
    @pytest.mark.asyncio
    async def test_corrupted_model_handling(self, temp_dir, conversion_engine):
        """Test handling of corrupted model files"""
        # Create a fake model file with invalid content
        fake_model_path = Path(temp_dir) / "corrupted_model.pth"
        fake_model_path.write_text("This is not a valid PyTorch model")
        
        config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path=str(fake_model_path),
            output_path=str(Path(temp_dir) / "output.onnx"),
            input_shapes={"input": [1, 10]}
        )
        
        result = await conversion_engine.convert_model(config)
        
        assert not result.success
        assert result.error_message is not None

# Integration test that can be run manually
@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_conversion_pipeline(self, temp_dir, conversion_engine):
        """Test complete conversion pipeline: PyTorch → ONNX → TorchScript wrapper"""
        
        # Create original model
        model = SimpleTestModel(input_size=20, hidden_size=50, output_size=10)
        original_path = Path(temp_dir) / "original_model.pth"
        torch.save(model, original_path)
        
        # Step 1: PyTorch → ONNX
        onnx_path = Path(temp_dir) / "converted_model.onnx"
        onnx_config = ConversionConfig(
            input_format="pytorch",
            output_format="onnx",
            input_path=str(original_path),
            output_path=str(onnx_path),
            input_shapes={"input": [1, 20]},
            optimize=True
        )
        
        onnx_result = await conversion_engine.convert_model(onnx_config)
        assert onnx_result.success
        
        # Step 2: PyTorch → TorchScript
        torchscript_path = Path(temp_dir) / "converted_model.pt"
        ts_config = ConversionConfig(
            input_format="pytorch",
            output_format="torchscript",
            input_path=str(original_path),
            output_path=str(torchscript_path),
            input_shapes={"input": [1, 20]}
        )
        
        ts_result = await conversion_engine.convert_model(ts_config)
        assert ts_result.success
        
        # Step 3: ONNX → PyTorch wrapper
        wrapper_path = Path(temp_dir) / "onnx_wrapper.py"
        wrapper_config = ConversionConfig(
            input_format="onnx",
            output_format="pytorch",
            input_path=str(onnx_path),
            output_path=str(wrapper_path)
        )
        
        wrapper_result = await conversion_engine.convert_model(wrapper_config)
        assert wrapper_result.success
        
        # Verify all files exist
        assert onnx_path.exists()
        assert torchscript_path.exists()
        assert wrapper_path.exists()
        
        # Check conversion summary
        summary = conversion_engine.get_conversion_summary()
        assert summary["total_conversions"] == 3
        assert summary["successful_conversions"] == 3

if __name__ == "__main__":
    # Run basic tests manually
    pytest.main([__file__, "-v"])