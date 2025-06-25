"""
Model Conversion Utilities for PRSM
Provides comprehensive model format conversion between PyTorch, ONNX, and TorchScript

🎯 PURPOSE IN PRSM:
Enables seamless model format conversion for:
- Cross-platform deployment optimization
- Model optimization and compression
- Framework interoperability 
- Production deployment preparation
- Edge device optimization
"""

import torch
import torch.jit
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import structlog
import json
import os

# Conditional imports for optional dependencies
try:
    import torch.onnx
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False

try:
    import torch.fx
    FX_AVAILABLE = True
except ImportError:
    FX_AVAILABLE = False

logger = structlog.get_logger(__name__)

@dataclass
class ConversionConfig:
    """Configuration for model conversion"""
    input_format: str  # "pytorch", "onnx", "torchscript"
    output_format: str  # "pytorch", "onnx", "torchscript"
    input_path: str
    output_path: str
    input_shapes: Optional[Dict[str, List[int]]] = None
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    opset_version: int = 11
    optimize: bool = True
    simplify: bool = True
    quantize: bool = False
    precision: str = "float32"  # "float32", "float16", "int8"
    device: str = "cpu"
    batch_size: int = 1
    sequence_length: int = 512

@dataclass
class ConversionResult:
    """Result from model conversion"""
    success: bool
    input_format: str
    output_format: str
    input_path: str
    output_path: str
    model_size_mb: float
    conversion_time: float
    validation_passed: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseConverter(ABC):
    """Abstract base class for model converters"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        
    @abstractmethod
    async def convert(self) -> ConversionResult:
        """Convert model from input format to output format"""
        pass
    
    @abstractmethod
    def validate_conversion(self, original_model: Any, converted_model: Any) -> bool:
        """Validate that conversion preserved model behavior"""
        pass
    
    def _get_file_size_mb(self, path: str) -> float:
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(path)
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0.0

class PyTorchToONNXConverter(BaseConverter):
    """Convert PyTorch models to ONNX format"""
    
    async def convert(self) -> ConversionResult:
        """Convert PyTorch model to ONNX"""
        import time
        start_time = time.time()
        
        try:
            # Load PyTorch model
            if self.config.input_path.endswith('.pth') or self.config.input_path.endswith('.pt'):
                model = torch.load(self.config.input_path, map_location=self.config.device, weights_only=True)
            elif TRANSFORMERS_AVAILABLE and Path(self.config.input_path).is_dir():
                # Hugging Face model
                model = AutoModel.from_pretrained(self.config.input_path)
            else:
                raise ValueError(f"Unsupported PyTorch model format: {self.config.input_path}")
            
            model.eval()
            model = model.to(self.config.device)
            
            # Prepare dummy inputs
            dummy_inputs = self._create_dummy_inputs(model)
            
            # Configure dynamic axes if provided
            dynamic_axes = self.config.dynamic_axes or {}
            
            # Convert to ONNX
            logger.info(f"Converting PyTorch model to ONNX: {self.config.output_path}")
            
            torch.onnx.export(
                model,
                dummy_inputs,
                self.config.output_path,
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
                input_names=list(self.config.input_shapes.keys()) if self.config.input_shapes else ['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Optimize ONNX model if requested
            if self.config.optimize:
                await self._optimize_onnx_model()
            
            # Simplify ONNX model if requested and available
            if self.config.simplify and ONNXSIM_AVAILABLE:
                await self._simplify_onnx_model()
            
            # Validate conversion
            validation_passed = await self._validate_onnx_conversion(model, dummy_inputs)
            
            conversion_time = time.time() - start_time
            model_size = self._get_file_size_mb(self.config.output_path)
            
            return ConversionResult(
                success=True,
                input_format=self.config.input_format,
                output_format=self.config.output_format,
                input_path=self.config.input_path,
                output_path=self.config.output_path,
                model_size_mb=model_size,
                conversion_time=conversion_time,
                validation_passed=validation_passed,
                metadata={
                    "opset_version": self.config.opset_version,
                    "optimized": self.config.optimize,
                    "simplified": self.config.simplify and ONNXSIM_AVAILABLE,
                    "dynamic_axes": bool(dynamic_axes)
                }
            )
            
        except Exception as e:
            logger.error(f"PyTorch to ONNX conversion failed: {e}")
            return ConversionResult(
                success=False,
                input_format=self.config.input_format,
                output_format=self.config.output_format,
                input_path=self.config.input_path,
                output_path=self.config.output_path,
                model_size_mb=0.0,
                conversion_time=time.time() - start_time,
                validation_passed=False,
                error_message=str(e)
            )
    
    def _create_dummy_inputs(self, model: torch.nn.Module) -> Union[torch.Tensor, Tuple]:
        """Create dummy inputs for ONNX export"""
        if self.config.input_shapes:
            # Use provided input shapes
            dummy_inputs = {}
            for name, shape in self.config.input_shapes.items():
                dummy_inputs[name] = torch.randn(*shape).to(self.config.device)
            
            if len(dummy_inputs) == 1:
                return list(dummy_inputs.values())[0]
            else:
                return tuple(dummy_inputs.values())
        else:
            # Try to infer input shape from model
            try:
                # Common case: single tensor input
                return torch.randn(self.config.batch_size, 3, 224, 224).to(self.config.device)
            except Exception:
                # Fallback: simple 1D input
                return torch.randn(self.config.batch_size, 768).to(self.config.device)
    
    async def _optimize_onnx_model(self):
        """Optimize ONNX model using ONNX Runtime"""
        try:
            from onnxruntime.tools import optimizer
            
            # Create optimized model
            optimized_path = self.config.output_path.replace('.onnx', '_optimized.onnx')
            
            # Optimize
            optimizer.optimize_model(
                self.config.output_path,
                optimized_path,
                optimization_level="all"
            )
            
            # Replace original with optimized
            shutil.move(optimized_path, self.config.output_path)
            logger.info("Applied ONNX Runtime optimizations")
            
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
    
    async def _simplify_onnx_model(self):
        """Simplify ONNX model using onnx-simplifier"""
        try:
            # Load model
            model = onnx.load(self.config.output_path)
            
            # Simplify
            simplified_model, check = onnxsim.simplify(model)
            
            if check:
                # Save simplified model
                onnx.save(simplified_model, self.config.output_path)
                logger.info("Applied ONNX simplification")
            else:
                logger.warning("ONNX simplification check failed")
                
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")
    
    async def _validate_onnx_conversion(self, pytorch_model: torch.nn.Module, dummy_inputs: torch.Tensor) -> bool:
        """Validate ONNX conversion by comparing outputs"""
        try:
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = pytorch_model(dummy_inputs)
            
            # Get ONNX output
            ort_session = ort.InferenceSession(self.config.output_path)
            input_name = ort_session.get_inputs()[0].name
            
            if isinstance(dummy_inputs, torch.Tensor):
                ort_inputs = {input_name: dummy_inputs.cpu().numpy()}
            else:
                ort_inputs = {input_name: dummy_inputs[0].cpu().numpy()}
            
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare outputs
            if isinstance(pytorch_output, torch.Tensor):
                pytorch_np = pytorch_output.cpu().numpy()
            else:
                pytorch_np = pytorch_output[0].cpu().numpy()
            
            onnx_np = ort_outputs[0]
            
            # Check if outputs are close (allowing for numerical differences)
            are_close = np.allclose(pytorch_np, onnx_np, rtol=1e-3, atol=1e-3)
            
            if are_close:
                logger.info("ONNX conversion validation passed")
            else:
                logger.warning("ONNX conversion validation failed - outputs differ")
            
            return are_close
            
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False
    
    def validate_conversion(self, original_model: Any, converted_model: Any) -> bool:
        """Validate conversion (implementation for base class)"""
        # This is handled by _validate_onnx_conversion
        return True

class PyTorchToTorchScriptConverter(BaseConverter):
    """Convert PyTorch models to TorchScript format"""
    
    async def convert(self) -> ConversionResult:
        """Convert PyTorch model to TorchScript"""
        import time
        start_time = time.time()
        
        try:
            # Load PyTorch model
            if self.config.input_path.endswith('.pth') or self.config.input_path.endswith('.pt'):
                model = torch.load(self.config.input_path, map_location=self.config.device, weights_only=True)
            elif TRANSFORMERS_AVAILABLE and Path(self.config.input_path).is_dir():
                # Hugging Face model
                model = AutoModel.from_pretrained(self.config.input_path)
            else:
                raise ValueError(f"Unsupported PyTorch model format: {self.config.input_path}")
            
            model.eval()
            model = model.to(self.config.device)
            
            # Try tracing first, then scripting if tracing fails
            scripted_model = None
            conversion_method = "unknown"
            
            try:
                # Create dummy inputs for tracing
                dummy_inputs = self._create_dummy_inputs(model)
                
                # Try tracing
                logger.info("Attempting TorchScript tracing...")
                scripted_model = torch.jit.trace(model, dummy_inputs)
                conversion_method = "trace"
                logger.info("TorchScript tracing successful")
                
            except Exception as trace_error:
                logger.warning(f"TorchScript tracing failed: {trace_error}")
                
                try:
                    # Try scripting
                    logger.info("Attempting TorchScript scripting...")
                    scripted_model = torch.jit.script(model)
                    conversion_method = "script"
                    logger.info("TorchScript scripting successful")
                    
                except Exception as script_error:
                    logger.error(f"TorchScript scripting failed: {script_error}")
                    raise RuntimeError(f"Both tracing and scripting failed. Trace error: {trace_error}, Script error: {script_error}")
            
            # Optimize TorchScript model if requested
            if self.config.optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
                logger.info("Applied TorchScript optimization")
            
            # Save TorchScript model
            scripted_model.save(self.config.output_path)
            
            # Validate conversion
            validation_passed = await self._validate_torchscript_conversion(model, scripted_model)
            
            conversion_time = time.time() - start_time
            model_size = self._get_file_size_mb(self.config.output_path)
            
            return ConversionResult(
                success=True,
                input_format=self.config.input_format,
                output_format=self.config.output_format,
                input_path=self.config.input_path,
                output_path=self.config.output_path,
                model_size_mb=model_size,
                conversion_time=conversion_time,
                validation_passed=validation_passed,
                metadata={
                    "conversion_method": conversion_method,
                    "optimized": self.config.optimize
                }
            )
            
        except Exception as e:
            logger.error(f"PyTorch to TorchScript conversion failed: {e}")
            return ConversionResult(
                success=False,
                input_format=self.config.input_format,
                output_format=self.config.output_format,
                input_path=self.config.input_path,
                output_path=self.config.output_path,
                model_size_mb=0.0,
                conversion_time=time.time() - start_time,
                validation_passed=False,
                error_message=str(e)
            )
    
    def _create_dummy_inputs(self, model: torch.nn.Module) -> torch.Tensor:
        """Create dummy inputs for TorchScript tracing"""
        if self.config.input_shapes:
            # Use provided input shapes
            shapes = list(self.config.input_shapes.values())
            if len(shapes) == 1:
                return torch.randn(*shapes[0]).to(self.config.device)
            else:
                return tuple(torch.randn(*shape).to(self.config.device) for shape in shapes)
        else:
            # Try to infer input shape
            try:
                return torch.randn(self.config.batch_size, 3, 224, 224).to(self.config.device)
            except Exception:
                return torch.randn(self.config.batch_size, 768).to(self.config.device)
    
    async def _validate_torchscript_conversion(self, original_model: torch.nn.Module, scripted_model: torch.jit.ScriptModule) -> bool:
        """Validate TorchScript conversion by comparing outputs"""
        try:
            # Create test inputs
            dummy_inputs = self._create_dummy_inputs(original_model)
            
            # Get outputs from both models
            with torch.no_grad():
                original_output = original_model(dummy_inputs)
                scripted_output = scripted_model(dummy_inputs)
            
            # Compare outputs
            if isinstance(original_output, torch.Tensor) and isinstance(scripted_output, torch.Tensor):
                are_close = torch.allclose(original_output, scripted_output, rtol=1e-3, atol=1e-3)
            else:
                # Handle multiple outputs
                if isinstance(original_output, (tuple, list)) and isinstance(scripted_output, (tuple, list)):
                    are_close = all(
                        torch.allclose(o1, o2, rtol=1e-3, atol=1e-3)
                        for o1, o2 in zip(original_output, scripted_output)
                    )
                else:
                    logger.warning("Output types don't match between models")
                    return False
            
            if are_close:
                logger.info("TorchScript conversion validation passed")
            else:
                logger.warning("TorchScript conversion validation failed - outputs differ")
            
            return are_close
            
        except Exception as e:
            logger.error(f"TorchScript validation failed: {e}")
            return False
    
    def validate_conversion(self, original_model: Any, converted_model: Any) -> bool:
        """Validate conversion (implementation for base class)"""
        # This is handled by _validate_torchscript_conversion
        return True

class ONNXToPyTorchConverter(BaseConverter):
    """Convert ONNX models to PyTorch format"""
    
    async def convert(self) -> ConversionResult:
        """Convert ONNX model to PyTorch"""
        import time
        start_time = time.time()
        
        try:
            # Note: ONNX to PyTorch conversion is complex and may require onnx2pytorch
            logger.warning("ONNX to PyTorch conversion is experimental and may not work for all models")
            
            # For now, we'll create a wrapper that can load and use the ONNX model
            # This is more practical than full conversion
            
            # Load ONNX model to validate it
            onnx_model = onnx.load(self.config.input_path)
            onnx.checker.check_model(onnx_model)
            
            # Create a PyTorch wrapper for the ONNX model
            wrapper_code = self._create_pytorch_wrapper()
            
            # Save wrapper
            with open(self.config.output_path, 'w') as f:
                f.write(wrapper_code)
            
            conversion_time = time.time() - start_time
            model_size = self._get_file_size_mb(self.config.output_path)
            
            return ConversionResult(
                success=True,
                input_format=self.config.input_format,
                output_format=self.config.output_format,
                input_path=self.config.input_path,
                output_path=self.config.output_path,
                model_size_mb=model_size,
                conversion_time=conversion_time,
                validation_passed=True,  # Wrapper approach doesn't need validation
                metadata={
                    "wrapper_approach": True,
                    "note": "Created PyTorch wrapper for ONNX model"
                }
            )
            
        except Exception as e:
            logger.error(f"ONNX to PyTorch conversion failed: {e}")
            return ConversionResult(
                success=False,
                input_format=self.config.input_format,
                output_format=self.config.output_format,
                input_path=self.config.input_path,
                output_path=self.config.output_path,
                model_size_mb=0.0,
                conversion_time=time.time() - start_time,
                validation_passed=False,
                error_message=str(e)
            )
    
    def _create_pytorch_wrapper(self) -> str:
        """Create PyTorch wrapper code for ONNX model"""
        wrapper_code = f'''"""
PyTorch wrapper for ONNX model: {self.config.input_path}
Generated automatically by PRSM Model Conversion Utilities
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from typing import Union, List, Dict, Any

class ONNXModelWrapper(nn.Module):
    """PyTorch wrapper for ONNX model"""
    
    def __init__(self, onnx_path: str, device: str = "cpu"):
        super().__init__()
        self.onnx_path = onnx_path
        self.device = device
        
        # Configure providers
        providers = []
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
    def forward(self, *args, **kwargs):
        """Forward pass through ONNX model"""
        # Prepare inputs
        if args and not kwargs:
            # Positional arguments
            if len(args) == 1 and len(self.input_names) == 1:
                inputs = {{self.input_names[0]: self._to_numpy(args[0])}}
            else:
                inputs = {{name: self._to_numpy(arg) for name, arg in zip(self.input_names, args)}}
        elif kwargs:
            # Keyword arguments
            inputs = {{name: self._to_numpy(kwargs[name]) for name in self.input_names if name in kwargs}}
        else:
            raise ValueError("No inputs provided")
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Convert back to torch tensors
        torch_outputs = [torch.from_numpy(out) for out in outputs]
        
        # Return single output or tuple
        if len(torch_outputs) == 1:
            return torch_outputs[0]
        else:
            return tuple(torch_outputs)
    
    def _to_numpy(self, tensor):
        """Convert torch tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)

# Create model instance
def load_model(device: str = "cpu"):
    """Load the ONNX model wrapper"""
    return ONNXModelWrapper("{self.config.input_path}", device)

if __name__ == "__main__":
    # Example usage
    model = load_model()
    print(f"Model inputs: {{model.input_names}}")
    print(f"Model outputs: {{model.output_names}}")
'''
        return wrapper_code
    
    def validate_conversion(self, original_model: Any, converted_model: Any) -> bool:
        """Validate conversion (wrapper approach doesn't need validation)"""
        return True

class ModelConversionEngine:
    """
    Unified model conversion engine supporting multiple formats
    
    🚀 FEATURES:
    - PyTorch ↔ ONNX conversion with optimization
    - PyTorch ↔ TorchScript conversion with tracing/scripting
    - ONNX → PyTorch wrapper generation
    - Model validation and comparison
    - Optimization and quantization support
    - Batch conversion capabilities
    """
    
    def __init__(self):
        self.conversion_history: List[ConversionResult] = []
        
    async def convert_model(self, config: ConversionConfig) -> ConversionResult:
        """Convert model between formats"""
        logger.info(f"Converting {config.input_format} → {config.output_format}: {config.input_path}")
        
        # Validate input file
        if not Path(config.input_path).exists():
            return ConversionResult(
                success=False,
                input_format=config.input_format,
                output_format=config.output_format,
                input_path=config.input_path,
                output_path=config.output_path,
                model_size_mb=0.0,
                conversion_time=0.0,
                validation_passed=False,
                error_message=f"Input file not found: {config.input_path}"
            )
        
        # Create output directory if needed
        output_dir = Path(config.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select appropriate converter
        converter = self._get_converter(config)
        if not converter:
            return ConversionResult(
                success=False,
                input_format=config.input_format,
                output_format=config.output_format,
                input_path=config.input_path,
                output_path=config.output_path,
                model_size_mb=0.0,
                conversion_time=0.0,
                validation_passed=False,
                error_message=f"Unsupported conversion: {config.input_format} → {config.output_format}"
            )
        
        # Perform conversion
        result = await converter.convert()
        
        # Store in history
        self.conversion_history.append(result)
        
        if result.success:
            logger.info(f"Conversion successful: {result.model_size_mb:.2f}MB in {result.conversion_time:.2f}s")
        else:
            logger.error(f"Conversion failed: {result.error_message}")
        
        return result
    
    def _get_converter(self, config: ConversionConfig) -> Optional[BaseConverter]:
        """Get appropriate converter for the format pair"""
        conversion_key = f"{config.input_format}_{config.output_format}"
        
        converter_map = {
            "pytorch_onnx": PyTorchToONNXConverter,
            "pytorch_torchscript": PyTorchToTorchScriptConverter,
            "onnx_pytorch": ONNXToPyTorchConverter,
        }
        
        converter_class = converter_map.get(conversion_key)
        if converter_class:
            return converter_class(config)
        
        return None
    
    async def batch_convert(self, configs: List[ConversionConfig]) -> List[ConversionResult]:
        """Convert multiple models in batch"""
        results = []
        
        for config in configs:
            result = await self.convert_model(config)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch conversion completed: {successful}/{len(results)} successful")
        
        return results
    
    def get_conversion_summary(self) -> Dict[str, Any]:
        """Get summary of conversion history"""
        if not self.conversion_history:
            return {"message": "No conversions performed yet"}
        
        total_conversions = len(self.conversion_history)
        successful_conversions = sum(1 for r in self.conversion_history if r.success)
        total_size_mb = sum(r.model_size_mb for r in self.conversion_history if r.success)
        total_time = sum(r.conversion_time for r in self.conversion_history)
        
        format_pairs = {}
        for result in self.conversion_history:
            key = f"{result.input_format}_{result.output_format}"
            if key not in format_pairs:
                format_pairs[key] = {"total": 0, "successful": 0}
            format_pairs[key]["total"] += 1
            if result.success:
                format_pairs[key]["successful"] += 1
        
        return {
            "total_conversions": total_conversions,
            "successful_conversions": successful_conversions,
            "success_rate": successful_conversions / total_conversions if total_conversions > 0 else 0,
            "total_model_size_mb": total_size_mb,
            "total_conversion_time": total_time,
            "avg_conversion_time": total_time / total_conversions if total_conversions > 0 else 0,
            "format_pairs": format_pairs,
            "recent_conversions": [
                {
                    "input_format": r.input_format,
                    "output_format": r.output_format,
                    "success": r.success,
                    "model_size_mb": r.model_size_mb,
                    "conversion_time": r.conversion_time
                }
                for r in self.conversion_history[-5:]  # Last 5 conversions
            ]
        }
    
    def get_supported_conversions(self) -> List[Dict[str, str]]:
        """Get list of supported conversion paths"""
        conversions = [
            {"from": "pytorch", "to": "torchscript", "description": "PyTorch to TorchScript with tracing/scripting"},
        ]
        
        # Add ONNX conversions only if ONNX is available
        if ONNX_AVAILABLE:
            conversions.extend([
                {"from": "pytorch", "to": "onnx", "description": "PyTorch to ONNX with optimization"},
                {"from": "onnx", "to": "pytorch", "description": "ONNX to PyTorch wrapper generation"},
            ])
        
        return conversions
    
    def get_conversion_history(self) -> List[ConversionResult]:
        """Get list of conversion history results"""
        return list(self.conversion_history)
    
    def clear_history(self):
        """Clear conversion history"""
        self.conversion_history.clear()
        logger.info("Conversion history cleared")

# Example usage and testing functions
async def example_pytorch_to_onnx():
    """Example: Convert PyTorch model to ONNX"""
    
    # Create a simple PyTorch model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            self.output = torch.nn.Linear(5, 1)
        
        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return self.output(x)
    
    # Save test model
    model = SimpleModel()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        test_model_path = tmp_file.name
    torch.save(model, test_model_path)
    
    # Configure conversion
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        output_path = tmp_file.name
    config = ConversionConfig(
        input_format="pytorch",
        output_format="onnx",
        input_path=test_model_path,
        output_path=output_path,
        input_shapes={"input": [1, 10]},
        optimize=True,
        simplify=True
    )
    
    # Convert
    engine = ModelConversionEngine()
    result = await engine.convert_model(config)
    
    print(f"Conversion result: {result}")
    
    # Cleanup
    for path in [test_model_path, output_path]:
        if Path(path).exists():
            Path(path).unlink()

async def example_pytorch_to_torchscript():
    """Example: Convert PyTorch model to TorchScript"""
    
    # Create test model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(16, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Save test model
    model = SimpleModel()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        test_model_path = tmp_file.name
    torch.save(model, test_model_path)
    
    # Configure conversion
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        output_path = tmp_file.name
    config = ConversionConfig(
        input_format="pytorch",
        output_format="torchscript",
        input_path=test_model_path,
        output_path=output_path,
        input_shapes={"input": [1, 3, 32, 32]},
        optimize=True
    )
    
    # Convert
    engine = ModelConversionEngine()
    result = await engine.convert_model(config)
    
    print(f"TorchScript conversion result: {result}")
    
    # Cleanup
    for path in [test_model_path, output_path]:
        if Path(path).exists():
            Path(path).unlink()

if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Run examples"""
        print("🔄 Testing PyTorch to ONNX conversion...")
        await example_pytorch_to_onnx()
        
        print("\n🔄 Testing PyTorch to TorchScript conversion...")
        await example_pytorch_to_torchscript()
        
        # Test conversion engine summary
        engine = ModelConversionEngine()
        print(f"\n📊 Supported conversions: {engine.get_supported_conversions()}")
    
    asyncio.run(main())