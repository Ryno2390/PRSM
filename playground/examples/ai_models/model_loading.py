#!/usr/bin/env python3
"""
AI Model Loading Example
This example demonstrates loading and managing different types of AI models with PRSM.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from prsm.distillation.model_inference_engine import ModelInferenceEngine
    from prsm.distillation.model_conversion_utilities import ModelConversionUtilities
    PRSM_AVAILABLE = True
except ImportError:
    PRSM_AVAILABLE = False

# Try ML framework imports
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class MockModelLoader:
    """Mock model loader when PRSM components aren't available"""
    
    def __init__(self):
        self.models = {}
        print("🔧 Using mock model loader (PRSM components not available)")
    
    async def load_model(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Load a mock model"""
        print(f"📦 Loading mock {model_type} model: {model_name}")
        await asyncio.sleep(0.5)  # Simulate loading time
        
        model_info = {
            "id": f"mock_{model_name}",
            "name": model_name,
            "type": model_type,
            "framework": "mock",
            "size_mb": 125.5,
            "parameters": 1000000,
            "accuracy": 0.95,
            "loaded_at": time.time()
        }
        
        self.models[model_info["id"]] = model_info
        print(f"✅ Mock model loaded: {model_name}")
        return model_info
    
    async def run_inference(self, model_id: str, input_data: List[float]) -> Dict[str, Any]:
        """Run mock inference"""
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}
        
        print(f"🧠 Running mock inference on {model_id}")
        await asyncio.sleep(0.1)  # Simulate inference time
        
        # Generate mock prediction
        prediction = [0.8, 0.15, 0.05] if len(input_data) > 5 else [sum(input_data) * 0.1]
        
        return {
            "model_id": model_id,
            "prediction": prediction,
            "confidence": 0.87,
            "processing_time": 0.1,
            "input_size": len(input_data)
        }

class SimplePyTorchModel(nn.Module):
    """Simple PyTorch model for demonstration"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

async def demonstrate_model_loading():
    """Demonstrate loading different types of AI models"""
    print("🚀 AI Model Loading Demo")
    print("=" * 50)
    
    if PRSM_AVAILABLE:
        print("✅ PRSM model infrastructure available")
        model_engine = ModelInferenceEngine()
    else:
        print("⚠️  Using mock model infrastructure")
        model_engine = MockModelLoader()
    
    # Track loaded models
    loaded_models = []
    
    print("\n📦 Loading Different Model Types...")
    
    # 1. Load PyTorch Model
    if PYTORCH_AVAILABLE:
        print("\n🔥 Loading PyTorch Neural Network...")
        try:
            # Create a simple PyTorch model
            pytorch_model = SimplePyTorchModel(input_size=10, hidden_size=32, output_size=1)
            
            model_info = {
                "id": "pytorch_demo_model",
                "name": "Simple Neural Network",
                "type": "neural_network",
                "framework": "pytorch",
                "model": pytorch_model,
                "input_size": 10,
                "output_size": 1,
                "parameters": sum(p.numel() for p in pytorch_model.parameters()),
                "loaded_at": time.time()
            }
            
            loaded_models.append(model_info)
            print(f"✅ PyTorch model loaded: {model_info['parameters']} parameters")
            
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {str(e)}")
    else:
        print("⚠️  PyTorch not available, loading mock model")
        mock_model = await model_engine.load_model("neural_network", "pytorch_mock")
        loaded_models.append(mock_model)
    
    # 2. Load Custom Mathematical Model
    print("\n🧮 Loading Custom Mathematical Model...")
    try:
        def linear_regression_model(inputs: List[float]) -> List[float]:
            """Simple linear regression function"""
            if not inputs:
                return [0.0]
            # y = 2x + 1 (simplified)
            result = sum(x * 2 + 1 for x in inputs) / len(inputs)
            return [result]
        
        custom_model_info = {
            "id": "custom_linear_model",
            "name": "Linear Regression",
            "type": "regression",
            "framework": "custom",
            "model": linear_regression_model,
            "input_size": "variable",
            "output_size": 1,
            "parameters": 2,  # slope and intercept
            "loaded_at": time.time()
        }
        
        loaded_models.append(custom_model_info)
        print(f"✅ Custom model loaded: Linear Regression")
        
    except Exception as e:
        print(f"❌ Error loading custom model: {str(e)}")
    
    # 3. Load Classification Model (mock)
    print("\n🎯 Loading Classification Model...")
    if hasattr(model_engine, 'load_model'):
        classification_model = await model_engine.load_model("classification", "text_classifier")
        loaded_models.append(classification_model)
    else:
        # Manual mock for non-PRSM case
        classification_info = {
            "id": "mock_classifier",
            "name": "Text Classifier",
            "type": "classification",
            "framework": "mock",
            "classes": ["positive", "negative", "neutral"],
            "accuracy": 0.92,
            "loaded_at": time.time()
        }
        loaded_models.append(classification_info)
        print(f"✅ Classification model loaded: {classification_info['name']}")
    
    # Display loaded models summary
    print(f"\n📊 Model Loading Summary:")
    print(f"   Total models loaded: {len(loaded_models)}")
    
    for i, model in enumerate(loaded_models, 1):
        print(f"   {i}. {model.get('name', 'Unknown')}")
        print(f"      Type: {model.get('type', 'unknown')}")
        print(f"      Framework: {model.get('framework', 'unknown')}")
        if 'parameters' in model:
            print(f"      Parameters: {model['parameters']:,}")
    
    return loaded_models

async def demonstrate_model_inference(loaded_models: List[Dict[str, Any]]):
    """Demonstrate running inference on loaded models"""
    print(f"\n🧠 Model Inference Demo")
    print("=" * 40)
    
    # Test data
    test_inputs = {
        "neural_network": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "regression": [1.5, 2.3, 3.1, 4.8, 5.2],
        "classification": [0.8, -0.2, 1.1, 0.5, -0.7]
    }
    
    inference_results = []
    
    for model in loaded_models:
        model_type = model.get('type', 'unknown')
        print(f"\n🔄 Running inference on {model.get('name', 'Unknown Model')}")
        
        # Get appropriate test input
        if model_type in test_inputs:
            input_data = test_inputs[model_type]
        else:
            input_data = test_inputs["neural_network"]  # Default
        
        try:
            start_time = time.time()
            
            if model.get('framework') == 'pytorch' and 'model' in model:
                # PyTorch inference
                pytorch_model = model['model']
                pytorch_model.eval()
                
                with torch.no_grad():
                    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                    output = pytorch_model(input_tensor)
                    prediction = output.squeeze().item()
                
                result = {
                    "model_id": model['id'],
                    "prediction": [prediction],
                    "confidence": 0.85,
                    "processing_time": time.time() - start_time,
                    "framework": "pytorch"
                }
                
            elif model.get('framework') == 'custom' and 'model' in model:
                # Custom model inference
                custom_function = model['model']
                prediction = custom_function(input_data)
                
                result = {
                    "model_id": model['id'],
                    "prediction": prediction,
                    "confidence": 0.90,
                    "processing_time": time.time() - start_time,
                    "framework": "custom"
                }
                
            else:
                # Mock inference
                await asyncio.sleep(0.1)  # Simulate processing
                
                if model_type == "classification":
                    prediction = ["positive"]
                    confidence = 0.78
                else:
                    prediction = [sum(input_data) * 0.1]
                    confidence = 0.82
                
                result = {
                    "model_id": model['id'],
                    "prediction": prediction,
                    "confidence": confidence,
                    "processing_time": time.time() - start_time,
                    "framework": model.get('framework', 'mock')
                }
            
            inference_results.append(result)
            
            print(f"✅ Inference completed:")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Time: {result['processing_time']:.3f}s")
            
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
            error_result = {
                "model_id": model['id'],
                "error": str(e),
                "framework": model.get('framework', 'unknown')
            }
            inference_results.append(error_result)
    
    return inference_results

async def demonstrate_model_conversion():
    """Demonstrate model format conversion"""
    print(f"\n🔄 Model Conversion Demo")
    print("=" * 40)
    
    if not PRSM_AVAILABLE:
        print("⚠️  PRSM conversion utilities not available")
        print("💡 Mock conversion demonstration:")
        
        conversions = [
            {"from": "pytorch", "to": "onnx", "status": "simulated"},
            {"from": "tensorflow", "to": "pytorch", "status": "simulated"},
            {"from": "onnx", "to": "torchscript", "status": "simulated"}
        ]
        
        for conv in conversions:
            print(f"   🔄 {conv['from']} → {conv['to']}: {conv['status']}")
        
        return conversions
    
    try:
        converter = ModelConversionUtilities()
        
        # Get available conversions
        if hasattr(converter, 'get_supported_conversions'):
            conversions = converter.get_supported_conversions()
            print(f"📋 Available conversions: {len(conversions)}")
            
            for conv in conversions:
                print(f"   🔄 {conv['from']} → {conv['to']}")
        else:
            print("💡 Conversion utilities available but no specific conversions listed")
        
        return []
        
    except Exception as e:
        print(f"❌ Error accessing conversion utilities: {str(e)}")
        return []

async def main():
    """Main function"""
    print("🚀 Starting AI Model Management Demo")
    print("🔧 Checking available components...")
    
    if PYTORCH_AVAILABLE:
        print("✅ PyTorch available")
    else:
        print("⚠️  PyTorch not available - using mocks")
    
    if NUMPY_AVAILABLE:
        print("✅ NumPy available")
    else:
        print("⚠️  NumPy not available")
    
    try:
        # Load models
        loaded_models = await demonstrate_model_loading()
        
        # Run inference
        inference_results = await demonstrate_model_inference(loaded_models)
        
        # Try model conversion
        conversion_results = await demonstrate_model_conversion()
        
        # Save results
        results = {
            "loaded_models": loaded_models,
            "inference_results": inference_results,
            "conversion_results": conversion_results,
            "demo_completed_at": time.time()
        }
        
        output_file = Path(__file__).parent / "model_demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Summary
        print(f"\n📈 Demo Summary:")
        print(f"   Models loaded: {len(loaded_models)}")
        print(f"   Inferences run: {len(inference_results)}")
        print(f"   Successful inferences: {len([r for r in inference_results if 'error' not in r])}")
        
        print(f"\n🎉 AI Model Management demo completed!")
        print(f"💡 Next steps:")
        print(f"   • Try: python playground_launcher.py --example ai_models/distributed_inference")
        print(f"   • Try: python playground_launcher.py --example p2p_network/basic_network")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        print("\n✅ Model Loading example completed successfully!")
    else:
        print("\n❌ Example failed. Check the logs for details.")
        sys.exit(1)