"""
Base Backend Interface for PRSM Distillation

🎯 PURPOSE IN PRSM:
This abstract base class defines the interface that all distillation backends
must implement. It ensures consistency across different ML frameworks while
allowing for framework-specific optimizations.

🔧 FRAMEWORK INTEGRATION:
Each backend (PyTorch, TensorFlow, etc.) implements this interface to provide:
- Model architecture generation
- Training data preparation  
- Knowledge distillation training
- Model evaluation and validation
- Export for PRSM deployment
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass

from ..models import DistillationRequest, TrainingConfig, StudentArchitecture, TeacherAnalysis


@dataclass
class TrainingMetrics:
    """Training metrics collected during distillation"""
    step: int
    epoch: int
    loss: float
    accuracy: float
    distillation_loss: float
    student_loss: float
    learning_rate: float
    temperature: float
    additional_metrics: Dict[str, float]


@dataclass
class ModelArtifacts:
    """Generated model artifacts for deployment"""
    model_path: str                    # Path to saved model
    model_config: Dict[str, Any]       # Model configuration
    tokenizer_path: Optional[str]      # Tokenizer if applicable
    onnx_path: Optional[str]          # ONNX export for cross-platform
    metadata: Dict[str, Any]          # Model metadata and performance info
    deployment_config: Dict[str, Any] # Deployment configuration


class DistillationBackend(ABC):
    """
    Abstract base class for distillation backends
    
    🔧 IMPLEMENTATION GUIDE FOR CONTRIBUTORS:
    
    When implementing a new backend (e.g., JAX, MLX), you must implement:
    1. generate_student_architecture() - Create model based on requirements
    2. prepare_training_data() - Format data for framework
    3. initialize_models() - Set up teacher and student models
    4. train_step() - Single training iteration with distillation loss
    5. evaluate_model() - Assessment against validation data
    6. export_model() - Package for PRSM deployment
    
    🧠 PRSM INTEGRATION:
    - Each backend integrates with PRSM's orchestrator for job management
    - Progress callbacks enable real-time user updates
    - Model artifacts are automatically deployed to P2P federation
    - Safety validation ensures circuit breaker compliance
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the distillation backend
        
        Args:
            device: Target device ("cpu", "cuda", "auto")
        """
        self.device = device
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (download dependencies, setup environment)"""
        pass
    
    @abstractmethod
    async def generate_student_architecture(
        self, 
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        architecture_spec: StudentArchitecture
    ) -> Dict[str, Any]:
        """
        Generate student model architecture
        
        🎯 PURPOSE: Create optimal model architecture based on:
        - User requirements (size, speed, accuracy targets)
        - Teacher model characteristics 
        - Target deployment environment
        
        Args:
            request: User's distillation requirements
            teacher_analysis: Analysis of teacher model capabilities
            architecture_spec: Generated architecture specifications
            
        Returns:
            Dict containing model architecture definition
        """
        pass
    
    @abstractmethod
    async def prepare_training_data(
        self,
        request: DistillationRequest,
        teacher_analysis: TeacherAnalysis,
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Prepare training data for distillation
        
        🎯 PURPOSE: Generate and format training data including:
        - Teacher model outputs for knowledge transfer
        - Data augmentation for robustness
        - Curriculum learning sequences
        - Validation and test sets
        
        Args:
            request: Distillation request with domain and requirements
            teacher_analysis: Teacher model capabilities and patterns
            config: Training configuration parameters
            
        Returns:
            Dict containing prepared training datasets
        """
        pass
    
    @abstractmethod
    async def initialize_models(
        self,
        teacher_config: Dict[str, Any],
        student_architecture: Dict[str, Any],
        config: TrainingConfig
    ) -> Tuple[Any, Any]:
        """
        Initialize teacher and student models
        
        🎯 PURPOSE: Set up models for training:
        - Load and configure teacher model
        - Initialize student model with architecture
        - Setup for framework-specific training
        
        Args:
            teacher_config: Teacher model configuration
            student_architecture: Student model architecture
            config: Training configuration
            
        Returns:
            Tuple of (teacher_model, student_model)
        """
        pass
    
    @abstractmethod
    async def train_step(
        self,
        teacher_model: Any,
        student_model: Any,
        batch_data: Dict[str, Any],
        optimizer: Any,
        config: TrainingConfig,
        step: int
    ) -> TrainingMetrics:
        """
        Execute single training step with knowledge distillation
        
        🎯 PURPOSE: Core distillation training including:
        - Forward pass through teacher and student
        - Knowledge distillation loss calculation
        - Student task loss calculation
        - Combined loss optimization
        - Metric collection
        
        Args:
            teacher_model: Teacher model for knowledge source
            student_model: Student model being trained
            batch_data: Training batch
            optimizer: Optimization algorithm
            config: Training configuration
            step: Current training step
            
        Returns:
            TrainingMetrics with loss and performance data
        """
        pass
    
    @abstractmethod
    async def evaluate_model(
        self,
        student_model: Any,
        eval_data: Dict[str, Any],
        teacher_model: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate student model performance
        
        🎯 PURPOSE: Comprehensive evaluation including:
        - Task-specific accuracy metrics
        - Comparison with teacher performance
        - Inference speed benchmarking
        - Memory usage assessment
        
        Args:
            student_model: Trained student model
            eval_data: Evaluation dataset
            teacher_model: Optional teacher for comparison
            
        Returns:
            Dict of evaluation metrics
        """
        pass
    
    @abstractmethod
    async def export_model(
        self,
        student_model: Any,
        model_config: Dict[str, Any],
        export_path: str,
        formats: List[str] = ["native", "onnx"]
    ) -> ModelArtifacts:
        """
        Export trained model for deployment
        
        🎯 PURPOSE: Package model for PRSM ecosystem:
        - Save in native framework format
        - Export to ONNX for cross-platform deployment
        - Generate deployment metadata
        - Create PRSM-compatible configuration
        
        Args:
            student_model: Trained student model
            model_config: Model configuration
            export_path: Directory for model artifacts
            formats: Export formats to generate
            
        Returns:
            ModelArtifacts with paths and metadata
        """
        pass
    
    @abstractmethod
    async def get_supported_architectures(self) -> List[str]:
        """Get list of supported model architectures for this backend"""
        pass
    
    @abstractmethod
    async def get_memory_requirements(
        self, 
        architecture: StudentArchitecture
    ) -> Dict[str, int]:
        """
        Estimate memory requirements for training and inference
        
        Returns:
            Dict with 'training_mb' and 'inference_mb' estimates
        """
        pass
    
    # === Helper Methods ===
    
    async def supports_strategy(self, strategy: str) -> bool:
        """Check if backend supports specific training strategy"""
        # Default implementation - backends can override
        supported = ["basic", "progressive", "ensemble"]
        return strategy.lower() in supported
    
    async def get_framework_info(self) -> Dict[str, str]:
        """Get information about the underlying ML framework"""
        return {
            "name": self.__class__.__name__,
            "device": self.device,
            "initialized": str(self.is_initialized)
        }
    
    def _validate_config(self, config: TrainingConfig) -> None:
        """Validate training configuration for this backend"""
        if config.num_epochs < 1:
            raise ValueError("Number of epochs must be positive")
        if config.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if not 0 < config.learning_rate < 1:
            raise ValueError("Learning rate must be between 0 and 1")


class BackendRegistry:
    """
    Registry for available distillation backends
    
    🎯 PURPOSE IN PRSM:
    Manages available backends and automatically selects the best one
    based on user requirements and system capabilities.
    """
    
    _backends: Dict[str, DistillationBackend] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: type):
        """Register a new backend"""
        cls._backends[name] = backend_class
    
    @classmethod
    def get_backend(cls, name: str, **kwargs) -> DistillationBackend:
        """Get a backend instance by name"""
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' not found. Available: {list(cls._backends.keys())}")
        return cls._backends[name](**kwargs)
    
    @classmethod
    def get_available_backends(cls) -> List[str]:
        """Get list of available backend names"""
        return list(cls._backends.keys())
    
    @classmethod
    async def auto_select_backend(
        cls, 
        request: DistillationRequest,
        system_info: Dict[str, Any]
    ) -> str:
        """
        Automatically select best backend based on requirements
        
        🧠 SELECTION LOGIC:
        - PyTorch: Best for most research and custom architectures
        - Transformers: Best for NLP tasks with pre-trained models
        - TensorFlow: Best for production deployment and mobile
        - ONNX: Best for cross-platform inference
        
        Args:
            request: User's distillation request
            system_info: Available hardware and frameworks
            
        Returns:
            Name of recommended backend
        """
        # Default priority order
        if "transformers" in cls._backends and request.domain in ["code_generation", "creative_writing"]:
            return "transformers"
        elif "pytorch" in cls._backends:
            return "pytorch"
        elif "tensorflow" in cls._backends:
            return "tensorflow"
        else:
            # Return first available backend
            return list(cls._backends.keys())[0]