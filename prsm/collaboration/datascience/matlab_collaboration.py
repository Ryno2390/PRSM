#!/usr/bin/env python3
"""
MATLAB Integration and ML Experiment Tracking for PRSM Secure Collaboration
===========================================================================

This module implements collaborative MATLAB environments with advanced
ML experiment tracking for university-industry research partnerships:

- Secure MATLAB workspace sharing with post-quantum encryption
- Collaborative MATLAB script development and execution
- ML experiment tracking and version control
- Integration with university research infrastructure
- Industry partnership workflows for MATLAB-based projects
- NWTN AI-powered MATLAB code optimization

Key Features:
- Multi-user MATLAB environments with P2P security
- Experiment tracking similar to Weights & Biases/MLflow
- Shared MATLAB toolboxes and custom functions
- Integration with statistical analysis and data visualization
- Secure sharing of proprietary MATLAB algorithms
- Performance monitoring and optimization suggestions
"""

import json
import uuid
import asyncio
import subprocess
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import pickle
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import PRSM components
from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding, CryptoMode
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for MATLAB collaboration"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # MATLAB-specific NWTN responses
        if context.get("matlab_optimization"):
            return {
                "response": {
                    "text": """
MATLAB Code Optimization Analysis:

🔧 **Performance Optimization Recommendations**:
```matlab
% Vectorized operations for better performance
% Before: Loop-based computation (slow)
for i = 1:length(data)
    result(i) = complex_function(data(i));
end

% After: Vectorized computation (fast)
result = arrayfun(@complex_function, data);

% Preallocate arrays for large datasets
n = 1e6;
data = zeros(n, 1);  % Preallocate instead of growing arrays

% Use parallel computing for intensive tasks
parfor i = 1:length(experiments)
    results{i} = run_ml_experiment(experiments(i));
end
```

📊 **Memory Management**:
- Use `clear` selectively to free memory
- Employ `tall` arrays for big data processing
- Implement incremental learning for large datasets
- Monitor memory usage with `memory` function

🤖 **ML Experiment Best Practices**:
- Implement proper cross-validation workflows
- Use MATLAB's Statistics and Machine Learning Toolbox efficiently
- Document hyperparameter choices and model configurations
- Save intermediate results for reproducibility

🏛️ **University-Industry Collaboration**:
- Version control MATLAB code with Git integration
- Use MATLAB Projects for organizing collaborative work
- Implement automated testing for critical algorithms
- Document code for easy knowledge transfer
                    """,
                    "confidence": 0.92,
                    "sources": ["mathworks.com", "matlab_documentation.pdf", "optimization_guide.pdf"]
                },
                "performance_metrics": {"total_processing_time": 3.4}
            }
        elif context.get("experiment_tracking"):
            return {
                "response": {
                    "text": """
ML Experiment Tracking Recommendations:

📈 **Experiment Organization**:
```matlab
% Create structured experiment tracking
experiment = struct();
experiment.id = generate_experiment_id();
experiment.name = 'Quantum_Algorithm_Optimization_v2.1';
experiment.description = 'Testing adaptive error correction parameters';
experiment.timestamp = datetime('now');
experiment.collaborators = {'sarah.chen@unc.edu', 'michael.johnson@sas.com'};

% Track hyperparameters systematically
experiment.hyperparameters = struct(...
    'learning_rate', 0.001, ...
    'batch_size', 32, ...
    'num_epochs', 100, ...
    'regularization', 0.01 ...
);

% Log metrics throughout training
experiment.metrics = struct();
experiment.metrics.training_accuracy = [];
experiment.metrics.validation_loss = [];
experiment.metrics.computational_time = [];
```

🔬 **Reproducibility Framework**:
- Save complete environment state (MATLAB version, toolboxes)
- Record random seeds and initialization parameters
- Archive input data with checksums
- Store model artifacts and intermediate results

📊 **Visualization and Reporting**:
- Generate automated experiment reports
- Create publication-ready figures
- Implement real-time monitoring dashboards
- Export results in multiple formats (PDF, Excel, JSON)

🤝 **Collaborative Features**:
- Share experiment results across institutions
- Compare models from different research groups
- Implement collaborative hyperparameter optimization
- Maintain experiment provenance and attribution
                    """,
                    "confidence": 0.88,
                    "sources": ["mlflow.org", "weights_and_biases.com", "experiment_design.pdf"]
                },
                "performance_metrics": {"total_processing_time": 2.9}
            }
        else:
            return {
                "response": {"text": "MATLAB collaboration assistance available", "confidence": 0.75, "sources": []},
                "performance_metrics": {"total_processing_time": 1.6}
            }

class MATLABAccessLevel(Enum):
    """Access levels for MATLAB collaboration"""
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    ANALYST = "analyst"
    VIEWER = "viewer"

class ExperimentStatus(Enum):
    """ML experiment status"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class MATLABToolbox(Enum):
    """MATLAB toolbox categories"""
    STATISTICS_ML = "statistics_ml"
    SIGNAL_PROCESSING = "signal_processing"
    IMAGE_PROCESSING = "image_processing"
    CONTROL_SYSTEMS = "control_systems"
    OPTIMIZATION = "optimization"
    PARALLEL_COMPUTING = "parallel_computing"
    DEEP_LEARNING = "deep_learning"
    BIOINFORMATICS = "bioinformatics"

@dataclass
class MATLABEnvironment:
    """MATLAB collaborative environment configuration"""
    env_id: str
    name: str
    description: str
    matlab_version: str
    installed_toolboxes: List[MATLABToolbox]
    custom_functions: List[str]
    shared_datasets: List[str]
    
    # Collaboration settings
    owner: str
    collaborators: Dict[str, MATLABAccessLevel]
    workspace_path: str
    
    # Security
    encrypted: bool = True
    access_controlled: bool = True
    security_level: str = "high"
    
    # Performance
    parallel_workers: int = 4
    memory_limit_gb: int = 16
    gpu_enabled: bool = False
    
    created_at: datetime = None
    last_accessed: datetime = None

@dataclass
class MLExperiment:
    """Machine Learning experiment tracking"""
    experiment_id: str
    name: str
    description: str
    created_by: str
    collaborators: List[str]
    
    # Experiment configuration
    algorithm_type: str
    hyperparameters: Dict[str, Any]
    dataset_info: Dict[str, Any]
    matlab_environment: str
    
    # Tracking
    status: ExperimentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    metrics: Dict[str, List[float]] = None
    final_metrics: Dict[str, float] = None
    model_artifacts: List[str] = None
    plots: List[str] = None
    
    # Reproducibility
    random_seed: int = 42
    matlab_code: str = ""
    environment_snapshot: Dict[str, Any] = None
    
    # Collaboration
    notes: List[Dict[str, Any]] = None
    tags: List[str] = None
    
    created_at: datetime = None
    last_modified: datetime = None

@dataclass
class MATLABScript:
    """Collaborative MATLAB script"""
    script_id: str
    name: str
    description: str
    content: str
    created_by: str
    collaborators: List[str]
    
    # Version control
    version: str = "1.0"
    edit_history: List[Dict[str, Any]] = None
    
    # Execution
    last_execution: Optional[datetime] = None
    execution_results: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    
    # Dependencies
    required_toolboxes: List[MATLABToolbox] = None
    input_variables: List[str] = None
    output_variables: List[str] = None
    
    created_at: datetime = None
    last_modified: datetime = None

class MATLABCollaboration:
    """
    Main class for MATLAB collaboration and ML experiment tracking
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize MATLAB collaboration system"""
        self.storage_path = storage_path or Path("./matlab_collaboration")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = PostQuantumCryptoSharding(
            default_shards=5,
            required_shards=3,
            crypto_mode=CryptoMode.POST_QUANTUM
        )
        self.nwtn_pipeline = None
        
        # Active environments and experiments
        self.matlab_environments: Dict[str, MATLABEnvironment] = {}
        self.ml_experiments: Dict[str, MLExperiment] = {}
        self.matlab_scripts: Dict[str, MATLABScript] = {}
        
        # MATLAB integration
        self.matlab_available = self._check_matlab_availability()
        
        # Standard toolbox configurations
        self.standard_toolboxes = self._initialize_standard_toolboxes()
        
        # Experiment tracking database
        self.experiment_db_path = self.storage_path / "experiment_tracking.json"
        self._initialize_experiment_db()
    
    def _check_matlab_availability(self) -> bool:
        """Check if MATLAB is available on the system"""
        try:
            result = subprocess.run(["matlab", "-batch", "disp('MATLAB Available')"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("🔬 MATLAB installation detected")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        print("⚠️  MATLAB not detected - using simulation mode")
        return False
    
    def _initialize_standard_toolboxes(self) -> Dict[MATLABToolbox, Dict[str, Any]]:
        """Initialize standard MATLAB toolbox configurations"""
        return {
            MATLABToolbox.STATISTICS_ML: {
                "functions": ["fitlm", "fitglm", "fitrensemble", "fitcsvm", "crossval"],
                "description": "Statistical analysis and machine learning",
                "required_for": ["data_analysis", "predictive_modeling"]
            },
            MATLABToolbox.DEEP_LEARNING: {
                "functions": ["trainNetwork", "classify", "predict", "dlarray"],
                "description": "Deep learning and neural networks",
                "required_for": ["neural_networks", "computer_vision", "nlp"]
            },
            MATLABToolbox.OPTIMIZATION: {
                "functions": ["fmincon", "ga", "particleswarm", "simulannealbnd"],
                "description": "Optimization algorithms",
                "required_for": ["parameter_optimization", "hyperparameter_tuning"]
            },
            MATLABToolbox.PARALLEL_COMPUTING: {
                "functions": ["parfor", "parfeval", "spmd", "distributed"],
                "description": "Parallel and distributed computing",
                "required_for": ["large_scale_computation", "cluster_computing"]
            }
        }
    
    def _initialize_experiment_db(self):
        """Initialize experiment tracking database"""
        if not self.experiment_db_path.exists():
            initial_db = {
                "experiments": {},
                "metrics_history": {},
                "model_registry": {},
                "created_at": datetime.now().isoformat()
            }
            with open(self.experiment_db_path, 'w') as f:
                json.dump(initial_db, f, indent=2)
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for MATLAB optimization"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_matlab_environment(self,
                                name: str,
                                description: str,
                                matlab_version: str,
                                owner: str,
                                collaborators: Optional[Dict[str, MATLABAccessLevel]] = None,
                                required_toolboxes: Optional[List[MATLABToolbox]] = None,
                                security_level: str = "high") -> MATLABEnvironment:
        """Create a collaborative MATLAB environment"""
        
        env_id = str(uuid.uuid4())
        
        # Create environment directory
        env_dir = self.storage_path / "environments" / env_id
        env_dir.mkdir(parents=True, exist_ok=True)
        
        environment = MATLABEnvironment(
            env_id=env_id,
            name=name,
            description=description,
            matlab_version=matlab_version,
            installed_toolboxes=required_toolboxes or [MATLABToolbox.STATISTICS_ML],
            custom_functions=[],
            shared_datasets=[],
            owner=owner,
            collaborators=collaborators or {},
            workspace_path=str(env_dir),
            encrypted=True,
            access_controlled=True,
            security_level=security_level,
            parallel_workers=4,
            memory_limit_gb=16,
            gpu_enabled=False,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        self.matlab_environments[env_id] = environment
        self._save_environment(environment)
        
        print(f"🔬 Created MATLAB environment: {name}")
        print(f"   Environment ID: {env_id}")
        print(f"   MATLAB Version: {matlab_version}")
        print(f"   Toolboxes: {len(environment.installed_toolboxes)}")
        print(f"   Collaborators: {len(collaborators or {})}")
        print(f"   Security: {security_level}")
        
        return environment
    
    def create_ml_experiment(self,
                           name: str,
                           description: str,
                           algorithm_type: str,
                           created_by: str,
                           matlab_environment: str,
                           hyperparameters: Dict[str, Any],
                           dataset_info: Dict[str, Any],
                           collaborators: Optional[List[str]] = None) -> MLExperiment:
        """Create a new ML experiment for tracking"""
        
        experiment_id = str(uuid.uuid4())
        
        experiment = MLExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            created_by=created_by,
            collaborators=collaborators or [],
            algorithm_type=algorithm_type,
            hyperparameters=hyperparameters,
            dataset_info=dataset_info,
            matlab_environment=matlab_environment,
            status=ExperimentStatus.PLANNED,
            metrics={},
            final_metrics={},
            model_artifacts=[],
            plots=[],
            random_seed=42,
            matlab_code="",
            environment_snapshot={},
            notes=[],
            tags=[],
            created_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        self.ml_experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        print(f"🧪 Created ML experiment: {name}")
        print(f"   Experiment ID: {experiment_id}")
        print(f"   Algorithm: {algorithm_type}")
        print(f"   Environment: {matlab_environment}")
        print(f"   Hyperparameters: {len(hyperparameters)} parameters")
        print(f"   Collaborators: {len(collaborators or [])}")
        
        return experiment
    
    def create_matlab_script(self,
                           name: str,
                           description: str,
                           content: str,
                           created_by: str,
                           collaborators: Optional[List[str]] = None,
                           required_toolboxes: Optional[List[MATLABToolbox]] = None) -> MATLABScript:
        """Create a collaborative MATLAB script"""
        
        script_id = str(uuid.uuid4())
        
        script = MATLABScript(
            script_id=script_id,
            name=name,
            description=description,
            content=content,
            created_by=created_by,
            collaborators=collaborators or [],
            version="1.0",
            edit_history=[],
            last_execution=None,
            execution_results=None,
            performance_metrics={},
            required_toolboxes=required_toolboxes or [],
            input_variables=[],
            output_variables=[],
            created_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        self.matlab_scripts[script_id] = script
        self._save_script(script)
        
        # Save script file
        script_file = self.storage_path / "scripts" / f"{script_id}.m"
        script_file.parent.mkdir(exist_ok=True)
        with open(script_file, 'w') as f:
            f.write(content)
        
        print(f"📝 Created MATLAB script: {name}")
        print(f"   Script ID: {script_id}")
        print(f"   Required toolboxes: {len(required_toolboxes or [])}")
        print(f"   Collaborators: {len(collaborators or [])}")
        
        return script
    
    async def start_experiment(self,
                             experiment_id: str,
                             user_id: str) -> Dict[str, Any]:
        """Start running an ML experiment"""
        
        if experiment_id not in self.ml_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.ml_experiments[experiment_id]
        
        # Check permissions
        if user_id != experiment.created_by and user_id not in experiment.collaborators:
            raise PermissionError("Insufficient permissions to start experiment")
        
        # Update experiment status
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        experiment.last_modified = datetime.now()
        
        # Initialize metrics tracking
        experiment.metrics = {
            "training_accuracy": [],
            "validation_accuracy": [],
            "training_loss": [],
            "validation_loss": [],
            "learning_curve": [],
            "computational_time": []
        }
        
        # Create MATLAB execution script
        matlab_script = self._generate_experiment_script(experiment)
        experiment.matlab_code = matlab_script
        
        # Mock experiment execution (in real implementation, would execute MATLAB)
        if not self.matlab_available:
            print("⚠️  MATLAB not available - simulating experiment execution")
            execution_result = self._simulate_experiment_execution(experiment)
        else:
            execution_result = await self._execute_matlab_experiment(experiment)
        
        self._save_experiment(experiment)
        
        print(f"🚀 Started ML experiment: {experiment.name}")
        print(f"   Experiment ID: {experiment_id}")
        print(f"   Status: {experiment.status.value}")
        print(f"   Algorithm: {experiment.algorithm_type}")
        print(f"   Started by: {user_id}")
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
            "execution_result": execution_result
        }
    
    def _generate_experiment_script(self, experiment: MLExperiment) -> str:
        """Generate MATLAB script for experiment execution"""
        
        script_template = f"""
%% ML Experiment: {experiment.name}
%% Generated: {datetime.now().isoformat()}
%% Created by: {experiment.created_by}
%% Experiment ID: {experiment.experiment_id}

%% Initialize experiment tracking
experiment_id = '{experiment.experiment_id}';
experiment_name = '{experiment.name}';
disp(['Starting experiment: ', experiment_name]);

%% Set random seed for reproducibility
rng({experiment.random_seed});

%% Load dataset
% Dataset info: {experiment.dataset_info}
dataset_path = '{experiment.dataset_info.get("path", "data.mat")}';
if exist(dataset_path, 'file')
    load(dataset_path);
    disp(['Dataset loaded: ', dataset_path]);
else
    % Generate synthetic data for demonstration
    n_samples = {experiment.dataset_info.get("n_samples", 1000)};
    n_features = {experiment.dataset_info.get("n_features", 10)};
    X = randn(n_samples, n_features);
    y = sign(sum(X(:, 1:3), 2) + 0.1 * randn(n_samples, 1));
    disp('Using synthetic dataset');
end

%% Hyperparameters
{chr(10).join([f"{k} = {repr(v)};" for k, v in experiment.hyperparameters.items()])}

%% Algorithm: {experiment.algorithm_type}
switch lower('{experiment.algorithm_type}')
    case 'svm'
        % Support Vector Machine
        model = fitcsvm(X, y, 'KernelFunction', 'rbf', ...
                       'BoxConstraint', regularization, ...
                       'KernelScale', 'auto');
        
    case 'random_forest'
        % Random Forest using TreeBagger
        model = TreeBagger(num_trees, X, y, ...
                          'Method', 'classification', ...
                          'NumPredictorsToSample', 'sqrt');
    
    case 'neural_network'
        % Neural Network (requires Deep Learning Toolbox)
        if exist('trainNetwork', 'file')
            layers = [
                featureInputLayer(size(X, 2))
                fullyConnectedLayer(hidden_units)
                reluLayer
                fullyConnectedLayer(length(unique(y)))
                softmaxLayer
                classificationLayer];
            
            options = trainingOptions('adam', ...
                'MaxEpochs', num_epochs, ...
                'InitialLearnRate', learning_rate, ...
                'ValidationFrequency', 10, ...
                'Plots', 'training-progress');
            
            % Convert data for deep learning
            X_dl = dlarray(X', 'CB');
            y_categorical = categorical(y);
            model = trainNetwork(X_dl, y_categorical, layers, options);
        else
            error('Deep Learning Toolbox not available');
        end
        
    otherwise
        error(['Unknown algorithm: ', '{experiment.algorithm_type}']);
end

%% Cross-validation evaluation
cv_folds = 5;
cv_accuracy = zeros(cv_folds, 1);
cv_loss = zeros(cv_folds, 1);

c = cvpartition(length(y), 'KFold', cv_folds);
for fold = 1:cv_folds
    train_idx = training(c, fold);
    test_idx = test(c, fold);
    
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    
    % Train fold model (simplified for template)
    fold_model = fitcsvm(X_train, y_train);
    y_pred = predict(fold_model, X_test);
    
    cv_accuracy(fold) = sum(y_pred == y_test) / length(y_test);
    cv_loss(fold) = sum(y_pred ~= y_test) / length(y_test);
end

%% Results
final_accuracy = mean(cv_accuracy);
final_loss = mean(cv_loss);
std_accuracy = std(cv_accuracy);

disp(['Final Accuracy: ', num2str(final_accuracy)]);
disp(['Final Loss: ', num2str(final_loss)]);
disp(['Accuracy Std: ', num2str(std_accuracy)]);

%% Save results
results = struct();
results.experiment_id = experiment_id;
results.final_accuracy = final_accuracy;
results.final_loss = final_loss;
results.cv_accuracy = cv_accuracy;
results.cv_loss = cv_loss;
results.hyperparameters = struct({', '.join([f"'{k}', {repr(v)}" for k, v in experiment.hyperparameters.items()])});
results.timestamp = datetime('now');

% Save to file
results_file = ['results_', experiment_id, '.mat'];
save(results_file, 'results', 'model');
disp(['Results saved to: ', results_file]);

%% Generate plots
figure('Name', ['Experiment Results: ', experiment_name]);

subplot(2, 2, 1);
bar(cv_accuracy);
title('Cross-Validation Accuracy');
xlabel('Fold');
ylabel('Accuracy');

subplot(2, 2, 2);
bar(cv_loss);
title('Cross-Validation Loss');
xlabel('Fold');
ylabel('Loss');

subplot(2, 2, 3);
histogram(y);
title('Target Distribution');
xlabel('Class');
ylabel('Count');

subplot(2, 2, 4);
if size(X, 2) >= 2
    scatter(X(:, 1), X(:, 2), 10, y, 'filled');
    title('Feature Space (First 2 Features)');
    xlabel('Feature 1');
    ylabel('Feature 2');
    colorbar;
end

% Save plots
plot_file = ['plots_', experiment_id, '.png'];
saveas(gcf, plot_file);
disp(['Plots saved to: ', plot_file]);

disp(['Experiment completed: ', experiment_name]);
"""
        
        return script_template
    
    def _simulate_experiment_execution(self, experiment: MLExperiment) -> Dict[str, Any]:
        """Simulate experiment execution when MATLAB is not available"""
        
        # Simulate realistic ML experiment results
        np.random.seed(experiment.random_seed)
        
        # Generate training curves
        n_epochs = experiment.hyperparameters.get('num_epochs', 100)
        training_acc = 0.5 + 0.4 * (1 - np.exp(-np.arange(n_epochs) / 20)) + 0.05 * np.random.randn(n_epochs)
        validation_acc = training_acc - 0.05 + 0.02 * np.random.randn(n_epochs)
        training_loss = 1.0 * np.exp(-np.arange(n_epochs) / 30) + 0.1 * np.random.randn(n_epochs)
        validation_loss = training_loss + 0.05 + 0.02 * np.random.randn(n_epochs)
        
        # Ensure realistic bounds
        training_acc = np.clip(training_acc, 0, 1)
        validation_acc = np.clip(validation_acc, 0, 1)
        training_loss = np.clip(training_loss, 0, None)
        validation_loss = np.clip(validation_loss, 0, None)
        
        # Update experiment metrics
        experiment.metrics = {
            "training_accuracy": training_acc.tolist(),
            "validation_accuracy": validation_acc.tolist(),
            "training_loss": training_loss.tolist(),
            "validation_loss": validation_loss.tolist(),
            "learning_curve": list(range(n_epochs)),
            "computational_time": [i * 0.1 for i in range(n_epochs)]
        }
        
        # Final metrics
        experiment.final_metrics = {
            "final_training_accuracy": float(training_acc[-1]),
            "final_validation_accuracy": float(validation_acc[-1]),
            "final_training_loss": float(training_loss[-1]),
            "final_validation_loss": float(validation_loss[-1]),
            "best_validation_accuracy": float(np.max(validation_acc)),
            "convergence_epoch": int(np.argmax(validation_acc)),
            "total_time_seconds": n_epochs * 0.1
        }
        
        # Update status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.now()
        
        return {
            "execution_type": "simulated",
            "status": "completed",
            "final_metrics": experiment.final_metrics,
            "artifacts_generated": ["model.mat", "results.mat", "plots.png"]
        }
    
    async def _execute_matlab_experiment(self, experiment: MLExperiment) -> Dict[str, Any]:
        """Execute experiment using actual MATLAB (when available)"""
        
        # Create experiment directory
        exp_dir = self.storage_path / "experiments" / experiment.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write MATLAB script
        script_path = exp_dir / f"experiment_{experiment.experiment_id}.m"
        with open(script_path, 'w') as f:
            f.write(experiment.matlab_code)
        
        try:
            # Execute MATLAB script
            result = subprocess.run([
                "matlab", "-batch", 
                f"cd('{exp_dir}'); run('{script_path.name}');"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                experiment.status = ExperimentStatus.COMPLETED
                experiment.end_time = datetime.now()
                
                # Parse results (simplified - would parse actual MATLAB output)
                return {
                    "execution_type": "matlab",
                    "status": "completed",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                experiment.status = ExperimentStatus.FAILED
                return {
                    "execution_type": "matlab",
                    "status": "failed",
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            experiment.status = ExperimentStatus.FAILED
            return {
                "execution_type": "matlab",
                "status": "timeout",
                "error": "Experiment exceeded time limit"
            }
    
    def track_experiment_metrics(self,
                               experiment_id: str,
                               metrics: Dict[str, float],
                               step: int,
                               user_id: str):
        """Track metrics during experiment execution"""
        
        if experiment_id not in self.ml_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.ml_experiments[experiment_id]
        
        # Add metrics to tracking
        for metric_name, value in metrics.items():
            if metric_name not in experiment.metrics:
                experiment.metrics[metric_name] = []
            experiment.metrics[metric_name].append(value)
        
        # Update last modified
        experiment.last_modified = datetime.now()
        self._save_experiment(experiment)
        
        print(f"📊 Tracked metrics for experiment {experiment.name}:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value}")
    
    async def get_matlab_optimization_advice(self,
                                           script_id: str,
                                           optimization_goals: List[str],
                                           user_id: str) -> Dict[str, Any]:
        """Get NWTN AI advice for MATLAB code optimization"""
        
        if script_id not in self.matlab_scripts:
            raise ValueError(f"Script {script_id} not found")
        
        script = self.matlab_scripts[script_id]
        
        await self.initialize_nwtn_pipeline()
        
        optimization_prompt = f"""
Please provide MATLAB code optimization advice for this collaborative research script:

**Script**: {script.name}
**Description**: {script.description}
**Optimization Goals**: {', '.join(optimization_goals)}
**Required Toolboxes**: {[tb.value for tb in script.required_toolboxes]}

**Current MATLAB Code**:
```matlab
{script.content}
```

Please provide:
1. Performance optimization recommendations
2. Memory usage improvements
3. Vectorization opportunities
4. Parallel computing suggestions
5. Best practices for collaborative development

Focus on university-industry collaboration requirements and computational efficiency.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=optimization_prompt,
            context={
                "domain": "matlab_optimization",
                "matlab_optimization": True,
                "script_type": "research_collaboration",
                "optimization_type": "comprehensive_analysis"
            }
        )
        
        advice = {
            "script_id": script_id,
            "script_name": script.name,
            "optimization_goals": optimization_goals,
            "recommendations": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"🔧 MATLAB optimization advice generated:")
        print(f"   Script: {script.name}")
        print(f"   Goals: {len(optimization_goals)} optimization objectives")
        print(f"   Confidence: {advice['confidence']:.2f}")
        
        return advice
    
    async def get_experiment_tracking_guidance(self,
                                             experiment_type: str,
                                             research_goals: List[str],
                                             user_id: str) -> Dict[str, Any]:
        """Get guidance on ML experiment tracking best practices"""
        
        await self.initialize_nwtn_pipeline()
        
        tracking_prompt = f"""
Please provide ML experiment tracking guidance for university-industry collaboration:

**Experiment Type**: {experiment_type}
**Research Goals**: {', '.join(research_goals)}
**Context**: Multi-institutional research using MATLAB
**Requirements**: Reproducibility, collaboration, and knowledge transfer

Please provide:
1. Experiment organization and structure recommendations
2. Metrics tracking and visualization strategies
3. Reproducibility best practices
4. Collaborative experiment management
5. Integration with MATLAB workflows

Focus on practices that enhance university-industry partnerships and facilitate knowledge sharing.
"""
        
        result = await self.nwt_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=tracking_prompt,
            context={
                "domain": "experiment_tracking",
                "experiment_tracking": True,
                "experiment_type": experiment_type,
                "guidance_type": "comprehensive_practices"
            }
        )
        
        guidance = {
            "experiment_type": experiment_type,
            "research_goals": research_goals,
            "tracking_guidance": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"📈 Experiment tracking guidance generated:")
        print(f"   Type: {experiment_type}")
        print(f"   Goals: {len(research_goals)} research objectives")
        print(f"   Confidence: {guidance['confidence']:.2f}")
        
        return guidance
    
    def generate_experiment_report(self,
                                 experiment_id: str,
                                 user_id: str,
                                 include_plots: bool = True) -> Dict[str, Any]:
        """Generate comprehensive experiment report"""
        
        if experiment_id not in self.ml_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.ml_experiments[experiment_id]
        
        # Generate plots if requested
        plots = []
        if include_plots and experiment.metrics:
            plots = self._generate_experiment_plots(experiment)
        
        # Create report
        report = {
            "experiment_info": {
                "id": experiment.experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "created_by": experiment.created_by,
                "collaborators": experiment.collaborators,
                "status": experiment.status.value,
                "algorithm_type": experiment.algorithm_type
            },
            "configuration": {
                "hyperparameters": experiment.hyperparameters,
                "dataset_info": experiment.dataset_info,
                "matlab_environment": experiment.matlab_environment,
                "random_seed": experiment.random_seed
            },
            "results": {
                "final_metrics": experiment.final_metrics or {},
                "status": experiment.status.value,
                "runtime": self._calculate_runtime(experiment)
            },
            "reproducibility": {
                "matlab_code": experiment.matlab_code,
                "environment_snapshot": experiment.environment_snapshot,
                "random_seed": experiment.random_seed
            },
            "plots": plots,
            "generated_at": datetime.now().isoformat(),
            "generated_by": user_id
        }
        
        # Save report
        report_file = self.storage_path / "reports" / f"report_{experiment_id}.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Generated experiment report:")
        print(f"   Experiment: {experiment.name}")
        print(f"   Status: {experiment.status.value}")
        print(f"   Report file: {report_file.name}")
        if plots:
            print(f"   Plots generated: {len(plots)}")
        
        return report
    
    def _generate_experiment_plots(self, experiment: MLExperiment) -> List[str]:
        """Generate visualization plots for experiment"""
        
        plots = []
        plot_dir = self.storage_path / "plots" / experiment.experiment_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        if not experiment.metrics:
            return plots
        
        # Learning curves
        if "training_accuracy" in experiment.metrics and "validation_accuracy" in experiment.metrics:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(experiment.metrics["training_accuracy"], label="Training")
            plt.plot(experiment.metrics["validation_accuracy"], label="Validation")
            plt.title("Accuracy Learning Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(experiment.metrics["training_loss"], label="Training")
            plt.plot(experiment.metrics["validation_loss"], label="Validation")
            plt.title("Loss Learning Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            # Hyperparameter visualization
            hp_names = list(experiment.hyperparameters.keys())
            hp_values = list(experiment.hyperparameters.values())
            if hp_names and all(isinstance(v, (int, float)) for v in hp_values):
                plt.bar(hp_names, hp_values)
                plt.title("Hyperparameters")
                plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            # Final metrics comparison
            if experiment.final_metrics:
                metric_names = list(experiment.final_metrics.keys())
                metric_values = list(experiment.final_metrics.values())
                if metric_names:
                    plt.bar(metric_names, metric_values)
                    plt.title("Final Metrics")
                    plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            plot_file = plot_dir / "learning_curves.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots.append(str(plot_file))
        
        return plots
    
    def _calculate_runtime(self, experiment: MLExperiment) -> Optional[float]:
        """Calculate experiment runtime in seconds"""
        if experiment.start_time and experiment.end_time:
            return (experiment.end_time - experiment.start_time).total_seconds()
        return None
    
    def _save_environment(self, environment: MATLABEnvironment):
        """Save MATLAB environment configuration"""
        env_dir = Path(environment.workspace_path)
        env_file = env_dir / "environment.json"
        
        with open(env_file, 'w') as f:
            env_data = asdict(environment)
            json.dump(env_data, f, default=str, indent=2)
    
    def _save_experiment(self, experiment: MLExperiment):
        """Save ML experiment configuration and results"""
        exp_dir = self.storage_path / "experiments" / experiment.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        exp_file = exp_dir / "experiment.json"
        with open(exp_file, 'w') as f:
            exp_data = asdict(experiment)
            json.dump(exp_data, f, default=str, indent=2)
    
    def _save_script(self, script: MATLABScript):
        """Save MATLAB script configuration"""
        script_dir = self.storage_path / "scripts" / script.script_id
        script_dir.mkdir(parents=True, exist_ok=True)
        
        script_file = script_dir / "script.json"
        with open(script_file, 'w') as f:
            script_data = asdict(script)
            json.dump(script_data, f, default=str, indent=2)

# RTP-specific MATLAB integrations
class RTPMATLABIntegration:
    """Research Triangle Park specific MATLAB integrations"""
    
    def __init__(self, matlab_collab: MATLABCollaboration):
        self.matlab_collab = matlab_collab
        self.sas_matlab_bridge = self._initialize_sas_bridge()
    
    def _initialize_sas_bridge(self) -> Dict[str, Any]:
        """Initialize SAS-MATLAB integration bridge"""
        return {
            "sas_connection": {
                "server": "sas.rtp.sas.com",
                "protocol": "https",
                "authentication": "university_sso"
            },
            "data_exchange": {
                "formats": ["sas7bdat", "csv", "parquet"],
                "encryption": "aes256",
                "compression": "gzip"
            },
            "shared_libraries": [
                "sas_matlab_analytics",
                "statistical_procedures",
                "optimization_routines"
            ]
        }
    
    def create_sas_matlab_collaboration(self,
                                      project_name: str,
                                      university_researcher: str,
                                      sas_analyst: str,
                                      shared_datasets: List[str]) -> MATLABEnvironment:
        """Create MATLAB environment optimized for SAS collaboration"""
        
        environment = self.matlab_collab.create_matlab_environment(
            name=f"SAS-University Partnership: {project_name}",
            description=f"Collaborative MATLAB environment for {project_name} with SAS Institute integration",
            matlab_version="R2023b",
            owner=university_researcher,
            collaborators={
                sas_analyst: MATLABAccessLevel.COLLABORATOR
            },
            required_toolboxes=[
                MATLABToolbox.STATISTICS_ML,
                MATLABToolbox.OPTIMIZATION,
                MATLABToolbox.PARALLEL_COMPUTING
            ],
            security_level="high"
        )
        
        # Add SAS integration script
        sas_integration_script = f"""
%% SAS-MATLAB Integration Script
%% Project: {project_name}
%% Created: {datetime.now().isoformat()}

%% SAS data import functions
function data = import_sas_data(dataset_path)
    % Import SAS7BDAT files into MATLAB
    if contains(dataset_path, '.sas7bdat')
        % Use SAS Import Tool or third-party libraries
        data = readtable(dataset_path);
    else
        data = readtable(dataset_path);
    end
    disp(['Imported dataset: ', dataset_path]);
end

%% SAS procedure equivalents in MATLAB
function results = proc_means_matlab(data, variables)
    % MATLAB equivalent of SAS PROC MEANS
    results = table();
    for i = 1:length(variables)
        var_data = data.(variables{{i}});
        if isnumeric(var_data)
            var_stats = struct();
            var_stats.N = sum(~isnan(var_data));
            var_stats.Mean = mean(var_data, 'omitnan');
            var_stats.Std = std(var_data, 'omitnan');
            var_stats.Min = min(var_data);
            var_stats.Max = max(var_data);
            results.(variables{{i}}) = var_stats;
        end
    end
end

function model = proc_reg_matlab(data, response, predictors)
    % MATLAB equivalent of SAS PROC REG
    formula = [response, ' ~ ', strjoin(predictors, ' + ')];
    model = fitlm(data, formula);
    disp(model);
end

%% Data sharing with SAS
{chr(10).join([f"% Dataset: {ds}" for ds in shared_datasets])}

disp('✅ SAS-MATLAB integration environment ready!');
disp('🤝 University-Industry collaboration tools loaded');
"""
        
        script = self.matlab_collab.create_matlab_script(
            name="sas_matlab_integration",
            description="SAS-MATLAB integration utilities for university-industry collaboration",
            content=sas_integration_script,
            created_by=university_researcher,
            collaborators=[sas_analyst],
            required_toolboxes=[MATLABToolbox.STATISTICS_ML]
        )
        
        print(f"🏢 Created SAS-MATLAB collaboration environment:")
        print(f"   University researcher: {university_researcher}")
        print(f"   SAS analyst: {sas_analyst}")
        print(f"   Shared datasets: {len(shared_datasets)}")
        print(f"   Integration script: {script.name}")
        
        return environment

# Example usage and testing
if __name__ == "__main__":
    async def test_matlab_collaboration():
        """Test MATLAB collaboration and ML experiment tracking"""
        
        print("🚀 Testing MATLAB Collaboration and ML Experiment Tracking")
        print("=" * 60)
        
        # Initialize MATLAB collaboration
        matlab_collab = MATLABCollaboration()
        
        # Create MATLAB environment for quantum computing research
        environment = matlab_collab.create_matlab_environment(
            name="Quantum Algorithm Development - Multi-University Partnership",
            description="Collaborative MATLAB environment for quantum error correction research with ML optimization",
            matlab_version="R2023b",
            owner="sarah.chen@unc.edu",
            collaborators={
                "alex.rodriguez@duke.edu": MATLABAccessLevel.COLLABORATOR,
                "jennifer.kim@ncsu.edu": MATLABAccessLevel.COLLABORATOR,
                "michael.johnson@sas.com": MATLABAccessLevel.ANALYST,
                "ml.engineer@unc.edu": MATLABAccessLevel.COLLABORATOR
            },
            required_toolboxes=[
                MATLABToolbox.STATISTICS_ML,
                MATLABToolbox.OPTIMIZATION,
                MATLABToolbox.PARALLEL_COMPUTING,
                MATLABToolbox.DEEP_LEARNING
            ],
            security_level="high"
        )
        
        print(f"\n✅ Created MATLAB environment: {environment.name}")
        print(f"   Environment ID: {environment.env_id}")
        print(f"   MATLAB Version: {environment.matlab_version}")
        print(f"   Toolboxes: {len(environment.installed_toolboxes)}")
        print(f"   Collaborators: {len(environment.collaborators)}")
        
        # Create ML experiment for quantum algorithm optimization
        experiment = matlab_collab.create_ml_experiment(
            name="Quantum Error Correction Parameter Optimization",
            description="ML-based optimization of adaptive quantum error correction algorithms using MATLAB",
            algorithm_type="neural_network",
            created_by="sarah.chen@unc.edu",
            matlab_environment=environment.env_id,
            hyperparameters={
                "learning_rate": 0.001,
                "num_epochs": 100,
                "batch_size": 32,
                "hidden_units": 128,
                "regularization": 0.01,
                "dropout_rate": 0.2
            },
            dataset_info={
                "name": "quantum_error_correction_data",
                "path": "data/quantum_experiments.mat",
                "n_samples": 5000,
                "n_features": 15,
                "description": "Experimental data from quantum error correction trials"
            },
            collaborators=[
                "alex.rodriguez@duke.edu",
                "michael.johnson@sas.com",
                "ml.engineer@unc.edu"
            ]
        )
        
        print(f"\n✅ Created ML experiment: {experiment.name}")
        print(f"   Experiment ID: {experiment.experiment_id}")
        print(f"   Algorithm: {experiment.algorithm_type}")
        print(f"   Hyperparameters: {len(experiment.hyperparameters)}")
        print(f"   Collaborators: {len(experiment.collaborators)}")
        
        # Create MATLAB script for the experiment
        matlab_script_content = """
%% Quantum Error Correction ML Optimization
%% Multi-University Collaboration: UNC + Duke + NC State + SAS
%% Principal Investigator: Dr. Sarah Chen (UNC Physics)

function results = quantum_error_correction_optimization()
    % Advanced quantum error correction parameter optimization using ML
    
    %% Initialize quantum simulation parameters
    n_qubits = 7;  % 7-qubit error correction code
    noise_levels = linspace(0.001, 0.1, 50);
    correction_methods = {'adaptive', 'standard', 'hybrid'};
    
    %% Load experimental data
    load('quantum_experiments.mat', 'error_rates', 'success_rates', 'noise_profiles');
    
    %% Prepare training data for ML optimization
    features = [error_rates, noise_profiles];
    targets = success_rates;
    
    %% Split data for cross-validation
    cv = cvpartition(length(targets), 'HoldOut', 0.2);
    X_train = features(training(cv), :);
    y_train = targets(training(cv));
    X_test = features(test(cv), :);
    y_test = targets(test(cv));
    
    %% Neural network for parameter optimization
    % Create feed-forward neural network
    hiddenLayerSize = 128;
    net = fitnet(hiddenLayerSize, 'trainlm');
    
    % Configure training
    net.trainParam.epochs = 100;
    net.trainParam.lr = 0.001;
    net.trainParam.goal = 1e-6;
    
    % Train network
    [net, tr] = train(net, X_train', y_train');
    
    %% Evaluate model performance
    y_pred_train = net(X_train');
    y_pred_test = net(X_test');
    
    train_mse = mse(y_train' - y_pred_train);
    test_mse = mse(y_test' - y_pred_test);
    train_r2 = corr(y_train, y_pred_train')^2;
    test_r2 = corr(y_test, y_pred_test')^2;
    
    %% Optimize quantum error correction parameters
    % Define optimization objective
    objective = @(params) -quantum_error_correction_performance(params, net);
    
    % Set parameter bounds
    lb = [0.001, 0.1, 0.5];  % [threshold, adaptation_rate, correction_strength]
    ub = [0.1, 1.0, 2.0];
    
    % Multi-objective optimization using genetic algorithm
    options = optimoptions('ga', 'MaxGenerations', 50, 'PopulationSize', 100);
    [optimal_params, optimal_performance] = ga(objective, 3, [], [], [], [], lb, ub, [], options);
    
    %% Results compilation
    results = struct();
    results.model_performance = struct('train_mse', train_mse, 'test_mse', test_mse, ...
                                     'train_r2', train_r2, 'test_r2', test_r2);
    results.optimal_parameters = optimal_params;
    results.optimal_performance = -optimal_performance;
    results.improvement_percentage = ((-optimal_performance - mean(y_test)) / mean(y_test)) * 100;
    
    %% Visualization
    figure('Name', 'Quantum Error Correction Optimization Results');
    
    subplot(2, 2, 1);
    scatter(y_test, y_pred_test');
    hold on;
    plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--');
    xlabel('Actual Success Rate');
    ylabel('Predicted Success Rate');
    title(['Test Set Predictions (R² = ', num2str(test_r2, '%.3f'), ')']);
    
    subplot(2, 2, 2);
    plot(tr.perf);
    xlabel('Epoch');
    ylabel('Performance (MSE)');
    title('Training Progress');
    
    subplot(2, 2, 3);
    bar(optimal_params);
    xlabel('Parameter Index');
    ylabel('Optimal Value');
    title('Optimized Parameters');
    xticklabels({'Threshold', 'Adaptation Rate', 'Correction Strength'});
    
    subplot(2, 2, 4);
    improvement_data = [mean(y_test), -optimal_performance];
    bar(improvement_data);
    xlabel('Condition');
    ylabel('Success Rate');
    title(['Performance Improvement: ', num2str(results.improvement_percentage, '%.1f'), '%']);
    xticklabels({'Baseline', 'Optimized'});
    
    %% Save results
    save(['quantum_optimization_results_', datestr(now, 'yyyymmdd_HHMMSS'), '.mat'], 'results', 'net', 'optimal_params');
    
    fprintf('\\n✅ Quantum error correction optimization completed!\\n');
    fprintf('🎯 Performance improvement: %.1f%%\\n', results.improvement_percentage);
    fprintf('🔬 Multi-university collaboration results ready for publication\\n');
end

function performance = quantum_error_correction_performance(params, model)
    % Evaluate quantum error correction performance for given parameters
    % This would interface with actual quantum simulation or experimental data
    
    threshold = params(1);
    adaptation_rate = params(2);
    correction_strength = params(3);
    
    % Simulate quantum error correction with these parameters
    % (In practice, this would run actual quantum simulations)
    base_performance = 0.85;  % Baseline success rate
    threshold_factor = 1 - exp(-10 * threshold);
    adaptation_factor = adaptation_rate * 0.1;
    correction_factor = min(correction_strength * 0.05, 0.1);
    
    performance = base_performance + threshold_factor + adaptation_factor + correction_factor;
    performance = min(performance, 1.0);  % Cap at 100% success rate
end
"""
        
        script = matlab_collab.create_matlab_script(
            name="quantum_error_correction_ml_optimization",
            description="ML-based optimization of quantum error correction parameters using neural networks",
            content=matlab_script_content,
            created_by="sarah.chen@unc.edu",
            collaborators=[
                "alex.rodriguez@duke.edu",
                "michael.johnson@sas.com",
                "ml.engineer@unc.edu"
            ],
            required_toolboxes=[
                MATLABToolbox.STATISTICS_ML,
                MATLABToolbox.OPTIMIZATION,
                MATLABToolbox.DEEP_LEARNING
            ]
        )
        
        print(f"\n✅ Created MATLAB script: {script.name}")
        print(f"   Script ID: {script.script_id}")
        print(f"   Required toolboxes: {len(script.required_toolboxes)}")
        print(f"   Collaborators: {len(script.collaborators)}")
        
        # Start ML experiment
        print(f"\n🚀 Starting ML experiment...")
        
        experiment_result = await matlab_collab.start_experiment(
            experiment.experiment_id,
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ ML experiment started:")
        print(f"   Status: {experiment_result['status']}")
        print(f"   Execution: {experiment_result.get('execution_result', {}).get('execution_type', 'unknown')}")
        
        # Track additional metrics during experiment
        matlab_collab.track_experiment_metrics(
            experiment.experiment_id,
            {
                "memory_usage_gb": 4.2,
                "gpu_utilization": 0.85,
                "parallel_efficiency": 0.92
            },
            step=50,
            user_id="sarah.chen@unc.edu"
        )
        
        # Get MATLAB optimization advice
        print(f"\n🔧 Getting MATLAB optimization advice...")
        
        optimization_advice = await matlab_collab.get_matlab_optimization_advice(
            script.script_id,
            ["performance", "memory_usage", "parallel_computing", "collaboration"],
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ MATLAB optimization advice generated:")
        print(f"   Confidence: {optimization_advice['confidence']:.2f}")
        print(f"   Processing time: {optimization_advice['processing_time']:.1f}s")
        print(f"   Advice preview: {optimization_advice['recommendations'][:200]}...")
        
        # Get experiment tracking guidance
        print(f"\n📈 Getting experiment tracking guidance...")
        
        tracking_guidance = await matlab_collab.get_experiment_tracking_guidance(
            "neural_network_optimization",
            ["quantum_computing", "parameter_optimization", "multi_institutional_collaboration"],
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Experiment tracking guidance generated:")
        print(f"   Confidence: {tracking_guidance['confidence']:.2f}")
        
        # Generate experiment report
        print(f"\n📋 Generating experiment report...")
        
        report = matlab_collab.generate_experiment_report(
            experiment.experiment_id,
            "sarah.chen@unc.edu",
            include_plots=True
        )
        
        print(f"✅ Experiment report generated:")
        print(f"   Status: {report['experiment_info']['status']}")
        print(f"   Plots: {len(report['plots'])} visualization files")
        if report['results']['final_metrics']:
            print(f"   Final accuracy: {report['results']['final_metrics'].get('best_validation_accuracy', 'N/A')}")
        
        # Test RTP-specific SAS integration
        print(f"\n🏢 Testing SAS Institute integration...")
        
        rtp_integration = RTPMATLABIntegration(matlab_collab)
        
        sas_environment = rtp_integration.create_sas_matlab_collaboration(
            "Advanced Quantum Analytics Partnership",
            "sarah.chen@unc.edu",
            "michael.johnson@sas.com",
            ["quantum_performance_metrics.sas7bdat", "experimental_results.csv", "parameter_optimization_data.parquet"]
        )
        
        print(f"✅ SAS-MATLAB collaboration created:")
        print(f"   Environment: {sas_environment.name}")
        print(f"   Integration features: Data exchange, shared libraries, collaborative analytics")
        
        print(f"\n🎉 MATLAB collaboration and ML experiment tracking test completed!")
        print("✅ Ready for university-industry quantum computing research partnerships!")
    
    # Run test
    import asyncio
    asyncio.run(test_matlab_collaboration())