# MLflow Integration Guide

Integrate PRSM with MLflow for comprehensive machine learning lifecycle management including experiment tracking, model versioning, and deployment.

## 🎯 Overview

This guide covers integrating PRSM with MLflow to track experiments, manage model versions, deploy models, and monitor ML workflows for better reproducibility and collaboration.

## 📋 Prerequisites

- PRSM instance configured
- MLflow installed
- Python 3.8+ installed
- Basic knowledge of machine learning workflows
- Database for MLflow backend (optional but recommended)

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Install MLflow and dependencies
pip install mlflow
pip install mlflow[extras]  # Additional integrations
pip install boto3  # For S3 artifact storage
pip install psycopg2-binary  # For PostgreSQL backend
pip install pymysql  # For MySQL backend

# Additional ML libraries
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
```

### 2. MLflow Configuration

```python
# prsm/integrations/mlflow/config.py
import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings
import mlflow

class MLflowConfig(BaseSettings):
    """MLflow integration configuration."""
    
    # MLflow Server Configuration
    tracking_uri: str = "http://localhost:5000"
    registry_uri: Optional[str] = None
    
    # Experiment Configuration
    default_experiment_name: str = "prsm_experiments"
    experiment_prefix: str = "prsm"
    
    # Model Registry Configuration
    default_model_name: str = "prsm_model"
    model_stage: str = "Staging"  # None, Staging, Production, Archived
    
    # Artifact Storage
    artifact_location: Optional[str] = None  # S3, Azure, GCS, or local path
    s3_bucket: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    
    # Database Backend (optional)
    backend_store_uri: Optional[str] = None  # PostgreSQL, MySQL, SQLite
    
    # Logging Configuration
    log_models: bool = True
    log_artifacts: bool = True
    log_metrics: bool = True
    log_params: bool = True
    log_tags: bool = True
    
    # AutoML Configuration
    enable_autolog: bool = True
    autolog_frameworks: list = ["sklearn", "tensorflow", "pytorch", "transformers"]
    
    class Config:
        env_prefix = "PRSM_MLFLOW_"

# Global configuration
mlflow_config = MLflowConfig()

# Configure MLflow
mlflow.set_tracking_uri(mlflow_config.tracking_uri)
if mlflow_config.registry_uri:
    mlflow.set_registry_uri(mlflow_config.registry_uri)
```

### 3. Basic MLflow Integration

```python
# prsm/integrations/mlflow/client.py
import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np

from prsm.integrations.mlflow.config import mlflow_config

logger = logging.getLogger(__name__)

class PRSMMLflowClient:
    """MLflow client for PRSM integration."""
    
    def __init__(self):
        self.config = mlflow_config
        self.client = MlflowClient(
            tracking_uri=self.config.tracking_uri,
            registry_uri=self.config.registry_uri
        )
        
        # Setup default experiment
        self._setup_default_experiment()
        
        # Enable autologging if configured
        if self.config.enable_autolog:
            self._enable_autologging()
    
    def _setup_default_experiment(self):
        """Setup default experiment for PRSM."""
        try:
            experiment = mlflow.get_experiment_by_name(self.config.default_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.config.default_experiment_name,
                    artifact_location=self.config.artifact_location,
                    tags={
                        "project": "PRSM",
                        "framework": "prsm",
                        "created_by": "prsm_integration"
                    }
                )
                logger.info(f"Created default experiment: {self.config.default_experiment_name}")
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_id=experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to setup default experiment: {e}")
    
    def _enable_autologging(self):
        """Enable MLflow autologging for supported frameworks."""
        try:
            for framework in self.config.autolog_frameworks:
                if framework == "sklearn":
                    mlflow.sklearn.autolog()
                elif framework == "tensorflow":
                    mlflow.tensorflow.autolog()
                elif framework == "pytorch":
                    mlflow.pytorch.autolog()
                elif framework == "transformers":
                    try:
                        mlflow.transformers.autolog()
                    except AttributeError:
                        logger.warning("MLflow transformers autolog not available")
            
            logger.info(f"Enabled autologging for: {self.config.autolog_frameworks}")
            
        except Exception as e:
            logger.warning(f"Failed to enable autologging: {e}")
    
    async def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:
        """Start a new MLflow run."""
        try:
            # Set experiment if specified
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            # Prepare tags
            run_tags = {
                "prsm.integration": "true",
                "prsm.version": "1.0.0",
                "prsm.timestamp": str(int(time.time()))
            }
            if tags:
                run_tags.update(tags)
            
            # Start run
            run = mlflow.start_run(
                run_name=run_name,
                tags=run_tags,
                nested=nested
            )
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    async def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        try:
            mlflow.end_run(status=status)
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    async def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    async def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    async def log_artifacts(self, artifacts: Dict[str, Any], artifact_path: str = ""):
        """Log artifacts to MLflow."""
        try:
            for name, content in artifacts.items():
                if isinstance(content, str):
                    # Text content
                    mlflow.log_text(content, f"{artifact_path}/{name}.txt")
                elif isinstance(content, dict):
                    # JSON content
                    mlflow.log_dict(content, f"{artifact_path}/{name}.json")
                elif isinstance(content, (pd.DataFrame, np.ndarray)):
                    # Data artifacts
                    if isinstance(content, pd.DataFrame):
                        content.to_csv(f"temp_{name}.csv", index=False)
                        mlflow.log_artifact(f"temp_{name}.csv", artifact_path)
                        os.remove(f"temp_{name}.csv")
                    else:
                        np.save(f"temp_{name}.npy", content)
                        mlflow.log_artifact(f"temp_{name}.npy", artifact_path)
                        os.remove(f"temp_{name}.npy")
                        
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
    
    async def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        model_type: str = "sklearn",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        pip_requirements: Optional[List[str]] = None
    ):
        """Log model to MLflow."""
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements
                )
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    tf_saved_model_dir=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements
                )
            else:
                # Generic model logging
                mlflow.log_artifact(model, artifact_path)
            
            logger.info(f"Logged {model_type} model to {artifact_path}")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    async def register_model(
        self,
        model_uri: str,
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """Register model in MLflow Model Registry."""
        try:
            model_name = model_name or self.config.default_model_name
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ):
        """Transition model to a different stage."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
    
    async def get_model_by_stage(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> Optional[Dict[str, Any]]:
        """Get model version by stage."""
        try:
            model_version = self.client.get_latest_versions(
                name=model_name,
                stages=[stage]
            )
            
            if model_version:
                mv = model_version[0]
                return {
                    "name": mv.name,
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "uri": f"models:/{mv.name}/{mv.version}",
                    "run_id": mv.run_id,
                    "creation_timestamp": mv.creation_timestamp,
                    "description": mv.description,
                    "tags": mv.tags
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model by stage: {e}")
            return None
    
    async def search_experiments(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Search experiments."""
        try:
            experiments = self.client.search_experiments(
                filter_string=filter_string,
                max_results=max_results,
                view_type=ViewType.ACTIVE_ONLY
            )
            
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "artifact_location": exp.artifact_location,
                    "tags": exp.tags
                }
                for exp in experiments
            ]
            
        except Exception as e:
            logger.error(f"Failed to search experiments: {e}")
            return []
    
    async def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search runs across experiments."""
        try:
            runs = self.client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by or ["start_time DESC"]
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                for run in runs
            ]
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []

# Global MLflow client
mlflow_client = PRSMMLflowClient()
```

## 🧪 Experiment Tracking

### 1. PRSM Query Tracking

```python
# prsm/integrations/mlflow/experiment_tracking.py
import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional
import mlflow
import pandas as pd
from datetime import datetime

from prsm.integrations.mlflow.client import mlflow_client
from prsm.models.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

class PRSMExperimentTracker:
    """Track PRSM experiments with MLflow."""
    
    def __init__(self):
        self.client = mlflow_client
        self.active_experiments = {}
    
    async def start_query_experiment(
        self,
        experiment_name: str,
        query_request: QueryRequest,
        model_config: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start tracking a query experiment."""
        try:
            # Prepare experiment tags
            exp_tags = {
                "experiment_type": "query_processing",
                "user_id": query_request.user_id,
                "model": query_request.model or "default",
                "timestamp": datetime.utcnow().isoformat()
            }
            if tags:
                exp_tags.update(tags)
            
            # Start MLflow run
            run_id = await self.client.start_run(
                run_name=f"query_exp_{int(time.time())}",
                experiment_name=experiment_name,
                tags=exp_tags
            )
            
            # Log query parameters
            params = {
                "prompt_length": len(query_request.prompt),
                "max_tokens": query_request.max_tokens,
                "temperature": query_request.temperature,
                "model": query_request.model or "default",
                "use_cache": query_request.use_cache
            }
            params.update(model_config)
            await self.client.log_params(params)
            
            # Log query artifacts
            artifacts = {
                "query_request": query_request.dict(),
                "model_config": model_config,
                "prompt": query_request.prompt
            }
            await self.client.log_artifacts(artifacts, "input")
            
            # Store active experiment
            self.active_experiments[run_id] = {
                "experiment_name": experiment_name,
                "start_time": time.time(),
                "query_request": query_request
            }
            
            logger.info(f"Started query experiment: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start query experiment: {e}")
            raise
    
    async def log_query_response(
        self,
        run_id: str,
        query_response: QueryResponse,
        execution_metrics: Dict[str, float]
    ):
        """Log query response and metrics."""
        try:
            # Set active run
            with mlflow.start_run(run_id=run_id):
                # Log response metrics
                metrics = {
                    "response_length": len(query_response.final_answer),
                    "token_usage": query_response.token_usage,
                    "execution_time": execution_metrics.get("execution_time", 0),
                    "processing_time": execution_metrics.get("processing_time", 0)
                }
                metrics.update(execution_metrics)
                await self.client.log_metrics(metrics)
                
                # Log response artifacts
                artifacts = {
                    "query_response": query_response.dict(),
                    "final_answer": query_response.final_answer,
                    "metadata": query_response.metadata or {}
                }
                await self.client.log_artifacts(artifacts, "output")
                
                # Calculate quality metrics if possible
                quality_metrics = await self._calculate_quality_metrics(
                    self.active_experiments[run_id]["query_request"],
                    query_response
                )
                if quality_metrics:
                    await self.client.log_metrics(quality_metrics)
            
            logger.info(f"Logged query response for run: {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log query response: {e}")
    
    async def end_query_experiment(
        self,
        run_id: str,
        status: str = "FINISHED",
        final_metrics: Optional[Dict[str, float]] = None
    ):
        """End query experiment tracking."""
        try:
            if run_id in self.active_experiments:
                exp_info = self.active_experiments[run_id]
                
                # Calculate total experiment time
                total_time = time.time() - exp_info["start_time"]
                
                # Set active run and log final metrics
                with mlflow.start_run(run_id=run_id):
                    final_exp_metrics = {"total_experiment_time": total_time}
                    if final_metrics:
                        final_exp_metrics.update(final_metrics)
                    
                    await self.client.log_metrics(final_exp_metrics)
                
                # Remove from active experiments
                del self.active_experiments[run_id]
            
            # End MLflow run
            await self.client.end_run(status=status)
            
            logger.info(f"Ended query experiment: {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to end query experiment: {e}")
    
    async def _calculate_quality_metrics(
        self,
        query_request: QueryRequest,
        query_response: QueryResponse
    ) -> Dict[str, float]:
        """Calculate quality metrics for the query/response pair."""
        try:
            metrics = {}
            
            # Response completeness (basic heuristic)
            if query_response.final_answer:
                metrics["response_completeness"] = min(
                    len(query_response.final_answer) / 100, 1.0
                )
            
            # Token efficiency
            if query_response.token_usage > 0:
                metrics["tokens_per_char"] = (
                    query_response.token_usage / len(query_response.final_answer)
                    if query_response.final_answer else 0
                )
            
            # Response relevance (placeholder - would need actual evaluation)
            metrics["estimated_relevance"] = 0.8  # Placeholder
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics: {e}")
            return {}
    
    async def compare_experiments(
        self,
        experiment_name: str,
        metric_names: List[str] = None,
        max_results: int = 50
    ) -> pd.DataFrame:
        """Compare experiments and return results as DataFrame."""
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment {experiment_name} not found")
            
            # Search runs
            runs = await self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            # Convert to DataFrame
            data = []
            for run in runs:
                row = {
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "start_time": pd.to_datetime(run["start_time"], unit="ms"),
                    "duration": (
                        (run["end_time"] - run["start_time"]) / 1000
                        if run["end_time"] else None
                    )
                }
                
                # Add parameters
                row.update({f"param_{k}": v for k, v in run["params"].items()})
                
                # Add metrics
                row.update({f"metric_{k}": v for k, v in run["metrics"].items()})
                
                # Add specific metrics if requested
                if metric_names:
                    for metric in metric_names:
                        row[metric] = run["metrics"].get(metric)
                
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to compare experiments: {e}")
            return pd.DataFrame()
    
    async def get_best_run(
        self,
        experiment_name: str,
        metric_name: str,
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get the best run based on a specific metric."""
        try:
            df = await self.compare_experiments(experiment_name)
            
            if df.empty or f"metric_{metric_name}" not in df.columns:
                return None
            
            # Find best run
            if maximize:
                best_idx = df[f"metric_{metric_name}"].idxmax()
            else:
                best_idx = df[f"metric_{metric_name}"].idxmin()
            
            best_run = df.iloc[best_idx]
            
            return {
                "run_id": best_run["run_id"],
                "metric_value": best_run[f"metric_{metric_name}"],
                "parameters": {
                    k.replace("param_", ""): v 
                    for k, v in best_run.items() 
                    if k.startswith("param_") and pd.notna(v)
                },
                "metrics": {
                    k.replace("metric_", ""): v 
                    for k, v in best_run.items() 
                    if k.startswith("metric_") and pd.notna(v)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None

# Global experiment tracker
prsm_experiment_tracker = PRSMExperimentTracker()
```

### 2. Model Training Tracking

```python
# prsm/integrations/mlflow/model_tracking.py
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from prsm.integrations.mlflow.client import mlflow_client

logger = logging.getLogger(__name__)

class PRSMModelTracker:
    """Track model training and evaluation with MLflow."""
    
    def __init__(self):
        self.client = mlflow_client
    
    async def track_model_training(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        experiment_name: str = "model_training",
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Track complete model training process."""
        try:
            # Start MLflow run
            run_id = await self.client.start_run(
                run_name=f"{model_name}_training",
                experiment_name=experiment_name,
                tags={
                    "model_type": type(model).__name__,
                    "model_name": model_name,
                    "training_type": "supervised"
                }
            )
            
            with mlflow.start_run(run_id=run_id):
                # Log hyperparameters
                if hyperparameters:
                    await self.client.log_params(hyperparameters)
                
                # Log model parameters
                if hasattr(model, 'get_params'):
                    model_params = model.get_params()
                    await self.client.log_params({
                        f"model_{k}": v for k, v in model_params.items()
                        if isinstance(v, (str, int, float, bool))
                    })
                
                # Log dataset information
                dataset_params = {
                    "n_train_samples": len(X_train),
                    "n_test_samples": len(X_test),
                    "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 1,
                    "feature_names": feature_names or [],
                    "target_names": target_names or []
                }
                await self.client.log_params(dataset_params)
                
                # Train model (if not already trained)
                if hasattr(model, 'fit') and not hasattr(model, 'is_fitted_'):
                    model.fit(X_train, y_train)
                
                # Log model
                await self.client.log_model(
                    model=model,
                    artifact_path="model",
                    model_type="sklearn"
                )
                
                # Evaluate model
                metrics = await self._evaluate_model(
                    model, X_train, y_train, X_test, y_test
                )
                await self.client.log_metrics(metrics)
                
                # Log artifacts
                artifacts = {
                    "training_summary": {
                        "model_name": model_name,
                        "model_type": type(model).__name__,
                        "training_samples": len(X_train),
                        "test_samples": len(X_test),
                        "features": X_train.shape[1] if len(X_train.shape) > 1 else 1
                    }
                }
                
                if feature_names:
                    # Feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        artifacts["feature_importance"] = feature_importance.to_dict('records')
                
                await self.client.log_artifacts(artifacts, "training")
            
            logger.info(f"Tracked model training: {model_name}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to track model training: {e}")
            raise
    
    async def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        try:
            metrics = {}
            
            # Training predictions
            y_train_pred = model.predict(X_train)
            
            # Test predictions
            y_test_pred = model.predict(X_test)
            
            # Classification metrics
            if len(np.unique(y_train)) <= 10:  # Likely classification
                # Training metrics
                metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
                metrics["train_precision"] = precision_score(
                    y_train, y_train_pred, average='weighted', zero_division=0
                )
                metrics["train_recall"] = recall_score(
                    y_train, y_train_pred, average='weighted', zero_division=0
                )
                metrics["train_f1"] = f1_score(
                    y_train, y_train_pred, average='weighted', zero_division=0
                )
                
                # Test metrics
                metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)
                metrics["test_precision"] = precision_score(
                    y_test, y_test_pred, average='weighted', zero_division=0
                )
                metrics["test_recall"] = recall_score(
                    y_test, y_test_pred, average='weighted', zero_division=0
                )
                metrics["test_f1"] = f1_score(
                    y_test, y_test_pred, average='weighted', zero_division=0
                )
                
                # Cross-validation score
                if hasattr(model, 'fit'):
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    metrics["cv_accuracy_mean"] = cv_scores.mean()
                    metrics["cv_accuracy_std"] = cv_scores.std()
            
            else:  # Regression metrics
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                
                # Training metrics
                metrics["train_mse"] = mean_squared_error(y_train, y_train_pred)
                metrics["train_rmse"] = np.sqrt(metrics["train_mse"])
                metrics["train_mae"] = mean_absolute_error(y_train, y_train_pred)
                metrics["train_r2"] = r2_score(y_train, y_train_pred)
                
                # Test metrics
                metrics["test_mse"] = mean_squared_error(y_test, y_test_pred)
                metrics["test_rmse"] = np.sqrt(metrics["test_mse"])
                metrics["test_mae"] = mean_absolute_error(y_test, y_test_pred)
                metrics["test_r2"] = r2_score(y_test, y_test_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
    
    async def track_hyperparameter_tuning(
        self,
        model_class: type,
        param_grid: Dict[str, List[Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str = "hyperparameter_tuning",
        cv_folds: int = 5,
        scoring: str = "accuracy"
    ) -> Dict[str, Any]:
        """Track hyperparameter tuning process."""
        try:
            from sklearn.model_selection import ParameterGrid, cross_val_score
            
            # Generate parameter combinations
            param_combinations = list(ParameterGrid(param_grid))
            
            best_score = -np.inf if scoring in ['accuracy', 'f1', 'precision', 'recall', 'r2'] else np.inf
            best_params = None
            best_run_id = None
            
            results = []
            
            for i, params in enumerate(param_combinations):
                # Start run for this parameter combination
                run_id = await self.client.start_run(
                    run_name=f"hp_tune_{i}",
                    experiment_name=experiment_name,
                    tags={
                        "tuning_iteration": str(i),
                        "total_combinations": str(len(param_combinations))
                    }
                )
                
                with mlflow.start_run(run_id=run_id):
                    # Create model with current parameters
                    model = model_class(**params)
                    
                    # Log parameters
                    await self.client.log_params(params)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds, scoring=scoring
                    )
                    
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Train on full training set and evaluate
                    model.fit(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    
                    # Log metrics
                    metrics = {
                        f"cv_{scoring}_mean": cv_mean,
                        f"cv_{scoring}_std": cv_std,
                        f"test_{scoring}": test_score
                    }
                    await self.client.log_metrics(metrics)
                    
                    # Track best model
                    is_better = (
                        (scoring in ['accuracy', 'f1', 'precision', 'recall', 'r2'] and cv_mean > best_score) or
                        (scoring in ['neg_mean_squared_error', 'neg_mean_absolute_error'] and cv_mean > best_score)
                    )
                    
                    if is_better:
                        best_score = cv_mean
                        best_params = params
                        best_run_id = run_id
                        
                        # Log best model
                        await self.client.log_model(
                            model=model,
                            artifact_path="best_model",
                            model_type="sklearn"
                        )
                    
                    results.append({
                        "run_id": run_id,
                        "params": params,
                        "cv_score": cv_mean,
                        "test_score": test_score
                    })
                
                await self.client.end_run()
            
            # Log summary of hyperparameter tuning
            summary_run_id = await self.client.start_run(
                run_name="hp_tuning_summary",
                experiment_name=experiment_name,
                tags={"summary": "true"}
            )
            
            with mlflow.start_run(run_id=summary_run_id):
                summary_metrics = {
                    "best_cv_score": best_score,
                    "total_combinations_tested": len(param_combinations),
                    "best_run_id": best_run_id
                }
                await self.client.log_metrics(summary_metrics)
                await self.client.log_params(best_params)
                
                # Log results summary
                results_df = pd.DataFrame(results)
                artifacts = {
                    "tuning_results": results,
                    "best_parameters": best_params,
                    "parameter_grid": param_grid
                }
                await self.client.log_artifacts(artifacts, "tuning_summary")
            
            await self.client.end_run()
            
            return {
                "best_params": best_params,
                "best_score": best_score,
                "best_run_id": best_run_id,
                "all_results": results
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning tracking failed: {e}")
            raise

# Global model tracker
prsm_model_tracker = PRSMModelTracker()
```

## 🚀 Model Deployment

### 1. Model Serving Integration

```python
# prsm/integrations/mlflow/deployment.py
import asyncio
import logging
import json
import requests
from typing import Dict, Any, List, Optional, Union
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from prsm.integrations.mlflow.client import mlflow_client

logger = logging.getLogger(__name__)

class PRSMModelDeployment:
    """Handle model deployment with MLflow."""
    
    def __init__(self):
        self.client = mlflow_client
        self.deployed_models = {}
    
    async def deploy_model_local(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        stage: str = "Production",
        port: int = 5001
    ) -> Dict[str, Any]:
        """Deploy model locally using MLflow serving."""
        try:
            # Get model URI
            if model_version:
                model_uri = f"models:/{model_name}/{model_version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            # Start MLflow model server
            import subprocess
            import threading
            
            cmd = [
                "mlflow", "models", "serve",
                "-m", model_uri,
                "-p", str(port),
                "--no-conda"
            ]
            
            # Start server in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            await asyncio.sleep(5)
            
            # Test if server is running
            try:
                response = requests.get(f"http://localhost:{port}/ping", timeout=5)
                if response.status_code == 200:
                    deployment_info = {
                        "model_name": model_name,
                        "model_uri": model_uri,
                        "endpoint": f"http://localhost:{port}",
                        "status": "running",
                        "process_id": process.pid
                    }
                    
                    self.deployed_models[model_name] = deployment_info
                    logger.info(f"Model {model_name} deployed locally on port {port}")
                    return deployment_info
                else:
                    raise Exception("Server not responding")
                    
            except requests.RequestException:
                process.terminate()
                raise Exception("Failed to start model server")
                
        except Exception as e:
            logger.error(f"Local model deployment failed: {e}")
            raise
    
    async def predict_with_deployed_model(
        self,
        model_name: str,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]], np.ndarray, pd.DataFrame],
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make predictions using deployed model."""
        try:
            # Get endpoint
            if not endpoint:
                if model_name not in self.deployed_models:
                    raise ValueError(f"Model {model_name} not deployed locally")
                endpoint = self.deployed_models[model_name]["endpoint"]
            
            # Prepare input data
            if isinstance(input_data, (np.ndarray, pd.DataFrame)):
                if isinstance(input_data, pd.DataFrame):
                    data = input_data.to_dict('split')
                else:
                    data = {"instances": input_data.tolist()}
            elif isinstance(input_data, dict):
                data = {"instances": [input_data]}
            else:
                data = {"instances": input_data}
            
            # Make prediction request
            response = requests.post(
                f"{endpoint}/invocations",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                predictions = response.json()
                return {
                    "predictions": predictions,
                    "model_name": model_name,
                    "status": "success"
                }
            else:
                raise Exception(f"Prediction failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Prediction with deployed model failed: {e}")
            raise
    
    async def load_model_for_inference(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        stage: str = "Production"
    ) -> Any:
        """Load model directly for inference."""
        try:
            # Get model URI
            if model_version:
                model_uri = f"models:/{model_name}/{model_version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            # Load model
            model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Loaded model {model_name} for inference")
            return model
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    async def create_model_endpoint(
        self,
        model_name: str,
        model_version: str,
        endpoint_name: str,
        cloud_provider: str = "aws",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create cloud model endpoint (placeholder for cloud deployment)."""
        try:
            # This is a placeholder for actual cloud deployment
            # In practice, you would integrate with AWS SageMaker, Azure ML, or GCP AI Platform
            
            deployment_config = {
                "model_name": model_name,
                "model_version": model_version,
                "endpoint_name": endpoint_name,
                "cloud_provider": cloud_provider,
                "config": config or {},
                "status": "creating"
            }
            
            # Simulate deployment process
            await asyncio.sleep(2)
            deployment_config["status"] = "deployed"
            deployment_config["endpoint_url"] = f"https://{endpoint_name}.{cloud_provider}.com/predict"
            
            logger.info(f"Created model endpoint: {endpoint_name}")
            return deployment_config
            
        except Exception as e:
            logger.error(f"Model endpoint creation failed: {e}")
            raise
    
    async def batch_predict(
        self,
        model_name: str,
        input_data: pd.DataFrame,
        output_path: str,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Perform batch predictions."""
        try:
            # Load model
            model = await self.load_model_for_inference(model_name)
            
            # Process in batches
            predictions = []
            total_batches = len(input_data) // batch_size + (1 if len(input_data) % batch_size > 0 else 0)
            
            for i in range(0, len(input_data), batch_size):
                batch_data = input_data.iloc[i:i+batch_size]
                batch_predictions = model.predict(batch_data)
                predictions.extend(batch_predictions)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{total_batches}")
            
            # Save predictions
            results_df = input_data.copy()
            results_df['predictions'] = predictions
            results_df.to_csv(output_path, index=False)
            
            return {
                "total_predictions": len(predictions),
                "output_path": output_path,
                "batches_processed": total_batches
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    async def monitor_model_performance(
        self,
        model_name: str,
        prediction_data: pd.DataFrame,
        ground_truth: Optional[pd.Series] = None,
        time_window: str = "1d"
    ) -> Dict[str, Any]:
        """Monitor deployed model performance."""
        try:
            metrics = {
                "total_predictions": len(prediction_data),
                "time_window": time_window,
                "model_name": model_name
            }
            
            # Basic prediction statistics
            if 'predictions' in prediction_data.columns:
                pred_series = prediction_data['predictions']
                metrics.update({
                    "prediction_mean": float(pred_series.mean()) if pred_series.dtype in ['int64', 'float64'] else None,
                    "prediction_std": float(pred_series.std()) if pred_series.dtype in ['int64', 'float64'] else None,
                    "unique_predictions": int(pred_series.nunique())
                })
            
            # Performance metrics if ground truth available
            if ground_truth is not None and 'predictions' in prediction_data.columns:
                from sklearn.metrics import accuracy_score, mean_squared_error
                
                predictions = prediction_data['predictions']
                
                # Classification or regression metrics
                if ground_truth.dtype == 'object' or ground_truth.nunique() <= 10:
                    # Classification
                    metrics["accuracy"] = float(accuracy_score(ground_truth, predictions))
                else:
                    # Regression
                    metrics["mse"] = float(mean_squared_error(ground_truth, predictions))
                    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            
            # Data drift detection (basic version)
            if hasattr(self, '_reference_data'):
                # Compare current data distribution with reference
                drift_score = self._calculate_drift_score(
                    prediction_data,
                    self._reference_data
                )
                metrics["drift_score"] = drift_score
            
            # Log monitoring metrics to MLflow
            run_id = await self.client.start_run(
                run_name=f"monitoring_{model_name}",
                experiment_name="model_monitoring",
                tags={
                    "model_name": model_name,
                    "monitoring_type": "performance",
                    "time_window": time_window
                }
            )
            
            with mlflow.start_run(run_id=run_id):
                await self.client.log_metrics(metrics)
                
                # Log monitoring artifacts
                artifacts = {
                    "monitoring_summary": metrics,
                    "prediction_stats": prediction_data.describe().to_dict() if not prediction_data.empty else {}
                }
                await self.client.log_artifacts(artifacts, "monitoring")
            
            await self.client.end_run()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model performance monitoring failed: {e}")
            return {}
    
    def _calculate_drift_score(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> float:
        """Calculate data drift score between current and reference data."""
        try:
            # Simple drift calculation using statistical distance
            # In practice, you might use more sophisticated methods
            
            drift_scores = []
            
            for column in current_data.select_dtypes(include=[np.number]).columns:
                if column in reference_data.columns:
                    # Kolmogorov-Smirnov test statistic as drift measure
                    from scipy.stats import ks_2samp
                    
                    current_values = current_data[column].dropna()
                    reference_values = reference_data[column].dropna()
                    
                    if len(current_values) > 0 and len(reference_values) > 0:
                        ks_stat, _ = ks_2samp(current_values, reference_values)
                        drift_scores.append(ks_stat)
            
            return float(np.mean(drift_scores)) if drift_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Drift calculation failed: {e}")
            return 0.0

# Global model deployment manager
prsm_model_deployment = PRSMModelDeployment()
```

## 📊 Analytics and Reporting

### MLflow Analytics Dashboard

```python
# prsm/integrations/mlflow/analytics.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from prsm.integrations.mlflow.client import mlflow_client

logger = logging.getLogger(__name__)

class MLflowAnalytics:
    """Analytics and reporting for MLflow experiments."""
    
    def __init__(self):
        self.client = mlflow_client
    
    async def generate_experiment_report(
        self,
        experiment_name: str,
        time_period: Optional[int] = 30  # days
    ) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment {experiment_name} not found")
            
            # Get time filter
            if time_period:
                start_time = int((datetime.now() - timedelta(days=time_period)).timestamp() * 1000)
                filter_string = f"attributes.start_time >= {start_time}"
            else:
                filter_string = None
            
            # Get runs
            runs = await self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=1000
            )
            
            if not runs:
                return {"message": "No runs found for the specified criteria"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(runs)
            
            # Basic statistics
            total_runs = len(runs)
            successful_runs = len([r for r in runs if r["status"] == "FINISHED"])
            failed_runs = total_runs - successful_runs
            
            # Time analysis
            start_times = pd.to_datetime([r["start_time"] for r in runs], unit="ms")
            duration_data = []
            for run in runs:
                if run["end_time"] and run["start_time"]:
                    duration = (run["end_time"] - run["start_time"]) / 1000  # seconds
                    duration_data.append(duration)
            
            # Metrics analysis
            all_metrics = {}
            for run in runs:
                for metric, value in run["metrics"].items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
            
            metrics_stats = {}
            for metric, values in all_metrics.items():
                if values:
                    metrics_stats[metric] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values)
                    }
            
            # Parameter analysis
            all_params = {}
            for run in runs:
                for param, value in run["params"].items():
                    if param not in all_params:
                        all_params[param] = []
                    all_params[param].append(value)
            
            param_stats = {}
            for param, values in all_params.items():
                unique_values = list(set(values))
                param_stats[param] = {
                    "unique_values": len(unique_values),
                    "values": unique_values[:10],  # First 10 unique values
                    "most_common": max(set(values), key=values.count) if values else None
                }
            
            report = {
                "experiment_name": experiment_name,
                "experiment_id": experiment.experiment_id,
                "report_period": f"Last {time_period} days" if time_period else "All time",
                "summary": {
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "failed_runs": failed_runs,
                    "success_rate": successful_runs / total_runs * 100 if total_runs > 0 else 0,
                    "avg_duration_seconds": np.mean(duration_data) if duration_data else 0,
                    "date_range": {
                        "start": start_times.min().isoformat() if len(start_times) > 0 else None,
                        "end": start_times.max().isoformat() if len(start_times) > 0 else None
                    }
                },
                "metrics_analysis": metrics_stats,
                "parameter_analysis": param_stats,
                "top_performing_runs": await self._get_top_performing_runs(runs, limit=5)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Experiment report generation failed: {e}")
            return {"error": str(e)}
    
    async def _get_top_performing_runs(
        self,
        runs: List[Dict[str, Any]],
        metric_name: str = "test_accuracy",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top performing runs based on a metric."""
        try:
            # Filter runs that have the specified metric
            runs_with_metric = [
                run for run in runs 
                if metric_name in run["metrics"]
            ]
            
            if not runs_with_metric:
                # Try alternative common metrics
                alternative_metrics = ["accuracy", "f1_score", "test_f1", "cv_accuracy_mean"]
                for alt_metric in alternative_metrics:
                    runs_with_metric = [
                        run for run in runs 
                        if alt_metric in run["metrics"]
                    ]
                    if runs_with_metric:
                        metric_name = alt_metric
                        break
            
            if not runs_with_metric:
                return []
            
            # Sort by metric value
            sorted_runs = sorted(
                runs_with_metric,
                key=lambda x: x["metrics"][metric_name],
                reverse=True
            )
            
            # Return top runs
            top_runs = []
            for run in sorted_runs[:limit]:
                top_runs.append({
                    "run_id": run["run_id"],
                    "metric_value": run["metrics"][metric_name],
                    "metric_name": metric_name,
                    "parameters": run["params"],
                    "start_time": pd.to_datetime(run["start_time"], unit="ms").isoformat()
                })
            
            return top_runs
            
        except Exception as e:
            logger.error(f"Failed to get top performing runs: {e}")
            return []
    
    async def compare_models_across_experiments(
        self,
        experiment_names: List[str],
        metric_name: str = "test_accuracy"
    ) -> Dict[str, Any]:
        """Compare model performance across multiple experiments."""
        try:
            comparison_data = []
            
            for exp_name in experiment_names:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if not experiment:
                    continue
                
                runs = await self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=100
                )
                
                for run in runs:
                    if metric_name in run["metrics"]:
                        comparison_data.append({
                            "experiment": exp_name,
                            "run_id": run["run_id"],
                            "metric_value": run["metrics"][metric_name],
                            "status": run["status"],
                            "start_time": pd.to_datetime(run["start_time"], unit="ms")
                        })
            
            if not comparison_data:
                return {"message": f"No runs found with metric {metric_name}"}
            
            df = pd.DataFrame(comparison_data)
            
            # Calculate statistics per experiment
            experiment_stats = df.groupby("experiment")[metric_name].agg([
                "count", "mean", "std", "min", "max", "median"
            ]).round(4)
            
            # Find best run overall
            best_run = df.loc[df[metric_name].idxmax()]
            
            return {
                "metric_name": metric_name,
                "total_runs": len(comparison_data),
                "experiments_compared": experiment_names,
                "experiment_statistics": experiment_stats.to_dict(),
                "best_run": {
                    "experiment": best_run["experiment"],
                    "run_id": best_run["run_id"],
                    "metric_value": best_run["metric_value"]
                },
                "comparison_data": comparison_data
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {"error": str(e)}
    
    async def generate_model_registry_report(self) -> Dict[str, Any]:
        """Generate report on model registry status."""
        try:
            # Get all registered models
            registered_models = self.client.client.search_registered_models()
            
            if not registered_models:
                return {"message": "No registered models found"}
            
            models_info = []
            
            for model in registered_models:
                # Get latest versions for each stage
                versions_by_stage = {}
                for version in model.latest_versions:
                    versions_by_stage[version.current_stage] = {
                        "version": version.version,
                        "run_id": version.run_id,
                        "creation_timestamp": pd.to_datetime(
                            version.creation_timestamp, unit="ms"
                        ).isoformat(),
                        "description": version.description
                    }
                
                models_info.append({
                    "name": model.name,
                    "description": model.description,
                    "creation_timestamp": pd.to_datetime(
                        model.creation_timestamp, unit="ms"
                    ).isoformat(),
                    "last_updated_timestamp": pd.to_datetime(
                        model.last_updated_timestamp, unit="ms"
                    ).isoformat(),
                    "versions_by_stage": versions_by_stage,
                    "total_versions": len(model.latest_versions),
                    "tags": model.tags
                })
            
            return {
                "total_registered_models": len(registered_models),
                "models": models_info,
                "stages_summary": self._summarize_model_stages(models_info)
            }
            
        except Exception as e:
            logger.error(f"Model registry report generation failed: {e}")
            return {"error": str(e)}
    
    def _summarize_model_stages(self, models_info: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize model counts by stage."""
        stage_counts = {}
        
        for model in models_info:
            for stage in model["versions_by_stage"].keys():
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return stage_counts

# Global analytics instance
mlflow_analytics = MLflowAnalytics()
```

## 📋 FastAPI Integration

### MLflow API Endpoints

```python
# prsm/api/mlflow_endpoints.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import pandas as pd

from prsm.integrations.mlflow.client import mlflow_client
from prsm.integrations.mlflow.experiment_tracking import prsm_experiment_tracker
from prsm.integrations.mlflow.model_tracking import prsm_model_tracker
from prsm.integrations.mlflow.deployment import prsm_model_deployment
from prsm.integrations.mlflow.analytics import mlflow_analytics

router = APIRouter(prefix="/api/v1/mlflow", tags=["MLflow"])

class ExperimentRequest(BaseModel):
    experiment_name: str
    query_request: Dict[str, Any]
    model_config: Dict[str, Any]
    tags: Optional[Dict[str, str]] = None

class ModelTrainingRequest(BaseModel):
    model_name: str
    experiment_name: str
    hyperparameters: Optional[Dict[str, Any]] = None

class ModelDeploymentRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    stage: str = "Production"
    deployment_type: str = "local"
    port: int = 5001

@router.post("/experiments/start")
async def start_experiment(request: ExperimentRequest):
    """Start MLflow experiment tracking."""
    try:
        # Convert dict to QueryRequest (you might need to adjust this)
        from prsm.models.query import QueryRequest
        query_request = QueryRequest(**request.query_request)
        
        run_id = await prsm_experiment_tracker.start_query_experiment(
            experiment_name=request.experiment_name,
            query_request=query_request,
            model_config=request.model_config,
            tags=request.tags
        )
        
        return {"run_id": run_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{run_id}/log-response")
async def log_query_response(
    run_id: str,
    response_data: Dict[str, Any],
    execution_metrics: Dict[str, float]
):
    """Log query response to experiment."""
    try:
        from prsm.models.query import QueryResponse
        query_response = QueryResponse(**response_data)
        
        await prsm_experiment_tracker.log_query_response(
            run_id=run_id,
            query_response=query_response,
            execution_metrics=execution_metrics
        )
        
        return {"status": "logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/{run_id}/end")
async def end_experiment(
    run_id: str,
    status: str = "FINISHED",
    final_metrics: Optional[Dict[str, float]] = None
):
    """End experiment tracking."""
    try:
        await prsm_experiment_tracker.end_query_experiment(
            run_id=run_id,
            status=status,
            final_metrics=final_metrics
        )
        
        return {"status": "ended"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_name}/compare")
async def compare_experiments(
    experiment_name: str,
    metric_names: Optional[List[str]] = Query(None),
    max_results: int = 50
):
    """Compare experiments and get results."""
    try:
        df = await prsm_experiment_tracker.compare_experiments(
            experiment_name=experiment_name,
            metric_names=metric_names,
            max_results=max_results
        )
        
        if df.empty:
            return {"message": "No experiments found"}
        
        return {
            "experiment_name": experiment_name,
            "total_runs": len(df),
            "results": df.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/deploy")
async def deploy_model(request: ModelDeploymentRequest):
    """Deploy model for serving."""
    try:
        if request.deployment_type == "local":
            result = await prsm_model_deployment.deploy_model_local(
                model_name=request.model_name,
                model_version=request.model_version,
                stage=request.stage,
                port=request.port
            )
        else:
            raise ValueError(f"Deployment type {request.deployment_type} not supported")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_name}/predict")
async def predict_with_model(
    model_name: str,
    input_data: Dict[str, Any],
    endpoint: Optional[str] = None
):
    """Make predictions with deployed model."""
    try:
        result = await prsm_model_deployment.predict_with_deployed_model(
            model_name=model_name,
            input_data=input_data,
            endpoint=endpoint
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/experiment-report/{experiment_name}")
async def get_experiment_report(
    experiment_name: str,
    time_period: Optional[int] = Query(30, description="Time period in days")
):
    """Get comprehensive experiment report."""
    try:
        report = await mlflow_analytics.generate_experiment_report(
            experiment_name=experiment_name,
            time_period=time_period
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/model-registry-report")
async def get_model_registry_report():
    """Get model registry status report."""
    try:
        report = await mlflow_analytics.generate_model_registry_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/list")
async def list_registered_models():
    """List all registered models."""
    try:
        models = mlflow_client.client.search_registered_models()
        
        model_list = []
        for model in models:
            model_list.append({
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id
                    }
                    for v in model.latest_versions
                ]
            })
        
        return {"models": model_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/list")
async def list_experiments():
    """List all experiments."""
    try:
        experiments = await mlflow_client.search_experiments()
        return {"experiments": experiments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

**Need help with MLflow integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).