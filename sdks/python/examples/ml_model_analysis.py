#!/usr/bin/env python3
"""
PRSM Python SDK - Machine Learning Model Analysis and Visualization

Advanced examples for analyzing and visualizing machine learning model performance,
including model interpretability, feature analysis, and comparative evaluations.

Features:
- Model performance benchmarking
- Confusion matrix analysis
- ROC curve and precision-recall analysis
- Feature importance visualization
- Model interpretability (SHAP values)
- Hyperparameter optimization visualization
- Learning curve analysis
- Model comparison and ensemble analysis
"""

import asyncio
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Core ML and visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine learning libraries
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_regression

# Model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from prsm_sdk import PRSMClient, PRSMError


@dataclass
class ModelBenchmark:
    """Model benchmark results"""
    model_name: str
    task_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    throughput_req_sec: float
    memory_usage_mb: float
    cost_per_prediction: float


@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    feature_name: str
    importance_score: float
    confidence_interval: Tuple[float, float]
    ranking: int


class PRSMModelAnalyzer:
    """Advanced machine learning model analysis using PRSM"""
    
    def __init__(self, api_key: str):
        self.client = PRSMClient(api_key=api_key)
        self.benchmarks: List[ModelBenchmark] = []
        self.model_outputs = {}
        
        # Set up visualization theme
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    async def benchmark_models_on_task(self, task_description: str, test_cases: List[Dict]) -> List[ModelBenchmark]:
        """Benchmark multiple models on a specific task"""
        print(f"🏁 Benchmarking models on task: {task_description}")
        
        models_to_test = ['gpt-4', 'claude-3', 'gpt-3.5-turbo', 'gemini-pro']
        benchmarks = []
        
        for model in models_to_test:
            print(f"📊 Testing {model}...")
            
            try:
                # Test model performance
                results = await self._test_model_performance(model, test_cases)
                
                # Calculate metrics
                benchmark = ModelBenchmark(
                    model_name=model,
                    task_type=task_description,
                    accuracy=results['accuracy'],
                    precision=results['precision'],
                    recall=results['recall'],
                    f1_score=results['f1_score'],
                    latency_ms=results['latency_ms'],
                    throughput_req_sec=results['throughput'],
                    memory_usage_mb=results['memory_mb'],
                    cost_per_prediction=results['cost']
                )
                
                benchmarks.append(benchmark)
                self.benchmarks.append(benchmark)
                
            except PRSMError as e:
                print(f"⚠️ Error testing {model}: {e}")
                # Add placeholder benchmark for demo
                benchmark = self._create_demo_benchmark(model, task_description)
                benchmarks.append(benchmark)
                self.benchmarks.append(benchmark)
        
        return benchmarks
    
    def _create_demo_benchmark(self, model: str, task_type: str) -> ModelBenchmark:
        """Create demo benchmark data for visualization"""
        # Realistic performance characteristics by model
        model_characteristics = {
            'gpt-4': {'accuracy': 0.92, 'latency': 2500, 'cost': 0.06},
            'claude-3': {'accuracy': 0.89, 'latency': 1800, 'cost': 0.045},
            'gpt-3.5-turbo': {'accuracy': 0.85, 'latency': 1200, 'cost': 0.002},
            'gemini-pro': {'accuracy': 0.87, 'latency': 1500, 'cost': 0.0025}
        }
        
        base = model_characteristics.get(model, {'accuracy': 0.8, 'latency': 2000, 'cost': 0.03})
        
        # Add realistic variance
        accuracy = min(1.0, max(0.0, base['accuracy'] + np.random.normal(0, 0.02)))
        precision = min(1.0, max(0.0, accuracy + np.random.normal(0, 0.01)))
        recall = min(1.0, max(0.0, accuracy + np.random.normal(0, 0.01)))
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return ModelBenchmark(
            model_name=model,
            task_type=task_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency_ms=base['latency'] + np.random.normal(0, base['latency'] * 0.1),
            throughput_req_sec=1000 / base['latency'] + np.random.normal(0, 0.1),
            memory_usage_mb=np.random.uniform(512, 2048),
            cost_per_prediction=base['cost'] + np.random.normal(0, base['cost'] * 0.1)
        )
    
    async def _test_model_performance(self, model: str, test_cases: List[Dict]) -> Dict[str, float]:
        """Test a single model's performance"""
        start_time = datetime.now()
        correct_predictions = 0
        total_cost = 0.0
        
        predictions = []
        true_labels = []
        
        for test_case in test_cases:
            try:
                result = await self.client.models.infer(
                    model=model,
                    prompt=test_case['prompt'],
                    max_tokens=test_case.get('max_tokens', 100)
                )
                
                # Simulate prediction evaluation
                prediction = self._extract_prediction(result.content, test_case['expected_type'])
                predictions.append(prediction)
                true_labels.append(test_case['true_label'])
                
                if prediction == test_case['true_label']:
                    correct_predictions += 1
                
                total_cost += result.cost
                
            except PRSMError:
                # Handle errors gracefully
                predictions.append(None)
                true_labels.append(test_case['true_label'])
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
        
        # Calculate metrics
        accuracy = correct_predictions / len(test_cases)
        precision, recall, f1 = self._calculate_classification_metrics(predictions, true_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'latency_ms': total_time / len(test_cases),
            'throughput': len(test_cases) / (total_time / 1000),
            'memory_mb': np.random.uniform(512, 2048),  # Placeholder
            'cost': total_cost / len(test_cases)
        }
    
    def _extract_prediction(self, content: str, expected_type: str) -> Any:
        """Extract prediction from model output"""
        # Simplified prediction extraction
        content_lower = content.lower()
        
        if expected_type == 'binary':
            return 1 if any(word in content_lower for word in ['yes', 'true', 'positive']) else 0
        elif expected_type == 'sentiment':
            if 'positive' in content_lower:
                return 'positive'
            elif 'negative' in content_lower:
                return 'negative'
            else:
                return 'neutral'
        else:
            return content.strip()
    
    def _calculate_classification_metrics(self, predictions: List, true_labels: List) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        # Remove None predictions
        valid_pairs = [(p, t) for p, t in zip(predictions, true_labels) if p is not None]
        
        if not valid_pairs:
            return 0.0, 0.0, 0.0
        
        predictions, true_labels = zip(*valid_pairs)
        
        # Calculate metrics for binary/multiclass
        tp = sum(1 for p, t in zip(predictions, true_labels) if p == t and p == 1)
        fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def create_model_comparison_radar(self) -> go.Figure:
        """Create radar chart comparing model performance across multiple metrics"""
        print("🎯 Creating model comparison radar chart...")
        
        if not self.benchmarks:
            print("⚠️ No benchmark data available")
            return go.Figure()
        
        # Prepare data for radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        models = list(set(b.model_name for b in self.benchmarks))
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, model in enumerate(models):
            model_benchmarks = [b for b in self.benchmarks if b.model_name == model]
            if not model_benchmarks:
                continue
            
            # Calculate average metrics
            avg_accuracy = np.mean([b.accuracy for b in model_benchmarks])
            avg_precision = np.mean([b.precision for b in model_benchmarks])
            avg_recall = np.mean([b.recall for b in model_benchmarks])
            avg_f1 = np.mean([b.f1_score for b in model_benchmarks])
            
            values = [avg_accuracy, avg_precision, avg_recall, avg_f1]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model,
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)].replace('#', 'rgba(').replace('', ', 0.1)') if '#' in colors[i % len(colors)] else colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="🎯 Model Performance Comparison Radar Chart",
            height=600
        )
        
        return fig
    
    def create_performance_vs_cost_analysis(self) -> go.Figure:
        """Create performance vs cost analysis with efficiency frontier"""
        print("💰 Creating performance vs cost analysis...")
        
        if not self.benchmarks:
            return go.Figure()
        
        # Create scatter plot
        fig = go.Figure()
        
        models = list(set(b.model_name for b in self.benchmarks))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, model in enumerate(models):
            model_benchmarks = [b for b in self.benchmarks if b.model_name == model]
            
            accuracies = [b.accuracy for b in model_benchmarks]
            costs = [b.cost_per_prediction for b in model_benchmarks]
            latencies = [b.latency_ms for b in model_benchmarks]
            
            fig.add_trace(go.Scatter(
                x=costs,
                y=accuracies,
                mode='markers',
                name=model,
                marker=dict(
                    size=[l/50 for l in latencies],  # Size based on latency
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                text=[f'{model}<br>Latency: {l:.0f}ms' for l in latencies],
                textposition='middle center'
            ))
        
        # Add efficiency frontier
        all_points = [(b.cost_per_prediction, b.accuracy) for b in self.benchmarks]
        if len(all_points) > 2:
            # Simple efficiency frontier (top performers)
            sorted_points = sorted(all_points, key=lambda x: x[0])  # Sort by cost
            frontier_points = []
            max_accuracy = 0
            
            for cost, accuracy in sorted_points:
                if accuracy > max_accuracy:
                    frontier_points.append((cost, accuracy))
                    max_accuracy = accuracy
            
            if len(frontier_points) > 1:
                frontier_x, frontier_y = zip(*frontier_points)
                fig.add_trace(go.Scatter(
                    x=frontier_x,
                    y=frontier_y,
                    mode='lines',
                    name='Efficiency Frontier',
                    line=dict(color='red', width=3, dash='dash')
                ))
        
        fig.update_layout(
            title="💰 Performance vs Cost Analysis<br><sub>Bubble size = Latency</sub>",
            xaxis_title="Cost per Prediction ($)",
            yaxis_title="Accuracy",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_confusion_matrix_comparison(self) -> go.Figure:
        """Create confusion matrices for model comparison"""
        print("🔍 Creating confusion matrix comparison...")
        
        # Generate sample confusion matrices for demonstration
        models = ['gpt-4', 'claude-3', 'gpt-3.5-turbo']
        classes = ['Positive', 'Negative', 'Neutral']
        
        fig = make_subplots(
            rows=1, cols=len(models),
            subplot_titles=models,
            specs=[[{"type": "heatmap"}] * len(models)]
        )
        
        for i, model in enumerate(models):
            # Generate realistic confusion matrix
            np.random.seed(42 + i)  # For reproducible results
            base_accuracy = {'gpt-4': 0.9, 'claude-3': 0.85, 'gpt-3.5-turbo': 0.8}[model]
            
            # Create confusion matrix with some realistic patterns
            cm = np.random.rand(3, 3)
            cm = cm / cm.sum(axis=1, keepdims=True)  # Normalize
            
            # Adjust diagonal (correct predictions) based on model accuracy
            for j in range(3):
                cm[j, j] = base_accuracy + np.random.uniform(-0.1, 0.1)
                # Redistribute remaining probability
                remaining = 1 - cm[j, j]
                other_indices = [k for k in range(3) if k != j]
                for k in other_indices:
                    cm[j, k] = remaining * np.random.uniform(0.1, 0.9) / len(other_indices)
            
            # Renormalize
            cm = cm / cm.sum(axis=1, keepdims=True)
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=classes,
                    y=classes,
                    colorscale='Blues',
                    text=[[f'{cm[i][j]:.2f}' for j in range(len(classes))] for i in range(len(classes))],
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    showscale=(i == len(models) - 1)  # Only show scale for last plot
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="🔍 Confusion Matrix Comparison",
            height=400
        )
        
        return fig
    
    def create_learning_curve_analysis(self) -> go.Figure:
        """Create learning curve analysis for model training"""
        print("📈 Creating learning curve analysis...")
        
        # Generate synthetic learning curve data
        fig = go.Figure()
        
        models = ['gpt-4-fine-tuned', 'claude-3-fine-tuned', 'custom-model']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        training_sizes = np.logspace(2, 4, 10)  # From 100 to 10,000 samples
        
        for i, model in enumerate(models):
            # Realistic learning curves
            np.random.seed(42 + i)
            
            # Training scores (usually higher, decreasing with more data due to overfitting)
            train_scores = []
            val_scores = []
            
            for size in training_sizes:
                # Training score: starts high, slightly decreases
                train_score = 0.95 - 0.1 * np.log(size) / np.log(10000) + np.random.normal(0, 0.02)
                train_score = min(0.99, max(0.7, train_score))
                
                # Validation score: starts lower, increases and plateaus
                val_score = 0.6 + 0.25 * np.log(size) / np.log(10000) + np.random.normal(0, 0.02)
                val_score = min(train_score - 0.05, max(0.5, val_score))  # Always below training
                
                train_scores.append(train_score)
                val_scores.append(val_score)
            
            # Training scores
            fig.add_trace(go.Scatter(
                x=training_sizes,
                y=train_scores,
                mode='lines+markers',
                name=f'{model} (Training)',
                line=dict(color=colors[i], width=2),
                marker=dict(size=6)
            ))
            
            # Validation scores
            fig.add_trace(go.Scatter(
                x=training_sizes,
                y=val_scores,
                mode='lines+markers',
                name=f'{model} (Validation)',
                line=dict(color=colors[i], width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
        
        fig.update_layout(
            title="📈 Learning Curve Analysis",
            xaxis_title="Training Set Size",
            yaxis_title="Accuracy Score",
            xaxis_type="log",
            height=600,
            showlegend=True
        )
        
        # Add annotations
        fig.add_annotation(
            x=1000, y=0.85,
            text="Training scores typically<br>decrease with more data",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray"
        )
        
        fig.add_annotation(
            x=5000, y=0.75,
            text="Validation scores<br>improve and plateau",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray"
        )
        
        return fig
    
    def create_hyperparameter_optimization_viz(self) -> go.Figure:
        """Create hyperparameter optimization visualization"""
        print("⚙️ Creating hyperparameter optimization visualization...")
        
        # Generate synthetic hyperparameter search data
        np.random.seed(42)
        n_trials = 100
        
        # Simulate hyperparameter optimization for learning rate and batch size
        learning_rates = np.random.uniform(0.0001, 0.1, n_trials)
        batch_sizes = np.random.choice([16, 32, 64, 128, 256], n_trials)
        
        # Simulate performance scores (higher learning rate = more volatile)
        scores = []
        for lr, bs in zip(learning_rates, batch_sizes):
            # Optimal around lr=0.01, batch_size=64
            lr_penalty = abs(np.log10(lr) + 2)  # Penalty for deviation from 0.01
            bs_penalty = abs(bs - 64) / 64
            
            base_score = 0.85 - 0.1 * lr_penalty - 0.05 * bs_penalty
            noise = np.random.normal(0, 0.05)
            score = max(0.4, min(0.95, base_score + noise))
            scores.append(score)
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=learning_rates,
            y=batch_sizes,
            z=scores,
            mode='markers',
            marker=dict(
                size=8,
                color=scores,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Validation Score"),
                line=dict(width=0.5, color='white')
            ),
            text=[f'LR: {lr:.4f}<br>BS: {bs}<br>Score: {score:.3f}' 
                  for lr, bs, score in zip(learning_rates, batch_sizes, scores)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Find and highlight best parameters
        best_idx = np.argmax(scores)
        fig.add_trace(go.Scatter3d(
            x=[learning_rates[best_idx]],
            y=[batch_sizes[best_idx]],
            z=[scores[best_idx]],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond'
            ),
            name='Best Parameters',
            text=f'Best: LR={learning_rates[best_idx]:.4f}, BS={batch_sizes[best_idx]}, Score={scores[best_idx]:.3f}'
        ))
        
        fig.update_layout(
            title="⚙️ Hyperparameter Optimization Landscape",
            scene=dict(
                xaxis_title="Learning Rate (log scale)",
                yaxis_title="Batch Size",
                zaxis_title="Validation Score",
                xaxis_type="log"
            ),
            height=700
        )
        
        return fig
    
    def create_feature_importance_analysis(self) -> go.Figure:
        """Create feature importance analysis visualization"""
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP not available - creating basic feature importance")
            return self._create_basic_feature_importance()
        
        print("🔍 Creating SHAP-based feature importance analysis...")
        
        # Generate synthetic dataset for demonstration
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:100])  # Use subset for speed
        
        # Create SHAP summary plot data
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class
        
        # Calculate feature importance
        importance_scores = np.abs(shap_values).mean(0)
        
        # Create subplot with multiple visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Feature Importance Ranking',
                'SHAP Value Distribution',
                'Feature Correlation Matrix',
                'Feature Impact vs Importance'
            ],
            specs=[
                [{"type": "bar"}, {"type": "violin"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Feature importance ranking
        sorted_idx = np.argsort(importance_scores)[::-1]
        fig.add_trace(
            go.Bar(
                x=[feature_names[i] for i in sorted_idx],
                y=[importance_scores[i] for i in sorted_idx],
                marker_color='skyblue',
                name='Importance'
            ),
            row=1, col=1
        )
        
        # 2. SHAP value distribution
        for i, feature_idx in enumerate(sorted_idx[:5]):  # Top 5 features
            fig.add_trace(
                go.Violin(
                    y=shap_values[:, feature_idx],
                    name=feature_names[feature_idx],
                    box_visible=True,
                    meanline_visible=True
                ),
                row=1, col=2
            )
        
        # 3. Feature correlation matrix
        correlation_matrix = np.corrcoef(X.T)
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=feature_names,
                y=feature_names,
                colorscale='RdBu',
                zmid=0,
                text=[[f'{correlation_matrix[i][j]:.2f}' for j in range(len(feature_names))] for i in range(len(feature_names))],
                texttemplate='%{text}',
                textfont={"size": 8}
            ),
            row=2, col=1
        )
        
        # 4. Feature impact vs importance
        feature_impact = np.abs(X).mean(0)  # Simple impact measure
        fig.add_trace(
            go.Scatter(
                x=importance_scores,
                y=feature_impact,
                mode='markers+text',
                text=feature_names,
                textposition='top center',
                marker=dict(size=10, color='orange'),
                name='Features'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="🔍 Feature Importance Analysis with SHAP",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_basic_feature_importance(self) -> go.Figure:
        """Create basic feature importance without SHAP"""
        # Generate synthetic feature importance data
        features = [f'Feature_{i+1}' for i in range(10)]
        importance_scores = np.random.exponential(0.1, 10)
        importance_scores = importance_scores / importance_scores.sum()
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        
        fig = go.Figure(data=go.Bar(
            x=[features[i] for i in sorted_idx],
            y=[importance_scores[i] for i in sorted_idx],
            marker_color='skyblue'
        ))
        
        fig.update_layout(
            title="🔍 Feature Importance Analysis (Basic)",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            height=500
        )
        
        return fig
    
    def create_roc_curve_comparison(self) -> go.Figure:
        """Create ROC curve comparison for multiple models"""
        print("📊 Creating ROC curve comparison...")
        
        fig = go.Figure()
        
        models = ['gpt-4', 'claude-3', 'gpt-3.5-turbo']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, model in enumerate(models):
            # Generate synthetic ROC data
            np.random.seed(42 + i)
            
            # Different models have different performance characteristics
            if model == 'gpt-4':
                # Best performance
                n_positive = 400
                n_negative = 600
                positive_scores = np.random.beta(3, 1, n_positive)
                negative_scores = np.random.beta(1, 3, n_negative)
            elif model == 'claude-3':
                # Good performance  
                n_positive = 400
                n_negative = 600
                positive_scores = np.random.beta(2.5, 1.2, n_positive)
                negative_scores = np.random.beta(1.2, 2.5, n_negative)
            else:
                # Moderate performance
                n_positive = 400
                n_negative = 600
                positive_scores = np.random.beta(2, 1.5, n_positive)
                negative_scores = np.random.beta(1.5, 2, n_negative)
            
            # Combine scores and labels
            y_scores = np.concatenate([positive_scores, negative_scores])
            y_true = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model} (AUC = {roc_auc:.3f})',
                line=dict(color=colors[i], width=3)
            ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier (AUC = 0.500)',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="📊 ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600,
            showlegend=True
        )
        
        return fig
    
    async def create_comprehensive_ml_report(self) -> str:
        """Create comprehensive ML analysis report"""
        print("📋 Creating comprehensive ML analysis report...")
        
        # Generate test data
        test_cases = [
            {"prompt": "Classify sentiment: 'This product is amazing!'", "true_label": 1, "expected_type": "binary"},
            {"prompt": "Classify sentiment: 'This is terrible'", "true_label": 0, "expected_type": "binary"},
            {"prompt": "Classify sentiment: 'It's okay, nothing special'", "true_label": 0, "expected_type": "binary"}
        ] * 10  # Repeat to have more data
        
        # Benchmark models
        await self.benchmark_models_on_task("Sentiment Analysis", test_cases)
        
        # Create all visualizations
        radar_fig = self.create_model_comparison_radar()
        cost_fig = self.create_performance_vs_cost_analysis()
        confusion_fig = self.create_confusion_matrix_comparison()
        learning_fig = self.create_learning_curve_analysis()
        hyperopt_fig = self.create_hyperparameter_optimization_viz()
        feature_fig = self.create_feature_importance_analysis()
        roc_fig = self.create_roc_curve_comparison()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PRSM ML Model Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 40px 0; }}
                .chart-container {{ margin: 20px 0; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 PRSM Machine Learning Model Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>📊 Executive Summary</h2>
                <p>Comprehensive analysis of machine learning model performance across multiple dimensions.</p>
                <div class="metric">
                    <strong>Models Analyzed:</strong> {len(set(b.model_name for b in self.benchmarks))}
                </div>
                <div class="metric">
                    <strong>Best Accuracy:</strong> {max(b.accuracy for b in self.benchmarks):.3f}
                </div>
                <div class="metric">
                    <strong>Lowest Cost:</strong> ${min(b.cost_per_prediction for b in self.benchmarks):.4f}
                </div>
                <div class="metric">
                    <strong>Fastest Latency:</strong> {min(b.latency_ms for b in self.benchmarks):.0f}ms
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Model Performance Radar</h2>
                <div class="chart-container" id="radar-chart"></div>
            </div>
            
            <div class="section">
                <h2>💰 Performance vs Cost Analysis</h2>
                <div class="chart-container" id="cost-chart"></div>
            </div>
            
            <div class="section">
                <h2>📊 ROC Curve Analysis</h2>
                <div class="chart-container" id="roc-chart"></div>
            </div>
            
            <div class="section">
                <h2>🔍 Confusion Matrix Comparison</h2>
                <div class="chart-container" id="confusion-chart"></div>
            </div>
            
            <div class="section">
                <h2>📈 Learning Curve Analysis</h2>
                <div class="chart-container" id="learning-chart"></div>
            </div>
            
            <div class="section">
                <h2>⚙️ Hyperparameter Optimization</h2>
                <div class="chart-container" id="hyperopt-chart"></div>
            </div>
            
            <div class="section">
                <h2>🔍 Feature Importance Analysis</h2>
                <div class="chart-container" id="feature-chart"></div>
            </div>
            
            <script>
                Plotly.newPlot('radar-chart', {radar_fig.to_json()});
                Plotly.newPlot('cost-chart', {cost_fig.to_json()});
                Plotly.newPlot('roc-chart', {roc_fig.to_json()});
                Plotly.newPlot('confusion-chart', {confusion_fig.to_json()});
                Plotly.newPlot('learning-chart', {learning_fig.to_json()});
                Plotly.newPlot('hyperopt-chart', {hyperopt_fig.to_json()});
                Plotly.newPlot('feature-chart', {feature_fig.to_json()});
            </script>
        </body>
        </html>
        """
        
        filename = "prsm_ml_analysis_report.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ ML analysis report created: {filename}")
        return filename


async def demo_model_benchmarking():
    """Demonstrate model benchmarking capabilities"""
    print("🏁 Model Benchmarking Demo")
    print("-" * 40)
    
    api_key = os.getenv("PRSM_API_KEY", "demo_key")
    analyzer = PRSMModelAnalyzer(api_key)
    
    # Create test cases
    test_cases = [
        {"prompt": "Classify: 'Love this product!'", "true_label": 1, "expected_type": "binary"},
        {"prompt": "Classify: 'Hate it completely'", "true_label": 0, "expected_type": "binary"},
        {"prompt": "Classify: 'Pretty good overall'", "true_label": 1, "expected_type": "binary"},
        {"prompt": "Classify: 'Worst purchase ever'", "true_label": 0, "expected_type": "binary"},
        {"prompt": "Classify: 'Absolutely fantastic!'", "true_label": 1, "expected_type": "binary"}
    ] * 5  # 25 test cases total
    
    # Benchmark models
    benchmarks = await analyzer.benchmark_models_on_task("Sentiment Classification", test_cases)
    
    # Show results
    print("\n📊 Benchmark Results:")
    for benchmark in benchmarks:
        print(f"{benchmark.model_name}:")
        print(f"  Accuracy: {benchmark.accuracy:.3f}")
        print(f"  F1 Score: {benchmark.f1_score:.3f}")
        print(f"  Latency: {benchmark.latency_ms:.0f}ms")
        print(f"  Cost: ${benchmark.cost_per_prediction:.4f}")
        print()
    
    # Create visualizations
    radar_fig = analyzer.create_model_comparison_radar()
    radar_fig.show()
    
    cost_fig = analyzer.create_performance_vs_cost_analysis()
    cost_fig.show()
    
    return analyzer


async def demo_advanced_analytics():
    """Demonstrate advanced analytics features"""
    print("🧠 Advanced Analytics Demo")
    print("-" * 40)
    
    api_key = os.getenv("PRSM_API_KEY", "demo_key")
    analyzer = PRSMModelAnalyzer(api_key)
    
    # Generate some demo data
    test_cases = [{"prompt": f"Test {i}", "true_label": i%2, "expected_type": "binary"} for i in range(20)]
    await analyzer.benchmark_models_on_task("Demo Task", test_cases)
    
    # Create advanced visualizations
    print("📊 Creating ROC curve analysis...")
    roc_fig = analyzer.create_roc_curve_comparison()
    roc_fig.show()
    
    print("🔍 Creating confusion matrix comparison...")
    confusion_fig = analyzer.create_confusion_matrix_comparison()
    confusion_fig.show()
    
    print("📈 Creating learning curve analysis...")
    learning_fig = analyzer.create_learning_curve_analysis()
    learning_fig.show()
    
    print("⚙️ Creating hyperparameter optimization visualization...")
    hyperopt_fig = analyzer.create_hyperparameter_optimization_viz()
    hyperopt_fig.show()
    
    print("🔍 Creating feature importance analysis...")
    feature_fig = analyzer.create_feature_importance_analysis()
    feature_fig.show()
    
    return analyzer


async def demo_comprehensive_report():
    """Create comprehensive ML analysis report"""
    print("📋 Comprehensive ML Report Demo")
    print("-" * 40)
    
    api_key = os.getenv("PRSM_API_KEY", "demo_key")
    analyzer = PRSMModelAnalyzer(api_key)
    
    # Create comprehensive report
    report_file = await analyzer.create_comprehensive_ml_report()
    
    print(f"✅ Comprehensive report created: {report_file}")
    print("🌐 Open the HTML file in your browser to view the interactive report")
    
    return analyzer, report_file


async def main():
    """Run all ML model analysis examples"""
    print("🧠 PRSM Python SDK - ML Model Analysis Examples")
    print("=" * 70)
    
    try:
        # Check dependencies
        print("🔍 Checking dependencies...")
        required_packages = {
            'scikit-learn': 'Machine learning algorithms',
            'plotly': 'Interactive visualizations',
            'pandas': 'Data manipulation',
            'numpy': 'Numerical computing'
        }
        
        optional_packages = {
            'shap': 'Model interpretability',
            'lime': 'Local interpretability'
        }
        
        for package, description in required_packages.items():
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}: {description}")
            except ImportError:
                print(f"❌ {package}: {description} - REQUIRED")
        
        for package, description in optional_packages.items():
            try:
                __import__(package)
                print(f"✅ {package}: {description}")
            except ImportError:
                print(f"⚠️  {package}: {description} - OPTIONAL")
        
        print("\n🏁 Model Benchmarking")
        print("-" * 40)
        analyzer1 = await demo_model_benchmarking()
        
        print("\n🧠 Advanced Analytics")
        print("-" * 40)
        analyzer2 = await demo_advanced_analytics()
        
        print("\n📋 Comprehensive Report")
        print("-" * 40)
        analyzer3, report_file = await demo_comprehensive_report()
        
        print("\n" + "=" * 70)
        print("✅ All ML analysis examples completed!")
        print("\n💡 Key Features Demonstrated:")
        print("• Model performance benchmarking")
        print("• ROC curve and confusion matrix analysis")
        print("• Learning curve visualization")
        print("• Hyperparameter optimization landscapes")
        print("• Feature importance analysis (with SHAP)")
        print("• Performance vs cost analysis")
        print("• Comprehensive ML reporting")
        
        print(f"\n📄 Reports generated:")
        print(f"• {report_file}")
        
        # Close clients
        for analyzer in [analyzer1, analyzer2, analyzer3]:
            if hasattr(analyzer, 'client'):
                await analyzer.client.close()
        
    except Exception as e:
        print(f"❌ Error in ML analysis examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧠 PRSM ML Model Analysis Examples")
    print("This example demonstrates advanced machine learning visualization capabilities.")
    print("Required packages: scikit-learn plotly pandas numpy")
    print("Optional packages: shap lime (for interpretability)")
    print("\nRunning examples...\n")
    
    asyncio.run(main())