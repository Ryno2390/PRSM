#!/usr/bin/env python3
"""
NWTN System Status Report
========================

This script provides a comprehensive status report of your NWTN system,
including optimized models, deployment status, and next steps.
"""

import os
import json
from datetime import datetime
from pathlib import Path

class NWTNSystemStatus:
    def __init__(self):
        self.models_dir = "models/nwtn_optimized"
        self.config_dir = "config/nwtn"
        self.deployment_dir = "models/deployed"
        self.data_dir = "data/nwtn_training"
        
    def get_system_overview(self):
        """Get overall system status"""
        overview = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational",
            "phases_completed": [],
            "models_optimized": 0,
            "models_deployed": 0,
            "total_optimization_time": 0,
            "average_performance": 0
        }
        
        # Check Phase 1: BYOA
        overview["phases_completed"].append("Phase 1: BYOA Implementation")
        
        # Check Phase 2: NWTN-optimized models
        if os.path.exists(self.models_dir):
            overview["phases_completed"].append("Phase 2: NWTN-optimized Models")
        
        # Check Phase 3: SEAL integration
        if os.path.exists(self.config_dir):
            overview["phases_completed"].append("Phase 3: SEAL Integration")
        
        # Count optimized models
        if os.path.exists(self.models_dir):
            metrics_files = [f for f in os.listdir(self.models_dir) if f.endswith("_metrics.json")]
            overview["models_optimized"] = len(metrics_files)
        
        # Count deployed models
        if os.path.exists(self.config_dir):
            deployment_files = [f for f in os.listdir(self.config_dir) if f.endswith("_deployment_config.json")]
            overview["models_deployed"] = len(deployment_files)
        
        return overview
    
    def get_optimized_models_status(self):
        """Get status of optimized models"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for file in os.listdir(self.models_dir):
            if file.endswith("_metrics.json"):
                model_name = file.replace("_metrics.json", "")
                metrics_file = os.path.join(self.models_dir, file)
                
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Check if deployed
                    deployment_config = os.path.join(self.config_dir, f"{model_name}_deployment_config.json")
                    is_deployed = os.path.exists(deployment_config)
                    
                    deployment_status = "not_deployed"
                    if is_deployed:
                        try:
                            with open(deployment_config, 'r') as f:
                                deploy_config = json.load(f)
                            deployment_status = deploy_config.get("deployment_status", "unknown")
                        except:
                            deployment_status = "error"
                    
                    models.append({
                        "name": model_name,
                        "optimization_completed": metrics.get("optimization_completed", False),
                        "validation_accuracy": metrics.get("validation_accuracy", 0),
                        "nwtn_integration_score": metrics.get("nwtn_integration_score", 0),
                        "optimization_time_hours": metrics.get("optimization_time_hours", 0),
                        "completed_at": metrics.get("completed_at", "unknown"),
                        "deployment_status": deployment_status,
                        "is_deployed": is_deployed
                    })
                except:
                    continue
        
        return models
    
    def get_deployment_status(self):
        """Get deployment status"""
        deployments = []
        
        if not os.path.exists(self.config_dir):
            return deployments
        
        for file in os.listdir(self.config_dir):
            if file.endswith("_deployment_config.json"):
                model_name = file.replace("_deployment_config.json", "")
                config_file = os.path.join(self.config_dir, file)
                
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    deployments.append({
                        "model_name": model_name,
                        "deployment_status": config.get("deployment_status", "unknown"),
                        "deployed_at": config.get("model_info", {}).get("deployed_at", "unknown"),
                        "performance_metrics": config.get("performance_metrics", {}),
                        "health_checks": config.get("health_checks", {}),
                        "seal_enabled": config.get("integration_settings", {}).get("seal_learning_enabled", False)
                    })
                except:
                    continue
        
        return deployments
    
    def get_available_models_analysis(self):
        """Get analysis of available models from external drive"""
        external_drive = "/Volumes/My Passport/OllamaModels"
        manifests_path = f"{external_drive}/manifests/registry.ollama.ai/library"
        
        available_models = []
        
        if os.path.exists(manifests_path):
            for model_dir in ["command-r", "deepseek-r1", "llama3.1", "llama3.2", "qwen3"]:
                model_path = f"{manifests_path}/{model_dir}"
                if os.path.exists(model_path):
                    available_models.append({
                        "name": model_dir,
                        "path": model_path,
                        "status": "available",
                        "optimized": any(m["name"] == model_dir for m in self.get_optimized_models_status()),
                        "deployed": any(d["model_name"] == model_dir for d in self.get_deployment_status())
                    })
        
        return available_models
    
    def get_next_steps_recommendations(self):
        """Get recommended next steps"""
        recommendations = []
        
        optimized_models = self.get_optimized_models_status()
        deployed_models = self.get_deployment_status()
        available_models = self.get_available_models_analysis()
        
        # Check if no models are optimized
        if not optimized_models:
            recommendations.append({
                "priority": "high",
                "action": "Optimize your first model",
                "description": "Start with llama3.1 for fastest results",
                "command": "python begin_optimization.py llama3.1"
            })
        
        # Check if models are optimized but not deployed
        undeployed_models = [m for m in optimized_models if not m["is_deployed"]]
        if undeployed_models:
            for model in undeployed_models:
                recommendations.append({
                    "priority": "medium",
                    "action": f"Deploy {model['name']} model",
                    "description": f"Deploy your optimized {model['name']} model to NWTN system",
                    "command": f"python deploy_optimized_model.py --deploy {model['name']}"
                })
        
        # Check for additional models to optimize
        unoptimized_models = [m for m in available_models if not m["optimized"]]
        if unoptimized_models and optimized_models:
            recommendations.append({
                "priority": "low",
                "action": "Optimize additional models",
                "description": "Optimize command-r for best performance or deepseek-r1 for cutting-edge features",
                "command": "python begin_optimization.py command-r"
            })
        
        # Check for production readiness
        if deployed_models:
            recommendations.append({
                "priority": "medium",
                "action": "Test production deployment",
                "description": "Test your deployed models in production scenarios",
                "command": "python test_production_deployment.py"
            })
        
        return recommendations
    
    def generate_status_report(self):
        """Generate comprehensive status report"""
        print("🎯 NWTN SYSTEM STATUS REPORT")
        print("=" * 60)
        
        # System Overview
        overview = self.get_system_overview()
        print(f"\n📊 SYSTEM OVERVIEW")
        print(f"   Status: {overview['system_status'].upper()}")
        print(f"   Phases Completed: {len(overview['phases_completed'])}/3")
        for phase in overview['phases_completed']:
            print(f"   ✅ {phase}")
        
        print(f"\n📈 METRICS")
        print(f"   Models Optimized: {overview['models_optimized']}")
        print(f"   Models Deployed: {overview['models_deployed']}")
        
        # Optimized Models Status
        optimized_models = self.get_optimized_models_status()
        if optimized_models:
            print(f"\n🧠 OPTIMIZED MODELS")
            for model in optimized_models:
                status_icon = "🚀" if model["is_deployed"] else "⏳"
                print(f"   {status_icon} {model['name'].upper()}")
                print(f"      Validation Accuracy: {model['validation_accuracy']:.1%}")
                print(f"      NWTN Integration: {model['nwtn_integration_score']:.1%}")
                print(f"      Optimization Time: {model['optimization_time_hours']:.1f} hours")
                print(f"      Deployment Status: {model['deployment_status']}")
        
        # Deployment Status
        deployments = self.get_deployment_status()
        if deployments:
            print(f"\n🚀 DEPLOYED MODELS")
            for deployment in deployments:
                health_status = "🟢" if all(deployment["health_checks"].values()) else "🟡"
                print(f"   {health_status} {deployment['model_name'].upper()}")
                print(f"      Deployed: {deployment['deployed_at'][:10] if deployment['deployed_at'] != 'unknown' else 'unknown'}")
                print(f"      Performance: {deployment['performance_metrics'].get('validation_accuracy', 0):.1%}")
                print(f"      SEAL Learning: {'Enabled' if deployment['seal_enabled'] else 'Disabled'}")
        
        # Available Models Analysis
        available_models = self.get_available_models_analysis()
        if available_models:
            print(f"\n💾 AVAILABLE MODELS")
            for model in available_models:
                opt_status = "✅" if model["optimized"] else "⏳"
                deploy_status = "🚀" if model["deployed"] else "📦"
                print(f"   {opt_status} {deploy_status} {model['name']}")
        
        # Next Steps
        recommendations = self.get_next_steps_recommendations()
        if recommendations:
            print(f"\n📋 RECOMMENDED NEXT STEPS")
            for i, rec in enumerate(recommendations, 1):
                priority_icon = {"high": "🔥", "medium": "⚡", "low": "💡"}[rec["priority"]]
                print(f"   {i}. {priority_icon} {rec['action']}")
                print(f"      {rec['description']}")
                print(f"      Command: {rec['command']}")
        
        # Performance Summary
        if optimized_models:
            avg_accuracy = sum(m["validation_accuracy"] for m in optimized_models) / len(optimized_models)
            avg_integration = sum(m["nwtn_integration_score"] for m in optimized_models) / len(optimized_models)
            total_time = sum(m["optimization_time_hours"] for m in optimized_models)
            
            print(f"\n🎯 PERFORMANCE SUMMARY")
            print(f"   Average Validation Accuracy: {avg_accuracy:.1%}")
            print(f"   Average NWTN Integration Score: {avg_integration:.1%}")
            print(f"   Total Optimization Time: {total_time:.1f} hours")
            print(f"   System Ready for Production: {'Yes' if deployments else 'No'}")
        
        print(f"\n⏰ Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


def main():
    status = NWTNSystemStatus()
    status.generate_status_report()

if __name__ == "__main__":
    main()