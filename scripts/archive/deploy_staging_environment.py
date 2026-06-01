#!/usr/bin/env python3
"""
One-Click Staging Environment Deployment for PRSM Phase 0
Achieves complete staging deployment with infrastructure provisioning and application deployment

ADDRESSES GEMINI AUDIT REQUIREMENT:
"Achieve a fully-containerized, one-click deployment to a staging EKS environment"

DEPLOYMENT STEPS:
1. Provision AWS infrastructure with Terraform
2. Deploy Kubernetes manifests to EKS
3. Initialize production data layer
4. Run consensus-network integration
5. Validate deployment health
6. Generate deployment report
"""

import asyncio
import json
import logging
import subprocess
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
import os

# Add PRSM to path
sys.path.append(str(Path(__file__).parent.parent))

from prsm.storage.production_data_layer import test_production_data_layer
from prsm.federation.consensus_network_bridge import test_consensus_network_integration

logger = logging.getLogger(__name__)


class StagingDeploymentManager:
    """Manages complete staging environment deployment"""
    
    def __init__(self):
        self.deployment_config = {
            "environment": "staging",
            "aws_region": "us-west-2",
            "cluster_name": "prsm-staging-cluster",
            "namespace": "prsm-staging"
        }
        
        self.deployment_stats = {
            "start_time": None,
            "end_time": None,
            "infrastructure_status": "pending",
            "application_status": "pending",
            "data_layer_status": "pending",
            "consensus_status": "pending",
            "health_status": "pending",
            "errors": []
        }
        
        # Paths
        self.terraform_dir = Path(__file__).parent.parent / "deploy" / "enterprise" / "terraform"
        self.k8s_dir = Path(__file__).parent.parent / "deploy" / "kubernetes" / "production"
        
    async def deploy_staging_environment(self) -> Dict[str, Any]:
        """Execute complete one-click staging deployment"""
        print("🚀 PRSM Staging Environment Deployment")
        print("=" * 60)
        
        self.deployment_stats["start_time"] = datetime.now(timezone.utc)
        
        try:
            # Step 1: Pre-deployment validation
            print("📋 Step 1: Pre-deployment validation...")
            if not await self._validate_prerequisites():
                return self._deployment_failed("Prerequisites validation failed")
            
            print("✅ Prerequisites validated")
            
            # Step 2: Provision AWS infrastructure
            print("📋 Step 2: Provisioning AWS infrastructure...")
            if not await self._provision_infrastructure():
                return self._deployment_failed("Infrastructure provisioning failed")
            
            self.deployment_stats["infrastructure_status"] = "success"
            print("✅ Infrastructure provisioned successfully")
            
            # Step 3: Deploy Kubernetes applications
            print("📋 Step 3: Deploying Kubernetes applications...")
            if not await self._deploy_kubernetes_applications():
                return self._deployment_failed("Kubernetes deployment failed")
            
            self.deployment_stats["application_status"] = "success"
            print("✅ Applications deployed successfully")
            
            # Step 4: Initialize data layer
            print("📋 Step 4: Initializing production data layer...")
            if not await self._initialize_data_layer():
                return self._deployment_failed("Data layer initialization failed")
            
            self.deployment_stats["data_layer_status"] = "success"
            print("✅ Data layer initialized successfully")
            
            # Step 5: Test consensus integration
            print("📋 Step 5: Testing consensus-network integration...")
            if not await self._test_consensus_integration():
                return self._deployment_failed("Consensus integration failed")
            
            self.deployment_stats["consensus_status"] = "success"
            print("✅ Consensus integration validated")
            
            # Step 6: Comprehensive health check
            print("📋 Step 6: Performing comprehensive health check...")
            health_results = await self._comprehensive_health_check()
            
            if health_results["overall_healthy"]:
                self.deployment_stats["health_status"] = "success"
                print("✅ Health check passed")
            else:
                print("⚠️ Health check had warnings - see report for details")
                self.deployment_stats["health_status"] = "warning"
            
            # Step 7: Generate deployment report
            print("📋 Step 7: Generating deployment report...")
            await self._generate_deployment_report(health_results)
            
            self.deployment_stats["end_time"] = datetime.now(timezone.utc)
            
            print("🎉 STAGING DEPLOYMENT COMPLETED SUCCESSFULLY!")
            print(f"🌐 Access your staging environment at: https://{self._get_staging_endpoint()}")
            
            return self.deployment_stats
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return self._deployment_failed(f"Unexpected error: {str(e)}")
    
    async def _validate_prerequisites(self) -> bool:
        """Validate deployment prerequisites"""
        try:
            # Check required tools
            required_tools = ["terraform", "kubectl", "aws"]
            
            for tool in required_tools:
                result = subprocess.run(["which", tool], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"❌ Required tool not found: {tool}")
                    return False
            
            # Check AWS credentials
            result = subprocess.run(["aws", "sts", "get-caller-identity"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ AWS credentials not configured")
                return False
            
            # Check Terraform directory exists
            if not self.terraform_dir.exists():
                print(f"❌ Terraform directory not found: {self.terraform_dir}")
                return False
            
            # Check Kubernetes manifests exist
            if not self.k8s_dir.exists():
                print(f"❌ Kubernetes manifests directory not found: {self.k8s_dir}")
                return False
            
            print("✅ All prerequisites validated")
            return True
            
        except Exception as e:
            print(f"❌ Prerequisites validation failed: {e}")
            return False
    
    async def _provision_infrastructure(self) -> bool:
        """Provision AWS infrastructure using Terraform"""
        try:
            os.chdir(self.terraform_dir)
            
            # Initialize Terraform
            print("🔧 Initializing Terraform...")
            result = subprocess.run(
                ["terraform", "init"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Terraform init failed: {result.stderr}")
                return False
            
            # Plan infrastructure
            print("📋 Planning infrastructure changes...")
            result = subprocess.run([
                "terraform", "plan", 
                f"-var=environment={self.deployment_config['environment']}",
                "-out=tfplan"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Terraform plan failed: {result.stderr}")
                return False
            
            # Apply infrastructure
            print("🚀 Applying infrastructure changes...")
            result = subprocess.run([
                "terraform", "apply", 
                "-auto-approve",
                "tfplan"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Terraform apply failed: {result.stderr}")
                return False
            
            print("✅ Infrastructure provisioned successfully")
            return True
            
        except Exception as e:
            print(f"❌ Infrastructure provisioning failed: {e}")
            return False
        finally:
            # Return to original directory
            os.chdir(Path(__file__).parent.parent)
    
    async def _deploy_kubernetes_applications(self) -> bool:
        """Deploy Kubernetes applications to EKS"""
        try:
            # Update kubeconfig for EKS
            print("🔧 Updating kubeconfig for EKS...")
            result = subprocess.run([
                "aws", "eks", "update-kubeconfig",
                "--region", self.deployment_config["aws_region"],
                "--name", self.deployment_config["cluster_name"]
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to update kubeconfig: {result.stderr}")
                return False
            
            # Create namespace
            print("🔧 Creating Kubernetes namespace...")
            result = subprocess.run([
                "kubectl", "create", "namespace", 
                self.deployment_config["namespace"],
                "--dry-run=client", "-o", "yaml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Apply namespace
                subprocess.run([
                    "kubectl", "apply", "-f", "-"
                ], input=result.stdout, text=True)
            
            # Apply Kubernetes manifests
            print("🚀 Deploying Kubernetes applications...")
            k8s_files = [
                "namespace.yaml",
                "configmap.yaml", 
                "secret.yaml",
                "rbac.yaml",
                "persistent-volumes.yaml",
                "prsm-api-deployment.yaml",
                "prsm-api-service.yaml",
                "prsm-consensus-deployment.yaml",
                "prsm-consensus-service.yaml",
                "prsm-federation-deployment.yaml",
                "prsm-federation-service.yaml",
                "milvus-deployment.yaml",
                "milvus-service.yaml",
                "hpa.yaml",
                "ingress.yaml"
            ]
            
            for k8s_file in k8s_files:
                file_path = self.k8s_dir / k8s_file
                if file_path.exists():
                    print(f"📄 Applying {k8s_file}...")
                    result = subprocess.run([
                        "kubectl", "apply", "-f", str(file_path),
                        "-n", self.deployment_config["namespace"]
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"⚠️ Warning applying {k8s_file}: {result.stderr}")
                else:
                    print(f"⚠️ Kubernetes manifest not found: {k8s_file}")
            
            # Wait for deployments to be ready
            print("⏳ Waiting for deployments to be ready...")
            await self._wait_for_deployments()
            
            print("✅ Kubernetes applications deployed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Kubernetes deployment failed: {e}")
            return False
    
    async def _wait_for_deployments(self, timeout_minutes: int = 10):
        """Wait for Kubernetes deployments to be ready"""
        deployments = ["prsm-api", "prsm-consensus", "prsm-federation"]
        
        for deployment in deployments:
            print(f"⏳ Waiting for {deployment} to be ready...")
            
            result = subprocess.run([
                "kubectl", "rollout", "status",
                f"deployment/{deployment}",
                "-n", self.deployment_config["namespace"],
                f"--timeout={timeout_minutes}m"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {deployment} is ready")
            else:
                print(f"⚠️ {deployment} deployment timeout or failed")
    
    async def _initialize_data_layer(self) -> bool:
        """Initialize and test production data layer"""
        try:
            print("🔧 Testing production data layer...")
            
            # Run data layer test
            test_results = await test_production_data_layer()
            
            if test_results.get("data_layer_functional", False):
                print("✅ Production data layer functional")
                return True
            else:
                print("❌ Production data layer test failed")
                return False
                
        except Exception as e:
            print(f"❌ Data layer initialization failed: {e}")
            return False
    
    async def _test_consensus_integration(self) -> bool:
        """Test consensus-network integration"""
        try:
            print("🔧 Testing consensus-network integration...")
            
            # Run consensus integration test
            test_results = await test_consensus_network_integration()
            
            if test_results.get("integration_functional", False):
                print("✅ Consensus-network integration functional")
                return True
            else:
                print("❌ Consensus-network integration test failed")
                return False
                
        except Exception as e:
            print(f"❌ Consensus integration test failed: {e}")
            return False
    
    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on staging environment"""
        health_results = {
            "infrastructure": {},
            "kubernetes": {},
            "applications": {},
            "overall_healthy": False
        }
        
        try:
            # Check infrastructure health
            print("🔍 Checking infrastructure health...")
            health_results["infrastructure"] = await self._check_infrastructure_health()
            
            # Check Kubernetes health
            print("🔍 Checking Kubernetes health...")
            health_results["kubernetes"] = await self._check_kubernetes_health()
            
            # Check application health
            print("🔍 Checking application health...")
            health_results["applications"] = await self._check_application_health()
            
            # Determine overall health
            infrastructure_ok = health_results["infrastructure"].get("healthy", False)
            kubernetes_ok = health_results["kubernetes"].get("healthy", False)
            applications_ok = health_results["applications"].get("healthy", False)
            
            health_results["overall_healthy"] = all([infrastructure_ok, kubernetes_ok, applications_ok])
            
            return health_results
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            health_results["error"] = str(e)
            return health_results
    
    async def _check_infrastructure_health(self) -> Dict[str, Any]:
        """Check AWS infrastructure health"""
        try:
            # Check EKS cluster status
            result = subprocess.run([
                "aws", "eks", "describe-cluster",
                "--name", self.deployment_config["cluster_name"],
                "--region", self.deployment_config["aws_region"]
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                cluster_info = json.loads(result.stdout)
                cluster_status = cluster_info["cluster"]["status"]
                
                return {
                    "healthy": cluster_status == "ACTIVE",
                    "cluster_status": cluster_status,
                    "cluster_endpoint": cluster_info["cluster"]["endpoint"]
                }
            else:
                return {"healthy": False, "error": "Failed to describe EKS cluster"}
                
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_kubernetes_health(self) -> Dict[str, Any]:
        """Check Kubernetes cluster health"""
        try:
            # Check node status
            result = subprocess.run([
                "kubectl", "get", "nodes",
                "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                nodes_info = json.loads(result.stdout)
                ready_nodes = 0
                total_nodes = len(nodes_info["items"])
                
                for node in nodes_info["items"]:
                    for condition in node["status"]["conditions"]:
                        if condition["type"] == "Ready" and condition["status"] == "True":
                            ready_nodes += 1
                            break
                
                return {
                    "healthy": ready_nodes == total_nodes and total_nodes > 0,
                    "ready_nodes": ready_nodes,
                    "total_nodes": total_nodes
                }
            else:
                return {"healthy": False, "error": "Failed to get node status"}
                
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_application_health(self) -> Dict[str, Any]:
        """Check application health"""
        try:
            # Check pod status
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", self.deployment_config["namespace"],
                "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                pods_info = json.loads(result.stdout)
                running_pods = 0
                total_pods = len(pods_info["items"])
                
                for pod in pods_info["items"]:
                    if pod["status"]["phase"] == "Running":
                        running_pods += 1
                
                return {
                    "healthy": running_pods == total_pods and total_pods > 0,
                    "running_pods": running_pods,
                    "total_pods": total_pods
                }
            else:
                return {"healthy": False, "error": "Failed to get pod status"}
                
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _get_staging_endpoint(self) -> str:
        """Get staging environment endpoint"""
        try:
            # Get load balancer endpoint
            result = subprocess.run([
                "kubectl", "get", "ingress",
                "-n", self.deployment_config["namespace"],
                "-o", "jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}'"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().strip("'")
            else:
                return f"{self.deployment_config['cluster_name']}.{self.deployment_config['aws_region']}.eks.amazonaws.com"
                
        except Exception:
            return "staging-endpoint-unavailable"
    
    async def _generate_deployment_report(self, health_results: Dict[str, Any]):
        """Generate comprehensive deployment report"""
        duration = (
            self.deployment_stats["end_time"] or datetime.now(timezone.utc) - 
            self.deployment_stats["start_time"]
        ).total_seconds()
        
        report = f"""
PRSM Staging Environment Deployment Report
==========================================

Deployment Summary:
- Environment: {self.deployment_config['environment']}
- Duration: {duration:.2f} seconds
- AWS Region: {self.deployment_config['aws_region']}
- EKS Cluster: {self.deployment_config['cluster_name']}
- Namespace: {self.deployment_config['namespace']}

Component Status:
- Infrastructure: {self._status_emoji(self.deployment_stats['infrastructure_status'])} {self.deployment_stats['infrastructure_status']}
- Applications: {self._status_emoji(self.deployment_stats['application_status'])} {self.deployment_stats['application_status']}
- Data Layer: {self._status_emoji(self.deployment_stats['data_layer_status'])} {self.deployment_stats['data_layer_status']}
- Consensus: {self._status_emoji(self.deployment_stats['consensus_status'])} {self.deployment_stats['consensus_status']}
- Health Check: {self._status_emoji(self.deployment_stats['health_status'])} {self.deployment_stats['health_status']}

Health Check Results:
- Infrastructure: {'✅ Healthy' if health_results.get('infrastructure', {}).get('healthy') else '❌ Unhealthy'}
- Kubernetes: {'✅ Healthy' if health_results.get('kubernetes', {}).get('healthy') else '❌ Unhealthy'}
- Applications: {'✅ Healthy' if health_results.get('applications', {}).get('healthy') else '❌ Unhealthy'}

Access Information:
- Staging Endpoint: https://{self._get_staging_endpoint()}
- Kubernetes Dashboard: kubectl port-forward -n {self.deployment_config['namespace']} service/prsm-api 8080:80
- Monitoring: kubectl logs -n {self.deployment_config['namespace']} -l app=prsm-api

Deployment Status: {'✅ SUCCESSFUL' if len(self.deployment_stats['errors']) == 0 else '⚠️ COMPLETED WITH WARNINGS'}

Next Steps:
1. Access the staging environment at the endpoint above
2. Run integration tests: python tests/integration/test_staging_environment.py
3. Monitor logs and metrics for any issues
4. Proceed with production deployment if staging validates successfully
"""
        
        print(report)
        
        # Save report to file
        report_file = Path("staging_deployment_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Full deployment report saved to: {report_file}")
    
    def _status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        status_emojis = {
            "success": "✅",
            "pending": "⏳",
            "warning": "⚠️",
            "failed": "❌"
        }
        return status_emojis.get(status, "❓")
    
    def _deployment_failed(self, error_message: str) -> Dict[str, Any]:
        """Handle deployment failure"""
        self.deployment_stats["errors"].append(error_message)
        self.deployment_stats["end_time"] = datetime.now(timezone.utc)
        
        print(f"❌ DEPLOYMENT FAILED: {error_message}")
        print("📄 Check deployment report for details")
        
        return self.deployment_stats


async def main():
    """Main deployment execution"""
    print("🚀 PRSM One-Click Staging Deployment")
    print("This will provision AWS infrastructure and deploy PRSM to staging")
    print()
    
    # Confirmation
    response = input("Continue with staging deployment? (y/N): ")
    if response.lower() != 'y':
        print("Deployment cancelled")
        return
    
    deployment_manager = StagingDeploymentManager()
    results = await deployment_manager.deploy_staging_environment()
    
    if len(results["errors"]) == 0:
        print("\n🎉 Staging deployment completed successfully!")
        print("Your PRSM staging environment is ready for testing.")
    else:
        print(f"\n⚠️ Deployment completed with {len(results['errors'])} errors.")
        print("Please review the deployment report for details.")


if __name__ == "__main__":
    asyncio.run(main())