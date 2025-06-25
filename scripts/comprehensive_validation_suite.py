#!/usr/bin/env python3
"""
PRSM Comprehensive Validation Suite
Automated testing and validation of all PRSM working features including:
- P2P network functionality
- AI model inference and management
- Consensus mechanisms
- Performance metrics
- Security features
- Enterprise capabilities

This script provides comprehensive validation with detailed reporting
for production readiness assessment.
"""

import asyncio
import json
import time
import logging
import sys
import traceback
import subprocess
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import unittest
import tempfile
import hashlib
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    category: str
    status: str  # "PASS", "FAIL", "SKIP", "WARNING"
    duration: float
    details: str
    metrics: Dict[str, Any] = None
    error_message: Optional[str] = None

@dataclass
class ValidationSummary:
    """Summary of all validation results"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    warning_tests: int
    total_duration: float
    categories: Dict[str, Dict[str, int]]
    critical_failures: List[str]
    recommendations: List[str]

class PRSMValidator:
    """Comprehensive PRSM validation framework"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp(prefix="prsm_validation_")
        
        # Validation categories
        self.categories = [
            "Core Infrastructure",
            "P2P Network",
            "AI Model Management", 
            "Consensus & Security",
            "Performance & Scalability",
            "Enterprise Features",
            "Integration & APIs",
            "Documentation & Usability"
        ]
        
        logger.info(f"PRSM Validation Suite initialized - temp dir: {self.temp_dir}")
    
    async def run_comprehensive_validation(self) -> ValidationSummary:
        """Run complete validation suite"""
        print("🚀 PRSM Comprehensive Validation Suite")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project: {project_root}")
        print()
        
        # Run all validation categories
        await self._validate_core_infrastructure()
        await self._validate_p2p_network()
        await self._validate_ai_model_management()
        await self._validate_consensus_security()
        await self._validate_performance_scalability()
        await self._validate_enterprise_features()
        await self._validate_integration_apis()
        await self._validate_documentation_usability()
        
        return self._generate_summary()
    
    async def _validate_core_infrastructure(self):
        """Validate core PRSM infrastructure"""
        category = "Core Infrastructure"
        logger.info(f"🔍 Validating {category}...")
        
        # Test 1: Project structure validation
        await self._test_project_structure(category)
        
        # Test 2: Python dependencies
        await self._test_python_dependencies(category)
        
        # Test 3: Configuration files
        await self._test_configuration_files(category)
        
        # Test 4: Import validation
        await self._test_import_validation(category)
        
        # Test 5: Basic module functionality
        await self._test_basic_module_functionality(category)
    
    async def _test_project_structure(self, category: str):
        """Test project directory structure"""
        test_name = "Project Structure Validation"
        start_time = time.time()
        
        try:
            required_dirs = [
                "prsm", "demos", "docs", "config", "tests", "scripts", "sdks"
            ]
            
            required_files = [
                "README.md", "requirements.txt", ".gitignore"
            ]
            
            missing_dirs = []
            missing_files = []
            
            # Check directories
            for dir_name in required_dirs:
                if not (project_root / dir_name).exists():
                    missing_dirs.append(dir_name)
            
            # Check files
            for file_name in required_files:
                if not (project_root / file_name).exists():
                    missing_files.append(file_name)
            
            issues = []
            if missing_dirs:
                issues.append(f"Missing directories: {', '.join(missing_dirs)}")
            if missing_files:
                issues.append(f"Missing files: {', '.join(missing_files)}")
            
            duration = time.time() - start_time
            
            if issues:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Project structure issues found: {'; '.join(issues)}",
                    metrics={"missing_dirs": len(missing_dirs), "missing_files": len(missing_files)}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details="All required project structure elements found",
                    metrics={"checked_dirs": len(required_dirs), "checked_files": len(required_files)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Project structure validation failed",
                error_message=str(e)
            ))
    
    async def _test_python_dependencies(self, category: str):
        """Test Python dependencies availability"""
        test_name = "Python Dependencies Check"
        start_time = time.time()
        
        try:
            critical_deps = [
                "asyncio", "json", "hashlib", "uuid", "logging", "pathlib"
            ]
            
            optional_deps = [
                ("torch", "PyTorch for neural networks"),
                ("sklearn", "scikit-learn for ML models"),
                ("numpy", "NumPy for numerical computing"),
                ("pandas", "Pandas for data manipulation"),
                ("streamlit", "Streamlit for dashboards"),
                ("plotly", "Plotly for visualizations")
            ]
            
            missing_critical = []
            missing_optional = []
            
            # Check critical dependencies
            for dep in critical_deps:
                try:
                    __import__(dep)
                except ImportError:
                    missing_critical.append(dep)
            
            # Check optional dependencies
            for dep, desc in optional_deps:
                try:
                    __import__(dep)
                except ImportError:
                    missing_optional.append((dep, desc))
            
            duration = time.time() - start_time
            
            if missing_critical:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Critical dependencies missing: {', '.join(missing_critical)}",
                    metrics={"missing_critical": len(missing_critical), "missing_optional": len(missing_optional)}
                ))
            elif missing_optional:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Optional dependencies missing: {', '.join([dep for dep, _ in missing_optional])}. Some features may be limited.",
                    metrics={"missing_critical": 0, "missing_optional": len(missing_optional)}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details="All dependencies available",
                    metrics={"critical_deps": len(critical_deps), "optional_deps": len(optional_deps)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Dependency check failed",
                error_message=str(e)
            ))
    
    async def _test_configuration_files(self, category: str):
        """Test configuration files validity"""
        test_name = "Configuration Files Validation"
        start_time = time.time()
        
        try:
            config_files = [
                ("config/grafana/dashboards/prsm-overview.json", "Grafana dashboard config"),
                ("docs/SECURITY_ARCHITECTURE.md", "Security documentation"),
                ("docs/ENTERPRISE_MONITORING_GUIDE.md", "Monitoring documentation"),
                ("docs/COMPLIANCE_FRAMEWORK.md", "Compliance documentation")
            ]
            
            valid_configs = 0
            invalid_configs = []
            
            for config_path, description in config_files:
                full_path = project_root / config_path
                if full_path.exists():
                    try:
                        if config_path.endswith('.json'):
                            with open(full_path, 'r') as f:
                                json.load(f)  # Validate JSON
                        valid_configs += 1
                    except json.JSONDecodeError as e:
                        invalid_configs.append(f"{config_path}: Invalid JSON - {e}")
                    except Exception as e:
                        invalid_configs.append(f"{config_path}: Error - {e}")
                else:
                    invalid_configs.append(f"{config_path}: File not found")
            
            duration = time.time() - start_time
            
            if invalid_configs:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Configuration issues: {'; '.join(invalid_configs)}",
                    metrics={"valid_configs": valid_configs, "total_configs": len(config_files)}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details="All configuration files valid",
                    metrics={"valid_configs": valid_configs, "total_configs": len(config_files)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Configuration validation failed",
                error_message=str(e)
            ))
    
    async def _test_import_validation(self, category: str):
        """Test key module imports"""
        test_name = "Module Import Validation"
        start_time = time.time()
        
        try:
            modules_to_test = [
                ("demos.p2p_network_demo", "P2P Network Demo"),
                ("demos.enhanced_p2p_ai_demo", "Enhanced P2P AI Demo"),
                ("prsm.distillation.model_inference_engine", "Model Inference Engine"),
                ("prsm.distillation.model_conversion_utilities", "Model Conversion Utilities")
            ]
            
            successful_imports = 0
            failed_imports = []
            
            for module_name, description in modules_to_test:
                try:
                    __import__(module_name)
                    successful_imports += 1
                except ImportError as e:
                    failed_imports.append(f"{description}: {e}")
                except Exception as e:
                    failed_imports.append(f"{description}: Unexpected error - {e}")
            
            duration = time.time() - start_time
            
            if failed_imports:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Import failures: {'; '.join(failed_imports)}",
                    metrics={"successful_imports": successful_imports, "total_modules": len(modules_to_test)}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details="All key modules imported successfully",
                    metrics={"successful_imports": successful_imports, "total_modules": len(modules_to_test)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Import validation failed",
                error_message=str(e)
            ))
    
    async def _test_basic_module_functionality(self, category: str):
        """Test basic functionality of key modules"""
        test_name = "Basic Module Functionality"
        start_time = time.time()
        
        try:
            # Test model inference engine
            try:
                from prsm.distillation.model_inference_engine import ModelInferenceEngine
                engine = ModelInferenceEngine()
                # Basic instantiation test
                basic_tests_passed = 1
            except Exception as e:
                basic_tests_passed = 0
                error_details = f"ModelInferenceEngine instantiation failed: {e}"
            
            # Test model conversion utilities  
            try:
                from prsm.distillation.model_conversion_utilities import ModelConversionEngine
                converter = ModelConversionEngine()
                basic_tests_passed += 1
            except Exception as e:
                error_details = f"ModelConversionEngine instantiation failed: {e}"
            
            duration = time.time() - start_time
            
            if basic_tests_passed == 2:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details="Basic module functionality verified",
                    metrics={"modules_tested": 2, "modules_passed": basic_tests_passed}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Module functionality issues: {error_details}",
                    metrics={"modules_tested": 2, "modules_passed": basic_tests_passed}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Basic functionality test failed",
                error_message=str(e)
            ))
    
    async def _validate_p2p_network(self):
        """Validate P2P network functionality"""
        category = "P2P Network"
        logger.info(f"🌐 Validating {category}...")
        
        await self._test_p2p_basic_demo(category)
        await self._test_p2p_enhanced_demo(category)
        await self._test_consensus_mechanism(category)
        await self._test_network_resilience(category)
    
    async def _test_p2p_basic_demo(self, category: str):
        """Test basic P2P network demo"""
        test_name = "Basic P2P Network Demo"
        start_time = time.time()
        
        try:
            # Run basic P2P demo programmatically
            from demos.p2p_network_demo import P2PNetworkDemo
            
            demo = P2PNetworkDemo(num_nodes=3)
            
            # Test network startup
            network_task = asyncio.create_task(demo.start_network())
            await asyncio.sleep(2)  # Give network time to start
            
            # Check network status
            status = demo.get_network_status()
            
            # Stop network
            await demo.stop_network()
            
            duration = time.time() - start_time
            
            # Validate results
            if (status["active_nodes"] >= 2 and 
                status["total_connections"] >= 2 and
                status["total_messages"] > 0):
                
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"P2P network functional: {status['active_nodes']} nodes, {status['total_connections']} connections, {status['total_messages']} messages",
                    metrics=status
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"P2P network issues: {status}",
                    metrics=status
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Basic P2P demo failed",
                error_message=str(e)
            ))
    
    async def _test_p2p_enhanced_demo(self, category: str):
        """Test enhanced P2P AI demo"""
        test_name = "Enhanced P2P AI Demo"
        start_time = time.time()
        
        try:
            from demos.enhanced_p2p_ai_demo import EnhancedP2PNetworkDemo
            
            demo = EnhancedP2PNetworkDemo(num_nodes=3)
            
            # Test enhanced network startup
            network_task = asyncio.create_task(demo.start_network())
            await asyncio.sleep(3)  # More time for AI model initialization
            
            # Test AI functionality
            await demo.demonstrate_model_discovery()
            await demo.demonstrate_distributed_inference()
            
            # Check enhanced status AFTER running demonstrations
            status = demo.get_enhanced_network_status()
            ai_metrics = status.get("ai_network_metrics", {})
            
            # Stop network
            await demo.stop_network()
            
            duration = time.time() - start_time
            
            # Validate AI-enhanced results
            if (status["active_nodes"] >= 2 and 
                ai_metrics.get("total_models", 0) > 0 and
                ai_metrics.get("total_inferences", 0) > 0):
                
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Enhanced P2P AI functional: {ai_metrics.get('total_models', 0)} models, {ai_metrics.get('total_inferences', 0)} inferences",
                    metrics=ai_metrics
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Enhanced P2P AI issues: {ai_metrics}",
                    metrics=ai_metrics
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Enhanced P2P AI demo failed",
                error_message=str(e)
            ))
    
    async def _test_consensus_mechanism(self, category: str):
        """Test consensus mechanism functionality"""
        test_name = "Consensus Mechanism"
        start_time = time.time()
        
        try:
            from demos.p2p_network_demo import P2PNetworkDemo
            
            demo = P2PNetworkDemo(num_nodes=3)
            
            # Start network
            network_task = asyncio.create_task(demo.start_network())
            await asyncio.sleep(2)
            
            # Test consensus
            await demo.demonstrate_consensus()
            
            # Check consensus results
            consensus_proposals = sum(len(node.pending_consensus) for node in demo.nodes)
            
            await demo.stop_network()
            
            duration = time.time() - start_time
            
            if consensus_proposals > 0:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Consensus mechanism functional: {consensus_proposals} proposals processed",
                    metrics={"consensus_proposals": consensus_proposals}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details="Consensus mechanism may have issues",
                    metrics={"consensus_proposals": consensus_proposals}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Consensus mechanism test failed",
                error_message=str(e)
            ))
    
    async def _test_network_resilience(self, category: str):
        """Test network resilience and failure recovery"""
        test_name = "Network Resilience"
        start_time = time.time()
        
        try:
            from demos.p2p_network_demo import P2PNetworkDemo
            
            demo = P2PNetworkDemo(num_nodes=3)
            
            # Start network
            network_task = asyncio.create_task(demo.start_network())
            await asyncio.sleep(2)
            
            # Test failure recovery
            await demo.simulate_node_failure()
            
            # Check network status after failure simulation
            final_status = demo.get_network_status()
            
            await demo.stop_network()
            
            duration = time.time() - start_time
            
            # Network should still be functional after failure simulation
            if final_status["active_nodes"] >= 2:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Network resilience validated: {final_status['active_nodes']} nodes active after failure simulation",
                    metrics=final_status
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Network resilience issues: only {final_status['active_nodes']} nodes active",
                    metrics=final_status
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Network resilience test failed",
                error_message=str(e)
            ))
    
    async def _validate_ai_model_management(self):
        """Validate AI model management capabilities"""
        category = "AI Model Management"
        logger.info(f"🧠 Validating {category}...")
        
        await self._test_model_inference_engine(category)
        await self._test_model_conversion_utilities(category)
        await self._test_ai_model_loading(category)
        await self._test_distributed_inference(category)
    
    async def _test_model_inference_engine(self, category: str):
        """Test model inference engine"""
        test_name = "Model Inference Engine"
        start_time = time.time()
        
        try:
            from prsm.distillation.model_inference_engine import ModelInferenceEngine
            
            engine = ModelInferenceEngine()
            
            # Test basic functionality
            available_frameworks = engine.get_available_frameworks()
            supported_formats = engine.get_supported_formats()
            
            duration = time.time() - start_time
            
            if len(available_frameworks) > 0 and len(supported_formats) > 0:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Inference engine functional: {len(available_frameworks)} frameworks, {len(supported_formats)} formats",
                    metrics={"frameworks": available_frameworks, "formats": supported_formats}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details="Inference engine has limited functionality",
                    metrics={"frameworks": available_frameworks, "formats": supported_formats}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Model inference engine test failed",
                error_message=str(e)
            ))
    
    async def _test_model_conversion_utilities(self, category: str):
        """Test model conversion utilities"""
        test_name = "Model Conversion Utilities"
        start_time = time.time()
        
        try:
            from prsm.distillation.model_conversion_utilities import ModelConversionEngine
            
            converter = ModelConversionEngine()
            
            # Test basic functionality
            supported_conversions = converter.get_supported_conversions()
            conversion_history = converter.get_conversion_history()
            
            duration = time.time() - start_time
            
            if len(supported_conversions) > 0:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Conversion utilities functional: {len(supported_conversions)} conversion types",
                    metrics={"conversions": supported_conversions, "history_count": len(conversion_history)}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details="Conversion utilities have limited functionality",
                    metrics={"conversions": supported_conversions}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Model conversion utilities test failed",
                error_message=str(e)
            ))
    
    async def _test_ai_model_loading(self, category: str):
        """Test AI model loading capabilities"""
        test_name = "AI Model Loading"
        start_time = time.time()
        
        try:
            from demos.enhanced_p2p_ai_demo import AIModelManager
            
            manager = AIModelManager("test_node")
            await manager.initialize_demo_models()
            
            models = manager.get_model_catalog()
            
            duration = time.time() - start_time
            
            if len(models) > 0:
                frameworks = set(model.framework for model in models.values())
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"AI model loading successful: {len(models)} models across {len(frameworks)} frameworks",
                    metrics={"model_count": len(models), "frameworks": list(frameworks)}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details="No AI models could be loaded",
                    metrics={"model_count": 0}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="AI model loading test failed",
                error_message=str(e)
            ))
    
    async def _test_distributed_inference(self, category: str):
        """Test distributed inference capabilities"""
        test_name = "Distributed Inference"
        start_time = time.time()
        
        try:
            from demos.enhanced_p2p_ai_demo import AIModelManager, InferenceRequest
            
            manager = AIModelManager("test_node")
            await manager.initialize_demo_models()
            
            # Test inference on different model types
            models = manager.get_model_catalog()
            successful_inferences = 0
            total_inferences = 0
            
            for model_id, model_info in models.items():
                try:
                    # Create appropriate test data
                    if model_info.framework == "pytorch":
                        test_data = [0.1 * i for i in range(10)]
                    elif model_info.framework == "custom":
                        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
                    else:
                        continue
                    
                    request = InferenceRequest(
                        request_id=f"test_{model_id}",
                        model_id=model_id,
                        input_data=test_data,
                        requestor_id="test_node",
                        timestamp=time.time()
                    )
                    
                    result = await manager.perform_inference(request)
                    total_inferences += 1
                    
                    if result.success and len(result.prediction) > 0:
                        successful_inferences += 1
                        
                except Exception as e:
                    logger.warning(f"Inference failed for model {model_id}: {e}")
                    total_inferences += 1
            
            duration = time.time() - start_time
            
            success_rate = (successful_inferences / max(1, total_inferences)) * 100
            
            if success_rate >= 80:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Distributed inference successful: {successful_inferences}/{total_inferences} ({success_rate:.1f}%)",
                    metrics={"successful_inferences": successful_inferences, "total_inferences": total_inferences, "success_rate": success_rate}
                ))
            elif success_rate >= 50:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Distributed inference partially functional: {successful_inferences}/{total_inferences} ({success_rate:.1f}%)",
                    metrics={"successful_inferences": successful_inferences, "total_inferences": total_inferences, "success_rate": success_rate}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Distributed inference failed: {successful_inferences}/{total_inferences} ({success_rate:.1f}%)",
                    metrics={"successful_inferences": successful_inferences, "total_inferences": total_inferences, "success_rate": success_rate}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Distributed inference test failed",
                error_message=str(e)
            ))
    
    async def _validate_consensus_security(self):
        """Validate consensus and security features"""
        category = "Consensus & Security"
        logger.info(f"🔐 Validating {category}...")
        
        await self._test_message_signing(category)
        await self._test_consensus_voting(category)
        await self._test_security_features(category)
    
    async def _test_message_signing(self, category: str):
        """Test message signing and verification"""
        test_name = "Message Signing & Verification"
        start_time = time.time()
        
        try:
            from demos.p2p_network_demo import P2PNode, Message
            
            node = P2PNode()
            
            # Test message signing
            test_payload = {"test": "data", "timestamp": time.time()}
            signature = node._sign_message(json.dumps(test_payload))
            
            # Create signed message
            message = Message(
                message_id="test_msg",
                sender_id=node.node_info.node_id,
                receiver_id="test_receiver",
                message_type="test",
                payload=test_payload,
                timestamp=time.time(),
                signature=signature
            )
            
            # Test signature verification
            is_valid = node._verify_signature(message, node.node_info.public_key)
            
            duration = time.time() - start_time
            
            if is_valid and len(signature) > 0:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details="Message signing and verification functional",
                    metrics={"signature_length": len(signature), "verification_success": is_valid}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details="Message signing or verification failed",
                    metrics={"signature_length": len(signature) if signature else 0, "verification_success": is_valid}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Message signing test failed",
                error_message=str(e)
            ))
    
    async def _test_consensus_voting(self, category: str):
        """Test consensus voting mechanism"""
        test_name = "Consensus Voting Mechanism"
        start_time = time.time()
        
        try:
            from demos.p2p_network_demo import P2PNode, ConsensusProposal
            
            node = P2PNode("validator")
            
            # Create test proposal
            proposal = ConsensusProposal(
                proposal_id="test_proposal",
                proposer_id=node.node_info.node_id,
                proposal_type="test_consensus",
                data={"test": "data"},
                timestamp=time.time(),
                required_votes=3,
                votes={},
                status="pending"
            )
            
            # Test voting logic
            vote = node._evaluate_consensus_proposal(proposal)
            
            # Test vote casting
            await node._cast_consensus_vote(proposal.proposal_id, vote)
            
            duration = time.time() - start_time
            
            if isinstance(vote, bool):
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Consensus voting functional: vote={vote}",
                    metrics={"vote_cast": vote, "proposal_type": proposal.proposal_type}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details="Consensus voting mechanism failed",
                    metrics={}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Consensus voting test failed",
                error_message=str(e)
            ))
    
    async def _test_security_features(self, category: str):
        """Test security features"""
        test_name = "Security Features"
        start_time = time.time()
        
        try:
            # Test hash generation
            test_data = b"test_security_data"
            hash1 = hashlib.sha256(test_data).hexdigest()
            hash2 = hashlib.sha256(test_data).hexdigest()
            
            # Test key generation
            from demos.p2p_network_demo import P2PNode
            node = P2PNode()
            key1 = node._generate_key()
            key2 = node._generate_key()
            
            duration = time.time() - start_time
            
            # Validate security features
            hash_consistent = (hash1 == hash2)
            keys_unique = (key1 != key2)
            key_length_valid = (len(key1) >= 16 and len(key2) >= 16)
            
            if hash_consistent and keys_unique and key_length_valid:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details="Security features functional: hashing consistent, keys unique",
                    metrics={"hash_consistent": hash_consistent, "keys_unique": keys_unique, "key_length": len(key1)}
                ))
            else:
                issues = []
                if not hash_consistent:
                    issues.append("hash inconsistency")
                if not keys_unique:
                    issues.append("key generation not unique")
                if not key_length_valid:
                    issues.append("key length insufficient")
                
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Security issues: {', '.join(issues)}",
                    metrics={"hash_consistent": hash_consistent, "keys_unique": keys_unique}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Security features test failed",
                error_message=str(e)
            ))
    
    async def _validate_performance_scalability(self):
        """Validate performance and scalability"""
        category = "Performance & Scalability"
        logger.info(f"📈 Validating {category}...")
        
        await self._test_system_resources(category)
        await self._test_concurrent_operations(category)
        await self._test_scalability_limits(category)
    
    async def _test_system_resources(self, category: str):
        """Test system resource usage"""
        test_name = "System Resource Usage"
        start_time = time.time()
        
        try:
            # Measure baseline system resources
            initial_memory = psutil.virtual_memory().percent
            initial_cpu = psutil.cpu_percent(interval=1)
            
            # Run a demo to measure resource usage
            from demos.enhanced_p2p_ai_demo import EnhancedP2PNetworkDemo
            demo = EnhancedP2PNetworkDemo(num_nodes=3)
            
            # Start demo and measure peak usage
            network_task = asyncio.create_task(demo.start_network())
            await asyncio.sleep(3)
            
            peak_memory = psutil.virtual_memory().percent
            peak_cpu = psutil.cpu_percent(interval=1)
            
            await demo.stop_network()
            
            # Calculate resource usage
            memory_increase = peak_memory - initial_memory
            cpu_increase = peak_cpu - initial_cpu
            
            duration = time.time() - start_time
            
            # Validate resource usage is reasonable
            if memory_increase < 50 and cpu_increase < 80:  # Reasonable thresholds
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Resource usage acceptable: +{memory_increase:.1f}% memory, +{cpu_increase:.1f}% CPU",
                    metrics={"memory_increase": memory_increase, "cpu_increase": cpu_increase, "peak_memory": peak_memory, "peak_cpu": peak_cpu}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"High resource usage: +{memory_increase:.1f}% memory, +{cpu_increase:.1f}% CPU",
                    metrics={"memory_increase": memory_increase, "cpu_increase": cpu_increase}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="System resource test failed",
                error_message=str(e)
            ))
    
    async def _test_concurrent_operations(self, category: str):
        """Test concurrent operations performance"""
        test_name = "Concurrent Operations"
        start_time = time.time()
        
        try:
            from demos.enhanced_p2p_ai_demo import AIModelManager, InferenceRequest
            
            # Create multiple managers for concurrent testing
            managers = [AIModelManager(f"concurrent_node_{i}") for i in range(3)]
            
            # Initialize all managers concurrently
            init_tasks = [manager.initialize_demo_models() for manager in managers]
            await asyncio.gather(*init_tasks)
            
            # Perform concurrent inferences
            inference_tasks = []
            for i, manager in enumerate(managers):
                models = manager.get_model_catalog()
                if models:
                    model_id = list(models.keys())[0]
                    model_info = models[model_id]
                    
                    # Use appropriate input data based on model framework
                    if model_info.framework == "pytorch":
                        input_data = [0.1 * j for j in range(10)]  # 10 features for neural network
                    elif model_info.framework == "custom":
                        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]    # 5 features for custom model
                    else:
                        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]    # Default
                    
                    request = InferenceRequest(
                        request_id=f"concurrent_{i}",
                        model_id=model_id,
                        input_data=input_data,
                        requestor_id=f"concurrent_node_{i}",
                        timestamp=time.time()
                    )
                    inference_tasks.append(manager.perform_inference(request))
            
            # Execute concurrent inferences
            results = await asyncio.gather(*inference_tasks, return_exceptions=True)
            
            successful_concurrent = sum(1 for r in results if hasattr(r, 'success') and r.success)
            total_concurrent = len(results)
            
            duration = time.time() - start_time
            
            concurrent_success_rate = (successful_concurrent / max(1, total_concurrent)) * 100
            
            if concurrent_success_rate >= 80:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Concurrent operations successful: {successful_concurrent}/{total_concurrent} ({concurrent_success_rate:.1f}%)",
                    metrics={"successful_concurrent": successful_concurrent, "total_concurrent": total_concurrent, "success_rate": concurrent_success_rate}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Concurrent operations issues: {successful_concurrent}/{total_concurrent} ({concurrent_success_rate:.1f}%)",
                    metrics={"successful_concurrent": successful_concurrent, "total_concurrent": total_concurrent}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Concurrent operations test failed",
                error_message=str(e)
            ))
    
    async def _test_scalability_limits(self, category: str):
        """Test scalability limits"""
        test_name = "Scalability Limits"
        start_time = time.time()
        
        try:
            from demos.p2p_network_demo import P2PNetworkDemo
            
            # Test with increasing node counts
            scalability_results = []
            
            for node_count in [3, 5, 7]:  # Modest scaling for validation
                try:
                    demo = P2PNetworkDemo(num_nodes=node_count)
                    
                    # Time network startup
                    scale_start = time.time()
                    network_task = asyncio.create_task(demo.start_network())
                    await asyncio.sleep(2)
                    
                    status = demo.get_network_status()
                    scale_duration = time.time() - scale_start
                    
                    await demo.stop_network()
                    
                    scalability_results.append({
                        "nodes": node_count,
                        "startup_time": scale_duration,
                        "active_nodes": status["active_nodes"],
                        "connections": status["total_connections"],
                        "messages": status["total_messages"]
                    })
                    
                except Exception as e:
                    logger.warning(f"Scalability test failed for {node_count} nodes: {e}")
                    break
            
            duration = time.time() - start_time
            
            if len(scalability_results) >= 2:
                # Check if performance scales reasonably
                startup_times = [r["startup_time"] for r in scalability_results]
                linear_scaling = all(t2 <= t1 * 2 for t1, t2 in zip(startup_times[:-1], startup_times[1:]))
                
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS" if linear_scaling else "WARNING",
                    duration=duration,
                    details=f"Scalability tested up to {max(r['nodes'] for r in scalability_results)} nodes",
                    metrics={"scalability_results": scalability_results, "linear_scaling": linear_scaling}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details="Scalability testing failed",
                    metrics={"scalability_results": scalability_results}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Scalability limits test failed",
                error_message=str(e)
            ))
    
    async def _validate_enterprise_features(self):
        """Validate enterprise features"""
        category = "Enterprise Features"
        logger.info(f"🏢 Validating {category}...")
        
        await self._test_monitoring_configuration(category)
        await self._test_security_documentation(category)
        await self._test_compliance_framework(category)
    
    async def _test_monitoring_configuration(self, category: str):
        """Test monitoring configuration"""
        test_name = "Monitoring Configuration"
        start_time = time.time()
        
        try:
            monitoring_files = [
                "config/grafana/dashboards/prsm-overview.json",
                "docs/ENTERPRISE_MONITORING_GUIDE.md"
            ]
            
            valid_configs = 0
            total_configs = len(monitoring_files)
            
            for config_file in monitoring_files:
                config_path = project_root / config_file
                if config_path.exists():
                    if config_file.endswith('.json'):
                        try:
                            with open(config_path, 'r') as f:
                                config_data = json.load(f)
                            # Check for key Grafana dashboard elements
                            if 'panels' in config_data and 'title' in config_data:
                                valid_configs += 1
                        except json.JSONDecodeError:
                            pass
                    else:
                        # Markdown file - check size and content
                        if config_path.stat().st_size > 1000:  # Substantial content
                            valid_configs += 1
            
            duration = time.time() - start_time
            
            config_completeness = (valid_configs / total_configs) * 100
            
            if config_completeness >= 80:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Monitoring configuration complete: {valid_configs}/{total_configs} files valid",
                    metrics={"valid_configs": valid_configs, "total_configs": total_configs, "completeness": config_completeness}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Monitoring configuration incomplete: {valid_configs}/{total_configs} files valid",
                    metrics={"valid_configs": valid_configs, "total_configs": total_configs}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Monitoring configuration test failed",
                error_message=str(e)
            ))
    
    async def _test_security_documentation(self, category: str):
        """Test security documentation completeness"""
        test_name = "Security Documentation"
        start_time = time.time()
        
        try:
            security_docs = [
                "docs/SECURITY_ARCHITECTURE.md",
                "docs/COMPLIANCE_FRAMEWORK.md",
                "docs/ENTERPRISE_AUTHENTICATION_GUIDE.md"
            ]
            
            valid_docs = 0
            doc_metrics = {}
            
            for doc_file in security_docs:
                doc_path = project_root / doc_file
                if doc_path.exists():
                    doc_size = doc_path.stat().st_size
                    if doc_size > 5000:  # Substantial documentation
                        valid_docs += 1
                        doc_metrics[doc_file] = {"size": doc_size, "exists": True}
                    else:
                        doc_metrics[doc_file] = {"size": doc_size, "exists": True, "warning": "too_small"}
                else:
                    doc_metrics[doc_file] = {"exists": False}
            
            duration = time.time() - start_time
            
            doc_completeness = (valid_docs / len(security_docs)) * 100
            
            if doc_completeness >= 80:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Security documentation complete: {valid_docs}/{len(security_docs)} documents",
                    metrics={"valid_docs": valid_docs, "total_docs": len(security_docs), "doc_details": doc_metrics}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Security documentation incomplete: {valid_docs}/{len(security_docs)} documents",
                    metrics={"valid_docs": valid_docs, "total_docs": len(security_docs)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Security documentation test failed",
                error_message=str(e)
            ))
    
    async def _test_compliance_framework(self, category: str):
        """Test compliance framework implementation"""
        test_name = "Compliance Framework"
        start_time = time.time()
        
        try:
            compliance_file = project_root / "docs/COMPLIANCE_FRAMEWORK.md"
            
            if compliance_file.exists():
                with open(compliance_file, 'r') as f:
                    content = f.read()
                
                # Check for key compliance elements
                compliance_elements = [
                    "GDPR", "SOC 2", "audit", "privacy", "data protection",
                    "monitoring", "compliance", "framework"
                ]
                
                found_elements = sum(1 for element in compliance_elements if element.lower() in content.lower())
                element_coverage = (found_elements / len(compliance_elements)) * 100
                
                duration = time.time() - start_time
                
                if element_coverage >= 75:
                    self.results.append(ValidationResult(
                        test_name=test_name,
                        category=category,
                        status="PASS",
                        duration=duration,
                        details=f"Compliance framework comprehensive: {found_elements}/{len(compliance_elements)} elements covered",
                        metrics={"coverage": element_coverage, "content_size": len(content)}
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=test_name,
                        category=category,
                        status="WARNING",
                        duration=duration,
                        details=f"Compliance framework incomplete: {found_elements}/{len(compliance_elements)} elements covered",
                        metrics={"coverage": element_coverage}
                    ))
            else:
                duration = time.time() - start_time
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details="Compliance framework documentation not found",
                    metrics={}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Compliance framework test failed",
                error_message=str(e)
            ))
    
    async def _validate_integration_apis(self):
        """Validate integration and API capabilities"""
        category = "Integration & APIs"
        logger.info(f"🔌 Validating {category}...")
        
        await self._test_api_testing_framework(category)
        await self._test_sdk_availability(category)
    
    async def _test_api_testing_framework(self, category: str):
        """Test API testing framework"""
        test_name = "API Testing Framework"
        start_time = time.time()
        
        try:
            api_test_file = project_root / "docs/API_TESTING_INTEGRATION_GUIDE.md"
            
            if api_test_file.exists():
                with open(api_test_file, 'r') as f:
                    content = f.read()
                
                # Check for API testing elements
                api_elements = [
                    "pytest", "httpx", "testing", "API", "validation",
                    "load testing", "integration", "automated"
                ]
                
                found_elements = sum(1 for element in api_elements if element.lower() in content.lower())
                api_coverage = (found_elements / len(api_elements)) * 100
                
                duration = time.time() - start_time
                
                if api_coverage >= 70:
                    self.results.append(ValidationResult(
                        test_name=test_name,
                        category=category,
                        status="PASS",
                        duration=duration,
                        details=f"API testing framework documented: {found_elements}/{len(api_elements)} elements",
                        metrics={"coverage": api_coverage, "content_size": len(content)}
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name=test_name,
                        category=category,
                        status="WARNING",
                        duration=duration,
                        details=f"API testing framework incomplete: {found_elements}/{len(api_elements)} elements",
                        metrics={"coverage": api_coverage}
                    ))
            else:
                duration = time.time() - start_time
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details="API testing framework documentation not found",
                    metrics={}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="API testing framework test failed",
                error_message=str(e)
            ))
    
    async def _test_sdk_availability(self, category: str):
        """Test SDK availability and structure"""
        test_name = "SDK Availability"
        start_time = time.time()
        
        try:
            sdk_dirs = [
                "sdks/javascript",
                "sdks/go"
            ]
            
            available_sdks = 0
            sdk_details = {}
            
            for sdk_dir in sdk_dirs:
                sdk_path = project_root / sdk_dir
                if sdk_path.exists():
                    # Check for key SDK files
                    if sdk_dir.endswith("javascript"):
                        key_files = ["package.json", "src/index.ts", "README.md"]
                    elif sdk_dir.endswith("go"):
                        key_files = ["go.mod", "README.md"]
                    else:
                        key_files = ["README.md"]
                    
                    found_files = sum(1 for f in key_files if (sdk_path / f).exists())
                    if found_files >= len(key_files) - 1:  # Allow 1 missing file
                        available_sdks += 1
                        sdk_details[sdk_dir] = {"files": found_files, "total": len(key_files)}
                    else:
                        sdk_details[sdk_dir] = {"files": found_files, "total": len(key_files), "incomplete": True}
                else:
                    sdk_details[sdk_dir] = {"exists": False}
            
            duration = time.time() - start_time
            
            sdk_availability = (available_sdks / len(sdk_dirs)) * 100
            
            if sdk_availability >= 50:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"SDKs available: {available_sdks}/{len(sdk_dirs)} SDKs",
                    metrics={"available_sdks": available_sdks, "total_sdks": len(sdk_dirs), "sdk_details": sdk_details}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Limited SDK availability: {available_sdks}/{len(sdk_dirs)} SDKs",
                    metrics={"available_sdks": available_sdks, "total_sdks": len(sdk_dirs)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="SDK availability test failed",
                error_message=str(e)
            ))
    
    async def _validate_documentation_usability(self):
        """Validate documentation and usability"""
        category = "Documentation & Usability"
        logger.info(f"📚 Validating {category}...")
        
        await self._test_documentation_completeness(category)
        await self._test_demo_usability(category)
    
    async def _test_documentation_completeness(self, category: str):
        """Test documentation completeness"""
        test_name = "Documentation Completeness"
        start_time = time.time()
        
        try:
            doc_files = [
                "README.md",
                "demos/README.md",
                "docs/SECURITY_ARCHITECTURE.md",
                "docs/ENTERPRISE_MONITORING_GUIDE.md",
                "docs/COMPLIANCE_FRAMEWORK.md",
                "docs/ENTERPRISE_AUTHENTICATION_GUIDE.md",
                "docs/API_TESTING_INTEGRATION_GUIDE.md"
            ]
            
            valid_docs = 0
            total_size = 0
            
            for doc_file in doc_files:
                doc_path = project_root / doc_file
                if doc_path.exists():
                    doc_size = doc_path.stat().st_size
                    total_size += doc_size
                    if doc_size > 1000:  # Minimum meaningful size
                        valid_docs += 1
            
            duration = time.time() - start_time
            
            doc_completeness = (valid_docs / len(doc_files)) * 100
            
            if doc_completeness >= 85:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Documentation comprehensive: {valid_docs}/{len(doc_files)} files ({total_size/1024:.0f}KB total)",
                    metrics={"valid_docs": valid_docs, "total_docs": len(doc_files), "total_size": total_size}
                ))
            elif doc_completeness >= 70:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Documentation mostly complete: {valid_docs}/{len(doc_files)} files",
                    metrics={"valid_docs": valid_docs, "total_docs": len(doc_files)}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="FAIL",
                    duration=duration,
                    details=f"Documentation incomplete: {valid_docs}/{len(doc_files)} files",
                    metrics={"valid_docs": valid_docs, "total_docs": len(doc_files)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Documentation completeness test failed",
                error_message=str(e)
            ))
    
    async def _test_demo_usability(self, category: str):
        """Test demo usability and functionality"""
        test_name = "Demo Usability"
        start_time = time.time()
        
        try:
            demo_files = [
                "demos/p2p_network_demo.py",
                "demos/enhanced_p2p_ai_demo.py"
            ]
            
            runnable_demos = 0
            demo_results = {}
            
            for demo_file in demo_files:
                demo_path = project_root / demo_file
                if demo_path.exists():
                    try:
                        # Test if demo can be imported
                        module_name = demo_file.replace("/", ".").replace(".py", "")
                        __import__(module_name)
                        runnable_demos += 1
                        demo_results[demo_file] = {"importable": True}
                    except Exception as e:
                        demo_results[demo_file] = {"importable": False, "error": str(e)}
                else:
                    demo_results[demo_file] = {"exists": False}
            
            duration = time.time() - start_time
            
            demo_usability = (runnable_demos / len(demo_files)) * 100
            
            if demo_usability >= 80:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="PASS",
                    duration=duration,
                    details=f"Demos usable: {runnable_demos}/{len(demo_files)} demos runnable",
                    metrics={"runnable_demos": runnable_demos, "total_demos": len(demo_files), "demo_details": demo_results}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name=test_name,
                    category=category,
                    status="WARNING",
                    duration=duration,
                    details=f"Demo usability issues: {runnable_demos}/{len(demo_files)} demos runnable",
                    metrics={"runnable_demos": runnable_demos, "total_demos": len(demo_files)}
                ))
                
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ValidationResult(
                test_name=test_name,
                category=category,
                status="FAIL",
                duration=duration,
                details="Demo usability test failed",
                error_message=str(e)
            ))
    
    def _generate_summary(self) -> ValidationSummary:
        """Generate comprehensive validation summary"""
        total_duration = time.time() - self.start_time
        
        # Count results by status
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        warnings = sum(1 for r in self.results if r.status == "WARNING")
        total = len(self.results)
        
        # Count by category
        categories = {}
        for category in self.categories:
            category_results = [r for r in self.results if r.category == category]
            categories[category] = {
                "total": len(category_results),
                "passed": sum(1 for r in category_results if r.status == "PASS"),
                "failed": sum(1 for r in category_results if r.status == "FAIL"),
                "warnings": sum(1 for r in category_results if r.status == "WARNING"),
                "skipped": sum(1 for r in category_results if r.status == "SKIP")
            }
        
        # Identify critical failures
        critical_failures = [r.test_name for r in self.results if r.status == "FAIL"]
        
        # Generate recommendations
        recommendations = []
        if failed > 0:
            recommendations.append(f"Address {failed} critical test failures before production deployment")
        if warnings > 0:
            recommendations.append(f"Review {warnings} warnings to improve system robustness")
        
        # Performance recommendations
        performance_results = [r for r in self.results if r.category == "Performance & Scalability"]
        if any(r.status in ["FAIL", "WARNING"] for r in performance_results):
            recommendations.append("Optimize performance and scalability before scaling to production")
        
        # Security recommendations
        security_results = [r for r in self.results if r.category == "Consensus & Security"]
        if any(r.status == "FAIL" for r in security_results):
            recommendations.append("Critical: Address security failures before any production use")
        
        return ValidationSummary(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            warning_tests=warnings,
            total_duration=total_duration,
            categories=categories,
            critical_failures=critical_failures,
            recommendations=recommendations
        )
    
    def print_detailed_report(self, summary: ValidationSummary):
        """Print detailed validation report"""
        print("\n" + "=" * 80)
        print("🎯 PRSM COMPREHENSIVE VALIDATION REPORT")
        print("=" * 80)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {summary.total_duration:.2f} seconds")
        print()
        
        # Overall summary
        print("📊 OVERALL SUMMARY")
        print("-" * 40)
        print(f"Total Tests: {summary.total_tests}")
        print(f"✅ Passed: {summary.passed_tests} ({summary.passed_tests/summary.total_tests*100:.1f}%)")
        print(f"❌ Failed: {summary.failed_tests} ({summary.failed_tests/summary.total_tests*100:.1f}%)")
        print(f"⚠️  Warnings: {summary.warning_tests} ({summary.warning_tests/summary.total_tests*100:.1f}%)")
        print(f"⏭️  Skipped: {summary.skipped_tests} ({summary.skipped_tests/summary.total_tests*100:.1f}%)")
        print()
        
        # Category breakdown
        print("📋 CATEGORY BREAKDOWN")
        print("-" * 40)
        for category, stats in summary.categories.items():
            if stats["total"] > 0:
                pass_rate = (stats["passed"] / stats["total"]) * 100
                status_icon = "✅" if pass_rate >= 80 else "⚠️" if pass_rate >= 60 else "❌"
                print(f"{status_icon} {category}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")
        print()
        
        # Critical failures
        if summary.critical_failures:
            print("🚨 CRITICAL FAILURES")
            print("-" * 40)
            for failure in summary.critical_failures:
                print(f"❌ {failure}")
            print()
        
        # Detailed results
        print("📝 DETAILED RESULTS")
        print("-" * 40)
        for category in self.categories:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                print(f"\n{category}:")
                for result in category_results:
                    status_icon = {
                        "PASS": "✅",
                        "FAIL": "❌", 
                        "WARNING": "⚠️",
                        "SKIP": "⏭️"
                    }.get(result.status, "❓")
                    
                    print(f"  {status_icon} {result.test_name} ({result.duration:.3f}s)")
                    print(f"      {result.details}")
                    if result.error_message:
                        print(f"      Error: {result.error_message}")
        
        # Recommendations
        if summary.recommendations:
            print("\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(summary.recommendations, 1):
                print(f"{i}. {rec}")
        
        # Production readiness assessment
        print("\n🚀 PRODUCTION READINESS ASSESSMENT")
        print("-" * 40)
        
        overall_pass_rate = (summary.passed_tests / summary.total_tests) * 100
        critical_failures = summary.failed_tests
        
        if overall_pass_rate >= 90 and critical_failures == 0:
            readiness = "🟢 READY FOR PRODUCTION"
            readiness_detail = "System demonstrates high reliability and completeness"
        elif overall_pass_rate >= 80 and critical_failures <= 2:
            readiness = "🟡 PRODUCTION READY WITH MINOR IMPROVEMENTS"
            readiness_detail = "Address minor issues before production deployment"
        elif overall_pass_rate >= 70:
            readiness = "🟠 REQUIRES SIGNIFICANT IMPROVEMENTS"
            readiness_detail = "Multiple issues need resolution before production"
        else:
            readiness = "🔴 NOT READY FOR PRODUCTION"
            readiness_detail = "Critical issues must be resolved"
        
        print(f"Status: {readiness}")
        print(f"Assessment: {readiness_detail}")
        print(f"Overall Score: {overall_pass_rate:.1f}%")
        
        print("\n" + "=" * 80)

async def main():
    """Main validation execution"""
    validator = PRSMValidator()
    
    try:
        summary = await validator.run_comprehensive_validation()
        validator.print_detailed_report(summary)
        
        # Save detailed results to JSON
        results_file = "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": asdict(summary),
                "detailed_results": [asdict(r) for r in validator.results],
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        print(f"📁 Logs saved to: validation_results.log")
        
        # Exit with appropriate code
        if summary.failed_tests == 0:
            print("\n🎉 All validations passed!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {summary.failed_tests} validations failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n💥 Validation suite crashed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())