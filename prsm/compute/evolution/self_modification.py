"""
Self-Modification Infrastructure

Core interfaces and validation systems for safe self-improving components.
Implements DGM-style empirical validation with rollback capabilities.
"""

import asyncio
import logging
import json
import copy
import pickle
import gzip
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import uuid
import traceback

from .models import (
    ComponentType, SafetyStatus, RiskLevel, ImpactLevel,
    ModificationProposal, SafetyValidationResult, ModificationResult,
    EvaluationResult, Checkpoint
)
# Optional imports for safety and governance
try:
    from prsm.core.safety.monitor import SafetyMonitor
except ImportError:
    SafetyMonitor = None

try:
    from prsm.economy.governance.voting import GovernanceSystem
except ImportError:
    GovernanceSystem = None


logger = logging.getLogger(__name__)

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = Path.home() / ".prsm" / "checkpoints"
MAX_CHECKPOINTS_PER_COMPONENT = 10


class SelfModifyingComponent(ABC):
    """
    Abstract base class for components that can modify themselves.
    Implements the core self-modification interface with safety constraints.
    """

    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
        self.modification_history: List[ModificationResult] = []
        self.checkpoints: List[Checkpoint] = []
        self.baseline_performance: float = 0.0
        self.safety_monitor = SafetyMonitor() if SafetyMonitor else None

        # Self-modification configuration
        self.max_modification_attempts = 3
        self.performance_threshold = 0.05  # 5% improvement required
        self.rollback_timeout_seconds = 300  # 5 minutes

        # Checkpoint storage
        self._checkpoint_dir = DEFAULT_CHECKPOINT_DIR / component_id
        self._checkpoint_metadata: List[Dict[str, Any]] = []

    @abstractmethod
    async def propose_modification(self, evaluation_logs: List[EvaluationResult]) -> Optional[ModificationProposal]:
        """
        Analyze performance and propose a self-modification.

        Args:
            evaluation_logs: Recent evaluation results to analyze

        Returns:
            ModificationProposal or None if no improvement needed
        """
        pass

    @abstractmethod
    async def apply_modification(self, modification: ModificationProposal) -> ModificationResult:
        """
        Apply a validated modification to this component.

        Args:
            modification: The modification to apply

        Returns:
            ModificationResult with success status and metrics
        """
        pass

    @abstractmethod
    async def validate_modification(self, modification: ModificationProposal) -> bool:
        """
        Validate that a modification preserves core functionality.

        Args:
            modification: The modification to validate

        Returns:
            True if modification is valid, False otherwise
        """
        pass

    async def create_checkpoint(self, label: str = "") -> Checkpoint:
        """
        Create a checkpoint of current component state for rollback.

        Serializes component state:
        - Captures self.__dict__ (shallow copy)
        - Excludes non-serializable items (locks, sockets, open files)
        - Uses pickle.dumps() → compress with gzip
        - Writes to ~/.prsm/checkpoints/{component_id}/{checkpoint_id}.pkl.gz
        - Stores checkpoint metadata (id, label, timestamp, size) in self._checkpoints list

        Args:
            label: Optional label for the checkpoint

        Returns:
            Checkpoint containing component state snapshot
        """
        try:
            checkpoint_id = f"ckpt_{uuid.uuid4().hex[:8]}_{int(datetime.now(timezone.utc).timestamp())}"

            # Capture serializable state
            state_snapshot = {}
            non_serializable_keys = []

            for key, value in self.__dict__.items():
                # Skip non-serializable items
                if key.startswith('_') and key.endswith('_lock'):
                    non_serializable_keys.append(key)
                    continue
                if key in ('safety_monitor', '_loop', '_executor'):
                    non_serializable_keys.append(key)
                    continue
                if callable(value):
                    non_serializable_keys.append(key)
                    continue

                try:
                    # Test if value is serializable
                    pickle.dumps(value)
                    state_snapshot[key] = copy.deepcopy(value)
                except (pickle.PicklingError, TypeError, AttributeError):
                    non_serializable_keys.append(key)

            # Capture configuration snapshot
            config_snapshot = {
                'component_id': self.component_id,
                'component_type': self.component_type.value if hasattr(self.component_type, 'value') else str(self.component_type),
                'max_modification_attempts': self.max_modification_attempts,
                'performance_threshold': self.performance_threshold,
                'rollback_timeout_seconds': self.rollback_timeout_seconds,
            }

            # Serialize and compress
            serialized = pickle.dumps({
                'state': state_snapshot,
                'config': config_snapshot,
                'label': label,
            })
            compressed = gzip.compress(serialized, compresslevel=6)

            # Save to disk
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self._checkpoint_dir / f"{checkpoint_id}.pkl.gz"

            with open(checkpoint_path, 'wb') as f:
                f.write(compressed)

            # Create checkpoint object
            checkpoint = Checkpoint(
                id=checkpoint_id,
                component_id=self.component_id,
                component_type=self.component_type,
                state_snapshot=state_snapshot,
                configuration_snapshot=config_snapshot,
                timestamp=datetime.now(timezone.utc),
                storage_location=str(checkpoint_path),
            )

            # Store metadata
            self._checkpoint_metadata.append({
                'checkpoint_id': checkpoint_id,
                'label': label,
                'timestamp': checkpoint.timestamp.isoformat(),
                'size_bytes': len(compressed),
                'storage_location': str(checkpoint_path),
            })

            # Enforce max checkpoints limit (evict oldest)
            while len(self._checkpoint_metadata) > MAX_CHECKPOINTS_PER_COMPONENT:
                oldest = self._checkpoint_metadata.pop(0)
                oldest_path = Path(oldest['storage_location'])
                if oldest_path.exists():
                    oldest_path.unlink()

            # Add to checkpoints list
            self.checkpoints.append(checkpoint)

            logger.info(f"Created checkpoint {checkpoint_id}",
                       component_id=self.component_id,
                       label=label,
                       size_bytes=len(compressed))

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise

    async def rollback_to_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Rollback component to a previous checkpoint state.

        Steps:
        1. Find checkpoint file
        2. Decompress and deserialize with pickle.loads()
        3. Restore self.__dict__ from checkpoint state
        4. Validate restored state with validate_modification()
        5. Log rollback event

        Args:
            checkpoint: The checkpoint to rollback to

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            checkpoint_path = Path(checkpoint.storage_location)

            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False

            # Read and decompress
            with open(checkpoint_path, 'rb') as f:
                compressed = f.read()

            decompressed = gzip.decompress(compressed)
            checkpoint_data = pickle.loads(decompressed)

            # Restore state
            state_snapshot = checkpoint_data.get('state', {})

            for key, value in state_snapshot.items():
                if key not in ('checkpoints', '_checkpoint_metadata', '_checkpoint_dir'):
                    try:
                        setattr(self, key, copy.deepcopy(value))
                    except Exception as e:
                        logger.warning(f"Failed to restore attribute {key}: {e}")

            # Validate restored state
            try:
                # Create a dummy modification for validation
                validation_proposal = ModificationProposal(
                    id=f"rollback_{checkpoint.id}",
                    component_id=self.component_id,
                    component_type=self.component_type,
                    modification_type="rollback",
                    description=f"Rollback to checkpoint {checkpoint.id}",
                    code_changes={},
                    config_changes=checkpoint.configuration_snapshot,
                    risk_level=RiskLevel.LOW,
                    impact_level=ImpactLevel.LOW,
                )

                validation_passed = await self.validate_modification(validation_proposal)

                if not validation_passed:
                    logger.warning(f"Rollback validation failed for checkpoint {checkpoint.id}")

            except Exception as e:
                logger.warning(f"Rollback validation error: {e}")

            # Log rollback event
            logger.info(f"Rolled back to checkpoint {checkpoint.id}",
                       component_id=self.component_id,
                       checkpoint_label=checkpoint_data.get('label', ''))

            return True

        except Exception as e:
            logger.error(f"Failed to rollback to checkpoint: {e}")
            return False

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints for this component.

        Returns:
            List of checkpoint metadata dicts
        """
        return self._checkpoint_metadata.copy()

    @abstractmethod
    async def evaluate_performance(self) -> EvaluationResult:
        """
        Evaluate current component performance.

        Returns:
            EvaluationResult with current performance metrics
        """
        pass
    
    async def self_improve(self, evaluation_logs: List[EvaluationResult]) -> Optional[ModificationResult]:
        """
        Execute complete self-improvement cycle with safety validation.
        
        Args:
            evaluation_logs: Recent performance evaluations
            
        Returns:
            ModificationResult or None if no improvement made
        """
        logger.info(f"Starting self-improvement cycle for {self.component_id}")
        
        try:
            # Step 1: Analyze performance and propose modification
            modification_proposal = await self.propose_modification(evaluation_logs)
            if not modification_proposal:
                logger.info("No modification proposed")
                return None
            
            # Step 2: Validate safety and constraints
            safety_result = await self._validate_modification_safety(modification_proposal)
            if not safety_result.passed:
                logger.warning(f"Safety validation failed: {safety_result.failed_checks}")
                return ModificationResult(
                    modification_id=modification_proposal.id,
                    success=False,
                    error_message=f"Safety validation failed: {safety_result.failed_checks}",
                    executor_id=self.component_id,
                    timestamp=datetime.utcnow()
                )
            
            # Step 3: Create checkpoint before modification
            checkpoint = await self.create_checkpoint()
            self.checkpoints.append(checkpoint)
            
            # Step 4: Apply modification with monitoring
            result = await self._apply_modification_with_monitoring(modification_proposal)
            
            # Step 5: Validate result and rollback if necessary
            if not result.success or result.performance_delta < -self.performance_threshold:
                logger.warning("Modification failed or degraded performance, rolling back")
                rollback_success = await self.rollback_to_checkpoint(checkpoint)
                result.rollback_required = True
                result.rollback_successful = rollback_success
            
            # Record modification in history
            self.modification_history.append(result)
            
            logger.info(f"Self-improvement cycle completed: success={result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Self-improvement cycle failed: {e}")
            return ModificationResult(
                modification_id=getattr(modification_proposal, 'id', 'unknown'),
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
                executor_id=self.component_id,
                timestamp=datetime.utcnow()
            )
    
    async def _validate_modification_safety(self, modification: ModificationProposal) -> SafetyValidationResult:
        """Comprehensive safety validation of modification."""
        validation_result = SafetyValidationResult(
            modification_id=modification.id,
            passed=True,
            risk_level=modification.risk_level,
            validator_id=self.component_id
        )
        
        # Capability bounds check
        capability_check = await self._check_capability_bounds(modification)
        validation_result.capability_bounds_check = capability_check
        if not capability_check:
            validation_result.failed_checks.append("capability_bounds")
            validation_result.passed = False
        
        # Resource limits check
        resource_check = await self._check_resource_limits(modification)
        validation_result.resource_limits_check = resource_check
        if not resource_check:
            validation_result.failed_checks.append("resource_limits")
            validation_result.passed = False
        
        # Behavioral constraints check
        behavioral_check = await self._check_behavioral_constraints(modification)
        validation_result.behavioral_constraints_check = behavioral_check
        if not behavioral_check:
            validation_result.failed_checks.append("behavioral_constraints")
            validation_result.passed = False
        
        # Impact assessment check
        impact_check = await self._assess_modification_impact(modification)
        validation_result.impact_assessment_check = impact_check
        if not impact_check:
            validation_result.failed_checks.append("impact_assessment")
            validation_result.passed = False
        
        # Component-specific validation
        component_check = await self.validate_modification(modification)
        if not component_check:
            validation_result.failed_checks.append("component_specific")
            validation_result.passed = False
        
        return validation_result
    
    async def _apply_modification_with_monitoring(self, modification: ModificationProposal) -> ModificationResult:
        """Apply modification with real-time safety monitoring."""
        start_time = datetime.utcnow()
        
        # Start monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_modification_execution(modification)
        )
        
        try:
            # Record pre-modification performance
            pre_performance = await self.evaluate_performance()
            
            # Apply the modification
            result = await self.apply_modification(modification)
            
            # Record post-modification performance
            post_performance = await self.evaluate_performance()
            
            # Calculate performance delta
            result.performance_before = pre_performance.performance_score
            result.performance_after = post_performance.performance_score
            result.performance_delta = post_performance.performance_score - pre_performance.performance_score
            
            # Check if functionality is preserved
            result.functionality_preserved = await self._verify_functionality_preserved()
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time_seconds = execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Modification application failed: {e}")
            return ModificationResult(
                modification_id=modification.id,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
                execution_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                executor_id=self.component_id,
                timestamp=datetime.utcnow()
            )
        finally:
            # Stop monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_modification_execution(self, modification: ModificationProposal):
        """Monitor modification execution for safety violations."""
        try:
            while True:
                await asyncio.sleep(1.0)  # Check every second
                
                # Check for safety violations (if safety monitor available)
                if self.safety_monitor:
                    safety_status = await self.safety_monitor.check_component_safety(self.component_id)
                    if safety_status.has_violations:
                        logger.error(f"Safety violation detected during modification: {safety_status.violations}")
                        # Emergency shutdown would be triggered here
                        break
                
                # Check resource usage
                resource_usage = await self._get_resource_usage()
                if self._exceeds_resource_limits(resource_usage, modification):
                    logger.error(f"Resource limits exceeded during modification: {resource_usage}")
                    break
                    
        except asyncio.CancelledError:
            logger.debug("Modification monitoring cancelled")
    
    async def _check_capability_bounds(self, modification: ModificationProposal) -> bool:
        """Check if modification stays within allowed capability bounds."""
        # Risk level check
        if modification.risk_level == RiskLevel.CRITICAL:
            return False

        # Type allowlist check (if configured)
        allowed = getattr(self, '_allowed_modification_types', set())
        if allowed and modification.modification_type not in allowed:
            logger.warning(
                "Modification type outside capability bounds",
                modification_type=modification.modification_type,
                allowed=list(allowed),
            )
            return False
        return True
    
    async def _check_resource_limits(self, modification: ModificationProposal) -> bool:
        """Check if modification respects resource consumption limits."""
        # Check compute requirements
        if modification.compute_requirements.get('cpu_cores', 0) > 16:
            return False
        if modification.compute_requirements.get('memory_gb', 0) > 64:
            return False
        
        # Check storage requirements
        if modification.storage_requirements.get('disk_gb', 0) > 1000:
            return False
        
        return True
    
    async def _check_behavioral_constraints(self, modification: ModificationProposal) -> bool:
        """Check if modification maintains required behavioral constraints."""
        # Reject modifications that claim to change the component interface
        breaking_keywords = ['interface_change', 'api_break', 'signature_change']
        tags = modification.tags if hasattr(modification, 'tags') else []
        for keyword in breaking_keywords:
            if keyword in tags:
                logger.warning(
                    "Modification marked as interface-breaking — failing behavioral constraint check",
                    modification_id=str(modification.id),
                )
                return False
        return True
    
    async def _assess_modification_impact(self, modification: ModificationProposal) -> bool:
        """Assess the potential impact of the modification."""
        # High impact modifications require additional validation
        if modification.impact_level == ImpactLevel.CRITICAL:
            # Would require governance approval
            return False
        
        # Check estimated performance impact
        if modification.estimated_performance_impact < -0.2:  # More than 20% degradation
            return False
        
        return True
    
    async def _verify_functionality_preserved(self) -> bool:
        """Verify that core functionality is preserved after modification."""
        try:
            # Attempt to verify core PRSM module importability as a smoke test
            import importlib.util
            core_modules = ['prsm.core.database', 'prsm.core.models']
            for mod_name in core_modules:
                spec = importlib.util.find_spec(mod_name)
                if spec is None:
                    logger.error(f"Core module no longer importable after modification: {mod_name}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Functionality verification failed: {e}")
            return False
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage metrics using psutil."""
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()

            try:
                disk = psutil.disk_io_counters()
                disk_read_mb = (disk.read_bytes / 1024 / 1024) if disk else 0.0
                disk_write_mb = (disk.write_bytes / 1024 / 1024) if disk else 0.0
            except Exception:
                disk_read_mb = disk_write_mb = 0.0

            try:
                net = psutil.net_io_counters()
                net_sent_mb = (net.bytes_sent / 1024 / 1024) if net else 0.0
                net_recv_mb = (net.bytes_recv / 1024 / 1024) if net else 0.0
            except Exception:
                net_sent_mb = net_recv_mb = 0.0

            return {
                'cpu_percent': cpu,
                'memory_percent': mem.percent,
                'memory_mb': mem.used / 1024 / 1024,
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'network_sent_mb': net_sent_mb,
                'network_recv_mb': net_recv_mb,
            }
        except Exception as e:
            logger.warning(f"Resource monitoring unavailable: {e}")
            return {
                'cpu_percent': 0.0, 'memory_percent': 0.0, 'memory_mb': 0.0,
                'disk_read_mb': 0.0, 'disk_write_mb': 0.0,
                'network_sent_mb': 0.0, 'network_recv_mb': 0.0,
            }
    
    def _exceeds_resource_limits(self, usage: Dict[str, float], modification: ModificationProposal) -> bool:
        """Check if current resource usage exceeds limits."""
        try:
            import psutil

            # Hard safety ceilings
            if usage.get('cpu_percent', 0) > 90:
                return True
            if usage.get('memory_percent', 0) > 90:
                return True

            # Modification-specific ceilings
            required_memory_gb = modification.compute_requirements.get('memory_gb', 0)
            available_memory_gb = (psutil.virtual_memory().available / 1024 / 1024 / 1024)
            if required_memory_gb > available_memory_gb:
                return True

            return False
        except Exception:
            # If we can't check, be conservative
            return False


class ModificationValidator:
    """
    Validates modifications for safety and correctness.
    Implements comprehensive validation pipeline for self-modifications.
    """
    
    def __init__(self, governance_system: Optional[Any] = None):
        self.governance_system = governance_system
        self.validation_cache: Dict[str, SafetyValidationResult] = {}
        
    async def validate_syntax(self, modification: ModificationProposal) -> bool:
        """Validate syntax of code changes in modification."""
        try:
            # Validate JSON syntax for configuration changes
            if modification.config_changes:
                json.dumps(modification.config_changes)
            
            # Validate Python syntax for code changes
            if modification.code_changes:
                for file_path, code_content in modification.code_changes.items():
                    if isinstance(code_content, str):
                        compile(code_content, file_path, 'exec')
            
            return True
            
        except (json.JSONDecodeError, SyntaxError) as e:
            logger.error(f"Syntax validation failed: {e}")
            return False
    
    async def validate_functionality(self, modification: ModificationProposal) -> bool:
        """Validate that modification preserves core functionality."""
        # This would implement comprehensive functionality testing
        # For now, return True for basic modifications
        return modification.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
    
    async def validate_safety_bounds(self, modification: ModificationProposal) -> bool:
        """Validate that modification stays within safety bounds."""
        # Check safety considerations
        if "recursive_self_modification" in modification.safety_considerations:
            # Prevent unbounded recursive modification
            return False
        
        if "capability_expansion" in modification.safety_considerations:
            # Require governance approval for capability expansion
            if self.governance_system:
                return await self.governance_system.approve_capability_expansion(modification.id)
            return False
        
        return True
    
    async def validate_performance_regression(self, modification: ModificationProposal) -> bool:
        """Validate that modification won't cause severe performance regression."""
        # If estimated impact is too negative, reject
        return modification.estimated_performance_impact > -0.5  # Max 50% degradation
    
    async def comprehensive_validation(self, modification: ModificationProposal) -> SafetyValidationResult:
        """Run comprehensive validation pipeline."""
        # Check cache first
        cache_key = f"{modification.id}_{hash(str(modification.dict()))}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        result = SafetyValidationResult(
            modification_id=modification.id,
            passed=True,
            risk_level=modification.risk_level,
            validator_id="ModificationValidator"
        )
        
        # Run all validation checks
        checks = [
            ("syntax", self.validate_syntax(modification)),
            ("functionality", self.validate_functionality(modification)),
            ("safety_bounds", self.validate_safety_bounds(modification)),
            ("performance_regression", self.validate_performance_regression(modification))
        ]
        
        for check_name, check_coro in checks:
            try:
                check_result = await check_coro
                if not check_result:
                    result.failed_checks.append(check_name)
                    result.passed = False
            except Exception as e:
                logger.error(f"Validation check {check_name} failed: {e}")
                result.failed_checks.append(check_name)
                result.passed = False
        
        # Add recommendations based on failed checks
        if result.failed_checks:
            result.recommendations = self._generate_recommendations(result.failed_checks)
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        return result
    
    def _generate_recommendations(self, failed_checks: List[str]) -> List[str]:
        """Generate recommendations based on failed validation checks."""
        recommendations = []
        
        if "syntax" in failed_checks:
            recommendations.append("Fix syntax errors in code changes")
        
        if "functionality" in failed_checks:
            recommendations.append("Ensure modification preserves core functionality")
        
        if "safety_bounds" in failed_checks:
            recommendations.append("Reduce modification scope to stay within safety bounds")
        
        if "performance_regression" in failed_checks:
            recommendations.append("Optimize modification to reduce performance impact")
        
        return recommendations


class EmergencyShutdownSystem:
    """
    Emergency system for handling critical failures during self-modification.
    Provides immediate intervention capabilities for safety violations.
    """
    
    def __init__(self):
        self.shutdown_triggers: List[Callable] = []
        self.emergency_contacts: List[str] = []
        self.incident_log: List[Dict[str, Any]] = []
        
    async def report_modification_failure(
        self, 
        modification: ModificationProposal, 
        error: Exception
    ):
        """Report critical modification failure and take emergency action."""
        incident = {
            'timestamp': datetime.utcnow().isoformat(),
            'modification_id': modification.id,
            'component_type': modification.component_type.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'risk_level': modification.risk_level.value,
            'impact_level': modification.impact_level.value
        }
        
        self.incident_log.append(incident)
        logger.critical(f"Emergency shutdown triggered: {incident}")
        
        # Trigger emergency procedures
        if modification.risk_level == RiskLevel.CRITICAL:
            await self._trigger_emergency_shutdown()
        
        # Notify emergency contacts
        await self._notify_emergency_contacts(incident)
    
    async def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown of all self-modification activities."""
        logger.critical("EMERGENCY SHUTDOWN: All self-modification halted")
        
        # Execute all shutdown triggers
        for trigger in self.shutdown_triggers:
            try:
                await trigger()
            except Exception as e:
                logger.error(f"Emergency shutdown trigger failed: {e}")
    
    async def _notify_emergency_contacts(self, incident: Dict[str, Any]):
        """Notify emergency contacts of critical incident."""
        # This would implement actual notification system
        logger.critical(f"Emergency notification sent: {incident}")
        
    def add_shutdown_trigger(self, trigger: Callable):
        """Add emergency shutdown trigger."""
        self.shutdown_triggers.append(trigger)
        
    def add_emergency_contact(self, contact: str):
        """Add emergency contact."""
        self.emergency_contacts.append(contact)