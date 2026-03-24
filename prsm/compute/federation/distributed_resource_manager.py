"""
PRSM Distributed Resource Manager
================================

Advanced resource management system for millions of distributed nodes supporting:
- User-controlled resource contribution settings
- Real-time resource verification and monitoring  
- Intelligent resource allocation and scheduling
- Trust-but-verify resource validation
- Economic incentives aligned with actual contributions

This system enables PRSM to scale to millions of nodes while maintaining
performance, reliability, and fair compensation for resource contributors.
"""

import asyncio
import time
import hashlib
import json
import random
import socket
import statistics
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

import aiohttp

import structlog
logger = structlog.get_logger(__name__)


# ============================================================================
# RESOURCE TYPES AND MEASUREMENT
# ============================================================================

class ResourceType(str, Enum):
    """Types of resources that nodes can contribute"""
    COMPUTE_CPU = "compute_cpu"           # CPU processing power
    COMPUTE_GPU = "compute_gpu"           # GPU processing power  
    COMPUTE_TPU = "compute_tpu"           # TPU/specialized AI hardware
    STORAGE_PERSISTENT = "storage_persistent"  # Long-term storage (SSD/HDD)
    STORAGE_MEMORY = "storage_memory"     # RAM/memory
    STORAGE_CACHE = "storage_cache"       # Fast cache storage
    BANDWIDTH_INGRESS = "bandwidth_ingress"    # Incoming network capacity
    BANDWIDTH_EGRESS = "bandwidth_egress"      # Outgoing network capacity
    SPECIALIZED_QUANTUM = "specialized_quantum"  # Quantum computing
    SPECIALIZED_EDGE = "specialized_edge"        # Edge computing devices


class ResourceMeasurement(str, Enum):
    """Standard units for resource measurement"""
    # Compute
    CPU_CORES = "cpu_cores"              # Number of CPU cores
    GPU_MEMORY_GB = "gpu_memory_gb"      # GPU memory in GB
    COMPUTE_UNITS = "compute_units"      # Standardized compute capacity
    
    # Storage  
    STORAGE_GB = "storage_gb"            # Storage in gigabytes
    MEMORY_GB = "memory_gb"              # Memory in gigabytes
    IOPS = "iops"                        # Input/output operations per second
    
    # Network
    MBPS = "mbps"                        # Megabits per second
    LATENCY_MS = "latency_ms"            # Network latency in milliseconds
    
    # Quality metrics
    UPTIME_PERCENTAGE = "uptime_percentage"      # Availability percentage
    RELIABILITY_SCORE = "reliability_score"     # Reliability index (0-1)


@dataclass
class ResourceSpec:
    """Specification for a specific resource type"""
    resource_type: ResourceType
    measurement_unit: ResourceMeasurement
    total_capacity: float                # Total available capacity
    allocated_capacity: float           # Currently allocated capacity
    reserved_capacity: float            # Reserved for system/user
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    verification_proofs: List[str] = field(default_factory=list)
    last_verified: Optional[datetime] = None
    verification_score: float = 1.0     # Trust score (0-1)


@dataclass
class NodeResourceProfile:
    """Complete resource profile for a network node"""
    node_id: str
    user_id: str
    node_type: str                       # micro, small, medium, large, massive
    geographic_region: str
    resources: Dict[ResourceType, ResourceSpec] = field(default_factory=dict)
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    reputation_score: float = 0.5       # Overall reputation (0-1)
    stake_amount: Decimal = Decimal('0') # Economic stake in FTNS
    contribution_settings: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# USER RESOURCE CONTROL INTERFACE
# ============================================================================

class ResourceContributionSettings(BaseModel):
    """User-configurable settings for resource contribution"""
    
    # Resource allocation percentages (0.0 to 1.0)
    cpu_allocation_percentage: float = Field(default=0.5, ge=0.0, le=1.0)
    gpu_allocation_percentage: float = Field(default=0.5, ge=0.0, le=1.0)
    storage_allocation_percentage: float = Field(default=0.3, ge=0.0, le=1.0)
    memory_allocation_percentage: float = Field(default=0.4, ge=0.0, le=1.0)
    bandwidth_allocation_percentage: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Operational constraints
    max_cpu_temperature: float = Field(default=80.0)  # Celsius
    max_power_consumption: float = Field(default=100.0)  # Watts
    allowed_time_windows: List[Dict[str, str]] = Field(default_factory=list)
    priority_level: int = Field(default=5, ge=1, le=10)  # 1=lowest, 10=highest
    
    # Economic settings
    minimum_hourly_rate: Decimal = Field(default=Decimal('0.1'))  # FTNS per hour
    automatic_scaling: bool = Field(default=True)
    market_participation: bool = Field(default=True)
    
    # Quality and reliability settings
    uptime_commitment: float = Field(default=0.95, ge=0.5, le=0.999)
    geographic_restrictions: List[str] = Field(default_factory=list)
    data_retention_days: int = Field(default=30)
    
    # Advanced settings
    specialized_capabilities: List[str] = Field(default_factory=list)
    security_level: str = Field(default="standard")  # standard, high, maximum
    redundancy_factor: float = Field(default=1.2)


# NVIDIA GPU Memory Bandwidth Lookup Table (GB/s)
# Source: NVIDIA published specifications per SKU
NVIDIA_MEMORY_BANDWIDTH = {
    # RTX 40 Series
    "RTX 4090": 1008,
    "RTX 4080": 717,
    "RTX 4080 SUPER": 768,
    "RTX 4070 Ti SUPER": 552,
    "RTX 4070 Ti": 504,
    "RTX 4070 SUPER": 480,
    "RTX 4070": 360,
    "RTX 4060 Ti": 288,
    "RTX 4060": 272,
    # RTX 30 Series
    "RTX 3090 Ti": 1008,
    "RTX 3090": 936,
    "RTX 3080 Ti": 912,
    "RTX 3080": 760,
    "RTX 3070 Ti": 608,
    "RTX 3070": 448,
    "RTX 3060 Ti": 448,
    "RTX 3060": 360,
    # RTX 20 Series
    "RTX 2080 Ti": 616,
    "RTX 2080 SUPER": 516,
    "RTX 2080": 448,
    "RTX 2070 SUPER": 448,
    "RTX 2070": 448,
    "RTX 2060 SUPER": 448,
    "RTX 2060": 336,
    # Data Center / Professional
    "A100": 2039,
    "A100-SXM4": 2039,
    "A100-PCIE": 1555,
    "A10": 600,
    "A10G": 600,
    "A30": 933,
    "A40": 696,
    "A2": 200,
    "V100": 900,
    "V100-SXM2": 900,
    "V100-PCIE": 897,
    "T4": 320,
    "L4": 300,
    "L40": 864,
    "H100": 3352,
    "H100-SXM5": 3352,
    "H100-PCIE": 2000,
    # Workstation
    "RTX 6000 Ada": 960,
    "RTX A6000": 768,
    "RTX A5000": 768,
    "RTX A4000": 448,
    "RTX A3000": 448,
    "Quadro RTX 8000": 624,
    "Quadro RTX 6000": 576,
}


class ResourceCapabilityDetector:
    """Automatically detect and benchmark node capabilities"""
    
    def __init__(self):
        self.benchmarks = {}
        self.detection_cache = {}
        # Disk benchmark cache with timestamps
        self._disk_read_cache: Optional[float] = None
        self._disk_read_cache_time: Optional[float] = None
        self._disk_write_cache: Optional[float] = None
        self._disk_write_cache_time: Optional[float] = None
        self._disk_cache_ttl: float = 3600.0  # Cache TTL in seconds
        # Network measurement cache with timestamps
        self._network_latency_cache: Optional[float] = None
        self._network_latency_cache_time: Optional[float] = None
        self._bandwidth_cache: Optional[Tuple[float, float]] = None
        self._bandwidth_cache_time: Optional[float] = None
        self._network_cache_ttl: float = 300.0  # Cache TTL in seconds (network changes more frequently)
        
    async def detect_system_resources(self) -> Dict[ResourceType, ResourceSpec]:
        """Automatically detect available system resources"""
        detected_resources = {}
        
        # CPU Detection
        cpu_spec = await self._detect_cpu_capabilities()
        if cpu_spec:
            detected_resources[ResourceType.COMPUTE_CPU] = cpu_spec
            
        # GPU Detection  
        gpu_spec = await self._detect_gpu_capabilities()
        if gpu_spec:
            detected_resources[ResourceType.COMPUTE_GPU] = gpu_spec
            
        # Storage Detection
        storage_specs = await self._detect_storage_capabilities()
        detected_resources.update(storage_specs)
        
        # Network Detection
        network_specs = await self._detect_network_capabilities()
        detected_resources.update(network_specs)
        
        # Memory Detection
        memory_spec = await self._detect_memory_capabilities()
        if memory_spec:
            detected_resources[ResourceType.STORAGE_MEMORY] = memory_spec
            
        return detected_resources
    
    async def _detect_cpu_capabilities(self) -> Optional[ResourceSpec]:
        """Detect CPU capabilities with benchmarking"""
        try:
            import psutil
            import multiprocessing
            
            # Basic CPU info
            cpu_count = multiprocessing.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Performance benchmark
            benchmark_score = await self._run_cpu_benchmark()
            
            return ResourceSpec(
                resource_type=ResourceType.COMPUTE_CPU,
                measurement_unit=ResourceMeasurement.CPU_CORES,
                total_capacity=float(cpu_count),
                allocated_capacity=0.0,
                reserved_capacity=1.0,  # Reserve 1 core for system
                quality_metrics={
                    "base_frequency_ghz": cpu_freq.current / 1000 if cpu_freq else 2.0,
                    "benchmark_score": benchmark_score,
                    "architecture": "x86_64"  # Would detect actual architecture
                }
            )
        except Exception as e:
            logger.warning("CPU detection failed", error=str(e))
            return None
    
    async def _detect_gpu_capabilities(self) -> Optional[ResourceSpec]:
        """
        Detect GPU capabilities using multiple fallback methods.

        Priority order:
        1. pynvml (NVIDIA Management Library) - most accurate for NVIDIA
        2. amdsmi (AMD System Management Interface) - for AMD GPUs
        3. torch.cuda (PyTorch) - good for ML workloads
        4. Return None if no GPU detected

        Returns:
            ResourceSpec with GPU details, or None if no GPU available
        """
        # Try pynvml first (NVIDIA - most detailed)
        gpu_info = self._get_gpu_info_via_pynvml()

        # Try AMD GPU detection if NVIDIA not found
        if gpu_info is None:
            gpu_info = self._get_gpu_info_via_amdsmi()

        # Fall back to torch.cuda if neither found
        if gpu_info is None:
            gpu_info = self._get_gpu_info_via_torch()

        if gpu_info is None:
            # No GPU detected
            return None

        return ResourceSpec(
            resource_type=ResourceType.COMPUTE_GPU,
            measurement_unit=ResourceMeasurement.GPU_MEMORY_GB,
            total_capacity=gpu_info["memory_gb"],
            allocated_capacity=0.0,
            reserved_capacity=1.0,  # Reserve 1GB for system
            quality_metrics={
                "cuda_cores": gpu_info["cuda_cores"],
                "memory_bandwidth": gpu_info["memory_bandwidth"],
                "compute_capability": gpu_info["compute_capability"],
                "device_name": gpu_info.get("device_name", "Unknown"),
                "gpu_vendor": gpu_info.get("gpu_vendor", "NVIDIA")
            }
        )
    
    async def _detect_storage_capabilities(self) -> Dict[ResourceType, ResourceSpec]:
        """Detect storage capabilities"""
        storage_specs = {}
        
        try:
            import psutil
            
            # Disk storage
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            
            if free_space_gb > 10:  # At least 10GB to contribute
                storage_specs[ResourceType.STORAGE_PERSISTENT] = ResourceSpec(
                    resource_type=ResourceType.STORAGE_PERSISTENT,
                    measurement_unit=ResourceMeasurement.STORAGE_GB,
                    total_capacity=free_space_gb,
                    allocated_capacity=0.0,
                    reserved_capacity=5.0,  # Reserve 5GB
                    quality_metrics={
                        "read_speed_mbps": await self._benchmark_disk_read(),
                        "write_speed_mbps": await self._benchmark_disk_write(),
                        "storage_type": "ssd"  # Would detect actual type
                    }
                )
                
        except Exception as e:
            logger.warning("Storage detection failed", error=str(e))
            
        return storage_specs
    
    async def _detect_memory_capabilities(self) -> Optional[ResourceSpec]:
        """Detect memory capabilities"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb > 2:  # At least 2GB to contribute
                return ResourceSpec(
                    resource_type=ResourceType.STORAGE_MEMORY,
                    measurement_unit=ResourceMeasurement.MEMORY_GB,
                    total_capacity=available_gb,
                    allocated_capacity=0.0,
                    reserved_capacity=2.0,  # Reserve 2GB for system
                    quality_metrics={
                        "memory_speed_mhz": 3200,  # Would detect actual speed
                        "memory_type": "DDR4"
                    }
                )
        except Exception as e:
            logger.warning("Memory detection failed", error=str(e))
            return None
    
    async def _detect_network_capabilities(self) -> Dict[ResourceType, ResourceSpec]:
        """Detect network capabilities"""
        network_specs = {}
        
        try:
            # Bandwidth test
            download_mbps, upload_mbps = await self._run_bandwidth_test()
            latency_ms = await self._measure_network_latency()
            
            if download_mbps > 10:  # At least 10 Mbps to contribute
                network_specs[ResourceType.BANDWIDTH_INGRESS] = ResourceSpec(
                    resource_type=ResourceType.BANDWIDTH_INGRESS,
                    measurement_unit=ResourceMeasurement.MBPS,
                    total_capacity=download_mbps,
                    allocated_capacity=0.0,
                    reserved_capacity=5.0,  # Reserve 5 Mbps
                    quality_metrics={
                        "latency_ms": latency_ms,
                        "packet_loss": 0.01,  # Would measure actual packet loss
                        "jitter_ms": 2.0
                    }
                )
                
            if upload_mbps > 5:  # At least 5 Mbps upload to contribute
                network_specs[ResourceType.BANDWIDTH_EGRESS] = ResourceSpec(
                    resource_type=ResourceType.BANDWIDTH_EGRESS,
                    measurement_unit=ResourceMeasurement.MBPS,
                    total_capacity=upload_mbps,
                    allocated_capacity=0.0,
                    reserved_capacity=2.0,  # Reserve 2 Mbps
                    quality_metrics={
                        "latency_ms": latency_ms,
                        "stability_score": 0.95
                    }
                )
                
        except Exception as e:
            logger.warning("Network detection failed", error=str(e))
            
        return network_specs
    
    # Benchmark methods
    async def _run_cpu_benchmark(self) -> float:
        """Run CPU performance benchmark"""
        # Simplified benchmark - would be more comprehensive
        start_time = time.time()
        
        # CPU-intensive calculation
        result = 0
        for i in range(100000):
            result += i * i
            
        execution_time = time.time() - start_time
        return 1.0 / execution_time  # Higher score for faster execution
    
    async def _detect_gpu_memory(self) -> float:
        """
        Detect GPU memory capacity using multiple fallback methods.

        Priority order:
        1. pynvml (NVIDIA Management Library) - most accurate for NVIDIA
        2. amdsmi (AMD System Management Interface) - for AMD GPUs
        3. torch.cuda (PyTorch) - good for ML workloads
        4. GPUtil - cross-platform fallback
        5. Return 0.0 if no GPU detected

        Returns:
            float: GPU memory in GB, or 0.0 if no GPU available
        """
        # Priority 1: Try pynvml (NVIDIA Management Library)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem_info.total / (1024**3)
            pynvml.nvmlShutdown()
            logger.debug("GPU memory detected via pynvml", total_gb=total_gb)
            return total_gb
        except ImportError:
            logger.debug("pynvml not available, trying next method")
        except Exception as e:
            logger.debug("pynvml detection failed", error=str(e))

        # Priority 2: Try amdsmi (AMD System Management Interface)
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            socket_handles = amdsmi.amdsmi_get_processor_handles()
            if socket_handles:
                # Get first GPU's memory info
                device_handle = socket_handles[0]
                memory_info = amdsmi.amdsmi_dev_gpu_memory_info_get(device_handle)
                total_gb = memory_info.get("vram_size", 0) / (1024**3)
                amdsmi.amdsmi_shut_down()
                if total_gb > 0:
                    logger.debug("GPU memory detected via amdsmi", total_gb=total_gb)
                    return total_gb
        except ImportError:
            logger.debug("amdsmi not available, trying next method")
        except Exception as e:
            logger.debug("amdsmi detection failed", error=str(e))

        # Priority 3: Try torch.cuda (PyTorch)
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory / (1024**3)
                logger.debug("GPU memory detected via torch.cuda", total_gb=total_gb)
                return total_gb
        except ImportError:
            logger.debug("torch not available, trying next method")
        except Exception as e:
            logger.debug("torch.cuda detection failed", error=str(e))
        
        # Priority 3: Try GPUtil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                # GPUtil returns memory in MB, convert to GB
                total_gb = gpus[0].memoryTotal / 1024
                logger.debug("GPU memory detected via GPUtil", total_gb=total_gb)
                return total_gb
        except ImportError:
            logger.debug("GPUtil not available")
        except Exception as e:
            logger.debug("GPUtil detection failed", error=str(e))
        
        # No GPU detected or all methods failed
        logger.debug("No GPU detected or all detection methods failed")
        return 0.0
    
    def _get_gpu_info_via_pynvml(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed GPU information via pynvml.
        
        Returns:
            Dict with keys: memory_gb, cuda_cores, compute_capability,
                           memory_bandwidth, device_name
            or None if detection fails
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = mem_info.total / (1024**3)
            
            # Device name
            device_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode('utf-8')
            
            # CUDA cores - try direct API first, then lookup
            try:
                cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
            except (AttributeError, pynvml.NVMLError):
                # Fallback: estimate from device name or use default
                cuda_cores = self._estimate_cuda_cores(device_name)
            
            # Compute capability
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"
            except (AttributeError, pynvml.NVMLError):
                compute_capability = "8.6"  # Default fallback
            
            # Memory bandwidth from lookup table
            memory_bandwidth = self._get_memory_bandwidth(device_name)
            
            pynvml.nvmlShutdown()
            
            return {
                "memory_gb": memory_gb,
                "cuda_cores": cuda_cores,
                "compute_capability": compute_capability,
                "memory_bandwidth": memory_bandwidth,
                "device_name": device_name
            }
        except Exception as e:
            logger.debug("pynvml GPU info detection failed", error=str(e))
            return None
    
    def _get_gpu_info_via_torch(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed GPU information via PyTorch.
        
        Returns:
            Dict with keys: memory_gb, cuda_cores, compute_capability,
                           memory_bandwidth, device_name
            or None if detection fails
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            
            # Compute capability
            compute_capability = f"{props.major}.{props.minor}"
            
            # Device name
            device_name = props.name
            
            # CUDA cores - approximate based on architecture
            # Modern NVIDIA GPUs: ~128 cores per SM (varies by architecture)
            # Ampere (8.x): 128 cores/SM, Ada (8.9): 128 cores/SM
            cuda_cores = props.multi_processor_count * 128
            
            # Memory bandwidth from lookup table
            memory_bandwidth = self._get_memory_bandwidth(device_name)
            
            return {
                "memory_gb": memory_gb,
                "cuda_cores": cuda_cores,
                "compute_capability": compute_capability,
                "memory_bandwidth": memory_bandwidth,
                "device_name": device_name
            }
        except Exception as e:
            logger.debug("torch GPU info detection failed", error=str(e))
            return None

    def _get_gpu_info_via_amdsmi(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed GPU information via AMD System Management Interface (amdsmi).

        Returns:
            Dict with keys: memory_gb, cuda_cores, compute_capability,
                           memory_bandwidth, device_name, gpu_vendor
            or None if detection fails
        """
        try:
            import amdsmi

            amdsmi.amdsmi_init()
            socket_handles = amdsmi.amdsmi_get_processor_handles()

            if not socket_handles:
                amdsmi.amdsmi_shut_down()
                return None

            # Get first AMD GPU
            device_handle = socket_handles[0]

            # Get device identification
            device_id = amdsmi.amdsmi_dev_id_get(device_handle)
            device_name = f"AMD GPU {device_id}"

            # Get memory info
            try:
                memory_info = amdsmi.amdsmi_dev_gpu_memory_info_get(device_handle)
                memory_gb = memory_info.get("vram_size", 0) / (1024**3)
            except Exception:
                # Fallback: estimate based on device
                memory_gb = 8.0  # Default 8GB for unknown AMD GPUs

            # AMD uses "stream processors" instead of CUDA cores
            # Approximate based on device family (simplified)
            cuda_cores = 2560  # Default for mid-range AMD GPU

            # ROCm compute capability (gfx architecture)
            try:
                target = amdsmi.amdsmi_dev_target_get(device_handle)
                compute_capability = f"gfx{target}" if target else "gfx900"
            except Exception:
                compute_capability = "gfx900"  # Default ROCm target

            # Memory bandwidth from lookup
            memory_bandwidth = self._get_amd_memory_bandwidth(device_name)

            amdsmi.amdsmi_shut_down()

            return {
                "memory_gb": memory_gb,
                "cuda_cores": cuda_cores,  # Actually stream processors
                "compute_capability": compute_capability,
                "memory_bandwidth": memory_bandwidth,
                "device_name": device_name,
                "gpu_vendor": "AMD"
            }
        except ImportError:
            logger.debug("amdsmi not available")
            return None
        except Exception as e:
            logger.debug("amdsmi GPU info detection failed", error=str(e))
            return None

    def _get_amd_memory_bandwidth(self, device_name: str) -> float:
        """
        Get memory bandwidth for AMD GPU devices from lookup table.

        Args:
            device_name: AMD GPU device name

        Returns:
            float: Memory bandwidth in GB/s, or 512.0 as default
        """
        # AMD GPU memory bandwidth lookup (GB/s)
        AMD_MEMORY_BANDWIDTH = {
            "RX 7900 XTX": 960,
            "RX 7900 XT": 800,
            "RX 7900 GRE": 576,
            "RX 7800 XT": 624,
            "RX 7700 XT": 432,
            "RX 7600": 288,
            "RX 6900 XT": 512,
            "RX 6800 XT": 512,
            "RX 6800": 512,
            "RX 6700 XT": 384,
            "RX 6600 XT": 288,
            "RX 6600": 224,
            "MI300X": 5300,
            "MI300A": 5300,
            "MI250X": 3200,
            "MI210": 1600,
            "MI100": 1200,
            "MI50": 1024,
        }

        clean_name = device_name.upper().replace("AMD", "").strip()

        for key, bandwidth in AMD_MEMORY_BANDWIDTH.items():
            if key.upper() in clean_name:
                return float(bandwidth)

        return 512.0  # Default for unknown AMD GPUs

    def _get_memory_bandwidth(self, device_name: str) -> float:
        """
        Get memory bandwidth for a GPU device from lookup table.
        
        Args:
            device_name: GPU device name (e.g., "NVIDIA GeForce RTX 4090")
            
        Returns:
            float: Memory bandwidth in GB/s, or 448.0 as default
        """
        # Clean up device name for matching
        clean_name = device_name.upper().replace("NVIDIA", "").replace("GEFORCE", "").strip()
        
        # Priority 1: Try exact match (case-insensitive)
        for key, bandwidth in NVIDIA_MEMORY_BANDWIDTH.items():
            if key.upper() == clean_name:
                return float(bandwidth)
        
        # Priority 2: Try to find the most specific match
        # Sort by key length descending to match more specific names first
        # (e.g., "RTX 3080 Ti" before "RTX 3080")
        sorted_items = sorted(NVIDIA_MEMORY_BANDWIDTH.items(), key=lambda x: len(x[0]), reverse=True)
        
        for key, bandwidth in sorted_items:
            key_upper = key.upper()
            # Check if the lookup key is contained in the device name
            if key_upper in clean_name:
                return float(bandwidth)
        
        # Priority 3: Try partial matching for common patterns
        for key, bandwidth in sorted_items:
            key_parts = key.upper().split()
            if any(part in clean_name for part in key_parts if len(part) > 3):
                return float(bandwidth)
        
        # Default fallback
        logger.debug("GPU memory bandwidth not found in lookup table, using default",
                    device_name=device_name)
        return 448.0
    
    def _estimate_cuda_cores(self, device_name: str) -> int:
        """
        Estimate CUDA cores based on device name.
        
        This is a fallback when nvmlDeviceGetNumGpuCores is not available.
        
        Args:
            device_name: GPU device name
            
        Returns:
            int: Estimated CUDA cores, or 2048 as default
        """
        # Known CUDA core counts for common GPUs
        cuda_cores_lookup = {
            "RTX 4090": 16384,
            "RTX 4080": 9728,
            "RTX 4080 SUPER": 10240,
            "RTX 4070 Ti SUPER": 8448,
            "RTX 4070 Ti": 7680,
            "RTX 4070 SUPER": 7168,
            "RTX 4070": 5888,
            "RTX 4060 Ti": 4352,
            "RTX 4060": 3072,
            "RTX 3090 Ti": 10752,
            "RTX 3090": 10496,
            "RTX 3080 Ti": 10240,
            "RTX 3080": 8704,
            "RTX 3070 Ti": 6144,
            "RTX 3070": 5888,
            "RTX 3060 Ti": 4864,
            "RTX 3060": 3584,
            "A100": 6912,
            "A10": 9216,
            "V100": 5120,
            "H100": 16896,
        }
        
        clean_name = device_name.upper().replace("NVIDIA", "").replace("GEFORCE", "").strip()
        
        for key, cores in cuda_cores_lookup.items():
            if key.upper() in clean_name:
                return cores
        
        return 2048  # Default fallback
    
    def _sync_benchmark_disk_write(self) -> float:
        """Synchronous disk write benchmark - runs in executor"""
        import tempfile
        import os
        
        test_size_bytes = 64 * 1024 * 1024  # 64 MB test file
        data = os.urandom(test_size_bytes)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.prsm_bench') as f:
            tmppath = f.name
            start = time.perf_counter()
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # force to physical disk
            elapsed = time.perf_counter() - start
        os.unlink(tmppath)
        
        return (test_size_bytes / (1024**2)) / elapsed  # MB/s
    
    def _sync_benchmark_disk_read(self) -> float:
        """Synchronous disk read benchmark - runs in executor"""
        import tempfile
        import os
        
        test_size_bytes = 64 * 1024 * 1024  # 64 MB test file
        data = os.urandom(test_size_bytes)
        
        # Write the test file first (without timing)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.prsm_bench') as f:
            tmppath = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        
        try:
            # Time the read operation
            start = time.perf_counter()
            with open(tmppath, 'rb') as f:
                while f.read(1024 * 1024):  # Read in 1MB chunks
                    pass
            elapsed = time.perf_counter() - start
            return (test_size_bytes / (1024**2)) / elapsed  # MB/s
        finally:
            os.unlink(tmppath)
    
    async def _benchmark_disk_read(self) -> float:
        """Benchmark disk read speed with caching"""
        current_time = time.time()
        
        # Check cache validity
        if (self._disk_read_cache is not None and
            self._disk_read_cache_time is not None and
            (current_time - self._disk_read_cache_time) < self._disk_cache_ttl):
            return self._disk_read_cache
        
        try:
            # Run blocking benchmark in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._sync_benchmark_disk_read)
            
            # Cache the result
            self._disk_read_cache = result
            self._disk_read_cache_time = current_time
            
            return result
        except Exception as e:
            logger.warning("Disk read benchmark failed, using fallback", error=str(e))
            return 100.0  # Conservative fallback
    
    async def _benchmark_disk_write(self) -> float:
        """Benchmark disk write speed with caching"""
        current_time = time.time()
        
        # Check cache validity
        if (self._disk_write_cache is not None and
            self._disk_write_cache_time is not None and
            (current_time - self._disk_write_cache_time) < self._disk_cache_ttl):
            return self._disk_write_cache
        
        try:
            # Run blocking benchmark in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._sync_benchmark_disk_write)
            
            # Cache the result
            self._disk_write_cache = result
            self._disk_write_cache_time = current_time
            
            return result
        except Exception as e:
            logger.warning("Disk write benchmark failed, using fallback", error=str(e))
            return 100.0  # Conservative fallback
    
    async def _run_bandwidth_test(self) -> Tuple[float, float]:
        """
        Test network bandwidth using psutil.net_io_counters() differential.
        
        Measures current network activity over a 3-second window.
        Note: This measures current throughput, not peak capacity.
        A proper speedtest (iperf3 to a PRSM bootstrap node) would be needed
        for accurate capacity measurement.
        
        TODO: Implement iperf3-based speedtest to PRSM bootstrap nodes for
        accurate peak bandwidth measurement.
        
        Returns:
            Tuple of (download_mbps, upload_mbps)
        """
        current_time = time.time()
        
        # Check cache validity
        if (self._bandwidth_cache is not None and
            self._bandwidth_cache_time is not None and
            (current_time - self._bandwidth_cache_time) < self._network_cache_ttl):
            return self._bandwidth_cache
        
        try:
            import psutil
            
            # Get initial counters
            counters_before = psutil.net_io_counters()
            
            # Wait 3 seconds to measure throughput
            await asyncio.sleep(3)
            
            # Get final counters
            counters_after = psutil.net_io_counters()
            
            # Calculate bytes transferred
            bytes_recv = counters_after.bytes_recv - counters_before.bytes_recv
            bytes_sent = counters_after.bytes_sent - counters_before.bytes_sent
            
            # Convert to Mbps (bits per second / 1,000,000)
            # Formula: (bytes * 8 bits) / (3 seconds * 1,000,000) = Mbps
            download_mbps = (bytes_recv * 8) / (3 * 1_000_000)
            upload_mbps = (bytes_sent * 8) / (3 * 1_000_000)
            
            # Apply minimum floor - if download is too low, return minimum viable
            if download_mbps < 0.1:
                result = (1.0, 0.5)
            else:
                result = (download_mbps, upload_mbps)
            
            # Cache the result
            self._bandwidth_cache = result
            self._bandwidth_cache_time = current_time
            
            return result
            
        except Exception as e:
            logger.warning("Bandwidth test failed, using fallback", error=str(e))
            return (10.0, 5.0)  # Conservative fallback
    
    def _sync_measure_latency(self) -> float:
        """
        Synchronous network latency measurement to well-known DNS endpoints.
        Runs in executor to avoid blocking the event loop.
        
        Returns:
            Median latency in milliseconds, or 999.0 if all targets fail.
        """
        targets = [
            ("8.8.8.8", 53),           # Google DNS
            ("1.1.1.1", 53),           # Cloudflare DNS
            ("208.67.222.222", 53),    # OpenDNS
        ]
        
        latencies = []
        
        for host, port in targets:
            try:
                start = time.perf_counter()
                sock = socket.create_connection((host, port), timeout=2)
                sock.close()
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
            except (socket.timeout, socket.error, OSError) as e:
                logger.debug("Latency measurement failed for target",
                           target=f"{host}:{port}", error=str(e))
                continue
        
        if latencies:
            return statistics.median(latencies)
        else:
            return 999.0  # High latency / offline signal
    
    async def _measure_network_latency(self) -> float:
        """
        Measure network latency to key nodes using connect-time measurement.
        
        Tests connectivity to 3 well-known stable DNS endpoints and returns
        the median latency. Uses executor for blocking socket operations.
        
        Returns:
            Median latency in milliseconds, or 999.0 if all targets fail.
        """
        current_time = time.time()
        
        # Check cache validity
        if (self._network_latency_cache is not None and
            self._network_latency_cache_time is not None and
            (current_time - self._network_latency_cache_time) < self._network_cache_ttl):
            return self._network_latency_cache
        
        try:
            # Run blocking latency measurement in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._sync_measure_latency)
            
            # Cache the result
            self._network_latency_cache = result
            self._network_latency_cache_time = current_time
            
            return result
            
        except Exception as e:
            logger.warning("Network latency measurement failed, using fallback", error=str(e))
            return 999.0  # High latency / offline signal


# ============================================================================
# TRUST-BUT-VERIFY RESOURCE VERIFICATION
# ============================================================================

class ResourceVerificationLevel(str, Enum):
    """Levels of resource verification"""
    SELF_REPORTED = "self_reported"       # User claims with signature
    PEER_VERIFIED = "peer_verified"       # Verified by other nodes
    BENCHMARKED = "benchmarked"           # Standardized benchmark results
    HARDWARE_ATTESTED = "hardware_attested"  # TPM/SGX hardware attestation
    ECONOMICALLY_STAKED = "economically_staked"  # Economic consequences for fraud


class ResourceVerificationProof(BaseModel):
    """Cryptographic proof of resource availability"""
    verification_type: ResourceVerificationLevel
    resource_type: ResourceType
    claimed_capacity: float
    proof_data: Dict[str, Any]           # Verification-specific data
    timestamp: datetime
    verifier_nodes: List[str] = Field(default_factory=list)
    cryptographic_signature: str
    validity_period: timedelta = Field(default=timedelta(hours=1))
    
    @property
    def is_valid(self) -> bool:
        """Check if proof is still valid"""
        return datetime.now(timezone.utc) < (self.timestamp + self.validity_period)


class ResourceVerificationEngine:
    """Engine for verifying resource claims with trust-but-verify approach"""
    
    def __init__(self):
        self.verification_strategies = {
            ResourceVerificationLevel.SELF_REPORTED: self._verify_self_reported,
            ResourceVerificationLevel.PEER_VERIFIED: self._verify_peer_consensus,
            ResourceVerificationLevel.BENCHMARKED: self._verify_benchmarks,
            ResourceVerificationLevel.HARDWARE_ATTESTED: self._verify_hardware,
            ResourceVerificationLevel.ECONOMICALLY_STAKED: self._verify_economic_stake
        }
        self.verification_cache = {}
        
    async def verify_resource_claim(
        self, 
        node_id: str, 
        resource_spec: ResourceSpec,
        verification_level: ResourceVerificationLevel = ResourceVerificationLevel.PEER_VERIFIED
    ) -> ResourceVerificationProof:
        """Verify a node's resource claim at specified verification level"""
        
        verification_func = self.verification_strategies[verification_level]
        proof_data = await verification_func(node_id, resource_spec)
        
        # Generate cryptographic proof
        proof = ResourceVerificationProof(
            verification_type=verification_level,
            resource_type=resource_spec.resource_type,
            claimed_capacity=resource_spec.total_capacity,
            proof_data=proof_data,
            timestamp=datetime.now(timezone.utc),
            cryptographic_signature=self._generate_proof_signature(proof_data)
        )
        
        # Cache verification result
        cache_key = f"{node_id}:{resource_spec.resource_type}:{verification_level}"
        self.verification_cache[cache_key] = proof
        
        logger.info("Resource verification completed", 
                   node_id=node_id, 
                   resource_type=resource_spec.resource_type,
                   verification_level=verification_level,
                   verification_score=proof_data.get("verification_score", 1.0))
        
        return proof
    
    async def _verify_self_reported(self, node_id: str, resource_spec: ResourceSpec) -> Dict[str, Any]:
        """Verify self-reported resource claims with basic validation"""
        return {
            "verification_method": "self_reported",
            "verification_score": 0.6,  # Lower trust for self-reported
            "node_signature": f"sig_{node_id}_{hash(str(resource_spec))}",
            "validation_checks": {
                "capacity_reasonable": resource_spec.total_capacity > 0,
                "quality_metrics_present": len(resource_spec.quality_metrics) > 0,
                "specification_complete": True
            }
        }
    
    async def _verify_peer_consensus(self, node_id: str, resource_spec: ResourceSpec) -> Dict[str, Any]:
        """Verify resource claims through peer node consensus"""
        # Select random peer nodes for verification
        verifier_peers = self._select_verification_peers(node_id, count=5)
        
        verification_results = []
        for peer_id in verifier_peers:
            # Each peer runs verification tests
            peer_result = await self._request_peer_verification(peer_id, node_id, resource_spec)
            verification_results.append(peer_result)
        
        # Calculate consensus score
        positive_votes = sum(1 for result in verification_results if result["verified"])
        consensus_score = positive_votes / len(verification_results)
        
        return {
            "verification_method": "peer_consensus",
            "verification_score": consensus_score,
            "verifier_peers": verifier_peers,
            "consensus_results": verification_results,
            "consensus_threshold": 0.6,
            "consensus_achieved": consensus_score >= 0.6
        }
    
    async def _verify_benchmarks(self, node_id: str, resource_spec: ResourceSpec) -> Dict[str, Any]:
        """Verify resources through standardized benchmark execution"""
        benchmark_type = self._get_benchmark_for_resource(resource_spec.resource_type)
        
        # Request node to run standardized benchmark
        benchmark_request = {
            "benchmark_type": benchmark_type,
            "expected_duration": 60,  # seconds
            "verification_nonce": self._generate_verification_nonce()
        }
        
        benchmark_result = await self._request_benchmark_execution(node_id, benchmark_request)
        
        # Validate benchmark authenticity and results
        is_authentic = await self._validate_benchmark_authenticity(benchmark_result)
        performance_score = self._calculate_performance_score(benchmark_result, resource_spec)
        
        return {
            "verification_method": "standardized_benchmark",
            "verification_score": 0.85 if is_authentic else 0.3,
            "benchmark_type": benchmark_type,
            "benchmark_results": benchmark_result,
            "performance_score": performance_score,
            "authenticity_verified": is_authentic
        }
    
    async def _verify_hardware(self, node_id: str, resource_spec: ResourceSpec) -> Dict[str, Any]:
        """Verify resources through hardware attestation (TPM/SGX)"""
        # Request hardware attestation
        attestation_request = {
            "attestation_type": "tpm_quote",
            "resource_type": resource_spec.resource_type,
            "nonce": self._generate_verification_nonce()
        }
        
        attestation_result = await self._request_hardware_attestation(node_id, attestation_request)
        
        # Validate attestation signature and contents
        is_valid_attestation = await self._validate_hardware_attestation(attestation_result)
        
        return {
            "verification_method": "hardware_attestation",
            "verification_score": 0.95 if is_valid_attestation else 0.1,
            "attestation_type": "tpm_quote",
            "attestation_valid": is_valid_attestation,
            "hardware_verified": True,
            "attestation_data": attestation_result
        }
    
    async def _verify_economic_stake(self, node_id: str, resource_spec: ResourceSpec) -> Dict[str, Any]:
        """Verify resources through economic stake and slashing conditions"""
        # Check node's economic stake
        stake_info = await self._get_node_stake_info(node_id)
        required_stake = self._calculate_required_stake(resource_spec)
        
        stake_adequate = stake_info["stake_amount"] >= required_stake
        
        # Set up slashing conditions
        slashing_conditions = {
            "availability_threshold": 0.95,
            "performance_threshold": 0.8,
            "verification_period_hours": 24,
            "slash_percentage": 0.1  # 10% of stake for violations
        }
        
        return {
            "verification_method": "economic_stake",
            "verification_score": 0.9 if stake_adequate else 0.4,
            "stake_amount": stake_info["stake_amount"],
            "required_stake": required_stake,
            "stake_adequate": stake_adequate,
            "slashing_conditions": slashing_conditions,
            "economic_incentive_aligned": True
        }
    
    # Helper methods for verification
    def _select_verification_peers(self, node_id: str, count: int) -> List[str]:
        """Select peers for verification, excluding the node being verified.
        
        Selects from the actual node registry, preferring peers with high
        reputation scores and using random sampling to avoid always picking
        the same peers.
        
        Args:
            node_id: The ID of the node being verified (to exclude)
            count: Number of peers to select
            
        Returns:
            List of peer node IDs to use for verification
        """
        # Pull from the actual node registry, exclude the target node
        candidates = [
            peer_id for peer_id in self.node_registry.keys()
            if peer_id != node_id
        ]
        
        if len(candidates) <= count:
            return candidates
        
        # Prefer peers with high reputation scores
        sorted_peers = sorted(
            candidates,
            key=lambda pid: self.node_registry[pid].reputation_score,
            reverse=True
        )
        
        # Take top 2x candidates and randomly sample from them
        # This avoids always picking the same peers
        pool = sorted_peers[:count * 2]
        return random.sample(pool, min(count, len(pool)))
    
    async def _request_peer_verification(self, peer_id: str, target_node: str, resource_spec: ResourceSpec) -> Dict[str, Any]:
        """Request a peer to verify another node's resources.
        
        Sends an HTTP POST request to the peer's verification endpoint
        with the resource specification and a verification nonce.
        
        Args:
            peer_id: The ID of the peer to request verification from
            target_node: The ID of the node being verified
            resource_spec: The resource specification to verify
            
        Returns:
            Dict with verification results or error information
        """
        peer_profile = self.node_registry.get(peer_id)
        if not peer_profile:
            return {"peer_id": peer_id, "verified": False, "error": "peer_not_found"}

        peer_address = f"http://{peer_profile.contribution_settings.get('host', 'localhost')}:{peer_profile.contribution_settings.get('port', 8080)}"

        payload = {
            "action": "verify_resource",
            "target_node": target_node,
            "resource_type": resource_spec.resource_type.value,
            "claimed_capacity": resource_spec.total_capacity,
            "nonce": self._generate_verification_nonce()
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{peer_address}/v1/verify",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"peer_id": peer_id, "verified": False, "error": f"HTTP {resp.status}"}
            except aiohttp.ClientError as e:
                return {"peer_id": peer_id, "verified": False, "error": str(e)}
    
    async def _request_benchmark_execution(self, node_id: str, benchmark_request: Dict[str, Any]) -> Dict[str, Any]:
        """Request a node to execute a standardized benchmark"""
        # Would send benchmark execution request
        return {
            "benchmark_completed": True,
            "execution_time": 45.2,
            "performance_metrics": {"ops_per_second": 1000, "throughput_mbps": 500},
            "verification_nonce": benchmark_request["verification_nonce"]
        }
    
    async def _request_hardware_attestation(self, node_id: str, attestation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Request hardware attestation from a node"""
        # Would request TPM quote or SGX attestation
        return {
            "attestation_signature": "mock_signature_data",
            "platform_configuration": {"tpm_version": "2.0", "secure_boot": True},
            "measurement_values": {"pcr0": "abc123", "pcr1": "def456"}
        }
    
    def _generate_proof_signature(self, proof_data: Dict[str, Any]) -> str:
        """Generate cryptographic signature for verification proof"""
        proof_hash = hashlib.sha256(json.dumps(proof_data, sort_keys=True).encode()).hexdigest()
        return f"sig_{proof_hash[:16]}"
    
    def _generate_verification_nonce(self) -> str:
        """Generate a unique nonce for verification requests"""
        return hashlib.sha256(f"{time.time()}_{uuid4()}".encode()).hexdigest()[:16]
    
    def _get_benchmark_for_resource(self, resource_type: ResourceType) -> str:
        """Get appropriate benchmark type for resource"""
        benchmark_map = {
            ResourceType.COMPUTE_CPU: "cpu_intensive_calculation",
            ResourceType.COMPUTE_GPU: "gpu_matrix_multiplication",
            ResourceType.STORAGE_PERSISTENT: "disk_io_benchmark",
            ResourceType.STORAGE_MEMORY: "memory_throughput_test",
            ResourceType.BANDWIDTH_INGRESS: "network_download_test",
            ResourceType.BANDWIDTH_EGRESS: "network_upload_test"
        }
        return benchmark_map.get(resource_type, "generic_performance_test")
    
    async def _validate_benchmark_authenticity(self, benchmark_result: Dict[str, Any]) -> bool:
        """Validate that benchmark results are authentic"""
        # Would implement sophisticated validation logic
        return benchmark_result.get("benchmark_completed", False)
    
    def _calculate_performance_score(self, benchmark_result: Dict[str, Any], resource_spec: ResourceSpec) -> float:
        """Calculate performance score based on benchmark results.
        
        Compares benchmark metrics against reference baselines per resource type,
        returning a normalized score in [0.0, 1.0] range.
        """
        # Define reference baselines per resource type
        BASELINES = {
            ResourceType.COMPUTE_CPU:        {"ops_per_second": 50_000},
            ResourceType.COMPUTE_GPU:        {"ops_per_second": 1_000_000},
            ResourceType.STORAGE_PERSISTENT: {"throughput_mbps": 500},
            ResourceType.STORAGE_MEMORY:     {"throughput_mbps": 20_000},
            ResourceType.BANDWIDTH_INGRESS:  {"throughput_mbps": 100},
            ResourceType.BANDWIDTH_EGRESS:   {"throughput_mbps": 100},
        }

        baseline = BASELINES.get(resource_spec.resource_type)
        if not baseline:
            return 0.5  # neutral for unknown types

        metrics = benchmark_result.get("performance_metrics", {})

        # Find the most relevant metric in the result
        for key, baseline_val in baseline.items():
            if key in metrics:
                raw_score = metrics[key] / baseline_val
                # Clamp to [0.0, 1.0] using sigmoid-like compression
                score = min(1.0, raw_score ** 0.5)   # sqrt dampens outliers
                return round(score, 4)

        return 0.5  # fallback if no matching metric key
    
    async def _validate_hardware_attestation(self, attestation_result: Dict[str, Any]) -> bool:
        """Validate hardware attestation signature and contents"""
        # Would validate TPM signature and measurement values
        return attestation_result.get("attestation_signature") is not None
    
    async def _get_node_stake_info(self, node_id: str) -> Dict[str, Any]:
        """Get node's economic stake information"""
        # Would query FTNS service for stake info
        return {
            "stake_amount": Decimal('100.0'),
            "stake_locked": True,
            "stake_history": []
        }
    
    def _calculate_required_stake(self, resource_spec: ResourceSpec) -> Decimal:
        """Calculate required economic stake based on resource value"""
        # Base stake calculation on resource capacity and type
        base_stake = Decimal('10.0')  # Base 10 FTNS
        capacity_multiplier = Decimal(str(resource_spec.total_capacity / 100))  # Scale with capacity
        return base_stake * capacity_multiplier


# ============================================================================
# INTELLIGENT RESOURCE ALLOCATION
# ============================================================================

class ResourceAllocationEngine:
    """Intelligent engine for optimal resource allocation across millions of nodes"""
    
    def __init__(self):
        self.allocation_strategies = {
            "cost_optimal": self._allocate_cost_optimal,
            "performance_optimal": self._allocate_performance_optimal,
            "geographic_optimal": self._allocate_geographic_optimal,
            "reliability_optimal": self._allocate_reliability_optimal,
            "hybrid_optimal": self._allocate_hybrid_optimal
        }
        self.allocation_cache = {}
        self.performance_history = {}
        
    async def allocate_resources_for_task(
        self,
        task_requirements: Dict[ResourceType, float],
        constraints: Dict[str, Any],
        strategy: str = "hybrid_optimal"
    ) -> Dict[str, Any]:
        """Allocate optimal resources for a specific task"""
        
        # Get available nodes matching requirements
        candidate_nodes = await self._find_candidate_nodes(task_requirements, constraints)
        
        # Apply allocation strategy
        allocation_func = self.allocation_strategies.get(strategy, self._allocate_hybrid_optimal)
        optimal_allocation = await allocation_func(task_requirements, candidate_nodes, constraints)
        
        # Reserve resources on selected nodes
        reservation_results = await self._reserve_resources(optimal_allocation)
        
        return {
            "allocation_id": str(uuid4()),
            "selected_nodes": optimal_allocation["nodes"],
            "total_cost": optimal_allocation["total_cost"],
            "expected_performance": optimal_allocation["expected_performance"],
            "allocation_strategy": strategy,
            "reservation_success": reservation_results["success"],
            "estimated_completion_time": optimal_allocation["estimated_completion_time"]
        }
    
    async def _find_candidate_nodes(
        self,
        task_requirements: Dict[ResourceType, float],
        constraints: Dict[str, Any]
    ) -> List[NodeResourceProfile]:
        """Find nodes that can satisfy task requirements."""
        try:
            from prsm.core.database import get_async_session, FederationPeerModel
            from sqlalchemy import select

            min_quality = constraints.get('min_quality_score', 0.3)
            limit = constraints.get('max_candidates', 50)

            async with get_async_session() as session:
                stmt = (
                    select(FederationPeerModel)
                    .where(
                        FederationPeerModel.is_active == True,
                        FederationPeerModel.quality_score >= min_quality,
                    )
                    .order_by(FederationPeerModel.quality_score.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                peers = result.scalars().all()

            candidates = []
            for peer in peers:
                resources = {}
                for resource_type, required_amount in task_requirements.items():
                    cap = peer.capabilities.get(resource_type.value, {})
                    total = cap.get('total_capacity', required_amount * 2)
                    allocated = cap.get('allocated_capacity', 0.0)
                    resources[resource_type] = ResourceSpec(
                        resource_type=resource_type,
                        measurement_unit=ResourceMeasurement.CPU_CORES,
                        total_capacity=float(total),
                        allocated_capacity=float(allocated),
                        reserved_capacity=float(total) * 0.1,
                        quality_metrics={
                            "performance_score": peer.quality_score,
                            "reliability_score": peer.quality_score,
                        }
                    )
                candidates.append(NodeResourceProfile(
                    node_id=peer.peer_id,
                    user_id=peer.peer_id,
                    node_type=peer.node_type,
                    geographic_region=getattr(peer, 'geographic_region', 'unknown'),
                    resources=resources,
                    last_updated=peer.last_seen,
                    reputation_score=peer.quality_score,
                ))

            return candidates

        except Exception as e:
            logger.warning(f"Node discovery failed, returning empty candidates: {e}")
            return []
    
    def _create_mock_node(self, node_id: str, requirements: Dict[ResourceType, float]) -> NodeResourceProfile:
        """Create a mock node for testing"""
        resources = {}
        for resource_type, required_amount in requirements.items():
            resources[resource_type] = ResourceSpec(
                resource_type=resource_type,
                measurement_unit=ResourceMeasurement.CPU_CORES,  # Simplified
                total_capacity=required_amount * 2,  # Double what's needed
                allocated_capacity=0.0,
                reserved_capacity=required_amount * 0.1,  # 10% reserved
                quality_metrics={"performance_score": 0.8, "reliability_score": 0.9}
            )
        
        return NodeResourceProfile(
            node_id=node_id,
            user_id=f"user_{node_id}",
            node_type="medium",
            geographic_region="us-west",
            resources=resources,
            reputation_score=0.85
        )
    
    async def _allocate_cost_optimal(
        self, 
        requirements: Dict[ResourceType, float], 
        candidates: List[NodeResourceProfile], 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allocate resources optimizing for minimum cost"""
        selected_nodes = []
        total_cost = Decimal('0')
        
        # Sort candidates by cost per unit
        sorted_candidates = sorted(candidates, 
                                 key=lambda n: self._calculate_node_cost_per_unit(n))
        
        remaining_requirements = requirements.copy()
        
        for node in sorted_candidates:
            if not remaining_requirements:
                break
                
            # Check what this node can provide
            node_contribution = {}
            node_cost = Decimal('0')
            
            for resource_type, needed_amount in remaining_requirements.items():
                if resource_type in node.resources:
                    available = (node.resources[resource_type].total_capacity - 
                               node.resources[resource_type].allocated_capacity)
                    contribution = min(needed_amount, available)
                    
                    if contribution > 0:
                        node_contribution[resource_type] = contribution
                        node_cost += self._calculate_resource_cost(node, resource_type, contribution)
            
            if node_contribution:
                selected_nodes.append({
                    "node_id": node.node_id,
                    "contribution": node_contribution,
                    "cost": node_cost
                })
                total_cost += node_cost
                
                # Update remaining requirements
                for resource_type, contribution in node_contribution.items():
                    remaining_requirements[resource_type] -= contribution
                    if remaining_requirements[resource_type] <= 0:
                        del remaining_requirements[resource_type]
        
        return {
            "nodes": selected_nodes,
            "total_cost": total_cost,
            "expected_performance": 0.75,  # Cost-optimized might be slower
            "estimated_completion_time": 120,  # seconds
            "allocation_efficiency": 0.8
        }
    
    async def _allocate_performance_optimal(
        self, 
        requirements: Dict[ResourceType, float], 
        candidates: List[NodeResourceProfile], 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allocate resources optimizing for maximum performance"""
        # Sort by performance score
        sorted_candidates = sorted(candidates, 
                                 key=lambda n: self._calculate_node_performance_score(n), 
                                 reverse=True)
        
        return await self._allocate_cost_optimal(requirements, sorted_candidates, constraints)
    
    async def _allocate_geographic_optimal(
        self, 
        requirements: Dict[ResourceType, float], 
        candidates: List[NodeResourceProfile], 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allocate resources optimizing for geographic distribution and latency"""
        user_location = constraints.get("user_location", "us-west")
        
        # Sort by geographic proximity
        sorted_candidates = sorted(candidates, 
                                 key=lambda n: self._calculate_geographic_distance(n.geographic_region, user_location))
        
        return await self._allocate_cost_optimal(requirements, sorted_candidates, constraints)
    
    async def _allocate_reliability_optimal(
        self, 
        requirements: Dict[ResourceType, float], 
        candidates: List[NodeResourceProfile], 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allocate resources optimizing for maximum reliability"""
        # Sort by reputation and reliability scores
        sorted_candidates = sorted(candidates, 
                                 key=lambda n: n.reputation_score, 
                                 reverse=True)
        
        return await self._allocate_cost_optimal(requirements, sorted_candidates, constraints)
    
    async def _allocate_hybrid_optimal(
        self, 
        requirements: Dict[ResourceType, float], 
        candidates: List[NodeResourceProfile], 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Allocate resources using hybrid optimization (cost + performance + reliability)"""
        # Calculate composite score for each node
        scored_candidates = []
        
        for node in candidates:
            cost_score = 1.0 / (1.0 + self._calculate_node_cost_per_unit(node))  # Lower cost = higher score
            performance_score = self._calculate_node_performance_score(node)
            reliability_score = node.reputation_score
            
            # Weighted combination
            composite_score = (cost_score * 0.3 + 
                             performance_score * 0.4 + 
                             reliability_score * 0.3)
            
            scored_candidates.append((node, composite_score))
        
        # Sort by composite score
        sorted_candidates = [node for node, score in sorted(scored_candidates, 
                                                           key=lambda x: x[1], 
                                                           reverse=True)]
        
        return await self._allocate_cost_optimal(requirements, sorted_candidates, constraints)
    
    async def _reserve_resources(self, allocation: Dict[str, Any]) -> Dict[str, bool]:
        """Reserve resources on selected nodes"""
        reservation_results = {"success": True, "failed_reservations": []}
        
        for node_allocation in allocation["nodes"]:
            node_id = node_allocation["node_id"]
            contribution = node_allocation["contribution"]
            
            # Send reservation request to node
            success = await self._send_reservation_request(node_id, contribution)
            
            if not success:
                reservation_results["success"] = False
                reservation_results["failed_reservations"].append(node_id)
        
        return reservation_results
    
    async def _send_reservation_request(self, node_id: str, resource_contribution: Dict[ResourceType, float]) -> bool:
        """Send resource reservation request to a node.
        
        Sends an HTTP POST request to the node's reservation endpoint
        with the resource amounts and a reservation ID.
        
        Args:
            node_id: The ID of the node to send the reservation to
            resource_contribution: Dict mapping ResourceType to amount to reserve
            
        Returns:
            True if reservation was successful, False otherwise
        """
        node_profile = self.node_registry.get(node_id)
        if not node_profile:
            logger.warning("Cannot reserve: node not in registry", node_id=node_id)
            return False

        node_address = f"http://{node_profile.contribution_settings.get('host', 'localhost')}:{node_profile.contribution_settings.get('port', 8080)}"

        payload = {
            "action": "reserve_resources",
            "resources": {rt.value: amt for rt, amt in resource_contribution.items()},
            "reservation_id": str(uuid4()),
            "ttl_seconds": 3600
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{node_address}/v1/reserve",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    return resp.status == 200
            except aiohttp.ClientError as e:
                logger.error("Reservation request failed", node_id=node_id, error=str(e))
                return False
    
    # Helper calculation methods
    def _calculate_node_cost_per_unit(self, node: NodeResourceProfile) -> float:
        """Calculate cost per resource unit for a node.
        
        Computes a weighted average cost across all contributed resources,
        adjusted by the node's reputation score. High-reputation nodes can
        charge a premium, while low-reputation nodes are discounted.
        """
        # Base rate per resource type (FTNS per normalized unit)
        BASE_RATES = {
            ResourceType.COMPUTE_CPU:        0.05,
            ResourceType.COMPUTE_GPU:        0.50,
            ResourceType.COMPUTE_TPU:        1.00,
            ResourceType.STORAGE_PERSISTENT: 0.01,
            ResourceType.STORAGE_MEMORY:     0.08,
            ResourceType.BANDWIDTH_INGRESS:  0.02,
            ResourceType.BANDWIDTH_EGRESS:   0.03,
        }

        if not node.resources:
            return 0.1  # default

        # Weighted average across contributed resources, adjusted by reputation
        total_cost = 0.0
        total_weight = 0.0

        for resource_type, resource_spec in node.resources.items():
            base = BASE_RATES.get(resource_type, 0.1)
            capacity_weight = resource_spec.total_capacity

            # Reputation adjustment: high-reputation nodes can charge a premium
            # Low-reputation nodes are discounted to incentivize proper behavior
            reputation_multiplier = 0.5 + node.reputation_score  # range: [0.5, 1.5]

            total_cost   += base * reputation_multiplier * capacity_weight
            total_weight += capacity_weight

        return total_cost / total_weight if total_weight > 0 else 0.1
    
    def _calculate_node_performance_score(self, node: NodeResourceProfile) -> float:
        """Calculate overall performance score for a node"""
        scores = []
        for resource in node.resources.values():
            performance = resource.quality_metrics.get("performance_score", 0.5)
            scores.append(performance)
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_geographic_distance(self, region1: str, region2: str) -> float:
        """Calculate geographic distance between regions"""
        # Simplified distance calculation
        distance_map = {
            ("us-west", "us-west"): 0,
            ("us-west", "us-east"): 1,
            ("us-west", "europe"): 2,
            ("us-west", "asia"): 3
        }
        return distance_map.get((region1, region2), 2)
    
    def _calculate_resource_cost(self, node: NodeResourceProfile, resource_type: ResourceType, amount: float) -> Decimal:
        """Calculate cost for using specific resource amount on a node"""
        base_rate = Decimal('0.1')  # Base rate per unit
        return base_rate * Decimal(str(amount))


# ============================================================================
# DISTRIBUTED RESOURCE MANAGER
# ============================================================================

class DistributedResourceManager:
    """Main coordinator for distributed resource management across millions of nodes"""
    
    def __init__(self):
        self.capability_detector = ResourceCapabilityDetector()
        self.verification_engine = ResourceVerificationEngine()
        self.allocation_engine = ResourceAllocationEngine()
        self.node_registry = {}  # Would be distributed database
        self.active_allocations = {}
        # Geographic region cache with timestamp
        self._geographic_region_cache: Optional[str] = None
        self._geographic_region_cache_time: Optional[float] = None
        self._geographic_region_cache_ttl: float = 86400.0  # Cache TTL in seconds (24 hours)
        
    async def initialize_node(self, user_id: str, contribution_settings: ResourceContributionSettings) -> str:
        """Initialize a new node in the network"""
        node_id = f"node_{uuid4().hex[:8]}"
        
        # Detect system capabilities
        detected_resources = await self.capability_detector.detect_system_resources()
        
        # Apply user contribution settings
        configured_resources = self._apply_contribution_settings(detected_resources, contribution_settings)
        
        # Create node profile
        node_profile = NodeResourceProfile(
            node_id=node_id,
            user_id=user_id,
            node_type=self._classify_node_size(configured_resources),
            geographic_region=await self._detect_geographic_region(),
            resources=configured_resources,
            contribution_settings=contribution_settings.dict()
        )
        
        # Verify initial resource claims
        for resource_type, resource_spec in configured_resources.items():
            verification_proof = await self.verification_engine.verify_resource_claim(
                node_id, resource_spec, ResourceVerificationLevel.BENCHMARKED
            )
            resource_spec.verification_proofs.append(verification_proof.cryptographic_signature)
            resource_spec.verification_score = verification_proof.proof_data["verification_score"]
        
        # Register node in network
        self.node_registry[node_id] = node_profile
        
        logger.info("Node initialized in PRSM network", 
                   node_id=node_id, 
                   user_id=user_id,
                   resource_types=list(configured_resources.keys()),
                   node_type=node_profile.node_type)
        
        return node_id
    
    async def update_node_settings(self, node_id: str, new_settings: ResourceContributionSettings) -> bool:
        """Update a node's resource contribution settings"""
        if node_id not in self.node_registry:
            return False
        
        node_profile = self.node_registry[node_id]
        
        # Re-detect capabilities and apply new settings
        detected_resources = await self.capability_detector.detect_system_resources()
        updated_resources = self._apply_contribution_settings(detected_resources, new_settings)
        
        # Verify updated resources
        for resource_type, resource_spec in updated_resources.items():
            verification_proof = await self.verification_engine.verify_resource_claim(
                node_id, resource_spec, ResourceVerificationLevel.PEER_VERIFIED
            )
            resource_spec.verification_proofs.append(verification_proof.cryptographic_signature)
        
        # Update node profile
        node_profile.resources = updated_resources
        node_profile.contribution_settings = new_settings.dict()
        node_profile.last_updated = datetime.now(timezone.utc)
        
        logger.info("Node settings updated", 
                   node_id=node_id, 
                   updated_resources=list(updated_resources.keys()))
        
        return True
    
    async def allocate_task_resources(
        self, 
        task_requirements: Dict[ResourceType, float], 
        user_constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Allocate optimal resources for a task across the network"""
        constraints = user_constraints or {}
        
        # Use allocation engine to find optimal resource allocation
        allocation_result = await self.allocation_engine.allocate_resources_for_task(
            task_requirements, constraints, strategy="hybrid_optimal"
        )
        
        # Track active allocation
        allocation_id = allocation_result["allocation_id"]
        self.active_allocations[allocation_id] = {
            "allocation": allocation_result,
            "start_time": datetime.now(timezone.utc),
            "status": "active"
        }
        
        return allocation_result
    
    async def continuous_verification_loop(self):
        """Continuously verify node resources to maintain trust"""
        while True:
            try:
                # Select nodes for verification (random sampling)
                nodes_to_verify = await self._select_nodes_for_verification()
                
                for node_id in nodes_to_verify:
                    if node_id in self.node_registry:
                        await self._verify_node_resources(node_id)
                
                # Wait before next verification cycle
                await asyncio.sleep(300)  # 5 minutes between cycles
                
            except Exception as e:
                logger.error("Verification loop error", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _verify_node_resources(self, node_id: str):
        """Verify all resources for a specific node"""
        node_profile = self.node_registry[node_id]
        
        for resource_type, resource_spec in node_profile.resources.items():
            # Choose verification level based on resource importance and node history
            verification_level = self._choose_verification_level(node_profile, resource_spec)
            
            try:
                verification_proof = await self.verification_engine.verify_resource_claim(
                    node_id, resource_spec, verification_level
                )
                
                # Update resource verification status
                resource_spec.verification_proofs.append(verification_proof.cryptographic_signature)
                resource_spec.last_verified = verification_proof.timestamp
                resource_spec.verification_score = verification_proof.proof_data["verification_score"]
                
                # Update node reputation based on verification results
                await self._update_node_reputation(node_id, verification_proof)
                
            except Exception as e:
                logger.warning("Resource verification failed", 
                             node_id=node_id, 
                             resource_type=resource_type, 
                             error=str(e))
                # Penalize node for verification failure
                node_profile.reputation_score *= 0.95  # 5% reputation penalty
    
    async def _select_nodes_for_verification(self) -> List[str]:
        """Select nodes for verification using intelligent sampling"""
        # Priority factors for verification selection:
        # 1. Nodes with high resource claims
        # 2. Nodes with declining reputation
        # 3. Nodes due for periodic verification
        # 4. Random sampling for unpredictability
        
        verification_candidates = []
        
        for node_id, node_profile in self.node_registry.items():
            # Calculate verification priority score
            priority_score = 0.0
            
            # High resource claims get higher priority
            total_resource_value = sum(
                spec.total_capacity * self._get_resource_value_multiplier(spec.resource_type)
                for spec in node_profile.resources.values()
            )
            priority_score += min(total_resource_value / 1000, 1.0)  # Normalize to 0-1
            
            # Lower reputation nodes get higher priority
            priority_score += (1.0 - node_profile.reputation_score)
            
            # Nodes not verified recently get higher priority
            time_since_verification = datetime.now(timezone.utc) - node_profile.last_updated
            if time_since_verification.total_seconds() > 3600:  # 1 hour
                priority_score += 0.5
            
            verification_candidates.append((node_id, priority_score))
        
        # Sort by priority and select top candidates
        verification_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 10% or at least 10 nodes
        selection_count = max(10, len(verification_candidates) // 10)
        selected_nodes = [node_id for node_id, _ in verification_candidates[:selection_count]]
        
        return selected_nodes
    
    def _apply_contribution_settings(
        self, 
        detected_resources: Dict[ResourceType, ResourceSpec], 
        settings: ResourceContributionSettings
    ) -> Dict[ResourceType, ResourceSpec]:
        """Apply user contribution settings to detected resources"""
        configured_resources = {}
        
        allocation_map = {
            ResourceType.COMPUTE_CPU: settings.cpu_allocation_percentage,
            ResourceType.COMPUTE_GPU: settings.gpu_allocation_percentage,
            ResourceType.STORAGE_PERSISTENT: settings.storage_allocation_percentage,
            ResourceType.STORAGE_MEMORY: settings.memory_allocation_percentage,
            ResourceType.BANDWIDTH_INGRESS: settings.bandwidth_allocation_percentage,
            ResourceType.BANDWIDTH_EGRESS: settings.bandwidth_allocation_percentage
        }
        
        for resource_type, resource_spec in detected_resources.items():
            if resource_type in allocation_map:
                allocation_percentage = allocation_map[resource_type]
                
                # Apply allocation percentage
                allocated_capacity = resource_spec.total_capacity * allocation_percentage
                
                # Ensure we don't over-allocate (keep some reserved)
                max_allocatable = resource_spec.total_capacity - resource_spec.reserved_capacity
                final_allocation = min(allocated_capacity, max_allocatable)
                
                if final_allocation > 0:
                    configured_spec = ResourceSpec(
                        resource_type=resource_spec.resource_type,
                        measurement_unit=resource_spec.measurement_unit,
                        total_capacity=final_allocation,
                        allocated_capacity=0.0,
                        reserved_capacity=resource_spec.reserved_capacity,
                        quality_metrics=resource_spec.quality_metrics.copy()
                    )
                    configured_resources[resource_type] = configured_spec
        
        return configured_resources
    
    def _classify_node_size(self, resources: Dict[ResourceType, ResourceSpec]) -> str:
        """Classify node size based on total resource contribution"""
        total_score = 0.0
        
        for resource_spec in resources.values():
            # Weight different resource types
            weight = self._get_resource_value_multiplier(resource_spec.resource_type)
            total_score += resource_spec.total_capacity * weight
        
        if total_score < 10:
            return "micro"
        elif total_score < 50:
            return "small"
        elif total_score < 200:
            return "medium"
        elif total_score < 1000:
            return "large"
        else:
            return "massive"
    
    def _get_resource_value_multiplier(self, resource_type: ResourceType) -> float:
        """Get value multiplier for different resource types"""
        multipliers = {
            ResourceType.COMPUTE_CPU: 1.0,
            ResourceType.COMPUTE_GPU: 5.0,  # GPUs are more valuable
            ResourceType.COMPUTE_TPU: 10.0,  # TPUs are very valuable
            ResourceType.STORAGE_PERSISTENT: 0.1,
            ResourceType.STORAGE_MEMORY: 2.0,
            ResourceType.BANDWIDTH_INGRESS: 0.5,
            ResourceType.BANDWIDTH_EGRESS: 0.5,
            ResourceType.SPECIALIZED_QUANTUM: 50.0  # Quantum is extremely valuable
        }
        return multipliers.get(resource_type, 1.0)
    
    async def _detect_geographic_region(self) -> str:
        """
        Detect node's geographic region using multiple strategies.
        
        Priority order:
        1. PRSM_REGION environment variable (operator override)
        2. Cached result (if within TTL)
        3. IP geolocation via ip-api.com (free, no API key)
        
        Returns:
            Region string (e.g., "us-west", "europe", "asia-pacific")
        """
        import os
        
        # Strategy 1: Environment variable override (highest priority)
        env_region = os.environ.get("PRSM_REGION")
        if env_region:
            logger.debug("Using PRSM_REGION environment variable", region=env_region)
            return env_region
        
        # Strategy 2: Check cache
        current_time = time.time()
        if (self._geographic_region_cache is not None and
            self._geographic_region_cache_time is not None and
            (current_time - self._geographic_region_cache_time) < self._geographic_region_cache_ttl):
            logger.debug("Using cached geographic region", region=self._geographic_region_cache)
            return self._geographic_region_cache
        
        # Strategy 3: IP geolocation via ip-api.com
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                resp = await session.get(
                    "http://ip-api.com/json/?fields=regionName,countryCode,lat,lon",
                    timeout=aiohttp.ClientTimeout(total=5)
                )
                data = await resp.json()
                
                country = data.get("countryCode", "")
                lat = data.get("lat", 0)
                lon = data.get("lon", 0)
                
                region = self._map_coords_to_region(lat, lon, country)
                
                # Cache the result
                self._geographic_region_cache = region
                self._geographic_region_cache_time = current_time
                
                logger.info("Detected geographic region via IP geolocation",
                           region=region, country=country, lat=lat, lon=lon)
                
                return region
                
        except Exception as e:
            logger.warning("IP geolocation failed, returning 'unknown'", error=str(e))
            return "unknown"
    
    def _map_coords_to_region(self, lat: float, lon: float, country: str) -> str:
        """
        Map geographic coordinates and country code to PRSM region string.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            country: ISO 3166-1 alpha-2 country code
            
        Returns:
            PRSM region string
        """
        # United States: split by longitude (roughly -100th meridian)
        if country == "US":
            return "us-west" if lon < -100 else "us-east"
        
        # Europe
        elif country in ["GB", "DE", "FR", "NL", "SE", "NO", "FI", "ES", "IT", "PT", "CH", "AT", "BE", "DK", "PL"]:
            return "europe"
        
        # Asia-Pacific
        elif country in ["JP", "KR", "CN", "TW", "HK", "SG", "AU", "NZ", "IN", "TH", "ID", "MY", "PH", "VN"]:
            return "asia-pacific"
        
        # South America
        elif country in ["BR", "AR", "CL", "CO", "PE", "MX"]:
            return "south-america"
        
        # Africa
        elif country in ["ZA", "NG", "KE", "EG", "MA"]:
            return "africa"
        
        # Middle East
        elif country in ["AE", "SA", "IL", "TR", "IR", "IQ"]:
            return "middle-east"
        
        # Fallback: lat/lon bounding boxes for unknown countries
        else:
            if -170 <= lon <= -50:
                return "americas"
            if -30 <= lon <= 60:
                return "europe-africa"
            return "asia-pacific"
    
    def _choose_verification_level(self, node_profile: NodeResourceProfile, resource_spec: ResourceSpec) -> ResourceVerificationLevel:
        """Choose appropriate verification level based on risk factors"""
        # High-value resources need stronger verification
        resource_value = resource_spec.total_capacity * self._get_resource_value_multiplier(resource_spec.resource_type)
        
        # Low reputation nodes need stronger verification
        reputation_factor = node_profile.reputation_score
        
        if resource_value > 100 or reputation_factor < 0.7:
            return ResourceVerificationLevel.HARDWARE_ATTESTED
        elif resource_value > 50 or reputation_factor < 0.8:
            return ResourceVerificationLevel.BENCHMARKED
        else:
            return ResourceVerificationLevel.PEER_VERIFIED
    
    async def _update_node_reputation(self, node_id: str, verification_proof: ResourceVerificationProof):
        """Update node reputation based on verification results"""
        node_profile = self.node_registry[node_id]
        verification_score = verification_proof.proof_data["verification_score"]
        
        # Weighted average of current reputation and verification score
        alpha = 0.1  # Learning rate
        node_profile.reputation_score = (
            (1 - alpha) * node_profile.reputation_score + 
            alpha * verification_score
        )
        
        logger.debug("Node reputation updated", 
                    node_id=node_id, 
                    new_reputation=node_profile.reputation_score,
                    verification_score=verification_score)


# ============================================================================
# GLOBAL INSTANCE AND INITIALIZATION
# ============================================================================

# Global distributed resource manager instance
distributed_resource_manager = DistributedResourceManager()


async def initialize_distributed_resource_management():
    """Initialize the distributed resource management system"""
    logger.info("Initializing PRSM Distributed Resource Management System")
    
    # Start continuous verification loop
    asyncio.create_task(distributed_resource_manager.continuous_verification_loop())
    
    logger.info("Distributed Resource Management System initialized")


# API functions for external use
async def register_node_resources(user_id: str, settings: ResourceContributionSettings) -> str:
    """Register a new node with specified resource contribution settings"""
    return await distributed_resource_manager.initialize_node(user_id, settings)


async def update_node_resource_settings(node_id: str, settings: ResourceContributionSettings) -> bool:
    """Update resource contribution settings for an existing node"""
    return await distributed_resource_manager.update_node_settings(node_id, settings)


async def allocate_resources_for_computation(
    requirements: Dict[ResourceType, float], 
    constraints: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Allocate optimal resources for a computational task"""
    return await distributed_resource_manager.allocate_task_resources(requirements, constraints)


def get_node_resource_profile(node_id: str) -> Optional[NodeResourceProfile]:
    """Get current resource profile for a node"""
    return distributed_resource_manager.node_registry.get(node_id)


def get_network_resource_summary() -> Dict[str, Any]:
    """Get summary of total network resources"""
    total_nodes = len(distributed_resource_manager.node_registry)
    
    resource_totals = {}
    for node_profile in distributed_resource_manager.node_registry.values():
        for resource_type, resource_spec in node_profile.resources.items():
            if resource_type not in resource_totals:
                resource_totals[resource_type] = {
                    "total_capacity": 0.0,
                    "allocated_capacity": 0.0,
                    "node_count": 0
                }
            
            resource_totals[resource_type]["total_capacity"] += resource_spec.total_capacity
            resource_totals[resource_type]["allocated_capacity"] += resource_spec.allocated_capacity
            resource_totals[resource_type]["node_count"] += 1
    
    return {
        "total_nodes": total_nodes,
        "resource_summary": resource_totals,
        "network_health": "operational",  # Would calculate actual health
        "average_node_reputation": sum(
            node.reputation_score for node in distributed_resource_manager.node_registry.values()
        ) / total_nodes if total_nodes > 0 else 0.0
    }