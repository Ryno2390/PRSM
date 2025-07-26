"""
Bandwidth Optimization System for PRSM P2P Collaboration

This module optimizes bandwidth usage across the P2P network by:
- Adaptive download/upload management
- Traffic shaping and QoS
- Bandwidth monitoring and prediction
- Dynamic load balancing
- Efficient chunk scheduling
- Network congestion avoidance

Key Features:
- Adaptive bitrate streaming for large files
- Intelligent peer selection based on bandwidth
- Traffic prioritization for collaborative sessions
- Bandwidth-aware redundancy management
- Real-time network condition monitoring
"""

import asyncio
import json
import logging
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
import heapq
import math

from .node_discovery import PeerNode

logger = logging.getLogger(__name__)


class TrafficPriority(Enum):
    """Traffic priority levels"""
    CRITICAL = "critical"      # Real-time collaboration
    HIGH = "high"             # Active file transfers
    NORMAL = "normal"         # Background sync
    LOW = "low"               # Maintenance traffic


class BandwidthUnit(Enum):
    """Bandwidth measurement units"""
    BYTES_PER_SEC = "bps"
    KILOBYTES_PER_SEC = "kbps"
    MEGABYTES_PER_SEC = "mbps"
    GIGABYTES_PER_SEC = "gbps"


@dataclass
class BandwidthMeasurement:
    """Bandwidth measurement data"""
    timestamp: float
    download_speed: float  # bytes per second
    upload_speed: float    # bytes per second
    latency: float         # seconds
    packet_loss: float     # percentage (0.0 to 1.0)
    node_id: Optional[str] = None
    
    def to_mbps(self, speed: float) -> float:
        """Convert bytes per second to Mbps"""
        return (speed * 8) / (1024 * 1024)
    
    @property
    def download_mbps(self) -> float:
        return self.to_mbps(self.download_speed)
    
    @property
    def upload_mbps(self) -> float:
        return self.to_mbps(self.upload_speed)


@dataclass
class TransferRequest:
    """Request for data transfer"""
    request_id: str
    source_node: str
    target_node: str
    data_size: int
    priority: TrafficPriority
    deadline: Optional[float] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
    
    @property
    def is_expired(self) -> bool:
        """Check if request has expired"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline
    
    @property
    def urgency_score(self) -> float:
        """Calculate urgency score for scheduling"""
        base_score = {
            TrafficPriority.CRITICAL: 1000,
            TrafficPriority.HIGH: 100,
            TrafficPriority.NORMAL: 10,
            TrafficPriority.LOW: 1
        }[self.priority]
        
        if self.deadline:
            time_remaining = max(0, self.deadline - time.time())
            if time_remaining > 0:
                urgency_multiplier = 1.0 / (time_remaining + 1)
                base_score *= (1 + urgency_multiplier)
        
        return base_score


@dataclass
class ActiveTransfer:
    """Information about an active transfer"""
    transfer_id: str
    request: TransferRequest
    start_time: float
    bytes_transferred: int = 0
    current_speed: float = 0  # bytes per second
    estimated_completion: Optional[float] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate transfer progress percentage"""
        if self.request.data_size == 0:
            return 100.0
        return min(100.0, (self.bytes_transferred / self.request.data_size) * 100)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed transfer time"""
        return time.time() - self.start_time
    
    def update_progress(self, bytes_transferred: int, current_speed: float):
        """Update transfer progress"""
        self.bytes_transferred = bytes_transferred
        self.current_speed = current_speed
        
        if current_speed > 0:
            remaining_bytes = self.request.data_size - bytes_transferred
            remaining_time = remaining_bytes / current_speed
            self.estimated_completion = time.time() + remaining_time


class NetworkMonitor:
    """Monitors network conditions and bandwidth usage"""
    
    def __init__(self, measurement_interval: float = 5.0):
        self.measurement_interval = measurement_interval
        self.measurements: Dict[str, List[BandwidthMeasurement]] = {}
        self.max_history = 100  # Keep last 100 measurements per node
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    def start_monitoring(self):
        """Start network monitoring"""
        if not self.running:
            self.running = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Network monitoring started")
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.running = False
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
        logger.info("Network monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._collect_measurements()
                await asyncio.sleep(self.measurement_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_measurements(self):
        """Collect bandwidth measurements"""
        # In a real implementation, this would:
        # 1. Measure actual network speeds to various peers
        # 2. Test latency and packet loss
        # 3. Monitor local network interface statistics
        
        # Placeholder implementation with simulated data
        timestamp = time.time()
        
        # Simulate measurements for demo purposes
        measurement = BandwidthMeasurement(
            timestamp=timestamp,
            download_speed=50 * 1024 * 1024,  # 50 MB/s
            upload_speed=25 * 1024 * 1024,    # 25 MB/s
            latency=0.05,  # 50ms
            packet_loss=0.01  # 1%
        )
        
        self._add_measurement("local", measurement)
    
    def _add_measurement(self, node_id: str, measurement: BandwidthMeasurement):
        """Add a measurement to history"""
        if node_id not in self.measurements:
            self.measurements[node_id] = []
        
        self.measurements[node_id].append(measurement)
        
        # Keep only recent measurements
        if len(self.measurements[node_id]) > self.max_history:
            self.measurements[node_id] = self.measurements[node_id][-self.max_history:]
    
    def get_recent_measurements(self, node_id: str, 
                              count: int = 10) -> List[BandwidthMeasurement]:
        """Get recent measurements for a node"""
        measurements = self.measurements.get(node_id, [])
        return measurements[-count:] if measurements else []
    
    def get_average_bandwidth(self, node_id: str, 
                            duration: float = 300) -> Optional[BandwidthMeasurement]:
        """Get average bandwidth over a time period"""
        measurements = self.measurements.get(node_id, [])
        if not measurements:
            return None
        
        cutoff_time = time.time() - duration
        recent_measurements = [
            m for m in measurements if m.timestamp > cutoff_time
        ]
        
        if not recent_measurements:
            return None
        
        avg_download = statistics.mean(m.download_speed for m in recent_measurements)
        avg_upload = statistics.mean(m.upload_speed for m in recent_measurements)
        avg_latency = statistics.mean(m.latency for m in recent_measurements)
        avg_loss = statistics.mean(m.packet_loss for m in recent_measurements)
        
        return BandwidthMeasurement(
            timestamp=time.time(),
            download_speed=avg_download,
            upload_speed=avg_upload,
            latency=avg_latency,
            packet_loss=avg_loss,
            node_id=node_id
        )
    
    def predict_bandwidth(self, node_id: str, 
                         prediction_window: float = 60) -> Optional[BandwidthMeasurement]:
        """Predict future bandwidth based on trends"""
        measurements = self.measurements.get(node_id, [])
        if len(measurements) < 3:
            return None
        
        # Simple linear trend prediction
        recent = measurements[-10:]  # Use last 10 measurements
        
        if len(recent) < 3:
            return recent[-1]
        
        # Calculate trends
        timestamps = [m.timestamp for m in recent]
        download_speeds = [m.download_speed for m in recent]
        upload_speeds = [m.upload_speed for m in recent]
        
        # Linear regression (simplified)
        def simple_trend(values, times):
            if len(values) < 2:
                return values[-1] if values else 0
            
            # Calculate slope
            time_diff = times[-1] - times[0]
            value_diff = values[-1] - values[0]
            
            if time_diff == 0:
                return values[-1]
            
            slope = value_diff / time_diff
            
            # Predict value at future time
            future_time = times[-1] + prediction_window
            predicted_value = values[-1] + slope * prediction_window
            
            # Clamp to reasonable bounds
            return max(0, min(predicted_value, values[-1] * 2))
        
        predicted_download = simple_trend(download_speeds, timestamps)
        predicted_upload = simple_trend(upload_speeds, timestamps)
        
        return BandwidthMeasurement(
            timestamp=time.time() + prediction_window,
            download_speed=predicted_download,
            upload_speed=predicted_upload,
            latency=recent[-1].latency,  # Use most recent latency
            packet_loss=recent[-1].packet_loss,
            node_id=node_id
        )


class AdaptiveRateController:
    """Controls adaptive transfer rates based on network conditions"""
    
    def __init__(self, initial_rate: float = 1024 * 1024):  # 1 MB/s default
        self.current_rate = initial_rate
        self.min_rate = 64 * 1024        # 64 KB/s minimum
        self.max_rate = 100 * 1024 * 1024  # 100 MB/s maximum
        
        # Adaptation parameters
        self.increase_factor = 1.2
        self.decrease_factor = 0.8
        self.stability_threshold = 0.1  # 10% variation threshold
        
        # Performance tracking
        self.rate_history: List[Tuple[float, float]] = []  # (timestamp, rate)
        self.performance_history: List[float] = []  # Recent throughput measurements
        
    def adapt_rate(self, measured_throughput: float, 
                   target_utilization: float = 0.8) -> float:
        """Adapt transfer rate based on measured performance"""
        current_time = time.time()
        
        # Record performance
        self.performance_history.append(measured_throughput)
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Calculate utilization
        if self.current_rate > 0:
            utilization = measured_throughput / self.current_rate
        else:
            utilization = 0
        
        # Adapt rate based on utilization
        if utilization > target_utilization:
            # We're achieving good throughput, try to increase
            new_rate = min(self.max_rate, self.current_rate * self.increase_factor)
        elif utilization < target_utilization * 0.5:
            # Poor utilization, decrease rate
            new_rate = max(self.min_rate, self.current_rate * self.decrease_factor)
        else:
            # Acceptable utilization, maintain rate
            new_rate = self.current_rate
        
        # Check for stability
        if len(self.performance_history) >= 3:
            recent_std = statistics.stdev(self.performance_history[-3:])
            recent_mean = statistics.mean(self.performance_history[-3:])
            
            if recent_mean > 0:
                coefficient_of_variation = recent_std / recent_mean
                
                if coefficient_of_variation > self.stability_threshold:
                    # High variation, be more conservative
                    new_rate = min(new_rate, self.current_rate * 1.05)
        
        # Update rate
        old_rate = self.current_rate
        self.current_rate = new_rate
        
        # Record rate change
        self.rate_history.append((current_time, new_rate))
        if len(self.rate_history) > 100:
            self.rate_history = self.rate_history[-100:]
        
        logger.debug(f"Rate adapted: {old_rate/1024/1024:.2f} -> {new_rate/1024/1024:.2f} MB/s "
                    f"(utilization: {utilization:.2%})")
        
        return new_rate
    
    def get_recommended_chunk_size(self) -> int:
        """Get recommended chunk size based on current rate"""
        # Aim for chunks that take 1-2 seconds to transfer
        target_duration = 1.5  # seconds
        chunk_size = int(self.current_rate * target_duration)
        
        # Clamp to reasonable bounds
        min_chunk = 64 * 1024      # 64 KB
        max_chunk = 10 * 1024 * 1024  # 10 MB
        
        return max(min_chunk, min(max_chunk, chunk_size))


class QoSManager:
    """Manages Quality of Service for different traffic types"""
    
    def __init__(self, total_bandwidth: float):
        self.total_bandwidth = total_bandwidth
        
        # Bandwidth allocation percentages by priority
        self.priority_allocations = {
            TrafficPriority.CRITICAL: 0.4,   # 40% for critical traffic
            TrafficPriority.HIGH: 0.3,       # 30% for high priority
            TrafficPriority.NORMAL: 0.2,     # 20% for normal traffic
            TrafficPriority.LOW: 0.1         # 10% for low priority
        }
        
        # Current usage tracking
        self.current_usage = {
            priority: 0.0 for priority in TrafficPriority
        }
        
        # Active transfers by priority
        self.active_transfers: Dict[TrafficPriority, Set[str]] = {
            priority: set() for priority in TrafficPriority
        }
    
    def get_allocated_bandwidth(self, priority: TrafficPriority) -> float:
        """Get allocated bandwidth for a priority level"""
        return self.total_bandwidth * self.priority_allocations[priority]
    
    def get_available_bandwidth(self, priority: TrafficPriority) -> float:
        """Get available bandwidth for a priority level"""
        allocated = self.get_allocated_bandwidth(priority)
        used = self.current_usage[priority]
        
        available = allocated - used
        
        # Allow borrowing from lower priorities if available
        if available <= 0:
            for lower_priority in self._get_lower_priorities(priority):
                lower_allocated = self.get_allocated_bandwidth(lower_priority)
                lower_used = self.current_usage[lower_priority]
                lower_available = lower_allocated - lower_used
                
                if lower_available > 0:
                    available += min(lower_available, -available)
                    break
        
        return max(0, available)
    
    def _get_lower_priorities(self, priority: TrafficPriority) -> List[TrafficPriority]:
        """Get priorities lower than the given priority"""
        priority_order = [
            TrafficPriority.CRITICAL,
            TrafficPriority.HIGH,
            TrafficPriority.NORMAL,
            TrafficPriority.LOW
        ]
        
        try:
            current_index = priority_order.index(priority)
            return priority_order[current_index + 1:]
        except ValueError:
            return []
    
    def reserve_bandwidth(self, transfer_id: str, priority: TrafficPriority, 
                         bandwidth: float) -> bool:
        """Reserve bandwidth for a transfer"""
        available = self.get_available_bandwidth(priority)
        
        if bandwidth <= available:
            self.current_usage[priority] += bandwidth
            self.active_transfers[priority].add(transfer_id)
            logger.debug(f"Reserved {bandwidth/1024/1024:.2f} MB/s for {transfer_id} "
                        f"(priority: {priority.value})")
            return True
        
        logger.warning(f"Insufficient bandwidth for {transfer_id}: "
                      f"requested {bandwidth/1024/1024:.2f} MB/s, "
                      f"available {available/1024/1024:.2f} MB/s")
        return False
    
    def release_bandwidth(self, transfer_id: str, priority: TrafficPriority, 
                         bandwidth: float):
        """Release bandwidth from a transfer"""
        self.current_usage[priority] = max(0, self.current_usage[priority] - bandwidth)
        self.active_transfers[priority].discard(transfer_id)
        
        logger.debug(f"Released {bandwidth/1024/1024:.2f} MB/s from {transfer_id}")
    
    def get_qos_stats(self) -> Dict[str, Any]:
        """Get QoS statistics"""
        stats = {}
        
        for priority in TrafficPriority:
            allocated = self.get_allocated_bandwidth(priority)
            used = self.current_usage[priority]
            available = self.get_available_bandwidth(priority)
            active_count = len(self.active_transfers[priority])
            
            stats[priority.value] = {
                'allocated_mbps': allocated / 1024 / 1024,
                'used_mbps': used / 1024 / 1024,
                'available_mbps': available / 1024 / 1024,
                'utilization_percent': (used / allocated * 100) if allocated > 0 else 0,
                'active_transfers': active_count
            }
        
        return stats


class TransferScheduler:
    """Schedules and manages file transfers based on priority and bandwidth"""
    
    def __init__(self, qos_manager: QoSManager, rate_controller: AdaptiveRateController):
        self.qos_manager = qos_manager
        self.rate_controller = rate_controller
        
        # Transfer queues by priority
        self.transfer_queues: Dict[TrafficPriority, List[TransferRequest]] = {
            priority: [] for priority in TrafficPriority
        }
        
        # Active transfers
        self.active_transfers: Dict[str, ActiveTransfer] = {}
        
        # Scheduler state
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
    
    def start_scheduler(self):
        """Start the transfer scheduler"""
        if not self.running:
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("Transfer scheduler started")
    
    def stop_scheduler(self):
        """Stop the transfer scheduler"""
        self.running = False
        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
        logger.info("Transfer scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                await self._process_transfer_queues()
                await self._update_active_transfers()
                await asyncio.sleep(1.0)  # Run every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    async def _process_transfer_queues(self):
        """Process pending transfer requests"""
        for priority in TrafficPriority:
            queue = self.transfer_queues[priority]
            
            # Remove expired requests
            queue[:] = [req for req in queue if not req.is_expired]
            
            # Sort by urgency
            queue.sort(key=lambda req: req.urgency_score, reverse=True)
            
            # Try to start transfers
            while queue:
                request = queue[0]
                
                # Estimate bandwidth requirement
                estimated_bandwidth = self._estimate_bandwidth_requirement(request)
                
                # Check if bandwidth is available
                if self.qos_manager.reserve_bandwidth(
                    request.request_id, request.priority, estimated_bandwidth
                ):
                    # Start transfer
                    queue.pop(0)
                    await self._start_transfer(request, estimated_bandwidth)
                else:
                    # No bandwidth available, wait
                    break
    
    def _estimate_bandwidth_requirement(self, request: TransferRequest) -> float:
        """Estimate bandwidth requirement for a transfer"""
        # Base estimate on file size and deadline
        if request.deadline:
            time_remaining = max(1, request.deadline - time.time())
            required_rate = request.data_size / time_remaining
            
            # Add buffer for reliability
            return required_rate * 1.2
        else:
            # Use adaptive rate as estimate
            return self.rate_controller.current_rate
    
    async def _start_transfer(self, request: TransferRequest, allocated_bandwidth: float):
        """Start an active transfer"""
        transfer = ActiveTransfer(
            transfer_id=request.request_id,
            request=request,
            start_time=time.time()
        )
        
        self.active_transfers[request.request_id] = transfer
        
        # Start transfer task
        transfer_task = asyncio.create_task(
            self._execute_transfer(transfer, allocated_bandwidth)
        )
        
        logger.info(f"Started transfer {request.request_id}: "
                   f"{request.data_size/1024/1024:.2f} MB "
                   f"(priority: {request.priority.value})")
    
    async def _execute_transfer(self, transfer: ActiveTransfer, allocated_bandwidth: float):
        """Execute a file transfer"""
        try:
            # Simulate file transfer with adaptive rate control
            total_size = transfer.request.data_size
            chunk_size = self.rate_controller.get_recommended_chunk_size()
            
            bytes_transferred = 0
            
            while bytes_transferred < total_size:
                chunk_size = min(chunk_size, total_size - bytes_transferred)
                
                # Simulate transfer time based on allocated bandwidth
                transfer_time = chunk_size / allocated_bandwidth
                await asyncio.sleep(transfer_time)
                
                bytes_transferred += chunk_size
                
                # Update progress
                current_speed = chunk_size / transfer_time
                transfer.update_progress(bytes_transferred, current_speed)
                
                # Adapt rate based on performance
                self.rate_controller.adapt_rate(current_speed)
                chunk_size = self.rate_controller.get_recommended_chunk_size()
                
                logger.debug(f"Transfer {transfer.transfer_id}: "
                           f"{transfer.progress_percentage:.1f}% complete")
            
            logger.info(f"Transfer {transfer.transfer_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Transfer {transfer.transfer_id} failed: {e}")
        
        finally:
            # Release bandwidth and cleanup
            self.qos_manager.release_bandwidth(
                transfer.transfer_id, transfer.request.priority, allocated_bandwidth
            )
            
            if transfer.transfer_id in self.active_transfers:
                del self.active_transfers[transfer.transfer_id]
    
    async def _update_active_transfers(self):
        """Update status of active transfers"""
        for transfer in list(self.active_transfers.values()):
            # Check for timeouts or other issues
            if transfer.request.is_expired:
                logger.warning(f"Transfer {transfer.transfer_id} expired")
                # Cancel transfer would go here
    
    def queue_transfer(self, request: TransferRequest):
        """Queue a transfer request"""
        self.transfer_queues[request.priority].append(request)
        logger.info(f"Queued transfer {request.request_id}: "
                   f"{request.data_size/1024/1024:.2f} MB "
                   f"(priority: {request.priority.value})")
    
    def get_transfer_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a transfer"""
        if request_id in self.active_transfers:
            transfer = self.active_transfers[request_id]
            return {
                'status': 'active',
                'progress_percent': transfer.progress_percentage,
                'bytes_transferred': transfer.bytes_transferred,
                'current_speed_mbps': transfer.current_speed / 1024 / 1024,
                'elapsed_time': transfer.elapsed_time,
                'estimated_completion': transfer.estimated_completion
            }
        
        # Check queues
        for priority, queue in self.transfer_queues.items():
            for i, request in enumerate(queue):
                if request.request_id == request_id:
                    return {
                        'status': 'queued',
                        'priority': priority.value,
                        'queue_position': i + 1,
                        'urgency_score': request.urgency_score
                    }
        
        return None
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        queue_stats = {}
        for priority, queue in self.transfer_queues.items():
            queue_stats[priority.value] = {
                'queued_transfers': len(queue),
                'total_queued_bytes': sum(req.data_size for req in queue)
            }
        
        active_stats = {
            'active_transfers': len(self.active_transfers),
            'total_active_bytes': sum(
                t.request.data_size for t in self.active_transfers.values()
            )
        }
        
        return {
            'queues': queue_stats,
            'active': active_stats,
            'qos': self.qos_manager.get_qos_stats()
        }


# Main bandwidth optimization system
class BandwidthOptimizer:
    """
    Main bandwidth optimization system
    
    Coordinates network monitoring, adaptive rate control, QoS management,
    and transfer scheduling for optimal P2P performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.network_monitor = NetworkMonitor(
            self.config.get('monitor_interval', 5.0)
        )
        
        total_bandwidth = self.config.get('total_bandwidth', 100 * 1024 * 1024)  # 100 MB/s
        self.qos_manager = QoSManager(total_bandwidth)
        
        initial_rate = self.config.get('initial_rate', 10 * 1024 * 1024)  # 10 MB/s
        self.rate_controller = AdaptiveRateController(initial_rate)
        
        self.scheduler = TransferScheduler(self.qos_manager, self.rate_controller)
        
        # System state
        self.running = False
    
    async def start(self):
        """Start the bandwidth optimization system"""
        if self.running:
            return
        
        self.running = True
        
        # Start components
        self.network_monitor.start_monitoring()
        self.scheduler.start_scheduler()
        
        logger.info("Bandwidth optimization system started")
    
    async def stop(self):
        """Stop the bandwidth optimization system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        self.network_monitor.stop_monitoring()
        self.scheduler.stop_scheduler()
        
        logger.info("Bandwidth optimization system stopped")
    
    def request_transfer(self, source_node: str, target_node: str, 
                        data_size: int, priority: TrafficPriority,
                        deadline: Optional[float] = None) -> str:
        """Request a data transfer"""
        request_id = f"transfer_{int(time.time())}_{hash((source_node, target_node))}"
        
        request = TransferRequest(
            request_id=request_id,
            source_node=source_node,
            target_node=target_node,
            data_size=data_size,
            priority=priority,
            deadline=deadline
        )
        
        self.scheduler.queue_transfer(request)
        return request_id
    
    def get_optimal_transfer_rate(self, target_node: str) -> float:
        """Get optimal transfer rate for a target node"""
        # Get recent bandwidth measurements
        avg_bandwidth = self.network_monitor.get_average_bandwidth(target_node)
        
        if avg_bandwidth:
            # Use conservative estimate (80% of available bandwidth)
            optimal_rate = min(
                avg_bandwidth.download_speed * 0.8,
                self.rate_controller.current_rate
            )
        else:
            # Fall back to current adaptive rate
            optimal_rate = self.rate_controller.current_rate
        
        return optimal_rate
    
    def get_network_quality_score(self, node_id: str) -> float:
        """Get quality score for a network node"""
        avg_measurement = self.network_monitor.get_average_bandwidth(node_id)
        
        if not avg_measurement:
            return 0.5  # Default neutral score
        
        # Calculate score based on multiple factors
        bandwidth_score = min(1.0, avg_measurement.download_mbps / 100)  # Normalize to 100 Mbps
        latency_score = max(0, 1.0 - avg_measurement.latency)  # Lower is better
        reliability_score = 1.0 - avg_measurement.packet_loss  # Lower loss is better
        
        # Weighted combination
        quality_score = (
            bandwidth_score * 0.5 +
            latency_score * 0.3 +
            reliability_score * 0.2
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'network_monitor': {
                'monitored_nodes': len(self.network_monitor.measurements),
                'total_measurements': sum(
                    len(measurements) for measurements in self.network_monitor.measurements.values()
                )
            },
            'rate_controller': {
                'current_rate_mbps': self.rate_controller.current_rate / 1024 / 1024,
                'recommended_chunk_size_kb': self.rate_controller.get_recommended_chunk_size() / 1024
            },
            'scheduler': self.scheduler.get_scheduler_stats()
        }


# Example usage
async def example_bandwidth_optimization():
    """Example of bandwidth optimization system usage"""
    
    config = {
        'total_bandwidth': 50 * 1024 * 1024,  # 50 MB/s total
        'initial_rate': 5 * 1024 * 1024,      # 5 MB/s initial
        'monitor_interval': 2.0
    }
    
    optimizer = BandwidthOptimizer(config)
    
    try:
        await optimizer.start()
        
        # Request some transfers
        transfer_id1 = optimizer.request_transfer(
            "node1", "node2", 100 * 1024 * 1024,  # 100 MB
            TrafficPriority.HIGH, deadline=time.time() + 300  # 5 minute deadline
        )
        
        transfer_id2 = optimizer.request_transfer(
            "node1", "node3", 50 * 1024 * 1024,   # 50 MB
            TrafficPriority.NORMAL
        )
        
        # Monitor progress
        for i in range(10):
            await asyncio.sleep(2)
            
            status1 = optimizer.scheduler.get_transfer_status(transfer_id1)
            status2 = optimizer.scheduler.get_transfer_status(transfer_id2)
            
            print(f"Transfer 1: {status1}")
            print(f"Transfer 2: {status2}")
            print(f"System stats: {json.dumps(optimizer.get_system_stats(), indent=2)}")
            print("---")
        
    finally:
        await optimizer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_bandwidth_optimization())