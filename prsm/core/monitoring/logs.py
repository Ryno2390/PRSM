"""
PRSM Log Analysis and Monitoring
===============================

Advanced log analysis system for PRSM applications.
Provides real-time log monitoring, pattern detection,
and automated insights generation.
"""

import asyncio
import re
import json
import time
from typing import Dict, List, Optional, Any, Callable, Pattern
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from pathlib import Path
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
from datetime import datetime, timedelta
import threading


@dataclass
class LogEntry:
    """Represents a single log entry"""
    timestamp: datetime
    level: str
    message: str
    logger_name: str
    thread_id: Optional[str] = None
    process_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogPattern:
    """Pattern for log analysis"""
    name: str
    pattern: Pattern[str]
    severity: str
    description: str
    action: Optional[str] = None


@dataclass
class LogMetrics:
    """Metrics derived from log analysis"""
    total_entries: int
    error_count: int
    warning_count: int
    info_count: int
    debug_count: int
    critical_count: int
    patterns_matched: Dict[str, int]
    time_range: Dict[str, datetime]
    top_loggers: Dict[str, int]
    error_rate: float
    events_per_second: float


class LogAnalyzer:
    """
    Comprehensive log analysis and monitoring system
    """
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.log_entries: deque = deque(maxlen=max_entries)
        self.patterns: List[LogPattern] = []
        self.pattern_matches: Dict[str, List[LogEntry]] = defaultdict(list)
        self.metrics_cache: Optional[LogMetrics] = None
        self.cache_timestamp: float = 0
        self.cache_ttl: float = 60  # Cache TTL in seconds
        self.lock = threading.Lock()
        
        # Initialize default patterns
        self._initialize_default_patterns()
        
        # Real-time monitoring
        self.alert_callbacks: List[Callable] = []
        self.monitoring_active = False
    
    def _initialize_default_patterns(self):
        """Initialize common log patterns"""
        default_patterns = [
            LogPattern(
                name="error_pattern",
                pattern=re.compile(r"ERROR|Exception|Traceback|Failed|Error", re.IGNORECASE),
                severity="error",
                description="General error pattern",
                action="investigate_error"
            ),
            LogPattern(
                name="warning_pattern", 
                pattern=re.compile(r"WARNING|WARN|Deprecated", re.IGNORECASE),
                severity="warning",
                description="Warning pattern"
            ),
            LogPattern(
                name="critical_pattern",
                pattern=re.compile(r"CRITICAL|FATAL|EMERGENCY", re.IGNORECASE),
                severity="critical",
                description="Critical error pattern",
                action="immediate_attention"
            ),
            LogPattern(
                name="performance_pattern",
                pattern=re.compile(r"slow|timeout|latency|performance", re.IGNORECASE),
                severity="warning",
                description="Performance issue pattern",
                action="performance_review"
            ),
            LogPattern(
                name="security_pattern",
                pattern=re.compile(r"authentication|authorization|security|breach|hack", re.IGNORECASE),
                severity="critical",
                description="Security-related pattern",
                action="security_review"
            ),
            LogPattern(
                name="memory_pattern",
                pattern=re.compile(r"memory|oom|out of memory|heap", re.IGNORECASE),
                severity="error",
                description="Memory-related issues",
                action="memory_investigation"
            )
        ]
        
        self.patterns.extend(default_patterns)
    
    def add_pattern(self, pattern: LogPattern):
        """Add a custom log pattern"""
        with self.lock:
            self.patterns.append(pattern)
    
    def remove_pattern(self, pattern_name: str):
        """Remove a log pattern"""
        with self.lock:
            self.patterns = [p for p in self.patterns if p.name != pattern_name]
            self.pattern_matches.pop(pattern_name, None)
    
    def parse_log_entry(self, log_line: str, logger_name: str = "unknown") -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry"""
        try:
            # Try to parse common log formats
            # Format: timestamp level logger_name message
            parts = log_line.strip().split(' ', 3)
            
            if len(parts) >= 3:
                timestamp_str = parts[0]
                level = parts[1].upper()
                message = parts[2] if len(parts) == 3 else ' '.join(parts[2:])
                
                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except ValueError:
                    timestamp = datetime.now()
                
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    logger_name=logger_name
                )
            else:
                # Fallback: treat entire line as message
                return LogEntry(
                    timestamp=datetime.now(),
                    level="INFO",
                    message=log_line.strip(),
                    logger_name=logger_name
                )
        
        except Exception:
            return None
    
    def add_log_entry(self, entry: LogEntry):
        """Add a log entry for analysis"""
        with self.lock:
            self.log_entries.append(entry)
            self._analyze_entry(entry)
            self._invalidate_cache()
    
    def add_log_line(self, log_line: str, logger_name: str = "unknown"):
        """Add a log line for analysis"""
        entry = self.parse_log_entry(log_line, logger_name)
        if entry:
            self.add_log_entry(entry)
    
    def _analyze_entry(self, entry: LogEntry):
        """Analyze a log entry against patterns"""
        message = entry.message
        
        for pattern in self.patterns:
            if pattern.pattern.search(message):
                self.pattern_matches[pattern.name].append(entry)
                
                # Trigger alerts for critical patterns
                if pattern.severity == "critical" and self.alert_callbacks:
                    asyncio.create_task(self._trigger_alerts(pattern, entry))
    
    async def _trigger_alerts(self, pattern: LogPattern, entry: LogEntry):
        """Trigger alert callbacks for critical patterns"""
        for callback in self.alert_callbacks:
            try:
                await callback(pattern, entry)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback for critical patterns"""
        self.alert_callbacks.append(callback)
    
    def _invalidate_cache(self):
        """Invalidate metrics cache"""
        self.metrics_cache = None
        self.cache_timestamp = 0
    
    def get_metrics(self, force_refresh: bool = False) -> LogMetrics:
        """Get comprehensive log metrics"""
        current_time = time.time()
        
        if (not force_refresh and 
            self.metrics_cache and 
            current_time - self.cache_timestamp < self.cache_ttl):
            return self.metrics_cache
        
        with self.lock:
            entries = list(self.log_entries)
        
        if not entries:
            return LogMetrics(
                total_entries=0,
                error_count=0,
                warning_count=0,
                info_count=0,
                debug_count=0,
                critical_count=0,
                patterns_matched={},
                time_range={},
                top_loggers={},
                error_rate=0.0,
                events_per_second=0.0
            )
        
        # Calculate metrics
        level_counts = defaultdict(int)
        logger_counts = defaultdict(int)
        
        for entry in entries:
            level_counts[entry.level] += 1
            logger_counts[entry.logger_name] += 1
        
        # Pattern matches
        pattern_counts = {name: len(matches) for name, matches in self.pattern_matches.items()}
        
        # Time range
        timestamps = [entry.timestamp for entry in entries]
        time_range = {
            'start': min(timestamps),
            'end': max(timestamps)
        }
        
        # Calculate rates
        time_span = (time_range['end'] - time_range['start']).total_seconds()
        events_per_second = len(entries) / max(time_span, 1)
        error_rate = (level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)) / len(entries)
        
        # Top loggers
        top_loggers = dict(sorted(logger_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        metrics = LogMetrics(
            total_entries=len(entries),
            error_count=level_counts.get('ERROR', 0),
            warning_count=level_counts.get('WARNING', 0) + level_counts.get('WARN', 0),
            info_count=level_counts.get('INFO', 0),
            debug_count=level_counts.get('DEBUG', 0),
            critical_count=level_counts.get('CRITICAL', 0) + level_counts.get('FATAL', 0),
            patterns_matched=pattern_counts,
            time_range=time_range,
            top_loggers=top_loggers,
            error_rate=error_rate,
            events_per_second=events_per_second
        )
        
        # Cache the results
        self.metrics_cache = metrics
        self.cache_timestamp = current_time
        
        return metrics
    
    def get_entries_by_level(self, level: str, limit: Optional[int] = None) -> List[LogEntry]:
        """Get log entries by level"""
        with self.lock:
            entries = [entry for entry in self.log_entries if entry.level.upper() == level.upper()]
        
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def get_entries_by_pattern(self, pattern_name: str, limit: Optional[int] = None) -> List[LogEntry]:
        """Get log entries matching a specific pattern"""
        matches = self.pattern_matches.get(pattern_name, [])
        
        if limit:
            matches = matches[-limit:]
        
        return matches
    
    def get_recent_entries(self, minutes: int = 60) -> List[LogEntry]:
        """Get log entries from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_entries = [
                entry for entry in self.log_entries 
                if entry.timestamp >= cutoff_time
            ]
        
        return recent_entries
    
    def search_logs(self, query: str, case_sensitive: bool = False) -> List[LogEntry]:
        """Search log entries by message content"""
        pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        
        with self.lock:
            matching_entries = [
                entry for entry in self.log_entries
                if pattern.search(entry.message)
            ]
        
        return matching_entries
    
    async def monitor_log_file(self, file_path: str, logger_name: str = None):
        """Monitor a log file for new entries"""
        if not HAS_AIOFILES:
            logging.warning("aiofiles not available, cannot monitor log files")
            return
            
        if logger_name is None:
            logger_name = Path(file_path).stem
        
        try:
            async with aiofiles.open(file_path, 'r') as file:
                # Go to end of file
                await file.seek(0, 2)
                
                while self.monitoring_active:
                    line = await file.readline()
                    if line:
                        self.add_log_line(line.strip(), logger_name)
                    else:
                        await asyncio.sleep(0.1)
        
        except Exception as e:
            logging.error(f"Error monitoring log file {file_path}: {e}")
    
    def start_monitoring(self):
        """Start real-time log monitoring"""
        self.monitoring_active = True
    
    def stop_monitoring(self):
        """Stop real-time log monitoring"""
        self.monitoring_active = False
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        metrics = self.get_metrics()
        
        # Convert to serializable format
        data = {
            'total_entries': metrics.total_entries,
            'error_count': metrics.error_count,
            'warning_count': metrics.warning_count,
            'info_count': metrics.info_count,
            'debug_count': metrics.debug_count,
            'critical_count': metrics.critical_count,
            'patterns_matched': metrics.patterns_matched,
            'time_range': {
                'start': metrics.time_range['start'].isoformat() if metrics.time_range else None,
                'end': metrics.time_range['end'].isoformat() if metrics.time_range else None
            },
            'top_loggers': metrics.top_loggers,
            'error_rate': metrics.error_rate,
            'events_per_second': metrics.events_per_second
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_logs(self):
        """Clear all log entries and pattern matches"""
        with self.lock:
            self.log_entries.clear()
            self.pattern_matches.clear()
            self._invalidate_cache()