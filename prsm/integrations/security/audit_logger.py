"""
Security Audit Logger
====================

Comprehensive audit logging for security events in the PRSM integration layer.
Tracks all security-related activities for monitoring and compliance.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4


class EventLevel(str, Enum):
    """Security event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SecurityEvent:
    """Security event data structure"""
    
    def __init__(self, event_type: str, level: EventLevel, 
                 user_id: str, platform: str, description: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.event_id = str(uuid4())
        self.event_type = event_type
        self.level = level
        self.user_id = user_id
        self.platform = platform
        self.description = description
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "level": self.level.value,
            "user_id": self.user_id,
            "platform": self.platform,
            "description": self.description,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class AuditLogger:
    """Security audit logging system"""
    
    def __init__(self, log_dir: str = "logs/security"):
        """Initialize audit logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.events_log = self.log_dir / "security_events.jsonl"
        self.summary_log = self.log_dir / "security_summary.json"
        
        # Event counters for statistics
        self.event_counts = {
            EventLevel.INFO: 0,
            EventLevel.WARNING: 0,
            EventLevel.ERROR: 0,
            EventLevel.CRITICAL: 0
        }
        
        print(f"🔐 Audit Logger initialized: {self.log_dir}")
    
    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event"""
        try:
            # Write event to JSONL log
            with open(self.events_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
            
            # Update counters
            self.event_counts[event.level] += 1
            
            # Update summary
            self._update_summary()
            
            # Log critical events to console
            if event.level == EventLevel.CRITICAL:
                print(f"🚨 CRITICAL SECURITY EVENT: {event.description}")
            
        except Exception as e:
            print(f"⚠️ Failed to log security event: {e}")
    
    def log_vulnerability_scan(self, user_id: str, platform: str, 
                             scan_result: Any, content_id: str) -> None:
        """Log vulnerability scan results"""
        level = EventLevel.INFO
        if hasattr(scan_result, 'risk_level'):
            if scan_result.risk_level.value in ['high', 'critical']:
                level = EventLevel.ERROR
            elif scan_result.risk_level.value == 'medium':
                level = EventLevel.WARNING
        
        event = SecurityEvent(
            event_type="vulnerability_scan",
            level=level,
            user_id=user_id,
            platform=platform,
            description=f"Vulnerability scan completed for content {content_id}",
            metadata={
                "content_id": content_id,
                "vulnerabilities_found": len(scan_result.vulnerabilities) if hasattr(scan_result, 'vulnerabilities') else 0,
                "risk_level": scan_result.risk_level.value if hasattr(scan_result, 'risk_level') else "unknown",
                "scan_method": scan_result.scan_method if hasattr(scan_result, 'scan_method') else "unknown"
            }
        )
        
        self.log_event(event)
    
    def log_license_check(self, user_id: str, platform: str,
                         license_result: Any, content_id: str) -> None:
        """Log license compliance check"""
        level = EventLevel.INFO if license_result.compliant else EventLevel.WARNING
        
        event = SecurityEvent(
            event_type="license_check",
            level=level,
            user_id=user_id,
            platform=platform,
            description=f"License check for content {content_id}: {'COMPLIANT' if license_result.compliant else 'NON-COMPLIANT'}",
            metadata={
                "content_id": content_id,
                "license_type": license_result.license_type.value if hasattr(license_result.license_type, 'value') else str(license_result.license_type),
                "compliant": license_result.compliant,
                "issues": license_result.issues
            }
        )
        
        self.log_event(event)
    
    def log_sandbox_execution(self, user_id: str, platform: str,
                            execution_result: Any, content_id: str) -> None:
        """Log sandbox execution"""
        level = EventLevel.INFO
        if hasattr(execution_result, 'success') and not execution_result.success:
            level = EventLevel.ERROR
        
        event = SecurityEvent(
            event_type="sandbox_execution",
            level=level,
            user_id=user_id,
            platform=platform,
            description=f"Sandbox execution for content {content_id}",
            metadata={
                "content_id": content_id,
                "success": execution_result.success if hasattr(execution_result, 'success') else None,
                "execution_time": execution_result.execution_time if hasattr(execution_result, 'execution_time') else None,
                "exit_code": execution_result.exit_code if hasattr(execution_result, 'exit_code') else None
            }
        )
        
        self.log_event(event)
    
    def log_threat_detection(self, user_id: str, platform: str,
                           threat_result: Any, content_id: str) -> None:
        """Log threat detection results"""
        level = EventLevel.INFO
        if hasattr(threat_result, 'threat_level'):
            if threat_result.threat_level.value in ['high', 'critical']:
                level = EventLevel.CRITICAL
            elif threat_result.threat_level.value == 'medium':
                level = EventLevel.ERROR
        
        event = SecurityEvent(
            event_type="threat_detection",
            level=level,
            user_id=user_id,
            platform=platform,
            description=f"Threat detection scan for content {content_id}",
            metadata={
                "content_id": content_id,
                "threats_found": len(threat_result.threats) if hasattr(threat_result, 'threats') else 0,
                "threat_level": threat_result.threat_level.value if hasattr(threat_result, 'threat_level') else "unknown",
                "scan_method": threat_result.scan_method if hasattr(threat_result, 'scan_method') else "unknown"
            }
        )
        
        self.log_event(event)
    
    def log_import_security_check(self, user_id: str, platform: str,
                                content_id: str, passed: bool, issues: List[str]) -> None:
        """Log overall import security validation"""
        level = EventLevel.INFO if passed else EventLevel.ERROR
        
        event = SecurityEvent(
            event_type="import_security_check",
            level=level,
            user_id=user_id,
            platform=platform,
            description=f"Security validation for import {content_id}: {'PASSED' if passed else 'FAILED'}",
            metadata={
                "content_id": content_id,
                "security_passed": passed,
                "issues_found": len(issues),
                "issues": issues
            }
        )
        
        self.log_event(event)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        return {
            "total_events": sum(self.event_counts.values()),
            "events_by_level": {
                level.value: count for level, count in self.event_counts.items()
            },
            "log_file": str(self.events_log),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def get_recent_events(self, limit: int = 50, level: Optional[EventLevel] = None) -> List[Dict[str, Any]]:
        """Get recent security events"""
        events = []
        
        try:
            if self.events_log.exists():
                with open(self.events_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Get last N lines and parse JSON
                for line in lines[-limit:]:
                    try:
                        event_data = json.loads(line.strip())
                        if level is None or event_data.get('level') == level.value:
                            events.append(event_data)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"⚠️ Failed to read security events: {e}")
        
        return events
    
    def _update_summary(self) -> None:
        """Update security summary file"""
        try:
            summary = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "event_counts": {
                    level.value: count for level, count in self.event_counts.items()
                },
                "total_events": sum(self.event_counts.values()),
                "log_file": str(self.events_log)
            }
            
            with open(self.summary_log, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to update security summary: {e}")


# Global audit logger instance
audit_logger = AuditLogger()